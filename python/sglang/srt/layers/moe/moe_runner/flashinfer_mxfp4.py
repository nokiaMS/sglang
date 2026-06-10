# FlashInfer SM90 cutlass 混合输入 W4A16 MXFP4 MoE 融合函数模块
# 本模块实现了基于 FlashInfer SM90 cutlass 的 W4A16 MXFP4 MoE 前向计算。
# 注册为 ("none", "flashinfer_mxfp4")，驱动 FlashInfer 的
# cutlass_fused_moe(use_w4_group_scaling=True)。

"""FlashInfer SM90 cutlass mixed-input W4A16 MXFP4 MoE fused func.
FlashInfer SM90 cutlass 混合输入 W4A16 MXFP4 MoE 融合函数。

Registered for ``("none", "flashinfer_mxfp4")``. Drives FlashInfer's
``cutlass_fused_moe(use_w4_group_scaling=True)`` (PR #3084 in flashinfer,
SM90 only). Quant methods build the quant_info each forward and call
``MoeRunner.run(dispatch_output, quant_info)``.
注册为 ``("none", "flashinfer_mxfp4")``。驱动 FlashInfer 的
``cutlass_fused_moe(use_w4_group_scaling=True)``（flashinfer PR #3084，
仅 SM90）。量化方法每次前向构建 quant_info 并调用
``MoeRunner.run(dispatch_output, quant_info)``。

Two production call sites share this fused func:
两个生产调用点共享此融合函数：
  - GPT-OSS via :class:`Mxfp4MoEMethod` (input pad/output trim + per-expert
    SwiGLU scalars + per-expert bias)
  - GPT-OSS 通过 :class:`Mxfp4MoEMethod`（输入填充/输出裁剪 + 每专家
    SwiGLU 标量 + 每专家偏置）
  - DSv4 via :class:`Mxfp4FlashinferCutlassMoEMethod` (no bias, optional
    SwiGLU scalars, no padding)
  - DSv4 通过 :class:`Mxfp4FlashinferCutlassMoEMethod`（无偏置，可选
    SwiGLU 标量，无填充）

The SM100 trtllm-gen path also lives under ``MoeRunnerBackend.FLASHINFER_MXFP4``
but is intentionally left in the legacy bypass path for now; migrating it is a
follow-up.
SM100 trtllm-gen 路径也位于 ``MoeRunnerBackend.FLASHINFER_MXFP4`` 下，
但目前有意保留在旧版绕过路径中；迁移它作为后续工作。
"""

from __future__ import annotations  # 启用延迟类型注解求值

from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架

from sglang.srt.distributed import get_tp_group  # 导入TP组获取函数
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,  # 导入对称内存上下文管理器
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric  # 导入分配对称性检查函数
from sglang.srt.layers.moe.moe_runner.base import (  # 从MoE运行器基类导入
    MoeQuantInfo,  # MoE量化信息基类
    MoeRunnerConfig,  # MoE运行器配置类
    register_fused_func,  # 融合函数注册装饰器
)
from sglang.srt.utils import is_flashinfer_available  # 导入FlashInfer可用性检查
from sglang.srt.utils.common import next_power_of_2  # 导入下一个2的幂函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput  # 标准分发输出
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 标准合并输入


@dataclass
class FlashInferMxfp4CutlassMoeQuantInfo(MoeQuantInfo):
    """Quantization payload for the SM90 cutlass W4A16 MXFP4 MoE path.
    SM90 cutlass W4A16 MXFP4 MoE 路径的量化负载。

    Weights and scales are pre-interleaved at load time via
    ``interleave_moe_{weights,scales}_for_sm90_mixed_gemm``; this dataclass
    only carries references plus the per-call routing/topology fields.
    权重和缩放因子在加载时通过
    ``interleave_moe_{weights,scales}_for_sm90_mixed_gemm`` 预交错；
    此数据类仅携带引用及每次调用的路由/拓扑字段。
    """

    # Pre-interleaved weights (uint8, packed FP4) / 预交错权重（uint8，打包FP4）
    w13_weight: torch.Tensor  # [E, 2*N, K/2] W13权重
    w2_weight: torch.Tensor  # [E, K, N/2] W2权重

    # Pre-interleaved E8M0 block scales (uint8; viewed as int32 at call time) / 预交错E8M0块缩放因子（uint8；调用时视为int32）
    w13_weight_scale: torch.Tensor  # [E, 2*N, K/32] W13权重缩放因子
    w2_weight_scale: torch.Tensor  # [E, K, N/32] W2权重缩放因子

    # Per-expert bias. GPT-OSS has both; DSv4 leaves both None. / 每专家偏置。GPT-OSS两者都有；DSv4两者都为None。
    w13_bias: Optional[torch.Tensor] = None  # bf16 [E, 2*N] W13偏置
    w2_bias: Optional[torch.Tensor] = None  # bf16 [E, K] W2偏置

    # Per-expert SwiGLU scalars (fp32 [E]). Either all three are present
    # (clamped SwiGLU) or all three are None (kernel default SwiGLU).
    # 每专家 SwiGLU 标量（fp32 [E]）。要么三个都存在
    # （钳位 SwiGLU），要么三个都为 None（内核默认 SwiGLU）。
    swiglu_alpha: Optional[torch.Tensor] = None  # SwiGLU alpha参数
    swiglu_beta: Optional[torch.Tensor] = None  # SwiGLU beta参数
    swiglu_limit: Optional[torch.Tensor] = None  # SwiGLU limit参数

    # TP/EP topology (forwarded to the FlashInfer kernel) / TP/EP拓扑（转发到FlashInfer内核）
    moe_tp_size: int = 1  # 张量并行大小
    moe_tp_rank: int = 0  # 张量并行rank
    moe_ep_size: int = 1  # 专家并行大小
    moe_ep_rank: int = 0  # 专家并行rank

    # GPT-OSS pads its input hidden dim up to the (pre-padded) loaded weight
    # width and trims the output back. DSv4 leaves this as ``None`` (no pad).
    # GPT-OSS 将输入隐藏维度填充到（预填充的）加载权重
    # 宽度并将输出裁剪回来。DSv4 将此设为 ``None``（无填充）。
    padded_hidden: Optional[int] = None  # 填充后的隐藏维度大小


def _flashinfer_cutlass_fused_moe():
    """Lazy import — keeps non-flashinfer wheels importable.
    惰性导入——保持非 FlashInfer 的 wheel 可导入。
    """
    if not is_flashinfer_available():  # 如果FlashInfer不可用
        raise RuntimeError(
            "flashinfer_mxfp4 runner backend requires flashinfer to be installed."
            "flashinfer_mxfp4 运行器后端需要安装 flashinfer。"
        )
    from flashinfer.fused_moe import cutlass_fused_moe  # 导入cutlass融合MoE函数
    from flashinfer.fused_moe.core import ActivationType  # 导入激活类型枚举

    return cutlass_fused_moe, ActivationType  # 返回融合MoE函数和激活类型


@register_fused_func("none", "flashinfer_mxfp4")  # 注册融合函数：none a2a后端 + flashinfer_mxfp4运行器
def fused_experts_none_to_flashinfer_mxfp4(
    dispatch_output: "StandardDispatchOutput",
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "StandardCombineInput":
    """SM90 W4A16 MXFP4 fused expert forward pass.
    SM90 W4A16 MXFP4 融合专家前向传播。

    Mirrors the legacy ``Mxfp4MoEMethod._apply_sm90_cutlass`` and DSv4's
    ``Mxfp4FlashinferCutlassMoEMethod.apply`` exactly; difference vs those is
    that all per-layer state arrives via ``quant_info`` rather than via the
    layer module, so this function is layer-agnostic.
    与旧版 ``Mxfp4MoEMethod._apply_sm90_cutlass`` 和 DSv4 的
    ``Mxfp4FlashinferCutlassMoEMethod.apply`` 完全一致；不同之处在于
    所有每层状态通过 ``quant_info`` 传入而非通过层模块，
    因此此函数与具体层无关。
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准合并输入
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出检查器

    assert isinstance(
        quant_info, FlashInferMxfp4CutlassMoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"  # 断言量化信息类型正确

    flashinfer_cutlass_fused_moe, ActivationType = _flashinfer_cutlass_fused_moe()  # 惰性导入FlashInfer函数

    x = dispatch_output.hidden_states  # 获取隐藏状态
    topk_output = dispatch_output.topk_output  # 获取TopK输出

    # Under ``--moe-runner-backend flashinfer_mxfp4`` topk may be in bypassed
    # form (the SM100 trtllm-gen path does routing internally). The cutlass
    # SM90 path needs explicit topk_ids / topk_weights; materialize here.
    # 在 ``--moe-runner-backend flashinfer_mxfp4`` 下，topk 可能是绕过
    # 形式（SM100 trtllm-gen 路径在内部进行路由）。cutlass
    # SM90 路径需要显式的 topk_ids / topk_weights；在此处具体化。
    if TopKOutputChecker.format_is_bypassed(topk_output):  # 如果TopK输出格式为绕过格式
        topk_output = topk_output.to_standard()  # 转换为标准格式
    topk_ids = topk_output.topk_ids  # 获取TopK ID
    topk_weights = topk_output.topk_weights  # 获取TopK权重

    # GPT-OSS: pad input hidden dim up to the loaded weight width. DSv4
    # leaves padded_hidden as None (or equal to origin_hidden), no pad.
    # GPT-OSS：将输入隐藏维度填充到加载权重宽度。DSv4
    # 将 padded_hidden 设为 None（或等于 origin_hidden），无填充。
    origin_hidden = x.shape[-1]  # 获取原始隐藏维度
    padded_hidden = quant_info.padded_hidden  # 获取填充后隐藏维度
    do_pad = padded_hidden is not None and padded_hidden != origin_hidden  # 判断是否需要填充
    if do_pad:  # 如果需要填充
        x = torch.nn.functional.pad(
            x,
            (0, padded_hidden - origin_hidden),
            mode="constant",
            value=0.0,
        )  # 对输入进行零填充

    out_hidden = padded_hidden if do_pad else origin_hidden  # 确定输出隐藏维度
    output_dtype = torch.bfloat16  # 输出数据类型为bfloat16
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):  # 在对称内存上下文中分配输出
        out = torch.empty(x.shape[0], out_hidden, dtype=output_dtype, device=x.device)  # 分配输出张量

    flashinfer_cutlass_fused_moe(
        input=x,  # 输入张量
        token_selected_experts=topk_ids.to(torch.int),  # 令牌选择的专家ID
        token_final_scales=topk_weights,  # 令牌最终缩放权重
        fc1_expert_weights=quant_info.w13_weight,  # FC1专家权重
        fc2_expert_weights=quant_info.w2_weight,  # FC2专家权重
        output_dtype=output_dtype,  # 输出数据类型
        quant_scales=[
            quant_info.w13_weight_scale.view(torch.int32),  # W13权重缩放因子（转为int32视图）
            quant_info.w2_weight_scale.view(torch.int32),  # W2权重缩放因子（转为int32视图）
        ],
        fc1_expert_biases=quant_info.w13_bias,  # FC1专家偏置
        fc2_expert_biases=quant_info.w2_bias,  # FC2专家偏置
        swiglu_alpha=quant_info.swiglu_alpha,  # SwiGLU alpha参数
        swiglu_beta=quant_info.swiglu_beta,  # SwiGLU beta参数
        swiglu_limit=quant_info.swiglu_limit,  # SwiGLU limit参数
        tp_size=quant_info.moe_tp_size,  # 张量并行大小
        tp_rank=quant_info.moe_tp_rank,  # 张量并行rank
        ep_size=quant_info.moe_ep_size,  # 专家并行大小
        ep_rank=quant_info.moe_ep_rank,  # 专家并行rank
        use_w4_group_scaling=True,  # 启用W4分组缩放
        activation_type=ActivationType.Swiglu,  # 激活类型为SwiGLU
        tune_max_num_tokens=next_power_of_2(x.shape[0]),  # 自适应调整最大令牌数
        output=out,  # 输出张量
    )  # 调用FlashInfer cutlass融合MoE

    if do_pad:  # 如果进行了填充
        out = out[:, :origin_hidden].contiguous()  # 裁剪输出到原始隐藏维度

    return StandardCombineInput(hidden_states=out)  # 返回标准合并输入
