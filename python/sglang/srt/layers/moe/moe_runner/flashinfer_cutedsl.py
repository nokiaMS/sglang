# FlashInfer CuteDSL FP4 MoE 运行器模块
# 本模块实现了基于 FlashInfer CuteDSL 的 FP4 量化 MoE（混合专家）前向计算。
# 支持 v2 标准路径（使用 CuteDslMoEWrapper）和 v1 DeepEP 低延迟路径。
# 包含权重/缩放因子准备工具函数、量化信息数据类和融合专家函数。

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Any, Optional  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.moe.moe_runner.base import (  # 从MoE运行器基类导入
    MoeQuantInfo,  # MoE量化信息基类
    MoeRunnerConfig,  # MoE运行器配置类
    register_fused_func,  # 融合函数注册装饰器
)
from sglang.srt.utils.common import log_info_on_rank0, print_warning_once  # 导入日志工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs  # 下行GEMM重叠参数
    from sglang.srt.layers.moe.token_dispatcher import (  # 从令牌分发器导入
        DeepEPLLCombineInput,  # DeepEP低延迟合并输入
        DeepEPLLDispatchOutput,  # DeepEP低延迟分发输出
        StandardCombineInput,  # 标准合并输入
        StandardDispatchOutput,  # 标准分发输出
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (  # 从FlashInfer分发器导入
        FlashinferCombineInput,  # FlashInfer合并输入
        FlashinferDispatchOutput,  # FlashInfer分发输出
    )

logger = logging.getLogger(__name__)  # 创建模块级日志记录器

_FP4_SF_VEC_SIZE = 16  # FP4缩放因子向量大小
_cutedsl_logged_scalarize: set = set()  # 记录已输出过标量化日志的名称集合


# ---------------------------------------------------------------------------
# Weight / scale preparation utilities (called from modelopt_quant.py during
# process_weights_after_loading and lazy wrapper init)
# 权重/缩放因子准备工具函数（在 modelopt_quant.py 的
# process_weights_after_loading 和惰性包装器初始化期间调用）
# ---------------------------------------------------------------------------


def interleave_w13_halves(
    tensor: torch.Tensor, group_size: int = 64, dim: int = 1
) -> torch.Tensor:
    """Interleave the two logical W13 halves for CuteDSL's SwiGLU GEMM1 layout.
    交错 W13 的两个逻辑半部分，用于 CuteDSL 的 SwiGLU GEMM1 布局。

    The caller is responsible for loading W13 in the expected two-half order.
    调用者负责以预期的两半顺序加载 W13。
    This helper only rewrites the first and second halves into alternating
    `group_size` chunks along `dim`.
    此辅助函数仅将前半部分和后半部分重写为沿 `dim` 维度交替的
    `group_size` 大小的块。
    """
    if tensor.shape[dim] % 2 != 0:  # 检查交错维度大小是否为偶数
        raise ValueError(
            "Expected even size on interleave dimension for W13 half split."
            "W13半分割要求交错维度大小为偶数。"
        )
    split = tensor.shape[dim] // 2  # 计算分割点
    if split % group_size != 0:  # 检查分割后大小是否能被group_size整除
        raise ValueError(
            f"Expected split dim divisible by group_size={group_size}, got {split}."
            f"期望分割维度能被 group_size={group_size} 整除，得到 {split}。"
        )
    first_half = tensor.narrow(dim, 0, split)  # 取前半部分
    second_half = tensor.narrow(dim, split, split)  # 取后半部分
    first_half_groups = first_half.split(group_size, dim=dim)  # 将前半部分按group_size分组
    second_half_groups = second_half.split(group_size, dim=dim)  # 将后半部分按group_size分组
    interleaved = [
        item for pair in zip(first_half_groups, second_half_groups) for item in pair
    ]  # 交错排列前后半部分的分组
    return torch.cat(interleaved, dim=dim)  # 沿指定维度拼接交错结果


def cutedsl_quant_scale_to_scalar(
    quant_scale: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Reduce per-expert quant-domain scale vector to a single scalar.
    将每个专家的量化域缩放向量归约为单个标量。

    The quant domain is the reciprocal of the raw checkpoint scale:
    量化域是原始检查点缩放因子的倒数：
        quant_scale = 1 / raw_scale

    Returns min(quant_scale) = 1/max(raw_scale), which is the TRTLLM CuteDSL
    convention for global scalar activation scales (see TRTLLM quantization.py
    lines 2137-2141: fc2_input_scale = tmp_fc2_input_scale.max().reciprocal()).
    返回 min(quant_scale) = 1/max(raw_scale)，这是 TRTLLM CuteDSL
    全局标量激活缩放因子的约定（参见 TRTLLM quantization.py
    第2137-2141行: fc2_input_scale = tmp_fc2_input_scale.max().reciprocal()）。

    If quant_scale is already scalar (numel==1), returns it unchanged.
    如果 quant_scale 已经是标量（numel==1），则原样返回。
    """
    quant_scale = quant_scale.to(torch.float32)  # 转换为float32精度
    if quant_scale.numel() == 0:  # 如果缩放因子为空
        print_warning_once(
            f"CuteDSL got empty {name}; using 1.0 fallback.",
        )  # 打印警告信息
        return torch.ones(1, device=quant_scale.device, dtype=torch.float32)  # 返回1.0作为回退值
    if quant_scale.numel() == 1:  # 如果缩放因子已经是标量
        return quant_scale.reshape(1)  # 重塑为1维张量并返回
    if name not in _cutedsl_logged_scalarize:  # 如果此名称尚未输出过日志
        log_info_on_rank0(
            logger,
            f"CuteDSL: reducing per-expert {name} to scalar via "
            "min(quant_scale) = 1/max(raw_scale), matching TRTLLM convention.",
        )  # 在rank0上记录标量化日志
        _cutedsl_logged_scalarize.add(name)  # 记录已输出日志的名称
    return quant_scale.min().reshape(1)  # 返回最小值作为标量缩放因子


def resolve_cutedsl_standard_scales(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve standard-path CuteDSL scales (baseline: scalar fc2/w13 input scales).
    解析标准路径 CuteDSL 缩放因子（基线：标量 fc2/w13 输入缩放因子）。

    Returns (w1_alpha, fc2_input_scale, w2_alpha, used_input_scale).
    返回 (w1_alpha, fc2_input_scale, w2_alpha, used_input_scale)。
    used_input_scale is the scalarized w13 input scale for FP4 quantize and GEMM1.
    used_input_scale 是用于 FP4 量化和 GEMM1 的标量化 w13 输入缩放因子。
    """

    def _to_fp32_tensor(x: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        """将输入转换为与参考张量相同设备的float32张量"""
        if not isinstance(x, torch.Tensor):  # 如果输入不是张量
            x = torch.tensor(x, device=ref.device)  # 创建新张量
        return x.to(device=ref.device, dtype=torch.float32)  # 转换为float32并移至参考设备

    def _align_scale_to_alpha(
        scale: torch.Tensor, alpha: torch.Tensor, scale_name: str
    ) -> torch.Tensor:
        """将缩放因子与alpha对齐，处理维度和专家并行差异"""
        scale = scale.to(device=alpha.device, dtype=torch.float32)  # 转换缩放因子设备和精度
        alpha = alpha.to(torch.float32)  # 转换alpha为float32
        if scale.ndim == 0:  # 如果缩放因子是标量
            return scale  # 直接返回
        # Gated weight scales may be (num_experts, 2) with separate gate/up
        # columns. Collapse to 1D by taking the first column (gate == up for
        # well-formed checkpoints; mismatch is warned in process_weights_after_loading).
        # 门控权重缩放因子可能是 (num_experts, 2)，包含分离的 gate/up
        # 列。通过取第一列折叠为1维（对于格式正确的检查点，gate == up；
        # 不匹配在 process_weights_after_loading 中发出警告）。
        if scale.ndim == 2 and scale.shape[1] <= 2:  # 如果是2维且第二维<=2
            scale = scale[:, 0]  # 取第一列（gate列）
        if scale.numel() == alpha.numel():  # 如果元素数量匹配
            return scale  # 直接返回
        if scale.numel() == 1:  # 如果缩放因子是单元素
            return scale.reshape(())  # 重塑为标量

        # Some EP setups may carry global-per-expert scale vectors while alphas are
        # local-per-expert vectors. Slice to this rank's local expert range.
        # 某些EP配置可能包含全局每专家缩放向量，而alpha是
        # 本地每专家向量。切片到当前rank的本地专家范围。
        num_local_experts = getattr(layer, "num_local_experts", None)  # 获取本地专家数
        num_experts = getattr(layer, "num_experts", None)  # 获取总专家数
        moe_ep_rank = getattr(layer, "moe_ep_rank", 0)  # 获取EP rank
        if (
            num_local_experts is not None
            and num_experts is not None
            and scale.numel() == num_experts
            and alpha.numel() == num_local_experts
        ):  # 如果需要对齐EP范围
            start = moe_ep_rank * num_local_experts  # 计算起始索引
            end = start + num_local_experts  # 计算结束索引
            return scale[start:end]  # 返回本地专家范围的切片

        raise ValueError(
            f"Unable to align {scale_name} shape={tuple(scale.shape)} "
            f"to alpha shape={tuple(alpha.shape)} for CuteDSL standard scale resolution."
            f"无法将 {scale_name} 形状={tuple(scale.shape)} "
            f"与 alpha 形状={tuple(alpha.shape)} 对齐，用于 CuteDSL 标准缩放解析。"
        )

    def _resolve_w1_alpha_from_scalar_input_scale(
        used_input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve GEMM1 alpha consistent with scalarized activation quant scale.
        解析与标量化激活量化缩放因子一致的 GEMM1 alpha。

        CuteDSL pre-quantizes x with a single scalar (used_input_scale), but
        g1_alphas was derived with per-expert activation scales:
        CuteDSL 使用单个标量（used_input_scale）预量化 x，但
        g1_alphas 是从每专家激活缩放因子推导的：
            g1_alphas[e] = (1/w13_isq[e]) * w13_ws2[e]
        Correct alpha for scalar quantization:
        标量量化的正确 alpha：
            w1_alpha[e] = w13_ws2[e] / used_input_scale
                         = g1_alphas[e] * w13_isq[e] / used_input_scale
        When w13_isq is already scalar, this is a no-op (ratio = 1).
        当 w13_isq 已经是标量时，此操作为空操作（比率 = 1）。
        """
        eps = 1e-12  # 防止除零的小值
        scalar = torch.clamp(used_input_scale.to(torch.float32).reshape(()), min=eps)  # 限制标量下界

        if hasattr(layer, "w13_weight_scale_2"):  # 如果有权重缩放因子2
            w13_weight_scale_2 = _align_scale_to_alpha(
                layer.w13_weight_scale_2, layer.g1_alphas, "w13_weight_scale_2"
            )  # 对齐w13权重缩放因子2
            return w13_weight_scale_2.to(torch.float32) / scalar  # 返回权重缩放因子2除以标量

        w13_isq = _align_scale_to_alpha(
            layer.w13_input_scale_quant, layer.g1_alphas, "w13_input_scale_quant"
        )  # 对齐w13输入缩放量化因子
        w13_isq = torch.clamp(_to_fp32_tensor(w13_isq, layer.g1_alphas), min=eps)  # 限制w13_isq下界
        return (layer.g1_alphas.to(torch.float32) * w13_isq / scalar).to(torch.float32)  # 计算并返回w1_alpha

    def _resolve_w2_alpha_from_scalar_fc2_input_scale(
        fc2_input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve GEMM2 alpha consistent with scalarized FC2 input scale.
        解析与标量化 FC2 输入缩放因子一致的 GEMM2 alpha。

        CuteDSL standard path uses a scalar global scale for GEMM1 FP4 output
        quantization (`fc2_input_scale`). GEMM2 alpha must use the same scalar
        convention: alpha2 = w2_weight_scale_2 / fc2_input_scale.
        CuteDSL 标准路径使用标量全局缩放因子对 GEMM1 FP4 输出
        量化（`fc2_input_scale`）。GEMM2 alpha 必须使用相同的标量
        约定：alpha2 = w2_weight_scale_2 / fc2_input_scale。
        """
        eps = 1e-12  # 防止除零的小值
        fc2_input_scale = fc2_input_scale.to(torch.float32)  # 转换为float32
        fc2_scalar = torch.clamp(fc2_input_scale.reshape(-1)[:1], min=eps).reshape(())  # 取标量并限制下界

        if hasattr(layer, "w2_weight_scale_2"):  # 如果有w2权重缩放因子2
            w2_weight_scale_2 = _align_scale_to_alpha(
                layer.w2_weight_scale_2, layer.g2_alphas, "w2_weight_scale_2"
            )  # 对齐w2权重缩放因子2
            w2_weight_scale_2 = w2_weight_scale_2.to(torch.float32)  # 转换为float32
            return w2_weight_scale_2 / fc2_scalar  # 返回权重缩放因子2除以fc2标量

        w2_q_for_w2 = _align_scale_to_alpha(
            layer.w2_input_scale_quant, layer.g2_alphas, "w2_input_scale_quant"
        )  # 对齐w2输入缩放量化因子
        w2_q_for_w2 = torch.clamp(
            _to_fp32_tensor(w2_q_for_w2, layer.g2_alphas), min=eps
        )  # 限制w2_q下界
        w2_weight_scale_2 = layer.g2_alphas.to(torch.float32) * w2_q_for_w2  # 计算w2权重缩放因子2
        return w2_weight_scale_2 / fc2_scalar  # 返回权重缩放因子2除以fc2标量

    fc2_input_scale = cutedsl_quant_scale_to_scalar(
        layer.w2_input_scale_quant,
        name="w2_input_scale_quant",
    )  # 将w2输入缩放量化因子归约为标量
    w2_alpha = _resolve_w2_alpha_from_scalar_fc2_input_scale(fc2_input_scale)  # 解析w2_alpha
    used_input_scale = cutedsl_quant_scale_to_scalar(
        layer.w13_input_scale_quant,
        name="w13_input_scale_quant",
    )  # 将w13输入缩放量化因子归约为标量
    w1_alpha = _resolve_w1_alpha_from_scalar_input_scale(used_input_scale)  # 解析w1_alpha
    return w1_alpha, fc2_input_scale, w2_alpha, used_input_scale  # 返回四个缩放因子


def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:
    """Lazily create CuteDslMoEWrapper and resolve scales on first forward.
    惰性创建 CuteDslMoEWrapper 并在首次前向时解析缩放因子。

    The wrapper is created lazily (not in __init__ / create_weights) because
    it depends on final weight shapes and EP configuration.  The wrapper's
    CUDA-graph buffers are allocated inside CuteDslMoEWrapper.__init__, which
    typically runs during the autotune dummy forward under inference_mode().
    包装器是惰性创建的（不在 __init__ / create_weights 中），因为它
    依赖于最终的权重形状和EP配置。包装器的CUDA图缓冲区在
    CuteDslMoEWrapper.__init__ 中分配，通常在推理模式下的
    autotune虚拟前向期间运行。
    We wrap the creation in inference_mode(False) so that those pre-allocated
    buffers are normal tensors -- inference tensors cannot be inplace-updated
    during later CUDA graph capture, which runs outside inference_mode.
    我们用 inference_mode(False) 包裹创建过程，以便这些预分配的
    缓冲区是普通张量——推理张量不能在后续的CUDA图捕获期间
    原地更新，而CUDA图捕获在 inference_mode 之外运行。
    """
    if getattr(layer, "_cutedsl_wrapper", None) is not None:  # 如果包装器已存在
        return  # 直接返回

    try:
        from flashinfer import CuteDslMoEWrapper  # 尝试导入CuteDSL MoE包装器
    except ImportError as e:  # 导入失败
        raise ImportError(
            "flashinfer_cutedsl backend requires FlashInfer with CuteDSL support. "
            "Install with: pip install flashinfer"
            "flashinfer_cutedsl 后端需要支持 CuteDSL 的 FlashInfer。"
            "安装命令: pip install flashinfer"
        ) from e

    from sglang.srt.server_args import get_global_server_args  # 导入服务器参数获取函数

    assert layer.intermediate_size_per_partition > 0, (  # 断言中间维度分区大小大于0
        f"CuteDSL MoE: intermediate_size_per_partition must be > 0, "
        f"got {layer.intermediate_size_per_partition}. Check EP/TP configuration."
        f"CuteDSL MoE: intermediate_size_per_partition 必须 > 0，"
        f"得到 {layer.intermediate_size_per_partition}。请检查EP/TP配置。"
    )

    server_args = get_global_server_args()  # 获取全局服务器参数
    use_cuda_graph = not server_args.disable_cuda_graph  # 确定是否使用CUDA图

    # Size the wrapper's CUDA-graph buffers for the largest number of tokens a
    # single forward can route through this layer.
    # 为包装器的CUDA图缓冲区分配单次前向可通过此层路由的最大令牌数。
    dispatcher = getattr(layer, "dispatcher", None)  # 获取分发器
    if hasattr(dispatcher, "max_num_tokens"):  # 如果分发器有max_num_tokens属性
        # A2A path: bounded by the dispatcher's own workspace limit.
        # A2A路径：受分发器自身工作空间限制约束。
        max_num_tokens = dispatcher.max_num_tokens * getattr(dispatcher, "ep_size", 1)  # 计算A2A路径最大令牌数
    else:
        # Standard allgather path: the MoE sees up to dp_size local forwards
        # gathered together, so scale the per-rank forward bound by dp_size.
        # 标准allgather路径：MoE最多看到dp_size个本地前向
        # 汇集在一起，因此按dp_size缩放每rank前向上限。
        max_num_tokens = server_args.dp_size * server_args.cutedsl_moe_max_num_tokens()  # 计算标准路径最大令牌数
    top_k = layer.top_k if layer.top_k is not None else layer.moe_runner_config.top_k  # 获取top_k值
    # inference_mode(False) ensures the wrapper's pre-allocated CUDA-graph
    # buffers are normal tensors.  This call typically happens inside
    # _dummy_run which runs under inference_mode(); inference tensors cannot
    # be inplace-updated during later CUDA graph capture (which runs outside
    # inference_mode), so we must opt out here.
    # inference_mode(False) 确保包装器预分配的CUDA图
    # 缓冲区是普通张量。此调用通常发生在
    # inference_mode() 下运行的 _dummy_run 中；推理张量不能
    # 在后续的CUDA图捕获期间（在 inference_mode 之外运行）
    # 原地更新，因此我们必须在此处退出推理模式。
    with torch.inference_mode(False):  # 临时关闭推理模式
        layer._cutedsl_wrapper = CuteDslMoEWrapper(
            num_experts=layer.num_experts,  # 专家总数
            top_k=top_k,  # top-k值
            hidden_size=layer.hidden_size,  # 隐藏层大小
            intermediate_size=layer.intermediate_size_per_partition,  # 中间层分区大小
            use_cuda_graph=use_cuda_graph,  # 是否使用CUDA图
            max_num_tokens=max_num_tokens,  # 最大令牌数
            num_local_experts=layer.num_local_experts,  # 本地专家数
            local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,  # 本地专家偏移
            output_dtype=layer.moe_runner_config.params_dtype,  # 输出数据类型
            device=str(layer.w13_weight.device),  # 设备
        )  # 创建CuteDslMoEWrapper实例

    w1_alpha, fc2_input_scale, w2_alpha, used_input_scale = (
        resolve_cutedsl_standard_scales(layer)
    )  # 解析标准路径缩放因子
    layer._cutedsl_scales = (w1_alpha, fc2_input_scale, w2_alpha)  # 存储CuteDSL缩放因子
    layer._cutedsl_input_scale = used_input_scale  # 存储输入缩放因子


# ---------------------------------------------------------------------------
# Dataclass + fused function for moe_runner dispatch
# 数据类 + 融合函数，用于 moe_runner 调度
# ---------------------------------------------------------------------------


@dataclass
class CuteDslFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload for FlashInfer CuteDSL FP4 MoE kernels.
    FlashInfer CuteDSL FP4 MoE 内核的量化负载。

    Shared by the two CuteDSL runner entries:
    由两个 CuteDSL 运行器入口共享：

    * "v2" standard path (a2a=``none``/``flashinfer``): consumed by the
      ``@register_fused_func("none", "flashinfer_cutedsl")`` entry, which
      drives ``CuteDslMoEWrapper.run``. Weights are ``[Up, Gate]``
      interleaved with MMA-layout blockscales. ``wrapper`` is set;
      ``w*_scale`` are scalarized.
    * "v2" 标准路径 (a2a=``none``/``flashinfer``)：由
      ``@register_fused_func("none", "flashinfer_cutedsl")`` 入口消费，
      驱动 ``CuteDslMoEWrapper.run``。权重为 ``[Up, Gate]``
      与 MMA 布局块缩放因子交错。``wrapper`` 已设置；
      ``w*_scale`` 已标量化。

    * "v1" DeepEP low-latency path (a2a=``deepep``): consumed by the
      ``@register_fused_func("deepep", "flashinfer_cutedsl")`` entry,
      which drives ``flashinfer_cutedsl_moe_masked``. Weights are
      ``[Gate, Up]`` non-interleaved with swizzled blockscales.
      ``wrapper`` is ``None``; ``w*_scale`` are per-expert.
    * "v1" DeepEP 低延迟路径 (a2a=``deepep``)：由
      ``@register_fused_func("deepep", "flashinfer_cutedsl")`` 入口消费，
      驱动 ``flashinfer_cutedsl_moe_masked``。权重为
      ``[Gate, Up]`` 非交错，使用 swizzled 块缩放因子。
      ``wrapper`` 为 ``None``；``w*_scale`` 为每专家级别。
    """

    # FP4 packed weights (uint8) / FP4打包权重（uint8）
    w13_weight: torch.Tensor  # W13权重（gate+up）
    w2_weight: torch.Tensor  # W2权重（down）

    # Block-scale factors (MMA layout for v2, swizzled for v1) / 块缩放因子（v2为MMA布局，v1为swizzled布局）
    w13_weight_sf: torch.Tensor  # W13权重块缩放因子
    w2_weight_sf: torch.Tensor  # W2权重块缩放因子

    # Per-expert GEMM dequant alphas (scalarized for v2, per-expert for v1) / 每专家GEMM反量化alpha（v2为标量，v1为每专家）
    w1_alpha: torch.Tensor  # GEMM1 alpha
    w2_alpha: torch.Tensor  # GEMM2 alpha

    # Activation quant scales (1 / raw_input_scale).
    # 激活量化缩放因子（1 / raw_input_scale）。
    #   - a1_scale: quantizes hidden_states before GEMM1
    #   - a1_scale: 在 GEMM1 前量化 hidden_states
    #   - a2_scale: quantizes GEMM1 output before GEMM2 (a.k.a. fc2 input)
    #   - a2_scale: 在 GEMM2 前量化 GEMM1 输出（即 fc2 输入）
    a1_scale: torch.Tensor  # 激活1缩放因子
    a2_scale: torch.Tensor  # 激活2缩放因子

    # v2 only: lazily-created CuteDslMoEWrapper (``None`` on the v1 path).
    # 仅v2：惰性创建的 CuteDslMoEWrapper（v1路径上为 ``None``）。
    wrapper: Optional[Any] = None  # CuteDSL MoE包装器

    # v1 only: ``True`` when DeepEP pre-quantizes activations to NVFP4.
    # 仅v1：当 DeepEP 预量化激活为 NVFP4 时为 ``True``。
    use_nvfp4_dispatch: bool = False  # 是否使用NVFP4分发

    # v1 only: SBO down-GEMM overlap args.
    # 仅v1：SBO下行GEMM重叠参数。
    down_gemm_overlap_args: Optional["DownGemmOverlapArgs"] = None  # 下行GEMM重叠参数


@register_fused_func("none", "flashinfer_cutedsl")  # 注册融合函数：none a2a后端 + flashinfer_cutedsl运行器
def fused_experts_none_to_flashinfer_cutedsl_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """标准分发到FlashInfer CuteDSL FP4融合专家前向函数"""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准合并输入
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出检查器
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize  # 导入FP4量化工具

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."  # 断言仅支持silu激活
    assert quant_info.wrapper is not None, "CuteDSL v2 path requires CuteDslMoEWrapper."  # 断言包装器必须存在

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_output = dispatch_output.topk_output  # 获取TopK输出
    assert TopKOutputChecker.format_is_standard(topk_output)  # 断言TopK输出格式为标准格式

    topk_ids = topk_output.topk_ids  # 获取TopK ID
    topk_weights = topk_output.topk_weights  # 获取TopK权重
    if topk_ids.dtype != torch.int32:  # 如果TopK ID不是int32类型
        topk_ids = topk_ids.to(torch.int32)  # 转换为int32

    x_fp4, x_sf = fp4_quantize(
        hidden_states,
        quant_info.a1_scale,
        sf_vec_size=_FP4_SF_VEC_SIZE,
        is_sf_swizzled_layout=False,
    )  # 对隐藏状态进行FP4量化

    output = quant_info.wrapper.run(
        x=x_fp4,  # FP4量化的输入
        x_sf=x_sf,  # 输入缩放因子
        token_selected_experts=topk_ids,  # 令牌选择的专家ID
        token_final_scales=topk_weights,  # 令牌最终缩放权重
        w1_weight=quant_info.w13_weight,  # W1权重
        w1_weight_sf=quant_info.w13_weight_sf,  # W1权重缩放因子
        w1_alpha=quant_info.w1_alpha,  # W1 alpha
        fc2_input_scale=quant_info.a2_scale,  # FC2输入缩放因子
        w2_weight=quant_info.w2_weight,  # W2权重
        w2_weight_sf=quant_info.w2_weight_sf,  # W2权重缩放因子
        w2_alpha=quant_info.w2_alpha,  # W2 alpha
    )  # 运行CuteDSL MoE包装器

    return StandardCombineInput(hidden_states=output)  # 返回标准合并输入


@register_fused_func("flashinfer", "flashinfer_cutedsl")  # 注册融合函数：flashinfer a2a后端 + flashinfer_cutedsl运行器
def fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput:
    """CuteDSL fused func for flashinfer alltoall dispatcher.
    FlashInfer alltoall 分发器的 CuteDSL 融合函数。

    Two cases depending on whether the dispatcher did FP4 quantization:
    两种情况取决于分发器是否进行了 FP4 量化：
    - bf16 input (SGLANG_MOE_NVFP4_DISPATCH=0): quantize with cutedsl's scale
    - bf16 输入 (SGLANG_MOE_NVFP4_DISPATCH=0): 使用 cutedsl 的缩放因子量化
    - FP4 input (SGLANG_MOE_NVFP4_DISPATCH=1): pass through (same fp4_quantize params)
    - FP4 输入 (SGLANG_MOE_NVFP4_DISPATCH=1): 直接传递（相同的 fp4_quantize 参数）
    """
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )  # 导入FlashInfer合并输入
    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出检查器
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize  # 导入FP4量化工具

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."  # 断言仅支持silu激活
    assert quant_info.wrapper is not None, "CuteDSL v2 path requires CuteDslMoEWrapper."  # 断言包装器必须存在

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    x_sf = dispatch_output.hidden_states_scale  # 获取隐藏状态缩放因子
    topk_output = dispatch_output.topk_output  # 获取TopK输出
    assert TopKOutputChecker.format_is_standard(topk_output)  # 断言TopK输出格式为标准格式

    topk_ids = topk_output.topk_ids  # 获取TopK ID
    topk_weights = topk_output.topk_weights  # 获取TopK权重
    if topk_ids.dtype != torch.int32:  # 如果TopK ID不是int32类型
        topk_ids = topk_ids.to(torch.int32)  # 转换为int32

    if x_sf is not None:  # 如果隐藏状态缩放因子存在
        # NVFP4 dispatch, inputs are already quantized.
        # NVFP4 分发，输入已量化。
        x_fp4 = hidden_states  # 直接使用已量化的输入
    else:
        x_fp4, x_sf = fp4_quantize(
            hidden_states,
            quant_info.a1_scale,
            sf_vec_size=_FP4_SF_VEC_SIZE,
            is_sf_swizzled_layout=False,
        )  # 对隐藏状态进行FP4量化

    output = quant_info.wrapper.run(
        x=x_fp4,  # FP4量化的输入
        x_sf=x_sf,  # 输入缩放因子
        token_selected_experts=topk_ids,  # 令牌选择的专家ID
        token_final_scales=topk_weights,  # 令牌最终缩放权重
        w1_weight=quant_info.w13_weight,  # W1权重
        w1_weight_sf=quant_info.w13_weight_sf,  # W1权重缩放因子
        w1_alpha=quant_info.w1_alpha,  # W1 alpha
        fc2_input_scale=quant_info.a2_scale,  # FC2输入缩放因子
        w2_weight=quant_info.w2_weight,  # W2权重
        w2_weight_sf=quant_info.w2_weight_sf,  # W2权重缩放因子
        w2_alpha=quant_info.w2_alpha,  # W2 alpha
    )  # 运行CuteDSL MoE包装器

    # Note: output contains routed expert results; shared_expert is handled separately
    # 注意：输出包含路由专家结果；共享专家单独处理

    # Write into pre-allocated workspace buffer if available
    # 如果可用，写入预分配的工作空间缓冲区
    if dispatch_output.moe_output is not None:  # 如果预分配输出缓冲区存在
        dispatch_output.moe_output.copy_(output)  # 将结果拷贝到预分配缓冲区
        output = dispatch_output.moe_output  # 使用预分配缓冲区作为输出

    return FlashinferCombineInput(hidden_states=output)  # 返回FlashInfer合并输入


@register_fused_func("deepep", "flashinfer_cutedsl")  # 注册融合函数：deepep a2a后端 + flashinfer_cutedsl运行器
def fused_experts_deepep_to_flashinfer_cutedsl_fp4(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> DeepEPLLCombineInput:
    """DeepEP低延迟分发到FlashInfer CuteDSL FP4融合专家前向函数"""
    from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
    )  # 导入CuteDSL掩码MoE函数
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput  # 导入DeepEP低延迟合并输入

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."  # 断言仅支持silu激活
    assert (
        not runner_config.apply_router_weight_on_input
    ), "apply_router_weight_on_input is not supported for Flashinfer"  # 断言不支持在输入上应用路由权重

    hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output  # 解包分发输出

    # flashinfer_cutedsl_moe_masked reinterprets scales as float8_e4m3fn.
    # Same-dtype .view is a no-op; only wider dtypes (e.g. int32-packed
    # UE8M0) need stride(-1)==1.
    # flashinfer_cutedsl_moe_masked 将缩放因子重新解释为 float8_e4m3fn。
    # 相同数据类型的 .view 是空操作；只有更宽数据类型（如 int32-packed
    # UE8M0）需要 stride(-1)==1。
    if (
        quant_info.use_nvfp4_dispatch
        and hidden_states_scale is not None
        and hidden_states_scale.element_size() != 1
        and hidden_states_scale.stride(-1) != 1
    ):  # 如果使用NVFP4分发且缩放因子步长不满足要求
        raise AssertionError(
            f"NVFP4 dispatch scale has stride(-1)={hidden_states_scale.stride(-1)}, "
            f"dtype={hidden_states_scale.dtype}; .view(float8_e4m3fn) requires stride(-1)==1. "
            "Try SGLANG_MOE_NVFP4_DISPATCH=0 or check DeepEP version."
            f"NVFP4 分发缩放因子步长 stride(-1)={hidden_states_scale.stride(-1)}，"
            f"dtype={hidden_states_scale.dtype}；.view(float8_e4m3fn) 需要 stride(-1)==1。"
            "请尝试 SGLANG_MOE_NVFP4_DISPATCH=0 或检查 DeepEP 版本。"
        )

    overlap = quant_info.down_gemm_overlap_args  # 获取下行GEMM重叠参数
    output = flashinfer_cutedsl_moe_masked(
        hidden_states=(hidden_states, hidden_states_scale),  # 隐藏状态及其缩放因子
        input_global_scale=(
            None if quant_info.use_nvfp4_dispatch else quant_info.a1_scale
        ),  # 输入全局缩放因子（NVFP4分发时不使用）
        w1=quant_info.w13_weight,  # W1权重
        w1_blockscale=quant_info.w13_weight_sf,  # W1块缩放因子
        w1_alpha=quant_info.w1_alpha,  # W1 alpha
        w2=quant_info.w2_weight,  # W2权重
        a2_global_scale=quant_info.a2_scale,  # A2全局缩放因子
        w2_blockscale=quant_info.w2_weight_sf,  # W2块缩放因子
        w2_alpha=quant_info.w2_alpha,  # W2 alpha
        masked_m=masked_m,  # 掩码矩阵
        **(
            dict(
                down_sm_count=overlap.num_sms,  # 下行GEMM SM数量
                down_signals=overlap.signal,  # 下行GEMM信号
                down_start_event=overlap.start_event,  # 下行GEMM启动事件
            )
            if overlap is not None  # 如果有重叠参数
            else {}  # 否则传递空字典
        ),
    )  # 运行CuteDSL掩码MoE

    return DeepEPLLCombineInput(
        hidden_states=output,  # 输出隐藏状态
        topk_ids=dispatch_output.topk_ids,  # TopK ID
        topk_weights=dispatch_output.topk_weights,  # TopK权重
    )  # 返回DeepEP低延迟合并输入
