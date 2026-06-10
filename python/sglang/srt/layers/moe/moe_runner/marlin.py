# Marlin 量化 MoE 运行器模块
# 本模块实现了基于 Marlin 量化格式的 MoE（混合专家）前向计算。
# 支持 GPTQ 和 AWQ 量化方法，包含量化信息数据类和融合专家函数。

from __future__ import annotations  # 启用延迟类型注解求值

from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Optional  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架

from sglang.srt.layers.moe.moe_runner.base import (  # 从MoE运行器基类导入
    MoeQuantInfo,  # MoE量化信息基类
    MoeRunnerConfig,  # MoE运行器配置类
    RunnerInput,  # 运行器输入基类
    RunnerOutput,  # 运行器输出基类
    register_fused_func,  # 融合函数注册装饰器
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # 导入MoE运行器后端枚举

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher import (  # 从令牌分发器导入
        StandardCombineInput,  # 标准合并输入
        StandardDispatchOutput,  # 标准分发输出
    )

MARLIN_MOE_WORKSPACE: Optional[torch.Tensor] = None  # Marlin MoE工作空间全局变量


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core.
    传递给 Marlin 运行器核心的输入束。
    """

    hidden_states: torch.Tensor  # 隐藏状态张量
    topk_weights: torch.Tensor  # Top-K权重张量
    topk_ids: torch.Tensor  # Top-K ID张量
    router_logits: torch.Tensor  # 路由器logits张量

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        """返回运行器后端类型为Marlin"""
        return MoeRunnerBackend.MARLIN  # 返回Marlin后端枚举值


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core.
    从 Marlin 运行器核心返回的输出束。
    """

    hidden_states: torch.Tensor  # 输出隐藏状态张量

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        """返回运行器后端类型为Marlin"""
        return MoeRunnerBackend.MARLIN  # 返回Marlin后端枚举值


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend.
    Marlin 后端消费的量化负载。
    """

    w13_qweight: torch.Tensor  # W13量化权重
    w2_qweight: torch.Tensor  # W2量化权重
    w13_scales: torch.Tensor  # W13缩放因子
    w2_scales: torch.Tensor  # W2缩放因子
    w13_g_idx_sort_indices: Optional[torch.Tensor]  # W13组索引排序索引
    w2_g_idx_sort_indices: Optional[torch.Tensor]  # W2组索引排序索引
    weight_bits: int  # 权重位数

    # GPTQ specific (Optional) / GPTQ特定字段（可选）
    w13_g_idx: Optional[torch.Tensor] = None  # W13组索引
    w2_g_idx: Optional[torch.Tensor] = None  # W2组索引
    is_k_full: bool = True  # K是否完整

    # AWQ specific (Optional) / AWQ特定字段（可选）
    w13_qzeros: Optional[torch.Tensor] = None  # W13量化零点
    w2_qzeros: Optional[torch.Tensor] = None  # W2量化零点

    # Optional / 可选字段
    expert_map: Optional[torch.Tensor] = None  # 专家映射


@register_fused_func("none", "marlin")  # 注册融合函数：none a2a后端 + marlin运行器
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """标准分发到Marlin量化MoE融合专家前向函数"""
    global MARLIN_MOE_WORKSPACE  # 声明使用全局工作空间变量
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe  # 导入Marlin融合MoE函数
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准合并输入
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace  # 导入Marlin工作空间创建函数

    hidden_states = dispatch_output.hidden_states  # 获取隐藏状态
    topk_output = dispatch_output.topk_output  # 获取TopK输出

    assert runner_config.activation == "silu", "Only SiLU activation is supported."  # 断言仅支持SiLU激活

    if (
        MARLIN_MOE_WORKSPACE is None
        or MARLIN_MOE_WORKSPACE.device != hidden_states.device
    ):  # 如果工作空间未创建或设备不匹配
        MARLIN_MOE_WORKSPACE = marlin_make_workspace(
            hidden_states.device, max_blocks_per_sm=4
        )  # 创建Marlin工作空间

    marlin_hidden_states = hidden_states  # 初始化Marlin隐藏状态
    # Avoid aliasing the MoE input buffer until Marlin output semantics are
    # fully validated across shared-expert and overlap paths.
    # 在 Marlin 输出语义在共享专家和重叠路径上完全验证之前，
    # 避免对 MoE 输入缓冲区进行别名操作。
    marlin_inplace = False  # 禁用原地操作
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):  # 如果是MXFP4(E8M0) Marlin且输入为fp16
        # MXFP4(E8M0) Marlin kernels are only numerically valid on the bf16
        # activation path. The fp16 + E8M0 path is intentionally not generated
        # in sgl-kernel, so upcast activations here and cast the result back.
        # MXFP4(E8M0) Marlin 内核仅在 bf16 激活路径上数值有效。
        # fp16 + E8M0 路径在 sgl-kernel 中有意不生成，
        # 因此在此处将激活上转换并将结果转换回来。
        marlin_hidden_states = hidden_states.to(torch.bfloat16)  # 上转换为bfloat16
        marlin_inplace = False  # 禁用原地操作

    output = fused_marlin_moe(
        hidden_states=marlin_hidden_states,  # Marlin隐藏状态
        w1=quant_info.w13_qweight,  # W1量化权重
        w2=quant_info.w2_qweight,  # W2量化权重
        w1_scale=quant_info.w13_scales,  # W1缩放因子
        w2_scale=quant_info.w2_scales,  # W2缩放因子
        gating_output=topk_output.router_logits,  # 门控输出（路由logits）
        topk_weights=topk_output.topk_weights,  # Top-K权重
        topk_ids=topk_output.topk_ids,  # Top-K ID
        expert_map=quant_info.expert_map,  # 专家映射
        g_idx1=quant_info.w13_g_idx,  # W1组索引
        g_idx2=quant_info.w2_g_idx,  # W2组索引
        sort_indices1=quant_info.w13_g_idx_sort_indices,  # W1组索引排序索引
        sort_indices2=quant_info.w2_g_idx_sort_indices,  # W2组索引排序索引
        w1_zeros=quant_info.w13_qzeros,  # W1量化零点
        w2_zeros=quant_info.w2_qzeros,  # W2量化零点
        workspace=MARLIN_MOE_WORKSPACE,  # Marlin工作空间
        num_bits=quant_info.weight_bits,  # 权重位数
        is_k_full=quant_info.is_k_full,  # K是否完整
        inplace=marlin_inplace,  # 是否原地操作
        routed_scaling_factor=runner_config.routed_scaling_factor,  # 路由缩放因子
        clamp_limit=runner_config.swiglu_limit,  # 钳位限制值
    ).to(hidden_states.dtype)  # 调用Marlin融合MoE并转换回原始数据类型

    return StandardCombineInput(
        hidden_states=output,
    )  # 返回标准合并输入
