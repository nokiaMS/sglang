# SPDX-License-Identifier: Apache-2.0  # 许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py  # 改编自 vLLM 项目的融合 MoE 实现

# 融合 MoE 内核模块 - 实现基于 Triton 的混合专家（MoE）层计算，支持多种量化格式、激活函数及硬件平台

"""Fused MoE kernel."""  # 英文文档字符串：融合 MoE 内核

from __future__ import annotations  # 启用延迟注解评估，支持前向引用类型

import functools  # 函数工具模块，提供 lru_cache 等装饰器
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 类型提示工具

import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # PyTorch 函数式接口
import triton.language as tl  # Triton 编程语言

from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # MoE 运行器配置
from sglang.srt.layers.moe.utils import get_moe_padding_size  # 获取 MoE 填充大小
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.utils import (  # 通用工具函数
    cpu_has_amx_support,  # 检查 CPU 是否支持 AMX 指令集
    get_bool_env_var,  # 获取布尔类型环境变量
    is_cpu,  # 判断是否为 CPU 平台
    is_cuda,  # 判断是否为 CUDA 平台
    is_hip,  # 判断是否为 HIP 平台
    is_musa,  # 判断是否为 MUSA 平台
    is_xpu,  # 判断是否为 XPU 平台
    use_intel_xpu_backend,  # 判断是否使用 Intel XPU 后端
)
from sglang.srt.utils.custom_op import register_custom_op  # 自定义算子注册装饰器

from .fused_moe_triton_config import get_config_dtype_str, try_get_optimal_moe_config  # MoE Triton 配置函数
from .fused_moe_triton_kernels import (  # MoE Triton 内核函数
    act_and_mul_triton,  # Triton 激活与乘法内核
    invoke_fused_moe_kernel,  # 调用融合 MoE 内核
    moe_sum_reduce_triton,  # Triton MoE 求和归约内核
    support_tensor_descriptor,  # 检查是否支持张量描述符
)
from .moe_align_block_size import moe_align_block_size  # MoE 块大小对齐函数

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.topk import StandardTopKOutput  # 标准 TopK 输出类型

_is_hip = is_hip()  # 是否为 HIP 平台
_is_cuda = is_cuda()  # 是否为 CUDA 平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU 是否支持 AMX
_is_cpu = is_cpu()  # 是否为 CPU 平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用 AITER（仅 HIP）
_is_xpu = is_xpu()  # 是否为 XPU 平台
_use_sgl_xpu = use_intel_xpu_backend()  # 是否使用 Intel XPU 后端
_is_musa = is_musa()  # 是否为 MUSA 平台


if _is_cuda:  # CUDA 平台
    from sgl_kernel import moe_sum_reduce  # 从 sgl_kernel 导入 MoE 求和归约

    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul  # 导入 GELU/SiLU 与乘法融合内核
elif _is_cpu and _is_cpu_amx_available:  # CPU 平台且支持 AMX
    pass  # 暂无额外导入
elif _is_hip:  # HIP 平台
    from sgl_kernel import gelu_and_mul, silu_and_mul  # 从 sgl_kernel 导入激活函数

    if _use_aiter:  # 使用 AITER 时
        try:
            from aiter import moe_sum  # 从 aiter 导入 MoE 求和
        except ImportError:
            raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")  # 缺少 aiter 包时报错
    # Note: vllm_ops is not needed for HIP when _use_aiter=False
    # 注意：当 _use_aiter=False 时 HIP 不需要 vllm_ops
    # because the code uses moe_sum_reduce_triton as fallback (line 619)
    # 因为代码使用 moe_sum_reduce_triton 作为回退（第619行）
elif _is_xpu:  # XPU 平台
    from sgl_kernel import moe_sum_reduce, silu_and_mul  # 从 sgl_kernel 导入 MoE 求和归约和 SiLU 乘法
elif _is_musa:  # MUSA 平台
    from sgl_kernel import moe_sum_reduce  # 从 sgl_kernel 导入 MoE 求和归约

    _silu_and_mul_musa = torch.nn.SwishGLU()  # MUSA 平台使用 PyTorch 的 SwishGLU 实现

# Try to import vllm_ops for non-CUDA/HIP/XPU platforms
# 尝试为非 CUDA/HIP/XPU 平台导入 vllm_ops
_has_vllm_ops = False  # 是否有 vllm_ops 可用
if not _is_cuda and not _is_hip and not _is_xpu:  # 非 CUDA/HIP/XPU 平台
    try:
        from vllm import _custom_ops as vllm_ops  # 从 vllm 导入自定义算子

        _has_vllm_ops = True  # vllm_ops 可用
    except ImportError:
        # Fallback: vllm not available, will use native PyTorch implementations
        # 回退：vllm 不可用，将使用原生 PyTorch 实现
        _has_vllm_ops = False  # vllm_ops 不可用

padding_size = get_moe_padding_size(_use_aiter)  # 获取 MoE 填充大小


@register_custom_op(mutates_args=["hidden_states"])  # 注册自定义算子，标记会修改 hidden_states
def inplace_fused_experts(  # 原地融合专家计算函数，结果直接写入 hidden_states
    hidden_states: torch.Tensor,  # 隐藏状态张量（会被原地修改）
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_weights: torch.Tensor,  # Top-K 路由权重
    topk_ids: torch.Tensor,  # Top-K 专家ID
    b1: Optional[torch.Tensor] = None,  # w1 偏置（可选）
    b2: Optional[torch.Tensor] = None,  # w2 偏置（可选）
    activation: str = "silu",  # 激活函数类型，默认 silu
    is_gated: bool = True,  # 是否为门控 MoE，默认 True
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: bool = False,  # 是否使用 INT8 W8A8 量化
    use_int8_w8a16: bool = False,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: bool = False,  # 是否使用 INT4 W4A16 量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    w1_scale: Optional[torch.Tensor] = None,  # w1 权重缩放因子（可选）
    w2_scale: Optional[torch.Tensor] = None,  # w2 权重缩放因子（可选）
    w1_zp: Optional[torch.Tensor] = None,  # w1 零点（可选）
    w2_zp: Optional[torch.Tensor] = None,  # w2 零点（可选）
    a1_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w1（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w2（可选）
    block_shape: Optional[List[int]] = None,  # 块状量化的块形状（可选）
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子（可选）
    gemm1_alpha: Optional[float] = None,  # GEMM1 alpha 参数（可选）
    gemm1_limit: Optional[float] = None,  # GEMM1 钳位限制（可选）
    filter_expert: bool = True,  # 是否过滤专家，默认 True
    swiglu_limit: Optional[float] = None,  # SwiGLU 钳位限制（可选）
) -> None:  # 无返回值，结果直接写入 hidden_states
    fused_experts_impl(  # 调用融合专家实现函数
        hidden_states,  # 隐藏状态
        w1,  # w1 权重
        w2,  # w2 权重
        topk_weights,  # Top-K 权重
        topk_ids,  # Top-K 专家ID
        b1,  # w1 偏置
        b2,  # w2 偏置
        True,  # inplace=True 原地操作
        activation,  # 激活函数
        is_gated,  # 是否门控
        apply_router_weight_on_input,  # 是否在输入上应用路由权重
        use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant,  # 逐通道量化标志
        w1_scale,  # w1 缩放因子
        w2_scale,  # w2 缩放因子
        w1_zp,  # w1 零点
        w2_zp,  # w2 零点
        a1_scale,  # w1 激活缩放
        a2_scale,  # w2 激活缩放
        block_shape,  # 块形状
        False,  # no_combine=False
        routed_scaling_factor,  # 路由缩放因子
        gemm1_alpha,  # GEMM1 alpha
        gemm1_limit,  # GEMM1 限制
        filter_expert,  # 是否过滤专家
        swiglu_limit=swiglu_limit,  # SwiGLU 限制
    )


@register_custom_op(out_shape="hidden_states")  # 注册自定义算子，输出形状与 hidden_states 相同
def outplace_fused_experts(  # 非原地融合专家计算函数，返回新张量
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_weights: torch.Tensor,  # Top-K 路由权重
    topk_ids: torch.Tensor,  # Top-K 专家ID
    b1: Optional[torch.Tensor] = None,  # w1 偏置（可选）
    b2: Optional[torch.Tensor] = None,  # w2 偏置（可选）
    activation: str = "silu",  # 激活函数类型，默认 silu
    is_gated: bool = True,  # 是否为门控 MoE，默认 True
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: bool = False,  # 是否使用 INT8 W8A8 量化
    use_int8_w8a16: bool = False,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: bool = False,  # 是否使用 INT4 W4A16 量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    w1_scale: Optional[torch.Tensor] = None,  # w1 权重缩放因子（可选）
    w2_scale: Optional[torch.Tensor] = None,  # w2 权重缩放因子（可选）
    w1_zp: Optional[torch.Tensor] = None,  # w1 零点（可选）
    w2_zp: Optional[torch.Tensor] = None,  # w2 零点（可选）
    a1_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w1（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w2（可选）
    block_shape: Optional[List[int]] = None,  # 块状量化的块形状（可选）
    no_combine: bool = False,  # 是否跳过合并步骤
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子（可选）
    gemm1_alpha: Optional[float] = None,  # GEMM1 alpha 参数（可选）
    gemm1_limit: Optional[float] = None,  # GEMM1 钳位限制（可选）
    filter_expert: bool = True,  # 是否过滤专家，默认 True
    swiglu_limit: Optional[float] = None,  # SwiGLU 钳位限制（可选）
) -> torch.Tensor:  # 返回输出张量
    return fused_experts_impl(  # 调用融合专家实现函数
        hidden_states,  # 隐藏状态
        w1,  # w1 权重
        w2,  # w2 权重
        topk_weights,  # Top-K 权重
        topk_ids,  # Top-K 专家ID
        b1,  # w1 偏置
        b2,  # w2 偏置
        False,  # inplace=False 非原地操作
        activation,  # 激活函数
        is_gated,  # 是否门控
        apply_router_weight_on_input,  # 是否在输入上应用路由权重
        use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant,  # 逐通道量化标志
        w1_scale,  # w1 缩放因子
        w2_scale,  # w2 缩放因子
        w1_zp,  # w1 零点
        w2_zp,  # w2 零点
        a1_scale,  # w1 激活缩放
        a2_scale,  # w2 激活缩放
        block_shape,  # 块形状
        no_combine=no_combine,  # 是否跳过合并
        routed_scaling_factor=routed_scaling_factor,  # 路由缩放因子
        gemm1_alpha=gemm1_alpha,  # GEMM1 alpha
        gemm1_limit=gemm1_limit,  # GEMM1 限制
        filter_expert=filter_expert,  # 是否过滤专家
        swiglu_limit=swiglu_limit,  # SwiGLU 限制
    )


def fused_experts(  # 融合专家计算入口函数，根据配置选择原地或非原地执行
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_output: StandardTopKOutput,  # 标准 Top-K 输出
    moe_runner_config: MoeRunnerConfig,  # MoE 运行器配置
    b1: Optional[torch.Tensor] = None,  # w1 偏置（可选）
    b2: Optional[torch.Tensor] = None,  # w2 偏置（可选）
    use_fp8_w8a8: bool = False,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: bool = False,  # 是否使用 INT8 W8A8 量化
    use_int8_w8a16: bool = False,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: bool = False,  # 是否使用 INT4 W4A16 量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    w1_scale: Optional[torch.Tensor] = None,  # w1 权重缩放因子（可选）
    w2_scale: Optional[torch.Tensor] = None,  # w2 权重缩放因子（可选）
    w1_zp: Optional[torch.Tensor] = None,  # w1 零点（可选）
    w2_zp: Optional[torch.Tensor] = None,  # w2 零点（可选）
    a1_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w1（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w2（可选）
    block_shape: Optional[List[int]] = None,  # 块状量化的块形状（可选）
):
    topk_weights, topk_ids, _ = topk_output  # 解包 TopK 输出：权重、ID、附加信息
    filter_expert = (  # 判断是否需要过滤专家
        moe_runner_config.num_experts is None  # 配置的专家数为None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts  # 或与本地专家数不一致
    )
    if moe_runner_config.inplace:  # 如果配置为原地操作
        assert not moe_runner_config.no_combine, "no combine + inplace makes no sense"  # 断言不能同时不合并和原地
        inplace_fused_experts(  # 调用原地融合专家函数
            hidden_states,  # 隐藏状态
            w1,  # w1 权重
            w2,  # w2 权重
            topk_weights,  # Top-K 权重
            topk_ids,  # Top-K 专家ID
            b1,  # w1 偏置
            b2,  # w2 偏置
            moe_runner_config.activation,  # 激活函数
            moe_runner_config.is_gated,  # 是否门控
            moe_runner_config.apply_router_weight_on_input,  # 是否在输入上应用路由权重
            use_fp8_w8a8,  # FP8 量化标志
            use_int8_w8a8,  # INT8 W8A8 量化标志
            use_int8_w8a16,  # INT8 W8A16 量化标志
            use_int4_w4a16,  # INT4 W4A16 量化标志
            per_channel_quant,  # 逐通道量化标志
            w1_scale,  # w1 缩放因子
            w2_scale,  # w2 缩放因子
            w1_zp,  # w1 零点
            w2_zp,  # w2 零点
            a1_scale,  # w1 激活缩放
            a2_scale,  # w2 激活缩放
            block_shape,  # 块形状
            moe_runner_config.routed_scaling_factor,  # 路由缩放因子
            moe_runner_config.gemm1_alpha,  # GEMM1 alpha
            moe_runner_config.gemm1_clamp_limit,  # GEMM1 钳位限制
            filter_expert,  # 是否过滤专家
            swiglu_limit=moe_runner_config.swiglu_limit,  # SwiGLU 限制
        )
        return hidden_states  # 返回原地修改后的隐藏状态
    else:  # 非原地操作
        return outplace_fused_experts(  # 调用非原地融合专家函数
            hidden_states,  # 隐藏状态
            w1,  # w1 权重
            w2,  # w2 权重
            topk_weights,  # Top-K 权重
            topk_ids,  # Top-K 专家ID
            b1,  # w1 偏置
            b2,  # w2 偏置
            moe_runner_config.activation,  # 激活函数
            moe_runner_config.is_gated,  # 是否门控
            moe_runner_config.apply_router_weight_on_input,  # 是否在输入上应用路由权重
            use_fp8_w8a8,  # FP8 量化标志
            use_int8_w8a8,  # INT8 W8A8 量化标志
            use_int8_w8a16,  # INT8 W8A16 量化标志
            use_int4_w4a16,  # INT4 W4A16 量化标志
            per_channel_quant,  # 逐通道量化标志
            w1_scale,  # w1 缩放因子
            w2_scale,  # w2 缩放因子
            w1_zp,  # w1 零点
            w2_zp,  # w2 零点
            a1_scale,  # w1 激活缩放
            a2_scale,  # w2 激活缩放
            block_shape,  # 块形状
            no_combine=moe_runner_config.no_combine,  # 是否跳过合并
            routed_scaling_factor=moe_runner_config.routed_scaling_factor,  # 路由缩放因子
            gemm1_alpha=moe_runner_config.gemm1_alpha,  # GEMM1 alpha
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,  # GEMM1 钳位限制
            filter_expert=filter_expert,  # 是否过滤专家
            swiglu_limit=moe_runner_config.swiglu_limit,  # SwiGLU 限制
        )


@torch.compile  # 使用 torch.compile 优化
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):  # 使用 torch.compile 优化的 MoE 求和归约
    torch.sum(x, dim=1, out=out)  # 沿 dim=1 求和，结果写入 out
    out.mul_(routed_scaling_factor)  # 原地乘以路由缩放因子


@torch.compile  # 使用 torch.compile 优化
def _swiglu_silu_clamp_mul(x, gemm1_limit):  # SwiGLU 激活：SiLU + 钳位 + 乘法
    gate, up = x.chunk(2, dim=-1)  # 将输入拆分为门控和上投影两部分
    gate = F.silu(gate)  # 对门控部分应用 SiLU 激活
    gate = gate.clamp(min=None, max=gemm1_limit)  # 对门控结果进行上界钳位
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)  # 对上投影进行双向钳位
    return gate * up  # 返回门控与上投影的逐元素乘积


@torch.compile  # 使用 torch.compile 优化
def swiglu_gpt_oss_sigmoid_alpha(x, gemm1_alpha, gemm1_limit):  # GPT-OSS 风格的 SwiGLU：sigmoid(alpha) 变体
    # NOTE: This variant uses gemm1_alpha, unlike _swiglu_silu_clamp_mul.
    # 注意：此变体使用 gemm1_alpha，不同于 _swiglu_silu_clamp_mul。
    # At present, only GPT-OSS uses this variant.
    # 目前仅 GPT-OSS 使用此变体。
    gate, up = x[..., ::2], x[..., 1::2]  # 交错拆分：偶数索引为门控，奇数索引为上投影
    gate = gate.clamp(min=None, max=gemm1_limit)  # 对门控进行上界钳位
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)  # 对上投影进行双向钳位
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)  # 返回 gate * sigmoid(gate*alpha) * (up+1)


@functools.lru_cache()  # 带缓存的函数，避免重复计算
def _down_moe_use_tma():  # 检查下投影 MoE 是否使用张量描述符加速（TMA）
    return support_tensor_descriptor()  # 返回是否支持张量描述符


def _prepare_fused_moe_run(  # 准备融合 MoE 运行所需的配置和对齐数据
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_ids: torch.Tensor,  # Top-K 专家ID
    *,  # 以下为仅关键字参数
    use_fp8_w8a8: bool,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: bool,  # 是否使用 INT8 W8A8 量化
    use_int8_w8a16: bool,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: bool,  # 是否使用 INT4 W4A16 量化
    per_channel_quant: bool,  # 是否使用逐通道量化
    block_shape: Optional[List[int]],  # 块状量化的块形状
):
    """Resolve config, down_config, TMA flag, and aligned expert routing ids.
    解析配置、下投影配置、TMA 标志及对齐后的专家路由ID。

    Shared by ``fused_experts_impl`` and ``pre_permute_standard_to_triton`` so
    由 ``fused_experts_impl`` 和 ``pre_permute_standard_to_triton`` 共享，
    both paths compute alignment from the same source.
    以确保两条路径从同一来源计算对齐。
    """
    padded_size = padding_size  # 获取默认填充大小
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:  # 非fp8/int8或块量化或使用aiter时
        padded_size = 0  # 不需要填充

    num_tokens = hidden_states.shape[0]  # 获取令牌数
    E = w1.shape[0]  # 获取专家数量
    config_dtype = get_config_dtype_str(  # 获取配置数据类型字符串
        use_fp8_w8a8=use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=use_int4_w4a16,  # INT4 W4A16 量化标志
        dtype=hidden_states.dtype,  # 隐藏状态数据类型
    )

    config, (down_config, _) = try_get_optimal_moe_config(  # 尝试获取最优 MoE 配置
        w1.shape,  # w1 形状
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),  # w2 形状（减去填充）
        topk_ids.shape[1],  # Top-K 值
        config_dtype,  # 配置数据类型
        num_tokens,  # 令牌数
        block_shape=block_shape,  # 块形状
        per_channel_quant=per_channel_quant,  # 逐通道量化
        return_down_config=True,  # 返回下投影配置
    )
    down_moe_use_tma = (  # 判断下投影是否使用 TMA
        _down_moe_use_tma()  # 是否支持张量描述符
        and down_config is not None  # 下投影配置存在
        and down_config.pop("USE_TMA", False)  # 配置中启用了 TMA
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(  # 对齐块大小并排序令牌
        topk_ids, config["BLOCK_SIZE_M"], E  # Top-K ID、块大小M、专家数
    )

    return (  # 返回准备好的所有数据
        config,  # 内核配置
        down_config,  # 下投影内核配置
        down_moe_use_tma,  # 下投影是否使用 TMA
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后令牌数
    )


def _fused_moe_kernel_sequence(  # 融合 MoE 内核序列：执行完整的 GEMM1 → 激活 → GEMM2 → 合并 流程
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_weights: torch.Tensor,  # Top-K 路由权重
    topk_ids: torch.Tensor,  # Top-K 专家ID
    sorted_token_ids: torch.Tensor,  # 排序后的令牌ID
    expert_ids: torch.Tensor,  # 专家ID
    num_tokens_post_padded: torch.Tensor,  # 填充后令牌数
    config: Dict[str, Any],  # 内核配置
    down_config: Optional[Dict[str, Any]],  # 下投影内核配置
    down_moe_use_tma: bool,  # 下投影是否使用 TMA
    *,  # 以下为仅关键字参数
    b1: Optional[torch.Tensor],  # w1 偏置
    b2: Optional[torch.Tensor],  # w2 偏置
    use_fp8_w8a8: bool,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: bool,  # 是否使用 INT8 W8A8 量化
    use_int8_w8a16: bool,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: bool,  # 是否使用 INT4 W4A16 量化
    per_channel_quant: bool,  # 是否使用逐通道量化
    w1_scale: Optional[torch.Tensor],  # w1 权重缩放因子
    w2_scale: Optional[torch.Tensor],  # w2 权重缩放因子
    w1_zp: Optional[torch.Tensor],  # w1 零点
    w2_zp: Optional[torch.Tensor],  # w2 零点
    a1_scale: Optional[torch.Tensor],  # 激活缩放因子用于w1
    a2_scale: Optional[torch.Tensor],  # 激活缩放因子用于w2
    block_shape: Optional[List[int]],  # 块状量化的块形状
    activation: str,  # 激活函数类型
    is_gated: bool,  # 是否为门控 MoE
    no_combine: bool,  # 是否跳过合并步骤
    inplace: bool,  # 是否原地操作
    apply_router_weight_on_input: bool,  # 是否在输入上应用路由权重
    routed_scaling_factor: Optional[float],  # 路由缩放因子
    gemm1_alpha: Optional[float],  # GEMM1 alpha 参数
    gemm1_limit: Optional[float],  # GEMM1 钳位限制
    filter_expert: bool,  # 是否过滤专家
    hooks: Optional[Any] = None,  # 可选的钩子函数
    swiglu_limit: Optional[float] = None,  # SwiGLU 钳位限制
) -> torch.Tensor:  # 返回输出隐藏状态张量
    """Run the MoE kernel/activation/kernel/combine sequence in a single shot.
    在单次调用中运行 MoE 内核/激活/内核/合并序列。

    Inputs are already aligned and the block-size config is already resolved.
    输入已对齐，块大小配置已解析。
    Supports optional LoRA hooks that fire between the two kernels and before
    支持可选的 LoRA 钩子，在两个内核之间和合并之前触发。
    combine. Returns ``out_hidden_states``.
    返回 ``out_hidden_states``。
    """
    num_tokens = hidden_states.shape[0]  # 获取令牌数
    E, N, _ = w1.shape  # 获取专家数E、中间维度N
    topk = topk_ids.shape[1]  # 获取 Top-K 值
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16  # 确定计算精度类型

    padded_tokens = (  # 计算 TMA 所需的填充令牌数
        min(num_tokens * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)  # TMA 填充计算公式
        if down_moe_use_tma  # 仅在使用 TMA 时填充
        else 0  # 不使用 TMA 时不填充
    )
    total_tokens = num_tokens * topk + padded_tokens  # 总令牌数 = 令牌数 * Top-K + 填充令牌数

    if no_combine:  # 不合并模式
        assert not inplace  # 断言不能同时原地操作
        out_hidden_states = torch.empty(  # 分配不合并的输出张量
            (num_tokens, topk, w2.shape[1]),  # 形状：(令牌数, Top-K, 输出维度)
            device=hidden_states.device,  # 设备
            dtype=hidden_states.dtype,  # 数据类型
        )
    elif inplace:  # 原地操作模式
        out_hidden_states = hidden_states  # 直接使用输入张量作为输出
    else:  # 非原地非不合并模式
        out_hidden_states = torch.empty_like(hidden_states)  # 分配与输入相同形状的空张量

    use_fused_moe_sum_all_reduce = (  # 是否使用融合 MoE 求和全归约
        get_global_server_args().enable_fused_moe_sum_all_reduce  # 全局参数启用
        and (not no_combine)  # 不是不合并模式
        and (topk > 2)  # Top-K 大于 2
        and (not use_int8_w8a16)  # 不使用 INT8 W8A16
        and (not use_int4_w4a16)  # 不使用 INT4 W4A16
    )

    intermediate_cache1 = torch.empty(  # 分配第一中间缓存，用于存储 GEMM1 结果
        (total_tokens, N),  # 形状：(总令牌数, 中间维度)
        device=hidden_states.device,  # 设备
        dtype=hidden_states.dtype,  # 数据类型
    )

    invoke_fused_moe_kernel(  # 执行第一组融合 MoE 内核（GEMM1）
        hidden_states,  # 输入隐藏状态
        w1,  # w1 权重
        b1,  # w1 偏置
        intermediate_cache1,  # 输出：第一中间缓存
        a1_scale,  # 激活缩放因子
        w1_scale,  # 权重缩放因子
        w1_zp,  # 权重零点
        topk_weights,  # Top-K 权重
        topk_ids,  # Top-K 专家ID
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后令牌数
        apply_router_weight_on_input,  # 是否在输入上应用路由权重
        topk,  # Top-K 值
        config,  # 内核配置
        compute_type=compute_type,  # 计算类型
        use_fp8_w8a8=use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant=per_channel_quant,  # 逐通道量化标志
        block_shape=block_shape,  # 块形状
        c_sorted=down_moe_use_tma,  # 是否为下投影使用 TMA 排序
        filter_expert=filter_expert,  # 是否过滤专家
    )

    if hooks and hooks.after_gate_up:  # 如果存在门控上投影后的钩子
        # Hooks expect intermediate_cache1 shaped (num_tokens, topk, N); the
        # 钩子期望 intermediate_cache1 形状为 (num_tokens, topk, N)；
        # underlying buffer is laid out as (total_tokens, N) where
        # 底层缓冲区布局为 (total_tokens, N)，
        # total_tokens = num_tokens * topk (+ TMA padding). Slice off any
        # 其中 total_tokens = num_tokens * topk（+ TMA 填充）。裁剪掉
        # padding and reshape for the hook, which writes in-place on the view.
        # 填充部分并重塑形状给钩子，钩子在视图上原地写入。
        hooks.after_gate_up(  # 调用门控上投影后的钩子
            hidden_states,  # 隐藏状态
            intermediate_cache1[: num_tokens * topk].view(num_tokens, topk, N),  # 裁剪并重塑为 (令牌数, Top-K, N)
            topk_weights,  # Top-K 权重
            topk_ids,  # Top-K 专家ID
        )

    intermediate_cache2 = torch.empty(  # 分配第二中间缓存，用于存储激活函数结果
        (total_tokens, N // 2),  # 形状：(总令牌数, 中间维度/2)，因为门控合并后维度减半
        device=hidden_states.device,  # 设备
        dtype=hidden_states.dtype,  # 数据类型
    )

    # Activation function with multiplication
    # 带乘法的激活函数
    if activation == "silu" and is_gated:  # SiLU 激活 + 门控模式
        # - gemm1_alpha != None: GPT-OSS-style swiglu(alpha, limit)
        # - gemm1_alpha != None：GPT-OSS 风格的 swiglu(alpha, limit)
        # - gemm1_alpha == None and gemm1_limit != None: silu+clamp+mul(limit-only)
        # - gemm1_alpha == None 且 gemm1_limit != None：silu+钳位+乘法（仅限limit）
        # - swiglu_limit != None: DeepSeek V4 swiglu clamp + silu_and_mul (CUDA/HIP only)
        # - swiglu_limit != None：DeepSeek V4 swiglu 钳位 + silu_and_mul（仅 CUDA/HIP）
        if gemm1_alpha is not None:  # GPT-OSS 风格：使用 alpha 参数
            assert gemm1_limit is not None  # 断言 gemm1_limit 必须同时存在
            intermediate_cache2 = swiglu_gpt_oss_sigmoid_alpha(  # 调用 GPT-OSS 风格 SwiGLU
                intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit  # 传入展平的中间缓存和 alpha/limit 参数
            )
        elif gemm1_limit is not None:  # 仅有限制模式：SiLU + 钳位 + 乘法
            intermediate_cache2 = _swiglu_silu_clamp_mul(  # 调用 SiLU + 钳位 + 乘法
                intermediate_cache1.view(-1, N), gemm1_limit  # 传入展平的中间缓存和 limit 参数
            )
        elif swiglu_limit is not None:  # DeepSeek V4 风格：SwiGLU 钳位
            # DeepSeek V4: swiglu clamp before silu_and_mul.
            # DeepSeek V4：在 silu_and_mul 之前进行 swiglu 钳位。
            # Two paths gated by SGLANG_OPT_SWIGLU_CLAMP_FUSION:
            # 两条路径由 SGLANG_OPT_SWIGLU_CLAMP_FUSION 控制：
            #   fusion=True: clamp fused into act_and_mul_triton or silu_and_mul_clamp
            #   fusion=True：钳位融合到 act_and_mul_triton 或 silu_and_mul_clamp
            #   fusion=False: explicit clamp_ on intermediate_cache1 (path checker)
            #   fusion=False：在 intermediate_cache1 上显式 clamp_（路径检查器）
            assert swiglu_limit == 10  # 断言 SwiGLU 限制值为 10
            assert intermediate_cache1.shape == (total_tokens, N)  # 断言中间缓存形状正确
            assert _is_cuda or _is_hip, "DeepSeek V4 only supports CUDA/HIP downstream"  # 断言仅支持 CUDA/HIP

            swiglu_limit_for_triton: Optional[float] = None  # 传递给 Triton 内核的 SwiGLU 限制
            swiglu_limit_for_silu_and_mul_clamp: Optional[float] = None  # 传递给 silu_and_mul_clamp 的限制

            if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():  # 如果启用了 SwiGLU 钳位融合优化
                if filter_expert:  # 过滤专家模式
                    swiglu_limit_for_triton = swiglu_limit  # 使用 Triton 内核融合钳位
                else:  # 不过滤专家模式
                    assert (  # 断言
                        _is_cuda
                    ), "fused silu_and_mul_clamp kernel is CUDA-only; HIP must disable SWIGLU_CLAMP_FUSION"  # 融合 silu_and_mul_clamp 内核仅支持 CUDA；HIP 必须禁用 SWIGLU_CLAMP_FUSION
                    swiglu_limit_for_silu_and_mul_clamp = swiglu_limit  # 使用 silu_and_mul_clamp 融合内核
            else:  # 未启用融合优化，显式钳位
                half = N // 2  # 计算半维度
                intermediate_cache1[:, :half].clamp_(max=swiglu_limit)  # 对门控部分进行上界钳位
                intermediate_cache1[:, half:].clamp_(  # 对上投影部分进行双向钳位
                    min=-swiglu_limit, max=swiglu_limit
                )

            if not filter_expert:  # 不过滤专家的路径
                if swiglu_limit_for_silu_and_mul_clamp is not None:  # 使用 silu_and_mul_clamp 融合内核
                    from sglang.jit_kernel.dsv4 import silu_and_mul_clamp  # 延迟导入 DeepSeek V4 专用内核

                    silu_and_mul_clamp(  # 调用带钳位的 SiLU 乘法内核
                        intermediate_cache1.view(-1, N),  # 输入
                        intermediate_cache2,  # 输出
                        swiglu_limit_for_silu_and_mul_clamp,  # 钳位限制
                    )
                else:  # 没有钳位融合内核
                    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)  # 调用标准 SiLU 乘法
            else:  # 过滤专家的路径
                act_and_mul_triton(  # 调用 Triton 激活乘法内核
                    intermediate_cache1.view(-1, N),  # 输入
                    intermediate_cache2,  # 输出
                    config,  # 内核配置
                    topk_ids,  # Top-K 专家ID
                    expert_ids,  # 专家ID
                    down_moe_use_tma,  # 是否使用 TMA
                    activation,  # 激活函数类型
                    swiglu_limit=swiglu_limit_for_triton,  # SwiGLU 限制值
                )
        elif _is_cuda or _is_hip or _is_xpu:  # CUDA/HIP/XPU 平台的普通 SiLU+门控
            if filter_expert and _is_cuda:  # CUDA 平台且过滤专家
                # HIP/XPU fall through to the unfiltered path: the down kernel
                # HIP/XPU 走不过滤路径：下投影内核
                # zeros filtered rows without reading their input.
                # 将过滤行的输出置零而不读取其输入。
                silu_and_mul(  # 调用 SiLU 乘法（带专家过滤）
                    intermediate_cache1.view(-1, N),  # 输入
                    intermediate_cache2,  # 输出
                    expert_ids=(expert_ids if down_moe_use_tma else topk_ids.view(-1)),  # 专家ID（根据TMA选择）
                    expert_step=(config["BLOCK_SIZE_M"] if down_moe_use_tma else 1),  # 专家步长（根据TMA选择）
                )
            else:  # 不过滤专家或其他平台
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)  # 标准SiLU乘法
        elif _is_musa:  # MUSA 平台
            intermediate_cache2 = _silu_and_mul_musa(intermediate_cache1.view(-1, N))  # 使用 MUSA 的 SwishGLU
        else:  # 其他平台
            if _has_vllm_ops:  # 有 vllm_ops 可用
                vllm_ops.silu_and_mul(  # 使用 vllm 的 SiLU 乘法
                    intermediate_cache2, intermediate_cache1.view(-1, N)  # 注意参数顺序：输出在前
                )
            else:  # 没有优化内核的回退方案
                # Fallback: native PyTorch silu_and_mul
                # 回退：原生 PyTorch silu_and_mul
                x = intermediate_cache1.view(-1, N)  # 展平中间缓存
                d = x.shape[-1] // 2  # 计算半维度
                intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])  # SiLU(d前半) * d后半
    elif activation == "gelu" and is_gated:  # GELU 激活 + 门控模式
        assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"  # GELU 不支持 gemm1_alpha
        assert gemm1_limit is None, "gemm1_limit is not supported for gelu"  # GELU 不支持 gemm1_limit
        if _is_cuda or _is_hip:  # CUDA/HIP 平台
            if filter_expert and _is_cuda:  # CUDA 且过滤专家
                gelu_and_mul(  # 调用 GELU 乘法（带专家过滤）
                    intermediate_cache1.view(-1, N),  # 输入
                    intermediate_cache2,  # 输出
                    expert_ids=(expert_ids if down_moe_use_tma else topk_ids.view(-1)),  # 专家ID
                    expert_step=(config["BLOCK_SIZE_M"] if down_moe_use_tma else 1),  # 专家步长
                )
            else:  # 不过滤或其他平台
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)  # 标准GELU乘法
        else:  # 非 CUDA/HIP 平台
            if _has_vllm_ops:  # 有 vllm_ops 可用
                vllm_ops.gelu_and_mul(  # 使用 vllm 的 GELU 乘法
                    intermediate_cache2, intermediate_cache1.view(-1, N)  # 输出在前
                )
            else:  # 没有优化内核的回退方案
                # Fallback: native PyTorch gelu_and_mul
                # 回退：原生 PyTorch gelu_and_mul
                x = intermediate_cache1.view(-1, N)  # 展平中间缓存
                d = x.shape[-1] // 2  # 计算半维度
                intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])  # GELU(d前半) * d后半
    # Activation function without multiplication
    # 不带乘法的激活函数
    elif activation == "silu" and not is_gated:  # SiLU 激活，非门控模式
        intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))  # 直接应用 SiLU
    elif activation == "gelu" and not is_gated:  # GELU 激活，非门控模式
        intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))  # 直接应用 GELU
    elif activation == "relu2" and not is_gated:  # ReLU2 激活（ReLU 的平方），非门控模式
        intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))  # ReLU 后取平方
    else:  # 不支持的激活函数组合
        raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")  # 抛出不支持错误

    del intermediate_cache1  # 释放第一中间缓存

    intermediate_cache3 = torch.empty(  # 分配第三中间缓存，用于存储 GEMM2 的逐专家结果
        (num_tokens, topk, w2.shape[1]),  # 形状：(令牌数, Top-K, 输出维度)
        device=hidden_states.device,  # 设备
        dtype=hidden_states.dtype,  # 数据类型
    )

    # LoRA hooks force the second kernel to write to intermediate_cache3 so
    # LoRA 钩子强制第二个内核写入 intermediate_cache3，以便
    # hooks.after_down can inspect/modify it before reduction.
    # hooks.after_down 可以在归约之前检查/修改它。
    _use_intermediate = not no_combine and (topk != 1 or hooks)  # 是否使用中间缓存（非不合并且topk>1或有钩子）

    out_slice = None  # 输出切片，用于融合全归约
    if use_fused_moe_sum_all_reduce:  # 使用融合全归约时
        out_slice = out_hidden_states  # 直接指向输出张量
        out_slice.zero_()  # 清零输出

    invoke_fused_moe_kernel(  # 执行第二组融合 MoE 内核（GEMM2）
        intermediate_cache2,  # 输入：激活后的中间缓存
        w2,  # w2 权重
        b2,  # w2 偏置
        (  # 根据不同模式选择输出目标
            out_slice  # 融合全归约模式：直接写入输出
            if use_fused_moe_sum_all_reduce
            else (  # 非融合全归约模式
                intermediate_cache3  # 需要中间缓存时：写入中间缓存
                if _use_intermediate
                else out_hidden_states.unsqueeze(0)  # topk=1且无钩子：直接写入输出（增加维度）
            )
        ),
        a2_scale,  # 激活缩放因子
        w2_scale,  # 权重缩放因子
        w2_zp,  # 权重零点
        topk_weights,  # Top-K 权重
        topk_ids,  # Top-K 专家ID
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后令牌数
        not apply_router_weight_on_input and not no_combine,  # 是否在下投影中应用路由权重
        1,  # Top-K 值为1（每个令牌每次只由一个专家处理）
        down_config or config,  # 使用下投影配置或默认配置
        compute_type=compute_type,  # 计算类型
        use_fp8_w8a8=use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant=per_channel_quant,  # 逐通道量化标志
        block_shape=block_shape,  # 块形状
        a_use_tma=down_moe_use_tma,  # 激活矩阵是否使用 TMA
        b_use_tma=down_moe_use_tma,  # 权重矩阵是否使用 TMA
        filter_expert=filter_expert,  # 是否过滤专家
        fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,  # 是否融合全归约
        router_topk=topk,  # 路由 Top-K 值
    )

    if hooks and hooks.after_down:  # 如果存在下投影后的钩子
        hooks.after_down(  # 调用下投影后的钩子
            intermediate_cache2, intermediate_cache3, topk_weights, topk_ids  # 传入中间缓存、结果、权重和ID
        )

    del intermediate_cache2  # 释放第二中间缓存

    if routed_scaling_factor is None:  # 路由缩放因子为None时
        routed_scaling_factor = 1.0  # 默认为1.0

    if no_combine:  # 不合并模式：跳过求和归约
        pass  # 不执行任何操作
    elif _is_cuda or _is_musa:  # CUDA 或 MUSA 平台
        if use_fused_moe_sum_all_reduce:  # 已使用融合全归约
            if routed_scaling_factor != 1.0:  # 路由缩放因子不为1时
                assert out_slice is not None  # 断言输出切片存在
                out_slice.mul_(routed_scaling_factor)  # 原地乘以路由缩放因子
        elif topk == 1 and routed_scaling_factor == 1.0 and not _use_intermediate:  # topk=1且缩放=1且无中间缓存
            pass  # we wrote directly into out_hidden_states  # 已直接写入输出，无需额外操作
        elif topk == 2 and routed_scaling_factor == 1.0:  # topk=2且缩放=1的快速路径
            torch.add(  # 直接使用加法合并两个专家结果
                intermediate_cache3[:, 0],  # 第一个专家结果
                intermediate_cache3[:, 1],  # 第二个专家结果
                out=out_hidden_states,  # 输出到隐藏状态
            ).squeeze(dim=1)  # 去除多余维度
        else:  # 通用路径
            # According to micro benchmark results, torch.compile can get better performance for small token.
            # 根据微基准测试结果，torch.compile 在少量令牌时性能更好。
            if num_tokens <= 32:  # 少量令牌时使用 torch.compile
                moe_sum_reduce_torch_compile(  # 使用 torch.compile 优化的求和归约
                    intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
                    out_hidden_states,  # 输出
                    routed_scaling_factor,  # 路由缩放因子
                )
            else:  # 大量令牌时使用专用内核
                moe_sum_reduce(  # 使用 sgl_kernel 的 MoE 求和归约
                    intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
                    out_hidden_states,  # 输出
                    routed_scaling_factor,  # 路由缩放因子
                )
    elif _is_hip:  # HIP 平台
        if _use_aiter:  # 使用 AITER 时
            moe_sum(  # 使用 aiter 的 MoE 求和
                intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
                out_hidden_states,  # 输出
            )
        else:  # 不使用 AITER 时
            # According to micro benchmark results, torch.compile can get better performance for small token.
            # 根据微基准测试结果，torch.compile 在少量令牌时性能更好。
            if num_tokens <= 32:  # 少量令牌时使用 torch.compile
                moe_sum_reduce_torch_compile(  # 使用 torch.compile 优化的求和归约
                    intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
                    out_hidden_states,  # 输出
                    routed_scaling_factor,  # 路由缩放因子
                )
            else:  # 大量令牌时使用 Triton 内核
                moe_sum_reduce_triton(  # 使用 Triton MoE 求和归约
                    intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
                    out_hidden_states,  # 输出
                    routed_scaling_factor,  # 路由缩放因子
                )
    elif _is_xpu:  # XPU 平台
        moe_sum_reduce(  # 使用 sgl_kernel 的 MoE 求和归约
            intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
            out_hidden_states,  # 输出
            routed_scaling_factor,  # 路由缩放因子
        )
    else:  # 其他平台
        if _has_vllm_ops:  # 有 vllm_ops 可用
            vllm_ops.moe_sum(  # 使用 vllm 的 MoE 求和
                intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
                out_hidden_states,  # 输出
            )
        else:  # 没有 vllm_ops 的回退方案
            # Fallback: use triton moe_sum_reduce when vllm is not available
            # 回退：当 vllm 不可用时使用 triton moe_sum_reduce
            moe_sum_reduce_triton(  # 使用 Triton MoE 求和归约
                intermediate_cache3.view(*intermediate_cache3.shape),  # 重塑中间缓存
                out_hidden_states,  # 输出
                routed_scaling_factor,  # 路由缩放因子
            )

    del intermediate_cache3  # 释放第三中间缓存

    return out_hidden_states  # 返回输出隐藏状态


def fused_experts_impl(  # 融合专家计算的实现函数，包含约束检查和调用核心序列
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_weights: torch.Tensor,  # Top-K 路由权重
    topk_ids: torch.Tensor,  # Top-K 专家ID
    b1: Optional[torch.Tensor] = None,  # w1 偏置（可选）
    b2: Optional[torch.Tensor] = None,  # w2 偏置（可选）
    inplace: bool = False,  # 是否原地操作，默认 False
    activation: str = "silu",  # 激活函数类型，默认 silu
    is_gated: bool = True,  # 是否为门控 MoE，默认 True
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: bool = False,  # 是否使用 INT8 W8A8 量化
    use_int8_w8a16: bool = False,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: bool = False,  # 是否使用 INT4 W4A16 量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    w1_scale: Optional[torch.Tensor] = None,  # w1 权重缩放因子（可选）
    w2_scale: Optional[torch.Tensor] = None,  # w2 权重缩放因子（可选）
    w1_zp: Optional[torch.Tensor] = None,  # w1 零点（可选）
    w2_zp: Optional[torch.Tensor] = None,  # w2 零点（可选）
    a1_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w1（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w2（可选）
    block_shape: Optional[List[int]] = None,  # 块状量化的块形状（可选）
    no_combine: bool = False,  # 是否跳过合并步骤
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子（可选）
    gemm1_alpha: Optional[float] = None,  # GEMM1 alpha 参数（可选）
    gemm1_limit: Optional[float] = None,  # GEMM1 钳位限制（可选）
    filter_expert: bool = True,  # 是否过滤专家，默认 True
    swiglu_limit: Optional[float] = None,  # SwiGLU 钳位限制（可选）
):
    padded_size = padding_size  # 获取默认填充大小
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:  # 非fp8/int8或块量化或使用aiter
        padded_size = 0  # 不需要填充

    # Check constraints.
    # 检查约束条件。
    if use_int4_w4a16:  # INT4 量化模式
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"  # 断言隐藏维度的一半等于w1输入维度
    else:  # 其他模式
        assert (  # 断言隐藏维度等于w1输入维度减去填充
            hidden_states.shape[1] == w1.shape[2] - padded_size
        ), f"Hidden size mismatch"  # 隐藏维度不匹配
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"  # 断言 Top-K 权重和ID形状一致
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"  # 断言隐藏状态内存连续
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"  # 断言w1权重内存连续
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"  # 断言w2权重内存连续
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]  # 断言数据类型合法

    (  # 调用准备函数获取配置和对齐数据
        config,  # 内核配置
        down_config,  # 下投影内核配置
        down_moe_use_tma,  # 下投影是否使用 TMA
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后令牌数
    ) = _prepare_fused_moe_run(  # 准备融合 MoE 运行
        hidden_states,  # 隐藏状态
        w1,  # w1 权重
        w2,  # w2 权重
        topk_ids,  # Top-K 专家ID
        use_fp8_w8a8=use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant=per_channel_quant,  # 逐通道量化标志
        block_shape=block_shape,  # 块形状
    )

    return _fused_moe_kernel_sequence(  # 调用融合 MoE 内核序列
        hidden_states,  # 隐藏状态
        w1,  # w1 权重
        w2,  # w2 权重
        topk_weights,  # Top-K 权重
        topk_ids,  # Top-K 专家ID
        sorted_token_ids,  # 排序后的令牌ID
        expert_ids,  # 专家ID
        num_tokens_post_padded,  # 填充后令牌数
        config,  # 内核配置
        down_config,  # 下投影内核配置
        down_moe_use_tma,  # 下投影是否使用 TMA
        b1=b1,  # w1 偏置
        b2=b2,  # w2 偏置
        use_fp8_w8a8=use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant=per_channel_quant,  # 逐通道量化标志
        w1_scale=w1_scale,  # w1 缩放因子
        w2_scale=w2_scale,  # w2 缩放因子
        w1_zp=w1_zp,  # w1 零点
        w2_zp=w2_zp,  # w2 零点
        a1_scale=a1_scale,  # w1 激活缩放
        a2_scale=a2_scale,  # w2 激活缩放
        block_shape=block_shape,  # 块形状
        activation=activation,  # 激活函数类型
        is_gated=is_gated,  # 是否门控
        no_combine=no_combine,  # 是否跳过合并
        inplace=inplace,  # 是否原地操作
        apply_router_weight_on_input=apply_router_weight_on_input,  # 是否在输入上应用路由权重
        routed_scaling_factor=routed_scaling_factor,  # 路由缩放因子
        gemm1_alpha=gemm1_alpha,  # GEMM1 alpha
        gemm1_limit=gemm1_limit,  # GEMM1 限制
        filter_expert=filter_expert,  # 是否过滤专家
        hooks=None,  # 无钩子
        swiglu_limit=swiglu_limit,  # SwiGLU 限制
    )


def fused_moe(  # MoE 层的完整计算函数，对外暴露的公共接口
    hidden_states: torch.Tensor,  # 隐藏状态张量
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_output: StandardTopKOutput,  # 标准 Top-K 输出
    moe_runner_config: MoeRunnerConfig = MoeRunnerConfig(),  # MoE 运行器配置，默认空配置
    b1: Optional[torch.Tensor] = None,  # w1 偏置（可选）
    b2: Optional[torch.Tensor] = None,  # w2 偏置（可选）
    use_fp8_w8a8: bool = False,  # 是否使用 FP8 W8A8 量化
    use_int8_w8a8: bool = False,  # 是否使用 INT8 W8A8 量化
    use_int8_w8a16: bool = False,  # 是否使用 INT8 W8A16 量化
    use_int4_w4a16: bool = False,  # 是否使用 INT4 W4A16 量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    w1_scale: Optional[torch.Tensor] = None,  # w1 权重缩放因子（可选）
    w2_scale: Optional[torch.Tensor] = None,  # w2 权重缩放因子（可选）
    w1_zp: Optional[torch.Tensor] = None,  # w1 零点（可选）
    w2_zp: Optional[torch.Tensor] = None,  # w2 零点（可选）
    a1_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w1（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 激活缩放因子用于w2（可选）
    block_shape: Optional[List[int]] = None,  # 块状量化的块形状（可选）
) -> torch.Tensor:  # 返回 MoE 层输出张量
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.
    此函数使用两组权重 w1 和 w2 以及 top-k 门控机制计算混合专家（MoE）层。

    Parameters:
    参数：
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - hidden_states (torch.Tensor)：MoE 层的输入张量。
    - w1 (torch.Tensor): The first set of expert weights.
    - w1 (torch.Tensor)：第一组专家权重。
    - w2 (torch.Tensor): The second set of expert weights.
    - w2 (torch.Tensor)：第二组专家权重。
    - topk_output (StandardTopKOutput): The top-k output of the experts.
    - topk_output (StandardTopKOutput)：专家的 top-k 输出。
    - moe_runner_config (MoeRunnerConfig): The configuration for the MoE runner.
    - moe_runner_config (MoeRunnerConfig)：MoE 运行器的配置。
    - b1 (Optional[torch.Tensor]): Optional bias for w1.
    - b1 (Optional[torch.Tensor])：w1 的可选偏置。
    - b2 (Optional[torch.Tensor]): Optional bias for w2.
    - b2 (Optional[torch.Tensor])：w2 的可选偏置。
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_fp8_w8a8 (bool)：如果为 True，使用 fp8 算术计算 w1 和 w2 的内积。默认为 False。
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a8 (bool)：如果为 True，使用 int8 算术计算 w1 和 w2 的内积。默认为 False。
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool)：如果为 True，使用 fp8 算术计算 w1 和 w2 的内积。默认为 False。
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int4_w4a16 (bool)：如果为 True，使用 int4 权重与 bf16/fp16 激活的矩阵乘法
        计算 w1 和 w2 的内积。默认为 False。
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w1_scale (Optional[torch.Tensor])：w1 的可选缩放因子。
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - w2_scale (Optional[torch.Tensor])：w2 的可选缩放因子。
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a1_scale (Optional[torch.Tensor])：a1 的可选缩放因子。
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - a2_scale (Optional[torch.Tensor])：a2 的可选缩放因子。
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.
    - block_shape: (Optional[List[int]])：块状量化的可选块大小。
    - gemm1_alpha (Optional[float]): Optional gemm1_alpha for the activation
        function.
    - gemm1_alpha (Optional[float])：激活函数的可选 gemm1_alpha 参数。
    - gemm1_limit (Optional[float]): Optional gemm1_limit for the swiglu activation
        function.
    - gemm1_limit (Optional[float])：swiglu 激活函数的可选 gemm1_limit 参数。

    Returns:
    返回：
    - torch.Tensor: The output tensor after applying the MoE layer.
    - torch.Tensor：应用 MoE 层后的输出张量。
    """
    if _use_sgl_xpu:  # Intel XPU 平台使用专用实现
        topk_weight, topk_ids, _ = topk_output  # 解包 TopK 输出
        from sgl_kernel import fused_experts as sgl_fused_experts  # 导入 XPU 专用融合专家函数

        return sgl_fused_experts(  # 调用 XPU 专用融合专家函数
            hidden_states,  # 隐藏状态
            w1,  # w1 权重
            w2,  # w2 权重
            topk_weight,  # Top-K 权重
            topk_ids,  # Top-K 专家ID
            b1=b1,  # w1 偏置
            b2=b2,  # w2 偏置
            use_fp8_w8a8=use_fp8_w8a8,  # FP8 量化标志
            w1_scale=w1_scale,  # w1 缩放因子
            w2_scale=w2_scale,  # w2 缩放因子
            w1_zp=w1_zp,  # w1 零点
            w2_zp=w2_zp,  # w2 零点
            a1_scale=a1_scale,  # w1 激活缩放
            a2_scale=a2_scale,  # w2 激活缩放
            block_shape=block_shape,  # 块形状
        )

    return fused_experts(  # 其他平台调用通用融合专家函数
        hidden_states,  # 隐藏状态
        w1,  # w1 权重
        w2,  # w2 权重
        topk_output,  # TopK 输出
        moe_runner_config=moe_runner_config,  # 运行器配置
        b1=b1,  # w1 偏置
        b2=b2,  # w2 偏置
        use_fp8_w8a8=use_fp8_w8a8,  # FP8 量化标志
        use_int8_w8a8=use_int8_w8a8,  # INT8 W8A8 量化标志
        use_int8_w8a16=use_int8_w8a16,  # INT8 W8A16 量化标志
        use_int4_w4a16=use_int4_w4a16,  # INT4 W4A16 量化标志
        per_channel_quant=per_channel_quant,  # 逐通道量化标志
        w1_scale=w1_scale,  # w1 缩放因子
        w2_scale=w2_scale,  # w2 缩放因子
        w1_zp=w1_zp,  # w1 零点
        w2_zp=w2_zp,  # w2 零点
        a1_scale=a1_scale,  # w1 激活缩放
        a2_scale=a2_scale,  # w2 激活缩放
        block_shape=block_shape,  # 块形状
    )
