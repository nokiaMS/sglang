# 文件说明：基于triton_kernels库的融合MoE计算模块
# 本模块使用triton_kernels库的matmul_ogs算子实现MoE前向计算，
# 支持带偏置和不带偏置两种模式，包含路由数据解析、BF16量化配置、
# 融合SwiGLU激活等功能，适配vLLM项目的triton_kernels接口

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/pull/18595/files#diff-f426a6de78c82ffec568eff6811bfbf0043dab5f87f1a8c0cffdbdcb8a81e035
# 适配自vLLM项目的triton_kernels MoE实现

from __future__ import annotations  # 启用延迟类型注解求值

from typing import TYPE_CHECKING, Optional  # 导入类型检查和可选类型注解

import torch  # 导入PyTorch库
from triton_kernels.matmul_ogs import (  # 从triton_kernels导入MoE矩阵乘法相关组件
    FlexCtx,  # 灵活上下文（用于量化配置）
    FnSpecs,  # 函数规格（用于融合激活）
    FusedActivation,  # 融合激活类
    GatherIndx,  # 收集索引类
    PrecisionConfig,  # 精度配置类
    RoutingData,  # 路由数据类
    ScatterIndx,  # 散射索引类
    matmul_ogs,  # 分组稀疏矩阵乘法函数
)
from triton_kernels.numerics import InFlexData  # 导入灵活数据输入类
from triton_kernels.swiglu import swiglu_fn  # 导入SwiGLU激活函数

from sglang.srt.utils import is_cuda  # 导入CUDA环境检测函数

if is_cuda():  # 如果是CUDA环境
    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul  # 导入JIT编译的激活函数内核
else:  # 非CUDA环境
    from sgl_kernel import gelu_and_mul, silu_and_mul  # 导入预编译的激活函数内核

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # 导入MoE运行器配置类
    from sglang.srt.layers.moe.topk import TopKOutput  # 导入TopK输出类型


def quantize(w, dtype, dev, **opt):  # 量化函数：将权重量化为指定格式
    # 将权重转换为指定数据类型，返回量化权重和灵活数据信息
    if dtype == "bf16":  # 如果目标类型为BF16
        return w.to(torch.bfloat16), InFlexData()  # 转为BF16，返回空灵活数据


def triton_kernel_moe_forward(  # 基于triton_kernels的MoE前向计算函数
    hidden_states: torch.Tensor,  # 输入隐藏状态
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    topk_output: TopKOutput,  # TopK输出（包含路由数据、收集和散射索引）
    moe_runner_config: MoeRunnerConfig,  # MoE运行器配置
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用FP8 W8A8量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    global_num_experts: int = -1,  # 全局专家数量
    expert_map: Optional[torch.Tensor] = None,  # 专家映射表
    w1_scale: Optional[torch.Tensor] = None,  # w1缩放因子
    w2_scale: Optional[torch.Tensor] = None,  # w2缩放因子
    a1_scale: Optional[torch.Tensor] = None,  # 激活1缩放因子
    a2_scale: Optional[torch.Tensor] = None,  # 激活2缩放因子
    block_shape: Optional[list[int]] = None,  # 块形状
) -> torch.Tensor:  # 返回MoE计算结果
    # 解析topk输出并委托给融合专家计算函数

    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出格式检查器

    assert TopKOutputChecker.format_is_triton_kernels(topk_output)  # 断言topk输出格式为triton_kernels格式

    routing_data, gather_idx, scatter_idx = topk_output  # 解包topk输出：路由数据、收集索引、散射索引

    return triton_kernel_fused_experts(  # 调用融合专家计算函数
        hidden_states,  # 输入隐藏状态
        w1,  # w1权重
        w2,  # w2权重
        routing_data,  # 路由数据
        gather_idx,  # 收集索引
        scatter_idx,  # 散射索引
        inplace=False,  # triton kernel doesn't support inplace  # triton内核不支持原地操作
        activation=moe_runner_config.activation,  # 激活函数类型
        apply_router_weight_on_input=apply_router_weight_on_input,  # 是否在输入上应用路由权重
        use_fp8_w8a8=use_fp8_w8a8,  # FP8量化标志
        per_channel_quant=per_channel_quant,  # 逐通道量化标志
        global_num_experts=global_num_experts,  # 全局专家数量
        expert_map=expert_map,  # 专家映射表
        w1_scale=w1_scale,  # w1缩放因子
        w2_scale=w2_scale,  # w2缩放因子
        a1_scale=a1_scale,  # 激活1缩放因子
        a2_scale=a2_scale,  # 激活2缩放因子
        block_shape=block_shape,  # 块形状
    )


# This is a triton implementation of the fused_experts function
# 这是fused_experts函数的triton实现
def triton_kernel_fused_experts(  # 基于triton_kernels的融合专家计算函数
    hidden_states: torch.Tensor,  # 输入隐藏状态
    w1: torch.Tensor,  # 第一组专家权重
    w2: torch.Tensor,  # 第二组专家权重
    routing_data: RoutingData,  # 路由数据
    gather_indx: GatherIndx,  # 收集索引
    scatter_indx: ScatterIndx,  # 散射索引
    inplace: bool = False,  # 是否原地操作
    activation: str = "silu",  # 激活函数类型
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用FP8量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    global_num_experts: int = -1,  # 全局专家数量
    expert_map: Optional[torch.Tensor] = None,  # 专家映射表
    w1_scale: Optional[torch.Tensor] = None,  # w1缩放因子
    w2_scale: Optional[torch.Tensor] = None,  # w2缩放因子
    a1_scale: Optional[torch.Tensor] = None,  # 激活1缩放因子
    a2_scale: Optional[torch.Tensor] = None,  # 激活2缩放因子
    block_shape: Optional[list[int]] = None,  # 块形状
) -> torch.Tensor:  # 返回MoE计算结果
    # 使用triton_kernels的matmul_ogs实现融合MoE计算

    assert use_fp8_w8a8 is False, "use_fp8_w8a8 is not supported"  # 断言不支持FP8量化
    assert per_channel_quant is False, "per_channel_quant is not supported"  # 断言不支持逐通道量化
    assert expert_map is None, "expert_map is not supported"  # 断言不支持专家映射
    assert w1_scale is None, "w1_scale is not supported"  # 断言不支持w1缩放
    assert w2_scale is None, "w2_scale is not supported"  # 断言不支持w2缩放
    assert a1_scale is None, "a1_scale is not supported"  # 断言不支持激活1缩放
    assert a2_scale is None, "a2_scale is not supported"  # 断言不支持激活2缩放
    assert block_shape is None, "block_shape is not supported"  # 断言不支持块形状

    # type check  # 类型检查
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"  # 断言输入为BF16
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"  # 断言w1为BF16
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"  # 断言w2为BF16

    # Shape check  # 形状检查
    assert hidden_states.ndim == 2, "hidden_states must be 2D"  # 断言输入为2D
    assert (  # 断言输入K维度与w1匹配
        hidden_states.shape[-1] == w1.shape[-2]
    ), f"hidden_states shape[-1] {hidden_states.shape} must be equal to w1 shape[-2] {w1.shape}"
    assert (  # 断言w2的输出维度与w1的中间维度匹配
        w2.shape[-1] == w1.shape[1]
    ), f"w2 shape[-1] {w2.shape[-1]} must be equal to w1 shape[1] {w1.shape[1]}"

    # feature check  # 功能检查
    assert inplace is False, "Inplace is not supported in new triton MoE kernel"  # 断言不支持原地操作

    M, K = hidden_states.shape  # 获取令牌数M和隐藏维度K
    E, _, N = w1.shape  # 获取专家数E、隐藏维度、中间维度N
    n_expts_act = routing_data.n_expts_act  # 获取每个令牌的激活专家数
    dtype = hidden_states.dtype  # 获取数据类型

    if global_num_experts == -1:  # 如果未指定全局专家数
        global_num_experts = E  # 使用w1中的专家数量

    # consistent with default implementation  # 与默认实现保持一致
    intermediate_cache2 = torch.empty(  # 创建中间缓存2（激活后输出）
        (M * n_expts_act, N // 2), device="cuda", dtype=dtype  # 形状为 [M*topk, N//2]
    )

    intermediate_cache1 = matmul_ogs(  # 执行w1矩阵乘法（门控+上投影）
        hidden_states,  # 输入隐藏状态
        w1,  # w1权重
        None,  # 无偏置
        routing_data,  # 路由数据
        gather_indx=gather_indx,  # 收集索引
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,  # 如果在输入上应用权重则传入门控缩放
    )

    if activation == "silu":  # 如果激活函数为SiLU
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)  # 应用SiLU+乘法激活
    elif activation == "gelu":  # 如果激活函数为GELU
        gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)  # 应用GELU+乘法激活
    else:  # 其他激活函数
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")  # 抛出不支持的激活函数异常

    intermediate_cache3 = matmul_ogs(  # 执行w2矩阵乘法（下投影）
        intermediate_cache2,  # 激活后的中间结果
        w2,  # w2权重
        None,  # 无偏置
        routing_data,  # 路由数据
        scatter_indx=scatter_indx,  # 散射索引
        gammas=None if apply_router_weight_on_input else routing_data.gate_scal,  # 如果不在输入上应用权重则在此处应用
    )

    return intermediate_cache3  # 返回最终输出


def triton_kernel_moe_with_bias_forward(  # 带偏置的triton_kernels MoE前向计算函数
    hidden_states: torch.Tensor,  # 输入隐藏状态
    w1: torch.Tensor,  # 第一组专家权重
    w1_pcg,  # w1精度配置
    b1: torch.Tensor,  # w1偏置
    w2: torch.Tensor,  # 第二组专家权重
    w2_pcg,  # w2精度配置
    b2: torch.Tensor,  # w2偏置
    topk_output: TopKOutput,  # TopK输出
    moe_runner_config: MoeRunnerConfig,  # MoE运行器配置
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用FP8量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    global_num_experts: int = -1,  # 全局专家数量
    expert_map: Optional[torch.Tensor] = None,  # 专家映射表
    w1_scale: Optional[torch.Tensor] = None,  # w1缩放因子
    w2_scale: Optional[torch.Tensor] = None,  # w2缩放因子
    a1_scale: Optional[torch.Tensor] = None,  # 激活1缩放因子
    a2_scale: Optional[torch.Tensor] = None,  # 激活2缩放因子
    block_shape: Optional[list[int]] = None,  # 块形状
) -> torch.Tensor:  # 返回MoE计算结果
    # 解析topk输出并委托给带偏置的融合专家计算函数

    from sglang.srt.layers.moe.topk import TopKOutputChecker  # 导入TopK输出格式检查器

    assert TopKOutputChecker.format_is_triton_kernels(topk_output)  # 断言topk输出格式为triton_kernels格式

    routing_data, gather_idx, scatter_idx = topk_output  # 解包topk输出：路由数据、收集索引、散射索引

    return triton_kernel_fused_experts_with_bias(  # 调用带偏置的融合专家计算函数
        hidden_states,  # 输入隐藏状态
        w1=w1,  # w1权重
        w1_pcg=w1_pcg,  # w1精度配置
        b1=b1,  # w1偏置
        w2=w2,  # w2权重
        w2_pcg=w2_pcg,  # w2精度配置
        b2=b2,  # w2偏置
        routing_data=routing_data,  # 路由数据
        gather_indx=gather_idx,  # 收集索引
        scatter_indx=scatter_idx,  # 散射索引
        inplace=False,  # triton kernel doesn't support inplace  # triton内核不支持原地操作
        activation=moe_runner_config.activation,  # 激活函数类型
        apply_router_weight_on_input=apply_router_weight_on_input,  # 是否在输入上应用路由权重
        use_fp8_w8a8=use_fp8_w8a8,  # FP8量化标志
        per_channel_quant=per_channel_quant,  # 逐通道量化标志
        global_num_experts=global_num_experts,  # 全局专家数量
        expert_map=expert_map,  # 专家映射表
        w1_scale=w1_scale,  # w1缩放因子
        w2_scale=w2_scale,  # w2缩放因子
        a1_scale=a1_scale,  # 激活1缩放因子
        a2_scale=a2_scale,  # 激活2缩放因子
        block_shape=block_shape,  # 块形状
        gemm1_alpha=moe_runner_config.gemm1_alpha,  # GEMM1的alpha参数
        gemm1_clamp_limit=moe_runner_config.gemm1_clamp_limit,  # GEMM1的clamp限制
    )


def triton_kernel_fused_experts_with_bias(  # 带偏置的triton_kernels融合专家计算函数
    hidden_states: torch.Tensor,  # 输入隐藏状态
    w1: torch.Tensor,  # 第一组专家权重
    w1_pcg,  # w1精度配置
    b1: torch.Tensor,  # w1偏置
    w2: torch.Tensor,  # 第二组专家权重
    w2_pcg,  # w2精度配置
    b2: torch.Tensor,  # w2偏置
    routing_data: RoutingData,  # 路由数据
    gather_indx: GatherIndx,  # 收集索引
    scatter_indx: ScatterIndx,  # 散射索引
    inplace: bool = False,  # 是否原地操作
    activation: str = "silu",  # 激活函数类型
    apply_router_weight_on_input: bool = False,  # 是否在输入上应用路由权重
    use_fp8_w8a8: bool = False,  # 是否使用FP8量化
    per_channel_quant: bool = False,  # 是否使用逐通道量化
    global_num_experts: int = -1,  # 全局专家数量
    expert_map: Optional[torch.Tensor] = None,  # 专家映射表
    w1_scale: Optional[torch.Tensor] = None,  # w1缩放因子
    w2_scale: Optional[torch.Tensor] = None,  # w2缩放因子
    a1_scale: Optional[torch.Tensor] = None,  # 激活1缩放因子
    a2_scale: Optional[torch.Tensor] = None,  # 激活2缩放因子
    block_shape: Optional[list[int]] = None,  # 块形状
    gemm1_alpha: Optional[float] = None,  # GEMM1的alpha参数
    gemm1_clamp_limit: Optional[float] = None,  # GEMM1的clamp限制值
) -> torch.Tensor:  # 返回MoE计算结果
    # 使用triton_kernels的matmul_ogs实现带偏置和融合激活的MoE计算

    assert use_fp8_w8a8 is False, "use_fp8_w8a8 is not supported"  # 断言不支持FP8量化
    assert per_channel_quant is False, "per_channel_quant is not supported"  # 断言不支持逐通道量化
    assert expert_map is None, "expert_map is not supported"  # 断言不支持专家映射
    assert w1_scale is None, "w1_scale is not supported"  # 断言不支持w1缩放
    assert w2_scale is None, "w2_scale is not supported"  # 断言不支持w2缩放
    assert a1_scale is None, "a1_scale is not supported"  # 断言不支持激活1缩放
    assert a2_scale is None, "a2_scale is not supported"  # 断言不支持激活2缩放
    assert block_shape is None, "block_shape is not supported"  # 断言不支持块形状

    # type check  # 类型检查
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"  # 断言输入为BF16
    for w in (w1, w2):  # 遍历w1和w2权重
        # TODO assert bf16 or mxfp4
        # TODO 断言为BF16或MXFP4
        # assert (w.dtype == torch.bfloat16) or check-is-mxfp4, f"w must be bfloat16 or mxfp4 {w1.dtype=}"
        pass  # 占位，暂不做类型检查

    # Shape check  # 形状检查
    assert hidden_states.ndim == 2, "hidden_states must be 2D"  # 断言输入为2D
    assert (  # 断言输入K维度与w1匹配
        hidden_states.shape[-1] == w1.shape[-2]
    ), f"hidden_states shape[-1] {hidden_states.shape} must be equal to w1 shape[-2] {w1.shape}"
    assert (  # 断言w2的输出维度与w1的中间维度匹配
        w2.shape[-1] == w1.shape[1]
    ), f"w2 shape[-1] {w2.shape[-1]} must be equal to w1 shape[1] {w1.shape[1]}"

    # feature check  # 功能检查
    assert inplace is False, "Inplace is not supported in new triton MoE kernel"  # 断言不支持原地操作

    M, K = hidden_states.shape  # 获取令牌数M和隐藏维度K
    E, _, N = w1.shape  # 获取专家数E、隐藏维度、中间维度N
    n_expts_act = routing_data.n_expts_act  # 获取每个令牌的激活专家数

    if global_num_experts == -1:  # 如果未指定全局专家数
        global_num_experts = E  # 使用w1中的专家数量

    # TODO maybe completely remove this branch  # TODO 可能完全移除此分支
    if w1.dtype == torch.bfloat16:  # 如果w1为BF16类型
        device = "cuda"  # 使用CUDA设备
        optg = dict()  # 初始化量化选项字典
        w1, w1_flex = quantize(w1, "bf16", device, **optg)  # 将w1量化为BF16格式
        w1_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex))  # 创建w1精度配置

        w2, w2_flex = quantize(w2, "bf16", device, **optg)  # 将w2量化为BF16格式
        w2_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex))  # 创建w2精度配置

    act = FusedActivation(  # 创建融合激活配置
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),  # 指定swiglu函数、参数名、归约因子
        (gemm1_alpha, gemm1_clamp_limit),  # 传入alpha和clamp_limit参数
    )

    intermediate_cache = torch.empty(  # 创建中间缓存（激活后输出）
        (1, M * n_expts_act, N // 2),  # 形状为 [1, M*topk, N//2]
        device=hidden_states.device,  # 设备与输入相同
        dtype=hidden_states.dtype,  # 数据类型与输入相同
    )
    output = torch.empty(  # 创建最终输出张量
        (1, M, K), device=hidden_states.device, dtype=hidden_states.dtype  # 形状为 [1, M, K]
    )

    matmul_ogs(  # 执行w1矩阵乘法（带偏置和融合激活）
        hidden_states,  # 输入隐藏状态
        w1,  # w1权重
        b1,  # w1偏置
        routing_data,  # 路由数据
        gather_indx=gather_indx,  # 收集索引
        precision_config=w1_pcg,  # w1精度配置
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,  # 如果在输入上应用权重则传入门控缩放
        fused_activation=act,  # 融合激活配置
        y=intermediate_cache,  # 输出到中间缓存
    )

    matmul_ogs(  # 执行w2矩阵乘法（带偏置和散射归约）
        intermediate_cache.view(M * n_expts_act, N // 2),  # 中间缓存reshape为2D
        w2,  # w2权重
        b2,  # w2偏置
        routing_data,  # 路由数据
        scatter_indx=scatter_indx,  # 散射索引
        precision_config=w2_pcg,  # w2精度配置
        gammas=None if apply_router_weight_on_input else routing_data.gate_scal,  # 如果不在输入上应用权重则在此处应用
        y=output,  # 输出到最终输出张量
    )
    return output.view(M, K)  # reshape为 [M, K] 并返回
