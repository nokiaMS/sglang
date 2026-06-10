# CUTLASS W4A8 MoE内核模块，提供INT4权重/FP8激活量化的混合专家层计算
# SPDX-License-Identifier: Apache-2.0
"""Cutlass W4A8 MoE kernel."""  # CUTLASS W4A8 MoE内核

from typing import Optional  # 导入可选类型提示

import torch  # 导入PyTorch

from sglang.srt.utils import is_cuda, is_cuda_alike  # 导入CUDA环境检测工具

_is_cuda = is_cuda()  # 检测当前是否为CUDA环境
_is_cuda_alike = is_cuda_alike()  # 检测当前是否为类CUDA环境

if _is_cuda_alike:  # 如果是类CUDA环境
    from sgl_kernel import (  # 从sgl_kernel导入CUDA内核函数
        cutlass_w4a8_moe_mm,  # CUTLASS W4A8 MoE矩阵乘法
        get_cutlass_w4a8_moe_mm_data,  # 获取CUTLASS W4A8 MoE矩阵乘法数据
    )

if _is_cuda:  # 如果是CUDA环境
    from sglang.jit_kernel.activation import silu_and_mul  # 导入SiLU激活与乘法JIT内核
else:  # 否则（非CUDA环境）
    from sgl_kernel import silu_and_mul  # 从sgl_kernel导入SiLU激活与乘法

from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8  # 导入逐张量FP8量化
from sglang.srt.distributed import get_moe_expert_parallel_world_size  # 导入获取MoE专家并行世界大小函数
from sglang.srt.layers.moe.ep_moe.kernels import (  # 导入EP MoE相关内核
    cutlass_w4_run_moe_ep_preproess,  # CUTLASS W4 MoE EP预处理
    deepep_ll_get_cutlass_w4a8_moe_mm_data,  # DeepEP低延迟获取CUTLASS W4A8 MoE矩阵乘法数据
    deepep_permute_triton_kernel,  # DeepEP排列Triton内核
    deepep_post_reorder_triton_kernel,  # DeepEP后重排序Triton内核
    deepep_run_moe_deep_preprocess,  # DeepEP运行MoE深度预处理
    fp8_per_token_to_per_tensor_quant_triton,  # FP8逐令牌到逐张量量化Triton内核
    post_reorder_for_cutlass_moe,  # CUTLASS MoE后重排序
    pre_reorder_for_cutlass_moe,  # CUTLASS MoE预重排序
    silu_and_mul_masked_post_per_tensor_quant_fwd,  # 带掩码SiLU乘法后逐张量量化前向
    silu_mul_static_tensorwise_quant_for_cutlass_moe,  # 静态张量级量化SiLU乘法（CUTLASS MoE用）
)


def cutlass_w4a8_moe(  # CUTLASS W4A8 MoE前向计算函数（标准模式）
    a: torch.Tensor,  # 输入张量 [M, K]
    w1_q: torch.Tensor,  # 第一个INT4量化专家权重 [num_experts, N*2, K//2]
    w2_q: torch.Tensor,  # 第二个INT4量化专家权重 [num_experts, K, N//2]
    w1_scale: torch.Tensor,  # w1_q的反量化缩放因子 [num_experts, K//512, N*8]
    w2_scale: torch.Tensor,  # w2_q的反量化缩放因子 [num_experts, N//512, K*4]
    topk_weights: torch.Tensor,  # 每个令牌到专家映射的权重
    topk_ids: torch.Tensor,  # 每个令牌到专家映射的索引
    a_strides1: torch.Tensor,  # 第一个分组GEMM的输入步幅
    b_strides1: torch.Tensor,  # 第一个分组GEMM的权重步幅
    c_strides1: torch.Tensor,  # 第一个分组GEMM的输出步幅
    a_strides2: torch.Tensor,  # 第二个分组GEMM的输入步幅
    b_strides2: torch.Tensor,  # 第二个分组GEMM的权重步幅
    c_strides2: torch.Tensor,  # 第二个分组GEMM的输出步幅
    s_strides13: torch.Tensor,  # 第一个分组GEMM的输入和缩放步幅
    s_strides2: torch.Tensor,  # 第二个分组GEMM的缩放步幅
    expert_offsets: torch.Tensor,  # 每个专家的计算起始偏移量
    problem_sizes1: torch.Tensor,  # 第一个分组GEMM的问题尺寸
    problem_sizes2: torch.Tensor,  # 第二个分组GEMM的问题尺寸
    a1_scale: Optional[torch.Tensor] = None,  # 输入a的FP8量化缩放因子（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 中间结果的FP8量化缩放因子（可选）
    apply_router_weight_on_input: bool = False,  # 是否将路由权重应用于输入（仅topk=1）
    routed_scaling_factor: float = 1.0,  # 路由缩放因子
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.
    # 本函数使用两组量化权重w1_q和w2_q以及top-k门控机制，计算W4A8量化的混合专家（MoE）层。
    # 矩阵乘法使用CUTLASS分组GEMM实现。

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    # a (torch.Tensor): MoE层的输入张量。形状: [M, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    # w1_q (torch.Tensor): 第一组INT4量化专家权重。形状: [num_experts, N*2, K//2]（转置且INT4打包）
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    # w2_q (torch.Tensor): 第二组INT4量化专家权重。形状: [num_experts, K, N//2]（转置且INT4打包）
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    # w1_scale (torch.Tensor): w1_q的反量化FP32缩放。形状: [num_experts, K//512, N*8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    # w2_scale (torch.Tensor): w2_q的反量化FP32缩放。形状: [num_experts, N//512, K*4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    # topk_weights (torch.Tensor): 每个令牌->专家映射的权重
    - topk_ids (torch.Tensor): The ids of each token->expert mapping.
    # topk_ids (torch.Tensor): 每个令牌->专家映射的索引
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    # a_strides1 (torch.Tensor): 第一个分组GEMM的输入步幅
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    # b_strides1 (torch.Tensor): 第一个分组GEMM的权重步幅
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    # c_strides1 (torch.Tensor): 第一个分组GEMM的输出步幅
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    # a_strides2 (torch.Tensor): 第二个分组GEMM的输入步幅
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    # b_strides2 (torch.Tensor): 第二个分组GEMM的权重步幅
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    # c_strides2 (torch.Tensor): 第二个分组GEMM的输出步幅
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    # s_strides13 (torch.Tensor): 第一个分组GEMM的输入和缩放步幅
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    # s_strides2 (torch.Tensor): 第二个分组GEMM的缩放步幅
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    # a1_scale (Optional[torch.Tensor]): 可选的量化a的FP32缩放。形状: 标量或[1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    # a2_scale (Optional[torch.Tensor]): 可选的量化GEMM之间中间结果的FP32缩放。形状: 标量或[1, N]
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.
    # apply_router_weight_on_input (bool): 为True时，topk权重直接应用于输入。仅适用于topk=1。

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    # 返回: torch.Tensor: 应用MoE层后的FP8输出张量。
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"  # 断言：topk形状匹配
    assert w1_q.dtype == torch.int8  # 断言：w1_q为int8（INT4打包为int8）
    assert w2_q.dtype == torch.int8  # 断言：w2_q为int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"  # 断言：隐藏维度与w1匹配
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"  # 断言：w1和w2隐藏维度匹配
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"  # 断言：专家数一致
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"  # 断言：w1缩放专家数一致
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"  # 断言：w2缩放专家数一致

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"  # 断言：A步幅1专家数一致
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"  # 断言：B步幅1专家数一致
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"  # 断言：A步幅2专家数一致
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"  # 断言：B步幅2专家数一致
    num_local_experts = w1_q.size(0)  # 获取本地专家数量
    m = a.size(0)  # 获取令牌总数
    k = w1_q.size(2) * 2  # w1_q is transposed and packed  # w1_q已转置且打包，实际隐藏维度乘2
    n = w2_q.size(2) * 2  # w2_q is transposed and packed  # w2_q已转置且打包，实际中间维度乘2
    topk = topk_ids.size(1)  # 获取top-k值

    if apply_router_weight_on_input:  # 如果将路由权重应用于输入
        assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"  # 断言：仅支持topk=1

    device = a.device  # 获取计算设备
    if get_moe_expert_parallel_world_size() > 1:  # 如果专家并行世界大小大于1
        topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)  # 将无效ID替换为本地专家数

    src2dst = cutlass_w4_run_moe_ep_preproess(  # 运行MoE EP预处理，获取源到目标映射
        topk_ids,  # top-k专家索引
    )

    gateup_input = torch.empty(  # 分配门控上投影输入缓冲区
        (m * topk, k),  # 形状: [m*topk, k]
        device=device,  # 设备
        dtype=torch.float8_e4m3fn,  # FP8数据类型
    )

    pre_reorder_for_cutlass_moe(  # 执行CUTLASS MoE预重排序
        a,  # 输入张量
        gateup_input,  # 预重排序输出
        src2dst,  # 源到目标映射
        topk_ids,  # top-k专家索引
        a1_scale,  # 输入缩放因子
        num_local_experts,  # 本地专家数
        topk,  # top-k值
        m,  # 令牌数
        k,  # 隐藏维度
    )

    # NOTE: a_map and c_map are not used in the get_cutlass_w4a8_moe_mm_data kernel,
    # they are kept to allow for a quick switch of the permutation logic
    # from the current triton kernel implementation to the cutlass-based one if needed.
    # 注意：a_map和c_map在get_cutlass_w4a8_moe_mm_data内核中未使用，
    # 保留它们是为了在需要时快速将排列逻辑从当前Triton内核实现切换到基于CUTLASS的实现。
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输入映射缓冲区
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输出映射缓冲区
    get_cutlass_w4a8_moe_mm_data(  # 获取CUTLASS W4A8 MoE矩阵乘法所需数据
        topk_ids,  # top-k专家索引
        expert_offsets,  # 专家偏移量
        problem_sizes1,  # 第一个问题尺寸
        problem_sizes2,  # 第二个问题尺寸
        a_map,  # 输入映射
        c_map,  # 输出映射
        num_local_experts,  # 本地专家数
        n,  # 中间维度
        k,  # 隐藏维度
    )

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)  # 分配第一个GEMM输出缓冲区
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)  # 分配第二个GEMM输出缓冲区

    cutlass_w4a8_moe_mm(  # 执行第一个CUTLASS W4A8 MoE矩阵乘法
        c1,  # 输出
        gateup_input,  # FP8量化输入
        w1_q,  # INT4量化权重
        a1_scale.float(),  # 输入缩放因子
        w1_scale,  # 权重缩放因子
        expert_offsets[:-1],  # 专家偏移（不含末尾）
        problem_sizes1,  # 问题尺寸
        a_strides1,  # 输入步幅
        b_strides1,  # 权重步幅
        c_strides1,  # 输出步幅
        s_strides13,  # 缩放步幅
        128,  # 块大小
        topk,  # top-k值
    )

    intermediate_q = torch.empty(  # 分配量化中间结果缓冲区
        (m * topk, n), dtype=torch.float8_e4m3fn, device=device
    )
    silu_mul_static_tensorwise_quant_for_cutlass_moe(  # 执行SiLU乘法及静态张量级量化
        c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
    )

    cutlass_w4a8_moe_mm(  # 执行第二个CUTLASS W4A8 MoE矩阵乘法
        c2,  # 输出
        intermediate_q,  # FP8量化中间结果
        w2_q,  # INT4量化权重
        a2_scale.float(),  # 中间结果缩放因子
        w2_scale,  # 权重缩放因子
        expert_offsets[:-1],  # 专家偏移（不含末尾）
        problem_sizes2,  # 问题尺寸
        a_strides2,  # 输入步幅
        b_strides2,  # 权重步幅
        c_strides2,  # 输出步幅
        s_strides2,  # 缩放步幅
        128,  # 块大小
        topk,  # top-k值
    )

    output = torch.empty_like(a)  # 分配输出张量（与输入形状和类型相同）

    post_reorder_for_cutlass_moe(  # 执行CUTLASS MoE后重排序
        c2,  # 第二个GEMM输出
        output,  # 最终输出
        src2dst,  # 源到目标映射
        topk_ids,  # top-k专家索引
        topk_weights,  # top-k路由权重
        num_local_experts,  # 本地专家数
        topk,  # top-k值
        m,  # 令牌数
        k,  # 隐藏维度
        routed_scaling_factor,  # 路由缩放因子
    )
    return output  # 返回MoE层输出


def cutlass_w4a8_moe_deepep_normal(  # CUTLASS W4A8 MoE DeepEP普通模式前向计算函数
    a: torch.Tensor,  # 输入张量 [M, K]
    w1_q: torch.Tensor,  # 第一个INT4量化专家权重 [num_experts, N*2, K//2]
    w2_q: torch.Tensor,  # 第二个INT4量化专家权重 [num_experts, K, N//2]
    w1_scale: torch.Tensor,  # w1_q的反量化缩放因子
    w2_scale: torch.Tensor,  # w2_q的反量化缩放因子
    topk_weights: torch.Tensor,  # 每个令牌到专家映射的权重
    topk_ids_: torch.Tensor,  # 每个令牌到专家映射的索引
    a_strides1: torch.Tensor,  # 第一个分组GEMM的输入步幅
    b_strides1: torch.Tensor,  # 第一个分组GEMM的权重步幅
    c_strides1: torch.Tensor,  # 第一个分组GEMM的输出步幅
    a_strides2: torch.Tensor,  # 第二个分组GEMM的输入步幅
    b_strides2: torch.Tensor,  # 第二个分组GEMM的权重步幅
    c_strides2: torch.Tensor,  # 第二个分组GEMM的输出步幅
    s_strides13: torch.Tensor,  # 第一个分组GEMM的输入和缩放步幅
    s_strides2: torch.Tensor,  # 第二个分组GEMM的缩放步幅
    expert_offsets: torch.Tensor,  # 每个专家的计算起始偏移量
    problem_sizes1: torch.Tensor,  # 第一个分组GEMM的问题尺寸
    problem_sizes2: torch.Tensor,  # 第二个分组GEMM的问题尺寸
    a1_scale: Optional[torch.Tensor] = None,  # 输入a的FP8量化缩放因子（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 中间结果的FP8量化缩放因子（可选）
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.
    # 本函数使用两组量化权重w1_q和w2_q以及top-k门控机制，计算W4A8量化的MoE层。
    # 矩阵乘法使用CUTLASS分组GEMM实现。

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    # a (torch.Tensor): MoE层的输入张量。形状: [M, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    # w1_q (torch.Tensor): 第一组INT4量化专家权重。形状: [num_experts, N*2, K//2]（转置且INT4打包）
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    # w2_q (torch.Tensor): 第二组INT4量化专家权重。形状: [num_experts, K, N//2]（转置且INT4打包）
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    # w1_scale (torch.Tensor): w1_q的反量化FP32缩放。形状: [num_experts, K//512, N*8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    # w2_scale (torch.Tensor): w2_q的反量化FP32缩放。形状: [num_experts, N//512, K*4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    # topk_weights (torch.Tensor): 每个令牌->专家映射的权重
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    # a_strides1 (torch.Tensor): 第一个分组GEMM的输入步幅
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    # b_strides1 (torch.Tensor): 第一个分组GEMM的权重步幅
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    # c_strides1 (torch.Tensor): 第一个分组GEMM的输出步幅
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    # a_strides2 (torch.Tensor): 第二个分组GEMM的输入步幅
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    # b_strides2 (torch.Tensor): 第二个分组GEMM的权重步幅
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    # c_strides2 (torch.Tensor): 第二个分组GEMM的输出步幅
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    # s_strides13 (torch.Tensor): 第一个分组GEMM的输入和缩放步幅
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    # s_strides2 (torch.Tensor): 第二个分组GEMM的缩放步幅
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    # a1_scale (Optional[torch.Tensor]): 可选的量化a的FP32缩放。形状: 标量或[1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    # a2_scale (Optional[torch.Tensor]): 可选的量化中间结果的FP32缩放。形状: 标量或[1, N]
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.
    # apply_router_weight_on_input (bool): 为True时，topk权重直接应用于输入。仅适用于topk=1。

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    # 返回: torch.Tensor: 应用MoE层后的FP8输出张量。
    """
    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"  # 断言：topk形状匹配
    assert w1_q.dtype == torch.int8  # 断言：w1_q为int8
    assert w2_q.dtype == torch.int8  # 断言：w2_q为int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"  # 断言：隐藏维度与w1匹配
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"  # 断言：w1和w2隐藏维度匹配
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"  # 断言：专家数一致
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"  # 断言：w1缩放专家数一致
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"  # 断言：w2缩放专家数一致

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"  # 断言：A步幅1专家数一致
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"  # 断言：B步幅1专家数一致
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"  # 断言：A步幅2专家数一致
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"  # 断言：B步幅2专家数一致
    num_experts = w1_q.size(0)  # 获取专家数量
    m = a.size(0)  # 获取令牌总数
    k = w1_q.size(2) * 2  # w1_q is transposed and packed  # w1_q已转置且打包
    n = w2_q.size(2) * 2  # w2_q is transposed and packed  # w2_q已转置且打包
    topk = topk_ids_.size(1)  # 获取top-k值

    num_experts = w1_q.size(0)  # 获取专家数量（重复赋值）
    m = a.size(0)  # 获取令牌总数（重复赋值）
    k = w1_q.size(2) * 2  # 获取隐藏维度（重复赋值）
    n = w2_q.size(2) * 2  # 获取中间维度（重复赋值）
    topk = topk_ids_.size(1)  # 获取top-k值（重复赋值）
    device = a.device  # 获取计算设备

    reorder_topk_ids, src2dst, _ = deepep_run_moe_deep_preprocess(  # 运行DeepEP MoE深度预处理
        topk_ids_, num_experts  # top-k专家索引和专家数
    )
    num_total_tokens = reorder_topk_ids.numel()  # 获取重排序后的令牌总数
    gateup_input_pre_reorder = torch.empty(  # 分配预重排序门控上投影输入缓冲区
        (int(num_total_tokens), a.shape[1]),  # 形状: [令牌总数, K]
        device=device,  # 设备
        dtype=a.dtype,  # 与输入相同的数据类型
    )
    deepep_permute_triton_kernel[(a.shape[0],)](  # 执行DeepEP排列Triton内核
        a,  # 输入张量
        gateup_input_pre_reorder,  # 排列输出
        src2dst,  # 源到目标映射
        topk_ids_.to(torch.int64),  # top-k专家索引（转为int64）
        None,  # 无额外参数
        topk,  # top-k值
        a.shape[1],  # 隐藏维度
        BLOCK_SIZE=512,  # 块大小
    )
    gateup_input = torch.empty(  # 分配FP8门控上投影输入缓冲区
        gateup_input_pre_reorder.shape, dtype=torch.float8_e4m3fn, device=device
    )
    per_tensor_quant_fp8(gateup_input_pre_reorder, gateup_input, a1_scale.float(), True)  # 执行逐张量FP8量化
    del gateup_input_pre_reorder  # 释放预重排序缓冲区
    local_topk_ids = topk_ids_  # 使用原始top-k索引
    local_topk_ids = (  # 将无效ID替换为专家数
        torch.where(local_topk_ids == -1, num_experts, topk_ids_).to(torch.int32)
    ).contiguous()  # 确保内存连续

    a_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输入映射缓冲区
    c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输出映射缓冲区
    get_cutlass_w4a8_moe_mm_data(  # 获取CUTLASS W4A8 MoE矩阵乘法所需数据
        local_topk_ids,  # 本地top-k专家索引
        expert_offsets,  # 专家偏移量
        problem_sizes1,  # 第一个问题尺寸
        problem_sizes2,  # 第二个问题尺寸
        a_map,  # 输入映射
        c_map,  # 输出映射
        num_experts,  # 专家数
        n,  # 中间维度
        k,  # 隐藏维度
    )
    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)  # 分配第一个GEMM输出缓冲区
    c2 = torch.zeros((m * topk, k), device=device, dtype=torch.bfloat16)  # 分配第二个GEMM输出缓冲区（初始化为零）

    cutlass_w4a8_moe_mm(  # 执行第一个CUTLASS W4A8 MoE矩阵乘法
        c1,  # 输出
        gateup_input,  # FP8量化输入
        w1_q,  # INT4量化权重
        a1_scale.float(),  # 输入缩放因子
        w1_scale,  # 权重缩放因子
        expert_offsets[:-1],  # 专家偏移（不含末尾）
        problem_sizes1,  # 问题尺寸
        a_strides1,  # 输入步幅
        b_strides1,  # 权重步幅
        c_strides1,  # 输出步幅
        s_strides13,  # 缩放步幅
        128,  # 块大小
        topk,  # top-k值
    )
    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.bfloat16)  # 分配中间结果缓冲区
    silu_and_mul(c1, intermediate)  # 应用SiLU激活与门控乘法

    intermediate_q = torch.empty(  # 分配量化中间结果缓冲区
        intermediate.shape, dtype=torch.float8_e4m3fn, device=device
    )
    per_tensor_quant_fp8(intermediate, intermediate_q, a2_scale.float(), True)  # 执行逐张量FP8量化

    cutlass_w4a8_moe_mm(  # 执行第二个CUTLASS W4A8 MoE矩阵乘法
        c2,  # 输出
        intermediate_q,  # FP8量化中间结果
        w2_q,  # INT4量化权重
        a2_scale.float(),  # 中间结果缩放因子
        w2_scale,  # 权重缩放因子
        expert_offsets[:-1],  # 专家偏移（不含末尾）
        problem_sizes2,  # 问题尺寸
        a_strides2,  # 输入步幅
        b_strides2,  # 权重步幅
        c_strides2,  # 输出步幅
        s_strides2,  # 缩放步幅
        128,  # 块大小
        topk,  # top-k值
    )
    num_tokens = src2dst.shape[0] // topk  # 计算令牌数
    output = torch.empty(  # 分配输出张量
        (num_tokens, c2.shape[1]),  # 形状: [令牌数, 隐藏维度]
        device=c2.device,  # 设备
        dtype=torch.bfloat16,  # bfloat16数据类型
    )
    deepep_post_reorder_triton_kernel[(num_tokens,)](  # 执行DeepEP后重排序Triton内核
        c2,  # 第二个GEMM输出
        output,  # 最终输出
        src2dst,  # 源到目标映射
        topk_ids_,  # top-k专家索引
        topk_weights,  # top-k路由权重
        topk,  # top-k值
        c2.shape[1],  # 隐藏维度
        BLOCK_SIZE=512,  # 块大小
    )

    return output  # 返回MoE层输出


def cutlass_w4a8_moe_deepep_ll(  # CUTLASS W4A8 MoE DeepEP低延迟模式前向计算函数
    a_states: torch.Tensor,  # 输入状态张量 [num_local_experts, num_max_dispatch_tokens_per_rank*num_ranks, K]
    a_scales: torch.Tensor,  # 输入缩放张量
    w1_q: torch.Tensor,  # 第一个INT4量化专家权重 [num_experts, N*2, K//2]
    w2_q: torch.Tensor,  # 第二个INT4量化专家权重 [num_experts, K, N//2]
    w1_scale: torch.Tensor,  # w1_q的反量化缩放因子
    w2_scale: torch.Tensor,  # w2_q的反量化缩放因子
    topk_ids_: torch.Tensor,  # 每个令牌到专家映射的索引
    masked_m: torch.Tensor,  # 掩码矩阵
    a_strides1: torch.Tensor,  # 第一个分组GEMM的输入步幅
    b_strides1: torch.Tensor,  # 第一个分组GEMM的权重步幅
    c_strides1: torch.Tensor,  # 第一个分组GEMM的输出步幅
    a_strides2: torch.Tensor,  # 第二个分组GEMM的输入步幅
    b_strides2: torch.Tensor,  # 第二个分组GEMM的权重步幅
    c_strides2: torch.Tensor,  # 第二个分组GEMM的输出步幅
    s_strides13: torch.Tensor,  # 第一个分组GEMM的输入和缩放步幅
    s_strides2: torch.Tensor,  # 第二个分组GEMM的缩放步幅
    expert_offsets: torch.Tensor,  # 每个专家的计算起始偏移量
    problem_sizes1: torch.Tensor,  # 第一个分组GEMM的问题尺寸
    problem_sizes2: torch.Tensor,  # 第二个分组GEMM的问题尺寸
    a1_scale: Optional[torch.Tensor] = None,  # 输入的FP8量化缩放因子（可选）
    a2_scale: Optional[torch.Tensor] = None,  # 中间结果的FP8量化缩放因子（可选）
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.
    # 本函数使用两组量化权重w1_q和w2_q以及top-k门控机制，计算W4A8量化的MoE层。
    # 矩阵乘法使用CUTLASS分组GEMM实现。

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, K]
    # a (torch.Tensor): MoE层的输入张量。形状: [本地专家数, 最大调度令牌数*rank数, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    # w1_q (torch.Tensor): 第一组INT4量化专家权重。形状: [num_experts, N*2, K//2]（转置且INT4打包）
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    # w2_q (torch.Tensor): 第二组INT4量化专家权重。形状: [num_experts, K, N//2]（转置且INT4打包）
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    # w1_scale (torch.Tensor): w1_q的反量化FP32缩放。形状: [num_experts, K//512, N*8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    # w2_scale (torch.Tensor): w2_q的反量化FP32缩放。形状: [num_experts, N//512, K*4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    # topk_weights (torch.Tensor): 每个令牌->专家映射的权重
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    # a_strides1 (torch.Tensor): 第一个分组GEMM的输入步幅
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    # b_strides1 (torch.Tensor): 第一个分组GEMM的权重步幅
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    # c_strides1 (torch.Tensor): 第一个分组GEMM的输出步幅
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    # a_strides2 (torch.Tensor): 第二个分组GEMM的输入步幅
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    # b_strides2 (torch.Tensor): 第二个分组GEMM的权重步幅
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    # c_strides2 (torch.Tensor): 第二个分组GEMM的输出步幅
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    # s_strides13 (torch.Tensor): 第一个分组GEMM的输入和缩放步幅
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    # s_strides2 (torch.Tensor): 第二个分组GEMM的缩放步幅
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    # a1_scale (Optional[torch.Tensor]): 可选的量化a的FP32缩放。形状: 标量或[1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    # a2_scale (Optional[torch.Tensor]): 可选的量化中间结果的FP32缩放。形状: 标量或[1, N]
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.
    # apply_router_weight_on_input (bool): 为True时，topk权重直接应用于输入。仅适用于topk=1。

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    # 返回: torch.Tensor: 应用MoE层后的FP8输出张量。
    """
    assert w1_q.dtype == torch.int8  # 断言：w1_q为int8
    assert w2_q.dtype == torch.int8  # 断言：w2_q为int8
    assert a_states.shape[2] // 2 == w1_q.shape[2], "Hidden size mismatch w1"  # 断言：隐藏维度与w1匹配
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"  # 断言：w1和w2隐藏维度匹配
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"  # 断言：专家数一致
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"  # 断言：w1缩放专家数一致
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"  # 断言：w2缩放专家数一致

    assert a_strides1.shape[0] == w1_q.shape[0], "A Strides 1 expert number mismatch"  # 断言：A步幅1专家数一致
    assert b_strides1.shape[0] == w1_q.shape[0], "B Strides 1 expert number mismatch"  # 断言：B步幅1专家数一致
    assert a_strides2.shape[0] == w2_q.shape[0], "A Strides 2 expert number mismatch"  # 断言：A步幅2专家数一致
    assert b_strides2.shape[0] == w2_q.shape[0], "B Strides 2 expert number mismatch"  # 断言：B步幅2专家数一致
    num_experts = w1_q.size(0)  # 获取专家数量
    m = a_states.size(1)  # 获取最大调度令牌数
    k = w1_q.size(2) * 2  # w1_q is transposed and packed  # w1_q已转置且打包
    n = w2_q.size(2) * 2  # w2_q is transposed and packed  # w2_q已转置且打包
    topk = topk_ids_.size(1)  # 获取top-k值

    device = a_states.device  # 获取计算设备

    problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(  # 获取DeepEP低延迟CUTLASS W4A8 MoE矩阵乘法数据
        masked_m,  # 掩码矩阵
        problem_sizes1,  # 第一个问题尺寸
        problem_sizes2,  # 第二个问题尺寸
        num_experts,  # 专家数
        n,  # 中间维度
        k,  # 隐藏维度
    )

    gateup_input = torch.empty(a_states.shape, dtype=torch.float8_e4m3fn, device=device)  # 分配FP8门控上投影输入
    fp8_per_token_to_per_tensor_quant_triton(  # 执行FP8逐令牌到逐张量量化
        x=a_states,  # 输入状态
        x_scale=a_scales,  # 输入缩放
        masked_m=masked_m,  # 掩码矩阵
        output_scale=a1_scale,  # 输出缩放
        output=gateup_input,  # 量化输出
    )
    c1 = torch.empty((num_experts, m, n * 2), device=device, dtype=torch.bfloat16)  # 分配第一个GEMM输出
    c2 = torch.empty((num_experts, m, k), device=device, dtype=torch.bfloat16)  # 分配第二个GEMM输出

    cutlass_w4a8_moe_mm(  # 执行第一个CUTLASS W4A8 MoE矩阵乘法
        c1,  # 输出
        gateup_input,  # FP8量化输入
        w1_q,  # INT4量化权重
        a1_scale.float(),  # 输入缩放因子
        w1_scale,  # 权重缩放因子
        expert_offsets[:-1],  # 专家偏移（不含末尾）
        problem_sizes1,  # 问题尺寸
        a_strides1,  # 输入步幅
        b_strides1,  # 权重步幅
        c_strides1,  # 输出步幅
        s_strides13,  # 缩放步幅
        128,  # 块大小
        topk,  # top-k值
    )

    intermediate_q = torch.empty(  # 分配量化中间结果缓冲区
        (num_experts, m, n), device=a_states.device, dtype=torch.float8_e4m3fn
    )
    silu_and_mul_masked_post_per_tensor_quant_fwd(  # 执行带掩码SiLU乘法后逐张量量化前向
        c1, intermediate_q, masked_m, a2_scale
    )
    cutlass_w4a8_moe_mm(  # 执行第二个CUTLASS W4A8 MoE矩阵乘法
        c2,  # 输出
        intermediate_q,  # FP8量化中间结果
        w2_q,  # INT4量化权重
        a2_scale.float(),  # 中间结果缩放因子
        w2_scale,  # 权重缩放因子
        expert_offsets[:-1],  # 专家偏移（不含末尾）
        problem_sizes2,  # 问题尺寸
        a_strides2,  # 输入步幅
        b_strides2,  # 权重步幅
        c_strides2,  # 输出步幅
        s_strides2,  # 缩放步幅
        128,  # 块大小
        topk,  # top-k值
    )

    return c2  # 返回第二个GEMM输出（低延迟模式下不执行后重排序）
