# CUTLASS融合MoE（混合专家）计算模块，提供基于FP8和FP4量化的MoE前向计算实现
"""CUTLASS based Fused MoE kernels."""  # 基于CUTLASS的融合MoE内核

from typing import Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams  # 导入CUTLASS MoE参数类
from sglang.srt.utils import is_cuda, is_sm90_supported, is_sm100_supported  # 导入CUDA及架构支持检测工具

_is_cuda = is_cuda()  # 检测当前是否为CUDA环境
if _is_cuda:  # 如果是CUDA环境
    from sgl_kernel import (  # 从sgl_kernel导入CUDA内核函数
        apply_shuffle_mul_sum,  # 应用洗牌乘加求和
        es_fp8_blockwise_scaled_grouped_mm,  # 专家特化FP8分块缩放分组矩阵乘法
        es_sm100_mxfp8_blockscaled_grouped_mm,  # SM100 MXFP8分块缩放分组矩阵乘法
        es_sm100_mxfp8_blockscaled_grouped_quant,  # SM100 MXFP8分块缩放分组量化
        fp8_blockwise_scaled_grouped_mm,  # FP8分块缩放分组矩阵乘法
        prepare_moe_input,  # 准备MoE输入
        shuffle_rows,  # 行洗牌
    )

    from sglang.jit_kernel.activation import silu_and_mul  # 导入SiLU激活与乘法JIT内核
    from sglang.jit_kernel.nvfp4 import (  # 导入NVFP4相关JIT内核
        cutlass_fp4_group_mm,  # CUTLASS FP4分组矩阵乘法
        scaled_fp4_experts_quant,  # 缩放FP4专家量化
    )


def cutlass_fused_experts_fp8(  # CUTLASS融合FP8专家前向计算函数
    a: torch.Tensor,  # 输入激活张量
    w1_q: torch.Tensor,  # 第一个GEMM的预量化FP8权重（上投影）
    w2_q: torch.Tensor,  # 第二个GEMM的预量化FP8权重（下投影）
    w1_scale: torch.Tensor,  # w1_q对应的分块缩放因子
    w2_scale: torch.Tensor,  # w2_q对应的分块缩放因子
    topk_weights: torch.Tensor,  # 路由器选择的top-k专家权重
    topk_ids: torch.Tensor,  # 路由器选择的top-k专家索引
    a1_strides: torch.Tensor,  # 第一个GEMM的输入步幅
    c1_strides: torch.Tensor,  # 第一个GEMM的输出步幅
    a2_strides: torch.Tensor,  # 第二个GEMM的输入步幅
    c2_strides: torch.Tensor,  # 第二个GEMM的输出步幅
    workspace: torch.Tensor,  # 内核可复用的工作空间
    a_ptrs: torch.Tensor,  # 输入激活指针容器
    b_ptrs: torch.Tensor,  # 输入权重指针容器
    out_ptrs: torch.Tensor,  # 输出激活指针容器
    a_scales_ptrs: torch.Tensor,  # 输入缩放指针容器
    b_scales_ptrs: torch.Tensor,  # 权重缩放指针容器
    expert_offsets: torch.Tensor,  # 每个专家的计算起始偏移量
    problem_sizes1: torch.Tensor,  # 第一个GEMM的问题尺寸
    problem_sizes2: torch.Tensor,  # 第二个GEMM的问题尺寸
    use_fp8_blockscale: bool = True,  # 是否使用FP8分块缩放
    use_mxfp8: bool = False,  # 是否使用MXFP8（UE8M0缩放）
    output: Optional[torch.Tensor] = None,  # 输出张量（可选）
    enable_es: Tuple[bool, bool] = (False, False),  # 是否启用专家特化内核（上投影，下投影）
) -> torch.Tensor:
    """Performs Fused MoE computation using CUTLASS-like kernels with FP8 weights and activations.
    # 使用类CUTLASS内核执行融合MoE计算，权重和激活均为FP8。

    This function implements a Mixture of Experts (MoE) layer with a SwiGLU/SiLU
    activation, leveraging custom kernels likely derived from CUTLASS principles
    for grouped matrix multiplication (`fp8_blockwise_scaled_grouped_mm`) and
    data preparation (`prepare_moe_input`, `silu_and_mul`).
    # 本函数实现了一个带有SwiGLU/SiLU激活的混合专家（MoE）层，
    # 利用可能源自CUTLASS原则的自定义内核进行分组矩阵乘法（fp8_blockwise_scaled_grouped_mm）
    # 和数据准备（prepare_moe_input, silu_and_mul）。

    It handles per-token routing, quantizes input activations to FP8 with
    per-token scales, performs the expert computations using FP8 GEMMs with
    pre-quantized FP8 weights (per-block scales), applies the SiLU activation,
    and combines the results weighted by the router scores.
    # 它处理逐令牌路由，将输入激活量化为FP8（逐令牌缩放），
    # 使用FP8 GEMM（带预量化FP8权重和逐块缩放）执行专家计算，
    # 应用SiLU激活，并按路由器分数加权合并结果。

    Args:
        a (torch.Tensor): Input activations. Shape: `(m, k)`, where `m` is the total
            number of tokens and `k` is the hidden size. Expected dtype: `torch.half`
            or `torch.bfloat16`.
        # a (torch.Tensor): 输入激活。形状：(m, k)，m为令牌总数，k为隐藏维度。dtype: half或bfloat16。
        w1_q (torch.Tensor): Pre-quantized FP8 weight tensor for the first GEMM
            (up-projection part of SwiGLU). Expected shape: `(E, k, n*2)`, where
            `E` is the number of experts, `k` is the hidden size, and `n*2` is the
            intermediate size (`I`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, hidden_size, intermediate_size).
        # w1_q (torch.Tensor): 第一个GEMM的预量化FP8权重（SwiGLU上投影部分）。
        #   形状：(E, k, n*2)，E为专家数，k为隐藏维度，n*2为中间维度。dtype: float8_e4m3fn。
        #   注意：形状表示权重存储为 (专家数, 隐藏维度, 中间维度)。
        w2_q (torch.Tensor): Pre-quantized FP8 weight tensor for the second GEMM
            (down-projection). Expected shape: `(E, n, k)`, where `n` is half the
            intermediate size (`I // 2`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, intermediate_size // 2, hidden_size).
        # w2_q (torch.Tensor): 第二个GEMM的预量化FP8权重（下投影）。
        #   形状：(E, n, k)，n为中间维度的一半。dtype: float8_e4m3fn。
        #   注意：形状表示权重存储为 (专家数, 中间维度//2, 隐藏维度)。
        w1_scale (torch.Tensor): Scales corresponding to `w1_q` (per-block scales).
            Shape: `(E, num_blocks_n, num_blocks_k)`. Dtype: `torch.float32`.
        # w1_scale (torch.Tensor): w1_q对应的缩放因子（逐块缩放）。形状：(E, num_blocks_n, num_blocks_k)。dtype: float32。
        w2_scale (torch.Tensor): Scales corresponding to `w2_q` (per-block scales).
             Shape: `(E, num_blocks_k, num_blocks_n)`. Dtype: `torch.float32`.
        # w2_scale (torch.Tensor): w2_q对应的缩放因子（逐块缩放）。形状：(E, num_blocks_k, num_blocks_n)。dtype: float32。
        topk_weights (torch.Tensor): Router weights for the selected top-k experts
            for each token. Shape: `(m, topk)`. Dtype should ideally match `a`.
        # topk_weights (torch.Tensor): 每个令牌所选top-k专家的路由权重。形状：(m, topk)。dtype应与a匹配。
        topk_ids (torch.Tensor): Indices of the selected top-k experts for each token.
            Shape: `(m, topk)`. Dtype: `torch.int32`.
        # topk_ids (torch.Tensor): 每个令牌所选top-k专家的索引。形状：(m, topk)。dtype: int32。
        a1_strides (torch.Tensor): Stride information for the first GEMM's 'a' input.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
            as it's passed as both a_stride and b_stride in the first call.
        # a1_strides (torch.Tensor): 第一个GEMM输入'a'的步幅信息。形状(E,)，dtype: int64。
        #   注意：在fp8_blockwise_scaled_grouped_mm中同时作为a_stride和b_stride传入。
        c1_strides (torch.Tensor): Stride information for the first GEMM's 'c' output.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
        # c1_strides (torch.Tensor): 第一个GEMM输出'c'的步幅信息。形状(E,)，dtype: int64。
        a2_strides (torch.Tensor): Stride information for the second GEMM's 'a' input.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
            as it's passed as both a_stride and b_stride in the second call.
        # a2_strides (torch.Tensor): 第二个GEMM输入'a'的步幅信息。形状(E,)，dtype: int64。
        #   注意：在fp8_blockwise_scaled_grouped_mm中同时作为a_stride和b_stride传入。
        c2_strides (torch.Tensor): Stride information for the second GEMM's 'c' output.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
        # c2_strides (torch.Tensor): 第二个GEMM输出'c'的步幅信息。形状(E,)，dtype: int64。
        workspace (torch.Tensor): Reusable workspace for the underlying kernel.
        # workspace (torch.Tensor): 内核可复用的工作空间。
        a_ptrs (torch.Tensor): Pointers container for calculating offsets of the input activations for each expert.
        # a_ptrs (torch.Tensor): 用于计算每个专家输入激活偏移量的指针容器。
        b_ptrs (torch.Tensor): Pointers container for calculating offsets of the input weights for each expert.
        # b_ptrs (torch.Tensor): 用于计算每个专家输入权重偏移量的指针容器。
        out_ptrs (torch.Tensor): Pointers container for calculating offsets of the output activations for each expert.
        # out_ptrs (torch.Tensor): 用于计算每个专家输出激活偏移量的指针容器。
        a_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
        # a_scales_ptrs (torch.Tensor): 用于计算每个专家输入缩放偏移量的指针容器。
        b_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
        # b_scales_ptrs (torch.Tensor): 用于计算每个专家权重缩放偏移量的指针容器。
        use_fp8_blockscale (bool, optional): Flag indicating usage of FP8 with
            block scaling. Currently, only `True` is supported. Defaults to `True`.
        # use_fp8_blockscale (bool, 可选): 是否使用FP8分块缩放的标志。目前仅支持True。默认True。
        use_mxfp8 (bool, optional): Flag indicating usage of MXFP8 (UE8M0 scales)
            with SM100 expert-specialization kernels. Defaults to `False`.
        # use_mxfp8 (bool, 可选): 是否使用MXFP8（UE8M0缩放）配合SM100专家特化内核。默认False。
        output (torch.Tensor, optional): Output tensor. If not provided, a new tensor will be created.
        # output (torch.Tensor, 可选): 输出张量。若未提供则创建新张量。
        enable_es (tuple(bool, bool)): Flag indicating usage of expert specialization kernel for (up-projection, down-projection)
        # enable_es (tuple(bool, bool)): 是否为（上投影，下投影）启用专家特化内核的标志
    Returns:
        torch.Tensor: The computed MoE layer output. Shape: `(m, k)`, dtype matches `a`.
    # 返回: torch.Tensor: MoE层计算输出。形状：(m, k)，dtype与a相同。

    Raises:
        AssertionError: If input shapes, dtypes, or flags are inconsistent or unsupported.
        NotImplementedError: If CUDA is not available or `sgl_kernel` is not properly installed.
    # 异常: AssertionError - 输入形状、dtype或标志不一致或不支持时抛出。
    #        NotImplementedError - CUDA不可用或sgl_kernel未正确安装时抛出。
    """
    assert use_fp8_blockscale, "Only support fp8 blockscale for now"  # 断言：目前仅支持FP8分块缩放
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"  # 断言：topk权重和索引形状必须匹配
    assert w1_q.dtype == torch.float8_e4m3fn  # 断言：w1_q必须是float8_e4m3fn类型
    assert w2_q.dtype == torch.float8_e4m3fn  # 断言：w2_q必须是float8_e4m3fn类型
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"  # 断言：输入隐藏维度与w1匹配
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"  # 断言：w1输出维度与w2输入维度匹配
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"  # 断言：w1和w2专家数一致
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"  # 断言：权重专家数一致
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"  # 断言：w1缩放专家数一致
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"  # 断言：w2缩放专家数一致
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"  # 断言：输出dtype必须为half或bfloat16

    if is_cuda:  # 如果是CUDA环境
        from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8量化内核
            sglang_per_token_group_quant_fp8,  # 逐令牌分组FP8量化
        )
    es_up, es_down = enable_es  # 解包专家特化启用标志（上投影，下投影）
    out_dtype = a.dtype  # 输出数据类型与输入相同
    num_experts = w1_q.size(0)  # 获取专家数量
    m = a.size(0)  # 获取令牌总数
    k = w1_q.size(1)  # 获取隐藏维度大小
    n = w2_q.size(1)  # 获取中间维度大小

    topk = topk_ids.size(1)  # 获取top-k值
    device = a.device  # 获取计算设备

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输入映射缓冲区
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输出映射缓冲区

    if use_mxfp8:  # 如果使用MXFP8
        assert es_up and es_down, "MXFP8 requires expert-specialization for both GEMMs"  # MXFP8要求两个GEMM都启用专家特化
        assert is_sm100_supported(), "MXFP8 requires SM100"  # MXFP8要求SM100架构
        assert k % 32 == 0, "MXFP8 requires hidden size to be divisible by 32"  # MXFP8要求隐藏维度可被32整除
        assert n % 32 == 0, "MXFP8 requires intermediate size to be divisible by 32"  # MXFP8要求中间维度可被32整除
        assert w1_scale.dtype == torch.uint8, "MXFP8 w1_scale must be uint8"  # MXFP8下w1缩放必须是uint8
        assert w2_scale.dtype == torch.uint8, "MXFP8 w2_scale must be uint8"  # MXFP8下w2缩放必须是uint8
        expected_w1_scale_shape = (  # 期望的w1缩放形状
            num_experts,  # 专家数
            w1_q.shape[1] // 32,  # 隐藏维度分块数
            w1_q.shape[2],  # 输出维度
        )
        expected_w2_scale_shape = (  # 期望的w2缩放形状
            num_experts,  # 专家数
            w2_q.shape[1] // 32,  # 中间维度分块数
            w2_q.shape[2],  # 输出维度
        )
        assert (  # 断言：w1缩放形状必须符合预期
            w1_scale.shape == expected_w1_scale_shape
        ), f"MXFP8 w1_scale must be {expected_w1_scale_shape}, got {w1_scale.shape}"
        assert (  # 断言：w2缩放形状必须符合预期
            w2_scale.shape == expected_w2_scale_shape
        ), f"MXFP8 w2_scale must be {expected_w2_scale_shape}, got {w2_scale.shape}"

        mxfp8_blockscale_align = 128  # MXFP8分块缩放对齐值
        total_tokens = m * topk  # 总令牌数（含top-k复制）
        nonzero_experts = min(num_experts, total_tokens)  # 非零专家数
        max_total = total_tokens + (mxfp8_blockscale_align - 1) * nonzero_experts  # 最大总数（含对齐填充）
        max_blockscale = (  # 最大分块缩放大小（对齐后）
            (max_total + mxfp8_blockscale_align - 1) // mxfp8_blockscale_align
        ) * mxfp8_blockscale_align

    blockscale_offsets = None  # 初始化分块缩放偏移为None
    if use_mxfp8 and (es_up or es_down):  # 如果使用MXFP8且启用了任一专家特化
        blockscale_offsets = torch.empty(  # 分配分块缩放偏移缓冲区
            (num_experts + 1,), dtype=torch.int32, device=device
        )

    prepare_moe_input(  # 准备MoE输入数据
        topk_ids,  # top-k专家索引
        expert_offsets,  # 专家偏移量
        problem_sizes1,  # 第一个GEMM问题尺寸
        problem_sizes2,  # 第二个GEMM问题尺寸
        a_map,  # 输入映射
        c_map,  # 输出映射
        num_experts,  # 专家数
        n,  # 中间维度
        k,  # 隐藏维度
        blockscale_offsets,  # 分块缩放偏移
    )

    if use_mxfp8 and es_up:  # 如果使用MXFP8且上投影启用专家特化
        rep_a = shuffle_rows(a, a_map, (m * topk, k))  # 按映射洗牌复制输入行
        rep_a_q = torch.empty_like(rep_a, dtype=torch.float8_e4m3fn)  # 分配量化后的复制输入缓冲区
        rep_a1_scales = torch.empty(  # 分配MXFP8分块缩放缓冲区
            (max_blockscale, k // 32), dtype=torch.uint8, device=device
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(  # 执行SM100 MXFP8分块缩放分组量化
            rep_a,  # 复制输入
            problem_sizes1,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            blockscale_offsets[:-1],  # 分块缩放偏移（不含末尾）
            rep_a_q,  # 量化输出
            rep_a1_scales,  # 缩放输出
        )
    else:  # 否则使用标准FP8量化
        a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)  # 逐令牌分组FP8量化（块大小128）
        rep_a_q = shuffle_rows(a_q, a_map, (m * topk, k))  # 按映射洗牌复制量化输入
        rep_a1_scales = shuffle_rows(a1_scale, a_map, (m * topk, int(k / 128)))  # 按映射洗牌复制缩放因子

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)  # 分配第一个GEMM输出缓冲区
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)  # 分配第二个GEMM输出缓冲区

    a_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)  # 激活缩放布局
    w_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)  # 权重缩放布局

    if is_sm90_supported() and es_up:  # SM90架构且上投影启用专家特化
        es_fp8_blockwise_scaled_grouped_mm(  # 执行专家特化FP8分块缩放分组矩阵乘法
            c1,  # 输出
            rep_a_q,  # 量化输入
            w1_q,  # 量化权重
            rep_a1_scales,  # 输入缩放
            w1_scale,  # 权重缩放
            a1_strides,  # 输入步幅
            a1_strides,  # 权重步幅（与输入步幅相同）
            c1_strides,  # 输出步幅
            problem_sizes1,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            workspace,  # 工作空间
        )
    elif use_mxfp8 and es_up:  # MXFP8且上投影启用专家特化
        es_sm100_mxfp8_blockscaled_grouped_mm(  # 执行SM100 MXFP8分块缩放分组矩阵乘法
            c1,  # 输出
            rep_a_q,  # 量化输入
            w1_q,  # 量化权重
            rep_a1_scales,  # 输入缩放
            w1_scale,  # 权重缩放
            problem_sizes1,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            blockscale_offsets[:-1],  # 分块缩放偏移（不含末尾）
        )
    else:  # 否则使用标准FP8分块缩放分组矩阵乘法
        fp8_blockwise_scaled_grouped_mm(  # 执行FP8分块缩放分组矩阵乘法
            c1,  # 输出
            a_ptrs,  # 输入指针
            b_ptrs,  # 权重指针
            out_ptrs,  # 输出指针
            a_scales_ptrs,  # 输入缩放指针
            b_scales_ptrs,  # 权重缩放指针
            rep_a_q,  # 量化输入
            w1_q,  # 量化权重
            rep_a1_scales,  # 输入缩放
            w1_scale,  # 权重缩放
            a1_strides,  # 输入步幅
            a1_strides,  # 权重步幅（与输入步幅相同）
            c1_strides,  # 输出步幅
            a_sf_layout,  # 激活缩放布局
            w_sf_layout,  # 权重缩放布局
            problem_sizes1,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            workspace,  # 工作空间
        )

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)  # 分配中间结果缓冲区
    silu_and_mul(c1, intermediate)  # 应用SiLU激活与门控乘法

    if use_mxfp8 and es_down:  # MXFP8且下投影启用专家特化
        intemediate_q = torch.empty_like(intermediate, dtype=torch.float8_e4m3fn)  # 分配量化中间结果缓冲区
        a2_scale = torch.empty(  # 分配MXFP8分块缩放缓冲区
            (max_blockscale, n // 32), dtype=torch.uint8, device=device
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(  # 执行SM100 MXFP8分块缩放分组量化
            intermediate,  # 中间结果输入
            problem_sizes2,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            blockscale_offsets[:-1],  # 分块缩放偏移（不含末尾）
            intemediate_q,  # 量化输出
            a2_scale,  # 缩放输出
        )
    else:  # 否则使用标准FP8量化
        intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(intermediate, 128)  # 逐令牌分组FP8量化

    if is_sm90_supported() and es_down:  # SM90架构且下投影启用专家特化
        es_fp8_blockwise_scaled_grouped_mm(  # 执行专家特化FP8分块缩放分组矩阵乘法
            c2,  # 输出
            intemediate_q,  # 量化中间结果
            w2_q,  # 量化权重
            a2_scale,  # 中间结果缩放
            w2_scale,  # 权重缩放
            a2_strides,  # 输入步幅
            a2_strides,  # 权重步幅（与输入步幅相同）
            c2_strides,  # 输出步幅
            problem_sizes2,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            workspace,  # 工作空间
        )
    elif use_mxfp8 and es_down:  # MXFP8且下投影启用专家特化
        es_sm100_mxfp8_blockscaled_grouped_mm(  # 执行SM100 MXFP8分块缩放分组矩阵乘法
            c2,  # 输出
            intemediate_q,  # 量化中间结果
            w2_q,  # 量化权重
            a2_scale,  # 中间结果缩放
            w2_scale,  # 权重缩放
            problem_sizes2,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            blockscale_offsets[:-1],  # 分块缩放偏移（不含末尾）
        )
    else:  # 否则使用标准FP8分块缩放分组矩阵乘法
        fp8_blockwise_scaled_grouped_mm(  # 执行FP8分块缩放分组矩阵乘法
            c2,  # 输出
            a_ptrs,  # 输入指针
            b_ptrs,  # 权重指针
            out_ptrs,  # 输出指针
            a_scales_ptrs,  # 输入缩放指针
            b_scales_ptrs,  # 权重缩放指针
            intemediate_q,  # 量化中间结果
            w2_q,  # 量化权重
            a2_scale,  # 中间结果缩放
            w2_scale,  # 权重缩放
            a2_strides,  # 输入步幅
            a2_strides,  # 权重步幅（与输入步幅相同）
            c2_strides,  # 输出步幅
            a_sf_layout,  # 激活缩放布局
            w_sf_layout,  # 权重缩放布局
            problem_sizes2,  # 问题尺寸
            expert_offsets[:-1],  # 专家偏移（不含末尾）
            workspace,  # 工作空间
        )

    if output is None:  # 如果未提供输出张量
        output = torch.empty((m, k), device=device, dtype=out_dtype)  # 创建新的输出张量

    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(out_dtype))  # 应用洗牌乘加求和，合并专家结果
    return output  # 返回MoE层输出


FLOAT4_E2M1_MAX = 6.0  # FP4 E2M1格式的最大值
FLOAT8_E4M3_MAX = 448.0  # FP8 E4M3格式的最大值


def cutlass_moe_fp4(  # CUTLASS FP4 MoE前向计算函数
    a: torch.Tensor,  # 输入激活张量 [m, k]
    a1_gscale: torch.Tensor,  # 第一个GEMM的逐专家全局缩放 [e]
    w1_fp4: torch.Tensor,  # 第一个GEMM的FP4量化权重 [e, 2*n, k//2]
    w1_blockscale: torch.Tensor,  # 第一个GEMM的分块缩放 [e, 2*n, k//block_size]
    w1_alphas: torch.Tensor,  # 第一个GEMM的逐专家alpha值
    a2_gscale: torch.Tensor,  # 第二个GEMM的逐专家全局缩放 [e]
    w2_fp4: torch.Tensor,  # 第二个GEMM的FP4量化权重 [e, k, n//2]
    w2_blockscale: torch.Tensor,  # 第二个GEMM的分块缩放 [e, k, n//block_size]
    w2_alphas: torch.Tensor,  # 第二个GEMM的逐专家alpha值
    topk_weights: torch.Tensor,  # top-k路由权重 [m, topk]
    topk_ids: torch.Tensor,  # top-k专家索引 [m, topk]
    params: CutlassMoEParams,  # CUTLASS MoE参数对象
    apply_router_weight_on_input: bool = False,  # 是否将路由权重应用于输入（仅topk=1时适用）
    no_combine: bool = False,  # 是否跳过专家结果合并
):
    """
    MoE implementation for FP4 Inputs
    # FP4输入的MoE实现

    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)  # 输入张量 [m, k]
    a1_gscale: Activation scale per expert: [e]  (float32)  # 逐专家激活缩放 [e]
    w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]  # w1（门控上投影，非函数参数）[e, 2*n, k]
    w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)  # FP4量化权重 [e, 2*n, k//2]，dtype: uint8
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)  # 注意：n为上投影输出维度，k为全精度输入维度
    w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)  # 分块缩放 [e, 2*n, k//block_size]
                   (Block size = 16 for NVFP4)  # NVFP4的块大小为16

    # Gemm 2
    a2_gscale: Activation scale per expert: [e]  # 逐专家激活缩放 [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]  # w2（下投影，非函数参数）[e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)  # FP4量化权重 [e, k, n//2]，dtype: uint8
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3  # 分块缩放 [e, k, n//block_size]

    Strides for activations, weights and output in logical number of elements.
    The activations & output stride is the number of elements to the next row.
    The weights stride is the number of elements to the next row per expert.
    For example, if the weight is [e, n, k], then the b_stride is a tensor of
    shape [e] with each element being k. Similarly for activations, if the
    shape is [m, k], then the a_stride has shape [e] with each value k.
    Similarly for output, if the output is [m, n], then the c_stride is a
    tensor of shape [e] with each element being k.
    # 激活、权重和输出的步幅（逻辑元素数）。
    # 激活和输出的步幅是到下一行的元素数。
    # 权重的步幅是每个专家到下一行的元素数。
    # 例如，权重为[e, n, k]，则b_stride为形状[e]的张量，每个元素为k。
    # 激活形状[m, k]，则a_stride形状[e]，每个值为k。
    # 输出形状[m, n]，则c_stride形状[e]，每个元素为k。

    Note: cutlass_fp4_group_mm is designed to accept the strides of
    activations and weights to be the same, so it is passed in as a single
    tensor.
    # 注意：cutlass_fp4_group_mm设计为接受激活和权重相同的步幅，因此作为单个张量传入。
    ab_strides_13: [e] dtype: int64 [Gemm 1: Activation / Weight strides]  # Gemm 1：激活/权重步幅
    ab_strides_2: [e] dtype: int64 [Gemm 2: Activation / Weight strides]  # Gemm 2：激活/权重步幅
    c_strides_13: [e] dtype: int64 [Gemm 1: Output Strides]  # Gemm 1：输出步幅
    c_strides_2: [e] dtype: int64 [Gemm 1: Output Strides]  # Gemm 2：输出步幅（原文标注为Gemm 1，实际为Gemm 2）

    topk_weights: [m, topk] dtype: float8  # top-k路由权重
    topk_ids: [m, topk] dtype: float8  # top-k专家索引

    m, n, k: Unquantized weight shapes, dtype: int  # 未量化权重维度
    e: number of experts for the current rank, dtype: int  # 当前rank的专家数
    assumes that topk < k < n to satisfy - up/down projection expectations.  # 假设 topk < k < n 以满足上/下投影期望
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"  # 断言：topk形状匹配
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"  # 断言：w1必须为uint8
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"  # 断言：w2必须为uint8
    assert (  # 断言：所有权重必须是3维的
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_fp4"
    m_a, k_a = a.shape  # 获取输入的令牌数和隐藏维度
    e_w1, nx2_w1, half_k_w1 = w1_fp4.shape  # 获取w1的专家数、输出维度和半隐藏维度
    e_w2, k_w2, half_n_w2 = w2_fp4.shape  # 获取w2的专家数、隐藏维度和半中间维度

    assert e_w1 == e_w2 and e_w1 == params.num_experts, (  # 断言：专家数一致
        "Number of experts must match",
        " between weights.",
    )
    assert (  # 断言：隐藏维度在a、w1和w2之间匹配
        k_a // 2 == half_k_w1 and params.hidden_size == k_w2
    ), "Hidden size mismatch between a, w1 and w2"
    assert (  # 断言：中间维度匹配
        nx2_w1 == params.intermediate_size_per_partition * 2
        and half_n_w2 == params.intermediate_size_per_partition // 2
    ), ("mismatch in " "expected `n`")
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"  # 断言：w1和w2隐藏维度匹配
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"  # 断言：输入dtype有效

    out_dtype = a.dtype  # 输出数据类型与输入相同
    num_topk = topk_ids.shape[1]  # 获取top-k值
    device = a.device  # 获取计算设备
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输入映射缓冲区
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)  # 分配输出映射缓冲区
    prepare_moe_input(  # 准备MoE输入数据
        topk_ids,  # top-k专家索引
        params.expert_offsets,  # 专家偏移量
        params.problem_sizes1,  # 第一个GEMM问题尺寸
        params.problem_sizes2,  # 第二个GEMM问题尺寸
        a_map,  # 输入映射
        c_map,  # 输出映射
        params.num_experts,  # 专家数
        params.intermediate_size_per_partition,  # 中间维度
        params.hidden_size,  # 隐藏维度
        params.blockscale_offsets,  # 分块缩放偏移
    )

    rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(  # 执行缩放FP4专家量化
        a,  # 输入激活
        a1_gscale,  # 逐专家全局缩放
        params.expert_offsets,  # 专家偏移量
        params.blockscale_offsets,  # 分块缩放偏移
        num_topk,  # top-k值
        expert_map=a_map,  # 专家映射
    )
    c1 = cutlass_fp4_group_mm(  # 执行第一个CUTLASS FP4分组矩阵乘法
        rep_a_fp4,  # FP4量化输入
        w1_fp4,  # FP4量化权重
        rep_a_blockscale,  # 输入分块缩放
        w1_blockscale,  # 权重分块缩放
        w1_alphas,  # 逐专家alpha值
        out_dtype,  # 输出数据类型
        params.to_gemm1_args(),  # 第一个GEMM参数
    )
    del rep_a_fp4, rep_a_blockscale  # 释放已使用的中间张量

    # hidden size dimension is split to one half sized tensor.  # 隐藏维度被拆分为一半大小的张量。
    intermediate = torch.empty(  # 分配中间结果缓冲区
        (m_a * num_topk, w1_fp4.shape[1] // 2), device=device, dtype=out_dtype
    )
    silu_and_mul(c1, intermediate)  # 应用SiLU激活与门控乘法

    int_fp4, int_blockscale = scaled_fp4_experts_quant(  # 执行缩放FP4专家量化（中间结果）
        intermediate,  # 中间结果
        a2_gscale,  # 第二个GEMM的逐专家全局缩放
        params.expert_offsets,  # 专家偏移量
        params.blockscale_offsets,  # 分块缩放偏移
        num_topk,  # top-k值
    )
    c2 = cutlass_fp4_group_mm(  # 执行第二个CUTLASS FP4分组矩阵乘法
        int_fp4,  # FP4量化中间结果
        w2_fp4,  # FP4量化权重
        int_blockscale,  # 中间结果分块缩放
        w2_blockscale,  # 权重分块缩放
        w2_alphas,  # 逐专家alpha值
        out_dtype,  # 输出数据类型
        params.to_gemm2_args(),  # 第二个GEMM参数
    )
    del int_fp4, int_blockscale  # 释放已使用的中间张量

    if no_combine:  # 如果不需要合并专家结果
        c2 = shuffle_rows(c2, c_map, (m_a * num_topk, params.hidden_size))  # 按映射洗牌行
        c2 = c2.view(m_a, num_topk, params.hidden_size)  # 重塑为 [m, topk, k]
        return c2.to(out_dtype)  # 返回未合并的结果
    output = torch.empty((m_a, k_a), device=device, dtype=out_dtype)  # 分配输出张量
    weights = topk_weights.to(out_dtype) if not apply_router_weight_on_input else None  # 路由权重（若不在输入上应用）
    apply_shuffle_mul_sum(c2, output, c_map, weights)  # 应用洗牌乘加求和，合并专家结果
    return output  # 返回MoE层输出
