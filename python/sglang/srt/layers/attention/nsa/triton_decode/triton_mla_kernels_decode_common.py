# 文件说明：triton_mla_kernels_decode_common.py - Triton MLA解码的通用工具函数和注意力内核模块
# 包含统一稀疏解码注意力内核、分块注意力辅助函数和基于内存的分块token范围计算
"""
Common utilities and attention kernels for Triton MLA Decode. # Triton MLA解码的通用工具和注意力内核

This module contains shared code for the DeepSeek V4 Triton decode implementation: # 本模块包含DeepSeek V4 Triton解码实现的共享代码：
- Attention kernels (unified sparse decode) # 注意力内核（统一稀疏解码）
- Helper functions for chunked attention # 分块注意力的辅助函数
- Token range computation for memory-based chunking # 基于内存分块的token范围计算
"""

from typing import List, Tuple  # 导入类型提示 # import type hints

import torch  # 导入PyTorch # import PyTorch
import triton  # 导入Triton # import Triton
import triton.language as tl  # 导入Triton语言 # import Triton language

LOG2E = tl.constexpr(1.4426950408889634)  # log2(e)的常量值，用于将自然指数转换为以2为底的指数 # log2(e) constant, for converting natural exp to base-2 exp


# ============================================================================
# Bucketing for autotune keys to avoid recompilation per unique batch size
# 自动调优键的分桶，避免每个唯一批次大小重新编译
# ============================================================================
def _bucket_total_tokens(total_tokens: int) -> int:  # 将total_tokens向上取整到最近的2的幂 # round total_tokens up to nearest power of 2
    """Round total_tokens up to the nearest power of 2 for autotune key stability. # 将total_tokens向上取整到最近的2的幂，以保证自动调优键的稳定性

    In serving, total_tokens (= batch_size * seq_len) varies with every batch. # 在服务中，total_tokens（= batch_size * seq_len）随每个批次变化
    Using the exact value as an autotune key causes recompilation for each unique # 使用精确值作为自动调优键会导致每个唯一值重新编译
    value. Bucketing to powers of 2 limits the number of unique keys to ~15, # 分桶到2的幂将唯一键数量限制在约15个
    dramatically reducing autotuning overhead. # 大幅减少自动调优开销

    Returns: # 返回值：
        Power-of-2 bucket: 1, 2, 4, 8, ..., up to the next power of 2. # 2的幂分桶：1, 2, 4, 8, ...，直到下一个2的幂
    """
    if total_tokens <= 0:  # 如果total_tokens非正 # if total_tokens is non-positive
        return 1  # 返回最小值1 # return minimum value 1
    # Round up to next power of 2 # 向上取整到下一个2的幂
    n = 1  # 初始化n为1 # initialize n to 1
    while n < total_tokens:  # 当n小于total_tokens时 # while n is less than total_tokens
        n <<= 1  # n左移1位，即乘以2 # left shift n by 1, i.e. multiply by 2
    return n  # 返回2的幂结果 # return power-of-2 result


# ============================================================================
# Helper function to compute workload size category for autotune
# 计算自动调优工作负载大小类别的辅助函数
# ============================================================================
def _get_workload_size_category(total_tokens: int, topk: int) -> int:  # 获取工作负载大小类别 # get workload size category
    """
    Compute workload size category for autotune key. # 为自动调优键计算工作负载大小类别
    Returns: # 返回值：
        0: small (< 10K elements) # 0：小（< 10K 元素）
        1: medium (10K - 100K elements) # 1：中（10K - 100K 元素）
        2: large (100K - 1M elements) # 2：大（100K - 1M 元素）
        3: very large (> 1M elements) # 3：非常大（> 1M 元素）
    """
    total_elements = total_tokens * topk  # 计算总元素数 # compute total elements
    if total_elements < 10000:  # 小于1万 # less than 10K
        return 0  # 返回类别0 # return category 0
    elif total_elements < 100000:  # 小于10万 # less than 100K
        return 1  # 返回类别1 # return category 1
    elif total_elements < 1000000:  # 小于100万 # less than 1M
        return 2  # 返回类别2 # return category 2
    else:  # 大于等于100万 # 1M or more
        return 3  # 返回类别3 # return category 3


# ============================================================================
# Unified Attention Kernels
# 统一注意力内核
# ============================================================================


# ============================================================================
# CDNA4 (gfx950) Optimized: Added high-performance configs for MI355X
# CDNA4 (gfx950) 优化：为MI355X添加了高性能配置
# Best config for h_q=128, large topk: BLOCK_H=64, BLOCK_N=256, num_warps=8
# h_q=128、大topk的最佳配置：BLOCK_H=64, BLOCK_N=256, num_warps=8
# ============================================================================
@triton.autotune(  # Triton自动调优装饰器 # Triton autotune decorator
    configs=[  # 配置列表 # config list
        # Selected based on CDNA4 architecture analysis: # 基于CDNA4架构分析选择：
        # - BLOCK_D=128 is fixed (matches KV tile structure for d_qk=512). # BLOCK_D=128固定（匹配d_qk=512的KV块结构）
        # - BLOCK_N=256: best for amortizing memory access over topk dimension. # BLOCK_N=256：在topk维度上摊销内存访问的最佳值
        #   (decode attention is memory-bound; larger BLOCK_N = fewer iterations) # （解码注意力是内存受限的；更大的BLOCK_N = 更少迭代）
        # - num_warps=8: memory-bound decode benefits from more warps for latency hiding. # num_warps=8：内存受限的解码从更多warp中受益以隐藏延迟
        # - BLOCK_H varies to cover different batch sizes: # BLOCK_H变化以覆盖不同的批次大小：
        #   * BLOCK_H=16: cdiv(128,16)=8 H-blocks, best for small batches (bs=1-8) # 适用于小批次(bs=1-8)
        #   * BLOCK_H=32: cdiv(128,32)=4 H-blocks, good for medium batches (bs=8-32) # 适用于中批次(bs=8-32)
        #   * BLOCK_H=64: cdiv(128,64)=2 H-blocks, best for large batches (bs=32+) # 适用于大批次(bs=32+)
        #     (original comment: "Best for h_q=128, large topk") # （原始注释："Best for h_q=128, large topk"）
        #   * BLOCK_H=128: cdiv(128,128)=1 H-block, for very large batches (bs=128+) # 适用于超大批次(bs=128+)
        triton.Config(  # Triton配置 # Triton config
            {"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1  # 小批次配置 # small batch config
        ),
        triton.Config(  # Triton配置 # Triton config
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1  # 中批次配置 # medium batch config
        ),
        triton.Config(  # Triton配置 # Triton config
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1  # 大批次配置 # large batch config
        ),
        triton.Config(  # Triton配置 # Triton config
            {"BLOCK_H": 128, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1  # 超大批次配置 # very large batch config
        ),
    ],
    key=["total_tokens_bucket", "h_q", "total_topk", "d_qk"],  # 自动调优键 # autotune keys
)
@triton.jit  # Triton JIT编译装饰器 # Triton JIT compile decorator
def _unified_sparse_decode_kernel(  # 统一稀疏解码注意力内核 # unified sparse decode attention kernel
    Q,  # 查询张量指针 # query tensor pointer
    KV,  # KV缓存张量指针 # KV cache tensor pointer
    Mask,  # 无效掩码张量指针 # invalid mask tensor pointer
    AttnSink,  # 注意力汇张量指针 # attention sink tensor pointer
    Output,  # 输出张量指针 # output tensor pointer
    LSE,  # log-sum-exp张量指针 # log-sum-exp tensor pointer
    sm_scale,  # softmax缩放因子 # softmax scale factor
    total_tokens,  # 总token数 # total token count
    total_tokens_bucket,  # 分桶后的总token数 # bucketed total token count
    h_q,  # 查询头数 # number of query heads
    total_topk,  # 总topk数 # total topk count
    d_qk,  # QK维度 # QK dimension
    d_v,  # V维度 # V dimension
    stride_q_t,  # Q的token步长 # Q token stride
    stride_q_h,  # Q的头步长 # Q head stride
    stride_q_d,  # Q的维度步长 # Q dimension stride
    stride_kv_t,  # KV的token步长 # KV token stride
    stride_kv_k,  # KV的k步长 # KV k stride
    stride_kv_d,  # KV的维度步长 # KV dimension stride
    stride_mask_t,  # 掩码的token步长 # mask token stride
    stride_mask_k,  # 掩码的k步长 # mask k stride
    stride_o_t,  # 输出的token步长 # output token stride
    stride_o_h,  # 输出的头步长 # output head stride
    stride_o_d,  # 输出的维度步长 # output dimension stride
    stride_lse_t,  # lse的token步长 # lse token stride
    stride_lse_h,  # lse的头步长 # lse head stride
    HAS_ATTN_SINK: tl.constexpr,  # 是否有注意力汇的编译时常量 # compile-time constant for attention sink
    BLOCK_H: tl.constexpr,  # 头维度块大小的编译时常量 # compile-time constant for head block size
    BLOCK_N: tl.constexpr,  # KV维度块大小的编译时常量 # compile-time constant for KV block size
    BLOCK_D: tl.constexpr,  # 维度块大小的编译时常量 # compile-time constant for dimension block size
):
    """Unified attention kernel with single KV buffer (int64 safe)."""  # 统一注意力内核，使用单个KV缓冲区（int64安全）
    pid_t = tl.program_id(0)  # 获取token维度的程序ID # get program ID for token dimension
    pid_h = tl.program_id(1)  # 获取头维度的程序ID # get program ID for head dimension
    pid_t_64 = pid_t.to(tl.int64)  # 将token程序ID转为int64以避免溢出 # cast token program ID to int64 to avoid overflow

    NEG_INF = float("-inf")  # 负无穷常量 # negative infinity constant
    POS_INF = float("+inf")  # 正无穷常量 # positive infinity constant

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # 计算头维度偏移量 # compute head dimension offsets
    mask_h = offs_h < h_q  # 头维度越界掩码 # head dimension out-of-bounds mask

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)  # 行最大值初始化为负无穷 # row max initialized to negative infinity
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)  # 行求和初始化为零 # row sum initialized to zero

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)  # 累加器第0块 # accumulator block 0
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)  # 累加器第1块 # accumulator block 1
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)  # 累加器第2块 # accumulator block 2
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)  # 累加器第3块 # accumulator block 3

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)  # Q步长转为int64 # cast Q stride to int64
    stride_kv_t_64 = tl.cast(stride_kv_t, tl.int64)  # KV步长转为int64 # cast KV stride to int64
    stride_mask_t_64 = tl.cast(stride_mask_t, tl.int64)  # 掩码步长转为int64 # cast mask stride to int64
    q_base = Q + pid_t_64 * stride_q_t_64  # Q的基地址 # Q base address
    kv_base = KV + pid_t_64 * stride_kv_t_64  # KV的基地址 # KV base address
    mask_base = Mask + pid_t_64 * stride_mask_t_64  # 掩码的基地址 # mask base address

    for n_start in range(0, total_topk, BLOCK_N):  # 遍历KV块 # iterate over KV blocks
        offs_n = n_start + tl.arange(0, BLOCK_N)  # 计算KV维度偏移量 # compute KV dimension offsets
        mask_n = offs_n < total_topk  # KV维度越界掩码 # KV dimension out-of-bounds mask

        mask_ptrs = mask_base + offs_n * stride_mask_k  # 掩码指针 # mask pointers
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True)  # 加载无效掩码，越界默认为True # load invalid mask, default True for out-of-bounds
        valid = mask_n & ~invalid  # 有效掩码 = 在界内且非无效 # valid mask = in-bounds and not invalid

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)  # QK点积初始化为零 # QK dot product initialized to zero

        for d_start in range(0, d_qk, BLOCK_D):  # 遍历维度块 # iterate over dimension blocks
            offs_d = d_start + tl.arange(0, BLOCK_D)  # 计算维度偏移量 # compute dimension offsets
            mask_d = offs_d < d_qk  # 维度越界掩码 # dimension out-of-bounds mask

            q_ptrs = (  # Q指针计算 # Q pointer computation
                q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
            )
            q_chunk = tl.load(  # 加载Q块 # load Q chunk
                q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0  # 越界填0 # fill 0 for out-of-bounds
            ).to(tl.bfloat16)  # 转为bfloat16 # cast to bfloat16

            k_ptrs = (  # K指针计算 # K pointer computation
                kv_base + offs_n[:, None] * stride_kv_k + offs_d[None, :] * stride_kv_d
            )
            k_chunk = tl.load(  # 加载K块 # load K chunk
                k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0  # 越界或无效填0 # fill 0 for out-of-bounds or invalid
            ).to(tl.bfloat16)  # 转为bfloat16 # cast to bfloat16

            qk += tl.dot(q_chunk, tl.trans(k_chunk))  # 累加QK点积 # accumulate QK dot product

        qk = qk * sm_scale  # 应用softmax缩放 # apply softmax scale
        qk = tl.where(valid[None, :], qk, NEG_INF)  # 无效位置设为负无穷 # set invalid positions to negative infinity

        m_ij = tl.max(qk, axis=1)  # 当前块的最大值 # max of current block
        m_new = tl.maximum(m_i, m_new) if False else tl.maximum(m_i, m_ij)  # 更新全局最大值 # update global max
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))  # 旧累加器的缩放因子 # scaling factor for old accumulator
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))  # 概率值 # probability values
        l_new = alpha * l_i + tl.sum(p, axis=1)  # 更新归一化分母 # update normalization denominator
        p_bf16 = p.to(tl.bfloat16)  # 概率转为bfloat16 # cast probabilities to bfloat16

        offs_v = tl.arange(0, BLOCK_D)  # V的第0块偏移 # V block 0 offsets
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d  # V指针 # V pointers
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)  # 加载V第0块 # load V block 0
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v)  # 累加输出第0块 # accumulate output block 0

        offs_v = BLOCK_D + tl.arange(0, BLOCK_D)  # V的第1块偏移 # V block 1 offsets
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d  # V指针 # V pointers
        v = tl.load(  # 加载V第1块 # load V block 1
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0  # 越界掩码 # out-of-bounds mask
        ).to(tl.bfloat16)  # 转为bfloat16 # cast to bfloat16
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v)  # 累加输出第1块 # accumulate output block 1

        offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)  # V的第2块偏移 # V block 2 offsets
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d  # V指针 # V pointers
        v = tl.load(  # 加载V第2块 # load V block 2
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0  # 越界掩码 # out-of-bounds mask
        ).to(tl.bfloat16)  # 转为bfloat16 # cast to bfloat16
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v)  # 累加输出第2块 # accumulate output block 2

        offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)  # V的第3块偏移 # V block 3 offsets
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d  # V指针 # V pointers
        v = tl.load(  # 加载V第3块 # load V block 3
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0  # 越界掩码 # out-of-bounds mask
        ).to(tl.bfloat16)  # 转为bfloat16 # cast to bfloat16
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v)  # 累加输出第3块 # accumulate output block 3

        m_i = m_new  # 更新行最大值 # update row max
        l_i = l_new  # 更新行求和 # update row sum

    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E  # 计算log-sum-exp # compute log-sum-exp
    is_lonely_q = l_i == 0.0  # 没有有效KV的孤独查询 # lonely query with no valid KV

    if HAS_ATTN_SINK:  # 如果使用注意力汇 # if using attention sink
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)  # 加载注意力汇值 # load attention sink values
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)  # 计算注意力汇的指数 # compute exp of attention sink
        denominator = l_i + exp_attn_sink_minus_m  # 分母 = 归一化项 + 注意力汇项 # denominator = normalization term + attention sink term
        denominator = tl.where(denominator == 0.0, 1.0, denominator)  # 避免除零 # avoid division by zero
        output_scale = 1.0 / denominator  # 输出缩放因子 # output scaling factor
    else:  # 不使用注意力汇 # not using attention sink
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)  # 孤独查询缩放为0 # scale lonely queries to 0

    # Pre-compute 2D versions for efficiency # 预计算2D版本以提高效率
    is_lonely_q_2d = is_lonely_q[:, None]  # 孤独查询2D掩码 # lonely query 2D mask
    output_scale_2d = output_scale[:, None]  # 输出缩放2D # output scale 2D
    acc_0 = tl.where(is_lonely_q_2d, 0.0, acc_0 * output_scale_2d)  # 缩放累加器第0块 # scale accumulator block 0
    acc_1 = tl.where(is_lonely_q_2d, 0.0, acc_1 * output_scale_2d)  # 缩放累加器第1块 # scale accumulator block 1
    acc_2 = tl.where(is_lonely_q_2d, 0.0, acc_2 * output_scale_2d)  # 缩放累加器第2块 # scale accumulator block 2
    acc_3 = tl.where(is_lonely_q_2d, 0.0, acc_3 * output_scale_2d)  # 缩放累加器第3块 # scale accumulator block 3
    lse = tl.where(is_lonely_q, POS_INF, lse)  # 孤独查询的lse设为正无穷 # set lse to positive infinity for lonely queries

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)  # lse步长转为int64 # cast lse stride to int64
    tl.store(LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h, lse, mask=mask_h)  # 存储lse # store lse

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)  # 输出步长转为int64 # cast output stride to int64
    o_base = Output + pid_t_64 * stride_o_t_64  # 输出基地址 # output base address
    # Pre-compute 2D versions # 预计算2D版本
    offs_h_2d = offs_h[:, None]  # 头偏移2D # head offsets 2D
    mask_h_2d = mask_h[:, None]  # 头掩码2D # head mask 2D
    offs_v_0 = tl.arange(0, BLOCK_D)  # V第0块偏移 # V block 0 offsets
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D)  # V第1块偏移 # V block 1 offsets
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)  # V第2块偏移 # V block 2 offsets
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)  # V第3块偏移 # V block 3 offsets
    tl.store(  # 存储输出第0块 # store output block 0
        o_base + offs_h_2d * stride_o_h + offs_v_0[None, :] * stride_o_d,
        acc_0.to(tl.bfloat16),  # 转为bfloat16存储 # cast to bfloat16 for storage
        mask=mask_h_2d,  # 头掩码 # head mask
    )
    tl.store(  # 存储输出第1块 # store output block 1
        o_base + offs_h_2d * stride_o_h + offs_v_1[None, :] * stride_o_d,
        acc_1.to(tl.bfloat16),  # 转为bfloat16存储 # cast to bfloat16 for storage
        mask=mask_h_2d & (offs_v_1[None, :] < d_v),  # 头掩码且V维度在界内 # head mask and V dimension in-bounds
    )
    tl.store(  # 存储输出第2块 # store output block 2
        o_base + offs_h_2d * stride_o_h + offs_v_2[None, :] * stride_o_d,
        acc_2.to(tl.bfloat16),  # 转为bfloat16存储 # cast to bfloat16 for storage
        mask=mask_h_2d & (offs_v_2[None, :] < d_v),  # 头掩码且V维度在界内 # head mask and V dimension in-bounds
    )
    tl.store(  # 存储输出第3块 # store output block 3
        o_base + offs_h_2d * stride_o_h + offs_v_3[None, :] * stride_o_d,
        acc_3.to(tl.bfloat16),  # 转为bfloat16存储 # cast to bfloat16 for storage
        mask=mask_h_2d & (offs_v_3[None, :] < d_v),  # 头掩码且V维度在界内 # head mask and V dimension in-bounds
    )


# ============================================================================
# Attention Runner Functions
# 注意力运行器函数
# ============================================================================


def run_unified_attention(  # 运行统一稀疏解码注意力 # run unified sparse decode attention
    q_reshaped,  # 重塑后的查询张量 # reshaped query tensor
    gathered_kv,  # 收集后的KV张量 # gathered KV tensor
    invalid_mask,  # 无效掩码张量 # invalid mask tensor
    d_v,  # V维度 # V dimension
    sm_scale,  # softmax缩放因子 # softmax scale factor
    total_tokens,  # 总token数 # total token count
    h_q,  # 查询头数 # number of query heads
    total_topk,  # 总topk数 # total topk count
    d_qk,  # QK维度 # QK dimension
    attn_sink=None,  # 注意力汇（可选） # attention sink (optional)
):
    """Run unified attention with single KV buffer. # 使用单个KV缓冲区运行统一注意力

    Run unified sparse decode attention kernel. # 运行统一稀疏解码注意力内核
    """
    output = torch.empty(  # 分配输出张量 # allocate output tensor
        (total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q_reshaped.device  # 形状：(token数, 头数, V维度) # shape: (tokens, heads, d_v)
    )
    lse = torch.empty(  # 分配lse张量 # allocate lse tensor
        (total_tokens, h_q), dtype=torch.float32, device=q_reshaped.device  # 形状：(token数, 头数) # shape: (tokens, heads)
    )

    HAS_ATTN_SINK = attn_sink is not None  # 是否使用注意力汇 # whether to use attention sink
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1]  # 注意力汇张量（无则用占位） # attention sink tensor (placeholder if absent)

    grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))  # 网格大小：(token数, 头块数) # grid: (tokens, head blocks)
    _unified_sparse_decode_kernel[grid](  # 启动统一稀疏解码内核 # launch unified sparse decode kernel
        q_reshaped,  # 查询张量 # query tensor
        gathered_kv,  # KV张量 # KV tensor
        invalid_mask,  # 无效掩码 # invalid mask
        attn_sink_tensor,  # 注意力汇张量 # attention sink tensor
        output,  # 输出张量 # output tensor
        lse,  # lse张量 # lse tensor
        sm_scale,  # softmax缩放 # softmax scale
        total_tokens,  # 总token数 # total tokens
        _bucket_total_tokens(total_tokens),  # 分桶后的总token数 # bucketed total tokens
        h_q,  # 查询头数 # query heads
        total_topk,  # 总topk数 # total topk
        d_qk,  # QK维度 # QK dimension
        d_v,  # V维度 # V dimension
        q_reshaped.stride(0),  # Q token步长 # Q token stride
        q_reshaped.stride(1),  # Q头步长 # Q head stride
        q_reshaped.stride(2),  # Q维度步长 # Q dimension stride
        gathered_kv.stride(0),  # KV token步长 # KV token stride
        gathered_kv.stride(1),  # KV k步长 # KV k stride
        gathered_kv.stride(2),  # KV维度步长 # KV dimension stride
        invalid_mask.stride(0),  # 掩码token步长 # mask token stride
        invalid_mask.stride(1),  # 掩码k步长 # mask k stride
        output.stride(0),  # 输出token步长 # output token stride
        output.stride(1),  # 输出头步长 # output head stride
        output.stride(2),  # 输出维度步长 # output dimension stride
        lse.stride(0),  # lse token步长 # lse token stride
        lse.stride(1),  # lse头步长 # lse head stride
        HAS_ATTN_SINK=HAS_ATTN_SINK,  # 是否有注意力汇 # whether has attention sink
    )
    return output, lse  # 返回输出和lse # return output and lse


def run_chunked_attention_triton(  # 使用Triton内核运行分块注意力 # run chunked attention using Triton kernels
    q_reshaped,  # 重塑后的查询张量 # reshaped query tensor
    gathered_kv,  # 收集后的KV张量 # gathered KV tensor
    invalid_mask,  # 无效掩码张量 # invalid mask tensor
    d_v,  # V维度 # V dimension
    sm_scale,  # softmax缩放因子 # softmax scale factor
    total_tokens,  # 总token数 # total token count
    h_q,  # 查询头数 # number of query heads
    total_topk,  # 总topk数 # total topk count
    d_qk,  # QK维度 # QK dimension
    attn_sink=None,  # 注意力汇（可选） # attention sink (optional)
    chunk_size=8192,  # 分块大小 # chunk size
):
    """Chunked attention using Triton kernels with cross-chunk softmax merging."""  # 使用Triton内核的分块注意力，支持跨块softmax合并
    device = q_reshaped.device  # 获取设备 # get device

    num_chunks = (total_topk + chunk_size - 1) // chunk_size  # 计算分块数量 # compute number of chunks

    kv_chunks = []  # KV分块列表 # KV chunk list
    mask_chunks = []  # 掩码分块列表 # mask chunk list
    chunk_sizes = []  # 分块大小列表 # chunk size list

    for chunk_idx in range(num_chunks):  # 遍历每个分块 # iterate over each chunk
        start_k = chunk_idx * chunk_size  # 分块起始位置 # chunk start position
        end_k = min(start_k + chunk_size, total_topk)  # 分块结束位置 # chunk end position
        chunk_topk = end_k - start_k  # 当前分块的topk数 # topk count of current chunk
        chunk_sizes.append(chunk_topk)  # 记录分块大小 # record chunk size
        kv_chunks.append(gathered_kv[:, start_k:end_k, :].contiguous())  # 切片KV并保证连续 # slice KV and ensure contiguous
        mask_chunks.append(invalid_mask[:, start_k:end_k].contiguous())  # 切片掩码并保证连续 # slice mask and ensure contiguous

    lse_acc = torch.full(  # 累积lse初始化为负无穷 # accumulated lse initialized to negative infinity
        (total_tokens, h_q), float("-inf"), dtype=torch.float32, device=device
    )
    acc = torch.zeros((total_tokens, h_q, d_v), dtype=torch.float32, device=device)  # 累积输出初始化为零 # accumulated output initialized to zero

    for chunk_idx in range(num_chunks):  # 遍历每个分块 # iterate over each chunk
        kv_chunk = kv_chunks[chunk_idx]  # 当前KV分块 # current KV chunk
        mask_chunk = mask_chunks[chunk_idx]  # 当前掩码分块 # current mask chunk
        chunk_topk = chunk_sizes[chunk_idx]  # 当前分块topk数 # current chunk topk count

        chunk_output, chunk_lse = run_unified_attention(  # 运行当前分块的统一注意力 # run unified attention for current chunk
            q_reshaped,  # 查询张量 # query tensor
            kv_chunk,  # KV分块 # KV chunk
            mask_chunk,  # 掩码分块 # mask chunk
            d_v,  # V维度 # V dimension
            sm_scale,  # softmax缩放 # softmax scale
            total_tokens,  # 总token数 # total tokens
            h_q,  # 查询头数 # query heads
            chunk_topk,  # 当前分块topk数 # current chunk topk
            d_qk,  # QK维度 # QK dimension
            attn_sink=None,  # 分块内不使用注意力汇 # no attention sink within chunk
        )

        is_chunk_lonely = torch.isinf(chunk_lse) & (chunk_lse > 0)  # 当前分块中的孤独查询 # lonely queries in current chunk

        chunk_lse_for_merge = torch.where(  # 孤独查询的lse设为负无穷以便合并 # set lse to -inf for lonely queries for merging
            is_chunk_lonely, torch.full_like(chunk_lse, float("-inf")), chunk_lse
        )

        lse_max = torch.maximum(lse_acc, chunk_lse_for_merge)  # 合并后的最大lse # max lse after merge

        exp_acc = torch.exp(lse_acc - lse_max)  # 旧累积的指数值 # exponential of old accumulated
        exp_acc = torch.where(torch.isnan(exp_acc), torch.zeros_like(exp_acc), exp_acc)  # NaN替换为0 # replace NaN with 0

        exp_chunk = torch.exp(chunk_lse_for_merge - lse_max)  # 当前分块的指数值 # exponential of current chunk
        exp_chunk = torch.where(  # NaN或孤独查询替换为0 # replace NaN or lonely with 0
            torch.isnan(exp_chunk) | is_chunk_lonely,
            torch.zeros_like(exp_chunk),
            exp_chunk,
        )

        sum_exp = exp_acc + exp_chunk  # 指数和 # sum of exponentials
        lse_new = lse_max + torch.log(  # 新的合并lse # new merged lse
            torch.where(sum_exp == 0, torch.ones_like(sum_exp), sum_exp)  # 避免log(0) # avoid log(0)
        )

        both_empty = (lse_acc == float("-inf")) & (chunk_lse_for_merge == float("-inf"))  # 两者都为空 # both are empty
        lse_new = torch.where(  # 两者都空时保持负无穷 # keep -inf when both empty
            both_empty, torch.full_like(lse_new, float("-inf")), lse_new
        )

        weight_acc = torch.exp(lse_acc - lse_new)  # 旧累积的权重 # weight of old accumulated
        weight_acc = torch.where(  # NaN或无穷替换为0 # replace NaN or inf with 0
            torch.isnan(weight_acc) | torch.isinf(weight_acc),
            torch.zeros_like(weight_acc),
            weight_acc,
        )

        weight_chunk = torch.exp(chunk_lse_for_merge - lse_new)  # 当前分块的权重 # weight of current chunk
        weight_chunk = torch.where(  # NaN/无穷/孤独替换为0 # replace NaN/inf/lonely with 0
            torch.isnan(weight_chunk) | torch.isinf(weight_chunk) | is_chunk_lonely,
            torch.zeros_like(weight_chunk),
            weight_chunk,
        )

        acc = (  # 加权合并输出 # weighted merge of outputs
            weight_acc.unsqueeze(-1) * acc  # 旧累积加权 # old accumulated weighted
            + weight_chunk.unsqueeze(-1) * chunk_output.float()  # 新分块加权 # new chunk weighted
        )

        lse_acc = lse_new  # 更新累积lse # update accumulated lse

    output = acc  # 最终输出 # final output
    lse = lse_acc  # 最终lse # final lse

    is_lonely_final = lse == float("-inf")  # 最终的孤独查询 # final lonely queries

    lse = torch.where(is_lonely_final, torch.full_like(lse, float("+inf")), lse)  # 孤独查询lse设为正无穷 # set lse to +inf for lonely queries

    if attn_sink is not None:  # 如果使用注意力汇 # if using attention sink
        attn_sink_expanded = attn_sink.view(1, h_q)  # 扩展注意力汇维度 # expand attention sink dimensions
        exp_diff = torch.exp(attn_sink_expanded - lse)  # 注意力汇与lse的指数差 # exponential diff between sink and lse
        exp_diff = torch.where(  # 孤独查询设为正无穷 # set to +inf for lonely queries
            is_lonely_final, torch.full_like(exp_diff, float("inf")), exp_diff
        )
        scale = 1.0 / (1.0 + exp_diff)  # 缩放因子 = 1/(1+exp_diff) # scale factor = 1/(1+exp_diff)
        output = output * scale.unsqueeze(-1)  # 应用缩放 # apply scaling

    output = torch.where(  # 孤独查询输出设为0 # set output to 0 for lonely queries
        is_lonely_final.unsqueeze(-1), torch.zeros_like(output), output
    )

    return output.to(torch.bfloat16), lse  # 返回bfloat16输出和lse # return bfloat16 output and lse


# ============================================================================
# Helper class and functions for token-range based chunking
# 基于token范围的分块辅助类和函数
# ============================================================================


class SlicedKVScope:  # 切片KV范围类 # Sliced KV scope class
    """A sliced view of KV scope for a specific token range."""  # 特定token范围的KV范围切片视图

    __slots__ = [  # 定义槽位以节省内存 # define slots to save memory
        "blocked_k",  # 块状K张量 # blocked K tensor
        "blocked_k_quantized",  # 量化后的块状K张量 # quantized blocked K tensor
        "indices_in_kvcache",  # KV缓存中的索引 # indices in KV cache
        "topk_length",  # topk有效长度 # valid topk length
    ]

    def __init__(self, blocked_k, blocked_k_quantized, indices_in_kvcache, topk_length):  # 初始化切片KV范围 # initialize sliced KV scope
        self.blocked_k = blocked_k  # 设置块状K # set blocked K
        self.blocked_k_quantized = blocked_k_quantized  # 设置量化块状K # set quantized blocked K
        self.indices_in_kvcache = indices_in_kvcache  # 设置KV缓存索引 # set KV cache indices
        self.topk_length = topk_length  # 设置topk长度 # set topk length


def slice_kv_scope_for_tokens(orig_scope, start_t: int, end_t: int, s_q: int):  # 按token范围切片KV范围 # slice KV scope by token range
    """Slice a KV scope to only include tokens in range [start_t, end_t)."""  # 将KV范围切片为仅包含[start_t, end_t)范围内的token
    if orig_scope is None:  # 如果原始范围为None # if original scope is None
        return None  # 返回None # return None

    orig_indices = orig_scope.indices_in_kvcache.reshape(  # 重塑原始索引为2D # reshape original indices to 2D
        -1, orig_scope.indices_in_kvcache.size(-1)
    )
    sliced_indices = orig_indices[start_t:end_t]  # 切片索引 # slice indices

    sliced_topk_length = None  # 初始化切片topk长度 # initialize sliced topk length
    if orig_scope.topk_length is not None:  # 如果原始topk长度存在 # if original topk length exists
        batch_start = start_t // s_q  # 计算批次起始索引 # compute batch start index
        batch_end = (end_t + s_q - 1) // s_q  # 计算批次结束索引 # compute batch end index
        batch_topk_length = orig_scope.topk_length[batch_start:batch_end]  # 切片批次topk长度 # slice batch topk length
        if s_q > 1:  # 如果序列长度大于1 # if sequence length > 1
            chunk_tokens = end_t - start_t  # 分块token数 # chunk token count
            expanded = batch_topk_length.unsqueeze(1).expand(-1, s_q).reshape(-1)  # 扩展topk长度到每个token # expand topk length to each token
            offset_in_first_batch = start_t % s_q  # 第一个批次内的偏移 # offset within first batch
            sliced_topk_length = expanded[  # 切片topk长度 # slice topk length
                offset_in_first_batch : offset_in_first_batch + chunk_tokens
            ]
        else:  # 序列长度为1 # sequence length is 1
            sliced_topk_length = batch_topk_length  # 直接使用批次topk长度 # directly use batch topk length

    return SlicedKVScope(  # 返回切片后的KV范围 # return sliced KV scope
        blocked_k=orig_scope.blocked_k,  # 保留原始块状K # keep original blocked K
        blocked_k_quantized=orig_scope.blocked_k_quantized,  # 保留原始量化块状K # keep original quantized blocked K
        indices_in_kvcache=sliced_indices,  # 使用切片后的索引 # use sliced indices
        topk_length=sliced_topk_length,  # 使用切片后的topk长度 # use sliced topk length
    )


def compute_token_ranges(  # 计算token处理范围 # compute token processing ranges
    total_tokens: int,  # 总token数 # total token count
    total_topk: int,  # 总topk数 # total topk count
    d_qk: int,  # QK维度 # QK dimension
    max_buffer_bytes: int = 2 * 1024 * 1024 * 1024,  # 最大缓冲区字节数（默认2GB） # max buffer bytes (default 2GB)
) -> List[Tuple[int, int]]:  # 返回(token起始, token结束)列表 # return list of (token_start, token_end)
    """Compute token ranges for processing, chunking if buffer would exceed limit."""  # 计算处理的token范围，如果缓冲区超过限制则分块
    buffer_size_bytes = total_tokens * total_topk * d_qk * 2  # 计算所需缓冲区大小（字节） # compute required buffer size in bytes

    if buffer_size_bytes <= max_buffer_bytes:  # 如果不超过限制 # if within limit
        return [(0, total_tokens)]  # 返回单个范围 # return single range

    max_tokens_per_chunk = max_buffer_bytes // (total_topk * d_qk * 2)  # 每个分块的最大token数 # max tokens per chunk
    chunk_size = max(1, max_tokens_per_chunk)  # 确保至少为1 # ensure at least 1

    token_ranges = []  # token范围列表 # token range list
    start_t = 0  # 起始token # start token
    while start_t < total_tokens:  # 遍历所有token # iterate over all tokens
        end_t = min(start_t + chunk_size, total_tokens)  # 结束token # end token
        token_ranges.append((start_t, end_t))  # 添加范围 # append range
        start_t = end_t  # 更新起始 # update start

    return token_ranges  # 返回token范围列表 # return token range list


# ============================================================================
# Split-K Attention for Large TopK
# 用于大TopK的Split-K注意力
# ============================================================================
def run_splitk_unified_attention(  # 运行Split-K统一注意力（用于大topk场景） # run split-K unified attention (for large topk cases)
    q_reshaped,  # 重塑后的查询张量 # reshaped query tensor
    gathered_kv,  # 收集后的KV张量 # gathered KV tensor
    invalid_mask,  # 无效掩码张量 # invalid mask tensor
    d_v,  # V维度 # V dimension
    sm_scale,  # softmax缩放因子 # softmax scale factor
    total_tokens,  # 总token数 # total token count
    h_q,  # 查询头数 # number of query heads
    total_topk,  # 总topk数 # total topk count
    d_qk,  # QK维度 # QK dimension
    attn_sink=None,  # 注意力汇（可选） # attention sink (optional)
    split_k=4,  # Split-K的分割数 # number of splits for split-K
):
    """Run split-K attention for large topk cases."""  # 为大topk场景运行Split-K注意力
    from .triton_mla_kernels_decode_splitk import run_splitk_attention  # 延迟导入split-k注意力函数 # lazy import split-k attention function

    return run_splitk_attention(  # 调用split-k注意力 # call split-k attention
        q_reshaped,  # 查询张量 # query tensor
        gathered_kv,  # KV张量 # KV tensor
        invalid_mask,  # 无效掩码 # invalid mask
        d_v,  # V维度 # V dimension
        sm_scale,  # softmax缩放 # softmax scale
        total_tokens,  # 总token数 # total tokens
        h_q,  # 查询头数 # query heads
        total_topk,  # 总topk数 # total topk
        d_qk,  # QK维度 # QK dimension
        attn_sink=attn_sink,  # 注意力汇 # attention sink
        split_k=split_k,  # 分割数 # number of splits
    )
