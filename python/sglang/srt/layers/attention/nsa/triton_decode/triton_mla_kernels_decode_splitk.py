# 大 TopK 情况的 Split-K 注意力内核模块
# 本模块实现了注意力内核的 Split-K 版本：
# 1. 将 K (topk) 维度分割到多个内核实例
# 2. 每个实例用自己的 m_i, l_i 和累加器计算部分结果
# 3. 合并内核使用在线 softmax 合并部分结果
# 这减少了寄存器压力，改善了大型 topk 情况的占用率和整体性能
"""
Split-K Attention Kernel for Large TopK Cases # 大 TopK 情况的 Split-K 注意力内核

This module implements a split-K version of the attention kernel that: # 本模块实现了注意力内核的 Split-K 版本：
1. Splits the K (topk) dimension across multiple kernel instances # 1. 将 K (topk) 维度分割到多个内核实例
2. Each instance computes partial results with its own m_i, l_i, and accumulators # 2. 每个实例用自己的 m_i, l_i 和累加器计算部分结果
3. A combine kernel merges the partial results using online softmax # 3. 合并内核使用在线 softmax 合并部分结果

This reduces register pressure by processing fewer K tokens per kernel instance, # 通过每个内核实例处理更少的 K 令牌来减少寄存器压力，
improving occupancy and overall performance for large topk cases. # 改善大型 topk 情况的占用率和整体性能
"""

from typing import Optional, Tuple # 导入类型提示

import torch # 导入 PyTorch
import triton # 导入 Triton
import triton.language as tl # 导入 Triton 语言

from .triton_mla_kernels_decode_common import _bucket_total_tokens # 导入分桶函数


# ============================================================================
# Split-K Attention Kernel # Split-K 注意力内核
# ============================================================================
@triton.autotune( # Triton 自动调优装饰器
    configs=[ # 配置列表
        # Split-K attention on already-gathered BF16 KV. # 对已收集的 BF16 KV 进行 Split-K 注意力
        # - BLOCK_N=256: amortizes memory access over KV tokens (memory-bound kernel). # BLOCK_N=256：在 KV 令牌上分摊内存访问（内存受限内核）
        # - BLOCK_D=128: matches KV tile structure. # BLOCK_D=128：匹配 KV tile 结构
        # - num_warps=8, num_stages=2: memory-bound kernel benefits from more warps # num_warps=8, num_stages=2：内存受限内核从更多 warp 受益
        #   and software pipelining (overlaps memory loads with compute). # 和软件流水线（重叠内存加载与计算）
        # - BLOCK_H varies for different batch sizes: # BLOCK_H 随批次大小变化
        triton.Config( # 配置
            {"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2 # BLOCK_H=16
        ),
        triton.Config( # 配置
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2 # BLOCK_H=32
        ),
        triton.Config( # 配置
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2 # BLOCK_H=64
        ),
        triton.Config( # 配置
            {"BLOCK_H": 128, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2 # BLOCK_H=128
        ),
    ],
    key=["total_tokens_bucket", "h_q", "topk_per_split", "d_qk"], # 自动调优键
)
@triton.jit # Triton JIT 编译装饰器
def _splitk_attention_kernel( # Split-K 注意力内核：处理 K 令牌子集
    Q, # 查询张量
    KV, # KV 张量
    Mask, # 无效掩码
    PartialOutput, # 部分输出
    PartialLSE, # 部分对数求和指数
    PartialM, # 部分最大值
    sm_scale, # softmax 缩放因子
    total_tokens, # 总令牌数
    total_tokens_bucket, # 分桶后的令牌数
    h_q, # 查询头数
    total_topk, # 总 topk 数
    d_qk, # QK 维度
    d_v, # 值维度
    topk_per_split, # 每次分割的 topk 数
    stride_q_t, # 查询令牌步长
    stride_q_h, # 查询头步长
    stride_q_d, # 查询维度步长
    stride_kv_t, # KV 令牌步长
    stride_kv_k, # KV topk 步长
    stride_kv_d, # KV 维度步长
    stride_mask_t, # 掩码令牌步长
    stride_mask_k, # 掩码 topk 步长
    stride_po_s, # 部分输出分割步长
    stride_po_t, # 部分输出令牌步长
    stride_po_h, # 部分输出头步长
    stride_po_d, # 部分输出维度步长
    stride_plse_s, # 部分LSE分割步长
    stride_plse_t, # 部分LSE令牌步长
    stride_plse_h, # 部分LSE头步长
    stride_pm_s, # 部分M分割步长
    stride_pm_t, # 部分M令牌步长
    stride_pm_h, # 部分M头步长
    BLOCK_H: tl.constexpr, # 头维度块大小
    BLOCK_N: tl.constexpr, # K维度块大小
    BLOCK_D: tl.constexpr, # 维度块大小
):
    """Split-K attention kernel that processes a subset of K tokens.""" # Split-K 注意力内核，处理 K 令牌的子集
    LOG2E: tl.constexpr = 1.4426950408889634 # log2(e) 常量

    pid_t = tl.program_id(0) # 令牌维度程序 ID
    pid_h = tl.program_id(1) # 头维度程序 ID
    pid_k = tl.program_id(2) # K 分割维度程序 ID
    pid_t_64 = pid_t.to(tl.int64) # 转换为 int64

    NEG_INF = float("-inf") # 负无穷

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H) # 头维度偏移
    mask_h = offs_h < h_q # 头维度越界掩码

    # Compute K range for this split # 计算本次分割的 K 范围
    k_start = pid_k * topk_per_split # K 起始位置
    k_end = tl.minimum(k_start + topk_per_split, total_topk) # K 结束位置

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32) # 在线 softmax 最大值
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32) # 在线 softmax 累加和

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 累加器0
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 累加器1
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 累加器2
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 累加器3

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64) # 查询步长转 int64
    stride_kv_t_64 = tl.cast(stride_kv_t, tl.int64) # KV 步长转 int64
    stride_mask_t_64 = tl.cast(stride_mask_t, tl.int64) # 掩码步长转 int64
    q_base = Q + pid_t_64 * stride_q_t_64 # 查询基地址
    kv_base = KV + pid_t_64 * stride_kv_t_64 # KV 基地址
    mask_base = Mask + pid_t_64 * stride_mask_t_64 # 掩码基地址

    for n_start in range(k_start, k_end, BLOCK_N): # 遍历 K 块
        offs_n = n_start + tl.arange(0, BLOCK_N) # K 偏移
        mask_n = offs_n < k_end # K 越界掩码

        mask_ptrs = mask_base + offs_n * stride_mask_k # 掩码指针
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True) # 加载无效标记
        valid = mask_n & ~invalid # 有效掩码

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32) # QK 点积

        for d_start in range(0, d_qk, BLOCK_D): # 遍历维度块
            offs_d = d_start + tl.arange(0, BLOCK_D) # 维度偏移
            mask_d = offs_d < d_qk # 维度越界掩码

            q_ptrs = ( # 查询指针
                q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d # 头×维度
            )
            q_chunk = tl.load( # 加载查询块
                q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0 # 带掩码
            ).to(tl.bfloat16) # 转为 BF16

            k_ptrs = ( # 键指针
                kv_base + offs_n[:, None] * stride_kv_k + offs_d[None, :] * stride_kv_d # topk×维度
            )
            k_chunk = tl.load( # 加载键块
                k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0 # 带掩码
            ).to(tl.bfloat16) # 转为 BF16

            qk += tl.dot(q_chunk, tl.trans(k_chunk)) # 累加 QK 点积

        qk = qk * sm_scale # 应用 softmax 缩放
        qk = tl.where(valid[None, :], qk, NEG_INF) # 无效位置设为负无穷

        m_ij = tl.max(qk, axis=1) # 当前块最大值
        m_new = tl.maximum(m_i, m_ij) # 更新全局最大值
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E)) # 修正因子
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E)) # 概率
        l_new = alpha * l_i + tl.sum(p, axis=1) # 更新归一化因子
        p_bf16 = p.to(tl.bfloat16) # 转为 BF16

        offs_v = tl.arange(0, BLOCK_D) # 值维度偏移0
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d # 值指针
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16) # 加载值
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v) # 累加器0更新

        offs_v = BLOCK_D + tl.arange(0, BLOCK_D) # 值维度偏移1
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d # 值指针
        v = tl.load( # 加载值
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0 # 带维度越界检查
        ).to(tl.bfloat16) # 转为 BF16
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v) # 累加器1更新

        offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D) # 值维度偏移2
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d # 值指针
        v = tl.load( # 加载值
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0 # 带维度越界检查
        ).to(tl.bfloat16) # 转为 BF16
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v) # 累加器2更新

        offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D) # 值维度偏移3
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d # 值指针
        v = tl.load( # 加载值
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0 # 带维度越界检查
        ).to(tl.bfloat16) # 转为 BF16
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v) # 累加器3更新

        m_i = m_new # 更新最大值
        l_i = l_new # 更新归一化因子

    # Store partial results # 存储部分结果
    stride_po_s_64 = tl.cast(stride_po_s, tl.int64) # 输出分割步长转 int64
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64) # 输出令牌步长转 int64
    po_base = PartialOutput + pid_k * stride_po_s_64 + pid_t_64 * stride_po_t_64 # 部分输出基地址

    offs_h_2d = offs_h[:, None] # 头偏移2D
    mask_h_2d = mask_h[:, None] # 头掩码2D
    offs_v_0 = tl.arange(0, BLOCK_D) # 值偏移0
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D) # 值偏移1
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D) # 值偏移2
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D) # 值偏移3

    tl.store( # 存储部分输出0
        po_base + offs_h_2d * stride_po_h + offs_v_0[None, :] * stride_po_d, # 指针
        acc_0, # 数据
        mask=mask_h_2d, # 掩码
    )
    tl.store( # 存储部分输出1
        po_base + offs_h_2d * stride_po_h + offs_v_1[None, :] * stride_po_d, # 指针
        acc_1, # 数据
        mask=mask_h_2d & (offs_v_1[None, :] < d_v), # 掩码（带维度检查）
    )
    tl.store( # 存储部分输出2
        po_base + offs_h_2d * stride_po_h + offs_v_2[None, :] * stride_po_d, # 指针
        acc_2, # 数据
        mask=mask_h_2d & (offs_v_2[None, :] < d_v), # 掩码（带维度检查）
    )
    tl.store( # 存储部分输出3
        po_base + offs_h_2d * stride_po_h + offs_v_3[None, :] * stride_po_d, # 指针
        acc_3, # 数据
        mask=mask_h_2d & (offs_v_3[None, :] < d_v), # 掩码（带维度检查）
    )

    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64) # LSE分割步长转 int64
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64) # LSE令牌步长转 int64
    plse_ptrs = ( # 部分LSE指针
        PartialLSE # 部分LSE基地址
        + pid_k * stride_plse_s_64 # 分割偏移
        + pid_t_64 * stride_plse_t_64 # 令牌偏移
        + offs_h * stride_plse_h # 头偏移
    )
    tl.store(plse_ptrs, l_i, mask=mask_h) # 存储部分 LSE

    stride_pm_s_64 = tl.cast(stride_pm_s, tl.int64) # M分割步长转 int64
    stride_pm_t_64 = tl.cast(stride_pm_t, tl.int64) # M令牌步长转 int64
    pm_ptrs = ( # 部分M指针
        PartialM # 部分M基地址
        + pid_k * stride_pm_s_64 # 分割偏移
        + pid_t_64 * stride_pm_t_64 # 令牌偏移
        + offs_h * stride_pm_h # 头偏移
    )
    tl.store(pm_ptrs, m_i, mask=mask_h) # 存储部分最大值


# ============================================================================
# Combine Kernel for Split-K # Split-K 合并内核
# ============================================================================
@triton.autotune( # Triton 自动调优装饰器
    configs=[ # 配置列表
        # Simple reduce kernel merging split-K results. # 简单归约内核，合并 split-K 结果
        # - BLOCK_D=128: 4 iterations to cover d_v=512. # BLOCK_D=128：4次迭代覆盖 d_v=512
        # - num_warps=4: sufficient for this simple reduce operation. # num_warps=4：对于简单归约操作足够
        # - BLOCK_H varies for different batch sizes: # BLOCK_H 随批次大小变化
        triton.Config({"BLOCK_H": 16, "BLOCK_D": 128}, num_warps=4, num_stages=1), # BLOCK_H=16
        triton.Config({"BLOCK_H": 32, "BLOCK_D": 128}, num_warps=4, num_stages=1), # BLOCK_H=32
        triton.Config({"BLOCK_H": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1), # BLOCK_H=64
    ],
    key=["total_tokens_bucket", "h_q", "split_k"], # 自动调优键
)
@triton.jit # Triton JIT 编译装饰器
def _combine_splitk_attention_kernel( # Split-K 合并注意力内核：合并部分结果
    PartialOutput, # 部分输出
    PartialLSE, # 部分对数求和指数
    PartialM, # 部分最大值
    AttnSink, # 注意力汇聚
    Output, # 最终输出
    LSE, # 最终 LSE
    total_tokens, # 总令牌数
    total_tokens_bucket, # 分桶后的令牌数
    h_q, # 查询头数
    d_v, # 值维度
    split_k, # 分割数
    stride_po_s, # 部分输出分割步长
    stride_po_t, # 部分输出令牌步长
    stride_po_h, # 部分输出头步长
    stride_po_d, # 部分输出维度步长
    stride_plse_s, # 部分LSE分割步长
    stride_plse_t, # 部分LSE令牌步长
    stride_plse_h, # 部分LSE头步长
    stride_pm_s, # 部分M分割步长
    stride_pm_t, # 部分M令牌步长
    stride_pm_h, # 部分M头步长
    stride_o_t, # 输出令牌步长
    stride_o_h, # 输出头步长
    stride_o_d, # 输出维度步长
    stride_lse_t, # LSE令牌步长
    stride_lse_h, # LSE头步长
    HAS_ATTN_SINK: tl.constexpr, # 是否有注意力汇聚
    BLOCK_H: tl.constexpr, # 头维度块大小
    BLOCK_D: tl.constexpr, # 维度块大小
):
    """Combine partial results from split-K attention kernel.""" # 合并来自 split-K 注意力内核的部分结果
    LOG2E: tl.constexpr = 1.4426950408889634 # log2(e) 常量
    NEG_INF = float("-inf") # 负无穷
    POS_INF = float("+inf") # 正无穷

    pid_t = tl.program_id(0) # 令牌维度程序 ID
    pid_h = tl.program_id(1) # 头维度程序 ID
    pid_t_64 = pid_t.to(tl.int64) # 转换为 int64

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H) # 头维度偏移
    mask_h = offs_h < h_q # 头维度越界掩码

    m_acc = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32) # 合并后的最大值
    l_acc = tl.zeros([BLOCK_H], dtype=tl.float32) # 合并后的归一化因子

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 合并累加器0
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 合并累加器1
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 合并累加器2
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32) # 合并累加器3

    stride_po_s_64 = tl.cast(stride_po_s, tl.int64) # 输出分割步长转 int64
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64) # 输出令牌步长转 int64
    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64) # LSE分割步长转 int64
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64) # LSE令牌步长转 int64
    stride_pm_s_64 = tl.cast(stride_pm_s, tl.int64) # M分割步长转 int64
    stride_pm_t_64 = tl.cast(stride_pm_t, tl.int64) # M令牌步长转 int64

    offs_h_2d = offs_h[:, None] # 头偏移2D
    mask_h_2d = mask_h[:, None] # 头掩码2D
    offs_v_0 = tl.arange(0, BLOCK_D) # 值偏移0
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D) # 值偏移1
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D) # 值偏移2
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D) # 值偏移3

    for k in range(split_k): # 遍历每个分割
        k_64 = tl.cast(k, tl.int64) # 转为 int64
        po_base = PartialOutput + k_64 * stride_po_s_64 + pid_t_64 * stride_po_t_64 # 部分输出基地址

        p_acc_0 = tl.load( # 加载部分输出0
            po_base + offs_h_2d * stride_po_h + offs_v_0[None, :] * stride_po_d, # 指针
            mask=mask_h_2d, # 掩码
            other=0.0, # 默认值
        )
        p_acc_1 = tl.load( # 加载部分输出1
            po_base + offs_h_2d * stride_po_h + offs_v_1[None, :] * stride_po_d, # 指针
            mask=mask_h_2d & (offs_v_1[None, :] < d_v), # 掩码（带维度检查）
            other=0.0, # 默认值
        )
        p_acc_2 = tl.load( # 加载部分输出2
            po_base + offs_h_2d * stride_po_h + offs_v_2[None, :] * stride_po_d, # 指针
            mask=mask_h_2d & (offs_v_2[None, :] < d_v), # 掩码（带维度检查）
            other=0.0, # 默认值
        )
        p_acc_3 = tl.load( # 加载部分输出3
            po_base + offs_h_2d * stride_po_h + offs_v_3[None, :] * stride_po_d, # 指针
            mask=mask_h_2d & (offs_v_3[None, :] < d_v), # 掩码（带维度检查）
            other=0.0, # 默认值
        )

        plse_ptrs = ( # 部分LSE指针
            PartialLSE # 部分LSE基地址
            + k_64 * stride_plse_s_64 # 分割偏移
            + pid_t_64 * stride_plse_t_64 # 令牌偏移
            + offs_h * stride_plse_h # 头偏移
        )
        p_l = tl.load(plse_ptrs, mask=mask_h, other=0.0) # 加载部分 LSE

        pm_ptrs = ( # 部分M指针
            PartialM # 部分M基地址
            + k_64 * stride_pm_s_64 # 分割偏移
            + pid_t_64 * stride_pm_t_64 # 令牌偏移
            + offs_h * stride_pm_h # 头偏移
        )
        p_m = tl.load(pm_ptrs, mask=mask_h, other=NEG_INF) # 加载部分最大值

        m_new = tl.maximum(m_acc, p_m) # 更新全局最大值
        alpha_acc = tl.where( # 累加器修正因子
            m_acc == NEG_INF, 0.0, tl.math.exp2((m_acc - m_new) * LOG2E) # 基于 softmax 的修正
        )
        alpha_p = tl.where(p_m == NEG_INF, 0.0, tl.math.exp2((p_m - m_new) * LOG2E)) # 部分结果修正因子
        l_new = alpha_acc * l_acc + alpha_p * p_l # 更新归一化因子

        acc_0 = acc_0 * alpha_acc[:, None] + p_acc_0 * alpha_p[:, None] # 合并累加器0
        acc_1 = acc_1 * alpha_acc[:, None] + p_acc_1 * alpha_p[:, None] # 合并累加器1
        acc_2 = acc_2 * alpha_acc[:, None] + p_acc_2 * alpha_p[:, None] # 合并累加器2
        acc_3 = acc_3 * alpha_acc[:, None] + p_acc_3 * alpha_p[:, None] # 合并累加器3

        m_acc = m_new # 更新最大值
        l_acc = l_new # 更新归一化因子

    lse = m_acc + tl.math.log2(tl.where(l_acc == 0.0, 1.0, l_acc)) / LOG2E # 计算 LSE
    is_lonely_q = l_acc == 0.0 # 孤立查询标记（无有效 KV）

    if HAS_ATTN_SINK: # 如果有注意力汇聚
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0) # 加载汇聚值
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_acc) * LOG2E) # 汇聚指数
        denominator = l_acc + exp_attn_sink_minus_m # 分母
        denominator = tl.where(denominator == 0.0, 1.0, denominator) # 避免除零
        output_scale = 1.0 / denominator # 输出缩放
    else: # 无注意力汇聚
        output_scale = tl.where(l_acc == 0.0, 0.0, 1.0 / l_acc) # 标准缩放

    is_lonely_q_2d = is_lonely_q[:, None] # 孤立查询2D
    output_scale_2d = output_scale[:, None] # 输出缩放2D
    acc_0 = tl.where(is_lonely_q_2d, 0.0, acc_0 * output_scale_2d) # 应用缩放到累加器0
    acc_1 = tl.where(is_lonely_q_2d, 0.0, acc_1 * output_scale_2d) # 应用缩放到累加器1
    acc_2 = tl.where(is_lonely_q_2d, 0.0, acc_2 * output_scale_2d) # 应用缩放到累加器2
    acc_3 = tl.where(is_lonely_q_2d, 0.0, acc_3 * output_scale_2d) # 应用缩放到累加器3
    lse = tl.where(is_lonely_q, POS_INF, lse) # 孤立查询 LSE 设为正无穷

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64) # 输出令牌步长转 int64
    o_base = Output + pid_t_64 * stride_o_t_64 # 输出基地址

    tl.store( # 存储输出0
        o_base + offs_h_2d * stride_o_h + offs_v_0[None, :] * stride_o_d, # 指针
        acc_0.to(tl.bfloat16), # 数据（转为BF16）
        mask=mask_h_2d, # 掩码
    )
    tl.store( # 存储输出1
        o_base + offs_h_2d * stride_o_h + offs_v_1[None, :] * stride_o_d, # 指针
        acc_1.to(tl.bfloat16), # 数据（转为BF16）
        mask=mask_h_2d & (offs_v_1[None, :] < d_v), # 掩码（带维度检查）
    )
    tl.store( # 存储输出2
        o_base + offs_h_2d * stride_o_h + offs_v_2[None, :] * stride_o_d, # 指针
        acc_2.to(tl.bfloat16), # 数据（转为BF16）
        mask=mask_h_2d & (offs_v_2[None, :] < d_v), # 掩码（带维度检查）
    )
    tl.store( # 存储输出3
        o_base + offs_h_2d * stride_o_h + offs_v_3[None, :] * stride_o_d, # 指针
        acc_3.to(tl.bfloat16), # 数据（转为BF16）
        mask=mask_h_2d & (offs_v_3[None, :] < d_v), # 掩码（带维度检查）
    )

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64) # LSE令牌步长转 int64
    tl.store(LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h, lse, mask=mask_h) # 存储 LSE


# ============================================================================
# Runner Function # 运行函数
# ============================================================================
def run_splitk_attention( # 运行 Split-K 注意力内核
    q_reshaped: torch.Tensor, # 重塑后的查询张量
    gathered_kv: torch.Tensor, # 收集后的 KV
    invalid_mask: torch.Tensor, # 无效掩码
    d_v: int, # 值维度
    sm_scale: float, # softmax 缩放因子
    total_tokens: int, # 总令牌数
    h_q: int, # 查询头数
    total_topk: int, # 总 topk 数
    d_qk: int, # QK 维度
    attn_sink: Optional[torch.Tensor] = None, # 可选的注意力汇聚
    split_k: int = 4, # 分割数，默认4
) -> Tuple[torch.Tensor, torch.Tensor]: # 返回输出和 LSE
    """Run split-K attention kernel.""" # 运行 Split-K 注意力内核
    device = q_reshaped.device # 获取设备

    topk_per_split = (total_topk + split_k - 1) // split_k # 计算每次分割的 topk 数

    partial_output = torch.empty( # 分配部分输出缓冲区
        split_k, total_tokens, h_q, d_v, dtype=torch.float32, device=device # float32 类型
    )
    partial_lse = torch.empty( # 分配部分 LSE 缓冲区
        split_k, total_tokens, h_q, dtype=torch.float32, device=device # float32 类型
    )
    partial_m = torch.empty( # 分配部分最大值缓冲区
        split_k, total_tokens, h_q, dtype=torch.float32, device=device # float32 类型
    )

    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device) # 最终输出
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device) # 最终 LSE

    grid_splitk = lambda meta: ( # Split-K 网格大小
        total_tokens, # 令牌维度
        triton.cdiv(h_q, meta["BLOCK_H"]), # 头维度
        split_k, # 分割维度
    )
    _splitk_attention_kernel[grid_splitk]( # 启动 Split-K 注意力内核
        q_reshaped, # 查询
        gathered_kv, # 收集的 KV
        invalid_mask, # 无效掩码
        partial_output, # 部分输出
        partial_lse, # 部分 LSE
        partial_m, # 部分最大值
        sm_scale, # softmax 缩放
        total_tokens, # 总令牌数
        _bucket_total_tokens(total_tokens), # 分桶令牌数
        h_q, # 头数
        total_topk, # 总 topk
        d_qk, # QK 维度
        d_v, # 值维度
        topk_per_split, # 每次分割的 topk
        q_reshaped.stride(0), # 查询令牌步长
        q_reshaped.stride(1), # 查询头步长
        q_reshaped.stride(2), # 查询维度步长
        gathered_kv.stride(0), # KV 令牌步长
        gathered_kv.stride(1), # KV topk 步长
        gathered_kv.stride(2), # KV 维度步长
        invalid_mask.stride(0), # 掩码令牌步长
        invalid_mask.stride(1), # 掩码 topk 步长
        partial_output.stride(0), # 部分输出分割步长
        partial_output.stride(1), # 部分输出令牌步长
        partial_output.stride(2), # 部分输出头步长
        partial_output.stride(3), # 部分输出维度步长
        partial_lse.stride(0), # 部分 LSE 分割步长
        partial_lse.stride(1), # 部分 LSE 令牌步长
        partial_lse.stride(2), # 部分 LSE 头步长
        partial_m.stride(0), # 部分M分割步长
        partial_m.stride(1), # 部分M令牌步长
        partial_m.stride(2), # 部分M头步长
    )

    HAS_ATTN_SINK = attn_sink is not None # 是否有注意力汇聚
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1] # 汇聚张量或占位

    grid_combine = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"])) # 合并网格大小
    _combine_splitk_attention_kernel[grid_combine]( # 启动合并内核
        partial_output, # 部分输出
        partial_lse, # 部分 LSE
        partial_m, # 部分最大值
        attn_sink_tensor, # 注意力汇聚
        output, # 最终输出
        lse, # 最终 LSE
        total_tokens, # 总令牌数
        _bucket_total_tokens(total_tokens), # 分桶令牌数
        h_q, # 头数
        d_v, # 值维度
        split_k, # 分割数
        partial_output.stride(0), # 部分输出分割步长
        partial_output.stride(1), # 部分输出令牌步长
        partial_output.stride(2), # 部分输出头步长
        partial_output.stride(3), # 部分输出维度步长
        partial_lse.stride(0), # 部分 LSE 分割步长
        partial_lse.stride(1), # 部分 LSE 令牌步长
        partial_lse.stride(2), # 部分 LSE 头步长
        partial_m.stride(0), # 部分M分割步长
        partial_m.stride(1), # 部分M令牌步长
        partial_m.stride(2), # 部分M头步长
        output.stride(0), # 输出令牌步长
        output.stride(1), # 输出头步长
        output.stride(2), # 输出维度步长
        lse.stride(0), # LSE令牌步长
        lse.stride(1), # LSE头步长
        HAS_ATTN_SINK=HAS_ATTN_SINK, # 是否有注意力汇聚
    )

    return output, lse # 返回输出和 LSE
