# 文件说明：absorbed-MLA路径下kv_b_proj的LoRA校正Triton内核
# 本文件实现了在MLA（Multi-head Latent Attention）吸收路径中，对kv_b_proj进行LoRA校正的四个Triton内核。
# 当LoRA适配器在kv_b_proj上激活时，通过分解矩阵乘法（不显式构造B@A），手动将LoRA增量加到q_nope_out和attn_bmm_output上。
# 四个内核分别为：step_a_q_fwd、step_b_q_fwd、step_a_v_fwd、step_b_v_fwd。

"""Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.

The absorbed-MLA path bypasses ``kv_b_proj.forward()`` and folds the K/V
sides as plain BMMs ``q_nope @ w_kc`` and ``attn_output @ w_vc``.  When a
LoRA adapter is active on ``kv_b_proj`` we add the LoRA delta to
``q_nope_out`` / ``attn_bmm_output`` manually.

Using the standard LoRA factored math we *never* materialize ``B @ A``:

    q_correction = q_nope     @ B_kc @ A   * scaling          # K-side
    v_correction = attn_output @ A.T @ B_vc.T * scaling       # V-side

where ``A: (slot, rank, kv_lora_rank)`` is the LoRA-A of ``kv_b_proj``
(shared across heads) and ``B: (slot, num_heads*(qk_nope+v_head_dim), rank)``
is the LoRA-B; ``B_kc`` / ``B_vc`` are its K-half / V-half slices.

Four kernels split the math along the factorization boundary, all using
the SGMM idiom from ``sgemm_lora_a`` / ``qkv_lora_b`` and the segment-indptr
routing used by ``chunked_sgmv_*``:

  * ``step_a_q_fwd``: per-head per-slot SGMM, ``(S,H,qk_nope) -> (S,H,rank)``
  * ``step_b_q_fwd``: shared-A per-slot SGMM, scaled+accumulated,
    ``(S,H,rank) -> (S,H,kv_lora_rank)``
  * ``step_a_v_fwd``: shared-A.T per-slot SGMM, ``(S,H,kv_lora_rank) -> (S,H,rank)``
  * ``step_b_v_fwd``: per-head per-slot SGMM with V-half of B, transposed,
    scaled+accumulated, ``(S,H,rank) -> (S,H,v_head_dim)``

Grid axes for each kernel:
  axis 0 : output tile in (S, N)         -- tile_id = pid_s * num_pid_n + pid_n
  axis 1 : head_id                       -- per-head weight slice
  axis 2 : batch_id (segment / request)  -- per-slot weight routing via weight_indices

Per-segment routing: each program derives its segment length from
``seg_indptr[segment_id + 1] - seg_indptr[segment_id]``, loads
``weight_indices[segment_id]`` once, and uses that slot's slice of the LoRA
weight stack.  When ``permutation`` is present, rows are routed through it,
matching the csgmv backend's adapter-grouped chunks.  No Python loops over slots
or heads.

The math also stays in the input dtype (no fp32 round-trip) -- the
contraction dim ``rank`` is small (typically 16-64), so bf16 accumulation
over it is acceptable.  ``tl.dot`` itself uses fp32 accumulation internally.
"""

from __future__ import annotations  # 启用延迟类型注解求值

import torch  # 导入PyTorch张量库
import triton  # 导入Triton JIT编译框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions  # 导入token位置解析工具
from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批量信息数据类

# ---------------------------------------------------------------------------
# Block sizes -- chosen per-kernel from the natural shape of each step.
# 块大小——根据每个步骤的自然形状为每个内核选择。
#
# The factored math gives the four kernels these contraction (K) and output
# (N) ranges (for Kimi-K2.5: rank=16-32, qk_nope=v_head_dim=128, kv_lora_rank=512):
# 分解数学给出四个内核的收缩(K)和输出(N)范围（以Kimi-K2.5为例: rank=16-32, qk_nope=v_head_dim=128, kv_lora_rank=512）:
#
#                              K (contraction)      N (output)
#                              K (收缩维度)          N (输出维度)
#   step_a_q                   qk_nope (~128)       rank (~16-32)
#   step_b_q                   rank (~16-32)        kv_lora_rank (~512)
#   step_a_v                   kv_lora_rank (~512)  rank (~16-32)
#   step_b_v                   rank (~16-32)        v_head_dim (~128)
#
# So the "step_a_*" kernels want a large BLOCK_K (to keep loop iters small)
# and a small BLOCK_N (matched to rank to avoid wasted tile lanes), while
# the "step_b_*" kernels are the inverse.  Kernels aren't autotuned -- the
# decode-shape workload is too small to benefit and the sweep surface is
# wide.
# 因此"step_a_*"内核需要大的BLOCK_K（减少循环迭代次数）和小的BLOCK_N（匹配rank避免浪费），
# 而"step_b_*"内核则相反。内核不进行自动调优——解码形状的工作负载太小，不值得调优。
# ---------------------------------------------------------------------------

_BLOCK_S = 16  # 序列维度的块大小
_STEP_A_BLOCK_K = 64  # step_a内核的收缩维度块大小，覆盖qk_nope(~128)或kv_lora_rank(~512)
_STEP_A_BLOCK_N = 16  # step_a内核的输出维度块大小，输出为rank
_STEP_B_BLOCK_K = 16  # step_b内核的收缩维度块大小，收缩维度为rank
_STEP_B_BLOCK_N = 64  # step_b内核的输出维度块大小，输出为kv_lora_rank(~512)或v_head_dim(~128)


def _num_segments(batch_info: LoRABatchInfo) -> int:  # 获取批量中的段数
    return batch_info.num_segments or batch_info.bs  # 优先使用num_segments，否则使用bs


def _max_segment_len(batch_info: LoRABatchInfo) -> int:  # 获取最大段长度
    if batch_info.max_len is not None:  # 如果max_len已指定
        return batch_info.max_len  # 直接返回
    if batch_info.seg_lens is not None:  # 如果seg_lens已指定
        return int(batch_info.seg_lens.max().item())  # 返回seg_lens中的最大值
    raise ValueError("LoRA batch_info must provide max_len or seg_lens.")  # 抛出异常：必须提供max_len或seg_lens


def _segment_grid_size(batch_info: LoRABatchInfo, num_segments: int) -> int:  # 计算段网格大小
    return batch_info.bs if batch_info.use_cuda_graph else num_segments  # 使用CUDA图时返回bs，否则返回段数


# ---------------------------------------------------------------------------
# Kernel 1 -- Step A_q: per-head per-slot SGMM, reads K-half of B
# 内核1 -- Step A_q：逐头逐槽SGMM，读取B的K半部分
#
#     q_lora_a[t, h, r] = sum_{i<qk_nope} q_nope[t, h, i] * B[slot, h*FULL_K + i, r]
#
# x      : (S, H, qk_nope)
# w (B)  : (num_lora, H*FULL_K, rank)   -- FULL_K = qk_nope + v_head_dim
# out    : (S, H, rank)                 -- fresh allocation, no accumulate
# 输出为新分配，不累加
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])  # JIT编译，不对num_segments进行特化
def _step_a_q_kernel(  # step A_q内核：计算q侧LoRA校正的第一步
    x,  # 输入张量x
    w,  # 权重张量w（B矩阵）
    out,  # 输出张量
    # dims 维度参数
    S,  # 总token数
    H_FULL_K,  # H * (qk_nope + v_head_dim)，行步幅地标值
    K,  # qk_nope（收缩维度）
    N,  # rank（输出维度）
    # strides 步幅参数
    x_stride_s,  # x的第0维步幅
    x_stride_h,  # x的第1维步幅
    x_stride_k,  # x的第2维步幅
    w_stride_l,  # w的lora维度步幅
    w_stride_n,  # w的行维度步幅
    w_stride_k,  # w的列维度步幅
    out_stride_s,  # 输出的第0维步幅
    out_stride_h,  # 输出的第1维步幅
    out_stride_n,  # 输出的第2维步幅
    # batch info 批量信息
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引
    lora_ranks,  # LoRA秩
    sorted_token_ids,  # 排序后的token ID
    num_segments,  # 段数量
    # meta 元参数
    FULL_K: tl.constexpr,  # B中每个头的行步幅（qk_nope + v_head_dim）
    SORTED_BY_ADAPTER: tl.constexpr,  # 是否按适配器排序
    BLOCK_S: tl.constexpr,  # 序列维度的块大小
    BLOCK_N: tl.constexpr,  # 输出维度的块大小
    BLOCK_K: tl.constexpr,  # 收缩维度的块大小
):
    batch_id = tl.program_id(axis=2)  # 获取批次ID（axis=2对应段/请求）
    head_id = tl.program_id(axis=1)  # 获取头ID（axis=1对应注意力头）
    pid = tl.program_id(axis=0)  # 获取程序ID（axis=0对应输出分片）

    if batch_id >= num_segments:  # 如果批次ID超出范围
        return  # 直接返回

    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次的权重索引
    cur_rank = tl.load(lora_ranks + w_index)  # 加载当前LoRA适配器的秩
    if cur_rank == 0:  # 如果秩为0
        return  # 无操作，直接返回

    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    seg_end = tl.load(seg_indptr + batch_id + 1)  # 加载当前段的结束位置
    seg_len = seg_end - seg_start  # 计算当前段的长度
    if seg_len == 0:  # 如果段长度为0
        return  # 无需计算，直接返回

    # Truncate output N to this slot's rank (allows mixed-rank batches).
    # 将输出N截断到当前槽的秩（支持混合秩批次）
    N_eff = tl.minimum(N, cur_rank)  # 计算有效的输出维度

    num_pid_n = tl.cdiv(N_eff, BLOCK_N)  # 计算N维度的程序数
    pid_s = pid // num_pid_n  # 计算序列维度的程序ID
    pid_n = pid % num_pid_n  # 计算输出维度的程序ID
    if pid_s * BLOCK_S >= seg_len:  # 如果序列维度超出段长度
        return  # 无需计算，直接返回

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S  # 序列维度的偏移量
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 输出维度的偏移量
    k_offset = tl.arange(0, BLOCK_K)  # 收缩维度的偏移量

    s_physical = _resolve_token_positions(  # 解析物理token位置
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    # Clamp masked-lane indices into the valid range so pointer arithmetic
    # stays in-bounds even before the load mask drops the values.
    # 将掩码通道的索引钳制到有效范围，确保指针运算在加载掩码丢弃值之前保持边界内
    row_mask = s_offset < seg_len  # 行掩码：哪些行在段内
    safe_row = tl.minimum(s_physical, S - 1)  # 安全行索引，防止越界
    safe_n = tl.minimum(n_offset, N_eff - 1)  # 安全列索引，防止越界

    head_row_base = (
        head_id * FULL_K
    )  # row offset for this head's K-half (i in [0, qk_nope))
    # 当前头在B矩阵中K半部分的行偏移量

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)  # 初始化部分和为0
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):  # 遍历收缩维度的块
        cur_k = k_block * BLOCK_K + k_offset  # 当前的K偏移
        k_mask = cur_k < K  # K维度掩码
        safe_k = tl.minimum(cur_k, K - 1)  # 安全K索引，防止越界

        # x[s, h, k]  读取输入分片
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],  # 掩码：行和K都有效
            other=0.0,  # 掩码外填充0
        )

        # B[slot, h*FULL_K + i, r]: row dim of B carries i (= GEMM K),
        # column dim carries r (= GEMM N).
        # B[slot, h*FULL_K + i, r]：B的行维度携带i（=GEMM K），列维度携带r（=GEMM N）
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + (head_row_base + safe_k[:, None]) * w_stride_n
            + safe_n[None, :] * w_stride_k,
            mask=k_mask[:, None] & (n_offset[None, :] < N_eff),  # 掩码：K和N都有效
            other=0.0,  # 掩码外填充0
        )

        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

    partial_sum = partial_sum.to(x.dtype.element_ty)  # 将结果转换回输入数据类型
    out_offs = (  # 计算输出的偏移量
        safe_row[:, None] * out_stride_s
        + head_id * out_stride_h
        + safe_n[None, :] * out_stride_n
    )
    out_mask = row_mask[:, None] & (n_offset[None, :] < N_eff)  # 输出掩码
    tl.store(out + out_offs, partial_sum, mask=out_mask)  # 将结果写入输出张量


def step_a_q_fwd(  # Q侧校正的步骤A：计算q_nope与B的K半部分的乘积
    q_nope: torch.Tensor,  # 输入：(S, H, qk_nope)，MLA吸收路径的q中间结果
    B_buf: torch.Tensor,  # B权重：(num_lora, H*full_K_per_head, rank)，来自LoRA池
    batch_info: LoRABatchInfo,  # 标准LoRA批量信息
    full_K_per_head: int,  # qk_nope + v_head_dim，B中每个头的行步幅
) -> torch.Tensor:  # 返回：(S, H, rank)，每个token每个头的低秩中间结果
    """Step A of the q-side correction.

    Args:
        q_nope: ``(S, H, qk_nope)``, the absorbed-MLA q intermediate.
        B_buf: ``(num_lora, H*full_K_per_head, rank)`` from the LoRA pool.
        batch_info: standard ``LoRABatchInfo``.
        full_K_per_head: ``qk_nope + v_head_dim``, the row stride per head in B.

    Returns:
        ``(S, H, rank)`` -- per-token, per-head low-rank intermediate, ready for step B_q.
    """
    S, H, qk_nope_dim = q_nope.shape  # 解析输入形状
    rank = B_buf.shape[-1]  # 获取LoRA秩
    out = torch.empty((S, H, rank), device=q_nope.device, dtype=q_nope.dtype)  # 分配输出张量
    num_segments = _num_segments(batch_info)  # 获取段数
    max_segment_len = _max_segment_len(batch_info)  # 获取最大段长度
    segment_grid = _segment_grid_size(batch_info, num_segments)  # 计算段网格大小

    grid = (  # 计算网格大小
        triton.cdiv(max_segment_len, _BLOCK_S) * triton.cdiv(rank, _STEP_A_BLOCK_N),
        H,  # 头数
        segment_grid,  # 段数
    )
    sorted_by_adapter = batch_info.permutation is not None  # 判断是否按适配器排序

    _step_a_q_kernel[grid](  # 启动step_a_q内核
        q_nope,  # 输入x
        B_buf,  # 权重w
        out,  # 输出
        S,  # 总token数
        H * full_K_per_head,  # H_FULL_K
        qk_nope_dim,  # 收缩维度K
        rank,  # 输出维度N
        q_nope.stride(0),  # x_stride_s
        q_nope.stride(1),  # x_stride_h
        q_nope.stride(2),  # x_stride_k
        B_buf.stride(0),  # w_stride_l
        B_buf.stride(1),  # w_stride_n
        B_buf.stride(2),  # w_stride_k
        out.stride(0),  # out_stride_s
        out.stride(1),  # out_stride_h
        out.stride(2),  # out_stride_n
        batch_info.seg_indptr,  # 段索引指针
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        batch_info.permutation,  # 排序后的token ID
        num_segments,  # 段数量
        FULL_K=full_K_per_head,  # 每个头的行步幅
        SORTED_BY_ADAPTER=sorted_by_adapter,  # 是否按适配器排序
        BLOCK_S=_BLOCK_S,  # 序列维度块大小
        BLOCK_N=_STEP_A_BLOCK_N,  # 输出维度块大小
        BLOCK_K=_STEP_A_BLOCK_K,  # 收缩维度块大小
    )
    return out  # 返回输出


# ---------------------------------------------------------------------------
# Kernel 2 -- Step B_q: shared-A per-slot SGMM, scaled + accumulated
# 内核2 -- Step B_q：共享A的逐槽SGMM，缩放并累加
#
#     base[t, h, k] += sum_r x[t, h, r] * A[slot, r, k] * scaling
#
# x      : (S, H, rank)
# w (A)  : (num_lora, rank, kv_lora_rank)
# base   : (S, H, kv_lora_rank), updated in-place (accumulated)
# base原地更新（累加）
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])  # JIT编译，不对num_segments进行特化
def _step_b_q_kernel(  # step B_q内核：计算q侧LoRA校正的第二步，累加到base_output
    x,  # 输入张量x（step A的输出）
    w,  # 权重张量w（A矩阵）
    base,  # 基础输出张量（原地更新）
    # dims 维度参数
    S,  # 总token数
    K,  # rank (contraction)  # rank（收缩维度）
    N,  # kv_lora_rank (output)  # kv_lora_rank（输出维度）
    # strides 步幅参数
    x_stride_s,  # x的第0维步幅
    x_stride_h,  # x的第1维步幅
    x_stride_k,  # x的第2维步幅
    w_stride_l,  # w的lora维度步幅
    w_stride_k,  # w的rank维度步幅
    w_stride_n,  # w的kv_lora_rank维度步幅
    b_stride_s,  # base的第0维步幅
    b_stride_h,  # base的第1维步幅
    b_stride_n,  # base的第2维步幅
    # batch info 批量信息
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引
    lora_ranks,  # LoRA秩
    sorted_token_ids,  # 排序后的token ID
    scalings,  # LoRA缩放因子
    num_segments,  # 段数量
    # meta 元参数
    SORTED_BY_ADAPTER: tl.constexpr,  # 是否按适配器排序
    BLOCK_S: tl.constexpr,  # 序列维度的块大小
    BLOCK_N: tl.constexpr,  # 输出维度的块大小
    BLOCK_K: tl.constexpr,  # 收缩维度的块大小
):
    batch_id = tl.program_id(axis=2)  # 获取批次ID
    head_id = tl.program_id(axis=1)  # 获取头ID
    pid = tl.program_id(axis=0)  # 获取程序ID

    if batch_id >= num_segments:  # 如果批次ID超出范围
        return  # 直接返回

    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次的权重索引
    cur_rank = tl.load(lora_ranks + w_index)  # 加载当前LoRA适配器的秩
    if cur_rank == 0:  # 如果秩为0
        return  # 无操作，直接返回

    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    seg_end = tl.load(seg_indptr + batch_id + 1)  # 加载当前段的结束位置
    seg_len = seg_end - seg_start  # 计算当前段的长度
    if seg_len == 0:  # 如果段长度为0
        return  # 无需计算，直接返回
    scaling = tl.load(scalings + w_index)  # 加载缩放因子

    # Truncate contraction K to this slot's rank.
    # 将收缩K截断到当前槽的秩
    K_eff = tl.minimum(K, cur_rank)  # 计算有效的收缩维度

    num_pid_n = tl.cdiv(N, BLOCK_N)  # 计算N维度的程序数
    pid_s = pid // num_pid_n  # 计算序列维度的程序ID
    pid_n = pid % num_pid_n  # 计算输出维度的程序ID
    if pid_s * BLOCK_S >= seg_len:  # 如果序列维度超出段长度
        return  # 无需计算，直接返回

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S  # 序列维度的偏移量
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 输出维度的偏移量
    k_offset = tl.arange(0, BLOCK_K)  # 收缩维度的偏移量

    s_physical = _resolve_token_positions(  # 解析物理token位置
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len  # 行掩码
    safe_row = tl.minimum(s_physical, S - 1)  # 安全行索引
    n_mask = n_offset[None, :] < N  # N维度掩码
    safe_n = tl.minimum(n_offset, N - 1)  # 安全N索引

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)  # 初始化部分和
    for k_block in range(0, tl.cdiv(K_eff, BLOCK_K)):  # 遍历收缩维度的块
        cur_k = k_block * BLOCK_K + k_offset  # 当前的K偏移
        k_mask = cur_k < K_eff  # K维度掩码
        safe_k = tl.minimum(cur_k, K_eff - 1)  # 安全K索引

        # x[s, h, k]  (k iterates over rank)
        # x[s, h, k]（k遍历rank维度）
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],  # 掩码
            other=0.0,  # 掩码外填充0
        )

        # A[slot, k, n]: read k along contraction, n along output.
        # A[slot, k, n]：沿收缩维度读k，沿输出维度读n
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + safe_n[None, :] * w_stride_n,
            mask=k_mask[:, None] & n_mask,  # 掩码
            other=0.0,  # 掩码外填充0
        )

        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

    partial_sum *= scaling  # 应用缩放因子
    partial_sum = partial_sum.to(x.dtype.element_ty)  # 转换回输入数据类型

    # Accumulate into base[s, h, n].
    # 累加到base[s, h, n]
    base_offs = (  # 计算base的偏移量
        safe_row[:, None] * b_stride_s
        + head_id * b_stride_h
        + safe_n[None, :] * b_stride_n
    )
    out_mask = row_mask[:, None] & n_mask  # 输出掩码
    partial_sum += tl.load(base + base_offs, mask=out_mask, other=0.0)  # 读取base值并加上部分和
    tl.store(base + base_offs, partial_sum, mask=out_mask)  # 将结果写回base


def step_b_q_fwd(  # Q侧校正的步骤B：将step A的结果与A矩阵相乘并累加到base_output
    q_lora_a: torch.Tensor,  # step A_q的输出：(S, H, rank)
    A_buf: torch.Tensor,  # A权重：(num_lora, rank, kv_lora_rank)，来自LoRA池
    batch_info: LoRABatchInfo,  # 标准LoRA批量信息
    base_output: torch.Tensor,  # 基础输出：(S, H, kv_lora_rank)，原地修改（吸收的q_nope @ w_kc结果）
) -> torch.Tensor:  # 返回base_output（同一对象，已修改）
    """Step B of the q-side correction, accumulating into ``base_output``.

    Args:
        q_lora_a: ``(S, H, rank)`` from step A_q.
        A_buf: ``(num_lora, rank, kv_lora_rank)`` from the LoRA pool.
        batch_info: standard ``LoRABatchInfo``.
        base_output: ``(S, H, kv_lora_rank)``, modified in-place
            (the absorbed ``q_nope @ w_kc`` result).

    Returns:
        ``base_output`` (same object, mutated).
    """
    S, H, rank = q_lora_a.shape  # 解析输入形状
    kv_lora_rank = A_buf.shape[-1]  # 获取kv_lora_rank维度
    num_segments = _num_segments(batch_info)  # 获取段数
    max_segment_len = _max_segment_len(batch_info)  # 获取最大段长度
    segment_grid = _segment_grid_size(batch_info, num_segments)  # 计算段网格大小

    grid = (  # 计算网格大小
        triton.cdiv(max_segment_len, _BLOCK_S)
        * triton.cdiv(kv_lora_rank, _STEP_B_BLOCK_N),
        H,  # 头数
        segment_grid,  # 段数
    )
    sorted_by_adapter = batch_info.permutation is not None  # 判断是否按适配器排序

    _step_b_q_kernel[grid](  # 启动step_b_q内核
        q_lora_a,  # 输入x
        A_buf,  # 权重w（A矩阵）
        base_output,  # 基础输出（原地更新）
        S,  # 总token数
        rank,  # 收缩维度K
        kv_lora_rank,  # 输出维度N
        q_lora_a.stride(0),  # x_stride_s
        q_lora_a.stride(1),  # x_stride_h
        q_lora_a.stride(2),  # x_stride_k
        A_buf.stride(0),  # w_stride_l
        A_buf.stride(1),  # w_stride_k
        A_buf.stride(2),  # w_stride_n
        base_output.stride(0),  # b_stride_s
        base_output.stride(1),  # b_stride_h
        base_output.stride(2),  # b_stride_n
        batch_info.seg_indptr,  # 段索引指针
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        batch_info.permutation,  # 排序后的token ID
        batch_info.scalings,  # 缩放因子
        num_segments,  # 段数量
        SORTED_BY_ADAPTER=sorted_by_adapter,  # 是否按适配器排序
        BLOCK_S=_BLOCK_S,  # 序列维度块大小
        BLOCK_N=_STEP_B_BLOCK_N,  # 输出维度块大小
        BLOCK_K=_STEP_B_BLOCK_K,  # 收缩维度块大小
    )
    return base_output  # 返回原地修改后的base_output


# ---------------------------------------------------------------------------
# Kernel 3 -- Step A_v: shared-A.T per-slot SGMM (no scaling, fresh output)
# 内核3 -- Step A_v：共享A.T的逐槽SGMM（无缩放，新输出）
#
#     attn_lora_a[t, h, r] = sum_k attn_output[t, h, k] * A[slot, r, k]
#
# x      : (S, H, kv_lora_rank)
# w (A)  : (num_lora, rank, kv_lora_rank) -- accessed transposed vs step B_q
# w (A)  : (num_lora, rank, kv_lora_rank) -- 相对于step B_q以转置方式访问
# out    : (S, H, rank), fresh allocation
# 输出为新分配
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])  # JIT编译，不对num_segments进行特化
def _step_a_v_kernel(  # step A_v内核：计算v侧LoRA校正的第一步，使用A的转置
    x,  # 输入张量x（attn_output）
    w,  # 权重张量w（A矩阵）
    out,  # 输出张量
    # dims 维度参数
    S,  # 总token数
    K,  # kv_lora_rank (contraction)  # kv_lora_rank（收缩维度）
    N,  # rank (output)  # rank（输出维度）
    # strides 步幅参数
    x_stride_s,  # x的第0维步幅
    x_stride_h,  # x的第1维步幅
    x_stride_k,  # x的第2维步幅
    w_stride_l,  # w的lora维度步幅
    w_stride_n,  # A's "rank" axis (= GEMM N)  # A的"rank"轴（= GEMM N）
    w_stride_k,  # A's "kv_lora_rank" axis (= GEMM K)  # A的"kv_lora_rank"轴（= GEMM K）
    out_stride_s,  # 输出的第0维步幅
    out_stride_h,  # 输出的第1维步幅
    out_stride_n,  # 输出的第2维步幅
    # batch info 批量信息
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引
    lora_ranks,  # LoRA秩
    sorted_token_ids,  # 排序后的token ID
    num_segments,  # 段数量
    # meta 元参数
    SORTED_BY_ADAPTER: tl.constexpr,  # 是否按适配器排序
    BLOCK_S: tl.constexpr,  # 序列维度的块大小
    BLOCK_N: tl.constexpr,  # 输出维度的块大小
    BLOCK_K: tl.constexpr,  # 收缩维度的块大小
):
    batch_id = tl.program_id(axis=2)  # 获取批次ID
    head_id = tl.program_id(axis=1)  # 获取头ID
    pid = tl.program_id(axis=0)  # 获取程序ID

    if batch_id >= num_segments:  # 如果批次ID超出范围
        return  # 直接返回

    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次的权重索引
    cur_rank = tl.load(lora_ranks + w_index)  # 加载当前LoRA适配器的秩
    if cur_rank == 0:  # 如果秩为0
        return  # 无操作，直接返回

    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    seg_end = tl.load(seg_indptr + batch_id + 1)  # 加载当前段的结束位置
    seg_len = seg_end - seg_start  # 计算当前段的长度
    if seg_len == 0:  # 如果段长度为0
        return  # 无需计算，直接返回

    # Truncate output N to this slot's rank.
    # 将输出N截断到当前槽的秩
    N_eff = tl.minimum(N, cur_rank)  # 计算有效的输出维度

    num_pid_n = tl.cdiv(N_eff, BLOCK_N)  # 计算N维度的程序数
    pid_s = pid // num_pid_n  # 计算序列维度的程序ID
    pid_n = pid % num_pid_n  # 计算输出维度的程序ID
    if pid_s * BLOCK_S >= seg_len:  # 如果序列维度超出段长度
        return  # 无需计算，直接返回

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S  # 序列维度的偏移量
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 输出维度的偏移量
    k_offset = tl.arange(0, BLOCK_K)  # 收缩维度的偏移量

    s_physical = _resolve_token_positions(  # 解析物理token位置
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len  # 行掩码
    safe_row = tl.minimum(s_physical, S - 1)  # 安全行索引
    safe_n = tl.minimum(n_offset, N_eff - 1)  # 安全N索引

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)  # 初始化部分和
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):  # 遍历收缩维度的块
        cur_k = k_block * BLOCK_K + k_offset  # 当前的K偏移
        k_mask = cur_k < K  # K维度掩码
        safe_k = tl.minimum(cur_k, K - 1)  # 安全K索引

        # x[s, h, k]  读取输入分片
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],  # 掩码
            other=0.0,  # 掩码外填充0
        )

        # A[slot, r, k] -- here we want each (k, r) so we read along k
        # (inner / contraction) and produce r as output.  Stride access:
        # the row dim is r (= GEMM N), column dim is k (= GEMM K).
        # A[slot, r, k] -- 这里我们需要每个(k, r)，所以沿k（内层/收缩）读取，
        # 产生r作为输出。步幅访问：行维度是r（= GEMM N），列维度是k（= GEMM K）
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + safe_n[None, :] * w_stride_n,
            mask=k_mask[:, None] & (n_offset[None, :] < N_eff),  # 掩码
            other=0.0,  # 掩码外填充0
        )

        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

    partial_sum = partial_sum.to(x.dtype.element_ty)  # 转换回输入数据类型
    out_offs = (  # 计算输出的偏移量
        safe_row[:, None] * out_stride_s
        + head_id * out_stride_h
        + safe_n[None, :] * out_stride_n
    )
    out_mask = row_mask[:, None] & (n_offset[None, :] < N_eff)  # 输出掩码
    tl.store(out + out_offs, partial_sum, mask=out_mask)  # 将结果写入输出张量


def step_a_v_fwd(  # V侧校正的步骤A：计算attn_output与A转置的乘积
    attn_output: torch.Tensor,  # 输入：(S, H, kv_lora_rank)，注意力后中间结果
    A_buf: torch.Tensor,  # A权重：(num_lora, rank, kv_lora_rank)
    batch_info: LoRABatchInfo,  # 标准LoRA批量信息
) -> torch.Tensor:  # 返回：(S, H, rank)，每个token每个头的低秩中间结果
    """Step A of the v-side correction.

    Args:
        attn_output: ``(S, H, kv_lora_rank)``, the post-attention intermediate.
        A_buf: ``(num_lora, rank, kv_lora_rank)``.
        batch_info: standard ``LoRABatchInfo``.

    Returns:
        ``(S, H, rank)`` -- per-token, per-head low-rank intermediate for step B_v.
    """
    S, H, kv_lora_rank = attn_output.shape  # 解析输入形状
    rank = A_buf.shape[1]  # 获取rank维度
    out = torch.empty((S, H, rank), device=attn_output.device, dtype=attn_output.dtype)  # 分配输出张量
    num_segments = _num_segments(batch_info)  # 获取段数
    max_segment_len = _max_segment_len(batch_info)  # 获取最大段长度
    segment_grid = _segment_grid_size(batch_info, num_segments)  # 计算段网格大小

    grid = (  # 计算网格大小
        triton.cdiv(max_segment_len, _BLOCK_S) * triton.cdiv(rank, _STEP_A_BLOCK_N),
        H,  # 头数
        segment_grid,  # 段数
    )
    sorted_by_adapter = batch_info.permutation is not None  # 判断是否按适配器排序

    _step_a_v_kernel[grid](  # 启动step_a_v内核
        attn_output,  # 输入x
        A_buf,  # 权重w
        out,  # 输出
        S,  # 总token数
        kv_lora_rank,  # 收缩维度K
        rank,  # 输出维度N
        attn_output.stride(0),  # x_stride_s
        attn_output.stride(1),  # x_stride_h
        attn_output.stride(2),  # x_stride_k
        A_buf.stride(0),  # w_stride_l
        A_buf.stride(1),  # w_stride_n
        A_buf.stride(2),  # w_stride_k
        out.stride(0),  # out_stride_s
        out.stride(1),  # out_stride_h
        out.stride(2),  # out_stride_n
        batch_info.seg_indptr,  # 段索引指针
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        batch_info.permutation,  # 排序后的token ID
        num_segments,  # 段数量
        SORTED_BY_ADAPTER=sorted_by_adapter,  # 是否按适配器排序
        BLOCK_S=_BLOCK_S,  # 序列维度块大小
        BLOCK_N=_STEP_A_BLOCK_N,  # 输出维度块大小
        BLOCK_K=_STEP_A_BLOCK_K,  # 收缩维度块大小
    )
    return out  # 返回输出


# ---------------------------------------------------------------------------
# Kernel 4 -- Step B_v: per-head per-slot SGMM with V-half of B (transposed),
# scaled + accumulated
# 内核4 -- Step B_v：逐头逐槽SGMM，使用B的V半部分（转置），缩放并累加
#
#     base[t, h, j] += sum_r x[t, h, r] * B[slot, h*FULL_K + qk_nope + j, r] * scaling
#
# x      : (S, H, rank)
# w (B)  : (num_lora, H*FULL_K, rank), V-half slice via offset
# w (B)  : (num_lora, H*FULL_K, rank)，通过偏移访问V半部分
# base   : (S, H, v_head_dim), updated in-place (accumulated)
# base原地更新（累加）
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])  # JIT编译，不对num_segments进行特化
def _step_b_v_kernel(  # step B_v内核：计算v侧LoRA校正的第二步，使用B的V半部分并累加到base
    x,  # 输入张量x（step A_v的输出）
    w,  # 权重张量w（B矩阵）
    base,  # 基础输出张量（原地更新）
    # dims 维度参数
    S,  # 总token数
    K,  # rank (contraction)  # rank（收缩维度）
    N,  # v_head_dim (output)  # v_head_dim（输出维度）
    # strides 步幅参数
    x_stride_s,  # x的第0维步幅
    x_stride_h,  # x的第1维步幅
    x_stride_k,  # x的第2维步幅
    w_stride_l,  # w的lora维度步幅
    w_stride_n,  # B's row dim (h*FULL_K + j) -- this is GEMM N  # B的行维度(h*FULL_K + j) -- 这是GEMM N
    w_stride_k,  # B's rank dim -- this is GEMM K  # B的rank维度 -- 这是GEMM K
    b_stride_s,  # base的第0维步幅
    b_stride_h,  # base的第1维步幅
    b_stride_n,  # base的第2维步幅
    # batch info 批量信息
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引
    lora_ranks,  # LoRA秩
    sorted_token_ids,  # 排序后的token ID
    scalings,  # LoRA缩放因子
    num_segments,  # 段数量
    # meta 元参数
    FULL_K: tl.constexpr,  # qk_nope + v_head_dim  # qk_nope + v_head_dim
    QK_NOPE_OFFSET: tl.constexpr,  # offset of V-half within each head's row block  # 每个头行块内V半部分的偏移量
    SORTED_BY_ADAPTER: tl.constexpr,  # 是否按适配器排序
    BLOCK_S: tl.constexpr,  # 序列维度的块大小
    BLOCK_N: tl.constexpr,  # 输出维度的块大小
    BLOCK_K: tl.constexpr,  # 收缩维度的块大小
):
    batch_id = tl.program_id(axis=2)  # 获取批次ID
    head_id = tl.program_id(axis=1)  # 获取头ID
    pid = tl.program_id(axis=0)  # 获取程序ID

    if batch_id >= num_segments:  # 如果批次ID超出范围
        return  # 直接返回

    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次的权重索引
    cur_rank = tl.load(lora_ranks + w_index)  # 加载当前LoRA适配器的秩
    if cur_rank == 0:  # 如果秩为0
        return  # 无操作，直接返回

    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    seg_end = tl.load(seg_indptr + batch_id + 1)  # 加载当前段的结束位置
    seg_len = seg_end - seg_start  # 计算当前段的长度
    if seg_len == 0:  # 如果段长度为0
        return  # 无需计算，直接返回
    scaling = tl.load(scalings + w_index)  # 加载缩放因子

    K_eff = tl.minimum(K, cur_rank)  # 计算有效的收缩维度

    num_pid_n = tl.cdiv(N, BLOCK_N)  # 计算N维度的程序数
    pid_s = pid // num_pid_n  # 计算序列维度的程序ID
    pid_n = pid % num_pid_n  # 计算输出维度的程序ID
    if pid_s * BLOCK_S >= seg_len:  # 如果序列维度超出段长度
        return  # 无需计算，直接返回

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S  # 序列维度的偏移量
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 输出维度的偏移量
    k_offset = tl.arange(0, BLOCK_K)  # 收缩维度的偏移量

    s_physical = _resolve_token_positions(  # 解析物理token位置
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len  # 行掩码
    safe_row = tl.minimum(s_physical, S - 1)  # 安全行索引
    n_mask = n_offset[None, :] < N  # N维度掩码
    safe_n = tl.minimum(n_offset, N - 1)  # 安全N索引

    # V-half row base for this head: h*FULL_K + qk_nope
    # 当前头V半部分的行基偏移：h*FULL_K + qk_nope
    head_row_base = head_id * FULL_K + QK_NOPE_OFFSET  # 计算V半部分的行基偏移

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)  # 初始化部分和
    for k_block in range(0, tl.cdiv(K_eff, BLOCK_K)):  # 遍历收缩维度的块
        cur_k = k_block * BLOCK_K + k_offset  # 当前的K偏移
        k_mask = cur_k < K_eff  # K维度掩码
        safe_k = tl.minimum(cur_k, K_eff - 1)  # 安全K索引

        # x[s, h, k]  读取输入分片
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],  # 掩码
            other=0.0,  # 掩码外填充0
        )

        # B[slot, h*FULL_K + qk_nope + j, r] -- row dim is j (= GEMM N),
        # column dim is r (= GEMM K).  Transposed access vs step A_q.
        # B[slot, h*FULL_K + qk_nope + j, r] -- 行维度是j（= GEMM N），
        # 列维度是r（= GEMM K）。相对于step A_q是转置访问
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + (head_row_base + safe_n[None, :]) * w_stride_n,
            mask=k_mask[:, None] & n_mask,  # 掩码
            other=0.0,  # 掩码外填充0
        )

        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

    partial_sum *= scaling  # 应用缩放因子
    partial_sum = partial_sum.to(x.dtype.element_ty)  # 转换回输入数据类型

    base_offs = (  # 计算base的偏移量
        safe_row[:, None] * b_stride_s
        + head_id * b_stride_h
        + safe_n[None, :] * b_stride_n
    )
    out_mask = row_mask[:, None] & n_mask  # 输出掩码
    partial_sum += tl.load(base + base_offs, mask=out_mask, other=0.0)  # 读取base值并加上部分和
    tl.store(base + base_offs, partial_sum, mask=out_mask)  # 将结果写回base


def step_b_v_fwd(  # V侧校正的步骤B：将step A_v的结果与B的V半部分相乘并累加到base_output
    attn_lora_a: torch.Tensor,  # step A_v的输出：(S, H, rank)
    B_buf: torch.Tensor,  # B权重：(num_lora, H*(qk_nope+v_head_dim), rank)
    batch_info: LoRABatchInfo,  # 标准LoRA批量信息
    base_output: torch.Tensor,  # 基础输出：(S, H, v_head_dim)，原地修改（吸收的attn_output @ w_vc结果）
    qk_nope_head_dim: int,  # B中每个头行块内V半部分的偏移量
    v_head_dim: int,  # 每个头的输出特征维度
) -> torch.Tensor:  # 返回base_output（同一对象，已修改）
    """Step B of the v-side correction, accumulating into ``base_output``.

    Args:
        attn_lora_a: ``(S, H, rank)`` from step A_v.
        B_buf: ``(num_lora, H*(qk_nope+v_head_dim), rank)``.
        batch_info: standard ``LoRABatchInfo``.
        base_output: ``(S, H, v_head_dim)``, modified in-place
            (the absorbed ``attn_output @ w_vc`` result).
        qk_nope_head_dim: offset of V-half within each head's row block of B.
        v_head_dim: output feature dim per head.

    Returns:
        ``base_output`` (same object, mutated).
    """
    S, H, rank = attn_lora_a.shape  # 解析输入形状
    full_K_per_head = qk_nope_head_dim + v_head_dim  # 计算每个头的完整K维度
    num_segments = _num_segments(batch_info)  # 获取段数
    max_segment_len = _max_segment_len(batch_info)  # 获取最大段长度
    segment_grid = _segment_grid_size(batch_info, num_segments)  # 计算段网格大小

    grid = (  # 计算网格大小
        triton.cdiv(max_segment_len, _BLOCK_S)
        * triton.cdiv(v_head_dim, _STEP_B_BLOCK_N),
        H,  # 头数
        segment_grid,  # 段数
    )
    sorted_by_adapter = batch_info.permutation is not None  # 判断是否按适配器排序

    _step_b_v_kernel[grid](  # 启动step_b_v内核
        attn_lora_a,  # 输入x
        B_buf,  # 权重w（B矩阵）
        base_output,  # 基础输出（原地更新）
        S,  # 总token数
        rank,  # 收缩维度K
        v_head_dim,  # 输出维度N
        attn_lora_a.stride(0),  # x_stride_s
        attn_lora_a.stride(1),  # x_stride_h
        attn_lora_a.stride(2),  # x_stride_k
        B_buf.stride(0),  # w_stride_l
        B_buf.stride(1),  # w_stride_n
        B_buf.stride(2),  # w_stride_k
        base_output.stride(0),  # b_stride_s
        base_output.stride(1),  # b_stride_h
        base_output.stride(2),  # b_stride_n
        batch_info.seg_indptr,  # 段索引指针
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        batch_info.permutation,  # 排序后的token ID
        batch_info.scalings,  # 缩放因子
        num_segments,  # 段数量
        FULL_K=full_K_per_head,  # 每个头的完整K维度
        QK_NOPE_OFFSET=qk_nope_head_dim,  # V半部分在行块内的偏移量
        SORTED_BY_ADAPTER=sorted_by_adapter,  # 是否按适配器排序
        BLOCK_S=_BLOCK_S,  # 序列维度块大小
        BLOCK_N=_STEP_B_BLOCK_N,  # 输出维度块大小
        BLOCK_K=_STEP_B_BLOCK_K,  # 收缩维度块大小
    )
    return base_output  # 返回原地修改后的base_output
