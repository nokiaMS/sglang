# 扩展（Prefill）阶段高效内存注意力Triton内核文件
# 本文件实现了预填充/扩展阶段的内存高效注意力计算，
# 支持页大小=1和带KV缓存的预填充（即extend操作）。
# 包含两种实现：
# - 两阶段内核：分别处理前缀KV和扩展KV
# - 统一单阶段内核：通过统一KV索引一次性处理所有KV
# 支持因果掩码、自定义掩码、滑动窗口、MLA位置编码等特性。

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).

预填充阶段的内存高效注意力。
支持页大小 = 1和带KV缓存的预填充（即extend）。
"""

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.layers.attention.triton_ops.prefill_attention import (  # 导入预填充注意力
    context_attention_fwd,  # 上下文注意力前向传播
)
from sglang.srt.utils import is_cuda, is_hip  # 导入平台检测函数

_is_cuda = is_cuda()  # 检测当前是否为CUDA平台
if _is_cuda:  # 如果是CUDA平台
    CUDA_CAPABILITY = torch.cuda.get_device_capability()  # 获取CUDA设备计算能力

_is_hip = is_hip()  # 检测当前是否为HIP(AMD ROCm)平台


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):  # 获取扩展注意力的块大小配置
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        Lq: Query head dimension
        Lv: Value head dimension

    Returns:
        tuple: (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)

    获取扩展注意力内核的块大小和配置。

    参数：
        Lq: Query头维度
        Lv: Value头维度

    返回：
        元组: (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    # Determine BLOCK_DMODEL and BLOCK_DPE based on head dimension
    # 根据头维度确定BLOCK_DMODEL和BLOCK_DPE
    if Lq == 576:  # MLA 576维
        BLOCK_DMODEL = 512  # 模型维度块大小
        BLOCK_DPE = 64  # 位置编码维度块大小
    elif Lq == 288:  # MLA 288维
        BLOCK_DMODEL = 256  # 模型维度块大小
        BLOCK_DPE = 32  # 位置编码维度块大小
    elif Lq == 192:  # MLA 192维
        BLOCK_DMODEL = 128  # 模型维度块大小
        BLOCK_DPE = 64  # 位置编码维度块大小
    else:  # 其他维度
        BLOCK_DMODEL = triton.next_power_of_2(Lq)  # 对齐到2的幂
        BLOCK_DPE = 0  # 无位置编码维度

    BLOCK_DV = triton.next_power_of_2(Lv)  # Value维度对齐到2的幂

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware
    # 根据硬件确定BLOCK_M、BLOCK_N和num_warps
    if _is_hip:  # HIP(AMD ROCm)平台
        BLOCK_M, BLOCK_N = (64, 64)  # 使用较小的块大小
        num_warps = 4  # 4个warp
    else:  # CUDA平台
        if _is_cuda and CUDA_CAPABILITY[0] == 12:  # sm120工作站Blackwell架构（RTX Pro 6000）
            # sm120 workstation Blackwell architecture (RTX Pro 6000) has a much smaller shared memory size (100K)
            # sm120工作站Blackwell架构（RTX Pro 6000）的共享内存大小（100K）要小得多
            if Lq <= 128:  # 小头维度
                BLOCK_M, BLOCK_N = (64, 128)  # 较大的块大小
            elif Lq <= 256:  # 中等头维度
                BLOCK_M, BLOCK_N = (64, 64)  # 中等块大小
            else:  # 大头维度
                BLOCK_M, BLOCK_N = (32, 32)  # 较小的块大小
        elif _is_cuda and CUDA_CAPABILITY[0] == 10:  # Blackwell数据中心架构
            # Blackwell data-center architecture (GB200, B200, sm_100a)
            # Blackwell数据中心架构（GB200, B200, sm_100a）
            # sm_100a has different register constraints from Hopper; Hopper block sizes
            # cause PTX register exhaustion (>255 regs) for large head dims (Lq=512).
            # sm_100a与Hopper有不同的寄存器约束；Hopper的块大小
            # 会导致大头维度（Lq=512）时PTX寄存器耗尽（>255个寄存器）。
            if Lq <= 256:  # 中小头维度
                BLOCK_M, BLOCK_N = (64, 64)  # 中等块大小
            else:  # 大头维度
                BLOCK_M, BLOCK_N = (16, 64)  # 减小M维度以避免寄存器溢出
        elif _is_cuda and CUDA_CAPABILITY[0] >= 9:  # Hopper架构（H100等）
            # Hopper architecture (H100, etc.)
            # Hopper架构（H100等）
            if Lq <= 256:  # 中小头维度
                BLOCK_M, BLOCK_N = (128, 64)  # 较大的块大小
            else:  # 大头维度
                BLOCK_M, BLOCK_N = (32, 64)  # 减小M维度
        elif _is_cuda and CUDA_CAPABILITY[0] >= 8:  # Ampere架构（A100等）
            # Ampere architecture (A100, etc.)
            # Ampere架构（A100等）
            # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
            # sm86/sm89的共享内存大小（100K）比sm80（160K）小得多
            if CUDA_CAPABILITY[1] == 9 or CUDA_CAPABILITY[1] == 6:  # sm86/sm89
                if Lq <= 128:  # 小头维度
                    BLOCK_M, BLOCK_N = (64, 128)  # 较大的块大小
                elif Lq <= 256:  # 中等头维度
                    BLOCK_M, BLOCK_N = (64, 64)  # 中等块大小
                else:  # 大头维度
                    BLOCK_M, BLOCK_N = (32, 32)  # 较小的块大小
            else:  # sm80等
                if Lq <= 128:  # 小头维度
                    BLOCK_M, BLOCK_N = (128, 128)  # 最大块大小
                elif Lq <= 256:  # 中等头维度
                    BLOCK_M, BLOCK_N = (64, 64)  # 中等块大小
                else:  # 大头维度
                    BLOCK_M, BLOCK_N = (32, 64)  # 减小M维度
        else:  # 更旧的架构
            # Older architectures
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)  # 根据头维度选择块大小

        num_warps = 4 if Lq <= 64 else 8  # 小头维度用4个warp，大头维度用8个

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps  # 返回所有块大小配置


@triton.jit  # Triton JIT编译装饰器
def tanh(x):  # Triton实现的tanh函数
    # Tanh is just a scaled sigmoid
    # Tanh只是缩放的sigmoid
    return 2 * tl.sigmoid(2 * x) - 1  # tanh(x) = 2*sigmoid(2x) - 1


@triton.jit  # Triton JIT编译装饰器
def _copy_unified_indices_kernel(  # 复制统一KV索引内核
    # Input buffers
    # 输入缓冲区
    prefix_kv_indptr,  # 前缀KV索引偏移
    prefix_kv_indices,  # 前缀KV索引
    extend_start_loc,  # 扩展起始位置
    extend_seq_lens,  # 扩展序列长度
    extend_kv_indices,  # 扩展KV索引
    unified_kv_indptr,  # 统一KV索引偏移
    # Output buffer
    # 输出缓冲区
    unified_kv_indices,  # 统一KV索引（输出）
    # Size
    # 大小
    bs,  # 批次大小
):  # 将前缀和扩展KV索引复制到统一缓冲区
    """
    Triton kernel to copy indices to unified buffer (parallel per sequence).
    Each thread block processes one sequence with vectorized loads/stores.

    将索引复制到统一缓冲区的Triton内核（按序列并行）。
    每个线程块处理一个序列，使用向量化加载/存储。
    """
    pid = tl.program_id(0)  # 获取当前程序ID（序列索引）

    if pid >= bs:  # 如果超出批次范围
        return  # 直接返回

    # Load sequence info
    # 加载序列信息
    prefix_start = tl.load(prefix_kv_indptr + pid)  # 加载前缀起始索引
    prefix_end = tl.load(prefix_kv_indptr + pid + 1)  # 加载前缀结束索引
    extend_start = tl.load(extend_start_loc + pid)  # 加载扩展起始位置
    extend_len = tl.load(extend_seq_lens + pid)  # 加载扩展序列长度

    prefix_len = prefix_end - prefix_start  # 计算前缀长度
    unified_start = tl.load(unified_kv_indptr + pid)  # 加载统一缓冲区起始位置

    # Copy indices in vectorized chunks
    # 以向量化块方式复制索引
    BLOCK_SIZE: tl.constexpr = 128  # 向量化块大小

    # Process prefix indices
    # 处理前缀索引
    for block_start in range(0, prefix_len, BLOCK_SIZE):  # 分块迭代
        offs = block_start + tl.arange(0, BLOCK_SIZE)  # 计算偏移
        mask = offs < prefix_len  # 生成掩码

        src_idx = prefix_start + offs  # 计算源索引
        dst_idx = unified_start + offs  # 计算目标索引

        vals = tl.load(prefix_kv_indices + src_idx, mask=mask, other=0)  # 加载前缀索引
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)  # 存储到统一缓冲区

    # Process extend indices
    # 处理扩展索引
    for block_start in range(0, extend_len, BLOCK_SIZE):  # 分块迭代
        offs = block_start + tl.arange(0, BLOCK_SIZE)  # 计算偏移
        mask = offs < extend_len  # 生成掩码

        src_idx = extend_start + offs  # 计算源索引
        dst_idx = unified_start + prefix_len + offs  # 计算目标索引（前缀之后）

        vals = tl.load(extend_kv_indices + src_idx, mask=mask, other=0)  # 加载扩展索引
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)  # 存储到统一缓冲区


def build_unified_kv_indices(  # 构建统一KV索引
    prefix_kv_indptr: torch.Tensor,  # 前缀KV索引偏移
    prefix_kv_indices: torch.Tensor,  # 前缀KV索引
    extend_start_loc: torch.Tensor,  # 扩展起始位置
    extend_seq_lens: torch.Tensor,  # 扩展序列长度
    extend_kv_indices: torch.Tensor,  # 扩展KV索引
    bs: int,  # 批次大小
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回统一索引偏移、索引和前缀长度
    """
    Build unified KV indices efficiently:
    - Use PyTorch's optimized cumsum (NVIDIA CUB) for indptr
    - Use Triton kernel for parallel index copying

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)

    高效构建统一KV索引：
    - 使用PyTorch优化的cumsum（NVIDIA CUB）计算indptr
    - 使用Triton内核并行复制索引

    返回：
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    device = prefix_kv_indptr.device  # 获取设备

    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]  # 计算每个序列的前缀长度

    # Create unified_kv_indptr avoiding direct assignment (for CUDA graph compatibility)
    # 创建unified_kv_indptr时避免直接赋值（为了CUDA图兼容性）
    unified_lens = prefix_lens + extend_seq_lens[:bs]  # 计算统一长度（前缀+扩展）
    unified_kv_indptr = torch.cat(  # 拼接创建统一索引偏移
        [
            torch.zeros(1, dtype=torch.int32, device=device),  # 起始偏移为0
            torch.cumsum(unified_lens, dim=0),  # 累积和作为偏移
        ]
    )

    max_unified_len = len(prefix_kv_indices) + len(extend_kv_indices)  # 计算最大统一长度

    unified_kv_indices = torch.empty(max_unified_len, dtype=torch.int64, device=device)  # 分配统一索引张量

    # Launch Triton kernel for parallel index copying
    # 启动Triton内核进行并行索引复制
    _copy_unified_indices_kernel[(bs,)](  # 调用内核
        prefix_kv_indptr,  # 前缀KV索引偏移
        prefix_kv_indices,  # 前缀KV索引
        extend_start_loc,  # 扩展起始位置
        extend_seq_lens,  # 扩展序列长度
        extend_kv_indices,  # 扩展KV索引
        unified_kv_indptr,  # 统一KV索引偏移
        unified_kv_indices,  # 统一KV索引（输出）
        bs,  # 批次大小
    )

    return unified_kv_indptr, unified_kv_indices, prefix_lens  # 返回统一索引和前缀长度


@triton.jit  # Triton JIT编译装饰器
def _fwd_kernel(  # 扩展注意力两阶段前向内核
    Q_Extend,  # 扩展Query指针
    K_Extend,  # 扩展Key指针
    V_Extend,  # 扩展Value指针
    O_Extend,  # 扩展输出指针
    K_Buffer,  # Key缓存指针（前缀+扩展）
    V_Buffer,  # Value缓存指针（前缀+扩展）
    qo_indptr,  # Query/Output索引偏移
    kv_indptr,  # KV索引偏移
    kv_indices,  # KV索引
    mask_ptr,  # 自定义掩码指针
    mask_indptr,  # 掩码索引偏移
    sink_ptr,  # 注意力汇聚指针
    window_kv_offset_ptr,  # 滑动窗口KV偏移指针
    sm_scale,  # softmax缩放因子
    k_scale,  # Key缩放因子
    v_scale,  # Value缩放因子
    kv_group_num,  # KV组数
    stride_qbs,  # Query批次步长
    stride_qh,  # Query头步长
    stride_kbs,  # Key批次步长
    stride_kh,  # Key头步长
    stride_vbs,  # Value批次步长
    stride_vh,  # Value头步长
    stride_obs,  # 输出批次步长
    stride_oh,  # 输出头步长
    stride_buf_kbs,  # Key缓存批次步长
    stride_buf_kh,  # Key缓存头步长
    stride_buf_vbs,  # Value缓存批次步长
    stride_buf_vh,  # Value缓存头步长
    SLIDING_WINDOW_SIZE: tl.constexpr,  # 滑动窗口大小（编译时常量）
    logit_cap: tl.constexpr,  # logit截断值（编译时常量）
    xai_temperature_len: tl.constexpr,  # XAI温度长度（编译时常量）
    Lq: tl.constexpr,  # Query头维度（编译时常量）
    Lv: tl.constexpr,  # Value头维度（编译时常量）
    BLOCK_DMODEL: tl.constexpr,  # 模型维度块大小（编译时常量）
    BLOCK_DPE: tl.constexpr,  # 位置编码维度块大小（编译时常量）
    BLOCK_DV: tl.constexpr,  # Value维度块大小（编译时常量）
    BLOCK_M: tl.constexpr,  # M维度块大小（编译时常量）
    BLOCK_N: tl.constexpr,  # N维度块大小（编译时常量）
    USE_CUSTOM_MASK: tl.constexpr,  # 是否使用自定义掩码（编译时常量）
    IS_CAUSAL: tl.constexpr,  # 是否因果注意力（编译时常量）
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,  # 是否跳过前缀的自定义掩码（编译时常量）
    STORE_TRANSPOSE: tl.constexpr,  # 是否转置存储（编译时常量）
    HAS_SINK: tl.constexpr,  # 是否有注意力汇聚（编译时常量）
):  # 扩展注意力两阶段内核：阶段1处理前缀KV，阶段2处理扩展KV的三角部分
    """扩展注意力两阶段前向内核，分别处理前缀KV和扩展KV"""
    cur_seq = tl.program_id(0)  # 获取当前序列索引
    cur_head = tl.program_id(1)  # 获取当前头索引
    cur_block_m = tl.program_id(2)  # 获取当前M维度块索引
    cur_kv_head = cur_head // kv_group_num  # 计算当前KV头索引

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)  # 加载扩展Query起始索引
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx  # 计算扩展序列长度
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)  # 加载KV起始索引
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx  # 计算前缀KV长度
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend  # 计算总序列长度

    if USE_CUSTOM_MASK:  # 如果使用自定义掩码
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)  # 加载掩码起始索引

    # For SWA, we should only load the mask in the sliding window
    # 对于滑动窗口注意力(SWA)，我们应该只加载滑动窗口内的掩码
    window_kv_offset = 0  # 初始化窗口KV偏移
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:  # 自定义掩码且有滑动窗口
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)  # 加载窗口KV偏移

    offs_d = tl.arange(0, BLOCK_DMODEL)  # 模型维度偏移
    offs_dv = tl.arange(0, BLOCK_DV)  # Value维度偏移
    offs_m = tl.arange(0, BLOCK_M)  # M维度偏移
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend  # M维度掩码

    mask_d = offs_d < Lq  # Key维度掩码
    mask_dv = offs_dv < Lv  # Value维度掩码

    if xai_temperature_len > 0:  # 如果启用XAI温度调节
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m  # Query位置索引
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))  # 温度缩放系数
        xai_temperature_reg = tl.where(  # 计算温度调节系数
            offs_qidx > xai_temperature_len,  # 超过温度长度
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,  # 应用缩放
            1.0,  # 不超过时为1
        )

    offs_q = (  # 计算Query偏移
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])  # token偏移
        * stride_qbs  # 批次步长
        + cur_head * stride_qh  # 头步长
        + offs_d[None, :]  # 维度偏移
    )
    q = tl.load(  # 加载Query向量
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0  # 带掩码加载
    )

    if BLOCK_DPE > 0:  # 如果有位置编码维度（MLA模式）
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)  # 位置编码维度偏移
        offs_qpe = (  # 计算Query位置编码偏移
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])  # token偏移
            * stride_qbs  # 批次步长
            + cur_head * stride_qh  # 头步长
            + offs_dpe[None, :]  # 位置编码维度偏移
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)  # 加载Query位置编码

    # stage 1: compute scores with prefix
    # 阶段1：计算与前缀KV的注意力分数
    offs_n = tl.arange(0, BLOCK_N)  # N维度偏移

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)  # 初始化Value累加器
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)  # 初始化分母（softmax归一化因子）
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 初始化最大指数值

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):  # 遍历前缀KV
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 提示编译器start_n是BLOCK_N的倍数
        mask_n = (start_n + offs_n) < cur_seq_len_prefix  # N维度掩码

        final_mask = mask_m[:, None] & mask_n[None, :]  # 合并M和N维度掩码
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:  # 使用自定义掩码且不跳过前缀掩码
            custom_mask = tl.load(  # 加载自定义掩码
                mask_ptr  # 掩码基址
                + cur_seq_mask_start_idx  # 掩码起始偏移
                + (cur_block_m * BLOCK_M + offs_m[:, None])  # Query位置
                * (cur_seq_len + window_kv_offset)  # 掩码行宽
                + window_kv_offset  # 窗口偏移
                + start_n  # KV起始位置
                + offs_n[None, :],  # KV偏移
                mask=(mask_m[:, None] & mask_n[None, :]),  # 掩码
                other=0,  # 越界填充值
            )
            final_mask &= custom_mask  # 合并自定义掩码
        if SLIDING_WINDOW_SIZE > 0:  # 如果有滑动窗口
            # Add mask where q_id <= kv_id + sliding_window_size
            # 添加q_id <= kv_id + sliding_window_size的掩码
            # q_id = prefix_len + cur_m, kv_id = cur_n
            # q_id = 前缀长度 + 当前M偏移, kv_id = 当前N偏移
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)  # 滑动窗口掩码
            final_mask &= window_mask  # 合并滑动窗口掩码

        SKIP_TILE = False  # 初始化跳过标记
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:  # 有额外掩码
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0  # 检查是否所有掩码为0

        if not SKIP_TILE:  # 不跳过该块
            offs_kv_loc = tl.load(  # 加载KV位置索引
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,  # 从索引数组获取
                mask=mask_n,  # 掩码
                other=0,  # 越界填充值
            )

            # load k in transposed way
            # 以转置方式加载Key
            offs_buf_k = (  # 计算Key缓存偏移
                offs_kv_loc[None, :] * stride_buf_kbs  # token位置偏移
                + cur_kv_head * stride_buf_kh  # 头偏移
                + offs_d[:, None]  # 维度偏移（转置）
            )
            k = tl.load(  # 加载Key向量
                K_Buffer + offs_buf_k,  # Key缓存地址
                mask=(mask_n[None, :]) & (mask_d[:, None]),  # 掩码
                other=0.0,  # 越界填充值
            )
            qk = tl.dot(q.to(k.dtype), k)  # 计算QK点积
            if BLOCK_DPE > 0:  # MLA模式：额外计算位置编码
                offs_kpe = (  # 计算Key位置编码偏移
                    offs_kv_loc[None, :] * stride_buf_kbs  # token位置偏移
                    + cur_kv_head * stride_buf_kh  # 头偏移
                    + offs_dpe[:, None]  # 位置编码维度偏移
                )
                kpe = tl.load(  # 加载Key位置编码
                    K_Buffer + offs_kpe,  # Key缓存地址
                    mask=mask_n[None, :],  # 掩码
                    other=0.0,  # 越界填充值
                )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)  # 累加位置编码的QK点积
            qk *= sm_scale * k_scale  # 乘以softmax和Key缩放因子

            if logit_cap > 0:  # 如果启用logit截断
                qk = logit_cap * tanh(qk / logit_cap)  # 应用logit截断

            if xai_temperature_len > 0:  # 如果启用XAI温度调节
                qk *= xai_temperature_reg[:, None]  # 乘以温度调节系数

            qk = tl.where(final_mask, qk, float("-inf"))  # 应用最终掩码

            row_max = tl.max(qk, 1)  # 计算每行最大值
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)  # 修复全负无穷行
            n_e_max = tl.maximum(row_max_fixed, e_max)  # 更新全局最大值

            re_scale = tl.exp(e_max - n_e_max)  # 计算重缩放因子
            p = tl.exp(qk - n_e_max[:, None])  # 计算softmax概率
            deno = deno * re_scale + tl.sum(p, 1)  # 更新分母

            offs_buf_v = (  # 计算Value缓存偏移
                offs_kv_loc[:, None] * stride_buf_vbs  # token位置偏移
                + cur_kv_head * stride_buf_vh  # 头偏移
                + offs_dv[None, :]  # 维度偏移
            )
            v = tl.load(  # 加载Value向量
                V_Buffer + offs_buf_v,  # Value缓存地址
                mask=mask_n[:, None] & mask_dv[None, :],  # 掩码
                other=0.0,  # 越界填充值
            )
            p = p.to(v.dtype)  # 转换概率数据类型
            acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale  # 累加加权Value并乘以V缩放因子

            e_max = n_e_max  # 更新最大指数值

    # stage 2: compute the triangle part
    # 阶段2：计算三角部分（扩展KV的自注意力）

    cur_block_m_end = (  # 计算当前M块的结束位置
        cur_seq_len_extend  # 非因果模式：处理所有扩展token
        if not IS_CAUSAL  # 非因果注意力
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)  # 因果注意力：只处理当前块及之前
    )
    for start_n in range(0, cur_block_m_end, BLOCK_N):  # 遍历扩展KV
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 提示编译器start_n是BLOCK_N的倍数
        mask_n = (start_n + offs_n) < cur_block_m_end  # N维度掩码

        final_mask = mask_m[:, None] & mask_n[None, :]  # 合并M和N维度掩码
        if USE_CUSTOM_MASK:  # 使用自定义掩码
            custom_mask = tl.load(  # 加载自定义掩码
                mask_ptr  # 掩码基址
                + cur_seq_mask_start_idx  # 掩码起始偏移
                + (cur_block_m * BLOCK_M + offs_m[:, None])  # Query位置
                * (cur_seq_len + window_kv_offset)  # 掩码行宽
                + window_kv_offset  # 窗口偏移
                + cur_seq_len_prefix  # 跳过前缀部分
                + start_n  # KV起始位置
                + offs_n[None, :],  # KV偏移
                mask=(mask_m[:, None] & mask_n[None, :]),  # 掩码
                other=0,  # 越界填充值
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]  # 确保自定义掩码不超出范围
            final_mask &= custom_mask  # 合并自定义掩码
        elif IS_CAUSAL:  # 因果注意力
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (  # 因果掩码：q位置 >= k位置
                start_n + offs_n[None, :]  # k位置
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]  # 合并范围掩码
            final_mask &= mask_causual  # 合并因果掩码
        else:  # 非因果非自定义掩码
            mask_non_causal = mask_m[:, None] & mask_n[None, :]  # 全连接掩码
            final_mask &= mask_non_causal  # 合并

        if SLIDING_WINDOW_SIZE > 0:  # 如果有滑动窗口
            # Add mask where q_id <= kv_id + sliding_window_size
            # 添加q_id <= kv_id + sliding_window_size的掩码
            window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (  # 滑动窗口掩码
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE  # k位置 + 窗口大小
            )
            final_mask &= window_mask  # 合并滑动窗口掩码

        SKIP_TILE = False  # 初始化跳过标记
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:  # 有额外掩码
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0  # 检查是否所有掩码为0

        if not SKIP_TILE:  # 不跳过该块
            # load k in transposed way
            # 以转置方式加载Key
            offs_k = (  # 计算扩展Key偏移
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs  # token偏移
                + cur_kv_head * stride_kh  # 头偏移
                + offs_d[:, None]  # 维度偏移（转置）
            )
            k = tl.load(  # 加载Key向量
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0  # 带掩码加载
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)  # 计算QK点积
            if BLOCK_DPE > 0:  # MLA模式：额外计算位置编码
                offs_kpe = (  # 计算扩展Key位置编码偏移
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs  # token偏移
                    + cur_kv_head * stride_kh  # 头偏移
                    + offs_dpe[:, None]  # 位置编码维度偏移
                )
                kpe = tl.load(  # 加载Key位置编码
                    K_Extend + offs_kpe,  # Key扩展地址
                    mask=mask_n[None, :],  # 掩码
                    other=0.0,  # 越界填充值
                )
                qk += tl.dot(qpe, kpe)  # 累加位置编码的QK点积

            qk *= sm_scale  # 乘以softmax缩放因子（注意：扩展KV不乘k_scale）

            if logit_cap > 0:  # 如果启用logit截断
                qk = logit_cap * tanh(qk / logit_cap)  # 应用logit截断

            if xai_temperature_len > 0:  # 如果启用XAI温度调节
                qk *= xai_temperature_reg[:, None]  # 乘以温度调节系数

            qk = tl.where(final_mask, qk, float("-inf"))  # 应用最终掩码

            row_max = tl.max(qk, 1)  # 计算每行最大值
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)  # 修复全负无穷行
            n_e_max = tl.maximum(row_max_fixed, e_max)  # 更新全局最大值

            re_scale = tl.exp(e_max - n_e_max)  # 计算重缩放因子
            p = tl.exp(qk - n_e_max[:, None])  # 计算softmax概率
            deno = deno * re_scale + tl.sum(p, 1)  # 更新分母

            offs_v = (  # 计算扩展Value偏移
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs  # token偏移
                + cur_kv_head * stride_vh  # 头偏移
                + offs_dv[None, :]  # 维度偏移
            )
            v = tl.load(  # 加载Value向量
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0  # 带掩码加载
            )
            p = p.to(v.dtype)  # 转换概率数据类型
            acc = acc * re_scale[:, None] + tl.dot(p, v)  # 累加加权Value

            e_max = n_e_max  # 更新最大指数值

    if HAS_SINK:  # 如果有注意力汇聚
        cur_sink = tl.load(sink_ptr + cur_head)  # 加载汇聚值
        deno += tl.exp(cur_sink - e_max)  # 将汇聚值加入分母

    offs_o = (  # 计算输出偏移
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])  # token偏移
        * stride_obs  # 输出批次步长
        + cur_head * stride_oh  # 输出头步长
        + offs_dv[None, :]  # Value维度偏移
    )
    if STORE_TRANSPOSE:  # 如果需要转置存储（HIP平台优化）
        tl.store(  # 转置方式存储输出
            O_Extend + offs_o.T,  # 转置输出地址
            (acc / deno[:, None]).T,  # 转置归一化结果
            mask=(mask_m[:, None] & mask_dv[None, :]).T,  # 转置掩码
        )
    else:  # 正常存储
        tl.store(  # 存储输出
            O_Extend + offs_o,  # 输出地址
            acc / deno[:, None],  # 归一化结果
            mask=mask_m[:, None] & mask_dv[None, :],  # 掩码
        )


def extend_attention_fwd(  # 扩展注意力两阶段前向传播
    q_extend,  # 扩展Query张量
    k_extend,  # 扩展Key张量
    v_extend,  # 扩展Value张量
    o_extend,  # 扩展输出张量
    k_buffer,  # Key缓存（前缀+扩展）
    v_buffer,  # Value缓存（前缀+扩展）
    qo_indptr,  # Query/Output索引偏移
    kv_indptr,  # KV索引偏移
    kv_indices,  # KV索引
    custom_mask,  # 自定义掩码
    is_causal,  # 是否因果注意力
    mask_indptr,  # 掩码索引偏移
    max_len_extend,  # 最大扩展长度
    k_scale,  # Key缩放因子
    v_scale,  # Value缩放因子
    sm_scale=None,  # softmax缩放因子（可选）
    logit_cap=0.0,  # logit截断值（默认0）
    skip_prefix_custom_mask=True,  # 是否跳过前缀自定义掩码（默认True）
    sliding_window_size=-1,  # 滑动窗口大小（默认-1，不使用）
    sinks=None,  # 注意力汇聚（可选）
    window_kv_offsets=None,  # 滑动窗口KV偏移（可选）
    xai_temperature_len=-1,  # XAI温度长度（默认-1）
):  # 扩展注意力两阶段前向传播入口函数
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager

    q_extend, k_extend, v_extend, o_extend: 连续张量

    k_buffer, v_buffer: mem_manager中的（前缀+扩展）张量
    """
    Lq, Lk, Lv = (  # 获取Query、Key、Value头维度
        q_extend.shape[-1],  # Query头维度
        k_extend.shape[-1],  # Key头维度
        v_extend.shape[-1],  # Value头维度
    )

    # Get block sizes and configuration
    # 获取块大小和配置
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)  # 根据头维度获取配置
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)  # 默认softmax缩放为1/sqrt(d)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]  # 获取批次大小和头数
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]  # 计算KV组数

    USE_CUSTOM_MASK = custom_mask is not None  # 是否使用自定义掩码
    # Skip custom mask for prefix part
    # 跳过前缀部分的自定义掩码
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask  # 是否跳过前缀掩码

    HAS_SINK = sinks is not None  # 是否有注意力汇聚

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))  # 定义3D网格
    num_stages = 1  # 流水线阶段数

    extra_kargs = {}  # 额外内核参数
    if _is_hip:  # HIP平台额外参数
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}  # ROCm优化参数

    _fwd_kernel[grid](  # 启动两阶段扩展注意力内核
        q_extend,  # 扩展Query
        k_extend,  # 扩展Key
        v_extend,  # 扩展Value
        o_extend,  # 扩展输出
        k_buffer,  # Key缓存
        v_buffer,  # Value缓存
        qo_indptr,  # Query/Output索引偏移
        kv_indptr,  # KV索引偏移
        kv_indices,  # KV索引
        custom_mask,  # 自定义掩码
        mask_indptr,  # 掩码索引偏移
        sinks,  # 注意力汇聚
        window_kv_offsets,  # 滑动窗口KV偏移
        sm_scale,  # softmax缩放因子
        k_scale,  # Key缩放因子
        v_scale,  # Value缩放因子
        kv_group_num,  # KV组数
        q_extend.stride(0),  # Query批次步长
        q_extend.stride(1),  # Query头步长
        k_extend.stride(0),  # Key批次步长
        k_extend.stride(1),  # Key头步长
        v_extend.stride(0),  # Value批次步长
        v_extend.stride(1),  # Value头步长
        o_extend.stride(0),  # 输出批次步长
        o_extend.stride(1),  # 输出头步长
        k_buffer.stride(0),  # Key缓存批次步长
        k_buffer.stride(1),  # Key缓存头步长
        v_buffer.stride(0),  # Value缓存批次步长
        v_buffer.stride(1),  # Value缓存头步长
        SLIDING_WINDOW_SIZE=sliding_window_size,  # 滑动窗口大小
        logit_cap=logit_cap,  # logit截断值
        xai_temperature_len=xai_temperature_len,  # XAI温度长度
        BLOCK_DMODEL=BLOCK_DMODEL,  # 模型维度块大小
        BLOCK_DPE=BLOCK_DPE,  # 位置编码维度块大小
        BLOCK_DV=BLOCK_DV,  # Value维度块大小
        BLOCK_M=BLOCK_M,  # M维度块大小
        BLOCK_N=BLOCK_N,  # N维度块大小
        Lq=Lq,  # Query头维度
        Lv=Lv,  # Value头维度
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,  # 是否使用自定义掩码
        IS_CAUSAL=is_causal,  # 是否因果注意力
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,  # 是否跳过前缀掩码
        HAS_SINK=HAS_SINK,  # 是否有注意力汇聚
        STORE_TRANSPOSE=_is_hip,  # HIP平台转置存储
        num_warps=num_warps,  # warp数量
        num_stages=num_stages,  # 流水线阶段数
        **extra_kargs,  # 额外参数
    )


def redundant_attention(  # 冗余注意力计算（用于调试/验证）
    q_extend,  # 扩展Query
    o_extend,  # 扩展输出
    k_buffer,  # Key缓存
    v_buffer,  # Value缓存
    b_req_idx,  # 请求索引
    b_start_loc,  # 起始位置
    b_seq_len,  # 序列长度
    b_seq_len_prefix,  # 前缀序列长度
    max_len_in_batch,  # 批次内最大长度
):  # 通过重构完整Query并调用上下文注意力来计算冗余注意力
    """冗余注意力计算，通过重构完整Query并调用上下文注意力实现，用于调试/验证"""
    total_token_num = k_buffer.shape[0]  # 总token数
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]  # 批次大小、头数、维度
    q_buffer = torch.empty(  # 分配完整Query缓冲区
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device  # 形状为(总token数, 头数, 维度)
    )

    pt = 0  # 初始化指针
    for i in range(B):  # 遍历每个序列
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]  # 计算扩展序列长度
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]  # 计算位置范围
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]  # 将扩展Query放入完整缓冲区
        pt += cur_seq_len_extend  # 推进指针

    o_buffer = torch.empty_like(q_buffer)  # 分配输出缓冲区
    context_attention_fwd(  # 调用上下文注意力前向传播
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch  # 传入所有参数
    )

    pt = 0  # 重置指针
    for i in range(B):  # 遍历每个序列
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]  # 计算扩展序列长度
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]  # 计算位置范围
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]  # 提取扩展部分的输出
        pt += cur_seq_len_extend  # 推进指针


@triton.jit  # Triton JIT编译装饰器
def _fwd_kernel_unified(  # 扩展注意力统一单阶段内核
    Q,  # Query指针
    O,  # 输出指针
    K_Buffer,  # Key缓存指针
    V_Buffer,  # Value缓存指针
    qo_indptr,  # Query/Output索引偏移
    kv_indptr,  # KV索引偏移
    kv_indices,  # 统一KV索引
    prefix_lens,  # 前缀长度
    mask_ptr,  # 自定义掩码指针
    mask_indptr,  # 掩码索引偏移
    sink_ptr,  # 注意力汇聚指针
    window_start_pos,  # 滑动窗口起始位置
    sm_scale_withk,  # 包含K缩放的softmax缩放因子
    v_scale,  # Value缩放因子
    kv_group_num,  # KV组数
    stride_qbs,  # Query批次步长
    stride_qh,  # Query头步长
    stride_obs,  # 输出批次步长
    stride_oh,  # 输出头步长
    stride_buf_kbs,  # Key缓存批次步长
    stride_buf_kh,  # Key缓存头步长
    stride_buf_vbs,  # Value缓存批次步长
    stride_buf_vh,  # Value缓存头步长
    SLIDING_WINDOW_SIZE: tl.constexpr,  # 滑动窗口大小（编译时常量）
    logit_cap: tl.constexpr,  # logit截断值（编译时常量）
    xai_temperature_len: tl.constexpr,  # XAI温度长度（编译时常量）
    Lq: tl.constexpr,  # Query头维度（编译时常量）
    Lv: tl.constexpr,  # Value头维度（编译时常量）
    BLOCK_DMODEL: tl.constexpr,  # 模型维度块大小（编译时常量）
    BLOCK_DPE: tl.constexpr,  # 位置编码维度块大小（编译时常量）
    BLOCK_DV: tl.constexpr,  # Value维度块大小（编译时常量）
    BLOCK_M: tl.constexpr,  # M维度块大小（编译时常量）
    BLOCK_N: tl.constexpr,  # N维度块大小（编译时常量）
    IS_CAUSAL: tl.constexpr,  # 是否因果注意力（编译时常量）
    USE_CUSTOM_MASK: tl.constexpr,  # 是否使用自定义掩码（编译时常量）
    HAS_SINK: tl.constexpr,  # 是否有注意力汇聚（编译时常量）
):  # 统一单阶段扩展注意力内核，通过统一KV索引一次性处理所有KV
    """
    Unified 1-stage kernel for deterministic extend attention.
    Both prefix and extend KV are accessed through the unified kv_indices.

    确定性扩展注意力的统一单阶段内核。
    前缀和扩展KV都通过统一的kv_indices访问。
    """
    cur_seq = tl.program_id(0)  # 获取当前序列索引
    cur_head = tl.program_id(1)  # 获取当前头索引
    cur_block_m = tl.program_id(2)  # 获取当前M维度块索引
    cur_kv_head = cur_head // kv_group_num  # 计算当前KV头索引

    # Load sequence information
    # 加载序列信息
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)  # 加载Query起始索引
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx  # 计算Query长度
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)  # 加载KV起始索引
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx  # 计算KV长度
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)  # 加载前缀长度

    # Load window start position for sliding window attention
    # This is the absolute position of the first key in the window (0 if no sliding window)
    # 加载滑动窗口注意力的窗口起始位置
    # 这是窗口中第一个key的绝对位置（0表示无滑动窗口）
    cur_window_start = 0  # 初始化窗口起始位置
    if SLIDING_WINDOW_SIZE > 0:  # 如果有滑动窗口
        cur_window_start = tl.load(window_start_pos + cur_seq)  # 加载窗口起始位置

    # Load custom mask start index if using custom mask (for speculative decoding)
    # 如果使用自定义掩码（用于推测解码），加载自定义掩码起始索引
    if USE_CUSTOM_MASK:  # 使用自定义掩码
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)  # 加载掩码起始索引

    offs_d = tl.arange(0, BLOCK_DMODEL)  # 模型维度偏移
    offs_dv = tl.arange(0, BLOCK_DV)  # Value维度偏移
    offs_m = tl.arange(0, BLOCK_M)  # M维度偏移
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len  # M维度掩码
    mask_d = offs_d < Lq  # Key维度掩码
    mask_dv = offs_dv < Lv  # Value维度掩码

    # XAI temperature handling
    # XAI温度处理
    if xai_temperature_len > 0:  # 如果启用XAI温度调节
        offs_qidx = cur_seq_prefix_len + cur_block_m * BLOCK_M + offs_m  # Query位置索引
        xai_temperature_reg = tl.where(  # 计算温度调节系数
            offs_qidx < xai_temperature_len,  # 位置小于温度长度
            1.0,  # 不缩放
            xai_temperature_len / (offs_qidx + 1.0),  # 按位置缩放
        )

    # Load Q
    # 加载Query
    offs_q = (  # 计算Query偏移
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs  # token偏移
        + cur_head * stride_qh  # 头偏移
        + offs_d[None, :]  # 维度偏移
    )
    q = tl.load(Q + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0)  # 加载Query向量

    if BLOCK_DPE > 0:  # 如果有位置编码维度（MLA模式）
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)  # 位置编码维度偏移
        offs_qpe = (  # 计算Query位置编码偏移
            (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs  # token偏移
            + cur_head * stride_qh  # 头偏移
            + offs_dpe[None, :]  # 位置编码维度偏移
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None], other=0.0)  # 加载Query位置编码

    # Initialize accumulators
    # 初始化累加器
    offs_n = tl.arange(0, BLOCK_N)  # N维度偏移
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)  # 初始化Value累加器
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)  # 初始化分母
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 初始化最大指数值

    # Unified loop: process all KV tokens (prefix + extend)
    # 统一循环：处理所有KV token（前缀+扩展）
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):  # 遍历所有KV token
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 提示编译器start_n是BLOCK_N的倍数
        mask_n = (start_n + offs_n) < cur_seq_kv_len  # N维度掩码

        # Compute mask
        # 计算掩码
        final_mask = mask_m[:, None] & mask_n[None, :]  # 合并M和N维度掩码

        # Apply custom mask if provided
        # 如果提供了自定义掩码，则应用
        if USE_CUSTOM_MASK:  # 使用自定义掩码
            custom_mask = tl.load(  # 加载自定义掩码
                mask_ptr  # 掩码基址
                + cur_seq_mask_start_idx  # 掩码起始偏移
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_kv_len  # Query位置 * KV长度
                + start_n  # KV起始位置
                + offs_n[None, :],  # KV偏移
                mask=(mask_m[:, None] & mask_n[None, :]),  # 掩码
                other=0,  # 越界填充值
            )
            final_mask &= custom_mask  # 合并自定义掩码

        # Apply causal mask for extend part
        # 对扩展部分应用因果掩码
        if IS_CAUSAL and not USE_CUSTOM_MASK:  # 因果注意力且无自定义掩码
            # Determine if current KV block is in extend region
            # Only apply causal mask when both Q and K are in extend region
            # 确定当前KV块是否在扩展区域
            # 仅当Q和K都在扩展区域时应用因果掩码
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]  # Query位置索引
            k_idx_in_total = start_n + offs_n[None, :]  # Key在统一数组中的位置索引

            # Causal mask: q_idx >= (k_idx - prefix_len) when k_idx >= prefix_len
            # For prefix region (k_idx < prefix_len), no causal mask
            # 因果掩码：当k_idx >= prefix_len时，q_idx >= (k_idx - prefix_len)
            # 对于前缀区域（k_idx < prefix_len），无因果掩码
            k_is_extend = k_idx_in_total >= cur_seq_prefix_len  # Key是否在扩展区域
            k_idx_in_extend = k_idx_in_total - cur_seq_prefix_len  # Key在扩展区域的索引
            causal_mask = tl.where(  # 计算因果掩码
                k_is_extend,  # Key在扩展区域
                q_idx >= k_idx_in_extend,  # Q位置 >= K位置（因果约束）
                True,  # No causal mask for prefix # 前缀区域无因果掩码
            )
            final_mask &= causal_mask  # 合并因果掩码

        if SLIDING_WINDOW_SIZE > 0:  # 如果有滑动窗口
            # Sliding window mask with correct absolute positions
            # Q absolute position: window_start + prefix_len + q_position_in_extend
            # 使用正确绝对位置的滑动窗口掩码
            # Q绝对位置：window_start + prefix_len + q在扩展中的位置
            q_abs_pos = (  # 计算Query绝对位置
                cur_window_start  # 窗口起始位置
                + cur_seq_prefix_len  # 前缀长度
                + cur_block_m * BLOCK_M  # 块偏移
                + offs_m[:, None]  # 块内偏移
            )

            # K absolute position: window_start + k_index_in_unified_array
            # K绝对位置：window_start + K在统一数组中的索引
            k_abs_pos = cur_window_start + start_n + offs_n[None, :]  # 计算Key绝对位置

            # Sliding window: query can attend to keys within window_size
            # 滑动窗口：query可以关注窗口内的key
            window_mask = q_abs_pos <= (k_abs_pos + SLIDING_WINDOW_SIZE)  # 滑动窗口掩码
            final_mask &= window_mask  # 合并滑动窗口掩码

        # Check if we can skip this tile
        # 检查是否可以跳过该块
        SKIP_TILE = False  # 初始化跳过标记
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:  # 有额外掩码
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0  # 检查是否所有掩码为0

        if not SKIP_TILE:  # 不跳过该块
            # Load KV indices
            # 加载KV索引
            offs_kv_loc = tl.load(  # 加载KV位置索引
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,  # 从统一索引数组获取
                mask=mask_n,  # 掩码
                other=0,  # 越界填充值
            )

            # Load K
            # 加载Key
            offs_buf_k = (  # 计算Key缓存偏移
                offs_kv_loc[None, :] * stride_buf_kbs  # token位置偏移
                + cur_kv_head * stride_buf_kh  # 头偏移
                + offs_d[:, None]  # 维度偏移（转置）
            )
            k = tl.load(  # 加载Key向量
                K_Buffer + offs_buf_k,  # Key缓存地址
                mask=(mask_n[None, :]) & (mask_d[:, None]),  # 掩码
                other=0.0,  # 越界填充值
            )

            qk = tl.dot(q.to(k.dtype), k)  # 计算QK点积
            if BLOCK_DPE > 0:  # MLA模式：额外计算位置编码
                offs_kpe = (  # 计算Key位置编码偏移
                    offs_kv_loc[None, :] * stride_buf_kbs  # token位置偏移
                    + cur_kv_head * stride_buf_kh  # 头偏移
                    + offs_dpe[:, None]  # 位置编码维度偏移
                )
                kpe = tl.load(  # 加载Key位置编码
                    K_Buffer + offs_kpe,  # Key缓存地址
                    mask=mask_n[None, :],  # 掩码
                    other=0.0,  # 越界填充值
                )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)  # 累加位置编码的QK点积

            qk *= sm_scale_withk  # 乘以包含K缩放的softmax缩放因子

            if logit_cap > 0:  # 如果启用logit截断
                qk = logit_cap * tanh(qk / logit_cap)  # 应用logit截断

            if xai_temperature_len > 0:  # 如果启用XAI温度调节
                qk *= xai_temperature_reg[:, None]  # 乘以温度调节系数

            qk = tl.where(final_mask, qk, float("-inf"))  # 应用最终掩码

            # Online softmax
            # 在线softmax
            row_max = tl.max(qk, 1)  # 计算每行最大值
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)  # 修复全负无穷行
            n_e_max = tl.maximum(row_max_fixed, e_max)  # 更新全局最大值

            re_scale = tl.exp(e_max - n_e_max)  # 计算重缩放因子
            p = tl.exp(qk - n_e_max[:, None])  # 计算softmax概率
            deno = deno * re_scale + tl.sum(p, 1)  # 更新分母

            # Load V
            # 加载Value
            offs_buf_v = (  # 计算Value缓存偏移
                offs_kv_loc[:, None] * stride_buf_vbs  # token位置偏移
                + cur_kv_head * stride_buf_vh  # 头偏移
                + offs_dv[None, :]  # 维度偏移
            )
            v = tl.load(  # 加载Value向量
                V_Buffer + offs_buf_v,  # Value缓存地址
                mask=mask_n[:, None] & mask_dv[None, :],  # 掩码
                other=0.0,  # 越界填充值
            )
            p = p.to(v.dtype)  # 转换概率数据类型
            acc = acc * re_scale[:, None] + tl.dot(p, v)  # 累加加权Value

            e_max = n_e_max  # 更新最大指数值

    # Handle sink tokens
    # 处理注意力汇聚token
    if HAS_SINK:  # 如果有注意力汇聚
        cur_sink = tl.load(sink_ptr + cur_head)  # 加载汇聚值
        deno += tl.exp(cur_sink - e_max)  # 将汇聚值加入分母

    # Store output
    # 存储输出
    offs_o = (  # 计算输出偏移
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs  # token偏移
        + cur_head * stride_oh  # 头偏移
        + offs_dv[None, :]  # Value维度偏移
    )
    tl.store(  # 存储最终输出
        O + offs_o,  # 输出地址
        acc / deno[:, None] * v_scale,  # 归一化后乘以Value缩放因子
        mask=mask_m[:, None] & mask_dv[None, :],  # 掩码
    )


def extend_attention_fwd_unified(  # 扩展注意力统一单阶段前向传播
    q,  # Query张量
    o,  # 输出张量
    k_buffer,  # Key缓存
    v_buffer,  # Value缓存
    k_scale,  # Key缩放因子
    v_scale,  # Value缩放因子
    qo_indptr,  # Query/Output索引偏移
    kv_indptr,  # KV索引偏移
    kv_indices,  # 统一KV索引
    prefix_lens,  # 前缀长度
    max_len_extend,  # 最大扩展长度
    custom_mask=None,  # 自定义掩码（可选）
    mask_indptr=None,  # 掩码索引偏移（可选）
    sm_scale=None,  # softmax缩放因子（可选）
    logit_cap=0.0,  # logit截断值（默认0）
    is_causal=True,  # 是否因果注意力（默认True）
    sliding_window_size=-1,  # 滑动窗口大小（默认-1）
    sinks=None,  # 注意力汇聚（可选）
    window_start_pos=None,  # 滑动窗口起始位置（可选）
    xai_temperature_len=-1,  # XAI温度长度（默认-1）
):  # 统一单阶段扩展注意力前向传播入口函数
    """
    Unified 1-stage extend attention for deterministic inference.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim]
        o: Output tensor [num_tokens, num_heads, head_dim]
        k_buffer: Key cache buffer
        v_buffer: Value cache buffer
        qo_indptr: Query offsets [batch_size + 1]
        kv_indptr: KV offsets [batch_size + 1] (includes both prefix and extend)
        kv_indices: Unified KV indices (both prefix and extend)
        prefix_lens: Prefix length for each sequence [batch_size]
        max_len_extend: Maximum extend length
        custom_mask: Custom attention mask (for speculative decoding tree attention)
        mask_indptr: Mask offsets [batch_size + 1]
        sm_scale: Softmax scale
        logit_cap: Logit capping value
        is_causal: Whether to apply causal mask
        sliding_window_size: Sliding window size (-1 for no sliding window)
        sinks: Sink tokens
        window_start_pos: Absolute position of first key in sliding window [batch_size]
                         (None if sliding window not used)
        xai_temperature_len: XAI temperature length

    确定性推理的统一单阶段扩展注意力。

    参数：
        q: Query张量 [num_tokens, num_heads, head_dim]
        o: 输出张量 [num_tokens, num_heads, head_dim]
        k_buffer: Key缓存缓冲区
        v_buffer: Value缓存缓冲区
        qo_indptr: Query偏移 [batch_size + 1]
        kv_indptr: KV偏移 [batch_size + 1]（包含前缀和扩展）
        kv_indices: 统一KV索引（前缀和扩展）
        prefix_lens: 每个序列的前缀长度 [batch_size]
        max_len_extend: 最大扩展长度
        custom_mask: 自定义注意力掩码（用于推测解码树注意力）
        mask_indptr: 掩码偏移 [batch_size + 1]
        sm_scale: Softmax缩放
        logit_cap: Logit截断值
        is_causal: 是否应用因果掩码
        sliding_window_size: 滑动窗口大小（-1表示不使用滑动窗口）
        sinks: 汇聚token
        window_start_pos: 滑动窗口中第一个key的绝对位置 [batch_size]
                         （如果不使用滑动窗口则为None）
        xai_temperature_len: XAI温度长度
    """
    Lq, Lv = q.shape[-1], v_buffer.shape[-1]  # 获取Query和Value头维度

    # Get block sizes and configuration
    # 获取块大小和配置
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)  # 根据头维度获取配置
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)  # 默认softmax缩放为1/sqrt(d)
    batch_size, head_num = qo_indptr.shape[0] - 1, q.shape[1]  # 获取批次大小和头数
    kv_group_num = q.shape[1] // k_buffer.shape[1]  # 计算KV组数

    USE_CUSTOM_MASK = custom_mask is not None  # 是否使用自定义掩码
    HAS_SINK = sinks is not None  # 是否有注意力汇聚

    # For sliding window attention, window_start_pos tracks the absolute position
    # of the first key in each sequence's window
    # 对于滑动窗口注意力，window_start_pos跟踪每个序列窗口中第一个key的绝对位置
    if sliding_window_size > 0 and window_start_pos is None:  # 有滑动窗口但未提供起始位置
        # If not provided, assume window starts at position 0
        # 如果未提供，假设窗口从位置0开始
        window_start_pos = torch.zeros(batch_size, dtype=torch.int32, device=q.device)  # 创建全零起始位置

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))  # 定义3D网格
    num_stages = 1  # 流水线阶段数

    extra_kargs = {}  # 额外内核参数
    if _is_hip:  # HIP平台额外参数
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}  # ROCm优化参数

    _fwd_kernel_unified[grid](  # 启动统一单阶段扩展注意力内核
        q,  # Query张量
        o,  # 输出张量
        k_buffer,  # Key缓存
        v_buffer,  # Value缓存
        qo_indptr,  # Query/Output索引偏移
        kv_indptr,  # KV索引偏移
        kv_indices,  # 统一KV索引
        prefix_lens,  # 前缀长度
        custom_mask,  # 自定义掩码
        mask_indptr,  # 掩码索引偏移
        sinks,  # 注意力汇聚
        window_start_pos,  # 滑动窗口起始位置
        sm_scale * k_scale,  # 合并softmax和Key缩放因子
        v_scale,  # Value缩放因子
        kv_group_num,  # KV组数
        q.stride(0),  # Query批次步长
        q.stride(1),  # Query头步长
        o.stride(0),  # 输出批次步长
        o.stride(1),  # 输出头步长
        k_buffer.stride(0),  # Key缓存批次步长
        k_buffer.stride(1),  # Key缓存头步长
        v_buffer.stride(0),  # Value缓存批次步长
        v_buffer.stride(1),  # Value缓存头步长
        SLIDING_WINDOW_SIZE=sliding_window_size,  # 滑动窗口大小
        logit_cap=logit_cap,  # logit截断值
        xai_temperature_len=xai_temperature_len,  # XAI温度长度
        BLOCK_DMODEL=BLOCK_DMODEL,  # 模型维度块大小
        BLOCK_DPE=BLOCK_DPE,  # 位置编码维度块大小
        BLOCK_DV=BLOCK_DV,  # Value维度块大小
        BLOCK_M=BLOCK_M,  # M维度块大小
        BLOCK_N=BLOCK_N,  # N维度块大小
        Lq=Lq,  # Query头维度
        Lv=Lv,  # Value头维度
        IS_CAUSAL=is_causal,  # 是否因果注意力
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,  # 是否使用自定义掩码
        HAS_SINK=HAS_SINK,  # 是否有注意力汇聚
        num_warps=num_warps,  # warp数量
        num_stages=num_stages,  # 流水线阶段数
        **extra_kargs,  # 额外参数
    )
