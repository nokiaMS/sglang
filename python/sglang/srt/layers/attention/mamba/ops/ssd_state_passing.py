# SSD状态传递模块 - 实现Mamba2状态空间模型中块间状态的递推传递
# 包含Triton GPU内核，用于高效计算SSD的分块状态传递（inter-chunk recurrence）

# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py
# 改编自: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py

# SPDX-License-Identifier: Apache-2.0
# SPDX许可证标识符: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX文件版权文本: vLLM项目的版权贡献者

# Copyright (c) 2024, Tri Dao, Albert Gu.
# 版权所有 (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py
# 改编自 https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# ruff: noqa: E501
# ruff: 忽略 E501(行长度)

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块，别名为tl


@triton.jit  # Triton JIT编译装饰器
def _state_passing_fwd_kernel(  # 状态传递前向计算内核
    # Pointers to matrices
    # 矩阵指针
    states_ptr,  # 分块状态输入指针
    out_ptr,  # 传递后状态输出指针
    final_states_ptr,  # 最终状态输出指针
    dA_cs_ptr,  # dA累积和指针
    initstates_ptr,  # 初始状态指针
    seq_idx_ptr,  # 序列索引指针
    chunk_offsets_ptr,  # 分块偏移指针
    chunk_meta_num,  # 分块元数据数量
    # Matrix dimensions
    # 矩阵维度
    dim,  # 状态维度（headdim * dstate展平后）
    nchunks,  # 分块数量
    seqlen,  # 序列长度
    chunk_size,  # 分块大小
    # Strides
    # 步幅
    stride_states_batch,  # 状态批次步幅
    stride_states_chunk,  # 状态分块步幅
    stride_states_head,  # 状态头步幅
    stride_states_dim,  # 状态维度步幅
    stride_out_batch,  # 输出批次步幅
    stride_out_chunk,  # 输出分块步幅
    stride_out_head,  # 输出头步幅
    stride_out_dim,  # 输出维度步幅
    stride_final_states_batch,  # 最终状态批次步幅
    stride_final_states_head,  # 最终状态头步幅
    stride_final_states_dim,  # 最终状态维度步幅
    stride_dA_cs_batch,  # dA累积和批次步幅
    stride_dA_cs_chunk,  # dA累积和分块步幅
    stride_dA_cs_head,  # dA累积和头步幅
    stride_dA_cs_csize,  # dA累积和分块大小步幅
    stride_initstates_batch,  # 初始状态批次步幅
    stride_initstates_head,  # 初始状态头步幅
    stride_initstates_dim,  # 初始状态维度步幅
    stride_seq_idx_batch,  # 序列索引批次步幅
    stride_seq_idx_seqlen,  # 序列索引序列步幅
    # Meta-parameters
    # 元参数
    HAS_INITSTATES: tl.constexpr,  # 是否有初始状态
    HAS_SEQ_IDX: tl.constexpr,  # 是否有序列索引
    IS_CONT_BATCHED: tl.constexpr,  # 是否连续批处理
    BLOCK_SIZE: tl.constexpr = 16,  # 块大小，默认16
):
    pid_b = tl.program_id(axis=1)  # 获取批次维度的程序ID
    pid_h = tl.program_id(axis=2)  # 获取头维度的程序ID
    pid_m = tl.program_id(axis=0)  # 获取维度方向的程序ID
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head  # 更新状态指针
    dA_cs_ptr += (  # 更新dA累积和指针
        pid_b * stride_dA_cs_batch  # 批次偏移
        + pid_h * stride_dA_cs_head  # 头偏移
        + (chunk_size - 1) * stride_dA_cs_csize  # 偏移到分块末尾
    )
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head  # 更新输出指针
    final_states_ptr += (  # 更新最终状态指针
        pid_b * stride_final_states_batch + pid_h * stride_final_states_head  # 批次和头偏移
    )
    if HAS_INITSTATES:  # 如果有初始状态
        initstates_ptr += pid_h * stride_initstates_head  # 头偏移
        if not IS_CONT_BATCHED:  # 如果不是连续批处理
            initstates_ptr += pid_b * stride_initstates_batch  # 加上批次偏移

    if HAS_SEQ_IDX:  # 如果有序列索引
        seq_idx_ptr += pid_b * stride_seq_idx_batch  # 更新序列索引指针

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算维度偏移
    states_ptrs = states_ptr + offs_m * stride_states_dim  # 计算状态加载位置
    out_ptrs = out_ptr + offs_m * stride_out_dim  # 计算输出存储位置
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim  # 计算最终状态存储位置

    # - states will be the past state of the sequence that continues on the current check
    # - states将是延续到当前分块的序列的过去状态
    if not HAS_INITSTATES:  # 如果没有初始状态
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)  # 初始化状态为零向量
    else:  # 有初始状态
        initstates_ptr += offs_m * stride_initstates_dim  # 计算初始状态加载位置
        initstates_ptrs = initstates_ptr  # 保存初始状态指针
        # - for cont batches, for the first chunk mean it will be the first batch's
        #   init state
        # - 对于连续批处理，第一个分块意味着它将是第一个批次的初始状态
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)  # 加载初始状态

    tl.store(out_ptrs, states, mask=offs_m < dim)  # 存储第一个分块的输出状态（初始状态或零）
    out_ptrs += stride_out_chunk  # 移动到下一个分块的输出位置
    prev_seq_idx_chunk_end = 0  # 前一个分块末尾的序列索引，初始化为0
    logical_chunk_idx = 0  # 逻辑分块索引，初始化为0
    for c in range(nchunks):  # 遍历所有分块
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)  # 加载当前分块的新状态
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)  # 加载当前分块末尾的dA累积和
        scale_mask = True  # 缩放掩码，默认为True
        if HAS_SEQ_IDX:  # 如果有序列索引
            # - the seq to pass forward is the one that is flushed to the right
            #   boundary.
            # - that is given by seq_idx_chunk_end below: the sequence index at the end of the chunk.
            # - 要向前传递的序列是已刷新到右边界的那个
            # - 由下面的seq_idx_chunk_end给出：分块末尾的序列索引
            seq_idx_chunk_end = tl.load(  # 加载分块末尾的序列索引
                seq_idx_ptr  # 序列索引基址
                + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen  # 分块末尾位置
            )
            if HAS_INITSTATES:  # 如果有初始状态
                if IS_CONT_BATCHED and prev_seq_idx_chunk_end != seq_idx_chunk_end:  # 连续批处理且序列发生变化
                    # this means in the current chunk the rightmost flushed seq
                    # has changed.
                    # - so we do not propagate the state from previous chunk
                    # - but rather we load that sequence's init state
                    # 这意味着当前分块中最右侧已刷新的序列已改变
                    # - 因此不从上一个分块传播状态
                    # - 而是加载该序列的初始状态
                    initstates_ptrs = (  # 计算新序列的初始状态指针
                        initstates_ptr + seq_idx_chunk_end * stride_initstates_batch  # 对应序列的偏移
                    )

                    # - update state with seq_idx_new's init state
                    # - 用seq_idx_new的初始状态更新状态
                    states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(  # 加载初始状态
                        tl.float32  # 转换为float32
                    )

                    # - we need to consider the cumsum only of the last sequence in the chunk
                    # - find its starting position (given by c_off of the logical chunk index)
                    # - and subtract the cumsum just before that position from the total cumsum
                    # - first, update the logical chunk index (add the number of sequences in the current physical chunk):
                    #   sequence index at the start of the current chunk
                    # - 需要只考虑分块中最后一个序列的累积和
                    # - 找到其起始位置（由逻辑分块索引的c_off给出）
                    # - 从总累积和中减去该位置之前的累积和
                    # - 首先，更新逻辑分块索引（加上当前物理分块中的序列数）：
                    #   当前分块起始处的序列索引
                    seq_idx_chunk_start = tl.load(  # 加载分块起始处的序列索引
                        seq_idx_ptr  # 序列索引基址
                        + min(c * chunk_size, seqlen) * stride_seq_idx_seqlen  # 分块起始位置
                    )
                    logical_chunk_idx += seq_idx_chunk_end - seq_idx_chunk_start  # 更新逻辑分块索引
                    # - load the chunk offset:
                    # - 加载分块偏移:
                    c_off = tl.load(  # 加载逻辑分块的偏移量
                        chunk_offsets_ptr + logical_chunk_idx,  # 逻辑分块索引位置
                        mask=logical_chunk_idx < chunk_meta_num,  # 掩码
                        other=0,  # 默认值0
                    )
                    # - if offset is 0, then the sequence starts at the beginning of the chunk, and we don't need to subtract anything
                    # - 如果偏移为0，则序列从分块开头开始，无需减去任何内容
                    if c_off > 0:  # 如果偏移大于0
                        # - dA_cs_ptr currently points to the cumsum at the end of the chunk - subtract the chunk size and add the offset
                        # - dA_cs_ptr当前指向分块末尾的累积和 - 减去分块大小并加上偏移
                        dA_cs_boundary = tl.load(  # 加载边界处的dA累积和
                            dA_cs_ptr  # dA累积和指针
                            - (chunk_size - 1) * stride_dA_cs_csize  # 回到分块起始
                            + (c_off - 1) * stride_dA_cs_csize,  # 加上偏移-1
                            mask=(c_off - 1) > -1 and c_off < chunk_size,  # 掩码条件
                            other=0.0,  # 默认值
                        )
                        dA_cs -= dA_cs_boundary  # 从总累积和中减去边界累积和

                # - increment logical chunk index for every physical chunk
                # - 每个物理分块增加逻辑分块索引
                logical_chunk_idx += 1  # 递增逻辑分块索引
            else:  # 没有初始状态（非连续批处理）
                scale_mask = seq_idx_chunk_end == prev_seq_idx_chunk_end  # 序列相同时掩码为True
            prev_seq_idx_chunk_end = seq_idx_chunk_end  # 更新前一个分块末尾的序列索引

        scale = tl.where(scale_mask, tl.exp(dA_cs), 0.0)  # 计算缩放因子：掩码为True时exp(dA_cs)，否则为0
        states = scale * states + new_states  # 状态递推: states = exp(dA_cs) * prev_states + new_states
        if c < nchunks - 1:  # 如果不是最后一个分块
            tl.store(out_ptrs, states, mask=offs_m < dim)  # 存储到输出
        else:  # 最后一个分块
            tl.store(final_states_ptrs, states, mask=offs_m < dim)  # 存储到最终状态
        states_ptrs += stride_states_chunk  # 移动到下一个分块的状态
        dA_cs_ptr += stride_dA_cs_chunk  # 移动到下一个分块的dA累积和
        out_ptrs += stride_out_chunk  # 移动到下一个分块的输出位置


def _state_passing_fwd(  # 状态传递前向计算的主机端调用函数
    states,  # 分块状态输入
    dA_cumsum,  # dA累积和
    initial_states=None,  # 初始状态，可选
    seq_idx=None,  # 序列索引，可选
    chunk_size=None,  # 分块大小，可选
    out_dtype=None,  # 输出数据类型，可选
    is_cont_batching=False,  # 是否连续批处理
    chunk_offsets=None,  # 分块偏移，可选
):
    batch, nchunks, nheads, dim = states.shape  # 解包状态的形状
    if chunk_size is None:  # 如果没有指定分块大小
        chunk_size = dA_cumsum.shape[-1]  # 从dA累积和推断
    else:  # 指定了分块大小
        assert chunk_size == dA_cumsum.shape[-1]  # 断言与dA_cumsum一致
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)  # 断言dA_cumsum的形状
    if initial_states is not None:  # 如果有初始状态
        if is_cont_batching:  # 如果是连续批处理
            # - if cu_seqlens is provided, then the initial states
            #   are used for continuous batching. In which case we
            #   require seq_idx to be provided
            # - 如果提供了cu_seqlens，则初始状态用于连续批处理
            #   此时要求提供seq_idx
            assert (
                seq_idx is not None
            ), "seq_idx must be provided for continuous batching"  # 断言seq_idx必须提供
            # - we also need chunk_offsets to be provided, to account
            #   for computation of dA_cumsum from the start of the
            #   sequence
            # - 还需要提供chunk_offsets，以处理从序列起始计算dA_cumsum
            assert (
                chunk_offsets is not None
            ), "chunk_offsets must be provided for continuous batching"  # 断言chunk_offsets必须提供
        else:  # 非连续批处理
            # - this is the regular batching case, where initial
            #   states are used are for each example of the batch.
            # - 这是常规批处理情况，初始状态用于批次中的每个样本
            assert initial_states.shape == (batch, nheads, dim)  # 断言初始状态的形状

    if seq_idx is not None:  # 如果有序列索引
        seqlen = seq_idx.shape[-1]  # 获取序列长度
        assert seq_idx.shape == (batch, seqlen)  # 断言seq_idx的形状
    out_dtype = states.dtype if out_dtype is None else out_dtype  # 确定输出数据类型
    out = torch.empty(  # 创建输出张量
        (batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype  # 形状和属性
    )
    final_states = torch.empty(  # 创建最终状态张量
        (batch, nheads, dim), device=states.device, dtype=torch.float32  # 使用float32保证精度
    )
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)  # 定义GPU内核的网格大小
    with torch.get_device_module(states.device).device(states.device.index):  # 在对应设备上执行
        _state_passing_fwd_kernel[grid](  # 启动状态传递前向内核
            states,  # 分块状态
            out,  # 输出
            final_states,  # 最终状态
            dA_cumsum,  # dA累积和
            initial_states,  # 初始状态
            seq_idx,  # 序列索引
            chunk_offsets,  # 分块偏移
            len(chunk_offsets) if chunk_offsets is not None else 0,  # 分块元数据数量
            dim,  # 维度
            nchunks,  # 分块数量
            seqlen if seq_idx is not None else 0,  # 序列长度或0
            chunk_size,  # 分块大小
            states.stride(0),  # 状态批次步幅
            states.stride(1),  # 状态分块步幅
            states.stride(2),  # 状态头步幅
            states.stride(3),  # 状态维度步幅
            out.stride(0),  # 输出批次步幅
            out.stride(1),  # 输出分块步幅
            out.stride(2),  # 输出头步幅
            out.stride(3),  # 输出维度步幅
            final_states.stride(0),  # 最终状态批次步幅
            final_states.stride(1),  # 最终状态头步幅
            final_states.stride(2),  # 最终状态维度步幅
            dA_cumsum.stride(0),  # dA累积和批次步幅
            dA_cumsum.stride(2),  # dA累积和分块步幅
            dA_cumsum.stride(1),  # dA累积和头步幅
            dA_cumsum.stride(3),  # dA累积和csize步幅
            *(  # 初始状态步幅
                (
                    initial_states.stride(0),  # 初始状态批次步幅
                    initial_states.stride(1),  # 初始状态头步幅
                    initial_states.stride(2),  # 初始状态维度步幅
                )
                if initial_states is not None  # 如果初始状态存在
                else (0, 0, 0)  # 否则步幅全为0
            ),
            *(  # 序列索引步幅
                (seq_idx.stride(0), seq_idx.stride(1))  # seq_idx各维度步幅
                if seq_idx is not None  # 如果seq_idx存在
                else (0, 0)  # 否则步幅为0
            ),
            HAS_INITSTATES=initial_states is not None,  # 是否有初始状态
            HAS_SEQ_IDX=seq_idx is not None,  # 是否有序列索引
            IS_CONT_BATCHED=is_cont_batching,  # 是否连续批处理
        )
    return out, final_states  # 返回传递后的状态和最终状态
