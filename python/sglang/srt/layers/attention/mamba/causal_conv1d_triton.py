# 因果一维卷积Triton核实现 - 提供基于Triton的因果一维卷积前向计算和更新核函数，
# 支持连续批处理、推测解码和Eagle树注意力掩码等功能
# SPDX-License-Identifier: Apache-2.0 # SPDX许可证标识符
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project # SPDX版权声明
# Copyright (c) 2024, Tri Dao. # 版权声明
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py # 改编自Dao-AILab的causal-conv1d项目
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py # 以及vLLM项目的实现

from typing import List, Optional, Union  # 导入类型提示工具 # 导入类型注解所需的类型

import torch  # 导入PyTorch深度学习框架 # 导入PyTorch
import triton  # 导入Triton GPU编程框架 # 导入Triton编译器
import triton.language as tl  # 导入Triton语言并简写为tl # 导入Triton语言API

PAD_SLOT_ID = -1  # 填充槽位ID，用于标识不需要处理的填充条目 # 填充槽位标识符，值为-1


@triton.jit()  # Triton JIT编译装饰器 # 使用Triton即时编译装饰器将函数编译为GPU核
def _causal_conv1d_fwd_kernel(  # continuous batching # 因果一维卷积前向核函数 - 连续批处理模式
    # Pointers to matrices # 矩阵指针
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences # 输入数据指针，形状为(dim, cu_seqlen)，包含实际序列和填充序列
    w_ptr,  # (dim, width) # 卷积权重指针，形状为(dim, width)
    bias_ptr,  # 偏置指针 # 偏置项指针
    initial_states_ptr,  # conv_states_ptr # 初始状态指针（卷积状态指针）
    cache_indices_ptr,  # conv_state_indices_ptr # 缓存索引指针（卷积状态索引指针）
    has_initial_states_ptr,  # 是否有初始状态的标志指针
    query_start_loc_ptr,  # 查询起始位置指针，用于变长序列索引
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr # 输出指针，形状为(dim, seqlen)，实际指向x_ptr
    # Matrix dimensions # 矩阵维度
    dim: tl.constexpr,  # 特征维度（通道数），编译时常量
    seqlen: tl.int32,  # cu_seqlen # 累积序列长度
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines # 缓存行数，用于支持vLLM更大的缓存行
    # Strides # 步长
    stride_x_seq: tl.constexpr,  # stride to get to next sequence, # x中跳转到下一个序列的步长
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value, # x中跳转到下一个特征值的步长
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index) # x中跳转到下一个token的步长（相同特征索引，相同序列索引）
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value # 权重中跳转到下一个维度轴值的步长
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value # 权重中跳转到下一个宽度轴值的步长
    stride_istate_seq: tl.constexpr,  # 初始状态中跳转到下一个序列的步长
    stride_istate_dim: tl.constexpr,  # 初始状态中跳转到下一个维度的步长
    stride_istate_token: tl.constexpr,  # 初始状态中跳转到下一个token的步长
    stride_o_seq: tl.constexpr,  # 输出中跳转到下一个序列的步长
    stride_o_dim: tl.constexpr,  # 输出中跳转到下一个维度的步长
    stride_o_token: tl.constexpr,  # 输出中跳转到下一个token的步长
    # others # 其他参数
    pad_slot_id: tl.constexpr,  # 填充槽位ID
    # Meta-parameters # 元参数
    HAS_BIAS: tl.constexpr,  # 是否有偏置的编译时常量标志
    KERNEL_WIDTH: tl.constexpr,  # 卷积核宽度的编译时常量
    SILU_ACTIVATION: tl.constexpr,  # 是否使用SiLU激活函数的编译时常量标志
    HAS_INITIAL_STATES: tl.constexpr,  # 是否有初始状态的编译时常量标志
    HAS_CACHE: tl.constexpr,  # 是否有缓存的编译时常量标志
    IS_CONTINUOUS_BATCHING: tl.constexpr,  # 是否为连续批处理模式的编译时常量标志
    USE_PAD_SLOT: tl.constexpr,  # 是否使用填充槽位的编译时常量标志
    NP2_STATELEN: tl.constexpr,  # 状态长度的2的幂次值
    BLOCK_M: tl.constexpr,  # token维度的块大小
    BLOCK_N: tl.constexpr,  # 特征维度的块大小
):
    conv_states_ptr = initial_states_ptr  # 卷积状态指针赋值为初始状态指针 # 将初始状态指针赋值给卷积状态指针
    conv_state_indices_ptr = cache_indices_ptr  # 卷积状态索引指针赋值为缓存索引指针 # 将缓存索引指针赋值给卷积状态索引指针
    stride_conv_state_seq = stride_istate_seq  # 卷积状态序列步长赋值为初始状态序列步长 # 复制初始状态的序列步长
    stride_conv_state_dim = stride_istate_dim  # 卷积状态维度步长赋值为初始状态维度步长 # 复制初始状态的维度步长
    stride_conv_state_tok = stride_istate_token  # 卷积状态token步长赋值为初始状态token步长 # 复制初始状态的token步长
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value # 状态长度等于卷积核宽度减1 # 如果不等于此值可通过参数传递

    # one program handles one chunk in a single sequence # 每个程序处理单个序列中的一个块
    # rather than mixing sequences - to make updating initial_states across sequences efficiently # 而非混合序列 - 以便高效地跨序列更新初始状态

    # single-sequence id # 单序列ID
    idx_seq = tl.program_id(0)  # 获取序列维度的程序ID # 获取当前程序在第0维的索引
    chunk_offset = tl.program_id(1)  # 获取块偏移的程序ID # 获取当前程序在第1维的索引

    # BLOCK_N elements along the feature-dimension (channel) # 特征维度（通道）上的BLOCK_N个元素
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)  # 计算特征索引 # 计算当前块处理的特征索引范围

    if idx_seq == pad_slot_id:  # 如果序列索引等于填充槽位ID # 检查是否为填充槽位
        return  # 直接返回，不处理填充序列 # 跳过填充槽位的处理

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)  # 加载序列起始索引 # 从查询起始位置数组中加载当前序列的起始位置
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)  # 加载序列结束索引 # 从查询起始位置数组中加载当前序列的结束位置
    # find the actual sequence length # 查找实际序列长度
    seqlen = sequence_end_index - sequence_start_index  # 计算实际序列长度 # 用结束索引减去起始索引得到序列长度

    token_offset = BLOCK_M * chunk_offset  # 计算token偏移量 # 当前块在序列中的token起始偏移
    segment_len = min(BLOCK_M, seqlen - token_offset)  # 计算段的实际长度 # 取块大小和剩余长度的较小值

    if segment_len <= 0:  # 如果段长度小于等于0 # 检查是否还有token需要处理
        return  # 直接返回 # 没有需要处理的token则退出

    # base of the sequence # 序列的基础地址
    x_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,] # 计算输入数据的基地址 # 根据序列起始位置和特征索引计算x的基地址

    if IS_CONTINUOUS_BATCHING:  # 如果是连续批处理模式 # 检查是否为连续批处理
        # cache_idx # 缓存索引
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)  # 加载卷积状态批次坐标 # 从索引数组中获取对应的缓存行索引
    else:
        # cache_idx # 缓存索引
        conv_state_batch_coord = idx_seq  # 卷积状态批次坐标等于序列索引 # 非连续批处理时直接使用序列索引
    if USE_PAD_SLOT:  # noqa # 如果使用填充槽位 # 检查是否需要处理填充槽位
        if conv_state_batch_coord == pad_slot_id:  # 如果批次坐标等于填充槽位ID # 检查缓存索引是否为填充槽位
            # not processing as this is not the actual sequence # 不处理，因为这不是实际序列
            return  # 直接返回 # 跳过非实际序列的处理
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,] # 计算卷积状态的基地址 # 根据批次坐标和特征索引计算卷积状态的基地址

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,] # 计算权重基地址 # 根据特征索引计算权重的基地址

    # Does 2 things: # 执行两件事：
    # 1. READ prior-block init-state data - [done by every Triton programs] # 1. 读取前一块的初始状态数据 - [每个Triton程序都要做]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0] # 2. 用新数据更新卷积状态 [仅由处理chunk_offset=0的Triton程序执行]
    if chunk_offset == 0:  # 如果块偏移为0（第一个块） # 处理第一个块
        # read from conv_states # 从卷积状态中读取
        load_init_state = False  # 初始化加载初始状态标志为False # 默认不加载初始状态
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES # 如果有初始状态（新的HAS_INITIAL_STATES标志） # 检查是否启用了初始状态
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)  # 加载是否有初始状态的标志 # 从标志数组中读取当前序列是否有初始状态
        if load_init_state:  # 如果需要加载初始状态 # 有初始状态时的处理逻辑
            # load from conv_states # 从卷积状态中加载
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok  # 计算前一个token的地址 # 指向卷积状态中最后一个位置的地址
            mask_w = idx_feats < dim  # 特征掩码，确保索引在维度范围内 # 生成特征维度的有效掩码
            if KERNEL_WIDTH == 2:  # 卷积核宽度为2的情况 # 处理卷积核宽度为2
                conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针 # 指向最后状态的指针
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第0列数据 # 加载最近的先前token
            if KERNEL_WIDTH == 3:  # 卷积核宽度为3的情况 # 处理卷积核宽度为3
                conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针 # 指向最后状态的指针
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第1列数据 # 加载最近的先前token
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N] # 前移一个token的指针
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第0列数据 # 加载更早的先前token
            if KERNEL_WIDTH == 4:  # 卷积核宽度为4的情况 # 处理卷积核宽度为4
                conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针 # 指向最后状态的指针
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第2列数据 # 加载最近的先前token
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N] # 前移一个token的指针
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第1列数据 # 加载次近的先前token
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N] # 前移两个token的指针
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第0列数据 # 加载最早的先前token
            if KERNEL_WIDTH == 5:  # 卷积核宽度为5的情况 # 处理卷积核宽度为5
                conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针 # 指向最后状态的指针
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第3列数据 # 加载最近的先前token
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N] # 前移一个token的指针
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第2列数据
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N] # 前移两个token的指针
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第1列数据
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N] # 前移三个token的指针
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第0列数据
        else:
            # prior-tokens are zeros # 先前token为零
            if KERNEL_WIDTH >= 2:  # STRATEGY1 # 卷积核宽度>=2的情况 # 策略1：无初始状态时设为零
                # first chunk and does not have prior-token, so just set to 0 # 第一个块且没有先前token，所以设为0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)  # 初始化第0列为零 # 创建全零的第0列
            if KERNEL_WIDTH >= 3:  # STRATEGY1 # 卷积核宽度>=3的情况
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)  # 初始化第1列为零 # 创建全零的第1列
            if KERNEL_WIDTH >= 4:  # STRATEGY1 # 卷积核宽度>=4的情况
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)  # 初始化第2列为零 # 创建全零的第2列
            if KERNEL_WIDTH >= 5:  # STRATEGY1 # 卷积核宽度>=5的情况
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)  # 初始化第3列为零 # 创建全零的第3列

        # STEP 2: # 第二步：
        # here prepare data for updating conv_state # 准备更新卷积状态的数据
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache) # 小缓存模式（仅将部分'x'数据移入卷积状态缓存）
            # just read from 'x' # 仅从'x'中读取
            # copy 'x' data to conv_state # 将'x'数据复制到卷积状态
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len) # 仅加载'x'数据（如果seqlen < state_len，在'x'之前设0）
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M] # 计算最后几个token的索引 # 计算需要从x中复制的token索引
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,] # 计算x数据的指针 # 构建二维指针用于加载x数据
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index # 计算加载掩码 # 生成复合掩码确保索引有效
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)  # 加载x数据 # 从x中加载数据
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)  # 加载新的卷积状态 # 再次加载作为新的卷积状态值
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M] # 卷积状态token索引 # 生成卷积状态的token索引范围
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )  # 计算卷积状态目标指针 # 计算存储新卷积状态的目标地址

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]  # 计算存储掩码 # 确保只存储有效范围内的数据
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler # 调试屏障 # 注意：由于Triton编译器的bug需要使用此屏障
            tl.store(conv_states_ptrs_target, new_conv_state, mask)  # 存储新的卷积状态 # 将新数据写入卷积状态缓存

        else:
            if load_init_state:  # 如果加载了初始状态 # 有初始状态时的更新逻辑
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x' # 通过左移更新卷积状态，即从卷积状态取最后几列加上'x'中的列
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M] # 卷积状态token索引 # 生成卷积状态的token索引范围

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N] # 计算卷积状态源指针 # 指向左移后保留的旧数据位置
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # 计算加载掩码 # 确保索引在有效范围内
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)  # 加载卷积状态 # 从旧位置加载数据

                VAL = state_len - seqlen  # 计算阈值 # 卷积状态中需要从旧状态保留的部分长度

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N] # 计算x数据指针 # 指向输入数据中需要写入的位置

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index # 计算x加载掩码 # 确保从x中加载有效数据
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)  # 加载x数据 # 从输入数据中加载对应位置的值

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load # 需要此屏障，因为tl.where的bug在数据来自另一个tl.load时不会强制执行此操作
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this # 合并旧状态和新数据 # tl.where的bug需要在此前加屏障
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N] # 计算卷积状态目标指针 # 指向要写入的目标位置
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]  # 计算存储掩码 # 确保只写入有效范围内的数据
                tl.store(conv_states_ptrs_target, new_conv_state, mask)  # 存储新的卷积状态 # 将合并后的数据写入卷积状态缓存
            else:  # load_init_state == False # 未加载初始状态的情况
                # update conv_state by shifting left, BUT # 通过左移更新卷积状态，但是
                # set cols prior to 'x' as zeros + cols from 'x' # 将'x'之前的列设为零 + 'x'中的列
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M] # 卷积状态token索引 # 生成卷积状态的token索引范围

                VAL = state_len - seqlen  # 计算阈值 # 卷积状态中需要补零的部分长度

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N] # 计算x数据指针 # 指向输入数据中需要写入的位置

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index # 计算x加载掩码 # 确保从x中加载有效数据
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)  # 加载新的卷积状态 # 从x中加载数据，其余位置自动为0

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N] # 计算卷积状态目标指针 # 指向要写入的目标位置
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]  # 计算存储掩码 # 确保只写入有效范围内的数据
                tl.store(conv_states_ptrs_target, new_conv_state, mask)  # 存储新的卷积状态 # 将新数据写入卷积状态缓存

    else:  # chunk_offset > 0 # 块偏移大于0（非第一个块）
        # read prior-token data from `x` # 从`x`中读取先前token数据
        load_init_state = True  # 需要加载初始状态 # 非首块总是需要加载先前数据
        prior_tokens = x_base + (token_offset - 1) * stride_x_token  # 计算先前token的地址 # 指向当前块前一个token的位置
        mask_w = idx_feats < dim  # 特征掩码 # 确保特征索引在有效范围内
        if KERNEL_WIDTH == 2:  # 卷积核宽度为2的情况
            conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第0列数据 # 使用缓存加速加载
        if KERNEL_WIDTH == 3:  # 卷积核宽度为3的情况
            conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第1列数据
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N] # 前移一个token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第0列数据
        if KERNEL_WIDTH == 4:  # 卷积核宽度为4的情况
            conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第2列数据
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N] # 前移一个token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第1列数据
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N] # 前移两个token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第0列数据
        if KERNEL_WIDTH == 5:  # 卷积核宽度为5的情况
            # ruff: noqa: F841 # ruff忽略未使用变量警告
            conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第3列数据
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N] # 前移一个token
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第2列数据
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N] # 前移两个token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第1列数据
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N] # 前移三个token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")  # 加载第0列数据

    if HAS_BIAS:  # 如果有偏置 # 处理偏置项
        bias = bias_ptr + idx_feats  # 计算偏置指针 # 指向当前特征对应的偏置
        mask_bias = idx_feats < dim  # 偏置掩码 # 确保特征索引在有效范围内
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N] # 预加载偏置到累加器 # 将偏置加载为float32类型
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)  # 无偏置时初始化累加器为零 # 创建全零的累加器

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk # 块起始地址 # 计算当前块的起始位置

    # PRE-LOAD WEIGHTS # 预加载权重
    mask_w = idx_feats < dim  # 权重掩码 # 确保特征索引在有效范围内
    if KERNEL_WIDTH >= 2:  # 卷积核宽度>=2时预加载权重
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor # 第0列权重指针
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第0列权重
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor # 第1列权重指针
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第1列权重
    if KERNEL_WIDTH >= 3:  # 卷积核宽度>=3时额外预加载权重
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor # 第2列权重指针
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第2列权重
    if KERNEL_WIDTH >= 4:  # 卷积核宽度>=4时额外预加载权重
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor # 第3列权重指针
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第3列权重
    mask_x_1d = idx_feats < dim  # 一维x掩码 # 确保特征索引在有效范围内
    for idx_token in range(segment_len):  # 遍历段中的每个token # 对当前块中的每个token进行卷积计算
        acc = acc_preload  # 初始化累加器 # 将偏置作为累加器的初始值

        matrix_w = w_col0  # 初始化权重矩阵为第0列 # 当前使用的权重列
        matrix_x = col0  # 初始化输入矩阵为第0列 # 当前使用的输入列
        for j in tl.static_range(KERNEL_WIDTH):  # 静态遍历卷积核宽度 # 遍历卷积核的每一列

            if KERNEL_WIDTH == 2:  # 卷积核宽度为2的情况
                if j == 1:  # KERNEL_WIDTH-1: # 当j=1时
                    matrix_w = w_col1  # 使用第1列权重
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N] # 计算当前token的x指针
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载当前token的输入
            elif KERNEL_WIDTH == 3:  # 卷积核宽度为3的情况
                if j == 1:  # 当j=1时
                    matrix_w = w_col1  # 使用第1列权重
                    matrix_x = col1  # 使用第1列输入
                elif j == 2:  # 当j=2时
                    matrix_w = w_col2  # 使用第2列权重
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N] # 计算当前token的x指针
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载当前token的输入
            elif KERNEL_WIDTH == 4:  # 卷积核宽度为4的情况
                if j == 1:  # 当j=1时
                    matrix_w = w_col1  # 使用第1列权重
                    matrix_x = col1  # 使用第1列输入
                elif j == 2:  # 当j=2时
                    matrix_w = w_col2  # 使用第2列权重
                    matrix_x = col2  # 使用第2列输入
                elif j == 3:  # 当j=3时
                    matrix_w = w_col3  # 使用第3列权重
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N] # 计算当前token的x指针
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载当前token的输入

            acc += matrix_x * matrix_w  # [BLOCK_N] # 累加乘积 # 将输入和权重的逐元素乘积累加到累加器

        if KERNEL_WIDTH == 2:  # 更新滑动窗口 - 卷积核宽度为2
            col0 = matrix_x  # 滑动窗口：第0列更新为当前token
        elif KERNEL_WIDTH == 3:  # 更新滑动窗口 - 卷积核宽度为3
            col0 = col1  # 第0列更新为第1列
            col1 = matrix_x  # 第1列更新为当前token
        elif KERNEL_WIDTH == 4:  # 更新滑动窗口 - 卷积核宽度为4
            col0 = col1  # 第0列更新为第1列
            col1 = col2  # 第1列更新为第2列
            col2 = matrix_x  # 第2列更新为当前token

        if SILU_ACTIVATION:  # 如果使用SiLU激活函数 # 应用SiLU激活
            acc = acc / (1 + tl.exp(-acc))  # 计算SiLU激活函数 # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index # 计算一维存储掩码 # 确保token和特征索引都在有效范围内
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )  # 计算输出指针 # 根据当前token位置计算输出地址

        tl.store(o_ptrs, acc, mask=mask_1d)  # 存储输出结果 # 将计算结果写入输出张量


def causal_conv1d_fn(  # 因果一维卷积前向函数 - 支持变长序列和连续批处理
    x: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 卷积权重
    bias: Union[torch.Tensor, None],  # 偏置项
    conv_states: torch.Tensor,  # 卷积状态缓存
    query_start_loc: torch.Tensor,  # 查询起始位置
    seq_lens_cpu: List[int],  # CPU上的序列长度列表
    cache_indices: Optional[torch.Tensor] = None,  # 缓存索引，可选
    has_initial_state: Optional[torch.Tensor] = None,  # 是否有初始状态，可选
    activation: Optional[str] = "silu",  # 激活函数类型，默认为silu
    pad_slot_id: int = PAD_SLOT_ID,  # 填充槽位ID
    validate_data=False,  # 是否验证数据
    **kwargs,  # 其他关键字参数
):
    """support varlen + continuous batching when x is 2D tensor # 支持变长序列+连续批处理（当x为2D张量时）

    x: (dim,cu_seq_len) # 输入张量，形状为(dim, cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch # cu_seq_len为批次中所有序列的总token数
        sequences are concatenated from left to right for varlen # 序列从左到右连接用于变长处理
    weight: (dim, width) # 权重张量，形状为(dim, width)
    conv_states: (...,dim,width - 1) itype # 卷积状态张量
        updated inplace if provided # 如果提供则原地更新
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence # [使用`cache_indices`获取该序列的卷积状态缓存索引

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True # conv_state[cache_indices[i]]用于第i个序列 - 当has_initial_state[i]=True时用作初始状态
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x' # 之后conv_state[cache_indices[i]]需要左移并用'x'中的值更新
        ] # ]
    query_start_loc: (batch + 1) int32 # 查询起始位置，形状为(batch+1)，int32类型
        The cumulative sequence lengths of the sequences in # 批次中序列的累积序列长度
        the batch, used to index into sequence. prepended by 0. # 用于索引序列，前面加0
        if # 如果
        x = [5, 1, 1, 1] <- continuous batching (batch=4) # x = [5, 1, 1, 1] <- 连续批处理（batch=4）
        then # 那么
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is # query_start_loc = [0, 5, 6, 7, 8] <- 下一个序列的起始索引；最后一个值是
           the ending index of the last sequence # 最后一个序列的结束索引
        [length(query_start_loc)-1 == batch] # [query_start_loc的长度-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]), # 例如：query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17) # x的形状为(dim,17)
    seq_lens_cpu: (batch) int32 # CPU上的序列长度，形状为(batch)，int32类型
        The sequence lengths of the sequences in the batch # 批次中各序列的序列长度
    cache_indices: (batch)  int32 # 缓存索引，形状为(batch)，int32类型
        indicates the corresponding state index, # 指示对应的状态索引，
        like so: conv_state = conv_states[cache_indices[batch_id]] # 如：conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool # 是否有初始状态，形状为(batch)，bool类型
        indicates whether should the kernel take the current state as initial # 指示核函数是否应将当前状态作为初始
        state for the calculations # 状态用于计算
        [single boolean for each sequence in the batch: True or False] # [批次中每个序列的布尔值：True或False]
    bias: (dim,) # 偏置项，形状为(dim,)
    activation: either None or "silu" or "swish" or True # 激活函数：None、"silu"、"swish"或True
    pad_slot_id: int # 填充槽位ID，整数
        if cache_indices is passed, lets the kernel identify padded # 如果传入了cache_indices，让核函数识别填充
        entries that will not be processed, # 的不需要处理的条目，
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id] # 例如：cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at # 在这种情况下，核函数不会处理
        indices 0 and 3 # 索引0和3处的条目

    out: same shape as `x` # 输出：与`x`形状相同
    """
    if isinstance(activation, bool) and activation:  # 如果激活是布尔值且为True # 处理布尔类型的激活参数
        activation = "silu"  # 将激活设为"silu" # 布尔True等价于silu

    out = torch.empty_like(x)  # 创建与x形状相同的空输出张量 # 分配输出张量

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)  # 检查是否为通道最后布局 # 判断x是否为channel-last内存布局
    dim, cu_seqlen = x.shape  # 获取维度和累积序列长度 # 解包x的形状
    _, width = weight.shape  # 获取卷积核宽度 # 解包权重的形状
    state_len = width - 1  # 状态长度等于卷积核宽度减1 # 卷积状态长度
    np2_statelen = triton.next_power_of_2(state_len)  # 计算状态长度的2的幂次值 # 获取大于等于state_len的最小2的幂

    stride_x_seq = 0  # x序列步长设为0 # 2D输入无序列维度
    stride_x_dim = x.stride(0)  # 获取x维度步长 # x在特征维度上的步长
    stride_x_token = x.stride(1)  # 获取x token步长 # x在token维度上的步长
    stride_w_dim = weight.stride(0)  # 获取权重维度步长 # 权重在特征维度上的步长
    stride_w_width = weight.stride(1)  # 获取权重宽度步长 # 权重在宽度维度上的步长
    stride_istate_seq = 0  # 初始状态序列步长 # 暂设为0
    stride_istate_dim = 0  # 初始状态维度步长 # 暂设为0
    stride_istate_token = 0  # 初始状态token步长 # 暂设为0
    num_cache_lines = 0  # 缓存行数初始化为0 # 暂设为0
    if conv_states is not None:  # 如果提供了卷积状态 # 处理卷积状态缓存
        # extensions to support vLLM: # 为支持vLLM的扩展：
        # 1. conv_states is used to replaced initial_states # 1. conv_states用于替代initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size # 2. conv_states作为缓存，缓存行数可以大于批次大小
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx] # 3. 通过cache_indices[idx]指定从序列x[idx]到缓存行的映射
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id # 4. 如果cache_indices[idx] == pad_slot_id则跳过计算
        num_cache_lines = conv_states.size(0)  # 获取缓存行数 # 读取卷积状态的第一维大小
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )  # 断言缓存形状有效 # 验证卷积状态缓存的形状
        stride_istate_seq = conv_states.stride(0)  # 获取初始状态序列步长 # 读取卷积状态在序列维度上的步长
        stride_istate_dim = conv_states.stride(1)  # 获取初始状态维度步长 # 读取卷积状态在特征维度上的步长
        stride_istate_token = conv_states.stride(2)  # 获取初始状态token步长 # 读取卷积状态在token维度上的步长
        # assert stride_istate_dim == 1 # 断言维度步长为1（已注释）
    if out.dim() == 2:  # 如果输出为2D # 处理2D输出的步长
        stride_o_seq = 0  # 输出序列步长为0 # 2D输出无序列维度
        stride_o_dim = out.stride(0)  # 获取输出维度步长
        stride_o_token = out.stride(1)  # 获取输出token步长
    else:  # 3D输出的情况
        stride_o_seq = out.stride(0)  # 获取输出序列步长
        stride_o_dim = out.stride(1)  # 获取输出维度步长
        stride_o_token = out.stride(2)  # 获取输出token步长

    if validate_data:  # 如果需要验证数据 # 数据验证逻辑
        assert x.dim() == 2  # 断言x为2D
        assert query_start_loc is not None  # 断言query_start_loc不为None
        assert query_start_loc.dim() == 1  # 断言query_start_loc为1D
        assert x.stride(0) == 1 or x.stride(1) == 1  # 断言x是连续的
        padded_batch = query_start_loc.size(0) - 1  # 计算填充批次大小
        if bias is not None:  # 如果有偏置
            assert bias.dim() == 1  # 断言偏置为1D
            assert dim == bias.size(0)  # 断言偏置维度匹配
        if cache_indices is not None:  # 如果有缓存索引
            assert cache_indices.dim() == 1  # 断言缓存索引为1D
            assert padded_batch == cache_indices.size(0)  # 断言缓存索引大小匹配
        if has_initial_state is not None:  # 如果有初始状态标志
            assert has_initial_state.size() == (padded_batch,)  # 断言初始状态大小匹配
            assert (
                conv_states is not None
            ), "ERROR: `has_initial_state` is used, which needs also `conv_states`"  # 断言卷积状态不为None
        assert weight.stride(1) == 1  # 断言权重宽度步长为1
        assert (dim, width) == weight.shape  # 断言权重形状匹配
        assert is_channel_last, "Need to run in channel-last layout"  # 断言为通道最后布局

    def grid(META):  # 定义网格函数 # 用于配置Triton核的启动网格
        max_seq_len = max(seq_lens_cpu)  # 获取最大序列长度 # 找到批次中最长的序列
        return (
            len(seq_lens_cpu),  # batch_size # 批次大小（序列数）
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],  # 块数 # 沿序列维度的块数
            triton.cdiv(dim, META["BLOCK_N"]),  # 特征维度块数 # 沿特征维度的块数
        )

    _causal_conv1d_fwd_kernel[grid](  # 启动因果一维卷积前向核 # 调用Triton核函数
        # Pointers to matrices # 矩阵指针
        x,  # 输入数据
        weight,  # 卷积权重
        bias,  # 偏置
        conv_states,  # 卷积状态
        cache_indices,  # 缓存索引
        has_initial_state,  # 初始状态标志
        query_start_loc,  # 查询起始位置
        out,  # 输出
        # Matrix dimensions # 矩阵维度
        dim,  # 特征维度
        cu_seqlen,  # 累积序列长度
        num_cache_lines,  # 缓存行数
        # stride # 步长
        stride_x_seq,  # x序列步长
        stride_x_dim,  # x维度步长
        stride_x_token,  # x token步长
        stride_w_dim,  # 权重维度步长
        stride_w_width,  # 权重宽度步长
        stride_istate_seq,  # 初始状态序列步长
        stride_istate_dim,  # 初始状态维度步长
        stride_istate_token,  # 初始状态token步长
        stride_o_seq,  # 输出序列步长
        stride_o_dim,  # 输出维度步长
        stride_o_token,  # 输出token步长
        # others # 其他参数
        pad_slot_id,  # 填充槽位ID
        # META # 元参数
        HAS_BIAS=bias is not None,  # 是否有偏置
        KERNEL_WIDTH=width,  # 卷积核宽度
        SILU_ACTIVATION=activation in ["silu", "swish"],  # 是否使用SiLU激活
        HAS_INITIAL_STATES=has_initial_state is not None,  # 是否有初始状态
        HAS_CACHE=conv_states is not None,  # 是否有缓存
        IS_CONTINUOUS_BATCHING=cache_indices is not None,  # 是否为连续批处理
        USE_PAD_SLOT=pad_slot_id is not None,  # 是否使用填充槽位
        NP2_STATELEN=np2_statelen,  # 状态长度的2的幂次值
        # launch_cooperative_grid=True # 启动协作网格（已注释）
        BLOCK_M=8,  # token维度块大小
        BLOCK_N=256,  # 特征维度块大小
        num_stages=2,  # 流水线阶段数
    )
    return out  # 返回输出张量


# HAS_EAGLE_TREE_CUSTOM_ATTN_MASK is added to support eagle tree attention mask # 添加HAS_EAGLE_TREE_CUSTOM_ATTN_MASK以支持eagle树注意力掩码
# retrieve_next_token_ptr: [N, NP2_T], retrieve_next_sibling_ptr: [N, NP2_T] # 检索下一个token指针和下一个兄弟token指针
# e.g. for a sequence of length 4, the eagle tree attention structure is: # 例如，对于长度为4的序列，eagle树注意力结构为：
# retrieve_next_token=[1, 3, -1, -1] -> retrieve_next_token[i]: the 1st child token of token i # retrieve_next_token[i]：token i的第一个子token
# retrieve_next_sibling=[-1, 2, -1, -1] -> retrieve_next_sibling[i]: the 1st tree sibling token of token i # retrieve_next_sibling[i]：token i的第一个树兄弟token
# retrieve_parent_token=[n/a, 0, 0, 1] -> retrieve_parent_token[i]: the parent token of token i # retrieve_parent_token[i]：token i的父token
# Tree: # 树结构：
#    0
#   / \
#  1   2
# /
# 3
# When calculating token 3's convolution, it should conv to token 1 (parent) and token 0 (grand-parent) # 计算token 3的卷积时，应该对token 1（父节点）和token 0（祖父节点）进行卷积
# When calculating token 2's convolution, it should conv to token 0 (parent) # 计算token 2的卷积时，应该对token 0（父节点）进行卷积
# This kernel is a fused kernel which will also produce retrieve_parent_token based on retrieve_next_token & retrieve_next_sibling # 此核为融合核，还将基于retrieve_next_token和retrieve_next_sibling生成retrieve_parent_token
@triton.jit()  # Triton JIT编译装饰器 # 使用Triton即时编译装饰器
def _causal_conv1d_update_kernel(  # 因果一维卷积更新核函数 - 用于解码阶段的单token或多token更新
    # Pointers to matrices # 矩阵指针
    x_ptr,  # (batch, dim, seqlen) # 输入数据指针，形状为(batch, dim, seqlen)
    w_ptr,  # (dim, width) # 卷积权重指针，形状为(dim, width)
    bias_ptr,  # 偏置指针
    conv_state_ptr,  # 卷积状态指针
    cache_seqlens_ptr,  # circular buffer # 循环缓冲区的缓存序列长度指针
    conv_state_indices_ptr,  # 卷积状态索引指针
    num_accept_tokens_ptr,  # 接受token数量指针（用于推测解码）
    intermediate_conv_window_ptr,  # 中间卷积窗口指针（用于推测解码回退）
    intermediate_state_indices_ptr,  # 中间状态索引指针
    retrieve_next_token_ptr,  # 检索下一个token指针（Eagle树）
    retrieve_next_sibling_ptr,  # 检索下一个兄弟token指针（Eagle树）
    retrieve_parent_token_ptr,  # 检索父token指针（Eagle树）
    o_ptr,  # (batch, dim, seqlen) # 输出指针，形状为(batch, dim, seqlen)
    # Matrix dimensions # 矩阵维度
    batch: int,  # 批次大小
    dim: tl.constexpr,  # 特征维度（编译时常量）
    seqlen: tl.constexpr,  # 序列长度（编译时常量）
    state_len: tl.constexpr,  # 状态长度（编译时常量）
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines # 缓存行数（编译时常量），用于支持vLLM更大的缓存行
    # Strides # 步长
    stride_x_seq: tl.constexpr,  # x序列步长
    stride_x_dim: tl.constexpr,  # x维度步长
    stride_x_token: tl.constexpr,  # x token步长
    stride_w_dim: tl.constexpr,  # 权重维度步长
    stride_w_width: tl.constexpr,  # 权重宽度步长
    stride_conv_state_seq: tl.constexpr,  # 卷积状态序列步长
    stride_conv_state_dim: tl.constexpr,  # 卷积状态维度步长
    stride_conv_state_tok: tl.constexpr,  # 卷积状态token步长
    stride_state_indices: tl.constexpr,  # 状态索引步长
    stride_inter_seq: tl.constexpr,  # 中间缓冲区序列步长
    stride_inter_step: tl.constexpr,  # 中间缓冲区步长
    stride_inter_dim: tl.constexpr,  # 中间缓冲区维度步长
    stride_inter_win: tl.constexpr,  # 中间缓冲区窗口步长
    stride_intermediate_state_indices: tl.constexpr,  # 中间状态索引步长
    stride_retrieve_next_token_seq: tl.constexpr,  # 检索下一个token序列步长
    stride_retrieve_next_token_token: tl.constexpr,  # 检索下一个token token步长
    stride_retrieve_next_sibling_seq: tl.constexpr,  # 检索下一个兄弟token序列步长
    stride_retrieve_next_sibling_token: tl.constexpr,  # 检索下一个兄弟token token步长
    stride_retrieve_parent_token_seq: tl.constexpr,  # 检索父token序列步长
    stride_retrieve_parent_token_token: tl.constexpr,  # 检索父token token步长
    stride_o_seq: tl.constexpr,  # 输出序列步长
    stride_o_dim: tl.constexpr,  # 输出维度步长
    stride_o_token: tl.constexpr,  # 输出token步长
    # others # 其他参数
    pad_slot_id: tl.constexpr,  # 填充槽位ID
    # Meta-parameters # 元参数
    HAS_BIAS: tl.constexpr,  # 是否有偏置
    KERNEL_WIDTH: tl.constexpr,  # 卷积核宽度
    SILU_ACTIVATION: tl.constexpr,  # 是否使用SiLU激活
    IS_CONTINUOUS_BATCHING: tl.constexpr,  # 是否为连续批处理
    IS_SPEC_DECODING: tl.constexpr,  # 是否为推测解码
    NP2_STATELEN: tl.constexpr,  # 状态长度的2的幂次值
    NP2_SEQLEN: tl.constexpr,  # 序列长度的2的幂次值
    USE_PAD_SLOT: tl.constexpr,  # 是否使用填充槽位
    BLOCK_N: tl.constexpr,  # 特征维度块大小
    SAVE_INTERMEDIATE: tl.constexpr,  # 是否保存中间状态
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,  # 是否有Eagle树自定义注意力掩码
):
    # ruff: noqa: E501 # ruff忽略行过长警告
    idx_seq = tl.program_id(0)  # 获取序列维度的程序ID # 当前处理的序列索引
    if idx_seq >= batch:  # 如果序列索引超出批次大小 # 边界检查
        return  # 直接返回 # 超出范围则退出

    # [BLOCK_N,] elements along the feature-dimension (channel) # 特征维度（通道）上的BLOCK_N个元素
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)  # 计算特征索引 # 当前块处理的特征索引范围

    if IS_CONTINUOUS_BATCHING:  # 如果是连续批处理模式 # 连续批处理时的索引映射
        # mask = idx_seq < batch # 掩码（已注释）
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)  # 加载卷积状态批次坐标 # 从索引数组获取对应的缓存行
        if SAVE_INTERMEDIATE:  # 如果需要保存中间状态 # 推测解码时保存中间卷积窗口
            intermediate_state_batch_coord = tl.load(
                intermediate_state_indices_ptr
                + idx_seq * stride_intermediate_state_indices
            ).to(tl.int64)  # 加载中间状态批次坐标 # 获取中间状态缓冲区的索引
    else:
        conv_state_batch_coord = idx_seq  # 非连续批处理时直接使用序列索引 # 卷积状态批次坐标等于序列索引
    if USE_PAD_SLOT:  # noqa # 如果使用填充槽位 # 检查填充槽位
        if conv_state_batch_coord == pad_slot_id:  # 如果批次坐标等于填充槽位ID # 跳过填充槽位
            # not processing as this is not the actual sequence # 不处理，因为这不是实际序列
            return  # 直接返回 # 跳过非实际序列

    if IS_SPEC_DECODING:  # 如果是推测解码模式 # 推测解码的卷积状态偏移处理
        # The rolling of conv state: # 卷积状态的滚动：
        #
        # Before forward, the conv_state is: # 前向传播前，卷积状态为：
        # [history1, history2, ..., historyM]. # [历史1, 历史2, ..., 历史M]。
        #
        # After forward, the conv_state becomes: # 前向传播后，卷积状态变为：
        # [history2, ..., historyM, draft1, draft2, ..., draftN]. # [历史2, ..., 历史M, 草稿1, 草稿2, ..., 草稿N]。
        #
        # After acceptance, it becomes: # 接受后，变为：
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1] # - 接受1个token：[历史2, ..., 历史M, 草稿1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2] # - 接受2个token：[历史3, ..., 历史M, 草稿1, 草稿2]
        # - and so on. # - 以此类推。
        conv_state_token_offset = tl.load(num_accept_tokens_ptr + idx_seq) - 1  # 加载卷积状态token偏移 # 根据接受的token数计算偏移
    else:
        conv_state_token_offset = 0  # 非推测解码时偏移为0 # 无偏移

    # STEP 1: READ init_state data # 第1步：读取初始状态数据
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # 计算卷积状态基地址 # 根据批次坐标和特征索引计算基地址
    mask_w = idx_feats < dim  # 特征掩码 # 确保特征索引在有效范围内

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok  # 计算先前token的地址 # 带偏移地指向先前token位置
    if KERNEL_WIDTH >= 2:  # 卷积核宽度>=2时加载第0列
        conv_states_ptrs = prior_tokens  # [BLOCK_N] # 卷积状态指针
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第0列数据 # 加载最近的先前token
    if KERNEL_WIDTH >= 3:  # 卷积核宽度>=3时加载第1列
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N] # 指针后移一位
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第1列数据
    if KERNEL_WIDTH >= 4:  # 卷积核宽度>=4时加载第2列
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N] # 指针后移两位
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第2列数据
    if KERNEL_WIDTH == 5:  # 卷积核宽度为5时加载第3列
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N] # 指针后移三位
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)  # 加载第3列数据

    # STEP 2: assume state_len > seqlen # 第2步：假设state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M] # 生成token索引范围

    # The conv_state updates works in a sliding window manner, # 卷积状态更新以滑动窗口方式工作，
    # at each forward pass, the tokens are shift by 1, so we # 每次前向传播时，token左移1位，所以我们
    # load since idx_tokens + 1. # 从idx_tokens + 1开始加载。
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N] # 计算卷积状态源指针 # 指向需要保留的旧数据位置
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )  # 计算加载掩码 # 确保索引在有效范围内
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)  # 加载卷积状态 # 从旧位置加载数据

    VAL = state_len - seqlen  # 计算阈值 # 卷积状态中需要从旧状态保留的部分长度
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N] # 计算x的基地址

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N] # 计算x数据指针 # 指向输入数据中需要写入的位置

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index # 计算x加载掩码 # 确保从x中加载有效数据
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)  # 加载x数据
    tl.debug_barrier()  # 调试屏障 # 确保加载完成

    new_conv_state = tl.where(mask, conv_state, loaded_x)  # 合并旧状态和新数据 # 根据掩码选择保留旧状态或使用新数据

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,] # 计算卷积状态基地址
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N] # 计算卷积状态目标指针
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]  # 计算存储掩码 # 确保只写入有效范围
    tl.store(conv_state_ptrs_target, new_conv_state, mask)  # 存储新的卷积状态 # 将更新后的数据写入卷积状态缓存

    # STEP 3: init accumulator # 第3步：初始化累加器
    if HAS_BIAS:  # 如果有偏置 # 处理偏置项
        bias = bias_ptr + idx_feats  # 计算偏置指针
        mask_bias = idx_feats < dim  # 偏置掩码
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N] # 预加载偏置到累加器 # 将偏置加载为float32
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)  # 无偏置时初始化累加器为零 # 创建全零累加器

    # STEP 4: # 第4步：
    # PRE-LOAD WEIGHTS # 预加载权重
    # first kernel column, configured for weights to handle BLOCK_N features in range # 第一列权重，配置为处理BLOCK_N范围内的特征
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:  # 如果有Eagle树自定义注意力掩码 # 处理Eagle树结构
        idx_tokens = tl.arange(0, NP2_SEQLEN)  # [BLOCK_M] # 生成token索引范围
        # Update parent mapping for all tokens at once using vectorized operations # 使用向量化操作一次性更新所有token的父映射
        mask_retrieve = idx_tokens < seqlen  # 检索掩码 # 只处理有效token
        retrieve_next_token_base = (
            retrieve_next_token_ptr
            + (idx_seq * stride_retrieve_next_token_seq)
            + idx_tokens * stride_retrieve_next_token_token
        )  # 计算检索下一个token的基地址
        retrieve_next_tokens = tl.load(retrieve_next_token_base, mask_retrieve)  # 加载下一个token索引
        retrieve_next_sibling_base = (
            retrieve_next_sibling_ptr
            + (idx_seq * stride_retrieve_next_sibling_seq)
            + idx_tokens * stride_retrieve_next_sibling_token
        )  # 计算检索下一个兄弟token的基地址
        retrieve_next_siblings = tl.load(retrieve_next_sibling_base, mask_retrieve)  # 加载下一个兄弟token索引
        parent_idx_tokens = tl.zeros((NP2_SEQLEN,), dtype=tl.int32)  # 初始化父token索引为零 # 创建全零的父索引数组

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,] # 计算权重基地址
    mask_w = idx_feats < dim  # 权重掩码 # 确保特征索引在有效范围内
    if KERNEL_WIDTH >= 2:  # 卷积核宽度>=2时预加载权重
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor # 第0列权重指针
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第0列权重
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor # 第1列权重指针
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第1列权重
    if KERNEL_WIDTH >= 3:  # 卷积核宽度>=3时额外预加载权重
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor # 第2列权重指针
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第2列权重
    if KERNEL_WIDTH >= 4:  # 卷积核宽度>=4时额外预加载权重
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor # 第3列权重指针
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)  # 加载第3列权重

    x_base_1d = x_base  # starting of chunk [BLOCK_N] # 块起始地址
    mask_x_1d = idx_feats < dim  # 一维x掩码 # 确保特征索引在有效范围内

    # STEP 5: compute each token # 第5步：计算每个token
    for idx_token in tl.static_range(seqlen):  # 静态遍历序列中的每个token # 对每个token进行卷积计算
        acc = acc_preload  # 初始化累加器 # 将偏置作为累加器的初始值

        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:  # Eagle树自定义注意力掩码的处理逻辑 # 按树结构计算卷积
            # set the parent index of the next token in the eagle tree # 设置eagle树中下一个token的父索引
            # next token's parent is the current token # 下一个token的父节点是当前token
            retrieve_next_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_tokens, 0)
            )  # 获取当前token的下一个子token索引 # 查找当前token的子节点
            if retrieve_next_token_idx != -1:  # pad slot id # 不是填充槽位时 # 有效子节点时
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_next_token_idx,
                    idx_token,
                    parent_idx_tokens,
                )  # 设置子token的父索引为当前token # 将当前token标记为子节点的父节点
            # next token's parent is the parent of the current token # 下一个token的父节点是当前token的父节点
            retrieve_sibling_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_siblings, 0)
            )  # 获取当前token的下一个兄弟token索引 # 查找当前token的兄弟节点
            if retrieve_sibling_token_idx != -1:  # pad slot id # 不是填充槽位时 # 有效兄弟节点时
                parent_idx_token = tl.sum(
                    tl.where(idx_tokens == idx_token, parent_idx_tokens, 0)
                )  # 获取当前token的父索引 # 获取当前token的父节点
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_sibling_token_idx,
                    parent_idx_token,
                    parent_idx_tokens,
                )  # 设置兄弟token的父索引与当前token相同 # 兄弟节点共享父节点
            # tl.device_print("am", parent_idx_tokens) # 设备打印（已注释）

            _idx_token = idx_token  # 临时token索引 # 用于沿树向上遍历
            x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N] # 计算当前token的x指针
            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载当前token的输入
            # convolution operation: itself * wcol[-1] + parent * wcol[-2] + grand-parent * wcol[-3] + ... # 卷积操作：自身*wcol[-1] + 父节点*wcol[-2] + 祖父节点*wcol[-3] + ...
            for j in tl.static_range(KERNEL_WIDTH):  # 静态遍历卷积核宽度 # 沿树向上遍历
                if KERNEL_WIDTH == 2:  # 卷积核宽度为2
                    if j == 0:  # 第0次迭代
                        matrix_w = w_col1  # 使用第1列权重
                    else:  # 第1次迭代
                        matrix_w = w_col0  # 使用第0列权重
                elif KERNEL_WIDTH == 3:  # 卷积核宽度为3
                    if j == 0:  # 第0次迭代
                        matrix_w = w_col2  # 使用第2列权重
                    elif j == 1:  # 第1次迭代
                        matrix_w = w_col1  # 使用第1列权重
                    else:  # 第2次迭代
                        matrix_w = w_col0  # 使用第0列权重
                elif KERNEL_WIDTH == 4:  # 卷积核宽度为4
                    if j == 0:  # 第0次迭代
                        matrix_w = w_col3  # 使用第3列权重
                    elif j == 1:  # 第1次迭代
                        matrix_w = w_col2  # 使用第2列权重
                    elif j == 2:  # 第2次迭代
                        matrix_w = w_col1  # 使用第1列权重
                    else:  # 第3次迭代
                        matrix_w = w_col0  # 使用第0列权重

                if SAVE_INTERMEDIATE:  # 如果需要保存中间状态 # 保存推测解码的中间卷积窗口
                    # Save the window state after consuming this token # 消费此token后保存窗口状态
                    # Layout: [seq(cache line), step, dim, win(K-1)] # 布局：[序列(缓存行), 步, 维度, 窗口(K-1)]
                    base_ptr = (
                        intermediate_conv_window_ptr
                        + intermediate_state_batch_coord * stride_inter_seq
                        + idx_token * stride_inter_step
                        + idx_feats * stride_inter_dim
                    )  # 计算中间状态基地址

                    # store itself in KERNEL_WIDTH-2 slot, parent in KERNEL_WIDTH-3 slot, grand-parent in KERNEL_WIDTH-4 slot, ... # 将自身存储在KERNEL_WIDTH-2槽，父节点在KERNEL_WIDTH-3槽，祖父节点在KERNEL_WIDTH-4槽，...
                    if KERNEL_WIDTH - j - 2 >= 0:  # 检查槽位是否有效 # 确保只存储有效的窗口位置
                        tl.store(
                            base_ptr + (KERNEL_WIDTH - j - 2) * stride_inter_win,
                            matrix_x,
                            mask=mask_w,
                        )  # 存储中间状态 # 将当前窗口值写入中间缓冲区

                acc += matrix_x * matrix_w  # 累加乘积 # 将输入和权重的逐元素乘积累加到累加器

                # move to parent for next iteration # 移动到父节点进行下一次迭代
                if _idx_token > 0:  # 如果当前token索引大于0 # 继续沿树向上
                    _idx_token = tl.sum(
                        tl.where(idx_tokens == _idx_token, parent_idx_tokens, 0)
                    )  # 获取父token索引 # 沿树向上移动到父节点
                    x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N] # 计算父token的x指针
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载父token的输入
                else:
                    # no parent within the current chunk, load from prev conv state: col[-1] (idx 0's parent), col[-2] (idx 0's grand parent), ... # 当前块内无父节点，从先前卷积状态加载：col[-1]（索引0的父节点），col[-2]（索引0的祖父节点），...
                    if KERNEL_WIDTH == 2:  # 卷积核宽度为2
                        if _idx_token == 0:  # 索引为0时
                            matrix_x = col0  # 使用第0列的缓存数据
                    elif KERNEL_WIDTH == 3:  # 卷积核宽度为3
                        if _idx_token == 0:  # 索引为0时
                            matrix_x = col1  # 使用第1列的缓存数据
                        else:  # 索引为-1时
                            matrix_x = col0  # 使用第0列的缓存数据
                    elif KERNEL_WIDTH == 4:  # 卷积核宽度为4
                        if _idx_token == 0:  # 索引为0时
                            matrix_x = col2  # 使用第2列的缓存数据
                        elif _idx_token == -1:  # 索引为-1时
                            matrix_x = col1  # 使用第1列的缓存数据
                        else:  # 索引为-2时
                            matrix_x = col0  # 使用第0列的缓存数据
                    _idx_token = _idx_token - 1  # 递减索引 # 继续向上查找
        else:  # 非Eagle树模式的常规卷积计算
            matrix_w = w_col0  # 初始化权重矩阵为第0列 # 当前使用的权重列
            matrix_x = col0  # 初始化输入矩阵为第0列 # 当前使用的输入列

            for j in tl.static_range(KERNEL_WIDTH):  # 静态遍历卷积核宽度 # 遍历卷积核的每一列
                if KERNEL_WIDTH == 2:  # 卷积核宽度为2的情况
                    if j == 1:  # KERNEL_WIDTH-1: # 当j=1时
                        matrix_w = w_col1  # 使用第1列权重
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N] # 计算当前token的x指针
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载当前token的输入
                elif KERNEL_WIDTH == 3:  # 卷积核宽度为3的情况
                    if j == 1:  # 当j=1时
                        matrix_w = w_col1  # 使用第1列权重
                        matrix_x = col1  # 使用第1列输入
                    elif j == 2:  # 当j=2时
                        matrix_w = w_col2  # 使用第2列权重
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N] # 计算当前token的x指针
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载当前token的输入
                elif KERNEL_WIDTH == 4:  # 卷积核宽度为4的情况
                    if j == 1:  # 当j=1时
                        matrix_w = w_col1  # 使用第1列权重
                        matrix_x = col1  # 使用第1列输入
                    elif j == 2:  # 当j=2时
                        matrix_w = w_col2  # 使用第2列权重
                        matrix_x = col2  # 使用第2列输入
                    elif j == 3:  # 当j=3时
                        matrix_w = w_col3  # 使用第3列权重
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N] # 计算当前token的x指针
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)  # 加载当前token的输入

                acc += matrix_x * matrix_w  # [BLOCK_N] # 累加乘积 # 将输入和权重的逐元素乘积累加

            if KERNEL_WIDTH == 2:  # 更新滑动窗口 - 卷积核宽度为2
                col0 = matrix_x  # 第0列更新为当前token
            elif KERNEL_WIDTH == 3:  # 更新滑动窗口 - 卷积核宽度为3
                col0 = col1  # 第0列更新为第1列
                col1 = matrix_x  # 第1列更新为当前token
            elif KERNEL_WIDTH == 4:  # 更新滑动窗口 - 卷积核宽度为4
                col0 = col1  # 第0列更新为第1列
                col1 = col2  # 第1列更新为第2列
                col2 = matrix_x  # 第2列更新为当前token

            if SAVE_INTERMEDIATE:  # 如果需要保存中间状态 # 保存推测解码的中间卷积窗口
                # Save the window state after consuming this token # 消费此token后保存窗口状态
                # Layout: [seq(cache line), step, dim, win(K-1)] # 布局：[序列(缓存行), 步, 维度, 窗口(K-1)]
                base_ptr = (
                    intermediate_conv_window_ptr
                    + intermediate_state_batch_coord * stride_inter_seq
                    + idx_token * stride_inter_step
                    + idx_feats * stride_inter_dim
                )  # 计算中间状态基地址
                if KERNEL_WIDTH >= 2:  # 保存第0列
                    tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)  # 存储col0
                if KERNEL_WIDTH >= 3:  # 保存第1列
                    tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)  # 存储col1
                if KERNEL_WIDTH >= 4:  # 保存第2列
                    tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)  # 存储col2

        if SILU_ACTIVATION:  # 如果使用SiLU激活函数 # 应用SiLU激活
            acc = acc / (1 + tl.exp(-acc))  # 计算SiLU激活函数 # SiLU(x) = x * sigmoid(x)
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index # 计算一维存储掩码 # 确保token和特征索引在有效范围内
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )  # 计算输出指针 # 根据当前位置计算输出地址

        tl.store(o_ptrs, acc, mask=mask_1d)  # 存储输出结果 # 将计算结果写入输出张量

        # fuse: store calculated retrieve_parent_token to tensor # 融合：将计算出的retrieve_parent_token存储到张量
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:  # 如果有Eagle树注意力掩码 # 保存父token索引
            tl.store(
                retrieve_parent_token_ptr
                + idx_seq * stride_retrieve_parent_token_seq
                + idx_tokens * stride_retrieve_parent_token_token,
                parent_idx_tokens,
                mask=mask_retrieve,
            )  # 存储父token索引 # 将计算出的父映射写入输出张量


def causal_conv1d_update(  # 因果一维卷积更新函数 - 用于解码阶段更新卷积状态并计算输出
    x: torch.Tensor,  # 输入张量
    conv_state: torch.Tensor,  # 卷积状态
    weight: torch.Tensor,  # 卷积权重
    bias: Optional[torch.Tensor] = None,  # 偏置项，可选
    activation: Union[bool, str, None] = None,  # 激活函数类型
    cache_seqlens: Optional[torch.Tensor] = None,  # 缓存序列长度（循环缓冲区），可选
    conv_state_indices: Optional[torch.Tensor] = None,  # 卷积状态索引，可选
    num_accept_tokens: Optional[torch.Tensor] = None,  # 接受token数量（推测解码），可选
    intermediate_conv_window: Optional[torch.Tensor] = None,  # 中间卷积窗口（推测解码），可选
    intermediate_state_indices: Optional[torch.Tensor] = None,  # 中间状态索引，可选
    retrieve_next_token: Optional[torch.Tensor] = None,  # 检索下一个token（Eagle树），可选
    retrieve_next_sibling: Optional[torch.Tensor] = None,  # 检索下一个兄弟token（Eagle树），可选
    retrieve_parent_token: Optional[torch.Tensor] = None,  # 检索父token（Eagle树），可选
    pad_slot_id: int = PAD_SLOT_ID,  # 填充槽位ID
    metadata=None,  # 元数据
    validate_data=False,  # 是否验证数据
):
    """ # 因果一维卷积更新函数的文档字符串
    x: (batch, dim) or (batch, dim, seqlen) # 输入张量，形状为(batch, dim)或(batch, dim, seqlen)
        [shape=2: single token prediction] # [形状为2：单token预测]
        [shape=3: single or multiple tokens prediction] # [形状为3：单或多token预测]
    conv_state: (..., dim, state_len), where state_len >= width - 1 # 卷积状态，形状为(..., dim, state_len)，其中state_len >= width - 1
    weight: (dim, width) # 权重，形状为(dim, width)
    bias: (dim,) # 偏置，形状为(dim,)
    cache_seqlens: (batch,), dtype int32. # 缓存序列长度，形状为(batch,)，int32类型。
        If not None, the conv_state is treated as a circular buffer. # 如果不为None，卷积状态被视为循环缓冲区。
        The conv_state will be updated by copying x to the conv_state # 卷积状态将通过将x复制到卷积状态来更新
        starting at the index # 从索引开始
        @cache_seqlens % state_len. # @cache_seqlens % state_len。
    conv_state_indices: (batch,), dtype int32 # 卷积状态索引，形状为(batch,)，int32类型
        If not None, the conv_state is a larger tensor along the batch dim, # 如果不为None，卷积状态沿批次维度是更大的张量，
        and we are selecting the batch coords specified by conv_state_indices. # 我们选择conv_state_indices指定的批次坐标。
        Useful for a continuous batching scenario. # 在连续批处理场景中很有用。
    pad_slot_id: int # 填充槽位ID，整数
            if cache_indices is passed, lets the kernel identify padded # 如果传入了cache_indices，让核函数识别填充
            entries that will not be processed, # 的不需要处理的条目，
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id] # 例如：cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at # 在这种情况下，核函数不会处理
            indices 0 and 3 # 索引0和3处的条目
    out: (batch, dim) or (batch, dim, seqlen) # 输出，形状为(batch, dim)或(batch, dim, seqlen)
    """
    if validate_data:  # 如果需要验证数据 # 数据验证
        assert cache_seqlens is None  # not implemented yet - ok for vLLM # 断言缓存序列长度为None - 尚未实现
        assert pad_slot_id is not None  # 断言填充槽位ID不为None
        assert x.stride(1) == 1  # 断言x的第二个维度步长为1
    if isinstance(activation, bool):  # 如果激活是布尔类型 # 处理布尔类型的激活参数
        activation = "silu" if activation is True else None  # True映射为"silu"，False映射为None
    elif activation is not None:  # 如果激活不是None # 验证激活函数名称
        assert activation in ["silu", "swish"]  # 断言激活函数为silu或swish
    unsqueeze = x.dim() == 2  # 检查是否需要增加维度 # 判断x是否为2D（需要扩展为3D）
    if unsqueeze:  # 如果需要增加维度
        # make it (batch, dim, seqlen) with seqlen == 1 # 将其变为(batch, dim, seqlen)，seqlen==1
        x = x.unsqueeze(-1)  # 在最后一维增加维度 # 扩展x为3D
    batch, dim, seqlen = x.shape  # 解包x的形状 # 获取批次大小、特征维度和序列长度
    _, width = weight.shape  # 解包权重的形状 # 获取卷积核宽度
    # conv_state: (..., dim, state_len), where state_len >= width - 1 # 卷积状态形状说明
    num_cache_lines, _, state_len = conv_state.size()  # 解包卷积状态的形状 # 获取缓存行数、维度和状态长度

    if validate_data:  # 数据验证逻辑
        assert dim == weight.size(0)  # 断言特征维度匹配
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"  # 断言卷积状态特征维度连续
        assert state_len >= width - 1  # 断言状态长度足够
        # when above happens, we don't shift-left to keep any records in conv_state # 当上述情况发生时，我们不移位以保留卷积状态中的记录
        assert dim == conv_state.size(1)  # 断言特征维度匹配
        if conv_state_indices is None:  # 如果没有卷积状态索引
            assert conv_state.size(0) >= batch  # 断言卷积状态足够大
        else:  # 有卷积状态索引时
            assert (batch,) == conv_state_indices.shape  # 断言索引形状匹配
            assert intermediate_state_indices is not None  # 断言中间状态索引不为None
            assert (batch,) == intermediate_state_indices.shape  # 断言中间状态索引形状匹配

        assert num_cache_lines >= batch  # 断言缓存行数足够
        assert weight.stride(1) == 1  # Need this # 需要此条件 # 断言权重连续
        assert cache_seqlens is None  # not needed for vLLM - circular buffer # vLLM不需要循环缓冲区

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o' # 采用vLLM策略直接在'x'上覆写，而非创建新张量'o'
    out = torch.empty_like(x)  # 创建与x形状相同的空输出张量 # 分配输出张量
    stride_w_dim, stride_w_width = weight.stride()  # 获取权重步长 # 读取权重的步长

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen) # 获取x的步长 # 读取输入的步长

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()  # 获取输出步长 # 读取输出的步长
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()  # 获取卷积状态步长 # 读取卷积状态的步长
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )  # 获取状态索引步长 # 如果有索引则读取步长，否则为0
    stride_intermediate_state_indices = (
        intermediate_state_indices.stride(0)
        if intermediate_state_indices is not None
        else 0
    )  # 获取中间状态索引步长 # 如果有中间索引则读取步长，否则为0
    if num_accept_tokens is not None:  # 如果有接受token数量（推测解码） # 计算有效状态长度
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed # 需要的有效状态长度
    else:
        state_len = width - 1  # 标准状态长度 # 普通解码时的状态长度
    np2_statelen = triton.next_power_of_2(state_len)  # 计算状态长度的2的幂次值 # 获取最小2的幂
    np2_seqlen = triton.next_power_of_2(seqlen)  # 计算序列长度的2的幂次值 # 获取最小2的幂

    def grid(META):  # 定义网格函数 # 用于配置Triton核的启动网格
        return (
            batch,  # 批次大小 # 序列数
            triton.cdiv(dim, META["BLOCK_N"]),  # 特征维度块数 # 沿特征维度的块数
        )

    # prepare intermediate buffer strides if provided # 如果提供了中间缓冲区则准备步长
    if intermediate_conv_window is not None:  # 如果有中间卷积窗口 # 读取中间缓冲区步长
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )  # 解包中间缓冲区步长
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0  # 无中间缓冲区时步长为0

    # prepare retrieve next token buffer strides if provided # 如果提供了检索下一个token缓冲区则准备步长
    if retrieve_next_token is not None:  # 如果有检索下一个token # 读取Eagle树token索引步长
        stride_retrieve_next_token_seq, stride_retrieve_next_token_token = (
            retrieve_next_token.stride(0),
            retrieve_next_token.stride(1),
        )  # 解包检索下一个token步长
    else:
        stride_retrieve_next_token_seq = stride_retrieve_next_token_token = 0  # 无Eagle树时步长为0

    # prepare retrieve next sibling buffer strides if provided # 如果提供了检索下一个兄弟token缓冲区则准备步长
    if retrieve_next_sibling is not None:  # 如果有检索下一个兄弟token
        stride_retrieve_next_sibling_seq, stride_retrieve_next_sibling_token = (
            retrieve_next_sibling.stride(0),
            retrieve_next_sibling.stride(1),
        )  # 解包检索下一个兄弟token步长
    else:
        stride_retrieve_next_sibling_seq = stride_retrieve_next_sibling_token = 0  # 无兄弟索引时步长为0

    # prepare retrieve parent token buffer strides if provided # 如果提供了检索父token缓冲区则准备步长
    if retrieve_parent_token is not None:  # 如果有检索父token
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (
            retrieve_parent_token.stride(0),
            retrieve_parent_token.stride(1),
        )  # 解包检索父token步长
    else:
        stride_retrieve_parent_token_seq = stride_retrieve_parent_token_token = 0  # 无父索引时步长为0

    _causal_conv1d_update_kernel[grid](  # 启动因果一维卷积更新核 # 调用Triton核函数
        # Pointers to matrices # 矩阵指针
        x,  # 输入数据
        weight,  # 卷积权重
        bias,  # 偏置
        conv_state,  # 卷积状态
        cache_seqlens,  # 缓存序列长度
        conv_state_indices,  # 卷积状态索引
        num_accept_tokens,  # 接受token数量
        intermediate_conv_window if intermediate_conv_window is not None else x,  # 中间卷积窗口（无则用x替代）
        intermediate_state_indices,  # 中间状态索引
        retrieve_next_token,  # 检索下一个token
        retrieve_next_sibling,  # 检索下一个兄弟token
        retrieve_parent_token,  # 检索父token
        out,  # 输出
        # Matrix dimensions # 矩阵维度
        batch,  # 批次大小
        dim,  # 特征维度
        seqlen,  # 序列长度
        state_len,  # 状态长度
        num_cache_lines,  # 缓存行数
        # stride # 步长
        stride_x_seq,  # x序列步长
        stride_x_dim,  # x维度步长
        stride_x_token,  # x token步长
        stride_w_dim,  # 权重维度步长
        stride_w_width,  # 权重宽度步长
        stride_istate_seq,  # 初始状态序列步长
        stride_istate_dim,  # 初始状态维度步长
        stride_istate_token,  # 初始状态token步长
        stride_state_indices,  # 状态索引步长
        stride_inter_seq,  # 中间缓冲区序列步长
        stride_inter_step,  # 中间缓冲区步长
        stride_inter_dim,  # 中间缓冲区维度步长
        stride_inter_win,  # 中间缓冲区窗口步长
        stride_intermediate_state_indices,  # 中间状态索引步长
        stride_retrieve_next_token_seq,  # 检索下一个token序列步长
        stride_retrieve_next_token_token,  # 检索下一个token token步长
        stride_retrieve_next_sibling_seq,  # 检索下一个兄弟token序列步长
        stride_retrieve_next_sibling_token,  # 检索下一个兄弟token token步长
        stride_retrieve_parent_token_seq,  # 检索父token序列步长
        stride_retrieve_parent_token_token,  # 检索父token token步长
        stride_o_seq,  # 输出序列步长
        stride_o_dim,  # 输出维度步长
        stride_o_token,  # 输出token步长
        # others # 其他参数
        pad_slot_id,  # 填充槽位ID
        # META # 元参数
        HAS_BIAS=bias is not None,  # 是否有偏置
        KERNEL_WIDTH=width,  # 卷积核宽度
        SILU_ACTIVATION=activation in ["silu", "swish"],  # 是否使用SiLU激活
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,  # 是否为连续批处理
        IS_SPEC_DECODING=num_accept_tokens is not None,  # 是否为推测解码
        NP2_STATELEN=np2_statelen,  # 状态长度的2的幂次值
        NP2_SEQLEN=np2_seqlen,  # 序列长度的2的幂次值
        USE_PAD_SLOT=pad_slot_id is not None,  # 是否使用填充槽位
        BLOCK_N=256,  # 特征维度块大小
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,  # 是否保存中间状态
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_next_token is not None,  # 是否有Eagle树自定义注意力掩码
    )
    if unsqueeze:  # 如果之前增加了维度 # 恢复原始维度
        out = out.squeeze(-1)  # 移除最后一维 # 将3D输出压缩回2D
    return out  # 返回输出张量
