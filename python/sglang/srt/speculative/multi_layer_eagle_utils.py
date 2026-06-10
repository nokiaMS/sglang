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
# 多层EAGLE工具函数模块。
# 包含用于推测解码的Triton CUDA核函数，实现输入ID旋转、
# 状态分配和隐藏状态池管理等操作。

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言


@triton.jit
def rotate_input_ids_kernel(
    input_ids_ptr,  # 输入ID指针
    extend_start_loc_ptr,  # 扩展起始位置指针
    extend_seq_lens_ptr,  # 扩展序列长度指针
    topk_index_ptr,  # topk索引指针
    select_index_ptr,  # 选择索引指针
    BLOCK_SIZE: tl.constexpr,  # 块大小常量
):
    """旋转输入ID的Triton核函数，将每个序列左移一位并在末尾填入新token。"""
    pid = tl.program_id(0)  # 获取程序ID

    start_loc = tl.load(extend_start_loc_ptr + pid)  # 加载扩展起始位置
    seq_len = tl.load(extend_seq_lens_ptr + pid)  # 加载扩展序列长度
    new_token = tl.load(topk_index_ptr + pid)  # 加载新token

    num_elements_to_shift = seq_len - 1  # 需要移动的元素数

    for off in range(0, num_elements_to_shift, BLOCK_SIZE):  # 分块移动
        offsets = off + tl.arange(0, BLOCK_SIZE)  # 计算偏移量
        mask = offsets < num_elements_to_shift  # 计算掩码

        read_ptr = input_ids_ptr + start_loc + offsets + 1  # 读取指针（偏移+1）
        val = tl.load(read_ptr, mask=mask)  # 加载值
        tl.debug_barrier()  # 调试屏障

        write_ptr = input_ids_ptr + start_loc + offsets  # 写入指针（偏移）
        tl.store(write_ptr, val, mask=mask)  # 存储值
        tl.debug_barrier()  # 调试屏障

    if seq_len > 0:  # 如果序列长度大于0
        if select_index_ptr is not None:  # 如果有选择索引
            last_pos_ptr = input_ids_ptr + tl.load(select_index_ptr + pid)  # 使用选择索引定位
        else:
            last_pos_ptr = input_ids_ptr + start_loc + seq_len - 1  # 使用末尾位置
        tl.store(last_pos_ptr, new_token)  # 在末尾存储新token


def rotate_input_ids_triton(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None  # 输入ID, # 扩展起始位置, # 扩展序列长度, # topk索引, # 选择索引，可选
):
    """旋转输入ID的Triton包装函数，将每个序列左移并填入topk token。"""
    batch_size = extend_seq_lens.shape[0]  # 获取批次大小
    BLOCK_SIZE = 4096 if select_index is not None else 8  # 根据是否有选择索引设置块大小
    grid = (batch_size,)  # 设置网格大小

    rotate_input_ids_kernel[grid](  # 启动核函数
        input_ids,  # 输入ID
        extend_start_loc,  # 扩展起始位置
        extend_seq_lens,  # 扩展序列长度
        topk_index,  # topk索引
        select_index,  # 选择索引
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小
    )
    return input_ids  # 返回旋转后的输入ID


@triton.jit
def assign_new_state_kernel(
    # Source pointers
    old_input_ids_ptr,  # 旧输入ID指针
    old_positions_ptr,  # 旧位置指针
    old_hidden_states_ptr,  # 旧隐藏状态指针
    old_out_cache_loc_ptr,  # 旧输出缓存位置指针
    old_extend_seq_lens_ptr,  # 旧扩展序列长度指针
    old_extend_start_loc_ptr,  # 旧扩展起始位置指针
    # Destination pointers
    input_ids_ptr,  # 输入ID指针
    positions_ptr,  # 位置指针
    hidden_states_ptr,  # 隐藏状态指针
    out_cache_loc_ptr,  # 输出缓存位置指针
    extend_seq_lens_ptr,  # 扩展序列长度指针
    extend_start_loc_ptr,  # 扩展起始位置指针
    # Auxiliary data pointers
    next_token_ids_ptr,  # 下一个token ID指针
    seq_lens_ptr,  # 序列长度指针
    padding_lens_ptr,  # 填充长度指针
    req_pool_indices_ptr,  # 请求池索引指针
    req_to_token_ptr,  # 请求到token映射指针
    req_to_hidden_states_pool_ptr,  # 请求到隐藏状态池指针
    # Scalars and Strides
    step,  # 步数
    stride_hidden_seq,  # 隐藏状态序列步长
    stride_hidden_dim,  # hidden_states strides  # 隐藏状态维度步长
    stride_pool_req,  # 池请求步长
    stride_pool_step,  # 池步数步长
    stride_pool_dim,  # pool strides  # 池维度步长
    stride_req_token_0,  # 请求到token第0步长
    stride_req_token_1,  # req_to_token strides  # 请求到token第1步长
    # Meta-parameters
    HIDDEN_DIM: tl.constexpr,  # 隐藏维度常量
    BLOCK_SEQ: tl.constexpr,  # 序列块大小常量
    BLOCK_HID: tl.constexpr,  # 隐藏维度块大小常量
):
    """分配新状态的Triton核函数，将旧状态复制到新位置并插入新token。"""
    pid = tl.program_id(0)  # 获取程序ID

    seq_len: tl.tensor = tl.load(seq_lens_ptr + pid)  # 加载序列长度
    old_extend_len = tl.load(old_extend_seq_lens_ptr + pid)  # 加载旧扩展长度
    old_start = tl.load(old_extend_start_loc_ptr + pid)  # 加载旧起始位置
    new_extend_len = old_extend_len + 1  # 新扩展长度加1
    new_start = old_start + pid  # 新起始位置

    tl.store(extend_seq_lens_ptr + pid, new_extend_len)  # 存储新扩展长度
    tl.store(extend_start_loc_ptr + pid, new_start)  # 存储新起始位置

    offs_seq = tl.arange(0, BLOCK_SEQ)  # 序列偏移量
    mask_seq = offs_seq < old_extend_len  # 序列掩码

    old_ids = tl.load(old_input_ids_ptr + old_start + offs_seq, mask=mask_seq)  # 加载旧输入ID
    tl.store(input_ids_ptr + new_start + offs_seq, old_ids, mask=mask_seq)  # 存储旧输入ID到新位置
    padding_len = tl.load(padding_lens_ptr + pid)  # 加载填充长度
    tl.store(  # 在新位置存储下一个token ID
        input_ids_ptr + new_start + old_extend_len - padding_len,  # 目标位置
        tl.load(next_token_ids_ptr + pid),  # 下一个token ID
    )

    old_pos = tl.load(old_positions_ptr + old_start + offs_seq, mask=mask_seq)  # 加载旧位置
    tl.store(positions_ptr + new_start + 1 + offs_seq, old_pos, mask=mask_seq)  # 存储旧位置（偏移+1）
    tl.store(  # 在起始位置存储位置
        positions_ptr + new_start, max(tl.load(old_positions_ptr + old_start) - 1, 0)  # 取最大值防止负数
    )

    old_cache = tl.load(old_out_cache_loc_ptr + old_start + offs_seq, mask=mask_seq)  # 加载旧缓存位置
    tl.store(out_cache_loc_ptr + new_start + 1 + offs_seq, old_cache, mask=mask_seq)  # 存储旧缓存位置（偏移+1）

    req_idx = tl.load(req_pool_indices_ptr + pid)  # 加载请求索引
    token_idx_col = seq_len - old_extend_len - 1  # 计算token索引列
    if token_idx_col >= 0:  # 如果索引有效
        req_token_ptr_loc = (  # 计算请求到token映射位置
            req_to_token_ptr
            + (req_idx * stride_req_token_0)
            + (token_idx_col * stride_req_token_1)
        )
        last_cache_loc = tl.load(req_token_ptr_loc)  # 加载最后缓存位置
        tl.store(out_cache_loc_ptr + new_start, last_cache_loc)  # 存储到新位置起始

    pool_vec_offset_base = ((req_idx + 1) * stride_pool_req) + (  # 计算池向量偏移量基址
        -(step + 1) * stride_pool_step
    )

    for off_h in range(0, HIDDEN_DIM, BLOCK_HID):  # 分块处理隐藏维度
        offs_h = off_h + tl.arange(0, BLOCK_HID)  # 隐藏维度偏移量
        mask_h = offs_h < HIDDEN_DIM  # 隐藏维度掩码

        for i in range(BLOCK_SEQ):  # 遍历序列块
            if i < old_extend_len:  # 如果在扩展长度内
                old_h_ptr = (  # 旧隐藏状态指针
                    old_hidden_states_ptr
                    + (old_start + i) * stride_hidden_seq
                    + (offs_h * stride_hidden_dim)
                )
                new_h_ptr = (  # 新隐藏状态指针
                    hidden_states_ptr
                    + (new_start + 1 + i) * stride_hidden_seq
                    + (offs_h * stride_hidden_dim)
                )

                chunk_old = tl.load(old_h_ptr, mask=mask_h)  # 加载旧隐藏状态块
                tl.store(new_h_ptr, chunk_old, mask=mask_h)  # 存储到新位置

        pool_ptrs = (  # 池指针
            req_to_hidden_states_pool_ptr
            + pool_vec_offset_base
            + (offs_h * stride_pool_dim)
        )
        pool_val = tl.load(pool_ptrs, mask=mask_h)  # 加载池值

        new_h_start_ptrs = (  # 新起始位置隐藏状态指针
            hidden_states_ptr
            + (new_start * stride_hidden_seq)
            + (offs_h * stride_hidden_dim)
        )
        tl.store(new_h_start_ptrs, pool_val, mask=mask_h)  # 存储池值到起始位置


def assign_new_state_triton(
    next_token_ids: torch.Tensor,  # 下一个token ID
    old_input_ids: torch.Tensor,  # 旧输入ID
    old_positions: torch.Tensor,  # 旧位置
    old_hidden_states: torch.Tensor,  # 旧隐藏状态
    old_out_cache_loc: torch.Tensor,  # 旧输出缓存位置
    old_extend_seq_lens: torch.Tensor,  # 旧扩展序列长度
    old_extend_start_loc: torch.Tensor,  # 旧扩展起始位置
    input_ids: torch.Tensor,  # 输入ID
    positions: torch.Tensor,  # 位置
    hidden_states: torch.Tensor,  # 隐藏状态
    out_cache_loc: torch.Tensor,  # 输出缓存位置
    extend_seq_lens: torch.Tensor,  # 扩展序列长度
    extend_start_loc: torch.Tensor,  # 扩展起始位置
    seq_lens: torch.Tensor,  # 序列长度
    padding_lens: torch.Tensor,  # 填充长度
    num_seqs: int,  # 序列数
    step: int,  # 步数
    req_pool_indices: torch.Tensor,  # 请求池索引
    req_to_token: torch.Tensor,  # 请求到token映射
    req_to_hidden_states_pool: torch.Tensor,  # 请求到隐藏状态池
):
    """分配新状态的Triton包装函数，计算偏移量和步长并启动核函数。"""
    """
    Wrapper function to calculate offsets and launch the Triton kernel.
    """
    hidden_dim = hidden_states.shape[1]  # 获取隐藏维度

    BLOCK_SEQ = 8  # 序列块大小
    BLOCK_HID = 64  # 隐藏维度块大小

    grid = (num_seqs,)  # 设置网格大小

    assign_new_state_kernel[grid](  # 启动核函数
        # Pointers
        old_input_ids,  # 旧输入ID
        old_positions,  # 旧位置
        old_hidden_states,  # 旧隐藏状态
        old_out_cache_loc,  # 旧输出缓存位置
        old_extend_seq_lens,  # 旧扩展序列长度
        old_extend_start_loc,  # 旧扩展起始位置
        input_ids,  # 输入ID
        positions,  # 位置
        hidden_states,  # 隐藏状态
        out_cache_loc,  # 输出缓存位置
        extend_seq_lens,  # 扩展序列长度
        extend_start_loc,  # 扩展起始位置
        next_token_ids,  # 下一个token ID
        seq_lens,  # 序列长度
        padding_lens,  # 填充长度
        req_pool_indices,  # 请求池索引
        req_to_token,  # 请求到token映射
        req_to_hidden_states_pool,  # 请求到隐藏状态池
        # Constants/Strides
        step,  # 步数
        old_hidden_states.stride(0),  # 旧隐藏状态序列步长
        old_hidden_states.stride(1),  # 旧隐藏状态维度步长
        req_to_hidden_states_pool.stride(0),  # 池请求步长
        req_to_hidden_states_pool.stride(1),  # 池步数步长
        req_to_hidden_states_pool.stride(2),  # 池维度步长
        req_to_token.stride(0),  # 请求到token第0步长
        req_to_token.stride(1),  # 请求到token第1步长
        # Meta
        HIDDEN_DIM=hidden_dim,  # 隐藏维度
        BLOCK_SEQ=BLOCK_SEQ,  # 序列块大小
        BLOCK_HID=BLOCK_HID,  # 隐藏维度块大小
    )


@triton.jit
def assign_hidden_states_pool_kernel(
    hidden_states_ptr,  # 隐藏状态指针
    req_pool_indices_ptr,  # 请求池索引指针
    req_to_hidden_states_pool_ptr,  # 请求到隐藏状态池指针
    extend_seq_lens_ptr,  # 扩展序列长度指针
    extend_start_loc_ptr,  # 扩展起始位置指针
    stride_hidden_seq,  # 隐藏状态序列步长
    stride_hidden_dim,  # 隐藏状态维度步长
    stride_pool_req,  # 池请求步长
    stride_pool_step,  # 池步数步长
    stride_pool_dim,  # 池维度步长
    HIDDEN_DIM: tl.constexpr,  # 隐藏维度常量
    pool_size: tl.constexpr,  # 池大小常量
    BLOCK_HID: tl.constexpr,  # 隐藏维度块大小常量
):
    """分配隐藏状态池的Triton核函数，将隐藏状态从扩展区域复制到池中。"""
    pid = tl.program_id(0)  # 获取程序ID

    extend_len = tl.load(extend_seq_lens_ptr + pid)  # 加载扩展长度
    start_loc = tl.load(extend_start_loc_ptr + pid)  # 加载起始位置
    end_loc = start_loc + extend_len  # 计算结束位置

    req_idx = tl.load(req_pool_indices_ptr + pid)  # 加载请求索引
    pool_vec_offset_base = req_idx * stride_pool_req  # 计算池向量偏移量基址

    for i in range(pool_size):  # 遍历池大小
        for off_h in range(0, HIDDEN_DIM, BLOCK_HID):  # 分块处理隐藏维度
            offs_h = off_h + tl.arange(0, BLOCK_HID)  # 隐藏维度偏移量
            mask_h = offs_h < HIDDEN_DIM  # 隐藏维度掩码

            hid_ptr = (  # 隐藏状态指针
                hidden_states_ptr
                + (end_loc - pool_size + i) * stride_hidden_seq
                + offs_h * stride_hidden_dim
            )
            hid_val = tl.load(hid_ptr, mask=mask_h)  # 加载隐藏状态值

            pool_ptr = (  # 池指针
                req_to_hidden_states_pool_ptr
                + pool_vec_offset_base
                + i * stride_pool_step
                + offs_h * stride_pool_dim
            )
            tl.store(pool_ptr, hid_val, mask=mask_h)  # 存储到池


def assign_hidden_states_pool_triton(
    hidden_states: torch.Tensor,  # 隐藏状态
    req_pool_indices: torch.Tensor,  # 请求池索引
    req_to_hidden_states_pool: torch.Tensor,  # 请求到隐藏状态池
    pool_size: int,  # 池大小
    num_seqs: int,  # 序列数
    extend_seq_lens: torch.Tensor,  # 扩展序列长度
    extend_start_loc: torch.Tensor,  # 扩展起始位置
):
    """分配隐藏状态池的Triton包装函数，用于KV缓存回退。"""
    grid = (num_seqs,)  # 设置网格大小
    assign_hidden_states_pool_kernel[grid](  # 启动核函数
        hidden_states,  # 隐藏状态
        req_pool_indices,  # 请求池索引
        req_to_hidden_states_pool,  # 请求到隐藏状态池
        extend_seq_lens,  # 扩展序列长度
        extend_start_loc,  # 扩展起始位置
        hidden_states.stride(0),  # 隐藏状态序列步长
        hidden_states.stride(1),  # 隐藏状态维度步长
        req_to_hidden_states_pool.stride(0),  # 池请求步长
        req_to_hidden_states_pool.stride(1),  # 池步数步长
        req_to_hidden_states_pool.stride(2),  # 池维度步长
        HIDDEN_DIM=hidden_states.shape[1],  # 隐藏维度
        pool_size=pool_size,  # 池大小
        BLOCK_HID=64,  # 隐藏维度块大小
    )


def assign_hidden_states_pool_torch(
    hidden_states: torch.Tensor,  # 隐藏状态
    req_pool_indices: torch.Tensor,  # 请求池索引
    req_to_hidden_states_pool: torch.Tensor,  # 请求到隐藏状态池
    pool_size: int,  # 池大小
    num_seqs: int,  # 序列数
    extend_seq_lens: torch.Tensor,  # 扩展序列长度
    extend_start_loc: torch.Tensor,  # 扩展起始位置
):
    """分配隐藏状态池的PyTorch实现，用于KV缓存回退（非Triton后备方案）。"""
    for req in range(num_seqs):  # 遍历每个请求
        pool_idx = req_pool_indices[req]  # 获取池索引
        extend_len = extend_seq_lens[req]  # 获取扩展长度
        start_loc = extend_start_loc[req]  # 获取起始位置
        end_loc = start_loc + extend_len  # 计算结束位置
        req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(  # 复制隐藏状态到池
            hidden_states[end_loc - pool_size : end_loc, :]  # 从扩展区域末尾取pool_size个
        )
