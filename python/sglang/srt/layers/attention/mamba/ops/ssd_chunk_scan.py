# SSD分块扫描前向计算模块 - 实现Mamba2状态空间模型中基于分块的扫描前向传播
# 包含Triton GPU内核，用于高效计算SSD（选择性状态空间）的分块扫描操作

# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py
# 改编自: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py

# SPDX-License-Identifier: Apache-2.0
# SPDX许可证标识符: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX文件版权文本: vLLM项目的版权贡献者

# Copyright (c) 2024, Tri Dao, Albert Gu.
# 版权所有 (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py
# 改编自 https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py

# ruff: noqa: E501,SIM102
# ruff: 忽略 E501(行长度),SIM102(合并if语句)

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块，别名为tl
from packaging import version  # 导入版本解析工具

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")  # 判断Triton版本是否>=2.2.0


@triton.jit  # Triton JIT编译装饰器，将函数编译为GPU内核
def _chunk_scan_fwd_kernel(  # 分块扫描前向计算内核函数
    # Pointers to matrices
    # 矩阵指针
    cb_ptr,  # CB矩阵指针（C^T * B的块矩阵）
    x_ptr,  # 输入x张量指针
    z_ptr,  # 门控z张量指针
    out_ptr,  # 输出张量指针
    out_x_ptr,  # 输出x（乘z之前）的指针
    dt_ptr,  # delta时间dt张量指针
    dA_cumsum_ptr,  # dA累积和指针
    seq_idx_ptr,  # 序列索引指针（用于连续批处理）
    C_ptr,  # C矩阵指针
    states_ptr,  # 状态张量指针
    D_ptr,  # D向量指针（跳跃连接）
    initstates_ptr,  # 初始状态指针
    chunk_indices_ptr,  # 分块索引指针
    chunk_offsets_ptr,  # 分块偏移指针
    chunk_meta_num,  # 分块元数据数量
    # Matrix dimensions
    # 矩阵维度
    chunk_size,  # 分块大小
    hdim,  # 头维度
    dstate,  # 状态维度
    batch,  # 批次大小
    seqlen,  # 序列长度
    nheads_ngroups_ratio,  # 头数与组数之比
    # Strides
    # 步幅
    stride_cb_batch,  # cb矩阵批次步幅
    stride_cb_chunk,  # cb矩阵分块步幅
    stride_cb_head,  # cb矩阵头步幅
    stride_cb_csize_m,  # cb矩阵分块大小m步幅
    stride_cb_csize_k,  # cb矩阵分块大小k步幅
    stride_x_batch,  # x张量批次步幅
    stride_x_seqlen,  # x张量序列步幅
    stride_x_head,  # x张量头步幅
    stride_x_hdim,  # x张量头维度步幅
    stride_z_batch,  # z张量批次步幅
    stride_z_seqlen,  # z张量序列步幅
    stride_z_head,  # z张量头步幅
    stride_z_hdim,  # z张量头维度步幅
    stride_out_batch,  # 输出批次步幅
    stride_out_seqlen,  # 输出序列步幅
    stride_out_head,  # 输出头步幅
    stride_out_hdim,  # 输出头维度步幅
    stride_dt_batch,  # dt批次步幅
    stride_dt_chunk,  # dt分块步幅
    stride_dt_head,  # dt头步幅
    stride_dt_csize,  # dt分块大小步幅
    stride_dA_cs_batch,  # dA累积和批次步幅
    stride_dA_cs_chunk,  # dA累积和分块步幅
    stride_dA_cs_head,  # dA累积和头步幅
    stride_dA_cs_csize,  # dA累积和分块大小步幅
    stride_seq_idx_batch,  # 序列索引批次步幅
    stride_seq_idx_seqlen,  # 序列索引序列步幅
    stride_C_batch,  # C矩阵批次步幅
    stride_C_seqlen,  # C矩阵序列步幅
    stride_C_head,  # C矩阵头步幅
    stride_C_dstate,  # C矩阵状态维度步幅
    stride_states_batch,  # 状态批次步幅
    stride_states_chunk,  # 状态分块步幅
    stride_states_head,  # 状态头步幅
    stride_states_hdim,  # 状态头维度步幅
    stride_states_dstate,  # 状态维度步幅
    stride_init_states_batch,  # 初始状态批次步幅
    stride_init_states_head,  # 初始状态头步幅
    stride_init_states_hdim,  # 初始状态头维度步幅
    stride_init_states_dstate,  # 初始状态维度步幅
    stride_D_head,  # D向量头步幅
    # Meta-parameters
    # 元参数
    IS_CAUSAL: tl.constexpr,  # 是否因果掩码
    HAS_D: tl.constexpr,  # 是否有D向量
    D_HAS_HDIM: tl.constexpr,  # D向量是否有头维度
    HAS_Z: tl.constexpr,  # 是否有z门控
    HAS_SEQ_IDX: tl.constexpr,  # 是否有序列索引
    BLOCK_SIZE_DSTATE: tl.constexpr,  # 状态维度块大小
    IS_TRITON_22: tl.constexpr,  # 是否Triton 2.2版本
    HAS_INITSTATES: tl.constexpr,  # 是否有初始状态
    BLOCK_SIZE_M: tl.constexpr = 16,  # M维度块大小，默认16
    BLOCK_SIZE_N: tl.constexpr = 16,  # N维度块大小，默认16
    BLOCK_SIZE_K: tl.constexpr = 16,  # K维度块大小，默认16
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)  # 获取批次-分块维度的程序ID，转为64位整数
    pid_c = pid_bc // batch  # 计算分块索引
    pid_b = pid_bc - pid_c * batch  # 计算批次索引
    if not HAS_INITSTATES:  # 如果没有初始状态
        c_idx = pid_c  # 分块索引直接使用pid_c
        c_off = 0  # 偏移量为0
    else:  # 如果有初始状态
        c_idx = tl.load(chunk_indices_ptr + pid_c, mask=pid_c > -1, other=0)  # 加载分块索引
        c_off = tl.load(chunk_offsets_ptr + pid_c, mask=pid_c > -1, other=0)  # 加载分块偏移

    pid_h = tl.program_id(axis=2)  # 获取头维度的程序ID
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)  # 计算N维度的块数
    pid_m = tl.program_id(axis=0) // num_pid_n  # 计算M维度的块索引
    pid_n = tl.program_id(axis=0) % num_pid_n  # 计算N维度的块索引
    cb_ptr += (  # 更新cb指针位置
        pid_b * stride_cb_batch  # 批次偏移
        + c_idx * stride_cb_chunk  # 分块偏移
        + (pid_h // nheads_ngroups_ratio) * stride_cb_head  # 头偏移
    )
    x_ptr += (  # 更新x指针位置
        pid_b * stride_x_batch  # 批次偏移
        + c_idx * chunk_size * stride_x_seqlen  # 分块偏移
        + pid_h * stride_x_head  # 头偏移
    )
    dt_ptr += pid_b * stride_dt_batch + c_idx * stride_dt_chunk + pid_h * stride_dt_head  # 更新dt指针位置
    dA_cumsum_ptr += (  # 更新dA累积和指针位置
        pid_b * stride_dA_cs_batch  # 批次偏移
        + c_idx * stride_dA_cs_chunk  # 分块偏移
        + pid_h * stride_dA_cs_head  # 头偏移
    )
    C_ptr += (  # 更新C指针位置
        pid_b * stride_C_batch  # 批次偏移
        + c_idx * chunk_size * stride_C_seqlen  # 分块偏移
        + (pid_h // nheads_ngroups_ratio) * stride_C_head  # 头偏移
    )

    # M-block offsets and prev states
    # M块偏移和前驱状态
    #  - logic in next block may override these if there is an active offset
    #  - 如果存在活跃偏移，下一块的逻辑可能会覆盖这些值
    offs_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)  # 计算M维度偏移
    prev_states_ptr = (  # 前驱状态指针
        states_ptr  # 状态基址
        + pid_b * stride_states_batch  # 批次偏移
        + c_idx * stride_states_chunk  # 分块偏移
        + pid_h * stride_states_head  # 头偏移
    )
    prev_states_hdim = stride_states_hdim  # 前驱状态头维度步幅
    prev_states_dstate = stride_states_dstate  # 前驱状态维度步幅

    chunk_size_limit = min(chunk_size, seqlen - c_idx * chunk_size)  # 计算当前分块的有效大小限制
    if HAS_SEQ_IDX:  # 如果有序列索引
        seq_idx_ptr += (  # 更新序列索引指针
            pid_b * stride_seq_idx_batch + c_idx * chunk_size * stride_seq_idx_seqlen  # 批次和分块偏移
        )

        # - we only need seq_idx_prev to be aligned to chunk boundary
        # - 我们只需要seq_idx_prev对齐到分块边界
        seq_idx_prev = tl.load(  # 加载前一个分块末尾的序列索引
            seq_idx_ptr - stride_seq_idx_seqlen, mask=c_idx >= 1, other=0  # 从当前分块起始位置前一个元素加载
        )

        if HAS_INITSTATES:  # 如果有初始状态
            # if there are init states, we only need seq_idx_m to point
            # what is the current seq_idx
            # 如果有初始状态，我们只需要seq_idx_m指向当前的序列索引

            # get current seq idx
            # 获取当前序列索引
            if (pid_m * BLOCK_SIZE_M + c_off) < chunk_size_limit:  # 如果当前M块在有效范围内
                seq_idx_m = tl.load(  # 加载当前M块的序列索引
                    seq_idx_ptr  # 序列索引基址
                    + (pid_m * BLOCK_SIZE_M + c_off) * stride_seq_idx_seqlen,  # 加上偏移
                )

                # - recall that in ssd_state_passing, for the case c_off == 0
                # i.e., the very first sequence, we made states_ptr hold its initial state
                # so this edge case is taken care of
                # - 回忆在ssd_state_passing中，对于c_off == 0的情况
                # 即第一个序列，我们让states_ptr保存其初始状态
                # 因此这个边界情况已处理
                if (  # 判断是否需要使用初始状态
                    (c_off == 0)  # 如果偏移为0
                    and (
                        seq_idx_prev != seq_idx_m
                    )  # if a seq is changed exactly on boundary
                    # 如果序列恰好在边界发生变化
                    or (c_off > 0)  # implies a new example (pseudo chunk)
                    # 或者偏移大于0（隐含一个新样本，即伪分块）
                ):

                    # - replace prev_states_ptr with init_states
                    # - 用init_states替换prev_states_ptr
                    prev_states_ptr = (  # 更新前驱状态指针为初始状态
                        initstates_ptr  # 初始状态基址
                        + seq_idx_m * stride_init_states_batch  # 对应序列的批次偏移
                        + pid_h * stride_init_states_head  # 头偏移
                    )
                    prev_states_hdim = stride_init_states_hdim  # override strides
                    # 覆盖步幅为初始状态步幅
                    prev_states_dstate = stride_init_states_dstate  # 覆盖状态维度步幅

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算N维度偏移
    dA_cs_m = tl.load(  # 加载dA累积和的M维度值
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0  # 带掩码加载
    ).to(tl.float32)  # 转换为float32

    # - handle chunk state limit
    # - 处理分块状态限制
    if HAS_INITSTATES:  # 如果有初始状态

        # have to split this if otherwise compilation will have problems
        # 必须拆分此if，否则编译会有问题
        dA_cs_m_boundary = 0.0  # dA累积和边界值，初始化为0

        # get the c_idx for the next (logica) chunk
        # 获取下一个（逻辑）分块的c_idx
        c_idx_n = tl.load(  # 加载下一个分块索引
            chunk_indices_ptr + (pid_c + 1),  # 读取pid_c+1位置
            mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,  # 掩码条件
            other=-1,  # to trigger different chunk
            # 默认值-1，触发不同分块判断
        )

        # - there are things to consider
        # - 有以下几点需要考虑
        # A. if c_off > 0 then we need to move the dA_cs boundary to ensure correct
        #    contribution of past states
        # A. 如果c_off > 0，则需要移动dA_cs边界以确保过去状态的正确贡献
        # B. if c_off_n < chunk_size_limit, then we need to adjust this so as not to
        #    encroach into the next sequence, where c_off_n is the offset of the next
        #    (logical) chunk.
        # B. 如果c_off_n < chunk_size_limit，则需要调整以不侵入下一个序列，
        #    其中c_off_n是下一个（逻辑）分块的偏移。
        # An equivalent check for B is c_idx == c_idx_n, where there is repetition in
        # (logical) chunk indices.
        # B的等效检查是c_idx == c_idx_n，即（逻辑）分块索引存在重复。

        if (c_idx == c_idx_n) or c_off > 0:  # 如果分块索引重复或偏移大于0

            # get the next offset
            # 获取下一个偏移
            c_off_n = tl.load(  # 加载下一个分块偏移
                chunk_offsets_ptr + (pid_c + 1),  # 读取pid_c+1位置
                mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,  # 掩码条件
                other=chunk_size,  # 默认值为分块大小
            )

            # in this case, adjust down the chunk_size_limit
            # 在这种情况下，向下调整chunk_size_limit
            if c_idx == c_idx_n:  # 如果当前和下一个分块索引相同
                chunk_size_limit = min(c_off_n, chunk_size_limit)  # 取偏移和限制的较小值

            # get the cs at the offset boundary
            # 获取偏移边界处的累积和
            # - c_off == 0 is a passthrough
            # - c_off == 0 是直通情况
            # - We need dA_cs at the boundary, defined by c_off - no need
            #   to increase pointer by pid_m (it is a constant offset,
            #   i.e. the same for all blocks)
            # - 我们需要边界处的dA_cs，由c_off定义 - 不需要按pid_m增加指针
            #   （它是一个常数偏移，对所有块都相同）
            dA_cs_m_boundary = tl.load(  # 加载边界处的dA累积和
                dA_cumsum_ptr + (c_off - 1) * stride_dA_cs_csize,  # c_off-1位置
                mask=(((c_off - 1) > -1) and ((c_off) < chunk_size)),  # 掩码条件
                other=0.0,  # 默认值0.0
            ).to(tl.float32)  # 转换为float32

    if HAS_SEQ_IDX:  # 如果有序列索引
        # - handle seq idx when HAS_INITSTATES==False
        # - 处理HAS_INITSTATES==False时的序列索引
        if not HAS_INITSTATES:  # 如果没有初始状态
            seq_idx_m = tl.load(  # 加载M维度的序列索引
                seq_idx_ptr + offs_m * stride_seq_idx_seqlen,  # 偏移位置
                mask=offs_m < chunk_size_limit,  # 掩码：在有效范围内
                other=-1,  # 默认值-1
            )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器为零矩阵

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # Without the if (pid_c > -1), with Triton 2.1.0, 会出现
    # 断言 `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' 失败。
    # With Triton 2.2.0, this works
    # 在Triton 2.2.0中，这可以正常工作
    if IS_TRITON_22 or c_idx > -1:  # 如果是Triton 2.2+或分块索引有效
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        # 使用更大的BLOCK_SIZE_K（最大128）进行1次迭代更快
        offs_k_dstate = tl.arange(  # 计算状态维度的K偏移
            0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K  # 小于128则用完整块，否则用BLOCK_SIZE_K
        )
        C_ptrs = C_ptr + (  # 计算C矩阵指针
            offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate  # M和K维度偏移
        )

        prev_states_ptrs = prev_states_ptr + (  # 计算前驱状态指针
            offs_n[None, :] * prev_states_hdim  # N维度偏移
            + offs_k_dstate[:, None] * prev_states_dstate  # K维度偏移
        )
        if HAS_SEQ_IDX:  # 如果有序列索引

            if not HAS_INITSTATES:  # 如果没有初始状态
                # - this is for continuous batching where there is no init states
                # - 这是用于没有初始状态的连续批处理
                scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)  # 序列相同则指数缩放，否则为0
            else:  # 如果有初始状态
                # - if there is initstates, we will rely on prev_states, no zeroing
                #   required.
                # - 如果有初始状态，将依赖prev_states，无需清零
                scale_m = tl.exp(dA_cs_m - dA_cs_m_boundary)  # 使用边界调整后的指数缩放
        else:  # 没有序列索引
            scale_m = tl.exp(dA_cs_m)  # 直接使用指数缩放
        if BLOCK_SIZE_DSTATE <= 128:  # 如果状态维度块大小<=128，可一次加载
            C = tl.load(  # 加载C矩阵块
                C_ptrs,  # C矩阵指针
                mask=(offs_m[:, None] < chunk_size_limit)  # M维度掩码
                & (offs_k_dstate[None, :] < dstate),  # K维度掩码
                other=0.0,  # 掩码外的默认值
            )

            prev_states = tl.load(  # 加载前驱状态
                prev_states_ptrs,  # 前驱状态指针
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),  # K和N维度掩码
                other=0.0,  # 掩码外的默认值
            )
            prev_states = prev_states.to(C_ptr.dtype.element_ty)  # 转换为C矩阵的数据类型
            acc = tl.dot(C, prev_states) * scale_m[:, None]  # 矩阵乘法并乘以缩放因子
        else:  # 状态维度块大小>128，需循环加载
            for k in range(0, dstate, BLOCK_SIZE_K):  # 按BLOCK_SIZE_K循环
                C = tl.load(  # 加载C矩阵块
                    C_ptrs,  # C矩阵指针
                    mask=(offs_m[:, None] < chunk_size_limit)  # M维度掩码
                    & (offs_k_dstate[None, :] < dstate - k),  # K维度掩码（减去已处理的k）
                    other=0.0,  # 掩码外的默认值
                )
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty) （此行被注释掉）
                prev_states = tl.load(  # 加载前驱状态
                    prev_states_ptrs,  # 前驱状态指针
                    mask=(offs_k_dstate[:, None] < dstate - k)  # K维度掩码
                    & (offs_n[None, :] < hdim),  # N维度掩码
                    other=0.0,  # 掩码外的默认值
                )
                prev_states = prev_states.to(C_ptr.dtype.element_ty)  # 转换为C矩阵的数据类型
                acc += tl.dot(C, prev_states)  # 累加矩阵乘法结果
                C_ptrs += BLOCK_SIZE_K  # C指针移动到下一个K块
                prev_states_ptrs += BLOCK_SIZE_K  # 前驱状态指针移动到下一个K块
            acc *= scale_m[:, None]  # 最后乘以缩放因子

    offs_k = tl.arange(0, BLOCK_SIZE_K) + c_off  # 计算K维度偏移（加上分块偏移）
    cb_ptrs = cb_ptr + (  # 计算cb矩阵指针
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k  # M和K维度偏移
    )
    x_ptrs = x_ptr + (  # 计算x张量指针
        offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim  # K和N维度偏移
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize  # 计算dt指针
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize  # 计算dA累积和指针
    K_MAX = (  # 计算K维度的最大循环次数
        chunk_size_limit  # 非因果模式：使用分块大小限制
        if not IS_CAUSAL  # 如果不是因果模式
        else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)  # 因果模式：限制到当前M块
    )
    for k in range(0, K_MAX, BLOCK_SIZE_K):  # 按BLOCK_SIZE_K循环K维度
        cb = tl.load(  # 加载cb矩阵块
            cb_ptrs,  # cb矩阵指针
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),  # 掩码
            other=0.0,  # 默认值
        ).to(tl.float32)  # 转换为float32
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(  # 加载K位置的dA累积和
            tl.float32  # 转换为float32
        )
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        # 如果存在seq_idx，我们已将seq_idx[i] != seq_idx[j]的cb[i, j]设为0。
        # 因此这里不需要对seq_idx进行掩码。
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])  # 乘以衰减因子exp(dA_cs_m - dA_cs_k)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)  # 加载dt值
        cb *= dt_k  # 乘以dt
        if IS_CAUSAL:  # 如果是因果模式
            mask = offs_m[:, None] >= k + offs_k[None, :]  # 因果掩码：只保留i>=j的位置
            cb = tl.where(mask, cb, 0.0)  # 应用因果掩码
        cb = cb.to(x_ptr.dtype.element_ty)  # 转换为x的数据类型
        x = tl.load(  # 加载x块
            x_ptrs,  # x指针
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),  # 掩码
            other=0.0,  # 默认值
        )
        acc += tl.dot(cb, x)  # 累加cb与x的矩阵乘法结果
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k  # cb指针移动到下一个K块
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen  # x指针移动到下一个K块
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize  # dt指针移动到下一个K块
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize  # dA累积和指针移动到下一个K块

    offs_out_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)  # 输出M维度偏移
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 输出N维度偏移

    if HAS_D:  # 如果有D向量（跳跃连接）
        if D_HAS_HDIM:  # 如果D有头维度
            D = tl.load(  # 加载D向量
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0  # 带掩码加载
            ).to(tl.float32)  # 转换为float32
        else:  # D没有头维度
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)  # 加载D的标量值
        x_residual = tl.load(  # 加载x残差
            x_ptr  # x基址
            + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),  # 偏移
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),  # 掩码
            other=0.0,  # 默认值
        ).to(tl.float32)  # 转换为float32
        acc += x_residual * D  # 累加跳跃连接项 x * D

    if HAS_Z:  # 如果有z门控
        out_x_ptr += (  # 更新out_x指针
            pid_b * stride_out_batch  # 批次偏移
            + c_idx * chunk_size * stride_out_seqlen  # 分块偏移
            + pid_h * stride_out_head  # 头偏移
        )
        out_x_ptrs = out_x_ptr + (  # 计算out_x的存储位置
            stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :]  # M和N维度偏移
        )
        tl.store(  # 存储out_x（门控前的输出）
            out_x_ptrs,  # 存储位置
            acc,  # 存储值
            mask=(offs_out_m[:, None] < chunk_size_limit)  # M维度掩码
            & (offs_out_n[None, :] < hdim),  # N维度掩码
        )

        z_ptr += (  # 更新z指针
            pid_b * stride_z_batch  # 批次偏移
            + c_idx * chunk_size * stride_z_seqlen  # 分块偏移
            + pid_h * stride_z_head  # 头偏移
        )
        z_ptrs = z_ptr + (  # 计算z的加载位置
            stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :]  # M和N维度偏移
        )
        z = tl.load(  # 加载z门控值
            z_ptrs,  # z指针
            mask=(offs_out_m[:, None] < chunk_size_limit)  # M维度掩码
            & (offs_out_n[None, :] < hdim),  # N维度掩码
            other=0.0,  # 默认值
        ).to(tl.float32)  # 转换为float32
        acc *= z * tl.sigmoid(z)  # 应用SiLU门控: y = x * z * sigmoid(z)

    out_ptr += (  # 更新输出指针
        pid_b * stride_out_batch  # 批次偏移
        + c_idx * chunk_size * stride_out_seqlen  # 分块偏移
        + pid_h * stride_out_head  # 头偏移
    )
    out_ptrs = out_ptr + (  # 计算输出的存储位置
        stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim  # M和N维度偏移
    )
    tl.store(  # 存储最终输出
        out_ptrs,  # 存储位置
        acc,  # 存储值
        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),  # 掩码
    )


def _chunk_scan_fwd(  # 分块扫描前向计算的主机端调用函数
    cb,  # CB矩阵（C^T * B）
    x,  # 输入张量
    dt,  # delta时间
    dA_cumsum,  # dA累积和
    C,  # C矩阵
    states,  # 状态张量
    D=None,  # D向量（跳跃连接），可选
    z=None,  # z门控张量，可选
    seq_idx=None,  # 序列索引，可选
    chunk_indices=None,  # 分块索引，可选
    chunk_offsets=None,  # 分块偏移，可选
    initial_states=None,  # 初始状态，可选
    out=None,  # 输出张量，可选
):
    batch, seqlen, nheads, headdim = x.shape  # 解包x的形状
    _, _, nchunks, chunk_size = dt.shape  # 解包dt的形状
    _, _, ngroups, dstate = C.shape  # 解包C的形状
    assert nheads % ngroups == 0  # 断言：头数必须能被组数整除
    assert C.shape == (batch, seqlen, ngroups, dstate)  # 断言C的形状
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)  # 断言cb的形状
    if z is not None:  # 如果z存在
        assert z.shape == x.shape  # 断言z的形状与x相同
    if D is not None:  # 如果D存在
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)  # 断言D的形状
    assert dt.shape == (batch, nheads, nchunks, chunk_size)  # 断言dt的形状
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)  # 断言dA_cumsum的形状
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)  # 断言states的形状

    if seq_idx is not None:  # 如果有序列索引
        assert seq_idx.shape == (batch, seqlen)  # 断言seq_idx的形状

        if initial_states is not None:  # 如果有初始状态
            # with initial states, we need to take care of how
            # seq_idx crosses the boundaries
            # 有初始状态时，需要注意seq_idx如何跨越边界
            assert batch == 1, "chunk scan only supports initial states with batch 1"  # 断言batch=1
            assert (  # 断言分块索引和偏移必须已设置
                chunk_indices is not None and chunk_offsets is not None
            ), "chunk_indices and chunk_offsets should have been set"  # 错误信息
        else:  # 没有初始状态
            chunk_indices, chunk_offsets = None, None  # 置空分块索引和偏移
    else:  # 没有序列索引
        chunk_indices, chunk_offsets = None, None  # 置空分块索引和偏移

    assert out.shape == x.shape  # 断言输出形状与x相同

    if z is not None:  # 如果z存在
        out_x = torch.empty_like(x)  # 创建与x相同形状的空张量
        assert out_x.stride() == out.stride()  # 断言步幅相同
    else:  # z不存在
        out_x = None  # out_x为None

    grid = lambda META: (  # 定义GPU内核的网格大小
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])  # M维度块数
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),  # 乘以N维度块数
        batch * nchunks if chunk_offsets is None else len(chunk_offsets),  # 批次*分块数 或 偏移数量
        nheads,  # 头数
    )
    z_strides = (  # 获取z的步幅
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))  # z各维度步幅
        if z is not None  # 如果z存在
        else (0, 0, 0, 0)  # 否则步幅全为0
    )
    _chunk_scan_fwd_kernel[grid](  # 启动分块扫描前向内核
        cb,  # CB矩阵
        x,  # 输入x
        z,  # z门控
        out,  # 输出
        out_x,  # 输出x（门控前）
        dt,  # delta时间
        dA_cumsum,  # dA累积和
        seq_idx,  # 序列索引
        C,  # C矩阵
        states,  # 状态
        D,  # D向量
        initial_states,  # 初始状态
        chunk_indices,  # 分块索引
        chunk_offsets,  # 分块偏移
        len(chunk_indices) if chunk_indices is not None else 0,  # 分块元数据数量
        chunk_size,  # 分块大小
        headdim,  # 头维度
        dstate,  # 状态维度
        batch,  # 批次大小
        seqlen,  # 序列长度
        nheads // ngroups,  # 头数与组数之比
        cb.stride(0),  # cb批次步幅
        cb.stride(1),  # cb分块步幅
        cb.stride(2),  # cb头步幅
        cb.stride(3),  # cb csize_m步幅
        cb.stride(4),  # cb csize_k步幅
        x.stride(0),  # x批次步幅
        x.stride(1),  # x序列步幅
        x.stride(2),  # x头步幅
        x.stride(3),  # x头维度步幅
        z_strides[0],  # z批次步幅
        z_strides[1],  # z序列步幅
        z_strides[2],  # z头步幅
        z_strides[3],  # z头维度步幅
        out.stride(0),  # 输出批次步幅
        out.stride(1),  # 输出序列步幅
        out.stride(2),  # 输出头步幅
        out.stride(3),  # 输出头维度步幅
        dt.stride(0),  # dt批次步幅
        dt.stride(2),  # dt分块步幅
        dt.stride(1),  # dt头步幅
        dt.stride(3),  # dt csize步幅
        dA_cumsum.stride(0),  # dA累积和批次步幅
        dA_cumsum.stride(2),  # dA累积和分块步幅
        dA_cumsum.stride(1),  # dA累积和头步幅
        dA_cumsum.stride(3),  # dA累积和csize步幅
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),  # seq_idx步幅
        C.stride(0),  # C批次步幅
        C.stride(1),  # C序列步幅
        C.stride(2),  # C头步幅
        C.stride(3),  # C状态维度步幅
        states.stride(0),  # 状态批次步幅
        states.stride(1),  # 状态分块步幅
        states.stride(2),  # 状态头步幅
        states.stride(3),  # 状态头维度步幅
        states.stride(4),  # 状态维度步幅
        *(  # 初始状态步幅
            (
                initial_states.stride(0),  # 初始状态批次步幅
                initial_states.stride(1),  # 初始状态头步幅
                initial_states.stride(2),  # 初始状态头维度步幅
                initial_states.stride(3),  # 初始状态维度步幅
            )
            if initial_states is not None  # 如果初始状态存在
            else (0, 0, 0, 0)  # 否则步幅全为0
        ),
        D.stride(0) if D is not None else 0,  # D头步幅或0
        True,  # IS_CAUSAL=True
        D is not None,  # HAS_D
        D.dim() == 2 if D is not None else True,  # D_HAS_HDIM
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),  # 状态维度块大小
        HAS_Z=z is not None,  # 是否有z门控
        HAS_SEQ_IDX=seq_idx is not None,  # 是否有序列索引
        IS_TRITON_22=TRITON_22,  # Triton版本标志
        HAS_INITSTATES=initial_states is not None,  # 是否有初始状态
    )
    return out_x  # 返回输出x（门控前的输出）
