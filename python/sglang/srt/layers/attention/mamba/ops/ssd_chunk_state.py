# SSD分块状态计算模块 - 实现Mamba2状态空间模型中基于分块的状态计算
# 包含三个核心Triton内核：累积和计算、块内状态计算、变长序列状态计算

# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py
# 改编自: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py

# SPDX-License-Identifier: Apache-2.0
# SPDX许可证标识符: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX文件版权文本: vLLM项目的版权贡献者

# Copyright (c) 2024, Tri Dao, Albert Gu.
# 版权所有 (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py
# 改编自 https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# ruff: noqa: E501
# ruff: 忽略 E501(行长度)

import math  # 导入数学模块

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块，别名为tl

from .mamba_ssm import softplus  # 从mamba_ssm模块导入softplus函数


@triton.jit  # Triton JIT编译装饰器
def _chunk_cumsum_fwd_kernel(  # 分块累积和前向计算内核
    # Pointers to matrices
    # 矩阵指针
    dt_ptr,  # dt输入指针
    A_ptr,  # A矩阵指针
    dt_bias_ptr,  # dt偏置指针
    dt_out_ptr,  # dt输出指针
    dA_cumsum_ptr,  # dA累积和输出指针
    # Matrix dimension
    # 矩阵维度
    batch,  # 批次大小
    seqlen,  # 序列长度
    nheads,  # 头数量
    chunk_size,  # 分块大小
    dt_min,  # dt最小值
    dt_max,  # dt最大值
    # Strides
    # 步幅
    stride_dt_batch,  # dt批次步幅
    stride_dt_seqlen,  # dt序列步幅
    stride_dt_head,  # dt头步幅
    stride_A_head,  # A头步幅
    stride_dt_bias_head,  # dt偏置头步幅
    stride_dt_out_batch,  # dt输出批次步幅
    stride_dt_out_chunk,  # dt输出分块步幅
    stride_dt_out_head,  # dt输出头步幅
    stride_dt_out_csize,  # dt输出分块大小步幅
    stride_dA_cs_batch,  # dA累积和批次步幅
    stride_dA_cs_chunk,  # dA累积和分块步幅
    stride_dA_cs_head,  # dA累积和头步幅
    stride_dA_cs_csize,  # dA累积和分块大小步幅
    # Meta-parameters
    # 元参数
    DT_SOFTPLUS: tl.constexpr,  # 是否对dt应用softplus
    HAS_DT_BIAS: tl.constexpr,  # 是否有dt偏置
    BLOCK_SIZE_CHUNK: tl.constexpr,  # 分块维度块大小
    BLOCK_SIZE_H: tl.constexpr = 16,  # 头维度块大小，默认16
):
    pid_b = tl.program_id(axis=0)  # 获取批次维度的程序ID

    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    # 如果dt很长可能会有问题，所以使用64位
    # https://github.com/triton-lang/triton/issues/1058
    pid_c = tl.program_id(axis=1).to(tl.int64)  # 获取分块维度的程序ID，转为64位
    pid_h = tl.program_id(axis=2)  # 获取头维度的程序ID
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen  # 更新dt指针
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk  # 更新dt输出指针
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk  # 更新dA累积和指针

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)  # 计算头维度偏移
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)  # 计算分块维度偏移
    dt_ptrs = dt_ptr + (  # 计算dt加载位置
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen  # 头和分块维度偏移
    )
    A_ptrs = A_ptr + offs_h * stride_A_head  # 计算A加载位置
    dt_out_ptrs = dt_out_ptr + (  # 计算dt输出存储位置
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize  # 头和分块维度偏移
    )
    dA_cs_ptrs = dA_cumsum_ptr + (  # 计算dA累积和存储位置
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize  # 头和分块维度偏移
    )
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)  # 计算当前分块的有效大小限制

    dt = tl.load(  # 加载dt值
        dt_ptrs,  # dt指针
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),  # 掩码
        other=0.0,  # 默认值
    ).to(tl.float32)  # 转换为float32
    if HAS_DT_BIAS:  # 如果有dt偏置
        dt_bias = tl.load(  # 加载dt偏置
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0  # 带掩码加载
        ).to(tl.float32)  # 转换为float32
        dt += dt_bias[:, None]  # 加上偏置
    if DT_SOFTPLUS:  # 如果需要softplus激活
        dt = tl.where(dt <= 20.0, softplus(dt), dt)  # 对<=20的值应用softplus，>20的值保持不变（避免数值问题）
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    # 截至 Triton 2.2.0，tl.clamp 尚不可用
    # dt = tl.clamp(dt, dt_min, dt_max) （此行被注释掉）
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)  # 将dt裁剪到[dt_min, dt_max]范围
    dt = tl.where(  # 对超出有效范围的值置零
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0  # 有效范围内保留dt，否则为0
    )
    tl.store(  # 存储处理后的dt
        dt_out_ptrs,  # 存储位置
        dt,  # 存储值
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),  # 掩码（使用完整chunk_size而非limit）
    )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)  # 加载A值
    dA = dt * A[:, None]  # 计算 dA = dt * A
    dA_cs = tl.cumsum(dA, axis=1)  # 沿序列维度计算累积和
    tl.store(  # 存储dA累积和
        dA_cs_ptrs,  # 存储位置
        dA_cs,  # 存储值
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),  # 掩码
    )


@triton.jit  # Triton JIT编译装饰器
def _chunk_state_fwd_kernel(  # 分块状态前向计算内核
    # Pointers to matrices
    # 矩阵指针
    x_ptr,  # 输入x指针
    b_ptr,  # B矩阵指针
    states_ptr,  # 状态输出指针
    dt_ptr,  # dt指针
    dA_cumsum_ptr,  # dA累积和指针
    seq_idx_ptr,  # 序列索引指针
    # Matrix dimensions
    # 矩阵维度
    hdim,  # 头维度
    dstate,  # 状态维度
    chunk_size,  # 分块大小
    batch,  # 批次大小
    seqlen,  # 序列长度
    nheads_ngroups_ratio,  # 头数与组数之比
    # Strides
    # 步幅
    stride_x_batch,  # x批次步幅
    stride_x_seqlen,  # x序列步幅
    stride_x_head,  # x头步幅
    stride_x_hdim,  # x头维度步幅
    stride_b_batch,  # B批次步幅
    stride_b_seqlen,  # B序列步幅
    stride_b_head,  # B头步幅
    stride_b_dstate,  # B状态维度步幅
    stride_states_batch,  # 状态批次步幅
    stride_states_chunk,  # 状态分块步幅
    stride_states_head,  # 状态头步幅
    stride_states_hdim,  # 状态头维度步幅
    stride_states_dstate,  # 状态维度步幅
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
    # Meta-parameters
    # 元参数
    HAS_SEQ_IDX: tl.constexpr,  # 是否有序列索引
    BLOCK_SIZE_M: tl.constexpr = 16,  # M维度块大小，默认16
    BLOCK_SIZE_N: tl.constexpr = 16,  # N维度块大小，默认16
    BLOCK_SIZE_K: tl.constexpr = 16,  # K维度块大小，默认16
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)  # 获取批次-分块维度的程序ID
    pid_c = pid_bc // batch  # 计算分块索引
    pid_b = pid_bc - pid_c * batch  # 计算批次索引
    pid_h = tl.program_id(axis=2)  # 获取头维度的程序ID
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)  # 计算N维度的块数
    pid_m = tl.program_id(axis=0) // num_pid_n  # 计算M维度的块索引
    pid_n = tl.program_id(axis=0) % num_pid_n  # 计算N维度的块索引
    b_ptr += (  # 更新B矩阵指针
        pid_b * stride_b_batch  # 批次偏移
        + pid_c * chunk_size * stride_b_seqlen  # 分块偏移
        + (pid_h // nheads_ngroups_ratio) * stride_b_head  # 头偏移
    )
    x_ptr += (  # 更新x指针
        pid_b * stride_x_batch  # 批次偏移
        + pid_c * chunk_size * stride_x_seqlen  # 分块偏移
        + pid_h * stride_x_head  # 头偏移
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head  # 更新dt指针
    dA_cumsum_ptr += (  # 更新dA累积和指针
        pid_b * stride_dA_cs_batch  # 批次偏移
        + pid_c * stride_dA_cs_chunk  # 分块偏移
        + pid_h * stride_dA_cs_head  # 头偏移
    )
    if HAS_SEQ_IDX:  # 如果有序列索引
        seq_idx_ptr += (  # 更新序列索引指针
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen  # 批次和分块偏移
        )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 计算M维度偏移
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算N维度偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # 计算K维度偏移
    x_ptrs = x_ptr + (  # 计算x加载位置
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen  # M和K维度偏移
    )
    b_ptrs = b_ptr + (  # 计算B加载位置
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen  # N和K维度偏移
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize  # 计算dt加载位置
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(  # 加载分块末尾的dA累积和
        tl.float32  # 转换为float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize  # 计算dA累积和加载位置
    if HAS_SEQ_IDX:  # 如果有序列索引
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen  # 计算序列索引加载位置

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)  # 计算当前分块的有效大小限制
    if HAS_SEQ_IDX:  # 如果有序列索引
        seq_idx_last = tl.load(  # 加载分块末尾的序列索引
            seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen  # 最后一个有效位置
        )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器为零矩阵
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):  # 按BLOCK_SIZE_K循环
        x = tl.load(  # 加载x块
            x_ptrs,  # x指针
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),  # 掩码
            other=0.0,  # 默认值
        )
        b = tl.load(  # 加载B块
            b_ptrs,  # B指针
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),  # 掩码
            other=0.0,  # 默认值
        ).to(tl.float32)  # 转换为float32
        dA_cs_k = tl.load(  # 加载K位置的dA累积和
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0  # 掩码
        ).to(tl.float32)  # 转换为float32
        if HAS_SEQ_IDX:  # 如果有序列索引
            seq_idx_k = tl.load(  # 加载K位置的序列索引
                seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1  # 掩码，默认-1
            )
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(  # 加载dt值
            tl.float32  # 转换为float32
        )
        if not HAS_SEQ_IDX:  # 如果没有序列索引
            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k  # 缩放因子 = exp(dA_cs_last - dA_cs_k) * dt
        else:  # 有序列索引
            scale = tl.where(  # 条件缩放
                seq_idx_k == seq_idx_last, tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0  # 序列匹配时缩放，否则为0
            )
        b *= scale[:, None]  # 将缩放因子应用到B
        b = b.to(x_ptr.dtype.element_ty)  # 转换B为x的数据类型
        acc += tl.dot(x, b)  # 累加x和B的矩阵乘法
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen  # x指针移动到下一个K块
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen  # B指针移动到下一个K块
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize  # dt指针移动到下一个K块
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize  # dA累积和指针移动到下一个K块
        if HAS_SEQ_IDX:  # 如果有序列索引
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen  # 序列索引指针移动到下一个K块
    states = acc.to(states_ptr.dtype.element_ty)  # 转换累加结果为状态的数据类型

    states_ptr += (  # 更新状态指针
        pid_b * stride_states_batch  # 批次偏移
        + pid_c * stride_states_chunk  # 分块偏移
        + pid_h * stride_states_head  # 头偏移
    )
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 重新计算M维度偏移
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 重新计算N维度偏移
    states_ptrs = states_ptr + (  # 计算状态的存储位置
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate  # M和N维度偏移
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)  # 计算存储掩码
    tl.store(states_ptrs, states, mask=c_mask)  # 存储计算得到的状态


@triton.jit  # Triton JIT编译装饰器
def _chunk_state_varlen_kernel(  # 变长序列分块状态计算内核
    # Pointers to matrices
    # 矩阵指针
    x_ptr,  # 输入x指针
    b_ptr,  # B矩阵指针
    dt_ptr,  # dt指针
    dA_cumsum_ptr,  # dA累积和指针
    chunk_states_ptr,  # 分块状态指针
    cu_seqlens_ptr,  # 累积序列长度指针
    states_ptr,  # 状态输出指针
    initstates_ptr,  # 初始状态指针
    # Matrix dimensions
    # 矩阵维度
    hdim,  # 头维度
    dstate,  # 状态维度
    chunk_size,  # 分块大小
    seqlen,  # 序列总长度
    nheads_ngroups_ratio,  # 头数与组数之比
    # Strides
    # 步幅
    stride_x_seqlen,  # x序列步幅
    stride_x_head,  # x头步幅
    stride_x_hdim,  # x头维度步幅
    stride_b_seqlen,  # B序列步幅
    stride_b_head,  # B头步幅
    stride_b_dstate,  # B状态维度步幅
    stride_dt_chunk,  # dt分块步幅
    stride_dt_head,  # dt头步幅
    stride_dt_csize,  # dt分块大小步幅
    stride_dA_cs_chunk,  # dA累积和分块步幅
    stride_dA_cs_head,  # dA累积和头步幅
    stride_dA_cs_csize,  # dA累积和分块大小步幅
    stride_chunk_states_chunk,  # 分块状态的分块步幅
    stride_chunk_states_head,  # 分块状态的头步幅
    stride_chunk_states_hdim,  # 分块状态的头维度步幅
    stride_chunk_states_dstate,  # 分块状态的状态维度步幅
    stride_states_batch,  # 状态批次步幅
    stride_states_head,  # 状态头步幅
    stride_states_hdim,  # 状态头维度步幅
    stride_states_dstate,  # 状态维度步幅
    stride_init_states_batch,  # 初始状态批次步幅
    stride_init_states_head,  # 初始状态头步幅
    stride_init_states_hdim,  # 初始状态头维度步幅
    stride_init_states_dstate,  # 初始状态维度步幅
    # Meta-parameters
    # 元参数
    HAS_INITSTATES: tl.constexpr,  # 是否有初始状态
    BLOCK_SIZE_M: tl.constexpr = 16,  # M维度块大小，默认16
    BLOCK_SIZE_N: tl.constexpr = 16,  # N维度块大小，默认16
    BLOCK_SIZE_K: tl.constexpr = 16,  # K维度块大小，默认16
):
    pid_b = tl.program_id(axis=1)  # 获取批次维度的程序ID
    pid_h = tl.program_id(axis=2)  # 获取头维度的程序ID
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)  # 计算N维度的块数
    pid_m = tl.program_id(axis=0) // num_pid_n  # 计算M维度的块索引
    pid_n = tl.program_id(axis=0) % num_pid_n  # 计算N维度的块索引
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)  # 加载当前序列的结束索引
    pid_c = (end_idx - 1) // chunk_size  # 计算当前序列所在的最后一个分块索引
    b_ptr += (  # 更新B矩阵指针
        pid_c * chunk_size * stride_b_seqlen  # 分块偏移
        + (pid_h // nheads_ngroups_ratio) * stride_b_head  # 头偏移
    )
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head  # 更新x指针
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head  # 更新dt指针
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head  # 更新dA累积和指针
    chunk_states_ptr += (  # 更新分块状态指针
        pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head  # 分块和头偏移
    )

    if HAS_INITSTATES:  # 如果有初始状态
        # if there are init states provided, we differentiate between states (which
        # are boundary conditions at a chunk boundary) and initstates (which are boundary
        # conditions when a new example in a cont batch starts)
        # 如果提供了初始状态，我们区分states（分块边界处的边界条件）
        # 和initstates（连续批处理中新样本开始时的边界条件）
        initstates_ptr += pid_h * stride_init_states_head  # 更新初始状态指针

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 计算M维度偏移
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算N维度偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # 计算K维度偏移
    x_ptrs = x_ptr + (  # 计算x加载位置
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen  # M和K维度偏移
    )
    b_ptrs = b_ptr + (  # 计算B加载位置
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen  # N和K维度偏移
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize  # 计算dt加载位置
    dA_cs_last = tl.load(  # 加载序列末尾的dA累积和
        dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize  # 序列末尾位置
    ).to(tl.float32)  # 转换为float32
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize  # 计算dA累积和加载位置

    chunk_size_limit = end_idx - pid_c * chunk_size  # 计算当前分块的有效大小限制
    start_idx = tl.load(cu_seqlens_ptr + pid_b)  # 加载当前序列的起始索引
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)  # 计算当前分块内的起始偏移

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器为零矩阵
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):  # 按BLOCK_SIZE_K循环
        x = tl.load(  # 加载x块
            x_ptrs,  # x指针
            mask=(offs_m[:, None] < hdim)  # M维度掩码
            & (offs_k[None, :] < chunk_size_limit - k)  # K维度上界掩码
            & (offs_k[None, :] >= start_idx_cur - k),  # K维度下界掩码
            other=0.0,  # 默认值
        )
        b = tl.load(  # 加载B块
            b_ptrs,  # B指针
            mask=(offs_k[:, None] < chunk_size_limit - k)  # K维度上界掩码
            & (offs_n[None, :] < dstate)  # N维度掩码
            & (offs_k[:, None] >= start_idx_cur - k),  # K维度下界掩码
            other=0.0,  # 默认值
        ).to(tl.float32)  # 转换为float32
        dA_cs_k = tl.load(  # 加载K位置的dA累积和
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0  # 掩码
        ).to(tl.float32)  # 转换为float32
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(  # 加载dt值
            tl.float32  # 转换为float32
        )
        scale = tl.where(  # 计算缩放因子
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),  # 在有效范围内
            tl.exp(dA_cs_last - dA_cs_k) * dt_k,  # 缩放因子 = exp(dA_cs_last - dA_cs_k) * dt
            0.0,  # 超出范围为0
        )
        b *= scale[:, None]  # 将缩放因子应用到B
        b = b.to(x_ptr.dtype.element_ty)  # 转换B为x的数据类型
        acc += tl.dot(x, b)  # 累加x和B的矩阵乘法
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen  # x指针移动到下一个K块
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen  # B指针移动到下一个K块
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize  # dt指针移动到下一个K块
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize  # dA累积和指针移动到下一个K块

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    # If HAS_INITSTATES==True need to consider two possibilities
    # - if start_idx < pid_c * chunk_size, then we need to take the past_states_ptrs
    # - if state_idx >= pid * chunk_size, then we need to insert initstates
    # 如果序列在最后一个分块索引之后开始，则不需要添加最后一个分块的贡献
    # 如果HAS_INITSTATES==True，需要考虑两种可能性
    # - 如果start_idx < pid_c * chunk_size，则需要取past_states_ptrs
    # - 如果state_idx >= pid * chunk_size，则需要插入initstates
    if (start_idx < pid_c * chunk_size) or (HAS_INITSTATES):  # first chunk
    # 如果序列起始在分块之前，或有初始状态（第一种情况）

        dA_cs_boundary = 0.0  # default
        # dA累积和边界值，默认0.0

        if not HAS_INITSTATES:  # 如果没有初始状态
            past_states_ptrs = chunk_states_ptr + (  # 使用分块状态作为前驱状态
                offs_m[:, None] * stride_chunk_states_hdim  # M维度偏移
                + offs_n[None, :] * stride_chunk_states_dstate  # N维度偏移
            )
        else:  # 有初始状态

            # - this seems repetitive, buts its to help the compiler
            # - 这看起来重复，但有助于编译器优化
            if start_idx < pid_c * chunk_size:  # 如果序列在当前分块之前开始
                past_states_ptrs = chunk_states_ptr + (  # 使用分块状态
                    offs_m[:, None] * stride_chunk_states_hdim  # M维度偏移
                    + offs_n[None, :] * stride_chunk_states_dstate  # N维度偏移
                )
            else:  # 序列在当前分块内开始
                past_states_ptrs = initstates_ptr + (  # 使用初始状态
                    pid_b * stride_init_states_batch  # 批次偏移
                    + offs_m[:, None] * stride_init_states_hdim  # M维度偏移
                    + offs_n[None, :] * stride_init_states_dstate  # N维度偏移
                )

                # need to adjust the boundary
                # 需要调整边界
                if start_idx > pid_c * chunk_size:  # 如果序列起始严格在分块内
                    dA_cs_boundary = tl.load(  # 加载边界处的dA累积和
                        dA_cumsum_ptr  # dA累积和基址
                        + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize  # 序列起始前一个位置
                    ).to(tl.float32)  # 转换为float32

        past_states = tl.load(  # 加载前驱状态
            past_states_ptrs,  # 前驱状态指针
            mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),  # 掩码
            other=0.0,  # 默认值
        ).to(tl.float32)  # 转换为float32

        scale = tl.exp(dA_cs_last - dA_cs_boundary)  # 计算缩放因子 = exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale  # 累加前驱状态乘以缩放因子

    states = acc.to(states_ptr.dtype.element_ty)  # 转换累加结果为状态的数据类型

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head  # 更新状态输出指针
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 重新计算M维度偏移
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 重新计算N维度偏移
    states_ptrs = states_ptr + (  # 计算状态的存储位置
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate  # M和N维度偏移
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)  # 计算存储掩码
    tl.store(states_ptrs, states, mask=c_mask)  # 存储计算得到的状态


def _chunk_cumsum_fwd(  # 分块累积和前向计算的主机端调用函数
    dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))  # dt, A矩阵, 分块大小, dt偏置, softplus标志, dt范围
):
    batch, seqlen, nheads = dt.shape  # 解包dt的形状
    assert A.shape == (nheads,)  # 断言A的形状
    if dt_bias is not None:  # 如果有dt偏置
        assert dt_bias.shape == (nheads,)  # 断言dt偏置的形状
    nchunks = math.ceil(seqlen / chunk_size)  # 计算分块数量
    dt_out = torch.empty(  # 创建dt输出张量
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32  # 批次,头,分块,分块大小
    )
    dA_cumsum = torch.empty(  # 创建dA累积和输出张量
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32  # 批次,头,分块,分块大小
    )
    grid_chunk_cs = lambda META: (  # 定义GPU内核的网格大小
        batch,  # 批次维度
        nchunks,  # 分块维度
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),  # 头维度块数
    )
    with torch.get_device_module(dt.device).device(dt.device.index):  # 在对应设备上执行
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](  # 启动累积和前向内核
            dt,  # dt输入
            A,  # A矩阵
            dt_bias,  # dt偏置
            dt_out,  # dt输出
            dA_cumsum,  # dA累积和输出
            batch,  # 批次大小
            seqlen,  # 序列长度
            nheads,  # 头数量
            chunk_size,  # 分块大小
            dt_limit[0],  # dt最小值
            dt_limit[1],  # dt最大值
            dt.stride(0),  # dt批次步幅
            dt.stride(1),  # dt序列步幅
            dt.stride(2),  # dt头步幅
            A.stride(0),  # A头步幅
            dt_bias.stride(0) if dt_bias is not None else 0,  # dt偏置步幅或0
            dt_out.stride(0),  # dt输出批次步幅
            dt_out.stride(2),  # dt输出分块步幅
            dt_out.stride(1),  # dt输出头步幅
            dt_out.stride(3),  # dt输出csize步幅
            dA_cumsum.stride(0),  # dA累积和批次步幅
            dA_cumsum.stride(2),  # dA累积和分块步幅
            dA_cumsum.stride(1),  # dA累积和头步幅
            dA_cumsum.stride(3),  # dA累积和csize步幅
            dt_softplus,  # 是否应用softplus
            HAS_DT_BIAS=dt_bias is not None,  # 是否有dt偏置
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),  # 分块维度块大小（2的幂次）
        )
    return dA_cumsum, dt_out  # 返回dA累积和和处理后的dt


def _chunk_state_fwd(  # 分块状态前向计算的主机端调用函数
    B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True  # B矩阵, x, dt, dA累积和, 序列索引, 状态, 是否fp32状态
):
    batch, seqlen, nheads, headdim = x.shape  # 解包x的形状
    _, _, nchunks, chunk_size = dt.shape  # 解包dt的形状
    _, _, ngroups, dstate = B.shape  # 解包B的形状
    assert nheads % ngroups == 0  # 断言：头数必须能被组数整除
    assert B.shape == (batch, seqlen, ngroups, dstate)  # 断言B的形状
    assert dt.shape == (batch, nheads, nchunks, chunk_size)  # 断言dt的形状
    assert dA_cumsum.shape == dt.shape  # 断言dA_cumsum与dt形状相同
    if seq_idx is not None:  # 如果有序列索引
        assert seq_idx.shape == (batch, seqlen)  # 断言seq_idx的形状
    if states is not None:  # 如果提供了状态张量
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)  # 断言states的形状
    else:  # 没有提供状态张量
        states_dtype = torch.float32 if states_in_fp32 else B.dtype  # 选择状态数据类型
        states = torch.empty(  # 创建状态张量
            (batch, nchunks, nheads, headdim, dstate),  # 形状
            device=x.device,  # 设备
            dtype=states_dtype,  # 数据类型
        )
    grid = lambda META: (  # 定义GPU内核的网格大小
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])  # M维度块数
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),  # 乘以N维度块数
        batch * nchunks,  # 批次*分块数
        nheads,  # 头数
    )
    with torch.get_device_module(x.device).device(x.device.index):  # 在对应设备上执行
        _chunk_state_fwd_kernel[grid](  # 启动分块状态前向内核
            x,  # 输入x
            B,  # B矩阵
            states,  # 状态
            dt,  # dt
            dA_cumsum,  # dA累积和
            seq_idx,  # 序列索引
            headdim,  # 头维度
            dstate,  # 状态维度
            chunk_size,  # 分块大小
            batch,  # 批次大小
            seqlen,  # 序列长度
            nheads // ngroups,  # 头数与组数之比
            x.stride(0),  # x批次步幅
            x.stride(1),  # x序列步幅
            x.stride(2),  # x头步幅
            x.stride(3),  # x头维度步幅
            B.stride(0),  # B批次步幅
            B.stride(1),  # B序列步幅
            B.stride(2),  # B头步幅
            B.stride(-1),  # B状态维度步幅
            states.stride(0),  # 状态批次步幅
            states.stride(1),  # 状态分块步幅
            states.stride(2),  # 状态头步幅
            states.stride(3),  # 状态头维度步幅
            states.stride(4),  # 状态维度步幅
            dt.stride(0),  # dt批次步幅
            dt.stride(2),  # dt分块步幅
            dt.stride(1),  # dt头步幅
            dt.stride(3),  # dt csize步幅
            dA_cumsum.stride(0),  # dA累积和批次步幅
            dA_cumsum.stride(2),  # dA累积和分块步幅
            dA_cumsum.stride(1),  # dA累积和头步幅
            dA_cumsum.stride(3),  # dA累积和csize步幅
            *(  # 序列索引步幅
                (seq_idx.stride(0), seq_idx.stride(1))  # seq_idx各维度步幅
                if seq_idx is not None  # 如果seq_idx存在
                else (0, 0)  # 否则步幅为0
            ),
            HAS_SEQ_IDX=seq_idx is not None,  # 是否有序列索引
        )
    return states  # 返回计算得到的状态


def chunk_state_varlen(  # 变长序列分块状态计算的主机端调用函数
    B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None  # B矩阵, x, dt, dA累积和, 累积序列长度, 分块状态, 初始状态
):
    total_seqlen, nheads, headdim = x.shape  # 解包x的形状
    _, nchunks, chunk_size = dt.shape  # 解包dt的形状
    _, ngroups, dstate = B.shape  # 解包B的形状
    batch = cu_seqlens.shape[0] - 1  # 从cu_seqlens推算批次大小
    cu_seqlens = cu_seqlens.contiguous()  # 确保cu_seqlens内存连续
    assert nheads % ngroups == 0  # 断言：头数必须能被组数整除
    assert B.shape == (total_seqlen, ngroups, dstate)  # 断言B的形状
    assert dt.shape == (nheads, nchunks, chunk_size)  # 断言dt的形状
    assert dA_cumsum.shape == dt.shape  # 断言dA_cumsum与dt形状相同
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)  # 断言分块状态的形状

    if initial_states is not None:  # 如果有初始状态
        assert initial_states.shape == (batch, nheads, headdim, dstate)  # 断言初始状态的形状

    states = torch.empty(  # 创建状态输出张量
        batch,  # 批次
        nheads,  # 头数
        headdim,  # 头维度
        dstate,  # 状态维度
        dtype=chunk_states.dtype,  # 与分块状态相同的数据类型
        device=chunk_states.device,  # 与分块状态相同的设备
    )
    grid = lambda META: (  # 定义GPU内核的网格大小
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])  # M维度块数
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),  # 乘以N维度块数
        batch,  # 批次维度
        nheads,  # 头维度
    )
    with torch.get_device_module(x.device).device(x.device.index):  # 在对应设备上执行
        _chunk_state_varlen_kernel[grid](  # 启动变长序列分块状态内核
            x,  # 输入x
            B,  # B矩阵
            dt,  # dt
            dA_cumsum,  # dA累积和
            chunk_states,  # 分块状态
            cu_seqlens,  # 累积序列长度
            states,  # 状态输出
            initial_states,  # 初始状态
            headdim,  # 头维度
            dstate,  # 状态维度
            chunk_size,  # 分块大小
            total_seqlen,  # 序列总长度
            nheads // ngroups,  # 头数与组数之比
            x.stride(0),  # x序列步幅
            x.stride(1),  # x头步幅
            x.stride(2),  # x头维度步幅
            B.stride(0),  # B序列步幅
            B.stride(1),  # B头步幅
            B.stride(2),  # B状态维度步幅
            dt.stride(1),  # dt分块步幅
            dt.stride(0),  # dt头步幅
            dt.stride(2),  # dt csize步幅
            dA_cumsum.stride(1),  # dA累积和分块步幅
            dA_cumsum.stride(0),  # dA累积和头步幅
            dA_cumsum.stride(2),  # dA累积和csize步幅
            chunk_states.stride(0),  # 分块状态分块步幅
            chunk_states.stride(1),  # 分块状态头步幅
            chunk_states.stride(2),  # 分块状态头维度步幅
            chunk_states.stride(3),  # 分块状态状态维度步幅
            states.stride(0),  # 状态批次步幅
            states.stride(1),  # 状态头步幅
            states.stride(2),  # 状态头维度步幅
            states.stride(3),  # 状态维度步幅
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
            HAS_INITSTATES=initial_states is not None,  # 是否有初始状态
        )
    return states  # 返回计算得到的状态
