# SSD分块矩阵乘法Triton内核实现
# 该模块实现了Mamba SSD（状态空间对偶）中的分块矩阵乘法Triton GPU内核，
# 支持因果掩码、序列索引和分组计算，用于SSM的分块前向传播。

# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_bmm.py  # 改编自vLLM项目的SSD分块矩阵乘法实现

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明

# Copyright (c) 2024, Tri Dao, Albert Gu.  # 版权声明
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py  # 改编自Mamba项目的SSD BMM实现

# ruff: noqa: E501,SIM102  # ruff检查忽略行过长和if嵌套优化规则

import math  # 导入数学库

import torch  # 导入PyTorch库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言并简写为tl


@triton.jit  # Triton JIT编译装饰器
def _bmm_chunk_fwd_kernel(  # 分块矩阵乘法前向传播内核
    # Pointers to matrices  # 矩阵指针
    a_ptr,  # 矩阵A指针
    b_ptr,  # 矩阵B指针
    out_ptr,  # 输出指针
    seq_idx_ptr,  # 序列索引指针
    # Matrix dimensions  # 矩阵维度
    seqlen,  # 序列长度
    chunk_size,  # 分块大小
    K,  # 内积维度
    ngroups,  # 组数
    stride_a_batch,  # A的批次步长
    stride_a_seqlen,  # A的序列步长
    stride_a_head,  # A的头步长
    stride_ak,  # A的内积维度步长
    stride_b_batch,  # B的批次步长
    stride_b_seqlen,  # B的序列步长
    stride_b_head,  # B的头步长
    stride_bk,  # B的内积维度步长
    stride_out_batch,  # 输出的批次步长
    stride_out_chunk,  # 输出的分块步长
    stride_out_head,  # 输出的头步长
    stride_outm,  # 输出的M维度步长
    stride_outn,  # 输出的N维度步长
    stride_seq_idx_batch,  # 序列索引的批次步长
    stride_seq_idx_seqlen,  # 序列索引的序列步长
    # Meta-parameters  # 元参数
    IS_CAUSAL: tl.constexpr,  # 是否因果（编译时常量）
    dot_dtype: tl.constexpr,  # 点积数据类型（编译时常量）
    HAS_SEQ_IDX: tl.constexpr,  # 是否有序列索引（编译时常量）
    BLOCK_SIZE_M: tl.constexpr = 16,  # M维度块大小（编译时常量，默认16）
    BLOCK_SIZE_N: tl.constexpr = 16,  # N维度块大小（编译时常量，默认16）
    BLOCK_SIZE_K: tl.constexpr = 16,  # K维度块大小（编译时常量，默认16）
):
    pid_b = tl.program_id(axis=1)  # 获取批次维度的程序ID
    pid_ch = tl.program_id(axis=2).to(tl.int64)  # 获取分块和头组合的程序ID
    pid_c = pid_ch // ngroups  # 计算分块索引
    pid_h = pid_ch - pid_c * ngroups  # 计算头索引
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)  # 计算N维度块数
    pid_m = tl.program_id(axis=0) // num_pid_n  # 计算M维度块索引
    pid_n = tl.program_id(axis=0) % num_pid_n  # 计算N维度块索引
    if IS_CAUSAL:  # 如果是因果模式
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:  # 如果N块索引超出因果范围
            return  # 跳过计算（因果上三角被屏蔽）
    a_ptr += (  # 计算矩阵A的基地址偏移
        pid_b * stride_a_batch
        + pid_c * chunk_size * stride_a_seqlen
        + pid_h * stride_a_head
    )
    b_ptr += (  # 计算矩阵B的基地址偏移
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + pid_h * stride_b_head
    )
    if HAS_SEQ_IDX:  # 如果有序列索引
        seq_idx_ptr += (  # 计算序列索引的基地址偏移
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 计算M维度偏移量
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算N维度偏移量
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # 计算K维度偏移量
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)  # 计算A矩阵元素地址
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)  # 计算B矩阵元素地址
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)  # 计算当前分块的有效大小（最后一个分块可能不完整）

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器为零
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  # 沿K维度分块循环
        a = tl.load(  # 加载A矩阵块
            a_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)  # M维度掩码
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),  # K维度掩码
            other=0.0,  # 越界位置填充0
        ).to(dot_dtype)  # 转换为目标数据类型
        b = tl.load(  # 加载B矩阵块
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)  # K维度掩码
            & (offs_n[None, :] < chunk_size_limit),  # N维度掩码
            other=0.0,  # 越界位置填充0
        ).to(dot_dtype)  # 转换为目标数据类型
        acc += tl.dot(a, b)  # 计算点积并累加
        a_ptrs += BLOCK_SIZE_K * stride_ak  # 移动A指针到下一个K块
        b_ptrs += BLOCK_SIZE_K * stride_bk  # 移动B指针到下一个K块

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 重新计算M维度偏移量
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 重新计算N维度偏移量
    if HAS_SEQ_IDX:  # 如果有序列索引
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)  # 重新计算当前分块的有效大小
        seq_idx_m = tl.load(  # 加载M维度对应的序列索引
            seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit,  # 掩码
            other=-1,  # 越界填充-1
        )
        seq_idx_n = tl.load(  # 加载N维度对应的序列索引
            seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
            mask=offs_n < chunk_size_limit,  # 掩码
            other=-2,  # 越界填充-2
        )
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)  # 不同序列间的结果置零
    out = acc.to(out_ptr.dtype.element_ty)  # 将累加结果转换为目标输出类型

    out_ptr += (  # 计算输出的基地址偏移
        pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    )
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)  # 计算输出元素地址
    tl.store(  # 存储输出结果
        out_ptrs,
        out,  # 输出数据
        mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size),  # 掩码
    )


def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False, output_dtype=None):  # 分块矩阵乘法前向传播主机函数
    """
    Argument:  # 参数说明：
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)  # 矩阵A
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)  # 矩阵B
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.  # 序列索引，不同序列间结果置零
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are  # 是否因果模式
            guaranteed to be correct.  # 因果模式下只有i<=j的结果保证正确
    Return:  # 返回值说明：
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)  # 输出张量
    """
    # Check constraints.  # 检查约束条件
    has_groups = a.dim() == 4  # 判断是否有分组维度
    if not has_groups:  # 无分组
        batch, seqlen, k = a.shape  # 解析无分组时的形状
    else:  # 有分组
        batch, seqlen, ngroups, k = a.shape  # 解析有分组时的形状
    assert b.shape == a.shape  # 断言B的形状与A相同
    if seq_idx is not None:  # 如果有序列索引
        assert seq_idx.shape == (batch, seqlen)  # 断言序列索引形状正确
    if a.stride(-1) != 1 and a.stride(1) != 1:  # 如果A在最内层或序列维度不连续
        a = a.contiguous()  # 转为连续张量
    if b.stride(-1) != 1 and b.stride(1) != 1:  # 如果B在最内层或序列维度不连续
        b = b.contiguous()  # 转为连续张量
    nchunks = math.ceil(seqlen / chunk_size)  # 计算分块数
    # Allocates output.  # 分配输出
    out_dtype = a.dtype if output_dtype is None else output_dtype  # 确定输出数据类型
    out = torch.empty(  # 创建输出张量
        (
            (batch, nchunks, chunk_size, chunk_size)  # 无分组时的输出形状
            if not has_groups
            else (batch, nchunks, ngroups, chunk_size, chunk_size)  # 有分组时的输出形状
        ),
        device=a.device,  # 设备与输入相同
        dtype=out_dtype,  # 数据类型
    )
    dot_dtype = (  # 确定点积计算的数据类型
        tl.bfloat16  # 如果输入有bfloat16则用bfloat16
        if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16
        else (
            tl.float16  # 如果输入有float16则用float16
            if a.dtype == torch.float16 or b.dtype == torch.float16
            else tl.float32  # 否则用float32
        )
    )
    grid = lambda META: (  # 定义内核启动网格
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])  # M维度块数
        * triton.cdiv(chunk_size, META["BLOCK_SIZE_N"]),  # 乘以N维度块数
        batch,  # 批次维度
        nchunks if not has_groups else nchunks * ngroups,  # 分块维度（有分组时乘以组数）
    )
    with torch.get_device_module(a.device).device(a.device.index):  # 在对应设备上执行
        _bmm_chunk_fwd_kernel[grid](  # 启动分块矩阵乘法内核
            a,  # 矩阵A
            b,  # 矩阵B
            out,  # 输出
            seq_idx,  # 序列索引
            seqlen,  # 序列长度
            chunk_size,  # 分块大小
            k,  # 内积维度
            ngroups if has_groups else 1,  # 组数（无分组时为1）
            a.stride(0),  # A的批次步长
            a.stride(1),  # A的序列步长
            0 if not has_groups else a.stride(2),  # A的头步长
            a.stride(-1),  # A的内积维度步长
            b.stride(0),  # B的批次步长
            b.stride(1),  # B的序列步长
            0 if not has_groups else b.stride(2),  # B的头步长
            b.stride(-1),  # B的内积维度步长
            out.stride(0),  # 输出的批次步长
            out.stride(1),  # 输出的分块步长
            0 if not has_groups else out.stride(2),  # 输出的头步长
            out.stride(-2),  # 输出的M维度步长
            out.stride(-1),  # 输出的N维度步长
            *(  # 序列索引步长
                (seq_idx.stride(0), seq_idx.stride(1))  # 有序列索引时的步长
                if seq_idx is not None  # 如果序列索引存在
                else (0, 0)  # 无序列索引时步长为0
            ),
            causal,  # 是否因果模式
            dot_dtype,  # 点积数据类型
            HAS_SEQ_IDX=seq_idx is not None,  # 是否有序列索引
        )
    return out  # 返回输出张量
