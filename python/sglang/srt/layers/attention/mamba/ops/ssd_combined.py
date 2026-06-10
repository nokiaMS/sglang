# SSD组合计算模块 - 整合Mamba2状态空间模型的完整前向计算流水线
# 将分块累积和、分块状态、状态传递、批量矩阵乘法和分块扫描五个子步骤组合为完整计算

# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_combined.py
# 改编自: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_combined.py

# SPDX-License-Identifier: Apache-2.0
# SPDX许可证标识符: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX文件版权文本: vLLM项目的版权贡献者

# Copyright (c) 2024, Tri Dao, Albert Gu.
# 版权所有 (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py
# 改编自 https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# ruff: noqa: E501
# ruff: 忽略 E501(行长度)

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架
from einops import rearrange  # 导入einops张量重排库
from packaging import version  # 导入版本解析工具

from .ssd_bmm import _bmm_chunk_fwd  # 导入分块批量矩阵乘法前向函数
from .ssd_chunk_scan import _chunk_scan_fwd  # 导入分块扫描前向函数
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd, chunk_state_varlen  # 导入分块累积和、分块状态、变长状态函数
from .ssd_state_passing import _state_passing_fwd  # 导入状态传递前向函数

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")  # 判断Triton版本是否>=2.2.0


def is_int_pow_2(n):  # 判断整数n是否为2的幂次
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0  # 正整数且二进制只有一个1


def _mamba_chunk_scan_combined_fwd(  # Mamba分块扫描组合前向计算函数（内部使用）
    x,  # 输入张量
    dt,  # delta时间
    A,  # A矩阵
    B,  # B矩阵
    C,  # C矩阵
    chunk_size,  # 分块大小
    D=None,  # D向量（跳跃连接），可选
    z=None,  # z门控张量，可选
    dt_bias=None,  # dt偏置，可选
    initial_states=None,  # 初始状态，可选
    seq_idx=None,  # 序列索引，可选
    chunk_indices=None,  # 分块索引，可选
    chunk_offsets=None,  # 分块偏移，可选
    cu_seqlens=None,  # 累积序列长度，可选
    dt_softplus=False,  # 是否对dt应用softplus
    dt_limit=(0.0, float("inf")),  # dt范围限制
    state_dtype=None,  # 状态数据类型，可选
    out=None,  # 输出张量，可选
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"  # 断言分块大小必须是2的幂次
    batch, seqlen, nheads, headdim = x.shape  # 解包x的形状
    _, _, ngroups, dstate = B.shape  # 解包B的形状
    assert nheads % ngroups == 0  # 断言：头数必须能被组数整除
    assert B.shape == (batch, seqlen, ngroups, dstate)  # 断言B的形状
    assert dt.shape == (batch, seqlen, nheads)  # 断言dt的形状
    assert A.shape == (nheads,)  # 断言A的形状
    assert C.shape == B.shape  # 断言C与B形状相同
    if z is not None:  # 如果z存在
        assert z.shape == x.shape  # 断言z与x形状相同
    if D is not None:  # 如果D存在
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)  # 断言D的形状
    if seq_idx is not None:  # 如果seq_idx存在
        assert seq_idx.shape == (batch, seqlen)  # 断言seq_idx的形状
    if B.stride(-1) != 1:  # 如果B的最后一个维度不连续
        B = B.contiguous()  # 使B内存连续
    if C.stride(-1) != 1:  # 如果C的最后一个维度不连续
        C = C.contiguous()  # 使C内存连续
    if (  # 如果x的M和K维度都不连续
        x.stride(-1) != 1 and x.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
    # M或K维度应该是连续的
        x = x.contiguous()  # 使x内存连续
    if (  # 如果z存在且M和K维度都不连续
        z is not None and z.stride(-1) != 1 and z.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
    # M或K维度应该是连续的
        z = z.contiguous()  # 使z内存连续
    if D is not None and D.stride(-1) != 1:  # 如果D存在且不连续
        D = D.contiguous()  # 使D内存连续
    if initial_states is not None:  # 如果有初始状态
        if cu_seqlens is None:  # 如果没有累积序列长度（非连续批处理）
            assert initial_states.shape == (batch, nheads, headdim, dstate)  # 断言初始状态形状
        else:  # 有累积序列长度（连续批处理）
            assert initial_states.shape == (  # 断言初始状态形状
                len(cu_seqlens) - 1,  # 批次大小
                nheads,  # 头数
                headdim,  # 头维度
                dstate,  # 状态维度
            )

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below
    # 此函数执行5个子函数来计算mamba
    # - 一个好的参考资源是博客 https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   其中有一个最小实现来理解以下操作
    # - 如博客所述，mamba是因果注意力的特殊情况
    # - 核心思想是将注意力矩阵分块，并使用不同优化分别计算每个子矩阵
    # - 参见博客和论文中我们在以下注释中提到的子矩阵可视化

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    # 1. 计算A * dt的分块累积和
    # - 此处dt可能会经过softplus激活
    dA_cumsum, dt = _chunk_cumsum_fwd(  # 计算dA累积和和处理后的dt
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit  # 传入参数
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    # 2. 计算每个块内的状态
    # （对角线外块的低秩分解的右项；B项）
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)  # 计算块内状态

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states
    #   ii) seq_idx iii) is_cont_batching and (iv) chunk_offsets to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    # - We will also make sure that the dA_cumsum is taken only from the start of the
    #   sequence (hence we need the full dA_cumsum tensor and not just the values at chunk boundaries)
    # - this will ensure that states will be updated with the rightmost flushed seq_idx
    #   of the previous chunk. This implies that the first chunk of states is either 0
    #   or equal to init_states of the first example.
    # 3. 计算块间SSM递推；在分块边界处生成正确的SSM状态
    # （对角线外块分解的中间项；A项）
    # - 为处理分块预填充，需要指定 i) initial_states ii) seq_idx iii) is_cont_batching iv) chunk_offsets
    # - 当检测到新的seq_idx时，停止传递prev_state并切换到对应新seq_idx的init_state
    # - 还需确保dA_cumsum只从序列起始处计算（因此需要完整的dA_cumsum张量，而非仅分块边界处的值）
    # - 这确保状态会用前一个分块最右侧已刷新的seq_idx更新。这意味着第一个分块的状态为0
    #   或等于第一个样本的init_states。
    states, final_states = _state_passing_fwd(  # 计算块间状态传递
        rearrange(states, "... p n -> ... (p n)"),  # 将状态展平最后两个维度
        dA_cumsum,  # dA累积和
        initial_states=(  # 初始状态
            rearrange(initial_states, "... p n -> ... (p n)")  # 展平初始状态
            if initial_states is not None  # 如果初始状态存在
            else None  # 否则为None
        ),
        seq_idx=seq_idx,  # 序列索引
        chunk_size=chunk_size,  # 分块大小
        out_dtype=state_dtype if state_dtype is not None else C.dtype,  # 输出数据类型
        is_cont_batching=cu_seqlens is not None,  # 是否连续批处理
        chunk_offsets=chunk_offsets,  # 分块偏移
    )
    states, final_states = (  # 将状态恢复为原始形状
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]  # 展开的维度重新分开
    )

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    # 4. 计算C_j^T * B_i项的批量矩阵乘法
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)  # 计算CB块矩阵

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    # 5. 扫描并计算对角块，考虑过去的因果状态
    # - 如果提供了初始状态，状态信息将与初始状态合并
    # - 为正确处理，需要考虑连续批处理中的样本变化，因此引入伪分块，
    #   即每次样本变化时分割的分块
    # - 在每个（伪）分块中，检测前一个（伪）分块是否有seq_idx变化，
    #   如果有则从init_states获取状态信息
    out_x = _chunk_scan_fwd(  # 执行分块扫描前向计算
        CB,  # CB矩阵
        x,  # 输入x
        dt,  # dt
        dA_cumsum,  # dA累积和
        C,  # C矩阵
        states,  # 状态
        D=D,  # D向量
        z=z,  # z门控
        seq_idx=seq_idx,  # 序列索引
        chunk_indices=chunk_indices,  # 分块索引
        chunk_offsets=chunk_offsets,  # 分块偏移
        initial_states=initial_states,  # 初始状态
        out=out,  # 输出张量
    )
    if cu_seqlens is None:  # 如果没有累积序列长度
        return out_x, dt, dA_cumsum, states, final_states  # 返回常规输出
    else:  # 有累积序列长度
        assert (  # 断言batch=1
            batch == 1
        ), "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"  # 错误信息
        varlen_states = chunk_state_varlen(  # 计算变长序列状态
            B.squeeze(0),  # 移除批次维度
            x.squeeze(0),  # 移除批次维度
            dt.squeeze(0),  # 移除批次维度
            dA_cumsum.squeeze(0),  # 移除批次维度
            cu_seqlens,  # 累积序列长度
            states.squeeze(0),  # 移除批次维度的状态
            initial_states=initial_states,  # 初始状态
        )
        return out_x, dt, dA_cumsum, states, final_states, varlen_states  # 返回包含变长状态的完整输出


def mamba_chunk_scan_combined(  # Mamba分块扫描组合计算的公共接口函数
    x,  # 输入张量
    dt,  # delta时间
    A,  # A矩阵
    B,  # B矩阵
    C,  # C矩阵
    chunk_size,  # 分块大小
    D=None,  # D向量（跳跃连接），可选
    z=None,  # z门控张量，可选
    dt_bias=None,  # dt偏置，可选
    initial_states=None,  # 初始状态，可选
    seq_idx=None,  # 序列索引，可选
    chunk_indices=None,  # 分块索引，可选
    chunk_offsets=None,  # 分块偏移，可选
    cu_seqlens=None,  # 累积序列长度，可选
    dt_softplus=False,  # 是否对dt应用softplus
    dt_limit=(0.0, float("inf")),  # dt范围限制
    out=None,  # 输出张量，可选
    return_final_states=False,  # 是否返回最终状态
    return_varlen_states=False,  # 是否返回变长序列状态
    return_intermediate_states=False,  # 是否返回中间状态
    state_dtype=None,  # 状态数据类型，可选
):
    """
    Argument:
    参数:
        x: (batch, seqlen, nheads, headdim)
        x: 输入张量 (批次, 序列长度, 头数, 头维度)
        dt: (batch, seqlen, nheads)
        dt: delta时间 (批次, 序列长度, 头数)
        A: (nheads)
        A: A矩阵 (头数)
        B: (batch, seqlen, ngroups, dstate)
        B: B矩阵 (批次, 序列长度, 组数, 状态维度)
        C: (batch, seqlen, ngroups, dstate)
        C: C矩阵 (批次, 序列长度, 组数, 状态维度)
        chunk_size: int
        chunk_size: 分块大小 (整数)
        D: (nheads, headdim) or (nheads,)
        D: D向量 (头数, 头维度) 或 (头数)
        z: (batch, seqlen, nheads, headdim)
        z: z门控张量 (批次, 序列长度, 头数, 头维度)
        dt_bias: (nheads,)
        dt_bias: dt偏置 (头数)
        initial_states: (batch, nheads, headdim, dstate)
        initial_states: 初始状态 (批次, 头数, 头维度, 状态维度)
        seq_idx: (batch, seqlen)
        seq_idx: 序列索引 (批次, 序列长度)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        cu_seqlens: 累积序列长度 (序列数+1) 或 None，仅在return_varlen_states为True时使用
        dt_softplus: Whether to apply softplus to dt
        dt_softplus: 是否对dt应用softplus
        out: Preallocated output tensor
        out: 预分配的输出张量
        state_dtype: The data type of the ssm state
        state_dtype: SSM状态的数据类型
    """

    if not return_varlen_states:  # 如果不需要返回变长状态
        cu_seqlens = None  # 置空cu_seqlens
    else:  # 需要返回变长状态
        assert (  # 断言cu_seqlens必须提供
            cu_seqlens is not None
        ), "cu_seqlens must be provided if return_varlen_states is True"  # 错误信息
    out_x, dt_out, dA_cumsum, states, final_states, *rest = (  # 调用内部前向计算函数
        _mamba_chunk_scan_combined_fwd(  # 内部前向计算
            x,  # 输入x
            dt,  # delta时间
            A,  # A矩阵
            B,  # B矩阵
            C,  # C矩阵
            chunk_size,  # 分块大小
            D=D,  # D向量
            z=z,  # z门控
            dt_bias=dt_bias,  # dt偏置
            initial_states=initial_states,  # 初始状态
            seq_idx=seq_idx,  # 序列索引
            chunk_indices=chunk_indices,  # 分块索引
            chunk_offsets=chunk_offsets,  # 分块偏移
            cu_seqlens=cu_seqlens,  # 累积序列长度
            dt_softplus=dt_softplus,  # softplus标志
            dt_limit=dt_limit,  # dt范围
            out=out,  # 输出张量
            state_dtype=state_dtype,  # 状态数据类型
        )
    )
    if return_intermediate_states:  # 如果需要返回中间状态
        if return_varlen_states:  # 如果需要变长状态
            varlen_states = rest[0]  # 获取变长状态
            if return_final_states:  # 如果还需要最终状态
                return states, final_states, varlen_states  # 返回中间状态、最终状态、变长状态
            else:  # 不需要最终状态
                return states, varlen_states  # 返回中间状态、变长状态
        else:  # 不需要变长状态
            if return_final_states:  # 如果需要最终状态
                return states, final_states  # 返回中间状态、最终状态
            else:  # 不需要最终状态
                return states  # 返回中间状态

    if not return_varlen_states:  # 如果不需要变长状态
        if not return_final_states:  # 不需要最终状态
            return  # 无返回值（输出已写入out张量）
        else:  # 需要最终状态
            return final_states  # 返回最终状态
    else:  # 需要变长状态
        varlen_states = rest[0]  # 获取变长状态
        return (  # 根据是否需要最终状态返回
            (varlen_states)  # 仅返回变长状态
            if not return_final_states  # 不需要最终状态
            else (final_states, varlen_states)  # 返回最终状态和变长状态
        )
