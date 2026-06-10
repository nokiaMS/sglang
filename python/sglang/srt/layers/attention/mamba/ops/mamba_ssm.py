# Mamba选择性状态更新SSM内核实现
# 该模块实现了Mamba模型中选择性扫描状态更新的Triton GPU内核，
# 支持多token处理、中间状态缓存、EAGLE树注意力掩码等功能，
# 用于SSM（状态空间模型）的增量推理。

# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/mamba_ssm.py  # 改编自vLLM项目的Mamba SSM操作

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明

# Copyright (c) 2024, Tri Dao, Albert Gu.  # 版权声明
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py  # 改编自Mamba项目的选择性状态更新实现

import torch  # 导入PyTorch库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言并简写为tl
from packaging import version  # 导入版本解析工具

PAD_SLOT_ID = -1  # 填充槽ID，用于标识无效的缓存槽位

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")  # 判断Triton版本是否>=3.0.0

if TRITON3:  # 如果是Triton 3.x版本

    @triton.jit  # Triton JIT编译装饰器
    def softplus(dt):  # softplus激活函数（Triton 3.x版本）
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)  # 当dt<=20时计算log(exp(dt)+1)，否则直接使用dt
        return dt  # 返回结果

else:  # 如果是Triton 2.x版本

    @triton.jit  # Triton JIT编译装饰器
    def softplus(dt):  # softplus激活函数（Triton 2.x版本）
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)  # 当dt<=20时使用log1p(exp(dt))提高数值精度，否则直接使用dt
        return dt  # 返回结果


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})  # 启发式：判断是否有时间步偏置
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})  # 启发式：判断是否有D跳跃连接参数
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})  # 启发式：判断是否有门控分支Z
@triton.heuristics(  # 启发式：判断是否有状态批次索引
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(  # 启发式：计算状态维度的块大小（2的幂次）
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(  # 启发式：判断是否缓存中间状态
    {
        "CACHE_INTERMEDIATE_STATES": lambda args: args["intermediate_states_buffer"]
        is not None
    }
)
@triton.heuristics(  # 启发式：判断是否有EAGLE树自定义注意力掩码
    {
        "HAS_EAGLE_TREE_CUSTOM_ATTN_MASK": lambda args: args[
            "retrieve_parent_token_ptr"
        ]
        is not None
    }
)
@triton.heuristics(  # 启发式：判断是否有中间状态索引
    {
        "HAS_INTERMEDIATE_STATE_INDICES": lambda args: args[
            "intermediate_state_indices_ptr"
        ]
        is not None
    }
)
@triton.jit(do_not_specialize=["T"])  # Triton JIT编译，不对T参数特化
def _selective_scan_update_kernel(  # 选择性扫描状态更新内核函数
    # Pointers to matrices  # 矩阵指针
    state_ptr,  # SSM状态指针
    x_ptr,  # 输入指针
    dt_ptr,  # 时间步指针
    dt_bias_ptr,  # 时间步偏置指针
    A_ptr,  # 状态转移矩阵A指针
    B_ptr,  # 输入矩阵B指针
    C_ptr,  # 输出矩阵C指针
    D_ptr,  # 跳跃连接D指针
    z_ptr,  # 门控分支指针
    out_ptr,  # 输出指针
    state_batch_indices_ptr,  # 状态批次索引指针
    pad_slot_id,  # 填充槽ID
    intermediate_states_buffer,  # 中间状态缓存缓冲区
    cache_steps,  # 缓存步数
    retrieve_parent_token_ptr,  # 检索父token指针
    intermediate_state_indices_ptr,  # 中间状态索引指针
    # Matrix dimensions  # 矩阵维度
    batch,  # 批次大小
    T,  # 时间步数
    nheads,  # 注意力头数
    dim,  # 隐藏维度
    dstate,  # 状态维度
    nheads_ngroups_ratio,  # 头数与组数的比率
    # Strides  # 步长
    stride_state_batch,  # 状态批次步长
    stride_state_head,  # 状态头步长
    stride_state_dim,  # 状态维度步长
    stride_state_dstate,  # 状态dstate步长
    stride_x_batch,  # 输入批次步长
    stride_x_T,  # 输入时间步长
    stride_x_head,  # 输入头步长
    stride_x_dim,  # 输入维度步长
    stride_dt_batch,  # 时间步批次步长
    stride_dt_T,  # 时间步的时间步长
    stride_dt_head,  # 时间步头步长
    stride_dt_dim,  # 时间步维度步长
    stride_dt_bias_head,  # 时间步偏置头步长
    stride_dt_bias_dim,  # 时间步偏置维度步长
    stride_A_head,  # A矩阵头步长
    stride_A_dim,  # A矩阵维度步长
    stride_A_dstate,  # A矩阵dstate步长
    stride_B_batch,  # B矩阵批次步长
    stride_B_T,  # B矩阵时间步长
    stride_B_group,  # B矩阵组步长
    stride_B_dstate,  # B矩阵dstate步长
    stride_C_batch,  # C矩阵批次步长
    stride_C_T,  # C矩阵时间步长
    stride_C_group,  # C矩阵组步长
    stride_C_dstate,  # C矩阵dstate步长
    stride_D_head,  # D矩阵头步长
    stride_D_dim,  # D矩阵维度步长
    stride_z_batch,  # 门控批次步长
    stride_z_T,  # 门控时间步长
    stride_z_head,  # 门控头步长
    stride_z_dim,  # 门控维度步长
    stride_out_batch,  # 输出批次步长
    stride_out_T,  # 输出时间步长
    stride_out_head,  # 输出头步长
    stride_out_dim,  # 输出维度步长
    stride_retrieve_parent_token_batch,  # 父token检索批次步长
    stride_retrieve_parent_token_T,  # 父token检索时间步长
    # Meta-parameters  # 元参数
    DT_SOFTPLUS: tl.constexpr,  # 是否对dt应用softplus（编译时常量）
    TIE_HDIM: tl.constexpr,  # 是否绑定隐藏维度（编译时常量）
    BLOCK_SIZE_M: tl.constexpr,  # M维度块大小（编译时常量）
    HAS_DT_BIAS: tl.constexpr,  # 是否有时间步偏置（编译时常量）
    HAS_D: tl.constexpr,  # 是否有D跳跃连接（编译时常量）
    HAS_Z: tl.constexpr,  # 是否有门控分支（编译时常量）
    HAS_STATE_BATCH_INDICES: tl.constexpr,  # 是否有状态批次索引（编译时常量）
    DISABLE_STATE_UPDATE: tl.constexpr,  # 是否禁用状态更新（编译时常量）
    CACHE_INTERMEDIATE_STATES: tl.constexpr,  # 是否缓存中间状态（编译时常量）
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,  # 是否有EAGLE树注意力掩码（编译时常量）
    HAS_INTERMEDIATE_STATE_INDICES: tl.constexpr,  # 是否有中间状态索引（编译时常量）
    BLOCK_SIZE_DSTATE: tl.constexpr,  # 状态维度块大小（编译时常量）
):
    pid_m = tl.program_id(axis=0)  # 获取第0轴程序ID（维度方向的块索引）
    pid_b = tl.program_id(axis=1)  # 获取第1轴程序ID（批次索引）
    pid_h = tl.program_id(axis=2)  # 获取第2轴程序ID（头索引）

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate  # 如果HAS_STATE_BATCH_INDICES为真，则SSM状态的批次坐标
    # is taken from the state_batch_indices_ptr Otherwise, the state coordinate  # 从state_batch_indices_ptr获取；否则，状态坐标
    # is the same as the batch id.  # 与批次ID相同
    if HAS_STATE_BATCH_INDICES:  # 如果有状态批次索引
        state_batch_indices_ptr += pid_b  # 偏移到当前批次的索引位置
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)  # 加载状态批次索引
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head  # 计算状态指针偏移
    else:  # 没有状态批次索引
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head  # 使用批次ID直接偏移

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head  # 计算输入指针偏移
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head  # 计算时间步指针偏移
    if HAS_DT_BIAS:  # 如果有时间步偏置
        dt_bias_ptr += pid_h * stride_dt_bias_head  # 计算时间步偏置指针偏移
    A_ptr += pid_h * stride_A_head  # 计算A矩阵指针偏移
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group  # 计算B矩阵指针偏移
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group  # 计算C矩阵指针偏移
    if HAS_Z:  # 如果有门控分支
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head  # 计算门控指针偏移
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head  # 计算输出指针偏移

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 计算M维度偏移量
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)  # 计算N维度（状态维度）偏移量
    state_ptrs = state_ptr + (  # 计算状态指针地址
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)  # 创建掩码，防止越界访问
    if HAS_STATE_BATCH_INDICES:  # 如果有状态批次索引
        mask &= state_batch_idx != pad_slot_id  # 排除填充槽位
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)  # 加载状态数据，越界位置填充0.0

    if HAS_DT_BIAS:  # 如果有时间步偏置
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim  # 计算时间步偏置指针地址
    if HAS_D:  # 如果有D跳跃连接
        D_ptr += pid_h * stride_D_head  # 计算D矩阵指针偏移
        D_ptrs = D_ptr + offs_m * stride_D_dim  # 计算D矩阵元素指针地址
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate  # 计算A矩阵指针地址

    cache_idx = -1  # 初始化缓存索引为-1
    if CACHE_INTERMEDIATE_STATES:  # 如果需要缓存中间状态
        if HAS_INTERMEDIATE_STATE_INDICES:  # 如果有中间状态索引
            intermediate_state_idx = tl.load(intermediate_state_indices_ptr + pid_b).to(  # 加载中间状态索引
                tl.int64
            )
            cache_idx = intermediate_state_idx  # 使用中间状态索引作为缓存索引
        elif HAS_STATE_BATCH_INDICES:  # 如果有状态批次索引
            cache_idx = state_batch_idx  # 使用状态批次索引作为缓存索引
        else:  # 其他情况
            cache_idx = pid_b  # 使用批次ID作为缓存索引

    current_step_idx = 0  # 初始化当前步索引为0
    for _ in range(T):  # 遍历每个时间步
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:  # 如果有EAGLE树自定义注意力掩码
            if current_step_idx != 0 and cache_idx >= 0:  # 非首步且缓存索引有效
                parent_ptr = (  # 计算父token指针地址
                    retrieve_parent_token_ptr
                    + pid_b * stride_retrieve_parent_token_batch
                    + current_step_idx * stride_retrieve_parent_token_T
                )
                parent_step_idx = tl.load(parent_ptr).to(tl.int32)  # 加载父token步索引

                if parent_step_idx >= 0 and parent_step_idx < T:  # 父步索引有效
                    step_offset = parent_step_idx * nheads * dim * dstate  # 计算步偏移量
                    cache_ptr = (  # 计算缓存指针地址
                        intermediate_states_buffer
                        + cache_idx * cache_steps * nheads * dim * dstate
                        + step_offset
                        + pid_h * dim * dstate
                        + offs_m[:, None] * dstate
                        + offs_n[None, :]
                    )
                    state = tl.load(cache_ptr, mask=mask, other=0.0).to(tl.float32)  # 从缓存加载状态

        x_ptrs = x_ptr + offs_m * stride_x_dim  # 计算输入元素指针地址
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim  # 计算时间步元素指针地址
        B_ptrs = B_ptr + offs_n * stride_B_dstate  # 计算B矩阵元素指针地址
        C_ptrs = C_ptr + offs_n * stride_C_dstate  # 计算C矩阵元素指针地址
        if HAS_Z:  # 如果有门控分支
            z_ptrs = z_ptr + offs_m * stride_z_dim  # 计算门控元素指针地址
        out_ptrs = out_ptr + offs_m * stride_out_dim  # 计算输出元素指针地址

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)  # 加载当前时间步的输入
        if not TIE_HDIM:  # 如果未绑定隐藏维度
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)  # 加载时间步
            if HAS_DT_BIAS:  # 如果有时间步偏置
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)  # 加上偏置
            if DT_SOFTPLUS:  # 如果需要对dt应用softplus
                dt = softplus(dt)  # 应用softplus激活
            A = tl.load(  # 加载A矩阵元素
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dA = tl.exp(A * dt[:, None])  # 计算dA = exp(A * dt)
        else:  # 如果绑定了隐藏维度
            dt = tl.load(dt_ptr).to(tl.float32)  # 加载标量时间步
            if HAS_DT_BIAS:  # 如果有时间步偏置
                dt += tl.load(dt_bias_ptr).to(tl.float32)  # 加上标量偏置
            if DT_SOFTPLUS:  # 如果需要对dt应用softplus
                dt = softplus(dt)  # 应用softplus激活
            A = tl.load(A_ptr).to(tl.float32)  # 加载标量A
            dA = tl.exp(A * dt)  # scalar, not a matrix  # 计算标量dA = exp(A * dt)，不是矩阵

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)  # 加载B矩阵元素
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)  # 加载C矩阵元素
        if HAS_D:  # 如果有D跳跃连接
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)  # 加载D矩阵元素
        if HAS_Z:  # 如果有门控分支
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)  # 加载门控元素

        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt  # 计算dB = B * dt
        state = state * dA + dB * x[:, None]  # 更新状态：state = state * dA + dB * x

        if CACHE_INTERMEDIATE_STATES:  # 如果需要缓存中间状态
            if HAS_STATE_BATCH_INDICES:  # 如果有状态批次索引
                if state_batch_idx != pad_slot_id:  # 非填充槽位才缓存
                    cache_ptr_base = (  # 计算缓存基地址
                        intermediate_states_buffer
                        + cache_idx * cache_steps * nheads * dim * dstate
                        + current_step_idx * nheads * dim * dstate
                        + pid_h * dim * dstate
                    )
                    cache_ptrs = cache_ptr_base + (  # 计算缓存元素地址
                        offs_m[:, None] * dstate + offs_n[None, :]
                    )
                    tl.store(  # 存储中间状态到缓存
                        cache_ptrs, state.to(cache_ptrs.dtype.element_ty), mask=mask
                    )

        out = tl.sum(state * C[None, :], axis=1)  # 计算输出：out = sum(state * C)
        if HAS_D:  # 如果有D跳跃连接
            out += x * D  # 加上跳跃连接：out += x * D
        if HAS_Z:  # 如果有门控分支
            out *= z * tl.sigmoid(z)  # 应用SiLU门控：out *= silu(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)  # 存储输出

        current_step_idx += 1  # 递增当前步索引

        x_ptr += stride_x_T  # 移动输入指针到下一个时间步
        dt_ptr += stride_dt_T  # 移动时间步指针到下一个时间步
        B_ptr += stride_B_T  # 移动B矩阵指针到下一个时间步
        C_ptr += stride_C_T  # 移动C矩阵指针到下一个时间步
        out_ptr += stride_out_T  # 移动输出指针到下一个时间步
        if HAS_Z:  # 如果有门控分支
            z_ptr += stride_z_T  # 移动门控指针到下一个时间步

    if not DISABLE_STATE_UPDATE:  # 如果未禁用状态更新
        tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)  # 将最终状态写回


def selective_state_update(  # 选择性状态更新主机函数
    state,  # SSM状态张量
    x,  # 输入张量
    dt,  # 时间步张量
    A,  # 状态转移矩阵A
    B,  # 输入矩阵B
    C,  # 输出矩阵C
    D=None,  # 跳跃连接D，默认为None
    z=None,  # 门控分支，默认为None
    dt_bias=None,  # 时间步偏置，默认为None
    dt_softplus=False,  # 是否对dt应用softplus，默认为False
    state_batch_indices=None,  # 状态批次索引，默认为None
    pad_slot_id=PAD_SLOT_ID,  # 填充槽ID，默认为PAD_SLOT_ID
    out=None,  # 预分配输出张量，默认为None
    disable_state_update=False,  # 是否禁用状态更新，默认为False
    intermediate_states_buffer=None,  # 中间状态缓存缓冲区，默认为None
    cache_steps=None,  # 缓存步数，默认为None
    retrieve_parent_token=None,  # 父token检索张量，默认为None
    intermediate_state_indices=None,  # 中间状态索引，默认为None
):
    """
    Argument:  # 参数说明：
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)  # SSM状态张量
        x: (batch, dim) or (batch, nheads, dim) for single-token or (batch, T, nheads, dim) for multi-token  # 输入张量
        dt: (batch, dim) or (batch, nheads, dim)  # 时间步张量
        A: (dim, dstate) or (nheads, dim, dstate)  # 状态转移矩阵A
        B: (batch, dstate) or (batch, ngroups, dstate) for single-token or (batch, T, ngroups, dstate) for multi-token  # 输入矩阵B
        C: (batch, dstate) or (batch, ngroups, dstate)  # 输出矩阵C
        D: (dim,) or (nheads, dim)  # 跳跃连接D
        z: (batch, dim) or (batch, nheads, dim)  # 门控分支
        dt_bias: (dim,) or (nheads, dim)  # 时间步偏置
        pad_slot_id: int  # 填充槽ID（整数）
            if cache_indices is passed, lets the kernel identify padded  # 如果传入了cache_indices，让内核识别填充的
            entries that will not be processed,  # 不会被处理的条目，
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]  # 例如：cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at  # 在这种情况下，内核不会处理索引
            indices 0 and 3  # 0和3处的条目
        out: Preallocated ssm output tensor. Assume same shape as x.  # 预分配的SSM输出张量，假设与x形状相同
             In-place updated.  # 原地更新
        disable_state_update: If True, don't write back to state (for speculative verify)  # 如果为True，不回写状态（用于投机验证）
        intermediate_states_buffer: Buffer to cache intermediate states  # 用于缓存中间状态的缓冲区
        cache_steps: Total number of steps in the buffer  # 缓冲区中的总步数
        retrieve_parent_token: (batch, T) tensor of parent token indices for EAGLE tree attention  # EAGLE树注意力的父token索引张量(batch, T)
        intermediate_state_indices: (batch,) tensor of indices for intermediate_states_buffer operations.  # 中间状态缓冲区操作的索引张量(batch,)
            If provided, uses these indices instead of state_batch_indices for the buffer.  # 如果提供，使用这些索引而非state_batch_indices操作缓冲区
    """
    if state.dim() == 3:  # 如果状态是3维的
        state = state.unsqueeze(1)  # 在第1维添加头维度
    if x.dim() == 2:  # 如果输入是2维的
        x = x.unsqueeze(1)  # 添加头维度
    if x.dim() == 3:  # 如果输入是3维的
        x = x.unsqueeze(1)  # 添加时间步维度
    if dt.dim() == 2:  # 如果时间步是2维的
        dt = dt.unsqueeze(1)  # 添加头维度
    if dt.dim() == 3:  # 如果时间步是3维的
        dt = dt.unsqueeze(1)  # 添加时间步维度
    if A.dim() == 2:  # 如果A矩阵是2维的
        A = A.unsqueeze(0)  # 添加头维度
    if B.dim() == 2:  # 如果B矩阵是2维的
        B = B.unsqueeze(1)  # 添加组维度
    if B.dim() == 3:  # 如果B矩阵是3维的
        B = B.unsqueeze(1)  # 添加时间步维度
    if C.dim() == 2:  # 如果C矩阵是2维的
        C = C.unsqueeze(1)  # 添加组维度
    if C.dim() == 3:  # 如果C矩阵是3维的
        C = C.unsqueeze(1)  # 添加时间步维度
    if D is not None and D.dim() == 1:  # 如果D存在且为1维
        D = D.unsqueeze(0)  # 添加头维度
    if z is not None:  # 如果门控存在
        if z.dim() == 2:  # 如果门控是2维的
            z = z.unsqueeze(1)  # 添加头维度
        if z.dim() == 3:  # 如果门控是3维的
            z = z.unsqueeze(1)  # 添加时间步维度
    if dt_bias is not None and dt_bias.dim() == 1:  # 如果时间步偏置存在且为1维
        dt_bias = dt_bias.unsqueeze(0)  # 添加头维度
    if out.dim() == 2:  # 如果输出是2维的
        out = out.unsqueeze(1)  # 添加头维度
    if out.dim() == 3:  # 如果输出是3维的
        out = out.unsqueeze(1)  # 添加时间步维度

    _, nheads, dim, dstate = state.shape  # 获取状态的维度信息
    batch, T, _, _ = x.shape  # 获取输入的批次大小和时间步数

    assert x.shape == (batch, T, nheads, dim)  # 断言输入形状正确
    assert dt.shape == x.shape  # 断言时间步形状与输入相同
    assert A.shape == (nheads, dim, dstate)  # 断言A矩阵形状正确
    ngroups = B.shape[2]  # 获取组数
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"  # 断言头数能被组数整除
    assert B.shape == (batch, T, ngroups, dstate)  # 断言B矩阵形状正确
    assert C.shape == B.shape  # 断言C矩阵形状与B相同
    if D is not None:  # 如果D存在
        assert D.shape == (nheads, dim)  # 断言D形状正确
    if z is not None:  # 如果门控存在
        assert z.shape == x.shape  # 断言门控形状与输入相同
    if dt_bias is not None:  # 如果时间步偏置存在
        assert dt_bias.shape == (nheads, dim)  # 断言偏置形状正确
    if state_batch_indices is not None:  # 如果状态批次索引存在
        assert state_batch_indices.shape == (batch,)  # 断言索引形状正确
    assert out.shape == x.shape  # 断言输出形状与输入相同

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)  # 定义内核启动网格
    z_strides = (  # 获取门控步长
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))  # 有门控时的步长
        if z is not None  # 如果门控存在
        else (0, 0, 0, 0)  # 无门控时步长为0
    )
    # We don't want autotune since it will overwrite the state  # 我们不希望自动调优，因为它会覆盖状态
    # We instead tune by hand.  # 我们改为手动调优。
    BLOCK_SIZE_M, num_warps = (  # 手动选择块大小和线程束数
        (32, 4)  # dstate <= 16时
        if dstate <= 16
        else (
            (16, 4)  # dstate <= 32时
            if dstate <= 32
            else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else ((4, 8))))  # 更大dstate时
        )
    )
    tie_hdim = (  # 判断是否绑定隐藏维度
        A.stride(-1) == 0  # A在最后维度步长为0
        and A.stride(-2) == 0  # A在倒数第二维度步长为0
        and dt.stride(-1) == 0  # dt在最后维度步长为0
        and dt_bias.stride(-1) == 0  # dt_bias在最后维度步长为0
    )

    retrieve_parent_token_strides = (  # 获取父token检索步长
        (retrieve_parent_token.stride(0), retrieve_parent_token.stride(1))  # 有父token检索时的步长
        if retrieve_parent_token is not None  # 如果父token检索存在
        else (0, 0)  # 无父token检索时步长为0
    )

    with torch.get_device_module(x.device).device(x.device.index):  # 在对应设备上执行
        _selective_scan_update_kernel[grid](  # 启动选择性扫描更新内核
            state,  # SSM状态
            x,  # 输入
            dt,  # 时间步
            dt_bias,  # 时间步偏置
            A,  # 状态转移矩阵A
            B,  # 输入矩阵B
            C,  # 输出矩阵C
            D,  # 跳跃连接D
            z,  # 门控分支
            out,  # 输出
            state_batch_indices,  # 状态批次索引
            pad_slot_id,  # 填充槽ID
            intermediate_states_buffer,  # 中间状态缓存缓冲区
            cache_steps if cache_steps is not None else 0,  # 缓存步数，None时传0
            retrieve_parent_token,  # 父token检索张量
            intermediate_state_indices,  # 中间状态索引
            batch,  # 批次大小
            T,  # 时间步数
            nheads,  # 注意力头数
            dim,  # 隐藏维度
            dstate,  # 状态维度
            nheads // ngroups,  # 头数与组数的比率
            state.stride(0),  # 状态批次步长
            state.stride(1),  # 状态头步长
            state.stride(2),  # 状态维度步长
            state.stride(3),  # 状态dstate步长
            x.stride(0),  # 输入批次步长
            x.stride(1),  # 输入时间步长
            x.stride(2),  # 输入头步长
            x.stride(3),  # 输入维度步长
            dt.stride(0),  # 时间步批次步长
            dt.stride(1),  # 时间步的时间步长
            dt.stride(2),  # 时间步头步长
            dt.stride(3),  # 时间步维度步长
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,  # 偏置步长
            A.stride(0),  # A矩阵头步长
            A.stride(1),  # A矩阵维度步长
            A.stride(2),  # A矩阵dstate步长
            B.stride(0),  # B矩阵批次步长
            B.stride(1),  # B矩阵时间步长
            B.stride(2),  # B矩阵组步长
            B.stride(3),  # B矩阵dstate步长
            C.stride(0),  # C矩阵批次步长
            C.stride(1),  # C矩阵时间步长
            C.stride(2),  # C矩阵组步长
            C.stride(3),  # C矩阵dstate步长
            *(D.stride(0), D.stride(1)) if D is not None else 0,  # D步长
            z_strides[0],  # 门控批次步长
            z_strides[1],  # 门控时间步长
            z_strides[2],  # 门控头步长
            z_strides[3],  # 门控维度步长
            out.stride(0),  # 输出批次步长
            out.stride(1),  # 输出时间步长
            out.stride(2),  # 输出头步长
            out.stride(3),  # 输出维度步长
            retrieve_parent_token_strides[0],  # 父token检索批次步长
            retrieve_parent_token_strides[1],  # 父token检索时间步长
            dt_softplus,  # 是否对dt应用softplus
            tie_hdim,  # 是否绑定隐藏维度
            BLOCK_SIZE_M,  # 块大小
            DISABLE_STATE_UPDATE=disable_state_update,  # 是否禁用状态更新
            num_warps=num_warps,  # 线程束数量
        )
