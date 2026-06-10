# 文件说明：预填充阶段的内存高效注意力Triton内核实现
# 本文件实现了预填充（prefill）阶段的内存高效注意力机制，支持page size = 1
# 基于lightllm的context_flashattention_nopad实现改编

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
Memory-efficient attention for prefill. # 预填充阶段的内存高效注意力
It supporst page size = 1. # 支持page size = 1
"""

# Adapted from # 改编自
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch  # 导入PyTorch库 # 深度学习框架
import triton  # 导入Triton库 # GPU内核编写框架
import triton.language as tl  # 导入Triton语言并别名为tl # Triton编程语言

from sglang.srt.utils import is_cuda, is_hip  # 导入CUDA和HIP检测函数 # 平台检测工具

_is_cuda = is_cuda()  # 检测当前是否为CUDA平台 # CUDA平台标志
_is_hip = is_hip()  # 检测当前是否为HIP(AMD)平台 # HIP平台标志

if _is_cuda or _is_hip:  # 如果是CUDA或HIP平台 # 平台判断
    CUDA_CAPABILITY = torch.cuda.get_device_capability()  # 获取GPU计算能力版本 # 读取GPU算力


@triton.jit  # Triton JIT编译装饰器 # 将函数编译为GPU内核
def _fwd_kernel(  # 前向计算内核函数 # 预填充注意力的前向计算内核
    Q,  # 查询张量 # Query张量
    K,  # 键张量 # Key张量
    V,  # 值张量 # Value张量
    sm_scale,  # softmax缩放因子 # softmax缩放系数
    B_Start_Loc,  # 批次起始位置数组 # 每个batch在扁平化序列中的起始位置
    B_Seqlen,  # 批次序列长度数组 # 每个batch的序列长度
    Out,  # 输出张量 # 注意力输出
    stride_qbs,  # Q的batch stride # Q张量在batch维度的步长
    stride_qh,  # Q的head stride # Q张量在head维度的步长
    stride_kbs,  # K的batch stride # K张量在batch维度的步长
    stride_kh,  # K的head stride # K张量在head维度的步长
    stride_vbs,  # V的batch stride # V张量在batch维度的步长
    stride_vh,  # V的head stride # V张量在head维度的步长
    stride_obs,  # Out的batch stride # 输出张量在batch维度的步长
    stride_oh,  # Out的head stride # 输出张量在head维度的步长
    kv_group_num: tl.constexpr,  # KV分组数（GQA中的组数） # GQA分组数
    BLOCK_M: tl.constexpr,  # M维度块大小 # Q序列分块大小
    BLOCK_DMODEL: tl.constexpr,  # 模型维度块大小 # 头维度分块大小
    BLOCK_N: tl.constexpr,  # N维度块大小 # K/V序列分块大小
    IS_CAUSAL: tl.constexpr,  # 是否使用因果掩码 # 因果注意力标志
    Lk: tl.constexpr,  # Key的实际维度长度 # Key维度长度
):
    cur_batch = tl.program_id(0)  # 获取当前batch索引 # 当前batch ID
    cur_head = tl.program_id(1)  # 获取当前head索引 # 当前head ID
    start_m = tl.program_id(2)  # 获取当前M块起始索引 # 当前Q序列块起始位置

    cur_kv_head = cur_head // kv_group_num  # 计算对应的KV头索引 # GQA中Q头到KV头的映射

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)  # 加载当前batch的序列长度 # 读取当前batch序列长度
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)  # 加载当前batch在扁平序列中的起始索引 # 读取起始位置

    block_start_loc = BLOCK_M * start_m  # 计算当前块的起始位置 # 当前Q块在序列中的起始位置

    # initialize offsets # 初始化偏移量 # 初始化各维度的索引偏移
    offs_n = tl.arange(0, BLOCK_N)  # N维度（KV序列）偏移量 # K/V序列维度索引
    offs_d = tl.arange(0, BLOCK_DMODEL)  # D维度（头维度）偏移量 # 头维度索引
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # M维度（Q序列）偏移量 # Q序列维度索引
    off_q = (  # 计算Q的内存偏移 # Q张量的内存地址偏移
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs  # batch维度偏移 # batch步长偏移
        + cur_head * stride_qh  # head维度偏移 # 头步长偏移
        + offs_d[None, :]  # dim维度偏移 # 维度步长偏移
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]  # 计算K的内存偏移 # K张量的内存地址偏移
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]  # 计算V的内存偏移 # V张量的内存地址偏移

    mask_d = offs_d < Lk  # 头维度有效掩码 # 遮蔽超出实际维度的部分

    q = tl.load(  # 加载Q数据 # 读取查询数据
        Q + off_q,  # Q的内存地址 # Q数据地址
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),  # 有效性掩码 # 遮蔽无效位置
        other=0.0,  # 无效位置填充0 # 无效位置填充值
    )

    k_ptrs = K + off_k  # K的指针基地址 # K数据基地址
    v_ptrs = V + off_v  # V的指针基地址 # V数据基地址

    # initialize pointer to m and l # 初始化m和l的指针 # 初始化在线softmax的中间变量
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 行最大值，初始化为负无穷 # 在线softmax最大值
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # 行求和，初始化为0 # 在线softmax归一化因子
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # 累加器，初始化为0 # 注意力加权累加器

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)  # 判断当前Q块是否在序列范围内 # 块有效性掩码

    end_n = (  # 计算KV序列的结束位置 # KV序列的遍历终点
        cur_batch_seq_len  # 非因果：遍历到序列末尾 # 非因果模式：遍历全部KV
        if not IS_CAUSAL  # 如果不是因果注意力 # 非因果模式
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)  # 因果：只遍历到Q位置 # 因果模式：只看当前位置之前的KV
    )
    for start_n in range(0, block_mask * end_n, BLOCK_N):  # 遍历KV序列块 # 按块遍历KV序列
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 提示编译器start_n是BLOCK_N的倍数 # 编译器优化提示
        # -- compute qk ---- # -- 计算QK点积 -- # 计算查询与键的点积
        k = tl.load(  # 加载K数据 # 读取键数据
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,  # 加上KV序列偏移 # K序列偏移地址
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),  # 有效性掩码 # 遮蔽无效位置
            other=0.0,  # 无效位置填充0 # 无效位置填充值
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0) # 加载掩码（已注释） # 额外掩码加载（未使用）

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # 初始化QK点积结果 # QK点积矩阵
        qk += tl.dot(q, k)  # 计算矩阵乘法 # Q和K的点积
        qk *= sm_scale  # 应用softmax缩放 # 缩放QK点积

        if IS_CAUSAL:  # 如果是因果注意力 # 因果模式处理
            qk += tl.where(  # 应用因果掩码 # 因果掩码：只允许看当前位置及之前
                (start_n + offs_n[None, :] < cur_batch_seq_len)  # 在序列范围内 # 序列有效性
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),  # Q位置>=KV位置（因果约束） # 因果约束条件
                0,  # 满足条件：不修改 # 允许的位置
                float("-inf"),  # 不满足条件：设为负无穷 # 禁止的位置
            )
        else:  # 非因果注意力 # 非因果模式处理
            qk += tl.where(  # 应用序列长度掩码 # 序列长度掩码
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")  # 超出序列的设为负无穷 # 遮蔽超出序列的位置
            )

        # -- compute m_ij, p, l_ij # -- 计算局部最大值、概率和局部求和 # 在线softmax的局部计算
        m_ij = tl.max(qk, 1)  # 计算当前块的行最大值 # 当前KV块的行最大值
        p = tl.exp(qk - m_ij[:, None])  # 计算指数概率（减去最大值保证数值稳定） # 减去最大值后取指数
        l_ij = tl.sum(p, 1)  # 计算当前块的行求和 # 当前KV块的概率和
        # -- update m_i and l_i # -- 更新全局最大值和求和 # 更新在线softmax状态
        m_i_new = tl.maximum(m_i, m_ij)  # 更新全局最大值 # 取旧最大值和新最大值的较大者
        alpha = tl.exp(m_i - m_i_new)  # 计算旧累加器的缩放因子 # 旧值的缩放系数
        beta = tl.exp(m_ij - m_i_new)  # 计算新值的缩放因子 # 新值的缩放系数
        l_i_new = alpha * l_i + beta * l_ij  # 更新全局求和 # 更新归一化因子
        # -- update output accumulator -- # -- 更新输出累加器 -- # 更新注意力输出
        # scale p # 缩放概率 # 缩放当前概率
        p_scale = beta / l_i_new  # 计算概率的缩放因子 # 当前块概率的缩放比
        p = p * p_scale[:, None]  # 缩放概率 # 应用缩放
        # scale acc # 缩放累加器 # 缩放已有累加结果
        acc_scale = l_i / l_i_new * alpha  # 计算累加器的缩放因子 # 历史累加结果的缩放比
        acc = acc * acc_scale[:, None]  # 缩放累加器 # 应用缩放
        # update acc # 更新累加器 # 累加新的加权值
        v = tl.load(  # 加载V数据 # 读取值数据
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,  # 加上KV序列偏移 # V序列偏移地址
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),  # 有效性掩码 # 遮蔽无效位置
            other=0.0,  # 无效位置填充0 # 无效位置填充值
        )

        p = p.to(v.dtype)  # 将概率转换为V的数据类型 # 类型转换以匹配V
        acc += tl.dot(p, v)  # 累加加权V值 # 加权求和
        # update m_i and l_i # 更新m_i和l_i # 更新在线softmax状态变量
        l_i = l_i_new  # 更新行求和 # 更新归一化因子
        m_i = m_i_new  # 更新行最大值 # 更新最大值
    # initialize pointers to output # 初始化输出指针 # 准备写入输出
    off_o = (  # 计算输出的内存偏移 # 输出张量的内存地址偏移
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs  # batch维度偏移 # batch步长偏移
        + cur_head * stride_oh  # head维度偏移 # 头步长偏移
        + offs_d[None, :]  # dim维度偏移 # 维度步长偏移
    )
    out_ptrs = Out + off_o  # 计算输出基地址 # 输出数据地址
    tl.store(  # 存储输出结果 # 写入注意力计算结果
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])  # 应用有效性掩码写入 # 仅写入有效位置
    )


def context_attention_fwd(  # 上下文注意力前向计算函数 # 预填充阶段的注意力前向计算入口
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True, sm_scale=None  # 输入参数 # Q/K/V/输出/起始位置/序列长度/最大长度/因果标志/缩放
):
    """
    q, k, v: [b * s, head, head_dim] # Q/K/V张量形状：[批次*序列长度, 头数, 头维度]
    b_start_loc: [b] # 每个batch的起始位置
    b_seq_len: [b] # 每个batch的序列长度
    out: [b * s, head, head_dim] # 输出形状：[批次*序列长度, 头数, 头维度]
    sm_scale: softmax scale, defaults to 1/sqrt(head_dim) # softmax缩放因子，默认为1/sqrt(头维度)
    """
    if (_is_cuda or _is_hip) and CUDA_CAPABILITY[0] > 8:  # 如果GPU计算能力大于8.x（如H100） # 高算力GPU
        BLOCK = 128  # 使用128的块大小 # 较大的分块大小
    else:
        BLOCK = 64  # 使用64的块大小 # 较小的分块大小

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]  # 获取Q/K/V的头维度 # 读取各张量的头维度

    if sm_scale is None:  # 如果未提供缩放因子 # 检查是否需要默认缩放
        sm_scale = 1.0 / (Lq**0.5)  # 默认使用1/sqrt(head_dim) # 默认缩放公式
    batch, head = b_seq_len.shape[0], q.shape[1]  # 获取batch大小和头数 # 批次大小和头数
    kv_group_num = q.shape[1] // k.shape[1]  # 计算KV分组数（GQA） # GQA中的组数

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # 设置内核启动网格 # 内核网格大小
    num_warps = 4 if Lk <= 64 else 8  # 根据头维度设置warp数量 # 较小维度用4个warp，较大用8个

    _fwd_kernel[grid](  # 启动前向计算内核 # 调用Triton内核
        q,  # 查询张量 # Query
        k,  # 键张量 # Key
        v,  # 值张量 # Value
        sm_scale,  # softmax缩放因子 # softmax缩放系数
        b_start_loc,  # 批次起始位置 # batch起始位置
        b_seq_len,  # 批次序列长度 # batch序列长度
        o,  # 输出张量 # 注意力输出
        q.stride(0),  # Q的batch步长 # Q在第0维的步长
        q.stride(1),  # Q的head步长 # Q在第1维的步长
        k.stride(0),  # K的batch步长 # K在第0维的步长
        k.stride(1),  # K的head步长 # K在第1维的步长
        v.stride(0),  # V的batch步长 # V在第0维的步长
        v.stride(1),  # V的head步长 # V在第1维的步长
        o.stride(0),  # Out的batch步长 # 输出在第0维的步长
        o.stride(1),  # Out的head步长 # 输出在第1维的步长
        kv_group_num=kv_group_num,  # KV分组数 # GQA分组数
        BLOCK_M=BLOCK,  # M维度块大小 # Q序列分块大小
        BLOCK_DMODEL=triton.next_power_of_2(Lk),  # 模型维度块大小（2的幂） # 对齐到2的幂的头维度分块
        BLOCK_N=BLOCK,  # N维度块大小 # KV序列分块大小
        IS_CAUSAL=is_causal,  # 是否因果注意力 # 因果注意力标志
        num_warps=num_warps,  # warp数量 # 线程束数量
        num_stages=1,  # 流水线阶段数 # 内核流水线级数
        Lk=Lk,  # Key维度长度 # Key维度
    )
