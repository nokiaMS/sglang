# 融合Q/K RMSNorm的Triton内核实现
# 本文件将Query和Key的RMSNorm归一化操作融合到单个Triton内核启动中，
# 从而将每个注意力层的归一化内核启动次数减半。
# 移植自ATOM项目(atom/model_ops/layernorm.py)。

"""Fused Q/K RMSNorm in a single Triton kernel launch.
将Q和K的RMSNorm融合为单次Triton内核启动。

Ported from ATOM (atom/model_ops/layernorm.py). Fuses per-head Q RMSNorm
移植自ATOM(atom/model_ops/layernorm.py)。将逐头Q RMSNorm
(optionally weightless) and KV RMSNorm into one kernel, halving the number
(可选无权重)和KV RMSNorm融合到一个内核中，将
of norm kernel launches per attention layer.
每个注意力层的归一化内核启动次数减半。
"""

from typing import Optional, Tuple  # 导入可选类型和元组类型 # 导入可选类型和元组类型

import torch  # 导入PyTorch库 # 导入PyTorch库
import triton  # 导入Triton库 # 导入Triton库
import triton.language as tl  # 导入Triton语言模块并简写为tl # 导入Triton语言模块并简写为tl


# 融合Q/K RMSNorm的Triton JIT内核
@triton.jit
def _fused_qk_norm_kernel(  # 融合Q/K归一化内核函数
    q_ptr,  # Q张量的输入指针 # Q张量的输入指针
    k_ptr,  # K张量的输入指针 # K张量的输入指针
    q_out_ptr,  # Q归一化后的输出指针 # Q归一化后的输出指针
    k_out_ptr,  # K归一化后的输出指针 # K归一化后的输出指针
    q_weight_ptr,  # Q归一化的权重指针 # Q归一化的权重指针
    k_weight_ptr,  # K归一化的权重指针 # K归一化的权重指针
    eps,  # 用于数值稳定性的epsilon值 # 用于数值稳定性的epsilon值
    num_tokens,  # token数量 # token数量
    head_dim,  # 每个注意力头的维度 # 每个注意力头的维度
    q_in_stride0,  # Q输入的第0步长 # Q输入的第0步长
    k_in_stride0,  # K输入的第0步长 # K输入的第0步长
    q_out_stride0,  # Q输出的第0步长 # Q输出的第0步长
    k_out_stride0,  # K输出的第0步长 # K输出的第0步长
    num_q_heads,  # Q头的数量 # Q头的数量
    num_k_heads,  # K头的数量 # K头的数量
    Q_HAS_WEIGHT: tl.constexpr,  # Q是否有权重的编译时常量 # Q是否有权重的编译时常量
    RBLOCK: tl.constexpr,  # 行块大小的编译时常量 # 行块大小的编译时常量
    XBLOCK: tl.constexpr,  # X块大小的编译时常量 # X块大小的编译时常量
):
    num_q_rows = num_tokens * num_q_heads  # Q的总行数 = token数 * Q头数 # Q的总行数 = token数 * Q头数
    total_rows = num_tokens * (num_q_heads + num_k_heads)  # 总行数 = token数 * (Q头数 + K头数) # 总行数 = token数 * (Q头数 + K头数)

    xoffset = tl.program_id(0) * XBLOCK  # 计算x偏移量 # 计算x偏移量
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # 计算x索引 # 计算x索引
    xmask = xindex < total_rows  # 生成行掩码 # 生成行掩码
    cols = tl.arange(0, RBLOCK)[None, :]  # 生成列索引 # 生成列索引
    col_mask = cols < head_dim  # 生成列掩码 # 生成列掩码

    is_q = xindex < num_q_rows  # 判断当前行是否属于Q # 判断当前行是否属于Q
    row_in_section = tl.where(is_q, xindex, xindex - num_q_rows)  # 计算在各自分区内的行号 # 计算在各自分区内的行号
    cur_num_heads = tl.where(is_q, num_q_heads, num_k_heads)  # 获取当前分区的头数 # 获取当前分区的头数

    tokens = row_in_section // cur_num_heads  # 计算token索引 # 计算token索引
    heads = row_in_section % cur_num_heads  # 计算头索引 # 计算头索引

    in_stride = tl.where(is_q, q_in_stride0, k_in_stride0)  # 选择输入步长 # 选择输入步长
    in_bases = tokens * in_stride + heads * head_dim  # 计算输入基地址 # 计算输入基地址

    out_stride0 = tl.where(is_q, q_out_stride0, k_out_stride0)  # 选择输出步长 # 选择输出步长
    out_bases = tokens * out_stride0 + heads * head_dim  # 计算输出基地址 # 计算输出基地址

    mask = xmask & col_mask  # 合并行掩码和列掩码 # 合并行掩码和列掩码

    if Q_HAS_WEIGHT:  # 如果Q有权重 # 如果Q有权重
        qw = tl.load(  # 加载Q权重
            q_weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"  # 加载Q权重，逐出策略为最后逐出
        ).to(tl.float32)  # 转换为float32 # 转换为float32
    else:  # 否则Q无权重 # 否则Q无权重
        qw = tl.full((RBLOCK,), 1.0, tl.float32)  # 用1.0填充，等效于无缩放 # 用1.0填充，等效于无缩放
    kw = tl.load(  # 加载K权重
        k_weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"  # 加载K权重，逐出策略为最后逐出
    ).to(tl.float32)  # 转换为float32 # 转换为float32
    w = tl.where(is_q, qw, kw)  # 根据是否为Q选择对应权重 # 根据是否为Q选择对应权重

    x = tl.load(  # 加载Q数据
        q_ptr + in_bases + cols,  # Q数据地址 # Q数据地址
        mask=mask & is_q,  # Q数据掩码 # Q数据掩码
        other=0.0,  # 掩码外的填充值 # 掩码外的填充值
        eviction_policy="evict_first",  # 逐出策略为首先逐出 # 逐出策略为首先逐出
    ).to(tl.float32)  # 转换为float32 # 转换为float32
    x = x + tl.load(  # 加载K数据并累加到x
        k_ptr + in_bases + cols,  # K数据地址 # K数据地址
        mask=mask & ~is_q,  # K数据掩码（非Q行） # K数据掩码（非Q行）
        other=0.0,  # 掩码外的填充值 # 掩码外的填充值
        eviction_policy="evict_first",  # 逐出策略为首先逐出 # 逐出策略为首先逐出
    ).to(tl.float32)  # 转换为float32 # 转换为float32

    var = tl.sum(x * x, 1)[:, None]  # 计算方差（平方和） # 计算方差（平方和）
    rstd = tl.rsqrt(var / head_dim + eps)  # 计算标准差的倒数（RMSNorm公式） # 计算标准差的倒数（RMSNorm公式）

    out = (x * rstd * w).to(q_out_ptr.dtype.element_ty)  # 计算归一化输出并转换为目标数据类型 # 计算归一化输出并转换为目标数据类型
    tl.store(  # 存储Q归一化结果
        q_out_ptr + out_bases + cols,  # Q输出地址 # Q输出地址
        out,  # 输出值 # 输出值
        mask=mask & is_q,  # Q输出掩码 # Q输出掩码
        eviction_policy="evict_first",  # 逐出策略为首先逐出 # 逐出策略为首先逐出
    )
    tl.store(  # 存储K归一化结果
        k_out_ptr + out_bases + cols,  # K输出地址 # K输出地址
        out,  # 输出值 # 输出值
        mask=mask & ~is_q,  # K输出掩码（非Q行） # K输出掩码（非Q行）
        eviction_policy="evict_first",  # 逐出策略为首先逐出 # 逐出策略为首先逐出
    )


# 融合Q/K RMSNorm的Python包装函数
def fused_qk_norm(  # 融合Q/K归一化函数
    q: torch.Tensor,  # Q张量 [num_tokens, num_heads, head_dim] # Q张量
    k: torch.Tensor,  # K张量 [num_tokens, num_kv_heads, head_dim] # K张量
    q_weight: Optional[torch.Tensor],  # Q归一化权重 [head_dim]，可为None表示无权重 # Q归一化权重
    k_weight: torch.Tensor,  # K归一化权重 [head_dim]（始终必须） # K归一化权重
    eps: float,  # 用于数值稳定性的epsilon值 # 用于数值稳定性的epsilon值
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回归一化后的Q和K元组 # 返回归一化后的Q和K元组
    """Fused Q/K RMSNorm in a single Triton kernel launch.
    将Q和K的RMSNorm融合为单次Triton内核启动。

    Args:
        q: [num_tokens, num_heads, head_dim]
        Q张量：[token数, 头数, 头维度]
        k: [num_tokens, num_kv_heads, head_dim]
        K张量：[token数, KV头数, 头维度]
        q_weight: [head_dim] norm weight, or None for weightless Q norm
        Q权重：[头维度] 归一化权重，或None表示Q归一化无权重
        k_weight: [head_dim] norm weight (always required)
        K权重：[头维度] 归一化权重（始终必须）
        eps: epsilon for numerical stability
        eps：用于数值稳定性的epsilon值

    Returns:
        (q_normed, k_normed) same shapes as inputs
        (归一化后的Q, 归一化后的K)，形状与输入相同
    """
    head_dim = k_weight.shape[0]  # 获取头维度 # 获取头维度
    if q_weight is not None:  # 如果Q权重存在 # 如果Q权重存在
        assert q_weight.shape[0] == head_dim  # 断言Q权重维度与头维度一致 # 断言Q权重维度与头维度一致
    num_tokens = q.shape[0]  # 获取token数量 # 获取token数量
    num_q_heads = q.shape[1]  # 获取Q头数量 # 获取Q头数量
    num_k_heads = k.shape[1]  # 获取K头数量 # 获取K头数量
    total_rows = num_tokens * (num_q_heads + num_k_heads)  # 计算总行数 # 计算总行数
    RBLOCK = triton.next_power_of_2(head_dim)  # 计算大于等于head_dim的最小2的幂 # 计算大于等于head_dim的最小2的幂

    q_out = torch.empty_like(q)  # 分配Q输出张量 # 分配Q输出张量
    k_out = torch.empty_like(k)  # 分配K输出张量 # 分配K输出张量

    XBLOCK = 2 if total_rows > 8192 else 1  # 行数大于8192时使用XBLOCK=2 # 行数大于8192时使用XBLOCK=2
    NUM_WARPS = 1  # 设置warp数量为1 # 设置warp数量为1
    q_weight_arg = q_weight if q_weight is not None else k_weight  # Q权重为None时用K权重占位（内核中不使用） # Q权重为None时用K权重占位
    _fused_qk_norm_kernel[((total_rows + XBLOCK - 1) // XBLOCK,)](  # 启动融合Q/K归一化内核
        q,  # Q输入 # Q输入
        k,  # K输入 # K输入
        q_out,  # Q输出 # Q输出
        k_out,  # K输出 # K输出
        q_weight_arg,  # Q权重参数 # Q权重参数
        k_weight,  # K权重 # K权重
        eps,  # epsilon值 # epsilon值
        num_tokens,  # token数量 # token数量
        head_dim,  # 头维度 # 头维度
        q.stride(0),  # Q输入第0步长 # Q输入第0步长
        k.stride(0),  # K输入第0步长 # K输入第0步长
        q_out.stride(0),  # Q输出第0步长 # Q输出第0步长
        k_out.stride(0),  # K输出第0步长 # K输出第0步长
        num_q_heads,  # Q头数量 # Q头数量
        num_k_heads,  # K头数量 # K头数量
        Q_HAS_WEIGHT=q_weight is not None,  # Q是否有权重 # Q是否有权重
        RBLOCK=RBLOCK,  # 行块大小 # 行块大小
        XBLOCK=XBLOCK,  # X块大小 # X块大小
        num_warps=NUM_WARPS,  # warp数量 # warp数量
    )
    return q_out, k_out  # 返回归一化后的Q和K # 返回归一化后的Q和K
