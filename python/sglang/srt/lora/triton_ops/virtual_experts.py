# LoRA虚拟专家Triton算子模块
# 本模块实现了LoRA虚拟专家的核心Triton GPU内核和辅助函数，
# 用于在混合专家(MoE)模型中融合虚拟专家路由、KV缓存对齐和LoRA计算。
# 主要功能包括：虚拟topk ID融合内核、专家ID清洗内核、
# Split-K分组GEMM收缩内核、块大小对齐以及合并专家融合MoE LoRA添加操作。

"""
LoRA Virtual Experts Triton Ops.
"""

import functools  # 导入functools模块，用于偏函数等工具 # 导入functools模块，用于偏函数等工具
from typing import Any  # 导入Any类型 # 导入Any类型

import torch  # 导入PyTorch库 # 导入PyTorch库
import triton  # 导入Triton库 # 导入Triton库
import triton.language as tl  # 导入Triton语言并简写为tl # 导入Triton语言并简写为tl

from sglang.jit_kernel.moe_align import moe_align_block_size as jit_moe_align_block_size  # 导入JIT编译的MoE块大小对齐函数 # 导入JIT编译的MoE块大小对齐函数


@triton.jit  # Triton JIT编译装饰器 # Triton JIT编译装饰器
def _fused_virtual_topk_ids_kernel(  # 融合虚拟topk IDs内核 # 融合虚拟topk IDs内核
    topk_ids_ptr,  # topk ID指针 # topk ID指针
    token_lora_mapping_ptr,  # token到LoRA的映射指针 # token到LoRA的映射指针
    virtual_topk_ids_ptr,  # 虚拟topk ID输出指针 # 虚拟topk ID输出指针
    token_lora_mask_ptr,  # token LoRA掩码输出指针 # token LoRA掩码输出指针
    num_experts_for_weight: tl.constexpr,  # 每个权重的专家数量（编译时常量） # 每个权重的专家数量（编译时常量）
    M,  # token数量M # token数量M
    top_k: tl.constexpr,  # top-k值（编译时常量） # top-k值（编译时常量）
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量） # 块大小（编译时常量）
):
    """
    Fuses _get_virtual_topk_ids: comparison + clamp + arithmetic into one kernel.
    融合_get_virtual_topk_ids：比较+裁剪+算术运算为一个内核。

    For each (m, k):
        lora_id = token_lora_mapping[m]
        mask[m] = (lora_id >= 0)
        safe_lora = max(lora_id, 0)
        if shared_outer:  (handled by num_experts_for_weight == 0 sentinel)
            virtual_topk_ids[m, k] = safe_lora * 1  (= safe_lora)
        else:
            virtual_topk_ids[m, k] = topk_ids[m, k] + safe_lora * num_experts_for_weight
    """
    pid = tl.program_id(0)  # 获取程序ID # 获取程序ID
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算偏移量 # 计算偏移量
    total = M * top_k  # 计算总元素数 # 计算总元素数
    valid = offs < total  # 计算有效掩码 # 计算有效掩码

    m = offs // top_k  # 计算行索引m # 计算行索引m
    # k = offs % top_k  # not needed directly  # k = offs % top_k  # 不需要直接使用

    lora_id = tl.load(token_lora_mapping_ptr + m, mask=valid, other=0)  # 加载LoRA ID # 加载LoRA ID
    mask_val = lora_id >= 0  # 计算掩码值（LoRA ID非负表示有效） # 计算掩码值（LoRA ID非负表示有效）
    safe_lora = tl.maximum(lora_id, 0)  # 将负值裁剪为0，得到安全的LoRA ID # 将负值裁剪为0，得到安全的LoRA ID

    base = tl.load(topk_ids_ptr + offs, mask=valid, other=0)  # 加载基础topk ID # 加载基础topk ID
    # Preserve negative sentinel topk_ids (e.g. -1 for non-local experts after
    # EP dispatch). Without this, `-1 + safe_lora * num_experts` would land on
    # a real virtual-expert slot belonging to another adapter and trigger OOB
    # loads in downstream LoRA kernels.
    # 保留负数哨兵topk_ids（例如EP分发后非本地专家的-1）。
    # 如果不保留，`-1 + safe_lora * num_experts` 会落在属于另一个适配器的
    # 真实虚拟专家槽位上，并在下游LoRA内核中触发越界加载。
    shifted = base + safe_lora * num_experts_for_weight  # 计算偏移后的值 # 计算偏移后的值
    result = tl.where(base < 0, base, shifted)  # 如果base为负则保留原值，否则使用偏移值 # 如果base为负则保留原值，否则使用偏移值
    tl.store(virtual_topk_ids_ptr + offs, result, mask=valid)  # 存储虚拟topk ID结果 # 存储虚拟topk ID结果

    # Write mask once per row (at first k position)
    # 每行只写一次掩码（在第一个k位置）
    k = offs % top_k  # 计算列索引k # 计算列索引k
    is_first_k = k == 0  # 判断是否为第一个k位置 # 判断是否为第一个k位置
    tl.store(token_lora_mask_ptr + m, mask_val, mask=valid & is_first_k)  # 存储token LoRA掩码 # 存储token LoRA掩码


def _fused_virtual_topk_ids(  # 融合虚拟topk IDs函数 # 融合虚拟topk IDs函数
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    token_lora_mapping: torch.Tensor,  # token到LoRA的映射张量 # token到LoRA的映射张量
    num_experts: int,  # 专家数量 # 专家数量
    shared_outer: bool,  # 是否共享外层 # 是否共享外层
    max_loras: int,  # 最大LoRA数量 # 最大LoRA数量
) -> tuple[torch.Tensor, torch.Tensor, int]:  # 返回虚拟topk IDs、掩码和虚拟专家数 # 返回虚拟topk IDs、掩码和虚拟专家数
    """
    Returns virtual topk_ids, token_lora_mask, and virtual_num_experts.
    返回虚拟topk_ids、token_lora_mask和virtual_num_experts。
    """
    M, top_k = topk_ids.shape  # 获取形状M和top_k # 获取形状M和top_k
    device = topk_ids.device  # 获取设备 # 获取设备

    if shared_outer:  # 如果共享外层 # 如果共享外层
        num_experts_for_weight = 1  # 共享外层时每个权重只有1个专家 # 共享外层时每个权重只有1个专家
        # For shared_outer, we need topk_ids to be zeros
        # 对于shared_outer，需要topk_ids为零
        zero_topk = torch.zeros_like(topk_ids)  # 创建全零的topk_ids # 创建全零的topk_ids
        input_topk = zero_topk  # 使用全零作为输入 # 使用全零作为输入
    else:  # 否则 # 否则
        num_experts_for_weight = num_experts  # 使用实际专家数量 # 使用实际专家数量
        input_topk = topk_ids  # 使用原始topk_ids # 使用原始topk_ids

    virtual_topk_ids = torch.empty_like(topk_ids)  # 创建虚拟topk IDs输出张量 # 创建虚拟topk IDs输出张量
    token_lora_mask = torch.empty(M, dtype=torch.bool, device=device)  # 创建token LoRA掩码输出张量 # 创建token LoRA掩码输出张量

    BLOCK_SIZE = 1024  # 设置块大小为1024 # 设置块大小为1024
    grid = ((M * top_k + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # 计算网格大小 # 计算网格大小

    _fused_virtual_topk_ids_kernel[grid](  # 启动融合虚拟topk IDs内核 # 启动融合虚拟topk IDs内核
        input_topk,  # 输入topk IDs # 输入topk IDs
        token_lora_mapping,  # token到LoRA的映射 # token到LoRA的映射
        virtual_topk_ids,  # 虚拟topk IDs输出 # 虚拟topk IDs输出
        token_lora_mask,  # token LoRA掩码输出 # token LoRA掩码输出
        num_experts_for_weight,  # 每个权重的专家数量 # 每个权重的专家数量
        M,  # token数量 # token数量
        top_k,  # top-k值 # top-k值
        BLOCK_SIZE,  # 块大小 # 块大小
    )

    virtual_num_experts = num_experts_for_weight * max_loras  # 计算虚拟专家总数 # 计算虚拟专家总数
    return virtual_topk_ids, token_lora_mask, virtual_num_experts  # 返回结果 # 返回结果


@triton.jit  # Triton JIT编译装饰器 # Triton JIT编译装饰器
def _fused_sanitize_expert_ids_kernel(  # 融合清洗专家ID内核 # 融合清洗专家ID内核
    expert_ids_ptr,  # 专家ID输入指针 # 专家ID输入指针
    output_ptr,  # 输出指针 # 输出指针
    num_virtual_experts,  # 虚拟专家数量 # 虚拟专家数量
    N,  # 元素总数 # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量） # 块大小（编译时常量）
):
    pid = tl.program_id(0)  # 获取程序ID # 获取程序ID
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算偏移量 # 计算偏移量
    valid = offs < N  # 计算有效掩码 # 计算有效掩码

    eid = tl.load(expert_ids_ptr + offs, mask=valid, other=0)  # 加载专家ID # 加载专家ID
    result = tl.where(eid < num_virtual_experts, eid, -1)  # 超出范围的ID替换为-1 # 超出范围的ID替换为-1
    tl.store(output_ptr + offs, result, mask=valid)  # 存储清洗结果 # 存储清洗结果


def fused_sanitize_expert_ids(  # 融合清洗专家ID函数 # 融合清洗专家ID函数
    expert_ids: torch.Tensor,  # 专家ID张量 # 专家ID张量
    num_virtual_experts: int,  # 虚拟专家数量 # 虚拟专家数量
) -> torch.Tensor:  # 返回清洗后的专家ID张量 # 返回清洗后的专家ID张量
    """
    Sanitize expert_ids by replacing values >= num_virtual_experts with -1.
    清洗专家ID，将>=num_virtual_experts的值替换为-1。

    Returns a new tensor with expert_ids >= num_virtual_experts replaced by -1.
    返回一个新张量，其中>=num_virtual_experts的expert_id被替换为-1。
    """
    N = expert_ids.numel()  # 获取元素总数 # 获取元素总数
    output = torch.empty_like(expert_ids)  # 创建输出张量 # 创建输出张量

    BLOCK_SIZE = 1024  # 设置块大小为1024 # 设置块大小为1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # 计算网格大小 # 计算网格大小

    _fused_sanitize_expert_ids_kernel[grid](  # 启动融合清洗专家ID内核 # 启动融合清洗专家ID内核
        expert_ids,  # 专家ID输入 # 专家ID输入
        output,  # 输出张量 # 输出张量
        num_virtual_experts,  # 虚拟专家数量 # 虚拟专家数量
        N,  # 元素总数 # 元素总数
        BLOCK_SIZE,  # 块大小 # 块大小
    )
    return output  # 返回清洗结果 # 返回清洗结果


@triton.jit  # Triton JIT编译装饰器 # Triton JIT编译装饰器
def _moe_lora_shrink_splitk_kernel(  # MoE LoRA收缩Split-K内核 # MoE LoRA收缩Split-K内核
    # Pointers
    # 指针
    a_ptr,  # type: ignore  # [num_tokens, K]  # 输入矩阵A指针 # 输入矩阵A指针
    b_ptr,  # type: ignore  # [num_virtual_experts, N, K]  # 权重矩阵B指针 # 权重矩阵B指针
    c_ptr,  # type: ignore  # [num_tokens * top_k, N]  (pre-zeroed when SPLIT_K > 1)  # 输出矩阵C指针 # 输出矩阵C指针
    sorted_token_ids_ptr,  # type: ignore  # 排序后的token ID指针 # 排序后的token ID指针
    expert_ids_ptr,  # type: ignore  # 专家ID指针 # 专家ID指针
    num_tokens_post_padded_ptr,  # type: ignore  # 填充后token数指针 # 填充后token数指针
    # Dimensions
    # 维度
    N,  # type: ignore  # N维度 # N维度
    K,  # type: ignore  # K维度 # K维度
    num_valid_tokens,  # type: ignore  # 有效token数 # 有效token数
    # Strides
    # 步长
    stride_am,  # type: ignore  # A矩阵m方向步长 # A矩阵m方向步长
    stride_ak,  # type: ignore  # A矩阵k方向步长 # A矩阵k方向步长
    stride_be,  # type: ignore  # B矩阵expert方向步长 # B矩阵expert方向步长
    stride_bn,  # type: ignore  # B矩阵n方向步长 # B矩阵n方向步长
    stride_bk,  # type: ignore  # B矩阵k方向步长 # B矩阵k方向步长
    stride_cm,  # type: ignore  # C矩阵m方向步长 # C矩阵m方向步长
    stride_cn,  # type: ignore  # C矩阵n方向步长 # C矩阵n方向步长
    # Constexprs
    # 编译时常量
    top_k: tl.constexpr,  # top-k值 # top-k值
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小 # M方向块大小
    BLOCK_SIZE_N: tl.constexpr,  # N方向块大小 # N方向块大小
    BLOCK_SIZE_K: tl.constexpr,  # K方向块大小 # K方向块大小
    GROUP_SIZE_M: tl.constexpr,  # M方向分组大小 # M方向分组大小
    SPLIT_K: tl.constexpr,  # K方向分割数 # K方向分割数
):
    """Split-K grouped GEMM for the LoRA A (shrink) stage with few virtual experts."""
    """Split-K分组GEMM，用于具有少量虚拟专家的LoRA A（收缩）阶段。"""
    pid = tl.program_id(0)  # 获取程序ID # 获取程序ID
    pid_sk = pid % SPLIT_K  # 计算Split-K索引 # 计算Split-K索引
    pid_mn = pid // SPLIT_K  # 计算mn索引 # 计算mn索引

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)  # 加载填充后token数 # 加载填充后token数
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)  # 计算M方向程序数 # 计算M方向程序数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # 计算N方向程序数 # 计算N方向程序数

    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 计算每组程序数 # 计算每组程序数
    group_id = pid_mn // num_pid_in_group  # 计算组ID # 计算组ID
    first_pid_m = group_id * GROUP_SIZE_M  # 计算组内第一个M方向PID # 计算组内第一个M方向PID
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 计算实际组大小 # 计算实际组大小
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)  # 计算M方向PID # 计算M方向PID
    pid_n = (pid_mn % num_pid_in_group) // group_size_m  # 计算N方向PID # 计算N方向PID

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:  # 如果超出范围则返回 # 如果超出范围则返回
        return  # 提前返回 # 提前返回

    # Token routing (same pattern as fused_moe_triton_kernels)
    # Token路由（与fused_moe_triton_kernels相同的模式）
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)  # 计算token ID偏移 # 计算token ID偏移
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)  # 加载排序后的token ID # 加载排序后的token ID
    token_mask = offs_token < num_valid_tokens  # 计算token有效掩码 # 计算token有效掩码

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)  # 加载专家偏移 # 加载专家偏移
    if off_expert == -1:  # 如果专家ID为-1则返回 # 如果专家ID为-1则返回
        return  # 提前返回 # 提前返回

    # Pointers
    # 指针
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N  # 计算B矩阵N方向偏移 # 计算B矩阵N方向偏移
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # 计算K方向偏移 # 计算K方向偏移

    a_ptrs = a_ptr + (  # 计算A矩阵指针 # 计算A矩阵指针
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (  # 计算B矩阵指针 # 计算B矩阵指针
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Accumulate
    # 累加
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器 # 初始化累加器
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)  # 计算K方向网格数 # 计算K方向网格数
    for k in range(0, grid_k):  # 遍历K方向网格 # 遍历K方向网格
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)  # 计算剩余K维度 # 计算剩余K维度
        k_mask = offs_k[:, None] < k_remaining  # 计算K方向掩码 # 计算K方向掩码
        a = tl.load(  # 加载A矩阵块 # 加载A矩阵块
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)  # 加载B矩阵块 # 加载B矩阵块
        accumulator += tl.dot(a, b.to(a.dtype))  # 矩阵乘法并累加 # 矩阵乘法并累加
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak  # 更新A矩阵指针 # 更新A矩阵指针
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk  # 更新B矩阵指针 # 更新B矩阵指针

    accumulator = accumulator.to(c_ptr.dtype.element_ty)  # 转换累加器数据类型 # 转换累加器数据类型

    # Write output
    # 写入输出
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算C矩阵N方向偏移 # 计算C矩阵N方向偏移
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  # 计算C矩阵指针 # 计算C矩阵指针
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  # 计算C矩阵掩码 # 计算C矩阵掩码
    if SPLIT_K == 1:  # 如果没有Split-K # 如果没有Split-K
        tl.store(c_ptrs, accumulator, mask=c_mask)  # 直接存储 # 直接存储
    else:  # 否则 # 否则
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")  # 原子加法存储 # 原子加法存储


def _invoke_moe_lora_shrink_splitk(  # 调用MoE LoRA收缩Split-K内核 # 调用MoE LoRA收缩Split-K内核
    hidden_states: torch.Tensor,  # 隐藏状态张量 # 隐藏状态张量
    weight: torch.Tensor,  # 权重张量 # 权重张量
    output: torch.Tensor,  # 输出张量 # 输出张量
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    sorted_token_ids: torch.Tensor,  # 排序后的token ID张量 # 排序后的token ID张量
    expert_ids: torch.Tensor,  # 专家ID张量 # 专家ID张量
    num_tokens_post_padded: torch.Tensor,  # 填充后token数张量 # 填充后token数张量
    top_k: int,  # top-k值 # top-k值
    config: dict[str, Any],  # 配置字典 # 配置字典
) -> None:  # 无返回值 # 无返回值
    """Launch split-K shrink kernel for LoRA A with few virtual experts."""
    """启动Split-K收缩内核，用于具有少量虚拟专家的LoRA A。"""
    N = weight.shape[1]  # 获取N维度 # 获取N维度
    K = weight.shape[2]  # 获取K维度 # 获取K维度
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]  # 获取M方向块大小 # 获取M方向块大小
    BLOCK_SIZE_N = min(config.get("BLOCK_SIZE_N", 64), max(16, N))  # 获取N方向块大小 # 获取N方向块大小
    BLOCK_SIZE_K = config.get("BLOCK_SIZE_K", 64)  # 获取K方向块大小 # 获取K方向块大小
    GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)  # 获取M方向分组大小 # 获取M方向分组大小

    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M)  # 计算M方向块数 # 计算M方向块数
    num_n_blocks = triton.cdiv(N, BLOCK_SIZE_N)  # 计算N方向块数 # 计算N方向块数
    base_grid = num_m_blocks * num_n_blocks  # 计算基础网格大小 # 计算基础网格大小
    max_split_k = max(1, K // BLOCK_SIZE_K)  # 计算最大Split-K数 # 计算最大Split-K数
    SPLIT_K = min(max_split_k, max(1, 128 // base_grid)) if base_grid < 128 else 1  # 计算实际Split-K数 # 计算实际Split-K数

    grid = (SPLIT_K * base_grid,)  # 计算网格大小 # 计算网格大小

    _moe_lora_shrink_splitk_kernel[grid](  # 启动MoE LoRA收缩Split-K内核 # 启动MoE LoRA收缩Split-K内核
        hidden_states,  # 隐藏状态 # 隐藏状态
        weight,  # 权重 # 权重
        output,  # 输出 # 输出
        sorted_token_ids,  # 排序后的token ID # 排序后的token ID
        expert_ids,  # 专家ID # 专家ID
        num_tokens_post_padded,  # 填充后token数 # 填充后token数
        N,  # N维度 # N维度
        K,  # K维度 # K维度
        topk_ids.numel(),  # 有效token数 # 有效token数
        hidden_states.stride(0),  # A矩阵m方向步长 # A矩阵m方向步长
        hidden_states.stride(1),  # A矩阵k方向步长 # A矩阵k方向步长
        weight.stride(0),  # B矩阵expert方向步长 # B矩阵expert方向步长
        weight.stride(1),  # B矩阵n方向步长 # B矩阵n方向步长
        weight.stride(2),  # B矩阵k方向步长 # B矩阵k方向步长
        output.stride(0),  # C矩阵m方向步长 # C矩阵m方向步长
        output.stride(1),  # C矩阵n方向步长 # C矩阵n方向步长
        top_k=top_k,  # top-k值 # top-k值
        BLOCK_SIZE_M=BLOCK_SIZE_M,  # M方向块大小 # M方向块大小
        BLOCK_SIZE_N=BLOCK_SIZE_N,  # N方向块大小 # N方向块大小
        BLOCK_SIZE_K=BLOCK_SIZE_K,  # K方向块大小 # K方向块大小
        GROUP_SIZE_M=GROUP_SIZE_M,  # M方向分组大小 # M方向分组大小
        SPLIT_K=SPLIT_K,  # Split-K数 # Split-K数
        num_warps=config.get("num_warps", 4),  # warp数量 # warp数量
        num_stages=config.get("num_stages", 4),  # 流水线阶段数 # 流水线阶段数
    )


def _align_block_size_jit(  # JIT编译的块大小对齐函数 # JIT编译的块大小对齐函数
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    block_size: int,  # 块大小 # 块大小
    num_experts: int,  # 专家数量 # 专家数量
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回排序token ID、专家ID和填充后token数 # 返回排序token ID、专家ID和填充后token数
    """CUDA JIT align_block_size for num_experts > 1024 (up to 8191).
    CUDA JIT块大小对齐函数，用于num_experts > 1024（最多8191）。

    Uses the v2 kernel from moe_align_kernel.cu which supports large expert
    counts via per-thread multi-expert processing and a two-level warp scan,
    replacing the previous pure-PyTorch fallback that had excessive CPU overhead
    from 15+ individual kernel launches and torch.argsort.
    使用moe_align_kernel.cu中的v2内核，通过每线程多专家处理和两级warp扫描
    支持大专家数量，替代了之前因15+个独立内核启动和torch.argsort导致
    CPU开销过大的纯PyTorch回退方案。

    The JIT kernel uses a +1 offset convention: topk_ids are shifted by +1 so
    that the EP sentinel value (-1) maps to bucket 0. The kernel internally
    handles histogram, padded prefix-sum, expert_ids assignment, and token
    scattering in just 2–3 CUDA kernel launches.
    JIT内核使用+1偏移约定：topk_ids偏移+1，使EP哨兵值(-1)映射到桶0。
    该内核内部仅通过2-3次CUDA内核启动处理直方图、填充前缀和、
    expert_ids分配和token散射。
    """
    assert num_experts <= 8191, (  # 断言专家数量不超过8191 # 断言专家数量不超过8191
        f"_align_block_size_jit supports at most 8191 experts "
        f"(num_moe_experts * max_loras), got {num_experts}"
    )

    device = topk_ids.device  # 获取设备 # 获取设备
    flat_topk_ids = topk_ids.reshape(-1)  # 展平topk_ids # 展平topk_ids
    if flat_topk_ids.dtype == torch.int64:  # 如果是int64类型则转换为int32 # 如果是int64类型则转换为int32
        flat_topk_ids = flat_topk_ids.to(torch.int32)  # 转换为int32 # 转换为int32
    num_total_tokens = flat_topk_ids.numel()  # 获取总token数 # 获取总token数

    if num_total_tokens == 0:  # 如果没有token则返回空张量 # 如果没有token则返回空张量
        empty = torch.empty(0, dtype=torch.int32, device=device)  # 创建空张量 # 创建空张量
        return empty, empty, torch.zeros(1, dtype=torch.int32, device=device)  # 返回空结果 # 返回空结果

    # JIT kernel uses +1 offset convention: -1 -> bucket 0 (sentinel),
    # expert i -> bucket i+1. So pass num_experts + 1 as the bucket count.
    # JIT内核使用+1偏移约定：-1映射到桶0（哨兵），专家i映射到桶i+1。
    # 因此传入num_experts + 1作为桶数。
    jit_num_experts = num_experts + 1  # 计算JIT专家数 # 计算JIT专家数

    if num_total_tokens < jit_num_experts:  # 如果总token数小于JIT专家数 # 如果总token数小于JIT专家数
        max_num_tokens_padded = num_total_tokens * block_size  # 计算最大填充token数 # 计算最大填充token数
    else:  # 否则 # 否则
        max_num_tokens_padded = num_total_tokens + jit_num_experts * (block_size - 1)  # 计算最大填充token数 # 计算最大填充token数

    # Align every sub-buffer offset to a multiple of 4 (VEC_SIZE). The CUDA
    # kernel fills sorted_token_ids with vectorized int4 writes whose last
    # store can spill up to 3 int32s past the logical end. With a fused
    # allocation the spill would corrupt the adjacent sub-buffer.
    # 将每个子缓冲区偏移对齐到4的倍数（VEC_SIZE）。CUDA内核使用向量化int4写入
    # 填充sorted_token_ids，其最后一次存储可能溢出逻辑末尾最多3个int32。
    # 使用融合分配时，溢出会破坏相邻的子缓冲区。
    _A4 = lambda n: (n + 3) & ~3  # noqa: E731  # 4字节对齐函数 # 4字节对齐函数
    max_num_tokens_padded = _A4(max_num_tokens_padded)  # 对齐填充token数 # 对齐填充token数
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size  # 计算最大M方向块数 # 计算最大M方向块数
    max_num_m_blocks_padded = _A4(max_num_m_blocks)  # 对齐M方向块数 # 对齐M方向块数
    num_post_pad_size = _A4(1)  # 1 element, padded to 4  # 1个元素，填充到4 # 1个元素，填充到4
    cumsum_size = _A4(jit_num_experts + 1)  # 前缀和缓冲区大小 # 前缀和缓冲区大小

    # Single allocation sliced into 4 views (zero-copy) to avoid
    # per-call Python overhead of 4 separate torch.empty calls.
    # 单次分配切分为4个视图（零拷贝），以避免4次独立torch.empty调用的
    # 每次调用Python开销。
    total_buf = (  # 计算总缓冲区大小 # 计算总缓冲区大小
        max_num_tokens_padded
        + max_num_m_blocks_padded
        + num_post_pad_size
        + cumsum_size
    )
    buf = torch.empty(total_buf, dtype=torch.int32, device=device)  # 分配统一缓冲区 # 分配统一缓冲区
    off = 0  # 初始化偏移量 # 初始化偏移量
    sorted_token_ids = buf[off : off + max_num_tokens_padded]  # 切片获取排序token ID视图 # 切片获取排序token ID视图
    off += max_num_tokens_padded  # 更新偏移量 # 更新偏移量
    expert_ids = buf[off : off + max_num_m_blocks]  # 切片获取专家ID视图 # 切片获取专家ID视图
    off += max_num_m_blocks_padded  # 更新偏移量 # 更新偏移量
    num_tokens_post_padded = buf[off : off + 1]  # 切片获取填充后token数视图 # 切片获取填充后token数视图
    off += num_post_pad_size  # 更新偏移量 # 更新偏移量
    cumsum_buffer = buf[off : off + jit_num_experts + 1]  # 切片获取前缀和缓冲区视图 # 切片获取前缀和缓冲区视图

    jit_moe_align_block_size(  # 调用JIT MoE块大小对齐内核 # 调用JIT MoE块大小对齐内核
        flat_topk_ids,  # 展平的topk_ids # 展平的topk_ids
        jit_num_experts,  # JIT专家数 # JIT专家数
        block_size,  # 块大小 # 块大小
        sorted_token_ids,  # 排序后的token ID # 排序后的token ID
        expert_ids,  # 专家ID # 专家ID
        num_tokens_post_padded,  # 填充后token数 # 填充后token数
        cumsum_buffer,  # 前缀和缓冲区 # 前缀和缓冲区
        True,  # pad_sorted_token_ids  # 填充排序token ID # 填充排序token ID
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded  # 返回结果 # 返回结果


@torch.compile(dynamic=True)  # 使用torch.compile动态编译装饰器 # 使用torch.compile动态编译装饰器
def _align_block_size_torch(  # PyTorch实现的块大小对齐函数 # PyTorch实现的块大小对齐函数
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    block_size: int,  # 块大小 # 块大小
    num_experts: int,  # 专家数量 # 专家数量
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回排序token ID、专家ID和填充后token数 # 返回排序token ID、专家ID和填充后token数
    """Pure-PyTorch align_block_size for num_experts > 1024, compiled via torch.compile.
    纯PyTorch实现的块大小对齐函数，用于num_experts > 1024，通过torch.compile编译。

    Fallback for platforms where the CUDA JIT kernel is unavailable (e.g. AMD/ROCm).
    在CUDA JIT内核不可用的平台（如AMD/ROCm）上的回退方案。

    Out-of-range topk_ids (negative sentinels left by EP dispatch, or virtual-
    expert IDs >= num_experts produced when those sentinels are combined with
    a per-adapter offset) are routed into a dedicated sentinel bucket. Without
    this, indexing ``padded_offsets[sorted_expert_ids]`` would wrap (-1) or
    OOB-read, and the bad expert ids would propagate into the downstream LoRA
    GEMM as real expert slots.
    超出范围的topk_ids（EP分发留下的负数哨兵，或当这些哨兵与每个适配器偏移
    组合时产生的>=num_experts的虚拟专家ID）被路由到专用哨兵桶中。
    如果不这样做，索引``padded_offsets[sorted_expert_ids]``会回绕（-1）或
    越界读取，并且错误的专家ID会作为真实专家槽位传播到下游LoRA GEMM中。
    """
    device = topk_ids.device  # 获取设备 # 获取设备
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)  # 展平并转换为int64 # 展平并转换为int64
    num_total_tokens = flat_topk_ids.numel()  # 获取总token数 # 获取总token数

    sentinel = num_experts  # 哨兵值等于专家数量 # 哨兵值等于专家数量
    valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < num_experts)  # 计算有效掩码 # 计算有效掩码
    safe_topk_ids = torch.where(  # 将无效ID替换为哨兵值 # 将无效ID替换为哨兵值
        valid_mask,
        flat_topk_ids,
        torch.full_like(flat_topk_ids, sentinel),
    )

    bucket_count = num_experts + 1  # 桶数量为专家数+1（含哨兵桶） # 桶数量为专家数+1（含哨兵桶）
    max_total_padded_tokens = (  # 计算最大填充token总数 # 计算最大填充token总数
        (num_total_tokens + bucket_count * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded_tokens // block_size  # 计算最大块数 # 计算最大块数

    sorted_token_ids = torch.full(  # 创建排序token ID张量，填充为总token数 # 创建排序token ID张量，填充为总token数
        (max_total_padded_tokens,),
        num_total_tokens,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.full(  # 创建专家ID张量，填充为-1 # 创建专家ID张量，填充为-1
        (max_num_blocks,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_total_tokens == 0:  # 如果没有token则返回空结果 # 如果没有token则返回空结果
        num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32, device=device)  # 创建零张量 # 创建零张量
        return sorted_token_ids, expert_ids, num_tokens_post_padded  # 返回结果 # 返回结果

    sorted_order = torch.argsort(safe_topk_ids)  # 按安全topk ID排序 # 按安全topk ID排序
    sorted_expert_ids = safe_topk_ids[sorted_order]  # 获取排序后的专家ID # 获取排序后的专家ID
    expert_range = torch.arange(bucket_count, device=device, dtype=torch.int64)  # 创建专家范围 # 创建专家范围
    counts_offsets = torch.searchsorted(sorted_expert_ids, expert_range, right=False)  # 搜索每个专家的起始位置 # 搜索每个专家的起始位置
    counts_end = torch.searchsorted(sorted_expert_ids, expert_range, right=True)  # 搜索每个专家的结束位置 # 搜索每个专家的结束位置
    counts = counts_end - counts_offsets  # 计算每个专家的token数 # 计算每个专家的token数
    padded_counts = ((counts + block_size - 1) // block_size) * block_size  # 计算填充后的计数 # 计算填充后的计数
    total_padded_tokens = padded_counts.sum().to(torch.int32).reshape(1)  # 计算总填充token数 # 计算总填充token数
    padded_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts  # 计算填充偏移 # 计算填充偏移

    token_ranks = (  # 计算每个token在其专家内的排名 # 计算每个token在其专家内的排名
        torch.arange(num_total_tokens, device=device, dtype=torch.int64)
        - counts_offsets[sorted_expert_ids]
    )
    output_positions = padded_offsets[sorted_expert_ids] + token_ranks  # 计算输出位置 # 计算输出位置
    sorted_token_ids.scatter_(  # 将排序后的token ID散列到输出位置 # 将排序后的token ID散列到输出位置
        0,
        output_positions.to(torch.int64),
        sorted_order.to(torch.int32),
    )

    block_counts = padded_counts // block_size  # 计算每个专家的块数 # 计算每个专家的块数
    real_block_counts = block_counts.clone()  # 克隆块计数 # 克隆块计数
    real_block_counts[sentinel] = 0  # 哨兵桶的块数设为0 # 哨兵桶的块数设为0
    actual_num_blocks = real_block_counts.sum()  # 计算实际总块数 # 计算实际总块数

    if max_num_blocks <= 0:  # 如果最大块数<=0则提前返回 # 如果最大块数<=0则提前返回
        return sorted_token_ids, expert_ids, total_padded_tokens  # 返回结果 # 返回结果

    block_offsets = torch.cumsum(real_block_counts, dim=0)  # 计算块偏移 # 计算块偏移
    all_block_positions = torch.arange(max_num_blocks, device=device, dtype=torch.int64)  # 创建所有块位置 # 创建所有块位置
    assigned_experts = torch.searchsorted(  # 搜索每个块分配的专家 # 搜索每个块分配的专家
        block_offsets, all_block_positions, right=True
    ).to(torch.int32)
    expert_ids.copy_(  # 复制专家ID到输出 # 复制专家ID到输出
        torch.where(
            all_block_positions < actual_num_blocks,
            assigned_experts,
            torch.full_like(assigned_experts, -1),
        )
    )

    return sorted_token_ids, expert_ids, total_padded_tokens  # 返回结果 # 返回结果


def _align_block_size_large(  # 大规模块大小对齐函数 # 大规模块大小对齐函数
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    block_size: int,  # 块大小 # 块大小
    num_experts: int,  # 专家数量 # 专家数量
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回排序token ID、专家ID和填充后token数 # 返回排序token ID、专家ID和填充后token数
    """Dispatch to the CUDA JIT kernel when available, otherwise fall back to
    the pure-PyTorch torch.compile path (needed on AMD/ROCm or when the JIT
    module fails to load)."""
    """当可用时分发到CUDA JIT内核，否则回退到纯PyTorch torch.compile路径
    （在AMD/ROCm上或JIT模块加载失败时需要）。"""
    try:  # 尝试使用JIT内核 # 尝试使用JIT内核
        return _align_block_size_jit(topk_ids, block_size, num_experts)  # 返回JIT结果 # 返回JIT结果
    except Exception:  # 如果JIT失败则回退 # 如果JIT失败则回退
        return _align_block_size_torch(topk_ids, block_size, num_experts)  # 返回PyTorch结果 # 返回PyTorch结果


def _merged_experts_fused_moe_lora_add_fake(  # 合并专家融合MoE LoRA添加的伪实现 # 合并专家融合MoE LoRA添加的伪实现
    output: torch.Tensor,  # 输出张量 # 输出张量
    hidden_states: torch.Tensor,  # 隐藏状态张量 # 隐藏状态张量
    lora_a: torch.Tensor,  # LoRA A权重张量 # LoRA A权重张量
    lora_b: torch.Tensor,  # LoRA B权重张量 # LoRA B权重张量
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    topk_weights: torch.Tensor,  # topk权重张量 # topk权重张量
    token_lora_mapping: torch.Tensor,  # token到LoRA的映射张量 # token到LoRA的映射张量
    mul_routed_weight: bool,  # 是否乘以路由权重 # 是否乘以路由权重
    experts_shared_outer_loras_a: bool,  # 专家是否共享外层LoRA A # 专家是否共享外层LoRA A
    experts_shared_outer_loras_b: bool,  # 专家是否共享外层LoRA B # 专家是否共享外层LoRA B
) -> None:  # 无返回值 # 无返回值
    return  # 伪实现，直接返回 # 伪实现，直接返回


def _merged_experts_fused_moe_lora_add_impl(  # 合并专家融合MoE LoRA添加的实现 # 合并专家融合MoE LoRA添加的实现
    output: torch.Tensor,  # 输出张量 # 输出张量
    hidden_states: torch.Tensor,  # 隐藏状态张量 # 隐藏状态张量
    lora_a: torch.Tensor,  # LoRA A权重张量 # LoRA A权重张量
    lora_b: torch.Tensor,  # LoRA B权重张量 # LoRA B权重张量
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    topk_weights: torch.Tensor,  # topk权重张量 # topk权重张量
    token_lora_mapping: torch.Tensor,  # token到LoRA的映射张量 # token到LoRA的映射张量
    mul_routed_weight: bool,  # 是否乘以路由权重 # 是否乘以路由权重
    experts_shared_outer_loras_a: bool,  # 专家是否共享外层LoRA A # 专家是否共享外层LoRA A
    experts_shared_outer_loras_b: bool,  # 专家是否共享外层LoRA B # 专家是否共享外层LoRA B
    routing_cache: dict | None = None,  # 路由缓存字典 # 路由缓存字典
) -> None:  # 无返回值 # 无返回值
    """
    1. Prepare virtual expert routing metadata from topk_ids + token_lora_mapping * num_experts.
    2. Flatten LoRA weights from [max_loras, num_experts, ...] to [max_loras * num_experts, ...].
    3. Run regular SGLang fused-MoE kernels for LoRA A and LoRA B.
    4. Mask out tokens with token_lora_mapping == -1 on the add path.
    """
    """
    1. 从topk_ids + token_lora_mapping * num_experts准备虚拟专家路由元数据。
    2. 将LoRA权重从[max_loras, num_experts, ...]展平为[max_loras * num_experts, ...]。
    3. 运行常规SGLang融合MoE内核进行LoRA A和LoRA B计算。
    4. 在添加路径上掩码掉token_lora_mapping == -1的token。
    """
    max_loras, _, max_lora_rank, _ = lora_a.shape  # 获取LoRA A的形状信息 # 获取LoRA A的形状信息
    input_top_k = 1 if hidden_states.shape[0] == topk_ids.numel() else topk_ids.shape[1]  # 计算输入top-k值 # 计算输入top-k值

    def _merge_lora_expert_weight(t: torch.Tensor) -> torch.Tensor:  # 合并LoRA专家权重 # 合并LoRA专家权重
        # [max_loras, num_experts, x, y] -> [max_loras * num_experts, x, y]
        # [max_loras, num_experts, x, y] -> [max_loras * num_experts, x, y]
        return t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])  # 展平前两个维度 # 展平前两个维度

    def _get_stage_config(  # 获取阶段配置 # 获取阶段配置
        weight: torch.Tensor,  # 权重张量 # 权重张量
        stage_top_k: int,  # 阶段top-k值 # 阶段top-k值
    ) -> dict[str, Any]:  # 返回配置字典 # 返回配置字典
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (  # 导入MoE Triton配置工具 # 导入MoE Triton配置工具
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        config_dtype = get_config_dtype_str(dtype=hidden_states.dtype)  # 获取配置数据类型字符串 # 获取配置数据类型字符串
        get_config_func = functools.partial(  # 创建偏函数 # 创建偏函数
            try_get_optimal_moe_config,
            weight.shape,
            weight.shape,
            stage_top_k,
            config_dtype,
        )
        try:  # 尝试获取最优配置 # 尝试获取最优配置
            cfg = get_config_func(token_lora_mapping.shape[0])  # 获取最优MoE配置 # 获取最优MoE配置
        except ValueError:  # 如果获取失败则使用默认配置 # 如果获取失败则使用默认配置
            K_dim = weight.shape[2]  # 获取K维度 # 获取K维度
            N_dim = weight.shape[1]  # 获取N维度 # 获取N维度
            if K_dim >= 1024:  # 如果K维度>=1024 # 如果K维度>=1024
                default_block_k = 256  # 默认K方向块大小为256 # 默认K方向块大小为256
            elif K_dim >= 64:  # 如果K维度>=64 # 如果K维度>=64
                default_block_k = 64  # 默认K方向块大小为64 # 默认K方向块大小为64
            else:  # 否则 # 否则
                default_block_k = max(16, K_dim)  # 默认K方向块大小为max(16, K_dim) # 默认K方向块大小为max(16, K_dim)
            cfg = {  # 创建默认配置 # 创建默认配置
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": min(64, max(16, N_dim)),
                "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        return cfg  # 返回配置 # 返回配置

    def _align_block_size(  # 块大小对齐函数 # 块大小对齐函数
        topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
        block_size: int,  # 块大小 # 块大小
        num_experts: int,  # 专家数量 # 专家数量
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回排序token ID、专家ID和填充后token数 # 返回排序token ID、专家ID和填充后token数
        # The native align kernel consumes num_experts + 1 internally for its
        # sentinel bucket, so the 1024-expert boundary must use the fallback path.
        # 原生对齐内核内部使用num_experts + 1作为哨兵桶，因此1024专家边界
        # 必须使用回退路径。
        if num_experts < 1024:  # 如果专家数量<1024 # 如果专家数量<1024
            from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (  # 导入原生MoE块大小对齐函数 # 导入原生MoE块大小对齐函数
                moe_align_block_size as native_moe_align_block_size,
            )

            return native_moe_align_block_size(topk_ids, block_size, num_experts)  # 使用原生对齐 # 使用原生对齐
        return _align_block_size_large(topk_ids, block_size, num_experts)  # 使用大规模对齐 # 使用大规模对齐

    def _get_routing(  # 获取路由信息 # 获取路由信息
        topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
        token_lora_mapping: torch.Tensor,  # token到LoRA的映射张量 # token到LoRA的映射张量
        num_experts: int,  # 专家数量 # 专家数量
        shared_outer: bool,  # 是否共享外层 # 是否共享外层
        block_size: int,  # 块大小 # 块大小
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回排序token ID、专家ID、填充后token数和掩码 # 返回排序token ID、专家ID、填充后token数和掩码
        # Check routing_cache for cross-call reuse (gate_up and down share routing)
        # 检查路由缓存以实现跨调用复用（gate_up和down共享路由）
        cache_key = (num_experts, shared_outer, block_size)  # 构建缓存键 # 构建缓存键
        if routing_cache is not None:  # 如果路由缓存不为空 # 如果路由缓存不为空
            cached = routing_cache.get(cache_key)  # 获取缓存结果 # 获取缓存结果
            if cached is not None:  # 如果缓存命中 # 如果缓存命中
                return cached  # 返回缓存结果 # 返回缓存结果

        virtual_topk_ids, token_lora_mask, virtual_num_experts = (  # 计算虚拟topk IDs # 计算虚拟topk IDs
            _fused_virtual_topk_ids(
                topk_ids, token_lora_mapping, num_experts, shared_outer, max_loras
            )
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(  # 对齐块大小 # 对齐块大小
            virtual_topk_ids,
            block_size=block_size,
            num_experts=virtual_num_experts,
        )
        # _align_block_size uses a worst-case padded allocation. Trim the routing buffers
        # to a tighter upper bound so we keep the real routed work but drop unused padding
        # _align_block_size使用最坏情况的填充分配。裁剪路由缓冲区到更紧的上限，
        # 以保留实际路由工作但丢弃未使用的填充
        num_tokens = topk_ids.numel()  # 获取token总数 # 获取token总数
        max_nonempty = min(num_tokens, virtual_num_experts)  # 计算最大非空专家数 # 计算最大非空专家数
        tight_padded = (  # 计算紧凑填充大小 # 计算紧凑填充大小
            triton.cdiv(num_tokens + max_nonempty * (block_size - 1), block_size)
            * block_size
        )
        sorted_token_ids = sorted_token_ids[:tight_padded]  # 裁剪排序token ID # 裁剪排序token ID
        expert_ids = expert_ids[: tight_padded // block_size]  # 裁剪专家ID # 裁剪专家ID
        expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)  # 清洗专家ID # 清洗专家ID
        result = (  # 构建结果元组 # 构建结果元组
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
        )

        if routing_cache is not None:  # 如果路由缓存不为空 # 如果路由缓存不为空
            routing_cache[cache_key] = result  # 存入缓存 # 存入缓存

        return result  # 返回路由结果 # 返回路由结果

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (  # 导入融合MoE Triton内核 # 导入融合MoE Triton内核
        invoke_fused_moe_kernel,
    )

    lora_a_virtual = _merge_lora_expert_weight(lora_a)  # 合并LoRA A专家权重 # 合并LoRA A专家权重
    lora_b_virtual = _merge_lora_expert_weight(lora_b)  # 合并LoRA B专家权重 # 合并LoRA B专家权重
    num_experts_a = lora_a.shape[1]  # 获取LoRA A的专家数 # 获取LoRA A的专家数
    num_experts_b = lora_b.shape[1]  # 获取LoRA B的专家数 # 获取LoRA B的专家数

    intermediate = torch.zeros(  # 创建中间结果张量 # 创建中间结果张量
        [token_lora_mapping.shape[0], topk_ids.shape[1], max_lora_rank],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    a_stage_config = _get_stage_config(lora_a_virtual, input_top_k)  # 获取LoRA A阶段配置 # 获取LoRA A阶段配置
    (  # 获取LoRA A阶段路由信息 # 获取LoRA A阶段路由信息
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_a,
        experts_shared_outer_loras_a,
        a_stage_config["BLOCK_SIZE_M"],
    )

    _invoke_moe_lora_shrink_splitk(  # 调用MoE LoRA收缩Split-K内核 # 调用MoE LoRA收缩Split-K内核
        hidden_states,
        lora_a_virtual,
        intermediate.view(-1, max_lora_rank),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        input_top_k,
        a_stage_config,
    )

    b_stage_config = _get_stage_config(lora_b_virtual, 1)  # 获取LoRA B阶段配置 # 获取LoRA B阶段配置
    (  # 获取LoRA B阶段路由信息 # 获取LoRA B阶段路由信息
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_b,
        experts_shared_outer_loras_b,
        b_stage_config["BLOCK_SIZE_M"],
    )

    invoke_fused_moe_kernel(  # 调用融合MoE内核 # 调用融合MoE内核
        intermediate.view(-1, max_lora_rank),  # 中间结果（展平） # 中间结果（展平）
        lora_b_virtual,  # LoRA B虚拟权重 # LoRA B虚拟权重
        None,  # 无激活函数 # 无激活函数
        output,  # 输出张量 # 输出张量
        None,  # 无偏置 # 无偏置
        None,  # 无B偏置 # 无B偏置
        None,  # 无A偏置 # 无A偏置
        topk_weights,  # topk权重 # topk权重
        topk_ids,  # topk ID # topk ID
        sorted_token_ids,  # 排序后的token ID # 排序后的token ID
        expert_ids,  # 专家ID # 专家ID
        num_tokens_post_padded,  # 填充后token数 # 填充后token数
        mul_routed_weight,  # 是否乘以路由权重 # 是否乘以路由权重
        1,  # top-k值为1 # top-k值为1
        b_stage_config,  # LoRA B阶段配置 # LoRA B阶段配置
        tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,  # 数据类型 # 数据类型
        False,  # 不使用激活函数 # 不使用激活函数
        False,  # 不使用专家偏置 # 不使用专家偏置
        False,  # 不使用A偏置 # 不使用A偏置
        False,  # 不使用B偏置 # 不使用B偏置
        False,  # 不使用重排序 # 不使用重排序
        None,  # 无自定义内核 # 无自定义内核
        fuse_add_to_output=True,  # 融合添加到输出 # 融合添加到输出
        add_output_mask=token_lora_mask,  # 添加输出掩码 # 添加输出掩码
        router_topk=topk_ids.shape[1],  # 路由器top-k值 # 路由器top-k值
    )


def _merged_experts_fused_moe_lora_add_op(  # 合并专家融合MoE LoRA添加算子 # 合并专家融合MoE LoRA添加算子
    output: torch.Tensor,  # 输出张量 # 输出张量
    hidden_states: torch.Tensor,  # 隐藏状态张量 # 隐藏状态张量
    lora_a: torch.Tensor,  # LoRA A权重张量 # LoRA A权重张量
    lora_b: torch.Tensor,  # LoRA B权重张量 # LoRA B权重张量
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    topk_weights: torch.Tensor,  # topk权重张量 # topk权重张量
    token_lora_mapping: torch.Tensor,  # token到LoRA的映射张量 # token到LoRA的映射张量
    mul_routed_weight: bool,  # 是否乘以路由权重 # 是否乘以路由权重
    experts_shared_outer_loras_a: bool,  # 专家是否共享外层LoRA A # 专家是否共享外层LoRA A
    experts_shared_outer_loras_b: bool,  # 专家是否共享外层LoRA B # 专家是否共享外层LoRA B
) -> None:  # 无返回值 # 无返回值
    _merged_experts_fused_moe_lora_add_impl(  # 调用实现函数 # 调用实现函数
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
    )


from sglang.srt.utils.common import direct_register_custom_op  # 导入自定义算子注册工具 # 导入自定义算子注册工具

direct_register_custom_op(  # 注册自定义算子 # 注册自定义算子
    op_name="merged_experts_fused_moe_lora_add",  # 算子名称 # 算子名称
    op_func=_merged_experts_fused_moe_lora_add_op,  # 算子函数 # 算子函数
    mutates_args=["output"],  # 可变参数列表 # 可变参数列表
    fake_impl=_merged_experts_fused_moe_lora_add_fake,  # 伪实现函数 # 伪实现函数
)


def merged_experts_fused_moe_lora_add(  # 合并专家融合MoE LoRA添加公共API # 合并专家融合MoE LoRA添加公共API
    output: torch.Tensor,  # 输出张量 # 输出张量
    hidden_states: torch.Tensor,  # 隐藏状态张量 # 隐藏状态张量
    lora_a: torch.Tensor,  # LoRA A权重张量 # LoRA A权重张量
    lora_b: torch.Tensor,  # LoRA B权重张量 # LoRA B权重张量
    topk_ids: torch.Tensor,  # topk ID张量 # topk ID张量
    topk_weights: torch.Tensor,  # topk权重张量 # topk权重张量
    token_lora_mapping: torch.Tensor,  # token到LoRA的映射张量 # token到LoRA的映射张量
    mul_routed_weight: bool,  # 是否乘以路由权重 # 是否乘以路由权重
    experts_shared_outer_loras_a: bool,  # 专家是否共享外层LoRA A # 专家是否共享外层LoRA A
    experts_shared_outer_loras_b: bool,  # 专家是否共享外层LoRA B # 专家是否共享外层LoRA B
    routing_cache: dict | None = None,  # 路由缓存字典 # 路由缓存字典
) -> None:  # 无返回值 # 无返回值
    """Public API: wraps the registered op with routing_cache support."""
    """公共API：封装已注册的算子，支持routing_cache。"""
    _merged_experts_fused_moe_lora_add_impl(  # 调用实现函数 # 调用实现函数
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
        routing_cache,
    )
