# LoRA MoE运行器的钩子机制实现
# 提供LoRA钩子闭包，在MoE流水线的gate_up投影后和down投影后注入LoRA增量
# 支持虚拟专家模式和经典LoRA对齐模式，包含Naive CPU回退路径
# Copyright 2023-2025 SGLang Team
# 版权所有 2023-2025 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache许可证2.0版（"许可证"）授权；
# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则您不得使用此文件。
# You may obtain a copy of the License at
# 您可以在以下地址获取许可证：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据适用法律或书面同意，按"原样"分发许可证下的软件，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不附带任何明示或暗示的保证或条件。
# See the License for the specific language governing permissions and
# 请参阅许可证以获取管理权限和限制的具体语言。
# limitations under the License.
# 许可证下的限制。
# ==============================================================================

"""LoRA hooks for MoE runners.
# MoE运行器的LoRA钩子。

LoRA deltas are injected at two points in the MoE pipeline:
# LoRA增量在MoE流水线的两个位置注入：
1. After gate_up projection, BEFORE activation
# 1. gate_up投影之后，激活之前
2. After down projection, BEFORE final reduction
# 2. down投影之后，最终归约之前

This module provides hook closures that any MoE backend can call at those points,
# 本模块提供钩子闭包，任何MoE后端都可以在这些位置调用，
without needing a per-backend LoRA runner subclass.
# 无需为每个后端创建LoRA运行器子类。
"""

from __future__ import annotations # 启用延迟注解评估

from dataclasses import dataclass # 导入数据类装饰器
from typing import Callable # 导入可调用类型

import torch # 导入PyTorch库

from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode # 导入CUDA Graph捕获模式检测
from sglang.srt.utils import is_cuda, is_hip, is_xpu, next_power_of_2 # 导入平台检测和2的幂工具

_is_cuda = is_cuda() # 检测是否为CUDA环境
_is_hip = is_hip() # 检测是否为HIP（AMD ROCm）环境
_is_xpu = is_xpu() # 检测是否为XPU（Intel）环境

if _is_cuda or _is_hip or _is_xpu: # 如果是CUDA/HIP/XPU环境
    from sglang.jit_kernel.moe_lora_align import moe_lora_align_block_size # 导入MoE LoRA块大小对齐内核


def _get_moe_lora_block_config(max_lora_rank: int) -> dict: # 根据最大LoRA秩计算块大小配置
    """Compute rank-aware block sizes for MoE LoRA kernels.
    # 计算MoE LoRA内核的秩感知块大小。

    Shrink: output dim is the rank -> cap BLOCK_SIZE_N to avoid waste.
    # Shrink：输出维度是秩 -> 限制BLOCK_SIZE_N以避免浪费。
    Expand: input dim is the rank -> cap BLOCK_SIZE_K similarly.
    # Expand：输入维度是秩 -> 同样限制BLOCK_SIZE_K。
    """
    if max_lora_rank <= 0: # 如果最大LoRA秩<=0
        rank_pow2 = 64 # 默认2的幂为64
    else: # 否则
        rank_pow2 = next_power_of_2(max_lora_rank) # 计算不小于最大秩的2的幂

    shrink_n = min(64, rank_pow2) # shrink的BLOCK_SIZE_N取秩的2的幂和64的较小值
    expand_k = max(16, min(64, rank_pow2)) # expand的BLOCK_SIZE_K取16到64之间

    return { # 返回块大小配置字典
        "shrink_block_size_n": shrink_n, # shrink的N块大小
        "expand_block_size_k": expand_k, # expand的K块大小
    }


_SPARSITY_FACTOR = 8 # 稀疏因子，用于决定是否使用Naive CPU回退


def _naive_moe_lora_align_block_size( # 在CPU上构建LoRA token-专家对齐（用于小批次）
    topk_ids: torch.Tensor, # topk专家ID
    seg_indptr: torch.Tensor, # 段索引指针
    req_to_lora: torch.Tensor, # 请求到LoRA的映射
    num_experts: int, # 专家数量
    block_size_m: int, # M维块大小
    max_loras: int, # 最大LoRA数量
    max_num_tokens_padded: int, # 最大填充token数
    max_num_m_blocks: int, # 最大M块数
    adapter_enabled: torch.Tensor, # 适配器启用状态
    device: torch.device, # 目标设备
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # 返回(sorted_token_ids, expert_ids_out, num_tokens_post_padded)
    """Construct LoRA token-expert alignment on CPU for small batches.
    # 为小批次在CPU上构建LoRA token-专家对齐。

    When the number of tokens is very small, the overhead of launching the
    CUDA-based moe_lora_align_block_size kernel exceeds the actual
    computation. This function builds the same data structures using simple
    Python loops on CPU and transfers the result to GPU in one shot.
    # 当token数量很少时，启动基于CUDA的moe_lora_align_block_size内核的开销
    # 超过了实际计算。此函数使用简单的Python循环在CPU上构建相同的数据结构，
    # 并一次性传输结果到GPU。
    """
    M, top_k = topk_ids.shape # 获取token数M和topk数量
    num_valid_tokens = M * top_k # 计算有效token总数

    sorted_token_ids = torch.full( # 创建排序token ID张量
        (max_loras * max_num_tokens_padded,), # 大小为max_loras * max_num_tokens_padded
        num_valid_tokens, # 填充值（有效token数，用作padding标记）
        dtype=torch.int32, # 32位整数
    )
    expert_ids_out = torch.full((max_loras * max_num_m_blocks,), -1, dtype=torch.int32) # 创建专家ID输出张量，填充-1
    num_tokens_post_padded = torch.zeros(max_loras, dtype=torch.int32) # 创建每LoRA填充后token数张量

    seg_indptr_list = seg_indptr.cpu().tolist() # 将段索引指针转为CPU列表
    req_to_lora_list = req_to_lora.cpu().tolist() # 将请求到LoRA映射转为CPU列表
    topk_ids_list = topk_ids.cpu().tolist() # 将topk ID转为CPU列表
    adapter_enabled_list = adapter_enabled.cpu().tolist() # 将适配器启用状态转为CPU列表

    for lora_id in range(max_loras): # 遍历每个LoRA适配器
        if not adapter_enabled_list[lora_id]: # 如果适配器未启用
            continue # 跳过

        pairs: list[tuple[int, int]] = [] # 存储(专家ID, token索引)对
        for seg_idx in range(len(seg_indptr_list) - 1): # 遍历每个段
            if req_to_lora_list[seg_idx] != lora_id: # 如果段不属于当前LoRA
                continue # 跳过
            start = seg_indptr_list[seg_idx] # 段起始位置
            end = seg_indptr_list[seg_idx + 1] # 段结束位置
            for m in range(start, end): # 遍历段内每个token
                for k in range(top_k): # 遍历每个topk选择
                    pairs.append((topk_ids_list[m][k], m * top_k + k)) # 添加(专家ID, token全局索引)对

        if not pairs: # 如果没有配对
            continue # 跳过

        pairs.sort() # 按专家ID排序

        base_t = lora_id * max_num_tokens_padded # 当前LoRA在sorted_token_ids中的基址
        base_e = lora_id * max_num_m_blocks # 当前LoRA在expert_ids_out中的基址
        pos = 0 # 当前写入位置
        block_idx = 0 # 当前块索引
        i = 0 # 遍历索引
        while i < len(pairs): # 遍历排序后的配对
            cur_expert = pairs[i][0] # 当前专家ID
            group_start = pos # 当前组起始位置
            while i < len(pairs) and pairs[i][0] == cur_expert: # 同一专家的配对
                sorted_token_ids[base_t + pos] = pairs[i][1] # 写入token索引
                pos += 1 # 位置递增
                i += 1 # 索引递增
            group_len = pos - group_start # 当前组长度
            padded_len = ((group_len + block_size_m - 1) // block_size_m) * block_size_m # 计算填充后的长度
            num_blocks = padded_len // block_size_m # 计算块数
            for b in range(num_blocks): # 遍历每个块
                expert_ids_out[base_e + block_idx + b] = cur_expert # 写入专家ID
            block_idx += num_blocks # 更新块索引
            pos = group_start + padded_len # 跳过填充部分

        num_tokens_post_padded[lora_id] = pos # 记录当前LoRA填充后的token数

    return ( # 返回结果元组
        sorted_token_ids.to(device), # 排序token ID（传输到GPU）
        expert_ids_out.to(device), # 专家ID输出（传输到GPU）
        num_tokens_post_padded.to(device), # 填充后token数（传输到GPU）
    )


@dataclass # 数据类装饰器
class LoRAInfo: # LoRA信息和分发数据
    """LoRA weights and dispatch info for MoE computation."""
    # MoE计算的LoRA权重和分发信息。

    # LoRA weights: [num_loras, num_experts_or_1, dim1, dim2]
    # LoRA权重: [LoRA数量, 专家数或1, 维度1, 维度2]
    # When experts_shared_outer_loras=True:
    # 当experts_shared_outer_loras=True时：
    #   gate_up_lora_a: [num_loras, 1, max_rank, hidden_dim] (shared)
    #   gate_up_lora_a: [LoRA数量, 1, 最大秩, 隐藏维度]（共享）
    #   down_lora_b: [num_loras, 1, hidden_dim, max_rank] (shared)
    #   down_lora_b: [LoRA数量, 1, 隐藏维度, 最大秩]（共享）
    gate_up_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts_or_1, max_rank, hidden_dim]
    # [LoRA数量, 专家数或1, 最大秩, 隐藏维度]
    gate_up_lora_b_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, gate_up_dim, max_rank]
    # [LoRA数量, 专家数, gate_up维度, 最大秩]
    down_lora_a_weights: (
        torch.Tensor
    )  # [num_loras, num_experts, max_rank, intermediate_dim]
    # [LoRA数量, 专家数, 最大秩, 中间维度]
    down_lora_b_weights: (
        torch.Tensor
    )  # [num_loras, num_experts_or_1, hidden_dim, max_rank]
    # [LoRA数量, 专家数或1, 隐藏维度, 最大秩]

    # Indice pointers of each segment in shape (num_segments + 1, )
    # 每个段的索引指针，形状为(num_segments + 1,)
    seg_indptr: torch.Tensor

    # The index of lora adapter used by each segment, in shape (num_segments,)
    # 每个段使用的LoRA适配器索引，形状为(num_segments,)
    req_to_lora: torch.Tensor

    # LoRA config per adapter
    # 每个适配器的LoRA配置
    lora_ranks: torch.Tensor  # [num_loras]
    # [LoRA数量] 每个LoRA的秩
    adapter_enabled: torch.Tensor  # [num_loras] - which adapters are enabled
    # [LoRA数量] - 哪些适配器已启用
    token_lora_mapping: torch.Tensor  # [num_tokens] - adapter used by each token
    # [token数] - 每个token使用的适配器
    max_lora_rank: int  # Maximum LoRA rank across all adapters
    # 所有适配器中的最大LoRA秩

    num_experts: int # 专家数量
    has_active_lora: bool = True # 是否有活跃的LoRA
    experts_shared_outer_loras: bool = False # 外部LoRA是否在专家间共享
    cg_buffers: dict | None = None # CUDA Graph缓冲区

    fully_sharded: bool = False # 是否完全分片
    tp_size: int = 1 # 张量并行大小
    tp_rank: int = 0 # 张量并行秩
    hidden_size: int = 0 # 隐藏层大小
    lora_use_virtual_experts: bool = False # 是否使用虚拟专家LoRA


@dataclass # 数据类装饰器
class LoRAHooks: # LoRA钩子回调
    """Hook callbacks for injecting LoRA deltas into the MoE pipeline."""
    # 用于向MoE流水线注入LoRA增量的钩子回调。

    after_gate_up: ( # gate_up投影后的钩子
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None
    ) = None # 可选回调：输入(hidden_states, intermediate_cache, topk_weights, topk_ids)
    after_down: ( # down投影后的钩子
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] | None
    ) = None # 可选回调：输入(intermediate_input, intermediate_cache, topk_weights, topk_ids)


def _compute_lora_alignment( # 计算LoRA对齐张量（经典路径，非虚拟专家）
    topk_ids: torch.Tensor, # topk专家ID
    lora_info: LoRAInfo, # LoRA信息
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # 返回(sorted_token_ids_reshaped, expert_ids_reshaped, num_tokens_post_padded_lora, lora_ids)
    """Compute LoRA alignment tensors for the non-virtual-expert (classic) path.
    # 计算非虚拟专家（经典）路径的LoRA对齐张量。

    Returns: (sorted_token_ids_reshaped, expert_ids_reshaped, num_tokens_post_padded_lora, lora_ids)
    # 返回：(重塑后的排序token ID, 重塑后的专家ID, 填充后token数, LoRA ID)
    """
    cg = lora_info.cg_buffers if get_is_capture_mode() else None # 获取CUDA Graph缓冲区（仅捕获模式）
    shrink_config = {"BLOCK_SIZE_M": 64} # shrink块大小配置
    M = topk_ids.shape[0] # 获取token数
    block_size_m = shrink_config["BLOCK_SIZE_M"] # 获取块大小
    max_loras = len(lora_info.lora_ranks) # 获取LoRA数量

    max_num_tokens_padded = topk_ids.numel() + lora_info.num_experts * (
        block_size_m - 1
    ) # 计算最大填充token数（含专家对齐填充）
    max_num_tokens_padded = (
        (max_num_tokens_padded + block_size_m - 1) // block_size_m
    ) * block_size_m # 向上对齐到块大小的倍数
    max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m # 计算最大M块数

    device = topk_ids.device # 获取设备

    use_naive = ( # 判断是否使用Naive CPU回退
        cg is None # 非CUDA Graph模式
        and M * topk_ids.shape[1] * _SPARSITY_FACTOR
        <= lora_info.num_experts * max_loras # 且token数*稀疏因子<=专家数*LoRA数
    )

    if use_naive: # 使用Naive CPU回退
        sorted_token_ids_lora, expert_ids_lora, num_tokens_post_padded_lora = (
            _naive_moe_lora_align_block_size( # 调用CPU对齐函数
                topk_ids,
                lora_info.seg_indptr,
                lora_info.req_to_lora,
                int(lora_info.num_experts),
                int(block_size_m),
                int(max_loras),
                int(max_num_tokens_padded),
                int(max_num_m_blocks),
                lora_info.adapter_enabled,
                device,
            )
        )
        lora_ids = torch.arange(max_loras, dtype=torch.int32, device=device) # 创建LoRA ID范围
    else: # 使用CUDA内核对齐
        if cg is not None: # CUDA Graph模式
            sorted_token_ids_lora = cg["sorted_token_ids_lora"][
                : max_loras * max_num_tokens_padded
            ] # 从预分配缓冲区获取排序token ID
            expert_ids_lora = cg["expert_ids_lora"][: max_loras * max_num_m_blocks] # 从预分配缓冲区获取专家ID
            num_tokens_post_padded_lora = cg["num_tokens_post_padded_lora"][:max_loras] # 从预分配缓冲区获取填充后token数
        else: # 非CUDA Graph模式
            sorted_token_ids_lora = torch.empty( # 分配排序token ID张量
                (max_loras * max_num_tokens_padded,),
                dtype=torch.int32,
                device=device,
            )
            expert_ids_lora = torch.empty( # 分配专家ID张量
                (max_loras * max_num_m_blocks,),
                dtype=torch.int32,
                device=device,
            )
            num_tokens_post_padded_lora = torch.empty( # 分配填充后token数张量
                (max_loras,), dtype=torch.int32, device=device
            )

        if cg is not None and "lora_ids" in cg: # CUDA Graph模式且缓冲区中有lora_ids
            lora_ids = cg["lora_ids"][:max_loras] # 从预分配缓冲区获取LoRA ID
        else: # 否则
            lora_ids = torch.arange(max_loras, dtype=torch.int32, device=device) # 创建LoRA ID范围

        moe_lora_align_block_size( # 调用CUDA内核进行LoRA块大小对齐
            topk_ids, # topk专家ID
            lora_info.seg_indptr, # 段索引指针
            lora_info.req_to_lora, # 请求到LoRA映射
            int(lora_info.num_experts), # 专家数量
            int(block_size_m), # M维块大小
            int(max_loras), # 最大LoRA数量
            int(max_num_tokens_padded), # 最大填充token数
            int(max_num_m_blocks), # 最大M块数
            sorted_token_ids_lora, # 排序token ID（输出）
            expert_ids_lora, # 专家ID（输出）
            num_tokens_post_padded_lora, # 填充后token数（输出）
            lora_info.adapter_enabled, # 适配器启用状态
            lora_ids, # LoRA ID
            cumsum_buffer=cg.get("cumsum_buffer") if cg is not None else None, # 累加和缓冲区
            token_mask=cg.get("token_mask") if cg is not None else None, # token掩码
        )

    return ( # 返回对齐结果
        sorted_token_ids_lora.view(max_loras, -1), # 重塑排序token ID为(max_loras, -1)
        expert_ids_lora.view(max_loras, -1), # 重塑专家ID为(max_loras, -1)
        num_tokens_post_padded_lora, # 填充后token数
        lora_ids, # LoRA ID
    )


def _add_lora_gate_up_delta( # 向gate_up中间缓存添加LoRA增量（原地操作）
    hidden_states: torch.Tensor, # 隐藏状态
    intermediate_cache: torch.Tensor, # gate_up中间缓存
    topk_weights: torch.Tensor, # topk权重
    topk_ids: torch.Tensor, # topk专家ID
    lora_info: LoRAInfo, # LoRA信息
    token_lora_mapping: torch.Tensor | None, # token到LoRA的映射
    sorted_token_ids_reshaped: torch.Tensor | None, # 重塑后的排序token ID
    expert_ids_reshaped: torch.Tensor | None, # 重塑后的专家ID
    num_tokens_post_padded_lora: torch.Tensor | None, # 填充后token数
    lora_ids: torch.Tensor | None, # LoRA ID
    routing_cache: dict | None = None, # 路由缓存
) -> None: # 无返回值（原地修改）
    """Add LoRA gate_up delta to intermediate_cache in-place."""
    # 原地向intermediate_cache添加LoRA gate_up增量。
    from sglang.srt.lora.triton_ops import ( # 导入Triton LoRA算子
        fused_moe_lora, # 融合MoE LoRA
        merged_experts_fused_moe_lora_add, # 合并专家融合MoE LoRA加法
    )

    if lora_info is None or lora_info.max_lora_rank == 0: # 如果没有LoRA信息或秩为0
        return # 直接返回
    if not get_is_capture_mode() and not lora_info.has_active_lora: # 非捕获模式且无活跃LoRA
        return # 直接返回

    M, top_k, gate_up_dim = intermediate_cache.shape # 获取中间缓存的形状
    r = lora_info.max_lora_rank # 获取最大LoRA秩
    gate_up_a = lora_info.gate_up_lora_a_weights # 获取gate_up LoRA A权重
    gate_up_b = lora_info.gate_up_lora_b_weights # 获取gate_up LoRA B权重

    if lora_info.experts_shared_outer_loras and not lora_info.lora_use_virtual_experts: # 外部LoRA共享且非虚拟专家
        gate_up_a = gate_up_a.expand(-1, lora_info.num_experts, -1, -1) # 扩展A权重到所有专家

    # Detect gated vs non-gated from A buffer rank dimension.
    # Gated: A has 2*r rows (gate + up). Non-gated: A has 1*r rows (w1 only).
    # 从A缓冲区的秩维度检测是否为门控模式。
    # 门控：A有2*r行（gate + up）。非门控：A有1*r行（仅w1）。
    is_gated = gate_up_a.shape[2] > r # 判断是否为门控模式
    if is_gated: # 门控模式
        inter_size = gate_up_b.shape[2] // 2 # 计算中间维度大小
        lora_a_stacked = [gate_up_a[:, :, :r, :], gate_up_a[:, :, r : 2 * r, :]] # 堆叠A权重：gate和up分开
        lora_b_stacked = [ # 堆叠B权重：gate和up分开
            gate_up_b[:, :, :inter_size, :],
            gate_up_b[:, :, inter_size:, :],
        ]
    else: # 非门控模式
        lora_a_stacked = [gate_up_a] # A权重不分割
        lora_b_stacked = [gate_up_b] # B权重不分割

    if lora_info.lora_use_virtual_experts: # 使用虚拟专家模式
        merged_experts_fused_moe_lora_add( # 调用合并专家融合LoRA加法
            output=intermediate_cache, # 输出为中间缓存（原地修改）
            hidden_states=hidden_states, # 隐藏状态
            lora_a=gate_up_a, # gate_up LoRA A权重
            lora_b=gate_up_b, # gate_up LoRA B权重
            topk_ids=topk_ids, # topk专家ID
            topk_weights=topk_weights, # topk权重
            token_lora_mapping=token_lora_mapping, # token到LoRA映射
            mul_routed_weight=False, # 不乘路由权重
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras, # A权重是否专家共享
            experts_shared_outer_loras_b=False, # B权重不共享
            routing_cache=routing_cache, # 路由缓存
        )
    else: # 经典模式
        blk = _get_moe_lora_block_config(r) # 获取块大小配置
        fused_moe_lora( # 调用融合MoE LoRA算子
            output=intermediate_cache, # 输出为中间缓存（原地修改）
            qcurr_hidden_states=hidden_states, # 当前隐藏状态
            lora_a_stacked=lora_a_stacked, # 堆叠的A权重
            lora_b_stacked=lora_b_stacked, # 堆叠的B权重
            topk_weights=topk_weights, # topk权重
            sorted_token_ids=sorted_token_ids_reshaped, # 排序token ID
            expert_ids=expert_ids_reshaped, # 专家ID
            num_tokens_post_padded=num_tokens_post_padded_lora, # 填充后token数
            max_lora_rank=r, # 最大LoRA秩
            top_k_num=top_k, # topk数量
            lora_ids=lora_ids, # LoRA ID
            adapter_enabled=lora_info.adapter_enabled, # 适配器启用状态
            shrink_block_size_m=64, # shrink的M块大小
            shrink_block_size_n=blk["shrink_block_size_n"], # shrink的N块大小
            shrink_block_size_k=64, # shrink的K块大小
            shrink_group_size_m=8, # shrink的M组大小
            shrink_num_warps=4, # shrink的warp数
            shrink_num_stages=2, # shrink的流水线级数
            shrink_split_k=1, # shrink的split-k值
            expand_block_size_m=64, # expand的M块大小
            expand_block_size_n=64, # expand的N块大小
            expand_block_size_k=blk["expand_block_size_k"], # expand的K块大小
            expand_group_size_m=8, # expand的M组大小
            expand_num_warps=4, # expand的warp数
            expand_num_stages=2, # expand的流水线级数
            expand_split_k=1, # expand的split-k值
            fully_sharded=lora_info.fully_sharded, # 是否完全分片
        )


def _add_lora_down_delta( # 向down中间缓存添加LoRA增量（原地操作）
    intermediate_input: torch.Tensor, # 中间层输入（激活后）
    intermediate_cache: torch.Tensor, # down中间缓存
    topk_weights: torch.Tensor, # topk权重
    topk_ids: torch.Tensor, # topk专家ID
    lora_info: LoRAInfo, # LoRA信息
    token_lora_mapping: torch.Tensor | None, # token到LoRA的映射
    sorted_token_ids_reshaped: torch.Tensor | None, # 重塑后的排序token ID
    expert_ids_reshaped: torch.Tensor | None, # 重塑后的专家ID
    num_tokens_post_padded_lora: torch.Tensor | None, # 填充后token数
    lora_ids: torch.Tensor | None, # LoRA ID
    routing_cache: dict | None = None, # 路由缓存
) -> None: # 无返回值（原地修改）
    """Add LoRA down delta to intermediate_cache in-place."""
    # 原地向intermediate_cache添加LoRA down增量。
    from sglang.srt.lora.triton_ops import ( # 导入Triton LoRA算子
        fused_moe_lora, # 融合MoE LoRA
        merged_experts_fused_moe_lora_add, # 合并专家融合MoE LoRA加法
    )

    if lora_info.max_lora_rank == 0: # 如果最大LoRA秩为0
        return # 直接返回

    M, top_k, hidden_dim = intermediate_cache.shape # 获取中间缓存的形状

    down_lora_a = lora_info.down_lora_a_weights # 获取down LoRA A权重
    down_lora_b = lora_info.down_lora_b_weights # 获取down LoRA B权重
    if lora_info.experts_shared_outer_loras and not lora_info.lora_use_virtual_experts: # 外部LoRA共享且非虚拟专家
        down_lora_b = down_lora_b.expand(-1, lora_info.num_experts, -1, -1) # 扩展B权重到所有专家

    if lora_info.fully_sharded and lora_info.tp_size > 1: # 完全分片且张量并行>1
        shard_size = lora_info.hidden_size // lora_info.tp_size # 计算分片大小
        offset = shard_size * lora_info.tp_rank # 计算当前分片偏移
    else: # 非分片
        offset = 0 # 偏移为0

    if lora_info.lora_use_virtual_experts: # 使用虚拟专家模式
        merged_experts_fused_moe_lora_add( # 调用合并专家融合LoRA加法
            output=intermediate_cache, # 输出为中间缓存（原地修改）
            hidden_states=intermediate_input, # 隐藏状态为中间层输入
            lora_a=down_lora_a, # down LoRA A权重
            lora_b=down_lora_b, # down LoRA B权重
            topk_ids=topk_ids, # topk专家ID
            topk_weights=topk_weights, # topk权重
            token_lora_mapping=token_lora_mapping, # token到LoRA映射
            mul_routed_weight=True, # 乘路由权重
            experts_shared_outer_loras_a=False, # A权重不共享
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras, # B权重是否专家共享
            routing_cache=routing_cache, # 路由缓存
        )
    else: # 经典模式
        blk = _get_moe_lora_block_config(lora_info.max_lora_rank) # 获取块大小配置
        fused_moe_lora( # 调用融合MoE LoRA算子
            output=intermediate_cache, # 输出为中间缓存（原地修改）
            qcurr_hidden_states=intermediate_input, # 当前隐藏状态为中间层输入
            lora_a_stacked=[down_lora_a], # A权重列表（单元素）
            lora_b_stacked=[down_lora_b], # B权重列表（单元素）
            topk_weights=topk_weights, # topk权重
            sorted_token_ids=sorted_token_ids_reshaped, # 排序token ID
            expert_ids=expert_ids_reshaped, # 专家ID
            num_tokens_post_padded=num_tokens_post_padded_lora, # 填充后token数
            max_lora_rank=lora_info.max_lora_rank, # 最大LoRA秩
            top_k_num=top_k, # topk数量
            lora_ids=lora_ids, # LoRA ID
            adapter_enabled=lora_info.adapter_enabled, # 适配器启用状态
            shrink_block_size_m=64, # shrink的M块大小
            shrink_block_size_n=blk["shrink_block_size_n"], # shrink的N块大小
            shrink_block_size_k=64, # shrink的K块大小
            shrink_group_size_m=8, # shrink的M组大小
            shrink_num_warps=4, # shrink的warp数
            shrink_num_stages=2, # shrink的流水线级数
            shrink_split_k=1, # shrink的split-k值
            expand_block_size_m=64, # expand的M块大小
            expand_block_size_n=64, # expand的N块大小
            expand_block_size_k=blk["expand_block_size_k"], # expand的K块大小
            expand_group_size_m=8, # expand的M组大小
            expand_num_warps=4, # expand的warp数
            expand_num_stages=2, # expand的流水线级数
            expand_split_k=1, # expand的split-k值
            mul_routed_weight=True, # 乘路由权重
            fully_sharded=lora_info.fully_sharded, # 是否完全分片
            offset=offset, # 分片偏移
        )


def build_lora_hooks( # 构建LoRA钩子闭包，用于注入到任意MoE运行器
    hidden_states: torch.Tensor, # 隐藏状态
    lora_info: LoRAInfo, # LoRA信息
    topk_ids: torch.Tensor, # topk专家ID
) -> LoRAHooks: # 返回LoRA钩子
    """Build LoRA hook closures for injection into any MoE runner.
    # 构建LoRA钩子闭包，用于注入到任意MoE运行器。

    Computes token_lora_mapping and alignment tensors once, then returns
    closures that capture them for the two injection points.
    # 一次性计算token_lora_mapping和对齐张量，然后返回捕获它们的两个注入点闭包。
    """
    if lora_info is None or lora_info.max_lora_rank == 0: # 如果没有LoRA信息或秩为0
        return LoRAHooks() # 返回空钩子
    # Skip alignment/mapping work entirely when the batch has no active adapter.
    # When the batch has no active adapters, skip all alignment/mapping work.
    # 当批次没有活跃适配器时，完全跳过对齐/映射工作。
    # During CUDA graph capture we still need to record the kernels into the
    # graph (adapter_enabled is all-zero, kernels early-exit on GPU).
    # 在CUDA Graph捕获期间，我们仍需将内核录制到图中
    # （adapter_enabled全为零，内核在GPU上提前退出）。
    if not get_is_capture_mode() and not lora_info.has_active_lora: # 非捕获模式且无活跃LoRA
        return LoRAHooks() # 返回空钩子

    # Compute alignment / mapping (once, shared by both hooks)
    # 计算对齐/映射（一次性，两个钩子共享）
    token_lora_mapping: torch.Tensor | None = None # 初始化token到LoRA映射
    sorted_token_ids_reshaped: torch.Tensor | None = None # 初始化重塑后的排序token ID
    expert_ids_reshaped: torch.Tensor | None = None # 初始化重塑后的专家ID
    num_tokens_post_padded_lora: torch.Tensor | None = None # 初始化填充后token数
    lora_ids: torch.Tensor | None = None # 初始化LoRA ID

    if lora_info.lora_use_virtual_experts: # 使用虚拟专家模式
        token_lora_mapping = lora_info.token_lora_mapping # 直接使用LoRA信息中的映射
    else: # 经典模式
        (
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
        ) = _compute_lora_alignment(topk_ids, lora_info) # 计算LoRA对齐张量

    # Shared routing cache: gate_up and down reuse routing for same (num_experts, shared_outer, block_size)
    # 共享路由缓存：gate_up和down为相同的(专家数, 共享外部, 块大小)重用路由
    routing_cache: dict = {} # 初始化路由缓存

    def after_gate_up( # gate_up投影后的钩子闭包
        hidden_states: torch.Tensor, # 隐藏状态
        intermediate_cache1: torch.Tensor, # gate_up中间缓存
        topk_weights: torch.Tensor, # topk权重
        topk_ids: torch.Tensor, # topk专家ID
    ) -> None: # 无返回值
        _add_lora_gate_up_delta( # 调用gate_up增量添加函数
            hidden_states,
            intermediate_cache1,
            topk_weights,
            topk_ids,
            lora_info,
            token_lora_mapping,
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
            routing_cache=routing_cache, # 传入路由缓存
        )

    def after_down( # down投影后的钩子闭包
        intermediate_input: torch.Tensor, # 中间层输入
        intermediate_cache3: torch.Tensor, # down中间缓存
        topk_weights: torch.Tensor, # topk权重
        topk_ids: torch.Tensor, # topk专家ID
    ) -> None: # 无返回值
        _add_lora_down_delta( # 调用down增量添加函数
            intermediate_input,
            intermediate_cache3,
            topk_weights,
            topk_ids,
            lora_info,
            token_lora_mapping,
            sorted_token_ids_reshaped,
            expert_ids_reshaped,
            num_tokens_post_padded_lora,
            lora_ids,
            routing_cache=routing_cache, # 传入路由缓存
        )

    return LoRAHooks(after_gate_up=after_gate_up, after_down=after_down) # 返回包含两个钩子的LoRAHooks