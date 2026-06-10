# DeepEP Waterfill模块：将共享专家作为第9个路由专家，调度到负载最低的rank
# Copyright 2023-2026 SGLang Team
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
"""DeepEP Waterfill: shared expert as 9th routed expert, dispatched to least-loaded rank."""  # DeepEP Waterfill：将共享专家作为第9个路由专家，调度到负载最低的rank

from typing import NamedTuple, Optional, Tuple  # 导入命名元组、可选类型和元组类型

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言
from torch import Tensor  # 导入张量类型

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.moe.topk import StandardTopKOutput  # 导入标准TopK输出类

LOCAL_SHARED_MARKER = -1  # Invalid expert ID; DeepEP ignores expert_id < 0.  # 无效专家ID标记；DeepEP忽略expert_id < 0
_LOCAL_PREF_NUMER = 11  # local-rank preference = 11/10  # 本地rank偏好系数分子 = 11/10
_LOCAL_PREF_DENOM = 10  # 本地rank偏好系数分母


class WaterfillDispatchPlan(NamedTuple):  # Waterfill调度计划命名元组
    """Inputs needed by the fused DeepEP Waterfill expansion path."""  # 融合DeepEP Waterfill扩展路径所需的输入

    # Effective rank load consumed by the fused kernel.  # 融合内核消耗的有效rank负载
    rank_load: Tensor  # rank负载张量
    allow_all_ranks: bool  # 是否允许调度到所有rank
    target_total: int  # 目标总负载


def _empty_expanded(topk_ids: Tensor, topk_weights: Tensor):  # 生成空扩展张量（零令牌批次用）
    """Return empty expanded tensors for zero-token batches."""  # 返回零令牌批次的空扩展张量
    topk, d = topk_ids.shape[1], topk_ids.device  # 获取top-k值和设备
    return (
        torch.empty(0, topk + 1, dtype=topk_ids.dtype, device=d),  # 空扩展专家索引
        torch.empty(0, topk + 1, dtype=topk_weights.dtype, device=d),  # 空扩展权重
    )


@triton.jit
def _count_routed_per_rank_kernel(  # 计算每个rank的路由令牌数的Triton内核
    topk_ids_ptr,  # [num_tokens, topk]  # top-k专家索引指针
    counts_ptr,  # [world_size] output (atomic add)  # 计数输出指针（原子加法）
    num_tokens,  # 令牌总数
    topk: tl.constexpr,  # top-k值（编译时常量）
    experts_per_rank,  # 每个rank的专家数
    world_size: tl.constexpr,  # 世界大小（编译时常量）
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    """Count routed tokens per rank using block-level histogram."""  # 使用块级直方图计算每个rank的路由令牌数
    pid = tl.program_id(0)  # 获取程序ID
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算当前块处理的令牌索引
    mask = token_idx < num_tokens  # 生成有效令牌掩码

    for r in range(world_size):  # 遍历每个rank
        rank_count = tl.zeros([BLOCK_SIZE], dtype=tl.int64)  # 初始化当前rank的计数

        for k in range(topk):  # 遍历每个top-k位置
            expert_id = tl.load(  # 加载专家ID
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)  # 转为int64
            valid = expert_id >= 0  # 判断专家ID是否有效
            target_rank = expert_id // experts_per_rank  # 计算目标rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)  # 钳制到合法范围
            rank_count += tl.where(  # 累加匹配当前rank的计数
                mask & valid & (target_rank == r),
                tl.full([BLOCK_SIZE], 1, dtype=tl.int64),  # 匹配则加1
                tl.zeros([BLOCK_SIZE], dtype=tl.int64),  # 不匹配则加0
            )

        block_total = tl.sum(rank_count)  # 计算块内总计数
        if block_total > 0:  # 如果有计数
            tl.atomic_add(counts_ptr + r, block_total)  # 原子加法更新全局计数


@triton.jit
def _waterfill_expand_kernel(  # 融合Waterfill+扩展的Triton内核
    topk_ids_ptr,  # top-k专家索引指针
    topk_weights_ptr,  # top-k权重指针
    rank_load_ptr,  # rank负载指针
    expanded_ids_ptr,  # 扩展专家索引输出指针
    expanded_weights_ptr,  # 扩展权重输出指针
    num_tokens,  # 令牌总数
    topk: tl.constexpr,  # top-k值（编译时常量）
    old_experts_per_rank,  # 旧每rank专家数
    new_experts_per_rank,  # 新每rank专家数（含共享专家）
    world_size: tl.constexpr,  # 世界大小（编译时常量）
    source_rank,  # 源rank
    shared_weight,  # 共享专家权重
    local_marker,  # 本地标记值
    local_pref_numer,  # 本地偏好系数分子
    local_pref_denom,  # 本地偏好系数分母
    precomputed_target_total,  # 预计算的目标总负载
    ALLOW_ALL_RANKS: tl.constexpr,  # 是否允许所有rank（编译时常量）
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    """Fused waterfill + expand. ID remap: old_id -> old_id + old_id // old_epr."""  # 融合waterfill和扩展。ID重映射：old_id -> old_id + old_id // old_epr
    pid = tl.program_id(0)  # 获取程序ID
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算当前块处理的令牌索引
    mask = token_idx < num_tokens  # 生成有效令牌掩码

    r_idx = tl.arange(0, world_size)  # rank索引范围
    rank_load_vec = tl.load(rank_load_ptr + r_idx, mask=r_idx < world_size, other=0).to(  # 加载所有rank负载
        tl.int64
    )
    total_effective_k = tl.sum(rank_load_vec)  # 计算总有效负载
    total_tokens_global_k = total_effective_k // topk  # 计算全局令牌数
    derived_target_total = (  # 推导目标总负载
        total_effective_k + total_tokens_global_k + world_size - 1
    ) // world_size
    target_total = tl.where(  # 使用预计算值或推导值
        precomputed_target_total > 0,
        precomputed_target_total,  # 预计算值
        derived_target_total,  # 推导值
    )

    # Step 1: Select destination rank for shared expert (waterfill sampling).  # 步骤1：选择共享专家的目标rank（waterfill采样）
    source_count = tl.load(rank_load_ptr + source_rank)  # 加载源rank负载
    best_count = tl.where(mask, source_count, 2**30)  # 初始化最佳计数为源rank计数
    best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)  # 初始化最佳rank为源rank
    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)  # 初始化有效标志
    src_rank_i32 = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int32)  # 源rank的int32表示

    if ALLOW_ALL_RANKS:  # 如果允许所有rank
        candidate_mask = tl.full([BLOCK_SIZE], (1 << world_size) - 1, dtype=tl.int32)  # 所有rank都可作为候选
        for r in range(world_size):  # 遍历所有rank
            target_count = tl.load(rank_load_ptr + r).to(tl.int64)  # 加载目标rank负载
            better = (  # 判断是否更好（考虑本地偏好）
                target_count * local_pref_numer < best_count * local_pref_denom
            ) & mask  # 且在有效范围内
            best_count = tl.where(better, target_count, best_count)  # 更新最佳计数
            best_rank = tl.where(  # 更新最佳rank
                better, tl.full([BLOCK_SIZE], r, dtype=tl.int64), best_rank
            )
    else:  # 否则仅源rank为候选
        candidate_mask = (tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << src_rank_i32).to(  # 仅源rank的位掩码
            tl.int32
        )

    for k in range(topk):  # 遍历每个top-k位置
        expert_id = tl.load(  # 加载专家ID
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)  # 转为int64
        valid = expert_id >= 0  # 判断专家ID是否有效
        has_valid = has_valid | valid  # 更新有效标志

        if not ALLOW_ALL_RANKS:  # 如果不允许所有rank
            target_rank = expert_id // old_experts_per_rank  # 计算目标rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)  # 钳制到合法范围
            target_rank_i32 = target_rank.to(tl.int32)  # 转为int32
            shift_amt = tl.where(valid, target_rank_i32, 0)  # 计算移位量
            bit = tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << shift_amt  # 生成位掩码
            candidate_mask = tl.where(  # 将目标rank加入候选
                valid & mask, candidate_mask | bit, candidate_mask
            )

            target_count = tl.load(  # 加载目标rank负载
                rank_load_ptr + target_rank, mask=mask & valid, other=2**30
            )

            better = (  # 判断是否更好（考虑本地偏好）
                (target_count * local_pref_numer < best_count * local_pref_denom)
                & valid
                & mask
            )
            best_count = tl.where(better, target_count, best_count)  # 更新最佳计数
            best_rank = tl.where(better, target_rank, best_rank)  # 更新最佳rank

    total_w = tl.zeros([BLOCK_SIZE], dtype=tl.int32)  # 初始化总权重
    for r in range(world_size):  # 遍历所有rank计算采样权重
        present = ((candidate_mask >> r) & 1) == 1  # 判断rank是否在候选中
        rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)  # 加载rank负载
        w = tl.where(target_total > rank_load_r, target_total - rank_load_r, 0).to(  # 计算权重（目标-当前负载）
            tl.int32
        )
        w_vec = tl.full([BLOCK_SIZE], w, dtype=tl.int32)  # 广播权重
        w_vec = tl.where(  # 非源rank降低权重（考虑本地偏好）
            src_rank_i32 == r,
            w_vec,  # 源rank保持原权重
            (w_vec * local_pref_denom) // local_pref_numer,  # 非源rank按偏好比例降低
        )
        total_w += tl.where(present, w_vec, 0)  # 累加有效权重

    token_seed = token_idx.to(tl.uint32) ^ (  # 生成令牌随机种子（异或哈希）
        src_rank_i32.to(tl.uint32) * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
    )
    token_seed = token_seed * tl.full([BLOCK_SIZE], 1664525, dtype=tl.uint32) + tl.full(  # 线性同余生成器步骤
        [BLOCK_SIZE], 1013904223, dtype=tl.uint32
    )
    u = tl.where(total_w > 0, token_seed % total_w.to(tl.uint32), 0).to(tl.int32)  # 生成均匀随机数

    chosen = src_rank_i32  # 初始化选中的rank为源rank
    cum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)  # 初始化累积权重
    for r in range(world_size):  # 遍历所有rank进行加权采样
        present = ((candidate_mask >> r) & 1) == 1  # 判断rank是否在候选中
        rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)  # 加载rank负载
        w = tl.where(target_total > rank_load_r, target_total - rank_load_r, 0).to(  # 计算权重
            tl.int32
        )
        w_vec = tl.full([BLOCK_SIZE], w, dtype=tl.int32)  # 广播权重
        w_vec = tl.where(  # 非源rank降低权重
            src_rank_i32 == r,
            w_vec,
            (w_vec * local_pref_denom) // local_pref_numer,
        )
        w_vec = tl.where(present, w_vec, 0)  # 仅对候选rank有效
        pick = (total_w > 0) & present & (u >= cum) & (u < (cum + w_vec))  # 判断是否选中当前rank
        chosen = tl.where(pick, r, chosen)  # 更新选中的rank
        cum += w_vec  # 累加权重

    best_rank = tl.where(total_w > 0, chosen.to(tl.int64), best_rank)  # 确定最终选中的rank

    # Step 2: Compute shared expert ID and local mask.  # 步骤2：计算共享专家ID和本地掩码
    is_local = best_rank == source_rank  # 判断是否为本地rank
    local_shared_id = source_rank * new_experts_per_rank + old_experts_per_rank  # 本地共享专家ID
    remote_shared_id = best_rank * new_experts_per_rank + old_experts_per_rank  # 远程共享专家ID
    shared_expert_id = tl.where(  # 选择本地或远程共享专家ID
        is_local,
        tl.full([BLOCK_SIZE], local_shared_id, dtype=tl.int64),  # 本地ID
        remote_shared_id,  # 远程ID
    ).to(tl.int64)
    shared_expert_id = tl.where(  # 无效令牌使用本地标记
        has_valid,
        shared_expert_id,  # 有效令牌使用计算的ID
        tl.full([BLOCK_SIZE], local_marker, dtype=tl.int64),  # 无效令牌使用标记
    )

    # Step 3: Copy and remap topk_ids, copy weights.  # 步骤3：复制并重映射topk_ids，复制权重
    for k in range(topk):  # 遍历每个top-k位置
        old_id = tl.load(topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1).to(  # 加载旧专家ID
            tl.int64
        )
        valid_id = old_id >= 0  # 判断是否为有效ID
        new_id = tl.where(valid_id, old_id + (old_id // old_experts_per_rank), old_id)  # 重映射ID：插入共享专家位置
        tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)  # 存储重映射后的ID

    for k in range(topk):  # 遍历每个top-k位置复制权重
        val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)  # 加载权重
        expert_id = tl.load(  # 加载专家ID用于有效性判断
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        val = tl.where(expert_id >= 0, val, 0.0)  # 无效专家权重置零
        tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)  # 存储权重

    # Step 4: Write shared expert column.  # 步骤4：写入共享专家列
    tl.store(  # 存储共享专家ID到最后一列
        expanded_ids_ptr + token_idx * (topk + 1) + topk,
        shared_expert_id,
        mask=mask,
    )
    tl.store(  # 存储共享专家权重到最后一列
        expanded_weights_ptr + token_idx * (topk + 1) + topk,
        tl.where(has_valid, shared_weight, 0.0),  # 有效令牌使用共享权重，否则为0
        mask=mask,
    )


def materialize_waterfill_dispatch_fused(  # 执行融合Waterfill rank选择和DeepEP TopK扩展
    topk_ids: Tensor,  # top-k专家索引 [N, topk]
    topk_weights: Tensor,  # top-k权重 [N, topk]
    rank_load: Tensor,  # rank负载 [world_size]
    num_routed_experts: int,  # 路由专家总数
    world_size: int,  # 世界大小
    source_rank: int,  # 源rank
    shared_weight: float,  # 共享专家权重
    allow_all_ranks: bool = False,  # 是否允许调度到所有rank
    target_total: int = 0,  # 目标总负载（0表示自动推导）
) -> Tuple[Tensor, Tensor]:
    """Run fused Waterfill rank selection and DeepEP TopK expansion.
    # 运行融合Waterfill rank选择和DeepEP TopK扩展。

    The Triton kernel intentionally selects each token's shared-expert rank and
    writes the expanded DeepEP TopK layout in one pass.
    # Triton内核在单次遍历中为每个令牌选择共享专家的rank，并写入扩展的DeepEP TopK布局。
    """
    num_tokens = topk_ids.shape[0]  # 获取令牌数
    topk = topk_ids.shape[1]  # 获取top-k值
    old_experts_per_rank = num_routed_experts // world_size  # 旧每rank专家数
    new_experts_per_rank = old_experts_per_rank + 1  # 新每rank专家数（含共享专家）
    device = topk_ids.device  # 获取设备

    if num_tokens == 0:  # 如果没有令牌
        return _empty_expanded(topk_ids, topk_weights)  # 返回空扩展张量

    expanded_topk_ids = torch.empty(  # 分配扩展专家索引缓冲区
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_topk_weights = torch.empty(  # 分配扩展权重缓冲区
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    BLOCK_SIZE = 256  # 设置块大小
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # 计算网格大小
    _waterfill_expand_kernel[grid](  # 启动waterfill扩展内核
        topk_ids,  # top-k专家索引
        topk_weights,  # top-k权重
        rank_load,  # rank负载
        expanded_topk_ids,  # 扩展专家索引输出
        expanded_topk_weights,  # 扩展权重输出
        num_tokens,  # 令牌数
        topk,  # top-k值
        old_experts_per_rank,  # 旧每rank专家数
        new_experts_per_rank,  # 新每rank专家数
        world_size,  # 世界大小
        source_rank,  # 源rank
        shared_weight,  # 共享专家权重
        LOCAL_SHARED_MARKER,  # 本地共享标记
        _LOCAL_PREF_NUMER,  # 本地偏好系数分子
        _LOCAL_PREF_DENOM,  # 本地偏好系数分母
        target_total,  # 目标总负载
        allow_all_ranks,  # 是否允许所有rank
        BLOCK_SIZE,  # 块大小
    )

    return expanded_topk_ids, expanded_topk_weights  # 返回扩展后的专家索引和权重


@torch.compile(dynamic=True)
def expand_topk_with_shared_expert(  # 扩展topk [N, 8] -> [N, 9]，带ID重映射；共享专家始终在本地
    topk_ids: Tensor,  # top-k专家索引 [N, topk]
    topk_weights: Tensor,  # top-k权重 [N, topk]
    num_routed_experts: int,  # 路由专家总数
    world_size: int,  # 世界大小
    source_rank: int,  # 源rank
    shared_weight: float,  # 共享专家权重
) -> Tuple[Tensor, Tensor]:
    """Expand topk [N, 8] → [N, 9] with ID remap; shared expert always local."""  # 扩展topk [N, 8] -> [N, 9]，带ID重映射；共享专家始终在本地
    num_tokens = topk_ids.shape[0]  # 获取令牌数
    topk = topk_ids.shape[1]  # 获取top-k值
    device = topk_ids.device  # 获取设备
    old_epr = num_routed_experts // world_size  # 旧每rank专家数
    new_epr = old_epr + 1  # 新每rank专家数（含共享专家）
    has_valid = (topk_ids >= 0).any(dim=1)  # 判断每行是否有有效专家
    valid_mask = topk_ids >= 0  # 有效专家掩码
    old_ranks = torch.where(valid_mask, topk_ids // old_epr, torch.zeros_like(topk_ids))  # 计算旧rank
    expanded_topk_ids = torch.empty(  # 分配扩展专家索引缓冲区
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_topk_ids[:, :topk] = torch.where(  # 重映射ID：在共享专家位置插入偏移
        valid_mask, topk_ids + old_ranks, topk_ids
    )

    shared_id = source_rank * new_epr + old_epr  # 计算本地共享专家ID
    expanded_topk_ids[:, topk] = torch.where(has_valid, shared_id, LOCAL_SHARED_MARKER)  # 写入共享专家ID
    expanded_topk_weights = torch.empty(  # 分配扩展权重缓冲区
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    expanded_topk_weights[:, :topk] = torch.where(valid_mask, topk_weights, 0.0)  # 复制权重（无效位置置零）
    expanded_topk_weights[:, topk] = torch.where(has_valid, shared_weight, 0.0).to(  # 写入共享专家权重
        topk_weights.dtype
    )
    return expanded_topk_ids, expanded_topk_weights  # 返回扩展后的专家索引和权重


class DeepEPWaterfillBalancer:  # DeepEP Waterfill负载均衡器类
    """Waterfill load balancer: shared expert fused as real routed expert (topk 8→9)."""  # Waterfill负载均衡器：共享专家融合为真实路由专家（topk 8→9）

    MIN_BATCH_FOR_BALANCE = 64  # 执行负载均衡的最小批次大小

    def __init__(  # 初始化方法
        self,
        num_routed_experts: int,  # 路由专家总数
        world_size: int,  # 世界大小
        rank: int,  # 当前rank
        layer_id: int,  # 层ID
        routed_scaling_factor: float = 1.0,  # 路由缩放因子
    ):
        self.num_routed_experts = num_routed_experts  # 设置路由专家总数
        self.world_size = world_size  # 设置世界大小
        self.rank = rank  # 设置当前rank
        self.layer_id = layer_id  # 设置层ID
        self.old_experts_per_rank = num_routed_experts // world_size  # 计算旧每rank专家数
        self.shared_weight = (  # 计算共享专家权重（考虑路由缩放因子）
            1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        )
        self._counts_buf: Optional[Tensor] = None  # 计数缓冲区（延迟初始化）
        self.use_static_waterfill = not envs.SGLANG_DISABLE_STATIC_WATERFILL.get()  # 是否使用静态waterfill

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:  # 计算每个rank的本地路由令牌数
        """Count routed tokens per rank via Triton kernel (uses original expert IDs)."""  # 通过Triton内核计算每个rank的路由令牌数（使用原始专家ID）
        if self._counts_buf is None:  # 如果计数缓冲区未初始化
            self._counts_buf = torch.zeros(  # 初始化计数缓冲区
                self.world_size, dtype=torch.int64, device=topk_ids.device
            )
        buf = self._counts_buf  # 获取计数缓冲区
        buf.zero_()  # 清零缓冲区
        num_tokens = topk_ids.shape[0]  # 获取令牌数
        if num_tokens == 0:  # 如果没有令牌
            return buf  # 返回零计数
        topk = topk_ids.shape[1]  # 获取top-k值
        BLOCK_SIZE = 256  # 设置块大小
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)  # 计算网格大小
        _count_routed_per_rank_kernel[grid](  # 启动计算内核
            topk_ids,  # top-k专家索引
            buf,  # 计数输出
            num_tokens,  # 令牌数
            topk,  # top-k值
            self.old_experts_per_rank,  # 每rank专家数
            self.world_size,  # 世界大小
            BLOCK_SIZE=BLOCK_SIZE,  # 块大小
        )
        return buf  # 返回计数结果

    def _is_low_batch(self, num_tokens: int) -> bool:  # 判断是否为小批次
        """Return whether waterfill should skip balancing for small batches."""  # 返回waterfill是否应跳过小批次的均衡
        return num_tokens < self.MIN_BATCH_FOR_BALANCE  # 令牌数小于最小均衡批次则跳过

    def _can_skip_dispatch_plan_for_low_batch(self, num_tokens: int) -> bool:  # 判断静态模式是否可跳过调度计划
        """Return whether static mode can skip dispatch-plan setup entirely."""  # 返回静态模式是否可以完全跳过调度计划设置
        return self.use_static_waterfill and self._is_low_batch(num_tokens)  # 静态模式且小批次则跳过

    def _build_static_dispatch_plan(  # 构建静态模式的Waterfill调度计划
        self, routed_counts: Tensor  # 本地路由计数
    ) -> WaterfillDispatchPlan:
        """Build static-mode Waterfill inputs from current local routed counts."""  # 从当前本地路由计数构建静态模式Waterfill输入
        return WaterfillDispatchPlan(
            rank_load=routed_counts,  # rank负载为本地路由计数
            allow_all_ranks=True,  # 静态模式允许所有rank
            target_total=0,  # 目标总负载为0（自动推导）
        )

    def _build_dynamic_dispatch_plan(  # 构建动态模式的Waterfill调度计划
        self,
        routed_counts: Tensor,  # 全局路由计数
        local_tokens_per_rank: Optional[Tensor],  # 每rank的本地令牌数（可选）
        topk: int,  # top-k值
    ) -> WaterfillDispatchPlan:
        """Build dynamic waterfill inputs from globally reduced routed counts."""  # 从全局归约的路由计数构建动态waterfill输入
        # Dynamic Waterfill balances against effective rank load: globally
        # reduced routed counts plus each rank's active token count.
        # 动态Waterfill根据有效rank负载进行均衡：全局归约的路由计数加上每rank的活跃令牌数。
        rank_load = (  # 计算有效rank负载
            routed_counts + local_tokens_per_rank  # 路由计数 + 本地令牌数
            if local_tokens_per_rank is not None
            else routed_counts  # 仅路由计数
        )
        total_routed_t = routed_counts.sum()  # 总路由令牌数
        total_tokens_global_t = total_routed_t // topk  # 全局令牌数
        total_effective_t = rank_load.sum()  # 总有效负载
        max_effective_t = rank_load.max()  # 最大有效负载
        target_total = int(  # 计算目标总负载（向上取整）
            (total_effective_t + total_tokens_global_t + self.world_size - 1)
            // self.world_size
        )
        allow_all_ranks = bool(max_effective_t <= target_total)  # 最大负载不超过目标则允许所有rank
        return WaterfillDispatchPlan(
            rank_load=rank_load,  # rank负载
            allow_all_ranks=allow_all_ranks,  # 是否允许所有rank
            target_total=target_total,  # 目标总负载
        )

    @staticmethod
    def _all_reduce_dynamic_rank_load(  # 使用SGLang EP通信聚合动态负载
        local_routed_counts: Tensor, num_tokens: int  # 本地路由计数和令牌数
    ) -> Tuple[Tensor, Tensor]:
        """Aggregate dynamic load with SGLang EP communication."""  # 使用SGLang EP通信聚合动态负载
        from sglang.srt.distributed import get_moe_ep_group  # 导入MoE EP组
        from sglang.srt.distributed.communication_op import (  # 导入通信操作
            moe_expert_parallel_all_reduce,  # MoE专家并行全归约
        )

        group = get_moe_ep_group()  # 获取MoE EP组
        world = group.world_size  # 获取世界大小
        buf = torch.zeros(  # 分配通信缓冲区 [world*2]
            world * 2, dtype=torch.int64, device=local_routed_counts.device
        )
        buf[:world] = local_routed_counts  # 前半部分存放路由计数
        rank = group.rank_in_group  # 获取组内rank
        buf[world + rank : world + rank + 1].fill_(num_tokens)  # 后半部分存放本地令牌数
        buf = moe_expert_parallel_all_reduce(buf)  # 执行全归约
        return buf[:world], buf[world:]  # 返回全局路由计数和每rank令牌数

    def _build_dispatch_plan(  # 构建调度计划
        self, topk_ids: Tensor, num_tokens: int  # top-k专家索引和令牌数
    ) -> Optional[WaterfillDispatchPlan]:
        """Prepare dispatch state for the waterfill selection boundary."""  # 准备waterfill选择边界的调度状态
        local_routed_counts = self.count_local_routed(topk_ids)  # 计算本地路由计数
        if self.use_static_waterfill:  # 如果使用静态waterfill
            return self._build_static_dispatch_plan(local_routed_counts)  # 构建静态调度计划

        global_routed_counts, local_tokens_per_rank = (  # 全局归约路由计数
            DeepEPWaterfillBalancer._all_reduce_dynamic_rank_load(
                local_routed_counts, num_tokens  # 本地路由计数和令牌数
            )
        )
        if self._is_low_batch(num_tokens):  # 如果是小批次
            return None  # 返回None（跳过调度计划）
        return self._build_dynamic_dispatch_plan(  # 构建动态调度计划
            global_routed_counts,  # 全局路由计数
            local_tokens_per_rank=local_tokens_per_rank,  # 每rank令牌数
            topk=topk_ids.shape[1],  # top-k值
        )

    def _materialize_dispatch(  # 实例化调度，执行TopK扩展
        self,
        topk_ids: Tensor,  # top-k专家索引
        topk_weights: Tensor,  # top-k权重
        dispatch_plan: WaterfillDispatchPlan,  # Waterfill调度计划
    ) -> Tuple[Tensor, Tensor]:
        """Expand TopK using local expansion or fused Waterfill."""  # 使用本地扩展或融合Waterfill扩展TopK
        num_tokens = topk_ids.shape[0]  # 获取令牌数
        if num_tokens == 0:  # 如果没有令牌
            return _empty_expanded(topk_ids, topk_weights)  # 返回空扩展张量

        if self._is_low_batch(num_tokens):  # 如果是小批次
            return expand_topk_with_shared_expert(  # 使用本地扩展（共享专家始终在本地）
                topk_ids,  # top-k专家索引
                topk_weights,  # top-k权重
                self.num_routed_experts,  # 路由专家总数
                self.world_size,  # 世界大小
                self.rank,  # 当前rank
                self.shared_weight,  # 共享专家权重
            )

        return materialize_waterfill_dispatch_fused(  # 使用融合Waterfill扩展
            topk_ids,  # top-k专家索引
            topk_weights,  # top-k权重
            dispatch_plan.rank_load,  # rank负载
            self.num_routed_experts,  # 路由专家总数
            self.world_size,  # 世界大小
            self.rank,  # 当前rank
            self.shared_weight,  # 共享专家权重
            allow_all_ranks=dispatch_plan.allow_all_ranks,  # 是否允许所有rank
            target_total=dispatch_plan.target_total,  # 目标总负载
        )

    @staticmethod
    def _with_expanded_topk(  # 将扩展张量包装回StandardTopKOutput
        topk_output: StandardTopKOutput,  # 原始TopK输出
        expanded_ids: Tensor,  # 扩展专家索引
        expanded_weights: Tensor,  # 扩展权重
    ) -> StandardTopKOutput:
        """Wrap expanded tensors back into SGLang's StandardTopKOutput."""  # 将扩展张量包装回SGLang的StandardTopKOutput
        return StandardTopKOutput(
            topk_weights=expanded_weights,  # 扩展权重
            topk_ids=expanded_ids,  # 扩展专家索引
            router_logits=topk_output.router_logits,  # 路由器logits（保持不变）
        )

    def _expand_local_shared(  # 本地共享专家扩展
        self, topk_output: StandardTopKOutput  # 原始TopK输出
    ) -> StandardTopKOutput:
        expanded_ids, expanded_weights = expand_topk_with_shared_expert(  # 执行本地扩展
            topk_output.topk_ids,  # top-k专家索引
            topk_output.topk_weights,  # top-k权重
            self.num_routed_experts,  # 路由专家总数
            self.world_size,  # 世界大小
            self.rank,  # 当前rank
            self.shared_weight,  # 共享专家权重
        )
        return self._with_expanded_topk(topk_output, expanded_ids, expanded_weights)  # 包装并返回

    def expand_topk(  # 扩展topk [N, 8] -> [N, 9]，带waterfill分配的共享专家
        self, topk_output: StandardTopKOutput, num_tokens: int  # TopK输出和令牌数
    ) -> StandardTopKOutput:
        """Expand topk [N, 8] -> [N, 9] with waterfill-assigned shared expert."""  # 扩展topk [N, 8] -> [N, 9]，使用waterfill分配的共享专家
        if self._can_skip_dispatch_plan_for_low_batch(num_tokens):  # 如果可以跳过调度计划
            # Static mode can use local expansion without communication for small
            # decode-sized batches. Dynamic mode still all-reduces before local
            # expansion so all ranks participate consistently.
            # 静态模式对小批次（解码规模）可使用无通信的本地扩展。
            # 动态模式仍在本地扩展前执行全归约以确保所有rank一致参与。
            return self._expand_local_shared(topk_output)  # 直接本地扩展

        dispatch_plan = self._build_dispatch_plan(topk_output.topk_ids, num_tokens)  # 构建调度计划
        if dispatch_plan is None:  # 如果调度计划为空（小批次动态模式）
            if num_tokens == 0:  # 如果没有令牌
                expanded_ids, expanded_weights = _empty_expanded(  # 生成空扩展
                    topk_output.topk_ids, topk_output.topk_weights
                )
                return self._with_expanded_topk(  # 包装空扩展
                    topk_output, expanded_ids, expanded_weights
                )
            else:  # 有令牌但小批次
                return self._expand_local_shared(topk_output)  # 本地扩展
        expanded_ids, expanded_weights = self._materialize_dispatch(  # 实例化调度执行扩展
            topk_output.topk_ids,  # top-k专家索引
            topk_output.topk_weights,  # top-k权重
            dispatch_plan,  # 调度计划
        )
        return self._with_expanded_topk(topk_output, expanded_ids, expanded_weights)  # 包装并返回
