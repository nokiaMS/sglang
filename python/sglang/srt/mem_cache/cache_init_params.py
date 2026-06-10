# KV缓存初始化参数数据类
# 定义了构建树形缓存（RadixCache）所需的所有初始化参数，
# 包括内存池、分页大小、并行组配置、淘汰策略等。

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        TreeComponent,
    )


@dataclasses.dataclass
class CacheInitParams:
    """KV缓存初始化参数数据类，包含构建树形缓存所需的所有配置。"""

    # 是否禁用基数缓存（radix cache）
    disable: bool
    # 请求到token的映射池，用于存储每个请求对应的token位置索引
    req_to_token_pool: ReqToTokenPool
    # token到KV张量的分配器，管理KV缓存槽位的分配与释放
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    # 分页大小，KV缓存以page_size为单位进行分配
    page_size: int

    # 是否使用EAGLE推测解码算法
    is_eagle: bool = False
    # 张量并行缓存通信组
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 注意力上下文并行缓存通信组
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 注意力张量并行缓存通信组
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 缓存淘汰策略，默认为LRU（最近最少使用）
    eviction_policy: str = "lru"
    # 是否禁用已完成请求的插入操作
    disable_finished_insert: bool = False

    # 是否启用指标收集
    enable_metrics: bool = False
    # 是否启用KV缓存事件记录（用于路由器感知KV缓存状态）
    enable_kv_cache_events: bool = False

    # 是否启用Mamba模型的额外缓冲区
    enable_mamba_extra_buffer: bool = False

    # 流水线并行的秩和大小
    pp_rank: int = 0
    pp_size: int = 1

    # 注意力上下文并行的秩和大小
    attn_cp_rank: int = 0
    attn_cp_size: int = 1

    # 分块预填充大小，None表示不分块
    chunked_prefill_size: Optional[int] = None

    # 滑动窗口大小，用于SWA（滑动窗口注意力）模型
    sliding_window_size: Optional[int] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    # 缓存条目的生存时间（秒），None表示禁用TTL
    cache_ttl_seconds: Optional[float] = None

    # 统一缓存中的树组件类型元组
    tree_components: Optional[tuple[ComponentType, ...]] = None
    # 组件注册表覆盖，用于替换默认的树组件实现
    component_registry_override: Optional[dict[ComponentType, type[TreeComponent]]] = (
        None
    )
