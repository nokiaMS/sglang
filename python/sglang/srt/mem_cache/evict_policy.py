# 本文件定义了内存缓存中前缀树的淘汰策略（Eviction Policy）。
# 当缓存空间不足时，需要根据一定的策略决定哪些节点应被优先淘汰。
# 提供了 LRU、LFU、FIFO、MRU、FILO、Priority 和 SLRU 等多种淘汰策略，
# 每种策略通过返回不同的优先级值来控制节点的淘汰顺序：
# 优先级值越小的节点越先被淘汰。

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


# 淘汰策略的抽象基类，所有具体策略都必须实现 get_priority 方法
class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        # 返回节点的淘汰优先级，值越小越先被淘汰
        pass


# 最近最少使用策略：根据节点最后一次被访问的时间排序，最久未被访问的节点优先淘汰
class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 返回最后访问时间，时间越小（越久远）越先淘汰
        return node.last_access_time


# 最不经常使用策略：综合考虑访问次数和最后访问时间，访问次数少的优先淘汰
class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # 返回 (访问次数, 最后访问时间) 元组，访问次数少的优先淘汰，次数相同时按 LRU 淘汰
        return (node.hit_count, node.last_access_time)


# 先进先出策略：根据节点创建时间排序，最早创建的节点优先淘汰
class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 返回创建时间，时间越小（越早创建）越先淘汰
        return node.creation_time


# 最近最常使用策略：与 LRU 相反，最近被访问的节点优先淘汰
class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 返回最后访问时间的负值，使最近访问的节点优先级最低（最先被淘汰）
        return -node.last_access_time


# 后进先出策略：与 FIFO 相反，最新创建的节点优先淘汰
class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 返回创建时间的负值，使最新创建的节点优先级最低（最先被淘汰）
        return -node.creation_time


# 基于优先级的淘汰策略：优先级值低的节点先淘汰，同一优先级内按 LRU 淘汰
class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        # 返回 (用户定义的优先级, 最后访问时间)，优先级值低的先淘汰，同优先级按 LRU 淘汰
        return (node.priority, node.last_access_time)


# 分段 LRU 策略（Segmented LRU）：将节点分为"试用段"和"保护段，
# 试用段中的节点优先于保护段中的节点被淘汰，同段内按 LRU 排序
class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        # 保护段的访问次数阈值，访问次数达到该阈值的节点晋升到保护段
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        # 优先级值越小，越先被淘汰
        #
        # Segment 0 (Probationary): hit_count < threshold
        # 段 0（试用段）：访问次数小于阈值的节点
        # Segment 1 (Protected): hit_count >= threshold
        # 段 1（保护段）：访问次数大于等于阈值的节点
        #
        # Tuple comparison: (segment, last_access_time)
        # 元组比较规则：(段号, 最后访问时间)
        # Nodes in segment 0 will always be evicted before segment 1.
        # 段 0 的节点总是先于段 1 的节点被淘汰
        # Inside the same segment, older nodes (smaller time) are evicted first.
        # 在同一段内，访问时间越早（值越小）的节点越先被淘汰

        # 判断节点是否属于保护段：访问次数达到阈值则为保护段（段1），否则为试用段（段0）
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        # 返回 (段号, 最后访问时间)，段号小的先淘汰，段号相同时按 LRU 淘汰
        return (is_protected, node.last_access_time)
