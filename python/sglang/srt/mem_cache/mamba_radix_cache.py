# Mamba前缀树缓存模块 - 实现混合（Full注意力 + Mamba状态）KV缓存的前缀树（Radix Tree）管理
# 本文件包含以下核心组件：
# - TreeNode: 前缀树节点，同时维护Full KV缓存和Mamba状态的引用
# - LRUList: 双向链表实现的LRU淘汰队列，支持Full和Mamba两种模式
# - MambaRadixCache: 混合KV缓存的前缀树管理器，支持前缀匹配、插入、淘汰等操作

from __future__ import annotations # 启用延迟类型注解求值

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
用于管理混合（Full和Mamba）KV缓存的前缀树数据结构。
"""
# 本文件实现了用于管理混合（Full注意力 + Mamba状态）KV缓存的前缀树（Radix Tree）数据结构。
# 主要包含以下核心组件：
# - TreeNode: 前缀树节点，同时维护Full KV缓存和Mamba状态的引用
# - LRUList: 双向链表实现的LRU淘汰队列，支持Full和Mamba两种模式
# - MambaRadixCache: 混合KV缓存的前缀树管理器，支持前缀匹配、插入、淘汰等操作

import heapq # 导入堆队列算法
from array import array # 导入数组类型
from collections import defaultdict # 导入默认字典
from typing import TYPE_CHECKING, List, Optional, Tuple # 导入类型注解

import torch # 导入PyTorch
from numpy import float64 # 导入64位浮点数类型

from sglang.srt.distributed import get_tensor_model_parallel_rank # 导入获取张量模型并行排名
from sglang.srt.mem_cache.allocator import ( # 导入内存分配器
    PagedTokenToKVPoolAllocator, # 分页Token到KV池分配器
    TokenToKVPoolAllocator, # Token到KV池分配器
)
from sglang.srt.mem_cache.base_prefix_cache import ( # 导入基础前缀缓存
    BasePrefixCache, # 基础前缀缓存类
    DecLockRefParams, # 减少锁引用参数
    DecLockRefResult, # 减少锁引用结果
    EvictParams, # 驱逐参数
    EvictResult, # 驱逐结果
    IncLockRefResult, # 增加锁引用结果
    InsertParams, # 插入参数
    InsertResult, # 插入结果
    MatchPrefixParams, # 匹配前缀参数
    MatchResult, # 匹配结果
)
from sglang.srt.mem_cache.events import KVCacheEventMixin # 导入KV缓存事件混入类
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool # 导入混合请求到Token池
from sglang.srt.mem_cache.radix_cache import RadixKey # 导入前缀树键
from sglang.srt.mem_cache.utils import split_node_hash_value # 导入节点哈希值分割工具
from sglang.srt.server_args import get_global_server_args # 导入获取全局服务器参数

if TYPE_CHECKING: # 类型检查时导入
    from sglang.srt.managers.schedule_batch import Req # 导入请求类
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams # 导入缓存初始化参数

import logging # 导入日志模块

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


class TreeNode: # 前缀树节点类，同时维护Full KV缓存和Mamba状态的引用

    counter = 0 # 节点计数器，用于生成唯一ID
    last_access_time_counter_float = float64(1.0) # 最后访问时间计数器（浮点数递增）

    def __init__(self, id: Optional[int] = None): # 初始化方法
        self.children = defaultdict(TreeNode) # 子节点字典（默认为TreeNode）
        self.parent: TreeNode = None # 父节点
        self.key: RadixKey = None # 节点键（token序列）
        self.value: Optional[torch.Tensor] = None # Full KV缓存值（设备索引）
        self.mamba_value: Optional[torch.Tensor] = None # Mamba状态值（设备索引）
        self.mamba_host_value: Optional[torch.Tensor] = None # Mamba主机状态值
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # 不变量：对于任何节点，如果mamba_lock_ref被锁定，full_lock_ref必须被锁定；
        # 如果full_lock_ref被锁定，mamba_lock_ref不需要被锁定。因此，
        # full_lock_ref总是 >= mamba_lock_ref。
        # for full_lock, once it is locked, its parent must be locked as well
        # 对于full_lock，一旦锁定，其父节点也必须被锁定
        # for mamba_lock, it only need lock node itself
        # 对于mamba_lock，只需锁定节点本身
        self.full_lock_ref = 0 # Full KV缓存锁引用计数
        self.mamba_lock_ref = 0 # Mamba状态锁引用计数
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        # 最后访问时间仅用于健全性检查。LRU由LRU列表维护。
        self.last_access_time = get_last_access_time() # 最后访问时间

        self.hit_count = 0 # 命中次数
        self.host_ref_counter = 0 # 主机KV引用计数
        self.host_mamba_ref_counter = 0 # 主机Mamba引用计数
        # store the host indices of KV cache
        # 存储KV缓存的主机索引
        self.host_value = None # 主机KV值
        # store hash values of each pages
        # 存储每页的哈希值
        self.hash_value: Optional[List[str]] = None # 哈希值列表

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        # 对于LRU列表，不变量：
        # 1. prev有更大的last_access_time
        # 2. next有更小的last_access_time
        self.prev = None # Full LRU前驱节点
        self.next = None # Full LRU后继节点
        self.mamba_prev = None # Mamba LRU前驱节点
        self.mamba_next = None # Mamba LRU后继节点
        self.host_mamba_prev = None # 主机Mamba LRU前驱节点
        self.host_mamba_next = None # 主机Mamba LRU后继节点

        self.id = TreeNode.counter if id is None else id # 设置节点ID
        TreeNode.counter += 1 # 递增节点计数器

    @property
    def evicted(self): # 判断Full KV缓存是否已被驱逐
        return self.value is None # 如果值为None则已驱逐

    @property
    def mamba_evicted(self): # 判断Mamba状态是否已被驱逐
        return self.mamba_value is None # 如果Mamba值为None则已驱逐

    @property
    def backuped(self): # 判断主机KV缓存是否有备份
        return self.host_value is not None # 如果主机值不为None则有备份

    @property
    def mamba_backuped(self): # 判断主机Mamba状态是否有备份
        return self.mamba_host_value is not None # 如果主机Mamba值不为None则有备份

    def protect_host(self): # 保护主机KV值不被驱逐
        """Protect the host KV value from eviction."""
        """保护主机KV值不被驱逐。"""
        self.host_ref_counter += 1 # 增加主机引用计数

    def release_host(self): # 释放主机KV值，允许驱逐
        """Release the host KV value, allowing it to be evicted."""
        """释放主机KV值，允许其被驱逐。"""
        if self.host_ref_counter > 0: # 如果引用计数大于0
            self.host_ref_counter -= 1 # 减少引用计数
        else: # 否则引用计数已为零
            raise RuntimeError("Host reference counter is already zero.") # 抛出运行时错误

    def protect_host_mamba(self): # 保护主机Mamba值不被驱逐
        """Protect the host mamba value from eviction."""
        """保护主机Mamba值不被驱逐。"""
        self.host_mamba_ref_counter += 1 # 增加主机Mamba引用计数

    def release_host_mamba(self): # 释放主机Mamba值，允许驱逐
        """Release the host mamba value, allowing it to be evicted."""
        """释放主机Mamba值，允许其被驱逐。"""
        if self.host_mamba_ref_counter > 0: # 如果引用计数大于0
            self.host_mamba_ref_counter -= 1 # 减少引用计数
        else: # 否则引用计数已为零
            raise RuntimeError("Host mamba reference counter is already zero.") # 抛出运行时错误

    def get_last_hash_value(self) -> Optional[str]: # 获取节点最后一个页面的哈希值
        """Returns the hash value of the last page in this node."""
        """返回此节点中最后一个页面的哈希值。"""
        if self.hash_value is None or len(self.hash_value) == 0: # 如果没有哈希值
            return None # 返回None
        return self.hash_value[-1] # 返回最后一个哈希值

    def get_prefix_hash_values(self, node: "TreeNode") -> List[str]: # 递归获取从根到指定节点的所有哈希值
        if node is None or node.hash_value is None: # 如果节点为空或没有哈希值
            return [] # 返回空列表
        return node.get_prefix_hash_values(node.parent) + node.hash_value # 递归拼接父节点和当前节点的哈希值

    def __lt__(self, other: "TreeNode"): # 小于比较运算符，用于堆排序
        return self.last_access_time < other.last_access_time # 按最后访问时间比较


def get_last_access_time() -> float64: # 获取递增的最后访问时间
    ret = TreeNode.last_access_time_counter_float # 获取当前计数器值
    TreeNode.last_access_time_counter_float += 1.0 # 递增计数器
    return ret # 返回当前值


class LRUList: # LRU列表类，双向链表实现，支持Full和Mamba两种模式
    def __init__(self, mamba: bool = False): # 初始化方法，mamba标志决定是Mamba LRU还是Full LRU
        self.mamba = mamba # 是否为Mamba模式的LRU列表
        if self.mamba: # 如果是Mamba模式
            self.prv = "mamba_prev" # 前驱属性名
            self.nxt = "mamba_next" # 后继属性名
            self.lock_ref = "mamba_lock_ref" # 锁引用属性名
        else: # 否则是Full模式
            self.prv = "prev" # 前驱属性名
            self.nxt = "next" # 后继属性名
            self.lock_ref = "full_lock_ref" # 锁引用属性名
        # Initialize dummy head and tail nodes
        # 初始化虚拟头尾节点
        self.head = TreeNode()  # Most recently used side # 最近使用端
        self.tail = TreeNode()  # Least recently used side # 最少使用端
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail # 头节点的下一个指向尾节点
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head # 尾节点的上一个指向头节点
        self.cache = {} # 节点ID到节点的映射缓存

    def _add_node(self, node): # 在头部之后添加节点（标记为最近使用）
        """Helper to add node right after head (most recently used)"""
        """在头部之后添加节点的辅助方法（标记为最近使用）"""
        self._add_node_after(self.head, node) # 委托给_add_node_after

    def _add_node_after(self, old_node, new_node): # 在指定节点之后添加新节点
        """Helper to add node right after old_node"""
        """在old_node之后添加new_node的辅助方法"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node # 新节点的前驱指向旧节点
        setattr( # 新节点的后继指向旧节点的后继
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr( # 旧节点后继的前驱指向新节点
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node # 旧节点的后继指向新节点

    def _remove_node(self, node): # 从链表中移除节点
        """Helper to remove node from linked list"""
        """从链表中移除节点的辅助方法"""
        setattr( # 节点前驱的后继指向节点的后继
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr( # 节点后继的前驱指向节点的前驱
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev
        # Clear self pointers to break reference cycles among evicted nodes.
        # 清除自身指针以打破被驱逐节点之间的引用循环。
        setattr(node, self.prv, None) # 清除前驱指针
        setattr(node, self.nxt, None) # 清除后继指针

    def _get_lru(self) -> Optional[TreeNode]: # 获取最少使用的节点
        """
        Get the least recently used node
        获取最少使用的节点
        """
        if len(self.cache) == 0: # 如果缓存为空
            return None # 返回None
        return getattr(self.tail, self.prv) # 返回尾节点的前驱（即最久未使用的节点）

    def reset_node_mru(self, node): # 将已有节点移到最近使用位置
        """
        Move a (existing) node to most recently used position
        将（已有）节点移到最近使用位置
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list" # 断言节点在LRU列表中
        assert ( # 断言不在Mamba模式或节点有Mamba值
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}" # 不能在Mamba LRU中重置墓碑节点
        self._remove_node(node) # 从当前位置移除
        self._add_node(node) # 添加到头部（最近使用位置）

    def reset_node_and_parents_mru(self, node, root_node): # 将节点及其父节点移到最近使用位置
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        将（已有）节点及其父节点移到最近使用位置。子节点比父节点更近使用。
        """
        prev_node = self.head # 前一个节点为头部
        while node != root_node: # 遍历到根节点
            if not self.mamba or node.mamba_value is not None: # 如果不是Mamba模式或节点有Mamba值
                assert ( # 断言节点在LRU列表中
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru" # 重置节点和父节点MRU时节点不在LRU列表中
                self._remove_node(node) # 从当前位置移除
                self._add_node_after(prev_node, node) # 添加到前一个节点之后
                prev_node = node # 更新前一个节点
            node = node.parent # 移到父节点

    def insert_mru(self, node): # 插入新节点到最近使用位置
        """
        Insert a (new) node as most recently used
        插入（新）节点为最近使用
        """
        assert ( # 断言不在Mamba模式或节点有Mamba值
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}" # 不能在Mamba LRU中插入墓碑节点
        assert ( # 断言节点不在LRU列表中
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}" # 插入已在LRU列表中的节点
        self.cache[node.id] = node # 添加到缓存
        self._add_node(node) # 添加到头部

    def remove_node(self, node: TreeNode): # 从LRU列表中移除节点
        """
        Remove node from lru list
        从LRU列表中移除节点
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list" # 断言节点在LRU列表中
        assert ( # 断言不在Mamba模式或节点有Mamba值
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}" # 不能从Mamba LRU中移除墓碑节点
        del self.cache[node.id] # 从缓存中删除
        self._remove_node(node) # 从链表中移除

    def get_lru_no_lock(self) -> Optional[TreeNode]: # 获取最少使用的未锁定节点
        """
        Get the least recently used node that is not locked
        获取最少使用的未锁定节点
        """
        return self.get_prev_no_lock(self.tail, check_id=False) # 从尾部向前查找

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]: # 获取最少使用的未锁定叶子节点
        """
        Get the least recently used leaf node that is not locked
        获取最少使用的未锁定叶子节点
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False) # 从尾部向前查找叶子节点

    def get_prev_no_lock( # 获取前一个未锁定的节点
        self, node: TreeNode, check_id: bool = True # 起始节点、是否检查ID
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        获取前一个（即更近使用的）未锁定节点
        """
        if check_id: # 如果需要检查ID
            assert ( # 断言节点在LRU列表中
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list" # 获取不在LRU列表中节点的前驱
        x = getattr(node, self.prv)  # x = node.prev # 从前驱开始
        while getattr(x, self.lock_ref) > 0: # 跳过锁定节点
            x = getattr(x, self.prv)  # x = x.prev # 继续向前
        # if x is the head, it means there is no node in the lru list without lock
        # 如果x是头节点，表示LRU列表中没有未锁定的节点
        if x == self.head: # 如果到达头部
            return None # 返回None
        return x # 返回找到的节点

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True): # 获取前一个未锁定的叶子节点
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        获取前一个（即更近使用的）未锁定叶子节点
        """
        if check_id: # 如果需要检查ID
            assert ( # 断言节点在LRU列表中
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list" # 获取不在LRU列表中节点的前驱
        x = getattr(node, self.prv)  # x = node.prev # 从前驱开始
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0: # 跳过锁定节点和非叶子节点
            x = getattr(x, self.prv)  # x = x.prev # 继续向前
        # if x is the head, it means there is no leaf node in the lru list without lock
        # 如果x是头节点，表示LRU列表中没有未锁定的叶子节点
        if x == self.head: # 如果到达头部
            return None # 返回None
        return x # 返回找到的叶子节点

    def in_list(self, node: Optional[TreeNode]): # 检查节点是否在LRU列表中
        """
        Check if the node is in the lru list
        检查节点是否在LRU列表中
        """
        if not node: # 如果节点为空
            return False # 返回False
        return node.id in self.cache # 检查节点ID是否在缓存中

    def pretty_print(self, tree_cache: Optional["MambaRadixCache"] = None): # 美化打印LRU列表
        """
        Pretty print the lru list
        美化打印LRU列表
        """
        msg = f"{self.mamba=} LRU list: " # 打印Mamba标志
        x_lru = self._get_lru() # 获取LRU节点
        while x_lru is not None and x_lru.id in self.cache: # 遍历LRU列表
            msg += f"[{x_lru.id}] {x_lru.last_access_time:f} -> " # 打印节点ID和访问时间
            x_lru = getattr(x_lru, self.prv) # 移到前驱
        print(msg) # 打印消息

        if not tree_cache: # 如果没有树缓存
            return # 返回
        msg = f"{self.mamba=} Nodes (sorted by last_access_time): " # 打印排序后的节点
        if self.mamba: # 如果是Mamba模式
            nodes = tree_cache._collect_nontombstone_nodes() # 收集非墓碑节点
        else: # 否则收集所有节点
            nodes = tree_cache._collect_all_nodes()
        heapq.heapify(nodes) # 建堆排序
        while len(nodes): # 遍历堆
            x = heapq.heappop(nodes) # 弹出最小元素
            msg += f"[{x.id}] {x.last_access_time:f} -> " # 打印节点ID和访问时间
        print(msg) # 打印消息

    # Note: this is expensive, only use for debug
    # 注意：此操作开销大，仅用于调试
    def sanity_check_evictable_size(self): # 检查可驱逐大小
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        检查可驱逐大小（即未锁定节点的大小）
        """
        node = self.get_lru_no_lock() # 获取第一个未锁定的LRU节点
        evictable_size = 0 # 初始化可驱逐大小
        while self.in_list(node): # 遍历所有未锁定节点
            evictable_size += ( # 累加可驱逐大小
                len(node.value) if not self.mamba else len(node.mamba_value) # 根据模式选择值
            )
            node = self.get_prev_no_lock(node) # 获取下一个未锁定节点
        return evictable_size # 返回可驱逐大小

    # Note: this is expensive, only use for debug or idle check
    # 注意：此操作开销大，仅用于调试或空闲检查
    def sanity_check(self, tree_cache: "MambaRadixCache"): # 健全性检查
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        通过从树重建LRU列表、建堆排序来检查LRU列表是否有效。
        """
        try: # 尝试执行
            if self.mamba: # 如果是Mamba模式
                nodes = tree_cache._collect_nontombstone_nodes() # 收集非墓碑节点
            else: # 否则收集所有节点
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes) # 总节点数
            total_lru = len(self.cache) # LRU列表中的节点数
            # heapify based on last_access_time
            # 基于last_access_time建堆
            heapq.heapify(nodes) # 建堆排序
            # the root node is not in the lru list
            # 根节点不在LRU列表中
            assert len(nodes) == ( # 断言节点数与LRU列表一致
                total_lru + (0 if self.mamba else 1) # Mamba模式不加1，Full模式加1（根节点）
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru() # 获取LRU节点
            while len(nodes): # 遍历堆
                x = heapq.heappop(nodes) # 弹出最小元素
                if x == tree_cache.root_node: # 如果是根节点
                    # root node is not in the lru list
                    # 根节点不在LRU列表中
                    continue # 跳过
                assert ( # 断言LRU节点有效
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}" # LRU列表不正确

                assert ( # 断言堆中的节点与LRU列表中的节点一致
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {x.last_access_time=}, {x_lru.last_access_time=}" # LRU列表不正确
                assert ( # 断言LRU节点未被锁定
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}" # 空闲时LRU节点不应被锁定
                assert ( # 断言LRU节点的Mamba锁未被锁定
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}" # 空闲时LRU节点的Mamba锁不应被锁定
                x_lru = getattr(x_lru, self.prv) # 移到前驱

            if self.mamba: # 如果是Mamba模式
                evictable_size = tree_cache.mamba_evictable_size() # 获取Mamba可驱逐大小
                lru_list_evictable_size = self.sanity_check_evictable_size() # 获取LRU列表可驱逐大小
            else: # 否则是Full模式
                evictable_size = tree_cache.full_evictable_size() # 获取Full可驱逐大小
                lru_list_evictable_size = self.sanity_check_evictable_size() # 获取LRU列表可驱逐大小

            assert ( # 断言可驱逐大小一致
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}" # 可驱逐大小不一致
        except Exception as e: # 捕获异常
            if get_tensor_model_parallel_rank() == 0: # 仅在rank 0上报告
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}" # Mamba前缀树健全性检查失败
                logger.error(msg) # 记录错误日志
                tree_cache.pretty_print() # 打印树结构
                tree_cache.full_lru_list.pretty_print(tree_cache) # 打印Full LRU列表
                tree_cache.mamba_lru_list.pretty_print(tree_cache) # 打印Mamba LRU列表
                raise Exception(msg) # 抛出异常


class MambaRadixCache(KVCacheEventMixin, BasePrefixCache): # Mamba前缀树缓存类，混合KV缓存的前缀树管理器
    def __init__(self, params: CacheInitParams): # 初始化方法
        assert isinstance( # 断言分配器类型正确
            params.token_to_kv_pool_allocator, TokenToKVPoolAllocator
        ) or isinstance(params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool # 请求到Token池
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator # Token到KV池分配器
        self.mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size # Mamba缓存块大小

        self.page_size = params.page_size # 页面大小
        self.disable = params.disable # 是否禁用缓存
        self.enable_kv_cache_events = params.enable_kv_cache_events # 是否启用KV缓存事件
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer # 是否启用Mamba额外缓冲区
        self.kv_event_queue = [] # KV事件队列

        if not self.enable_mamba_extra_buffer: # 如果未启用Mamba额外缓冲区
            assert ( # 断言页面大小必须为1
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}" # MambaRadixCache v1页面大小必须为1

        if self.token_to_kv_pool_allocator: # 如果有分配器
            self.device = self.token_to_kv_pool_allocator.device # 获取设备
        else: # 否则使用CPU
            self.device = torch.device("cpu") # 设备为CPU

        if params.enable_metrics: # 如果启用指标
            self.init_metrics_collector() # 初始化指标收集器

        self.reset() # 重置缓存

    ##### Public API ##### # 公共API

    def supports_mamba(self) -> bool: # 是否支持Mamba
        return True # 返回True

    def reset(self) -> None: # 重置缓存
        self.root_node = TreeNode() # 创建根节点
        self.root_node.key = RadixKey(array("q"), None) # 设置根节点键为空
        self.root_node.value = [] # 设置根节点值为空列表
        self.root_node.hash_value = [] # 设置根节点哈希值为空列表
        self.root_node.full_lock_ref = 1 # 锁定根节点（Full）
        self.root_node.mamba_lock_ref = 1 # 锁定根节点（Mamba）
        self.full_evictable_size_ = 0 # Full可驱逐大小
        self.mamba_evictable_size_ = 0 # Mamba可驱逐大小
        self.full_protected_size_ = 0 # Full受保护大小
        self.mamba_protected_size_ = 0 # Mamba受保护大小
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        # LRU列表用于维护树中节点的驱逐顺序
        self.full_lru_list = LRUList(mamba=False) # 创建Full LRU列表
        self.mamba_lru_list = LRUList(mamba=True) # 创建Mamba LRU列表
        self._record_all_cleared_event() # 记录全部清除事件

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult: # 前缀匹配
        """Find the matching prefix from the radix tree.
        从前缀树中查找匹配的前缀。
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
            参数：MatchPrefixParams，包含键和可选的Mamba特定参数。
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
            返回一个包含匹配前缀token ID张量和包含前缀值的最后一个节点的元组。
            注意此API可能修改Radix树的内部状态。如果前缀短于最后节点的值，
            最后节点会创建一个新的子节点。
        """
        key = self._match_pre_processor(params) # 预处理键
        if key is None: # 如果键为空
            return MatchResult( # 返回空匹配结果
                device_indices=torch.empty( # 空设备索引张量
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node, # 最后设备节点为根节点
                last_host_node=self.root_node, # 最后主机节点为根节点
                best_match_node=self.root_node, # 最佳匹配节点为根节点
            )

        value, last_node, best_value_len = self._match_prefix_helper(key) # 执行前缀匹配
        return self._match_post_processor(params, value, last_node, best_value_len) # 后处理匹配结果

    def insert(self, params: InsertParams) -> InsertResult: # 插入键值对
        if self.disable: # 如果禁用缓存
            return InsertResult(prefix_len=0, mamba_exist=False) # 返回空插入结果

        key = params.key # 获取键
        value = params.value # 获取值
        mamba_value = params.mamba_value # 获取Mamba值
        prev_prefix_len = params.prev_prefix_len # 获取前一个前缀长度

        if value is None: # 如果值为空
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64) # 从键的token ID创建值
        prefix_len, mamba_exist = self._insert_helper( # 执行插入
            self.root_node, key, value, mamba_value, params.chunked, prev_prefix_len # 传入根节点、键、值、Mamba值、是否分块、前一个前缀长度
        )
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist) # 返回插入结果

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None: # 缓存已完成的请求
        """Cache request when it finishes."""
        """请求完成时缓存。"""
        kv_committed_len = req.pop_committed_kv_cache() # 获取已提交的KV缓存长度
        if self.disable: # 如果禁用缓存
            kv_indices = self.req_to_token_pool.req_to_token[ # 获取KV索引
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices) # 释放KV索引
            self.req_to_token_pool.free_mamba_cache(req) # 释放Mamba缓存
            return # 返回

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len] # 获取token ID
        kv_indices = self.req_to_token_pool.req_to_token[ # 获取KV索引
            req.req_pool_idx, :kv_committed_len
        ]

        if is_insert: # 如果需要插入
            cache_len = ( # 计算缓存长度
                req.mamba_last_track_seqlen # Mamba最后跟踪序列长度
                if self.enable_mamba_extra_buffer # 如果启用Mamba额外缓冲区
                else len(token_ids) # 否则使用token ID长度
            )
            if cache_len is None: # 如果缓存长度为空
                cache_len = 0 # 设为0
            if cache_len != len(token_ids): # 如果缓存长度不等于token ID长度
                cache_end_idx = max(cache_len, req.cache_protected_len) # 计算缓存结束索引
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:]) # 释放不需要的KV索引
                token_ids = token_ids[:cache_len] # 截断token ID
                kv_indices = kv_indices[:cache_len] # 截断KV索引

            if self.page_size != 1: # 如果页面大小不为1
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size # 计算页面对齐长度
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to( # 页面对齐的KV索引
                    dtype=torch.int64, copy=True # 转换类型并复制
                )
            else: # 否则页面大小为1
                page_aligned_len = len(kv_indices) # 页面对齐长度等于KV索引长度
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True) # 转换类型并复制

            assert ( # 断言缓存长度等于页面对齐长度
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this" # 如果看到此消息请联系@yizhang2077

            # Radix Cache takes one ref in memory pool
            # Radix缓存在内存池中持有一个引用
            # insert the token_ids and kv_indices into the radix tree
            # 将token_ids和kv_indices插入前缀树
            if self.enable_mamba_extra_buffer: # 如果启用Mamba额外缓冲区
                mamba_ping_pong_track_buffer_to_keep = ( # 获取Mamba乒乓跟踪缓冲区中要保留的索引
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                mamba_value = ( # 获取Mamba值
                    req.mamba_ping_pong_track_buffer[
                        mamba_ping_pong_track_buffer_to_keep
                    ]
                    .unsqueeze(-1) # 增加一个维度
                    .clone() # 克隆
                )
            else: # 否则未启用Mamba额外缓冲区
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone() # 获取Mamba池索引并克隆
                mamba_ping_pong_track_buffer_to_keep = None # 不需要保留

            result = self.insert( # 插入到前缀树
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key), # 键
                    value=page_aligned_kv_indices, # 值
                    mamba_value=mamba_value, # Mamba值
                    prev_prefix_len=req.cache_protected_len, # 前一个前缀长度
                )
            )
            mamba_exist = result.mamba_exist # 获取Mamba是否存在
        else: # 否则不插入
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :]) # 释放不需要的KV索引
            mamba_exist = True # Mamba值已存在

        if mamba_exist: # 如果Mamba值已存在
            mamba_ping_pong_track_buffer_to_keep = None # 不需要保留

        free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist # 是否释放Mamba缓存

        if free_mamba_cache: # 如果需要释放Mamba缓存
            self.req_to_token_pool.free_mamba_cache( # 释放Mamba缓存
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep, # 要保留的索引
            )

        self.dec_lock_ref(req.last_node) # 减少锁引用

    def cache_unfinished_req(self, req: Req, chunked=False) -> None: # 缓存未完成的请求
        """Cache request when it is unfinished."""
        """请求未完成时缓存。"""

        def _skip_cache_unfinished_req(req: Req) -> None: # 跳过缓存未完成请求
            kv_indices = self.req_to_token_pool.req_to_token[ # 获取KV索引
                req.req_pool_idx, : len(req.fill_ids)
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            # `req.prefix_indices`稍后将在`PrefillAdder::add_chunked_req`中使用
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True) # 转换类型并复制
            return # 返回

        token_ids = req.fill_ids # 获取填充ID
        cache_len = ( # 计算缓存长度
            req.mamba_last_track_seqlen # Mamba最后跟踪序列长度
            if self.enable_mamba_extra_buffer # 如果启用Mamba额外缓冲区
            else len(token_ids) # 否则使用token ID长度
        )
        if self.disable or cache_len is None: # 如果禁用或缓存长度为空
            return _skip_cache_unfinished_req(req) # 跳过缓存

        kv_indices_orig = self.req_to_token_pool.req_to_token[ # 获取原始KV索引
            req.req_pool_idx, : len(token_ids)
        ]
        # kv_indices is the kv indices to be cached
        # kv_indices是要缓存的KV索引
        kv_indices = kv_indices_orig[:cache_len] # 截取需要缓存的KV索引
        if self.page_size != 1: # 如果页面大小不为1
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size # 计算页面对齐长度
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to( # 页面对齐的KV索引
                dtype=torch.int64, copy=True # 转换类型并复制
            )
        else: # 否则页面大小为1
            page_aligned_len = len(kv_indices) # 页面对齐长度等于KV索引长度
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True) # 转换类型并复制

        assert page_aligned_len == len( # 断言页面对齐长度等于KV索引长度
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {self.mamba_cache_chunk_size=}" # 页面对齐长度不等于KV索引长度

        page_aligned_token_ids = token_ids[:page_aligned_len] # 页面对齐的token ID

        # Donate the mamba index to the radix cache instead of copying.
        # This avoids a data copy that would race with the forward stream.
        # 将Mamba索引捐赠给前缀缓存而不是复制。
        # 这避免了与前向流竞争的数据复制。
        if self.enable_mamba_extra_buffer: # 如果启用Mamba额外缓冲区
            mamba_ping_pong_track_buffer_to_keep = ( # 获取Mamba乒乓跟踪缓冲区中要保留的索引
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            mamba_value_donated = ( # 获取要捐赠的Mamba值
                req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
                .unsqueeze(-1) # 增加一个维度
                .clone() # 克隆
            )
            new_slot = self._alloc_mamba_slot() # 分配新的Mamba槽位
            req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep] = ( # 更新乒乓缓冲区
                new_slot[0]
            )
            self.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[ # 更新映射
                req.req_pool_idx
            ] = req.mamba_ping_pong_track_buffer # 设置新的乒乓缓冲区
        else: # 否则未启用Mamba额外缓冲区
            mamba_value_donated = self._alloc_mamba_slot() # 分配新的Mamba槽位
            self.req_to_token_pool.mamba_pool.copy_from( # 从请求的Mamba池索引复制到新槽位
                req.mamba_pool_idx.unsqueeze(0), mamba_value_donated
            )

        result = self.insert( # 插入到前缀树
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key), # 键
                value=page_aligned_kv_indices, # 值
                mamba_value=mamba_value_donated, # 捐赠的Mamba值
                prev_prefix_len=req.cache_protected_len, # 前一个前缀长度
                chunked=chunked, # 是否分块
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist # 获取新前缀长度和Mamba是否存在
        if mamba_exist: # 如果Mamba值已存在
            self.req_to_token_pool.mamba_pool.free(mamba_value_donated) # 释放捐赠的Mamba值

        # The prefix indices could be updated, reuse it
        # 前缀索引可能已更新，重新使用
        match_result = self.match_prefix( # 重新匹配前缀
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key)) # 传入键
        )
        new_indices, new_last_node = ( # 获取新的索引和最后节点
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist: # 如果Mamba值不存在
            assert torch.equal(new_last_node.mamba_value, mamba_value_donated) # 断言新节点的Mamba值等于捐赠的值

        assert ( # 断言缓存保护长度不超过新索引长度+页面大小-1
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}" # 缓存保护长度检查失败
        assert new_prefix_len <= len( # 断言新前缀长度不超过新索引长度
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}" # 新前缀长度检查失败

        self.req_to_token_pool.write( # 写入请求到Token池
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))), # 位置
            new_indices[req.cache_protected_len :], # 新索引
        )

        self.dec_lock_ref(req.last_node) # 减少旧节点的锁引用
        self.inc_lock_ref(new_last_node) # 增加新节点的锁引用

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # `req.prefix_indices`稍后将在`PrefillAdder::add_chunked_req`中使用
        # NOTE: this is needed for both page_size == 1 and page_size > 1
        # 注意：page_size == 1和page_size > 1都需要此操作
        req.prefix_indices = torch.cat( # 设置前缀索引
            [new_indices, kv_indices_orig[len(new_indices) :]] # 拼接新索引和剩余的原始索引
        )
        req.cache_protected_len = len(new_indices) # 更新缓存保护长度
        req.mamba_last_track_seqlen = None # 重置Mamba最后跟踪序列长度
        req.last_node = new_last_node # 更新最后节点

    def pretty_print(self) -> None: # 美化打印前缀树
        self._print_helper(self.root_node, 0) # 从根节点开始打印
        total_size, total_mamba_size = self._total_size_helper() # 计算总大小
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}") # 打印大小信息

    def total_size(self) -> Tuple[int, int]: # 获取总大小
        return self._total_size_helper() # 返回总大小

    def _evict_leaf_node( # 驱逐叶子节点
        self, x: TreeNode, is_evict_mamba: bool # 要驱逐的节点、是否驱逐Mamba
    ) -> Tuple[int, int, TreeNode, TreeNode]: # 返回驱逐的Full token数、Mamba数、当前节点、下一个节点
        assert ( # 断言叶子节点未被锁定
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}" # 驱逐锁定叶子节点无效

        assert x.mamba_value is not None, f"leaf node mamba value is not None, {x.id=}" # 断言叶子节点有Mamba值
        # 1. a leaf node, free full tokens and mamba
        # 1. 叶子节点，释放Full token和Mamba
        self._record_remove_event(x) # 记录移除事件
        self.token_to_kv_pool_allocator.free(x.value) # 释放Full KV索引
        full_num_evicted = len(x.value) # 记录驱逐的Full token数
        self.req_to_token_pool.mamba_pool.free(x.mamba_value) # 释放Mamba值
        mamba_num_evicted = len(x.mamba_value) # 记录驱逐的Mamba数

        # 2. get the next node, update the lru lists
        # 2. 获取下一个节点，更新LRU列表
        if is_evict_mamba: # 如果驱逐Mamba
            x_next = self.mamba_lru_list.get_prev_no_lock(x) # 从Mamba LRU获取下一个节点
        else: # 否则驱逐Full
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x) # 从Full LRU获取下一个叶子节点
        self.full_lru_list.remove_node(x) # 从Full LRU列表中移除
        self.mamba_lru_list.remove_node(x) # 从Mamba LRU列表中移除

        # 3. delete the leaf node
        # 3. 删除叶子节点
        self._delete_leaf(x) # 删除叶子节点

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        # 4. 迭代删除墓碑叶子节点，维护叶子节点不是墓碑的不变量
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x) # 迭代删除墓碑叶子
        full_num_evicted += leaf_full_num_evicted # 累加驱逐的Full token数
        return full_num_evicted, mamba_num_evicted, x, x_next # 返回驱逐数和下一个节点

    def evict(self, params: EvictParams) -> EvictResult: # 驱逐缓存
        if self.disable: # 如果禁用缓存
            return EvictResult() # 返回空结果

        full_num_evicted = 0 # 驱逐的Full token数
        mamba_num_evicted = 0 # 驱逐的Mamba数

        if params.num_tokens > 0: # 如果需要驱逐Full token
            full_num_evicted = self.evict_full(params.num_tokens) # 驱逐Full KV
        if params.mamba_num > 0: # 如果需要驱逐Mamba
            mamba_num_evicted = self.evict_mamba(params.mamba_num) # 驱逐Mamba

        return EvictResult( # 返回驱逐结果
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted # Full和Mamba驱逐数
        )

    def evict_mamba(self, mamba_num: int) -> int: # 驱逐Mamba状态
        """Evict mamba states. Returns the number of mamba states evicted."""
        """驱逐Mamba状态。返回驱逐的Mamba状态数。"""
        if self.disable or mamba_num <= 0: # 如果禁用或不需要驱逐
            return 0 # 返回0
        # get the least recently used node that is not locked, doesn't have to be a leaf
        # 获取最少使用的未锁定节点，不一定是叶子节点
        x = self.mamba_lru_list.get_lru_no_lock() # 获取LRU节点
        mamba_num_evicted = 0 # 驱逐的Mamba数
        # evict lru leaf nodes until mamba_num_tokens is reached
        # 驱逐LRU叶子节点直到达到mamba_num_tokens
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)): # 循环驱逐
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}" # 断言节点有Mamba值
            assert ( # 断言Mamba值长度为1
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}" # 节点Mamba长度异常
            assert x != self.root_node, f"root node is not evictable, {x.id=}" # 断言不是根节点
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}" # 断言Mamba锁未锁定

            if len(x.children) > 0: # 如果是内部节点
                # 1. an internal node, free mamba tokens.
                # 1. 内部节点，释放Mamba token。
                self.req_to_token_pool.mamba_pool.free(x.mamba_value) # 释放Mamba值
                mamba_num_evicted += len(x.mamba_value) # 累加驱逐数

                # 2. get the next node, update the lru lists
                # 2. 获取下一个节点，更新LRU列表
                x_next = self.mamba_lru_list.get_prev_no_lock(x) # 获取下一个节点
                self.mamba_lru_list.remove_node(x) # 从Mamba LRU列表中移除

                # 3. tombstone the node
                # 3. 将节点标记为墓碑
                self._tombstone_internal_node(x) # 标记为墓碑
            else: # 否则是叶子节点
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True) # 驱逐叶子节点
                mamba_num_evicted += mamba_evicted_delta # 累加驱逐数

            x = x_next # 移到下一个节点

        return mamba_num_evicted # 返回驱逐的Mamba数

    def evict_full(self, full_num_tokens: int) -> int: # 驱逐Full KV缓存
        """Evict full KV cache. Returns the number of tokens evicted."""
        """驱逐Full KV缓存。返回驱逐的token数。"""
        if self.disable or full_num_tokens <= 0: # 如果禁用或不需要驱逐
            return 0 # 返回0

        full_num_evicted = 0 # 驱逐的Full token数
        # get the least recently used leaf node that is not locked
        # 获取最少使用的未锁定叶子节点
        x = self.full_lru_list.get_leaf_lru_no_lock() # 获取LRU叶子节点

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x): # 循环驱逐
            assert ( # 断言不是根节点
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}" # 根节点不应存在于Full LRU列表中
            full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False) # 驱逐叶子节点
            full_num_evicted += full_num_evicted_delta # 累加驱逐数

            # if parent has no more children, it is a leaf. It is possible that this node is lru, so
            # we need to get the first leaf node in the lru list
            # 如果父节点没有子节点了，它就是叶子。这个节点可能是LRU的，
            # 所以需要获取LRU列表中的第一个叶子节点
            if len(x.parent.children) == 0: # 如果父节点没有子节点
                x_next = self.full_lru_list.get_leaf_lru_no_lock() # 重新获取LRU叶子节点

            x = x_next # 移到下一个节点

        return full_num_evicted # 返回驱逐的Full token数

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult: # 增加锁引用计数
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        增加节点的锁引用计数。
        锁定[last node, root)之间节点的full_lock_ref（不包含root）。
        如果当前节点的mamba_value存在，锁定mamba_lock_ref。
        """
        if self.disable: # 如果禁用缓存
            return IncLockRefResult() # 返回空结果

        # protect mamba value in current node if it exists
        # 如果当前节点有Mamba值，保护它
        if node.mamba_value is not None: # 如果有Mamba值
            if node.mamba_lock_ref == 0: # 如果Mamba锁引用为0
                self.mamba_evictable_size_ -= len(node.mamba_value) # 减少可驱逐大小
                self.mamba_protected_size_ += len(node.mamba_value) # 增加受保护大小
            node.mamba_lock_ref += 1 # 增加Mamba锁引用

        while node != self.root_node: # 从节点到根节点
            # lock full from node to root
            # 从节点到根节点锁定Full
            assert ( # 断言Full锁引用非负
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}" # 在Full锁引用为负的节点上增加锁引用
            if node.full_lock_ref == 0: # 如果Full锁引用为0
                self.full_evictable_size_ -= len(node.value) # 减少可驱逐大小
                self.full_protected_size_ += len(node.value) # 增加受保护大小
            node.full_lock_ref += 1 # 增加Full锁引用
            node = node.parent # 移到父节点
        return IncLockRefResult() # 返回结果

    def dec_lock_ref( # 减少锁引用计数
        self, node: TreeNode, params: Optional[DecLockRefParams] = None # 节点、参数
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        减少节点的锁引用计数。
        解锁[last node, root)之间节点的full_lock_ref（不包含root）。
        如果当前节点的mamba_value存在，解锁mamba_lock_ref。
        """
        if self.disable: # 如果禁用缓存
            return DecLockRefResult() # 返回空结果

        if node.mamba_value is not None: # 如果有Mamba值
            assert ( # 断言Mamba锁引用大于0
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}" # 在Mamba锁引用为0的节点上减少锁引用
            if node.mamba_lock_ref == 1: # 如果Mamba锁引用将为0
                self.mamba_evictable_size_ += len(node.mamba_value) # 增加可驱逐大小
                self.mamba_protected_size_ -= len(node.mamba_value) # 减少受保护大小
            node.mamba_lock_ref -= 1 # 减少Mamba锁引用

        while node != self.root_node: # 从节点到根节点
            assert ( # 断言Full锁引用大于0
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}" # 在Full锁引用为0的节点上减少锁引用
            if node.full_lock_ref == 1: # 如果Full锁引用将为0
                self.full_evictable_size_ += len(node.value) # 增加可驱逐大小
                self.full_protected_size_ -= len(node.value) # 减少受保护大小
            node.full_lock_ref -= 1 # 减少Full锁引用
            node = node.parent # 移到父节点

        return DecLockRefResult() # 返回结果

    def sanity_check(self): # 健全性检查
        if self.disable: # 如果禁用缓存
            return # 返回
        self.full_lru_list.sanity_check(self) # 检查Full LRU列表
        self.mamba_lru_list.sanity_check(self) # 检查Mamba LRU列表

    def evictable_size(self) -> Tuple[int, int]: # 获取可驱逐大小（已弃用）
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        # 注意：请使用full_evictable_size()和mamba_evictable_size()。
        raise NotImplementedError # 抛出未实现异常

    def full_evictable_size(self) -> int: # 获取Full可驱逐大小
        return self.full_evictable_size_ # 返回Full可驱逐大小

    def mamba_evictable_size(self) -> int: # 获取Mamba可驱逐大小
        return self.mamba_evictable_size_ # 返回Mamba可驱逐大小

    def protected_size(self) -> Tuple[int, int]: # 获取受保护大小（已弃用）
        # Note: use full_protected_size() and mamba_protected_size() instead.
        # 注意：请使用full_protected_size()和mamba_protected_size()。
        raise NotImplementedError # 抛出未实现异常

    def full_protected_size(self) -> int: # 获取Full受保护大小
        # protected size refers to the size of the full cache that is locked
        # 受保护大小指被锁定的Full缓存大小
        return self.full_protected_size_ # 返回Full受保护大小

    def mamba_protected_size(self) -> int: # 获取Mamba受保护大小
        # protected size refers to the size of the mamba cache that is locked
        # 受保护大小指被锁定的Mamba缓存大小
        return self.mamba_protected_size_ # 返回Mamba受保护大小

    def all_values_flatten(self) -> torch.Tensor: # 获取所有Full KV值的扁平化张量
        values = [] # 值列表

        def _dfs_helper(node: TreeNode): # 深度优先遍历辅助函数
            for _, child in node.children.items(): # 遍历子节点
                values.append(child.value) # 添加子节点的值
                _dfs_helper(child) # 递归遍历

        _dfs_helper(self.root_node) # 从根节点开始遍历
        return torch.cat(values) if len(values) > 0 else torch.tensor([]) # 拼接所有值或返回空张量

    def all_mamba_values_flatten(self) -> torch.Tensor: # 获取所有Mamba值的扁平化张量
        values = [] # 值列表

        def _dfs_helper(node: TreeNode): # 深度优先遍历辅助函数
            if node.mamba_value is not None: # 如果有Mamba值
                values.append(node.mamba_value) # 添加Mamba值
            for _, child in node.children.items(): # 遍历子节点
                _dfs_helper(child) # 递归遍历

        _dfs_helper(self.root_node) # 从根节点开始遍历
        return torch.cat(values) if len(values) > 0 else torch.tensor([]) # 拼接所有值或返回空张量

    def available_and_evictable_str(self) -> str: # 获取可用和可驱逐大小的字符串
        full_available_size = self.token_to_kv_pool_allocator.available_size() # Full可用大小
        full_evictable_size = self.full_evictable_size() # Full可驱逐大小
        return ( # 返回格式化字符串
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n" # 可用Full token数
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n" # Full LRU列表可驱逐大小
        )

    ##### Internal Helper Functions ##### # 内部辅助函数

    def _alloc_mamba_slot(self) -> torch.Tensor: # 分配一个Mamba池槽位
        """Allocate one mamba pool slot, evicting if necessary."""
        """分配一个Mamba池槽位，必要时进行驱逐。"""
        slot = self.req_to_token_pool.mamba_pool.alloc(1) # 尝试分配1个槽位
        if slot is None: # 如果分配失败
            self.evict(EvictParams(num_tokens=0, mamba_num=1)) # 驱逐1个Mamba状态
            slot = self.req_to_token_pool.mamba_pool.alloc(1) # 再次尝试分配
            assert slot is not None, "Can not alloc mamba cache" # 断言分配成功
        return slot # 返回分配的槽位

    def _match_prefix_helper( # 前缀匹配辅助函数
        self, key: RadixKey # 匹配键
    ) -> Tuple[List[torch.Tensor], TreeNode, int]: # 返回值列表、最佳匹配节点、最佳匹配长度
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        Mamba前缀匹配辅助函数。它考虑滑动窗口大小，保证匹配的节点要么
        1. 连接到根节点且没有Mamba墓碑，要么
        2. 从匹配节点到最后一个Mamba墓碑节点的匹配token数大于等于滑动窗口大小。
        """
        node = self.root_node # 从根节点开始
        child_key = key.child_key(self.page_size) # 获取子键

        value: List[torch.Tensor] = [] # 值列表
        best_value_len = 0 # 最佳匹配长度
        best_last_node = node # 最佳匹配节点
        while len(key) > 0 and child_key in node.children.keys(): # 循环匹配
            child = node.children[child_key] # 获取子节点
            # update best_value_len and best_last_node if needed
            # 如果需要，更新best_value_len和best_last_node
            if node.mamba_value is not None: # 如果当前节点有Mamba值
                best_value_len = len(value) # 更新最佳匹配长度
                best_last_node = node # 更新最佳匹配节点

            prefix_len = child.key.match(key, page_size=self.page_size) # 计算匹配前缀长度
            if prefix_len < len(child.key): # 如果匹配不完整
                new_node = self._split_node(child.key, child, prefix_len) # 分裂子节点
                value.append(new_node.value) # 添加分裂后新节点的值
                node = new_node # 移到新节点
                break # 跳出循环
            else: # 否则完全匹配
                value.append(child.value) # 添加子节点的值
                node = child # 移到子节点
                key = key[prefix_len:] # 截断已匹配的键

                if len(key): # 如果还有未匹配的键
                    child_key = key.child_key(self.page_size) # 获取下一个子键
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        # 处理best_value_len和best_last_node，针对最后节点完全匹配的情况
        if node.mamba_value is not None: # 如果最后节点有Mamba值
            best_value_len = len(value) # 更新最佳匹配长度
            best_last_node = node # 更新最佳匹配节点

        return value, best_last_node, best_value_len # 返回值列表、最佳匹配节点、最佳匹配长度

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]: # 匹配前预处理
        """Preprocess the key before matching."""
        """匹配前预处理键。"""
        key = params.key # 获取键

        if self.disable or len(key) == 0: # 如果禁用或键为空
            return None # 返回None

        return key # 返回键

    def _match_post_processor( # 匹配后处理
        self, # 自身实例
        params: MatchPrefixParams, # 匹配参数
        value: List[torch.Tensor], # 值列表
        last_node: TreeNode, # 最后节点
        best_value_len: int, # 最佳匹配长度
    ) -> MatchResult: # 返回匹配结果
        """Post-process the matched result."""
        """后处理匹配结果。"""
        cow_mamba = params.cow_mamba # 是否需要写时复制Mamba
        req = params.req # 请求对象

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows mamba to evict nodes closer to root first
        # 更新匹配节点的访问时间，使更接近根的节点为最少使用
        # 这允许Mamba优先驱逐更接近根的节点
        node_update = last_node # 从最后节点开始
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node) # 更新Full LRU
        self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node) # 更新Mamba LRU

        # This last_access_time is for sanity check, can be deleted after validation in production
        # 此last_access_time用于健全性检查，生产环境验证后可删除
        cur_time = get_last_access_time() # 获取当前时间
        while node_update: # 遍历到根
            node_update.last_access_time = cur_time # 更新访问时间
            cur_time -= ( # 递减时间
                0.00001  # assuming less than 100000 nodes in a branch of the tree # 假设树的分支中少于100000个节点
            )
            node_update = node_update.parent # 移到父节点

        # Calculate the branching point. It is defined as the last aligned position that
        # does not have a mamba value.
        # 计算分支点。定义为没有Mamba值的最后一个对齐位置。
        if len(value) > best_value_len: # 如果值列表超过最佳匹配长度
            chunk_aligned_seqlen = ( # 计算块对齐的序列长度
                sum(len(v) for v in value) // self.mamba_cache_chunk_size
            ) * self.mamba_cache_chunk_size
            mamba_branching_seqlen = ( # Mamba分支序列长度
                chunk_aligned_seqlen if chunk_aligned_seqlen > 0 else None # 如果为0则为None
            )
        else: # 否则无需分支
            mamba_branching_seqlen = None # 无分支

        # Defer COW to forward stream: record source index, allocate destination
        # 将写时复制延迟到前向流：记录源索引，分配目标
        if cow_mamba and last_node.mamba_value is not None: # 如果需要COW且有Mamba值
            if req.mamba_pool_idx is None: # 如果请求没有Mamba池索引
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1) # 分配目标索引
                if dst_index is None: # 如果分配失败
                    self.inc_lock_ref(last_node) # 增加锁引用（防止被驱逐）
                    self.evict(EvictParams(num_tokens=0, mamba_num=1)) # 驱逐1个Mamba状态
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1) # 再次分配
                    self.dec_lock_ref(last_node) # 减少锁引用
                    assert dst_index is not None, "Can not alloc mamba cache" # 断言分配成功
                req.mamba_pool_idx = dst_index[0] # 设置请求的Mamba池索引
            req.mamba_cow_src_index = last_node.mamba_value # 记录COW源索引
            req.mamba_needs_clear = False # 不需要清除

        value = value[:best_value_len] # 截断到最佳匹配长度
        if value: # 如果有值
            value = torch.cat(value) # 拼接值
        else: # 否则无匹配
            value = torch.empty((0,), dtype=torch.int64, device=self.device) # 创建空张量

        return MatchResult( # 返回匹配结果
            device_indices=value, # 设备索引
            last_device_node=last_node, # 最后设备节点
            last_host_node=last_node, # 最后主机节点
            best_match_node=last_node, # 最佳匹配节点
            mamba_branching_seqlen=mamba_branching_seqlen, # Mamba分支序列长度
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode: # 分裂节点
        # new_node -> child
        # new_node -> child（新节点成为child的父节点）
        new_node = TreeNode() # 创建新节点
        new_node.children = {key[split_len:].child_key(self.page_size): child} # 新节点的子节点为原child
        new_node.parent = child.parent # 新节点的父节点为原child的父节点
        new_node.mamba_value = None  # mamba cache can not be split # Mamba缓存不能被分裂
        new_node.full_lock_ref = child.full_lock_ref # 继承Full锁引用
        new_node.mamba_lock_ref = 0 # Mamba锁引用为0
        new_node.key = child.key[:split_len] # 新节点的键为前半部分
        new_node.value = child.value[:split_len].clone() # 新节点的值为前半部分（克隆）

        # child time should be later than parent's time for mamba tombstone
        # 子节点的时间应晚于父节点的时间（用于Mamba墓碑）
        child.last_access_time = get_last_access_time() # 更新子节点访问时间

        self.full_lru_list.remove_node(child) # 从Full LRU中移除子节点
        if child.mamba_value is not None: # 如果子节点有Mamba值
            self.mamba_lru_list.remove_node(child) # 从Mamba LRU中移除子节点
        child.parent = new_node # 子节点的父节点为新节点
        child.key = child.key[split_len:] # 子节点的键为后半部分
        child.value = child.value[split_len:].clone() # 子节点的值为后半部分（克隆）
        new_node.parent.children[key.child_key(self.page_size)] = new_node # 更新父节点的子节点引用
        new_node.hash_value, child.hash_value = split_node_hash_value( # 分裂哈希值
            child.hash_value, split_len, self.page_size
        )

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        # 将新节点和子节点插入LRU列表，先插入父节点
        # 使父节点在LRU列表中位于子节点之后
        self.full_lru_list.insert_mru(new_node) # 插入新节点到Full LRU
        self.full_lru_list.insert_mru(child) # 插入子节点到Full LRU
        if child.mamba_value is not None: # 如果子节点有Mamba值
            self.mamba_lru_list.insert_mru(child) # 插入子节点到Mamba LRU
        return new_node # 返回新节点

    def _insert_helper( # 插入辅助函数
        self, # 自身实例
        node: TreeNode, # 当前节点
        key: RadixKey, # 键
        value, # 值
        mamba_value, # Mamba值
        chunked: bool = False, # 是否分块
        prev_prefix_len: int = 0, # 前一个前缀长度
    ) -> Tuple[int, bool]: # 返回前缀长度和Mamba是否存在
        # Update the last access time from root to leaf, so that
        # mamba will tombstone the node closer to root first
        # 从根到叶更新最后访问时间，使Mamba优先将更接近根的节点标记为墓碑
        assert mamba_value is not None, "Mamba value should not be None here." # 断言Mamba值不为空
        node.last_access_time = get_last_access_time() # 更新当前节点访问时间
        if node != self.root_node: # 如果不是根节点
            self.full_lru_list.reset_node_mru(node) # 更新Full LRU位置
            if node.mamba_value is not None: # 如果有Mamba值
                self.mamba_lru_list.reset_node_mru(node) # 更新Mamba LRU位置
        if len(key) == 0: # 如果键为空
            return 0, True # 返回前缀长度0和Mamba存在

        child_key = key.child_key(self.page_size) # 获取子键

        total_prefix_length = 0 # 总前缀长度
        while len(key) > 0 and child_key in node.children.keys(): # 循环匹配已有节点
            node = node.children[child_key] # 获取子节点
            node.last_access_time = get_last_access_time() # 更新访问时间
            self.full_lru_list.reset_node_mru(node) # 更新Full LRU位置
            if node.mamba_value is not None: # 如果有Mamba值
                self.mamba_lru_list.reset_node_mru(node) # 更新Mamba LRU位置
            prefix_len = node.key.match(key, page_size=self.page_size) # 计算匹配前缀长度

            if prev_prefix_len < total_prefix_length + prefix_len: # 如果之前的前缀长度在当前范围内
                start = max(0, prev_prefix_len - total_prefix_length) # 计算起始位置
                self.token_to_kv_pool_allocator.free(value[start:prefix_len]) # 释放重复的KV索引

            total_prefix_length += prefix_len # 累加前缀长度
            key = key[prefix_len:] # 截断已匹配的键
            value = value[prefix_len:] # 截断已匹配的值

            if prefix_len < len(node.key): # 如果匹配不完整
                new_node = self._split_node(node.key, node, prefix_len) # 分裂节点
                node = new_node # 移到新节点

            if len(key): # 如果还有未匹配的键
                child_key = key.child_key(self.page_size) # 获取下一个子键

        mamba_value_exist = False # Mamba值是否存在
        if len(key): # 如果还有未匹配的键，需要创建新节点
            new_node = TreeNode() # 创建新节点
            new_node.parent = node # 设置父节点
            new_node.key = key # 设置键
            new_node.value = value.clone() # 设置值（克隆）
            new_node.mamba_value = mamba_value # 设置Mamba值
            self.full_lru_list.insert_mru(new_node) # 插入Full LRU
            self.mamba_lru_list.insert_mru(new_node) # 插入Mamba LRU
            node.children[child_key] = new_node # 添加为子节点
            self.full_evictable_size_ += len(value) # 增加Full可驱逐大小
            self.mamba_evictable_size_ += len(mamba_value) # 增加Mamba可驱逐大小
            self._record_store_event(new_node) # 记录存储事件
        elif node.mamba_value is None:  # add for mamba tombstone # 为Mamba墓碑添加Mamba值
            node.mamba_value = mamba_value # 设置Mamba值
            self.full_lru_list.reset_node_mru(node) # 更新Full LRU位置
            self.mamba_lru_list.insert_mru(node) # 插入Mamba LRU
            self.mamba_evictable_size_ += len(mamba_value) # 增加Mamba可驱逐大小
            node.last_access_time = get_last_access_time() # 更新访问时间
        else:  # mamba value already exists # Mamba值已存在
            mamba_value_exist = True # 标记为已存在
            self.full_lru_list.reset_node_mru(node) # 更新Full LRU位置
            self.mamba_lru_list.reset_node_mru(node) # 更新Mamba LRU位置
            node.last_access_time = get_last_access_time() # 更新访问时间

        return total_prefix_length, mamba_value_exist # 返回总前缀长度和Mamba是否存在

    def _iteratively_delete_tombstone_leaf( # 迭代删除墓碑叶子节点
        self, node: TreeNode # 起始节点
    ) -> Tuple[TreeNode, int]: # 返回最终节点和驱逐的Full token数
        full_num_evicted = 0 # 驱逐的Full token数
        while node.parent.mamba_value is None and len(node.parent.children) == 0: # 父节点是墓碑叶子
            # root node is not evictable
            # 根节点不可驱逐
            if node.parent == self.root_node: # 如果父节点是根节点
                break # 跳出
            # if locked, means node is in use, skip
            # 如果被锁定，表示节点正在使用，跳过
            if node.parent.full_lock_ref > 0: # 如果父节点的Full锁引用大于0
                break # 跳出
            assert ( # 断言墓碑节点的Mamba锁引用为0
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}" # 墓碑节点的Mamba锁引用应始终为0
            # delete tombstone node evicts full tokens
            # 删除墓碑节点会驱逐Full token
            self._record_remove_event(node.parent) # 记录移除事件
            self.token_to_kv_pool_allocator.free(node.parent.value) # 释放Full KV索引
            full_num_evicted += len(node.parent.value) # 累加驱逐数
            self.full_lru_list.remove_node(node.parent) # 从Full LRU中移除
            self._delete_tombstone_leaf(node.parent) # 删除墓碑叶子节点
            node = node.parent # 移到父节点

        return node, full_num_evicted # 返回最终节点和驱逐数

    def _delete_leaf(self, node: TreeNode) -> None: # 删除叶子节点
        assert ( # 断言叶子节点不是墓碑
            node.mamba_value is not None
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}" # 不变量违反：叶子节点是墓碑
        assert len(node.children) == 0, f"leaf node has children, {node.id=}" # 断言叶子节点没有子节点
        key = node.key.child_key(self.page_size) # 获取节点的子键
        v = node.parent.children.pop(key, None) # 从父节点的子节点中弹出
        assert v == node, f"parent does not have child key, {key}" # 断言弹出的节点就是当前节点

        self.full_evictable_size_ -= len(node.key) # 减少Full可驱逐大小
        self.mamba_evictable_size_ -= len(node.mamba_value) # 减少Mamba可驱逐大小

    def _tombstone_internal_node(self, node: TreeNode) -> None: # 将内部节点标记为墓碑
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}" # 断言不能对叶子节点标记墓碑
        self.mamba_evictable_size_ -= len(node.mamba_value) # 减少Mamba可驱逐大小
        node.mamba_value = None # 清除Mamba值（标记为墓碑）

    def _delete_tombstone_leaf(self, node: TreeNode) -> None: # 删除墓碑叶子节点
        assert ( # 断言是墓碑节点
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}" # 删除意外的非墓碑叶子节点
        assert len(node.children) == 0, f"leaf node has children, {node.id=}" # 断言没有子节点
        key = node.key.child_key(self.page_size) # 获取节点的子键
        v = node.parent.children.pop(key, None) # 从父节点的子节点中弹出
        assert v == node, f"parent does not have child key, {key}" # 断言弹出的节点就是当前节点

        self.full_evictable_size_ -= len(node.key) # 减少Full可驱逐大小

    def _collect_nontombstone_nodes(self) -> List[TreeNode]: # 收集所有非墓碑节点
        ret_list = [] # 返回列表
        stack = [self.root_node] # 栈初始化

        while stack: # 遍历栈
            cur_node = stack.pop() # 弹出节点
            if cur_node.mamba_value is not None: # 如果有Mamba值（非墓碑）
                ret_list.append(cur_node) # 添加到返回列表
            stack.extend(cur_node.children.values()) # 添加子节点到栈

        return ret_list # 返回非墓碑节点列表

    def _collect_all_nodes(self) -> List[TreeNode]: # 收集所有节点
        ret_list = [] # 返回列表
        stack = [self.root_node] # 栈初始化
        while stack: # 遍历栈
            cur_node = stack.pop() # 弹出节点
            ret_list.append(cur_node) # 添加到返回列表
            stack.extend(cur_node.children.values()) # 添加子节点到栈
        return ret_list # 返回所有节点列表

    def _print_helper(self, node: TreeNode, indent: int) -> None: # 打印辅助函数
        """Prints the radix tree in a human-readable format."""
        """以人类可读的格式打印前缀树。"""
        stack = [(node, indent)] # 栈初始化
        while stack: # 遍历栈
            current_node, current_indent = stack.pop() # 弹出节点和缩进
            print( # 打印节点信息
                " " * current_indent,
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items(): # 遍历子节点
                stack.append((child, current_indent + 2)) # 添加子节点到栈

                assert key == child.key.child_key( # 断言键与子节点键一致
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _total_size_helper(self) -> Tuple[int, int]: # 计算总大小辅助函数
        total_size = 0 # 总Full大小
        total_mamba_size = 0 # 总Mamba大小
        stack = [self.root_node] # 栈初始化
        while stack: # 遍历栈
            current_node = stack.pop() # 弹出节点
            total_size += len(current_node.value) # 累加Full大小
            if current_node.mamba_value is not None: # 如果有Mamba值
                total_mamba_size += len(current_node.mamba_value) # 累加Mamba大小
            for child in current_node.children.values(): # 遍历子节点
                if child.evicted: # 如果子节点已被驱逐
                    continue # 跳过
                stack.append(child) # 添加子节点到栈
        return total_size, total_mamba_size # 返回总Full大小和总Mamba大小
