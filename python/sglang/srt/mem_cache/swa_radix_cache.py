# 文件说明：SWA（Sliding Window Attention）Radix Tree缓存实现
# 本文件实现了混合full/SWA注意力架构下的基数树KV缓存管理，包含TreeNode节点、LRUList双向链表、
# SWARadixCache基数树缓存三大核心类，支持full和SWA独立的LRU淘汰、swa_tombstone机制、
# 以及基于滑动窗口大小的前缀匹配和锁引用计数管理。

from __future__ import annotations  # 启用延迟类型注解求值

"""
Copyright 2023-2024 SGLang Team  # 版权所有 2023-2024 SGLang团队
Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
You may obtain a copy of the License at  # 可在以下地址获取许可证

    http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证URL

Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不附带任何明示或暗示的担保或条件
See the License for the specific language governing permissions and  # 详见许可证中关于权限和
limitations under the License.  # 限制的条款
"""

"""
The radix tree data structure for managing the hybrid (full and SWA) KV cache.  # 管理混合（full和SWA）KV缓存的基数树数据结构
"""

import heapq  # 堆队列算法
import time  # 时间相关功能
from collections import defaultdict  # 默认字典
from typing import TYPE_CHECKING, List, Optional, Tuple  # 类型提示

import torch  # PyTorch深度学习框架
from numpy import float64  # NumPy的float64类型

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.mem_cache.base_prefix_cache import (  # 导入前缀缓存基类及相关参数/结果类
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # 导入缓存初始化参数
from sglang.srt.mem_cache.events import KVCacheEventMixin  # 导入KV缓存事件混入类
from sglang.srt.mem_cache.radix_cache import RadixKey  # 导入基数树键类
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator  # 导入SWA内存池分配器
from sglang.srt.mem_cache.utils import split_node_hash_value  # 导入节点哈希值分割工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.managers.schedule_batch import Req  # 导入请求类

import logging  # 日志记录库

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class TreeNode:  # 基数树节点类

    counter = 0  # 节点ID计数器
    swa_uuid_counter = 1  # SWA UUID计数器（从1开始，0表示未设置）
    last_access_time_counter_float = float64(1.0)  # 最后访问时间浮点计数器

    def __init__(self, id: Optional[int] = None):  # 初始化树节点
        self.children = defaultdict(TreeNode)  # 子节点字典
        self.parent: TreeNode = None  # 父节点
        self.key: RadixKey = None  # 节点的基数键
        self.value: Optional[torch.Tensor] = None  # KV缓存索引值张量
        # swa_tombstone is used to indicate the kv indices have been freed for swa layers  # swa_tombstone用于标记SWA层的KV索引已被释放
        self.swa_tombstone = False  # SWA墓碑标志
        # invariant: for any node, if swa_lock_ref is locked, full_lock_ref must be locked;  # 不变量：对于任何节点，如果swa_lock_ref被锁定，full_lock_ref也必须被锁定；
        # if full_lock_ref is locked, swa_lock_ref doesn't need to be locked. So,  # 如果full_lock_ref被锁定，swa_lock_ref不一定需要被锁定。因此，
        # full_lock_ref is always >= swa_lock_ref.  # full_lock_ref始终 >= swa_lock_ref。
        self.full_lock_ref = 0  # full侧锁引用计数
        self.swa_lock_ref = 0  # SWA侧锁引用计数
        # last access time is only used for sanity check. LRU is maintained by the lru list.  # 最后访问时间仅用于健全性检查。LRU由lru列表维护。
        self.last_access_time = get_last_access_time()  # 最后访问时间

        self.hit_count = 0  # 命中计数
        # store the host indices of KV cache  # 存储KV缓存的主机端索引
        self.host_value = None  # 主机端值
        # store hash values of each page  # 存储每页的哈希值
        self.hash_value: Optional[List[str]] = None  # 哈希值列表

        # for lru list, invariant:  # 用于LRU列表，不变量：
        # 1. prev has greater last_access_time  # 1. prev有更大的last_access_time
        # 2. next has smaller last_access_time  # 2. next有更小的last_access_time
        self.prev = None  # full LRU前驱节点
        self.next = None  # full LRU后继节点
        self.swa_prev = None  # SWA LRU前驱节点
        self.swa_next = None  # SWA LRU后继节点

        self.id = TreeNode.counter if id is None else id  # 节点ID
        TreeNode.counter += 1  # 递增ID计数器
        self.swa_uuid = None  # SWA唯一标识（用于dec_lock_ref）

    @property
    def evicted(self):  # 节点是否已被淘汰（value为None）
        return self.value is None  # 如果value为None则已淘汰

    @property
    def backuped(self):  # 节点是否已备份到主机
        return self.host_value is not None  # 如果host_value不为None则已备份

    def __lt__(self, other: "TreeNode"):  # 小于比较运算符（按最后访问时间排序）
        return self.last_access_time < other.last_access_time  # 比较最后访问时间


def gen_swa_uuid() -> int:  # 生成SWA唯一标识符
    TreeNode.swa_uuid_counter += 1  # 递增SWA UUID计数器
    return TreeNode.swa_uuid_counter  # 返回新的UUID


def get_last_access_time() -> float64:  # 获取单调递增的最后访问时间戳
    ret = TreeNode.last_access_time_counter_float  # 获取当前计数器值
    TreeNode.last_access_time_counter_float += 1.0  # 递增计数器
    return ret  # 返回时间戳


class LRUList:  # LRU双向链表，用于维护节点淘汰顺序
    def __init__(self, is_swa_list: bool = False):  # 初始化LRU列表，区分full和SWA列表
        self.is_swa_list = is_swa_list  # 是否为SWA LRU列表
        if self.is_swa_list:  # 如果是SWA列表
            self.prv = "swa_prev"  # SWA前驱属性名
            self.nxt = "swa_next"  # SWA后继属性名
            self.lock_ref = "swa_lock_ref"  # SWA锁引用属性名
        else:  # 如果是full列表
            self.prv = "prev"  # full前驱属性名
            self.nxt = "next"  # full后继属性名
            self.lock_ref = "full_lock_ref"  # full锁引用属性名
        # Initialize dummy head and tail nodes  # 初始化虚拟头尾节点
        self.head = TreeNode()  # Most recently used side  # 最近使用端（虚拟头节点）
        self.tail = TreeNode()  # Least recently used side  # 最久未使用端（虚拟尾节点）
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail  # 头节点后继指向尾节点
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head  # 尾节点前驱指向头节点
        self.cache = {}  # 节点ID到节点的映射缓存

    def _add_node(self, node):  # 将节点添加到头部之后（最近使用位置）
        """Helper to add node right after head (most recently used)"""  # 辅助方法：在头节点之后添加节点（最近使用位置）
        self._add_node_after(self.head, node)  # 在头节点后添加

    def _add_node_after(self, old_node, new_node):  # 在指定节点之后添加新节点
        """Helper to add node right after old_node"""  # 辅助方法：在old_node之后添加节点
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node  # 新节点前驱指向旧节点
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next  # 新节点后继指向旧节点的原后继
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node  # 旧节点原后继的前驱指向新节点
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node  # 旧节点后继指向新节点

    def _remove_node(self, node):  # 从链表中移除节点
        """Helper to remove node from linked list"""  # 辅助方法：从链表中移除节点
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next  # 前驱的后继指向后继
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev  # 后继的前驱指向前驱
        # Clear self pointers to break reference cycles among evicted nodes.  # 清除自身指针以打破被淘汰节点之间的引用循环。
        setattr(node, self.prv, None)  # 清除前驱指针
        setattr(node, self.nxt, None)  # 清除后继指针

    def _get_lru(self) -> Optional[TreeNode]:  # 获取最久未使用的节点
        """
        Get the least recently used node  # 获取最久未使用的节点
        """
        if len(self.cache) == 0:  # 如果缓存为空
            return None  # 返回None
        return getattr(self.tail, self.prv)  # 返回尾节点的前驱（即最久未使用的节点）

    def reset_node_mru(self, node):  # 将已有节点移到最近使用位置
        """
        Move a (existing) node to most recently used position  # 将（已有）节点移动到最近使用位置
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"  # 断言节点在LRU列表中
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Resetting swa tombstone node in swa lru list: {node.id=}"  # SWA列表中不允许重置墓碑节点
        self._remove_node(node)  # 从当前位置移除
        self._add_node(node)  # 添加到头部（最近使用位置）

    def reset_node_and_parents_mru(self, node, root_node):  # 将节点及其父节点移到最近使用位置
        """
        Move an (existing) node and its parents to most recently used position. Child node is  # 将（已有）节点及其父节点移动到最近使用位置。子节点比
        more recently used than parent node.  # 父节点更最近使用。
        """
        prev_node = self.head  # 上一个插入的节点（从头节点开始）
        while node != root_node:  # 从当前节点遍历到根节点
            # for swa lru list, only reset non-tombstone nodes  # 对于SWA LRU列表，仅重置非墓碑节点
            if not self.is_swa_list or not node.swa_tombstone:  # 非墓碑节点才处理
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"  # 断言节点在LRU列表中
                self._remove_node(node)  # 从当前位置移除
                self._add_node_after(prev_node, node)  # 插入到上一个节点之后
                prev_node = node  # 更新上一个节点
            node = node.parent  # 移动到父节点

    def insert_mru(self, node):  # 将新节点插入到最近使用位置
        """
        Insert a (new) node as most recently used  # 将（新）节点插入为最近使用
        """
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Inserting swa tombstone node in swa lru list: {node.id=}"  # SWA列表中不允许插入墓碑节点
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"  # 断言节点不在LRU列表中
        self.cache[node.id] = node  # 将节点添加到缓存
        self._add_node(node)  # 添加到头部

    def remove_node(self, node: TreeNode):  # 从LRU列表中移除节点
        """
        Remove node from lru list  # 从LRU列表中移除节点
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"  # 断言节点在LRU列表中
        assert (
            not self.is_swa_list or not node.swa_tombstone
        ), f"Removing swa tombstone node from swa lru list: {node.id=}"  # SWA列表中不允许移除墓碑节点
        del self.cache[node.id]  # 从缓存中删除
        self._remove_node(node)  # 从链表中移除

    def get_lru_no_lock(self) -> Optional[TreeNode]:  # 获取最久未使用且未锁定的节点
        """
        Get the least recently used node that is not locked  # 获取最久未使用且未被锁定的节点
        """
        return self.get_prev_no_lock(self.tail, check_id=False)  # 从尾节点向前查找

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:  # 获取最久未使用且未锁定的叶子节点
        """
        Get the least recently used leaf node that is not locked  # 获取最久未使用且未被锁定的叶子节点
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)  # 从尾节点向前查找叶子节点

    def get_prev_no_lock(  # 获取指定节点之前（更近使用）的未锁定节点
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked  # 获取之前（即更近使用的）未被锁定的节点
        """
        if check_id:  # 如果需要检查ID
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"  # 断言节点在LRU列表中
        x = getattr(node, self.prv)  # x = node.prev  # 获取前驱节点
        while getattr(x, self.lock_ref) > 0:  # 当前驱节点被锁定
            x = getattr(x, self.prv)  # x = x.prev  # 继续向前查找
        # if x is the head, it means there is no node in the lru list without lock  # 如果x是头节点，说明LRU列表中没有未锁定的节点
        if x == self.head:  # 到达头节点
            return None  # 返回None
        return x  # 返回找到的未锁定节点

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):  # 获取指定节点之前未锁定的叶子节点
        """
        Get the previous (i.e. more recently used) leaf node that is not locked  # 获取之前（即更近使用的）未被锁定的叶子节点
        """
        if check_id:  # 如果需要检查ID
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"  # 断言节点在LRU列表中
        x = getattr(node, self.prv)  # x = node.prev  # 获取前驱节点
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:  # 节点被锁定或有子节点（非叶子）
            x = getattr(x, self.prv)  # x = x.prev  # 继续向前查找
        # if x is the head, it means there is no leaf node in the lru list without lock  # 如果x是头节点，说明LRU列表中没有未锁定的叶子节点
        if x == self.head:  # 到达头节点
            return None  # 返回None
        return x  # 返回找到的未锁定叶子节点

    def in_list(self, node: Optional[TreeNode]):  # 检查节点是否在LRU列表中
        """
        Check if the node is in the lru list  # 检查节点是否在LRU列表中
        """
        if not node:  # 如果节点为None
            return False  # 返回False
        return node.id in self.cache  # 检查节点ID是否在缓存中

    # Note: this is expensive, only use for debug  # 注意：此操作开销大，仅用于调试
    def sanity_check_evictable_size(self):  # 健全性检查：计算可淘汰节点的大小
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)  # 检查可淘汰大小（即未被锁定的节点大小）
        """
        node = self.get_lru_no_lock()  # 获取最久未使用的未锁定节点
        evictable_size = 0  # 可淘汰大小
        while self.in_list(node):  # 遍历所有可淘汰节点
            evictable_size += len(node.value)  # 累加节点值大小
            node = self.get_prev_no_lock(node)  # 移动到下一个可淘汰节点
        return evictable_size  # 返回可淘汰大小

    # Note: this is expensive, only use for debug or idle check  # 注意：此操作开销大，仅用于调试或空闲检查
    def sanity_check(self, tree_cache: "SWARadixCache"):  # 健全性检查：验证LRU列表与树结构一致
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and  # 通过从树重建LRU列表、建堆并
        checking if the lru list is valid.  # 检查LRU列表是否有效。
        """
        try:
            if self.is_swa_list:  # 如果是SWA列表
                nodes = tree_cache._collect_nontombstone_nodes()  # 收集所有非墓碑节点
            else:  # 如果是full列表
                nodes = tree_cache._collect_all_nodes()  # 收集所有节点
            total_nodes = len(nodes)  # 节点总数
            total_lru_plus_1 = len(self.cache) + 1  # LRU列表大小加1（包含根节点）
            # heapify based on last_access_time  # 基于last_access_time建堆
            heapq.heapify(nodes)  # 将节点列表转为最小堆
            # the root node is not in the lru list  # 根节点不在LRU列表中
            assert (
                len(nodes) == len(self.cache) + 1
            ), f"len(nodes): {len(nodes)} != len(self.cache) + 1: {len(self.cache) + 1}"  # 断言节点数等于LRU列表大小+1

            x_lru = self._get_lru()  # 获取LRU列表中最久未使用的节点
            while len(nodes):  # 遍历堆中的所有节点
                x = heapq.heappop(nodes)  # 弹出最小元素（最早访问时间）
                if x == tree_cache.root_node:  # 如果是根节点
                    # root node is not in the lru list  # 根节点不在LRU列表中
                    continue  # 跳过
                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.is_swa_list=}, x: {x.id=} != x_lru: {x_lru.id=}"  # 断言堆顺序与LRU列表一致
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"  # 断言空闲时未锁定
                assert (
                    x_lru.swa_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.swa_lock_ref=}, {x_lru.swa_uuid=}, {x_lru.id=}"  # 断言空闲时SWA未锁定
                x_lru = getattr(x, self.prv)  # 移动到LRU列表中的前一个节点

            if self.is_swa_list:  # 如果是SWA列表
                evictable_size = tree_cache.swa_evictable_size()  # 获取SWA可淘汰大小
                lru_list_evictable_size = self.sanity_check_evictable_size()  # 获取LRU列表可淘汰大小
            else:  # 如果是full列表
                evictable_size = tree_cache.full_evictable_size()  # 获取full可淘汰大小
                lru_list_evictable_size = self.sanity_check_evictable_size()  # 获取LRU列表可淘汰大小

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.is_swa_list=}, total nodes: {total_nodes}, total lru plus 1: {total_lru_plus_1}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"  # 断言可淘汰大小一致
        except Exception as e:  # 捕获异常
            msg = f"SWA Radix tree sanity check failed, ping @hanming-lu: {e}"  # 构造错误消息
            logger.error(msg)  # 记录错误
            raise Exception(msg)  # 抛出异常


class SWARadixCache(KVCacheEventMixin, BasePrefixCache):  # SWA基数树缓存，管理混合full/SWA KV缓存
    def __init__(self, params: CacheInitParams):  # 初始化SWA基数树缓存
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)  # 断言使用SWA分配器
        self.req_to_token_pool = params.req_to_token_pool  # 请求到token的映射池
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator  # token到KV池的分配器
        self.page_size = params.page_size  # 页大小
        self.disable = params.disable  # 是否禁用缓存
        self.is_eagle = params.is_eagle  # 是否为EAGLE推测解码
        self.enable_kv_cache_events = params.enable_kv_cache_events  # 是否启用KV缓存事件
        self.kv_event_queue = []  # KV事件队列

        if self.token_to_kv_pool_allocator:  # 如果分配器存在
            self.device = self.token_to_kv_pool_allocator.device  # 获取设备类型
        else:  # 分配器不存在
            self.device = torch.device("cpu")  # 默认使用CPU

        if params.enable_metrics:  # 如果启用指标收集
            self.init_metrics_collector()  # 初始化指标收集器

        self.sliding_window_size = params.sliding_window_size  # 滑动窗口大小
        self.reset()  # 重置缓存状态

    ##### Public API #####  ##### 公共API #####

    def supports_swa(self) -> bool:  # 检查是否支持SWA
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for SWARadixCache"  # 断言滑动窗口大小已设置
        return True  # 返回True（始终支持SWA）

    def reset(self) -> None:  # 重置缓存状态
        self.root_node = TreeNode()  # 创建根节点
        self.root_node.key = []  # 根节点键为空列表
        self.root_node.value = []  # 根节点值为空列表
        self.root_node.hash_value = []  # 根节点哈希值为空列表
        self.root_node.full_lock_ref = 1  # 根节点full锁引用为1（永不被淘汰）
        self.root_node.swa_lock_ref = 1  # 根节点SWA锁引用为1（永不被淘汰）
        self.full_evictable_size_ = 0  # full可淘汰大小
        self.swa_evictable_size_ = 0  # SWA可淘汰大小
        self.full_protected_size_ = 0  # full受保护大小
        self.swa_protected_size_ = 0  # SWA受保护大小
        # LRU lists are used to maintain the order of eviction of the nodes in the tree  # LRU列表用于维护树中节点的淘汰顺序
        self.full_lru_list = LRUList(is_swa_list=False)  # full LRU列表
        self.swa_lru_list = LRUList(is_swa_list=True)  # SWA LRU列表
        self._record_all_cleared_event()  # 记录缓存清空事件

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # 前缀匹配
        """Find the matching prefix from the radix tree.  # 从基数树中查找匹配的前缀
        Args:  # 参数：
            params: MatchPrefixParams containing key.  # params：包含键的MatchPrefixParams
        Returns:  # 返回：
            A tuple of a tensor of matching prefix token IDs and  # 匹配前缀token ID的张量和
            the last node that contains the prefix values. Note that  # 包含前缀值的最后节点的元组。注意
            this API can modify the internal state of the Radix tree.  # 此API可能修改基数树的内部状态。
            The last node create a new child if the prefix is shorter  # 如果前缀短于最后节点的值，
            than the last node's value.  # 最后节点会创建一个新子节点。
        """

        key = self._match_pre_processor(params)  # 预处理键
        if key is None:  # 如果键无效
            return MatchResult(  # 返回空匹配结果
                device_indices=torch.empty(  # 空的设备索引张量
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,  # 最后设备节点为根节点
                last_host_node=self.root_node,  # 最后主机节点为根节点
                best_match_node=self.root_node,  # 最佳匹配节点为根节点
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)  # 执行前缀匹配
        return self._match_post_processor(params, value, last_node, best_value_len)  # 后处理匹配结果

    def insert(self, params: InsertParams) -> InsertResult:  # 插入KV缓存到基数树
        if self.disable:  # 如果缓存被禁用
            return InsertResult(prefix_len=0)  # 返回前缀长度为0

        key = params.key  # 获取键
        value = params.value  # 获取值
        prev_prefix_len = params.prev_prefix_len  # 获取之前的前缀长度
        swa_evicted_seqlen = params.swa_evicted_seqlen  # 获取SWA已淘汰的序列长度

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)  # 处理bigram视图
        key = key.page_aligned(self.page_size)  # 页对齐键
        if value is not None:  # 如果值不为None
            value = value[: len(key)]  # 截断值到键长度
        else:  # 值为None
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)  # 从键创建值张量

        prefix_len = self._insert_helper(  # 执行插入
            self.root_node, key, value, prev_prefix_len, swa_evicted_seqlen  # 从根节点开始插入
        )
        return InsertResult(prefix_len=prefix_len)  # 返回插入结果

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:  # 缓存已完成的请求
        """Cache request when it finishes."""  # 请求完成时缓存
        kv_committed_len = req.pop_committed_kv_cache()  # 获取已提交的KV缓存长度
        if self.disable:  # 如果缓存被禁用
            kv_indices = self.req_to_token_pool.req_to_token[  # 获取KV索引
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)  # 直接释放KV索引
            return  # 返回

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]  # 获取token ID
        kv_indices = self.req_to_token_pool.req_to_token[  # 获取KV索引
            req.req_pool_idx, :kv_committed_len
        ]

        radix_key = RadixKey(  # 创建基数键
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)  # 页对齐
        page_aligned_len = len(radix_key)  # 页对齐后的长度
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)  # 创建值张量的副本
        old_prefix_len = req.cache_protected_len  # 获取旧的前缀保护长度

        # Radix Cache takes one ref in memory pool  # 基数缓存在内存池中持有一个引用
        # Note: the insert function already frees the overlapped kv_indices  # 注意：insert函数已释放重叠的kv_indices
        if is_insert:  # 如果需要插入
            self.insert(  # 插入到基数树
                InsertParams(
                    key=radix_key,  # 基数键
                    value=values,  # 值
                    prev_prefix_len=old_prefix_len,  # 旧前缀长度
                    swa_evicted_seqlen=req.swa_evicted_seqlen,  # SWA已淘汰序列长度
                )
            )
        else:  # 不插入
            self.token_to_kv_pool_allocator.free(  # 释放KV索引
                kv_indices[old_prefix_len:page_aligned_len]
            )

        # free the unaligned tail  # 释放未对齐的尾部
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])  # 释放页对齐后的剩余索引

        # Remove req slot release the cache lock  # 移除请求槽位，释放缓存锁
        self.dec_lock_ref(  # 减少锁引用
            req.last_node,  # 最后节点
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),  # SWA UUID参数
            skip_swa=req.swa_prefix_lock_released,  # 是否跳过SWA
        )
        req.swa_prefix_lock_released = False  # 重置SWA前缀锁释放标志

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:  # 缓存未完成的请求
        """Cache request when it is unfinished."""  # 请求未完成时缓存
        if self.disable:  # 如果缓存被禁用
            kv_indices = self.req_to_token_pool.req_to_token[  # 获取KV索引
                req.req_pool_idx, : len(req.fill_ids)
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later  # `req.prefix_indices`稍后将在`PrefillAdder::add_chunked_req`中使用
            req.prefix_indices = kv_indices  # 设置前缀索引
            return  # 返回

        token_ids = req.fill_ids  # 获取填充ID
        kv_indices = self.req_to_token_pool.req_to_token[  # 获取KV索引
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(  # 创建基数键
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)  # 页对齐
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)  # 创建值张量的副本
        old_prefix_len = req.cache_protected_len  # 获取旧的前缀保护长度

        # Radix Cache takes one ref in memory pool  # 基数缓存在内存池中持有一个引用
        # Note: the insert function already frees the overlapped kv_indices  # 注意：insert函数已释放重叠的kv_indices
        result = self.insert(  # 插入到基数树
            InsertParams(
                key=radix_key,  # 基数键
                value=values,  # 值
                prev_prefix_len=old_prefix_len,  # 旧前缀长度
            )
        )
        new_prefix_len = result.prefix_len  # 获取新的前缀长度

        # The prefix indices could be updated, reuse it  # 前缀索引可能已更新，重新使用
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))  # 重新匹配前缀
        new_indices, new_last_node = (  # 获取新的索引和最后节点
            match_result.device_indices,
            match_result.last_device_node,
        )

        assert old_prefix_len <= len(new_indices), f"{old_prefix_len=}, {new_indices=}"  # 断言旧前缀长度不大于新索引长度
        assert new_prefix_len <= len(new_indices), f"{new_prefix_len=}, {new_indices=}"  # 断言新前缀长度不大于新索引长度
        self.req_to_token_pool.write(  # 写入请求到token映射
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),  # 写入位置
            new_indices[old_prefix_len:],  # 写入新索引
        )

        req.cache_protected_len = len(new_indices)  # 更新缓存保护长度

        self.dec_lock_ref(  # 减少旧节点的锁引用
            req.last_node,  # 旧的最后节点
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),  # SWA UUID参数
            skip_swa=req.swa_prefix_lock_released,  # 是否跳过SWA
        )
        req.swa_prefix_lock_released = False  # 重置SWA前缀锁释放标志
        result = self.inc_lock_ref(new_last_node)  # 增加新节点的锁引用
        swa_uuid_for_lock = result.swa_uuid_for_lock  # 获取新的SWA UUID

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later  # `req.prefix_indices`稍后将在`PrefillAdder::add_chunked_req`中使用
        if len(new_indices) < len(kv_indices):  # 如果新索引短于原索引
            req.prefix_indices = torch.cat(  # 拼接新旧索引
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:  # 新索引足够长
            req.prefix_indices = new_indices  # 直接使用新索引
        req.last_node = new_last_node  # 更新最后节点
        req.swa_uuid_for_lock = swa_uuid_for_lock  # 更新SWA UUID

    def pretty_print(self) -> None:  # 以可读格式打印基数树
        self._print_helper(self.root_node, 0)  # 从根节点开始打印
        total_size, total_swa_size = self._total_size_helper()  # 获取总大小
        print(f"#full_tokens: {total_size}, #swa_tokens: {total_swa_size}")  # 打印full和SWA token数

    def total_size(self) -> Tuple[int, int]:  # 获取缓存的总大小
        return self._total_size_helper()  # 返回(full总大小, SWA总大小)

    def evict(self, params: EvictParams) -> EvictResult:  # 淘汰KV缓存
        if self.disable:  # 如果缓存被禁用
            return EvictResult()  # 返回空结果
        start_time = time.perf_counter()  # 记录开始时间
        full_num_tokens = params.num_tokens  # 需要淘汰的full token数
        swa_num_tokens = params.swa_num_tokens  # 需要淘汰的SWA token数
        full_num_evicted = 0  # 已淘汰的full token数
        swa_num_evicted = 0  # 已淘汰的SWA token数
        if full_num_tokens > 0:  # 如果需要淘汰full token
            # get the least recently used leaf node that is not locked  # 获取最久未使用且未锁定的叶子节点
            x = self.full_lru_list.get_leaf_lru_no_lock()  # 获取LRU叶子节点

            while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):  # 循环淘汰直到满足需求
                assert (
                    x != self.root_node
                ), f"root node should not exist in full lru list, {x.id=}"  # 断言根节点不在LRU列表中
                assert x.full_lock_ref == 0, f"node is in use, {x.id=}"  # 断言节点未被锁定

                # 1. free node kv indices, evict full and swa tokens  # 1. 释放节点的KV索引，淘汰full和SWA token
                self._record_remove_event(x)  # 记录移除事件
                self.token_to_kv_pool_allocator.free(x.value)  # 释放KV索引
                full_num_evicted += len(x.value)  # 累加full淘汰数
                # Tombstoned leaves had their SWA freed earlier in `dec_swa_lock_only`  # 墓碑叶子的SWA已在`dec_swa_lock_only`中提前释放
                if not x.swa_tombstone:  # 如果非墓碑节点
                    swa_num_evicted += len(x.value)  # 累加SWA淘汰数

                # 2. get the next leaf, update the lru lists  # 2. 获取下一个叶子节点，更新LRU列表
                x_next = self.full_lru_list.get_prev_leaf_no_lock(x)  # 获取下一个可淘汰叶子
                self.full_lru_list.remove_node(x)  # 从full LRU列表中移除
                if not x.swa_tombstone:  # 如果非墓碑节点
                    self.swa_lru_list.remove_node(x)  # 从SWA LRU列表中移除

                # 3. delete the leaf node  # 3. 删除叶子节点
                self._delete_leaf(x)  # 从树中删除叶子节点

                # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone  # 4. 迭代删除墓碑叶子节点，维持叶子节点非墓碑的不变量
                x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)  # 删除墓碑叶子
                full_num_evicted += leaf_full_num_evicted  # 累加淘汰数

                # 5. if parent has no more children, it is a leaf. It is possible that this node is lru, so  # 5. 如果父节点没有子节点，它就是叶子。该节点可能是LRU，因此
                # we need to get the first leaf node in the lru list  # 我们需要获取LRU列表中的第一个叶子节点
                if len(x.parent.children) == 0:  # 如果父节点没有子节点
                    x_next = self.full_lru_list.get_leaf_lru_no_lock()  # 重新获取LRU叶子

                x = x_next  # 移动到下一个节点

        if swa_num_evicted < swa_num_tokens:  # 如果SWA淘汰不足
            # get the least recently used node that is not locked, doesn't have to be a leaf  # 获取最久未使用且未锁定的节点，不要求是叶子
            x = self.swa_lru_list.get_lru_no_lock()  # 获取SWA LRU节点

            # evict lru leaf nodes until swa_num_tokens is reached  # 淘汰LRU节点直到SWA淘汰数达标
            while swa_num_evicted < swa_num_tokens and (self.swa_lru_list.in_list(x)):  # 循环淘汰
                assert not x.swa_tombstone, f"duplicate swa tombstone node, {x.id=}"  # 断言非墓碑节点
                assert x != self.root_node, f"root node is not evictable, {x.id=}"  # 断言不是根节点
                assert x.swa_lock_ref == 0, f"node is in use by swa kv indices, {x.id=}"  # 断言SWA锁为0

                if len(x.children) > 0:  # 如果是内部节点（有子节点）
                    # 1. an internal node, free swa tokens.  # 1. 内部节点，释放SWA token
                    self.token_to_kv_pool_allocator.free_swa(x.value)  # 仅释放SWA侧KV
                    swa_num_evicted += len(x.value)  # 累加SWA淘汰数

                    # 2. get the next node, update the lru lists  # 2. 获取下一个节点，更新LRU列表
                    x_next = self.swa_lru_list.get_prev_no_lock(x)  # 获取下一个可淘汰节点
                    self.swa_lru_list.remove_node(x)  # 从SWA LRU列表移除

                    # 3. tombstone the node  # 3. 将节点标记为墓碑
                    self._tombstone_internal_node(x)  # 标记内部节点为墓碑
                elif x.full_lock_ref > 0:  # 叶子节点仍持有full锁
                    # Leaf still holds a full-side lock (can happen when the  # 叶子节点仍持有full侧锁（可能发生在
                    # SWA leaf-lock early-release optimization revived a  # SWA叶子锁提前释放优化恢复了
                    # tombstoned leaf. Treat it like an internal tombstone.  # 墓碑叶子。将其视为内部墓碑。
                    self.token_to_kv_pool_allocator.free_swa(x.value)  # 释放SWA侧KV
                    swa_num_evicted += len(x.value)  # 累加SWA淘汰数

                    x_next = self.swa_lru_list.get_prev_no_lock(x)  # 获取下一个节点
                    self.swa_lru_list.remove_node(x)  # 从SWA LRU列表移除

                    self.swa_evictable_size_ -= len(x.value)  # 更新SWA可淘汰大小
                    x.swa_tombstone = True  # 标记为墓碑
                else:  # 普通叶子节点（无full锁）
                    assert (
                        x.full_lock_ref == 0
                    ), f"leaf node with full lock must also have swa lock, {x.id=}"  # 断言full锁为0
                    # 1. a leaf node, free full and swa tokens  # 1. 叶子节点，释放full和SWA token
                    self._record_remove_event(x)  # 记录移除事件
                    self.token_to_kv_pool_allocator.free(x.value)  # 释放full和SWA KV
                    full_num_evicted += len(x.value)  # 累加full淘汰数
                    swa_num_evicted += len(x.value)  # 累加SWA淘汰数

                    # 2. get the next node, update the lru lists  # 2. 获取下一个节点，更新LRU列表
                    x_next = self.swa_lru_list.get_prev_no_lock(x)  # 获取下一个节点
                    self.full_lru_list.remove_node(x)  # 从full LRU列表移除
                    self.swa_lru_list.remove_node(x)  # 从SWA LRU列表移除

                    # 3. delete the leaf node  # 3. 删除叶子节点
                    self._delete_leaf(x)  # 从树中删除

                    # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone  # 4. 迭代删除墓碑叶子以维持叶子非墓碑不变量
                    self._iteratively_delete_tombstone_leaf(x)  # 删除墓碑叶子

                x = x_next  # 移动到下一个节点

        self.update_eviction_metrics(full_num_evicted + swa_num_evicted, start_time)  # 更新淘汰指标
        return EvictResult(  # 返回淘汰结果
            num_tokens_evicted=full_num_evicted, swa_num_tokens_evicted=swa_num_evicted  # full和SWA淘汰数
        )

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:  # 增加锁引用计数
        """
        Increment the lock reference count for the node. Returns the swa_uuid_for_lock, which needs  # 增加节点的锁引用计数。返回swa_uuid_for_lock，需要
        to be passed to dec_lock_ref.  # 传递给dec_lock_ref。
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.  # 锁定[last node, root)之间节点的full_lock_ref，不含root。
        It locks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.  # 锁定[last node, swa_uuid_for_lock]之间节点的swa_lock_ref，包含两端。
        """
        if self.disable:  # 如果缓存被禁用
            return IncLockRefResult()  # 返回空结果

        swa_lock_size = 0  # 已锁定的SWA token数
        swa_uuid_for_lock = None  # SWA UUID
        while node != self.root_node:  # 从当前节点遍历到根节点
            # lock full from node to root  # 从节点到根锁定full锁
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"  # 断言full锁引用非负
            if node.full_lock_ref == 0:  # 如果是首次锁定
                self.full_evictable_size_ -= len(node.value)  # 减少full可淘汰大小
                self.full_protected_size_ += len(node.value)  # 增加full受保护大小
            node.full_lock_ref += 1  # 增加full锁引用

            # lock swa if we have not reached the sliding window size.  # 如果尚未达到滑动窗口大小，则锁定SWA。
            # When we reach the sliding window size, we will set the swa_uuid_for_lock.  # 当达到滑动窗口大小时，设置swa_uuid_for_lock。
            # caller needs to pass the swa_uuid_for_lock to dec_lock_ref  # 调用者需要将swa_uuid_for_lock传递给dec_lock_ref
            if swa_lock_size < self.sliding_window_size:  # 如果SWA锁定大小未达到窗口大小
                assert (
                    not node.swa_tombstone
                ), f"inc_lock_swa on swa_tombstone node, {node.id=}"  # 断言非墓碑节点
                if node.swa_lock_ref == 0:  # 如果是首次锁定SWA
                    self.swa_evictable_size_ -= len(node.value)  # 减少SWA可淘汰大小
                    self.swa_protected_size_ += len(node.value)  # 增加SWA受保护大小
                node.swa_lock_ref += 1  # 增加SWA锁引用
                swa_lock_size += len(node.value)  # 累加SWA锁定大小
                if swa_lock_size >= self.sliding_window_size:  # 如果达到窗口大小
                    if node.swa_uuid is None:  # 如果节点没有SWA UUID
                        node.swa_uuid = gen_swa_uuid()  # 生成SWA UUID
                    swa_uuid_for_lock = node.swa_uuid  # 记录SWA UUID
            node = node.parent  # 移动到父节点
        return IncLockRefResult(swa_uuid_for_lock=swa_uuid_for_lock)  # 返回结果（包含SWA UUID）

    def dec_lock_ref(  # 减少锁引用计数
        self,
        node: TreeNode,  # 起始节点
        params: Optional[DecLockRefParams] = None,  # 解锁参数
        skip_swa: bool = False,  # 是否跳过SWA解锁
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.  # 减少节点的锁引用计数。
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.  # 解锁[last node, root)之间节点的full_lock_ref，不含root。
        It unlocks the swa_lock_ref for nodes between the [last node, swa_uuid_for_lock], inclusive.  # 解锁[last node, swa_uuid_for_lock]之间节点的swa_lock_ref，包含两端。
        If swa_uuid_for_lock is None, it unlocks to the root, exclusive.  # 如果swa_uuid_for_lock为None，则解锁到根节点（不含）。

        If skip_swa is True, only the full_lock_ref is decremented; the SWA lock is  # 如果skip_swa为True，仅减少full_lock_ref；SWA锁
        assumed to have been released already (e.g. via `dec_swa_lock_only`).  # 假设已通过`dec_swa_lock_only`释放。
        """
        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None  # 获取SWA UUID

        if self.disable:  # 如果缓存被禁用
            return DecLockRefResult()  # 返回空结果

        dec_lock_swa = not skip_swa  # 是否减少SWA锁
        while node != self.root_node:  # 从当前节点遍历到根节点
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"  # 断言full锁引用大于0
            if node.full_lock_ref == 1:  # 如果这是最后一次解锁
                self.full_evictable_size_ += len(node.value)  # 增加full可淘汰大小
                self.full_protected_size_ -= len(node.value)  # 减少full受保护大小
            node.full_lock_ref -= 1  # 减少full锁引用

            if dec_lock_swa:  # 如果需要减少SWA锁
                assert (
                    not node.swa_tombstone
                ), f"dec_lock_ref on swa_tombstone node, {node.id=}"  # 断言非墓碑节点
                assert (
                    node.swa_lock_ref > 0
                ), f"dec_lock_ref on node with {node.swa_lock_ref=}, {node.id=}"  # 断言SWA锁引用大于0

                if node.swa_lock_ref == 1:  # 如果这是最后一次SWA解锁
                    self.swa_evictable_size_ += len(node.value)  # 增加SWA可淘汰大小
                    self.swa_protected_size_ -= len(node.value)  # 减少SWA受保护大小
                node.swa_lock_ref -= 1  # 减少SWA锁引用
                if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:  # 如果到达SWA UUID节点
                    dec_lock_swa = False  # 停止减少SWA锁

            node = node.parent  # 移动到父节点

        return DecLockRefResult()  # 返回结果

    def dec_swa_lock_only(  # 仅减少SWA锁引用（保留full锁）
        self, node: TreeNode, swa_uuid_for_lock: Optional[int] = None
    ):
        """
        Decrement only the swa_lock_ref (and swa_protected_size_) along the chain  # 仅沿链路减少swa_lock_ref（和swa_protected_size_）
        [node, swa_uuid_for_lock], inclusive. The full_lock_ref is left untouched  # [node, swa_uuid_for_lock]，包含两端。full_lock_ref保持不变，
        so the caller's full-cache protection is preserved.  # 以保留调用者的full缓存保护。

        Used to early-release the SWA portion of a request's tree lock once the  # 用于在请求的解码位置超过滑动窗口后，
        request's decode position has advanced past the sliding window, so the  # 提前释放请求树锁的SWA部分，使
        protected window can be reclaimed.  # 受保护窗口可被回收。

        For internal nodes, the standard protected -> evictable transition is  # 对于内部节点，应用标准的受保护→可淘汰转换
        applied (node stays in swa_lru_list and may be evicted by SWA LRU later).  # （节点留在swa_lru_list中，可能稍后被SWA LRU淘汰）。
        For leaf nodes, since `swa_lru_list` cannot contain a leaf with  # 对于叶子节点，由于`swa_lru_list`不能包含持有
        `full_lock_ref > 0` (SWA-eviction would also delete the still-referenced  # `full_lock_ref > 0`的叶子（SWA淘汰也会删除仍被引用的
        leaf), we instead free the SWA pool slots immediately and mark the leaf  # 叶子），我们改为立即释放SWA池槽位并标记叶子为
        as `swa_tombstone=True`. The full kv stays alive until the full-side  # `swa_tombstone=True`。full KV在full侧锁
        lock drops; future prefix-matches stop before this tombstoned leaf.  # 释放前保持活跃；未来的前缀匹配在此墓碑叶子前停止。

        Caller must ensure this is invoked at most once per (node, swa_uuid_for_lock)  # 调用者必须确保每个(node, swa_uuid_for_lock)对最多调用一次
        pair (track via e.g. `Req.swa_prefix_lock_released`). When the request  # （通过如`Req.swa_prefix_lock_released`跟踪）。当请求
        finally releases its full lock via `dec_lock_ref`, pass `skip_swa=True`  # 最终通过`dec_lock_ref`释放full锁时，传递`skip_swa=True`
        to avoid touching SWA state again.  # 以避免再次触碰SWA状态。
        """
        if self.disable:  # 如果缓存被禁用
            return  # 直接返回

        while node != self.root_node:  # 从当前节点遍历到根节点
            assert (
                not node.swa_tombstone
            ), f"dec_swa_lock_only on swa_tombstone node, {node.id=}"  # 断言非墓碑节点
            assert (
                node.swa_lock_ref > 0
            ), f"dec_swa_lock_only on node with {node.swa_lock_ref=}, {node.id=}"  # 断言SWA锁引用大于0

            if node.swa_lock_ref == 1:  # 如果这是最后一次SWA解锁
                self.swa_protected_size_ -= len(node.value)  # 减少SWA受保护大小
                if len(node.children) == 0:  # 如果是叶子节点
                    # Leaf: free SWA pool slots and tombstone, and remove from  # 叶子：释放SWA池槽位并标记墓碑，从
                    # swa_lru_list so SWA-eviction won't pick this tombstoned  # swa_lru_list中移除，使SWA淘汰不会选到这个墓碑
                    # leaf (which still holds full_lock_ref > 0). The full kv  # 叶子（仍持有full_lock_ref > 0）。full KV
                    # stays alive until the request releases its full lock.  # 在请求释放full锁前保持活跃。
                    self.token_to_kv_pool_allocator.free_swa(node.value)  # 释放SWA池槽位
                    self.swa_lru_list.remove_node(node)  # 从SWA LRU列表移除
                    node.swa_tombstone = True  # 标记为墓碑
                else:  # 内部节点
                    # Internal: standard protected -> evictable.  # 内部：标准的受保护→可淘汰转换。
                    self.swa_evictable_size_ += len(node.value)  # 增加SWA可淘汰大小
            node.swa_lock_ref -= 1  # 减少SWA锁引用

            if swa_uuid_for_lock and node.swa_uuid == swa_uuid_for_lock:  # 如果到达SWA UUID节点
                break  # 停止遍历
            node = node.parent  # 移动到父节点

    def sanity_check(self):  # 执行健全性检查
        self.full_lru_list.sanity_check(self)  # 检查full LRU列表
        self.swa_lru_list.sanity_check(self)  # 检查SWA LRU列表

    def evictable_size(self) -> Tuple[int, int]:  # 获取可淘汰大小（已废弃）
        # Note: use full_evictable_size() and swa_evictable_size() instead.  # 注意：请使用full_evictable_size()和swa_evictable_size()替代。
        raise NotImplementedError  # 抛出未实现异常

    def full_evictable_size(self) -> int:  # 获取full可淘汰大小
        return self.full_evictable_size_  # 返回full可淘汰大小

    def swa_evictable_size(self) -> int:  # 获取SWA可淘汰大小
        return self.swa_evictable_size_  # 返回SWA可淘汰大小

    def protected_size(self) -> Tuple[int, int]:  # 获取受保护大小（已废弃）
        # Note: use full_protected_size() and swa_protected_size() instead.  # 注意：请使用full_protected_size()和swa_protected_size()替代。
        raise NotImplementedError  # 抛出未实现异常

    def full_protected_size(self) -> int:  # 获取full受保护大小
        # protected size refers to the size of the full cache that is locked  # 受保护大小是指被锁定的full缓存大小
        return self.full_protected_size_  # 返回full受保护大小

    def swa_protected_size(self) -> int:  # 获取SWA受保护大小
        # protected size refers to the size of the swa cache that is locked  # 受保护大小是指被锁定的SWA缓存大小
        return self.swa_protected_size_  # 返回SWA受保护大小

    def all_values_flatten(self) -> torch.Tensor:  # 将所有节点的值展平为单个张量
        values = []  # 值列表

        def _dfs_helper(node: TreeNode):  # 深度优先遍历辅助函数
            for _, child in node.children.items():  # 遍历所有子节点
                values.append(child.value)  # 添加子节点的值
                _dfs_helper(child)  # 递归遍历子节点

        _dfs_helper(self.root_node)  # 从根节点开始遍历
        return torch.cat(values)  # 拼接所有值并返回

    def available_and_evictable_str(self) -> str:  # 获取可用和可淘汰大小的字符串表示
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()  # full可用大小
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()  # SWA可用大小
        full_evictable_size = self.full_evictable_size()  # full可淘汰大小
        swa_evictable_size = self.swa_evictable_size()  # SWA可淘汰大小
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"  # full可用+可淘汰
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"  # SWA可用+可淘汰
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"  # full LRU可淘汰大小
            f"SWA LRU list evictable size: {self.swa_lru_list.sanity_check_evictable_size()}\n"  # SWA LRU可淘汰大小
        )

    ##### Internal Helper Functions #####  ##### 内部辅助函数 #####

    def _match_prefix_helper(  # SWA前缀匹配辅助函数
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        SWA prefix matching helper. It factors in the sliding window size such that  # SWA前缀匹配辅助函数。考虑滑动窗口大小，使得
        the matched node is guaranteed to either 1. connected to root without swa tombstone,  # 匹配的节点保证要么1. 连接到根节点且无SWA墓碑，
        or 2. the number of matching tokens from the matched node to the last swa tombstone  # 要么2. 从匹配节点到最后一个SWA墓碑节点的匹配token数
        node is greater than or equal to the sliding window size.  # 大于或等于滑动窗口大小。
        """
        node = self.root_node  # 从根节点开始
        child_key = key.child_key(self.page_size)  # 获取子键

        value = []  # 匹配的值列表
        # for path connected to root without tombstone, always match, so set to inf  # 对于连接到根节点且无墓碑的路径，始终匹配，设为无穷大
        match_len_since_tombstone = float("inf")  # 自上次墓碑以来的匹配长度
        best_value_len = 0  # 最佳匹配值长度
        best_last_node = node  # 最佳匹配的最后节点
        enable_compact = envs.SGLANG_OPT_SWA_RADIX_CACHE_COMPACT.get()  # 获取紧凑化环境变量
        while len(key) > 0 and child_key in node.children.keys():  # 循环匹配
            child = node.children[child_key]  # 获取子节点

            if enable_compact:  # 如果启用紧凑化
                self._compact_single_child_chain(child)  # 紧凑化单子节点链

            if child.swa_tombstone:  # 如果子节点是墓碑节点
                # update best_value_len and best_last_node if needed  # 如需要则更新最佳匹配值长度和最后节点
                if match_len_since_tombstone >= self.sliding_window_size:  # 如果自上次墓碑以来的匹配长度达到窗口大小
                    best_value_len = len(value)  # 更新最佳值长度
                    best_last_node = node  # 更新最佳最后节点
                # reset match_len_since_tombstone if we hit a tombstone node  # 如果遇到墓碑节点则重置匹配长度
                match_len_since_tombstone = 0  # 重置匹配长度

            prefix_len = child.key.match(key, page_size=self.page_size)  # 计算前缀匹配长度
            if prefix_len < len(child.key):  # 如果部分匹配（需要分裂）
                new_node = self._split_node(child.key, child, prefix_len)  # 分裂子节点
                value.append(new_node.value)  # 添加新节点的值
                if not new_node.swa_tombstone:  # 如果非墓碑节点
                    match_len_since_tombstone += len(new_node.value)  # 累加匹配长度
                node = new_node  # 移动到新节点
                break  # 跳出循环
            else:  # 完全匹配
                value.append(child.value)  # 添加子节点的值
                if not child.swa_tombstone:  # 如果非墓碑节点
                    match_len_since_tombstone += len(child.value)  # 累加匹配长度
                node = child  # 移动到子节点
                key = key[prefix_len:]  # 截取剩余键

                if len(key):  # 如果还有剩余键
                    child_key = key.child_key(self.page_size)  # 获取下一个子键

        # handle best_value_len and best_last_node, for the case that last node is fully matched  # 处理best_value_len和best_last_node，针对最后一个节点完全匹配的情况
        if match_len_since_tombstone >= self.sliding_window_size:  # 如果匹配长度达到窗口大小
            best_value_len = len(value)  # 更新最佳值长度
            best_last_node = node  # 更新最佳最后节点

        return value, best_last_node, best_value_len  # 返回值列表、最后节点和最佳值长度

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:  # 匹配前预处理
        """Preprocess the key before matching."""  # 匹配前预处理键
        key = params.key  # 获取键
        key, _ = key.maybe_to_bigram_view(self.is_eagle)  # 处理bigram视图
        if self.disable or len(key) == 0:  # 如果禁用或键为空
            return None  # 返回None
        key = key.page_aligned(self.page_size)  # 页对齐键
        if len(key) == 0:  # 如果对齐后键为空
            return None  # 返回None
        return key  # 返回预处理后的键

    def _match_post_processor(  # 匹配后处理
        self,
        params: MatchPrefixParams,  # 匹配参数
        value: List[torch.Tensor],  # 匹配的值列表
        last_node: TreeNode,  # 最后匹配节点
        best_value_len: int,  # 最佳值长度
    ) -> MatchResult:
        """Post-process the matched result."""  # 后处理匹配结果
        node_update = last_node  # 需要更新的节点
        # update time for matched nodes, and make nodes closer to root to be least recently used  # 更新匹配节点的访问时间，使接近根的节点最久未使用
        # this allows swa to evict nodes closer to root first  # 这允许SWA优先淘汰接近根的节点
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)  # 更新full LRU
        self.swa_lru_list.reset_node_and_parents_mru(node_update, self.root_node)  # 更新SWA LRU

        # This last_access_time is for sanity check, can be deleted after validation in production  # last_access_time用于健全性检查，生产验证后可删除
        cur_time = get_last_access_time()  # 获取当前时间戳
        while node_update:  # 遍历从当前节点到根的路径
            node_update.last_access_time = cur_time  # 更新访问时间
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree  # 假设树的分支中不超过100000个节点
            )
            node_update = node_update.parent  # 移动到父节点

        value = value[:best_value_len]  # 截取到最佳匹配长度
        if value:  # 如果有匹配值
            value = torch.cat(value)  # 拼接为张量
        else:  # 无匹配值
            value = torch.empty((0,), dtype=torch.int64, device=self.device)  # 创建空张量

        return MatchResult(  # 返回匹配结果
            device_indices=value,  # 设备索引
            last_device_node=last_node,  # 最后设备节点
            last_host_node=last_node,  # 最后主机节点
            best_match_node=last_node,  # 最佳匹配节点
        )

    def _compact_single_child_chain(self, node: TreeNode) -> None:  # 紧凑化单子节点链（合并连续单子节点）
        # FIXME(ispobock): drifts retract pool accounting (commit 6348cb506);  # FIXME(ispobock)：会导致retract池核算偏移（提交6348cb506）；
        # also overwrites active swa_uuid when window > page_size. Off by  # 也会在窗口 > page_size时覆盖活跃的swa_uuid。默认关闭，
        # default via SGLANG_OPT_SWA_RADIX_CACHE_COMPACT.  # 通过SGLANG_OPT_SWA_RADIX_CACHE_COMPACT控制。
        while len(node.children) == 1:  # 当节点只有一个子节点时
            child = next(iter(node.children.values()))  # 获取唯一子节点
            if len(child.children) == 0:  # 如果子节点是叶子
                break  # 不合并叶子
            sum_gc_full_lock_ref = sum(  # 计算孙子节点的full锁引用之和
                gc.full_lock_ref for gc in child.children.values()
            )
            if child.full_lock_ref > sum_gc_full_lock_ref:  # 如果子节点的锁引用大于孙子节点之和
                break  # 不合并（有外部引用）

            if (  # 检查节点和子节点的锁状态是否一致
                child.swa_tombstone != node.swa_tombstone  # 墓碑状态不同
                or child.full_lock_ref != node.full_lock_ref  # full锁引用不同
                or child.swa_lock_ref != node.swa_lock_ref  # SWA锁引用不同
            ):
                break  # 不合并

            # Preserve is_bigram: main #23106 made bigram an O(1) flag on RadixKey;  # 保留is_bigram：主分支#23106将bigram设为RadixKey的O(1)标志；
            # the constructor defaults to False, so concat without explicit flag  # 构造函数默认为False，因此不带显式标志的拼接
            # silently demotes EAGLE/MTP bigram keys → match() returns 0 →  # 会静默降级EAGLE/MTP bigram键 → match()返回0 →
            # _split_node assert.  # _split_node断言失败。
            node.key = RadixKey(  # 合并键
                node.key.token_ids + child.key.token_ids,  # 拼接token ID
                node.key.extra_key,  # 保留extra_key
                is_bigram=node.key.is_bigram,  # 保留bigram标志
            )
            node.value = torch.cat([node.value, child.value])  # 合并值
            node.children = child.children  # 继承子节点的子节点
            for grandchild in node.children.values():  # 更新孙子节点的父节点
                grandchild.parent = node  # 设置父节点为当前节点

            if child.swa_uuid is not None:  # 如果子节点有SWA UUID
                node.swa_uuid = child.swa_uuid  # 继承SWA UUID

            if node.hash_value is not None and child.hash_value is not None:  # 如果两者都有哈希值
                node.hash_value = list(node.hash_value) + list(child.hash_value)  # 合并哈希值
            else:  # 任一无哈希值
                node.hash_value = None  # 设为None

            self.full_lru_list.remove_node(child)  # 从full LRU列表移除子节点
            if not child.swa_tombstone:  # 如果子节点非墓碑
                self.swa_lru_list.remove_node(child)  # 从SWA LRU列表移除子节点

    def _maybe_split_leaf_for_swa_lock(self, leaf: TreeNode) -> TreeNode:  # 可能分裂叶子节点以优化SWA锁定
        """``inc_lock_ref`` protects ``len(leaf.value)`` SWA tokens for the  # `inc_lock_ref`为叶子保护`len(leaf.value)`个SWA token，尽管
        leaf even though SWA only actually needs the last  # SWA实际只需要最后
        ``sliding_window_size`` tokens. With chunked prefill, leaves can be  # `sliding_window_size`个token。在分块预填充中，叶子可能有
        thousands of tokens long, which inflates ``swa_protected_size_`` by  # 数千个token长，这会使`swa_protected_size_`膨胀约
        ~``chunked_prefill_size / sliding_window_size`` and causes premature  # `chunked_prefill_size / sliding_window_size`倍，导致SWA池
        SWA pool exhaustion / retract thrashing.  # 过早耗尽/retract抖动。
        """
        if (
            leaf is self.root_node  # 是根节点
            or leaf.swa_lock_ref > 0  # 已有SWA锁
            or leaf.swa_tombstone  # 是墓碑节点
            or len(leaf.value) == 0  # 值为空
        ):
            return leaf  # 不需要分裂

        # Smallest page-aligned size that still covers the sliding window.  # 仍能覆盖滑动窗口的最小页对齐大小。
        tail_size = (  # 计算尾部长度
            (self.sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size  # 向上取整到页大小
        )
        if len(leaf.value) <= tail_size:  # 如果叶子大小不超过尾部长度
            return leaf  # 不需要分裂

        split_at = len(leaf.value) - tail_size  # 计算分裂位置

        if split_at <= 0 or split_at >= len(leaf.value):  # 分裂位置不合法
            return leaf  # 不需要分裂
        if self.page_size > 1 and (  # 分页模式下检查对齐
            split_at % self.page_size != 0 or len(leaf.value) % self.page_size != 0  # 不页对齐
        ):
            return leaf  # 不需要分裂

        self._split_node(leaf.key, leaf, split_at)  # 执行分裂
        return leaf  # 返回叶子节点

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:  # 分裂节点
        # new_node -> child  # 新节点 -> 原子节点
        new_node = TreeNode()  # 创建新节点
        new_node.children = {key[split_len:].child_key(self.page_size): child}  # 新节点的子节点为原节点
        new_node.parent = child.parent  # 新节点的父节点为原节点的父节点
        new_node.swa_tombstone = child.swa_tombstone  # 继承墓碑状态
        new_node.full_lock_ref = child.full_lock_ref  # 继承full锁引用
        new_node.swa_lock_ref = child.swa_lock_ref  # 继承SWA锁引用
        new_node.key = child.key[:split_len]  # 新节点的键为前半部分
        assert len(new_node.key) > 0, f"new_node.key should not be empty"  # 断言新键非空
        new_node.value = child.value[:split_len].clone()  # 新节点的值为前半部分的副本
        # parent inherits the swa_uuid from child for swa lock ref  # 父节点从子节点继承swa_uuid用于SWA锁引用
        new_node.swa_uuid = child.swa_uuid  # 继承SWA UUID
        child.swa_uuid = None  # 子节点的SWA UUID置空
        # child time should be later than parent's time for swa tombstone  # 子节点时间应晚于父节点时间（用于SWA墓碑）
        child.last_access_time = get_last_access_time()  # 更新子节点访问时间

        # remove the child from the lru lists because it is being split  # 从LRU列表中移除子节点（因为正在分裂）
        self.full_lru_list.remove_node(child)  # 从full LRU列表移除
        if not new_node.swa_tombstone:  # 如果非墓碑节点
            self.swa_lru_list.remove_node(child)  # 从SWA LRU列表移除
        child.parent = new_node  # 子节点的父节点设为新节点
        child.key = child.key[split_len:]  # 子节点的键为后半部分
        assert len(child.key) > 0, f"child.key should not be empty"  # 断言子键非空
        child.value = child.value[split_len:].clone()  # 子节点的值为后半部分的副本
        new_node.parent.children[key.child_key(self.page_size)] = new_node  # 替换父节点的子节点引用
        new_node.hash_value, child.hash_value = split_node_hash_value(  # 分裂哈希值
            child.hash_value, split_len, self.page_size
        )

        # insert the new node and child into the lru lists, insert  # 将新节点和子节点插入LRU列表，先插入
        # parent first so that parent is after child in the lru list  # 父节点，使父节点在LRU列表中位于子节点之后
        self.full_lru_list.insert_mru(new_node)  # 插入新节点到full LRU
        self.full_lru_list.insert_mru(child)  # 插入子节点到full LRU
        if not new_node.swa_tombstone:  # 如果非墓碑节点
            self.swa_lru_list.insert_mru(new_node)  # 插入新节点到SWA LRU
            self.swa_lru_list.insert_mru(child)  # 插入子节点到SWA LRU
        return new_node  # 返回新节点

    def _insert_helper(  # 插入辅助函数
        self,
        node: TreeNode,  # 当前节点
        key: RadixKey,  # 要插入的键
        value,  # 要插入的值
        update_kv_after_len: int,  # 更新KV的起始长度
        swa_evicted_seqlen: int = 0,  # SWA已淘汰的序列长度
    ) -> int:
        # Update the last access time from root to leaf, so that  # 从根到叶更新最后访问时间，使
        # swa will tombstone the node closer to root first  # SWA优先淘汰接近根的节点
        node.last_access_time = get_last_access_time()  # 更新访问时间
        if node != self.root_node:  # 如果不是根节点
            self.full_lru_list.reset_node_mru(node)  # 重置full LRU位置
            if not node.swa_tombstone:  # 如果非墓碑节点
                self.swa_lru_list.reset_node_mru(node)  # 重置SWA LRU位置
        if len(key) == 0:  # 如果键为空
            return 0  # 返回0

        child_key = key.child_key(self.page_size)  # 获取子键

        total_prefix_length = 0  # 总前缀长度
        while len(key) > 0 and child_key in node.children.keys():  # 循环匹配已有节点
            node = node.children[child_key]  # 移动到子节点
            node.last_access_time = get_last_access_time()  # 更新访问时间
            self.full_lru_list.reset_node_mru(node)  # 重置full LRU位置
            if not node.swa_tombstone:  # 如果非墓碑节点
                self.swa_lru_list.reset_node_mru(node)  # 重置SWA LRU位置
            prefix_len = node.key.match(key, page_size=self.page_size)  # 计算前缀匹配长度

            if prefix_len < len(node.key):  # 如果部分匹配（需要分裂）
                new_node = self._split_node(node.key, node, prefix_len)  # 分裂节点
                node = new_node  # 移动到新节点

            # if tombstone after update_kv_after_len, update node.value to be the input value.  # 如果update_kv_after_len之后是墓碑，将node.value更新为输入值。
            # This is needed because it is possible that the last sliding window size tokens  # 这是必要的，因为最后滑动窗口大小的token可能
            # contains tombstone. If this is the case and we don't update the kv value, then  # 包含墓碑。如果是这种情况且不更新KV值，则
            # the prefill prefix matching will stuck.  # 预填充前缀匹配会卡住。
            if update_kv_after_len < total_prefix_length + prefix_len:  # 如果需要更新KV值
                # For page_size > 1 and chunked prefill case, update_kv_after_len may be not page-aligned due to a trailing partial page  # 对于page_size > 1和分块预填充情况，update_kv_after_len可能由于尾部不完整页而未页对齐
                # (kept in the request but not inserted into the radix tree) appended to prefix_indices.  # （保留在请求中但未插入基数树）附加到prefix_indices。
                if node.swa_tombstone:  # 如果是墓碑节点
                    assert (
                        node.swa_lock_ref == 0
                    ), f"tombstone swa_lock_ref should always be 0, {node.full_lock_ref=}, {node.swa_lock_ref=}, {node.id=}"  # 断言墓碑节点的SWA锁引用为0
                    assert (
                        swa_evicted_seqlen % self.page_size == 0
                    ), f"swa_evicted_seqlen must be page aligned, {swa_evicted_seqlen=}, {self.page_size=}"  # 断言SWA淘汰序列长度页对齐
                    if swa_evicted_seqlen <= total_prefix_length:  # 分支1：所有SWA token未淘汰
                        # Branch 1: all swa tokens of value[:prefix_len] are not evicted, so we can insert it to the tree directly.  # 分支1：value[:prefix_len]的所有SWA token未被淘汰，可直接插入。
                        # Free full tokens in the original tree node.  # 释放原始树节点中的full token。
                        self.token_to_kv_pool_allocator.free(node.value[:prefix_len])  # 释放full token
                        # Overwrite the new value in request to the tree node.  # 用请求中的新值覆盖树节点的值。
                        node.value = value[:prefix_len].clone()  # 更新节点值
                        node.swa_tombstone = False  # 清除墓碑标记
                        self.swa_lru_list.insert_mru(node)  # 插入SWA LRU列表
                        self.swa_evictable_size_ += len(node.value)  # 增加SWA可淘汰大小
                    elif swa_evicted_seqlen < total_prefix_length + prefix_len:  # 分支2：部分SWA token被淘汰
                        # Branch 2: part of swa tokens of value[:prefix_len] are evicted, so we need to split the node and insert the value to new node.  # 分支2：value[:prefix_len]的部分SWA token被淘汰，需分裂节点并插入新值。
                        start_update_idx = swa_evicted_seqlen - total_prefix_length  # 计算开始更新索引
                        self.token_to_kv_pool_allocator.free(  # 释放部分full token
                            node.value[start_update_idx:prefix_len]
                        )
                        self._split_node(node.key, node, start_update_idx)  # 分裂节点
                        # Here node is the new node after split, so we can overwrite the value to the new node.  # 此处node是分裂后的新节点，可以覆盖其值。
                        # The old node is still swa tombstone and the full token is not freed.  # 旧节点仍是SWA墓碑且full token未释放。
                        node.value = value[start_update_idx:prefix_len].clone()  # 更新节点值
                        self.token_to_kv_pool_allocator.free(value[:start_update_idx])  # 释放旧full token
                        node.swa_tombstone = False  # 清除墓碑标记
                        self.swa_lru_list.insert_mru(node)  # 插入SWA LRU列表
                        self.swa_evictable_size_ += len(node.value)  # 增加SWA可淘汰大小
                    else:  # 分支3：所有SWA token都被淘汰
                        # Branch 3: all swa tokens of value[:prefix_len] are evicted, so we don't need to update the node.  # 分支3：value[:prefix_len]的所有SWA token已被淘汰，无需更新节点。
                        self.token_to_kv_pool_allocator.free(value[:prefix_len])  # 释放新值
                else:  # 非墓碑节点
                    # The node is not tombstone, so we don't need to update the node.  # 节点非墓碑，无需更新节点。
                    self.token_to_kv_pool_allocator.free(value[:prefix_len])  # 释放重复的值

            total_prefix_length += prefix_len  # 累加前缀长度
            key = key[prefix_len:]  # 截取剩余键
            value = value[prefix_len:]  # 截取剩余值

            if len(key):  # 如果还有剩余键
                child_key = key.child_key(self.page_size)  # 获取下一个子键

        if len(key):  # 如果还有键未匹配（需要创建新节点）
            # Layout: |--- total_prefix_length ---|--- len(key) ---|  # 布局：|--- total_prefix_length ---|--- len(key) ---|
            #         ^                           ^                ^  #         ^                           ^                ^
            #         0              total_prefix_length     total_length  #         0              total_prefix_length     total_length
            #
            # Cases based on swa_evicted_seqlen position:  # 根据swa_evicted_seqlen位置的情况：
            # 1. swa_evicted_seqlen <= total_prefix_length:  # 1. swa_evicted_seqlen <= total_prefix_length：
            #    Already handled in the while loop above. All of len(key) is non-tombstone.  #    已在上面的while循环中处理。len(key)全部为非墓碑。
            # 2. total_prefix_length < swa_evicted_seqlen < total_length:  # 2. total_prefix_length < swa_evicted_seqlen < total_length：
            #    Split: [total_prefix_length, swa_evicted_seqlen) as tombstone,  #    分裂：[total_prefix_length, swa_evicted_seqlen)作为墓碑，
            #           [swa_evicted_seqlen, total_length) as non-tombstone.  #           [swa_evicted_seqlen, total_length)作为非墓碑。
            # 3. swa_evicted_seqlen == total_length:  # 3. swa_evicted_seqlen == total_length：
            #    All remaining tokens are evicted. Free value and return without  #    所有剩余token都被淘汰。释放值并返回而不
            #    creating a node (leaf nodes must not be tombstone).  #    创建节点（叶子节点不能是墓碑）。
            #    Note: the -page_size fix in _evict_swa prevents this case from  #    注意：_evict_swa中的-page_size修复阻止此情况在
            #    occurring in normal operation. This check is a defensive guard  #    正常操作中发生。此检查是针对其他代码路径
            #    against unexpected eviction states from other code paths.  #    意外淘汰状态的防御性保护。
            if swa_evicted_seqlen == total_prefix_length + len(key):  # 情况3：所有剩余token被淘汰
                self.token_to_kv_pool_allocator.free(value)  # 释放所有值
                return total_prefix_length  # 返回前缀长度

            if (  # 情况2：部分token被淘汰
                swa_evicted_seqlen > total_prefix_length
                and swa_evicted_seqlen < total_prefix_length + len(key)
            ):
                swa_tombstone_len = swa_evicted_seqlen - total_prefix_length  # 计算墓碑长度
                node = self._add_new_node(  # 创建墓碑节点
                    node,
                    key[:swa_tombstone_len],  # 墓碑部分的键
                    value[:swa_tombstone_len],  # 墓碑部分的值
                    swa_tombstone=True,  # 标记为墓碑
                )
                key = key[swa_tombstone_len:]  # 截取非墓碑部分的键
                value = value[swa_tombstone_len:]  # 截取非墓碑部分的值

            new_leaf = self._add_new_node(node, key, value, swa_tombstone=False)  # 创建新的非墓碑叶子节点

            if envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.get():  # 如果启用插入时分裂叶子优化
                # Cap the leaf at one (page-aligned) sliding window so a future  # 将叶子限制为一个（页对齐的）滑动窗口，使未来的
                # inc_lock_ref only protects `sliding_window_size` tokens of SWA pool.  # inc_lock_ref仅保护`sliding_window_size`个SWA池token。
                self._maybe_split_leaf_for_swa_lock(new_leaf)  # 可能分裂叶子

        return total_prefix_length  # 返回总前缀长度

    def _add_new_node(  # 向树中添加新节点
        self,
        parent: TreeNode,  # 父节点
        key: RadixKey,  # 键
        value: torch.Tensor,  # 值
        swa_tombstone: bool = False,  # 是否为SWA墓碑节点
    ) -> TreeNode:
        assert len(key) > 0, f"key should not be empty"  # 断言键非空
        new_node = TreeNode()  # 创建新节点
        new_node.parent = parent  # 设置父节点
        new_node.key = key  # 设置键
        new_node.value = value.clone()  # 设置值（克隆）
        new_node.swa_tombstone = swa_tombstone  # 设置墓碑标志
        parent.children[key.child_key(self.page_size)] = new_node  # 添加到父节点的子节点字典
        self.full_lru_list.insert_mru(new_node)  # 插入full LRU列表
        self.full_evictable_size_ += len(value)  # 增加full可淘汰大小
        if not swa_tombstone:  # 如果非墓碑节点
            self.swa_lru_list.insert_mru(new_node)  # 插入SWA LRU列表
            self.swa_evictable_size_ += len(value)  # 增加SWA可淘汰大小
        self._record_store_event(new_node)  # 记录存储事件
        return new_node  # 返回新节点

    def _iteratively_delete_tombstone_leaf(  # 迭代删除墓碑叶子节点
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0  # 淘汰的full token数
        while node.parent.swa_tombstone and len(node.parent.children) == 0:  # 父节点是墓碑且无子节点
            # root node is not evictable  # 根节点不可淘汰
            if node.parent == self.root_node:  # 如果父节点是根节点
                break  # 停止
            # if locked, means node is in use, skip  # 如果被锁定，说明节点正在使用，跳过
            if node.parent.full_lock_ref > 0:  # 如果父节点full锁引用大于0
                break  # 停止
            assert (
                node.parent.swa_lock_ref == 0
            ), f"tombstone swa_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.swa_lock_ref=}, {node.parent.id=}"  # 断言墓碑父节点SWA锁为0
            # delete tombstone node evicts full tokens  # 删除墓碑节点会淘汰full token
            self._record_remove_event(node.parent)  # 记录移除事件
            self.token_to_kv_pool_allocator.free(node.parent.value)  # 释放full KV
            full_num_evicted += len(node.parent.value)  # 累加淘汰数
            self.full_lru_list.remove_node(node.parent)  # 从full LRU列表移除
            self._delete_tombstone_leaf(node.parent)  # 删除墓碑叶子
            node = node.parent  # 移动到父节点

        return node, full_num_evicted  # 返回最终节点和淘汰数

    def _delete_leaf(self, node: TreeNode) -> None:  # 删除叶子节点
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"  # 断言是叶子节点
        key = node.key.child_key(self.page_size)  # 获取子键
        v = node.parent.children.pop(key, None)  # 从父节点中移除
        assert v == node, f"parent does not have child key, {key}"  # 断言移除的节点正确
        self.full_evictable_size_ -= len(node.key)  # 减少full可淘汰大小
        # Tombstoned leaves were never (re-)added to swa_lru_list and were  # 墓碑叶子从未（重新）添加到swa_lru_list，并且
        # already removed from swa_evictable_size_ when they were tombstoned.  # 在被标记为墓碑时已从swa_evictable_size_中移除。
        if not node.swa_tombstone:  # 如果非墓碑节点
            self.swa_evictable_size_ -= len(node.key)  # 减少SWA可淘汰大小

    def _tombstone_internal_node(self, node: TreeNode) -> None:  # 将内部节点标记为墓碑
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"  # 断言不是叶子节点
        node.swa_tombstone = True  # 设置墓碑标志
        self.swa_evictable_size_ -= len(node.key)  # 减少SWA可淘汰大小

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:  # 删除墓碑叶子节点
        assert (
            node.swa_tombstone
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"  # 断言是墓碑节点
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"  # 断言是叶子节点
        key = node.key.child_key(self.page_size)  # 获取子键
        v = node.parent.children.pop(key, None)  # 从父节点中移除
        assert v == node, f"parent does not have child key, {key}"  # 断言移除的节点正确

        self.full_evictable_size_ -= len(node.key)  # 减少full可淘汰大小

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:  # 收集所有非墓碑节点
        ret_list = []  # 结果列表
        stack = [self.root_node]  # 栈（从根节点开始）

        while stack:  # 深度优先遍历
            cur_node = stack.pop()  # 弹出当前节点
            if not cur_node.swa_tombstone:  # 如果非墓碑
                ret_list.append(cur_node)  # 添加到结果列表
            stack.extend(cur_node.children.values())  # 将子节点压入栈

        return ret_list  # 返回非墓碑节点列表

    def _collect_all_nodes(self) -> List[TreeNode]:  # 收集所有节点（包含墓碑）
        ret_list = []  # 结果列表
        stack = [self.root_node]  # 栈（从根节点开始）
        while stack:  # 深度优先遍历
            cur_node = stack.pop()  # 弹出当前节点
            ret_list.append(cur_node)  # 添加到结果列表
            stack.extend(cur_node.children.values())  # 将子节点压入栈
        return ret_list  # 返回所有节点列表

    def _print_helper(self, node: TreeNode, indent: int) -> None:  # 打印树结构的辅助函数
        """Prints the radix tree in a human-readable format."""  # 以可读格式打印基数树
        stack = [(node, indent)]  # 栈（节点和缩进）
        while stack:  # 遍历栈
            current_node, current_indent = stack.pop()  # 弹出当前节点和缩进
            print(  # 打印节点信息
                " " * current_indent,  # 缩进空格
                current_node.id,  # 节点ID
                len(current_node.key),  # 键长度
                f"fr={current_node.full_lock_ref}",  # full锁引用
                f"sr={current_node.swa_lock_ref}",  # SWA锁引用
                f"fll={self.full_lru_list.in_list(current_node)}",  # 是否在full LRU列表中
                f"sll={self.swa_lru_list.in_list(current_node)}",  # 是否在SWA LRU列表中
                f"ts={current_node.swa_tombstone}",  # 墓碑标志
            )
            for key, child in current_node.children.items():  # 遍历子节点
                stack.append((child, current_indent + 2))  # 压入栈（增加缩进）

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"  # 断言键一致性

    def _total_size_helper(self) -> Tuple[int, int]:  # 计算树中full和SWA的总token数
        total_size = 0  # full总大小
        total_swa_size = 0  # SWA总大小
        stack = [self.root_node]  # 栈（从根节点开始）
        while stack:  # 深度优先遍历
            current_node = stack.pop()  # 弹出当前节点
            total_size += len(current_node.value)  # 累加full大小
            if not current_node.swa_tombstone:  # 如果非墓碑节点
                total_swa_size += len(current_node.value)  # 累加SWA大小
            for child in current_node.children.values():  # 遍历子节点
                if child.evicted:  # 如果子节点已被淘汰
                    continue  # 跳过
                stack.append(child)  # 压入栈
        return total_size, total_swa_size  # 返回full和SWA总大小
