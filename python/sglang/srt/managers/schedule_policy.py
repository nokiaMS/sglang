from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from array import array  # 导入数组类型

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor  # 导入预填充延迟器
from sglang.srt.utils import get_bool_env_var  # 导入布尔环境变量获取工具

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")  # 路由键策略调试日志开关
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

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
"""Request scheduler policy"""
# 请求调度策略模块，定义了请求的调度策略和预填充添加器

import os  # 导入操作系统接口
import random  # 导入随机数模块
from collections import Counter, defaultdict  # 导入计数器和默认字典
from contextlib import contextmanager  # 导入上下文管理器装饰器
from enum import Enum, auto  # 导入枚举类型和自动赋值
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.dllm.config import DllmConfig  # 导入扩散语言模型配置
from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split  # 导入DSA预填充上下文并行判断工具
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled  # 导入预填充上下文并行开关判断
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # 导入请求和调度批次类
from sglang.srt.mem_cache.base_prefix_cache import (  # 导入前缀缓存基类及相关参数
    BasePrefixCache,  # 前缀缓存基类
    InitLoadBackParams,  # 初始化回加载参数
    InsertParams,  # 插入参数
    MatchPrefixParams,  # 前缀匹配参数
    zero_match_result,  # 零匹配结果生成函数
)
from sglang.srt.mem_cache.hisparse_memory_pool import (  # 导入HiSparse内存池
    DeepSeekV4HiSparseTokenToKVPoolAllocator,  # DeepSeek V4 HiSparse KV池分配器
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode  # 导入基数缓存、基数键和树节点
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator  # 导入滑动窗口注意力KV池分配器
from sglang.srt.server_args import ServerArgs  # 导入服务器参数

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator  # 导入KV池分配器基类

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
# 裁剪max_new_tokens的估计值，防止调度器过于保守，但不改变实际停止条件
CLIP_MAX_NEW_TOKENS = int(  # 裁剪后的最大新token数
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")  # 从环境变量读取，默认4096
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
# 批次内前缀缓存检查阈值；匹配前缀长度小于此值时触发检查，设为-1则禁用
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(  # 批次内前缀缓存检查阈值
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")  # 从环境变量读取，默认32
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
# 批次内前缀缓存降优先级阈值；等待队列中匹配前缀长度超过此值时降低该请求优先级
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(  # 批次内前缀缓存降优先级阈值
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")  # 从环境变量读取，默认32
)


IGNORE_EOS_RESERVE_TOKENS = 1  # 忽略EOS时预留的token数


def match_prefix_for_req(  # 为请求匹配前缀
    tree_cache: BasePrefixCache,  # 树形前缀缓存
    req: Req,  # 请求对象
    token_ids: Optional[array[int]] = None,  # token ID列表，可选
    *,  # 以下为仅关键字参数
    cow_mamba: bool = False,  # 是否使用Mamba的写时复制
    include_req: bool = False,  # 是否在匹配时包含请求本身
):
    """为请求在前缀缓存中匹配前缀，并更新请求的缓存匹配信息"""
    if token_ids is None:  # 如果未提供token_ids
        token_ids = req.origin_input_ids + req.output_ids  # 则使用原始输入ID加上输出ID

    match_result = tree_cache.match_prefix(  # 在树缓存中进行前缀匹配
        MatchPrefixParams(  # 构造匹配参数
            key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),  # 使用token_ids和extra_key构造基数键
            cow_mamba=cow_mamba,  # 传入Mamba写时复制标志
            req=req if include_req else None,  # 根据include_req决定是否传入请求
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():  # 如果强制基数缓存未命中
        match_result = zero_match_result(tree_cache, match_result)  # 则将匹配结果置零
    (
        req.prefix_indices,  # 请求的前缀索引
        req.last_node,  # 请求的最后一个树节点
        req.last_host_node,  # 请求的最后一个主机节点
        req.best_match_node,  # 请求的最佳匹配节点
        req.host_hit_length,  # 请求的主机命中长度
    ) = (
        match_result.device_indices,  # 匹配结果的设备索引
        match_result.last_device_node,  # 匹配结果的最后一个设备节点
        match_result.last_host_node,  # 匹配结果的最后一个主机节点
        match_result.best_match_node,  # 匹配结果的最佳匹配节点
        match_result.host_hit_length,  # 匹配结果的主机命中长度
    )
    if match_result.mamba_branching_seqlen is not None:  # 如果匹配结果包含Mamba分支序列长度
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen  # 更新请求的Mamba分支序列长度
    if match_result.cache_protected_len is not None:  # 如果匹配结果包含缓存保护长度
        req.cache_protected_len = match_result.cache_protected_len  # 更新请求的缓存保护长度
    return match_result  # 返回匹配结果


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""
    """感知树缓存的调度策略"""

    LPM = "lpm"  # 最长前缀匹配
    DFS_WEIGHT = "dfs-weight"  # 深度优先搜索加权


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""
    """不感知树缓存的调度策略"""

    FCFS = "fcfs"  # 先来先服务
    LOF = "lof"  # 最长输出优先
    RANDOM = "random"  # 随机调度
    ROUTING_KEY = "routing-key"  # 按运行批次中路由键频率优先


class SchedulePolicy:
    """请求调度策略类，根据配置的策略对等待队列中的请求进行排序"""
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]  # 策略类型为缓存感知或缓存不感知

    def __init__(  # 初始化方法
        self,
        policy: str,  # 策略名称字符串
        tree_cache: BasePrefixCache,  # 树形前缀缓存
        enable_hierarchical_cache: bool,  # 是否启用分层缓存
        enable_priority_scheduling: bool,  # 是否启用优先级调度
        schedule_low_priority_values_first: bool,  # 是否先调度低优先级值
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)  # 验证并调整策略
        self.tree_cache = tree_cache  # 保存树缓存引用
        self.enable_hierarchical_cache = enable_hierarchical_cache  # 保存分层缓存开关
        self.enable_priority_scheduling = enable_priority_scheduling  # 保存优先级调度开关
        self.schedule_low_priority_values_first = schedule_low_priority_values_first  # 保存低优先级优先开关
        self.priority_sign = 1 if schedule_low_priority_values_first else -1  # 优先级符号：先调度低优先级为1，否则为-1

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache.create_simulated()  # 创建模拟的基数缓存用于批次内前缀匹配

    def calc_priority(  # 计算等待队列中请求的优先级排序
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> None:
        """根据策略计算并排序等待队列中请求的优先级"""
        if self.policy == CacheAgnosticPolicy.FCFS:  # 如果是先来先服务策略
            if self.enable_priority_scheduling:  # 如果启用了优先级调度
                SchedulePolicy._sort_by_priority_and_fcfs(  # 按优先级和先来先服务排序
                    waiting_queue, self.priority_sign
                )
            return  # 直接返回

        policy = self._determine_active_policy(waiting_queue)  # 确定当前活跃的策略

        if isinstance(policy, CacheAwarePolicy):  # 如果是缓存感知策略
            temporary_deprioritized = self._compute_prefix_matches(  # 计算前缀匹配，获取临时降优先级集合
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:  # 如果是最长前缀匹配策略
                SchedulePolicy._sort_by_longest_prefix(  # 按最长前缀排序
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:  # 如果是DFS加权策略
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)  # 按DFS权重排序
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")  # 未知的缓存感知策略
        else:
            if policy == CacheAgnosticPolicy.FCFS:  # 如果是先来先服务
                pass  # 无需额外操作
            elif policy == CacheAgnosticPolicy.LOF:  # 如果是最长输出优先
                SchedulePolicy._sort_by_longest_output(  # 按最长输出排序
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:  # 如果是随机策略
                SchedulePolicy._sort_randomly(waiting_queue)  # 随机打乱
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:  # 如果是路由键策略
                if running_batch is not None:  # 如果有运行中的批次
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)  # 按路由键频率排序
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")  # 未知的缓存不感知策略

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        """确定当前活跃的调度策略，当等待队列过长时自动降级为FCFS"""
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS  # 队列过大时关闭昂贵的前缀匹配，降级为先来先服务
        return self.policy  # 返回原始策略

    def _validate_and_adjust_policy(  # 验证并调整策略
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        验证策略名称，如果树缓存被禁用则将缓存感知策略降级为FCFS。
        """
        try:
            policy_enum = CacheAwarePolicy(policy)  # 尝试解析为缓存感知策略
            if getattr(tree_cache, "disable", True):  # 如果树缓存被禁用
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS  # 降级为先来先服务
            return policy_enum  # 返回缓存感知策略枚举
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)  # 尝试解析为缓存不感知策略
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")  # 未知的策略名称

    def _compute_prefix_matches(  # 计算前缀匹配
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        计算并缓存等待队列中请求的前缀匹配结果，处理批次内前缀缓存逻辑，
        返回需要临时降低优先级的请求ID集合。
        """
        temporary_deprioritized: Set[int] = set()  # 临时降优先级的请求ID集合
        self.waiting_queue_radix_tree.reset()  # 重置等待队列基数树

        for r in waiting_queue:  # 遍历等待队列中的每个请求
            prefix_ids = r.origin_input_ids + r.output_ids  # 拼接原始输入ID和输出ID
            extra_key = r.extra_key  # 获取额外键
            match_result = match_prefix_for_req(self.tree_cache, r, prefix_ids)  # 对请求进行前缀匹配

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            # 注释：此逻辑用于批次内前缀缓存；当多个请求的缓存匹配前缀较短但共享相同前缀时，
            # 优先调度其中一个以提高缓存命中率。
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:  # 如果前缀匹配长度低于检查阈值
                match_result = self.waiting_queue_radix_tree.match_prefix(  # 在等待队列基数树中匹配前缀
                    MatchPrefixParams(
                        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)  # 构造匹配键
                    )
                )
                if envs.SGLANG_RADIX_FORCE_MISS.get():  # 如果强制基数缓存未命中
                    match_result = zero_match_result(  # 将匹配结果置零
                        self.waiting_queue_radix_tree, match_result
                    )
                in_batch_matching_prefixes = match_result.device_indices  # 获取批次内匹配前缀索引
                if (
                    len(in_batch_matching_prefixes)  # 如果批次内匹配前缀长度
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD  # 大于等于降优先级阈值
                ):
                    temporary_deprioritized.add(r.rid)  # 将该请求加入临时降优先级集合
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(  # 否则将请求插入等待队列基数树
                        InsertParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),  # 使用token_ids和extra_key构造键
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),  # 使用空布尔张量作为值（占位）
                        )
                    )
        return temporary_deprioritized  # 返回临时降优先级集合

    @staticmethod
    def _sort_by_longest_prefix(  # 按最长前缀匹配排序
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        """按最长前缀匹配长度对等待队列进行降序排序，被降优先级的请求排到最后"""
        waiting_queue.sort(  # 对等待队列排序
            key=lambda r: (
                -len(r.prefix_indices)  # 前缀长度取负，实现降序排列
                if r.rid not in temporary_deprioritized  # 如果请求未被降优先级
                else float("inf")  # 否则排到最后（正无穷大取负后仍最小，但此处取负长度的相反方向）
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(  # 按DFS权重排序
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        """基于深度优先搜索权重对等待队列进行排序"""
        last_node_to_reqs = defaultdict(list)  # 建立树节点到请求列表的映射
        for req in waiting_queue:  # 遍历等待队列
            last_node_to_reqs[req.last_node].append(req)  # 将请求按最后节点分组

        node_to_weight = defaultdict(int)  # 建立节点到权重的映射
        for node in last_node_to_reqs:  # 遍历每个有请求的节点
            node_to_weight[node] = len(last_node_to_reqs[node])  # 权重初始为该节点对应的请求数量
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)  # 递归计算所有节点的累积权重

        waiting_queue.clear()  # 清空等待队列
        SchedulePolicy._get_dfs_priority(  # 按DFS优先级重建等待队列
            tree_cache.root_node,  # 从根节点开始
            node_to_weight,  # 节点权重映射
            last_node_to_reqs,  # 节点到请求列表的映射
            waiting_queue,  # 输出队列
        )

    @staticmethod
    def _sort_by_longest_output(  # 按最长输出排序
        waiting_queue: List[Req],
        enable_priority_scheduling: bool,  # 是否启用优先级调度
        priority_sign: int,  # 优先级符号
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first."""
        """按最长输出（max_new_tokens）对等待队列排序；启用优先级调度时先按优先级排序"""
        if enable_priority_scheduling:  # 如果启用了优先级调度
            waiting_queue.sort(  # 按优先级和最长输出排序
                key=lambda x: (
                    x.priority * priority_sign,  # 第一排序键：优先级乘以符号
                    -x.sampling_params.max_new_tokens,  # 第二排序键：最大新token数取负（降序）
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)  # 仅按最大新token数降序排列

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        """随机打乱等待队列的顺序"""
        random.shuffle(waiting_queue)  # 随机打乱

    @staticmethod
    def _sort_by_priority_and_fcfs(  # 按优先级和先来先服务排序
        waiting_queue: List[Req], priority_sign: int
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp."""
        """按请求优先级和到达时间戳对等待队列排序"""
        waiting_queue.sort(
            key=lambda x: (
                x.priority * priority_sign,  # 第一排序键：优先级乘以符号
                x.time_stats.wait_queue_entry_time,  # 第二排序键：等待队列进入时间（先来先服务）
            )
        )

    @staticmethod
    def _sort_by_routing_key(  # 按路由键排序
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        """按运行批次中路由键的出现频率对等待队列排序，频率高的优先"""
        routing_key_counts = Counter(  # 统计运行批次中各路由键的出现次数
            r.routing_key for r in running_batch.reqs if r.routing_key  # 仅统计有路由键的请求
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:  # 如果开启了路由键调试日志
            waiting_keys_before = [r.routing_key for r in waiting_queue]  # 记录排序前的等待队列路由键
            logger.info(  # 输出调试信息
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:  # 如果运行批次中没有路由键
            return  # 直接返回，不排序

        def sort_key(req: Req):  # 定义排序键函数
            key = req.routing_key  # 获取请求的路由键
            if key and key in routing_key_counts:  # 如果路由键存在且在运行批次中出现
                count = routing_key_counts[key]  # 获取出现次数
                return (0, -count, key)  # 有匹配的排在前面，按频率降序
            else:
                return (1, 0, key or "")  # 无匹配的排在后面

        waiting_queue.sort(key=sort_key)  # 按排序键排序等待队列

        if _ROUTING_KEY_POLICY_DEBUG_LOG:  # 如果开启了路由键调试日志
            waiting_keys_after = [r.routing_key for r in waiting_queue]  # 记录排序后的等待队列路由键
            logger.info(f"waiting_keys_after={waiting_keys_after}")  # 输出排序后的调试信息

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        """递归计算树节点的累积权重（将子节点权重累加到父节点）"""
        for child in cur_node.children.values():  # 遍历当前节点的所有子节点
            SchedulePolicy._calc_weight(child, node_to_weight)  # 递归计算子节点的权重
            node_to_weight[cur_node] += node_to_weight[child]  # 将子节点权重累加到当前节点

    @staticmethod
    def _get_dfs_priority(  # 按DFS优先级重建队列
        cur_node: TreeNode,  # 当前树节点
        node_to_priority: Dict[TreeNode, int],  # 节点到优先级（权重）的映射
        last_node_to_reqs: Dict[TreeNode, List[Req]],  # 节点到请求列表的映射
        q: List,  # 输出队列
    ) -> None:
        """按深度优先顺序遍历树节点，按权重降序处理子节点，将请求加入队列"""
        children = [child for child in cur_node.children.values()]  # 获取当前节点的所有子节点
        children.sort(key=lambda x: -node_to_priority[x])  # 按权重降序排列子节点
        for child in children:  # 遍历排序后的子节点
            SchedulePolicy._get_dfs_priority(  # 递归处理每个子节点
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])  # 将当前节点对应的请求添加到队列末尾


class AddReqResult(Enum):
    """添加请求的结果枚举"""
    CONTINUE = auto()  # Continue to add requests  # 可以继续添加请求
    NO_TOKEN = auto()  # No token left  # 没有剩余token
    OTHER = auto()  # Other reasons to stop adding requests  # 其他原因停止添加


class PrefillAdder:
    """预填充添加器，负责在调度过程中添加预填充请求并管理token预算"""
    def __init__(  # 初始化方法
        self,
        page_size: int,  # 页大小
        tree_cache: BasePrefixCache,  # 树形前缀缓存
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,  # KV池分配器
        running_batch: ScheduleBatch,  # 运行中的批次
        new_token_ratio: float,  # 新token比例
        rem_input_tokens: int,  # 剩余输入token数
        rem_chunk_tokens: Optional[int],  # 剩余分块token数
        num_mixed_decode_tokens: int = 0,  # 混合解码token数，默认0
        priority_scheduling_preemption_threshold: int = 0,  # 优先级调度抢占阈值，默认0
        max_prefill_bs: int = 0,  # 最大预填充批次大小，默认0
        max_running_requests: Optional[int] = None,  # 最大运行请求数，可选
        prefill_max_requests: Optional[int] = None,  # 预填充最大请求数，可选
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,  # 预填充延迟器，可选
        dllm_config: Optional[DllmConfig] = None,  # 扩散语言模型配置，可选
        waiting_queue_len: int = 0,  # 等待队列长度，默认0
    ):
        self.page_size = page_size  # 保存页大小
        self.tree_cache = tree_cache  # 保存树缓存引用
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator  # 保存KV池分配器
        self.running_batch = running_batch  # 保存运行批次引用
        self.new_token_ratio = new_token_ratio  # 保存新token比例
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens  # 剩余输入token减去混合解码token
        self.rem_chunk_tokens = rem_chunk_tokens  # 保存剩余分块token数
        self.dllm_config = dllm_config  # 保存DLLM配置

        if self.dllm_config is not None:  # 如果配置了DLLM
            self._init_dllm_meta(dllm_config)  # 初始化DLLM元数据

        if self.rem_chunk_tokens is not None:  # 如果启用了分块预填充
            self.rem_chunk_tokens -= num_mixed_decode_tokens  # 从剩余分块token中减去混合解码token
        self.rem_total_token_offset = num_mixed_decode_tokens  # 剩余总token偏移量初始化为混合解码token数
        self.cur_rem_token_offset = num_mixed_decode_tokens  # 当前剩余token偏移量初始化为混合解码token数

        self.req_states = None  # 请求状态列表，初始为None
        self.can_run_list = []  # 可以运行的请求列表
        self.preempt_list = []  # 被抢占的请求列表
        self.new_chunked_req = None  # 新的分块请求
        self.log_hit_tokens = 0  # 日志记录的命中token数
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0  # 日志记录的输入token数

        if running_batch is not None:  # 如果有运行中的批次
            # Estimate the offset in the remaining token space
            self.rem_total_token_offset += sum(  # 累加运行批次中每个请求的token偏移量
                [
                    self._get_running_request_total_token_offset(r)  # 计算每个运行请求的总token偏移
                    for r in running_batch.reqs  # 遍历运行批次中的请求
                ]
            )

        # DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and
        # exposes the full SWA allocator interface.
        # DeepSeek V4 HiSparse内部封装了SWA分配器，并暴露完整的SWA接口
        self.is_hybrid_swa = isinstance(  # 判断是否为混合SWA模式
            self.token_to_kv_pool_allocator,  # 检查KV池分配器类型
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),  # 是否为SWA或HiSparse分配器
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()  # 判断是否支持Mamba SSM缓存

        self.rem_swa_token_offset = 0  # 剩余SWA token偏移量初始化为0

        self.priority_scheduling_preemption_threshold = (  # 保存优先级调度抢占阈值
            priority_scheduling_preemption_threshold
        )
        self.dsa_prefill_cp_in_seq_split = is_dsa_prefill_cp_in_seq_split()  # DSA预填充上下文并行序列拆分标志
        self.max_running_requests = max_running_requests  # 保存最大运行请求数
        self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()  # 预填充上下文并行是否启用
        self.prefill_max_requests = prefill_max_requests  # 保存预填充最大请求数
        self.prefill_delayer_single_pass = prefill_delayer_single_pass  # 保存预填充延迟器
        self.max_prefill_bs = max_prefill_bs  # 保存最大预填充批次大小
        # Snapshot of scheduler waiting_queue length at the start of this
        # prefill pass. Used by PrefillDelayer's queue-based trigger.
        # 调度器等待队列长度快照，用于预填充延迟器的基于队列的触发
        self.waiting_queue_len = waiting_queue_len  # 保存等待队列长度

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        """初始化扩散语言模型（DLLM）的元数据"""
        self.dllm_block_size = dllm_config.block_size  # 保存DLLM块大小
        max_running_reqs = dllm_config.max_running_requests  # 获取最大运行请求数

        self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size  # 计算剩余DLLM token数

    def _get_running_request_total_token_offset(self, req: Req) -> int:
        """计算运行中请求的总token偏移量（预估该请求未来需要的解码token数）"""
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),  # 剩余待生成的token数
                CLIP_MAX_NEW_TOKENS,  # 上限裁剪
            )
            * self.new_token_ratio  # 乘以新token比例
        )

    @property
    def rem_total_tokens(self):
        """剩余总token数属性：可用token + 可驱逐token - 总偏移量"""
        if self.is_hybrid_swa:  # 如果是混合SWA模式
            available_and_evictable = (  # 计算可用和可驱逐token
                self.token_to_kv_pool_allocator.full_available_size()  # 完全可用大小
                + self.tree_cache.full_evictable_size()  # 完全可驱逐大小
            )
        elif self.is_hybrid_ssm_cache:  # 如果是混合SSM缓存模式
            available_and_evictable = (  # 计算可用和可驱逐token
                self.token_to_kv_pool_allocator.available_size()  # 可用大小
                + self.tree_cache.full_evictable_size()  # 完全可驱逐大小
            )
        else:  # 普通模式
            available_and_evictable = (  # 计算可用和可驱逐token
                self.token_to_kv_pool_allocator.available_size()  # 可用大小
                + self.tree_cache.evictable_size()  # 可驱逐大小
            )
        return available_and_evictable - self.rem_total_token_offset  # 返回扣除偏移后的剩余总token数

    @property
    def rem_swa_tokens(self):
        """剩余SWA token数属性"""
        return (
            self.token_to_kv_pool_allocator.swa_available_size()  # SWA可用大小
            + self.tree_cache.swa_evictable_size()  # SWA可驱逐大小
            - self.rem_swa_token_offset  # 减去SWA偏移量
        )

    @property
    def cur_rem_tokens(self):
        """当前剩余token数属性：可用token + 可驱逐token - 当前偏移量"""
        if self.is_hybrid_swa:  # 如果是混合SWA模式
            available_and_evictable = (  # 计算可用和可驱逐token
                self.token_to_kv_pool_allocator.full_available_size()  # 完全可用大小
                + self.tree_cache.full_evictable_size()  # 完全可驱逐大小
            )
        elif self.is_hybrid_ssm_cache:  # 如果是混合SSM缓存模式
            available_and_evictable = (  # 计算可用和可驱逐token
                self.token_to_kv_pool_allocator.available_size()  # 可用大小
                + self.tree_cache.full_evictable_size()  # 完全可驱逐大小
            )
        else:  # 普通模式
            available_and_evictable = (  # 计算可用和可驱逐token
                self.token_to_kv_pool_allocator.available_size()  # 可用大小
                + self.tree_cache.evictable_size()  # 可驱逐大小
            )

        return available_and_evictable - self.cur_rem_token_offset  # 返回扣除当前偏移后的剩余token数

    def _swa_budget_for_req(self, extend_input_len: int) -> int:
        """SWA pool budget per request. Only valid when is_hybrid_swa is True.

        With chunked prefill + overlap scheduler, the peak SWA occupancy is:
          chunk N (running, not yet in tree) + sliding window (locked in tree)
          + chunk N+1 (new allocation)
        Since chunk N and locked tokens are already excluded from
        swa_available + swa_evictable, the budget only needs to cover the
        chunk N+1 allocation. We floor at sliding_window_size to reserve
        room for the decode phase.
        计算单个请求的SWA池预算，仅在混合SWA模式下有效。
        在分块预填充+重叠调度器下，峰值SWA占用包括：运行中的分块N + 滑动窗口 + 新分配的分块N+1。
        预算至少为滑动窗口大小，以确保解码阶段有足够空间。
        """
        if self.rem_chunk_tokens is not None:  # 如果启用了分块预填充
            alloc = min(extend_input_len, self.rem_chunk_tokens)  # 取扩展输入长度和剩余分块token的较小值
        else:
            alloc = extend_input_len  # 否则直接使用扩展输入长度
        return max(alloc, self.tree_cache.sliding_window_size) + self.page_size  # 至少为滑动窗口大小，加上一页开销

    def ceil_paged_tokens(self, tokens: int) -> int:
        """将token数向上取整到页大小的整数倍"""
        return -(-tokens // self.page_size) * self.page_size  # 利用负数除法实现向上取整

    def budget_state(self):
        """检查当前预算状态，返回是否可以继续添加请求"""
        no_token = self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0  # 判断是否有可用token
        if not no_token and self.is_hybrid_swa:  # 如果还有token且是混合SWA模式
            no_token = self.rem_swa_tokens <= 0  # 进一步检查SWA token
        if no_token:  # 如果没有可用token
            return AddReqResult.NO_TOKEN  # 返回无token结果

        if self.rem_input_tokens <= 0:  # 如果剩余输入token不足
            return AddReqResult.OTHER  # 返回其他原因

        if self.dllm_config is not None:  # 如果配置了DLLM
            if self.rem_dllm_tokens <= 0:  # 如果剩余DLLM token不足
                return AddReqResult.OTHER  # 返回其他原因
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:  # 如果启用了分块且剩余分块token不足
                return AddReqResult.OTHER  # 返回其他原因

        return AddReqResult.CONTINUE  # 可以继续添加请求

    def _update_prefill_budget(  # 更新预填充预算
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        """更新预填充预算：扣除已使用的token，记录命中和输入token数"""
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        extend_input_len = self.ceil_paged_tokens(extend_input_len)  # 将扩展输入长度向上取整到页大小倍数

        # alloc_extend reserves an extra page_size per request to make sure the budget doesn't over-commit
        page_overhead = self.page_size  # 每个请求预留一页的开销
        self.rem_total_token_offset += extend_input_len + max_new_tokens + page_overhead  # 更新总token偏移量
        self.cur_rem_token_offset += extend_input_len + page_overhead  # 更新当前token偏移量
        self.rem_input_tokens -= extend_input_len  # 扣除已使用的输入token

        if self.is_hybrid_swa:  # 如果是混合SWA模式
            self.rem_swa_token_offset += self._swa_budget_for_req(extend_input_len)  # 更新SWA token偏移量

        if self.dllm_config is not None:  # 如果配置了DLLM
            self.rem_dllm_tokens -= extend_input_len  # 扣除DLLM token
        elif self.rem_chunk_tokens is not None:  # 如果启用了分块预填充
            self.rem_chunk_tokens -= extend_input_len  # 扣除分块token

        self.log_hit_tokens += prefix_len  # 累加命中token数
        self.log_input_tokens += extend_input_len  # 累加输入token数

    def _get_dllm_remain_tokens(self) -> int:
        """获取DLLM剩余可用token数"""
        _rem_tokens = min(  # 取以下三个值的最小值
            self.rem_dllm_tokens,  # 剩余DLLM token数
            self.dllm_block_size,  # DLLM块大小
            int(self.rem_total_tokens),  # 剩余总token数
        )
        if _rem_tokens <= 0:  # 如果计算结果非正
            _rem_tokens = self.rem_dllm_tokens  # 回退为剩余DLLM token数

        return _rem_tokens  # 返回剩余DLLM token数

    def _add_dllm_req(self, req: Req, prefix_len: int):
        """添加DLLM请求到可运行列表，处理输入截断和预算更新"""
        # FIXME: consider the case when rem_dllm_tokens < dllm_block_size,
        # the diffusion unmask process may have some problems
        # Make sure at least one page is available
        trunc_len = (  # 计算截断长度，确保至少一页可用
            min(self.rem_dllm_tokens, self.dllm_block_size)  # 取剩余DLLM token和块大小的较小值
            // self.page_size  # 整除页大小
            * self.page_size  # 再乘以页大小，向下对齐
        )

        req.extend_input_len = trunc_len  # 设置请求的扩展输入长度为截断长度
        req.fill_ids = req.fill_ids[: prefix_len + trunc_len]  # 截断fill_ids到前缀长度+截断长度

        self.can_run_list.append(req)  # 将请求加入可运行列表

        self._update_prefill_budget(prefix_len, trunc_len, 0)  # 更新预填充预算（DLLM不需要max_new_tokens）

    def _req_inc_lock_ref(self, req: Req):
        """增加请求最后节点在树缓存中的锁引用计数"""
        result = self.tree_cache.inc_lock_ref(req.last_node)  # 增加锁引用
        if self.is_hybrid_swa:  # 如果是混合SWA模式
            req.swa_uuid_for_lock = result.swa_uuid_for_lock  # 保存SWA锁的UUID

    def add_dllm_staging_req(self, req: Req):
        """添加DLLM暂存请求，处理输入截断和预算更新，返回添加结果"""
        assert self.dllm_config is not None  # 断言DLLM配置不为空
        _rem_tokens = self._get_dllm_remain_tokens()  # 获取剩余DLLM token数

        if _rem_tokens <= 0:  # 如果没有剩余token
            return AddReqResult.NO_TOKEN  # 返回无token结果

        # Truncate input length to available tokens and update request metadata
        truncated = req.extend_input_len > _rem_tokens  # 判断是否需要截断
        req.extend_input_len = min(req.extend_input_len, _rem_tokens)  # 截断扩展输入长度
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]  # 截断fill_ids
        self.can_run_list.append(req)  # 将请求加入可运行列表

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (  # 仅在未截断时预留max_new_tokens
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)  # 取max_new_tokens和裁剪上限的较小值
            if not truncated  # 未截断时
            else 0  # 截断时不预留
        )
        self._update_prefill_budget(0, req.extend_input_len, max_new_tokens)  # 更新预填充预算

        # Return based on remaining token availability
        return (  # 根据剩余token可用性返回结果
            AddReqResult.NO_TOKEN  # 如果剩余DLLM token不足
            if self._get_dllm_remain_tokens() <= 0  # 检查剩余DLLM token
            else AddReqResult.CONTINUE  # 否则可以继续
        )

    def add_chunked_req(self, req: Req):
        """添加分块预填充请求，处理输入截断和预算更新，返回截断后的请求或None"""
        if self.dllm_config is not None:  # 如果配置了DLLM
            _rem_tokens = self._get_dllm_remain_tokens()  # 使用DLLM剩余token数
        else:
            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))  # 取分块和总token的较小值
            if self.is_hybrid_swa:  # 如果是混合SWA模式
                # alloc_extend needs extend_num_tokens + page_size per request,
                # so reserve one page here to avoid OOM
                _rem_tokens = min(  # 进一步限制为SWA token减一页
                    _rem_tokens, int(self.rem_swa_tokens) - self.page_size
                )
            # The chunked_req must be added to the list; otherwise, it will cause a memory leak.
            # Therefore, in certain cases where _rem_tokens <= 0, it should be replaced with rem_chunk_tokens.
            if _rem_tokens <= 0:  # 如果剩余token非正
                if self.is_hybrid_swa:  # 如果是混合SWA模式
                    return req  # 返回请求（SWA模式下无法替代）
                _rem_tokens = self.rem_chunk_tokens  # 用剩余分块token替代，避免内存泄漏

        truncated = req.extend_input_len > _rem_tokens  # 判断是否需要截断
        req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))  # 截断扩展输入长度
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]  # 截断fill_ids
        self.can_run_list.append(req)  # 将请求加入可运行列表
        self._update_prefill_budget(  # 更新预填充预算
            0,  # 前缀长度为0（分块请求的前缀已在前一次处理中计算）
            req.extend_input_len,  # 扩展输入长度
            (  # max_new_tokens
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)  # 未截断时预留
                if not truncated
                else 0  # 截断时不预留
            ),
        )

        # Return if chunked prefill not finished
        return req if truncated else None  # 如果被截断则返回请求（表示预填充未完成），否则返回None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        """上下文管理器：临时锁定树节点，退出时自动释放锁"""
        dec_lock_params = None  # 解锁参数初始化为None
        try:
            result = self.tree_cache.inc_lock_ref(last_node)  # 增加锁引用
            if self.tree_cache.is_tree_cache():  # 如果是树缓存
                # init_load_back may revive SWA/Mamba tombstones while this
                # temporary admission lock is held. Release must mirror the
                # exact nodes skipped at acquire time.
                dec_lock_params = result.to_dec_params()  # 保存解锁参数以镜像获取时跳过的节点
            yield None  # 产出None，执行with块内的代码
        finally:
            if dec_lock_params is not None:  # 如果有解锁参数
                self.tree_cache.dec_lock_ref(last_node, dec_lock_params)  # 使用参数减少锁引用
            else:
                self.tree_cache.dec_lock_ref(last_node)  # 无参数减少锁引用

    def add_one_req_ignore_eos(self, req: Req):
        """添加一个忽略EOS的请求，进行更严格的内存检查以防止OOM"""
        paged_input = self.ceil_paged_tokens(req.extend_input_len)  # 将扩展输入长度向上取整到页大小
        if paged_input > min(self.cur_rem_tokens, self.rem_total_tokens):  # 如果分页后输入超过剩余token
            return AddReqResult.NO_TOKEN  # 返回无token结果
        if self.is_hybrid_swa:  # 如果是混合SWA模式
            if self._swa_budget_for_req(req.extend_input_len) > self.rem_swa_tokens:  # 如果SWA预算不足
                return AddReqResult.NO_TOKEN  # 返回无token结果

        def add_req_state(r, insert_sort=False):  # 添加请求状态的内嵌函数
            new_token_ratio = (  # 计算新token比例
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio  # 忽略EOS时比例为1.0
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(  # 计算剩余待生成token
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)  # 计算已占用token数

            if tokens_left <= 0:  # 如果没有剩余待生成token
                return  # 跳过

            if not insert_sort:  # 如果不是插入排序模式
                self.req_states.append((tokens_left, tokens_occupied))  # 直接追加到状态列表
            else:
                i = 0  # 插入位置初始化为0
                for i in range(len(self.req_states)):  # 遍历已有状态
                    if tokens_left <= self.req_states[i][0]:  # 找到第一个大于等于的位置
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))  # 在正确位置插入

        if self.req_states is None:  # 如果状态列表尚未初始化
            self.req_states = []  # 初始化为空列表
            add_req_state(req)  # 添加当前请求状态
            if self.running_batch is not None:  # 如果有运行中的批次
                for r in self.running_batch.reqs:  # 遍历运行中的请求
                    add_req_state(r)  # 添加每个运行请求的状态
            for r in self.can_run_list:  # 遍历可运行列表
                add_req_state(r)  # 添加每个可运行请求的状态
            self.req_states.sort(key=lambda x: x[0])  # 按剩余token数排序
        else:
            add_req_state(req, insert_sort=True)  # 已有状态列表时使用插入排序

        if not self.is_hybrid_swa:  # 如果不是混合SWA模式
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.cur_rem_tokens - self.ceil_paged_tokens(  # 计算扣除当前请求后的剩余token
                req.extend_input_len
            )
            tokens_freed = 0  # 已释放token数初始化为0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):  # 遍历所有请求状态
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i  # 当前及之后的请求数（批大小）
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs  # 最小空闲token
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:  # 如果空闲token不足以预留
                    return AddReqResult.NO_TOKEN  # 返回无token结果
                tokens_freed += tokens_occupied  # 累加释放的token数

        if (self.prefill_delayer_single_pass is not None) and (  # 如果配置了预填充延迟器
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(  # 且延迟器不允许预填充
                local_prefillable=True
            )
        ):
            return AddReqResult.OTHER  # 返回其他原因

        if self.dllm_config is not None:  # 如果配置了DLLM
            if self.rem_dllm_tokens <= 0:  # 如果剩余DLLM token不足
                return AddReqResult.OTHER  # 返回其他原因

            self._add_dllm_req(req, 0)  # 添加DLLM请求
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled  # 未启用分块预填充
            or req.extend_input_len <= self.rem_chunk_tokens  # it is the last chunk  # 或者是最后一个分块
        ):
            # Non-chunked prefill
            self.can_run_list.append(req)  # 非分块预填充：直接加入可运行列表
            self._update_prefill_budget(  # 更新预填充预算
                0,  # 前缀长度为0
                req.extend_input_len,  # 扩展输入长度
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),  # 预留max_new_tokens
            )
        else:
            if self.rem_chunk_tokens <= 0:  # 如果剩余分块token不足
                return AddReqResult.OTHER  # 返回其他原因

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens  # 截断长度为剩余分块token数

            req.set_extend_input_len(trunc_len)  # 设置请求的扩展输入长度为截断长度
            req.fill_ids = req.fill_ids[:trunc_len]  # 截断fill_ids
            self.can_run_list.append(req)  # 将请求加入可运行列表
            self.new_chunked_req = req  # 记录新的分块请求
            self._update_prefill_budget(0, trunc_len, 0)  # 更新预填充预算（不预留max_new_tokens）

        return self.budget_state()  # 返回当前预算状态

    def add_one_req(  # 添加单个请求
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        """添加单个预填充请求，处理前缀缓存加载、预算检查、分块截断等逻辑"""
        if (self.prefill_delayer_single_pass is not None) and (  # 如果配置了预填充延迟器
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(  # 且延迟器不允许预填充
                local_prefillable=True,  # 本地可预填充
                running_batch=self.running_batch.batch_size(),  # 运行批大小
                max_prefill_bs=self.max_prefill_bs,  # 最大预填充批大小
                max_running_requests=self.max_running_requests,  # 最大运行请求数
                waiting_queue_len=self.waiting_queue_len,  # 等待队列长度
            )
        ):
            return AddReqResult.OTHER  # 返回其他原因
        # TODO support cp with multiple requests
        # Enabling context parallelism currently presents precision issues;
        # therefore, the prefill-batch setting is temporarily set to 1.
        if (self.dsa_prefill_cp_in_seq_split) and len(self.can_run_list) >= 1:  # 如果启用了DSA预填充上下文并行且已有请求
            return AddReqResult.OTHER  # 返回其他原因（CP仅支持单请求）

        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:  # 如果达到预填充最大请求数
            return AddReqResult.OTHER  # 返回其他原因

        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):  # 如果忽略EOS且缓存被禁用
            return self.add_one_req_ignore_eos(req)  # 使用忽略EOS的添加逻辑

        # Reserve page_size for page-alignment overhead. The paged allocator
        # may consume up to one extra page per request (see alloc_extend), and
        # _update_prefill_budget already accounts for this in the deduction.
        # Without this, admission is more optimistic than the actual budget
        # deduction, allowing over-admission when the pool is nearly full.
        max_new = min(  # 计算裁剪后的最大新token数
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),  # 剩余待生成token数，最小为0
            CLIP_MAX_NEW_TOKENS,  # 上限裁剪
        )
        total_tokens = req.extend_input_len + max_new + self.page_size  # 总token需求 = 扩展输入 + 最大新token + 页开销

        # adjusting the input_tokens based on host_hit_length and page_size
        real_input_tokens = req.extend_input_len - req.host_hit_length  # 减去主机命中长度得到实际需要分配的输入token
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)  # 向上取整到页大小倍数
        prefix_len = len(req.prefix_indices)  # 获取前缀索引长度

        if total_tokens >= self.rem_total_tokens:  # 如果总token需求超过剩余总token
            return AddReqResult.NO_TOKEN  # 返回无token结果

        if self.is_hybrid_swa:  # 如果是混合SWA模式
            swa_needed = self._swa_budget_for_req(req.extend_input_len)  # 计算所需的SWA预算
            if swa_needed >= self.rem_swa_tokens:  # 如果SWA预算不足
                return AddReqResult.NO_TOKEN  # 返回无token结果

        if (
            self.rem_chunk_tokens is None  # 未启用分块预填充
            and len(self.can_run_list) != 0  # 可运行列表不为空
            and real_input_tokens >= self.rem_input_tokens  # 实际输入token超过剩余输入token
        ):
            # If without chunked prefill:
            # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
            # - if the can_run_list is empty, always accept the first prefill request
            return AddReqResult.OTHER  # 不满足约束，返回其他原因

        with self._lock_node(req.last_node):  # 锁定请求的最后节点
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens:  # 锁定后重新检查总token（可能减少）
                return AddReqResult.NO_TOKEN  # 返回无token结果

            if self.is_hybrid_swa:  # 如果是混合SWA模式
                swa_needed = self._swa_budget_for_req(req.extend_input_len)  # 重新计算SWA预算
                if swa_needed >= self.rem_swa_tokens:  # 如果SWA预算不足
                    return AddReqResult.NO_TOKEN  # 返回无token结果

            if req.host_hit_length > 0:  # 如果有主机命中长度（分层缓存命中）
                new_indices, req.last_node = self.tree_cache.init_load_back(  # 从主机加载缓存回设备
                    InitLoadBackParams(
                        best_match_node=req.best_match_node,  # 最佳匹配节点
                        host_hit_length=req.host_hit_length,  # 主机命中长度
                        req=req,  # 请求对象
                    )
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])  # 将新索引拼接到前缀索引
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))  # 更新扩展输入长度
                prefix_len = len(req.prefix_indices)  # 更新前缀长度
                req.cache_protected_len = prefix_len  # 设置缓存保护长度

            input_tokens = self.ceil_paged_tokens(req.extend_input_len)  # 将扩展输入长度向上取整

            if (
                self.rem_chunk_tokens is None  # 未启用分块预填充
                and len(self.can_run_list) != 0  # 可运行列表不为空
                and input_tokens >= self.rem_input_tokens  # 输入token超过剩余输入token
            ):
                # If without chunked prefill:
                # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
                # - if the can_run_list is empty, always accept the first prefill request
                return AddReqResult.OTHER  # 不满足约束，返回其他原因

            if self.dllm_config is not None:  # 如果配置了DLLM
                if self.rem_dllm_tokens <= 0:  # 如果剩余DLLM token不足
                    return AddReqResult.OTHER  # 返回其他原因

                assert (
                    truncation_align_size is None
                ), "truncation_align_size is not supported for dllm prefill"  # DLLM不支持截断对齐

                self._add_dllm_req(req, prefix_len)  # 添加DLLM请求
                self._req_inc_lock_ref(req)  # 增加锁引用
            elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:  # 非分块或最后一个分块
                # Non-chunked prefill
                self.can_run_list.append(req)  # 非分块预填充：加入可运行列表

                self._req_inc_lock_ref(req)  # 增加锁引用
                self._update_prefill_budget(  # 更新预填充预算
                    prefix_len,  # 前缀长度
                    input_tokens,  # 输入token数
                    min(  # 预留max_new_tokens
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS,
                    ),
                )
            else:
                # Make sure at least one page is available
                trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size  # 将剩余分块token向下对齐到页大小

                if trunc_len <= 0:  # 如果截断后长度非正
                    return AddReqResult.OTHER  # 返回其他原因

                # When truncation align size is set, we want to assert that the prefill prefix length is multiple of truncation align size
                # A typical use case is when deterministic inference is enabled with flashinfer attention backend,
                # we need the prefill prefix length to be multiple of attention split size
                if truncation_align_size is not None:  # 如果设置了截断对齐大小
                    if trunc_len < truncation_align_size:  # 如果截断长度小于对齐大小
                        return AddReqResult.OTHER  # 返回其他原因
                    else:
                        trunc_len = truncation_align_size * (  # 将截断长度对齐到对齐大小的倍数
                            trunc_len // truncation_align_size
                        )

                now_input_len = trunc_len + len(req.prefix_indices)  # 当前输入长度 = 截断长度 + 前缀长度
                now_input_len = now_input_len // self.page_size * self.page_size  # 向下对齐到页大小
                trunc_len = now_input_len - len(req.prefix_indices)  # 重新计算截断长度

                if trunc_len <= 0:  # 如果重新计算后截断长度非正
                    return AddReqResult.OTHER  # 返回其他原因

                # Chunked prefill
                req.set_extend_input_len(trunc_len)  # 设置请求的扩展输入长度为截断长度
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]  # 截断fill_ids

                self.can_run_list.append(req)  # 将请求加入可运行列表
                self.new_chunked_req = req  # 记录新的分块请求

                self._req_inc_lock_ref(req)  # 增加锁引用
                self._update_prefill_budget(prefix_len, trunc_len, 0)  # 更新预填充预算（不预留max_new_tokens）

        return self.budget_state()  # 返回当前预算状态

    def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.
        抢占运行中的请求以服务新请求。当优先级差满足阈值且token总量验证通过时执行抢占。
        返回True表示已执行抢占，新请求可以被调度。
        """
        # Iterate running requests to find preemptible requests
        priority_sign = 1 if server_args.schedule_low_priority_values_first else -1  # 计算优先级符号

        # NOTE: A request finishes in two phases:
        #   1) update_finish_state + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        valid_running_reqs = (  # 过滤有效的运行中请求
            r  # 请求对象
            for r in self.running_batch.reqs  # 遍历运行批次中的请求
            if r not in self.preempt_list and not r.finished()  # 排除已抢占和已完成的请求
        )

        sorted_valid_running_reqs = sorted(  # 对有效请求按优先级排序
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),  # 按相反优先级排序（低优先级的先被抢占）
                -x.time_stats.wait_queue_entry_time,  # 相同优先级时后到的先被抢占
            ),
        )

        preemptible_reqs = []  # 可抢占的请求列表
        min_tokens_to_remove = (  # 需要释放的最小token数
            req.extend_input_len  # 新请求的扩展输入长度
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)  # 加上新请求的预估解码token
            - self.rem_total_tokens  # 减去剩余总token
        )
        for running_req in sorted_valid_running_reqs:  # 遍历排序后的有效运行请求
            # Priority difference needs to meet the threshold to be preemptible.
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)  # 计算优先级差

            if priority_diff > self.priority_scheduling_preemption_threshold:  # 如果优先级差超过阈值
                preemptible_reqs.append(running_req)  # 加入可抢占列表
                min_tokens_to_remove -= self._get_running_request_total_token_offset(  # 减去该请求释放的token
                    running_req
                )
                if min_tokens_to_remove <= 0:  # 如果已释放足够的token
                    break  # 停止遍历
            else:
                break  # 优先级差不足，停止遍历（因为已排序）

        # Check max token count limit can be met
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:  # 如果没有可抢占的请求或释放的token不够
            return False  # 抢占失败

        # Preempt running requests. Release allocated resources for immediate usage.
        preemptible_reqs = set(preemptible_reqs)  # 转为集合以便快速查找
        keep_indices = []  # 保留的请求索引列表
        release_counter = 0  # 释放计数器
        for i, running_req in enumerate(self.running_batch.reqs):  # 遍历运行批次中的请求
            if running_req in preemptible_reqs:  # 如果该请求在可抢占集合中
                self.rem_total_token_offset -= (  # 减少总token偏移量
                    self._get_running_request_total_token_offset(running_req)  # 扣除该请求的token偏移
                )
                release_counter += 1  # 释放计数器加1
                self.running_batch.release_req(  # 释放该请求的资源
                    i, len(self.running_batch.reqs) - release_counter, server_args
                )
            else:
                keep_indices.append(i)  # 否则保留该索引
        self.running_batch.filter_batch(keep_indices=keep_indices)  # 过滤批次，只保留未被抢占的请求
        self.preempt_list.extend(preemptible_reqs)  # 将被抢占的请求加入抢占列表
        return True  # 抢占成功
