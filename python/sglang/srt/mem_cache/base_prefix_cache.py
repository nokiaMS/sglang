from __future__ import annotations  # 启用延迟类型注解评估，允许在类型注解中使用尚未定义的类型

import dataclasses  # 导入dataclasses模块，用于定义数据类
import time  # 导入time模块，用于时间相关操作
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import (  # 导入类型提示相关模块
    TYPE_CHECKING,  # 类型检查标志，用于条件导入类型
    Any,  # 任意类型
    NamedTuple,  # 具名元组基类
    Optional,  # 可选类型
    Protocol,  # 协议基类，用于结构化子类型
    Tuple,  # 元组类型
    runtime_checkable,  # 运行时可检查协议装饰器
)

import torch  # 导入PyTorch深度学习框架

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator  # 导入KV池分配器基类
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # 导入请求到token的映射池
from sglang.srt.observability.metrics_collector import (  # 导入可观测性指标收集器相关模块
    STAT_LOGGER_ROLE_RADIX_CACHE,  # 基数缓存统计日志角色标识
    RadixCacheMetricsCollector,  # 基数缓存指标收集器类
    resolve_collector_class,  # 解析收集器类的工具函数
)

if TYPE_CHECKING:  # 仅在类型检查时导入，运行时不导入
    from sglang.srt.managers.schedule_batch import Req  # 导入请求类
    from sglang.srt.mem_cache.radix_cache import RadixKey  # 导入基数缓存键类型
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # 导入树组件类型
        ComponentType,  # 组件类型枚举
    )


# 前缀缓存协议，定义了前缀缓存必须具备的属性
@runtime_checkable  # 标记为运行时可检查的协议
class PrefixCacheTrait(Protocol):  # 前缀缓存特征协议类
    req_to_token_pool: ReqToTokenPool  # 请求到token的映射池
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator  # token到KV池的分配器
    page_size: int  # 页大小
    disable: bool  # 是否禁用缓存


# 前缀匹配参数数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""
    # 不同缓存类型统一的前缀匹配参数

    key: RadixKey  # 基数缓存键

    # Mamba specific
    # Mamba模型特有的参数
    cow_mamba: bool = False  # 是否对Mamba状态进行写时复制
    req: Optional[Req] = None  # 关联的请求对象


# 插入参数数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class InsertParams:
    """Unified parameters for insert across different cache types"""
    # 不同缓存类型统一的插入参数

    key: Optional[RadixKey] = None  # 基数缓存键（可选）
    value: Optional[torch.Tensor] = None  # 插入的KV值张量（可选）

    # Mamba specific
    # Mamba模型特有的参数
    mamba_value: Optional[torch.Tensor] = None  # Mamba状态值张量（可选）

    # SWA specific
    # 滑动窗口注意力特有的参数
    prev_prefix_len: int = 0  # 先前前缀长度
    swa_evicted_seqlen: int = 0  # 滑动窗口中被驱逐的序列长度

    # General
    # 通用参数
    chunked: bool = False  # 是否为分块插入
    priority: int = 0  # 插入优先级


# 插入结果数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class InsertResult:
    """Result of an insert operation"""
    # 插入操作的结果

    prefix_len: int  # 匹配到的前缀长度
    mamba_exist: bool = False  # Mamba状态是否已存在
    inserted_host_node: Any = None  # 在主机端插入的节点


# 驱逐参数数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class EvictParams:
    """Unified parameters for evict across different cache types"""
    # 不同缓存类型统一的驱逐参数

    num_tokens: int = 0  # 需要驱逐的token数量
    swa_num_tokens: int = 0  # 需要驱逐的滑动窗口token数量
    mamba_num: int = 0  # 需要驱逐的Mamba状态数量


# 驱逐结果数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class EvictResult:
    """Result of an evict operation"""
    # 驱逐操作的结果

    num_tokens_evicted: int = 0  # 已驱逐的token数量
    swa_num_tokens_evicted: int = 0  # 已驱逐的滑动窗口token数量
    mamba_num_evicted: int = 0  # 已驱逐的Mamba状态数量


# 增加锁引用结果数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class IncLockRefResult:
    """Result of an inc_lock_ref operation."""
    # 增加锁引用操作的结果

    delta: Optional[int] = None  # 锁引用计数的变化量
    swa_uuid_for_lock: Optional[int] = None  # 滑动窗口设备端锁的UUID
    swa_uuid_for_host_lock: Optional[int] = None  # 滑动窗口主机端锁的UUID
    # Component nodes that were tombstones at acquire time. Replaying this set
    # at release prevents a short-lived lock from consuming a later load-back or
    # request lock after that tombstone becomes a valid device value.
    # 在获取时为墓碑状态的组件节点。在释放时重放此集合可防止短命锁
    # 在墓碑变为有效设备值后消耗后续的加载回写或请求锁。
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(  # 需要跳过加锁的节点ID集合
        default_factory=dict  # 默认工厂函数为空字典
    )

    def to_dec_params(self) -> "DecLockRefParams":  # 转换为减少锁引用的参数对象
        """Convert to the corresponding DecLockRefParams for dec_lock_ref."""
        # 转换为对应的DecLockRefParams，用于dec_lock_ref操作
        return DecLockRefParams(  # 构造并返回DecLockRefParams实例
            swa_uuid_for_lock=self.swa_uuid_for_lock,  # 传递滑动窗口设备端锁UUID
            swa_uuid_for_host_lock=self.swa_uuid_for_host_lock,  # 传递滑动窗口主机端锁UUID
            skip_lock_node_ids={  # 构建需要跳过加锁的节点ID字典
                component_type: set(node_ids)  # 为每个组件类型创建节点ID集合的副本
                for component_type, node_ids in self.skip_lock_node_ids.items()  # 遍历所有组件类型及其节点ID
            },
        )


# 减少锁引用参数数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class DecLockRefParams:
    """Parameters for dec_lock_ref operation."""
    # 减少锁引用操作的参数

    swa_uuid_for_lock: Optional[int] = None  # 滑动窗口设备端锁的UUID
    swa_uuid_for_host_lock: Optional[int] = None  # 滑动窗口主机端锁的UUID
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(  # 需要跳过加锁的节点ID集合
        default_factory=dict  # 默认工厂函数为空字典
    )


# 减少锁引用结果数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""
    # 减少锁引用操作的结果

    delta: Optional[int] = None  # 锁引用计数的变化量


# 初始化加载回写参数数据类
@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types."""
    # 不同缓存类型统一的初始化加载回写参数

    best_match_node: Any  # 最佳匹配节点
    host_hit_length: int  # 主机端命中长度
    mem_quota: Optional[int] = None  # 内存配额限制（可选）
    req: Optional[Req] = None  # 关联的请求对象（可选）


# 前缀匹配结果具名元组
class MatchResult(NamedTuple):  # 继承NamedTuple定义具名元组
    """Result of a prefix match operation.
    # 前缀匹配操作的结果

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        # device_indices: 设备上通过公共前缀匹配到的KV缓存索引
        last_device_node:   The last TreeNode on the device that was matched.
        # last_device_node: 设备上匹配到的最后一个树节点
        last_host_node  :   The last TreeNode on the host that was matched.
        # last_host_node: 主机上匹配到的最后一个树节点
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            # 注意：如果未启用HiCache，此值必须与last_device_node相同
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
                            # 保留用于L3存储预取锚定；L2加载回写使用best_match_node
        best_match_node :   Deepest node accepted by all component validators
        # best_match_node: 所有组件验证器接受的最深节点
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
                            # 在match_prefix期间。每个L2主机到设备加载回写遍历的锚点
                            # 对于不运行多组件验证的旧缓存，将其设为last_host_node
        host_hit_length :   Length of the host cache hit. For pure-KV caches this is the
        # host_hit_length: 主机端缓存命中的长度。对于纯KV缓存，这是
                            number of evicted KV tokens on CPU. For hybrid Mamba models this
                            is max(kv_host_tokens, 1-if-mamba-on-host) so that a mamba-only
                            host hit still triggers load-back without adding a separate field.
                            0 if HiCache is not enabled.
                            # CPU上被驱逐的KV token数量。对于混合Mamba模型，这是
                            # max(kv_host_tokens, 1-if-mamba-on-host)，以便纯Mamba主机命中
                            # 仍能触发加载回写而无需添加单独字段。未启用HiCache时为0
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
        # mamba_branching_seqlen: Mamba基数缓存的分支点，即最长的
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
                                # 页对齐位置，如果存在Mamba状态则可能缓存命中
    """

    device_indices: torch.Tensor  # 设备上匹配到的KV缓存索引张量
    last_device_node: Any  # 设备上匹配到的最后一个树节点
    last_host_node: Any  # 主机上匹配到的最后一个树节点
    best_match_node: Any  # 所有组件验证器接受的最佳匹配节点
    host_hit_length: int = 0  # 主机端缓存命中长度，默认为0
    mamba_branching_seqlen: Optional[int] = None  # Mamba基数缓存分支点序列长度，默认为None
    cache_protected_len: Optional[int] = None  # 缓存受保护长度，默认为None


# 将匹配结果置为零匹配的辅助函数
def zero_match_result(tree_cache, match_result: "MatchResult") -> "MatchResult":  # 将匹配结果清零
    if tree_cache.is_chunk_cache():  # 如果是分块缓存
        # Chunk caches' match_prefix already returns a miss; no root_node to walk back to.
        # 分块缓存的match_prefix已经返回未命中；没有root_node可以回溯
        return match_result  # 直接返回原匹配结果
    root = tree_cache.root_node  # 获取树缓存的根节点
    return match_result._replace(  # 返回替换字段后的新匹配结果
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        # [:0]保留原始张量的数据类型和设备（如CUDA int64），而无需分配新的空张量
        device_indices=match_result.device_indices[:0],  # 设备索引设为空张量，保留类型和设备
        last_device_node=root,  # 最后设备节点设为根节点
        last_host_node=root,  # 最后主机节点设为根节点
        best_match_node=root,  # 最佳匹配节点设为根节点
        host_hit_length=0,  # 主机命中长度设为0
    )


# 前缀缓存基类
class BasePrefixCache(ABC, PrefixCacheTrait):  # 继承抽象基类和前缀缓存特征协议
    """Cache can be indexed by either rid or key."""
    # 缓存可以通过请求ID或键进行索引

    metrics_collector: Optional[RadixCacheMetricsCollector] = (  # 指标收集器，用于收集缓存相关指标
        None  # metrics collector for the cache
        # 缓存的指标收集器，默认为None
    )

    def init_metrics_collector(self):  # 初始化指标收集器
        from sglang.srt.server_args import get_global_server_args  # 导入获取全局服务器参数的函数

        server_args = get_global_server_args()  # 获取全局服务器参数
        labels = {"cache_type": self.__class__.__name__}  # 创建标签字典，包含缓存类型名称
        if server_args.extra_metric_labels:  # 如果服务器参数中有额外的指标标签
            labels.update(server_args.extra_metric_labels)  # 将额外标签更新到标签字典中
        radix_cache_cls = resolve_collector_class(  # 解析收集器类
            server_args,  # 服务器参数
            STAT_LOGGER_ROLE_RADIX_CACHE,  # 基数缓存统计日志角色
            RadixCacheMetricsCollector,  # 基数缓存指标收集器基类
        )
        self.metrics_collector = radix_cache_cls(labels=labels)  # 实例化指标收集器并赋值

    def update_eviction_metrics(self, num_evicted: int, start_time: float):  # 更新驱逐指标
        if self.metrics_collector is not None and num_evicted > 0:  # 如果指标收集器存在且有token被驱逐
            self.metrics_collector.observe_eviction_duration(  # 记录驱逐操作的持续时间
                time.perf_counter() - start_time  # 计算从开始时间到现在的耗时
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)  # 增加驱逐token计数

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def reset(self):  # 重置缓存
        pass  # 由子类实现

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # 前缀匹配
        pass  # 由子类实现

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):  # 缓存已完成的请求
        pass  # 由子类实现

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def cache_unfinished_req(self, req: Req, **kwargs):  # 缓存未完成的请求
        pass  # 由子类实现

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def evict(self, params: EvictParams) -> EvictResult:  # 驱逐缓存条目
        pass  # 由子类实现

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:  # 增加节点的锁引用计数
        pass  # 由子类实现

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def dec_lock_ref(  # 减少节点的锁引用计数
        self, node: Any, params: Optional[DecLockRefParams] = None  # 节点和可选的减少锁引用参数
    ) -> DecLockRefResult:  # 返回减少锁引用结果
        pass  # 由子类实现

    def evictable_size(self):  # 获取可驱逐的大小
        return 0  # 默认返回0

    def full_evictable_size(self):  # 获取完整可驱逐的大小
        return 0  # 默认返回0

    def swa_evictable_size(self):  # 获取滑动窗口可驱逐的大小
        return 0  # 默认返回0

    def protected_size(self):  # 获取受保护的大小
        return 0  # 默认返回0

    def full_protected_size(self):  # 获取完整受保护的大小
        return 0  # 默认返回0

    def swa_protected_size(self):  # 获取滑动窗口受保护的大小
        return 0  # 默认返回0

    def total_size(self):  # 获取总大小
        raise NotImplementedError()  # 默认抛出未实现异常

    def pretty_print(self):  # 格式化打印缓存信息
        raise NotImplementedError()  # 默认抛出未实现异常

    def init_load_back(  # 初始化从主机到设备的KV缓存加载
        self,
        params: InitLoadBackParams,  # 初始化加载回写参数
    ) -> Tuple[torch.Tensor, Any]:  # 返回张量和任意类型的元组
        """
        Preparing KV cache loading from host to device.
        """
        # 准备从主机到设备的KV缓存加载
        raise NotImplementedError()  # 默认抛出未实现异常

    def ready_to_load_host_cache(self) -> Any:  # 通知缓存控制器开始加载主机KV缓存
        """
        Notify the cache controller to start the KV cache loading
        """
        # 通知缓存控制器开始KV缓存加载
        raise NotImplementedError()  # 默认抛出未实现异常

    def flush_write_through_acks(self) -> None:  # 刷新直写确认，释放已完成直写的节点的锁引用
        """Release lock_ref on radix-tree nodes whose write-through has completed.
        # 释放已完成直写的基数树节点的锁引用

        Lightweight operation that only processes finished write acks.
        # 轻量级操作，仅处理已完成的写确认
        No-op for caches without hierarchical write-through support.
        # 对于不支持分层直写的缓存，此操作为空操作
        """
        pass  # 默认空操作

    def check_hicache_events(self) -> Any:  # 检查HiCache相关事件
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        # 检查HiCache相关活动以更新基数树，并在需要时跨TP工作器同步
        raise NotImplementedError()  # 默认抛出未实现异常

    def take_events(self):  # 获取缓存事件
        return []  # 默认返回空列表

    def supports_swa(self) -> bool:  # 检查是否支持滑动窗口注意力
        return False  # 默认不支持

    def supports_mamba(self) -> bool:  # 检查是否支持Mamba模型
        return False  # 默认不支持

    def supports_streaming_session(self) -> bool:  # 检查是否支持流式会话
        return False  # 默认不支持

    def release_session(self, session_id: str) -> None:  # 释放指定会话
        pass  # 默认空操作

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:  # 获取会话持有的token数量
        return 0  # 默认返回0

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:  # 获取会话持有的完整token数量
        return 0  # 默认返回0

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:  # 获取会话持有的滑动窗口token数量
        return 0  # 默认返回0

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:  # 获取会话持有的请求数量
        return 0  # 默认返回0

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:  # 获取会话持有的Mamba槽位数量
        return 0  # 默认返回0

    def is_chunk_cache(self) -> bool:  # 检查是否为分块缓存
        return False  # 默认不是分块缓存

    def is_tree_cache(self) -> bool:  # 检查是否为树缓存
        return not self.is_chunk_cache()  # 如果不是分块缓存则为树缓存

    def available_and_evictable_str(self) -> str:  # 生成可用和可驱逐token的字符串描述
        available_size = self.token_to_kv_pool_allocator.available_size()  # 获取KV池分配器的可用大小
        evictable_size = self.evictable_size()  # 获取可驱逐大小
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"  # 返回格式化的字符串