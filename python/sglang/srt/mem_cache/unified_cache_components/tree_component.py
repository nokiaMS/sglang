# 树组件抽象基类与公共数据结构
# 本文件定义了统一基数缓存系统的核心抽象和公共数据结构。
# 包含：ComponentType组件类型枚举、ComponentData节点组件数据类、
# EvictLayer驱逐层级枚举、CacheTransferPhase缓存传输阶段枚举、
# LRURefreshPhase LRU刷新阶段枚举，以及TreeComponent抽象基类。
# TreeComponent定义了所有树节点组件的通用接口，包括匹配验证、
# 驱逐、锁定/解锁、插入、HiCache传输等方法的抽象声明。

from __future__ import annotations  # 启用延迟类型注解求值

import dataclasses  # 导入数据类模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from enum import Enum, IntFlag  # 导入枚举和整数标志枚举
from typing import TYPE_CHECKING, Any, Callable, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch张量库
from numpy import float64  # 导入NumPy的float64类型

from sglang.srt.mem_cache.base_prefix_cache import (  # 从基础前缀缓存模块导入数据结构
    DecLockRefParams,  # 减锁引用参数
    EvictParams,  # 驱逐参数
    IncLockRefResult,  # 增锁引用结果
    InsertParams,  # 插入参数
    InsertResult,  # 插入结果
    MatchPrefixParams,  # 匹配前缀参数
    MatchResult,  # 匹配结果
)
from sglang.srt.mem_cache.hicache_storage import PoolTransfer  # 导入HiCache存储传输描述

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.managers.schedule_batch import Req  # 请求类型
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # 缓存初始化参数
    from sglang.srt.mem_cache.unified_radix_cache import (  # 统一基数缓存
        UnifiedRadixCache,  # 统一基数缓存类
        UnifiedTreeNode,  # 统一树节点类
    )


class ComponentType(int, Enum):  # 组件类型整数枚举，可直接用于节点列表/元组的索引
    """Integer enum so that per-node list/tuple storage can be indexed directly."""  # 整数枚举，使每节点列表/元组存储可直接索引

    FULL = 0  # 全注意力类型，索引0
    SWA = 1  # 滑动窗口注意力类型，索引1
    MAMBA = 2  # Mamba状态类型，索引2

    def __str__(self) -> str:  # keep human-readable logging  # 保持人类可读的日志输出
        return self.name.lower()  # 返回小写名称

    @property
    def is_full(self) -> bool:  # 判断是否为全注意力类型
        return self == ComponentType.FULL  # 比较是否为FULL

    @property
    def is_swa(self) -> bool:  # 判断是否为滑动窗口注意力类型
        return self == ComponentType.SWA  # 比较是否为SWA

    @property
    def is_mamba(self) -> bool:  # 判断是否为Mamba类型
        return self == ComponentType.MAMBA  # 比较是否为MAMBA


BASE_COMPONENT_TYPE = ComponentType.FULL  # 基础组件类型为FULL
_NUM_COMPONENT_TYPES = len(ComponentType)  # 组件类型总数

_LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)  # 最后访问时间计数器（浮点），初始1.0
_COMPONENT_UUID_COUNTER = 1  # 组件UUID计数器，初始1


@dataclasses.dataclass  # 数据类装饰器
class ComponentData:  # 组件数据类，存储每个节点上每种组件类型的数据
    value: Optional[torch.Tensor] = None  # 设备端KV索引值
    lock_ref: int = 0  # 设备端锁引用计数
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)  # 元数据字典（如SWA UUID）
    host_value: Optional[torch.Tensor] = None  # 主机端KV索引值
    host_lock_ref: int = 0  # 主机端锁引用计数


class EvictLayer(IntFlag):  # 驱逐层级整数标志枚举，可通过位运算组合
    """Which storage layer(s) to evict.  Combinable via bitwise OR."""  # 驱逐哪些存储层。可通过位或运算组合

    DEVICE = 1  # 设备层，值1
    HOST = 2  # 主机层，值2
    ALL = DEVICE | HOST  # 所有层，值3


class CacheTransferPhase(str, Enum):  # 缓存传输阶段字符串枚举

    BACKUP_HOST = "backup_host"  # D→H  # 设备到主机备份
    LOAD_BACK = "load_back"  # H→D  # 主机到设备加载
    BACKUP_STORAGE = "backup_storage"  # H→Storage  # 主机到外部存储备份
    PREFETCH = "prefetch"  # Storage→H  # 外部存储到主机预取


class LRURefreshPhase(str, Enum):  # LRU刷新阶段字符串枚举

    WALKDOWN = "walkdown"  # touching a node while walking through the tree  # 遍历树时触碰节点
    MATCH_END = "match_end"  # end of a successful prefix match  # 前缀匹配成功结束
    INSERT_END = "insert_end"  # after a new/updated leaf is committed  # 新/更新叶子提交后


def get_and_increase_time_counter() -> float64:  # 获取并递增最后访问时间计数器
    global _LAST_ACCESS_TIME_COUNTER_FLOAT  # 声明全局变量
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT  # 保存当前值
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0  # 递增计数器
    return ret  # 返回旧值


def next_component_uuid() -> int:  # 生成下一个组件UUID
    global _COMPONENT_UUID_COUNTER  # 声明全局变量
    _COMPONENT_UUID_COUNTER += 1  # 递增计数器
    return _COMPONENT_UUID_COUNTER  # 返回新UUID


class TreeComponent(ABC):  # 树组件抽象基类
    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):  # 初始化树组件
        self.cache = cache  # 保存缓存引用

    # Subclasses MUST set this as a class attribute (not @property)  # 子类必须将其设置为类属性（而非@property）
    component_type: ComponentType  # 组件类型

    def node_has_component_data(  # 判断节点是否有指定层级的组件数据
        self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE  # 节点和目标层级
    ) -> bool:  # 返回是否有数据
        cd = node.component_data[self.component_type]  # 获取组件数据
        if target is EvictLayer.DEVICE:  # 如果是设备层
            return cd.value is not None  # 返回设备值是否存在
        return cd.host_value is not None  # 返回主机值是否存在

    def value_len(self, node: UnifiedTreeNode) -> int:  # 获取节点组件值的长度
        value = node.component_data[self.component_type].value  # 获取设备值
        return len(value) if value is not None else 0  # 有值返回长度，否则返回0

    def refresh_lru(  # 刷新LRU位置
        self,
        phase: LRURefreshPhase,  # LRU刷新阶段
        node: UnifiedTreeNode,  # 目标节点
        root_node: UnifiedTreeNode,  # 根节点
    ) -> None:
        ct = self.component_type  # 获取组件类型
        match phase:  # 根据阶段处理
            case LRURefreshPhase.WALKDOWN:  # 下行遍历阶段
                if node.component_data[ct].value is None:  # 如果设备值不存在
                    return  # 无需刷新
                self.cache.lru_lists[ct].reset_node_mru(node)  # 重置为MRU位置
            case LRURefreshPhase.MATCH_END:  # 匹配结束阶段
                self.cache.lru_lists[ct].reset_node_and_parents_mru(  # 重置节点及其祖先为MRU
                    node, root_node, self.node_has_component_data
                )
            case LRURefreshPhase.INSERT_END:  # 插入结束阶段
                # WALKDOWN already refreshed every node on the insert path  # WALKDOWN已刷新插入路径上的每个节点
                # (including the new leaf), so there is nothing more to do.  # （包括新叶子），因此无需额外操作
                return  # 不执行任何操作
            case _:  # 未知阶段
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")  # 抛出异常

    @abstractmethod
    def create_match_validator(  # 创建匹配验证器（抽象方法）
        self, match_device_only: bool = False  # 是否仅匹配设备数据
    ) -> Callable[[UnifiedTreeNode], bool]:  # 返回节点验证谓词
        """Return a per-match stateful predicate that decides whether a node  # 返回每次匹配的有状态谓词，判断节点
        is a valid match boundary for this component.  # 是否为此组件的有效匹配边界
        Called once per match_prefix; the returned closure may carry state.  # 每次match_prefix调用一次；返回的闭包可携带状态
        When match_device_only is true, host-backed nodes must not be accepted  # 当match_device_only为True时，主机备份节点不能作为
        as valid match boundaries.  # 有效匹配边界
        - Full: returns True if the node has full component data.  # - Full：如果节点有全注意力组件数据则返回True
        - SWA: tracks accumulated length since last gap; returns True only  # - SWA：跟踪自上次间隔以来的累积长度；仅当
          when the contiguous window reaches swa_sliding_window_size.  # 连续窗口达到swa_sliding_window_size时返回True
        - Mamba: returns True iff the node has mamba component data."""  # - Mamba：当且仅当节点有Mamba组件数据时返回True
        ...

    def finalize_match_result(  # 最终处理匹配结果（默认透传）
        self,
        result: MatchResult,  # 原始匹配结果
        params: MatchPrefixParams,  # 匹配前缀参数
        value_chunks: list[torch.Tensor],  # 值张量分块列表
        best_value_len: int,  # 最佳值长度
    ) -> MatchResult:  # 返回匹配结果
        """Post-process the match result after prefix matching completes.  # 前缀匹配完成后的后处理
        - Full & SWA: pass through unchanged.  # - Full和SWA：透传不变
        - Mamba: performs copy-on-write — allocates a new mamba slot, copies  # - Mamba：执行写时复制——分配新Mamba槽位，将
          the matched node's mamba state into the request pool, and records  # 匹配节点的Mamba状态复制到请求池，并记录
          branching_seqlen in result."""  # 结果中的branching_seqlen
        return result  # 默认直接返回结果

    def update_component_on_insert_overlap(  # 插入重叠时更新组件数据（默认不消耗）
        self,
        node: UnifiedTreeNode,  # 目标节点
        prefix_len: int,  # 前缀长度
        total_prefix_len: int,  # 总前缀长度
        value_slice: torch.Tensor,  # 值切片
        params: InsertParams,  # 插入参数
    ) -> int:  # 返回组件消耗的起始索引
        """Called per-node when an insert's key overlaps an existing node.  # 当插入的键与现有节点重叠时，对每个节点调用
        Returns the index within value_slice from which this component  # 返回此组件从value_slice中消耗（接管底层KV池槽位）的
        consumed (took ownership of) the underlying KV pool slots.  # 起始索引
        Returns prefix_len if nothing was consumed (default).  # 如果未消耗任何内容则返回prefix_len（默认）
        _insert_helper uses this to free only the non-consumed duplicate  # _insert_helper使用此值仅释放未消耗的重复
        portion: value_slice[dup_start:consumed_from]."""  # 部分：value_slice[dup_start:consumed_from]
        return prefix_len  # 默认不消耗

    def should_skip_leaf_creation(  # 判断是否应跳过新叶子创建（默认不跳过）
        self, total_prefix_len: int, key_len: int, params: InsertParams  # 总前缀长度、键长度和插入参数
    ) -> bool:
        """Return True to veto leaf creation when the entire new leaf would  # 当整个新叶子对此组件
        be a tombstone for this component."""  # 将是墓碑时返回True以否决叶子创建
        return False  # 默认不跳过

    def recover_after_unevict(  # 反驱逐后恢复组件数据（默认无操作）
        self,
        node: UnifiedTreeNode,  # 目标节点
        prefix_len: int,  # 前缀长度
        total_prefix_len: int,  # 总前缀长度
        params: InsertParams,  # 插入参数
    ) -> None:
        """Called after _unevict_node_on_insert restores the base (Full) value  # 在_unevict_node_on_insert恢复基础（Full）值后调用
        on an evicted node. Aux components (e.g. SWA) override this to rebuild  # 辅助组件（如SWA）重写此方法从新分配的基础值
        their own data from the freshly assigned base value when their entry  # 重建自身数据，当它们的条目
        is still tombstoned. Default no-op."""  # 仍处于墓碑状态时。默认无操作
        return None  # 默认无操作

    def commit_insert_component_data(  # 提交插入后的组件数据（默认无操作）
        self,
        node: UnifiedTreeNode,  # 目标节点
        is_new_leaf: bool,  # 是否为新叶子节点
        params: InsertParams,  # 插入参数
        result: InsertResult,  # 插入结果
    ) -> None:
        """Finalize component data on the target (leaf) node after the insert  # 插入遍历完成后，在目标（叶子）节点上最终确定组件数据
        walk completes. Called once per insert.  # 每次插入调用一次
        - Full: no-op (full data is handled by _add_new_node).  # - Full：无操作（全注意力数据由_add_new_node处理）
        - SWA: for new leaves, checks whether the node straddles the SWA  # - SWA：对于新叶子，检查节点是否跨越SWA
          eviction boundary (swa_evicted_seqlen). If so, splits the node  # 驱逐边界（swa_evicted_seqlen）。如果是，通过
          via _split_node — the parent becomes a tombstone (no SWA) and the  # _split_node分裂节点——父节点变为墓碑（无SWA），
          child (the deeper portion) receives SWA data. If the entire node  # 子节点（较深部分）接收SWA数据。如果整个节点
          is within the window, sets SWA directly. If entirely outside,  # 在窗口内，直接设置SWA。如果完全在窗口外，
          leaves SWA as None (tombstone).  # SWA保持None（墓碑）
        - Mamba: sets the mamba component value from params, inserts into  # - Mamba：从params设置Mamba组件值，插入
          mamba LRU list, and increments evictable size. If the node already  # Mamba LRU列表，并增加可驱逐大小。如果节点已有
          has mamba data, resets its LRU position instead."""  # Mamba数据，则重置其LRU位置
        pass  # 默认无操作

    @abstractmethod
    def redistribute_on_node_split(  # 节点分裂时重分布组件数据（抽象方法）
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode  # 新父节点和子节点
    ):
        """Redistribute component data between new_parent and child when a  # 当节点分裂时，在new_parent和child之间重分布组件数据
        node is split. new_parent is the newly created prefix node.  # new_parent是新创建的前缀节点
        - Full: copies child's lock_ref to new_parent.  # - Full：将child的lock_ref复制到new_parent
        - SWA: slices (or clones) the swa value for new_parent, copies  # - SWA：切片（或克隆）new_parent的swa值，复制
          lock_ref and component_uuid metadata, then syncs child's swa  # lock_ref和component_uuid元数据，然后同步child的swa
          value with its (now-trimmed) full_value.  # 值与其（已裁剪的）full_value
        - Mamba: sets new_parent's mamba value to None and lock_ref to 0  # - Mamba：设置new_parent的mamba值为None，lock_ref为0
          (mamba data stays on the original leaf, not on prefix nodes)."""  # （Mamba数据保留在原始叶子节点上，不在前缀节点上）
        ...

    @abstractmethod
    def evict_component(  # 驱逐节点上的组件KV资源（抽象方法）
        self,
        node: UnifiedTreeNode,  # 待驱逐节点
        target: EvictLayer = EvictLayer.DEVICE,  # 驱逐目标层级
    ) -> tuple[int, int]:  # 返回(设备释放数量, 主机释放数量)
        """Free this component's KV resources on a node being evicted.  # 释放被驱逐节点上此组件的KV资源

        *target* controls which layer(s) to evict:  # *target*控制驱逐哪些层
          - DEVICE: free device memory and tombstone (value = None).  # - DEVICE：释放设备内存并置墓碑（value = None）
                    Host data is untouched.  # 主机数据不受影响
          - HOST:   free host memory (host_value = None).  # - HOST：释放主机内存（host_value = None）
                    Device data is untouched.  # 设备数据不受影响
          - ALL:    free both device and host memory.  # - ALL：释放设备和主机内存
                    No tombstone — caller will delete the node.  # 无墓碑——调用者将删除节点

        Returns (device_freed, host_freed) token counts."""  # 返回(设备释放数, 主机释放数)token计数
        ...

    def eviction_priority(self, is_leaf: bool) -> int:  # 获取驱逐优先级，越高驱逐越晚
        """Eviction priority on this node type. Higher = evicted later.  # 此节点类型的驱逐优先级。越高驱逐越晚
        When a component is evicted, all other components with equal or  # 当一个组件被驱逐时，同一节点上所有优先级
        lower priority on the same node are also cascade-evicted.  # 相等或更低的组件也会被级联驱逐

        Leaf: all components equal (0) — evicting any cascades to all,  # 叶子：所有组件优先级相等（0）——驱逐任一会级联到所有
        because the node will be deleted.  # 因为节点将被删除

        Internal: full=2 > swa=1 > mamba=0.  # 内部节点：full=2 > swa=1 > mamba=0
        Why swa > mamba: SWA data on internal nodes is *path data* —  # 为什么swa > mamba：内部节点上的SWA数据是*路径数据*
        the sliding window needs continuous SWA coverage along the path  # 滑动窗口需要沿路径从根到匹配边界的连续SWA覆盖
        from root to the match boundary. E.g. A->B->C->D->E where C  # 例如A->B->C->D->E，其中C
        and E both have mamba and the window covers C->E: if C's mamba  # 和E都有Mamba且窗口覆盖C->E：如果C的Mamba
        is evicted, C's SWA must stay so E remains reachable.  # 被驱逐，C的SWA必须保留以便E仍可达
        Mamba data, by contrast, is only meaningful at the match  # 相比之下，Mamba数据仅在匹配
        boundary node; on internal nodes it  # 边界节点上有意义；在内部节点上它
        contributes nothing to the path. So SWA is more valuable to  # 对路径无贡献。因此SWA更有价值
        keep and should be evicted later.  # 应该更晚驱逐

        Cascade consequences:  # 级联后果：
        - Mamba evict internal: no cascade.  # - Mamba驱逐内部节点：无级联
        - SWA evict internal: cascades to Mamba. SWA gone -> SWA  # - SWA驱逐内部节点：级联到Mamba。SWA消失 -> SWA
          validator fails -> mamba data is useless (match requires all  # 验证器失败 -> Mamba数据无用（匹配要求所有
          validators to pass).  # 验证器通过）
        - Full evict internal: cascades to SWA + Mamba."""  # - Full驱逐内部节点：级联到SWA + Mamba
        return 0  # 默认优先级0

    @abstractmethod
    def drive_eviction(  # 驱动驱逐流程（抽象方法）
        self, params: EvictParams, tracker: dict[ComponentType, int]  # 驱逐参数和追踪器
    ) -> None:
        """Drive eviction from this component's LRU list.  # 从此组件的LRU列表驱动驱逐
        Each component extracts its own request from params, walks its own  # 每个组件从params提取自己的请求，遍历自己的LRU，
        LRU, evicts, and calls cache._cascade_evict for priority cascade.  # 执行驱逐，并调用cache._cascade_evict进行优先级级联
        Updates the shared tracker with freed amounts for all components.  # 用所有组件的释放量更新共享追踪器
        - Full: walks leaf LRU, evicts full then cascades entire leaf.  # - Full：遍历叶子LRU，驱逐full后级联整个叶子
        - Mamba: walks full LRU; tombstones internal nodes (with cascade  # - Mamba：遍历full LRU；墓碑内部节点（级联到
          to equal-priority components like swa), cascades leaves to all."""  # 同优先级组件如swa），级联叶子到所有组件
        ...

    @abstractmethod
    def acquire_component_lock(  # 获取组件锁引用（抽象方法）
        self,
        node: UnifiedTreeNode,  # 目标节点
        result: IncLockRefResult,  # 增锁结果
        lock_host: bool = False,  # 是否锁定主机数据
    ) -> IncLockRefResult:  # 返回更新后的增锁结果
        """Increment component lock refs, protecting nodes from  # 递增组件锁引用，保护节点不被驱逐
        eviction. Updates evictable → protected size on first lock.  # 首次加锁时更新可驱逐→受保护大小
        - Full: path-lock — walks from node up to root, incrementing  # - Full：路径锁——从节点向上走到根，递增
          lock_ref on every ancestor.  # 每个祖先的lock_ref
        - SWA: path-lock — walks upward collecting swa values until the  # - SWA：路径锁——向上收集swa值直到
          sliding window is filled; records a component_uuid at the  # 滑动窗口填满；在
          boundary for release_component_lock to know where to stop.  # 边界记录component_uuid以便release_component_lock知道在哪里停止
        - Mamba: single-node lock — only increments lock_ref on the  # - Mamba：单节点锁——仅递增
          node itself (mamba state is per-leaf, not per-path).  # 节点本身的lock_ref（Mamba状态是每叶子而非每路径）

        When ``lock_host`` is True, the lock applies to host-side state:  # 当``lock_host``为True时，锁应用于主机端状态：
        - Full: single-node host lock.  # - Full：单节点主机锁
        - SWA: host window-lock with a dedicated host UUID boundary.  # - SWA：带专用主机UUID边界的主机窗口锁
        - Mamba: single-node host lock with host LRU detach."""  # - Mamba：带主机LRU分离的单节点主机锁
        ...

    @abstractmethod
    def release_component_lock(  # 释放组件锁引用（抽象方法）
        self,
        node: UnifiedTreeNode,  # 目标节点
        params: Optional[DecLockRefParams],  # 减锁参数
        lock_host: bool = False,  # 是否释放主机锁
    ) -> None:
        """Decrement component lock refs, un-protecting nodes.  # 递减组件锁引用，取消对节点的保护
        Updates protected → evictable size when lock_ref drops to 0.  # 当lock_ref降为0时更新受保护→可驱逐大小
        - Full: path-unlock — walks from node up to root, decrementing  # - Full：路径解锁——从节点向上走到根，递减
          lock_ref on every ancestor.  # 每个祖先的lock_ref
        - SWA: path-unlock — walks upward, stopping at the node whose  # - SWA：路径解锁——向上遍历，在component_uuid
          component_uuid matches the one recorded during acquire.  # 与获取时记录的匹配的节点处停止
        - Mamba: single-node unlock — only decrements lock_ref on the  # - Mamba：单节点解锁——仅递减
          node itself.  # 节点本身的lock_ref

        When ``lock_host`` is True, the inverse host-side semantics apply."""  # 当``lock_host``为True时，应用反向主机端语义
        ...

    def prepare_for_caching_req(  # 在请求缓存前准备组件特定数据（默认无操作）
        self,
        req: Req,  # 请求对象
        insert_params: InsertParams,  # 插入参数
        token_ids_len: int,  # token ID长度
        is_finished: bool,  # 请求是否已完成
    ) -> Optional[int]:  # 返回有效缓存长度或None
        """Prepare component-specific data before insert, fill component  # 插入前准备组件特定数据，填充insert_params中的
        fields in insert_params, return effective cache_len.  # 组件字段，返回有效cache_len
        Return None for no truncation opinion (use full length);  # 返回None表示无截断意见（使用完整长度）
        return int >= 0 for effective cache length.  # 返回int >= 0表示有效缓存长度
        - Full: no-op, returns None.  # - Full：无操作，返回None
        - SWA: sets insert_params.swa_evicted_seqlen on finished; returns None.  # - SWA：完成时设置insert_params.swa_evicted_seqlen；返回None
        - Mamba: prepares mamba_value (finished from ping-pong buffer,  # - Mamba：准备mamba_value（完成时从乒乓缓冲，
          unfinished fork from req); returns mamba_last_track_seqlen."""  # 未完成时从req分叉）；返回mamba_last_track_seqlen
        return None  # 默认不截断

    def cleanup_after_caching_req(  # 请求缓存后的组件资源清理（默认无操作）
        self,
        req: Req,  # 请求对象
        is_finished: bool,  # 请求是否已完成
        insert_result: Optional[InsertResult] = None,  # 插入结果
        insert_params: Optional[InsertParams] = None,  # 插入参数
    ) -> None:
        """Post-cache cleanup for component-specific resources.  # 缓存后的组件特定资源清理

        ``is_finished`` — whether the request has finished generation.  # ``is_finished``——请求是否已完成生成
        True means the request is complete and its resources can be released;  # True表示请求完成，其资源可被释放
        ``insert_result`` is None when insert was skipped (cache disabled  # ``insert_result``为None当插入被跳过（缓存禁用
        or effective_cache_len <= 0); treat as "no insert happened".  # 或effective_cache_len <= 0时）；视为"未发生插入"
        ``insert_params`` is None only on the disabled path; on early-return  # ``insert_params``仅在禁用路径上为None；在提前返回
        paths it is still provided so components can free their resources."""  # 路径上仍会提供以便组件释放资源
        pass  # 默认无操作

    # ---- HiCache Hooks ----  # ---- HiCache钩子函数 ----

    def build_hicache_transfers(  # 构建HiCache传输描述符（默认返回None）
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw  # 节点、传输阶段和额外参数
    ) -> Optional[list[PoolTransfer]]:  # 返回传输描述列表或None
        """Build transfer descriptors for this component in the given phase.  # 为此组件在给定阶段构建传输描述符
        Returns None if the component has nothing to transfer."""  # 如果组件无内容需传输则返回None
        return None  # 默认无传输

    def commit_hicache_transfer(  # 提交HiCache传输结果（默认无操作）
        self,
        node: UnifiedTreeNode,  # 目标节点
        phase: CacheTransferPhase,  # 传输阶段
        transfers: list[PoolTransfer] = (),  # 传输描述列表
        **kw,  # 额外参数
    ) -> None:
        """Post-transfer bookkeeping: store host indices, update LRU, etc."""  # 传输后记账：存储主机索引，更新LRU等
        pass  # 默认无操作

    def drive_host_eviction(  # 驱动主机层驱逐（默认无操作）
        self, num_tokens: int, tracker: dict[ComponentType, int]  # 需释放token数和追踪器
    ) -> None:
        """Evict from this component's host-side resources.  # 从此组件的主机端资源驱逐
        Called by HostPoolGroup when the host pool is full.  # 主机池满时由HostPoolGroup调用
        Default no-op for components without host storage."""  # 对于无主机存储的组件默认无操作
        pass  # 默认无操作
