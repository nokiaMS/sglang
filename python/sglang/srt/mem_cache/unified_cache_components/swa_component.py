# 滑动窗口注意力(SWA)缓存组件
# 本文件实现了SWAComponent类，继承自TreeComponent，用于管理滑动窗口注意力的KV缓存数据。
# 每个SWA节点存储翻译后的SWA池索引作为其组件值，独立于同一树节点的全注意力索引。
# 当SWA数据从内部节点被驱逐时，节点变为墓碑状态——SWA组件值变为None，
# 而全注意力值保持不变。主要功能包括：滑动窗口匹配验证、设备值恢复、
# 插入重叠处理、节点分裂时数据重分布、设备/主机层驱逐、SWA窗口锁定等。

from __future__ import annotations  # 启用延迟类型注解求值

from typing import TYPE_CHECKING, Callable, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch张量库

from sglang.srt.mem_cache.base_prefix_cache import (  # 从基础前缀缓存模块导入数据结构
    DecLockRefParams,  # 减锁引用参数
    EvictParams,  # 驱逐参数
    IncLockRefResult,  # 增锁引用结果
    InsertParams,  # 插入参数
    InsertResult,  # 插入结果
    MatchPrefixParams,  # 匹配前缀参数
    MatchResult,  # 匹配结果
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # 导入HiCache存储池名和传输描述
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # 从树组件模块导入基类和枚举
    BASE_COMPONENT_TYPE,  # 基础组件类型（FULL）
    CacheTransferPhase,  # 缓存传输阶段枚举
    ComponentType,  # 组件类型枚举
    EvictLayer,  # 驱逐层级枚举
    LRURefreshPhase,  # LRU刷新阶段枚举
    TreeComponent,  # 树组件抽象基类
    next_component_uuid,  # 生成下一个组件UUID
)

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.managers.schedule_batch import Req  # 请求类型
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # 缓存初始化参数
    from sglang.srt.mem_cache.unified_radix_cache import (  # 统一基数缓存
        UnifiedRadixCache,  # 统一基数缓存类
        UnifiedTreeNode,  # 统一树节点类
    )


class SWAComponent(TreeComponent):  # SWA组件类，继承自TreeComponent
    """Sliding window attention component.  # 滑动窗口注意力组件

    Each SWA node stores translated SWA pool indices as its component  # 每个SWA节点存储翻译后的SWA池索引作为其组件
    value, independent of the full attention indices on the same tree node.  # 值，独立于同一树节点上的全注意力索引
    When SWA data is evicted from an internal node the node is tombstoned  # 当SWA数据从内部节点被驱逐时，节点变为墓碑状态
    — its SWA component value becomes None while the full attention  # ——其SWA组件值变为None，而全注意力
    value stays intact.  # 值保持不变
    """

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):  # 初始化SWAComponent实例
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator  # 导入SWA内存池分配器

        assert isinstance(  # 断言分配器为SWA类型
            cache.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(cache.token_to_kv_pool_allocator)}"  # SWAComponent需要SWATokenToKVPoolAllocator
        super().__init__(cache, params)  # 调用父类初始化
        self.sliding_window_size = params.sliding_window_size  # 保存滑动窗口大小
        # HiCache state: set to host SWA pool when HiCache enabled  # HiCache状态：启用HiCache时设置为主机SWA池
        self._swa_kv_pool_host = None  # 主机端SWA KV池，默认为None

    component_type = ComponentType.SWA  # 组件类型为SWA

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:  # 将全注意力索引翻译为SWA索引
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(  # 调用分配器的翻译方法
            full_indices
        )

    def refresh_lru(  # 刷新SWA组件的LRU位置
        self,
        phase: LRURefreshPhase,  # LRU刷新阶段
        node: UnifiedTreeNode,  # 目标节点
        root_node: UnifiedTreeNode,  # 根节点
    ) -> None:
        match phase:  # 根据阶段处理
            case LRURefreshPhase.WALKDOWN:  # 下行遍历阶段
                # Walk-down would refresh every visited ancestor to MRU,  # 下行遍历会将每个访问的祖先刷新为MRU
                # but most are outside the active sliding window and must  # 但大多数不在活跃滑动窗口内，必须
                # stay evictable. Window-bounded refresh runs at  # 保持可驱逐。窗口范围刷新在
                # MATCH_END / INSERT_END instead.  # MATCH_END / INSERT_END阶段执行
                return  # 不执行任何操作
            case LRURefreshPhase.MATCH_END | LRURefreshPhase.INSERT_END:  # 匹配结束或插入结束阶段
                self.cache.lru_lists[  # 刷新节点及其窗口内祖先的MRU位置
                    self.component_type
                ].reset_node_and_window_ancestors_mru(
                    node,  # 起始节点
                    root_node,  # 根节点
                    self.sliding_window_size + self.cache.page_size,  # 窗口大小加上页大小
                    self.node_has_component_data,  # 节点是否有组件数据的判断函数
                )
            case _:  # 未知阶段
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")  # 抛出异常

    def _restore_device_value(self, node: UnifiedTreeNode, value: torch.Tensor) -> None:  # 恢复节点的SWA设备值
        ct = self.component_type  # 获取组件类型
        node.component_data[ct].value = value  # 设置设备值
        host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU列表
        if host_lru.in_list(node):  # 如果在主机LRU中
            host_lru.remove_node(node)  # 从主机LRU中移除
        self.cache.lru_lists[ct].insert_mru(node)  # 插入设备LRU的MRU端
        self.cache.component_evictable_size_[ct] += len(value)  # 增加可驱逐大小

    def create_match_validator(  # 创建匹配验证器，基于滑动窗口累积长度判断有效性
        self, match_device_only: bool = False  # 是否仅匹配设备数据
    ) -> Callable[[UnifiedTreeNode], bool]:  # 返回节点验证谓词
        sliding_window_size = self.sliding_window_size  # 获取滑动窗口大小
        ct = self.component_type  # 获取组件类型
        state = {"len": float("inf")}  # 状态字典，跟踪累积长度

        def validator(node: UnifiedTreeNode) -> bool:  # 验证器函数
            cd = node.component_data[ct]  # 获取组件数据
            # HiCache: a host-only tombstone is a valid match boundary too  # HiCache：仅主机端的墓碑也是有效的匹配边界
            # — load_back will restore SWA from host before use.  # ——load_back将在使用前从主机恢复SWA
            if cd.value is None and (match_device_only or cd.host_value is None):  # 设备值空且（仅设备模式或主机值也空）
                state["len"] = 0  # 重置累积长度
                return False  # 无效匹配
            state["len"] += len(node.key)  # 累加键长度
            return state["len"] >= sliding_window_size  # 累积长度达到窗口大小则为有效

        return validator  # 返回验证器

    def finalize_match_result(  # 最终处理匹配结果，计算SWA的主机命中
        self,
        result: MatchResult,  # 原始匹配结果
        params: MatchPrefixParams,  # 匹配前缀参数
        value_chunks: list[torch.Tensor],  # 值张量分块列表
        best_value_len: int,  # 最佳值长度
    ) -> MatchResult:  # 返回更新后的匹配结果
        ct = self.component_type  # 获取组件类型
        n_swa = 0  # SWA累积长度初始化
        node = result.best_match_node  # 从最佳匹配节点开始
        root = self.cache.root_node  # 获取根节点
        while node is not root and n_swa < self.sliding_window_size:  # 遍历直到根节点或窗口填满
            cd = node.component_data[ct]  # 获取组件数据
            if cd.value is None and cd.host_value is not None:  # 设备值空但主机值存在
                # TODO(ispobock): refactor host_hit_length usage  # 待办：重构host_hit_length的使用方式
                return result._replace(host_hit_length=max(result.host_hit_length, 1))  # 确保主机命中长度至少为1
            if cd.value is not None:  # 设备值存在
                n_swa += len(cd.value)  # 累加设备值长度
            elif cd.host_value is not None:  # 主机值存在
                n_swa += len(cd.host_value)  # 累加主机值长度
            else:  # 两者都不存在
                break  # 中断遍历
            node = node.parent  # 向上移动
        return result  # 返回结果

    def update_component_on_insert_overlap(  # 插入重叠时更新SWA组件数据，处理墓碑恢复
        self,
        node: UnifiedTreeNode,  # 目标节点
        prefix_len: int,  # 前缀长度
        total_prefix_len: int,  # 总前缀长度
        value_slice: torch.Tensor,  # 值切片
        params: InsertParams,  # 插入参数
    ) -> int:  # 返回组件消耗的值切片起始索引
        if params.prev_prefix_len >= total_prefix_len + prefix_len:  # 如果之前的前缀长度已覆盖此段
            return prefix_len  # 不消耗任何内容

        is_tombstone = node.component_data[self.component_type].value is None  # 判断是否为墓碑状态
        if not is_tombstone:  # 如果不是墓碑
            return prefix_len  # 不消耗任何内容

        swa_evicted_seqlen = params.swa_evicted_seqlen  # 获取SWA驱逐序列长度
        assert (  # 断言墓碑节点的锁引用为0
            node.component_data[self.component_type].lock_ref == 0
        ), f"tombstone {self.component_type} lock_ref should be 0, node {node.id}"  # 墓碑组件锁引用应为0
        assert (  # 断言驱逐序列长度页对齐
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{self.component_type}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"  # swa_evicted_seqlen必须页对齐

        if swa_evicted_seqlen <= total_prefix_len:  # 分支1：整个值切片在SWA窗口内——完全恢复
            # Branch 1: entire value_slice is within SWA window — recover  # 分支1：整个value_slice在SWA窗口内——恢复
            self.cache.token_to_kv_pool_allocator.free(  # 释放旧的Full KV索引
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice.clone()  # 用新值替换
            swa_value = self._translate_full_to_swa(  # 将新Full值翻译为SWA值
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)  # 恢复SWA设备值
            return 0  # 消耗从0开始
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:  # 分支2：部分值切片在窗口内——部分恢复
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover  # 分支2：value_slice[start_idx:]在SWA窗口内——部分恢复
            start_idx = swa_evicted_seqlen - total_prefix_len  # 计算SWA窗口内起始索引
            self.cache.token_to_kv_pool_allocator.free(  # 释放窗口外部分的Full KV索引
                node.component_data[BASE_COMPONENT_TYPE].value[start_idx:]
            )
            self.cache._split_node(node.key, node, start_idx)  # 在窗口边界分裂节点
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice[  # 用窗口内的新值替换
                start_idx:
            ].clone()
            swa_value = self._translate_full_to_swa(  # 将新Full值翻译为SWA值
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)  # 恢复SWA设备值
            return start_idx  # 返回消耗起始索引
        else:  # 分支3：整个值切片在SWA窗口外——不消耗
            # Branch 3: entire value_slice is outside SWA window — not consumed  # 分支3：整个value_slice在SWA窗口外——未被消耗
            return prefix_len  # 不消耗任何内容

    def should_skip_leaf_creation(  # 判断是否应跳过新叶子节点创建（当整个叶子都在SWA窗口外时）
        self, total_prefix_len: int, key_len: int, params: InsertParams  # 总前缀长度、键长度和插入参数
    ) -> bool:
        return params.swa_evicted_seqlen >= total_prefix_len + key_len  # 驱逐序列长度覆盖整个新叶子则跳过

    def recover_after_unevict(  # 反驱逐后恢复SWA组件数据（从恢复的Full值重建SWA）
        self,
        node: UnifiedTreeNode,  # 目标节点
        prefix_len: int,  # 前缀长度
        total_prefix_len: int,  # 总前缀长度
        params: InsertParams,  # 插入参数
    ) -> None:
        # _unevict_node_on_insert already wrote the request's fresh KV slice  # _unevict_node_on_insert已写入请求的新KV切片
        # into the base value. We just need to rebuild SWA from that slice for  # 到基础值中。我们只需从该切片重建SWA
        # the in-window portion. There is no old SWA slot to free here.  # 对窗口内部分。此处无需释放旧的SWA槽位
        ct = self.component_type  # 获取组件类型
        if node.component_data[ct].value is not None:  # 如果SWA值已存在
            return  # 无需恢复
        assert (  # 断言墓碑节点的锁引用为0
            node.component_data[ct].lock_ref == 0
        ), f"tombstone {ct} lock_ref should be 0 on unevict, node {node.id}"  # 反驱逐时墓碑组件锁引用应为0
        swa_evicted_seqlen = params.swa_evicted_seqlen  # 获取SWA驱逐序列长度
        assert (  # 断言驱逐序列长度页对齐
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"  # swa_evicted_seqlen必须页对齐

        full_value = node.component_data[BASE_COMPONENT_TYPE].value  # 获取Full KV值
        if swa_evicted_seqlen <= total_prefix_len:  # 整个值在窗口内
            swa_value = self._translate_full_to_swa(full_value)  # 直接翻译
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:  # 部分在窗口内
            start_idx = swa_evicted_seqlen - total_prefix_len  # 计算起始索引
            self.cache._split_node(node.key, node, start_idx)  # 在窗口边界分裂节点
            full_value = node.component_data[BASE_COMPONENT_TYPE].value  # 获取分裂后的Full值
            swa_value = self._translate_full_to_swa(full_value)  # 翻译
        else:  # 全部在窗口外
            return  # 无需恢复
        self._restore_device_value(node, swa_value)  # 恢复SWA设备值

    def commit_insert_component_data(  # 提交插入后的SWA组件数据
        self,
        node: UnifiedTreeNode,  # 目标节点
        is_new_leaf: bool,  # 是否为新叶子节点
        params: InsertParams,  # 插入参数
        result: InsertResult,  # 插入结果
    ) -> None:
        if not is_new_leaf:  # 如果不是新叶子节点
            return  # 直接返回

        node_start = result.prefix_len  # 节点起始位置
        split_pos = params.swa_evicted_seqlen - node_start  # 计算SWA窗口边界位置

        if split_pos <= 0:  # 整个节点在SWA窗口内
            swa_value = self._translate_full_to_swa(  # 将Full值翻译为SWA值
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value  # 设置SWA值
            self.cache.lru_lists[self.component_type].insert_mru(node)  # 插入设备LRU
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)  # 增加可驱逐大小
        elif split_pos < len(node.key):  # 节点跨越SWA窗口边界
            # Node straddles the SWA eviction boundary  # 节点跨越SWA驱逐边界
            # Split into parent (tombstone, no SWA) and child (with SWA)  # 分裂为父节点（墓碑，无SWA）和子节点（有SWA）
            # After _split_node, `node` becomes the child  # _split_node后，`node`变为子节点
            self.cache._split_node(node.key, node, split_pos)  # 在边界处分裂节点
            swa_value = self._translate_full_to_swa(  # 将Full值翻译为SWA值
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value  # 设置SWA值
            self.cache.lru_lists[self.component_type].insert_mru(node)  # 插入设备LRU
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)  # 增加可驱逐大小
        else:  # 整个叶子在SWA窗口外
            # Entire leaf is outside the SWA window — left as a tombstone.  # 整个叶子在SWA窗口外——保持墓碑状态
            return  # 不设置SWA值

        self._maybe_split_leaf_for_swa_lock(node)  # 可能需要为SWA锁定分裂叶子

    def _maybe_split_leaf_for_swa_lock(self, leaf: UnifiedTreeNode) -> None:  # 可能将新SWA叶子上限截断为一个页对齐窗口大小
        """Cap a fresh SWA leaf at one page-aligned window so locking it pins  # 将新SWA叶子上限截断为一个页对齐窗口，使锁定时仅固定
        only one window of SWA pool, not the whole (long chunked-prefill) leaf.  # 一个窗口的SWA池，而非整个（长分块预填充）叶子
        """
        ct = self.component_type  # 获取组件类型
        cd = leaf.component_data[ct]  # 获取组件数据
        if leaf is self.cache.root_node or cd.value is None or cd.lock_ref > 0:  # 根节点、无值或有锁
            return  # 不分裂

        page_size = self.cache.page_size  # 获取页大小
        # Smallest page-aligned size that still covers the sliding window.  # 覆盖滑动窗口的最小页对齐大小
        tail_size = (self.sliding_window_size + page_size - 1) // page_size * page_size  # 向上取整到页大小
        leaf_len = len(leaf.key)  # 获取叶子键长度
        if leaf_len <= tail_size:  # 如果叶子长度不超过窗口大小
            return  # 不分裂
        split_at = leaf_len - tail_size  # 计算分裂位置
        if page_size > 1 and (split_at % page_size != 0 or leaf_len % page_size != 0):  # 页对齐检查
            return  # 不对齐则不分裂

        self.cache._split_node(leaf.key, leaf, split_at)  # 在计算位置分裂叶子

    def redistribute_on_node_split(  # 节点分裂时重分布SWA组件数据
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode  # 新父节点和子节点
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[  # 新父节点继承子节点的锁引用
            self.component_type
        ].lock_ref

        child_swa_value = child.component_data[self.component_type].value  # 获取子节点的SWA设备值
        if child_swa_value is not None:  # 如果子节点有SWA设备值
            split_len = len(new_parent.key)  # 获取分裂长度
            new_parent.component_data[self.component_type].value = child_swa_value[  # 新父节点获得前半部分
                :split_len
            ].clone()
            child.component_data[self.component_type].value = child_swa_value[  # 子节点保留后半部分
                split_len:
            ].clone()
        else:  # 子节点无SWA设备值
            new_parent.component_data[self.component_type].value = None  # 新父节点也无SWA设备值

        child_swa_host_value = child.component_data[self.component_type].host_value  # 获取子节点的主机值
        if child_swa_host_value is not None:  # 如果子节点有主机值
            split_len = len(new_parent.key)  # 获取分裂长度
            new_parent.component_data[self.component_type].host_value = (  # 新父节点获得前半部分主机值
                child_swa_host_value[:split_len].clone()
            )
            child.component_data[self.component_type].host_value = child_swa_host_value[  # 子节点保留后半部分主机值
                split_len:
            ].clone()
            host_lru = self.cache.host_lru_lists[self.component_type]  # 获取主机LRU列表
            if new_parent.component_data[self.component_type].value is None:  # 如果新父节点无设备值
                host_lru.insert_mru(new_parent)  # 插入主机LRU
            if child.component_data[  # 如果子节点无设备值且不在主机LRU中
                self.component_type
            ].value is None and not host_lru.in_list(child):
                host_lru.insert_mru(child)  # 插入主机LRU

        # parent inherits the swa_uuid from child for swa lock ref  # 父节点从子节点继承swa_uuid用于SWA锁引用
        new_parent.component_data[self.component_type].metadata["uuid"] = (  # 新父节点获得UUID
            child.component_data[self.component_type].metadata.get("uuid")
        )
        child.component_data[self.component_type].metadata.pop("uuid", None)  # 子节点移除UUID

    def evict_component(  # 驱逐节点上的SWA组件数据
        self,
        node: UnifiedTreeNode,  # 待驱逐节点
        target: EvictLayer = EvictLayer.DEVICE,  # 驱逐目标层级
    ) -> tuple[int, int]:  # 返回(设备释放数量, 主机释放数量)
        ct = self.component_type  # 获取组件类型
        cd = node.component_data[ct]  # 获取组件数据
        freed = 0  # 设备释放计数初始化
        host_freed = 0  # 主机释放计数初始化

        # Device layer  # 设备层驱逐
        if EvictLayer.DEVICE in target and cd.value is not None:  # 如果目标是设备层且设备值存在
            # Pass full indices to free_swa so slots with no SWA pair are  # 传递全注意力索引给free_swa，使无SWA配对的槽位
            # skipped. Freeing swa_value directly would double free those  # 被跳过。直接释放swa_value会导致双重释放
            # entries since they all map to the same sentinel slot.  # 因为它们都映射到同一个哨兵槽位
            self.cache.token_to_kv_pool_allocator.free_swa(  # 使用free_swa释放SWA槽位
                node.component_data[BASE_COMPONENT_TYPE].value  # 传递Full KV索引
            )
            freed = len(cd.value)  # 记录释放的token数
            self.cache.component_evictable_size_[ct] -= freed  # 减少可驱逐大小
            cd.value = None  # 置空设备值

        # Host layer  # 主机层驱逐
        host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU列表
        if EvictLayer.HOST in target and cd.host_value is not None:  # 如果目标是主机层且主机值存在
            host_freed = len(cd.host_value)  # 记录主机释放的token数
            if self._swa_kv_pool_host is not None:  # 如果主机SWA KV池存在
                self._swa_kv_pool_host.free(cd.host_value)  # 释放主机SWA KV池中的值
            cd.host_value = None  # 置空主机值
            if host_lru.in_list(node):  # 如果在主机LRU中
                host_lru.remove_node(node)  # 从主机LRU中移除

        # After device tombstone: if host_value remains, move into host LRU  # 设备驱逐后：如果主机值仍存在，移入主机LRU
        if (
            target is EvictLayer.DEVICE  # 仅设备层驱逐
            and cd.value is None  # 设备值已清空
            and cd.host_value is not None  # 主机值仍存在
        ):
            if not host_lru.in_list(node):  # 如果不在主机LRU中
                host_lru.insert_mru(node)  # 插入主机LRU

        return freed, host_freed  # 返回设备和主机释放数量

    def eviction_priority(self, is_leaf: bool) -> int:  # 获取驱逐优先级，叶子0，内部1
        return 0 if is_leaf else 1  # 叶子0，内部1（SWA内部优先级高于Mamba但低于Full）

    def drive_eviction(  # 驱动SWA组件的驱逐流程，使用LRU策略
        self, params: EvictParams, tracker: dict[ComponentType, int]  # 驱逐参数和追踪器
    ) -> None:
        request = params.swa_num_tokens  # 需要释放的SWA token数
        ct = self.component_type  # 获取组件类型
        lru = self.cache.lru_lists[ct]  # 获取设备LRU列表
        x = lru.get_lru_no_lock()  # 获取LRU端节点
        while tracker[ct] < request and x is not None and lru.in_list(x):  # 循环直到满足需求
            assert x.component_data[ct].value is not None  # 断言SWA设备值存在
            if x in self.cache.evictable_device_leaves:  # 如果是可驱逐的设备叶子
                # D-leaf: atomic eviction of all components  # 设备叶子：原子驱逐所有组件
                x_next = lru.get_prev_no_lock(x)  # 获取前驱节点
                self.cache._evict_device_leaf(x, tracker)  # 驱逐设备叶子
                if not lru.in_list(x_next):  # 如果前驱已不在LRU中
                    x_next = lru.get_lru_no_lock()  # 重新获取LRU端
                x = x_next  # 移动到下一个
            else:  # 内部节点
                # Internal: tombstone SWA + cascade  # 内部节点：墓碑SWA + 级联驱逐
                x_next = lru.get_prev_no_lock(x)  # 获取前驱节点
                self.cache._evict_component_and_detach_lru(  # 驱逐组件并从LRU分离
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)  # 级联驱逐低优先级组件
                x = x_next  # 移动到下一个

    def acquire_component_lock(  # 获取SWA组件锁引用，沿路径向上锁定直到滑动窗口填满
        self,
        node: UnifiedTreeNode,  # 目标节点
        result: IncLockRefResult,  # 增锁结果
        lock_host: bool = False,  # 是否锁定主机数据
    ) -> IncLockRefResult:  # 返回更新后的增锁结果
        ct = self.component_type  # 获取组件类型
        root = self.cache.root_node  # 获取根节点
        sliding_window_size = self.sliding_window_size  # 获取滑动窗口大小
        swa_lock_size = 0  # SWA锁定大小初始化
        swa_uuid_for_lock = None  # SWA锁UUID初始化

        # Tombstoned nodes (cd.value is None) have no SWA chunk to protect  # 墓碑节点（cd.value为None）无SWA块需保护
        # skip them and keep walking up. This path is hit when HiCache  # 跳过它们并继续向上。此路径在HiCache
        # backs up a FULL present internal node whose SWA was already evicted.  # 备份了SWA已被驱逐的FULL存在内部节点时触发
        cur = node  # 从当前节点开始
        while cur != root and swa_lock_size < sliding_window_size:  # 向上遍历直到根或窗口填满
            comp = cur.component_data[ct]  # 获取组件数据
            if comp.value is None:  # 如果是墓碑
                result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)  # 记录跳过的节点
                cur = cur.parent  # 继续向上
                continue  # 继续
            if comp.lock_ref == 0:  # 如果当前无锁
                key_len = len(cur.key)  # 获取键长度
                self.cache.component_evictable_size_[ct] -= key_len  # 减少可驱逐大小
                self.cache.component_protected_size_[ct] += key_len  # 增加受保护大小
            comp.lock_ref += 1  # 增加锁引用
            swa_lock_size += len(cur.key)  # 累加锁定大小
            if swa_lock_size >= sliding_window_size:  # 如果达到窗口大小
                if comp.metadata.get("uuid") is None:  # 如果无UUID
                    comp.metadata["uuid"] = next_component_uuid()  # 生成新UUID
                swa_uuid_for_lock = comp.metadata["uuid"]  # 记录锁边界UUID
            cur = cur.parent  # 向上移动

        result.swa_uuid_for_lock = swa_uuid_for_lock  # 设置SWA锁UUID
        return result  # 返回结果

    def release_component_lock(  # 释放SWA组件锁引用，沿路径向上直到UUID边界
        self,
        node: UnifiedTreeNode,  # 目标节点
        params: Optional[DecLockRefParams],  # 减锁参数
        lock_host: bool = False,  # 是否释放主机锁
    ) -> None:
        ct = self.component_type  # 获取组件类型
        root = self.cache.root_node  # 获取根节点
        swa_uuid_for_lock = params.swa_uuid_for_lock if params else None  # 获取SWA锁UUID
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()  # 获取跳过的节点ID
        dec_swa = True  # 是否继续减少锁引用

        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.  # skip_lock_node_ids中的节点在获取锁时是墓碑
        cur = node  # 从当前节点开始
        while cur != root and dec_swa:  # 向上遍历直到根或到达UUID边界
            comp = cur.component_data[ct]  # 获取组件数据
            if cur.id in skip_lock_node_ids:  # 如果在跳过集合中
                cur = cur.parent  # 继续向上
                continue  # 继续
            if comp.lock_ref == 0:  # 如果锁引用为0
                cur = cur.parent  # 继续向上
                continue  # 继续
            if comp.lock_ref == 1:  # 如果将降为0
                key_len = len(cur.key)  # 获取键长度
                self.cache.component_evictable_size_[ct] += key_len  # 增加可驱逐大小
                self.cache.component_protected_size_[ct] -= key_len  # 减少受保护大小
            comp.lock_ref -= 1  # 减少锁引用
            if swa_uuid_for_lock and comp.metadata.get("uuid") == swa_uuid_for_lock:  # 如果到达UUID边界
                dec_swa = False  # 停止减少
            cur = cur.parent  # 向上移动

    def prepare_for_caching_req(  # 在请求缓存前准备SWA组件数据
        self,
        req: Req,  # 请求对象
        insert_params: InsertParams,  # 插入参数
        token_ids_len: int,  # token ID长度
        is_finished: bool,  # 请求是否已完成
    ) -> Optional[int]:  # 返回有效缓存长度或None
        if is_finished:  # 如果请求已完成
            insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen  # 设置SWA驱逐序列长度
        return None  # 不限制缓存长度

    # ---- HiCache Hooks ----  # ---- HiCache钩子函数 ----

    def build_hicache_transfers(  # 构建HiCache传输描述符
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw  # 节点、传输阶段和额外参数
    ) -> Optional[list[PoolTransfer]]:  # 返回传输描述列表或None
        ct = self.component_type  # 获取组件类型

        if phase == CacheTransferPhase.BACKUP_HOST:  # 备份到主机阶段
            cd = node.component_data[ct]  # 获取组件数据
            if cd.value is None:  # 如果设备值不存在
                return None  # 无需备份
            # cd.value already holds SWA-pool indices (translated at insert time).  # cd.value已持有SWA池索引（在插入时翻译）
            # Host pool indexing wants int64.  # 主机池索引需要int64类型
            return [  # 返回SWA备份传输
                PoolTransfer(
                    name=PoolName.SWA,  # SWA池
                    device_indices=cd.value.to(torch.int64),  # 转为int64的设备端索引
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:  # 从主机加载回设备阶段
            # `node` is best_match_node; the SWA validator guarantees every  # `node`是最佳匹配节点；SWA验证器保证
            # ancestor within `sliding_window_size` has value or host_value.  # sliding_window_size内的每个祖先有value或host_value
            n_swa = 0  # SWA累积长度初始化
            backed_up: list[torch.Tensor] = []  # 已备份的主机值列表
            nodes: list = []  # 对应节点列表
            cur = node  # 从当前节点开始
            while cur is not self.cache.root_node and n_swa < self.sliding_window_size:  # 遍历直到根或窗口填满
                cd = cur.component_data[ct]  # 获取组件数据
                assert cd.host_value is not None or cd.value is not None  # 断言主机或设备值存在
                if cd.value is not None:  # 如果设备值存在
                    # device exists, skip it  # 设备数据已存在，跳过
                    n_swa += len(cd.value)  # 累加设备值长度
                else:  # 仅主机值存在
                    # host only, collect it  # 仅主机数据，收集它
                    backed_up.append(cd.host_value)  # 收集主机值
                    nodes.append(cur)  # 收集节点
                    n_swa += len(cd.host_value)  # 累加主机值长度
                cur = cur.parent  # 向上移动

            if not backed_up:  # 如果无需加载
                return None  # 返回None

            backed_up.reverse()  # 反转，按从根到叶顺序排列
            nodes.reverse()  # 反转节点列表

            return [  # 返回加载传输
                PoolTransfer(
                    name=PoolName.SWA,  # SWA池
                    host_indices=torch.cat(backed_up),  # 拼接所有主机值
                    device_indices=None,  # 设备端索引暂为None
                    nodes_to_load=nodes,  # 需要加载的节点列表
                )
            ]

        return None  # 其他阶段不处理

    def commit_hicache_transfer(  # 提交HiCache传输结果
        self,
        node: UnifiedTreeNode,  # 目标节点
        phase: CacheTransferPhase,  # 传输阶段
        transfers: list[PoolTransfer] = (),  # 传输描述列表
        **kw,  # 额外参数
    ) -> None:
        ct = self.component_type  # 获取组件类型

        if phase == CacheTransferPhase.BACKUP_HOST:  # 备份到主机阶段
            if transfers and transfers[0].host_indices is not None:  # 如果传输成功且有主机索引
                cd = node.component_data[ct]  # 获取组件数据
                if cd.host_value is None:  # 如果主机值尚不存在
                    cd.host_value = transfers[0].host_indices.clone()  # 保存主机索引
            return  # 返回

        if phase == CacheTransferPhase.LOAD_BACK:  # 从主机加载回设备阶段
            assert transfers and transfers[0].device_indices is not None  # 断言设备索引存在
            xfer = transfers[0]  # 获取第一个传输
            device_indices = xfer.device_indices  # 获取设备索引
            allocator = self.cache.token_to_kv_pool_allocator  # 获取KV池分配器

            offset = 0  # 偏移量初始化
            for n in xfer.nodes_to_load or []:  # 遍历需要加载的节点
                cd_n = n.component_data[ct]  # 获取SWA组件数据
                cd_full_n = n.component_data[BASE_COMPONENT_TYPE]  # 获取Full组件数据
                n_tokens = len(cd_n.host_value)  # 获取token数
                swa_chunk = device_indices[offset : offset + n_tokens].clone()  # 切出对应的设备索引
                self._restore_device_value(n, swa_chunk)  # 恢复SWA设备值
                assert cd_full_n.value is not None and len(cd_full_n.value) == n_tokens  # 断言Full值存在且长度匹配
                # rebuild the mapping for the loaded SWA chunk  # 重建加载的SWA块的映射关系
                allocator.set_full_to_swa_mapping(cd_full_n.value, swa_chunk)  # 设置Full到SWA的映射
                offset += n_tokens  # 更新偏移量
            assert offset == len(xfer.host_indices)  # 断言偏移量等于主机索引长度
            return  # 返回

    def drive_host_eviction(  # 驱动SWA主机层驱逐
        self, num_tokens: int, tracker: dict[ComponentType, int]  # 需释放token数和追踪器
    ) -> None:
        """Evict SWA host resources.  # 驱逐SWA主机资源
        Internal nodes: private tombstone (free SWA host only).  # 内部节点：私有墓碑（仅释放SWA主机数据）
        Host leaves: atomic eviction via _evict_host_leaf."""  # 主机叶子：通过_evict_host_leaf原子驱逐
        ct = self.component_type  # 获取组件类型
        host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU列表
        x = host_lru.get_lru_no_lock()  # 获取LRU端节点
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):  # 循环直到满足需求
            x_next = host_lru.get_prev_no_lock(x)  # 获取前驱节点
            cd = x.component_data[ct]  # 获取组件数据
            if x in self.cache.evictable_host_leaves:  # 如果是可驱逐的主机叶子
                self.cache._evict_host_leaf(x, tracker)  # 驱逐主机叶子
            else:  # 内部节点
                assert cd.host_value is not None  # 断言主机值存在
                self.cache._evict_component_and_detach_lru(  # 驱逐组件并从LRU分离
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)  # 级联驱逐
            x = x_next  # 移动到下一个节点
