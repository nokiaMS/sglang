# 全注意力(Full Attention)KV缓存组件
# 本文件实现了FullComponent类，继承自TreeComponent，用于管理树节点的全注意力KV缓存数据。
# 主要功能包括：前缀匹配验证、匹配结果后处理、节点分裂时数据重分布、
# 设备/主机层驱逐、驱逐优先级、驱逐驱动、组件锁定/解锁、HiCache备份与加载等。

from __future__ import annotations  # 启用延迟类型注解求值

import heapq  # 导入堆队列模块，用于驱逐优先级排序
from typing import TYPE_CHECKING, Callable, Optional  # 导入类型检查相关工具

import torch  # 导入PyTorch张量库

from sglang.srt.mem_cache.base_prefix_cache import (  # 从基础前缀缓存模块导入数据结构
    DecLockRefParams,  # 减锁引用参数
    EvictParams,  # 驱逐参数
    IncLockRefResult,  # 增锁引用结果
    MatchPrefixParams,  # 匹配前缀参数
    MatchResult,  # 匹配结果
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # 导入HiCache存储池名和传输描述
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # 从树组件模块导入基类和枚举
    CacheTransferPhase,  # 缓存传输阶段枚举
    ComponentType,  # 组件类型枚举
    EvictLayer,  # 驱逐层级枚举
    TreeComponent,  # 树组件抽象基类
)

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,  # 统一树节点类型
    )


class FullComponent(TreeComponent):  # 全注意力组件类，继承自TreeComponent
    component_type = ComponentType.FULL  # 组件类型为FULL

    def __init__(self, cache, params):  # 初始化FullComponent实例
        super().__init__(cache, params)  # 调用父类初始化
        allocator = cache.token_to_kv_pool_allocator  # 获取token到KV池的分配器
        # When SWA is present, only free full-attention KV here;  # 当SWA存在时，此处仅释放全注意力KV
        # SWA KV will be freed by cascade via SWAComponent.evict_component.  # SWA KV将通过SWAComponent.evict_component级联释放
        if ComponentType.SWA in cache.tree_components:  # 如果缓存中存在SWA组件
            self._free_full = allocator.full_attn_allocator.free  # 使用全注意力分配器的free方法
        else:  # 否则
            self._free_full = allocator.free  # 使用通用分配器的free方法
        # HiCache state: set to host KV pool when HiCache enabled  # HiCache状态：启用HiCache时设置为主机KV池
        self._full_kv_pool_host = None  # 主机端全注意力KV池，默认为None

    def create_match_validator(  # 创建匹配验证器，返回用于判断节点是否为有效匹配边界的谓词函数
        self, match_device_only: bool = False  # 是否仅匹配设备上的数据
    ) -> Callable[[UnifiedTreeNode], bool]:  # 返回节点验证谓词
        if match_device_only:  # 如果仅匹配设备数据
            return (  # 返回仅检查设备数据的验证器
                lambda node: node.component_data[self.component_type].value is not None  # 节点的全注意力组件数据非空则为有效
            )

        # HiCache: evicted + backuped nodes are valid match boundaries.  # HiCache：已驱逐但已备份的节点也是有效的匹配边界
        return lambda node: (  # 返回检查设备数据或备份状态的验证器
            node.component_data[self.component_type].value is not None or node.backuped  # 设备数据非空或节点已备份则为有效
        )

    def finalize_match_result(  # 最终处理匹配结果，计算Full KV主机命中长度
        self,
        result: MatchResult,  # 原始匹配结果
        params: MatchPrefixParams,  # 匹配前缀参数
        value_chunks: list[torch.Tensor],  # 值张量分块列表
        best_value_len: int,  # 最佳值长度
    ) -> MatchResult:  # 返回更新后的匹配结果
        # Compute Full KV host hit length: walk from last_host_node up to  # 计算Full KV主机命中长度：从last_host_node向上遍历
        # last_device_node, summing host_value lengths of evicted nodes.  # 到last_device_node，累加已驱逐节点的主机值长度
        ct = self.component_type  # 获取组件类型
        kv_host_hit = 0  # 主机KV命中计数初始化
        node = result.best_match_node  # 从最佳匹配节点开始
        root_node = self.cache.root_node  # 获取根节点
        while node is not result.last_device_node and node is not root_node:  # 遍历直到设备节点或根节点
            full_host = node.component_data[ct].host_value  # 获取节点的主机端全注意力值
            if full_host is not None:  # 如果主机值存在
                kv_host_hit += len(full_host)  # 累加主机值长度
            node = node.parent  # 向上移动到父节点
        if kv_host_hit > 0:  # 如果有主机命中
            return result._replace(  # 返回更新后的匹配结果
                host_hit_length=max(result.host_hit_length, kv_host_hit)  # 取现有和新的主机命中长度的最大值
            )
        return result  # 无主机命中，返回原结果

    def redistribute_on_node_split(  # 节点分裂时重分布组件数据
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode  # 新父节点和子节点
    ):
        ct = self.component_type  # 获取组件类型
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref  # 新父节点继承子节点的锁引用
        child_cd = child.component_data[ct]  # 获取子节点的组件数据
        split_len = len(new_parent.key)  # 分裂位置长度
        if child_cd.value is not None:  # 如果子节点有设备值
            new_parent.component_data[ct].value = child_cd.value[:split_len].clone()  # 新父节点获得前半部分设备值
            child_cd.value = child_cd.value[split_len:].clone()  # 子节点保留后半部分设备值
        if child_cd.host_value is not None:  # 如果子节点有主机值
            new_parent.component_data[ct].host_value = child_cd.host_value[  # 新父节点获得前半部分主机值
                :split_len
            ].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()  # 子节点保留后半部分主机值

    def evict_component(  # 驱逐节点上的组件数据，释放设备/主机资源
        self,
        node: UnifiedTreeNode,  # 待驱逐节点
        target: EvictLayer = EvictLayer.DEVICE,  # 驱逐目标层级
    ) -> tuple[int, int]:  # 返回(设备释放数量, 主机释放数量)
        cd = node.component_data[self.component_type]  # 获取节点的组件数据
        freed = 0  # 设备释放计数初始化
        host_freed = 0  # 主机释放计数初始化

        # Device layer  # 设备层驱逐
        if EvictLayer.DEVICE in target and cd.value is not None:  # 如果目标是设备层且设备值存在
            self._free_full(cd.value)  # 释放全注意力设备值
            freed = len(cd.value)  # 记录释放的token数
            self.cache.component_evictable_size_[self.component_type] -= freed  # 减少可驱逐大小
            # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)  # 注意：cd.value = None 延迟到_cascade_evict执行（Full作为触发器）
            # because SWA's free_swa still needs to read Full.value.  # 因为SWA的free_swa仍需读取Full.value
            # cd.value = None  # 不在此处置空，延迟到级联驱逐

        # Host layer  # 主机层驱逐
        if EvictLayer.HOST in target and cd.host_value is not None:  # 如果目标是主机层且主机值存在
            host_freed = len(cd.host_value)  # 记录主机释放的token数
            if self._full_kv_pool_host is not None:  # 如果主机KV池存在
                self._full_kv_pool_host.free(cd.host_value)  # 释放主机KV池中的值
            cd.host_value = None  # 置空主机值
        return freed, host_freed  # 返回设备和主机释放数量

    def eviction_priority(self, is_leaf: bool) -> int:  # 获取驱逐优先级，叶子节点为0，内部节点为2
        return 0 if is_leaf else 2  # 叶子0，内部2（内部节点优先级更高，驱逐更晚）

    def drive_eviction(  # 驱动全注意力组件的驱逐流程，使用堆按优先级驱逐叶子节点
        self, params: EvictParams, tracker: dict[ComponentType, int]  # 驱逐参数和各组件释放量追踪器
    ) -> None:
        request = params.num_tokens  # 需要释放的token数
        heap = [  # 构建优先级堆
            (self.cache.eviction_strategy.get_priority(n), n)  # (优先级, 节点)元组
            for n in self.cache.evictable_device_leaves  # 遍历可驱逐的设备叶子节点
        ]
        heapq.heapify(heap)  # 将列表转化为堆
        ct = self.component_type  # 获取组件类型
        while tracker[ct] < request and heap:  # 循环直到释放量满足需求或堆为空
            _, x = heapq.heappop(heap)  # 弹出最低优先级节点
            if x not in self.cache.evictable_device_leaves:  # 如果节点已不可驱逐
                continue  # 跳过
            self.cache._evict_device_leaf(x, tracker)  # 驱逐该设备叶子节点
            if x.parent is not None and x.parent in self.cache.evictable_device_leaves:  # 如果父节点变为可驱逐叶子
                heapq.heappush(  # 将父节点加入堆
                    heap,
                    (self.cache.eviction_strategy.get_priority(x.parent), x.parent),  # (父节点优先级, 父节点)
                )

    def drive_host_eviction(  # 驱动主机层驱逐，释放KV主机池空间
        self, num_tokens: int, tracker: dict[ComponentType, int]  # 需释放token数和追踪器
    ) -> None:
        """Evict host leaves to free KV host pool space."""  # 驱逐主机叶子节点以释放KV主机池空间
        heap = [  # 构建优先级堆
            (self.cache.eviction_strategy.get_priority(n), n)  # (优先级, 节点)元组
            for n in self.cache.evictable_host_leaves  # 遍历可驱逐的主机叶子节点
        ]
        heapq.heapify(heap)  # 将列表转化为堆
        ct = self.component_type  # 获取组件类型
        while tracker[ct] < num_tokens and heap:  # 循环直到释放量满足需求或堆为空
            _, x = heapq.heappop(heap)  # 弹出最低优先级节点
            if x not in self.cache.evictable_host_leaves:  # 如果节点已不可驱逐
                continue  # 跳过
            self.cache._evict_host_leaf(x, tracker)  # 驱逐该主机叶子节点
            if x.parent is not None and x.parent in self.cache.evictable_host_leaves:  # 如果父节点变为主机叶子
                heapq.heappush(  # 将父节点加入堆
                    heap,
                    (self.cache.eviction_strategy.get_priority(x.parent), x.parent),  # (父节点优先级, 父节点)
                )

    def acquire_component_lock(  # 获取组件锁引用，保护节点不被驱逐
        self,
        node: UnifiedTreeNode,  # 目标节点
        result: IncLockRefResult,  # 增锁结果
        lock_host: bool = False,  # 是否锁定主机数据
    ) -> IncLockRefResult:  # 返回更新后的增锁结果
        ct = self.component_type  # 获取组件类型

        # Only the last host node needs to be protected.  # 仅最后一个主机节点需要保护
        if lock_host:  # 如果锁定主机数据
            cd = node.component_data[ct]  # 获取组件数据
            if cd.host_value is None:  # 如果主机值不存在
                return result  # 直接返回，无需锁定
            cd.host_lock_ref += 1  # 增加主机锁引用
            self.cache._update_evictable_leaf_sets(node)  # 更新可驱逐叶子集合
            return result  # 返回结果

        root = self.cache.root_node  # 获取根节点
        cur = node  # 从当前节点开始

        # Skip the bottom evicted segment  # 跳过底部已驱逐段
        while cur is not root and cur.component_data[ct].value is None:  # 向上遍历直到根节点或找到设备值
            result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)  # 记录跳过锁定的节点ID
            cur = cur.parent  # 移动到父节点

        # Lock the device-on segment up to root  # 锁定设备在线段直到根节点
        delta = 0  # 从可驱逐转为受保护的大小增量
        while cur is not root:  # 向上遍历到根节点
            cd = cur.component_data[ct]  # 获取组件数据
            assert (  # 断言设备值存在
                cd.value is not None
            ), f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"  # FULL不变性被破坏：设备在线段上方存在已驱逐祖先
            if cd.lock_ref == 0:  # 如果当前无锁
                key_len = len(cd.value)  # 获取值长度
                self.cache.component_evictable_size_[ct] -= key_len  # 减少可驱逐大小
                self.cache.component_protected_size_[ct] += key_len  # 增加受保护大小
                delta += key_len  # 累加增量
            cd.lock_ref += 1  # 增加锁引用
            self.cache.evictable_device_leaves.discard(cur)  # 从可驱逐叶子集合中移除
            cur = cur.parent  # 移动到父节点
        result.delta = delta  # 设置增量
        return result  # 返回结果

    def release_component_lock(  # 释放组件锁引用，允许节点被驱逐
        self,
        node: UnifiedTreeNode,  # 目标节点
        params: Optional[DecLockRefParams],  # 减锁参数
        lock_host: bool = False,  # 是否释放主机锁
    ) -> None:
        ct = self.component_type  # 获取组件类型
        if lock_host:  # 如果释放主机锁
            cd = node.component_data[ct]  # 获取组件数据
            if cd.host_value is None or cd.host_lock_ref == 0:  # 如果主机值不存在或无主机锁
                return  # 直接返回
            cd.host_lock_ref -= 1  # 减少主机锁引用
            self.cache._update_evictable_leaf_sets(node)  # 更新可驱逐叶子集合
            return  # 返回

        root = self.cache.root_node  # 获取根节点
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()  # 获取跳过锁定的节点ID集合
        cur = node  # 从当前节点开始
        while cur != root:  # 向上遍历到根节点
            if cur.id in skip_lock_node_ids:  # 如果节点在跳过集合中
                cur = cur.parent  # 跳过，移到父节点
                continue  # 继续
            cd = cur.component_data[ct]  # 获取组件数据
            assert cd.value is not None  # 断言设备值存在
            assert cd.lock_ref > 0  # 断言锁引用大于0

            if cd.lock_ref == 1:  # 如果锁引用将降为0
                key_len = len(cd.value)  # 获取值长度
                self.cache.component_evictable_size_[ct] += key_len  # 增加可驱逐大小
                self.cache.component_protected_size_[ct] -= key_len  # 减少受保护大小
            cd.lock_ref -= 1  # 减少锁引用
            if cd.lock_ref == 0:  # 如果锁引用降为0
                self.cache._update_evictable_leaf_sets(cur)  # 更新可驱逐叶子集合
            cur = cur.parent  # 移动到父节点

    # ---- HiCache Hooks ----  # ---- HiCache钩子函数 ----

    def build_hicache_transfers(  # 构建HiCache传输描述符，用于设备与主机之间的数据传输
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw  # 节点、传输阶段和额外参数
    ) -> Optional[list[PoolTransfer]]:  # 返回传输描述列表或None
        ct = self.component_type  # 获取组件类型

        if phase == CacheTransferPhase.BACKUP_HOST:  # 如果是备份到主机阶段
            # Full KV backup is handled by the main flow  # Full KV备份由主流程处理
            # (write_backup → cache_controller.write on host_value directly).  # (write_backup → cache_controller.write直接写入host_value)
            # No extra PoolTransfer needed.  # 不需要额外的PoolTransfer
            return None  # 返回None

        if phase == CacheTransferPhase.LOAD_BACK:  # 如果是从主机加载回设备阶段
            # `node` is best_match_node. FULL device evict only from leaves,  # `node`是最佳匹配节点。FULL仅从叶子节点驱逐设备数据
            # so once we hit a device-on node, everything above is also device-on  # 因此一旦遇到设备在线节点，其上方所有节点也在线
            backed_up: list[torch.Tensor] = []  # 已备份的主机值列表
            nodes: list = []  # 对应节点列表
            cur = node  # 从当前节点开始遍历
            while cur.evicted:  # 遍历已驱逐节点
                cd = cur.component_data[ct]  # 获取组件数据
                assert cd.host_value is not None  # 断言主机值存在
                backed_up.append(cd.host_value)  # 收集主机值
                nodes.append(cur)  # 收集节点
                cur = cur.parent  # 向上移动
            backed_up.reverse()  # 反转列表，使其按从根到叶顺序排列
            nodes.reverse()  # 反转节点列表
            return [  # 返回传输描述
                PoolTransfer(
                    name=PoolName.KV,  # KV池
                    host_indices=(  # 主机端索引
                        torch.cat(backed_up)  # 拼接所有主机值
                        if backed_up  # 如果有备份值
                        else torch.empty((0,), dtype=torch.int64, device="cpu")  # 否则返回空张量
                    ),
                    device_indices=None,  # 设备端索引暂为None，加载时填充
                    nodes_to_load=nodes,  # 需要加载的节点列表
                )
            ]

        return None  # 其他阶段不处理

    def commit_hicache_transfer(  # 提交HiCache传输结果，更新组件数据和LRU状态
        self,
        node: UnifiedTreeNode,  # 目标节点
        phase: CacheTransferPhase,  # 传输阶段
        transfers: list[PoolTransfer] = (),  # 传输描述列表
        **kw,  # 额外参数
    ) -> None:
        ct = self.component_type  # 获取组件类型

        if phase == CacheTransferPhase.BACKUP_HOST:  # 如果是备份到主机阶段
            if transfers and transfers[0].host_indices is not None:  # 如果传输成功且有主机索引
                node.component_data[ct].host_value = transfers[0].host_indices.clone()  # 保存主机索引到组件数据

        elif phase == CacheTransferPhase.LOAD_BACK:  # 如果是从主机加载回设备阶段
            if not transfers or transfers[0].device_indices is None:  # 如果无传输或设备索引为空
                self.cache._update_evictable_leaf_sets(node)  # 仅更新叶子集合
                return  # 返回

            xfer = transfers[0]  # 获取第一个传输描述
            device_indices = xfer.device_indices  # 获取设备端索引
            offset = 0  # 偏移量初始化
            for n in xfer.nodes_to_load or []:  # 遍历需要加载的节点
                cd = n.component_data[ct]  # 获取组件数据
                n_len = len(cd.host_value)  # 获取主机值长度
                cd.value = device_indices[offset : offset + n_len].clone()  # 从设备索引中恢复设备值
                offset += n_len  # 更新偏移量
                # Full uses leaf sets, not LRU  # Full使用叶子集合而非LRU
                self.cache.component_evictable_size_[ct] += n_len  # 增加可驱逐大小
                self.cache._update_evictable_leaf_sets(n)  # 更新可驱逐叶子集合

            self.cache._update_evictable_leaf_sets(node)  # 更新目标节点的叶子集合
