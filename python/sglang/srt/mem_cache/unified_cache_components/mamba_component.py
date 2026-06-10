# Mamba状态缓存组件
# 本文件实现了MambaComponent类，继承自TreeComponent，用于管理Mamba模型的SSM状态缓存。
# 主要功能包括：前缀匹配验证、匹配结果后处理（CoW分支）、插入时Mamba数据提交、
# 节点分裂时数据重分布、设备/主机层驱逐、LRU驱逐驱动、组件锁定/解锁、
# 请求缓存前后的Mamba槽位分配与清理、HiCache备份/加载/存储/预取等。

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
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer  # 导入HiCache存储策略、池名和传输描述
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # 从树组件模块导入基类和枚举
    CacheTransferPhase,  # 缓存传输阶段枚举
    ComponentType,  # 组件类型枚举
    EvictLayer,  # 驱逐层级枚举
    TreeComponent,  # 树组件抽象基类
    get_and_increase_time_counter,  # 获取并递增时间计数器
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.managers.schedule_batch import Req  # 请求类型
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # 缓存初始化参数
    from sglang.srt.mem_cache.unified_radix_cache import (  # 统一基数缓存
        UnifiedRadixCache,  # 统一基数缓存类
        UnifiedTreeNode,  # 统一树节点类
    )


class MambaComponent(TreeComponent):  # Mamba组件类，继承自TreeComponent
    component_type = ComponentType.MAMBA  # 组件类型为MAMBA

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):  # 初始化MambaComponent实例
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool  # 导入混合请求到token池

        assert isinstance(  # 断言请求到token池为混合类型
            cache.req_to_token_pool, HybridReqToTokenPool
        ), f"MambaComponent requires HybridReqToTokenPool, got {type(cache.req_to_token_pool)}"  # MambaComponent需要HybridReqToTokenPool
        if not params.enable_mamba_extra_buffer:  # 如果未启用Mamba额外缓冲
            assert (  # 断言页大小为1
                cache.page_size == 1
            ), f"MambaComponent requires page_size=1 when mamba_extra_buffer is disabled, got {cache.page_size}"  # 禁用mamba_extra_buffer时page_size必须为1
        super().__init__(cache, params)  # 调用父类初始化
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer  # 保存是否启用Mamba额外缓冲
        # HiCache state  # HiCache状态
        self._mamba_pool_host = None  # set to host mamba pool when HiCache enabled  # 启用HiCache时设置为主机Mamba池

    def create_match_validator(  # 创建匹配验证器，返回判断节点是否为有效匹配边界的谓词
        self, match_device_only: bool = False  # 是否仅匹配设备数据
    ) -> Callable[[UnifiedTreeNode], bool]:  # 返回节点验证谓词
        ct = self.component_type  # 获取组件类型
        if match_device_only:  # 如果仅匹配设备数据
            return lambda node: node.component_data[ct].value is not None  # 仅设备值非空为有效

        # HiCache: evicted + backuped (host_value present) is also a valid match  # HiCache：已驱逐但已备份（host_value存在）的节点也是有效匹配
        return lambda node: (  # 返回检查设备或主机值的验证器
            node.component_data[ct].value is not None  # 设备值非空
            or node.component_data[ct].host_value is not None  # 或主机值非空
        )

    def finalize_match_result(  # 最终处理匹配结果，处理Mamba的CoW分支和HiCache命中
        self,
        result: MatchResult,  # 原始匹配结果
        params: MatchPrefixParams,  # 匹配前缀参数
        value_chunks: list[torch.Tensor],  # 值张量分块列表
        best_value_len: int,  # 最佳值长度
    ) -> MatchResult:  # 返回更新后的匹配结果
        cow_mamba = params.cow_mamba  # 获取是否启用Mamba写时复制
        req = params.req  # 获取请求对象
        last_node = result.best_match_node  # 获取最佳匹配节点

        # HiCache can still use prefix matches and load back host-backed Mamba  # HiCache仍可使用前缀匹配并加载回主机备份的Mamba
        # states. We temporarily skip branching-state fill in that mode and can  # 状态。在此模式下暂时跳过分支状态填充
        # add a HiCache-aware branching policy later.  # 后续可添加HiCache感知的分支策略
        if self.cache.cache_controller is None and len(value_chunks) > best_value_len:  # 非HiCache且值分块超过最佳长度
            chunk_size = get_global_server_args().mamba_cache_chunk_size  # 获取Mamba缓存块大小
            aligned_seqlen = (  # 计算对齐后的序列长度
                sum(len(v) for v in value_chunks) // chunk_size  # 总长度整除块大小
            ) * chunk_size  # 乘以块大小
            branching_seqlen = aligned_seqlen if aligned_seqlen > 0 else None  # 对齐长度大于0则使用，否则为None
        else:  # HiCache模式或无额外值
            branching_seqlen = None  # 不进行分支

        mamba_value = last_node.component_data[self.component_type].value  # 获取匹配节点的Mamba值
        if cow_mamba and mamba_value is not None:  # 如果启用CoW且Mamba值存在
            assert req is not None  # 断言请求对象存在
            if req.mamba_pool_idx is None:  # 如果请求尚未分配Mamba池索引
                dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)  # 分配一个新的Mamba池槽位
                if dst_index is None:  # 如果分配失败
                    self.cache.inc_lock_ref(last_node)  # 锁定匹配节点
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))  # 驱逐一个Mamba槽位
                    dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)  # 重新分配
                    self.cache.dec_lock_ref(last_node)  # 解锁匹配节点
                    assert dst_index is not None, "Can not alloc mamba cache"  # 断言分配成功
                req.mamba_pool_idx = dst_index[0]  # 保存Mamba池索引
            req.mamba_cow_src_index = mamba_value  # 设置CoW源索引
            req.mamba_needs_clear = False  # 不需要清空

        # HiCache: if mamba was evicted from device but has host backup,  # HiCache：如果Mamba已从设备驱逐但有主机备份
        # ensure host_hit_length >= 1 so load_back is triggered.  # 确保host_hit_length >= 1以触发load_back
        cd = last_node.component_data[self.component_type]  # 获取组件数据
        if cd.value is None and cd.host_value is not None:  # 设备值空但主机值存在
            result = result._replace(host_hit_length=max(result.host_hit_length, 1))  # 确保主机命中长度至少为1

        return result._replace(mamba_branching_seqlen=branching_seqlen)  # 返回带分支序列长度的结果

    def commit_insert_component_data(  # 提交插入后的Mamba组件数据
        self,
        node: UnifiedTreeNode,  # 目标节点
        is_new_leaf: bool,  # 是否为新叶子节点
        params: InsertParams,  # 插入参数
        result: InsertResult,  # 插入结果
    ) -> None:
        assert params.mamba_value is not None  # 断言Mamba值存在
        if is_new_leaf:  # 如果是新叶子节点
            node.component_data[self.component_type].value = params.mamba_value  # 设置Mamba值
            self.cache.lru_lists[self.component_type].insert_mru(node)  # 插入设备LRU的MRU端
            self.cache.component_evictable_size_[self.component_type] += len(  # 增加可驱逐大小
                params.mamba_value
            )
            return  # 返回
        if node.component_data[self.component_type].value is None:  # 如果节点尚无Mamba设备值
            node.component_data[self.component_type].value = params.mamba_value  # 设置Mamba值
            # move from host LRU to device LRU  # 从主机LRU移动到设备LRU
            host_lru = self.cache.host_lru_lists[self.component_type]  # 获取主机LRU列表
            if host_lru.in_list(node):  # 如果节点在主机LRU中
                host_lru.remove_node(node)  # 从主机LRU中移除
            self.cache.lru_lists[self.component_type].insert_mru(node)  # 插入设备LRU的MRU端
            self.cache.component_evictable_size_[self.component_type] += len(  # 增加可驱逐大小
                params.mamba_value
            )
            node.last_access_time = get_and_increase_time_counter()  # 更新最后访问时间
            return  # 返回
        self.cache.lru_lists[self.component_type].reset_node_mru(node)  # 重置节点为MRU位置
        node.last_access_time = get_and_increase_time_counter()  # 更新最后访问时间
        result.mamba_exist = True  # 标记Mamba数据已存在

    def redistribute_on_node_split(  # 节点分裂时重分布Mamba组件数据
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode  # 新父节点和子节点
    ):
        ct = self.component_type  # 获取组件类型
        new_parent.component_data[ct].value = None  # 新父节点无Mamba设备值
        new_parent.component_data[ct].lock_ref = 0  # 新父节点锁引用为0
        # HiCache: mamba host_value stays on child (mamba = leaf-only data)  # HiCache：Mamba主机值保留在子节点（Mamba仅为叶子数据）
        new_parent.component_data[ct].host_value = None  # 新父节点无Mamba主机值
        new_parent.component_data[ct].host_lock_ref = 0  # 新父节点主机锁引用为0

    def evict_component(  # 驱逐节点上的Mamba组件数据
        self,
        node: UnifiedTreeNode,  # 待驱逐节点
        target: EvictLayer = EvictLayer.DEVICE,  # 驱逐目标层级
    ) -> tuple[int, int]:  # 返回(设备释放数量, 主机释放数量)
        cd = node.component_data[self.component_type]  # 获取组件数据
        freed = 0  # 设备释放计数初始化
        host_freed = 0  # 主机释放计数初始化

        # Device layer  # 设备层驱逐
        if EvictLayer.DEVICE in target and cd.value is not None:  # 如果目标是设备层且设备值存在
            self.cache.req_to_token_pool.mamba_pool.free(cd.value)  # 释放Mamba池中的设备值
            freed = len(cd.value)  # 记录释放的token数
            self.cache.component_evictable_size_[self.component_type] -= freed  # 减少可驱逐大小
            cd.value = None  # 置空设备值（Mamba可立即置空，不像Full需延迟）

        # Host layer  # 主机层驱逐
        host_lru = self.cache.host_lru_lists[self.component_type]  # 获取主机LRU列表
        if EvictLayer.HOST in target and cd.host_value is not None:  # 如果目标是主机层且主机值存在
            host_freed = len(cd.host_value)  # 记录主机释放的token数
            if self._mamba_pool_host is not None:  # 如果主机Mamba池存在
                self._mamba_pool_host.free(cd.host_value)  # 释放主机Mamba池中的值
            cd.host_value = None  # 置空主机值
            if host_lru.in_list(node):  # 如果节点在主机LRU中
                host_lru.remove_node(node)  # 从主机LRU中移除

        # After device tombstone: if only host_value remains, insert into host LRU  # 设备驱逐后：如果仅剩主机值，插入主机LRU
        if (
            target is EvictLayer.DEVICE  # 仅设备层驱逐
            and cd.value is None  # 设备值已清空
            and cd.host_value is not None  # 主机值仍存在
        ):
            if not host_lru.in_list(node):  # 如果不在主机LRU中
                host_lru.insert_mru(node)  # 插入主机LRU的MRU端

        return freed, host_freed  # 返回设备和主机释放数量

    def drive_eviction(  # 驱动Mamba组件的驱逐流程，使用LRU策略
        self, params: EvictParams, tracker: dict[ComponentType, int]  # 驱逐参数和追踪器
    ) -> None:
        request = params.mamba_num  # 需要释放的Mamba槽位数
        ct = self.component_type  # 获取组件类型
        lru = self.cache.lru_lists[ct]  # 获取设备LRU列表
        x = lru.get_lru_no_lock()  # 获取LRU端节点
        while tracker[ct] < request and x is not None and lru.in_list(x):  # 循环直到满足需求
            assert x.component_data[ct].value is not None  # 断言Mamba设备值存在
            if x in self.cache.evictable_device_leaves:  # 如果是可驱逐的设备叶子
                # D-leaf: atomic eviction of all components  # 设备叶子：原子驱逐所有组件
                x_next = lru.get_prev_no_lock(x)  # 获取前驱节点
                self.cache._evict_device_leaf(x, tracker)  # 驱逐设备叶子
                if not lru.in_list(x_next):  # 如果前驱已不在LRU中
                    x_next = lru.get_lru_no_lock()  # 重新获取LRU端
                x = x_next  # 移动到下一个
            else:  # 内部节点
                # Internal: tombstone Mamba + cascade  # 内部节点：墓碑Mamba + 级联驱逐
                x_next = lru.get_prev_no_lock(x)  # 获取前驱节点
                self.cache._evict_component_and_detach_lru(  # 驱逐组件并从LRU分离
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)  # 级联驱逐低优先级组件
                x = x_next  # 移动到下一个

    def acquire_component_lock(  # 获取Mamba组件锁引用
        self,
        node: UnifiedTreeNode,  # 目标节点
        result: IncLockRefResult,  # 增锁结果
        lock_host: bool = False,  # 是否锁定主机数据
    ) -> IncLockRefResult:  # 返回更新后的增锁结果
        ct = self.component_type  # 获取组件类型
        if node is self.cache.root_node:  # 根节点不需要锁定
            return result  # 直接返回
        cd = node.component_data[ct]  # 获取组件数据
        value = cd.host_value if lock_host else cd.value  # 根据锁定类型获取对应值
        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.  # skip_lock_node_ids中的节点在获取锁时是墓碑状态
        if value is None:  # 如果值为空（墓碑）
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)  # 记录跳过锁定的节点
            return result  # 返回

        if lock_host:  # 如果锁定主机数据
            if cd.host_lock_ref == 0:  # 如果主机锁引用为0
                host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU列表
                if host_lru.in_list(node):  # 如果在主机LRU中
                    host_lru.remove_node(node)  # 从主机LRU中移除
            cd.host_lock_ref += 1  # 增加主机锁引用
        else:  # 锁定设备数据
            if cd.lock_ref == 0:  # 如果设备锁引用为0
                vlen = len(value)  # 获取值长度
                self.cache.component_evictable_size_[ct] -= vlen  # 减少可驱逐大小
                self.cache.component_protected_size_[ct] += vlen  # 增加受保护大小
            cd.lock_ref += 1  # 增加设备锁引用
        return result  # 返回结果

    def release_component_lock(  # 释放Mamba组件锁引用
        self,
        node: UnifiedTreeNode,  # 目标节点
        params: Optional[DecLockRefParams],  # 减锁参数
        lock_host: bool = False,  # 是否释放主机锁
    ) -> None:
        ct = self.component_type  # 获取组件类型
        if node is self.cache.root_node:  # 根节点无需释放
            return  # 直接返回
        cd = node.component_data[ct]  # 获取组件数据
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()  # 获取跳过锁定的节点ID
        if node.id in skip_lock_node_ids:  # 如果节点在跳过集合中
            return  # 直接返回

        value = cd.host_value if lock_host else cd.value  # 根据锁定类型获取对应值
        if lock_host:  # 如果释放主机锁
            cd.host_lock_ref -= 1  # 减少主机锁引用
            if cd.host_lock_ref == 0 and cd.value is None and cd.host_value is not None:  # 主机锁归零、设备值空、主机值存在
                host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU
                if not host_lru.in_list(node):  # 如果不在主机LRU中
                    host_lru.insert_mru(node)  # 插入主机LRU
            return  # 返回

        if cd.lock_ref > 0:  # 如果设备锁引用大于0
            if cd.lock_ref == 1:  # 如果将降为0
                vlen = len(value)  # 获取值长度
                self.cache.component_evictable_size_[ct] += vlen  # 增加可驱逐大小
                self.cache.component_protected_size_[ct] -= vlen  # 减少受保护大小
            cd.lock_ref -= 1  # 减少设备锁引用

    def _alloc_mamba_slot(self) -> torch.Tensor:  # 分配一个Mamba池槽位，必要时驱逐
        """Allocate one mamba pool slot, evicting if necessary."""  # 分配一个Mamba池槽位，必要时进行驱逐
        slot = self.cache.req_to_token_pool.mamba_pool.alloc(1)  # 尝试分配一个槽位
        if slot is None:  # 如果分配失败
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))  # 驱逐一个Mamba槽位
            slot = self.cache.req_to_token_pool.mamba_pool.alloc(1)  # 重新分配
            assert slot is not None, "Can not alloc mamba cache"  # 断言分配成功
        return slot  # 返回分配的槽位

    def prepare_for_caching_req(  # 在请求缓存前准备Mamba组件数据
        self,
        req: Req,  # 请求对象
        insert_params: InsertParams,  # 插入参数
        token_ids_len: int,  # token ID长度
        is_finished: bool,  # 请求是否已完成
    ) -> Optional[int]:  # 返回有效缓存长度或None
        cache_len = (  # 计算缓存长度
            req.mamba_last_track_seqlen  # 如果启用额外缓冲，使用上次跟踪序列长度
            if self.enable_mamba_extra_buffer  # 判断是否启用Mamba额外缓冲
            else token_ids_len  # 否则使用token ID长度
        )
        if is_finished:  # 如果请求已完成
            if cache_len is None:  # 如果缓存长度为None
                cache_len = 0  # 设为0
            if self.enable_mamba_extra_buffer:  # 如果启用额外缓冲
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(  # 获取乒乓缓冲的另一个索引
                    req.mamba_next_track_idx
                )
                mamba_value = (  # 从乒乓缓冲中取出Mamba值
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:  # 未启用额外缓冲
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()  # 从Mamba池索引克隆值
            insert_params.mamba_value = mamba_value  # 设置插入参数的Mamba值
            return cache_len  # 返回缓存长度
        else:  # 请求未完成
            if cache_len is None:  # 如果缓存长度为None
                return 0  # 返回0
            # Donate the mamba index to the radix cache instead of copying.  # 将Mamba索引捐赠给基数缓存而非复制
            if self.enable_mamba_extra_buffer:  # 如果启用额外缓冲
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(  # 获取乒乓缓冲的另一个索引
                    req.mamba_next_track_idx
                )
                mamba_value_donated = (  # 从乒乓缓冲取出捐赠的Mamba值
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
                req.mamba_ping_pong_track_buffer[keep_idx] = self._alloc_mamba_slot()[0]  # 分配新槽位替换原缓冲
                self.cache.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[  # 更新映射
                    req.req_pool_idx
                ] = req.mamba_ping_pong_track_buffer
            else:  # 未启用额外缓冲
                mamba_value_donated = self._alloc_mamba_slot()  # 分配新Mamba槽位
                self.cache.req_to_token_pool.mamba_pool.copy_from(  # 从请求Mamba池复制到新槽位
                    req.mamba_pool_idx.unsqueeze(0), mamba_value_donated
                )
            insert_params.mamba_value = mamba_value_donated  # 设置插入参数的Mamba值
            return cache_len  # 返回缓存长度

    def cleanup_after_caching_req(  # 请求缓存后的Mamba资源清理
        self,
        req: Req,  # 请求对象
        is_finished: bool,  # 请求是否已完成
        insert_result: Optional[InsertResult] = None,  # 插入结果
        insert_params: Optional[InsertParams] = None,  # 插入参数
    ) -> None:
        if is_finished:  # 如果请求已完成
            mamba_exist = (  # 判断Mamba数据是否已存在于缓存中
                insert_result.mamba_exist if insert_result is not None else True  # 有插入结果则取其值，否则视为已存在
            )
            if self.enable_mamba_extra_buffer:  # 如果启用额外缓冲
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(  # 获取保留的乒乓索引
                    req.mamba_next_track_idx
                )
            else:  # 未启用额外缓冲
                keep_idx = None  # 无保留索引
            if mamba_exist:  # 如果Mamba已存在
                keep_idx = None  # 不保留任何索引
            free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist  # 决定是否释放Mamba缓存
            if free_mamba_cache:  # 如果需要释放
                self.cache.req_to_token_pool.free_mamba_cache(  # 释放Mamba缓存
                    req, mamba_ping_pong_track_buffer_to_keep=keep_idx  # 指定保留的乒乓索引
                )
        else:  # 请求未完成
            if insert_params.mamba_value is not None and (  # 如果有Mamba值且
                insert_result is None or insert_result.mamba_exist  # 无插入结果或Mamba已存在
            ):
                self.cache.req_to_token_pool.mamba_pool.free(insert_params.mamba_value)  # 释放捐赠的Mamba槽位
            req.mamba_last_track_seqlen = None  # 重置上次跟踪序列长度

    # ---- HiCache Hooks ----  # ---- HiCache钩子函数 ----

    def build_hicache_transfers(  # 构建HiCache传输描述符
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw  # 节点、传输阶段和额外参数
    ) -> Optional[list[PoolTransfer]]:  # 返回传输描述列表或None
        ct = self.component_type  # 获取组件类型

        if phase == CacheTransferPhase.BACKUP_HOST:  # 备份到主机阶段
            cd = node.component_data[ct]  # 获取组件数据
            if cd.value is None:  # 如果设备值不存在
                return None  # 无需备份
            return [  # 返回Mamba备份传输
                PoolTransfer(
                    name=PoolName.MAMBA,  # Mamba池
                    device_indices=cd.value,  # 设备端索引
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:  # 从主机加载回设备阶段
            req = kw.get("req")  # 获取请求对象
            transfers: list[PoolTransfer] = []  # 传输列表

            cd = node.component_data[ct]  # 获取组件数据
            if cd.value is not None:  # 如果设备值已存在
                return None  # 无需加载

            # restore single node if host_value exists  # 如果主机值存在则恢复单个节点
            if cd.host_value is not None and cd.value is None:  # 主机值存在且设备值不存在
                transfers.append(  # 添加恢复传输
                    PoolTransfer(
                        name=PoolName.MAMBA,  # Mamba池
                        host_indices=cd.host_value,  # 主机端索引
                        nodes_to_load=[node],  # 需要加载的节点
                    )
                )

            # Per-request mamba CoW (H→D copy into request's device slot)  # 每请求Mamba CoW（H→D复制到请求的设备槽位）
            cd = node.component_data[ct]  # 重新获取组件数据
            if req is not None and cd.host_value is not None:  # 如果有请求且主机值存在
                if req.mamba_pool_idx is None:  # 如果请求尚未分配Mamba池索引
                    dst = self.cache.req_to_token_pool.mamba_pool.alloc(1)  # 分配设备槽位
                    if dst is None:  # 分配失败
                        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))  # 驱逐一个Mamba槽位
                        dst = self.cache.req_to_token_pool.mamba_pool.alloc(1)  # 重新分配
                        assert dst is not None, "Cannot alloc mamba for load_back"  # 断言分配成功
                    req.mamba_pool_idx = dst[0]  # 保存Mamba池索引
                transfers.append(  # 添加CoW传输
                    PoolTransfer(
                        name=PoolName.MAMBA,  # Mamba池
                        host_indices=cd.host_value,  # 主机端索引
                        device_indices=req.mamba_pool_idx.unsqueeze(0),  # 请求的设备端索引
                    )
                )

            return transfers if transfers else None  # 返回传输列表或None

        if phase == CacheTransferPhase.BACKUP_STORAGE:  # 备份到存储阶段
            cd = node.component_data[ct]  # 获取组件数据
            if cd.host_value is None or not node.hash_value:  # 如果主机值或哈希值不存在
                return None  # 无需备份
            return [  # 返回存储备份传输
                PoolTransfer(
                    name=PoolName.MAMBA,  # Mamba池
                    host_indices=cd.host_value,  # 主机端索引
                    keys=[node.hash_value[-1]],  # 哈希键（最后一个）
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,  # 尾页命中策略
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:  # 预取阶段
            host_indices = self._mamba_pool_host.alloc(1)  # 分配主机Mamba池空间
            if host_indices is None:  # 分配失败
                self.cache.evict_host(1, ComponentType.MAMBA)  # 驱逐一个Mamba主机槽位
                host_indices = self._mamba_pool_host.alloc(1)  # 重新分配
            if host_indices is None:  # 仍然失败
                return []  # 返回空列表表示分配失败
            return [  # 返回预取传输
                PoolTransfer(
                    name=PoolName.MAMBA,  # Mamba池
                    host_indices=host_indices,  # 主机端索引
                    keys=["__placeholder__"],  # 占位键
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,  # 尾页命中策略
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

        elif phase == CacheTransferPhase.LOAD_BACK:  # 从主机加载回设备阶段
            if not transfers:  # 如果无传输
                return  # 直接返回
            transfer = transfers[0]  # 获取第一个传输
            if transfer.device_indices is not None:  # 如果设备索引存在（加载成功）
                cd = node.component_data[ct]  # 获取组件数据
                cd.value = transfer.device_indices.clone()  # 保存设备索引
                count = len(cd.value)  # 获取加载的token数
                # Move from host LRU to device LRU  # 从主机LRU移动到设备LRU
                host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU列表
                if host_lru.in_list(node):  # 如果在主机LRU中
                    host_lru.remove_node(node)  # 从主机LRU中移除
                self.cache.lru_lists[ct].insert_mru(node)  # 插入设备LRU的MRU端
                self.cache.component_evictable_size_[ct] += count  # 增加可驱逐大小

        elif phase == CacheTransferPhase.PREFETCH:  # 预取阶段
            if not transfers:  # 如果无传输
                return  # 直接返回
            transfer = transfers[0]  # 获取第一个传输
            host_indices = transfer.host_indices  # 获取主机索引
            insert_result = kw.get("insert_result")  # 获取插入结果
            pool_storage_result = kw.get("pool_storage_result")  # 获取存储池结果
            loaded = (  # 判断是否成功加载
                pool_storage_result is not None  # 存储池结果存在
                and pool_storage_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0) >= 1  # Mamba池命中页数>=1
            )
            target_node = (  # 获取目标插入节点
                insert_result.inserted_host_node if insert_result is not None else None  # 插入的主机节点或None
            )
            if (  # 如果任何条件不满足
                host_indices is None  # 无主机索引
                or target_node is None  # 无目标节点
                or not loaded  # 未加载
                or target_node.component_data[ct].host_value is not None  # 目标节点已有主机值
            ):
                self.cache.cache_controller.append_host_mem_release(  # 释放预分配的主机内存
                    extra_pools=[transfer]
                )
                if insert_result is not None:  # 如果有插入结果
                    insert_result.mamba_exist = True  # 标记Mamba已存在（无需插入）
                return  # 返回

            target_node.component_data[ct].host_value = host_indices.clone()  # 保存主机索引到目标节点
            if target_node.component_data[ct].value is None:  # 如果目标节点无设备值
                host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU列表
                if not host_lru.in_list(target_node):  # 如果不在主机LRU中
                    host_lru.insert_mru(target_node)  # 插入主机LRU
            if insert_result is not None:  # 如果有插入结果
                insert_result.mamba_exist = False  # 标记Mamba不存在（需要新建）

    def drive_host_eviction(  # 驱动Mamba主机层驱逐
        self, num_tokens: int, tracker: dict[ComponentType, int]  # 需释放token数和追踪器
    ) -> None:
        """Evict mamba host resources.  # 驱逐Mamba主机资源
        Internal nodes: private tombstone (free host mamba only).  # 内部节点：私有墓碑（仅释放主机Mamba）
        Host leaves: atomic eviction via _evict_host_leaf."""  # 主机叶子：通过_evict_host_leaf原子驱逐
        ct = self.component_type  # 获取组件类型
        host_lru = self.cache.host_lru_lists[ct]  # 获取主机LRU列表
        x = host_lru.get_lru_no_lock()  # 获取LRU端节点
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):  # 循环直到满足需求
            x_next = host_lru.get_prev_no_lock(x)  # 获取前驱节点
            cd = x.component_data[ct]  # 获取组件数据
            if x in self.cache.evictable_host_leaves:  # 如果是可驱逐的主机叶子
                # Host leaf: atomic eviction (all components host + delete)  # 主机叶子：原子驱逐（所有组件主机层 + 删除节点）
                self.cache._evict_host_leaf(x, tracker)  # 驱逐主机叶子
            else:  # 内部节点
                # Internal: tombstone Mamba + cascade  # 内部节点：墓碑Mamba + 级联驱逐
                assert cd.host_value is not None  # 断言主机值存在
                self.cache._evict_component_and_detach_lru(  # 驱逐组件并从LRU分离
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)  # 级联驱逐
                self.cache._update_evictable_leaf_sets(x)  # 更新可驱逐叶子集合
            x = x_next  # 移动到下一个节点
