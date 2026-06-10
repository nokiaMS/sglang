# HiMamba基数缓存模块
# 实现混合Mamba模型的分层基数缓存，支持GPU/CPU/存储三级缓存层次结构
# 包含主机LRU管理、设备/主机淘汰策略、写穿透/写回机制、存储后端预取与备份
# 支持Mamba状态的主机端和设备端联合管理，以及推测解码和流水线并行

from __future__ import annotations  # 启用延迟类型注解评估

import atexit  # 导入退出处理模块
import heapq  # 导入堆队列模块（用于LRU淘汰）
import json  # 导入JSON处理模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
import threading  # 导入线程模块
import time  # 导入时间模块
from queue import Empty  # 导入队列空异常
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch张量库

from sglang.srt.disaggregation.kv_events import StorageMedium  # 导入存储介质枚举
from sglang.srt.mem_cache.base_prefix_cache import (  # 导入前缀缓存基类及相关参数类型
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (  # 导入HiCache存储相关类型
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PrefetchTimeoutConfig,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (  # 导入混合缓存控制器
    PrefetchOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (  # 导入混合池组装器
    attach_hybrid_pool_to_mamba_cache,
)
from sglang.srt.mem_cache.mamba_radix_cache import (  # 导入Mamba基数缓存相关类型
    LRUList,
    MambaRadixCache,
    TreeNode,
    get_last_access_time,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool  # 导入混合线性KV池和请求映射池
from sglang.srt.mem_cache.radix_cache import (  # 导入基数键类型
    RadixKey,
)
from sglang.srt.mem_cache.utils import compute_node_hash_values, split_node_hash_value  # 导入哈希计算工具
from sglang.srt.observability.metrics_collector import (  # 导入可观测性指标收集器
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:  # 仅在类型检查时执行的代码块
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # 导入缓存初始化参数类型
    from sglang.srt.server_args import ServerArgs  # 导入服务器参数类型

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class HostLRUList(LRUList):  # 主机端Mamba LRU列表
    def __init__(self):  # 初始化主机Mamba LRU列表
        super().__init__(mamba=True)  # 调用父类初始化，启用mamba模式
        self.prv = "host_mamba_prev"  # 前驱指针属性名
        self.nxt = "host_mamba_next"  # 后继指针属性名
        setattr(self.head, self.nxt, self.tail)  # 初始化头节点的后继为尾节点
        setattr(self.tail, self.prv, self.head)  # 初始化尾节点的前驱为头节点

    def reset_node_mru(self, node):  # 重置节点为最近使用（MRU）位置
        assert node.id in self.cache, f"Resetting node {node.id=} not in host mamba lru"  # 断言节点在缓存中
        assert (
            node.mamba_host_value is not None
        ), f"Resetting host mamba tombstone node in lru list: {node.id=}"  # 断言节点不是墓碑节点
        self._remove_node(node)  # 从当前位置移除
        self._add_node(node)  # 添加到MRU位置

    def insert_mru(self, node):  # 将节点插入到MRU位置
        assert (
            node.mamba_host_value is not None
        ), f"Inserting host mamba tombstone node in lru list: {node.id=}"  # 断言节点不是墓碑节点
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in host mamba lru list"  # 断言节点不在列表中
        self.cache[node.id] = node  # 将节点加入缓存
        self._add_node(node)  # 添加到MRU位置

    def remove_node(self, node: TreeNode):  # 从LRU列表中移除节点
        assert node.id in self.cache, f"Removing node {node.id=} not in host mamba lru"  # 断言节点在缓存中
        assert (
            node.mamba_host_value is not None
        ), f"Removing host mamba tombstone node from lru list: {node.id=}"  # 断言节点不是墓碑节点
        del self.cache[node.id]  # 从缓存中删除
        self._remove_node(node)  # 从链表中移除


class HiMambaRadixCache(MambaRadixCache):  # 分层Mamba基数缓存
    """Hierarchical cache for hybrid Mamba models.
    混合Mamba模型的分层缓存。"""

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):  # 初始化分层Mamba基数缓存
        self._enable_metrics_flag = params.enable_metrics  # 保存指标启用标志
        if server_args.hicache_io_backend == "direct":  # 如果IO后端为direct模式
            if server_args.hicache_mem_layout == "page_first":  # 如果内存布局为page_first
                server_args.hicache_mem_layout = "page_first_direct"  # 切换为page_first_direct
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )  # 记录警告：直接IO后端不支持page_first布局

        self.page_size = params.page_size  # 保存页面大小
        self.hybrid_kv_cache = params.token_to_kv_pool_allocator.get_kvcache()  # 获取混合KV缓存
        if not isinstance(self.hybrid_kv_cache, HybridLinearKVPool):  # 断言为混合线性KV池
            raise ValueError(
                "HiMambaRadixCache requires HybridLinearKVPool for hybrid SSM models."
            )
        if not isinstance(params.req_to_token_pool, HybridReqToTokenPool):  # 断言为混合请求映射池
            raise ValueError(
                "HiMambaRadixCache requires HybridReqToTokenPool for hybrid SSM models."
            )

        self.kvcache = self.hybrid_kv_cache.full_kv_pool  # 获取完整KV池

        self.tp_group = params.tp_cache_group  # 保存张量并行通信组
        self.tp_world_size = (  # 计算张量并行世界大小
            1
            if self.tp_group is None  # 无通信组则大小为1
            else torch.distributed.get_world_size(group=self.tp_group)  # 否则获取世界大小
        )

        self.enable_storage = server_args.hicache_storage_backend is not None  # 是否启用存储后端
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics  # 是否启用存储指标
        self.extra_metric_labels = server_args.extra_metric_labels  # 额外指标标签

        (  # 解析存储后端额外配置
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config  # 传入额外配置JSON
        )
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func  # 设置预取超时检查函数
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy  # 保存预取停止策略

        self.load_cache_event = threading.Event()  # 创建加载缓存事件
        attach_hybrid_pool_to_mamba_cache(  # 将混合池附加到Mamba缓存
            self,
            params,  # 缓存初始化参数
            server_args,  # 服务器参数
            extra_config=extra_config,  # 额外配置
            prefetch_threshold=prefetch_threshold,  # 预取阈值
            load_cache_event=self.load_cache_event,  # 加载缓存事件
            enable_storage_metrics=self.enable_storage_metrics,  # 存储指标启用标志
            attn_cp_group=params.attn_cp_cache_group,  # 注意力CP通信组
            attn_tp_group=params.attn_tp_cache_group,  # 注意力TP通信组
        )
        self._apply_storage_runtime_config(  # 应用存储运行时配置
            storage_backend=server_args.hicache_storage_backend,  # 存储后端类型
            prefetch_threshold=prefetch_threshold,  # 预取阈值
            prefetch_timeout_config=prefetch_timeout_config,  # 预取超时配置
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,  # 是否传递前缀键
            enable_storage=self.enable_storage,  # 是否启用存储
            enable_storage_metrics=self.enable_storage_metrics,  # 是否启用存储指标
            extra_metric_labels=self.extra_metric_labels,  # 额外指标标签
        )

        self.ongoing_write_through = {}  # 进行中的写穿透操作字典
        self.ongoing_load_back = {}  # 进行中的加载回传操作字典
        self.ongoing_prefetch = {}  # 进行中的预取操作字典
        self.ongoing_backup = {}  # 进行中的备份操作字典
        # track per-request tokens loaded from storage (L3 hits)
        # 跟踪每个请求从存储加载的token数（L3命中）
        # key: request_id, value: number of tokens actually loaded from storage
        # 键：请求ID，值：实际从存储加载的token数量
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}  # 预取加载token数按请求ID索引

        self.write_through_threshold = (  # 写穿透阈值
            1 if server_args.hicache_write_policy == "write_through" else 2  # 写穿透策略阈值为1，否则为2
        )
        self.load_back_threshold = 10  # 加载回传阈值

        self.evictable_full_device_leaves: set[TreeNode] = set()  # 可淘汰的设备端叶子节点集合
        self.evictable_full_host_leaves: set[TreeNode] = set()  # 可淘汰的主机端叶子节点集合
        self.mamba_host_lru_list = HostLRUList()  # Mamba主机LRU列表

        # Detach storage backend automatically on process shutdown
        # 进程退出时自动分离存储后端
        atexit.register(self.shutdown)  # 注册退出处理函数

        super().__init__(params=params)  # 调用父类初始化

    def reset(self) -> None:  # 重置缓存
        TreeNode.counter = 0  # 重置树节点计数器
        self._flush_pending_storage_backups_before_reset()  # 在重置前刷新待处理的存储备份
        self.cache_controller.reset()  # 重置缓存控制器
        self.full_kv_pool_host.clear()  # 清空主机KV池
        self.mamba_pool_host.clear()  # 清空主机Mamba池
        self.ongoing_write_through = {}  # 清空进行中的写穿透
        self.ongoing_load_back = {}  # 清空进行中的加载回传
        self.ongoing_prefetch = {}  # 清空进行中的预取
        self.ongoing_backup = {}  # 清空进行中的备份
        self.prefetch_loaded_tokens_by_reqid.clear()  # 清空预取加载token计数
        self.evictable_full_device_leaves.clear()  # 清空可淘汰设备叶子集合
        self.evictable_full_host_leaves.clear()  # 清空可淘汰主机叶子集合
        self.mamba_host_lru_list = HostLRUList()  # 重新创建主机Mamba LRU列表
        logger.info(
            "HiMambaRadixCache reset completed: host_kv_available=%s host_mamba_available=%s",
            self.full_kv_pool_host.available_size(),  # 主机KV可用大小
            self.mamba_pool_host.available_size(),  # 主机Mamba可用大小
        )
        super().reset()  # 调用父类重置

    def write_backup(self, node: TreeNode, write_back=False) -> int:  # 将节点备份到主机
        # Backup invariant (for write-through mode): backed-up nodes must form a
        # contiguous prefix from root — no gaps.  Skip if parent isn't backed
        # up yet;
        # 备份不变式（写穿透模式）：已备份的节点必须从根节点形成连续前缀——无间隙。
        # 如果父节点尚未备份则跳过；
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0  # 父节点未备份，跳过

        # If mamba host slot already exists, refresh its LRU position.
        # 如果Mamba主机槽已存在，刷新其LRU位置。
        if node.mamba_value is not None and node.mamba_host_value is not None:
            if self.mamba_host_lru_list.in_list(node):
                self.mamba_host_lru_list.reset_node_mru(node)  # 刷新MRU位置

        extra_pools = self.mamba_backup_transfers(node)  # 构建Mamba备份传输描述符
        host_indices = self.cache_controller.write(  # 执行写操作
            device_indices=node.value,  # 设备端索引
            node_id=node.id,  # 节点ID
            extra_pools=extra_pools,  # 额外池传输
        )
        if host_indices is None:  # 主机内存不足
            self.evict_host(len(node.value))  # 淘汰主机端空间
            host_indices = self.cache_controller.write(  # 重试写操作
                device_indices=node.value,
                node_id=node.id,
                extra_pools=extra_pools,
            )
        if host_indices is not None:  # 写入成功
            node.host_value = host_indices.clone()  # 保存主机端索引
            if extra_pools is not None:  # 如果有额外池传输
                self.mamba_backup_commit(node, extra_pools)  # 提交Mamba备份
            assert len(node.host_value) > 0  # 断言主机值非空
            self.ongoing_write_through[node.id] = node  # 记录进行中的写穿透
            if not write_back:  # 非写回模式
                # no need to lock nodes if write back
                # 写回模式下不需要锁定节点
                self.inc_lock_ref(node)  # 增加引用计数锁定节点
        else:  # 写入失败
            return 0  # 返回0

        return len(host_indices)  # 返回写入的索引数量

    def load_back(  # 从主机加载完整KV回设备
        self, node: TreeNode, mem_quota: Optional[int] = None, req=None  # 节点、内存配额、请求对象
    ) -> Optional[torch.Tensor]:
        """Load full KV back from host.
        从主机加载完整KV回设备。"""
        last_hit_node = node  # 保存最后命中节点
        nodes_to_load = []  # 需要加载的节点列表

        while node.evicted:  # 向上遍历直到找到未淘汰的祖先
            assert node.backuped, f"No backup on evicted node {node.id}"  # 断言被淘汰节点已备份
            nodes_to_load.insert(0, node)  # 将节点插入列表头部（保持顺序）
            node = node.parent  # 向上移动
        else:
            ancestor_node = node  # 保存祖先节点

        mamba_restore_nodes = []  # 需要恢复Mamba状态的节点列表
        if last_hit_node.mamba_backuped and last_hit_node.mamba_evicted:  # 如果Mamba状态需要恢复
            mamba_restore_nodes.append(last_hit_node)  # 添加到恢复列表

        result = self.inc_lock_ref(ancestor_node)  # 增加祖先节点引用计数
        delta = result.delta  # 获取引用计数增量

        if nodes_to_load:  # 如果有需要加载的节点
            full_host_indices = torch.cat([n.host_value for n in nodes_to_load])  # 拼接所有主机索引
        else:
            full_host_indices = torch.empty((0,), dtype=torch.int64, device="cpu")  # 创建空张量

        if (  # 检查是否跳过加载
            len(full_host_indices) > 0
            and (
                (len(full_host_indices) < self.load_back_threshold)  # 加载量太小
                or (
                    len(full_host_indices) > mem_quota + delta
                    if mem_quota is not None
                    else False
                )  # 超出内存配额
            )
            and len(mamba_restore_nodes) == 0  # 无Mamba需要恢复
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            # 如果总大小太小或超出内存配额，则跳过加载
            self.dec_lock_ref(ancestor_node)  # 释放引用
            return None  # 返回None

        logger.debug(
            f"Init load back from cpu -> gpu, kv hit length: {len(full_host_indices)}, mamba host hit length: {len(mamba_restore_nodes)}"
        )  # 记录调试信息
        mamba_pools = self.mamba_restore_transfers(  # 构建Mamba恢复传输描述符
            last_hit_node, mamba_restore_nodes, req
        )
        full_device_indices = self.cache_controller.load(  # 执行加载操作
            host_indices=full_host_indices,
            node_id=last_hit_node.id,
            extra_pools=mamba_pools,
        )
        if full_device_indices is None:  # 设备内存不足
            if len(full_host_indices) > 0:  # 如果有主机索引
                self.evict(EvictParams(num_tokens=len(full_host_indices)))  # 淘汰设备端空间

            mamba_pools = self.mamba_restore_transfers(  # 重新构建Mamba恢复传输
                last_hit_node, mamba_restore_nodes, req
            )
            full_device_indices = self.cache_controller.load(  # 重试加载
                host_indices=full_host_indices,
                node_id=last_hit_node.id,
                extra_pools=mamba_pools,
            )
        self.dec_lock_ref(ancestor_node)  # 释放祖先节点引用
        if full_device_indices is None:  # 加载失败
            # no sufficient GPU memory to load back KV caches
            # GPU内存不足，无法加载回KV缓存
            return None  # 返回None

        self.mamba_restore_commit(mamba_restore_nodes, mamba_pools)  # 提交Mamba恢复

        offset = 0  # 偏移量
        for n in nodes_to_load:  # 遍历需要加载的节点
            n_len = len(n.host_value)  # 获取节点的主机值长度
            n.value = full_device_indices[offset : offset + n_len].clone()  # 分配设备索引
            offset += n_len  # 更新偏移量

            self._record_store_event(n, medium=StorageMedium.GPU)  # 记录存储事件
            self.full_lru_list.insert_mru(n)  # 插入全量LRU列表
            self.full_evictable_size_ += n_len  # 更新可淘汰大小
            self._update_leaf_status(n)  # 更新叶子状态

        for n in mamba_restore_nodes:  # 遍历Mamba恢复节点
            if self.mamba_lru_list.in_list(n):  # 如果在LRU列表中
                self.mamba_lru_list.reset_node_mru(n)  # 刷新MRU位置
            else:  # 不在LRU列表中
                self.mamba_lru_list.insert_mru(n)  # 插入MRU位置
                self.mamba_evictable_size_ += len(n.mamba_value)  # 更新可淘汰大小

        self._update_leaf_status(ancestor_node)  # 更新祖先节点叶子状态

        self.inc_lock_ref(last_hit_node)  # 增加最后命中节点的引用计数
        self.ongoing_load_back[last_hit_node.id] = last_hit_node  # 记录进行中的加载回传

        return full_device_indices  # 返回设备索引

    def init_load_back(  # 初始化加载回传操作
        self,
        params: InitLoadBackParams,  # 加载回传参数
    ):
        last_node = params.best_match_node  # 获取最佳匹配节点
        mem_quota = params.mem_quota  # 获取内存配额
        req = params.req  # 获取请求对象
        if last_node.evicted or (last_node.mamba_evicted and last_node.mamba_backuped):  # 如果需要加载
            loading_values = self.load_back(last_node, mem_quota, req=req)  # 执行加载回传
            if loading_values is not None:  # 加载成功
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )  # 记录调试信息
                return loading_values, last_node  # 返回加载结果

            while last_node is not self.root_node and (  # 向上查找可用的祖先节点
                last_node.evicted or last_node.mamba_evicted
            ):
                last_node = last_node.parent  # 向上移动

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),  # 空设备索引
            last_node,  # 可用节点
        )

    def _inc_hit_count(self, node: TreeNode, chunked=False):  # 增加节点命中计数
        if self.cache_controller.write_policy == "write_back" or chunked:  # 写回模式或分块模式
            return  # 不计数
        node.hit_count += 1  # 增加命中计数

        if not node.backuped and node.hit_count >= self.write_through_threshold:  # 达到写穿透阈值
            # write to host if the node is not backuped
            # 如果节点未备份，则写入主机
            self.write_backup(node)  # 执行写备份

    def writing_check(self, write_back=False):  # 检查写穿透完成状态
        if write_back:  # 写回模式
            # blocking till all write back complete
            # 阻塞直到所有写回完成
            while len(self.ongoing_write_through) > 0:  # 等待所有写穿透完成
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:  # 遍历确认队列
                    finish_event.synchronize()  # 同步完成事件
                    for ack_id in ack_list:  # 遍历确认ID
                        backuped_node = self.ongoing_write_through.pop(ack_id)  # 取出已备份节点
                        self._record_store_event(
                            backuped_node, medium=StorageMedium.CPU
                        )  # 记录存储事件
                        if self.enable_storage:  # 如果启用存储
                            self.write_backup_storage(backuped_node)  # 写备份到存储
                self.cache_controller.ack_write_queue.clear()  # 清空确认队列
                assert len(self.ongoing_write_through) == 0  # 断言所有写穿透已完成
            return  # 返回

        if len(self.ongoing_write_through) == 0:  # 没有进行中的写穿透
            return  # 直接返回

        finish_count = 0  # 已完成计数
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:  # 遍历确认队列
            if not finish_event.query():  # 事件未完成
                break  # 退出循环
            finish_count += 1  # 增加计数

        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")  # 创建张量
        if self.tp_world_size > 1:  # 多卡并行
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )  # 全局取最小完成数
        finish_count = int(queue_size.item())  # 转换为整数

        while finish_count > 0:  # 处理已完成的写穿透
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)  # 取出队首
            finish_event.synchronize()  # 同步完成事件
            for ack_id in ack_list:  # 遍历确认ID
                backuped_node = self.ongoing_write_through.pop(ack_id)  # 取出已备份节点
                self._record_store_event(backuped_node, medium=StorageMedium.CPU)  # 记录存储事件
                self.dec_lock_ref(backuped_node)  # 释放引用
                if self.enable_storage:  # 如果启用存储
                    self.write_backup_storage(backuped_node)  # 写备份到存储
            finish_count -= 1  # 减少计数

    def loading_check(self):  # 检查加载回传完成状态
        finish_count = 0  # 已完成计数
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:  # 遍历确认队列
            if not finish_event.query():  # 事件未完成
                # the KV cache loading is still ongoing
                # KV缓存加载仍在进行中
                break  # 退出循环
            finish_count += 1  # 增加计数
            for ack_id in ack_list:  # 遍历确认ID
                end_node = self.ongoing_load_back.pop(ack_id)  # 取出已加载节点
                self.dec_lock_ref(end_node)  # 释放引用

        del self.cache_controller.ack_load_queue[:finish_count]  # 删除已处理的队列项

    def ready_to_load_host_cache(self) -> int:  # 检查主机缓存是否准备好加载
        return self.cache_controller.start_loading()  # 返回可加载的token数

    def flush_write_through_acks(self) -> None:  # 刷新写穿透确认
        self.writing_check()  # 调用写检查

    def check_hicache_events(self):  # 检查HiCache事件（写穿透确认、加载确认、存储控制队列）
        self.writing_check()  # 检查写穿透
        self.loading_check()  # 检查加载回传

        if self.enable_storage:  # 如果启用存储
            self.drain_storage_control_queues()  # 排空存储控制队列
        if self.enable_storage_metrics:  # 如果启用存储指标
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )  # 记录存储指标

    def _protect_host_node(self, node: TreeNode, protect_mamba: bool = True):  # 保护主机节点不被淘汰
        node.protect_host()  # 增加主机引用计数
        self.evictable_full_host_leaves.discard(node)  # 从可淘汰集合中移除
        if protect_mamba:  # 如果需要保护Mamba
            node.protect_host_mamba()  # 增加主机Mamba引用计数
            if self.mamba_host_lru_list.in_list(node):  # 如果在Mamba主机LRU列表中
                self.mamba_host_lru_list.remove_node(node)  # 从列表中移除

    def _release_host_node(self, node: TreeNode, release_mamba: bool = True):  # 释放主机节点保护
        node.release_host()  # 减少主机引用计数
        if release_mamba:  # 如果需要释放Mamba
            node.release_host_mamba()  # 减少主机Mamba引用计数
            if node.host_mamba_ref_counter == 0 and node.mamba_host_value is not None:  # 引用归零且有主机值
                if not self.mamba_host_lru_list.in_list(node):  # 不在LRU列表中
                    self.mamba_host_lru_list.insert_mru(node)  # 插入LRU列表
        if node.host_ref_counter == 0 and node.host_mamba_ref_counter == 0:  # 所有引用归零
            self._update_full_host_leaf_status(node)  # 更新主机叶子状态

    def _discard_from_leaf_sets(self, node: TreeNode):  # 从叶子节点集合中丢弃节点
        self.evictable_full_device_leaves.discard(node)  # 从设备叶子集合移除
        self.evictable_full_host_leaves.discard(node)  # 从主机叶子集合移除

    def _update_leaf_status(self, node: TreeNode):  # 更新节点叶子状态
        self._update_full_device_leaf_status(node)  # 更新设备叶子状态
        self._update_full_host_leaf_status(node)  # 更新主机叶子状态

    def _update_full_device_leaf_status(self, node: TreeNode):  # 更新设备端叶子节点状态
        if node == self.root_node or node.evicted or node.full_lock_ref > 0:  # 根节点、已淘汰或被锁定
            self.evictable_full_device_leaves.discard(node)  # 从可淘汰集合移除
            return
        for child in node.children.values():  # 遍历子节点
            if not child.evicted:  # 有未淘汰的子节点
                self.evictable_full_device_leaves.discard(node)  # 非叶子节点，移除
                return
        self.evictable_full_device_leaves.add(node)  # 添加到可淘汰叶子集合

    def _update_full_host_leaf_status(self, node: TreeNode):  # 更新主机端叶子节点状态
        if (
            not node.evicted  # 未淘汰
            or not node.backuped  # 未备份
            or node == self.root_node  # 根节点
            or node.host_ref_counter > 0  # 主机引用计数>0
            or node.host_mamba_ref_counter > 0  # 主机Mamba引用计数>0
            or len(node.children) > 0  # 有子节点
        ):
            self.evictable_full_host_leaves.discard(node)  # 从可淘汰集合移除
            return
        self.evictable_full_host_leaves.add(node)  # 添加到可淘汰叶子集合

    def _free_device_mamba(self, node: TreeNode) -> int:  # 释放设备端Mamba状态
        if node.mamba_value is None:  # 无Mamba值
            return 0  # 返回0
        mamba_num = len(node.mamba_value)  # 获取Mamba值长度
        self.req_to_token_pool.mamba_pool.free(node.mamba_value)  # 释放Mamba池空间
        if node.mamba_lock_ref > 0:  # Mamba被锁定
            self.mamba_protected_size_ -= mamba_num  # 减少受保护大小
            node.mamba_lock_ref = 0  # 重置锁定计数
        else:  # Mamba未被锁定
            self.mamba_evictable_size_ -= mamba_num  # 减少可淘汰大小
        if self.mamba_lru_list.in_list(node):  # 在LRU列表中
            self.mamba_lru_list.remove_node(node)  # 从列表中移除
        node.mamba_value = None  # 清空Mamba值
        return mamba_num  # 返回释放的Mamba数量

    def _evict_to_host(self, node: TreeNode) -> Tuple[int, int]:  # 将设备端节点降级到主机
        # GPU -> CPU demotion: node stays in tree as evicted+backuped
        # GPU -> CPU 降级：节点在树中保留为evicted+backuped状态
        assert not node.evicted, f"already evicted, {node.id=}"  # 断言未被淘汰
        assert node.backuped, f"not backuped, {node.id=}"  # 断言已备份

        num_full = len(node.value)  # 获取KV值长度

        self._record_remove_event(node, medium=StorageMedium.GPU)  # 记录移除事件
        self.cache_controller.evict_device(node.value)  # 淘汰设备端KV
        self.full_evictable_size_ -= num_full  # 更新可淘汰大小
        if self.full_lru_list.in_list(node):  # 在全量LRU列表中
            self.full_lru_list.remove_node(node)  # 从列表中移除

        mamba_num = self._free_device_mamba(node)  # 释放设备端Mamba

        node.value = None  # 清空设备端值
        self._update_leaf_status(node)  # 更新叶子状态
        self._update_full_device_leaf_status(node.parent)  # 更新父节点设备叶子状态
        return num_full, mamba_num  # 返回淘汰的KV和Mamba数量

    def _evict_regular(self, node: TreeNode) -> Tuple[int, int]:  # 淘汰未备份的设备端叶子节点
        # evict a non-backuped device leaf — free GPU KV + mamba, delete from tree
        # 淘汰未备份的设备叶子——释放GPU KV + mamba，从树中删除
        assert not node.evicted, f"already evicted, {node.id=}"  # 断言未被淘汰
        assert not node.backuped, f"backuped node, {node.id=}"  # 断言未备份
        assert len(node.children) == 0, f"non-leaf, {node.id=}"  # 断言为叶子节点

        full_num_evicted = len(node.value)  # 获取被淘汰的KV数量

        self._record_remove_event(node, medium=StorageMedium.GPU)  # 记录移除事件
        self.cache_controller.evict_device(node.value)  # 淘汰设备端KV
        self.full_evictable_size_ -= full_num_evicted  # 更新可淘汰大小
        if self.full_lru_list.in_list(node):  # 在全量LRU列表中
            self.full_lru_list.remove_node(node)  # 从列表中移除

        mamba_num_evicted = self._free_device_mamba(node)  # 释放设备端Mamba

        if node.mamba_host_value is not None:  # 有主机端Mamba值
            if self.mamba_host_lru_list.in_list(node):  # 在Mamba主机LRU列表中
                self.mamba_host_lru_list.remove_node(node)  # 从列表中移除
            self.mamba_pool_host.free(node.mamba_host_value)  # 释放主机Mamba池空间
            node.mamba_host_value = None  # 清空主机Mamba值

        node.value = None  # 清空设备端值
        self._discard_from_leaf_sets(node)  # 从叶子集合中丢弃

        parent = node.parent  # 获取父节点
        key = node.key.child_key(self.page_size)  # 获取子键
        v = parent.children.pop(key, None)  # 从父节点子节点中移除
        assert v == node, f"parent does not have child key, {key}"  # 断言移除的节点正确

        self._update_leaf_status(parent)  # 更新父节点叶子状态
        _, cascade_full_num_evicted, cascade_mamba_num_evicted = (
            self._iteratively_delete_tombstone_leaf(node)  # 级联删除墓碑叶子
        )
        return (
            full_num_evicted + cascade_full_num_evicted,  # 总淘汰KV数量
            mamba_num_evicted + cascade_mamba_num_evicted,  # 总淘汰Mamba数量
        )

    def _evict_host_leaf(self, node: TreeNode) -> int:  # 淘汰主机端叶子节点
        # evict a host-resident leaf: free host KV + mamba, delete from tree, cascade
        # 淘汰主机端叶子：释放主机KV + mamba，从树中删除，级联
        assert node.evicted, f"not evicted, {node.id=}"  # 断言已淘汰
        assert node.backuped, f"not backuped, {node.id=}"  # 断言已备份
        assert node.mamba_value is None, f"has device mamba, {node.id=}"  # 断言无设备Mamba
        assert (
            node.host_ref_counter == 0
        ), f"host kv in use, {node.id=} {node.host_ref_counter=}"  # 断言主机KV未使用
        assert (
            node.host_mamba_ref_counter == 0
        ), f"host mamba in use, {node.id=} {node.host_mamba_ref_counter=}"  # 断言主机Mamba未使用

        self._record_remove_event(node, medium=StorageMedium.CPU)  # 记录移除事件
        full_num_evicted = self.cache_controller.evict_host(node.host_value)  # 淘汰主机端KV
        node.host_value = None  # 清空主机值

        if node.mamba_host_value is not None:  # 有主机Mamba值
            if self.mamba_host_lru_list.in_list(node):  # 在Mamba主机LRU列表中
                self.mamba_host_lru_list.remove_node(node)  # 从列表中移除
            self.mamba_pool_host.free(node.mamba_host_value)  # 释放主机Mamba池空间
            node.mamba_host_value = None  # 清空主机Mamba值

        self._discard_from_leaf_sets(node)  # 从叶子集合中丢弃
        parent = node.parent  # 获取父节点
        key = node.key.child_key(self.page_size)  # 获取子键
        v = parent.children.pop(key, None)  # 从父节点子节点中移除
        assert v == node, f"parent does not have child key, {key}"  # 断言移除的节点正确

        self._update_leaf_status(parent)  # 更新父节点叶子状态
        _, cascade_full_num_evicted, _ = self._iteratively_delete_tombstone_leaf(node)  # 级联删除墓碑叶子

        return full_num_evicted + cascade_full_num_evicted  # 返回总淘汰数量

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:  # 删除墓碑叶子节点
        assert node.mamba_value is None, f"has mamba value, {node.id=}"  # 断言无设备Mamba值
        assert node.mamba_host_value is None, f"has mamba host value, {node.id=}"  # 断言无主机Mamba值
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"  # 断言无子节点
        parent = node.parent  # 获取父节点
        key = node.key.child_key(self.page_size)  # 获取子键
        v = parent.children.pop(key, None)  # 从父节点子节点中移除
        assert v == node, f"parent does not have child key, {key}"  # 断言移除的节点正确

        self._discard_from_leaf_sets(node)  # 从叶子集合中丢弃

        if (
            node.backuped  # 已备份
            and node.host_ref_counter == 0  # 主机KV未使用
            and node.host_mamba_ref_counter == 0  # 主机Mamba未使用
        ):
            self._record_remove_event(node, medium=StorageMedium.CPU)  # 记录移除事件
            self.cache_controller.evict_host(node.host_value)  # 淘汰主机端KV
            node.host_value = None  # 清空主机值

        self._update_leaf_status(parent)  # 更新父节点叶子状态

    def _iteratively_delete_tombstone_leaf(  # 迭代删除墓碑叶子节点（级联清理）
        self, node: TreeNode
    ) -> Tuple[TreeNode, int, int]:
        full_num_evicted = 0  # 淘汰的KV数量
        mamba_num_evicted = 0  # 淘汰的Mamba数量

        while len(node.parent.children) == 0:  # 父节点无其他子节点
            if node.parent == self.root_node:  # 父节点为根节点
                break  # 停止级联
            if node.parent.mamba_value is not None:  # 父节点有设备Mamba值
                break  # 停止级联
            if node.parent.mamba_host_value is not None:  # 父节点有主机Mamba值
                break  # 停止级联
            if node.parent.full_lock_ref > 0 or node.parent.mamba_lock_ref > 0:  # 父节点被锁定
                break  # 停止级联
            if (
                node.parent.host_ref_counter > 0
                or node.parent.host_mamba_ref_counter > 0
            ):  # 父节点主机引用>0
                break  # 停止级联

            parent = node.parent  # 获取父节点

            if not parent.evicted:  # 父节点在设备上
                self._record_remove_event(parent, medium=StorageMedium.GPU)  # 记录移除事件
                full_num_evicted += len(parent.value)  # 累加淘汰KV数量
                self.full_evictable_size_ -= len(parent.value)  # 更新可淘汰大小
                self.cache_controller.evict_device(parent.value)  # 淘汰设备端KV
                if self.full_lru_list.in_list(parent):  # 在全量LRU列表中
                    self.full_lru_list.remove_node(parent)  # 从列表中移除

            self._discard_from_leaf_sets(parent)  # 从叶子集合中丢弃
            self._delete_tombstone_leaf(parent)  # 删除墓碑叶子
            node = parent  # 继续向上级联

        return node, full_num_evicted, mamba_num_evicted  # 返回最终节点和淘汰数量

    def _evict_device_leaf(self, x: TreeNode) -> Tuple[int, int]:  # 淘汰设备端叶子节点（选择合适策略）
        """Evict a device leaf node, choosing the right strategy:
        淘汰设备端叶子节点，选择合适的策略：

        - backuped: demote to host via _evict_to_host (node stays in tree)
          已备份：通过_evict_to_host降级到主机（节点保留在树中）
        - not backuped + write_back: write_backup first, then demote
          未备份 + 写回模式：先写备份，再降级
        - not backuped + write_through: _evict_regular (delete from tree)
          未备份 + 写穿透模式：_evict_regular（从树中删除）
        """
        if not x.backuped:  # 未备份
            if self.cache_controller.write_policy == "write_back":  # 写回模式
                self.write_backup(x, write_back=True)  # 先写备份
                self.writing_check(write_back=True)  # 检查写回完成
                return self._evict_to_host(x)  # 降级到主机
            else:  # 写穿透模式
                return self._evict_regular(x)  # 常规淘汰
        return self._evict_to_host(x)  # 已备份，降级到主机

    def evict(self, params: EvictParams) -> EvictResult:  # 执行淘汰操作
        if self.disable:  # 缓存已禁用
            return EvictResult()  # 返回空结果

        full_num_tokens = params.num_tokens  # 需要淘汰的token数
        full_num_evicted = 0  # 已淘汰的KV数量
        mamba_num_evicted = 0  # 已淘汰的Mamba数量

        if full_num_tokens > 0:  # 需要淘汰KV
            leaves = list(self.evictable_full_device_leaves)  # 获取可淘汰叶子列表
            eviction_heap = [(n.last_access_time, n) for n in leaves]  # 构建淘汰堆
            heapq.heapify(eviction_heap)  # 堆化

            while full_num_evicted < full_num_tokens and eviction_heap:  # 未达目标且堆非空
                _, x = heapq.heappop(eviction_heap)  # 取出最久未访问的节点
                if x not in self.evictable_full_device_leaves:  # 已不可淘汰
                    continue  # 跳过

                evicted_full, evicted_mamba = self._evict_device_leaf(x)  # 淘汰节点
                full_num_evicted += evicted_full  # 累加KV数量
                mamba_num_evicted += evicted_mamba  # 累加Mamba数量

                parent = x.parent  # 获取父节点
                if parent in self.evictable_full_device_leaves:  # 父节点变为可淘汰
                    heapq.heappush(eviction_heap, (parent.last_access_time, parent))  # 加入堆

        if params.mamba_num > 0:  # 需要淘汰Mamba
            mamba_num_evicted += self.evict_mamba(params.mamba_num)  # 淘汰Mamba

        return EvictResult(
            num_tokens_evicted=full_num_evicted,  # 淘汰的KV数量
            mamba_num_evicted=mamba_num_evicted,  # 淘汰的Mamba数量
        )

    def evict_host(self, num_tokens: int):  # 淘汰主机端叶子节点
        """Evict host-resident leaf nodes: free host KV + mamba, delete from tree, cascade.
        淘汰主机端叶子节点：释放主机KV + mamba，从树中删除，级联。"""
        heap = [(n.last_access_time, n) for n in self.evictable_full_host_leaves]  # 构建淘汰堆
        heapq.heapify(heap)  # 堆化

        num_evicted = 0  # 已淘汰数量
        while num_evicted < num_tokens and heap:  # 未达目标且堆非空
            _, x = heapq.heappop(heap)  # 取出最久未访问的节点
            if x not in self.evictable_full_host_leaves:  # 已不可淘汰
                continue  # 跳过

            num_evicted += self._evict_host_leaf(x)  # 淘汰节点

            if x.parent in self.evictable_full_host_leaves:  # 父节点变为可淘汰
                heapq.heappush(heap, (x.parent.last_access_time, x.parent))  # 加入堆

    def evict_mamba_host(self, num_mamba_hosts: int) -> int:  # 淘汰主机端Mamba状态
        """Evict host mamba states.
        淘汰主机端Mamba状态。

        Internal host node: free host mamba only (tombstone).
        内部主机节点：仅释放主机Mamba（墓碑）。
        Host leaf node: same as Full host evict — _evict_host_leaf_node frees
                        host KV + mamba, deletes from tree, cascades.
        主机叶子节点：与完整主机淘汰相同——_evict_host_leaf_node释放
                      主机KV + mamba，从树中删除，级联。
        """
        if self.disable or num_mamba_hosts <= 0:  # 禁用或无需淘汰
            return 0  # 返回0

        x = self.mamba_host_lru_list.get_lru_no_lock()  # 获取LRU节点
        num_evicted = 0  # 已淘汰数量
        while num_evicted < num_mamba_hosts and self.mamba_host_lru_list.in_list(x):  # 未达目标
            x_next = self.mamba_host_lru_list.get_prev_no_lock(x)  # 获取下一个LRU节点
            if x in self.evictable_full_host_leaves:  # 叶子节点
                # Leaf: evictable_full_host_leaves guarantees both counters == 0
                # 叶子：evictable_full_host_leaves保证两个计数器都为0
                assert (
                    x.host_ref_counter == 0
                ), f"evict host leaf: host_ref_counter != 0 with {x.id=} {x.host_ref_counter=}"
                assert (
                    x.host_mamba_ref_counter == 0
                ), f"evict host leaf: host_mamba_ref_counter != 0 with {x.id=} {x.host_mamba_ref_counter=}"
                self._evict_host_leaf(x)  # 淘汰主机叶子
                num_evicted += 1  # 增加计数
            else:  # 内部节点
                # Internal host node
                # 内部主机节点
                assert (
                    x.host_mamba_ref_counter == 0
                ), f"evict host mamba internal: host_mamba_ref_counter != 0 with {x.id=} {x.host_mamba_ref_counter=}"
                self.mamba_host_lru_list.remove_node(x)  # 从LRU列表中移除
                self.mamba_pool_host.free(x.mamba_host_value)  # 释放主机Mamba池
                x.mamba_host_value = None  # 清空主机Mamba值
                num_evicted += 1  # 增加计数

            x = x_next  # 移动到下一个节点
        return num_evicted  # 返回淘汰数量

    def evict_mamba(self, mamba_num: int) -> int:  # 淘汰设备端Mamba状态
        """Evict mamba states.
        淘汰Mamba状态。

        Internal node: tombstone — free GPU mamba only, KV stays on GPU.
        内部节点：墓碑——仅释放GPU Mamba，KV保留在GPU上。
        Leaf node: same as Full evict — _evict_to_host moves KV+mamba to host,
                   node stays in tree, then cascade tombstone parent device leaves.
        叶子节点：与完整淘汰相同——_evict_to_host将KV+mamba移到主机，
                 节点保留在树中，然后级联淘汰父设备叶子。
        """
        if self.disable or mamba_num <= 0:  # 禁用或无需淘汰
            return 0  # 返回0

        x = self.mamba_lru_list.get_lru_no_lock()  # 获取LRU节点
        mamba_num_evicted = 0  # 已淘汰Mamba数量
        while mamba_num_evicted < mamba_num and self.mamba_lru_list.in_list(x):  # 未达目标
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"  # 断言有Mamba值
            assert x != self.root_node, f"root node is not evictable, {x.id=}"  # 断言非根节点
            assert x.mamba_lock_ref == 0, f"node is in use, {x.id=}"  # 断言未被锁定
            assert (
                not x.evicted
            ), f"evicted node should not be in mamba_lru_list, {x.id=}"  # 断言未淘汰

            if len(x.children) > 0:  # 内部节点
                # Internal: free device mamba only, KV stays on device (tombstone)
                # 内部节点：仅释放设备Mamba，KV保留在设备上（墓碑）
                x_next = self.mamba_lru_list.get_prev_no_lock(x)  # 获取下一个LRU节点
                mamba_num_evicted += len(x.mamba_value)  # 累加淘汰数量
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)  # 释放Mamba池空间
                self.mamba_lru_list.remove_node(x)  # 从LRU列表中移除
                self._tombstone_internal_node(x)  # 标记为墓碑节点
            else:  # 叶子节点
                # Leaf: evict KV + mamba atomically
                # 叶子：原子性地淘汰KV + mamba
                assert (
                    x.full_lock_ref == 0
                ), f"evict device leaf: full_lock_ref mismatch with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

                x_next = self.mamba_lru_list.get_prev_no_lock(x)  # 获取下一个LRU节点
                _, mamba_evicted = self._evict_device_leaf(x)  # 淘汰设备叶子
                mamba_num_evicted += mamba_evicted  # 累加淘汰数量

                if not self.mamba_lru_list.in_list(x_next):  # 下一个节点不在列表中
                    x_next = self.mamba_lru_list.get_lru_no_lock()  # 获取新的LRU节点

            x = x_next  # 移动到下一个节点

        return mamba_num_evicted  # 返回淘汰数量

    def _unevict_node(self, node: TreeNode, fresh_value: torch.Tensor):  # 反淘汰节点（恢复到设备）
        assert node.evicted, f"not evicted, {node.id=}"  # 断言已淘汰
        assert node.mamba_value is None, f"evicted node has device mamba, {node.id=}"  # 断言无设备Mamba
        n = len(fresh_value)  # 获取值长度

        node.value = fresh_value.clone()  # 设置新的设备值
        self.full_lru_list.insert_mru(node)  # 插入全量LRU列表
        self.full_evictable_size_ += n  # 更新可淘汰大小
        self._record_store_event(node, medium=StorageMedium.GPU)  # 记录存储事件

        self._update_leaf_status(node)  # 更新叶子状态
        if node.parent is not None:  # 有父节点
            self._update_leaf_status(node.parent)  # 更新父节点叶子状态

    def _insert_helper(  # 插入辅助方法
        self,
        node: TreeNode,  # 当前节点
        key: RadixKey,  # 基数键
        value,  # 值
        mamba_value,  # Mamba值
        chunked: bool = False,  # 是否分块
        prev_prefix_len: int = 0,  # 先前前缀长度
    ) -> Tuple[int, bool]:
        assert mamba_value is not None, "Mamba value should not be None here."  # 断言Mamba值非空
        node.last_access_time = get_last_access_time()  # 更新最后访问时间
        if node != self.root_node:  # 非根节点
            if not node.evicted:  # 未淘汰
                self.full_lru_list.reset_node_mru(node)  # 刷新全量LRU
            if node.mamba_value is not None:  # 有Mamba值
                self.mamba_lru_list.reset_node_mru(node)  # 刷新Mamba LRU
        if len(key) == 0:  # 键为空
            return 0, True  # 返回0

        child_key = key.child_key(self.page_size)  # 获取子键

        total_prefix_length = 0  # 总前缀匹配长度
        while len(key) > 0 and child_key in node.children.keys():  # 遍历匹配路径
            node = node.children[child_key]  # 移动到子节点
            node.last_access_time = get_last_access_time()  # 更新访问时间

            if not node.evicted:  # 未淘汰
                self.full_lru_list.reset_node_mru(node)  # 刷新全量LRU
            if node.mamba_value is not None:  # 有Mamba值
                self.mamba_lru_list.reset_node_mru(node)  # 刷新Mamba LRU

            prefix_len = node.key.match(key, page_size=self.page_size)  # 计算前缀匹配长度

            if prefix_len < len(node.key):  # 部分匹配，需要分裂
                new_node = self._split_node(node.key, node, prefix_len)  # 分裂节点
                node = new_node  # 移动到新节点

            if node.evicted:  # 节点已淘汰
                self._unevict_node(node, value[:prefix_len])  # 反淘汰恢复
            else:  # 节点在设备上
                if prev_prefix_len < total_prefix_length + prefix_len:  # 有新的匹配部分
                    start = max(0, prev_prefix_len - total_prefix_length)  # 计算释放起点
                    self.token_to_kv_pool_allocator.free(value[start:prefix_len])  # 释放重复的KV
                total_prefix_length += prefix_len  # 累加前缀长度
                self._inc_hit_count(node, chunked)  # 增加命中计数

            key = key[prefix_len:]  # 截断已匹配的键
            value = value[prefix_len:]  # 截断已匹配的值

            if len(key):  # 还有剩余键
                child_key = key.child_key(self.page_size)  # 获取下一个子键

        mamba_value_exist = False  # Mamba值是否已存在
        if len(key):  # 有剩余键，需要创建新节点
            new_node = self._add_new_node(node, key, value, mamba_value)  # 添加新节点
            self._inc_hit_count(new_node, chunked)  # 增加命中计数
        elif node.mamba_value is None:  # 无剩余键且节点无Mamba值
            node.mamba_value = mamba_value  # 设置Mamba值
            if not node.evicted:  # 未淘汰
                self.full_lru_list.reset_node_mru(node)  # 刷新全量LRU
            self.mamba_lru_list.insert_mru(node)  # 插入Mamba LRU
            self.mamba_evictable_size_ += len(mamba_value)  # 更新可淘汰大小
            node.last_access_time = get_last_access_time()  # 更新访问时间
        else:  # 节点已有Mamba值
            mamba_value_exist = True  # 标记Mamba值已存在
            if not node.evicted:  # 未淘汰
                self.full_lru_list.reset_node_mru(node)  # 刷新全量LRU
            self.mamba_lru_list.reset_node_mru(node)  # 刷新Mamba LRU
            node.last_access_time = get_last_access_time()  # 更新访问时间

        return total_prefix_length, mamba_value_exist  # 返回前缀长度和Mamba存在标志

    def _add_new_node(  # 添加新节点到基数树
        self,
        parent: TreeNode,  # 父节点
        key: RadixKey,  # 基数键
        value: torch.Tensor,  # KV值
        mamba_value: torch.Tensor,  # Mamba值
    ) -> TreeNode:
        child_key = key.child_key(self.page_size)  # 获取子键
        new_node = TreeNode()  # 创建新树节点
        new_node.parent = parent  # 设置父节点
        new_node.key = key  # 设置键
        new_node.value = value.clone()  # 设置KV值（克隆）
        new_node.mamba_value = mamba_value  # 设置Mamba值
        self.full_lru_list.insert_mru(new_node)  # 插入全量LRU列表
        self.mamba_lru_list.insert_mru(new_node)  # 插入Mamba LRU列表
        parent.children[child_key] = new_node  # 添加到父节点的子节点
        self.full_evictable_size_ += len(value)  # 更新可淘汰大小
        self.mamba_evictable_size_ += len(mamba_value)  # 更新Mamba可淘汰大小
        if self.enable_storage or self.enable_kv_cache_events:  # 启用存储或KV事件
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)  # 计算哈希值
        self._record_store_event(new_node, medium=StorageMedium.GPU)  # 记录存储事件
        self._update_full_device_leaf_status(new_node)  # 更新新节点叶子状态
        self._update_full_device_leaf_status(parent)  # 更新父节点叶子状态
        return new_node  # 返回新节点

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # 前缀匹配
        key = params.key  # 获取匹配键

        if self.disable or len(key) == 0:  # 禁用或键为空
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),  # 空设备索引
                last_device_node=self.root_node,  # 根节点
                last_host_node=self.root_node,  # 根节点
                best_match_node=self.root_node,  # 根节点
                host_hit_length=0,  # 主机命中长度为0
            )

        if self.page_size != 1:  # 非单页大小
            page_aligned_len = len(key) // self.page_size * self.page_size  # 对齐到页面
            key = key[:page_aligned_len]  # 截断键

        value, best_last_node, best_value_len = self._match_prefix_helper(key)  # 执行前缀匹配
        return self._match_post_processor(params, value, best_last_node, best_value_len)  # 后处理

    def _match_prefix_helper(  # 前缀匹配辅助方法
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """Walk tree to find best_last_node (mamba boundary).
        遍历树查找best_last_node（Mamba边界）。"""
        node = self.root_node  # 从根节点开始
        child_key = key.child_key(self.page_size)  # 获取子键

        value: List[torch.Tensor] = []  # 匹配到的值列表
        best_value_len = 0  # 最佳值长度
        best_last_node = node  # 最佳最后节点

        while len(key) > 0 and child_key in node.children.keys():  # 遍历匹配路径
            child = node.children[child_key]  # 获取子节点

            if child.evicted and not child.backuped:  # 已淘汰且未备份
                break  # 停止匹配

            if node.mamba_value is not None or node.mamba_backuped:  # 当前节点有Mamba边界
                best_value_len = len(value)  # 更新最佳值长度
                best_last_node = node  # 更新最佳最后节点

            prefix_len = child.key.match(key, page_size=self.page_size)  # 计算前缀匹配长度
            if prefix_len < len(child.key):  # 部分匹配，需要分裂
                new_node = self._split_node(child.key, child, prefix_len)  # 分裂节点
                if not new_node.evicted:  # 新节点未淘汰
                    value.append(new_node.value)  # 添加值
                node = new_node  # 移动到新节点
                break  # 停止匹配
            else:  # 完全匹配
                if not child.evicted:  # 子节点未淘汰
                    value.append(child.value)  # 添加值
                node = child  # 移动到子节点
                key = key[prefix_len:]  # 截断已匹配的键
                if len(key):  # 还有剩余键
                    child_key = key.child_key(self.page_size)  # 获取下一个子键

        if node.mamba_value is not None or node.mamba_backuped:  # 最终节点有Mamba边界
            best_value_len = len(value)  # 更新最佳值长度
            best_last_node = node  # 更新最佳最后节点

        return value, best_last_node, best_value_len  # 返回匹配结果

    def _match_post_processor(  # 前缀匹配后处理器
        self,
        params: MatchPrefixParams,  # 匹配参数
        value: List[torch.Tensor],  # 匹配值列表
        best_last_node: TreeNode,  # 最佳最后节点
        best_value_len: int,  # 最佳值长度
    ) -> MatchResult:
        cow_mamba = params.cow_mamba  # 是否写时复制Mamba
        req = params.req  # 请求对象

        # Full LRU: skip evicted nodes for full_lru_list
        # 全量LRU：跳过已淘汰的节点
        lru_node = best_last_node  # 从最佳最后节点开始
        while lru_node != self.root_node and lru_node.evicted:  # 向上查找未淘汰节点
            lru_node = lru_node.parent  # 移动到父节点
        self.full_lru_list.reset_node_and_parents_mru(lru_node, self.root_node)  # 刷新LRU
        self.mamba_lru_list.reset_node_and_parents_mru(best_last_node, self.root_node)  # 刷新Mamba LRU

        cur_time = get_last_access_time()  # 获取当前时间
        node_update = best_last_node  # 从最佳最后节点开始
        while node_update:  # 更新所有祖先的访问时间
            node_update.last_access_time = cur_time  # 设置访问时间
            cur_time -= 0.00001  # 递减时间确保顺序
            node_update = node_update.parent  # 移动到父节点

        if len(value) > best_value_len:  # 超过Mamba边界的匹配
            from sglang.srt.server_args import get_global_server_args  # 延迟导入

            mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size  # 获取Mamba缓存块大小
            mamba_cache_chunk_aligned_seqlen = (
                sum(len(v) for v in value) // mamba_cache_chunk_size
            ) * mamba_cache_chunk_size  # 对齐到块大小
            mamba_branching_seqlen = (
                mamba_cache_chunk_aligned_seqlen
                if mamba_cache_chunk_aligned_seqlen > 0
                else None
            )  # Mamba分支序列长度
        else:
            mamba_branching_seqlen = None  # 无分支

        kv_host_hit_length = 0  # 主机KV命中长度
        last_device_node = best_last_node  # 从最佳最后节点开始
        while last_device_node is not self.root_node and last_device_node.evicted:  # 向上查找设备节点
            kv_host_hit_length += len(last_device_node.host_value)  # 累加主机值长度
            last_device_node = last_device_node.parent  # 移动到父节点

        last_host_node = best_last_node  # 从最佳最后节点开始
        while last_host_node is not self.root_node and not last_host_node.backuped:  # 向上查找已备份节点
            last_host_node = last_host_node.parent  # 移动到父节点

        mamba_host_hit = (
            1 if (last_host_node.mamba_evicted and last_host_node.mamba_backuped) else 0
        )  # Mamba主机命中标志
        host_hit_length = max(kv_host_hit_length, mamba_host_hit)  # 主机命中长度

        mamba_node = best_last_node  # Mamba节点
        if cow_mamba and mamba_node.mamba_value is not None:  # 写时复制Mamba
            if req.mamba_pool_idx is None:  # 请求无Mamba池索引
                dst_index = self._alloc_with_evict(
                    self.req_to_token_pool.mamba_pool,
                    1,
                    self.evict_mamba,
                    lock_node=mamba_node,
                    error_message="Can not alloc mamba cache",
                )  # 分配Mamba缓存
                req.mamba_pool_idx = dst_index[0]  # 保存索引
            req.mamba_cow_src_index = mamba_node.mamba_value  # 设置写时复制源
            req.mamba_needs_clear = False  # 不需要清除

        value = value[:best_value_len]  # 截断到最佳值长度
        if value:  # 有匹配值
            value = torch.cat(value)  # 拼接值
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)  # 创建空张量

        return MatchResult(
            device_indices=value,  # 设备索引
            last_device_node=last_device_node,  # 最后设备节点
            last_host_node=last_host_node,  # 最后主机节点
            # TODO(ispobock): use best_match_node as start node for load_back
            # TODO(ispobock): 使用best_match_node作为load_back的起始节点
            best_match_node=last_host_node,  # 最佳匹配节点
            host_hit_length=host_hit_length,  # 主机命中长度
            mamba_branching_seqlen=mamba_branching_seqlen,  # Mamba分支序列长度
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:  # 分裂节点
        if child.evicted:  # 已淘汰节点
            return self._split_evicted_node(key, child, split_len)  # 调用已淘汰节点分裂

        self.evictable_full_device_leaves.discard(child)  # 从可淘汰集合移除

        new_node = super()._split_node(key, child, split_len)  # 调用父类分裂

        if child.backuped:  # 已备份
            new_node.host_value = child.host_value[:split_len].clone()  # 分裂主机值
            child.host_value = child.host_value[split_len:].clone()  # 更新子节点主机值

        self._update_leaf_status(new_node)  # 更新新节点叶子状态
        self._update_leaf_status(child)  # 更新子节点叶子状态

        return new_node  # 返回新节点

    def _split_evicted_node(  # 分裂已淘汰节点
        self, key: RadixKey, child: TreeNode, split_len: int
    ) -> TreeNode:
        self.evictable_full_host_leaves.discard(child)  # 从可淘汰集合移除

        new_node = TreeNode()  # 创建新节点
        new_node.children = {key[split_len:].child_key(self.page_size): child}  # 设置子节点
        new_node.parent = child.parent  # 设置父节点
        new_node.value = None  # 无设备值（已淘汰）
        new_node.mamba_value = None  # 无设备Mamba值
        new_node.full_lock_ref = child.full_lock_ref  # 继承全量锁定计数
        new_node.mamba_lock_ref = 0  # Mamba锁定计数为0
        new_node.key = child.key[:split_len]  # 设置键（截断）

        if child.backuped:  # 已备份
            new_node.host_value = child.host_value[:split_len].clone()  # 分裂主机值
            child.host_value = child.host_value[split_len:].clone()  # 更新子节点主机值

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )  # 分裂哈希值

        child.last_access_time = get_last_access_time()  # 更新子节点访问时间
        if child.mamba_value is not None:  # 子节点有Mamba值
            self.mamba_lru_list.remove_node(child)  # 从Mamba LRU移除
        child.parent = new_node  # 设置子节点的父节点
        child.key = child.key[split_len:]  # 更新子节点键
        new_node.parent.children[key.child_key(self.page_size)] = new_node  # 替换父节点的子节点
        if child.mamba_value is not None:  # 子节点有Mamba值
            self.mamba_lru_list.insert_mru(child)  # 重新插入Mamba LRU

        self._update_full_host_leaf_status(new_node)  # 更新新节点主机叶子状态
        self._update_full_host_leaf_status(child)  # 更新子节点主机叶子状态

        return new_node  # 返回新节点

    def _collect_all_nodes(self) -> list:  # 收集所有非淘汰节点
        ret = []  # 结果列表
        stack = [self.root_node]  # 使用栈遍历
        while stack:  # 栈非空
            cur = stack.pop()  # 取出节点
            if not cur.evicted:  # 未淘汰
                ret.append(cur)  # 添加到结果
            stack.extend(cur.children.values())  # 添加子节点到栈
        return ret  # 返回结果

    def _collect_mamba_nontombstone_nodes(self) -> list:  # 收集所有有Mamba值的非墓碑节点
        ret = []  # 结果列表
        stack = [self.root_node]  # 使用栈遍历
        while stack:  # 栈非空
            cur = stack.pop()  # 取出节点
            if cur.mamba_value is not None:  # 有Mamba值
                ret.append(cur)  # 添加到结果
            stack.extend(cur.children.values())  # 添加子节点到栈
        return ret  # 返回结果

    def all_values_flatten(self) -> torch.Tensor:  # 将所有设备端值展平
        values = []  # 值列表

        def _dfs(node: TreeNode):  # 深度优先遍历
            for child in node.children.values():  # 遍历子节点
                if not child.evicted:  # 未淘汰
                    values.append(child.value)  # 添加值
                _dfs(child)  # 递归遍历

        _dfs(self.root_node)  # 从根节点开始遍历
        return torch.cat(values) if values else torch.tensor([])  # 拼接或返回空张量

    def sanity_check(self):  # 健全性检查
        """Skip if async operations are pending (those nodes are still locked).
        如果有异步操作待处理则跳过（这些节点仍被锁定）。"""
        self.loading_check()  # 检查加载状态
        if self.ongoing_load_back or self.ongoing_write_through:  # 有进行中的操作
            return  # 跳过检查
        super().sanity_check()  # 调用父类健全性检查

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:  # 增加节点引用计数
        if self.disable:  # 缓存已禁用
            return IncLockRefResult(delta=0)  # 返回0增量

        delta = 0  # 引用计数增量
        if node.mamba_value is not None:  # 有Mamba值
            if node.mamba_lock_ref == 0:  # Mamba从可淘汰变为受保护
                self.mamba_evictable_size_ -= len(node.mamba_value)  # 减少可淘汰大小
                self.mamba_protected_size_ += len(node.mamba_value)  # 增加受保护大小
            node.mamba_lock_ref += 1  # 增加Mamba锁定计数

        while node != self.root_node:  # 向上遍历到根节点
            if node.evicted:  # 已淘汰节点跳过
                node = node.parent  # 移动到父节点
                continue

            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"  # 断言锁定计数非负
            if node.full_lock_ref == 0:  # 从可淘汰变为受保护
                self.full_evictable_size_ -= len(node.value)  # 减少可淘汰大小
                self.full_protected_size_ += len(node.value)  # 增加受保护大小
                delta -= len(node.value)  # 计算增量
                self.evictable_full_device_leaves.discard(node)  # 从可淘汰集合移除
            node.full_lock_ref += 1  # 增加全量锁定计数
            node = node.parent  # 移动到父节点
        return IncLockRefResult(delta=delta)  # 返回增量结果

    def dec_lock_ref(  # 减少节点引用计数
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:  # 缓存已禁用
            return DecLockRefResult(delta=0)  # 返回0增量

        delta = 0  # 引用计数增量

        if node.mamba_value is not None and node.mamba_lock_ref > 0:  # 有Mamba值且被锁定
            if node.mamba_lock_ref == 1:  # Mamba从受保护变为可淘汰
                self.mamba_evictable_size_ += len(node.mamba_value)  # 增加可淘汰大小
                self.mamba_protected_size_ -= len(node.mamba_value)  # 减少受保护大小
            node.mamba_lock_ref -= 1  # 减少Mamba锁定计数

        while node != self.root_node:  # 向上遍历到根节点
            if node.evicted:  # 已淘汰节点跳过
                node = node.parent  # 移动到父节点
                continue

            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"  # 断言锁定计数>0
            if node.full_lock_ref == 1:  # 从受保护变为可淘汰
                self.full_evictable_size_ += len(node.value)  # 增加可淘汰大小
                self.full_protected_size_ -= len(node.value)  # 减少受保护大小
                delta += len(node.value)  # 计算增量
            node.full_lock_ref -= 1  # 减少全量锁定计数
            if node.full_lock_ref == 0:  # 锁定计数归零
                self._update_full_device_leaf_status(node)  # 更新叶子状态
            node = node.parent  # 移动到父节点
        return DecLockRefResult(delta=delta)  # 返回增量结果

    # ---- L3 Support ----  # ---- L3存储支持 ----

    def shutdown(self):  # 关闭缓存，分离存储后端
        try:
            if self.enable_storage:  # 如果启用存储
                self.detach_storage_backend()  # 分离存储后端
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")  # 记录异常

    def _apply_storage_runtime_config(  # 应用存储运行时配置
        self,
        *,
        storage_backend: Optional[str],  # 存储后端类型
        prefetch_threshold: int,  # 预取阈值
        prefetch_timeout_config: PrefetchTimeoutConfig,  # 预取超时配置
        hicache_storage_pass_prefix_keys: bool,  # 是否传递前缀键
        enable_storage: bool,  # 是否启用存储
        enable_storage_metrics: bool,  # 是否启用存储指标
        extra_metric_labels: Optional[Dict[str, str]],  # 额外指标标签
    ) -> None:
        storage_metrics_collector = None  # 存储指标收集器
        if enable_storage_metrics:  # 如果启用存储指标
            labels = {  # 构建指标标签
                "storage_backend": storage_backend,  # 存储后端
                "tp_rank": self.cache_controller.tp_rank,  # TP排名
                "dp_rank": self.cache_controller.dp_rank,  # DP排名
                "pp_rank": self.cache_controller.pp_rank,  # PP排名
                "pp_size": self.cache_controller.pp_size,  # PP大小
            }
            if extra_metric_labels:  # 有额外标签
                labels.update(extra_metric_labels)  # 合并标签
            from sglang.srt.server_args import get_global_server_args  # 延迟导入

            storage_cls = resolve_collector_class(  # 解析指标收集器类
                get_global_server_args(),
                STAT_LOGGER_ROLE_STORAGE,
                StorageMetricsCollector,
            )
            storage_metrics_collector = storage_cls(labels=labels)  # 创建指标收集器

        self.enable_storage = enable_storage  # 保存存储启用标志
        self.prefetch_threshold = prefetch_threshold  # 保存预取阈值
        self.prefetch_timeout_config = prefetch_timeout_config  # 保存预取超时配置
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys  # 保存前缀键传递标志
        self.enable_storage_metrics = enable_storage_metrics  # 保存存储指标启用标志
        if self.enable_storage_metrics:  # 如果启用存储指标
            self.storage_metrics_collector = storage_metrics_collector  # 保存指标收集器
        else:
            self.storage_metrics_collector = None  # 清空指标收集器

    def attach_storage_backend(  # 附加存储后端
        self,
        storage_backend: str,  # 存储后端类型
        storage_backend_extra_config_json: Optional[str] = None,  # 额外配置JSON
        served_model_name: Optional[str] = None,  # 服务模型名称
        hicache_storage_prefetch_policy: Optional[str] = None,  # 预取策略
        hicache_write_policy: Optional[str] = None,  # 写策略
    ) -> tuple[bool, str]:
        if hicache_storage_prefetch_policy is not None:  # 验证预取策略
            allowed = ["best_effort", "wait_complete", "timeout"]  # 允许的策略列表
            if hicache_storage_prefetch_policy not in allowed:  # 无效策略
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:  # 验证写策略
            allowed = ["write_back", "write_through", "write_through_selective"]  # 允许的策略列表
            if hicache_write_policy not in allowed:  # 无效策略
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if self.enable_storage:  # 已启用存储
            current_backend = self.cache_controller.storage_backend_type  # 当前存储后端类型
            if current_backend == storage_backend:  # 相同后端
                if hicache_storage_prefetch_policy is not None:  # 更新预取策略
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                if hicache_write_policy is not None:  # 更新写策略
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )  # 已启用相同后端，策略已更新
            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )  # 不同后端，需要先分离

        if hicache_storage_prefetch_policy is not None:  # 设置预取策略
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
        if hicache_write_policy is not None:  # 设置写策略
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")  # 记录附加存储后端信息
        try:
            (  # 解析额外配置
                extra_config,
                prefetch_threshold,
                prefetch_timeout_config,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json  # 传入额外配置JSON
            )
        except Exception as e:  # 解析失败
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")  # 记录异常
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json "
                f"'{storage_backend_extra_config_json}': {e}",
            )  # 返回失败信息

        try:
            self.cache_controller.attach_storage_backend(  # 附加存储后端到缓存控制器
                storage_backend=storage_backend,  # 存储后端类型
                prefetch_threshold=prefetch_threshold,  # 预取阈值
                model_name=served_model_name,  # 模型名称
                storage_backend_extra_config=extra_config,  # 额外配置
                host_pools=self.host_pool_group.entries,  # 主机池条目
            )
        except Exception as e:  # 附加失败
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )  # 记录异常
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"  # 返回失败信息

        self._apply_storage_runtime_config(  # 应用存储运行时配置
            storage_backend=storage_backend,  # 存储后端类型
            prefetch_threshold=prefetch_threshold,  # 预取阈值
            prefetch_timeout_config=prefetch_timeout_config,  # 预取超时配置
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,  # 前缀键传递标志
            enable_storage=True,  # 启用存储
            enable_storage_metrics=self._enable_metrics_flag,  # 存储指标标志
            extra_metric_labels=self.extra_metric_labels,  # 额外指标标签
        )
        return True, "Attached HiCache storage backend successfully."  # 返回成功信息

    def detach_storage_backend(self) -> tuple:  # 分离存储后端
        try:
            self._drain_storage_control_queues_local()  # 排空本地存储控制队列
            self.cache_controller.detach_storage_backend()  # 分离缓存控制器的存储后端
        except Exception as e:  # 分离失败
            logger.exception("Failed to detach storage backend.")  # 记录异常
            return False, f"Failed to detach HiCache storage backend: {e}"  # 返回失败信息

        self._drain_storage_control_queues_local()  # 再次排空队列
        self._force_release_pending_storage_ops()  # 强制释放待处理操作

        self.enable_storage = False  # 禁用存储
        self.enable_storage_metrics = False  # 禁用存储指标
        if hasattr(self, "storage_metrics_collector"):  # 有指标收集器
            self.storage_metrics_collector = None  # 清空指标收集器
        return True, "Detached HiCache storage backend successfully."  # 返回成功信息

    def prefetch_abort(self, pool_transfers: Optional[list[PoolTransfer]]) -> None:  # 中止预取操作
        """Free any allocated mamba host slots on prefetch abort/revoke.
        在预取中止/撤销时释放已分配的Mamba主机槽。"""
        for transfer in pool_transfers or []:  # 遍历传输描述符
            if transfer.name == PoolName.MAMBA:  # Mamba池
                if transfer.host_indices is not None:  # 有主机索引
                    self.mamba_pool_host.free(transfer.host_indices)  # 释放主机Mamba池
                break  # 只处理第一个Mamba传输

    def _force_release_pending_storage_ops(self):  # 强制释放所有待处理的存储操作
        cc = self.cache_controller  # 获取缓存控制器

        try:
            for req_id, info in list(self.ongoing_prefetch.items()):  # 遍历进行中的预取
                try:
                    last_host_node, token_ids, host_indices, _operation = info  # 解包预取信息
                except Exception:  # 解包失败
                    self.ongoing_prefetch.pop(req_id, None)  # 移除无效条目
                    continue
                try:
                    if host_indices is not None:  # 有主机索引
                        cc.mem_pool_host.free(host_indices)  # 释放主机池
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )  # 记录异常
                try:
                    self.prefetch_abort(getattr(_operation, "pool_transfers", None))  # 中止预取
                except Exception:
                    logger.exception(
                        "Failed to release mamba host indices for prefetch %s", req_id
                    )  # 记录异常
                try:
                    self._release_host_node(last_host_node)  # 释放主机节点保护
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )  # 记录异常
                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)  # 减少预取占用token数
                    if cc.prefetch_tokens_occupied < 0:  # 不应为负
                        cc.prefetch_tokens_occupied = 0  # 重置为0
                except Exception:
                    pass  # 忽略异常
                self.ongoing_prefetch.pop(req_id, None)  # 移除预取条目
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")  # 记录异常

        try:
            for ack_id, entry in list(self.ongoing_backup.items()):  # 遍历进行中的备份
                try:
                    node, mamba_host_protected = entry  # 解包备份信息
                    self._release_host_node(node, release_mamba=mamba_host_protected)  # 释放主机节点
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )  # 记录异常
                self.ongoing_backup.pop(ack_id, None)  # 移除备份条目
        except Exception:
            logger.exception("Force release pending backup ops failed.")  # 记录异常

    def _drain_storage_control_queues_local(self):  # 排空本地存储控制队列
        self._drain_storage_control_queues_impl(
            n_revoke=None,  # 不限制撤销数量
            n_backup=None,  # 不限制备份数量
            n_release=None,  # 不限制释放数量
            log_metrics=False,  # 不记录指标
        )

    def _drain_storage_control_queues_impl(  # 排空存储控制队列的实现
        self,
        n_revoke: Optional[int],  # 撤销数量限制
        n_backup: Optional[int],  # 备份数量限制
        n_release: Optional[int],  # 释放数量限制
        log_metrics: bool,  # 是否记录指标
    ):
        cc = self.cache_controller  # 获取缓存控制器

        def _drain_queue(q, limit: Optional[int]):  # 从队列中排空指定数量的项目
            drained = 0  # 已排空计数
            while limit is None or drained < limit:  # 未达限制
                try:
                    item = q.get_nowait()  # 非阻塞获取
                except Empty:  # 队列为空
                    break  # 退出
                drained += 1  # 增加计数
                yield item  # 产出项目

        def _drain_revoke():  # 排空撤销队列
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):  # 遍历撤销项
                info = self.ongoing_prefetch.pop(req_id, None)  # 获取预取信息
                if info is not None:  # 有预取信息
                    last_host_node, token_ids, _, operation = info  # 解包
                    self.prefetch_abort(operation.pool_transfers)  # 中止预取
                    self._release_host_node(last_host_node)  # 释放主机节点
                    cc.prefetch_tokens_occupied -= len(token_ids)  # 减少占用
                    if cc.prefetch_tokens_occupied < 0:  # 不应为负
                        cc.prefetch_tokens_occupied = 0  # 重置为0

        def _drain_backup():  # 排空备份确认队列
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):  # 遍历备份确认
                ack_id = operation.id  # 获取确认ID
                entry = self.ongoing_backup.pop(ack_id, None)  # 获取备份条目
                if entry is not None:  # 有条目
                    node, mamba_host_protected = entry  # 解包
                    self._release_host_node(node, release_mamba=mamba_host_protected)  # 释放主机节点
                if log_metrics and self.enable_storage_metrics:  # 需要记录指标
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )  # 记录已备份token数

        def _drain_release():  # 排空主机内存释放队列
            host_indices_list = []  # 主机索引列表
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):  # 遍历释放项
                host_indices_list.append(host_indices)  # 添加到列表
            if host_indices_list:  # 有需要释放的索引
                host_indices = torch.cat(host_indices_list, dim=0)  # 拼接索引
                cc.mem_pool_host.free(host_indices)  # 释放主机池

        _drain_revoke()  # 执行撤销排空
        _drain_backup()  # 执行备份排空
        _drain_release()  # 执行释放排空

    def _parse_storage_backend_extra_config(  # 解析存储后端额外配置
        self, storage_backend_extra_config: Optional[str]
    ):
        extra_config = {}  # 额外配置字典
        if storage_backend_extra_config:  # 有额外配置
            try:
                if storage_backend_extra_config.startswith("@"):  # 文件路径格式
                    path = storage_backend_extra_config[1:]  # 提取路径
                    ext = os.path.splitext(path)[1].lower()  # 获取文件扩展名
                    with open(path, "rb" if ext == ".toml" else "r") as f:  # 打开文件
                        if ext == ".json":  # JSON格式
                            extra_config = json.load(f)
                        elif ext == ".toml":  # TOML格式
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):  # YAML格式
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )  # 不支持的配置格式
                else:  # 内联JSON格式
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:  # 解析失败
                logger.error(f"Invalid backend extra config JSON: {e}")  # 记录错误
                raise e  # 重新抛出异常

        defaults = PrefetchTimeoutConfig()  # 获取默认超时配置
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # 提取预取阈值
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", defaults.base)  # 提取超时基础值
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", defaults.per_ki_token
        )  # 提取每千token超时值
        prefetch_timeout_max = extra_config.pop("prefetch_timeout_max", defaults.max)  # 提取超时上限
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )  # 提取前缀键传递标志

        if not isinstance(prefetch_threshold, int):  # 验证预取阈值类型
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):  # 验证超时基础值类型
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):  # 验证每千token超时值类型
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(prefetch_timeout_max, (int, float)):  # 验证超时上限类型
            raise ValueError(
                f"prefetch_timeout_max must be number, got "
                f"{type(prefetch_timeout_max).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):  # 验证前缀键传递标志类型
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        prefetch_timeout_config = PrefetchTimeoutConfig(  # 创建预取超时配置
            base=float(prefetch_timeout_base),  # 基础超时
            per_ki_token=float(prefetch_timeout_per_ki_token),  # 每千token超时
            max=float(prefetch_timeout_max),  # 最大超时
        )

        return (
            extra_config,  # 额外配置
            prefetch_threshold,  # 预取阈值
            prefetch_timeout_config,  # 预取超时配置
            hicache_storage_pass_prefix_keys,  # 前缀键传递标志
        )

    def clear_storage_backend(self) -> bool:  # 清空存储后端
        if self.enable_storage:  # 如果启用存储
            try:
                if hasattr(self.cache_controller.storage_backend, "clear"):  # 存储后端支持清空
                    self.cache_controller.storage_backend.clear()  # 清空存储
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )  # 记录成功
                    return True  # 返回成功
                else:  # 不支持清空
                    logger.warning(
                        f"Storage backend "
                        f"{type(self.cache_controller.storage_backend).__name__} "
                        "does not support clear operation."
                    )  # 记录警告
                    return False  # 返回失败
            except Exception as e:  # 清空失败
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")  # 记录错误
                return False  # 返回失败
        else:  # 存储未启用
            logger.warning("Hierarchical cache storage backend is not enabled.")  # 记录警告
            return False  # 返回失败

    def drain_storage_control_queues(self):  # 排空存储控制队列（跨TP同步）
        cc = self.cache_controller  # 获取缓存控制器

        qsizes = torch.tensor(  # 创建队列大小张量
            [
                cc.prefetch_revoke_queue.qsize(),  # 撤销队列大小
                cc.ack_backup_queue.qsize(),  # 备份确认队列大小
                cc.host_mem_release_queue.qsize(),  # 内存释放队列大小
            ],
            dtype=torch.int,  # 整数类型
        )
        if self.tp_world_size > 1:  # 多卡并行
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )  # 全局取最小队列大小

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())  # 转换为整数
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,  # 撤销数量
            n_backup=n_backup,  # 备份数量
            n_release=n_release,  # 释放数量
            log_metrics=True,  # 记录指标
        )

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):  # 线性预取超时检查
        cfg = self.prefetch_timeout_config  # 获取超时配置
        num_tokens = len(operation.hash_value) * self.page_size  # 计算token数
        timeout = min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)  # 计算超时时间
        return time.monotonic() - operation.start_time > timeout  # 返回是否超时

    def can_terminate_prefetch(self, operation: PrefetchOperation):  # 判断是否可以终止预取
        can_terminate = True  # 默认可以终止

        if self.prefetch_stop_policy == "best_effort":  # 最佳努力策略
            return can_terminate  # 总是可以终止

        if len(operation.hash_value) == 0:  # 无哈希值
            completed = False  # 未完成
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )  # 判断是否完成

        if self.prefetch_stop_policy == "wait_complete":  # 等待完成策略
            can_terminate = completed  # 完成后才能终止
        elif self.prefetch_stop_policy == "timeout":  # 超时策略
            can_terminate = completed or self.is_prefetch_timeout(operation)  # 完成或超时后可终止
        else:
            return True  # 其他策略总是可以终止

        operation_terminated = operation.is_terminated()  # 检查操作是否已终止
        if self.tp_world_size > 1:  # 多卡并行
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )  # 创建状态张量
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )  # 全局取最大状态
            can_terminate = states[0].item() == 0  # 更新终止标志
            operation_terminated = states[1].item() == 1  # 更新终止状态
        can_terminate = can_terminate or operation_terminated  # 任何一个终止即可
        return can_terminate  # 返回是否可以终止

    def terminate_prefetch(self, req_id: str):  # 终止指定请求的预取
        if req_id not in self.ongoing_prefetch:  # 请求不在进行中
            return  # 直接返回

        _, _, _, operation = self.ongoing_prefetch[req_id]  # 获取预取操作
        if operation.host_indices is None:  # 无主机索引
            return  # 直接返回
        operation.mark_terminate()  # 标记操作为终止

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:  # 弹出预取加载的token数
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)  # 返回并移除token数

    def write_backup_storage(self, node: TreeNode):  # 将节点备份写入存储后端
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )  # 获取前缀哈希键（如果启用）
        extra_pools = self.mamba_archive_transfers(node)  # 构建Mamba归档传输描述符
        operation_id = self.cache_controller.write_storage(  # 执行存储写入
            node.host_value,  # 主机端索引
            node.key,  # 节点键
            node.hash_value,  # 节点哈希值
            prefix_keys,  # 前缀键
            extra_pools=extra_pools,  # 额外池传输
        )
        mamba_host_protected = extra_pools is not None  # Mamba主机是否被保护
        self.ongoing_backup[operation_id] = (node, mamba_host_protected)  # 记录进行中的备份
        self._protect_host_node(node, protect_mamba=mamba_host_protected)  # 保护主机节点

    def prefetch_from_storage(  # 从存储后端预取数据
        self,
        req_id: str,  # 请求ID
        last_host_node: TreeNode,  # 最后主机节点
        new_input_tokens: List[int],  # 新输入token列表
        last_hash: Optional[str] = None,  # 最后哈希值
        prefix_keys: Optional[List[str]] = None,  # 前缀键列表
    ):
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )  # 对齐到页面大小的预取长度
        new_input_tokens = new_input_tokens[:prefetch_length]  # 截断token列表
        if (
            not self.enable_storage  # 存储未启用
            or prefetch_length < self.prefetch_threshold  # 预取长度小于阈值
            or self.cache_controller.prefetch_rate_limited()  # 预取被限流
        ):
            return  # 跳过预取

        self._protect_host_node(last_host_node, protect_mamba=False)  # 保护主机节点（不保护Mamba）

        # Allocate host KV memory
        # 分配主机KV内存
        host_indices = self._alloc_with_evict(
            self.cache_controller.mem_pool_host,
            prefetch_length,
            self.evict_host,
        )  # 分配主机索引
        if host_indices is None:  # 分配失败
            self._release_host_node(last_host_node, release_mamba=False)  # 释放主机节点
            return  # 跳过预取

        # Allocate host mamba slot
        # 分配主机Mamba槽
        extra_pools = self.mamba_prefetch_alloc(new_input_tokens, last_hash)  # 分配Mamba预取槽
        if extra_pools is None:  # 分配失败
            self.cache_controller.mem_pool_host.free(host_indices)  # 释放主机KV
            self._release_host_node(last_host_node, release_mamba=False)  # 释放主机节点
            return  # 跳过预取

        # mamba is also being loaded, protect host mamba as well
        # Mamba也正在加载，同时保护主机Mamba
        last_host_node.protect_host_mamba()  # 增加主机Mamba引用
        if self.mamba_host_lru_list.in_list(last_host_node):  # 在Mamba主机LRU列表中
            self.mamba_host_lru_list.remove_node(last_host_node)  # 从列表中移除

        operation = self.cache_controller.prefetch(  # 执行预取
            req_id,  # 请求ID
            host_indices,  # 主机索引
            new_input_tokens,  # 新输入token
            last_hash,  # 最后哈希值
            prefix_keys,  # 前缀键
            extra_pools=extra_pools,  # 额外池传输
        )
        self.ongoing_prefetch[req_id] = (  # 记录进行中的预取
            last_host_node,  # 最后主机节点
            new_input_tokens,  # 新输入token
            host_indices,  # 主机索引
            operation,  # 预取操作
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)  # 增加预取占用token数

    def check_prefetch_progress(self, req_id: str) -> bool:  # 检查预取进度
        if req_id not in self.ongoing_prefetch:  # 请求不在进行中
            return True  # 已完成

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]  # 获取预取信息

        if operation.host_indices is None:  # 无主机索引
            return True  # 已完成

        if not self.can_terminate_prefetch(operation):  # 不能终止
            return False  # 仍在进行

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )  # 终止预取

        min_completed_tokens = completed_tokens  # 最小完成token数
        if self.tp_world_size > 1:  # 多卡并行
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )  # 创建张量
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )  # 全局取最小完成数
            min_completed_tokens = completed_tokens_tensor.item()  # 转换为整数

        mamba_host_indices = None  # Mamba主机索引
        mamba_loaded = False  # Mamba是否加载
        for transfer in operation.pool_transfers or []:  # 遍历传输描述符
            if transfer.name == PoolName.MAMBA:  # Mamba池
                mamba_host_indices = transfer.host_indices  # 获取Mamba主机索引
                mamba_loaded = (
                    operation.pool_storage_result.extra_pool_hit_pages.get(
                        PoolName.MAMBA, 0
                    )
                    >= 1
                )  # 判断Mamba是否加载成功
                break  # 退出循环

        fetched_token_ids = token_ids[:min_completed_tokens]  # 获取已获取的token ID
        written_indices = host_indices[:min_completed_tokens]  # 获取已写入的索引
        matched_length = self._insert_helper_host(  # 将预取结果插入主机端树
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids,
                extra_key=last_host_node.key.extra_key,
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            mamba_host_indices,
            mamba_loaded,
        )

        # Free host KV memory: matched portion is already in tree, tail was unused
        # 释放主机KV内存：匹配部分已在树中，尾部未使用
        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])  # 释放匹配部分
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )  # 追加到内存释放队列

        # Free mamba host slot if it wasn't inserted into the tree
        # 如果Mamba主机槽未插入树中则释放
        if mamba_host_indices is not None:  # 有Mamba主机索引
            inserted_new = matched_length < min_completed_tokens  # 是否插入了新节点
            if not inserted_new or not mamba_loaded:  # 未插入新节点或Mamba未加载
                self.mamba_pool_host.free(mamba_host_indices)  # 释放Mamba主机池

        self._release_host_node(last_host_node)  # 释放主机节点保护
        del self.ongoing_prefetch[req_id]  # 移除预取条目
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)  # 减少预取占用数

        loaded_from_storage = min_completed_tokens - matched_length  # 从存储加载的token数
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage  # 记录加载数

        if self.enable_storage_metrics:  # 如果启用存储指标
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)  # 记录预取token数
        if loaded_from_storage > 0 and operation.pool_transfers:  # 有加载且有传输
            logger.debug(
                "HiCache mamba prefetch completed for request %s: prefetched_tokens=%s mamba_states=%s",
                req_id,  # 请求ID
                loaded_from_storage,  # 预取token数
                int(mamba_loaded),  # Mamba加载标志
            )

        return True  # 返回完成

    def _insert_helper_host(  # 主机端插入辅助方法
        self,
        node: TreeNode,  # 当前节点
        key: RadixKey,  # 基数键
        host_value,  # 主机值
        hash_value,  # 哈希值
        mamba_host_value: Optional[torch.Tensor] = None,  # Mamba主机值
        mamba_loaded: bool = False,  # Mamba是否加载
    ):
        node.last_access_time = get_last_access_time()  # 更新访问时间
        if len(key) == 0:  # 键为空
            return 0  # 返回0

        child_key = key.child_key(self.page_size)  # 获取子键

        matched_length = 0  # 匹配长度
        while len(key) > 0 and child_key in node.children.keys():  # 遍历匹配路径
            node = node.children[child_key]  # 移动到子节点
            node.last_access_time = get_last_access_time()  # 更新访问时间
            if node != self.root_node and node.mamba_value is not None:  # 非根节点且有Mamba值
                self.mamba_lru_list.reset_node_mru(node)  # 刷新Mamba LRU
            prefix_len = node.key.match(key, page_size=self.page_size)  # 计算前缀匹配长度

            key = key[prefix_len:]  # 截断已匹配的键
            host_value = host_value[prefix_len:]  # 截断已匹配的主机值
            hash_value = hash_value[prefix_len // self.page_size :]  # 截断哈希值
            matched_length += prefix_len  # 累加匹配长度

            if prefix_len < len(node.key):  # 部分匹配，需要分裂
                new_node = self._split_node(node.key, node, prefix_len)  # 分裂节点
                node = new_node  # 移动到新节点

            if len(key):  # 还有剩余键
                child_key = key.child_key(self.page_size)  # 获取下一个子键

        leaf_node: Optional[TreeNode] = None  # 新创建的叶子节点
        if len(key):  # 有剩余键，创建新节点
            new_node = TreeNode()  # 创建新节点
            new_node.parent = node  # 设置父节点
            new_node.key = key  # 设置键
            new_node.value = None  # 无设备值
            new_node.mamba_value = None  # 无设备Mamba值
            new_node.host_value = host_value.clone()  # 设置主机值
            new_node.hash_value = hash_value  # 设置哈希值
            node.children[child_key] = new_node  # 添加到父节点子节点
            leaf_node = new_node  # 记录叶子节点
            self._update_full_host_leaf_status(new_node)  # 更新叶子状态
            self._update_full_host_leaf_status(node)  # 更新父节点叶子状态
            self._record_store_event(new_node, medium=StorageMedium.CPU)  # 记录存储事件

        # Attach mamba state to the new leaf
        # 将Mamba状态附加到新叶子节点
        if leaf_node is not None and mamba_host_value is not None and mamba_loaded:  # 有叶子节点且Mamba已加载
            leaf_node.mamba_host_value = mamba_host_value.clone()  # 设置Mamba主机值
            if not self.mamba_host_lru_list.in_list(leaf_node):  # 不在LRU列表中
                self.mamba_host_lru_list.insert_mru(leaf_node)  # 插入LRU列表
        return matched_length  # 返回匹配长度

    def release_aborted_request(self, rid: str):  # 释放中止请求的资源
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)  # 移除预取加载计数

        if rid not in self.ongoing_prefetch:  # 请求不在进行中
            return  # 直接返回

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]  # 获取预取信息
        if operation.host_indices is None:  # 无主机索引
            return  # 直接返回

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)  # 终止预取
        if self.tp_world_size > 1:  # 多卡并行
            torch.distributed.barrier(group=self.tp_group)  # 同步屏障
        self._release_host_node(last_host_node)  # 释放主机节点
        del self.ongoing_prefetch[rid]  # 移除预取条目
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])  # 追加到内存释放队列
        self.prefetch_abort(operation.pool_transfers)  # 中止预取的Mamba分配
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)  # 减少预取占用数

    def _flush_pending_storage_backups_before_reset(self) -> None:  # 在重置前刷新待处理的存储备份
        if not self.enable_storage:  # 存储未启用
            return  # 直接返回

        self.writing_check(write_back=True)  # 检查写回完成
        deadline = time.monotonic() + 30.0  # 设置30秒超时
        while time.monotonic() < deadline:  # 等待备份完成
            self.drain_storage_control_queues()  # 排空存储控制队列
            backup_qsize = self.cache_controller.backup_queue.qsize()  # 备份队列大小
            ack_backup_qsize = self.cache_controller.ack_backup_queue.qsize()  # 确认队列大小
            ongoing_backup = len(self.ongoing_backup)  # 进行中的备份数
            ongoing_write = len(self.ongoing_write_through)  # 进行中的写穿透数
            if (
                backup_qsize == 0  # 备份队列为空
                and ack_backup_qsize == 0  # 确认队列为空
                and ongoing_backup == 0  # 无进行中的备份
                and ongoing_write == 0  # 无进行中的写穿透
            ):
                return  # 所有备份已完成
            time.sleep(0.05)  # 等待50ms

        logger.warning(
            "Timed out waiting for HiCache storage backups to drain before reset: "
            "ongoing_write=%s ongoing_backup=%s backup_queue=%s ack_backup_queue=%s",
            len(self.ongoing_write_through),  # 进行中的写穿透数
            len(self.ongoing_backup),  # 进行中的备份数
            self.cache_controller.backup_queue.qsize(),  # 备份队列大小
            self.cache_controller.ack_backup_queue.qsize(),  # 确认队列大小
        )  # 记录超时警告

    def _alloc_with_evict(  # 分配内存并在不足时淘汰
        self,
        pool,  # 内存池
        size: int,  # 分配大小
        evict_fn,  # 淘汰函数
        lock_node: Optional[TreeNode] = None,  # 要锁定的节点
        error_message: Optional[str] = None,  # 错误信息
    ) -> Optional[torch.Tensor]:
        indices = pool.alloc(size)  # 尝试分配
        if indices is None:  # 分配失败
            if lock_node is not None:  # 有要锁定的节点
                self.inc_lock_ref(lock_node)  # 增加引用
            evict_fn(size)  # 执行淘汰
            indices = pool.alloc(size)  # 重试分配
            if lock_node is not None:  # 有要锁定的节点
                self.dec_lock_ref(lock_node)  # 释放引用
        if indices is None and error_message is not None:  # 仍然失败
            raise RuntimeError(error_message)  # 抛出运行时错误
        return indices  # 返回分配的索引

    # -- mamba PoolTransfer builders (D↔H↔S) ----------------------------------
    # -- Mamba PoolTransfer构建器（设备↔主机↔存储）----------------------------------

    def mamba_backup_transfers(self, node: TreeNode) -> Optional[list[PoolTransfer]]:  # 构建Mamba D→H备份传输描述符
        # build D→H transfer descriptor for mamba state
        # 构建Mamba状态的D→H传输描述符
        if node.mamba_value is None:  # 无Mamba值
            return None  # 返回None
        return [
            PoolTransfer(
                name=PoolName.MAMBA,  # Mamba池
                host_indices=node.mamba_host_value,  # 主机索引
                device_indices=node.mamba_value,  # 设备索引
            )
        ]

    def mamba_backup_commit(  # 提交Mamba D→H备份
        self, node: TreeNode, transfers: list[PoolTransfer]
    ) -> None:
        # store auto-allocated mamba host indices into the node after D→H backup
        # D→H备份后将自动分配的Mamba主机索引存储到节点中
        if not transfers:  # 无传输
            return  # 直接返回
        host_indices = transfers[0].host_indices  # 获取主机索引
        if node.mamba_host_value is None and host_indices is not None:  # 节点无主机值且有新索引
            node.mamba_host_value = host_indices.clone()  # 设置Mamba主机值
            self.mamba_host_lru_list.insert_mru(node)  # 插入LRU列表

    def mamba_archive_transfers(self, node: TreeNode) -> Optional[list[PoolTransfer]]:  # 构建Mamba H→存储归档传输描述符
        # build H→Storage transfer descriptor for mamba state
        # 构建Mamba状态的H→存储传输描述符
        if node.mamba_host_value is None or not node.hash_value:  # 无主机值或无哈希值
            return None  # 返回None
        return [
            PoolTransfer(
                name=PoolName.MAMBA,  # Mamba池
                host_indices=node.mamba_host_value,  # 主机索引
                keys=[node.hash_value[-1]],  # 最后一个哈希值作为键
                hit_policy=PoolHitPolicy.TRAILING_PAGES,  # 尾部页面命中策略
            )
        ]

    def mamba_prefetch_alloc(  # 分配Mamba预取主机槽并构建存储→H传输描述符
        self,
        token_ids: List[int],  # token ID列表
        last_hash: Optional[str],  # 最后哈希值
    ) -> Optional[list[PoolTransfer]]:
        # allocate a mamba host slot and build Storage→H transfer descriptor
        # 分配Mamba主机槽并构建存储→H传输描述符
        if not token_ids:  # 无token
            return None  # 返回None
        host_indices = self._alloc_with_evict(
            self.mamba_pool_host, 1, self.evict_mamba_host
        )  # 分配Mamba主机槽
        if host_indices is None:  # 分配失败
            return None  # 返回None
        # placeholder key; I/O thread replaces with correct hash after hit query
        # 占位键；I/O线程在命中查询后替换为正确的哈希
        return [
            PoolTransfer(
                name=PoolName.MAMBA,  # Mamba池
                host_indices=host_indices,  # 主机索引
                keys=["__placeholder__"],  # 占位键
                hit_policy=PoolHitPolicy.TRAILING_PAGES,  # 尾部页面命中策略
            )
        ]

    def mamba_restore_transfers(  # 构建Mamba H→D恢复传输描述符
        self,
        last_hit_node: TreeNode,  # 最后命中节点
        nodes_to_restore: list[TreeNode],  # 需要恢复的节点列表
        req,  # 请求对象
    ) -> Optional[list[PoolTransfer]]:
        # build H→D transfer descriptors for mamba state
        # 构建Mamba状态的H→D传输描述符
        backed_up_host_indices: list[torch.Tensor] = []  # 已备份的主机索引列表
        for node in nodes_to_restore:  # 遍历需要恢复的节点
            if not node.mamba_backuped:  # 未备份Mamba
                continue  # 跳过
            backed_up_host_indices.append(node.mamba_host_value)  # 添加主机索引

        transfers: list[PoolTransfer] = []  # 传输描述符列表
        if backed_up_host_indices:  # 有已备份的索引
            transfers.append(
                PoolTransfer(
                    name=PoolName.MAMBA,  # Mamba池
                    host_indices=torch.cat(backed_up_host_indices),  # 拼接主机索引
                    device_indices=None,  # 设备索引待分配
                )
            )

        if (
            req is not None  # 有请求
            and last_hit_node in nodes_to_restore  # 最后命中节点需要恢复
            and last_hit_node.mamba_host_value is not None  # 有Mamba主机值
        ):
            if req.mamba_pool_idx is None:  # 请求无Mamba池索引
                req.mamba_pool_idx = self._alloc_with_evict(
                    self.req_to_token_pool.mamba_pool,
                    len(last_hit_node.mamba_host_value),
                    self.evict_mamba,
                    lock_node=last_hit_node,
                    error_message="Cannot alloc request mamba cache for host load back",
                )[0]  # 分配请求Mamba缓存
            transfers.append(
                PoolTransfer(
                    name=PoolName.MAMBA,  # Mamba池
                    host_indices=last_hit_node.mamba_host_value,  # 主机索引
                    device_indices=req.mamba_pool_idx.unsqueeze(0),  # 设备索引
                )
            )

        return transfers if transfers else None  # 返回传输列表或None

    def mamba_restore_commit(  # 提交Mamba H→D恢复
        self,
        restored_nodes: list[TreeNode],  # 已恢复的节点列表
        transfers: Optional[list[PoolTransfer]],  # 传输描述符列表
    ) -> None:
        # write back controller-allocated device indices after H→D restore
        # H→D恢复后写回控制器分配的设备索引
        if not restored_nodes or not transfers or transfers[0].device_indices is None:  # 无恢复数据
            return  # 直接返回
        device_indices = transfers[0].device_indices  # 获取设备索引
        offset = 0  # 偏移量
        for node in restored_nodes:  # 遍历已恢复的节点
            count = len(node.mamba_host_value)  # 获取Mamba主机值长度
            node.mamba_value = device_indices[offset : offset + count].clone()  # 设置设备Mamba值
            offset += count  # 更新偏移量
