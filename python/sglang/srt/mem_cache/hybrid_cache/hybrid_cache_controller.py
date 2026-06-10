# 混合缓存控制器模块 - 实现分层KV缓存的读写、预取和存储后端控制
# 本文件扩展了基础HiCacheController，支持多池（KV、SWA、Mamba等）的混合传输操作，
# 包含CacheOperation、StorageOperation、PrefetchOperation等操作类，
# 以及HybridCacheController控制器，负责设备与主机之间的数据传输管理

from __future__ import annotations # 启用延迟类型注解求值

import json # 导入JSON处理模块
import logging # 导入日志模块
import os # 导入操作系统模块
import threading # 导入线程模块
import time # 导入时间模块
from queue import Queue # 导入队列
from typing import TYPE_CHECKING, Any, Callable, List, Optional # 导入类型注解

import torch # 导入PyTorch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation # 导入基础缓存操作类
from sglang.srt.managers.cache_controller import ( # 导入HiCache确认
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import ( # 导入HiCache控制器基类
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import ( # 导入层级完成计数器
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import ( # 导入基础存储操作类
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import ( # 导入HiCache存储相关类
    HiCacheStorageExtraInfo, # HiCache存储额外信息
    PoolHitPolicy, # 池命中策略
    PoolName, # 池名称枚举
    PoolTransfer, # 池传输对象
    PoolTransferResult, # 池传输结果
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry # 导入池条目
from sglang.srt.utils import get_device_module # 导入获取设备模块工具

if TYPE_CHECKING: # 类型检查时导入
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator # 导入基础Token到KV池分配器

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器
device_module = get_device_module() # 获取设备模块（CUDA/HIP等）


class CacheOperation(BaseCacheOperation): # 缓存操作类，扩展基础缓存操作，增加池传输支持
    def __init__( # 初始化方法
        self, # 自身实例
        host_indices: torch.Tensor, # 主机索引
        device_indices: torch.Tensor, # 设备索引
        node_id: int, # 节点ID
        priority: Optional[int] = None, # 优先级（可选）
        pool_transfers: Optional[list[PoolTransfer]] = None, # 池传输列表（可选）
    ):
        super().__init__(host_indices, device_indices, node_id, priority) # 调用父类初始化
        self.pool_transfers = pool_transfers # 保存池传输列表

    @staticmethod
    def merge_pool_transfers( # 合并多个操作中的池传输
        ops: List[CacheOperation], # 缓存操作列表
    ) -> Optional[list[PoolTransfer]]: # 返回合并后的池传输列表或None
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {} # 按池名称和源池分组
        for op in ops: # 遍历所有操作
            for t in op.pool_transfers or []: # 遍历操作中的每个池传输
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t) # 按分组键归类
        if not grouped: # 如果没有池传输
            return None # 返回None

        def cat_or_none(tensors): # 拼接张量列表，过滤None值
            parts = [x for x in tensors if x is not None] # 过滤非None张量
            return torch.cat(parts) if parts else None # 拼接或返回None

        return [ # 返回合并后的池传输列表
            PoolTransfer( # 创建新的池传输对象
                name=ts[0].name, # 使用第一个传输的名称
                host_indices=cat_or_none(t.host_indices for t in ts), # 合并主机索引
                device_indices=cat_or_none(t.device_indices for t in ts), # 合并设备索引
                keys=[k for t in ts if t.keys for k in t.keys] or None, # 合并键列表
                hit_policy=ts[0].hit_policy, # 使用第一个传输的命中策略
                indices_from_pool=ts[0].indices_from_pool, # 使用第一个传输的源池
            )
            for ts in grouped.values() # 遍历每组传输
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation: # 合并多个缓存操作为一个操作
        if len(ops) == 1: # 如果只有一个操作
            return ops[0] # 直接返回
        host_indices = torch.cat([op.host_indices for op in ops]) # 合并主机索引
        device_indices = torch.cat([op.device_indices for op in ops]) # 合并设备索引
        node_ids = [] # 节点ID列表
        priority = min(op.priority for op in ops) # 取最低优先级
        for op in ops: # 遍历所有操作
            node_ids.extend(op.node_ids) # 收集所有节点ID
        merged = CacheOperation( # 创建合并后的操作
            host_indices, # 合并的主机索引
            device_indices, # 合并的设备索引
            -1, # 节点ID设为-1（合并操作）
            priority, # 最低优先级
            pool_transfers=CacheOperation.merge_pool_transfers(ops), # 合并池传输
        )
        merged.node_ids = node_ids # 设置合并后的节点ID列表
        return merged # 返回合并后的操作


class StorageOperation(BaseStorageOperation): # 存储操作类，扩展基础存储操作，增加池传输支持
    def __init__( # 初始化方法
        self, # 自身实例
        host_indices: torch.Tensor, # 主机索引
        token_ids: List[int], # token ID列表
        last_hash: Optional[str] = None, # 上一个哈希值（可选）
        hash_value: Optional[List[str]] = None, # 哈希值列表（可选）
        prefix_keys: Optional[List[str]] = None, # 前缀键列表（可选）
        pool_transfers: Optional[list[PoolTransfer]] = None, # 池传输列表（可选）
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys) # 调用父类初始化
        self.pool_transfers = pool_transfers # 保存池传输列表
        self.pool_storage_result = PoolTransferResult.empty() # 初始化池存储结果为空


class PrefetchOperation(StorageOperation): # 预取操作类，扩展存储操作，支持中断和进度跟踪
    def __init__( # 初始化方法
        self, # 自身实例
        request_id: str, # 请求ID
        host_indices: torch.Tensor, # 主机索引
        token_ids: List[int], # token ID列表
        last_hash: Optional[str] = None, # 上一个哈希值（可选）
        prefix_keys: Optional[List[str]] = None, # 前缀键列表（可选）
        pool_transfers: Optional[list[PoolTransfer]] = None, # 池传输列表（可选）
    ):
        self.request_id = request_id # 保存请求ID
        self._lock = threading.Lock() # 创建线程锁
        self._terminated_flag = False # 终止标志
        self.start_time = time.monotonic() # 记录开始时间
        super().__init__( # 调用父类初始化
            host_indices, # 主机索引
            token_ids, # token ID列表
            last_hash, # 上一个哈希值
            prefix_keys=prefix_keys, # 前缀键列表
            pool_transfers=pool_transfers, # 池传输列表
        )

    def increment(self, num_tokens: int): # 增加已完成的token数量
        with self._lock: # 加锁
            if self._terminated_flag: # 如果已终止
                return False # 返回False
            self.completed_tokens += num_tokens # 增加已完成token数
            return True # 返回True

    def mark_terminate(self): # 标记操作为终止状态
        with self._lock: # 加锁
            self._terminated_flag = True # 设置终止标志

    def is_terminated(self) -> bool: # 检查操作是否已终止
        return self._terminated_flag # 返回终止标志


class HybridCacheController(BaseHiCacheController): # 混合缓存控制器，扩展基础HiCache控制器，支持多池混合传输
    def __init__( # 初始化方法
        self, # 自身实例
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator, # Token到KV池分配器
        mem_pool_host: Any, # 主机内存池
        page_size: int, # 页面大小
        tp_group: torch.distributed.ProcessGroup, # 张量并行进程组
        load_cache_event: threading.Event, # 加载缓存事件
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力上下文并行进程组（可选）
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None, # 注意力张量并行进程组（可选）
        write_policy: str = "write_through_selective", # 写策略，默认为选择性写通
        io_backend: str = "", # IO后端
        storage_backend: Optional[str] = None, # 存储后端（可选）
        prefetch_threshold: int = 256, # 预取阈值，默认256
        model_name: Optional[str] = None, # 模型名称（可选）
        storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
        pp_rank: int = 0, # 流水线并行排名，默认0
        pp_size: int = 1, # 流水线并行大小，默认1
        transfer_layer_num: Optional[int] = None, # 传输层数（可选）
        enable_storage_metrics: bool = False, # 是否启用存储指标，默认False
    ):
        startup_storage_backend = storage_backend # 保存初始存储后端
        self.extra_host_mem_release_queues: dict[PoolName, Queue] = {} # 额外主机内存释放队列字典
        super().__init__( # 调用父类初始化
            token_to_kv_pool_allocator=token_to_kv_pool_allocator, # Token到KV池分配器
            mem_pool_host=mem_pool_host, # 主机内存池
            page_size=page_size, # 页面大小
            tp_group=tp_group, # 张量并行进程组
            load_cache_event=load_cache_event, # 加载缓存事件
            attn_cp_group=attn_cp_group, # 注意力上下文并行进程组
            attn_tp_group=attn_tp_group, # 注意力张量并行进程组
            write_policy=write_policy, # 写策略
            io_backend=io_backend, # IO后端
            storage_backend=None, # 先不初始化存储后端
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=model_name, # 模型名称
            storage_backend_extra_config=storage_backend_extra_config, # 存储后端额外配置
            pp_rank=pp_rank, # 流水线并行排名
            pp_size=pp_size, # 流水线并行大小
            enable_storage_metrics=enable_storage_metrics, # 存储指标开关
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        # 覆盖layer_num：混合模型传输所有层（例如线性模型（KV + Mamba）），
        # 而不仅仅是full_kv_pool报告的全注意力层。
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num: # 如果指定了传输层数且与默认不同
            self.layer_num = transfer_layer_num # 覆盖层数
            self.layer_done_counter = LayerDoneCounter(self.layer_num) # 重新创建层级完成计数器

        if startup_storage_backend is not None: # 如果指定了初始存储后端
            self.attach_storage_backend( # 附加存储后端
                storage_backend=startup_storage_backend, # 存储后端类型
                prefetch_threshold=prefetch_threshold, # 预取阈值
                model_name=model_name, # 模型名称
                storage_backend_extra_config=storage_backend_extra_config, # 额外配置
                host_pools=getattr(mem_pool_host, "entries", None), # 主机池列表
            )

    def _start_storage_threads(self): # 启动存储线程
        super()._start_storage_threads() # 调用父类方法启动存储线程
        self._init_extra_host_mem_release_queues() # 初始化额外主机内存释放队列

    def attach_storage_backend( # 附加存储后端
        self, # 自身实例
        storage_backend: str, # 存储后端类型
        prefetch_threshold: int = 256, # 预取阈值
        model_name: Optional[str] = None, # 模型名称（可选）
        storage_backend_extra_config: Optional[dict] = None, # 存储后端额外配置（可选）
        host_pools: Optional[list[PoolEntry]] = None, # 主机池列表（可选）
    ):
        super().attach_storage_backend( # 调用父类方法附加存储后端
            storage_backend=storage_backend, # 存储后端类型
            prefetch_threshold=prefetch_threshold, # 预取阈值
            model_name=model_name, # 模型名称
            storage_backend_extra_config=storage_backend_extra_config, # 额外配置
        )

        for entry in host_pools or []: # 遍历主机池列表
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name) # 注册主机内存池

    @staticmethod
    def parse_storage_backend_extra_config( # 解析存储后端额外配置
        storage_backend_extra_config: Optional[str], # 存储后端额外配置字符串
    ) -> tuple[dict, int, float, float, bool]: # 返回配置字典、预取阈值、超时基数、每千token超时、是否传递前缀键
        extra_config = {} # 初始化额外配置字典
        if storage_backend_extra_config: # 如果有额外配置
            if storage_backend_extra_config.startswith("@"): # 如果以@开头，表示从文件读取
                path = storage_backend_extra_config[1:] # 获取文件路径
                ext = os.path.splitext(path)[1].lower() # 获取文件扩展名（小写）
                with open(path, "rb" if ext == ".toml" else "r") as f: # 打开文件
                    if ext == ".json": # JSON格式
                        extra_config = json.load(f) # 加载JSON
                    elif ext == ".toml": # TOML格式
                        import tomllib # 导入TOML解析器

                        extra_config = tomllib.load(f) # 加载TOML
                    elif ext in (".yaml", ".yml"): # YAML格式
                        import yaml # 导入YAML解析器

                        extra_config = yaml.safe_load(f) # 加载YAML
                    else: # 不支持的格式
                        raise ValueError( # 抛出值错误
                            f"Unsupported config file {path} (config format: {ext})" # 不支持的配置文件
                        )
            else: # 否则直接解析JSON字符串
                extra_config = json.loads(storage_backend_extra_config) # 解析JSON字符串

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256) # 提取预取阈值，默认256
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1) # 提取预取超时基数，默认1
        prefetch_timeout_per_ki_token = extra_config.pop( # 提取每千token预取超时，默认0.25
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop( # 提取是否传递前缀键，默认False
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int): # 验证预取阈值类型
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}" # 预取阈值必须是整数
            )
        if not isinstance(prefetch_timeout_base, (int, float)): # 验证超时基数类型
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}" # 超时基数必须是数字
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)): # 验证每千token超时类型
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got " # 每千token超时必须是数字
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool): # 验证前缀键传递标志类型
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got " # 前缀键传递标志必须是布尔值
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return ( # 返回解析结果
            extra_config, # 剩余配置字典
            prefetch_threshold, # 预取阈值
            float(prefetch_timeout_base), # 超时基数（浮点数）
            float(prefetch_timeout_per_ki_token), # 每千token超时（浮点数）
            hicache_storage_pass_prefix_keys, # 是否传递前缀键
        )

    def clear_storage_backend(self) -> bool: # 清空存储后端
        if not self.enable_storage: # 如果未启用存储
            logger.warning("Hierarchical cache storage backend is not enabled.") # 记录警告
            return False # 返回False
        if not hasattr(self.storage_backend, "clear"): # 如果存储后端不支持清空操作
            logger.warning( # 记录警告
                "Storage backend %s does not support clear operation.", # 存储后端不支持清空操作
                type(self.storage_backend).__name__,
            )
            return False # 返回False
        self.storage_backend.clear() # 清空存储后端
        return True # 返回True

    def _init_extra_host_mem_release_queues(self) -> None: # 初始化额外主机内存释放队列
        self.extra_host_mem_release_queues = {} # 清空释放队列字典
        entries = getattr(self.mem_pool_host, "entries", None) or [] # 获取主机池条目列表
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None) # 获取锚点条目
        for entry in entries: # 遍历所有条目
            if entry is anchor_entry or entry.is_primary_index_anchor: # 如果是锚点条目
                continue # 跳过
            self.extra_host_mem_release_queues[entry.name] = Queue() # 为非锚点条目创建释放队列

    def _append_host_mem_release_pages( # 将主机索引按页面大小分割后加入释放队列
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int # 释放队列、主机索引、页面大小
    ) -> None:
        if host_indices.numel() == 0: # 如果没有需要释放的索引
            return # 直接返回
        for page in host_indices.split(page_size): # 按页面大小分割索引
            release_queue.put(page) # 将每个页面加入释放队列

    def append_host_mem_release( # 追加主机内存释放请求
        self, # 自身实例
        host_indices: Optional[torch.Tensor] = None, # 主机索引（可选）
        extra_pools: Optional[list[PoolTransfer]] = None, # 额外池传输列表（可选）
    ):
        if host_indices is not None: # 如果有主机索引
            self._append_host_mem_release_pages( # 将主机索引加入释放队列
                self.host_mem_release_queue, # 主机内存释放队列
                host_indices, # 主机索引
                self.mem_pool_host.page_size, # 页面大小
            )
        for transfer in extra_pools or []: # 遍历额外池传输
            if transfer.host_indices is None or transfer.host_indices.numel() == 0: # 如果没有主机索引
                continue # 跳过
            entry = self.mem_pool_host.entry_map.get(transfer.name) # 获取对应的池条目
            if ( # 如果条目不存在、是锚点、或有源池指定
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue # 跳过
            release_queue = self.extra_host_mem_release_queues.get(transfer.name) # 获取对应的释放队列
            if release_queue is None: # 如果没有释放队列
                continue # 跳过
            self._append_host_mem_release_pages( # 将主机索引加入对应的释放队列
                release_queue, transfer.host_indices, entry.host_pool.page_size # 释放队列、主机索引、页面大小
            )

    def reset(self): # 重置控制器状态
        super().reset() # 调用父类重置方法
        if self.enable_storage: # 如果启用了存储
            self.host_mem_release_queue.queue.clear() # 清空主机内存释放队列
            for release_queue in self.extra_host_mem_release_queues.values(): # 遍历所有额外释放队列
                release_queue.queue.clear() # 清空释放队列
            self.prefetch_tokens_occupied = 0 # 重置预取占用token数

    def write( # 写入操作：将设备数据备份到主机
        self, # 自身实例
        device_indices: torch.Tensor, # 设备索引
        priority: Optional[int] = None, # 优先级（可选）
        node_id: int = -1, # 节点ID，默认-1
        extra_pools: Optional[list[PoolTransfer]] = None, # 额外池传输列表（可选）
    ) -> Optional[torch.Tensor]: # 返回主机索引或None
        host_indices = self.mem_pool_host.alloc(len(device_indices)) # 分配主机索引
        if host_indices is None: # 如果主机内存不足
            return None # 返回None
        pool_transfers = self._resolve_pool_transfers_allocation( # 解析池传输分配
            extra_pools, # 额外池传输
            alloc_host=True, # 分配主机索引
            kv_device_indices=device_indices, # KV设备索引
            kv_host_indices=host_indices, # KV主机索引
        )
        if pool_transfers is None and extra_pools: # 如果池传输分配失败且有额外池
            self.mem_pool_host.free(host_indices) # 释放已分配的主机索引
            return None # 返回None

        self.write_queue.append( # 将写入操作加入写入队列
            CacheOperation( # 创建缓存操作
                host_indices, # 主机索引
                device_indices, # 设备索引
                node_id, # 节点ID
                priority, # 优先级
                pool_transfers=pool_transfers or None, # 池传输列表
            )
        )
        self.start_writing() # 开始写入
        return host_indices # 返回主机索引

    def start_writing(self) -> None: # 开始执行写入操作
        if not self.write_queue: # 如果写入队列为空
            return # 直接返回
        op = CacheOperation.merge_ops(self.write_queue) # 合并写入队列中的操作
        host_indices, device_indices, resolved_pool_transfers = ( # 获取混合索引
            self.move_hybrid_indices(op) # 移动混合索引
        )
        self.write_queue.clear() # 清空写入队列
        start_event = device_module.Event() # 创建开始事件
        finish_event = device_module.Event() # 创建完成事件
        start_event.record() # 记录开始事件
        with device_module.stream(self.write_stream): # 在写入流中执行
            start_event.wait(self.write_stream) # 等待开始事件
            self.mem_pool_host.backup_from_device_all_layer( # 从设备备份所有层到主机
                self.mem_pool_device, # 设备内存池
                host_indices, # 主机索引
                device_indices, # 设备索引
                self.io_backend, # IO后端
                pool_transfers=resolved_pool_transfers, # 池传输列表
            )
            finish_event.record() # 记录完成事件
            self._record_transfer_indices_on_stream( # 在流上记录传输索引
                self.write_stream, # 写入流
                host_indices, # 主机索引
                device_indices, # 设备索引
                resolved_pool_transfers, # 池传输列表
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids)) # 将确认信息加入队列

    def load( # 加载操作：从主机加载数据到设备
        self, # 自身实例
        host_indices: torch.Tensor, # 主机索引
        priority: Optional[int] = None, # 优先级（可选）
        node_id: int = -1, # 节点ID，默认-1
        extra_pools: Optional[list[PoolTransfer]] = None, # 额外池传输列表（可选）
    ) -> Optional[torch.Tensor]: # 返回设备索引或None
        need_load_kv = host_indices.numel() > 0 # 是否需要加载KV数据

        full_allocator = getattr( # 获取完整注意力分配器
            self.mem_pool_device_allocator,
            "full_attn_allocator", # 优先获取full_attn_allocator属性
            self.mem_pool_device_allocator, # 否则使用自身
        )
        if not need_load_kv: # 如果不需要加载KV
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device) # 创建空设备索引
        else: # 否则需要加载KV
            device_indices = full_allocator.alloc(len(host_indices)) # 分配设备索引
            if device_indices is None: # 如果设备内存不足
                return None # 返回None

        pool_transfers = self._resolve_pool_transfers_allocation( # 解析池传输分配
            extra_pools, # 额外池传输
            alloc_host=False, # 分配设备索引
            kv_device_indices=device_indices, # KV设备索引
            kv_host_indices=host_indices, # KV主机索引
        )
        if pool_transfers is None and extra_pools: # 如果池传输分配失败且有额外池
            if need_load_kv: # 如果需要加载KV
                full_allocator.free(device_indices) # 释放已分配的设备索引
            return None # 返回None

        self.load_queue.append( # 将加载操作加入加载队列
            CacheOperation( # 创建缓存操作
                host_indices, # 主机索引
                device_indices, # 设备索引
                node_id, # 节点ID
                priority, # 优先级
                pool_transfers=pool_transfers or None, # 池传输列表
            )
        )
        return device_indices # 返回设备索引

    def start_loading(self) -> int: # 开始执行加载操作，返回生产者ID
        if not self.load_queue: # 如果加载队列为空
            return -1 # 返回-1
        producer_id = self.layer_done_counter.update_producer() # 更新生产者计数器
        op = CacheOperation.merge_ops(self.load_queue) # 合并加载队列中的操作
        host_indices, device_indices, resolved_pool_transfers = ( # 获取混合索引
            self.move_hybrid_indices(op) # 移动混合索引
        )
        self.load_queue.clear() # 清空加载队列
        producer_event = self.layer_done_counter.events[producer_id] # 获取生产者事件
        producer_event.start_event.record() # 记录开始事件
        with device_module.stream(self.load_stream): # 在加载流中执行
            producer_event.start_event.wait(self.load_stream) # 等待开始事件
            for i in range(self.layer_num): # 逐层加载
                self.mem_pool_host.load_to_device_per_layer( # 从主机按层加载到设备
                    self.mem_pool_device, # 设备内存池
                    host_indices, # 主机索引
                    device_indices, # 设备索引
                    i, # 当前层索引
                    self.io_backend, # IO后端
                    pool_transfers=resolved_pool_transfers, # 池传输列表
                )
                producer_event.complete(i) # 标记当前层完成
            self._record_transfer_indices_on_stream( # 在流上记录传输索引
                self.load_stream, # 加载流
                host_indices, # 主机索引
                device_indices, # 设备索引
                resolved_pool_transfers, # 池传输列表
            )
        self.ack_load_queue.append( # 将确认信息加入队列
            HiCacheAck( # 创建确认对象
                producer_event.start_event, # 开始事件
                producer_event.finish_event, # 完成事件
                op.node_ids, # 节点ID列表
            )
        )
        return producer_id # 返回生产者ID

    def _record_transfer_indices_on_stream( # 在CUDA流上记录传输索引，防止张量被提前回收
        self, # 自身实例
        stream: torch.Stream, # CUDA流
        host_indices: torch.Tensor, # 主机索引
        device_indices: torch.Tensor, # 设备索引
        pool_transfers: Optional[list[PoolTransfer]] = None, # 池传输列表（可选）
    ) -> None:
        if host_indices.is_cuda: # 如果主机索引在CUDA上
            host_indices.record_stream(stream) # 记录流引用
        if device_indices.is_cuda: # 如果设备索引在CUDA上
            device_indices.record_stream(stream) # 记录流引用
        for transfer in pool_transfers or []: # 遍历池传输
            if transfer.host_indices is not None and transfer.host_indices.is_cuda: # 如果主机索引在CUDA上
                transfer.host_indices.record_stream(stream) # 记录流引用
            if transfer.device_indices is not None and transfer.device_indices.is_cuda: # 如果设备索引在CUDA上
                transfer.device_indices.record_stream(stream) # 记录流引用

    def prefetch( # 预取操作：预取主机数据到存储后端
        self, # 自身实例
        request_id: str, # 请求ID
        host_indices: torch.Tensor, # 主机索引
        new_input_tokens: List[int], # 新输入token列表
        last_hash: Optional[str] = None, # 上一个哈希值（可选）
        prefix_keys: Optional[List[str]] = None, # 前缀键列表（可选）
        extra_pools: Optional[list[PoolTransfer]] = None, # 额外池传输列表（可选）
    ) -> PrefetchOperation: # 返回预取操作对象
        operation = PrefetchOperation( # 创建预取操作
            request_id, # 请求ID
            host_indices, # 主机索引
            new_input_tokens, # 新输入token列表
            last_hash, # 上一个哈希值
            prefix_keys=prefix_keys, # 前缀键列表
            pool_transfers=extra_pools, # 池传输列表
        )
        self.prefetch_queue.put(operation) # 将预取操作加入队列
        return operation # 返回预取操作对象

    def write_storage( # 写入存储：将主机数据备份到存储后端
        self, # 自身实例
        host_indices: torch.Tensor, # 主机索引
        token_ids: List[int], # token ID列表
        hash_value: Optional[List[str]] = None, # 哈希值列表（可选）
        prefix_keys: Optional[List[str]] = None, # 前缀键列表（可选）
        extra_pools: Optional[list[PoolTransfer]] = None, # 额外池传输列表（可选）
    ) -> int: # 返回操作ID
        operation = StorageOperation( # 创建存储操作
            host_indices, # 主机索引
            token_ids, # token ID列表
            hash_value=hash_value, # 哈希值列表
            prefix_keys=prefix_keys, # 前缀键列表
            pool_transfers=extra_pools, # 池传输列表
        )
        self.backup_queue.put(operation) # 将存储操作加入备份队列
        return operation.id # 返回操作ID

    def _storage_hit_query(self, operation) -> tuple[list[str], int]: # 存储命中查询
        last_hash = operation.last_hash # 获取上一个哈希值
        hash_value = [] # 哈希值列表
        for start in range(0, len(operation.token_ids), self.page_size): # 按页面大小遍历token
            last_hash = self.get_hash_str( # 计算当前页面的哈希值
                operation.token_ids[start : start + self.page_size], last_hash # 当前页面token和上一个哈希值
            )
            hash_value.append(last_hash) # 添加哈希值

        extra_info = HiCacheStorageExtraInfo( # 创建存储额外信息
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None # 复制前缀键列表
        )
        if operation.pool_transfers: # 如果有池传输
            hit_result = self.storage_backend.batch_exists_v2( # 批量查询存在性（V2版本）
                hash_value, operation.pool_transfers, extra_info # 哈希值、池传输、额外信息
            )
        else: # 否则使用V1版本
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info) # 批量查询KV存在性
            hit_result = PoolTransferResult( # 创建池传输结果
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={} # KV命中页面数、额外池命中页面为空
            )

        kv_hit_pages = hit_result.kv_hit_pages # 获取KV命中页面数
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages) # 更新操作的池存储结果

        if kv_hit_pages > 0 and operation.pool_transfers: # 如果有KV命中且有池传输
            self._sync_trailing_keys(operation.pool_transfers, hash_value, kv_hit_pages) # 同步尾部键

        return ( # 返回命中哈希值和命中token数
            hash_value[:kv_hit_pages], # 命中的哈希值列表
            kv_hit_pages * self.page_size, # 命中的token数=命中页面数*页面大小
        )

    def move_hybrid_indices( # 移动混合索引：将主机/设备索引从树所有者转移到控制器
        self, operation: CacheOperation # 缓存操作
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]: # 返回主机索引、设备索引、解析后的池传输
        host_indices, device_indices = self.move_indices( # 移动KV索引
            operation.host_indices, operation.device_indices # 主机索引、设备索引
        )
        resolved_pool_transfers = None # 初始化解析后的池传输
        if operation.pool_transfers: # 如果有池传输
            resolved_pool_transfers = [] # 初始化列表
            for transfer in operation.pool_transfers: # 遍历每个池传输
                transfer_host_indices, transfer_device_indices = self.move_indices( # 移动传输索引
                    transfer.host_indices, transfer.device_indices # 传输的主机索引、设备索引
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                # 保持原始PoolTransfer不变，因为树拥有的传输可能仍引用radix树主机状态。
                # 控制器只需要一个规范化的执行时副本。
                resolved_pool_transfers.append( # 添加解析后的池传输
                    PoolTransfer( # 创建新的池传输对象
                        name=transfer.name, # 池名称
                        host_indices=transfer_host_indices, # 主机索引
                        device_indices=transfer_device_indices, # 设备索引
                        keys=transfer.keys, # 键列表
                        hit_policy=transfer.hit_policy, # 命中策略
                        indices_from_pool=transfer.indices_from_pool, # 源池
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers # 返回主机索引、设备索引、解析后的池传输

    def _page_transfer(self, operation): # 页面传输：从存储后端加载额外池数据
        # Transfer extra pools
        # 传输额外池
        if operation.pool_transfers and not operation.is_terminated(): # 如果有池传输且操作未终止
            self._resolve_sidecar_derived_pool_transfers(operation) # 解析侧车派生池传输
            results = self.storage_backend.batch_get_v2(operation.pool_transfers) # 批量获取存储数据
            operation.pool_storage_result.update_extra_pool_hit_pages(results) # 更新额外池命中页面

        # Transfer kv pools
        # 传输KV池
        super()._page_transfer(operation) # 调用父类方法传输KV池

    def _page_backup(self, operation): # 页面备份：将额外池数据写入存储后端
        # Backup extra pools
        # 备份额外池
        if operation.pool_transfers: # 如果有池传输
            self._resolve_sidecar_derived_pool_transfers(operation) # 解析侧车派生池传输
            results = self.storage_backend.batch_set_v2(operation.pool_transfers) # 批量写入存储数据
            operation.pool_storage_result.update_extra_pool_hit_pages(results) # 更新额外池命中页面

        # Backup kv pools
        # 备份KV池
        super()._page_backup(operation) # 调用父类方法备份KV池

    def _resolve_sidecar_derived_pool_transfers(self, operation): # 解析侧车派生池传输：为派生池填充主机索引和键
        for transfer in operation.pool_transfers: # 遍历所有池传输
            if transfer.indices_from_pool is None: # 如果不是派生池
                continue # 跳过
            if transfer.indices_from_pool != PoolName.KV: # 如果源池不是KV
                # TODO(hzh): Support storage sidecar derived pools from other sources
                # TODO(hzh): 支持从其他来源的存储侧车派生池
                raise AssertionError( # 抛出断言错误
                    "Storage sidecar derived pool currently only supports KV-shared " # 存储侧车派生池目前仅支持KV共享索引
                    f"indices, got {transfer.name} from {transfer.indices_from_pool}."
                )
            transfer.host_indices = operation.host_indices # 使用操作的主机索引
            if transfer.keys is None: # 如果键为空
                transfer.keys = operation.hash_value # 使用操作的哈希值

    def _sync_trailing_keys( # 同步尾部键：在KV命中截断后重新对齐侧车键
        self, # 自身实例
        pool_transfers: list[PoolTransfer], # 池传输列表
        all_hashes: list[str], # 所有哈希值列表
        kv_hit_pages: int, # KV命中页面数
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.
        在KV命中截断后重新对齐尾部页面侧车键。

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        当存储命中短于原始目标前缀时，每个池传输的键必须更新为实际命中范围的最后N个哈希值，
        而不是原始目标范围的最后N个哈希值。对于mamba（N=1），这只是最后一个命中页面的哈希；
        对于SWA（N>1），它是最后N个命中页面的滑动窗口。
        """
        for transfer in pool_transfers: # 遍历所有池传输
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES: # 如果不是尾部页面策略
                continue # 跳过
            trailing_n = len(transfer.keys) if transfer.keys else 1 # 获取尾部页面数
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages] # 更新为命中范围的最后N个哈希值

    def _resolve_pool_transfers_allocation( # 解析池传输分配：自动分配主机或设备索引
        self, # 自身实例
        extra_pools: Optional[list[PoolTransfer]], # 额外池传输列表
        alloc_host: bool, # 是否分配主机索引
        kv_device_indices: Optional[torch.Tensor] = None, # KV设备索引（可选）
        kv_host_indices: Optional[torch.Tensor] = None, # KV主机索引（可选）
    ) -> Optional[list[PoolTransfer]]: # 返回解析后的池传输列表或None
        """Auto-alloc host or device indices for PoolTransfers where they are None.
        为host_indices或device_indices为None的PoolTransfer自动分配主机或设备索引。
        """
        if not extra_pools: # 如果没有额外池
            return None # 返回None
        # (pool, free_fn, indices) for atomic rollback on failure.
        # (池, 释放函数, 索引) 用于失败时的原子回滚。
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = [] # 新分配的索引列表
        derived_transfers: list[PoolTransfer] = [] # 派生池传输列表

        def rollback_allocated() -> None: # 回滚已分配的索引
            for prev_pool, prev_free_fn, prev_indices in newly_allocated: # 遍历所有已分配的索引
                prev_free_fn(prev_indices) # 调用释放函数
                if alloc_host: # 如果是主机索引
                    prev_pool.host_indices = None # 清空主机索引
                else: # 否则是设备索引
                    prev_pool.device_indices = None # 清空设备索引

        for pool in extra_pools: # 遍历所有额外池
            if pool.indices_from_pool is not None: # 如果是派生池
                derived_transfers.append(pool) # 添加到派生池列表
                continue # 跳过
            entry = self.mem_pool_host.entry_map.get(pool.name) # 获取对应的池条目
            if entry is None: # 如果条目不存在
                continue # 跳过
            if alloc_host: # 如果分配主机索引
                if pool.host_indices is not None or pool.device_indices is None: # 如果主机索引已分配或设备索引为空
                    continue # 跳过
                alloc_fn = entry.host_pool.alloc # 主机分配函数
                free_fn = entry.host_pool.free # 主机释放函数
                evict_fn = entry.host_evict_fn # 主机驱逐函数
                size = len(pool.device_indices) # 需要分配的数量
            else: # 否则分配设备索引
                if pool.device_indices is not None or pool.host_indices is None: # 如果设备索引已分配或主机索引为空
                    continue # 跳过
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                # device_alloc_fn / device_free_fn覆盖entry.device_pool的方法，
                # 用于device_pool是原始KV池（布局）而非分配器（如SWA）的池。
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc # 设备分配函数
                free_fn = entry.device_free_fn or entry.device_pool.free # 设备释放函数
                evict_fn = entry.device_evict_fn # 设备驱逐函数
                size = len(pool.host_indices) # 需要分配的数量
            indices = alloc_fn(size) # 尝试分配索引
            if indices is None and evict_fn: # 如果分配失败且有驱逐函数
                evict_fn(size) # 驱逐指定大小的数据
                indices = alloc_fn(size) # 再次尝试分配
            if indices is None: # 如果仍然分配失败
                # Atomic rollback: free everything we successfully allocated.
                # 原子回滚：释放所有成功分配的索引。
                rollback_allocated() # 执行回滚
                return None # 返回None
            if alloc_host: # 如果是主机索引
                pool.host_indices = indices # 设置主机索引
            else: # 否则是设备索引
                pool.device_indices = indices # 设置设备索引
            newly_allocated.append((pool, free_fn, indices)) # 记录新分配的索引

        # Assign indices to deferred pools from their source.
        # 为延迟的派生池分配源池的索引。
        for pool in derived_transfers: # 遍历派生池
            if pool.indices_from_pool == PoolName.KV: # 如果源池是KV
                pool.host_indices = kv_host_indices # 使用KV主机索引
                pool.device_indices = kv_device_indices # 使用KV设备索引
                continue # 跳过

            source = next( # 查找源池传输
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None # 源池不是派生池
                    and transfer.name == pool.indices_from_pool # 名称匹配
                ),
                None, # 默认None
            )
            if source is None: # 如果找不到源池
                rollback_allocated() # 执行回滚
                return None # 返回None
            pool.host_indices = source.host_indices # 使用源池主机索引
            pool.device_indices = source.device_indices # 使用源池设备索引
        return extra_pools # 返回额外池列表
