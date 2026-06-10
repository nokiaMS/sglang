# HF3FS存储后端实现模块 - HiCacheHF3FS
# 本文件实现了基于HF3FS文件系统的HiCache存储后端，用于KV缓存的磁盘换入换出
# 主要功能包括：
# 1. Hf3fsMetadataInterface - 元数据操作抽象接口
# 2. AtomicCounter - 原子计数器，用于客户端轮询选择
# 3. create_hf3fs_client - 客户端工厂函数
# 4. HiCacheHF3FS - 核心存储类，支持v1/v2批量读写、混合池存储、零拷贝、MLA模型优化等

import atexit  # 导入退出处理模块 # 退出处理库
import concurrent.futures  # 导入并发线程池模块 # 并发线程池
import json  # 导入JSON解析模块 # JSON库
import logging  # 导入日志模块 # 日志库
import os  # 导入操作系统接口模块 # 操作系统接口库
import signal  # 导入信号处理模块 # 信号处理库
import threading  # 导入线程模块 # 线程库
import time  # 导入时间模块 # 时间库
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器 # 抽象基类
from dataclasses import dataclass  # 导入数据类装饰器 # 数据类
from functools import wraps  # 导入函数装饰器工具 # 装饰器工具
from typing import Any, List, Optional, Tuple  # 导入类型注解 # 类型注解

import torch  # 导入PyTorch张量库 # PyTorch张量库

from sglang.srt.mem_cache.hicache_storage import (  # 导入HiCache存储相关类 # 导入HiCache存储类
    HiCacheStorage,  # HiCache存储基类 # 存储基类
    HiCacheStorageConfig,  # HiCache存储配置类 # 存储配置
    HiCacheStorageExtraInfo,  # HiCache存储额外信息类 # 额外信息
    PoolHitPolicy,  # 池命中策略枚举 # 命中策略
    PoolName,  # 池名称枚举 # 池名称
    PoolTransfer,  # 池传输数据类 # 池传输
    PoolTransferResult,  # 池传输结果数据类 # 池传输结果
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # 导入主机KV缓存类 # 主机KV缓存
from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient  # 导入HF3FS客户端抽象基类 # HF3FS客户端基类
from sglang.srt.observability.metrics_collector import StorageMetrics  # 导入存储指标收集器 # 存储指标

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器 # 创建日志记录器


class Hf3fsMetadataInterface(ABC):  # HF3FS元数据操作抽象接口 # 元数据抽象接口
    """Interface for HF3FS metadata operations."""  # HF3FS元数据操作接口 # 元数据操作接口

    @abstractmethod
    def initialize(  # 初始化元数据服务
        self, rank: int, num_pages: int, namespace: PoolName = PoolName.KV  # rank、页面数、命名空间 # 参数
    ) -> None:
        """Initialize the metadata service with specified number of pages."""  # 使用指定页面数初始化元数据服务 # 初始化元数据服务
        pass  # 抽象方法占位 # 占位

    @abstractmethod
    def reserve_and_allocate_page_indices(  # 预留并分配页索引
        self,
        rank: int,  # 进程rank # rank
        keys: List[Tuple[str, str]],  # 键列表 # 键列表
        namespace: PoolName = PoolName.KV,  # 命名空间 # 命名空间
    ) -> List[Tuple[bool, int]]:
        """  # 预留并分配页索引的文档字符串 # 文档
        Reserve and allocate page indices for the specified keys.  # 为指定键预留并分配页索引 # 预留分配页索引
        Args:
            rank: The rank of the process.  # 参数rank：进程的rank # 进程rank
            keys: The keys to reserve and allocate page indices for. Each tuple contains a key and the key of its prefix block.  # 参数keys：要预留分配的键，每个元组包含键及其前缀块的键 # 键列表
            namespace: The namespace (pool type) for the metadata.  # 参数namespace：元数据的命名空间（池类型） # 命名空间
        Returns:
            List[Tuple[bool, int]]: A list of tuples, where each tuple contains a boolean indicating whether the key has existed and an integer indicating the allocated page index.  # 返回值：元组列表，布尔值表示键是否已存在，整数表示分配的页索引 # 返回结果
        """
        pass  # 抽象方法占位 # 占位

    @abstractmethod
    def confirm_write(  # 确认写入操作
        self,
        rank: int,  # 进程rank # rank
        written_keys_to_confirm: List[Tuple[str, int]],  # 已写入的键和页索引 # 已写入键
        pages_to_release: List[int],  # 需要释放的页面 # 释放页面
        namespace: PoolName = PoolName.KV,  # 命名空间 # 命名空间
    ) -> None:
        """  # 确认写入的文档字符串 # 文档
        Confirm that key-value pairs have been successfully written to storage.  # 确认键值对已成功写入存储 # 确认写入
        Args:
            rank: The rank of the process.  # 参数rank：进程的rank # 进程rank
            written_keys_to_confirm: A list of tuples, where each tuple contains a key and its corresponding page index.  # 参数written_keys_to_confirm：元组列表，每个元组包含键和对应页索引 # 已写入键列表
            pages_to_release: A list of page indices to be released.  # 参数pages_to_release：需要释放的页索引列表 # 释放页面
            namespace: The namespace (pool type) for the metadata.  # 参数namespace：元数据的命名空间（池类型） # 命名空间
        """
        pass  # 抽象方法占位 # 占位

    @abstractmethod
    def get_page_indices(  # 获取页索引
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[Optional[int]]:
        """  # 获取页索引的文档字符串 # 文档
        Get page indices for the specified keys.  # 获取指定键的页索引 # 获取页索引
        Args:
            rank: The rank of the process.  # 参数rank：进程的rank # 进程rank
            keys: A list of keys.  # 参数keys：键列表 # 键列表
            namespace: The namespace (pool type) for the metadata.  # 参数namespace：元数据的命名空间（池类型） # 命名空间
        Returns:
            List[Optional[int]]: A list of integers representing the page indices for the specified keys.  # 返回值：整数列表，表示指定键的页索引 # 页索引列表
                                 If a key is not found, the corresponding index will be None.  # 如果键未找到，对应的索引为None # 未找到则为None
        """
        pass  # 抽象方法占位 # 占位

    @abstractmethod
    def delete_keys(  # 删除键
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> None:
        """Delete specified keys and their associated pages."""  # 删除指定键及其关联的页面 # 删除键和页面
        pass  # 抽象方法占位 # 占位

    @abstractmethod
    def exists(  # 检查键是否存在
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV  # rank、键列表、命名空间 # 参数
    ) -> List[bool]:
        """Check if the specified keys exist."""  # 检查指定键是否存在 # 检查键是否存在
        pass  # 抽象方法占位 # 占位

    @abstractmethod
    def clear(self, rank: int, namespace: PoolName = PoolName.KV) -> None:  # 清除指定rank的所有键值和页面分配
        """Clear all key-value pairs and page allocations for the specified rank."""  # 清除指定rank的所有键值对和页面分配 # 清除rank数据
        pass  # 抽象方法占位 # 占位


class AtomicCounter:  # 原子计数器类，用于客户端轮询选择 # 原子计数器
    def __init__(self, n: int):  # 初始化原子计数器
        assert n > 0  # 计数器范围必须大于0 # 范围检查
        self.n = n  # 计数器范围 # 范围
        self._value = 0  # 当前值 # 当前值
        self._lock = threading.Lock()  # 线程锁 # 线程锁

    def next(self) -> int:  # 获取下一个计数值
        with self._lock:  # 获取锁 # 获取锁
            current = self._value  # 保存当前值 # 当前值
            self._value = (current + 1) % self.n  # 循环递增 # 循环递增
            return current  # 返回当前值 # 返回值


def synchronized():  # 同步装饰器，用锁保护方法
    def _decorator(func):  # 内部装饰器函数 # 内部装饰器
        @wraps(func)  # 保留原函数元信息 # 保留元信息
        def wrapper(self, *args, **kwargs):  # 包装函数 # 包装函数
            with self.lock:  # 获取锁 # 获取锁
                return func(self, *args, **kwargs)  # 执行原函数并返回结果 # 执行原函数

        return wrapper  # 返回包装函数 # 返回包装函数

    return _decorator  # 返回装饰器 # 返回装饰器


def create_hf3fs_client(  # 创建HF3FS客户端的工厂函数
    path: str,  # 存储文件路径 # 文件路径
    size: int,  # 存储文件总大小 # 存储总大小
    bytes_per_page: int,  # 每页字节数 # 每页字节数
    entries: int,  # 批量操作条目数 # 批量操作条目数
    client_timeout: int,  # 客户端超时时间 # 客户端超时
    use_mock: bool = False,  # 是否使用Mock客户端 # 是否使用Mock
) -> Hf3fsClient:
    """Factory function to create appropriate HF3FS client.  # 创建合适HF3FS客户端的工厂函数 # 客户端工厂函数

    Args:
        path: File path for storage  # 参数path：存储文件路径 # 文件路径
        size: Total size of storage file  # 参数size：存储文件总大小 # 存储总大小
        bytes_per_page: Bytes per page  # 参数bytes_per_page：每页字节数 # 每页字节数
        entries: Number of entries for batch operations  # 参数entries：批量操作条目数 # 批量操作条目数
        use_mock: Whether to use mock client instead of real usrbio client  # 参数use_mock：是否使用模拟客户端而非真实usrbio客户端 # 是否Mock

    Returns:
    """  # 返回值 # 返回
    if use_mock:  # 使用Mock客户端 # 使用Mock
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsMockClient  # 导入Mock客户端类 # 导入Mock客户端

        logger.info(f"[Rank Using Hf3fsMockClient for testing")  # 记录使用Mock客户端 # 记录日志
        return Hf3fsMockClient(path, size, bytes_per_page, entries)  # 创建并返回Mock客户端 # 返回Mock客户端
    else:  # 使用真实usrbio客户端 # 使用真实客户端
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_usrbio_client import (  # 导入UsrBio客户端类 # 导入UsrBio客户端
            Hf3fsUsrBioClient,
        )

        return Hf3fsUsrBioClient(path, size, bytes_per_page, entries, client_timeout)  # 创建并返回UsrBio客户端 # 返回UsrBio客户端


@dataclass  # 数据类装饰器 # 数据类
class _PoolStorageCtx:  # 每个池的存储上下文数据类 # 池存储上下文
    """Per-pool storage context for hybrid KV cache pools."""  # 混合KV缓存池的每池存储上下文 # 混合池存储上下文

    pool_name: str  # 池名称 # 池名称
    bytes_per_page: int  # 每页字节数 # 每页字节数
    num_pages: int  # 页面总数 # 页面总数
    namespace: PoolName  # 命名空间 # 命名空间
    clients: List[Hf3fsClient]  # HF3FS客户端列表 # 客户端列表
    gb_per_page: float  # 每页GB大小 # 每页GB大小


class HiCacheHF3FS(HiCacheStorage):  # 基于HF3FS的HiCache存储后端类 # HF3FS存储后端
    """HiCache backend that stores KV cache pages in HF3FS files."""  # 将KV缓存页面存储在HF3FS文件中的HiCache后端 # HF3FS文件存储后端

    default_env_var: str = "SGLANG_HICACHE_HF3FS_CONFIG_PATH"  # 默认环境变量名 # 默认环境变量

    def __init__(  # 初始化HiCacheHF3FS实例
        self,
        rank: int,  # 进程rank # 进程rank
        file_path: str,  # 存储文件路径 # 文件路径
        file_size: int,  # 文件大小 # 文件大小
        numjobs: int,  # 并发作业数 # 并发作业数
        bytes_per_page: int,  # 每页字节数 # 每页字节数
        entries: int,  # 批量操作条目数 # 批量操作条目数
        client_timeout: int,  # 客户端超时时间 # 客户端超时
        dtype: torch.dtype,  # 数据类型 # 数据类型
        metadata_client: Hf3fsMetadataInterface,  # 元数据客户端 # 元数据客户端
        is_mla_model: bool = False,  # 是否为MLA模型 # 是否MLA
        is_page_first_layout: bool = False,  # 是否为page_first布局 # 是否page_first布局
        use_mock_client: bool = False,  # 是否使用Mock客户端 # 是否Mock客户端
        enable_storage_metrics: bool = False,  # 是否启用存储指标 # 是否启用指标
    ):
        self.rank = rank  # 进程rank # rank
        self.file_path = file_path  # 存储文件路径 # 文件路径
        self.file_size = file_size  # 文件大小 # 文件大小
        self.numjobs = numjobs  # 并发作业数 # 作业数
        self.bytes_per_page = bytes_per_page  # 每页字节数 # 每页字节数
        self.gb_per_page = bytes_per_page / (1 << 30)  # 每页GB大小 # 每页GB
        self.entries = entries  # 批量操作条目数 # 条目数
        self.client_timeout = client_timeout  # 客户端超时时间 # 超时
        self.dtype = dtype  # 数据类型 # 数据类型
        self.metadata_client = metadata_client  # 元数据客户端 # 元数据客户端
        self.is_mla_model = is_mla_model  # 是否为MLA模型 # MLA标志
        self.is_page_first_layout = is_page_first_layout  # 是否为page_first布局 # page_first标志
        self.enable_storage_metrics = enable_storage_metrics  # 是否启用存储指标 # 指标标志
        self.use_mock_client = use_mock_client  # 是否使用Mock客户端 # Mock标志
        self.numel = self.bytes_per_page // self.dtype.itemsize  # 每页元素数 # 每页元素数
        self.num_pages = self.file_size // self.bytes_per_page  # 总页面数 # 总页面数
        self.skip_backup = False  # 是否跳过备份 # 跳过备份标志
        if self.is_mla_model and self.rank != 0:  # MLA模型且非rank 0 # MLA非rank0
            self.skip_backup = True  # 跳过备份 # 跳过备份
            self.rank = 0  # 设置rank为0 # 设置rank为0

        self.is_zero_copy = False  # 是否零拷贝 # 零拷贝标志

        logger.info(  # 记录初始化信息 # 记录日志
            f"[Rank {self.rank}] HiCacheHF3FS Client Initializing: "  # 初始化信息 # 初始化信息
            f"file_path={self.file_path}, "  # 文件路径 # 文件路径
            f"file_size={self.file_size / (2 ** 30):.2f} GB, "  # 文件大小（GB） # 文件大小
            f"num_pages={self.num_pages}, "  # 页面数 # 页面数
            f"is_mla_model={self.is_mla_model}"  # 是否MLA模型 # MLA标志
        )

        self.ac = AtomicCounter(self.numjobs)  # 创建原子计数器 # 创建计数器
        self.clients = [  # 创建多个HF3FS客户端 # 创建客户端列表
            create_hf3fs_client(  # 调用工厂函数创建客户端 # 创建客户端
                self.file_path,  # 文件路径 # 文件路径
                self.file_size,  # 文件大小 # 文件大小
                self.bytes_per_page,  # 每页字节数 # 每页字节数
                self.entries,  # 条目数 # 条目数
                self.client_timeout,  # 超时时间 # 超时
                use_mock_client,  # 是否Mock # Mock标志
            )
            for _ in range(numjobs)  # 循环numjobs次 # 循环
        ]
        self.executor = concurrent.futures.ThreadPoolExecutor(  # 创建线程池 # 创建线程池
            max_workers=self.numjobs, thread_name_prefix=f"HiCacheHF3FS-Rank{self.rank}"  # 最大线程数和线程名前缀 # 线程数和前缀
        )

        self.metadata_client.initialize(self.rank, self.num_pages)  # 初始化元数据 # 初始化元数据
        self.lock = threading.RLock()  # 创建可重入锁 # 创建锁
        self._pool_storage_ctx: dict = {}  # 池存储上下文字典 # 池上下文字典

        atexit.register(self.close)  # 注册退出回调 # 注册退出回调

        signal.signal(signal.SIGINT, lambda sig, frame: self.close())  # 捕获SIGINT信号 # 捕获中断信号
        signal.signal(signal.SIGTERM, lambda sig, frame: self.close())  # 捕获SIGTERM信号 # 捕获终止信号
        signal.signal(signal.SIGQUIT, lambda sig, frame: self.close())  # 捕获SIGQUIT信号 # 捕获退出信号

        self.prefetch_pgs = []  # 预取页面数列表 # 预取页面数
        self.backup_pgs = []  # 备份页面数列表 # 备份页面数
        self.prefetch_bandwidth = []  # 预取带宽列表 # 预取带宽
        self.backup_bandwidth = []  # 备份带宽列表 # 备份带宽

    @staticmethod
    def from_env_config(  # 从环境配置创建HiCacheHF3FS实例
        bytes_per_page: int,  # 每页字节数 # 每页字节数
        dtype: torch.dtype,  # 数据类型 # 数据类型
        storage_config: HiCacheStorageConfig = None,  # 存储配置 # 存储配置
    ) -> "HiCacheHF3FS":
        """Create a HiCacheHF3FS instance from environment configuration.  # 从环境配置创建HiCacheHF3FS实例 # 从环境配置创建实例

        Environment:
            - Uses env var stored in `HiCacheHF3FS.default_env_var` to locate a JSON config.  # 使用HiCacheHF3FS.default_env_var中存储的环境变量定位JSON配置 # 使用环境变量定位配置
            - Falls back to a local single-machine config when the env var is not set.  # 环境变量未设置时回退到本地单机配置 # 回退到本地配置

        Raises:
            ValueError: If MLA Model is requested without global metadata server or required keys are missing.  # 如果请求MLA模型但缺少全局元数据服务器或必要键则抛出ValueError # 缺少必要配置时抛出异常
        """
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (  # 导入元数据客户端类 # 导入元数据客户端
            Hf3fsGlobalMetadataClient,  # 全局元数据客户端 # 全局客户端
            Hf3fsLocalMetadataClient,  # 本地元数据客户端 # 本地客户端
        )

        use_mock_client = False  # 默认不使用Mock客户端 # Mock标志
        if storage_config is not None:  # 存储配置不为空 # 检查配置
            rank, is_mla_model, is_page_first_layout = (  # 从配置中提取参数 # 提取参数
                storage_config.tp_rank,  # 张量并行rank # tp_rank
                storage_config.is_mla_model,  # 是否MLA模型 # MLA标志
                storage_config.is_page_first_layout,  # 是否page_first布局 # page_first标志
            )

            if storage_config.extra_config is not None:  # 额外配置不为空 # 检查额外配置
                use_mock_client = storage_config.extra_config.get(  # 获取Mock客户端标志 # 获取Mock标志
                    "use_mock_hf3fs_client", False  # 默认False # 默认值
                )
        else:  # 存储配置为空 # 无配置
            rank, is_mla_model, is_page_first_layout = (  # 使用默认值 # 默认值
                0,  # rank默认为0 # 默认rank
                False,  # 非MLA模型 # 默认非MLA
                False,  # 非page_first布局 # 默认非page_first
            )

        mla_unsupported_msg = f"MLA model is not supported without global metadata server, please refer to https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/hf3fs/docs/deploy_sglang_3fs_multinode.md"  # MLA不支持的消息 # MLA不支持提示

        config_path = os.getenv(HiCacheHF3FS.default_env_var)  # 获取配置文件路径环境变量 # 获取环境变量
        if not config_path:  # 环境变量未设置 # 未设置
            if is_mla_model:  # MLA模型 # 检查MLA
                raise ValueError(mla_unsupported_msg)  # 抛出不支持异常 # 抛出异常

            return HiCacheHF3FS(  # 使用默认配置创建实例 # 创建默认实例
                rank=rank,  # rank # rank
                file_path=f"/data/hicache.{rank}.bin",  # 默认文件路径 # 默认路径
                file_size=1 << 40,  # 默认1TB文件大小 # 默认1TB
                numjobs=16,  # 默认16个作业 # 默认作业数
                bytes_per_page=bytes_per_page,  # 每页字节数 # 每页字节数
                entries=8,  # 默认8个条目 # 默认条目数
                client_timeout=5,  # 默认5秒超时 # 默认超时
                dtype=dtype,  # 数据类型 # 数据类型
                metadata_client=Hf3fsLocalMetadataClient(),  # 本地元数据客户端 # 本地客户端
                is_page_first_layout=is_page_first_layout,  # 是否page_first布局 # page_first标志
                use_mock_client=use_mock_client,  # 是否Mock # Mock标志
            )

        try:  # 尝试加载配置文件 # 尝试加载
            with open(config_path, "r") as f:  # 打开配置文件 # 打开文件
                config = json.load(f)  # 加载JSON配置 # 加载配置
        except Exception as e:  # 捕获异常 # 捕获异常
            raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")  # 抛出加载失败异常 # 抛出异常

        # Check required keys (metadata_server_url is now optional)  # 检查必要键（metadata_server_url现在是可选的） # 检查必要键
        required_keys = {  # 必要键集合 # 必要键
            "file_path_prefix",  # 文件路径前缀 # 路径前缀
            "file_size",  # 文件大小 # 文件大小
            "numjobs",  # 作业数 # 作业数
            "entries",  # 条目数 # 条目数
        }
        missing_keys = required_keys - set(config.keys())  # 计算缺失的键 # 缺失键
        if missing_keys:  # 有缺失的键 # 检查缺失
            raise ValueError(f"Missing required keys in config: {missing_keys}")  # 抛出缺失键异常 # 抛出异常

        # Choose metadata client based on configuration  # 根据配置选择元数据客户端 # 选择元数据客户端
        if config.get("metadata_server_url"):  # 配置了元数据服务器URL # 检查URL
            # Use global metadata client to connect to metadata server  # 使用全局元数据客户端连接元数据服务器 # 使用全局客户端
            metadata_server_url = config["metadata_server_url"]  # 获取URL # 获取URL
            metadata_client = Hf3fsGlobalMetadataClient(metadata_server_url)  # 创建全局客户端 # 创建全局客户端

            logger.info(  # 记录日志 # 记录日志
                f"Using global metadata client with server url: {metadata_server_url}"  # 使用全局客户端的URL # 全局客户端URL
            )
        else:  # 未配置元数据服务器URL # 无URL
            # Enable MLA optimization only when using the global metadata client  # 仅在使用全局元数据客户端时启用MLA优化 # MLA仅全局客户端支持
            if is_mla_model:  # MLA模型 # 检查MLA
                raise ValueError(mla_unsupported_msg)  # 抛出不支持异常 # 抛出异常

            # Use local metadata client for single-machine deployment  # 单机部署使用本地元数据客户端 # 本地客户端
            metadata_client = Hf3fsLocalMetadataClient()  # 创建本地客户端 # 创建本地客户端

        rank_for_path = 0 if is_mla_model else rank  # MLA模型使用rank 0的路径，否则使用当前rank # 路径rank
        return HiCacheHF3FS(  # 创建HiCacheHF3FS实例 # 创建实例
            rank=rank,  # rank # rank
            # Let all ranks use the same file path for MLA model  # MLA模型让所有rank使用相同文件路径 # MLA共享文件
            file_path=f"{config['file_path_prefix']}.{rank_for_path}.bin",  # 文件路径 # 文件路径
            file_size=int(config["file_size"]),  # 文件大小 # 文件大小
            numjobs=int(config["numjobs"]),  # 作业数 # 作业数
            bytes_per_page=bytes_per_page,  # 每页字节数 # 每页字节数
            entries=int(config["entries"]),  # 条目数 # 条目数
            client_timeout=config.get("client_timeout", 5),  # 客户端超时 # 超时
            dtype=dtype,  # 数据类型 # 数据类型
            metadata_client=metadata_client,  # 元数据客户端 # 元数据客户端
            is_mla_model=is_mla_model,  # 是否MLA模型 # MLA标志
            is_page_first_layout=is_page_first_layout,  # 是否page_first布局 # page_first标志
            use_mock_client=use_mock_client,  # 是否Mock # Mock标志
            enable_storage_metrics=storage_config.enable_storage_metrics,  # 是否启用存储指标 # 指标标志
        )

    def _batch_get(  # 批量获取（预取）KV缓存页面
        self,
        keys: List[str],  # 键列表 # 键列表
        values: List[torch.Tensor],  # 目标张量列表 # 目标张量
    ) -> List[bool]:
        page_indices = self.metadata_client.get_page_indices(self.rank, keys)  # 从元数据获取页索引 # 获取页索引
        if len(page_indices) != len(keys):  # 页索引数量与键数量不匹配 # 检查数量
            logger.error(  # 记录错误日志 # 记录错误
                f"[Rank {self.rank}] HiCacheHF3FS get: page_indices length {len(page_indices)} mismatch keys length {len(keys)}."  # 长度不匹配 # 长度不匹配
            )
            return [False] * len(keys)  # 返回全部失败 # 返回失败
        batch_indices, file_offsets = [], []  # 批量索引和文件偏移量列表 # 索引和偏移
        for i, page_index in enumerate(page_indices):  # 遍历页索引 # 遍历
            if page_index is not None:  # 页索引不为空 # 检查非空
                batch_indices.append(i)  # 添加批量索引 # 添加索引
                file_offsets.append(page_index * self.bytes_per_page)  # 计算文件偏移量 # 计算偏移

        for target_location in values:  # 遍历目标张量 # 遍历张量
            assert target_location.is_contiguous()  # 断言张量是连续的 # 连续性检查
        file_results = values  # 文件读取结果直接写入目标张量 # 文件结果

        start_time = time.perf_counter()  # 记录开始时间 # 开始计时

        futures = [  # 创建异步读取任务列表 # 创建任务
            self.executor.submit(  # 提交任务到线程池 # 提交任务
                self.clients[self.ac.next()].batch_read,  # 轮询选择客户端并批量读取 # 轮询读取
                file_offsets[i : i + self.entries],  # 文件偏移量切片 # 偏移切片
                file_results[i : i + self.entries],  # 目标张量切片 # 结果切片
            )
            for i in range(0, len(batch_indices), self.entries)  # 按条目数分批 # 分批
        ]
        read_results = [result for future in futures for result in future.result()]  # 等待所有任务完成并收集结果 # 收集结果

        end_time = time.perf_counter()  # 记录结束时间 # 结束计时
        ionum = len(batch_indices)  # IO操作数量 # IO数量

        if self.enable_storage_metrics:  # 启用存储指标 # 检查指标
            self.prefetch_pgs.append(ionum)  # 记录预取页面数 # 记录页面数
            self.prefetch_bandwidth.append(  # 记录预取带宽 # 记录带宽
                ionum / (end_time - start_time) * self.gb_per_page  # 计算带宽 # 计算带宽
            )

        results = [False] * len(keys)  # 初始化结果列表 # 初始化结果
        for batch_index, read_result in zip(batch_indices, read_results):  # 遍历结果 # 遍历
            if read_result == self.bytes_per_page:  # 读取成功（字节数等于页大小） # 检查成功
                results[batch_index] = True  # 标记成功 # 标记成功
            else:  # 读取失败 # 读取失败
                logger.error(  # 记录错误日志 # 记录错误
                    f"[Rank {self.rank}] HiCacheHF3FS get {keys[batch_index]} failed"  # 读取失败 # 读取失败
                )

        return results  # 返回结果列表 # 返回结果

    def _batch_set(  # 批量设置（备份）KV缓存页面
        self,
        keys: List[str],  # 键列表 # 键列表
        values: Optional[Any] = None,  # 值列表 # 值列表
    ) -> List[bool]:
        # In MLA backend, only one rank needs to backup the KV cache  # 在MLA后端，只有一个rank需要备份KV缓存 # MLA只需一个rank备份
        if self.skip_backup:  # 跳过备份 # 检查跳过
            return True  # 直接返回成功 # 返回成功

        # Todo: Add prefix block's hash key  # 待办：添加前缀块的哈希键 # 待办：添加前缀哈希
        key_with_prefix = [(key, "") for key in keys]  # 构建键和空前缀的元组列表 # 构建键元组
        indices = self.metadata_client.reserve_and_allocate_page_indices(  # 预留并分配页索引 # 预留分配
            self.rank, key_with_prefix  # rank和键元组列表 # 参数
        )
        if len(indices) != len(keys):  # 分配索引数量不匹配 # 检查数量
            logger.error(  # 记录错误日志 # 记录错误
                f"[Rank {self.rank}] HiCacheHF3FS batch_get: mismatched lengths {len(indices)} != {len(keys)}"  # 长度不匹配 # 长度不匹配
            )
            # free allocated pages  # 释放已分配的页面 # 释放已分配页面
            if indices:  # 有已分配的索引 # 检查索引
                self.metadata_client.confirm_write(  # 确认写入以释放页面 # 确认写入
                    self.rank, [], [index[1] for index in indices]  # 空写入列表，释放所有页索引 # 释放页面
                )
            return [False] * len(keys)  # 返回全部失败 # 返回失败
        batch_indices, file_offsets, file_values = [], [], []  # 批量索引、文件偏移量、文件值列表 # 列表
        pages_to_release = []  # 需要释放的页面列表 # 释放页面列表

        for i, (value, (is_written, page_index)) in enumerate(zip(values, indices)):  # 遍历值和索引 # 遍历
            if is_written or page_index == -1:  # 已写入或页索引无效 # 检查跳过
                continue  # 跳过 # 跳过

            batch_indices.append(i)  # 添加批量索引 # 添加索引
            file_offsets.append(page_index * self.bytes_per_page)  # 计算文件偏移量 # 计算偏移
            assert value.is_contiguous()  # 断言值张量是连续的 # 连续性检查
            file_values.append(value)  # 添加文件值 # 添加值

        start_time = time.perf_counter()  # 记录开始时间 # 开始计时

        futures = [  # 创建异步写入任务列表 # 创建任务
            self.executor.submit(  # 提交任务到线程池 # 提交任务
                self.clients[self.ac.next()].batch_write,  # 轮询选择客户端并批量写入 # 轮询写入
                file_offsets[i : i + self.entries],  # 文件偏移量切片 # 偏移切片
                file_values[i : i + self.entries],  # 值张量切片 # 值切片
            )
            for i in range(0, len(batch_indices), self.entries)  # 按条目数分批 # 分批
        ]
        write_results = [  # 等待所有任务完成并收集结果 # 收集结果
            result == self.bytes_per_page  # 检查写入字节数是否等于页大小 # 检查成功
            for future in futures  # 遍历Future # 遍历
            for result in future.result()  # 遍历结果 # 遍历结果
        ]

        end_time = time.perf_counter()  # 记录结束时间 # 结束计时
        ionum = len(batch_indices)  # IO操作数量 # IO数量

        if self.enable_storage_metrics:  # 启用存储指标 # 检查指标
            self.backup_pgs.append(ionum)  # 记录备份页面数 # 记录页面数
            self.backup_bandwidth.append(  # 记录备份带宽 # 记录带宽
                ionum / (end_time - start_time) * self.gb_per_page  # 计算带宽 # 计算带宽
            )

        written_keys_to_confirm = []  # 需要确认写入的键列表 # 确认写入列表
        results = [index[0] for index in indices]  # 初始化结果为索引的是否已存在标志 # 初始化结果
        for batch_index, write_result in zip(batch_indices, write_results):  # 遍历结果 # 遍历
            key = keys[batch_index]  # 获取键 # 获取键
            page_index = indices[batch_index][1]  # 获取页索引 # 获取页索引
            if write_result:  # 写入成功 # 检查成功
                written_keys_to_confirm.append((key, page_index))  # 添加到确认列表 # 添加确认
            else:  # 写入失败 # 写入失败
                logger.error(f"[Rank {self.rank}] HiCacheHF3FS set {key} failed")  # 记录错误 # 记录错误
                pages_to_release.append(page_index)  # 添加到释放列表 # 添加释放
            results[batch_index] = write_result  # 更新结果 # 更新结果

        if len(written_keys_to_confirm) > 0 or len(pages_to_release) > 0:  # 有需要确认或释放的页面 # 检查
            self.metadata_client.confirm_write(  # 确认写入 # 确认写入
                self.rank, written_keys_to_confirm, pages_to_release  # rank、确认列表、释放列表 # 参数
            )

        return results  # 返回结果列表 # 返回结果

    def delete(self, key: str) -> None:  # 删除指定键
        self.metadata_client.delete_keys(self.rank, [key])  # 从元数据中删除键 # 删除键

    def exists(self, key: str) -> bool:  # 检查指定键是否存在
        result = self.metadata_client.exists(self.rank, [key])  # 查询键是否存在 # 查询存在
        return result[0] if result else False  # 返回结果或False # 返回结果

    def batch_exists(  # 批量检查键是否存在
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None  # 键列表和额外信息 # 参数
    ) -> int:
        factor = 1  # 因子，用于零拷贝时调整 # 因子
        if self.is_zero_copy and not self.is_mla_model:  # 零拷贝且非MLA # 检查条件
            keys = self._get_mha_zero_copy_keys(keys)  # 获取MHA零拷贝键 # 获取零拷贝键
            factor = 2  # 因子设为2（K和V分开存储） # 因子为2

        results = self.metadata_client.exists(self.rank, keys)  # 查询键是否存在 # 查询存在

        i = 0  # 计数器 # 计数器
        while i < len(keys) and results[i]:  # 遍历直到第一个不存在的键 # 遍历
            i += 1  # 递增 # 递增

        return i // factor  # 返回命中页面数 # 返回命中数

    def clear(self) -> None:  # 清除所有元数据
        try:  # 尝试清除 # 尝试
            self.metadata_client.clear(self.rank)  # 清除主KV池元数据 # 清除元数据
            for ctx in getattr(self, "_pool_storage_ctx", {}).values():  # 遍历所有池存储上下文 # 遍历池上下文
                self.metadata_client.clear(self.rank, namespace=ctx.namespace)  # 清除池元数据 # 清除池元数据
            logger.info(f"Cleared HiCacheHF3FS for rank {self.rank}")  # 记录清除成功 # 记录日志
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"Failed to clear HiCacheHF3FS: {e}")  # 记录清除失败 # 记录错误

    def close(self) -> None:  # 关闭存储并清理资源
        try:  # 尝试关闭 # 尝试
            for c in self.clients:  # 遍历所有客户端 # 遍历客户端
                c.close()  # 关闭客户端 # 关闭客户端
            for ctx in getattr(self, "_pool_storage_ctx", {}).values():  # 遍历所有池存储上下文 # 遍历池上下文
                for c in ctx.clients:  # 遍历池客户端 # 遍历池客户端
                    c.close()  # 关闭池客户端 # 关闭客户端
            self.executor.shutdown(wait=True)  # 关闭线程池 # 关闭线程池
        except Exception as e:  # 捕获异常 # 捕获异常
            logger.error(f"close HiCacheHF3FS: {e}")  # 记录关闭错误 # 记录错误
        logger.info("close HiCacheHF3FS")  # 记录关闭信息 # 记录日志

    def get_stats(self):  # 获取存储统计信息
        storage_metrics = StorageMetrics()  # 创建存储指标对象 # 创建指标
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)  # 复制预取页面数 # 复制预取页面数
        storage_metrics.backup_pgs.extend(self.backup_pgs)  # 复制备份页面数 # 复制备份页面数
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)  # 复制预取带宽 # 复制预取带宽
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)  # 复制备份带宽 # 复制备份带宽
        self.prefetch_pgs.clear()  # 清空预取页面数 # 清空预取页面数
        self.backup_pgs.clear()  # 清空备份页面数 # 清空备份页面数
        self.prefetch_bandwidth.clear()  # 清空预取带宽 # 清空预取带宽
        self.backup_bandwidth.clear()  # 清空备份带宽 # 清空备份带宽
        return storage_metrics  # 返回存储指标 # 返回指标

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):  # 注册主机内存池
        super().register_mem_pool_host(mem_pool_host)  # 调用父类方法 # 调用父类
        self.is_zero_copy = self.mem_pool_host.layout in [  # 检查布局是否为零拷贝 # 检查零拷贝
            "page_first",  # page_first布局 # page_first布局
            "page_first_direct",  # page_first_direct布局 # page_first_direct布局
        ]

        logger.info(f"{self.is_zero_copy=}, layout={self.mem_pool_host.layout}")  # 记录零拷贝和布局信息 # 记录日志

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):  # 注册v2版本的主机内存池
        if host_pool_name == PoolName.KV:  # KV池不在此处理 # 检查KV池
            return  # 直接返回 # 返回
        super().register_mem_host_pool_v2(host_pool, host_pool_name)  # 调用父类方法 # 调用父类

        pool_page_size = getattr(host_pool, "page_size", 1) or 1  # 获取池页面大小 # 获取页面大小
        pool_bytes_per_page = host_pool.get_ksize_per_token() * pool_page_size  # 计算每页字节数 # 计算每页字节数
        pool_num_pages = self.file_size // pool_bytes_per_page  # 计算池页面数 # 计算页面数
        pool_file_path = f"{self.file_path}.{host_pool_name}"  # 池文件路径 # 文件路径
        namespace = host_pool_name  # e.g. PoolName.MAMBA, PoolName.INDEXER  # 命名空间 # 命名空间

        pool_clients = [  # 创建池客户端列表 # 创建池客户端
            create_hf3fs_client(  # 调用工厂函数 # 创建客户端
                pool_file_path,  # 池文件路径 # 文件路径
                self.file_size,  # 文件大小 # 文件大小
                pool_bytes_per_page,  # 每页字节数 # 每页字节数
                self.entries,  # 条目数 # 条目数
                self.client_timeout,  # 超时时间 # 超时
                self.use_mock_client,  # 是否Mock # Mock标志
            )
            for _ in range(self.numjobs)  # 循环numjobs次 # 循环
        ]

        self.metadata_client.initialize(self.rank, pool_num_pages, namespace=namespace)  # 初始化池元数据 # 初始化元数据

        self._pool_storage_ctx[host_pool_name] = _PoolStorageCtx(  # 创建池存储上下文 # 创建上下文
            pool_name=host_pool_name,  # 池名称 # 池名称
            bytes_per_page=pool_bytes_per_page,  # 每页字节数 # 每页字节数
            num_pages=pool_num_pages,  # 页面数 # 页面数
            namespace=namespace,  # 命名空间 # 命名空间
            clients=pool_clients,  # 客户端列表 # 客户端列表
            gb_per_page=pool_bytes_per_page / (1 << 30),  # 每页GB大小 # 每页GB
        )
        logger.info(  # 记录注册信息 # 记录日志
            f"[Rank {self.rank}] Registered hybrid pool '{host_pool_name}': "  # 注册混合池 # 注册混合池
            f"bytes_per_page={pool_bytes_per_page}, num_pages={pool_num_pages}, "  # 每页字节数和页面数 # 字节数和页面数
            f"namespace={namespace}, file={pool_file_path}"  # 命名空间和文件路径 # 命名空间和路径
        )

    def _get_mha_zero_copy_keys(self, keys: List[str]) -> List[str]:  # 获取MHA零拷贝模式的键列表（K和V分开）
        _keys = []  # 键列表 # 键列表
        for k in keys:  # 遍历原始键 # 遍历键
            _keys.append(f"{k}-k")  # 添加K键 # 添加K键
            _keys.append(f"{k}-v")  # 添加V键 # 添加V键
        return _keys  # 返回键列表 # 返回键

    def _get_mha_zero_copy_values(  # 获取MHA零拷贝模式的值列表（K和V分开）
        self, values: List[torch.Tensor]  # 原始值张量列表 # 值列表
    ) -> List[torch.Tensor]:
        _values = []  # 值列表 # 值列表
        for value in values:  # 遍历原始值 # 遍历值
            _values.append(value[0])  # 添加K张量 # 添加K
            _values.append(value[1])  # 添加V张量 # 添加V
        return _values  # 返回值列表 # 返回值

    def _batch_get_preprocess(self, keys, host_indices):  # 批量获取的预处理
        page_num = len(host_indices) // self.mem_pool_host.page_size  # 计算页面数 # 计算页面数
        # host_indices to kv_buffer  # host索引转kv缓冲区 # 索引转缓冲区
        flat = not self.is_zero_copy  # 非零拷贝时使用flat模式 # flat标志
        values = (  # 构建值张量列表 # 构建值列表
            [  # 零拷贝模式 # 零拷贝模式
                self.mem_pool_host.get_data_page(  # 获取数据页面 # 获取数据页
                    host_indices[i * self.mem_pool_host.page_size], flat=flat  # 主机索引和flat标志 # 索引和标志
                )
                for i in range(page_num)  # 遍历页面 # 遍历页面
            ]
            if self.is_zero_copy  # 零拷贝模式 # 零拷贝
            else [  # 非零拷贝模式 # 非零拷贝模式
                self.mem_pool_host.get_dummy_flat_data_page() for _ in range(page_num)  # 获取虚拟flat数据页 # 虚拟数据页
            ]
        )

        if self.is_zero_copy and not self.is_mla_model:  # 零拷贝且非MLA # 检查条件
            keys = self._get_mha_zero_copy_keys(keys)  # 获取MHA零拷贝键 # 获取零拷贝键
            values = self._get_mha_zero_copy_values(values)  # 获取MHA零拷贝值 # 获取零拷贝值

        return keys, values  # 返回键和值 # 返回键值

    def _batch_get_postprocess(self, host_indices, values, results):  # 批量获取的后处理
        page_num = len(host_indices) // self.mem_pool_host.page_size  # 计算页面数 # 计算页面数

        if self.is_zero_copy:  # 零拷贝模式 # 零拷贝模式
            if not self.is_mla_model:  # 非MLA模型 # 非MLA
                results = [  # 合并K和V的结果 # 合并结果
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)  # K和V都成功才算成功 # K和V都成功
                ]
                results = results[:page_num]  # 截取到页面数 # 截取结果
            return results  # 返回结果 # 返回结果

        for i in range(page_num):  # 遍历页面 # 遍历页面
            if not results[i]:  # 结果为False # 检查失败
                break  # 跳出循环 # 跳出
            self.mem_pool_host.set_from_flat_data_page(  # 从flat数据页设置数据 # 设置数据
                host_indices[i * self.mem_pool_host.page_size], values[i]  # 主机索引和值 # 索引和值
            )

        return results  # 返回结果 # 返回结果

    def batch_exists_v2(  # v2版本批量检查键是否存在，支持多池
        self,
        keys: List[str],  # 键列表 # 键列表
        pool_transfers: Optional[List[PoolTransfer]] = None,  # 池传输列表 # 池传输列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息 # 额外信息
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)  # 检查KV池命中 # 检查KV命中

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}  # 命中计数字典 # 命中计数
        final_pages = kv_pages  # 最终命中页面数 # 最终页面数

        for transfer in pool_transfers or []:  # 遍历池传输 # 遍历传输
            if final_pages == 0:  # 已无命中页面 # 检查命中
                break  # 跳出循环 # 跳出

            pool_name = transfer.name  # 池名称 # 池名称
            ctx = self._pool_storage_ctx.get(pool_name)  # 获取池存储上下文 # 获取上下文
            if ctx is None:  # 上下文不存在 # 检查上下文
                final_pages = 0  # 命中页面置0 # 置0
                break  # 跳出循环 # 跳出

            component_keys = [f"{key}_{pool_name}" for key in keys[:kv_pages]]  # 构建组件键 # 构建组件键
            exists_results = self.metadata_client.exists(  # 查询键是否存在 # 查询存在
                self.rank, component_keys, namespace=ctx.namespace  # rank、键列表、命名空间 # 参数
            )

            boundary = 0  # 边界值 # 边界
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:  # 所有页面必须命中 # 全部命中策略
                try:  # 尝试查找第一个False # 尝试
                    boundary = exists_results.index(False)  # 第一个不存在的索引 # 不存在索引
                except ValueError:  # 没有False，全部存在 # 全部存在
                    boundary = kv_pages  # 边界等于KV页面数 # 边界
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:  # 尾部页面命中策略 # 尾部命中策略
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)  # 尾部页面数 # 尾部数
                for prefix_len in range(kv_pages, 0, -1):  # 从大到小搜索 # 反向搜索
                    if all(  # 检查尾部页面是否全部存在 # 检查尾部
                        exists_results[i]  # 页面存在 # 存在
                        for i in range(max(0, prefix_len - trailing), prefix_len)  # 尾部范围 # 尾部范围
                    ):
                        boundary = prefix_len  # 设置边界 # 设置边界
                        break  # 跳出循环 # 跳出

            if boundary:  # 边界大于0 # 检查边界
                hit_count[pool_name] = boundary  # 记录池命中数 # 记录命中
            final_pages = min(final_pages, boundary)  # 取最小命中页面数 # 取最小

        return PoolTransferResult(final_pages, hit_count)  # 返回池传输结果 # 返回结果

    def _pool_batch_get(self, transfer: PoolTransfer) -> List[bool]:  # 单个池的批量获取
        pool_name = transfer.name  # 池名称 # 池名称
        ctx = self._pool_storage_ctx[pool_name]  # 获取池存储上下文 # 获取上下文
        host_pool = self.registered_pools[pool_name]  # 获取主机内存池 # 获取主机池
        keys = transfer.keys  # 键列表 # 键列表
        host_indices = transfer.host_indices  # 主机索引 # 主机索引
        page_size = getattr(host_pool, "page_size", 1) or 1  # 页面大小 # 页面大小
        page_num = len(keys)  # 页面数 # 页面数

        component_keys = [f"{key}_{pool_name}" for key in keys]  # 构建组件键 # 组件键
        page_indices = self.metadata_client.get_page_indices(  # 获取页索引 # 获取页索引
            self.rank, component_keys, namespace=ctx.namespace  # rank、键列表、命名空间 # 参数
        )

        batch_indices, file_offsets, values = [], [], []  # 批量索引、文件偏移量、值张量列表 # 列表
        for i, page_index in enumerate(page_indices):  # 遍历页索引 # 遍历
            if page_index is not None:  # 页索引不为空 # 检查非空
                batch_indices.append(i)  # 添加批量索引 # 添加索引
                file_offsets.append(page_index * ctx.bytes_per_page)  # 计算文件偏移量 # 计算偏移
                values.append(host_pool.get_dummy_flat_data_page())  # 获取虚拟flat数据页 # 获取虚拟页

        if not batch_indices:  # 没有需要读取的页面 # 检查空
            return [False] * page_num  # 返回全部失败 # 返回失败

        start_time = time.perf_counter()  # 记录开始时间 # 开始计时
        futures = [  # 创建异步读取任务列表 # 创建任务
            self.executor.submit(  # 提交任务到线程池 # 提交任务
                ctx.clients[self.ac.next()].batch_read,  # 轮询选择客户端并批量读取 # 轮询读取
                file_offsets[j : j + self.entries],  # 文件偏移量切片 # 偏移切片
                values[j : j + self.entries],  # 值张量切片 # 值切片
            )
            for j in range(0, len(batch_indices), self.entries)  # 按条目数分批 # 分批
        ]
        read_results = [r for f in futures for r in f.result()]  # 等待所有任务完成并收集结果 # 收集结果
        end_time = time.perf_counter()  # 记录结束时间 # 结束计时
        ionum = len(batch_indices)  # IO操作数量 # IO数量

        if self.enable_storage_metrics:  # 启用存储指标 # 检查指标
            self.prefetch_pgs.append(ionum)  # 记录预取页面数 # 记录页面数
            self.prefetch_bandwidth.append(  # 记录预取带宽 # 记录带宽
                ionum / (end_time - start_time) * ctx.gb_per_page  # 计算带宽 # 计算带宽
            )

        results = [False] * page_num  # 初始化结果列表 # 初始化结果
        for idx, (batch_idx, read_result) in enumerate(  # 遍历结果 # 遍历
            zip(batch_indices, read_results)  # 批量索引和读取结果 # 索引和结果
        ):
            if read_result == ctx.bytes_per_page:  # 读取成功 # 检查成功
                host_idx = host_indices[batch_idx * page_size].item()  # 获取主机索引 # 获取主机索引
                host_pool.set_from_flat_data_page(host_idx, values[idx])  # 从flat数据页设置数据 # 设置数据
                results[batch_idx] = True  # 标记成功 # 标记成功
            else:  # 读取失败 # 读取失败
                logger.error(  # 记录错误日志 # 记录错误
                    f"[Rank {self.rank}][Pool {pool_name.upper()}] HiCacheHF3FS get {keys[batch_idx]} failed"  # 读取失败 # 读取失败
                )

        return results  # 返回结果列表 # 返回结果

    def _pool_batch_set(self, transfer: PoolTransfer) -> List[bool]:  # 单个池的批量设置
        pool_name = transfer.name  # 池名称 # 池名称
        ctx = self._pool_storage_ctx[pool_name]  # 获取池存储上下文 # 获取上下文
        host_pool = self.registered_pools[pool_name]  # 获取主机内存池 # 获取主机池
        keys = transfer.keys  # 键列表 # 键列表
        host_indices = transfer.host_indices  # 主机索引 # 主机索引
        page_size = getattr(host_pool, "page_size", 1) or 1  # 页面大小 # 页面大小
        page_num = len(keys)  # 页面数 # 页面数

        component_keys = [f"{key}_{pool_name}" for key in keys]  # 构建组件键 # 组件键
        key_with_prefix = [(k, "") for k in component_keys]  # 构建键和空前缀的元组列表 # 键元组
        indices = self.metadata_client.reserve_and_allocate_page_indices(  # 预留并分配页索引 # 预留分配
            self.rank, key_with_prefix, namespace=ctx.namespace  # rank、键元组、命名空间 # 参数
        )

        if len(indices) != page_num:  # 分配索引数量不匹配 # 检查数量
            logger.error(  # 记录错误日志 # 记录错误
                f"[Rank {self.rank}] Pool {pool_name}: mismatched indices length"  # 索引长度不匹配 # 长度不匹配
            )
            if indices:  # 有已分配的索引 # 检查索引
                self.metadata_client.confirm_write(  # 确认写入以释放页面 # 确认写入
                    self.rank, [], [idx[1] for idx in indices], namespace=ctx.namespace  # 空写入列表、释放页索引、命名空间 # 释放页面
                )
            return [False] * page_num  # 返回全部失败 # 返回失败

        batch_indices, file_offsets, file_values = [], [], []  # 批量索引、文件偏移量、文件值列表 # 列表
        for i, (is_written, page_index) in enumerate(indices):  # 遍历索引 # 遍历
            if is_written or page_index == -1:  # 已写入或页索引无效 # 检查跳过
                continue  # 跳过 # 跳过
            batch_indices.append(i)  # 添加批量索引 # 添加索引
            file_offsets.append(page_index * ctx.bytes_per_page)  # 计算文件偏移量 # 计算偏移
            host_idx = host_indices[i * page_size].item()  # 获取主机索引 # 获取主机索引
            data = host_pool.get_data_page(host_idx, flat=True)  # 获取flat数据页 # 获取数据页
            assert data.is_contiguous()  # 断言数据是连续的 # 连续性检查
            file_values.append(data)  # 添加文件值 # 添加值

        start_time = time.perf_counter()  # 记录开始时间 # 开始计时
        futures = [  # 创建异步写入任务列表 # 创建任务
            self.executor.submit(  # 提交任务到线程池 # 提交任务
                ctx.clients[self.ac.next()].batch_write,  # 轮询选择客户端并批量写入 # 轮询写入
                file_offsets[j : j + self.entries],  # 文件偏移量切片 # 偏移切片
                file_values[j : j + self.entries],  # 值张量切片 # 值切片
            )
            for j in range(0, len(batch_indices), self.entries)  # 按条目数分批 # 分批
        ]
        write_results = [r == ctx.bytes_per_page for f in futures for r in f.result()]  # 等待并检查结果 # 收集结果
        end_time = time.perf_counter()  # 记录结束时间 # 结束计时
        ionum = len(batch_indices)  # IO操作数量 # IO数量

        if self.enable_storage_metrics:  # 启用存储指标 # 检查指标
            self.backup_pgs.append(ionum)  # 记录备份页面数 # 记录页面数
            self.backup_bandwidth.append(  # 记录备份带宽 # 记录带宽
                ionum / (end_time - start_time) * ctx.gb_per_page  # 计算带宽 # 计算带宽
            )

        written_keys_to_confirm = []  # 需要确认写入的键列表 # 确认列表
        pages_to_release = []  # 需要释放的页面列表 # 释放列表
        results = [idx[0] for idx in indices]  # 初始化结果为索引的是否已存在标志 # 初始化结果
        for batch_idx, write_ok in zip(batch_indices, write_results):  # 遍历结果 # 遍历
            key = component_keys[batch_idx]  # 获取组件键 # 获取键
            page_index = indices[batch_idx][1]  # 获取页索引 # 获取页索引
            if write_ok:  # 写入成功 # 检查成功
                written_keys_to_confirm.append((key, page_index))  # 添加到确认列表 # 添加确认
            else:  # 写入失败 # 写入失败
                logger.error(  # 记录错误日志 # 记录错误
                    f"[Rank {self.rank}][Pool {pool_name.upper()}] HiCacheHF3FS set {keys[batch_idx]} failed"  # 写入失败 # 写入失败
                )
                pages_to_release.append(page_index)  # 添加到释放列表 # 添加释放
            results[batch_idx] = write_ok  # 更新结果 # 更新结果

        if written_keys_to_confirm or pages_to_release:  # 有需要确认或释放的页面 # 检查
            self.metadata_client.confirm_write(  # 确认写入 # 确认写入
                self.rank,  # rank # rank
                written_keys_to_confirm,  # 确认列表 # 确认列表
                pages_to_release,  # 释放列表 # 释放列表
                namespace=ctx.namespace,  # 命名空间 # 命名空间
            )

        return results  # 返回结果列表 # 返回结果

    def batch_get_v2(  # v2版本批量获取，支持多池
        self,
        transfers: List[PoolTransfer],  # 池传输列表 # 池传输列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息 # 额外信息
    ) -> dict:
        results = {}  # 结果字典 # 结果字典
        for transfer in transfers:  # 遍历池传输 # 遍历传输
            results[transfer.name] = self._pool_batch_get(transfer)  # 执行单个池批量获取 # 执行获取
        return results  # 返回结果 # 返回结果

    def batch_set_v2(  # v2版本批量设置，支持多池
        self,
        transfers: List[PoolTransfer],  # 池传输列表 # 池传输列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息 # 额外信息
    ) -> dict:
        results = {}  # 结果字典 # 结果字典
        for transfer in transfers:  # 遍历池传输 # 遍历传输
            results[transfer.name] = self._pool_batch_set(transfer)  # 执行单个池批量设置 # 执行设置
        return results  # 返回结果 # 返回结果

    def batch_get_v1(  # v1版本批量获取
        self,
        keys: List[str],  # 键列表 # 键列表
        host_indices: torch.Tensor,  # 主机索引张量 # 主机索引
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息 # 额外信息
    ) -> List[bool]:
        keys, values = self._batch_get_preprocess(keys, host_indices)  # 预处理 # 预处理
        results = self._batch_get(keys, values)  # 批量获取 # 批量获取
        return self._batch_get_postprocess(host_indices, values, results)  # 后处理并返回 # 后处理

    def _batch_set_preprocess(self, keys, host_indices):  # 批量设置的预处理
        page_num = len(host_indices) // self.mem_pool_host.page_size  # 计算页面数 # 计算页面数
        # host_indices to kv_buffer  # host索引转kv缓冲区 # 索引转缓冲区
        flat = not self.is_zero_copy  # 非零拷贝时使用flat模式 # flat标志
        values = [  # 构建值张量列表 # 构建值列表
            self.mem_pool_host.get_data_page(  # 获取数据页面 # 获取数据页
                host_indices[i * self.mem_pool_host.page_size], flat=flat  # 主机索引和flat标志 # 索引和标志
            )
            for i in range(page_num)  # 遍历页面 # 遍历页面
        ]

        if self.is_zero_copy and not self.is_mla_model:  # 零拷贝且非MLA # 检查条件
            keys = self._get_mha_zero_copy_keys(keys)  # 获取MHA零拷贝键 # 获取零拷贝键
            values = self._get_mha_zero_copy_values(values)  # 获取MHA零拷贝值 # 获取零拷贝值

        return keys, values  # 返回键和值 # 返回键值

    def batch_set_v1(  # v1版本批量设置
        self,
        keys: List[str],  # 键列表 # 键列表
        host_indices: torch.Tensor,  # 主机索引张量 # 主机索引
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息 # 额外信息
    ) -> List[bool]:
        len_keys = len(keys)  # 键数量 # 键数量
        keys, values = self._batch_set_preprocess(keys, host_indices)  # 预处理 # 预处理
        results = self._batch_set(keys, values)  # 批量设置 # 批量设置
        return results  # 返回结果 # 返回结果

    # Deprecated  # 已弃用 # 已弃用
    def get(  # 获取单个键的值（已弃用）
        self,
        key: str,  # 键 # 键
        target_location: Optional[Any] = None,  # 目标位置 # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小 # 目标大小
    ) -> torch.Tensor | None:
        pass  # 占位 # 占位

    # Deprecated  # 已弃用 # 已弃用
    def batch_get(  # 批量获取（已弃用）
        self,
        keys: List[str],  # 键列表 # 键列表
        target_locations: Optional[Any] = None,  # 目标位置列表 # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小列表 # 目标大小
    ) -> List[torch.Tensor | None] | int:
        pass  # 占位 # 占位

    # Deprecated  # 已弃用 # 已弃用
    def set(  # 设置单个键值（已弃用）
        self,
        key: str,  # 键 # 键
        value: Optional[Any] = None,  # 值 # 值
        target_location: Optional[Any] = None,  # 目标位置 # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小 # 目标大小
    ) -> bool:
        pass  # 占位 # 占位

    # Deprecated  # 已弃用 # 已弃用
    def batch_set(  # 批量设置（已弃用）
        self,
        keys: List[str],  # 键列表 # 键列表
        values: Optional[Any] = None,  # 值列表 # 值列表
        target_locations: Optional[Any] = None,  # 目标位置列表 # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小列表 # 目标大小
    ) -> bool:
        pass  # 占位 # 占位
