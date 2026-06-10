# Mooncake存储后端实现模块
# 本模块实现了基于Mooncake分布式存储的HiCache存储后端，
# 提供零拷贝的KV缓存读写、批量操作、SSD卸载等功能，
# 支持MLA和MHA两种注意力机制模型，以及多种配置加载方式。

import ctypes  # 导入ctypes模块，用于C类型内存操作
import json  # 导入json模块，用于解析JSON配置文件
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于操作系统接口
import time  # 导入time模块，用于时间相关操作
import uuid  # 导入uuid模块，用于生成唯一标识符
from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import Any, List, Optional, Tuple  # 从typing模块导入类型注解

import requests  # 导入requests模块，用于HTTP请求
import torch  # 导入torch模块，用于张量操作

from sglang.srt.environ import envs  # 从environ模块导入环境变量配置
from sglang.srt.mem_cache.hicache_storage import (  # 从hicache_storage模块导入存储相关类和枚举
    HiCacheStorage,  # HiCache存储基类
    HiCacheStorageConfig,  # HiCache存储配置类
    HiCacheStorageExtraInfo,  # HiCache存储额外信息类
    PoolHitPolicy,  # 池命中策略枚举
    PoolName,  # 池名称枚举
    PoolTransfer,  # 池传输类
    PoolTransferResult,  # 池传输结果类
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache, HostTensorAllocator  # 从memory_pool_host模块导入主机KV缓存和主机张量分配器
from sglang.srt.observability.metrics_collector import StorageMetrics  # 从metrics_collector模块导入存储指标收集器

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB  # 默认本地缓冲区大小为16MB
SETUP_TIMEOUT = 600  # 10min  # 设置超时时间为600秒（10分钟）

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class MooncakeHostTensorAllocator(HostTensorAllocator):  # Mooncake主机张量分配器类，继承自HostTensorAllocator
    def __init__(self):  # 初始化方法
        super().__init__()  # 调用父类初始化方法
        from mooncake.store import MooncakeHostMemAllocator  # 从mooncake.store导入Mooncake主机内存分配器

        self.allocator = MooncakeHostMemAllocator()  # 创建Mooncake主机内存分配器实例
        self.ptr = None  # 初始化指针为None

    def allocate(  # 分配内存方法 # 使用MooncakeHostMemAllocator分配内存并包装为PyTorch张量
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"  # 参数：维度、数据类型、设备（默认CPU）
    ) -> torch.Tensor:  # 返回PyTorch张量
        """
        Allocates memory using MooncakeHostMemAllocator and wraps it in a PyTorch tensor.
        使用MooncakeHostMemAllocator分配内存并包装为PyTorch张量。
        """
        self.dims = dims  # 保存维度信息
        self.dtype = dtype  # 保存数据类型
        size = 1  # 初始化大小为1
        for d in dims:  # 遍历每个维度
            size *= d  # 计算总元素数
        size *= torch.tensor([], dtype=self.dtype).element_size()  # 乘以每个元素的字节大小得到总字节数
        ptr_int = self.allocator.alloc(size)  # 使用Mooncake分配器分配内存，返回指针地址
        self.ptr = ptr_int  # 保存分配的内存指针
        c_type = ctypes.c_byte * size  # 创建C类型的字节数组类型，长度为size
        c_array = c_type.from_address(ptr_int)  # 从内存地址创建C数组

        tensor = torch.frombuffer(c_array, dtype=torch.uint8, count=size)  # 从C数组创建uint8类型的张量

        if dtype != torch.uint8:  # 如果目标类型不是uint8
            element_size = torch.tensor([], dtype=dtype).element_size()  # 获取目标类型的元素大小
            assert size % element_size == 0, "Size must be divisible by element size"  # 断言总大小能被元素大小整除 # 大小必须能被元素大小整除
            tensor = tensor.view(dtype)  # 将张量视图转换为目标数据类型

        return tensor.view(dims)  # 返回重塑为目标维度的张量


def _parse_global_segment_size(value) -> int:  # 解析全局段大小配置值，支持整数和带"gb"后缀的字符串 # 将全局段大小配置值解析为整数（字节数）
    if isinstance(value, int):  # 如果值是整数类型
        return value  # 直接返回
    if isinstance(value, str):  # 如果值是字符串类型
        s = value.strip().lower()  # 去除首尾空格并转为小写
        if s.endswith("gb"):  # 如果以"gb"结尾
            num = s[:-2].strip()  # 提取"gb"前面的数字部分
            if not num:  # 如果数字部分为空
                raise ValueError(  # 抛出数值错误
                    "Invalid global_segment_size: missing number before 'gb'"  # 无效的全局段大小：'gb'前缺少数字 # 'gb'前缺少数字
                )
            return int(num) * 1024 * 1024 * 1024  # 将GB转换为字节并返回
        return int(s)  # 否则直接转为整数返回
    return int(value)  # 其他类型强制转为整数返回


@dataclass  # 数据类装饰器
class MooncakeStoreConfig:  # Mooncake存储配置数据类
    local_hostname: str  # 本地主机名
    metadata_server: str  # 元数据服务器地址
    global_segment_size: int  # 全局段大小（字节）
    protocol: str  # 通信协议（如rdma）
    device_name: str  # 设备名称
    master_server_address: str  # 主服务器地址
    master_metrics_port: int  # 主服务器指标端口
    check_server: bool  # 是否检查服务器状态
    standalone_storage: bool  # 是否使用独立存储模式
    client_server_address: str  # 客户端服务器地址
    enable_ssd_offload: bool = False  # 是否启用SSD卸载，默认False
    ssd_offload_path: Optional[str] = None  # SSD卸载路径，默认None

    @staticmethod  # 静态方法
    def from_file() -> "MooncakeStoreConfig":  # 从JSON配置文件加载配置 # 从文件加载Mooncake存储配置
        """Load the config from a JSON file."""  # 从JSON文件加载配置。 # 从JSON文件加载配置
        if not envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():  # 如果配置文件路径环境变量未设置
            raise RuntimeError(  # 抛出运行时错误
                f"Config file path not set. Please set {envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.name}"  # 配置文件路径未设置，请设置对应环境变量
            )
        file_path = envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.get()  # 获取配置文件路径
        try:  # 尝试加载配置
            with open(file_path) as fin:  # 打开配置文件
                config = json.load(fin)  # 加载JSON配置
        except Exception as e:  # 捕获异常
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")  # 抛出加载失败错误 # 从文件加载配置失败

        if (  # 检查是否包含必要的服务器地址
            "master_server_address" not in config  # 配置中没有master_server_address
            and "client_server_address" not in config  # 也没有client_server_address
        ):
            raise ValueError(  # 抛出数值错误
                "Either master_server_address or client_server_address is required in config file"  # 配置文件中必须包含master_server_address或client_server_address
            )

        return MooncakeStoreConfig(  # 返回MooncakeStoreConfig实例
            local_hostname=config.get(  # 本地主机名
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default  # 优先从配置读取，否则使用默认值
            ),
            metadata_server=config.get(  # 元数据服务器地址
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default  # 优先从配置读取，否则使用默认值
            ),
            global_segment_size=_parse_global_segment_size(  # 全局段大小，解析配置值
                config.get(  # 从配置获取值
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default  # 优先从配置读取，否则使用默认值
                )
            ),
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.default),  # 通信协议
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),  # 设备名称
            master_server_address=config.get(  # 主服务器地址
                "master_server_address", envs.MOONCAKE_MASTER.default  # 优先从配置读取，否则使用默认值
            ),
            master_metrics_port=config.get(  # 主服务器指标端口
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default  # 优先从配置读取，否则使用默认值
            ),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),  # 是否检查服务器
            standalone_storage=config.get(  # 是否独立存储
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default  # 优先从配置读取，否则使用默认值
            ),
            client_server_address=config.get(  # 客户端服务器地址
                "client_server_address", envs.MOONCAKE_CLIENT.default  # 优先从配置读取，否则使用默认值
            ),
            enable_ssd_offload=config.get(  # 是否启用SSD卸载
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default  # 优先从配置读取，否则使用默认值
            ),
            ssd_offload_path=config.get(  # SSD卸载路径
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default  # 优先从配置读取，否则使用默认值
            ),
        )

    @staticmethod  # 静态方法
    def load_from_env() -> "MooncakeStoreConfig":  # 从环境变量加载配置 # 从环境变量加载Mooncake存储配置
        """Load config from a file specified in the environment variable.
        从环境变量指定的文件加载配置。
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        # other required environment variables...  # 其他必需的环境变量...
        if not envs.MOONCAKE_MASTER.is_set() and not envs.MOONCAKE_CLIENT.is_set():  # 如果主服务器和客户端地址都未设置
            raise ValueError(  # 抛出数值错误
                "Either the environment variable 'MOONCAKE_MASTER' or 'MOONCAKE_CLIENT' is not set."  # 必须设置MOONCAKE_MASTER或MOONCAKE_CLIENT环境变量
            )

        # Special handling for local_hostname: try MOONCAKE_LOCAL_HOSTNAME first,
        # then fall back to LOCAL_HOSTNAME if not set.
        # This is for forward compatibility with the legacy LOCAL_HOSTNAME environment variable.
        # 对local_hostname的特殊处理：优先尝试MOONCAKE_LOCAL_HOSTNAME，
        # 如果未设置则回退到LOCAL_HOSTNAME。
        # 这是为了向前兼容旧的LOCAL_HOSTNAME环境变量。
        if envs.MOONCAKE_LOCAL_HOSTNAME.is_set():  # 如果MOONCAKE_LOCAL_HOSTNAME环境变量已设置
            local_hostname = envs.MOONCAKE_LOCAL_HOSTNAME.get()  # 使用该值
        else:  # 否则
            local_hostname = os.getenv(  # 从os.getenv获取LOCAL_HOSTNAME
                "LOCAL_HOSTNAME", envs.MOONCAKE_LOCAL_HOSTNAME.default  # 优先使用LOCAL_HOSTNAME，否则使用默认值
            )

        return MooncakeStoreConfig(  # 返回MooncakeStoreConfig实例
            local_hostname=local_hostname,  # 本地主机名
            metadata_server=envs.MOONCAKE_TE_META_DATA_SERVER.get(),  # 元数据服务器地址
            global_segment_size=_parse_global_segment_size(  # 全局段大小，解析环境变量值
                envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.get()  # 获取环境变量值
            ),
            protocol=envs.MOONCAKE_PROTOCOL.get(),  # 通信协议
            device_name=envs.MOONCAKE_DEVICE.get(),  # 设备名称
            master_server_address=envs.MOONCAKE_MASTER.get(),  # 主服务器地址
            master_metrics_port=envs.MOONCAKE_MASTER_METRICS_PORT.get(),  # 主服务器指标端口
            check_server=envs.MOONCAKE_CHECK_SERVER.get(),  # 是否检查服务器
            standalone_storage=envs.MOONCAKE_STANDALONE_STORAGE.get(),  # 是否独立存储
            client_server_address=envs.MOONCAKE_CLIENT.get(),  # 客户端服务器地址
            enable_ssd_offload=envs.MOONCAKE_ENABLE_SSD_OFFLOAD.get(),  # 是否启用SSD卸载
            ssd_offload_path=envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.get(),  # SSD卸载路径
        )

    @staticmethod  # 静态方法
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":  # 从额外配置字典加载配置 # 从extra_config字典加载Mooncake存储配置
        """Load config from extra_config dictionary."""  # 从extra_config字典加载配置。 # 从extra_config字典加载配置
        if (  # 检查是否包含必要的服务器地址
            "master_server_address" not in extra_config  # 额外配置中没有master_server_address
            and "client_server_address" not in extra_config  # 也没有client_server_address
        ):
            raise ValueError(  # 抛出数值错误
                "Either master_server_address or client_server_address is required in extra_config"  # extra_config中必须包含master_server_address或client_server_address
            )

        return MooncakeStoreConfig(  # 返回MooncakeStoreConfig实例
            local_hostname=extra_config.get(  # 本地主机名
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default  # 优先从配置读取，否则使用默认值
            ),
            metadata_server=extra_config.get(  # 元数据服务器地址
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default  # 优先从配置读取，否则使用默认值
            ),
            global_segment_size=_parse_global_segment_size(  # 全局段大小，解析配置值
                extra_config.get(  # 从额外配置获取值
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default  # 优先从配置读取，否则使用默认值
                )
            ),
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.default),  # 通信协议
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),  # 设备名称
            master_server_address=extra_config.get(  # 主服务器地址
                "master_server_address", envs.MOONCAKE_MASTER.default  # 优先从配置读取，否则使用默认值
            ),
            master_metrics_port=extra_config.get(  # 主服务器指标端口
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default  # 优先从配置读取，否则使用默认值
            ),
            check_server=extra_config.get(  # 是否检查服务器
                "check_server", envs.MOONCAKE_CHECK_SERVER.default  # 优先从配置读取，否则使用默认值
            ),
            standalone_storage=extra_config.get(  # 是否独立存储
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default  # 优先从配置读取，否则使用默认值
            ),
            client_server_address=extra_config.get(  # 客户端服务器地址
                "client_server_address", envs.MOONCAKE_CLIENT.default  # 优先从配置读取，否则使用默认值
            ),
            enable_ssd_offload=extra_config.get(  # 是否启用SSD卸载
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default  # 优先从配置读取，否则使用默认值
            ),
            ssd_offload_path=extra_config.get(  # SSD卸载路径
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default  # 优先从配置读取，否则使用默认值
            ),
        )


class MooncakeBaseStore:  # Mooncake基础存储类，提供通用的存储操作
    def __init__(self):  # 初始化方法
        self.store = None  # Mooncake分布式存储实例，初始为None
        self.config = None  # 存储配置，初始为None

    def _import_mooncake_store(self):  # 导入Mooncake分布式存储模块 # 导入MooncakeDistributedStore类
        try:  # 尝试导入
            from mooncake.store import MooncakeDistributedStore  # 从mooncake.store导入Mooncake分布式存储类

            return MooncakeDistributedStore  # 返回导入的类
        except ImportError as e:  # 捕获导入错误
            raise ImportError(  # 抛出导入错误
                "Please install mooncake by following the instructions at "  # 请按照以下地址安装mooncake
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "  # 安装说明链接
                "to run SGLang with MooncakeConnector."  # 以便使用MooncakeConnector运行SGLang
            ) from e

    def _load_config(self, storage_config: Any = None):  # 加载Mooncake存储配置 # 根据不同来源加载存储配置
        extra_config = (  # 获取额外配置
            getattr(storage_config, "extra_config", None) if storage_config else None  # 如果有storage_config则获取extra_config属性
        )

        if extra_config and (  # 如果额外配置存在且包含服务器地址
            extra_config.get("master_server_address") is not None  # master_server_address不为空
            or extra_config.get("client_server_address") is not None  # 或client_server_address不为空
        ):
            config = MooncakeStoreConfig.load_from_extra_config(extra_config)  # 从额外配置加载
            logger.info("Mooncake Configuration loaded from extra_config successfully.")  # 记录从extra_config加载成功

        elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():  # 否则如果配置文件路径已设置
            config = MooncakeStoreConfig.from_file()  # 从文件加载配置
            logger.info("Mooncake Configuration loaded from file successfully.")  # 记录从文件加载成功

        else:  # 否则
            config = MooncakeStoreConfig.load_from_env()  # 从环境变量加载配置
            logger.info("Mooncake Configuration loaded from env successfully.")  # 记录从环境变量加载成功

        return config  # 返回加载的配置

    def register_buffer(self, tensor: torch.Tensor):  # 注册张量缓冲区到Mooncake存储 # 将PyTorch张量注册到Mooncake存储中
        if self.store is None:  # 如果存储实例为空
            raise RuntimeError("Mooncake store is not initialized.")  # 抛出运行时错误：Mooncake存储未初始化
        ptr = tensor.data_ptr()  # 获取张量的数据指针
        size = tensor.numel() * tensor.element_size()  # 计算张量的总字节大小
        ret_code = self.store.register_buffer(ptr, size)  # 向Mooncake存储注册缓冲区
        if ret_code != 0:  # 如果返回码不为0（注册失败）
            logger.error(f"Failed to register buffer, error code: {ret_code}")  # 记录注册失败日志
            raise RuntimeError(  # 抛出运行时错误
                f"Failed to register buffer to Mooncake Store, error code: {ret_code}"  # 向Mooncake存储注册缓冲区失败
            )


class MooncakeStore(HiCacheStorage, MooncakeBaseStore):  # Mooncake存储类，继承自HiCacheStorage和MooncakeBaseStore，实现完整的存储功能

    @staticmethod  # 静态方法
    def _standalone_required_bytes(mem_pool: Any) -> int:  # 计算独立模式下需要注册的总字节数 # 计算独立（虚拟客户端）模式下需要暴露给真实客户端的主机缓冲区总字节数
        """Compute total bytes of host buffers that must be visible to the real client.
        计算必须对真实客户端可见的主机缓冲区总字节数。

        In standalone (dummy client) mode, the real mooncake_client process needs
        to map any host buffers we will later pass by pointer via register_buffer().
        For hybrid models, that includes KV + sidecar pools (e.g. Mamba temporal/conv).
        在独立（虚拟客户端）模式下，真实的mooncake_client进程需要映射我们稍后通过register_buffer()
        以指针方式传递的所有主机缓冲区。对于混合模型，包括KV和辅助池（如Mamba的temporal/conv）。
        """
        # Prefer a generic "hybrid pool" accessor when present.  # 优先使用通用的"混合池"访问器（如果存在）
        total = 0  # 初始化总字节数为0
        seen_ptrs: set[int] = set()  # 已处理的指针集合，避免重复计算

        def _add_tensor(t: Optional[torch.Tensor]):  # 内部函数：将张量的字节数累加到total # 将单个张量的字节数添加到总计中（去重）
            nonlocal total  # 声明total为外部变量
            if t is None:  # 如果张量为None
                return  # 直接返回
            try:  # 尝试获取数据指针
                ptr = int(t.data_ptr())  # 获取张量数据指针
            except Exception:  # 如果获取失败
                return  # 直接返回
            if ptr in seen_ptrs:  # 如果指针已处理过
                return  # 跳过避免重复计算
            seen_ptrs.add(ptr)  # 将指针添加到已处理集合
            total += int(t.numel() * t.element_size())  # 累加张量的字节数

        # Always include the anchor KV buffer if present.  # 始终包含锚点KV缓冲区（如果存在）
        _add_tensor(getattr(mem_pool, "kv_buffer", None))  # 添加KV缓冲区张量

        # HostPoolGroup: include each pool's hybrid buffers when available.  # HostPoolGroup：当可用时包含每个池的混合缓冲区
        entries = getattr(mem_pool, "entries", None)  # 获取池条目列表
        if entries:  # 如果条目存在
            for entry in entries:  # 遍历每个条目
                host_pool = getattr(entry, "host_pool", None)  # 获取主机池
                if host_pool is None:  # 如果主机池为空
                    continue  # 跳过
                # KV pool anchor memory is already covered, but harmless if added twice.  # KV池锚点内存已被覆盖，但重复添加也无害
                _add_tensor(getattr(host_pool, "kv_buffer", None))  # 添加主机池的KV缓冲区
                for buf in getattr(host_pool, "get_hybrid_pool_buffer", lambda: [])():  # 遍历混合池缓冲区
                    _add_tensor(buf)  # 添加混合池缓冲区
            return total  # 返回总字节数

        # Single HostKVCache-like pool: add its sidecar buffers if any.  # 单个HostKVCache类池：添加其辅助缓冲区（如果有）
        for buf in getattr(mem_pool, "get_hybrid_pool_buffer", lambda: [])():  # 遍历混合池缓冲区
            _add_tensor(buf)  # 添加混合池缓冲区
        return total  # 返回总字节数

    def __init__(  # 初始化方法 # 初始化MooncakeStore实例
        self, storage_config: HiCacheStorageConfig = None, mem_pool: HostKVCache = None  # 参数：存储配置和主机KV缓存
    ):
        MooncakeBaseStore.__init__(self)  # 调用MooncakeBaseStore的初始化方法
        MooncakeDistributedStore = self._import_mooncake_store()  # 导入MooncakeDistributedStore类
        try:  # 尝试初始化
            self.store = MooncakeDistributedStore()  # 创建Mooncake分布式存储实例

            self.config = self._load_config(storage_config)  # 加载存储配置
            extra_config = (  # 获取额外配置
                getattr(storage_config, "extra_config", None)  # 从storage_config获取extra_config属性
                if storage_config  # 如果storage_config不为None
                else None  # 否则为None
            )
            tp_scale_factor = 1 if storage_config is None else storage_config.tp_size  # 张量并行缩放因子

            per_tp_global_segment_size = (  # 计算每个TP的全局段大小
                self.config.global_segment_size // tp_scale_factor  # 总全局段大小除以TP大小
            )

            # Check if extra_backend_tag should be passed to MooncakeDistributedStore  # 检查是否应将extra_backend_tag传递给MooncakeDistributedStore
            self.extra_backend_tag = None  # 初始化额外后端标签为None
            if extra_config and "extra_backend_tag" in extra_config:  # 如果额外配置中包含extra_backend_tag
                self.extra_backend_tag = extra_config["extra_backend_tag"]  # 设置额外后端标签
                logger.info(f"Using extra_backend_tag: {self.extra_backend_tag}")  # 记录使用的额外后端标签

            # Check server status  # 检查服务器状态
            if self.config.check_server:  # 如果配置要求检查服务器
                self.check_server()  # 调用检查服务器方法

            # Handle JSON device_name configuration  # 处理JSON格式的device_name配置
            device_name = self.config.device_name  # 获取设备名称配置
            if device_name and device_name.strip().startswith("{"):  # 如果设备名称以"{"开头（JSON格式）
                try:  # 尝试解析JSON
                    device_config = json.loads(device_name)  # 解析JSON配置
                    if storage_config and hasattr(storage_config, "tp_rank"):  # 如果有TP rank信息
                        tp_rank = storage_config.tp_rank  # 获取TP rank
                        # Try both integer and string keys since JSON parsing may convert keys  # 同时尝试整数和字符串键，因为JSON解析可能转换键类型
                        device_name = device_config.get(tp_rank, "")  # 尝试用整数键获取设备名
                        if not device_name:  # 如果未获取到
                            device_name = device_config.get(str(tp_rank), "")  # 尝试用字符串键获取设备名
                    else:  # 否则
                        device_name = ""  # 设备名为空
                except (json.JSONDecodeError, AttributeError):  # 捕获JSON解析错误和属性错误
                    logger.warning(  # 记录警告
                        f"Failed to parse device_name as JSON: {device_name}"  # 解析device_name为JSON失败
                    )
                    device_name = ""  # 设备名置为空
            if self.config.standalone_storage:  # 如果使用独立存储模式
                if not isinstance(mem_pool.allocator, MooncakeHostTensorAllocator):  # 检查分配器类型
                    raise RuntimeError(  # 抛出运行时错误
                        "MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator. "  # 独立存储模式需要MooncakeHostTensorAllocator
                        "Please set standalone_storage=False "  # 请设置standalone_storage=False
                        "or upgrade Mooncake by 'pip install mooncake --upgrade'."  # 或升级Mooncake
                    )
                required_bytes = self._standalone_required_bytes(mem_pool)  # 计算所需的字节数
                ret_code = self.store.setup_dummy(  # 设置虚拟客户端模式
                    required_bytes,  # 所需字节数
                    DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer  # 零拷贝接口不需要本地缓冲区 # 默认本地缓冲区大小
                    self.config.client_server_address,  # 客户端服务器地址
                )
            else:  # 非独立存储模式
                try:  # 尝试获取共享的Mooncake传输引擎
                    from sglang.srt.distributed.parallel_state import (  # 从parallel_state模块导入
                        get_mooncake_transfer_engine,  # 获取Mooncake传输引擎函数
                    )

                    self._shared_mooncake_transfer_engine = (  # 保存共享的传输引擎
                        get_mooncake_transfer_engine()  # 获取共享的Mooncake传输引擎
                    )
                except Exception:  # 捕获异常
                    self._shared_mooncake_transfer_engine = None  # 共享传输引擎设为None
                    logger.debug("Failed to reuse initialized mooncake transfer engine")  # 记录复用传输引擎失败

                # Only reuse the shared MooncakeTransferEngine when its
                # configuration matches the one used by MooncakeStore.
                # 仅当共享的MooncakeTransferEngine的配置与MooncakeStore使用的配置匹配时才复用。
                if (  # 检查是否可以复用共享传输引擎
                    self._shared_mooncake_transfer_engine is not None  # 共享传输引擎存在
                    and device_name  # 设备名不为空
                    == self._shared_mooncake_transfer_engine.get_ib_device()  # 设备名与共享引擎的IB设备匹配
                    and self.config.metadata_server == "P2PHANDSHAKE"  # 元数据服务器为P2P握手模式
                    and self.config.protocol == "rdma"  # 协议为RDMA
                ):
                    client_hostname = (  # 使用共享引擎的会话ID作为客户端主机名
                        self._shared_mooncake_transfer_engine.get_session_id()  # 获取会话ID
                    )
                    transfer_engine = self._shared_mooncake_transfer_engine.get_engine()  # 获取传输引擎
                    logger.info(  # 记录复用传输引擎信息
                        f"Reuse initialized mooncake transfer engine: {self._shared_mooncake_transfer_engine}"  # 复用已初始化的mooncake传输引擎
                    )
                else:  # 不能复用共享引擎
                    client_hostname = self.config.local_hostname  # 使用本地主机名
                    transfer_engine = None  # 传输引擎为None

                setup_kwargs = {}  # 初始化setup的额外参数
                if self.config.enable_ssd_offload:  # 如果启用SSD卸载
                    setup_kwargs["enable_ssd_offload"] = True  # 设置SSD卸载标志
                if self.config.ssd_offload_path is not None:  # 如果SSD卸载路径不为空
                    setup_kwargs["ssd_offload_path"] = self.config.ssd_offload_path  # 设置SSD卸载路径

                while True:  # 循环尝试setup
                    try:  # 尝试设置存储
                        ret_code = self.store.setup(  # 调用setup方法
                            client_hostname,  # 客户端主机名
                            self.config.metadata_server,  # 元数据服务器
                            per_tp_global_segment_size,  # 每个TP的全局段大小
                            DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer  # 零拷贝接口不需要本地缓冲区 # 默认本地缓冲区大小
                            self.config.protocol,  # 通信协议
                            device_name,  # 设备名
                            self.config.master_server_address,  # 主服务器地址
                            transfer_engine,  # 传输引擎
                            **setup_kwargs,  # 额外参数
                        )
                        break  # 设置成功，跳出循环
                    except TypeError as e:  # 捕获类型错误（不支持的参数）
                        unsupported_kwargs = [  # 查找不支持的参数
                            key for key in list(setup_kwargs) if key in str(e)  # 在错误信息中查找不支持的参数键
                        ]
                        if not unsupported_kwargs:  # 如果没有不支持的参数
                            raise  # 重新抛出异常
                        logger.warning(  # 记录警告
                            "The installed Mooncake version does not support the "  # 安装的Mooncake版本不支持
                            f"{', '.join(unsupported_kwargs)} parameter(s) in setup(). "  # setup()中不支持的参数
                            f"Retrying without {', '.join(unsupported_kwargs)}. "  # 不使用不支持的参数重试
                            "Please upgrade Mooncake to enable SSD offload support."  # 请升级Mooncake以启用SSD卸载支持
                        )
                        for key in unsupported_kwargs:  # 移除不支持的参数
                            setup_kwargs.pop(key, None)  # 从额外参数中移除不支持的键
            if ret_code:  # 如果返回码非零（设置失败）
                raise RuntimeError(  # 抛出运行时错误
                    f"Failed to setup Mooncake store, error code: {ret_code}"  # 设置Mooncake存储失败
                )
            logger.info("Mooncake store setup successfully.")  # 记录设置成功

            self.local_rank = (  # 设置本地rank
                storage_config.tp_rank if storage_config is not None else 0  # 使用配置中的tp_rank，否则为0
            )
            self.warmup()  # 执行预热操作
            logger.info("Mooncake store warmup successfully.")  # 记录预热成功

            self.enable_storage_metrics = False  # 初始化存储指标标志为False
            if storage_config is not None:  # 如果存储配置不为空
                self.is_mla_backend = storage_config.is_mla_model  # 是否为MLA后端
                self.pp_rank = storage_config.pp_rank  # 流水线并行rank
                self.pp_size = storage_config.pp_size  # 流水线并行大小
                self.attn_cp_rank = storage_config.attn_cp_rank  # 注意力上下文并行rank
                self.attn_cp_size = storage_config.attn_cp_size  # 注意力上下文并行大小
                self.enable_storage_metrics = storage_config.enable_storage_metrics  # 是否启用存储指标
            else:  # 存储配置为空
                self.is_mla_backend = False  # 默认非MLA后端
                self.local_rank = 0  # 默认本地rank为0
                self.pp_rank = 0  # 默认流水线rank为0
                self.pp_size = 1  # 默认流水线大小为1
                self.attn_cp_rank = 0  # 默认注意力CP rank为0
                self.attn_cp_size = 1  # 默认注意力CP大小为1

            self.enable_pp = self.pp_size > 1  # 是否启用流水线并行
            if self.enable_pp:  # 如果启用流水线并行
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"  # MHA后缀包含本地rank和流水线rank
                self.mla_suffix = f"{self.pp_rank}"  # MLA后缀只包含流水线rank
            else:  # 未启用流水线并行
                self.mha_suffix = f"{self.local_rank}"  # MHA后缀只包含本地rank
                self.mla_suffix = ""  # MLA后缀为空

            self.storage_config = storage_config  # 保存存储配置
            self.split_factor = 0  # 初始化头分割因子为0
            if self.storage_config.should_split_heads:  # 如果需要分割注意力头
                self.split_factor = (  # 计算分割因子
                    self.storage_config.tp_lcm_size // self.storage_config.tp_size  # TP最小公倍数大小除以TP大小
                )
                base_rank = self.local_rank * self.split_factor  # 计算基础rank
                target_ranks = [base_rank + i for i in range(self.split_factor)]  # 生成目标rank列表
                if self.enable_pp:  # 如果启用流水线并行
                    self.mha_suffix = [  # MHA后缀列表
                        f"{rank}_{self.pp_rank}" for rank in target_ranks  # 每个目标rank加上流水线rank
                    ]
                else:  # 未启用流水线并行
                    self.mha_suffix = [f"{rank}" for rank in target_ranks]  # MHA后缀列表只包含目标rank

            self.registered_pools = {}  # 初始化已注册池字典

            self.gb_per_page = None  # 每页GB数，稍后计算
            self.prefetch_pgs = []  # 预取页数列表
            self.backup_pgs = []  # 备份页数列表
            self.prefetch_bandwidth = []  # 预取带宽列表
            self.backup_bandwidth = []  # 备份带宽列表

        except ValueError as e:  # 捕获数值错误
            logger.error("Configuration loading failed: %s", e)  # 记录配置加载失败
            raise  # 重新抛出异常
        except Exception as exc:  # 捕获其他异常
            logger.error("An error occurred while loading the configuration: %s", exc)  # 记录加载配置时的错误
            raise  # 重新抛出异常

    def check_server(self):  # 检查Mooncake存储服务器是否已启动 # 检查Mooncake存储服务器状态，等待直到服务器启动或超时
        master_server_ip = self.config.master_server_address.split(":")[0]  # 从主服务器地址提取IP
        segments_url = f"http://{master_server_ip}:{self.config.master_metrics_port}/get_all_segments"  # 构造获取所有段的URL
        start_time = time.perf_counter()  # 记录开始时间

        check_result = False  # 初始化检查结果为False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:  # 在超时时间内循环
            try:  # 尝试请求
                check_segments_resp = requests.get(segments_url, timeout=3)  # 发送GET请求，超时3秒
            except Exception:  # 捕获请求异常
                logger.info(  # 记录等待信息
                    "waiting mooncake store server started, cost_time: %.2f seconds.",  # 等待Mooncake存储服务器启动，已耗时
                    time.perf_counter() - start_time,  # 已耗时秒数
                )
                time.sleep(3)  # 等待3秒后重试
                continue  # 继续循环

            if check_segments_resp.text == "":  # 如果响应为空
                logger.info(  # 记录等待信息
                    "waiting mooncake store server started, cost_time: %.2f seconds.",  # 等待Mooncake存储服务器启动，已耗时
                    time.perf_counter() - start_time,  # 已耗时秒数
                )
                time.sleep(3)  # 等待3秒后重试
                continue  # 继续循环

            logger.info("Mooncake store server started successfully.")  # 记录服务器启动成功
            check_result = True  # 设置检查结果为True
            break  # 跳出循环

        if not check_result:  # 如果检查结果为False（超时）
            logger.error("Launch mooncake store server timeout")  # 记录启动Mooncake存储服务器超时
            raise ValueError("Launch mooncake store server timeout")  # 抛出数值错误

    def warmup(self):  # 预热Mooncake存储 # 通过写入和读取测试键值对来预热Mooncake存储
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex  # 生成唯一的预热键
        warmup_value = bytes(4 * 1024)  # 4 KB  # 预热值为4KB的字节数据

        # Retry logic to handle Transfer Engine startup race condition  # 重试逻辑，处理传输引擎启动竞争条件
        max_retries = 10  # 最大重试次数
        retry_delay = 1.0  # seconds  # 重试延迟（秒）

        for attempt in range(max_retries):  # 遍历重试次数
            ret = self.store.put(warmup_key, warmup_value)  # 尝试写入预热数据
            if ret == 0:  # 如果写入成功
                break  # 跳出循环
            logger.warning(  # 记录警告
                f"[TP{self.local_rank}] Warmup put failed (attempt {attempt + 1}/{max_retries}), "  # 预热写入失败（尝试次数）
                f"ret={ret}, retrying in {retry_delay}s..."  # 返回码和重试延迟
            )
            time.sleep(retry_delay)  # 等待重试延迟时间
        else:  # 所有重试都失败
            raise RuntimeError(  # 抛出运行时错误
                f"[TP{self.local_rank}] Warmup put failed after {max_retries} attempts, "  # 预热写入在多次尝试后失败
                "Transfer Engine might not be ready"  # 传输引擎可能未就绪
            )

        assert self.store.is_exist(warmup_key) == 1  # 断言预热键存在
        assert self.store.get(warmup_key) == warmup_value  # 断言预热值正确

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):  # 注册主机内存池到Mooncake存储 # 将主机KV缓存注册到Mooncake存储，并注册其缓冲区
        super().register_mem_pool_host(mem_pool_host)  # 调用父类方法注册主机内存池
        assert self.mem_pool_host.layout in [  # 断言内存布局为支持的类型
            "page_first",  # 页优先布局
            "page_first_direct",  # 页优先直接布局
            "page_head",  # 页头布局
            "page_first_kv_split",  # 页优先KV分割布局
        ], "mooncake store storage backend only support page first, page first direct, page head and  page_first_kv_split layout"  # mooncake存储后端只支持page_first、page_first_direct、page_head和page_first_kv_split布局
        buffer = self.mem_pool_host.kv_buffer  # 获取KV缓冲区
        try:  # 尝试注册缓冲区
            super().register_buffer(buffer)  # 调用父类方法注册缓冲区
        except TypeError as err:  # 捕获类型错误
            logger.error("Failed to register buffer to Mooncake Store: %s", err)  # 记录注册缓冲区失败
            raise TypeError("Mooncake Store Register Buffer Error.") from err  # 抛出类型错误

        bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size  # 计算每页字节数
        self.gb_per_page = bytes_per_page / (1 << 30)  # 将每页字节数转换为GB

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):  # 注册v2版本的主机内存池 # 注册额外的混合池到Mooncake存储（v2版本）
        # KV anchor memory is already registered via register_mem_pool_host().  # KV锚点内存已通过register_mem_pool_host()注册。
        # v2 here only registers additional hybrid pools.  # v2版本只注册额外的混合池。
        if host_pool_name == PoolName.KV:  # 如果是KV池
            return  # 已注册，直接返回
        # Keep a name->pool mapping so batch v2 can resolve PoolTransfer.name to
        # the corresponding host pool implementation at runtime.
        # 保持名称到池的映射，以便批量v2在运行时将PoolTransfer.name解析为对应的主机池实现。
        self.registered_pools[host_pool_name] = host_pool  # 将池添加到已注册池字典

        # Hybrid pools expose the tensors that Mooncake needs for zero-copy I/O.
        # The storage backend only depends on this accessor, not concrete fields.
        # 混合池暴露Mooncake零拷贝I/O所需的张量。
        # 存储后端仅依赖此访问器，而不依赖具体字段。
        buf_list = host_pool.get_hybrid_pool_buffer()  # 获取混合池缓冲区列表
        for buf in buf_list:  # 遍历每个缓冲区
            super().register_buffer(buf)  # 调用父类方法注册缓冲区

    def _tag_keys(self, keys: List[str]) -> List[str]:  # 为键添加额外后端标签前缀 # 如果有extra_backend_tag，则为所有键添加标签前缀
        if self.extra_backend_tag is None:  # 如果没有额外后端标签
            return keys  # 直接返回原键列表
        return [f"{ self.extra_backend_tag}_{key}" for key in keys]  # 为每个键添加标签前缀

    def _get_hybrid_page_component_keys(  # 获取混合页组件键列表 # 根据池传输类型生成混合页的组件存储键
        self, page_keys: List[str], transfer: PoolTransfer  # 参数：页键列表和池传输对象
    ) -> Tuple[List[str], int]:  # 返回组件键列表和每个页的键乘数
        # A logical "page" may map to multiple physical objects in storage.
        # - INDEXER: one key per page
        # - MAMBA  : one temporal key + N conv keys per page
        # key_multiplier records how many component keys are generated per page.
        # 一个逻辑"页"可以映射到存储中的多个物理对象。
        # - INDEXER：每页一个键
        # - MAMBA：每页一个temporal键 + N个conv键
        # key_multiplier记录每页生成多少个组件键。
        name = transfer.name  # 获取池名称
        suffixes = []  # 初始化后缀列表
        if name == PoolName.INDEXER:  # 如果是INDEXER池
            suffixes = [f"_{self.mla_suffix}_{PoolName.INDEXER}"]  # INDEXER使用MLA后缀
        elif name == PoolName.MAMBA:  # 如果是MAMBA池
            pools = getattr(self, "registered_pools", {})  # 获取已注册池
            mamba_pool = pools.get(PoolName.MAMBA)  # 获取MAMBA池
            conv_num = len(getattr(mamba_pool, "conv_buffer", None) or [])  # 获取conv缓冲区数量
            base_suffix = f"_{self.mha_suffix}"  # 基础后缀
            suffixes = [f"{base_suffix}_temporal"] + [  # temporal键后缀
                f"{base_suffix}_conv_{i}" for i in range(conv_num)  # conv键后缀列表
            ]
        key_multiplier = len(suffixes)  # 键乘数等于后缀数量
        component_keys = [  # 生成组件键列表
            f"{page_key}{suffix}" for page_key in page_keys for suffix in suffixes  # 每个页键配上每个后缀
        ]
        return component_keys, key_multiplier  # 返回组件键列表和键乘数

    def batch_exists_v2(  # 批量检查键是否存在（v2版本） # 批量检查多个池的键是否存在，支持混合页组件
        self,
        keys: List[str],  # 键列表
        pool_transfers: Optional[List[PoolTransfer]] = None,  # 池传输列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> PoolTransferResult:  # 返回池传输结果
        kv_pages = self.batch_exists(keys, extra_info)  # 先检查KV页是否存在

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}  # KV池命中数
        final_pages = kv_pages  # 最终匹配页数

        for transfer in pool_transfers or []:  # 遍历每个池传输
            if final_pages == 0:  # 如果没有匹配页
                break  # 跳出循环
            component_keys, key_multiplier = self._get_hybrid_page_component_keys(  # 获取组件键
                keys, transfer  # 传入键和传输对象
            )
            component_keys = self._tag_keys(component_keys)  # 为组件键添加标签前缀
            ex = self._batch_exist(component_keys)  # 批量检查组件键是否存在
            if key_multiplier > 0:  # 如果键乘数大于0
                page_exists = [  # 计算每个页是否存在
                    all(  # 所有组件键都存在才算页存在
                        r == 1  # 结果为1表示存在
                        for r in ex[i * key_multiplier : (i + 1) * key_multiplier]  # 取该页对应的所有组件键结果
                    )
                    for i in range(kv_pages)  # 遍历每个KV页
                ]
            else:  # 键乘数为0
                page_exists = [False] * kv_pages  # 所有页不存在
            boundary = 0  # 边界初始化为0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:  # 如果命中策略为所有页
                try:  # 尝试找到第一个不存在的页
                    boundary = page_exists.index(False)  # 找到第一个False的索引
                except ValueError:  # 如果所有页都存在
                    boundary = kv_pages  # 边界为总页数
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:  # 如果命中策略为尾随页
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)  # 尾随页数
                for prefix_len in range(kv_pages, 0, -1):  # 从后向前遍历
                    if all(  # 检查尾随页是否都存在
                        page_exists[i]  # 页存在
                        for i in range(max(0, prefix_len - trailing), prefix_len)  # 尾随页范围
                    ):
                        boundary = prefix_len  # 设置边界
                        break  # 跳出循环
            if boundary:  # 如果边界大于0
                hit_count[transfer.name] = boundary  # 记录该池的命中数
            final_pages = min(final_pages, boundary)  # 取所有池的最小命中数

        return PoolTransferResult(final_pages, hit_count)  # 返回池传输结果

    def _batch_io_v2(self, transfers: List[PoolTransfer], is_set: bool):  # v2版本的批量I/O操作 # 统一的v2批量读写路径，每个PoolTransfer可以展开为多个存储对象
        # Unified v2 I/O path: each PoolTransfer can expand to one or more
        # storage objects per logical page, but API still reports page-level result.
        # 统一的v2 I/O路径：每个PoolTransfer可以展开为每个逻辑页的一个或多个存储对象，
        # 但API仍报告页级别的结果。
        results: dict = {}  # 初始化结果字典
        for transfer in transfers:  # 遍历每个传输
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)  # 获取对应的主机池
            keys = transfer.keys  # 获取键列表
            page_size = getattr(host_pool, "page_size", 1) or 1  # 获取页大小
            host_indices = transfer.host_indices  # 获取主机索引
            assert len(keys) > 0  # 断言键列表不为空
            assert len(keys) == len(host_indices) // page_size  # 断言键数量与索引数量匹配

            ptr_list, element_size_list = host_pool.get_page_buffer_meta(host_indices)  # 获取页缓冲区元数据
            key_strs, key_multiplier = self._get_hybrid_page_component_keys(  # 获取组件键
                keys, transfer  # 传入键和传输对象
            )
            key_strs = self._tag_keys(key_strs)  # 为键添加标签前缀

            if is_set:  # 如果是写操作
                exist_result = self._batch_exist(key_strs)  # 批量检查键是否存在
                io_results = [0 if state == 1 else -1 for state in exist_result]  # 已存在的标记为0，不存在的标记为-1
                missing_idx = [i for i, state in enumerate(exist_result) if state != 1]  # 获取不存在的键的索引
                if missing_idx:  # 如果有不存在的键
                    put_results = self._put_batch_zero_copy_impl(  # 批量写入不存在的键
                        [key_strs[i] for i in missing_idx],  # 缺失的键
                        [ptr_list[i] for i in missing_idx],  # 对应的指针
                        [element_size_list[i] for i in missing_idx],  # 对应的大小
                    )
                    for i, res in zip(missing_idx, put_results):  # 更新写入结果
                        io_results[i] = res  # 将写入结果填入对应位置
            else:  # 如果是读操作
                io_results = self._get_batch_zero_copy_impl(  # 批量零拷贝读取
                    key_strs, ptr_list, element_size_list  # 键、指针和大小列表
                )
            results[transfer.name] = self._batch_postprocess(  # 后处理结果
                io_results, is_set_operate=is_set, key_multiplier=key_multiplier  # I/O结果、操作类型和键乘数
            )
        return results  # 返回结果字典

    def batch_get_v2(  # 批量读取v2版本 # 批量从存储中读取多个池的数据（v2版本）
        self,
        transfers: List[PoolTransfer],  # 池传输列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> dict:  # 返回结果字典
        return self._batch_io_v2(transfers, is_set=False)  # 调用v2批量I/O，设置为读操作

    def batch_set_v2(  # 批量写入v2版本 # 批量向存储中写入多个池的数据（v2版本）
        self,
        transfers: List[PoolTransfer],  # 池传输列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> dict:  # 返回结果字典
        return self._batch_io_v2(transfers, is_set=True)  # 调用v2批量I/O，设置为写操作

    def _get_mha_split_heads_buffer_meta(self, keys, indices):  # 获取MHA分割头的缓冲区元数据 # 获取多头注意力分割头模式下的键列表和缓冲区指针/大小
        ptr_list, element_size_list = (  # 获取分割头的页缓冲区元数据
            self.mem_pool_host.get_split_heads_page_buffer_meta(  # 调用分割头页缓冲区元数据方法
                indices, self.split_factor  # 传入索引和分割因子
            )
        )
        key_list = []  # 初始化键列表
        for key_ in keys:  # 遍历每个键
            for suffix in self.mha_suffix:  # 遍历每个MHA后缀
                key_list.append(f"{key_}_{suffix}_k")  # 添加K键
                key_list.append(f"{key_}_{suffix}_v")  # 添加V键
        assert len(key_list) == len(ptr_list)  # 断言键数量与指针数量匹配
        return key_list, ptr_list, element_size_list  # 返回键列表、指针列表和大小列表

    def _get_mha_buffer_meta(self, keys, indices):  # 获取MHA缓冲区元数据 # 获取多头注意力模式下的键列表和缓冲区指针/大小
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)  # 获取页缓冲区元数据
        key_list = []  # 初始化键列表
        for key_ in keys:  # 遍历每个键
            key_list.append(f"{key_}_{self.mha_suffix}_k")  # 添加K键
            key_list.append(f"{key_}_{self.mha_suffix}_v")  # 添加V键
        assert len(key_list) == len(ptr_list)  # 断言键数量与指针数量匹配
        return key_list, ptr_list, element_size_list  # 返回键列表、指针列表和大小列表

    def _get_mla_buffer_meta(self, keys, indices):  # 获取MLA缓冲区元数据 # 获取多潜在注意力模式下的键列表和缓冲区指针/大小
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)  # 获取页缓冲区元数据
        key_list = []  # 初始化键列表
        for key_ in keys:  # 遍历每个键
            key_list.append(f"{key_}_{self.mla_suffix}_k")  # 添加K键（MLA只有K键）
        assert len(key_list) == len(ptr_list)  # 断言键数量与指针数量匹配
        return key_list, ptr_list, element_size_list  # 返回键列表、指针列表和大小列表

    def _batch_preprocess(self, keys, host_indices):  # 批量预处理，根据模型类型选择缓冲区元数据获取方式 # 为批量读写操作预处理键和缓冲区元数据
        assert len(keys) > 0  # 断言键列表不为空
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size  # 断言键数量与索引数量匹配
        if self.is_mla_backend:  # 如果是MLA后端
            return self._get_mla_buffer_meta(keys, host_indices)  # 使用MLA缓冲区元数据
        else:  # 非MLA后端
            if self.storage_config.should_split_heads:  # 如果需要分割头
                return self._get_mha_split_heads_buffer_meta(keys, host_indices)  # 使用MHA分割头缓冲区元数据
            else:  # 不需要分割头
                return self._get_mha_buffer_meta(keys, host_indices)  # 使用MHA缓冲区元数据

    def _batch_postprocess(  # 批量后处理，将NIXL返回值转换为布尔结果 # 将批量I/O操作的原始结果转换为布尔值列表
        self, results: List[int], is_set_operate=False, key_multiplier=None  # 参数：原始结果列表、是否为写操作、键乘数
    ):
        """
        refer to https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-store/include/pybind_client.h
        for batch_get_into, results is Vector of integers,
            where each element is the number of bytes read on success, or a negative value on error
        for batch_put_from, results is Vector of integers,
            where each element is 0 on success, or a negative value on error
        参见 https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-store/include/pybind_client.h
        对于batch_get_into，结果是整数向量，
            其中每个元素是成功时读取的字节数，或错误时的负值
        对于batch_put_from，结果是整数向量，
            其中每个元素是成功时为0，或错误时的负值
        """
        if key_multiplier is None:  # 如果键乘数未指定
            if self.is_mla_backend:  # 如果是MLA后端
                key_multiplier = 1  # MLA每个页一个键
            else:  # 非MLA后端
                key_multiplier = 2  # MHA每个页两个键（K和V）
                if self.storage_config.should_split_heads:  # 如果需要分割头
                    key_multiplier *= self.split_factor  # 键乘数乘以分割因子

        result_groups = [  # 将结果按key_multiplier分组
            results[i : i + key_multiplier]  # 每组包含key_multiplier个结果
            for i in range(0, len(results), key_multiplier)  # 按步长key_multiplier遍历
        ]
        return [  # 返回布尔值列表
            (
                all(res == 0 for res in group)  # 写操作：所有结果为0表示成功
                if is_set_operate  # 如果是写操作
                else all(res > 0 for res in group)  # 读操作：所有结果大于0表示成功
            )
            for group in result_groups  # 遍历每个结果组
        ]

    def batch_get_v1(  # 批量读取v1版本 # 批量从存储中读取KV缓存数据（v1版本）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:  # 返回布尔值列表
        # Apply extra_backend_tag prefix if available  # 如果有extra_backend_tag则应用前缀
        keys = self._tag_keys(keys)  # 为键添加标签前缀

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)  # 批量预处理

        start_time = time.perf_counter()  # 记录开始时间
        get_results = self._get_batch_zero_copy_impl(  # 批量零拷贝读取
            key_strs, buffer_ptrs, buffer_sizes  # 键、指针和大小列表
        )
        end_time = time.perf_counter()  # 记录结束时间

        if self.enable_storage_metrics:  # 如果启用存储指标
            self.prefetch_pgs.append(len(keys))  # 记录预取页数
            self.prefetch_bandwidth.append(  # 记录预取带宽
                len(keys) / (end_time - start_time) * self.gb_per_page  # 页数/耗时*每页GB数=带宽GB/s
            )

        return self._batch_postprocess(get_results, is_set_operate=False)  # 后处理并返回结果

    def batch_set_v1(  # 批量写入v1版本 # 批量向存储中写入KV缓存数据（v1版本）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:  # 返回布尔值列表
        # Apply extra_backend_tag prefix if available  # 如果有extra_backend_tag则应用前缀
        keys = self._tag_keys(keys)  # 为键添加标签前缀

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)  # 批量预处理
        exist_result = self._batch_exist(key_strs)  # 批量检查键是否存在

        set_keys = []  # 需要写入的键列表
        set_buffer_ptrs = []  # 需要写入的指针列表
        set_buffer_sizes = []  # 需要写入的大小列表
        set_indices = []  # 需要写入的索引列表
        set_results = [-1] * len(key_strs)  # 初始化所有结果为-1（失败）
        for i in range(len(key_strs)):  # 遍历所有键
            if exist_result[i] != 1:  # 如果键不存在于存储中
                set_keys.append(key_strs[i])  # 添加到写入键列表
                set_buffer_ptrs.append(buffer_ptrs[i])  # 添加到写入指针列表
                set_buffer_sizes.append(buffer_sizes[i])  # 添加到写入大小列表
                set_indices.append(i)  # 添加到写入索引列表
            else:  # 如果键已存在
                set_results[i] = 0  # 标记为成功（无需写入）

        # Only set non-existing keys to storage  # 只写入存储中不存在的键
        if len(set_keys) > 0:  # 如果有需要写入的键
            start_time = time.perf_counter()  # 记录开始时间
            put_results = self._put_batch_zero_copy_impl(  # 批量零拷贝写入
                set_keys, set_buffer_ptrs, set_buffer_sizes  # 键、指针和大小列表
            )
            end_time = time.perf_counter()  # 记录结束时间

            if self.enable_storage_metrics:  # 如果启用存储指标
                self.backup_pgs.append(len(set_keys))  # 记录备份页数
                self.backup_bandwidth.append(  # 记录备份带宽
                    len(set_keys) / (end_time - start_time) * self.gb_per_page  # 页数/耗时*每页GB数=带宽GB/s
                )

            for i in range(len(set_indices)):  # 遍历写入索引
                set_results[set_indices[i]] = put_results[i]  # 更新写入结果

        return self._batch_postprocess(set_results, is_set_operate=True)  # 后处理并返回结果

    def set(  # 写入单个键值对 # 向存储中写入单个键值对（零拷贝方式）
        self,
        key,  # 键
        value: Optional[Any] = None,  # 值（未使用）
        target_location: Optional[List[int]] = None,  # 目标位置指针
        target_sizes: Optional[List[int]] = None,  # 目标大小
    ) -> bool:  # 返回是否成功
        # Only support zero copy set for now  # 目前只支持零拷贝写入
        assert target_location is not None and target_sizes is not None  # 断言目标位置和大小不为空
        exist_result = self._batch_exist([key])  # 检查键是否已存在
        if exist_result[0] == 1:  # 如果键已存在
            return True  # 返回成功
        put_result = self._put_batch_zero_copy_impl(  # 零拷贝写入
            [key], [target_location], [target_sizes]  # 键、目标位置和大小
        )
        return put_result[0] == 0  # 返回是否成功（0表示成功）

    def batch_set(  # 批量写入键值对 # 向存储中批量写入键值对（零拷贝方式）
        self,
        keys: List[str],  # 键列表
        values: Optional[List[torch.Tensor]] = None,  # 值列表（未使用）
        target_locations: Optional[List[int]] = None,  # 目标位置指针列表
        target_sizes: Optional[List[int]] = None,  # 目标大小列表
    ) -> bool:  # 返回是否全部成功
        # Only support zero copy set for now  # 目前只支持零拷贝写入
        assert target_locations is not None and target_sizes is not None  # 断言目标位置和大小不为空
        assert len(keys) == len(target_locations) == len(target_sizes)  # 断言列表长度一致

        if len(keys) == 0:  # 如果键列表为空
            return False  # 返回失败

        for i in range(len(keys)):  # 遍历所有键
            if (  # 检查键和目标是否有效
                keys[i] is None  # 键为None
                or target_locations[i] is None  # 目标位置为None
                or target_sizes[i] is None  # 目标大小为None
            ):
                return False  # 返回失败

        exist_result = self._batch_exist(keys)  # 批量检查键是否存在
        set_keys = []  # 需要写入的键列表
        set_target_locations = []  # 需要写入的目标位置列表
        set_target_sizes = []  # 需要写入的目标大小列表
        set_indices = []  # 需要写入的索引列表
        for i in range(len(keys)):  # 遍历所有键
            if exist_result[i] != 1:  # 如果键不存在
                set_keys.append(keys[i])  # 添加到写入列表
                set_target_locations.append(target_locations[i])  # 添加目标位置
                set_target_sizes.append(target_sizes[i])  # 添加目标大小
                set_indices.append(i)  # 添加索引
        # Only set non-existing keys to storage  # 只写入存储中不存在的键
        start_time = time.perf_counter()  # 记录开始时间
        put_result = self._put_batch_zero_copy_impl(  # 批量零拷贝写入
            set_keys, set_target_locations, set_target_sizes  # 键、目标位置和大小列表
        )
        end_time = time.perf_counter()  # 记录结束时间

        if self.enable_storage_metrics:  # 如果启用存储指标
            self.backup_pgs.append(len(set_keys))  # 记录备份页数
            self.backup_bandwidth.append(  # 记录备份带宽
                len(set_keys) / (end_time - start_time) * self.gb_per_page  # 页数/耗时*每页GB数=带宽GB/s
            )

        for i in range(len(set_indices)):  # 遍历写入索引
            if put_result[i] == 0:  # 如果写入成功
                exist_result[set_indices[i]] = 1  # 更新存在结果

        success_count = 0  # 连续成功计数
        for i in range(len(keys)):  # 遍历所有键
            if exist_result[i] == 0:  # 如果键不存在（写入失败）
                break  # 跳出循环
            success_count += 1  # 成功计数加1
        # TODO: return the number of consecutive successful operations from the start.  # TODO：返回从开始连续成功操作的数量。
        return success_count == len(keys)  # 返回是否全部成功

    def get(  # 读取单个键值对 # 从存储中读取单个键值对（零拷贝方式）
        self,
        key,  # 键
        target_location: Optional[Any] = None,  # 目标位置指针
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> bool:  # 返回是否成功
        assert target_location is not None and target_sizes is not None  # 断言目标位置和大小不为空
        get_result = self._get_batch_zero_copy_impl(  # 零拷贝读取
            [key], [target_location], [target_sizes]  # 键、目标位置和大小
        )
        return get_result[0] >= 0  # 返回是否成功（非负值表示成功）

    def batch_get(  # 批量读取键值对 # 从存储中批量读取键值对（零拷贝方式），返回连续成功读取的页数
        self,
        keys: List[str],  # 键列表
        target_locations: Optional[Any] = None,  # 目标位置指针列表
        target_sizes: Optional[Any] = None,  # 目标大小列表
    ) -> int:  # 返回连续成功读取的页数
        assert len(keys) == len(target_locations) == len(target_sizes)  # 断言列表长度一致
        if len(keys) == 0:  # 如果键列表为空
            return 0  # 返回0

        start_time = time.perf_counter()  # 记录开始时间
        get_result = self._get_batch_zero_copy_impl(  # 批量零拷贝读取
            keys, target_locations, target_sizes  # 键、目标位置和大小列表
        )
        end_time = time.perf_counter()  # 记录结束时间

        if self.is_mla_backend:  # 如果是MLA后端
            key_multiplier = 1  # 键乘数为1
        else:  # 非MLA后端
            key_multiplier = 2  # 键乘数为2

        if self.enable_storage_metrics:  # 如果启用存储指标
            self.prefetch_pgs.append(len(keys))  # 记录预取页数
            self.prefetch_bandwidth.append(  # 记录预取带宽
                len(keys) / (end_time - start_time) * self.gb_per_page  # 页数/耗时*每页GB数=带宽GB/s
            )

        for i in range(len(keys)):  # 遍历读取结果
            if get_result[i] < 0:  # 如果读取失败
                return i // key_multiplier  # 返回失败前的连续成功页数
        return len(keys) // key_multiplier  # 返回总成功页数

    def exists(self, key) -> bool:  # 检查单个键是否存在 # 检查指定键是否存在于存储中
        exist_result = self._batch_exist([key])  # 批量检查键是否存在
        return exist_result[0] == 1  # 返回是否存在（1表示存在）

    def batch_exists(  # 批量检查键是否存在 # 批量检查多个键是否存在于存储中，返回连续存在的页数
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None  # 参数：键列表和额外信息
    ) -> int:  # 返回连续存在的页数
        # Apply extra_backend_tag prefix if available  # 如果有extra_backend_tag则应用前缀
        keys = self._tag_keys(keys)  # 为键添加标签前缀

        if self.is_mla_backend:  # 如果是MLA后端
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]  # MLA只查询K键
            key_multiplier = 1  # 键乘数为1
        else:  # 非MLA后端
            query_keys = []  # 初始化查询键列表
            if self.storage_config.should_split_heads:  # 如果需要分割头
                for key in keys:  # 遍历每个键
                    for suffix in self.mha_suffix:  # 遍历每个MHA后缀
                        query_keys.append(f"{key}_{suffix}_k")  # 添加K键
                        query_keys.append(f"{key}_{suffix}_v")  # 添加V键
                key_multiplier = 2 * self.split_factor  # 键乘数为2*分割因子
            else:  # 不需要分割头
                for key in keys:  # 遍历每个键
                    query_keys.append(f"{key}_{self.mha_suffix}_k")  # 添加K键
                    query_keys.append(f"{key}_{self.mha_suffix}_v")  # 添加V键
                key_multiplier = 2  # 键乘数为2

        exist_result = self._batch_exist(query_keys)  # 批量检查查询键是否存在
        for i in range(len(query_keys)):  # 遍历查询结果
            if exist_result[i] != 1:  # 如果键不存在
                return i // key_multiplier  # 返回连续存在的页数
        return len(query_keys) // key_multiplier  # 返回总存在的页数

    def close(self):  # 关闭存储 # 关闭Mooncake存储（无需手动关闭，析构函数会自动调用）
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        # MooncakeDistributedStore会自动调用析构函数，因此无需手动关闭。
        pass  # 空操作

    def clear(self) -> None:  # 清除存储中的所有数据 # 清除Mooncake存储中的所有数据
        self.store.remove_all()  # 调用remove_all删除所有数据

    def _put_batch_zero_copy_impl(  # 批量零拷贝写入实现 # 批量零拷贝写入的底层实现，调用Mooncake的batch_put_from
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]  # 参数：键列表、指针列表、大小列表
    ) -> List[int]:  # 返回结果列表
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)  # 调用Mooncake的批量写入方法

    def _get_batch_zero_copy_impl(  # 批量零拷贝读取实现 # 批量零拷贝读取的底层实现，调用Mooncake的batch_get_into
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]  # 参数：键列表、指针列表、大小列表
    ) -> List[int]:  # 返回结果列表
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)  # 调用Mooncake的批量读取方法

    def _batch_exist(self, key_strs: List[str]) -> List[int]:  # 批量检查键是否存在实现 # 批量检查键是否存在的底层实现，调用Mooncake的batch_is_exist
        return self.store.batch_is_exist(key_strs)  # 调用Mooncake的批量存在检查方法

    def get_stats(self):  # 获取存储统计数据 # 获取并重置存储性能指标（预取和备份的页数及带宽）
        storage_metrics = StorageMetrics()  # 创建存储指标对象
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)  # 复制预取页数数据
        storage_metrics.backup_pgs.extend(self.backup_pgs)  # 复制备份页数数据
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)  # 复制预取带宽数据
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)  # 复制备份带宽数据
        self.prefetch_pgs.clear()  # 清空预取页数
        self.backup_pgs.clear()  # 清空备份页数
        self.prefetch_bandwidth.clear()  # 清空预取带宽
        self.backup_bandwidth.clear()  # 清空备份带宽
        return storage_metrics  # 返回存储指标对象
