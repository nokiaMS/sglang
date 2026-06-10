# NIXL存储后端实现模块
# 本模块实现了基于NIXL（NVIDIA IO）插件的HiCache存储后端，
# 提供零拷贝和非零拷贝两种模式的KV缓存读写，
# 支持FILE和OBJ两种存储类型，以及多种NIXL后端插件（POSIX/GDS/3FS/OBJ等）。

import logging  # 导入logging模块，用于日志记录
import time  # 导入time模块，用于时间相关操作
import uuid  # 导入uuid模块，用于生成唯一标识符
from typing import Any, List, Optional  # 从typing模块导入类型注解

import torch  # 导入torch模块，用于张量操作

from sglang.srt.environ import envs  # 从environ模块导入环境变量配置
from sglang.srt.mem_cache.hicache_storage import (  # 从hicache_storage模块导入存储相关类
    STORAGE_BATCH_SIZE,  # 存储批量大小常量
    HiCacheStorage,  # HiCache存储基类
    HiCacheStorageConfig,  # HiCache存储配置类
    HiCacheStorageExtraInfo,  # HiCache存储额外信息类
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # 从memory_pool_host模块导入主机KV缓存类
from sglang.srt.mem_cache.mmap_allocator import alloc_mmap  # 从mmap_allocator模块导入mmap分配函数

from .nixl_registry import NixlRegistry  # 从当前包导入NIXL注册表类
from .nixl_utils import NixlBackendConfig, NixlBackendSelection, NixlFileManager  # 从当前包导入NIXL工具类

try:  # 尝试导入NIXL库
    from nixl._api import nixl_agent, nixl_agent_config, nixlBind  # 从nixl._api导入NIXL代理和配置类
except ImportError as e:  # 捕获导入错误
    raise ImportError(  # 抛出导入错误
        "Please install NIXL by following the instructions at "  # 请按照以下地址安装NIXL
        "https://github.com/ai-dynamo/nixl/blob/main/README.md "  # 安装说明链接
        "to use HiCacheNixl storage backend."  # 以使用HiCacheNixl存储后端
    ) from e

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class HiCacheNixl(HiCacheStorage):  # HiCacheNixl类，继承自HiCacheStorage，提供基于NIXL插件的高性能存储
    """HiCacheNixl provides high-performance storage using NIXL plugins."""  # HiCacheNixl使用NIXL插件提供高性能存储。

    def __init__(  # 初始化方法
        self,
        storage_config: HiCacheStorageConfig,  # 存储配置
        file_path: str = "/tmp/hicache_storage",  # 文件存储路径，默认/tmp/hicache_storage
    ):
        """Initialize NIXL storage connector."""  # 初始化NIXL存储连接器。

        # create nixlconfig from the --hicache-storage-backend-extra-config  # 从--hicache-storage-backend-extra-config创建nixlconfig
        nixlconfig = NixlBackendConfig(storage_config.extra_config)  # 从额外配置创建NIXL后端配置

        # select the NIXL backend plugin from extra_config or environment variable  # 从extra_config或环境变量选择NIXL后端插件
        plugin = nixlconfig.get_specified_plugin()  # 获取指定的插件名称

        use_direct_io = nixlconfig.get_use_direct_io()  # 获取是否使用直接I/O

        # Might be better to be unified across HiCache backends and moved to HiCacheController  # 可能最好在所有HiCache后端统一处理并移至HiCacheController
        file_path = envs.SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR.get() or file_path  # 从环境变量获取文件路径或使用默认值
        self.file_manager = (  # 创建文件管理器（仅FILE类型需要）
            NixlFileManager(file_path, use_direct_io=use_direct_io)  # 创建文件管理器实例
            if plugin not in NixlBackendSelection.OBJ_PLUGINS  # 如果不是OBJ插件
            else None  # OBJ插件不需要文件管理器
        )

        tp_rank, tp_size, model_name = (  # 获取张量并行和模型信息
            storage_config.tp_rank,  # TP rank
            storage_config.tp_size,  # TP大小
            storage_config.model_name,  # 模型名称
        )

        self.is_mla_model = storage_config.is_mla_model  # 是否为MLA模型
        self.is_zero_copy = False  # 是否为零拷贝模式，初始为False
        self.storage_config = storage_config  # 保存存储配置
        self.backup_skip = self.is_mla_model and storage_config.tp_rank != 0  # MLA模式下非零rank跳过备份

        model_name = "-".join(model_name.split("/")) if model_name else ""  # 将模型名中的"/"替换为"-"

        if self.is_mla_model:  # 如果是MLA模型
            self.config_suffix = f"_{model_name}"  # 配置后缀只包含模型名
        else:  # 非MLA模型
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"  # 配置后缀包含模型名和TP信息

        sync_mode = getattr(  # 获取同步模式
            nixlBind, "NIXL_THREAD_SYNC_RW", nixlBind.NIXL_THREAD_SYNC_STRICT  # 优先使用RW模式，否则使用STRICT模式
        )
        agent_config = nixl_agent_config(backends=[])  # 创建代理配置，后端列表为空
        self.agent_name = f"hicache_nixl_{str(uuid.uuid4())}"  # 生成唯一的代理名称
        self.agent = nixl_agent(self.agent_name, agent_config)  # 创建NIXL代理
        bind_cfg = nixlBind.nixlAgentConfig()  # 创建NIXL绑定配置
        bind_cfg.useProgThread = agent_config.enable_pthread  # 是否启用进度线程
        bind_cfg.useListenThread = agent_config.enable_listen  # 是否启用监听线程
        bind_cfg.listenPort = agent_config.port  # 监听端口
        bind_cfg.syncMode = sync_mode  # 同步模式
        bind_cfg.pthrDelay = 0  # 进度线程延迟
        bind_cfg.lthrDelay = 100000  # 监听线程延迟
        bind_cfg.captureTelemetry = agent_config.capture_telemetry  # 是否捕获遥测数据
        self.agent.agent = nixlBind.nixlAgent(self.agent_name, bind_cfg)  # 创建底层NIXL代理
        self.agent.plugin_list = self.agent.agent.getAvailPlugins()  # 获取可用插件列表

        self.backend_selector = NixlBackendSelection(plugin, nixlconfig)  # 创建后端选择器
        if not self.backend_selector.create_backend(self.agent):  # 创建后端，如果失败
            raise RuntimeError("Failed to create NIXL backend")  # 抛出运行时错误

        self.registry = NixlRegistry(  # 创建NIXL注册表
            self.agent,  # NIXL代理
            self.backend_selector.mem_type,  # 内存类型
            self.file_manager,  # 文件管理器
        )
        # O_DIRECT requires OS-page-aligned I/O buffers on all file-based backends
        # (POSIX, GDS, GDS_MT, 3FS). OBJ backends never open files so they are exempt
        # (file_manager is None for OBJ).
        # O_DIRECT要求所有基于文件的后端（POSIX、GDS、GDS_MT、3FS）使用OS页对齐的I/O缓冲区。
        # OBJ后端从不打开文件，因此不受此限制（OBJ的file_manager为None）。
        self.needs_page_alignment = use_direct_io and self.file_manager is not None  # 是否需要页对齐
        if self.needs_page_alignment:  # 如果需要页对齐
            logger.info(  # 记录页对齐信息
                "HiCacheNixl: O_DIRECT is active with a file-based backend (%s). "  # O_DIRECT在使用文件后端时激活
                "Page-aligned host buffers are required (needs_page_alignment=True).",  # 需要页对齐的主机缓冲区
                self.backend_selector.backend_name,  # 后端名称
            )
        # Pre-registered host regions (set by register_mem_pool_host):
        # zero-copy: one registration covering mem_pool_host.kv_buffer
        # non-zero-copy: two registrations, one bounce buffer per direction
        # (set/get) so the two storage threads never share slots.
        # 预注册的主机区域（由register_mem_pool_host设置）：
        # 零拷贝：一个注册覆盖mem_pool_host.kv_buffer
        # 非零拷贝：两个注册，每个方向一个弹跳缓冲区
        # （set/get），这样两个存储线程不会共享槽位。
        self._host_regs: List[Any] = []  # 预注册的主机区域列表
        self._bounce_set: Optional[torch.Tensor] = None  # 写方向的弹跳缓冲区
        self._bounce_get: Optional[torch.Tensor] = None  # 读方向的弹跳缓冲区
        self._bounce_page_bytes: Optional[int] = None  # 每页弹跳缓冲区字节数

    def _get_suffixed_key(self, key: str) -> str:  # 获取带后缀的键 # 将键添加配置后缀
        return key + self.config_suffix  # 返回带配置后缀的键

    def _create_query_tuple(self, key: str) -> tuple:  # 创建NIXL查询元组 # 构建用于NIXL查询内存的元组
        """Build the NIXL query_memory tuple for a single key."""  # 为单个键构建NIXL query_memory元组。
        if self.backend_selector.mem_type == "FILE":  # 如果内存类型为FILE
            return (0, 0, 0, self.file_manager.get_file_path(key))  # 返回包含文件路径的元组
        return (0, 0, 0, key)  # 返回包含键名的元组（OBJ类型）

    def _xfer_and_wait(  # 执行传输并等待完成 # 初始化NIXL传输并轮询直到完成
        self,
        host_descs: Any,  # 主机端描述符
        storage_descs: Any,  # 存储端描述符
        direction: str,  # 传输方向
    ) -> bool:  # 返回是否成功
        """Initialize and poll a NIXL transfer to completion."""  # 初始化并轮询NIXL传输直到完成。
        try:  # 尝试初始化传输
            xfer_req = self.agent.initialize_xfer(  # 初始化传输请求
                direction, host_descs, storage_descs, self.agent_name  # 方向、主机描述符、存储描述符、代理名
            )
        except Exception as e:  # 捕获初始化异常
            logger.error(f"Failed to create transfer request: {e}")  # 记录创建传输请求失败
            return False  # 返回失败

        try:  # 尝试执行传输
            state = self.agent.transfer(xfer_req)  # 启动传输
            while state != "DONE":  # 等待传输完成
                state = self.agent.check_xfer_state(xfer_req)  # 检查传输状态
                if state == "ERR":  # 如果传输出错
                    logger.error("Transfer failed")  # 记录传输失败
                    return False  # 返回失败
                # Best would be to have a better notification mechanism from NIXL,
                # but we only have polling for now.
                # 最好能有NIXL的更好通知机制，
                # 但目前只有轮询方式。
                time.sleep(0.0001)  # 短暂等待后再次轮询
            return True  # 传输完成，返回成功
        except Exception as e:  # 捕获传输异常
            logger.error(f"Failed to execute transfer: {e}")  # 记录执行传输失败
            import traceback  # 导入traceback模块

            logger.error(f"Traceback: {traceback.format_exc()}")  # 记录异常堆栈
            return False  # 返回失败
        finally:  # 最终清理
            self.agent.release_xfer_handle(xfer_req)  # 释放传输句柄

    def _xfer_pre_registered(  # 使用预注册主机区域的传输 # 执行主机端已预注册的传输操作
        self,
        host_buffers: List[tuple],  # 主机缓冲区列表（addr, size元组）
        keys: List[str],  # 键列表
        direction: str,  # 传输方向
    ) -> bool:  # 返回是否成功
        """Run a transfer where the host side is already pre-registered.
        执行主机端已预注册的传输。

        ``host_buffers`` is a list of ``(addr, size)`` tuples within the
        pre-registered host region (kv_buffer for zero-copy, bounce buffer
        otherwise). Only the storage side is registered per transfer.
        ``host_buffers``是预注册主机区域（零拷贝为kv_buffer，否则为弹跳缓冲区）
        中的``(addr, size)``元组列表。每次传输只注册存储端。
        """
        if len(host_buffers) != len(keys):  # 如果主机缓冲区和键数量不匹配
            logger.error("Mismatch between number of host buffers and keys")  # 记录数量不匹配错误
            return False  # 返回失败

        host_descs = self.agent.get_xfer_descs(  # 获取主机端传输描述符
            [(addr, size, 0) for (addr, size) in host_buffers], "DRAM"  # 将缓冲区转为DRAM描述符格式
        )
        if host_descs is None:  # 如果获取描述符失败
            logger.error("Failed to build host xfer descs")  # 记录构建描述符失败
            return False  # 返回失败

        with self.registry.storage(host_buffers, keys, direction) as storage_descs:  # 使用注册表的storage上下文管理器
            if storage_descs is None:  # 如果存储描述符为空
                return False  # 返回失败
            return self._xfer_and_wait(host_descs, storage_descs, direction)  # 执行传输并等待完成

    def get(  # 单键读取（已弃用） # 已弃用的单键get方法
        self,
        key: str,  # 键
        target_location: Optional[Any] = None,  # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> torch.Tensor | None:  # 返回张量或None
        raise NotImplementedError("deprecated; use batch_get_v1")  # 抛出未实现错误，提示使用batch_get_v1

    def batch_get(  # 批量读取（已弃用） # 已弃用的批量get方法
        self,
        keys: List[str],  # 键列表
        target_locations: Optional[Any] = None,  # 目标位置列表
        target_sizes: Optional[Any] = None,  # 目标大小列表
    ) -> List[torch.Tensor | None]:  # 返回张量列表
        raise NotImplementedError("deprecated; use batch_get_v1")  # 抛出未实现错误，提示使用batch_get_v1

    def set(  # 单键写入（已弃用） # 已弃用的单键set方法
        self,
        key: str,  # 键
        value: Optional[Any] = None,  # 值
        target_location: Optional[Any] = None,  # 目标位置
        target_sizes: Optional[Any] = None,  # 目标大小
    ) -> bool:  # 返回是否成功
        raise NotImplementedError("deprecated; use batch_set_v1")  # 抛出未实现错误，提示使用batch_set_v1

    def batch_set(  # 批量写入（已弃用） # 已弃用的批量set方法
        self,
        keys: List[str],  # 键列表
        values: Optional[Any] = None,  # 值列表
        target_locations: Optional[Any] = None,  # 目标位置列表
        target_sizes: Optional[Any] = None,  # 目标大小列表
    ) -> bool:  # 返回是否成功
        raise NotImplementedError("deprecated; use batch_set_v1")  # 抛出未实现错误，提示使用batch_set_v1

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):  # 注册主机内存池 # 将主机KV缓存注册到NIXL存储后端
        super().register_mem_pool_host(mem_pool_host)  # 调用父类方法注册主机内存池

        # enable zero-copy automatically if mem layout is page_first or page_first_direct  # 如果内存布局为page_first或page_first_direct，自动启用零拷贝
        self.is_zero_copy = self.mem_pool_host.layout in [  # 检查是否支持零拷贝的布局
            "page_first",  # 页优先布局
            "page_first_direct",  # 页优先直接布局
        ]

        if self.needs_page_alignment and self.is_zero_copy:  # 如果需要页对齐且为零拷贝模式
            # Check that the kv_buffer base AND per-page strides are multiples of
            # the OS page size so every pointer passed to NIXL (base + p * stride)
            # is page-aligned. The base is whatever torch.empty() happened to give
            # us -- it is not guaranteed to be page-aligned. Fall back to copy mode
            # if either condition fails.
            # 4096: O_DIRECT alignment is FS-dependent (some allow 512 B); 4 KiB
            # is the safe lower bound all known FSes accept, and real page-sizes meet it.
            # 检查kv_buffer基址和每页步长是否是OS页大小的倍数，以确保传递给NIXL的每个指针
            # （base + p * stride）都是页对齐的。基址是torch.empty()分配的——
            # 不保证页对齐。如果任一条件不满足，则回退到拷贝模式。
            # 4096：O_DIRECT对齐要求取决于文件系统（有些允许512B）；4 KiB是所有已知
            # 文件系统都接受的安全下限，实际页大小也满足此要求。
            if not self.mem_pool_host.is_stride_page_aligned(4096):  # 如果步长不是4096对齐的
                logger.warning(  # 记录警告
                    "HiCacheNixl: O_DIRECT is active but the host kv_buffer is "  # O_DIRECT已激活但主机kv_buffer
                    "not OS-page-aligned (base or per-page stride). Falling back "  # 不是OS页对齐的（基址或每页步长）。回退
                    "to copy mode for this pool."  # 到此池的拷贝模式。
                )
                self.is_zero_copy = False  # 回退到非零拷贝模式

        if self.is_zero_copy:  # 如果为零拷贝模式
            kv = mem_pool_host.kv_buffer  # 获取KV缓冲区
            self._pre_register_host(  # 预注册主机区域
                kv.data_ptr(), kv.numel() * kv.element_size(), "kv_buffer"  # 传入基址、大小和标签
            )
        else:  # 非零拷贝模式
            # One bounce buffer per direction so set/get run lock-free across
            # the prefetch and backup threads. Sized from get_dummy_flat_data_page()
            # so each slot matches what the v1 path would otherwise allocate.
            # 每个方向一个弹跳缓冲区，使set/get在预取和备份线程之间无锁运行。
            # 大小由get_dummy_flat_data_page()确定，使每个槽位与v1路径分配的大小匹配。
            sample = mem_pool_host.get_dummy_flat_data_page()  # 获取样本数据页
            page_numel = sample.numel()  # 每页元素数
            self._bounce_page_bytes = page_numel * sample.element_size()  # 每页字节数
            del sample  # 删除样本
            pin_memory = bool(getattr(mem_pool_host, "pin_memory", False))  # 是否锁页内存
            self._bounce_set = self._alloc_registered(  # 分配并注册写方向弹跳缓冲区
                page_numel, mem_pool_host.dtype, pin_memory, "bounce_set"  # 元素数、数据类型、是否锁页、标签
            )
            self._bounce_get = self._alloc_registered(  # 分配并注册读方向弹跳缓冲区
                page_numel, mem_pool_host.dtype, pin_memory, "bounce_get"  # 元素数、数据类型、是否锁页、标签
            )

        logger.info(  # 记录注册信息
            f"HiCacheNixl: pre-registered host regions for "  # HiCacheNixl：预注册主机区域
            f"layout={mem_pool_host.layout} zero_copy={self.is_zero_copy}"  # 布局和零拷贝模式
        )

    def _alloc_registered(  # 分配并预注册弹跳缓冲区 # 分配页对齐的弹跳缓冲区并预注册为DRAM区域
        self,
        page_numel: int,  # 每页元素数
        dtype: torch.dtype,  # 数据类型
        pin_memory: bool,  # 是否锁页内存
        kind: str,  # 缓冲区类型标签
    ) -> torch.Tensor:  # 返回张量
        """Allocate a ``(STORAGE_BATCH_SIZE, page_numel)`` bounce buffer and
        pre-register it as a DRAM region with NIXL. Uses alloc_mmap so the
        buffer is page-aligned -- required when O_DIRECT is on for any
        file-based backend (POSIX/GDS/GDS_MT/3FS). pin_memory is currently
        unused (alloc_mmap does not support it)."""
        # 分配一个(STORAGE_BATCH_SIZE, page_numel)的弹跳缓冲区，
        # 并预注册为NIXL的DRAM区域。使用alloc_mmap确保缓冲区页对齐——
        # 这在O_DIRECT启用时对任何基于文件的后端（POSIX/GDS/GDS_MT/3FS）是必需的。
        # pin_memory当前未使用（alloc_mmap不支持）。
        buf = alloc_mmap((STORAGE_BATCH_SIZE, page_numel), dtype)  # 使用mmap分配页对齐缓冲区
        self._pre_register_host(buf.data_ptr(), buf.numel() * buf.element_size(), kind)  # 预注册缓冲区
        return buf  # 返回缓冲区张量

    def _pre_register_host(self, base_addr: int, total_size: int, kind: str) -> None:  # 预注册主机DRAM区域 # 将单个DRAM区域预先注册到NIXL并保存句柄
        """Register a single DRAM region up-front and remember the handle."""  # 预先注册单个DRAM区域并记住句柄。
        reg_descs = self.agent.get_reg_descs([(base_addr, total_size, 0, "")], "DRAM")  # 获取注册描述符
        if reg_descs is None:  # 如果获取描述符失败
            raise RuntimeError(f"Failed to build reg descs for host {kind}")  # 抛出运行时错误
        try:  # 尝试注册内存
            self._host_regs.append(self.agent.register_memory(reg_descs))  # 注册内存并保存句柄
        except Exception as e:  # 捕获注册异常
            raise RuntimeError(f"Failed to pre-register host {kind} with NIXL") from e  # 抛出运行时错误

    def clear(self) -> None:  # 清除存储 # 清除文件管理器中的所有文件
        if self.file_manager is None:  # 如果文件管理器为空（OBJ模式）
            return  # 直接返回
        self.file_manager.clear()  # 清除文件管理器中的文件

    def close(self):  # 关闭存储 # 注销所有预注册的主机区域并清理弹跳缓冲区
        while self._host_regs:  # 遍历所有预注册的主机区域
            reg = self._host_regs.pop()  # 弹出最后一个注册句柄
            try:  # 尝试注销
                self.agent.deregister_memory(reg)  # 注销内存区域
            except Exception as e:  # 捕获注销异常
                logger.debug("deregister of pre-registered host region failed: %s", e)  # 记录注销失败
        self._bounce_set = None  # 清空写方向弹跳缓冲区
        self._bounce_get = None  # 清空读方向弹跳缓冲区
        self._bounce_page_bytes = None  # 清空每页字节数

    def __del__(self):  # 析构方法
        try:  # 尝试关闭
            self.close()  # 调用close方法
        except Exception:  # 捕获异常
            pass  # 忽略异常

    def exists(self, key: str) -> bool:  # 检查键是否存在 # 检查指定键是否存在于存储中
        results = self.batch_exists([key])  # 通过batch_exists检查
        return results > 0  # 返回结果大于0表示存在

    def batch_exists(  # 批量检查键是否存在 # 批量检查多个键是否存在于存储中，返回连续存在的页数
        self,
        keys: List[str],  # 键列表
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> int:  # 返回连续存在的页数
        if self.is_zero_copy:  # 如果为零拷贝模式
            key_list = self._get_key_list_from_meta(keys)  # 从元数据获取键列表
            key_denominator = (  # 键分母（每页的键数）
                1 if self.is_mla_model else 2  # MLA: 1 key per page (_k only), non-MLA: 2 NIXL keys per page (_k + _v)  # MLA每页1个键（仅_k），非MLA每页2个NIXL键（_k + _v）
            )
        else:  # 非零拷贝模式
            key_list = [self._get_suffixed_key(key) for key in keys]  # 为每个键添加后缀
            key_denominator = 1  # 非零拷贝模式每页1个键

        tuples = [self._create_query_tuple(key) for key in key_list]  # 创建查询元组列表

        query_res = self.agent.query_memory(  # 查询NIXL内存
            tuples,  # 查询元组列表
            self.backend_selector.backend_name,  # 后端名称
            mem_type=self.backend_selector.mem_type,  # 内存类型
        )

        for i in range(len(query_res)):  # 遍历查询结果
            if query_res[i] is None:  # 如果键不存在
                return i // key_denominator  # 返回连续存在的页数
        return len(query_res) // key_denominator  # 所有键都存在，返回总页数

    def _get_key_list_from_meta(self, keys: List[str]) -> List[str]:  # 从键生成存储键列表 # 根据模型类型为每个键生成_k和_v存储键
        # Each key maps to a `_k` entry, plus a `_v` entry on non-MLA models
        # (MLA stores k/v interleaved in a single buffer).
        # 每个键映射到一个`_k`条目，非MLA模型还有`_v`条目
        # （MLA将k/v交错存储在单个缓冲区中）。
        key_list = []  # 初始化键列表
        for key in keys:  # 遍历每个键
            suffixed_key = self._get_suffixed_key(key)  # 获取带后缀的键
            key_list.append(f"{suffixed_key}_k")  # 添加K键
            if not self.is_mla_model:  # 非MLA模型
                key_list.append(f"{suffixed_key}_v")  # 添加V键
        return key_list  # 返回键列表

    def _get_location_and_size_list_from_meta(  # 从元数据获取位置和大小列表 # 获取零拷贝模式下的键列表和缓冲区元数据
        self, keys: List[str], host_indices: torch.Tensor  # 参数：键列表和主机索引
    ):
        # zero copy: mem_pool_host.get_data_page() does not work due to non-contiguous tensors, causing issues for NIXL transfer
        # 零拷贝：mem_pool_host.get_data_page()因非连续张量而无法工作，会导致NIXL传输问题
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(  # 获取页缓冲区元数据
            host_indices  # 主机索引
        )
        key_list = self._get_key_list_from_meta(keys)  # 从元数据获取键列表

        if len(key_list) != len(ptr_list):  # 如果键数量与指针数量不匹配
            logger.error(  # 记录错误
                f"HiCacheNixl: mismatch between number of keys and number of buffer meta entries, keys: {len(keys)}, key_list: {len(key_list)}, buffer meta entries: {len(ptr_list)}"  # 键和缓冲区元数据条目数量不匹配
            )
            return [], [], []  # 返回空列表

        return key_list, ptr_list, element_size_list  # 返回键列表、指针列表和大小列表

    def _bounce_slot_buffers(self, buf: torch.Tensor, page_num: int) -> List[tuple]:  # 获取弹跳缓冲区槽位 # 返回弹跳缓冲区中指定数量槽位的(addr, size)元组列表
        """Return ``page_num`` ``(addr, size)`` tuples pointing at the first
        ``page_num`` slots of ``buf``.
        返回``page_num``个``(addr, size)``元组，指向``buf``的前``page_num``个槽位。
        """
        base = buf.data_ptr()  # 获取缓冲区基址
        return [  # 返回槽位列表
            (base + i * self._bounce_page_bytes, self._bounce_page_bytes)  # 每个槽位的地址和大小
            for i in range(page_num)  # 遍历每个槽位
        ]

    def _batch_preprocess(self, keys: List[str], host_indices: torch.Tensor, op: str):  # 批量预处理 # 为v1路径构建键列表和主机缓冲区
        """Build (key_list, host_buffers) for the v1 path.
        为v1路径构建(key_list, host_buffers)。

        For zero-copy: ``host_buffers`` are ``(addr, size)`` tuples inside the
        pre-registered ``kv_buffer``.
        For non-zero-copy: ``host_buffers`` are slots of the direction-specific
        pre-registered bounce buffer (``_bounce_set`` for set, ``_bounce_get``
        for get); for ``op == "set"`` we copy the host pages into those slots
        here so the subsequent transfer reads from the bounce buffer.
        Returns ``([], [])`` on validation failure.
        对于零拷贝：``host_buffers``是预注册``kv_buffer``内的``(addr, size)``元组。
        对于非零拷贝：``host_buffers``是方向特定的预注册弹跳缓冲区（set为``_bounce_set``，
        get为``_bounce_get``）的槽位；对于``op == "set"``，我们在此处将主机页复制
        到这些槽位，以便后续传输从弹跳缓冲区读取。
        验证失败时返回``([], [])``。
        """
        page_size = self.mem_pool_host.page_size  # 获取页大小
        page_num = len(host_indices) // page_size  # 计算页数

        if len(keys) == 0 or len(keys) != page_num:  # 如果键为空或数量与页数不匹配
            logger.warning(  # 记录警告
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {page_size}"  # 键为空或键与host_indices长度不匹配
            )
            return [], []  # 返回空列表

        if self.is_zero_copy:  # 如果为零拷贝模式
            key_list, ptr_list, size_list = self._get_location_and_size_list_from_meta(  # 从元数据获取位置和大小
                keys, host_indices  # 传入键和索引
            )
            host_buffers = list(zip(ptr_list, size_list))  # 将指针和大小组合为主机缓冲区列表
            return key_list, host_buffers  # 返回键列表和主机缓冲区

        if page_num > STORAGE_BATCH_SIZE:  # 如果页数超过批量大小
            logger.error(  # 记录错误
                f"HiCacheNixl: batch size {page_num} exceeds bounce buffer capacity {STORAGE_BATCH_SIZE}"  # 批量大小超过弹跳缓冲区容量
            )
            return [], []  # 返回空列表

        bounce = self._bounce_set if op == "set" else self._bounce_get  # 根据操作类型选择弹跳缓冲区
        if op == "set":  # 如果是写操作
            for i in range(page_num):  # 遍历每页
                src = self.mem_pool_host.get_data_page(  # 获取源数据页
                    host_indices[i * page_size], flat=True  # 获取扁平化数据页
                )
                bounce[i].copy_(src)  # 将源数据复制到弹跳缓冲区

        host_buffers = self._bounce_slot_buffers(bounce, page_num)  # 获取弹跳缓冲区槽位
        key_list = [self._get_suffixed_key(key) for key in keys]  # 为每个键添加后缀
        return key_list, host_buffers  # 返回键列表和主机缓冲区

    def _batch_xfer(  # 批量传输 # 对预注册主机区域执行批量读或写操作
        self,
        keys: List[str],  # 原始键列表
        key_strs: List[str],  # 存储键字符串列表
        host_buffers: List[tuple],  # 主机缓冲区列表
        direction: str,  # 传输方向（READ或WRITE）
    ) -> List[bool]:  # 返回布尔结果列表
        """Run a batch READ or WRITE for the v1 path against the pre-registered
        host region (no per-transfer host registration).
        对预注册主机区域执行v1路径的批量READ或WRITE（无需每次传输注册主机）。
        """
        if not key_strs or not host_buffers:  # 如果键或缓冲区为空
            return [False] * len(keys)  # 返回全False列表

        if len(key_strs) != len(host_buffers):  # 如果键和缓冲区数量不匹配
            logger.error("Mismatch between number of key_strs and host_buffers")  # 记录数量不匹配错误
            return [False] * len(keys)  # 返回全False列表

        if self.backend_selector.mem_type == "FILE":  # 如果存储类型为FILE
            file_paths = [self.file_manager.get_file_path(key) for key in key_strs]  # 获取文件路径列表
            success = self._xfer_pre_registered(host_buffers, file_paths, direction)  # 执行FILE类型传输
        else:  # mem_type == "OBJ"  # 存储类型为OBJ
            success = self._xfer_pre_registered(host_buffers, key_strs, direction)  # 执行OBJ类型传输

        # READ results are consumed by _batch_get_postprocess, which pairs
        # entries 2*i / 2*i+1 for non-MLA zero-copy: it needs one bool per
        # key_str (i.e. per `_k`/`_v` buffer). WRITE results map 1:1 to
        # pages, i.e. to `keys`.
        # READ结果由_batch_get_postprocess处理，对于非MLA零拷贝模式它会配对
        # 2*i / 2*i+1条目：每个key_str需要一个布尔值（即每个`_k`/`_v`缓冲区）。
        # WRITE结果与页一一映射，即与`keys`一一映射。
        result_len = len(key_strs) if direction == "READ" else len(keys)  # READ返回key_str数量个结果，WRITE返回keys数量个结果
        return [success] * result_len  # 返回统一的结果列表

    def _batch_get_postprocess(  # 批量读取后处理 # 处理批量读取的结果，包括零拷贝模式的结果合并和非零拷贝模式的数据拷贝
        self,
        host_indices: torch.Tensor,  # 主机索引张量
        results: List[bool],  # 传输结果列表
    ) -> List[bool]:  # 返回布尔结果列表
        page_size = self.mem_pool_host.page_size  # 获取页大小
        page_num = len(host_indices) // page_size  # 计算页数

        if self.is_zero_copy:  # 如果为零拷贝模式
            # zero copy: update final results based on the boolean results from NIXL transfer  # 零拷贝：根据NIXL传输的布尔结果更新最终结果
            if self.is_mla_model:  # 如果是MLA模型
                return results  # MLA直接返回结果
            return [(results[2 * i] and results[2 * i + 1]) for i in range(page_num)]  # 非MLA需要K和V都成功

        # non zero copy: copy data from the get-side bounce buffer to mem_pool_host  # 非零拷贝：从读方向弹跳缓冲区拷贝数据到主机内存池
        for i in range(page_num):  # 遍历每页
            if not results[i]:  # 如果该页读取失败
                break  # 跳出循环
            self.mem_pool_host.set_from_flat_data_page(  # 将弹跳缓冲区数据写入主机内存池
                host_indices[i * page_size], self._bounce_get[i]  # 目标索引和源弹跳缓冲区页
            )
        return results  # 返回结果列表

    def _log_xfer_stats(  # 记录传输统计信息 # 记录传输操作的统计信息（键数、字节数、耗时、带宽）
        self,
        op_name: str,  # 操作名称
        num_keys: int,  # 键数量
        host_indices: torch.Tensor,  # 主机索引
        buffer_sizes: List[int],  # 缓冲区大小列表
        elapsed_ms: float,  # 耗时（毫秒）
    ) -> None:  # 无返回值
        total_bytes = sum(s for s in buffer_sizes if s is not None)  # 计算总传输字节数
        bw = total_bytes / (elapsed_ms / 1000) / (1024 * 1024) if elapsed_ms else 0.0  # 计算带宽MB/s
        logger.debug(  # 记录调试信息
            f"HiCacheNixl {op_name} transferred: {num_keys} keys (pages), "  # 传输的键数（页数）
            f"{host_indices.numel()} host_indices, {total_bytes} bytes, "  # 主机索引数量和总字节数
            f"total time: {elapsed_ms:.3f} ms, effective bandwidth: {bw:.2f} MB/s"  # 总耗时和有效带宽
        )

    def batch_get_v1(  # 批量读取v1版本 # 从存储中批量读取KV缓存数据（v1版本）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:  # 返回布尔结果列表
        if not self._host_regs:  # 如果主机区域未注册
            logger.error(  # 记录错误
                "HiCacheNixl batch_get_v1: register_mem_pool_host must be called first"  # 必须先调用register_mem_pool_host
            )
            return [False] * len(keys)  # 返回全False列表

        key_strs, host_buffers = self._batch_preprocess(keys, host_indices, "get")  # 批量预处理
        if not key_strs or not host_buffers:  # 如果预处理结果为空
            return [False] * len(keys)  # 返回全False列表

        start_time = time.perf_counter()  # 记录开始时间
        results = self._batch_xfer(keys, key_strs, host_buffers, "READ")  # 执行批量读取
        elapsed_ms = (time.perf_counter() - start_time) * 1000  # 计算耗时（毫秒）
        self._log_xfer_stats(  # 记录传输统计
            "batch_get_v1",  # 操作名称
            len(keys),  # 键数量
            host_indices,  # 主机索引
            [s for _, s in host_buffers],  # 缓冲区大小列表
            elapsed_ms,  # 耗时
        )

        return self._batch_get_postprocess(host_indices, results)  # 后处理并返回结果

    def batch_set_v1(  # 批量写入v1版本 # 向存储中批量写入KV缓存数据（v1版本）
        self,
        keys: List[str],  # 键列表
        host_indices: torch.Tensor,  # 主机索引张量
        extra_info: Optional[HiCacheStorageExtraInfo] = None,  # 额外信息
    ) -> List[bool]:  # 返回布尔结果列表
        # skip on MLA backup rank  # 在MLA备份rank上跳过
        if self.backup_skip:  # 如果需要跳过备份
            return [True] * len(keys)  # 返回全True列表

        if len(keys) == 0:  # 如果键列表为空
            return []  # 返回空列表

        if not self._host_regs:  # 如果主机区域未注册
            logger.error(  # 记录错误
                "HiCacheNixl batch_set_v1: register_mem_pool_host must be called first"  # 必须先调用register_mem_pool_host
            )
            return [False] * len(keys)  # 返回全False列表

        key_strs, host_buffers = self._batch_preprocess(keys, host_indices, "set")  # 批量预处理
        if not key_strs or not host_buffers:  # 如果预处理结果为空
            return [False] * len(keys)  # 返回全False列表

        start_time = time.perf_counter()  # 记录开始时间
        results = self._batch_xfer(keys, key_strs, host_buffers, "WRITE")  # 执行批量写入
        elapsed_ms = (time.perf_counter() - start_time) * 1000  # 计算耗时（毫秒）
        self._log_xfer_stats(  # 记录传输统计
            "batch_set_v1",  # 操作名称
            len(keys),  # 键数量
            host_indices,  # 主机索引
            [s for _, s in host_buffers],  # 缓冲区大小列表
            elapsed_ms,  # 耗时
        )

        return results  # 返回结果列表
