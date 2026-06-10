# 存储后端工厂模块
# 该模块实现了存储后端的工厂模式，支持内置后端的注册与创建，以及动态加载外部后端

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project  # SPDX版权声明：SGLang项目的贡献者

import importlib  # 导入模块动态导入工具
import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any, Dict  # 导入类型提示

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig  # 导入HiCache存储基类和配置类

if TYPE_CHECKING:  # 类型检查时执行
    pass  # 占位语句，无实际操作

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class StorageBackendFactory:  # 存储后端工厂类
    """Factory for creating storage backend instances with support for dynamic loading."""  # 用于创建存储后端实例的工厂，支持动态加载

    _registry: Dict[str, Dict[str, Any]] = {}  # 后端注册表，存储已注册后端的元信息

    @staticmethod  # 静态方法
    def _load_backend_class(  # 加载并验证后端类
        module_path: str, class_name: str, backend_name: str  # 模块路径、类名、后端名称
    ) -> type[HiCacheStorage]:  # 返回HiCacheStorage的子类类型
        """Load and validate a backend class from module path."""  # 从模块路径加载并验证后端类
        try:  # 尝试导入模块
            module = importlib.import_module(module_path)  # 动态导入指定模块
            backend_class = getattr(module, class_name)  # 从模块中获取指定类
            if not issubclass(backend_class, HiCacheStorage):  # 检查类是否为HiCacheStorage的子类
                raise TypeError(  # 如果不是则抛出类型错误
                    f"Backend class {class_name} must inherit from HiCacheStorage"  # 后端类必须继承自HiCacheStorage
                )
            return backend_class  # 返回后端类
        except ImportError as e:  # 捕获导入错误
            raise ImportError(  # 抛出更详细的导入错误
                f"Failed to import backend '{backend_name}' from '{module_path}': {e}"  # 从指定模块导入后端失败
            ) from e  # 保留原始异常链
        except AttributeError as e:  # 捕获属性错误
            raise AttributeError(  # 抛出更详细的属性错误
                f"Class '{class_name}' not found in module '{module_path}': {e}"  # 在指定模块中未找到类
            ) from e  # 保留原始异常链

    @classmethod  # 类方法
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:  # 注册存储后端，支持懒加载
        """Register a storage backend with lazy loading.  # 注册支持懒加载的存储后端

        Args:  # 参数说明
            name: Backend identifier  # 后端标识符
            module_path: Python module path containing the backend class  # 包含后端类的Python模块路径
            class_name: Name of the backend class  # 后端类名
        """
        if name in cls._registry:  # 如果后端已注册
            logger.warning(f"Backend '{name}' is already registered, overwriting")  # 记录覆盖警告

        def loader() -> type[HiCacheStorage]:  # 懒加载函数
            """Lazy loader function to import the backend class."""  # 懒加载函数，用于导入后端类
            return cls._load_backend_class(module_path, class_name, name)  # 调用静态方法加载后端类

        cls._registry[name] = {  # 将后端信息注册到注册表
            "loader": loader,  # 懒加载函数
            "module_path": module_path,  # 模块路径
            "class_name": class_name,  # 类名
        }

    @classmethod  # 类方法
    def create_backend(  # 创建存储后端实例
        cls,
        backend_name: str,  # 后端名称
        storage_config: HiCacheStorageConfig,  # 存储配置
        mem_pool_host: Any,  # 主机端内存池
        **kwargs,  # 额外参数
    ) -> HiCacheStorage:  # 返回HiCacheStorage实例
        """Create a storage backend instance.  # 创建存储后端实例
        Args:  # 参数说明
            backend_name: Name of the backend to create  # 要创建的后端名称
            storage_config: Storage configuration  # 存储配置
            mem_pool_host: Memory pool host object  # 主机端内存池对象
            **kwargs: Additional arguments passed to external backends  # 传递给外部后端的额外参数
        Returns:  # 返回值
            Initialized storage backend instance  # 初始化的存储后端实例
        Raises:  # 异常
            ValueError: If backend is not registered and cannot be dynamically loaded  # 后端未注册且无法动态加载
            ImportError: If backend module cannot be imported  # 后端模块无法导入
            Exception: If backend initialization fails  # 后端初始化失败
        """
        # First check if backend is already registered  # 首先检查后端是否已注册
        if backend_name in cls._registry:  # 如果后端在注册表中
            registry_entry = cls._registry[backend_name]  # 获取注册表条目
            backend_class = registry_entry["loader"]()  # 调用懒加载函数获取后端类
            logger.info(  # 记录创建信息
                f"Creating storage backend '{backend_name}' "  # 正在创建存储后端
                f"({registry_entry['module_path']}.{registry_entry['class_name']})"  # 显示模块路径和类名
            )
            return cls._create_builtin_backend(  # 创建内置后端实例
                backend_name, backend_class, storage_config, mem_pool_host  # 传入后端名称、类、配置和内存池
            )

        # Try to dynamically load backend from extra_config  # 尝试从额外配置动态加载后端
        if backend_name == "dynamic" and storage_config.extra_config is not None:  # 如果是动态后端且额外配置不为空
            backend_config = storage_config.extra_config  # 获取后端配置
            return cls._create_dynamic_backend(  # 创建动态后端实例
                backend_config, storage_config, mem_pool_host, **kwargs  # 传入配置和参数
            )

        # Backend not found  # 未找到后端
        available_backends = list(cls._registry.keys())  # 获取所有已注册的后端名称

        raise ValueError(  # 抛出值错误
            f"Unknown storage backend '{backend_name}'. "  # 未知的存储后端
            f"Registered backends: {available_backends}. "  # 已注册的后端列表
        )

    @classmethod  # 类方法
    def _create_dynamic_backend(  # 从配置动态创建后端
        cls,
        backend_config: Dict[str, Any],  # 后端配置字典
        storage_config: HiCacheStorageConfig,  # 存储配置
        mem_pool_host: Any,  # 主机端内存池
        **kwargs,  # 额外参数
    ) -> HiCacheStorage:  # 返回HiCacheStorage实例
        """Create a backend dynamically from configuration."""  # 从配置动态创建后端
        required_fields = ["backend_name", "module_path", "class_name"]  # 必需字段列表
        for field in required_fields:  # 遍历必需字段
            if field not in backend_config:  # 如果字段缺失
                raise ValueError(  # 抛出值错误
                    f"Missing required field '{field}' in backend config for 'dynamic' backend"  # 动态后端配置缺少必需字段
                )

        backend_name = backend_config["backend_name"]  # 获取后端名称
        module_path = backend_config["module_path"]  # 获取模块路径
        class_name = backend_config["class_name"]  # 获取类名

        try:  # 尝试创建动态后端
            # Import the backend class  # 导入后端类
            backend_class = cls._load_backend_class(  # 加载后端类
                module_path, class_name, backend_name  # 传入模块路径、类名和后端名称
            )

            logger.info(  # 记录创建信息
                f"Creating dynamic storage backend '{backend_name}' "  # 正在创建动态存储后端
                f"({module_path}.{class_name})"  # 显示模块路径和类名
            )

            # Create the backend instance with storage_config  # 使用存储配置创建后端实例
            return backend_class(storage_config, kwargs)  # 调用后端类构造函数
        except Exception as e:  # 捕获异常
            logger.error(  # 记录错误日志
                f"Failed to create dynamic storage backend '{backend_name}': {e}"  # 创建动态存储后端失败
            )
            raise  # 重新抛出异常

    @classmethod  # 类方法
    def _create_builtin_backend(  # 创建内置后端
        cls,
        backend_name: str,  # 后端名称
        backend_class: type[HiCacheStorage],  # 后端类
        storage_config: HiCacheStorageConfig,  # 存储配置
        mem_pool_host: Any,  # 主机端内存池
    ) -> HiCacheStorage:  # 返回HiCacheStorage实例
        """Create built-in backend with original initialization logic."""  # 使用原始初始化逻辑创建内置后端
        if backend_name == "file":  # 如果是文件后端
            return backend_class(storage_config)  # 仅传入存储配置
        elif backend_name == "nixl":  # 如果是NIXL后端
            return backend_class(storage_config)  # 仅传入存储配置
        elif backend_name == "mooncake":  # 如果是Mooncake后端
            backend = backend_class(storage_config, mem_pool_host)  # 传入存储配置和内存池
            return backend  # 返回后端实例
        elif backend_name == "aibrix":  # 如果是AIBrix后端
            backend = backend_class(storage_config, mem_pool_host)  # 传入存储配置和内存池
            return backend  # 返回后端实例
        elif backend_name == "hf3fs":  # 如果是HF3FS后端
            # Calculate bytes_per_page based on memory pool layout  # 根据内存池布局计算每页字节数
            if mem_pool_host.layout in ["page_first", "page_first_direct"]:  # 页优先布局
                bytes_per_page = (  # 计算每页字节数
                    mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size  # KV分离模式下每页字节数
                )
            elif mem_pool_host.layout == "layer_first":  # 层优先布局
                bytes_per_page = (  # 计算每页字节数
                    mem_pool_host.get_size_per_token() * mem_pool_host.page_size  # 每token大小乘以页大小
                )

            dtype = mem_pool_host.dtype  # 获取数据类型
            return backend_class.from_env_config(bytes_per_page, dtype, storage_config)  # 从环境配置创建HF3FS后端
        elif backend_name == "eic":  # 如果是EIC后端
            return backend_class(storage_config, mem_pool_host)  # 传入存储配置和内存池
        elif backend_name == "simm":  # 如果是SiMM后端
            return backend_class(storage_config, mem_pool_host)  # 传入存储配置和内存池
        else:  # 未知后端
            raise ValueError(f"Unknown built-in backend: {backend_name}")  # 抛出未知内置后端错误


# Register built-in storage backends  # 注册内置存储后端
StorageBackendFactory.register_backend(  # 注册文件后端
    "file", "sglang.srt.mem_cache.hicache_storage", "HiCacheFile"  # 文件后端：HiCacheFile类
)

StorageBackendFactory.register_backend(  # 注册NIXL后端
    "nixl",  # 后端名称
    "sglang.srt.mem_cache.storage.nixl.hicache_nixl",  # 模块路径
    "HiCacheNixl",  # 类名
)

StorageBackendFactory.register_backend(  # 注册Mooncake后端
    "mooncake",  # 后端名称
    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store",  # 模块路径
    "MooncakeStore",  # 类名
)

StorageBackendFactory.register_backend(  # 注册HF3FS后端
    "hf3fs",  # 后端名称
    "sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs",  # 模块路径
    "HiCacheHF3FS",  # 类名
)

StorageBackendFactory.register_backend(  # 注册AIBrix后端
    "aibrix",  # 后端名称
    "sglang.srt.mem_cache.storage.aibrix_kvcache.aibrix_kvcache_storage",  # 模块路径
    "AibrixKVCacheStorage",  # 类名
)

StorageBackendFactory.register_backend(  # 注册EIC后端
    "eic",  # 后端名称
    "sglang.srt.mem_cache.storage.eic.eic_storage",  # 模块路径
    "EICStorage",  # 类名
)

StorageBackendFactory.register_backend(  # 注册SiMM后端
    "simm",  # 后端名称
    "sglang.srt.mem_cache.storage.simm.hicache_simm",  # 模块路径
    "HiCacheSiMM",  # 类名
)
