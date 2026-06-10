# NIXL工具模块
# 本模块提供NIXL存储后端的辅助工具类，包括：
# - NixlBackendConfig：处理NIXL后端配置
# - NixlBackendSelection：处理后端选择和创建
# - NixlFileManager：处理文件系统操作

import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于操作系统接口
from typing import Optional  # 从typing模块导入可选类型注解

from sglang.srt.environ import envs  # 从environ模块导入环境变量配置

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class NixlBackendConfig:  # NIXL后端配置类，处理后端配置解析
    """Handles NIXL backend configurations"""  # 处理NIXL后端配置

    def __init__(self, config: Optional[dict[str, str]] = None):  # 初始化方法 # 使用可选的配置字典初始化后端配置
        """Initialize backend configuration.
        初始化后端配置。
        Args:
            config: configurations in a dictionary. This config comes from --hicache-storage-backend-extra-config
            config: 字典形式的配置。此配置来自--hicache-storage-backend-extra-config

            config can be in two forms:
            1. fully qualified form (for all plugins, some of them are enabled, others not):
                {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            2. flat form (for a specific selected plugin), assuming all params apply to a selected plugin
                {'param1': 'value1', 'param2': 'value2', ...}
            配置可以有两种形式：
            1. 完全限定形式（用于所有插件，部分启用，部分不启用）：
                {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            2. 扁平形式（用于特定选定的插件），假设所有参数适用于选定的插件
                {'param1': 'value1', 'param2': 'value2', ...}
        """
        self.config = config or {}  # 保存配置字典，如果为None则使用空字典

    def get_use_direct_io(self) -> bool:  # 获取是否使用直接I/O # 检查是否应在打开文件时使用O_DIRECT
        """Return True if O_DIRECT should be requested when opening files.
        如果在打开文件时应请求O_DIRECT，则返回True。

        Checks the top-level ``use_direct_io`` key in the long-form JSON config first,
        then falls back to the ``SGLANG_HICACHE_NIXL_USE_DIRECT_IO`` environment variable
        (default: enabled).
        首先检查长格式JSON配置中的顶级``use_direct_io``键，
        然后回退到``SGLANG_HICACHE_NIXL_USE_DIRECT_IO``环境变量（默认：启用）。
        """
        if "use_direct_io" in self.config:  # 如果配置中有use_direct_io键
            return bool(self.config["use_direct_io"])  # 返回其布尔值
        return envs.SGLANG_HICACHE_NIXL_USE_DIRECT_IO.get()  # 否则从环境变量获取

    def get_specified_plugin(self) -> str:  # 获取指定的插件名称 # 决定使用哪个插件：优先从配置获取，否则从环境变量获取，默认为"auto"
        """decide which plugin to use: either config or SGLANG_HICACHE_NIXL_BACKEND_PLUGIN specifies the plugin, if not, use "auto" """
        # 决定使用哪个插件：从config或SGLANG_HICACHE_NIXL_BACKEND_PLUGIN指定插件，否则使用"auto"

        if "plugin" in self.config:  # 如果配置中有plugin键（完全限定形式）
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            # choose the FIRST active plugin
            # 完全限定形式：{'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            # 选择第一个激活的插件
            for key, item in self.config["plugin"].items():  # 遍历插件配置
                if item.get("active", False) in [True, "true", "True"]:  # 如果插件处于激活状态
                    plugin = key.upper()  # 将插件名转为大写
                    break  # 跳出循环
        else:  # 否则（扁平形式或空配置）
            # config is empty, or in flat form {'param1': 'value1', 'param2': 'value2', ...}
            # 配置为空，或为扁平形式{'param1': 'value1', 'param2': 'value2', ...}
            plugin = os.getenv("SGLANG_HICACHE_NIXL_BACKEND_PLUGIN", "auto")  # 从环境变量获取插件名，默认"auto"

        return plugin  # 返回插件名称

    def get_backend_initparams(self, backend_name) -> dict:  # 获取后端初始化参数 # 从配置中获取指定后端的初始化参数
        """Get initialization parameters from config of NIXL backend for backend creation.
        从NIXL后端的配置中获取用于后端创建的初始化参数。
        Args:
            backend_name: a specific backend's name (already converted "auto" into a specific backend name)
            backend_name: 特定后端的名称（已将"auto"转换为具体的后端名称）
        """

        initparams = {}  # 初始化参数字典

        # config can be in two forms:  # 配置可以有两种形式：
        if "plugin" in self.config:  # 如果配置中有plugin键（完全限定形式）
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            # 完全限定形式：{'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            if backend_name.lower() in self.config["plugin"]:  # 如果后端名在plugin配置中
                config_data = self.config["plugin"][backend_name.lower()]  # 获取该后端的配置数据
            else:  # 否则
                logger.debug(  # 记录调试信息
                    f"No specific config found for plugin {backend_name} in extra_config. Use default init params."  # 在extra_config中未找到插件{backend_name}的特定配置，使用默认初始化参数。
                )
                config_data = {}  # 使用空配置
        else:  # 扁平形式
            # flat form {'param1': 'value1', 'param2': 'value2', ...}
            # 扁平形式{'param1': 'value1', 'param2': 'value2', ...}
            config_data = self.config  # 使用整个配置作为该后端的配置

        for key, value in config_data.items():  # 遍历配置数据
            initparams[key] = str(value)  # 将所有值转为字符串

        return initparams  # 返回初始化参数字典


class NixlBackendSelection:  # NIXL后端选择类，处理后端的选择和创建
    """Handles NIXL backend selection and creation."""  # 处理NIXL后端选择和创建。

    # Priority order for File-based plugins in case of auto selection  # 自动选择时基于文件的插件优先级顺序
    FILE_PLUGINS = ["3FS", "POSIX", "GDS_MT", "GDS"]  # 文件插件优先级列表
    # Priority order for File-based plugins in case of auto selection (add more as needed)  # 自动选择时基于文件的插件优先级顺序（按需添加更多）
    OBJ_PLUGINS = ["OBJ"]  # Based on Amazon S3 SDK  # 基于Amazon S3 SDK的对象插件列表

    def __init__(  # 初始化方法
        self, plugin: str = "auto", nixlconfig: Optional[NixlBackendConfig] = None  # 参数：插件名称和配置
    ):
        """Initialize backend selection.
        初始化后端选择。
        Args:
            plugin: Plugin to use (default "auto" selects best available).
                   Can be a file plugin (3FS, POSIX, GDS, GDS_MT) or
                   an object plugin (OBJ).
            plugin: 要使用的插件（默认"auto"选择最佳可用插件）。
                   可以是文件插件（3FS、POSIX、GDS、GDS_MT）或
                   对象插件（OBJ）。
        """
        self.plugin = plugin  # 保存插件名称
        self.backend_name = None  # 后端名称，初始化为None
        self.mem_type = None  # 内存类型，初始化为None
        self.nixlconfig = nixlconfig  # 保存NIXL配置

    def create_backend(self, agent) -> bool:  # 创建NIXL后端 # 根据配置创建适当的NIXL后端
        """Create the appropriate NIXL backend based on configuration."""  # 根据配置创建适当的NIXL后端。
        try:  # 尝试创建后端
            plugin_list = agent.get_plugin_list()  # 获取可用插件列表
            logger.debug(f"Available NIXL plugins: {plugin_list}")  # 记录可用插件列表

            # Handle explicit plugin selection or auto priority  # 处理显式插件选择或自动优先级
            if self.plugin == "auto":  # 如果为自动选择模式
                # Try all file plugins first  # 优先尝试所有文件插件
                for plugin in self.FILE_PLUGINS:  # 遍历文件插件优先级列表
                    if plugin in plugin_list:  # 如果插件可用
                        self.backend_name = plugin  # 设置后端名称
                        break  # 跳出循环
                # If no file plugin found, try object plugins  # 如果未找到文件插件，尝试对象插件
                if not self.backend_name:  # 如果后端名称仍为None
                    for plugin in self.OBJ_PLUGINS:  # 遍历对象插件列表
                        if plugin in plugin_list:  # 如果插件可用
                            self.backend_name = plugin  # 设置后端名称
                            break  # 跳出循环
            else:  # 显式指定插件
                # Use explicitly requested plugin  # 使用显式请求的插件
                self.backend_name = self.plugin  # 使用指定的插件名称

            if self.backend_name not in plugin_list:  # 如果选定的后端不在可用插件列表中
                logger.error(  # 记录错误
                    f"Backend {self.backend_name} not available in plugins: {plugin_list}"  # 后端在可用插件列表中不可用
                )
                return False  # 返回失败

            # obtain initparams for the backend from the NIXL config  # 从NIXL配置获取后端的初始化参数
            initparams = (  # 获取初始化参数
                self.nixlconfig.get_backend_initparams(self.backend_name)  # 从配置获取参数
                if self.nixlconfig  # 如果配置存在
                else {}  # 否则使用空字典
            )

            # Create backend and set memory type  # 创建后端并设置内存类型
            if self.backend_name in self.OBJ_PLUGINS and "bucket" not in initparams:  # 如果是OBJ插件且未指定bucket
                bucket = os.environ.get("AWS_DEFAULT_BUCKET")  # 从环境变量获取AWS bucket
                if not bucket:  # 如果bucket不存在
                    logger.error(  # 记录错误
                        "AWS_DEFAULT_BUCKET environment variable must be set for object storage"  # 必须为对象存储设置AWS_DEFAULT_BUCKET环境变量
                    )
                    return False  # 返回失败

                initparams["bucket"] = bucket  # 将bucket添加到初始化参数

            # create backend using initialization parameters  # 使用初始化参数创建后端
            agent.create_backend(self.backend_name, initparams)  # 调用agent创建后端

            logger.info(  # 记录创建信息
                f"NixlBackendSelection.create_backend: backend_name {self.backend_name} initparams {initparams} customParams {agent.get_backend_params(self.backend_name)} supported plugins {plugin_list}"  # 后端创建信息
            )

            self.mem_type = "OBJ" if self.backend_name in self.OBJ_PLUGINS else "FILE"  # 设置内存类型
            logger.debug(  # 记录调试信息
                f"Created NIXL backend: {self.backend_name} with memory type: {self.mem_type}"  # 创建的NIXL后端及其内存类型
            )
            return True  # 返回成功

        except Exception as e:  # 捕获异常
            logger.error(  # 记录错误
                f"Failed to create NIXL backend: {e}, backend_name {self.backend_name}, supported plugins {plugin_list} initparams {initparams}"  # 创建NIXL后端失败
            )
            return False  # 返回失败


class NixlFileManager:  # NIXL文件管理器类，处理文件系统操作
    """Handles file system operations for NIXL."""  # 处理NIXL的文件系统操作。

    def __init__(self, base_dir: str, use_direct_io: bool = True):  # 初始化方法 # 使用基础目录和直接I/O选项初始化文件管理器
        """
        Initialize file manager.
        初始化文件管理器。
        Args:
            base_dir: Base directory for storing tensor files
            base_dir: 存储张量文件的基础目录
            use_direct_io: If True, open files with O_DIRECT (bypasses OS page cache).
                Falls back to buffered I/O with a warning when O_DIRECT is unavailable.
            use_direct_io: 如果为True，使用O_DIRECT打开文件（绕过OS页缓存）。
                当O_DIRECT不可用时回退到缓冲I/O并发出警告。
        """
        self.base_dir = base_dir  # 保存基础目录
        self.use_direct_io = use_direct_io  # 保存是否使用直接I/O
        if base_dir == "":  # 如果基础目录为空字符串
            logger.debug(  # 记录调试信息
                f"Initialized file manager without a base directory. Direct I/O: {use_direct_io}"  # 无基础目录初始化文件管理器
            )
        else:  # 基础目录不为空
            os.makedirs(base_dir, exist_ok=True)  # 创建基础目录（如不存在）
            logger.debug(  # 记录调试信息
                f"Initialized file manager with base directory: {base_dir}. Direct I/O: {use_direct_io}"  # 使用基础目录初始化文件管理器
            )

    def clear(self) -> None:  # 清除目录中的所有文件 # 清除基础目录中的所有文件
        """Clear all files in the base directory."""  # 清除基础目录中的所有文件。
        if self.base_dir == "":  # 如果基础目录为空
            logger.warning("Base directory is empty, skipping clear operation")  # 基础目录为空，跳过清除操作
            return  # 直接返回

        try:  # 尝试清除文件
            for root, dirs, files in os.walk(self.base_dir):  # 遍历目录树
                for file in files:  # 遍历每个文件
                    os.remove(os.path.join(root, file))  # 删除文件
            logger.debug(f"Cleared all files in base directory: {self.base_dir}")  # 记录清除成功
        except Exception as e:  # 捕获异常
            logger.error(  # 记录错误
                f"Failed to clear files in base directory {self.base_dir}: {e}"  # 清除基础目录中的文件失败
            )

    def get_file_path(self, key: str) -> str:  # 获取键对应的文件路径 # 根据键获取完整的文件路径
        """Get full file path for a given key."""  # 获取给定键的完整文件路径。
        return os.path.join(self.base_dir, key)  # 拼接基础目录和键名得到完整路径

    def open_file(self, file_path: str, create: bool = False) -> Optional[int]:  # 打开文件并返回文件描述符 # 打开文件，支持O_DIRECT和O_CREAT选项
        """Open a file and return its file descriptor.
        打开文件并返回其文件描述符。

        If ``create`` is True, the file is created if it does not exist
        (mode 0o644, no truncation). When ``self.use_direct_io`` is True,
        the file is opened with ``O_DIRECT`` (bypasses the OS page cache);
        falls back to buffered I/O with a warning if ``O_DIRECT`` is
        unavailable on this platform.
        如果``create``为True，文件不存在时创建（模式0o644，不截断）。
        当``self.use_direct_io``为True时，文件使用``O_DIRECT``打开
        （绕过OS页缓存）；如果此平台不支持``O_DIRECT``，
        则回退到缓冲I/O并发出警告。
        """
        flags = os.O_RDWR | os.O_CREAT if create else os.O_RDWR  # 设置文件打开标志
        if self.use_direct_io:  # 如果使用直接I/O
            if hasattr(os, "O_DIRECT"):  # 如果系统支持O_DIRECT
                flags |= os.O_DIRECT  # 添加O_DIRECT标志
            else:  # 系统不支持O_DIRECT
                logger.warning(  # 记录警告
                    "use_direct_io is True, but O_DIRECT is not available on "  # use_direct_io为True，但O_DIRECT在此系统上不可用
                    "this system. Falling back to buffered I/O."  # 回退到缓冲I/O。
                )
        try:  # 尝试打开文件
            return os.open(file_path, flags, 0o644)  # 打开文件并返回文件描述符
        except Exception as e:  # 捕获异常
            logger.error(f"Failed to open file {file_path}: {e}")  # 记录打开文件失败
            return None  # 返回None

    def close_file(self, fd: int) -> bool:  # 关闭文件描述符 # 关闭指定的文件描述符
        """Close a file descriptor."""  # 关闭文件描述符。
        try:  # 尝试关闭
            os.close(fd)  # 关闭文件描述符
            return True  # 返回成功
        except Exception as e:  # 捕获异常
            logger.error(f"Failed to close file descriptor {fd}: {e}")  # 记录关闭文件描述符失败
            return False  # 返回失败
