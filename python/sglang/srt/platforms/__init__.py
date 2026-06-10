# SGLang平台发现与延迟初始化模块
# 本模块提供current_platform作为模块级延迟单例，首次访问时通过
# entry_points发现平台插件并实例化相应的SRTPlatform子类。
"""
SGLang Platform Discovery and Lazy Initialization.

Provides `current_platform` as a module-level lazy singleton. On first access,
it discovers platform plugins via entry_points and instantiates the appropriate
SRTPlatform subclass.

Usage:
    from sglang.srt.platforms import current_platform
    print(current_platform.device_name)
"""

import logging  # 导入日志模块
import pkgutil  # 导入包工具模块
from importlib.metadata import entry_points  # 导入入口点发现功能

import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.platforms.cuda import CudaSRTPlatform  # 导入CUDA平台类
from sglang.srt.platforms.interface import SRTPlatform  # 导入SRT平台基类
from sglang.srt.platforms.rocm import RocmSRTPlatform  # 导入ROCm平台类
from sglang.srt.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group  # 导入插件相关功能

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

_current_platform: SRTPlatform | None = None  # 当前平台实例，初始为None


def _is_cuda_available() -> bool:  # 检查CUDA是否可用
    """检查CUDA是否可用（非ROCm环境）"""  # Check if CUDA is available (non-ROCm)
    return bool(torch.cuda.is_available() and torch.version.hip is None)  # CUDA可用且非HIP


def _is_rocm_available() -> bool:  # 检查ROCm是否可用
    """检查ROCm是否可用"""  # Check if ROCm is available
    return bool(torch.cuda.is_available() and torch.version.hip is not None)  # CUDA可用且是HIP


def _resolve_platform() -> SRTPlatform:
    """
    Discover and instantiate the active platform.  # 发现并实例化活动平台

    Discovery flow:  # 发现流程
    1. Branch on SGLANG_PLATFORM:  # 根据SGLANG_PLATFORM分支

       SGLANG_PLATFORM set (front-loading filter):  # SGLANG_PLATFORM已设置（前置过滤）
         - Enumerate entry_points without importing any plugin modules  # 枚举入口点而不导入任何插件模块
         - Only ep.load() + activate() the named plugin  # 仅加载并激活指定插件
         - Other plugins are never imported (avoids pulling their dependencies)  # 其他插件不会被导入（避免拉取依赖）
         - Plugin name not found → RuntimeError  # 插件名未找到则抛出运行时错误
         - activate() returns None → RuntimeError (hardware unavailable)  # activate()返回None则抛出运行时错误（硬件不可用）

       SGLANG_PLATFORM unset (auto-discover):  # SGLANG_PLATFORM未设置（自动发现）
         - Import and activate all discovered plugins  # 导入并激活所有发现的插件
         - 0 activated + CUDA available → fallback CudaSRTPlatform  # 0个激活且CUDA可用则回退到CUDA平台
         - 0 activated + ROCm available → fallback RocmSRTPlatform  # 0个激活且ROCm可用则回退到ROCm平台
         - 0 activated + neither → fallback base SRTPlatform  # 0个激活且都不可用则回退到基类平台
         - 1 activated → use it  # 1个激活则使用它
         - N activated → RuntimeError (must set SGLANG_PLATFORM)  # N个激活则抛出错误（必须设置SGLANG_PLATFORM）

       SGLANG_PLATFORM matches against entry_point names.  # SGLANG_PLATFORM与入口点名称匹配
    """
    selected = envs.SGLANG_PLATFORM.get()  # 获取SGLANG_PLATFORM环境变量

    if selected:  # 如果指定了平台
        # Front-loading filter: only import and activate the specified plugin.  # 前置过滤：仅导入并激活指定插件
        # Other plugins' modules are never loaded — avoids pulling their deps.  # 其他插件的模块不会被加载——避免拉取依赖
        discovered = entry_points(group=PLATFORM_PLUGINS_GROUP)  # 发现所有平台插件入口点
        ep_map = {ep.name: ep for ep in discovered}  # 构建名称到入口点的映射

        if selected not in ep_map:  # 如果指定名称不在映射中
            available = ", ".join(f"'{n}'" for n in ep_map) if ep_map else "none"  # 构建可用列表
            raise RuntimeError(  # 抛出运行时错误
                f"SGLANG_PLATFORM={selected!r} not found in discovered platform plugins "
                f"(available: {available}). Install the plugin with 'pip install -e' "
                f"to register its entry_points."
            )

        try:  # 尝试加载并激活插件
            plugin_fn = ep_map[selected].load()  # 加载插件函数
            result = plugin_fn()  # 调用插件激活函数
        except Exception:  # 加载或激活失败
            logger.exception("Failed to activate platform plugin: %s", selected)  # 记录异常
            raise  # 重新抛出异常

        if result is None:  # 如果激活函数返回None
            raise RuntimeError(  # 抛出运行时错误
                f"Platform plugin {selected!r} is installed but activate() returned None "
                f"(hardware not available on this machine?)."
            )
        logger.info("OOT platform plugin activated: %s -> %s", selected, result)  # 记录激活成功
        return _load_platform_class(result)()  # 加载并实例化平台类

    # Auto-discover: import and activate all plugins, expect exactly one  # 自动发现：导入并激活所有插件，期望恰好一个
    all_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)  # 加载所有平台插件

    activated: dict[str, str] = {}  # 已激活的插件映射
    for name, (plugin_fn, _dist) in all_plugins.items():  # 遍历所有插件
        try:  # 尝试激活
            result = plugin_fn()  # 调用插件激活函数
            if result is not None:  # 如果激活成功
                activated[name] = result  # 记录激活结果
                logger.info("OOT platform plugin activated: %s -> %s", name, result)  # 记录日志
        except Exception:  # 激活失败
            logger.exception("Failed to activate platform plugin: %s", name)  # 记录异常

    if len(activated) == 0:  # 没有插件激活
        if _is_cuda_available():  # 如果CUDA可用
            logger.debug(
                "No platform plugin detected. Using CUDA SRTPlatform defaults."
            )
            return CudaSRTPlatform()  # 返回CUDA平台实例
        if _is_rocm_available():  # 如果ROCm可用
            logger.debug(
                "No platform plugin detected. Using ROCm SRTPlatform defaults."
            )
            return RocmSRTPlatform()  # 返回ROCm平台实例
        logger.debug("No platform detected. Using base SRTPlatform.")  # 记录调试日志
        return SRTPlatform()  # 返回基类平台实例

    if len(activated) == 1:  # 恰好一个插件激活
        name, qualname = next(iter(activated.items()))  # 获取唯一的激活结果
        return _load_platform_class(qualname)()  # 加载并实例化平台类

    # Multiple activated without SGLANG_PLATFORM  # 多个插件激活但未设置SGLANG_PLATFORM
    names_str = ", ".join(f"'{n}'" for n in activated)  # 构建名称列表
    raise RuntimeError(  # 抛出运行时错误
        f"Multiple platform plugins activated: {names_str}. "
        f"Set SGLANG_PLATFORM to select one."
    )


def _load_platform_class(qualname: str) -> type:  # 根据全限定名加载平台类
    """Load an SRTPlatform subclass from its fully-qualified class name."""  # 根据全限定类名加载SRTPlatform子类
    cls = pkgutil.resolve_name(qualname)  # 解析全限定名称获取类
    if not isinstance(cls, type) or not issubclass(cls, SRTPlatform):  # 检查是否是SRTPlatform子类
        raise TypeError(
            f"Expected an SRTPlatform subclass, got {type(cls)}: {qualname}"
        )
    return cls  # 返回平台类


def __getattr__(name: str):  # 模块属性访问钩子
    """Lazy initialization of current_platform on first access."""  # 首次访问时延迟初始化current_platform
    if name == "current_platform":  # 如果访问的是current_platform
        global _current_platform  # 声明使用全局变量
        if _current_platform is None:  # 如果尚未初始化
            _current_platform = _resolve_platform()  # 解析并实例化平台
        return _current_platform  # 返回平台实例
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # 抛出属性错误
