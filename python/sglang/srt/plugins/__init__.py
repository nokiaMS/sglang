# SGLang统一插件框架模块
# 本模块支持通过setuptools entry_points注册两类插件：
# 1. 硬件平台插件（sglang.srt.platforms）——注册自定义硬件平台
# 2. 通用插件（sglang.srt.plugins）——注入钩子到函数/方法、替换类等
# 插件通过pip安装后自动发现，可通过环境变量选择启用哪些插件。
"""
SGLang Unified Plugin Framework.

Supports two types of plugins via setuptools entry_points:
1. Hardware Platform Plugins (sglang.srt.platforms) - register custom hardware platforms  # 硬件平台插件——注册自定义硬件平台
2. General Plugins (sglang.srt.plugins) - inject hooks into functions/methods, replace classes, etc.  # 通用插件——注入钩子、替换类等

Plugins are discovered automatically when installed via pip.  # 插件通过pip安装后自动发现
- Platform plugins: use ``SGLANG_PLATFORM`` to select when multiple are installed.  # 平台插件：安装多个时用SGLANG_PLATFORM选择
- General plugins: use ``SGLANG_PLUGINS`` (comma-separated) to restrict which are loaded.  # 通用插件：用SGLANG_PLUGINS限制加载哪些
"""

import logging  # 导入日志模块
from collections.abc import Callable  # 导入可调用类型
from importlib.metadata import entry_points  # 导入入口点发现功能
from typing import Any  # 导入任意类型

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.plugins.hook_registry import (  # 导入钩子注册表相关组件
    HookRegistry,
    HookSource,
    _current_plugin_source,
)

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

# Entry point group names  # 入口点组名
PLATFORM_PLUGINS_GROUP = "sglang.srt.platforms"  # 硬件平台插件入口点组名
GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"  # 通用插件入口点组名

# Guard against multiple loads in the same process  # 防止在同一进程中多次加载
_plugins_loaded = False  # 插件是否已加载标志


def load_plugins_by_group(
    group: str,
    excluded_dists: set[str] | None = None,
) -> dict[str, tuple[Callable[[], Any], str | None]]:
    """
    Discover and load plugins registered under the given entry point group.  # 发现并加载指定入口点组下注册的插件

    Args:
        group: The setuptools entry_point group name.  # setuptools入口点组名
        excluded_dists: Distribution names to skip. Plugins from these  # 要跳过的发行版名称
            distributions are never ``ep.load()``-ed (avoids importing  # 这些发行版的插件不会被加载（避免导入
            their modules and pulling hardware-specific dependencies).  # 它们的模块并拉取硬件特定依赖）

    Returns:
        Dictionary mapping plugin name to ``(callable, dist_name)``.  # 插件名到(callable, dist_name)的映射字典
    """
    # SGLANG_PLUGINS whitelist (comma-separated plugin names)  # SGLANG_PLUGINS白名单（逗号分隔的插件名称）
    allowed_set: set[str] | None = None  # 允许的插件名集合
    allowed_str = envs.SGLANG_PLUGINS.get()  # 获取SGLANG_PLUGINS环境变量
    if allowed_str:  # 如果设置了白名单
        allowed_set = {x.strip() for x in allowed_str.split(",") if x.strip()}  # 解析为集合

    discovered = entry_points(group=group)  # 发现指定组的所有入口点
    if len(discovered) == 0:  # 如果没有发现插件
        logger.debug("No plugins found for group %s.", group)  # 记录调试日志
        return {}  # 返回空字典

    logger.info("Available plugins for group %s:", group)  # 记录可用插件信息
    for ep in discovered:  # 遍历所有入口点
        logger.info("  - %s -> %s", ep.name, ep.value)  # 记录每个入口点

    plugins: dict[str, tuple[Callable[[], Any], str | None]] = {}  # 已加载插件字典
    for ep in discovered:  # 遍历所有入口点
        if allowed_set is not None and ep.name not in allowed_set:  # 如果白名单存在且插件不在白名单中
            logger.info("Skipping plugin %s (not in SGLANG_PLUGINS)", ep.name)  # 记录跳过信息
            continue  # 跳过
        dist_name = ep.dist.name if ep.dist else None  # 获取发行版名称
        if excluded_dists and dist_name in excluded_dists:  # 如果在排除列表中
            logger.info(
                "Skipping plugin %s (dist %s excluded by SGLANG_PLATFORM)",
                ep.name,
                dist_name,
            )
            continue  # 跳过
        try:  # 尝试加载插件
            func = ep.load()  # 加载入口点函数
            plugins[ep.name] = (func, dist_name)  # 存入插件字典
            logger.info("Loaded plugin %s from group %s", ep.name, group)  # 记录加载成功
        except Exception:  # 加载失败
            logger.exception("Failed to load plugin %s from group %s", ep.name, group)  # 记录异常

    return plugins  # 返回已加载插件字典


def _get_excluded_dists() -> set[str]:  # 获取应排除的发行版名称集合
    """Compute dist names to skip when ``SGLANG_PLATFORM`` is set.  # 当SGLANG_PLATFORM设置时计算要跳过的发行版名称

    Returns dist names that provide a platform plugin but are NOT the one
    selected by ``SGLANG_PLATFORM``.  This prevents unselected platform
    packages from registering hooks that pull their hardware dependencies.  # 返回提供了平台插件但不是SGLANG_PLATFORM选择的发行版名称
    """
    selected = envs.SGLANG_PLATFORM.get()  # 获取选中的平台
    if not selected:  # 如果未选择
        return set()  # 返回空集合
    platform_eps = entry_points(group=PLATFORM_PLUGINS_GROUP)  # 发现所有平台插件入口点
    return {ep.dist.name for ep in platform_eps if ep.dist and ep.name != selected}  # 返回未选中的发行版名称集合


def load_plugins():  # 加载并执行所有通用插件
    """
    Load and execute all general plugins, then apply registered hooks.  # 加载并执行所有通用插件，然后应用已注册的钩子

    Idempotent - safe to call multiple times. General plugins are functions
    whose side effects (registering hooks, replacing classes, etc.) are the
    desired behavior. Return values are ignored.  # 幂等——可安全多次调用，通用插件的副作用（注册钩子、替换类等）即为预期行为，返回值被忽略

    When ``SGLANG_PLATFORM`` is set, general plugins from unselected platform
    packages are automatically skipped (avoids pulling their dependencies).  # 当SGLANG_PLATFORM设置时，自动跳过未选中平台包的通用插件

    After all plugins execute, ``HookRegistry.apply_hooks()`` is called
    automatically so callers only need this single function call.  # 所有插件执行后，自动调用HookRegistry.apply_hooks()，调用者只需调用此函数

    This should be called early in every process (main, engine core, workers).  # 应在每个进程（主进程、引擎核心、worker）中尽早调用
    """
    global _plugins_loaded  # 声明使用全局变量
    if _plugins_loaded:  # 如果已加载
        return  # 直接返回
    _plugins_loaded = True  # 标记为已加载

    plugins = load_plugins_by_group(  # 加载通用插件
        GENERAL_PLUGINS_GROUP,
        excluded_dists=_get_excluded_dists(),  # 排除未选中平台的发行版
    )

    for name, (func, dist_name) in plugins.items():  # 遍历所有插件
        source = HookSource(plugin_name=name, dist_name=dist_name)  # 创建钩子来源信息
        token = _current_plugin_source.set(source)  # 设置当前插件来源上下文变量
        try:  # 尝试执行插件
            func()  # 执行插件函数
            logger.info("Executed general plugin: %s", name)  # 记录执行成功
        except Exception:  # 执行失败
            logger.exception("Failed to execute general plugin: %s", name)  # 记录异常
        finally:  # 无论成功失败
            _current_plugin_source.reset(token)  # 重置上下文变量

    # Apply all registered hooks (idempotent — already-patched targets are skipped).  # 应用所有已注册的钩子（幂等——已修补的目标会被跳过）
    HookRegistry.apply_hooks()  # 应用钩子
