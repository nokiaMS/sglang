# SGLang插件钩子注册表模块
# 本模块提供before/after/around/replace四种钩子类型，可应用于sglang代码库中的
# 任何函数、方法或类。钩子在插件加载期间注册，在引擎启动前应用。
"""
Hook registry for SGLang plugins.

Provides before/after/around/replace hooks that can be applied to any
function, method, or class in the sglang codebase. Hooks are registered
during plugin loading and applied before the engine starts.

Usage:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def my_timer(original_fn, *args, **kwargs):
        start = time.perf_counter()
        result = original_fn(*args, **kwargs)
        print(f"Elapsed: {time.perf_counter() - start:.3f}s")
        return result

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.schedule",
        my_timer,
        HookType.AROUND,
    )
"""

import contextvars  # 导入上下文变量模块
import functools  # 导入函数工具模块
import logging  # 导入日志模块
import pkgutil  # 导入包工具模块
import sys  # 导入系统模块
import types  # 导入类型模块
from collections import defaultdict  # 导入默认字典
from collections.abc import Callable  # 导入可调用类型
from enum import Enum  # 导入枚举类型
from typing import NamedTuple  # 导入命名元组

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class HookSource(NamedTuple):
    """Identifies which plugin registered a hook."""  # 标识注册钩子的插件来源

    plugin_name: str  # entry_point name, e.g. "xpu_hooks"  # 入口点名称
    dist_name: str | None  # distribution name, e.g. "sglang_xpu_platform"  # 发行版名称


# Set by load_plugins() around each plugin's func() call, read by register().  # 由load_plugins()在每个插件的func()调用前后设置，由register()读取
_current_plugin_source: contextvars.ContextVar[HookSource | None] = (
    contextvars.ContextVar("_current_plugin_source", default=None)  # 当前插件来源上下文变量
)


def _format_source(source: HookSource | None) -> str:  # 格式化来源信息用于日志
    """Format source info for log messages."""  # 格式化来源信息用于日志消息
    if source is None:  # 如果来源为空
        return "unknown"  # 返回"unknown"
    if source.dist_name:  # 如果有发行版名称
        return f"plugin={source.plugin_name}, dist={source.dist_name}"  # 返回完整信息
    return f"plugin={source.plugin_name}"  # 仅返回插件名称


class HookType(Enum):
    """Types of hooks that can be applied to functions or classes."""  # 可应用于函数或类的钩子类型

    BEFORE = "before"  # Execute before original; can modify args  # 在原始函数前执行；可修改参数
    AFTER = "after"  # Execute after original; can modify return value  # 在原始函数后执行；可修改返回值
    AROUND = "around"  # Wrap original; full control over execution  # 包裹原始函数；完全控制执行
    REPLACE = "replace"  # Replace the original function or class entirely  # 完全替换原始函数或类


class HookRegistry:
    """
    Global registry for function/method/class hooks.  # 函数/方法/类钩子的全局注册表

    Thread safety: All registration should happen during load_plugins()
    phase (single-threaded). apply_hooks() should be called once before the
    engine starts serving requests.  # 线程安全：所有注册应在load_plugins()阶段（单线程）完成，apply_hooks()应在引擎开始服务请求前调用一次
    """

    _hooks: dict[str, list[tuple[HookType, Callable, HookSource | None]]] = defaultdict(
        list
    )  # 钩子注册表：目标路径到钩子列表的映射
    _patched: set[str] = set()  # 已修补的目标路径集合

    @classmethod
    def register(
        cls,
        target: str,
        hook: Callable,
        hook_type: HookType = HookType.AFTER,
        *,
        source: HookSource | None = None,
    ):  # 注册钩子
        """
        Register a hook on a target function, method, or class.  # 在目标函数、方法或类上注册钩子

        Args:
            target: Fully-qualified dotted path to the target.  # 目标的完全限定点分路径
                    e.g. "sglang.srt.managers.scheduler.Scheduler.schedule"
                    or   "sglang.srt.managers.scheduler.Scheduler" (class)
            hook: The hook callable (function or class). Signature depends on hook_type:  # 钩子可调用对象，签名取决于钩子类型
                - BEFORE:  fn(*args, **kwargs) -> (args, kwargs) or None
                - AFTER:   fn(result, *args, **kwargs) -> new_result or None
                - AROUND:  fn(original_fn, *args, **kwargs) -> result
                - REPLACE: fn(*args, **kwargs) -> result   (function replacement)
                           MyClass                         (class replacement)
            hook_type: Type of hook (default: AFTER).  # 钩子类型（默认AFTER）
            source: Optional source info. If None, auto-read from context var  # 可选来源信息，为None时自动从上下文变量读取
                set by ``load_plugins()``.

        Raises:
            TypeError: If a class is passed with a hook_type other than REPLACE.  # 如果类与非REPLACE钩子类型一起传入
        """
        if isinstance(hook, type) and hook_type != HookType.REPLACE:  # 如果传入类但钩子类型不是REPLACE
            raise TypeError(
                f"Class {hook.__name__} can only be used with HookType.REPLACE, "
                f"got HookType.{hook_type.name}. "
                f"Use a function for BEFORE/AFTER/AROUND hooks."
            )
        resolved_source = source or _current_plugin_source.get()  # 解析来源信息
        # Warn on duplicate REPLACE for the same target  # 对同一目标的重复REPLACE发出警告
        if hook_type == HookType.REPLACE:  # 如果是REPLACE类型
            existing_replace = [
                (h, src) for ht, h, src in cls._hooks[target] if ht == HookType.REPLACE
            ]  # 查找已有的REPLACE钩子
            if existing_replace:  # 如果存在重复REPLACE
                prev, prev_src = existing_replace[-1]  # 获取上一个REPLACE钩子
                prev_name = getattr(prev, "__qualname__", None) or repr(prev)  # 获取上一个钩子名称
                new_name = getattr(hook, "__qualname__", None) or repr(hook)  # 获取新钩子名称
                logger.warning(
                    "Multiple REPLACE hooks on '%s': previous (%s [%s]) will be "
                    "overridden by (%s [%s]). The last registered REPLACE takes effect.",
                    target,
                    prev_name,
                    _format_source(prev_src),
                    new_name,
                    _format_source(resolved_source),
                )  # 记录警告
        cls._hooks[target].append((hook_type, hook, resolved_source))  # 添加到钩子列表
        logger.debug(
            "Registered %s hook on %s [%s]",
            hook_type.value,
            target,
            _format_source(resolved_source),
        )  # 记录注册信息

    @classmethod
    def apply_hooks(cls):  # 应用所有已注册的钩子
        """
        Apply all registered hooks to their target functions/classes.  # 将所有已注册的钩子应用到目标函数/类

        This performs the actual monkey-patching. Should be called once after
        all plugins have been loaded and before the engine starts.  # 执行实际的猴子补丁，应在所有插件加载后、引擎启动前调用一次

        Targets with class REPLACE hooks are applied first, so that
        subsequent method-level hooks (AROUND, BEFORE, AFTER) on child
        attributes resolve against the *replaced* class rather than the
        original.  # 类REPLACE钩子优先应用，以便后续方法级钩子解析到替换后的类
        """
        sorted_items = sorted(cls._hooks.items(), key=cls._target_sort_key)  # 按排序键排序
        for target, hooks in sorted_items:  # 遍历排序后的目标
            if target in cls._patched:  # 如果已修补
                continue  # 跳过
            try:  # 尝试应用钩子
                cls._apply_target(target, hooks)  # 应用钩子到目标
                cls._patched.add(target)  # 标记为已修补
            except Exception:  # 应用失败
                logger.exception("Failed to apply hooks to %s", target)  # 记录异常

    @staticmethod
    def _target_sort_key(item):  # 目标排序键
        """Sort key: class REPLACE targets (tier 0) before all others (tier 1).  # 排序键：类REPLACE目标（层级0）优先于其他（层级1）

        This ensures that when a class is replaced, subsequent method-level
        hooks on ``ClassName.method`` resolve against the replacement class.  # 确保类被替换后，后续方法级钩子解析到替换类
        """
        _target, hooks = item  # 解构目标路径和钩子列表
        has_class_replace = any(
            isinstance(h, type) and ht == HookType.REPLACE for ht, h, _ in hooks
        )  # 检查是否有类REPLACE钩子
        return (0 if has_class_replace else 1, _target)  # 类REPLACE排前面

    @classmethod
    def _apply_target(cls, target: str, hooks: list):  # 应用钩子到指定目标
        """Resolve target, build wrapper chain, and replace the original."""  # 解析目标，构建包装链，替换原始对象
        parts = target.rsplit(".", 1)  # 从右侧分割目标路径
        if len(parts) != 2:  # 如果分割结果不是两部分
            raise ValueError(
                f"Invalid target path (need at least module.attr): {target}"
            )  # 抛出值错误

        obj_path, attr_name = parts  # 获取对象路径和属性名
        obj = pkgutil.resolve_name(obj_path)  # 解析对象路径获取对象

        # Check if the original is a classmethod or staticmethod by  # 检查原始对象是否是类方法或静态方法
        # inspecting __dict__ before getattr() triggers the descriptor  # 在getattr()触发描述符协议之前检查__dict__
        # protocol (which would lose the wrapper type for classmethod).  # 否则会丢失类方法的包装类型
        original = getattr(obj, attr_name)  # 获取原始属性
        is_classmethod = False  # 是否是类方法标志
        is_staticmethod = False  # 是否是静态方法标志
        if isinstance(obj, type):  # 如果对象是类
            raw_attr = obj.__dict__.get(attr_name)  # 从类__dict__获取原始属性
            if isinstance(raw_attr, classmethod):  # 如果是类方法
                is_classmethod = True  # 设置标志
                original = raw_attr.__func__  # 获取底层函数
            elif isinstance(raw_attr, staticmethod):  # 如果是静态方法
                is_staticmethod = True  # 设置标志
                original = raw_attr.__func__  # 获取底层函数

        # Cross-target conflict detection: if the parent object is a class  # 跨目标冲突检测：如果父对象是类
        # that was already class-REPLACE'd, and the replacement class defines  # 且已被类REPLACE替换，替换类定义了
        # its own version of this method, a method REPLACE here will silently  # 自己版本的此方法，方法REPLACE将静默
        # override the replacement class's implementation.  # 覆盖替换类的实现
        if isinstance(obj, type) and obj_path in cls._patched:  # 如果对象是已替换的类
            has_method_replace = any(ht == HookType.REPLACE for ht, _, _ in hooks)  # 检查是否有方法REPLACE
            if has_method_replace and attr_name in obj.__dict__:  # 如果替换类有自己的方法实现
                replace_sources = [
                    _format_source(src)
                    for ht, _, src in hooks
                    if ht == HookType.REPLACE
                ]  # 获取REPLACE钩子来源
                logger.warning(
                    "Method REPLACE on '%s' will override the class REPLACE's "
                    "own implementation of '%s'. If this is unintended, remove "
                    "the method REPLACE and modify the replacement class "
                    "directly, or use AROUND to wrap it. (from: %s)",
                    target,
                    attr_name,
                    ", ".join(replace_sources),
                )  # 记录警告

        # Guard: if the target is a class, only REPLACE is safe. Wrapping a  # 保护：如果目标是类，只有REPLACE是安全的
        # class in a function would break isinstance/issubclass/inheritance.  # 用函数包装类会破坏isinstance/issubclass/继承
        if isinstance(original, type):  # 如果原始对象是类
            bad = [ht for ht, _, _ in hooks if ht != HookType.REPLACE]  # 查找非REPLACE类型的钩子
            if bad:  # 如果存在非REPLACE类型钩子
                raise TypeError(
                    f"Target '{target}' is a class. Only HookType.REPLACE is "
                    f"allowed for class targets (got {bad[0].value}). "
                    f"To hook a method, use '{target}.<method_name>' instead."
                )  # 抛出类型错误

        # Warn about risky hook combinations  # 警告有风险的钩子组合
        hook_types = [ht for ht, _, _ in hooks]  # 获取所有钩子类型
        around_count = hook_types.count(HookType.AROUND)  # AROUND钩子数量
        has_replace = HookType.REPLACE in hook_types  # 是否有REPLACE钩子
        has_others = any(ht != HookType.REPLACE for ht in hook_types)  # 是否有非REPLACE钩子

        if around_count > 1:  # 多个AROUND钩子
            around_sources = [
                _format_source(src) for ht, _, src in hooks if ht == HookType.AROUND
            ]  # 获取AROUND钩子来源
            logger.warning(
                "Multiple AROUND hooks on '%s' (%d hooks, from: %s). If any AROUND hook "
                "skips calling original_fn, inner hooks will be bypassed.",
                target,
                around_count,
                ", ".join(around_sources),
            )  # 记录警告
        if has_replace and has_others:  # 同时有REPLACE和其他类型钩子
            logger.info(
                "Target '%s' has both REPLACE and %s hooks. "
                "REPLACE will be applied first, then wrapped by other hooks.",
                target,
                ", ".join(
                    sorted({ht.value for ht in hook_types if ht != HookType.REPLACE})
                ),
            )  # 记录信息

        # Build the wrapper chain.  # 构建包装链
        # Sort: REPLACE hooks first (stable sort preserves registration order  # 排序：REPLACE钩子优先（稳定排序保持注册顺序）
        # within the same type). This ensures AROUND/BEFORE/AFTER always wrap  # 确保AROUND/BEFORE/AFTER总是包裹
        # the replaced function, regardless of registration order.  # 被替换的函数，无论注册顺序如何
        sorted_hooks = sorted(
            hooks, key=lambda h: (0 if h[0] == HookType.REPLACE else 1)
        )  # REPLACE排前面
        wrapped = original  # 初始化为原始对象
        for hook_type, hook, _src in sorted_hooks:  # 遍历排序后的钩子
            if isinstance(hook, type) and hook_type == HookType.REPLACE:  # 类替换
                # Class replacement: direct substitution to preserve type identity.  # 类替换：直接替换以保持类型标识
                # This keeps isinstance(), issubclass(), and inheritance working.  # 保持isinstance()、issubclass()和继承正常工作
                wrapped = hook  # 直接替换
            else:  # 函数钩子
                wrapped = _wrap_fn(wrapped, hook, hook_type)  # 构建包装函数

        # Restore classmethod/staticmethod decorator if the original had one.  # 如果原始对象有类方法/静态方法装饰器则恢复
        if is_classmethod:  # 如果原始是类方法
            wrapped = classmethod(wrapped)  # 恢复类方法装饰器
            logger.debug("Preserved @classmethod decorator for %s", target)  # 记录调试日志
        elif is_staticmethod:  # 如果原始是静态方法
            wrapped = staticmethod(wrapped)  # 恢复静态方法装饰器
            logger.debug("Preserved @staticmethod decorator for %s", target)  # 记录调试日志

        setattr(obj, attr_name, wrapped)  # 设置替换后的属性

        # Propagate the patch to all other modules that imported the original  # 将补丁传播到所有导入了原始对象的其他模块
        # via ``from source_module import name``.  Python's ``from X import Y``  # 通过from source_module import name导入
        # copies the reference at import time; patching X alone leaves  # Python的from X import Y在导入时复制引用
        # importers with a stale binding.  # 仅修补X会使导入者持有过时的绑定
        if wrapped is not original:  # 如果替换后不同
            extra = _propagate_patch(original, wrapped, obj)  # 传播补丁
            if extra:  # 如果有额外的模块被修补
                logger.debug(
                    "Propagated patch for %s to %d additional module(s)",
                    target,
                    extra,
                )  # 记录调试日志

        sources = sorted({_format_source(src) for _, _, src in hooks})  # 获取所有来源
        logger.info(
            "Applied %d hook(s) to %s (from: %s)",
            len(hooks),
            target,
            ", ".join(sources),
        )  # 记录应用信息

    @classmethod
    def reset(cls):  # 重置所有钩子和补丁
        """Reset all hooks and patches. Primarily for testing."""  # 重置所有钩子和补丁，主要用于测试
        cls._hooks.clear()  # 清空钩子注册表
        cls._patched.clear()  # 清空已修补集合


def _propagate_patch(original: object, wrapped: object, source_module: object) -> int:
    """Propagate a monkey-patch to all modules holding a stale ``from X import Y`` binding.  # 将猴子补丁传播到所有持有过时from X import Y绑定的模块

    After ``setattr(source_module, name, wrapped)`` updates the defining module,  # setattr更新定义模块后
    other modules that did ``from source_module import name`` still hold a direct  # 其他通过from source_module import name导入的模块仍持有
    reference to the old *original* object.  This walks ``sys.modules`` and  # 对旧原始对象的直接引用，遍历sys.modules
    replaces every such stale binding with *wrapped*.  # 替换所有过时绑定

    Returns the number of additional module attributes that were patched.  # 返回被修补的额外模块属性数量
    """
    patched_count = 0  # 修补计数
    for mod in list(sys.modules.values()):  # 遍历所有已加载模块
        if mod is source_module or mod is None:  # 跳过源模块和None
            continue  # 继续
        if not isinstance(mod, types.ModuleType):  # 跳过非模块对象
            continue  # 继续
        try:  # 尝试获取模块变量
            mod_vars = vars(mod)  # 获取模块变量字典
        except TypeError:  # 获取失败
            continue  # 继续
        for attr_name, attr_value in list(mod_vars.items()):  # 遍历模块变量
            if attr_value is original:  # 如果引用的是原始对象
                try:  # 尝试修补
                    setattr(mod, attr_name, wrapped)  # 替换为新对象
                    patched_count += 1  # 增加计数
                except (AttributeError, TypeError):  # 修补失败
                    pass  # 忽略
    return patched_count  # 返回修补计数


def _wrap_fn(original_fn: Callable, hook: Callable, hook_type: HookType) -> Callable:
    """Create a wrapper function based on the hook type."""  # 根据钩子类型创建包装函数
    if hook_type == HookType.REPLACE:  # REPLACE类型

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):  # 替换包装器
            return hook(*args, **kwargs)  # 调用钩子函数

        wrapper.__wrapped__ = original_fn  # 保存原始函数引用
        return wrapper  # 返回包装器

    elif hook_type == HookType.BEFORE:  # BEFORE类型

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):  # 前置包装器
            result = hook(*args, **kwargs)  # 调用前置钩子
            if result is not None:  # 如果钩子返回了修改后的参数
                args, kwargs = result  # 更新参数
            return original_fn(*args, **kwargs)  # 调用原始函数

        wrapper.__wrapped__ = original_fn  # 保存原始函数引用
        return wrapper  # 返回包装器

    elif hook_type == HookType.AFTER:  # AFTER类型

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):  # 后置包装器
            result = original_fn(*args, **kwargs)  # 调用原始函数
            modified = hook(result, *args, **kwargs)  # 调用后置钩子
            return modified if modified is not None else result  # 返回修改后的结果或原始结果

        wrapper.__wrapped__ = original_fn  # 保存原始函数引用
        return wrapper  # 返回包装器

    elif hook_type == HookType.AROUND:  # AROUND类型

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):  # 环绕包装器
            return hook(original_fn, *args, **kwargs)  # 钩子控制原始函数的调用

        wrapper.__wrapped__ = original_fn  # 保存原始函数引用
        return wrapper  # 返回包装器

    else:  # 未知钩子类型
        raise ValueError(f"Unknown hook type: {hook_type}")  # 抛出值错误


def plugin_hook(
    target: str,
    type: HookType = HookType.AFTER,
) -> Callable:  # 插件钩子装饰器
    """Decorator that registers a function or class as a hook on *target*.  # 将函数或类注册为*target*上钩子的装饰器

    Usage::

        # Function hook (AROUND)  # 函数钩子（AROUND类型）
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler.schedule",
                      type=HookType.AROUND)
        def my_timer(original_fn, *args, **kwargs):
            start = time.perf_counter()
            result = original_fn(*args, **kwargs)
            print(f"Elapsed: {time.perf_counter() - start:.3f}s")
            return result

        # Class replacement (REPLACE)  # 类替换（REPLACE类型）
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler",
                      type=HookType.REPLACE)
        class MyScheduler(Scheduler):
            ...

    The decorated function/class is returned unchanged so it can still be
    used directly if needed.  # 装饰后的函数/类原样返回，仍可直接使用
    """

    def decorator(hook: Callable) -> Callable:  # 装饰器函数
        HookRegistry.register(target, hook, type)  # 注册钩子
        return hook  # 返回原函数/类

    return decorator  # 返回装饰器
