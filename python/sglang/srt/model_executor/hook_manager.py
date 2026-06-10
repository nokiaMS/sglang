# 本文件实现了模型前向钩子（forward hook）的注册与管理机制。
# 它根据用户提供的 hook_specs 配置，将前向钩子函数动态挂载到模型的指定模块上，
# 支持通配符模式匹配模块名称，并通过可调用路径动态加载钩子工厂函数。

import fnmatch
import importlib
import logging
from typing import Any, Callable, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def register_forward_hooks(model: nn.Module, hook_specs: List[dict[str, Any]]) -> None:
    """
    hook_specs is a list of dicts from server_args.forward_hooks.
    Attaches forward hooks to the matching modules.
    """
    # 将模型的所有命名模块构建为 字典，方便按名称查找
    name_to_module = dict(model.named_modules())

    # 遍历每个钩子规格配置
    for spec in hook_specs:
        spec_name = spec.get("name", "")
        # 获取目标模块的匹配模式列表
        target_patterns = spec.get("target_modules", [])
        if not target_patterns:
            logger.warning(f"Hook spec '{spec_name}' has no 'target_modules', skipping")
            continue

        # 获取钩子工厂函数的路径字符串
        hook_factory_path = spec.get("hook_factory")
        if not hook_factory_path:
            logger.warning(f"Hook spec '{spec_name}' has no 'hook_factory', skipping")
            continue

        # 获取钩子的额外配置参数，默认为空字典
        config = spec.get("config") or {}
        # 通过路径动态解析钩子工厂函数
        hook_factory = resolve_callable(hook_factory_path)

        # 调用工厂函数，传入配置，生成实际的钩子函数
        hook = hook_factory(config) if hook_factory else None
        if hook is None:
            logger.warning(
                f"Hook factory '{hook_factory_path}' for spec '{spec_name}' "
                "returned None, not registering any hook"
            )
            continue

        # Resolve patterns like "model.layers.*.mlp"
        # 使用通配符模式匹配，找到所有符合模式的模块
        matched = []
        for name, module in name_to_module.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in target_patterns):
                matched.append((name, module))

        if not matched:
            logger.warning(
                f"No modules matched hook spec '{spec_name}' "
                f"patterns={target_patterns}"
            )
            continue

        # 将钩子注册到每个匹配的模块上
        for module_name, module in matched:
            _ = module.register_forward_hook(hook)
            logger.info(f"Registered forward hook '{spec_name}' " f"on {module_name}")


def resolve_callable(path: Optional[str]) -> Optional[Callable]:
    """根据路径字符串动态解析并返回可调用对象（函数/类方法）。"""

    if path is None:
        return None

    # 支持 "module.submodule:factory" 格式，用冒号分隔模块路径与函数名
    if ":" in path:
        module_name, fn_name = path.split(":", 1)
    else:
        # 也支持 "module.submodule.factory" 格式，最后一个点之后为函数名
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid hook callable path '{path}'. "
                "Expected 'module.submodule:factory' or 'module.submodule.factory'."
            )
        *mod_parts, fn_name = parts
        module_name = ".".join(mod_parts)

    # 动态导入模块并获取其中的属性（函数/方法）
    module = importlib.import_module(module_name)
    try:
        return getattr(module, fn_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{fn_name}' "
            f"(from hook path '{path}')"
        ) from e
