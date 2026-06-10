"""Internal storage backing ``SpeculativeAlgorithm.register``. Plugins
should use that classmethod API; do not import from this module directly.
"""
# 内部存储，支撑 SpeculativeAlgorithm.register 的注册机制。插件应使用该类方法 API；请勿直接从此模块导入。

from __future__ import annotations  # 启用延迟注解评估，允许在类型注解中使用尚未定义的类型

from typing import TYPE_CHECKING, Callable, Dict, Optional, Type  # 导入类型检查相关工具：TYPE_CHECKING 用于条件导入，Callable 可调用类型，Dict 字典类型，Optional 可选类型，Type 类型对象

import torch  # 导入 PyTorch 深度学习框架

if TYPE_CHECKING:  # 仅在类型检查时执行以下导入，运行时不导入以避免循环依赖
    from sglang.srt.managers.overlap_utils import FutureMap  # 导入 FutureMap 类型，用于异步未来结果映射
    from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入 ScheduleBatch 类型，表示调度批次
    from sglang.srt.server_args import ServerArgs  # 导入 ServerArgs 类型，表示服务器参数
    from sglang.srt.speculative.spec_info import SpecInput  # 导入 SpecInput 类型，表示投机解码的输入信息

WorkerFactory = Callable[["ServerArgs"], Type]  # 工厂函数类型：接受 ServerArgs 参数，返回一个类型（通常是 Worker 类）
ServerArgsValidator = Callable[["ServerArgs"], None]  # 服务器参数验证器类型：接受 ServerArgs 参数，无返回值，通常用于校验参数合法性


class CustomSpecAlgo:
    """A plugin-registered speculative algorithm. Duck-types
    ``SpeculativeAlgorithm`` enum values (same ``is_*()`` / ``create_worker``
    interface).

    Plugins may subclass this to override any ``is_*()`` / ``supports_*()`` /
    ``create_worker`` method (e.g. to integrate with builtin-specific
    branches like ``if spec_algorithm.is_eagle():`` in scheduler /
    model_runner). Pass the subclass via ``spec_class=...`` at registration.

    Defaults: all ``is_*()`` return ``False`` except ``is_speculative``;
    ``supports_spec_v2`` follows ``supports_overlap``.
    """
    # 插件注册的投机解码算法。采用鸭子类型与 SpeculativeAlgorithm 枚举值兼容
    # （相同的 is_*() / create_worker 接口）。
    #
    # 插件可以继承此类并覆盖任意 is_*() / supports_*() / create_worker 方法
    # （例如，为了与调度器/model_runner 中的内置特定分支集成，如
    # if spec_algorithm.is_eagle():）。注册时通过 spec_class=... 传入子类。
    #
    # 默认行为：所有 is_*() 返回 False，除了 is_speculative 返回 True；
    # supports_spec_v2 跟随 supports_overlap 的值。

    def __init__(
        self,
        name: str,  # 算法名称
        factory: WorkerFactory,  # Worker 工厂函数
        *,
        supports_overlap: bool = False,  # 是否支持重叠调度，默认不支持
        validate_server_args: Optional[ServerArgsValidator] = None,  # 可选的服务器参数验证函数，默认为 None
    ):
        self.name = name  # 保存算法名称
        self.factory = factory  # 保存 Worker 工厂函数
        self.supports_overlap = supports_overlap  # 保存是否支持重叠调度的标志
        self.validate_server_args = validate_server_args  # 保存服务器参数验证函数

    def __repr__(self) -> str:  # 定义对象的字符串表示形式
        return f"CustomSpecAlgo({self.name!r})"  # 返回包含算法名称的可读字符串表示

    def is_none(self) -> bool:  # 判断是否为"无算法"类型
        return False  # 自定义算法不是"无算法"，返回 False

    def is_speculative(self) -> bool:  # 判断是否为投机解码算法
        return True  # 自定义算法默认是投机解码算法，返回 True

    def is_eagle(self) -> bool:  # 判断是否为 EAGLE 算法
        return False  # 自定义算法默认不是 EAGLE，返回 False

    def is_eagle3(self) -> bool:  # 判断是否为 EAGLE3 算法
        return False  # 自定义算法默认不是 EAGLE3，返回 False

    def is_dflash(self) -> bool:  # 判断是否为 DFLASH 算法
        return False  # 自定义算法默认不是 DFLASH，返回 False

    def is_standalone(self) -> bool:  # 判断是否为 STANDALONE 算法
        return False  # 自定义算法默认不是 STANDALONE，返回 False

    def is_ngram(self) -> bool:  # 判断是否为 NGRAM 算法
        return False  # 自定义算法默认不是 NGRAM，返回 False

    def supports_target_verify_for_draft(self) -> bool:  # 判断是否支持 draft 模式的 target verify
        return False  # 自定义算法默认不支持 target verify for draft，返回 False

    def supports_spec_v2(self) -> bool:  # 判断是否支持投机解码 V2 接口
        return self.supports_overlap  # V2 支持跟随重叠调度标志，支持重叠则支持 V2

    def create_worker(self, server_args: "ServerArgs") -> Type:  # 创建 Worker 实例
        if not server_args.disable_overlap_schedule and not self.supports_overlap:  # 如果未禁用重叠调度且算法不支持重叠调度
            raise ValueError(  # 抛出值错误异常
                f"Speculative algorithm {self.name} does not support overlap scheduling."  # 提示该算法不支持重叠调度
            )
        return self.factory(server_args)  # 调用工厂函数创建并返回 Worker 实例

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool  # draft token 数量和是否为 draft worker 标志
    ) -> int:  # 返回 target verify 阶段每个批次的 token 数
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        # 待修复：在前向模式重构后移除此方法。Target verify 本质上是固定序列长度的
        # prefill/extend，具有完整的 CUDA graph 支持。我们可以将其用于 target verify，
        # 也可以用于非 target verify 但固定长度 prefill 的其他场景。
        # 此处暴露此接口以允许其他用例。
        return num_draft_tokens  # 默认返回 draft token 数量

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,  # 调度批次
        server_args: ServerArgs,  # 服务器参数
        last_tokens_tensor: torch.Tensor,  # 最近的 token 张量
        future_map: FutureMap,  # 异步未来结果映射
    ) -> Optional[SpecInput]:  # 返回可选的投机解码输入
        return None  # 默认返回 None，表示不支持分离式 draft 输入构建


_REGISTRY: Dict[str, CustomSpecAlgo] = {}  # 全局注册表，存储名称到自定义投机算法的映射

# Builtin enum members + the NEXTN alias; plugins cannot shadow these.
_RESERVED_NAMES = frozenset(  # 保留名称集合，内置枚举成员及 NEXTN 别名，插件不能覆盖这些名称
    {"DFLASH", "EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM", "NONE"}  # 内置算法名称的保留集合
)


def register_algorithm(
    name: str,  # 算法名称
    *,
    supports_overlap: bool = False,  # 是否支持重叠调度，默认不支持
    validate_server_args: Optional[ServerArgsValidator] = None,  # 可选的服务器参数验证函数
    spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,  # 自定义算法类，默认为 CustomSpecAlgo
) -> Callable[[WorkerFactory], WorkerFactory]:  # 返回一个装饰器函数
    """Return a decorator that registers a plugin algorithm under ``name``.

    Pass a ``spec_class`` subclass of ``CustomSpecAlgo`` to override any
    ``is_*()`` / ``supports_*()`` / ``create_worker`` method.
    """
    # 返回一个装饰器，用于在指定名称下注册插件算法。
    # 传入 CustomSpecAlgo 的子类作为 spec_class，以覆盖任意
    # is_*() / supports_*() / create_worker 方法。
    upper = name.upper()  # 将算法名称转为大写作为注册键
    if upper in _RESERVED_NAMES:  # 如果名称与保留名称冲突
        raise ValueError(  # 抛出值错误异常
            f"'{upper}' is a reserved speculative algorithm name; cannot be re-registered."  # 提示该名称为保留名称，无法重新注册
        )
    if upper in _REGISTRY:  # 如果名称已被注册
        raise ValueError(f"Speculative algorithm '{upper}' already registered.")  # 提示该算法已注册，抛出异常

    def decorator(factory: WorkerFactory) -> WorkerFactory:  # 定义装饰器函数，接受工厂函数并返回它
        _REGISTRY[upper] = spec_class(  # 在注册表中以大写名称为键，创建算法实例并存入
            name=upper,  # 算法名称（大写）
            factory=factory,  # Worker 工厂函数
            supports_overlap=supports_overlap,  # 是否支持重叠调度
            validate_server_args=validate_server_args,  # 服务器参数验证函数
        )
        return factory  # 返回原始工厂函数，使其仍可正常使用

    return decorator  # 返回装饰器函数


def get_spec(name: Optional[str]) -> Optional[CustomSpecAlgo]:  # 根据名称获取已注册的投机算法，返回 Optional
    """Return the registered spec for ``name``, or ``None`` for builtin /
    unknown names."""
    # 返回指定名称的已注册投机算法，若为内置/未知名称则返回 None。
    if name is None:  # 如果名称为 None
        return None  # 直接返回 None
    return _REGISTRY.get(name.upper())  # 以大写名称为键查找注册表，返回对应的算法实例或 None
