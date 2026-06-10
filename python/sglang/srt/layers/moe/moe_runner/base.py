# MoE Runner基础模块：定义MoE Runner的核心抽象类（MoeRunnerCore、RunnerInput、RunnerOutput、MoeQuantInfo）、
# 配置类（MoeRunnerConfig）、置换方法池（PermuteMethodPool）、融合算子池（FusedOpPool），
# 以及注册装饰器（register_pre_permute、register_post_permute、register_fused_func）等基础设施。
from __future__ import annotations  # 启用延迟类型注解求值

import contextvars  # 上下文变量模块
from abc import ABC, abstractmethod  # 抽象基类和抽象方法
from contextlib import contextmanager  # 上下文管理器工具
from dataclasses import dataclass  # 数据类装饰器
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Tuple, TypeGuard  # 类型提示工具

import torch  # PyTorch深度学习框架

from sglang.srt.layers.moe.utils import (  # 导入MoE工具模块
    MoeA2ABackend,  # MoE A2A后端枚举
    MoeRunnerBackend,  # MoE Runner后端枚举
    RoutingMethodType,  # 路由方法类型枚举
)

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.moe_runner.triton import (  # Triton Runner类型
        TritonRunnerCore,  # Triton Runner核心类
        TritonRunnerInput,  # Triton Runner输入类
        TritonRunnerOutput,  # Triton Runner输出类
    )
    from sglang.srt.layers.moe.token_dispatcher import (  # Token调度器类型
        CombineInput,  # Combine输入基类
        CombineInputFormat,  # Combine输入格式
        DispatchOutput,  # 调度输出基类
        DispatchOutputFormat,  # 调度输出格式
    )


_moe_output_buf: contextvars.ContextVar[Optional[torch.Tensor]] = (  # MoE输出缓冲区上下文变量
    contextvars.ContextVar("moe_output_buf", default=None)  # 默认为None
)


@contextmanager
def moe_output_buffer_ctx(buf: torch.Tensor) -> Generator[None, None, None]:  # 上下文管理器：设置MoE输出缓冲区
    token = _moe_output_buf.set(buf)  # 设置上下文变量值并获取令牌
    try:
        yield  # 执行上下文内代码
    finally:
        _moe_output_buf.reset(token)  # 恢复上下文变量为原始值


@dataclass
class MoeRunnerConfig:  # MoE Runner配置数据类
    # MoE parameters  # MoE参数
    num_experts: Optional[int] = None  # 专家总数
    num_local_experts: Optional[int] = None  # 本地专家数
    hidden_size: Optional[int] = None  # 隐藏层大小
    intermediate_size_per_partition: Optional[int] = None  # 每分区中间维度大小
    layer_id: Optional[int] = None  # 层ID
    top_k: Optional[int] = None  # Top-K值
    num_fused_shared_experts: Optional[int] = None  # 融合共享专家数
    params_dtype: Optional[torch.dtype] = None  # 参数数据类型
    routing_method_type: Optional[RoutingMethodType] = None  # 路由方法类型

    # Runner configuration  # Runner配置
    activation: str = "silu"  # 激活函数，默认为SiLU
    is_gated: bool = True  # 是否使用门控结构
    apply_router_weight_on_input: bool = False  # 是否在输入端应用路由权重
    inplace: bool = True  # 是否原地操作
    no_combine: bool = False  # 是否跳过combine阶段
    routed_scaling_factor: Optional[float] = None  # 路由缩放因子
    gemm1_alpha: Optional[float] = None  # GEMM1 alpha参数
    gemm1_clamp_limit: Optional[float] = None  # GEMM1限幅值
    swiglu_limit: Optional[float] = None  # SwiGLU限幅值


@dataclass
class RunnerInput(ABC):  # Runner输入抽象基类
    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...  # 返回Runner后端类型（抽象属性）

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerInput]:  # 类型守卫：判断是否为Triton Runner输入
        return self.runner_backend == MoeRunnerBackend.TRITON  # 比较后端类型


class RunnerOutput(ABC):  # Runner输出抽象基类
    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...  # 返回Runner后端类型（抽象属性）

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerOutput]:  # 类型守卫：判断是否为Triton Runner输出
        return self.runner_backend == MoeRunnerBackend.TRITON  # 比较后端类型


@dataclass
class MoeQuantInfo(ABC):  # MoE量化信息抽象基类
    """Moe quantization data."""  # MoE量化数据。

    pass  # 空基类，由子类扩展


class MoeRunnerCore(ABC):  # MoE Runner核心抽象基类
    def __init__(self, config: MoeRunnerConfig):  # 初始化，保存Runner配置
        self.config = config

    @abstractmethod
    def run(  # 执行MoE计算（抽象方法）
        self,
        runner_input: RunnerInput,  # Runner输入
        quant_info: MoeQuantInfo,  # 量化信息
        running_state: dict,  # 运行时状态
        hooks: Optional[Any] = None,  # 钩子函数
    ) -> RunnerOutput:  # 返回Runner输出
        pass

    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...  # 返回Runner后端类型（抽象属性）

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerCore]:  # 类型守卫：判断是否为Triton Runner核心
        return self.runner_backend == MoeRunnerBackend.TRITON  # 比较后端类型


class FusedOpPool:  # 融合算子注册池
    _fused_funcs: dict[str, Callable] = {}  # 存储已注册的融合函数

    @classmethod
    def register_fused_func(  # 注册融合函数
        cls, a2a_backend_name: str, runner_backend_name: str, fused_func: Callable
    ):
        key = (a2a_backend_name, runner_backend_name)  # 以(A2A后端名, Runner后端名)为键
        if key in cls._fused_funcs:  # 如果键已存在
            raise ValueError(  # 抛出重复注册异常
                f"Fused function for {a2a_backend_name} to {runner_backend_name} is already registered."
            )
        assert MoeA2ABackend(  # 验证A2A后端名合法
            a2a_backend_name
        ), f"Invalid dispatch name: {a2a_backend_name}"
        assert MoeRunnerBackend(  # 验证Runner后端名合法
            runner_backend_name
        ), f"Invalid runner name: {runner_backend_name}"
        cls._fused_funcs[key] = fused_func  # 注册融合函数

    @classmethod
    def get_fused_func(cls, dispatch_name: str, runner_name: str) -> Optional[Callable]:  # 获取已注册的融合函数
        key = (dispatch_name, runner_name)  # 构造查找键
        fused_func = cls._fused_funcs.get(key)  # 查找融合函数
        return fused_func  # 返回找到的函数或None


class PermuteMethodPool:  # 置换方法注册池
    _pre_permute_methods: dict[  # 前置换方法字典
        Tuple[DispatchOutputFormat, MoeRunnerBackend], Callable
    ] = {}
    _post_permute_methods: dict[  # 后置换方法字典
        Tuple[MoeRunnerBackend, CombineInputFormat], Callable
    ] = {}

    @classmethod
    def register_pre_permute(  # 注册前置换方法
        cls,
        dispatch_output_name: str,  # 调度输出格式名称
        runner_backend_name: str,  # Runner后端名称
        permute_func: Callable,  # 置换函数
    ):
        """
        Register a customized pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.  # 为给定的DispatchOutputFormat和MoeRunnerBackend注册自定义前置换函数。

        :param dispatch_output_name: The DispatchOutputFormat name.  # :param dispatch_output_name: DispatchOutputFormat名称。
        :param runner_backend_name: The MoeRunnerBackend name.  # :param runner_backend_name: MoeRunnerBackend名称。
        :param permute_func: The permute function to register.  # :param permute_func: 要注册的置换函数。
        """
        # TODO: check if registration is valid  # TODO: 检查注册是否有效
        key = (dispatch_output_name, runner_backend_name)  # 构造注册键
        if key in cls._pre_permute_methods:  # 如果键已存在
            raise ValueError(  # 抛出重复注册异常
                f"Pre-permute method for {dispatch_output_name} to {runner_backend_name} is already registered."
            )
        cls._pre_permute_methods[key] = permute_func  # 注册前置换方法

    @classmethod
    def register_post_permute(  # 注册后置换方法
        cls,
        runner_backend_name: str,  # Runner后端名称
        combine_input_name: str,  # Combine输入格式名称
        permute_func: Callable,  # 置换函数
    ):
        """
        Register a customized post-permute function for the given MoeRunnerBackend and CombineInputFormat.  # 为给定的MoeRunnerBackend和CombineInputFormat注册自定义后置换函数。

        :param runner_backend_name: The MoeRunnerBackend name.  # :param runner_backend_name: MoeRunnerBackend名称。
        :param combine_input_name: The CombineInputFormat name.  # :param combine_input_name: CombineInputFormat名称。
        :param permute_func: The permute function to register.  # :param permute_func: 要注册的置换函数。
        """
        # TODO: check if registration is valid  # TODO: 检查注册是否有效
        key = (runner_backend_name, combine_input_name)  # 构造注册键
        if key in cls._post_permute_methods:  # 如果键已存在
            raise ValueError(  # 抛出重复注册异常
                f"Post-permute method for {runner_backend_name} to {combine_input_name} is already registered."
            )
        cls._post_permute_methods[key] = permute_func  # 注册后置换方法

    @classmethod
    def get_pre_permute(  # 获取前置换方法
        cls,
        dispatch_output_format: DispatchOutputFormat,  # 调度输出格式
        runner_input_format: MoeRunnerBackend,  # Runner后端格式
    ) -> Callable:  # 返回置换函数
        """
        Retrieve the pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.  # 获取给定DispatchOutputFormat和MoeRunnerBackend的前置换函数。

        :param dispatch_output_format: The DispatchOutputFormat type.  # :param dispatch_output_format: DispatchOutputFormat类型。
        :param runner_input_format: The MoeRunnerBackend type.  # :param runner_input_format: MoeRunnerBackend类型。
        :return: The registered permute function or None if not found.  # :return: 已注册的置换函数，未找到则为None。
        """
        key = (dispatch_output_format, runner_input_format)  # 构造查找键
        pre_permute_func = cls._pre_permute_methods.get(key)  # 查找前置换方法
        assert (
            pre_permute_func is not None
        ), f"Pre-permute function for {dispatch_output_format} to {runner_input_format} is not registered"  # 断言方法已注册
        return pre_permute_func  # 返回前置换方法

    @classmethod
    def get_post_permute(  # 获取后置换方法
        cls,
        runner_output_format: MoeRunnerBackend,  # Runner输出格式
        combine_input_format: CombineInputFormat,  # Combine输入格式
    ) -> Callable:  # 返回置换函数
        """
        Retrieve the post-permute function for the given MoeRunnerBackend and CombineInputFormat.  # 获取给定MoeRunnerBackend和CombineInputFormat的后置换函数。

        :param runner_output_format: The MoeRunnerBackend type.  # :param runner_output_format: MoeRunnerBackend类型。
        :param combine_input_format: The CombineInputFormat type.  # :param combine_input_format: CombineInputFormat类型。
        :return: The registered permute function or None if not found.  # :return: 已注册的置换函数，未找到则为None。
        """
        key = (runner_output_format, combine_input_format)  # 构造查找键
        post_permute_func = cls._post_permute_methods.get(key)  # 查找后置换方法
        assert (
            post_permute_func is not None
        ), f"Post-permute function for {runner_output_format} to {combine_input_format} is not registered"  # 断言方法已注册
        return post_permute_func  # 返回后置换方法


def register_fused_func(  # 装饰器：注册融合函数
    a2a_backend_name: str,  # A2A后端名称
    runner_backend_name: str,  # Runner后端名称
) -> Callable:  # 返回装饰器函数
    """
    Decorator to register a fused function for the given DispatchOutputFormat and MoeRunnerBackend.  # 装饰器：为给定的DispatchOutputFormat和MoeRunnerBackend注册融合函数。

    :param a2a_backend_name: The A2A backend name.  # :param a2a_backend_name: A2A后端名称。
    :param runner_backend_name: The MoeRunnerBackend name.  # :param runner_backend_name: MoeRunnerBackend名称。
    :return: The decorator function.  # :return: 装饰器函数。
    """

    def decorator(fused_func: Callable):  # 装饰器内部函数
        FusedOpPool.register_fused_func(  # 在融合算子池中注册
            a2a_backend_name, runner_backend_name, fused_func
        )
        return fused_func  # 返回原函数

    return decorator  # 返回装饰器


def register_pre_permute(  # 装饰器：注册前置换函数
    dispatch_output_name: str,  # 调度输出格式名称
    runner_backend_name: str,  # Runner后端名称
) -> Callable:  # 返回装饰器函数
    """
    Decorator to register a pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.  # 装饰器：为给定的DispatchOutputFormat和MoeRunnerBackend注册前置换函数。

    :param dispatch_output_name: The DispatchOutputFormat name.  # :param dispatch_output_name: DispatchOutputFormat名称。
    :param runner_backend_name: The MoeRunnerBackend name.  # :param runner_backend_name: MoeRunnerBackend名称。
    :return: The decorator function.  # :return: 装饰器函数。
    """

    def decorator(  # 装饰器内部函数
        permute_func: Callable[
            [DispatchOutput, MoeQuantInfo, MoeRunnerConfig, dict], RunnerInput
        ],
    ) -> Callable:
        PermuteMethodPool.register_pre_permute(  # 在置换方法池中注册前置换方法
            dispatch_output_name, runner_backend_name, permute_func
        )
        return permute_func  # 返回原函数

    return decorator  # 返回装饰器


def register_post_permute(  # 装饰器：注册后置换函数
    runner_backend_name: str,  # Runner后端名称
    combine_input_name: str,  # Combine输入格式名称
) -> Callable:  # 返回装饰器函数
    """
    Decorator to register a post-permute function for the given MoeRunnerBackend and CombineInputFormat.  # 装饰器：为给定的MoeRunnerBackend和CombineInputFormat注册后置换函数。

    :param runner_backend_name: The MoeRunnerBackend name.  # :param runner_backend_name: MoeRunnerBackend名称。
    :param combine_input_name: The CombineInputFormat name.  # :param combine_input_name: CombineInputFormat名称。
    :return: The decorator function.  # :return: 装饰器函数。
    """

    def decorator(  # 装饰器内部函数
        permute_func: Callable[
            [RunnerOutput, MoeQuantInfo, MoeRunnerConfig, dict], CombineInput
        ],
    ) -> Callable:
        PermuteMethodPool.register_post_permute(  # 在置换方法池中注册后置换方法
            runner_backend_name, combine_input_name, permute_func
        )
        return permute_func  # 返回原函数

    return decorator  # 返回装饰器