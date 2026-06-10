# 本文件实现了自定义算子(Custom Op)的注册机制，用于将 Python 函数注册为
# PyTorch 自定义算子，以支持 torch.compile 兼容性。
# 主要提供两种注册方式：
# - register_custom_op：装饰器方式，用于注册内部自定义算子
# - register_custom_op_from_extern：函数方式，用于包装外部库函数（如 flashinfer）
# 注册时会自动生成 fake implementation 用于 torch.compile 的形状/类型推导。

from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, TypeVar, Union, overload

import torch
import torch.library

from sglang.kernel_api_logging import debug_torch_op

F = TypeVar("F", bound=Callable)


@overload
def register_custom_op(
    fn: F,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    eager: bool = True,
) -> F: ...


@overload
def register_custom_op(
    fn: F,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    fake_impl: Optional[Callable],
    eager: bool = True,
) -> F: ...


@overload
def register_custom_op(
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    eager: bool = True,
) -> Callable[[F], F]: ...


@overload
def register_custom_op(
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    fake_impl: Optional[Callable],
    eager: bool = True,
) -> Callable[[F], F]: ...


# Real implementation
def register_custom_op(
    fn: Optional[Callable] = None,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    eager: bool = True,
    **extra_kwargs,
) -> Any:
    """
    A decorator to register a custom operator.

    注册自定义算子的装饰器。支持就地(inplace)算子和有输出的算子。

    Example usage:
    ```python
    # inplace operator, out_shape is None by default
    @register_custom_op(mutates_args=["x"])
    def add_1_(x: torch.Tensor) -> None:
        x.add_(1)

    # operator with output, out_shape indicates the position of output
    @register_custom_op(mutates_args=["x"], out_shape=0)
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.add_(y)
    ```

    :param fn: The function to be registered as a custom operator.
               If None, return a decorator.
               要注册为自定义算子的函数，为 None 时返回装饰器
    :type fn: Callable
    :param op_name: The name of the operator. If None, use the function name
                    算子名称，为 None 时使用函数名
    :type op_name: Optional[str]
    :param mutates_args: A list of argument names that are mutated in-place.
                         就地修改的参数名列表
    :type mutates_args: List[str]
    :param out_shape: The position (int for positional, str for keyword) of the output-shape tensor.
                      It is used to generate a fake implementation for torch.compile compatibility.
                      If the operator is inplace and has no output, set to None.
                      输出形状对应的参数位置（整数表示位置参数，字符串表示关键字参数），
                      用于自动生成 fake implementation；就地算子无输出时设为 None
    :type out_shape: Optional[List[Union[int, str]]]
    :param fake_impl: A fake implementation for the operator.
                      Only one of `out_shape` or `fake_impl` should be provided.
                      自定义的 fake implementation，与 out_shape 二选一
    :type fake_impl: Optional[Callable]
    :param eager: Whether to register the operator eagerly.
                  If False, the registration will be deferred until the first call.
                  If you met any issue with torch.compile, try to set eager=True.
                  Currently, to avoid misuse, we set eager=True by default.
                  是否立即注册算子；为 False 时延迟到首次调用时注册，
                  遇到 torch.compile 问题时可尝试设为 True
    :type eager: bool
    :return: The registered JIT custom operator, or a decorator.
             NOTE: the real register will occur at the first call of the function.
             注册后的自定义算子可调用对象或装饰器；注意实际注册在首次调用时发生
    :rtype: Callable
    """
    # 检查额外的关键字参数是否合法
    extra_kwarg_keys = set(extra_kwargs.keys())
    expected_kwarg_keys = set({"out_shape", "fake_impl"})
    assert (
        expected_kwarg_keys >= extra_kwarg_keys
    ), f"Unexpected extra kwargs: {extra_kwarg_keys - expected_kwarg_keys}"

    has_out_shape = "out_shape" in extra_kwargs
    has_fake_impl = "fake_impl" in extra_kwargs
    assert not (
        has_out_shape and has_fake_impl
    ), "Only one of `out_shape` or `fake_impl` should be provided."
    # Assume inplace if neither out_shape nor fake_impl is provided
    # 如果既没有 out_shape 也没有 fake_impl，假定为就地算子
    if not (has_out_shape or has_fake_impl):
        extra_kwargs["out_shape"] = None

    def decorator(op_func: Callable) -> Callable:
        wrapper = CustomOpWrapper(
            op_name=op_name or op_func.__name__,
            op_func=op_func,
            mutates_args=mutates_args or [],
            **extra_kwargs,
        )
        # eager=True 时直接返回 real_impl（首次调用时注册），否则返回 wrapper（延迟注册）
        return wrapper.real_impl if eager else wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


class CustomOpWrapper:
    """自定义算子包装器，封装算子函数及其注册信息。
    支持延迟注册：在首次调用时才向 PyTorch 注册自定义算子。"""

    def __init__(
        self,
        op_name: str,
        op_func: Callable,
        mutates_args: List[str],
        **extra_kwargs,
    ):
        self.op_name = op_name  # 算子名称
        self.op_func = op_func  # 原始算子函数
        self.mutates_args = mutates_args  # 就地修改的参数名列表
        self.extra_kwargs = extra_kwargs  # 额外参数（out_shape 或 fake_impl）
        self._impl: Optional[Callable] = None  # 缓存的已注册实现

    def __call__(self, *args, **kwargs):
        return self.real_impl(*args, **kwargs)

    @property
    def real_impl(self) -> Callable:
        """获取实际的算子实现。首次访问时执行注册，包括向 torch.ops.sglang 注册
        自定义算子和生成 fake implementation。"""
        if self._impl is None:
            if not hasattr(torch.ops.sglang, self.op_name):
                from sglang.srt.utils.common import direct_register_custom_op

                # NOTE(dark): if torch compile fail here, mark the decorator as eager
                # lazy registration does not work with torch compile
                # 注意：如果 torch.compile 失败，将装饰器标记为 eager；
                # 延迟注册与 torch.compile 不兼容
                direct_register_custom_op(
                    op_name=self.op_name,
                    op_func=self.op_func,
                    mutates_args=self.mutates_args,
                    fake_impl=self.fake_impl,
                )
            # 使用 debug_torch_op 包装以支持调试日志
            self._impl = debug_torch_op(self.op_func, self.op_name)
            assert self._impl is not None
        return self._impl

    @property
    def fake_impl(self) -> Callable:
        """生成或返回 fake implementation，用于 torch.compile 的形状/类型推导。
        如果提供了自定义 fake_impl 则直接返回；否则根据 out_shape 自动生成。"""
        if "fake_impl" in self.extra_kwargs:
            return self.extra_kwargs["fake_impl"]
        assert "out_shape" in self.extra_kwargs
        signature = inspect.signature(self.op_func)
        out_shape = self.extra_kwargs["out_shape"]
        # check out_shape in signature

        def fake_impl(*args, **kwargs):
            # 就地算子无输出，返回 None
            if out_shape is None:
                return None
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                # 根据 out_shape 位置创建形状相同的空张量
                return torch.empty_like(
                    bound.args[out_shape]
                    if isinstance(out_shape, int)
                    else bound.arguments[out_shape]
                )
            except (IndexError, KeyError):
                raise RuntimeError(
                    f"Cannot find output argument at position `{out_shape}` for "
                    f"custom operator `{self.op_name}` with signature `{signature}`."
                )

        return fake_impl


def register_custom_op_from_extern(
    fn: Callable,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    out_dtype: Optional[torch.dtype] = None,
    fake_impl: Optional[Callable] = None,
    computed_args: Optional[dict] = None,
) -> Callable:
    """Wrap an external library function as a custom op for torch.compile compatibility.

    Use this to wrap functions from external libraries (e.g. flashinfer kernels) that
    perform operations incompatible with torch.compile/dynamo tracing, such as JIT
    compilation, file I/O, or dynamic module loading.

    The wrapped function becomes an opaque node in the compiled graph. Dynamo will
    not trace inside it, avoiding tracing failures. A fake implementation is used
    for shape/dtype propagation during compilation.

    The external function must have type annotations compatible with
    ``torch.library.infer_schema`` (``torch.Tensor``, ``int``, ``float``, ``bool``,
    ``Optional[torch.Tensor]``, etc.).

    This function is idempotent: calling it multiple times with the same ``op_name``
    (or ``fn.__name__``) safely skips re-registration.

    将外部库函数包装为自定义算子，使其兼容 torch.compile。
    包装后的函数在编译图中成为不透明节点，Dynamo 不会追踪其内部实现，
    从而避免因 JIT 编译、文件 I/O 或动态模块加载等操作导致的追踪失败。
    此函数是幂等的：使用相同 op_name 多次调用会安全地跳过重复注册。

    Example usage::

        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

        trtllm_fp8_block_scale_moe = register_custom_op_from_extern(
            trtllm_fp8_block_scale_moe,
            out_shape="hidden_states",
            out_dtype=torch.bfloat16,
            computed_args={
                "tune_max_num_tokens": lambda hidden_states, **kw: next_power_of_2(
                    hidden_states.shape[0]
                ),
            },
        )

    :param fn: The external function to wrap.
               要包装的外部函数
    :param op_name: The name of the custom operator.
                    Defaults to ``fn.__name__``.
                    自定义算子名称，默认为函数名
    :param mutates_args: A list of argument names that are mutated in-place.
                         Defaults to ``[]``.
                         就地修改的参数名列表
    :param out_shape: The position (int) or name (str) of the argument whose shape
                      matches the output tensor. Used to auto-generate a fake
                      implementation. Set to ``None`` for inplace-only operators.
                      输出形状对应的参数位置或名称，用于自动生成 fake implementation；
                      就地算子设为 None
    :param out_dtype: Override the output dtype in the fake implementation.
                      If ``None``, ``torch.empty_like`` is used (same dtype as the
                      reference tensor). Useful when the output dtype differs from
                      the input (e.g. fp8 input -> bf16 output).
                      覆盖 fake implementation 中的输出数据类型；当输出与输入数据类型
                      不同时使用（如 fp8 输入 -> bf16 输出）
    :param fake_impl: A custom fake implementation for shape/dtype propagation.
                      Only one of ``out_shape`` or ``fake_impl`` should be provided.
                      自定义 fake implementation，与 out_shape 二选一
    :param computed_args: A dict mapping argument names to callables. These arguments
                          are excluded from the custom op schema and computed inside
                          the op body at runtime. Each callable receives the other
                          arguments as keyword args and returns the computed value.
                          Use this for arguments that vary dynamically (e.g.
                          ``tune_max_num_tokens``) to avoid torch.compile recompilation.
                          参数名到计算函数的映射；这些参数从算子 schema 中排除，
                          在运行时由计算函数生成；用于动态变化的参数以避免重编译
    :return: The registered custom op callable (``torch.ops.sglang.<op_name>``).
             注册后的自定义算子可调用对象
    """
    name = op_name or fn.__name__
    computed_args = computed_args or {}

    assert not (
        out_shape is not None and fake_impl is not None
    ), "Only one of `out_shape` or `fake_impl` should be provided."

    # If computed_args specified, create a wrapper with a reduced signature
    # that computes the excluded args inside the op body.
    # 如果指定了 computed_args，创建一个精简签名的包装函数，
    # 在算子体内计算被排除的参数
    if computed_args:
        original_fn = fn
        original_sig = inspect.signature(fn)

        # Build new signature excluding computed args
        # 构建排除 computed_args 后的新函数签名
        new_params = [
            p
            for param_name, p in original_sig.parameters.items()
            if param_name not in computed_args
        ]
        new_sig = original_sig.replace(parameters=new_params)

        def wrapper(*args, **kwargs):
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            # Compute excluded args from the bound arguments
            # 根据已绑定的参数计算被排除的参数值
            for arg_name, compute_fn in computed_args.items():
                bound.arguments[arg_name] = compute_fn(**bound.arguments)
            return original_fn(**bound.arguments)

        # 保留原始函数的元信息
        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        wrapper.__module__ = fn.__module__
        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        # Build annotations without computed args, preserving return type
        # 构建排除 computed_args 后的类型注解，保留返回类型
        wrapper.__annotations__ = {
            k: v
            for k, v in getattr(fn, "__annotations__", {}).items()
            if k not in computed_args
        }
        fn = wrapper

    # Generate fake_impl from out_shape if needed
    # 如果需要，根据 out_shape 生成 fake implementation
    fake_sig = inspect.signature(fn)
    if fake_impl is None and out_shape is not None:

        def _fake_impl(*args, **kwargs):
            bound = fake_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                ref = (
                    bound.args[out_shape]
                    if isinstance(out_shape, int)
                    else bound.arguments[out_shape]
                )
            except (IndexError, KeyError):
                raise RuntimeError(
                    f"Cannot find output argument at position `{out_shape}` for "
                    f"external function `{name}` with signature `{fake_sig}`."
                )
            # 如果指定了 out_dtype，使用指定类型；否则使用与参考张量相同的类型
            if out_dtype is not None:
                return torch.empty(ref.shape, dtype=out_dtype, device=ref.device)
            return torch.empty_like(ref)

        fake_impl = _fake_impl
    elif fake_impl is None:
        # 就地算子无输出，fake implementation 返回 None
        fake_impl = lambda *args, **kwargs: None

    from sglang.srt.utils.common import direct_register_custom_op

    # 向 PyTorch 注册自定义算子
    direct_register_custom_op(
        op_name=name,
        op_func=fn,
        mutates_args=mutates_args or [],
        fake_impl=fake_impl,
    )

    return debug_torch_op(fn, name)
