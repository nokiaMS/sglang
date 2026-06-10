# 多CUDA流并行执行工具模块
# 改编自trtllm，提供在两个CUDA流上并行运行函数的机制
# 仅在CUDA图开启时启用多流，因为切换流有额外的主机开销
# 主要用于低延迟场景
# Adapted from trtllm.

import threading  # 导入线程模块
from contextlib import contextmanager  # 导入上下文管理器装饰器
from typing import Any, Callable, Optional  # 导入类型提示

import torch  # 导入PyTorch


class do_multi_stream_local(threading.local):  # 线程局部存储类，保存多流启用状态

    def __init__(self):  # 初始化线程局部变量
        self.do_multi_stream = False  # 默认禁用多流


_local = do_multi_stream_local()  # 创建线程局部存储实例


def set_do_multi_stream(enable: bool):  # 设置是否启用多流
    _local.do_multi_stream = enable


def do_multi_stream() -> bool:  # 查询是否启用多流
    return _local.do_multi_stream


@contextmanager
def with_multi_stream(enable: bool):  # 临时启用或禁用多流的上下文管理器
    prev_do_multi_stream = _local.do_multi_stream  # 保存之前的状态
    set_do_multi_stream(enable)  # 设置新状态
    try:
        yield  # 执行上下文中的代码
    finally:
        set_do_multi_stream(prev_do_multi_stream)  # 恢复之前的状态


def maybe_execute_in_parallel(  # 在两个CUDA流上并行运行两个函数
    fn0: Callable,
    fn1: Callable,
    events: list[torch.cuda.Event],
    aux_stream: Optional[torch.cuda.Stream] = None,
) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        events (list[torch.cuda.Event]): cuda events for callables
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    multi_stream = do_multi_stream() and aux_stream is not None  # 判断是否启用多流

    if multi_stream:  # 如果启用多流
        events[0].record()  # 在默认流上记录事件0
        result0 = fn0()  # 在默认流上执行fn0

        with torch.cuda.stream(aux_stream):  # 切换到辅助流
            events[0].wait()  # 等待事件0完成
            result1 = fn1()  # 在辅助流上执行fn1
            events[1].record()  # 在辅助流上记录事件1
        events[1].wait()  # 在默认流上等待事件1完成
    else:  # 如果不启用多流
        result0 = fn0()  # 顺序执行fn0
        result1 = fn1()  # 顺序执行fn1
    return (result0, result1)  # 返回两个函数的结果
