# 启动函数日志和计时器模块
# 本模块提供启动阶段各组件的延迟记录功能
# 支持上下文管理器和装饰器两种方式记录启动延迟

"""
Records startup latency breakdown by context using gauge metrics in seconds
"""

import logging  # 导入日志模块
import time  # 导入时间模块
from contextlib import contextmanager  # 导入上下文管理器工具
from functools import wraps  # 导入装饰器工具
from typing import Any, Callable, Dict, Generator, Optional  # 导入类型提示

logger = logging.getLogger(__name__)  # 创建日志记录器

enable_startup_metrics = False  # 是否启用启动指标
STARTUP_LATENCY_SECONDS = None  # 启动延迟仪表，初始化为None
# Track maximum durations for each context
_max_durations: Dict[str, float] = {}  # 追踪每个上下文的最大持续时间


def enable_startup_timer():  # 启用启动计时器
    """Initialize startup latency metrics when metrics are enabled"""  # 在启用指标时初始化启动延迟指标
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import Gauge  # 导入Prometheus仪表

    global enable_startup_metrics, STARTUP_LATENCY_SECONDS  # 声明全局变量
    enable_startup_metrics = True  # 启用启动指标

    STARTUP_LATENCY_SECONDS = Gauge(  # 创建启动延迟仪表
        "sglang:startup_latency_breakdown_seconds_max",  # 指标名称
        "Startup latency breakdown in seconds by context, only records the maximum duration if the context is called multiple times.",  # 文档描述
        labelnames=["context"],  # 标签名称
        multiprocess_mode="mostrecent",  # 多进程模式
    )


def set_startup_metric(context: str, value: float, should_log: bool = True):  # 设置启动指标
    """Set the startup metric for a given context"""  # 设置指定上下文的启动指标
    if should_log:  # 如果需要记录日志
        logger.info(f"Setting startup metric: {context} took {value:.3f}s")  # 记录信息

    if not enable_startup_metrics:  # 如果未启用启动指标
        return  # 直接返回
    current_max = _max_durations.get(context, 0.0)  # 获取当前最大值
    if value > current_max:  # 如果新值更大
        _max_durations[context] = value  # 更新最大值
        STARTUP_LATENCY_SECONDS.labels(context=context).set(value)  # 设置仪表值


def reset_startup_timers():  # 重置所有记录的最大持续时间
    """Reset all recorded maximum durations. Useful for testing or reinitialization."""  # 重置所有记录的最大持续时间，用于测试或重新初始化
    global _max_durations  # 声明全局变量
    _max_durations.clear()  # 清空字典


def get_max_duration(context: str) -> Optional[float]:  # 获取指定上下文的最大持续时间
    """Get the maximum recorded duration for a context name."""  # 获取指定上下文名称的最大记录持续时间
    return _max_durations.get(context)  # 返回最大持续时间


@contextmanager
def startup_timer(name: str, log_only: bool = False) -> Generator[None, None, None]:  # 启动计时器上下文管理器
    """
    Context manager to measure startup latency for arbitrary code blocks.
    Only records the maximum duration if the context is called multiple times.

    Usage:
        with startup_timer("model_loading"):
            # model loading code
            model = load_model()

        with startup_timer("memory_allocation"):
            # memory setup code
            allocate_memory()
    """
    start_time = time.monotonic()  # 记录开始时间
    try:  # 尝试执行
        yield  # 执行代码块
    finally:  # 最终
        duration_seconds = time.monotonic() - start_time  # 计算持续时间

        # Track the maximum duration for this context name
        current_max = _max_durations.get(name, 0.0)  # 获取当前最大值
        is_new_max = duration_seconds > current_max  # 判断是否为新最大值

        if is_new_max:  # 如果是新最大值
            _max_durations[name] = duration_seconds  # 更新最大值

            # Only update Prometheus gauge if this is a new maximum
            if enable_startup_metrics and not log_only:  # 如果启用指标且非仅日志
                STARTUP_LATENCY_SECONDS.labels(context=name).set(duration_seconds)  # 设置仪表值

        # Log with indication if this was a new max
        logger.info(f"Startup timing: {name} took {duration_seconds:.3f}s")  # 记录时间信息


def time_startup_latency(  # 启动延迟计时装饰器
    func: Callable = None, name: Optional[str] = None, log_only: bool = False  # 被装饰函数、名称、是否仅日志
) -> Callable[..., Any]:
    """
    A decorator to measure startup context latency and record it in seconds.
    Only records the maximum duration if the context is called multiple times.

    Usage:
        @time_startup_latency
        def load_model():
            # model loading code

        @time_startup_latency(name="custom_init")
        def initialize_something():
            # initialization code

        @time_startup_latency(name="debug_only", log_only=True)
        def debug_function():
            # This will only log, not record to Prometheus
    """

    def measure(func: Callable[..., Any]) -> Callable[..., Any]:  # 测量函数包装器
        nonlocal name  # 使用外部name变量

        name = name or func.__name__  # 使用函数名作为名称

        @wraps(func)  # 保留函数元信息
        def wrapper(*args, **kwargs):  # 包装器函数
            start_time = time.monotonic()  # 记录开始时间
            try:  # 尝试执行
                result = func(*args, **kwargs)  # 执行函数
                return result  # 返回结果
            finally:  # 最终
                duration_seconds = time.monotonic() - start_time  # 计算持续时间

                # Track the maximum duration for this context name
                current_max = _max_durations.get(name, 0.0)  # 获取当前最大值
                is_new_max = duration_seconds > current_max  # 判断是否为新最大值

                if is_new_max:  # 如果是新最大值
                    _max_durations[name] = duration_seconds  # 更新最大值

                    # Only update Prometheus gauge if this is a new maximum
                    if enable_startup_metrics and not log_only:  # 如果启用指标且非仅日志
                        STARTUP_LATENCY_SECONDS.labels(context=name).set(  # 设置仪表值
                            duration_seconds
                        )

                # Log the timing
                logger.info(f"Startup timing: {name} took {duration_seconds:.3f}s")  # 记录时间信息

        return wrapper  # 返回包装器

    if func:  # 如果直接装饰函数
        return measure(func)  # 返回测量结果
    else:  # 如果带参数装饰
        return measure  # 返回测量函数
