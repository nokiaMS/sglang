# 函数延迟计时器模块
# 本模块提供函数执行延迟的记录功能
# 支持同步和异步函数的延迟装饰器

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Records the latency of some functions
"""

import asyncio  # 导入异步IO模块
import time  # 导入时间模块
from functools import wraps  # 导入装饰器工具
from typing import Any, Callable, Optional  # 导入类型提示

from sglang.srt.observability.utils import exponential_buckets  # 导入指数桶工具

enable_metrics = False  # 是否启用指标记录


def enable_func_timer():  # 启用函数计时器
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import Histogram  # 导入Prometheus直方图

    global enable_metrics, FUNC_LATENCY  # 声明全局变量
    enable_metrics = True  # 启用指标

    FUNC_LATENCY = Histogram(  # 创建函数延迟直方图
        "sglang:func_latency_seconds",  # 指标名称
        "Function latency in seconds",  # 文档描述
        # captures latency in range [50ms - ~50s]
        buckets=exponential_buckets(start=0.05, width=1.5, length=18),  # 指数桶配置，覆盖50ms到50s
        labelnames=["name"],  # 标签名称
    )


FUNC_LATENCY = None  # 函数延迟直方图，初始化为None


def time_func_latency(  # 函数延迟计时装饰器
    func: Callable = None, name: Optional[str] = None  # 被装饰函数和名称
) -> Callable[..., Any]:
    """
    A decorator to observe the latency of a function's execution. Supports both sync and async functions.

    NOTE: We use our own implementation of a timer decorator since prometheus_client does not support async
    context manager yet.

    Overhead: The overhead introduced here in case of an async function could likely be because of `await` introduced
    which will return in another coroutine object creation and under heavy load could see longer wall time
    (scheduling delays due to introduction of another awaitable).
    """

    def measure(func: Callable[..., Any]) -> Callable[..., Any]:  # 测量函数包装器
        nonlocal name  # 使用外部name变量

        name = name or func.__name__  # 使用函数名作为名称

        @wraps(func)  # 保留函数元信息
        async def async_wrapper(*args, **kwargs):  # 异步函数包装器
            if not enable_metrics:  # 如果未启用指标
                return await func(*args, **kwargs)  # 直接执行函数

            metric = FUNC_LATENCY  # 获取指标对象
            start = time.monotonic()  # 记录开始时间
            ret = func(*args, **kwargs)  # 调用函数（可能返回协程）
            if isinstance(ret, asyncio.Future) or asyncio.iscoroutine(ret):  # 如果是异步对象
                try:  # 尝试执行
                    ret = await ret  # 等待结果
                finally:  # 无论是否异常
                    metric.labels(name=name).observe(time.monotonic() - start)  # 记录延迟
            return ret  # 返回结果

        @wraps(func)  # 保留函数元信息
        def sync_wrapper(*args, **kwargs):  # 同步函数包装器
            if not enable_metrics:  # 如果未启用指标
                return func(*args, **kwargs)  # 直接执行函数

            metric = FUNC_LATENCY  # 获取指标对象
            start = time.monotonic()  # 记录开始时间
            try:  # 尝试执行
                ret = func(*args, **kwargs)  # 调用函数
            finally:  # 无论是否异常
                metric.labels(name=name).observe(time.monotonic() - start)  # 记录延迟
            return ret  # 返回结果

        if asyncio.iscoroutinefunction(func):  # 如果是异步函数
            return async_wrapper  # 返回异步包装器
        return sync_wrapper  # 返回同步包装器

    if func:  # 如果直接装饰函数
        return measure(func)  # 返回测量结果
    else:  # 如果带参数装饰
        return measure  # 返回测量函数
