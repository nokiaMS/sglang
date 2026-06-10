# 该文件实现了SGLang请求追踪功能，基于OpenTelemetry框架
# 提供了分布式追踪的初始化、跨度管理、线程上下文管理等功能
# 支持将追踪数据导出到OTLP端点（gRPC或HTTP协议）
# 包含自定义ID生成器以避免多TP调度进程间的trace ID冲突

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
"""package for sglang requests tracing"""  # SGLang请求追踪包

from __future__ import annotations  # 启用延迟注解评估

import logging  # 日志模块
import os  # 操作系统接口
import random  # 随机数生成
import threading  # 线程支持
import time  # 时间相关
import uuid  # UUID生成
from dataclasses import dataclass  # 数据类装饰器
from typing import Any, Dict, List, Mapping, Optional  # 类型注解

from sglang.srt.utils import get_int_env_var  # 获取整数环境变量的工具函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
opentelemetry_imported = False  # OpenTelemetry是否成功导入的标志
opentelemetry_initialized = False  # OpenTelemetry是否已初始化的标志
_trace_context_propagator = None  # 追踪上下文传播器
tracer: Optional[trace.Tracer] = None  # 全局追踪器实例

global_trace_level = get_int_env_var("SGLANG_TRACE_LEVEL", 3)  # 全局追踪级别，默认为3

TRACE_HEADERS = ["traceparent", "tracestate"]  # 追踪相关的HTTP头名称列表

try:
    from opentelemetry import context, propagate, trace  # 导入OpenTelemetry核心模块
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,  # gRPC协议的OTLP跨度导出器
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,  # HTTP协议的OTLP跨度导出器
    )
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,  # OTLP导出协议的环境变量名
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # 资源相关
    from opentelemetry.sdk.trace import TracerProvider, id_generator  # 追踪提供者和ID生成器
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # 批量跨度处理器
    from opentelemetry.trace import Status, StatusCode  # 跨度状态相关
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,  # 追踪上下文文本映射传播器
    )

    _trace_context_propagator = TraceContextTextMapPropagator()  # 初始化追踪上下文传播器

    opentelemetry_imported = True  # 标记OpenTelemetry已成功导入
except ImportError:

    class id_generator:  # 未安装OpenTelemetry时的占位ID生成器
        class IdGenerator:  # 占位ID生成器类
            pass

    logger.debug("opentelemetry package is not installed, tracing disabled")  # 记录调试信息


def extract_trace_headers(headers: Mapping[str, str]) -> Optional[Dict]:
    """从HTTP头中提取追踪相关的头部信息"""
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}  # 返回包含追踪头的字典


def set_global_trace_level(level: int):
    """设置全局追踪级别"""
    global global_trace_level  # 声明使用全局变量
    global_trace_level = level  # 更新全局追踪级别


@dataclass
class TraceThreadInfo:  # 追踪线程信息数据类
    host_id: str  # 主机标识
    pid: int  # 进程ID
    thread_label: str  # 线程标签
    tp_rank: int  # 张量并行排名
    dp_rank: int  # 数据并行排名
    pp_rank: int  # 流水线并行排名


@dataclass
class TraceEvent:  # 追踪事件数据类
    event_name: str  # 事件名称
    ts: int  # 时间戳（纳秒）
    attrs: Dict[str, Any]  # 事件属性


@dataclass
class TraceSliceContext:  # 追踪切片上下文数据类
    slice_name: str  # 切片名称
    start_time_ns: int  # 开始时间（纳秒）
    end_time_ns: Optional[int] = None  # 结束时间（纳秒），可选
    span: Optional[trace.span.Span] = None  # 关联的跨度对象，可选
    level: int = 1  # 追踪级别，默认为1
    attrs: Optional[Dict[str, Any]] = None  # 切片属性，可选
    events: Optional[List[TraceEvent]] = None  # 事件列表，可选


@dataclass
class TraceThreadContext:  # 追踪线程上下文数据类
    thread_info: TraceThreadInfo  # 线程信息
    cur_slice_stack: Optional[List[TraceSliceContext]] = None  # 当前切片栈，可选
    thread_span: Optional[trace.span.Span] = None  # 线程跨度，可选


class TraceCustomIdGenerator(id_generator.IdGenerator):
    """
    The default IdGenerator may produce duplicate trace IDs across multiple TP scheduler processes,
    hence a custom IdGenerator is implemented.
    """  # 默认ID生成器可能在多个TP调度进程间产生重复的trace ID，因此实现了自定义ID生成器

    def __init__(self):  # 初始化自定义ID生成器
        super().__init__()  # 调用父类初始化
        self.local_random = random.Random()  # 创建本地随机数生成器
        self.local_random.seed(time.time())  # 使用当前时间作为随机种子

    def generate_trace_id(self) -> int:  # 生成64位随机trace ID
        return self.local_random.getrandbits(64)  # 返回64位随机整数

    def generate_span_id(self) -> int:  # 生成64位随机span ID
        return self.local_random.getrandbits(64)  # 返回64位随机整数


# global variables  # 全局变量
threads_info: Dict[int, TraceThreadInfo] = {}  # 线程信息字典，键为进程ID

get_cur_time_ns = lambda: int(time.time() * 1e9)  # 获取当前纳秒时间戳的lambda（备用实现）
if hasattr(time, "time_ns"):  # 如果系统支持time_ns
    get_cur_time_ns = lambda: int(time.time_ns())  # 使用更高精度的time_ns函数


def __get_host_id() -> str:
    """
    In distributed tracing systems, obtain a unique node identifier
    and inject it into all subsequently generated spans
    to prevent PID conflicts between threads on different nodes.
    """  # 在分布式追踪系统中，获取唯一节点标识符并注入到所有后续生成的跨度中，以防止不同节点上线程间的PID冲突
    if os.path.exists("/etc/machine-id"):  # 如果存在machine-id文件
        try:
            with open("/etc/machine-id", "r") as f:  # 以读模式打开文件
                return f.read().strip()  # 读取并去除首尾空白后返回
        except:
            pass  # 忽略异常

    mac = uuid.getnode()  # 获取MAC地址
    if mac != 0:  # 如果MAC地址有效
        return uuid.UUID(int=mac).hex  # 将MAC转换为UUID的十六进制形式返回

    return "unknown"  # 无法获取时返回"unknown"


# Should be called by each tracked process.  # 应由每个被追踪的进程调用
def process_tracing_init(otlp_endpoint, server_name):  # 初始化进程追踪
    """初始化OpenTelemetry追踪，配置导出器和处理器"""
    global opentelemetry_initialized  # 声明使用全局变量
    global get_cur_time_ns  # 声明使用全局变量
    global tracer  # 声明使用全局变量
    if not opentelemetry_imported:  # 如果OpenTelemetry未导入
        opentelemetry_initialized = False  # 标记为未初始化
        raise RuntimeError(
            "opentelemetry package is not installed!!! Please not enable tracing or install opentelemetry"  # 抛出运行时错误
        )

    try:
        resource = Resource.create(  # 创建资源
            attributes={
                SERVICE_NAME: server_name,  # 服务名称属性
            }
        )
        tracer_provider = TracerProvider(  # 创建追踪提供者
            resource=resource, id_generator=TraceCustomIdGenerator()  # 使用自定义ID生成器
        )

        schedule_delay_millis = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", 500  # 批量导出调度延迟，默认500毫秒
        )
        max_export_batch_size = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", 64  # 最大导出批量大小，默认64
        )

        processor = BatchSpanProcessor(  # 创建批量跨度处理器
            span_exporter=get_otlp_span_exporter(otlp_endpoint),  # 获取OTLP跨度导出器
            schedule_delay_millis=schedule_delay_millis,  # 调度延迟
            max_export_batch_size=max_export_batch_size,  # 最大批量大小
        )
        tracer_provider.add_span_processor(processor)  # 添加跨度处理器
        trace.set_tracer_provider(tracer_provider)  # 设置全局追踪提供者
    except Exception as e:
        opentelemetry_initialized = False  # 标记为未初始化
        raise RuntimeError(
            f"initialize opentelemetry error:{e}. Please set correct otlp endpoint."  # 抛出运行时错误
        )

    opentelemetry_initialized = True  # 标记为已初始化
    tracer = trace.get_tracer("sglang server")  # 获取名为"sglang server"的追踪器


def get_global_tracing_enabled():  # 检查全局追踪是否已启用
    """检查全局追踪是否已启用"""
    return opentelemetry_initialized  # 返回OpenTelemetry初始化状态


def get_otlp_span_exporter(endpoint):  # 根据协议获取OTLP跨度导出器
    """根据配置的协议类型获取相应的OTLP跨度导出器"""
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")  # 从环境变量获取协议，默认grpc
    supported_protocols = {"grpc", "http/protobuf"}  # 支持的协议集合

    if protocol not in supported_protocols:  # 如果协议不受支持
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' configured. "
            f"Supported protocols are: {', '.join(sorted(supported_protocols))}"  # 抛出值错误
        )

    if protocol == "grpc":  # 如果是gRPC协议
        return GRPCSpanExporter(endpoint=endpoint, insecure=True)  # 返回gRPC导出器
    elif protocol == "http/protobuf":  # 如果是HTTP协议
        return HTTPSpanExporter(endpoint=endpoint)  # 返回HTTP导出器


# Should be called by each tracked thread.  # 应由每个被追踪的线程调用
def trace_set_thread_info(
    thread_label: str,  # 线程标签
    tp_rank: Optional[int] = None,  # 张量并行排名，可选
    dp_rank: Optional[int] = None,  # 数据并行排名，可选
    pp_rank: Optional[int] = None,  # 流水线并行排名，可选
):  # 设置追踪线程信息
    """为当前线程设置追踪信息，包括标签和并行排名"""
    if not opentelemetry_initialized:  # 如果OpenTelemetry未初始化
        return  # 直接返回

    pid = threading.get_native_id()  # 获取原生线程ID
    if pid in threads_info:  # 如果该线程已有信息
        return  # 直接返回

    threads_info[pid] = TraceThreadInfo(  # 存储线程信息
        host_id=__get_host_id(),  # 主机标识
        pid=pid,  # 进程ID
        thread_label=thread_label,  # 线程标签
        tp_rank=tp_rank,  # 张量并行排名
        dp_rank=dp_rank,  # 数据并行排名
        pp_rank=pp_rank,  # 流水线并行排名
    )


class TraceReqContext:  # 追踪请求上下文类
    def __init__(
        self,
        rid,  # 请求ID
        bootstrap_room=None,  # 引导房间号，可选
        role="unified",  # 角色，默认为"unified"
        module_name="",  # 模块名称，默认为空
        external_trace_header: Optional[Dict[str, str]] = None,  # 外部追踪头，可选
    ):  # 初始化追踪请求上下文
        """初始化追踪请求上下文，管理请求级别的追踪状态"""
        self.rid: str = str(rid)  # 请求ID，转为字符串
        self.trace_level = global_trace_level  # 追踪级别
        self.tracing_enable: bool = opentelemetry_initialized and self.trace_level > 0  # 追踪是否启用

        if not self.tracing_enable:  # 如果追踪未启用
            return  # 直接返回

        self.start_time_ns: Optional[int] = None  # 开始时间（纳秒），可选
        self.thread_context: Optional[TraceThreadContext] = None  # 线程上下文，可选
        self.bootstrap_room: Optional[int] = bootstrap_room  # 引导房间号
        self.role: str = role  # 角色
        self.module_name = module_name  # 模块名称

        # Indicates whether this instance is a replica from the main process.  # 指示此实例是否是主进程的副本
        # When True, root_span is None and only root_span_context is preserved.  # 为True时，root_span为None，仅保留root_span_context
        self.is_copy: bool = False  # 是否为副本
        self.root_span: Optional[trace.span.Span] = None  # 根跨度，可选
        self.root_span_context: Optional[context.Context] = None  # 根跨度上下文，可选
        # Record the most recently completed span as the previous span for the next span to be created.  # 记录最近完成的跨度作为下一个要创建的跨度的前一个跨度
        self.last_span_context: Optional[trace.span.SpanContext] = None  # 上一个跨度上下文，可选
        self.external_trace_header: Optional[Dict[str, str]] = external_trace_header  # 外部追踪头

        self.events_cache: List[TraceEvent] = []  # 事件缓存列表

        self.pid: int = threading.get_native_id()  # 当前线程的原生ID

    def is_tracing_enabled(self) -> bool:  # 检查追踪是否已启用
        """检查当前请求的追踪是否已启用"""
        return self.tracing_enable  # 返回追踪启用状态

    def __create_thread_context(self, ts: int):  # 创建线程上下文
        """创建并初始化线程上下文和线程跨度"""
        if self.pid not in threads_info:  # 如果当前线程没有追踪信息
            trace_set_thread_info("unknown")  # 使用"unknown"标签设置线程信息

        thread_info = threads_info[self.pid]  # 获取线程信息
        thread_context = TraceThreadContext(  # 创建线程上下文
            thread_info=thread_info,  # 线程信息
            cur_slice_stack=[],  # 空的切片栈
        )

        thread_name = f"{thread_info.thread_label}"  # 线程名称基础部分
        if thread_info.tp_rank is not None:  # 如果有TP排名
            thread_name += f" [TP {thread_info.tp_rank}] "  # 添加TP排名
        if thread_info.pp_rank is not None:  # 如果有PP排名
            thread_name += f" [PP {thread_info.pp_rank}] "  # 添加PP排名
        if thread_info.dp_rank is not None:  # 如果有DP排名
            thread_name += f" [DP {thread_info.dp_rank}] "  # 添加DP排名
        thread_name += f"(host:{thread_info.host_id[:8]} | pid:{self.pid})"  # 添加主机和PID信息
        thread_context.thread_span = tracer.start_span(  # 创建线程跨度
            name=thread_name,  # 跨度名称
            start_time=ts,  # 开始时间
            context=self.root_span_context,  # 根跨度上下文
        )

        rank_attrs = {}  # 排名属性字典
        if thread_info.tp_rank is not None:  # 如果有TP排名
            rank_attrs["tp_rank"] = thread_info.tp_rank  # 添加TP排名属性
        if thread_info.pp_rank is not None:  # 如果有PP排名
            rank_attrs["pp_rank"] = thread_info.pp_rank  # 添加PP排名属性
        if thread_info.dp_rank is not None:  # 如果有DP排名
            rank_attrs["dp_rank"] = thread_info.dp_rank  # 添加DP排名属性
        if rank_attrs:  # 如果有排名属性
            thread_context.thread_span.set_attributes(rank_attrs)  # 设置排名属性

        thread_context.thread_span.set_attributes(  # 设置线程基础属性
            {
                "host_id": thread_info.host_id,  # 主机标识
                "pid": thread_info.pid,  # 进程ID
                "thread_label": thread_info.thread_label,  # 线程标签
            }
        )

        return thread_context  # 返回线程上下文

    def __getstate__(self) -> Optional[Dict[str, Any]]:  # 序列化状态
        """序列化追踪请求上下文用于进程间传输"""
        if not self.tracing_enable:  # 如果追踪未启用
            return {"tracing_enable": False}  # 返回禁用状态

        if not self.root_span_context:  # 如果没有根跨度上下文
            return {"tracing_enable": False}  # 返回禁用状态

        state = {  # 构建序列化状态字典
            "tracing_enable": self.tracing_enable,  # 追踪启用状态
            "rid": self.rid,  # 请求ID
            "bootstrap_room": self.bootstrap_room,  # 引导房间号
            "start_time_ns": self.start_time_ns,  # 开始时间
            "role": self.role,  # 角色
            "trace_level": self.trace_level,  # 追踪级别
            "module_name": self.module_name,  # 模块名称
            "is_copy": self.is_copy,  # 是否为副本
            "pid": self.pid,  # 进程ID
            "thread_context": None,  # 线程上下文置空
            "root_span": None,  # 根跨度置空
            "last_span_context": None,  # 上一个跨度上下文置空
        }

        carrier: dict[str, str] = {}  # 传播载体
        propagate.inject(carrier, self.root_span_context)  # 注入追踪上下文到载体
        state["root_span_context"] = carrier  # 存储传播后的上下文

        prev_span_context = self.last_span_context  # 获取上一个跨度上下文
        if self.thread_context and self.thread_context.cur_slice_stack:  # 如果有线程上下文和切片栈
            cur_slice = self.thread_context.cur_slice_stack[0]  # 获取栈底的切片
            if cur_slice.span:  # 如果切片有跨度
                prev_span_context = cur_slice.span.get_span_context()  # 获取其跨度上下文

        if prev_span_context:  # 如果有上一个跨度上下文
            state["last_span_context"] = {  # 存储跨度上下文信息
                "span_id": prev_span_context.span_id,  # 跨度ID
                "trace_id": prev_span_context.trace_id,  # 追踪ID
            }

        return state  # 返回序列化状态

    def __setstate__(self, state: Dict[str, Any]):  # 反序列化状态
        """从序列化状态恢复追踪请求上下文"""
        self.__dict__.update(state)  # 更新实例字典
        if not opentelemetry_initialized:  # 如果OpenTelemetry未初始化
            self.tracing_enable = False  # 禁用追踪
        if not self.tracing_enable:  # 如果追踪未启用
            return  # 直接返回

        self.is_copy = True  # 标记为副本
        self.pid = threading.get_native_id()  # 获取当前线程ID
        self.root_span_context = propagate.extract(self.root_span_context)  # 从载体中提取追踪上下文
        if self.last_span_context:  # 如果有上一个跨度上下文
            self.last_span_context = trace.span.SpanContext(  # 重建跨度上下文
                trace_id=self.last_span_context["trace_id"],  # 追踪ID
                span_id=self.last_span_context["span_id"],  # 跨度ID
                is_remote=True,  # 标记为远程
            )
        self.events_cache = []  # 重置事件缓存

    def rebuild_thread_context(self, ts: Optional[int] = None):  # 重建线程上下文
        """在反序列化后重建线程上下文"""
        if not self.tracing_enable:  # 如果追踪未启用
            return  # 直接返回

        ts = ts or get_cur_time_ns()  # 获取当前时间戳
        self.thread_context = self.__create_thread_context(ts)  # 创建新的线程上下文

    def trace_req_start(
        self,
        ts: Optional[int] = None,  # 时间戳，可选
    ):  # 开始追踪请求
        """开始追踪一个请求，创建根跨度和线程上下文"""
        if not self.tracing_enable:  # 如果追踪未启用
            return  # 直接返回

        ts = ts or get_cur_time_ns()  # 获取当前时间戳

        # create req context and root span  # 创建请求上下文和根跨度
        self.start_time_ns = ts  # 记录开始时间

        external_trace_context = _trace_context_propagator.extract(
            self.external_trace_header or {}  # 从外部追踪头提取上下文
        )

        # Drop the worker_id added by MultiTokenizer  # 去除MultiTokenizer添加的worker_id
        orig_rid = self.rid.split("_")[-1]  # 获取原始请求ID
        role = "" if self.role == "unified" else self.role  # 统一角色时为空字符串
        attrs = {"rid": orig_rid, "module": f"sglang::{self.module_name}"}  # 跨度属性
        if self.bootstrap_room:  # 如果有引导房间号
            attrs["bootstrap_room"] = str(hex(self.bootstrap_room))  # 添加十六进制引导房间属性
        root_span = tracer.start_span(  # 创建根跨度
            name=f"{role} Req {orig_rid[:8]}",  # 跨度名称
            start_time=ts,  # 开始时间
            context=external_trace_context,  # 外部追踪上下文
            attributes=attrs,  # 跨度属性
        )

        self.root_span = root_span  # 保存根跨度
        self.root_span_context = trace.set_span_in_context(root_span)  # 设置根跨度上下文

        # create thread context and thread span  # 创建线程上下文和线程跨度
        self.thread_context = self.__create_thread_context(ts)  # 创建线程上下文

    def trace_req_finish(
        self, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None  # 时间戳和属性，可选
    ):  # 结束追踪请求
        """结束追踪一个请求，关闭所有未关闭的跨度"""
        if not self.tracing_enable:  # 如果追踪未启用
            return  # 直接返回

        if not self.root_span:  # 如果没有根跨度
            return  # 直接返回

        ts = ts or get_cur_time_ns()  # 获取当前时间戳

        # End all unclosed thread spans.  # 结束所有未关闭的线程跨度
        self.abort()  # 中止所有未关闭的跨度

        if attrs:  # 如果有额外属性
            self.root_span.set_attributes(attrs)  # 设置根跨度属性

        self.root_span.end(end_time=ts)  # 结束根跨度
        self.root_span = None  # 清空根跨度引用

    def __check_fast_return(self, level=None):  # 检查是否可以快速返回
        """检查追踪是否禁用或级别不足，用于快速返回"""
        if not self.tracing_enable:  # 如果追踪未启用
            return True  # 快速返回

        if not self.thread_context:  # 如果没有线程上下文
            return True  # 快速返回

        if level and level > self.trace_level:  # 如果级别超过追踪级别
            return True  # 快速返回

        return False  # 不快速返回

    def trace_slice_start(
        self,
        name: str,  # 切片名称
        level: int,  # 追踪级别
        ts: Optional[int] = None,  # 时间戳，可选
    ):  # 开始追踪一个切片
        """开始追踪一个切片（时间片段），创建对应的跨度"""
        if self.__check_fast_return(level):  # 检查是否快速返回
            return  # 直接返回

        ts = ts or get_cur_time_ns()  # 获取当前时间戳

        cur_slice = TraceSliceContext(  # 创建切片上下文
            slice_name=name,  # 切片名称
            start_time_ns=ts,  # 开始时间
            level=level,  # 追踪级别
            attrs={},  # 空属性字典
            events=[],  # 空事件列表
        )

        parent_span = self.thread_context.thread_span  # 默认父跨度为线程跨度
        prev_span_context = None  # 上一个跨度上下文
        if not self.thread_context.cur_slice_stack:  # 如果切片栈为空
            if self.last_span_context:  # 如果有上一个跨度上下文
                prev_span_context = self.last_span_context  # 使用上一个跨度上下文
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span  # 使用栈顶跨度的span作为父跨度

        parent_span_context = trace.set_span_in_context(parent_span)  # 设置父跨度上下文

        span = tracer.start_span(  # 创建新跨度
            name=cur_slice.slice_name,  # 跨度名称
            start_time=cur_slice.start_time_ns,  # 开始时间
            context=parent_span_context,  # 父跨度上下文
        )
        cur_slice.span = span  # 保存跨度引用

        if prev_span_context:  # 如果有上一个跨度上下文
            span.add_link(prev_span_context)  # 添加跨度链接

        self.thread_context.cur_slice_stack.append(cur_slice)  # 将切片压入栈

    def trace_slice_end(
        self,
        name: str,  # 切片名称
        level: int,  # 追踪级别
        ts: Optional[int] = None,  # 时间戳，可选
        attrs: Optional[Dict[str, Any]] = None,  # 属性，可选
        thread_finish_flag: bool = False,  # 线程结束标志
    ):  # 结束追踪一个切片
        """结束追踪一个切片，关闭对应的跨度并设置属性"""
        if self.__check_fast_return(level):  # 检查是否快速返回
            return  # 直接返回

        if not self.thread_context.cur_slice_stack:  # 如果切片栈为空
            logger.warning(
                f"No matching with the SLICE_START event {name} is required."  # 记录警告
            )
            return  # 直接返回

        cur_slice = self.thread_context.cur_slice_stack[-1]  # 获取栈顶切片
        ts = ts or get_cur_time_ns()  # 获取当前时间戳

        # check if slice_name matching and level matching  # 检查切片名称和级别是否匹配
        # unlikely path, excepting error API usage  # 不太可能发生，除非API使用错误
        if cur_slice.slice_name != name or cur_slice.level != level:  # 如果名称或级别不匹配
            logger.warning(
                f"Slice name mismatch: {name} != {cur_slice.slice_name} or level mismatch: {level} != {cur_slice.level}"  # 记录警告
            )
            self.thread_context.cur_slice_stack.pop()  # 弹出栈顶切片
            return  # 直接返回

        span = cur_slice.span  # 获取当前切片的跨度

        if attrs:  # 如果有属性
            span.set_attributes(attrs)  # 设置跨度属性

        if self.events_cache:  # 如果有缓存的事件
            new_events_cache = []  # 新的事件缓存
            for event in self.events_cache:  # 遍历所有缓存事件
                if event.ts >= cur_slice.start_time_ns and event.ts < ts:  # 如果事件在当前切片时间范围内
                    span.add_event(  # 添加事件到跨度
                        name=event.event_name,  # 事件名称
                        timestamp=event.ts,  # 事件时间戳
                        attributes=event.attrs,  # 事件属性
                    )
                else:
                    new_events_cache.append(event)  # 保留不在范围内的事件
            self.events_cache = new_events_cache  # 更新事件缓存

        span.end(end_time=ts)  # 结束跨度

        self.thread_context.cur_slice_stack.pop()  # 弹出栈顶切片
        # only for first level slice  # 仅对第一级切片
        if not self.thread_context.cur_slice_stack:  # 如果切片栈为空
            self.last_span_context = span.get_span_context()  # 记录最后一个跨度上下文

        if thread_finish_flag:  # 如果设置了线程结束标志
            self.abort(ts)  # 中止追踪

    def trace_slice(
        self,
        slice: TraceSliceContext,  # 追踪切片上下文
        thread_finish_flag: bool = False,  # 线程结束标志
    ):  # 追踪一个完整的切片（开始和结束）
        """追踪一个完整的切片，一次性创建并结束跨度"""
        if self.__check_fast_return(slice.level):  # 检查是否快速返回
            return  # 直接返回

        parent_span = self.thread_context.thread_span  # 默认父跨度为线程跨度
        prev_span_context = None  # 上一个跨度上下文
        if not self.thread_context.cur_slice_stack:  # 如果切片栈为空
            if self.last_span_context:  # 如果有上一个跨度上下文
                prev_span_context = self.last_span_context  # 使用上一个跨度上下文
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span  # 使用栈顶跨度的span

        parent_span_context = trace.set_span_in_context(parent_span)  # 设置父跨度上下文

        span = tracer.start_span(  # 创建新跨度
            name=slice.slice_name,  # 跨度名称
            start_time=slice.start_time_ns,  # 开始时间
            context=parent_span_context,  # 父跨度上下文
        )

        if prev_span_context:  # 如果有上一个跨度上下文
            span.add_link(prev_span_context)  # 添加跨度链接

        if slice.attrs:  # 如果切片有属性
            span.set_attributes(slice.attrs)  # 设置跨度属性

        if slice.events:  # 如果切片有事件
            for event in slice.events:  # 遍历事件
                span.add_event(
                    name=event.event_name, timestamp=event.ts, attributes=event.attrs  # 添加事件
                )

        if self.events_cache:  # 如果有缓存的事件
            new_events_cache = []  # 新的事件缓存
            for event in self.events_cache:  # 遍历所有缓存事件
                if event.ts >= slice.start_time_ns and event.ts < slice.end_time_ns:  # 如果事件在切片时间范围内
                    span.add_event(  # 添加事件到跨度
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                else:
                    new_events_cache.append(event)  # 保留不在范围内的事件
            self.events_cache = new_events_cache  # 更新事件缓存

        span.end(end_time=slice.end_time_ns)  # 结束跨度

        # only for first level slice  # 仅对第一级切片
        if not self.thread_context.cur_slice_stack:  # 如果切片栈为空
            self.last_span_context = span.get_span_context()  # 记录最后一个跨度上下文

        if thread_finish_flag:  # 如果设置了线程结束标志
            self.abort(slice.end_time_ns)  # 中止追踪

    # Add event to the current slice on the same thread with the same rid.  # 在同一线程和同一请求ID的当前切片中添加事件
    def trace_event(
        self,
        name: str,  # 事件名称
        level: int,  # 追踪级别
        ts: Optional[int] = None,  # 时间戳，可选
        attrs: Dict[str, Any] = None,  # 事件属性
    ):  # 添加追踪事件
        """添加一个追踪事件到缓存，将在切片结束时关联到对应跨度"""
        if self.__check_fast_return(level):  # 检查是否快速返回
            return  # 直接返回

        ts = ts or get_cur_time_ns()  # 获取当前时间戳

        if attrs is None:  # 如果属性为None
            attrs = {}  # 使用空字典
        self.events_cache.append(TraceEvent(name, ts, attrs))  # 将事件添加到缓存

    def trace_set_root_attrs(self, attrs: Dict[str, Any]):  # 设置根跨度属性
        """设置根跨度的属性"""
        if not self.tracing_enable:  # 如果追踪未启用
            return  # 直接返回

        if self.root_span:  # 如果有根跨度
            self.root_span.set_attributes(attrs)  # 设置根跨度属性

    def trace_set_thread_attrs(self, attrs: Dict[str, Any]):  # 设置线程跨度属性
        """设置线程跨度的属性"""
        if self.__check_fast_return():  # 检查是否快速返回
            return  # 直接返回

        if self.thread_context.thread_span:  # 如果有线程跨度
            self.thread_context.thread_span.set_attributes(attrs)  # 设置线程跨度属性

    def abort(self, ts=None, abort_info: Optional[Dict] = None):  # 中止追踪
        """中止追踪，关闭所有未关闭的跨度并设置中止信息"""
        if self.__check_fast_return():  # 检查是否快速返回
            return  # 直接返回

        # close all slice spans (unlikely, except error API usage)  # 关闭所有切片跨度（不太可能，除非API使用错误）
        ts = ts or get_cur_time_ns()  # 获取当前时间戳
        while len(self.thread_context.cur_slice_stack) > 0:  # 当切片栈不为空
            if self.thread_context.cur_slice_stack[-1].span:  # 如果栈顶切片有跨度
                self.thread_context.cur_slice_stack[-1].span.end(end_time=ts)  # 结束该跨度
            self.thread_context.cur_slice_stack.pop()  # 弹出栈顶切片

        # set abort info into thread span  # 将中止信息设置到线程跨度
        if self.thread_context.thread_span:  # 如果有线程跨度
            if abort_info:  # 如果有中止信息
                from sglang.srt.managers.schedule_batch import BaseFinishReason  # 导入完成原因类

                if isinstance(abort_info, BaseFinishReason):  # 如果是BaseFinishReason实例
                    abort_info = abort_info.to_json()  # 转换为JSON
                self.thread_context.thread_span.set_status(Status(StatusCode.ERROR))  # 设置错误状态
                self.thread_context.thread_span.set_attributes(abort_info)  # 设置中止信息属性

            if self.events_cache:  # 如果有缓存的事件
                for event in self.events_cache:  # 遍历所有缓存事件
                    self.thread_context.thread_span.add_event(  # 添加事件到线程跨度
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                self.events_cache = []  # 清空事件缓存

            self.thread_context.thread_span.end(end_time=ts)  # 结束线程跨度
        self.thread_context = None  # 清空线程上下文

    def __del__(self):  # 析构函数
        """析构时自动关闭未关闭的跨度"""
        self.abort(abort_info={"reason": "have unclosed span, auto closed"})  # 中止并记录原因


@dataclass
class TraceNullContext:  # 空追踪上下文数据类，用于禁用追踪时
    tracing_enable: bool = False  # 追踪启用标志，默认为False

    def __getattr__(self, name):  # 属性访问代理
        """属性访问代理，返回自身以支持链式调用"""
        return self  # 返回自身

    def __call__(self, *args, **kwargs):  # 调用代理
        """调用代理，返回自身以忽略所有调用"""
        return self  # 返回自身


class SpanAttributes:  # 跨度属性名称常量类
    # Attribute names copied from here to avoid version conflicts:  # 从此处复制属性名以避免版本冲突：
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"  # 生成AI补全token数
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"  # 生成AI提示token数
    GEN_AI_USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"  # 生成AI缓存token数
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"  # 生成AI请求最大token数
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"  # 生成AI请求top_p参数
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"  # 生成AI请求top_k参数
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"  # 生成AI请求温度参数
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"  # 生成AI响应模型
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"  # 生成AI响应完成原因
    GEN_AI_REQUEST_ID = "gen_ai.request.id"  # 生成AI请求ID
    GEN_AI_REQUEST_N = "gen_ai.request.n"  # 生成AI请求n参数
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"  # 生成AI队列等待延迟
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"  # 生成AI首token延迟
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"  # 生成AI端到端延迟
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"  # 生成AI预填充延迟
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"  # 生成AI解码延迟
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"  # 生成AI推理延迟
