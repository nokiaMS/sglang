# 请求时间统计模块
# 本模块提供请求各阶段时间戳的记录和统计功能
# 支持统一模式、预填充分离模式和解码分离模式

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
"""Utilities for Request Time Stats."""

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
import time  # 导入时间模块
import uuid  # 导入UUID模块
from dataclasses import dataclass, field  # 导入数据类装饰器和字段
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union  # 导入类型提示

from sglang.srt.disaggregation.utils import DisaggregationMode  # 导入分离模式
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # 导入前向模式
from sglang.srt.observability.metrics_collector import (  # 导入指标收集器
    SchedulerMetricsCollector,
    TokenizerMetricsCollector,
)
from sglang.srt.observability.trace import (  # 导入追踪相关类
    SpanAttributes,
    TraceNullContext,
    TraceReqContext,
    TraceSliceContext,
    get_global_tracing_enabled,
)
from sglang.srt.utils import get_bool_env_var  # 导入布尔环境变量工具

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.disaggregation.base.conn import KVTransferMetric  # 导入KV传输指标
    from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")  # 是否测试请求时间统计


logger = logging.getLogger(__name__)  # 创建日志记录器

# Reduce system time calls by computing time.time() based on calibrated perf_counter() values.
global_diff_realtime_monotonic = time.time() - time.perf_counter()  # 全局时间差：真实时间与单调时间之差


def calibrate_time_diff():  # 校准时间差
    # due to NTP, the diff between time.time() and time.perf_counter() can change
    # periodically calibrate the diff
    global global_diff_realtime_monotonic  # 声明全局变量
    global_diff_realtime_monotonic = time.time() - time.perf_counter()  # 更新时间差


real_time = time.time  # 真实时间函数引用
monotonic_time = time.perf_counter  # 单调时间函数引用


def convert_time_to_realtime(time_value: float) -> float:  # 将单调时间转换为真实时间
    # note: Within the time scale of a single request's latency,
    # we assume that the diff does not change significantly.
    return time_value + global_diff_realtime_monotonic  # 加上时间差


def convert_time_to_realtime_ns(time_value: float) -> int:  # 将单调时间转换为真实时间（纳秒）
    return int((time_value + global_diff_realtime_monotonic) * 1e9)  # 转换为纳秒整数


def convert_time_cross_thread(  # 跨线程转换时间
    time_value: float, old_diff: float, new_diff: float
) -> float:
    # note: precision loss
    return time_value + old_diff - new_diff  # 调整时间差


@dataclass
class RequestStageConfig:  # 请求阶段配置
    """Configuration for a request pipeline stage.

    Attributes:
        stage_name: Name used for metrics labels and trace span names.
        level: Trace hierarchy depth.
            1 = leaf stages (atomic operations, e.g. TOKENIZE, PREFILL_FORWARD),
            2 = parent/dispatch stages (e.g. API_SERVER_DISPATCH, REQUEST_PROCESS),
            3 = composite/nested stages (e.g. DECODE_LOOP, PREFILL_CHUNKED_FORWARD).
        metrics_is_observed: Whether to call metrics_collector.observe_per_stage_req_latency.
    """

    stage_name: str  # 阶段名称
    level: int = 0  # 追踪层级深度
    metrics_is_observed: bool = False  # 是否观测指标


class RequestStage:  # 请求阶段定义
    # Tokenizer/gRPC Server
    TOKENIZE = RequestStageConfig(  # 分词阶段
        "tokenize",
        level=1,
    )
    API_SERVER_DISPATCH = RequestStageConfig(  # API服务器分发阶段
        "api_server_dispatch",
        level=2,
    )

    # DP controller
    DPC_DISPATCH = RequestStageConfig(  # DP控制器分发阶段
        "dpc_dispatch",
        level=2,
    )

    # common/non-disaggregation
    REQUEST_PROCESS = RequestStageConfig(  # 请求处理阶段
        "request_process",
        level=2,
        metrics_is_observed=True,
    )
    PREFILL_WAITING = RequestStageConfig(  # 预填充等待阶段
        "prefill_waiting",
        level=1,
        # equal to "observe_queue_time"
        metrics_is_observed=False,
    )
    DECODE_FORWARD = RequestStageConfig(  # 解码前向阶段
        "decode_forward",
        level=1,
    )
    DECODE_LOOP = RequestStageConfig(  # 解码循环阶段
        "decode_loop",
        level=3,
    )
    PREFILL_FORWARD = RequestStageConfig(  # 预填充前向阶段
        "prefill_forward",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_CHUNKED_FORWARD = RequestStageConfig(  # 分块预填充前向阶段
        "chunked_prefill",
        level=3,
        metrics_is_observed=True,
    )

    # disaggregation prefill
    PREFILL_PREPARE = RequestStageConfig(  # 预填充准备阶段
        "prefill_prepare",
        level=1,
    )
    PREFILL_BOOTSTRAP = RequestStageConfig(  # 预填充引导阶段
        "prefill_bootstrap",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_TRANSFER_KV_CACHE = RequestStageConfig(  # 预填充KV缓存传输阶段
        "prefill_transfer_kv_cache",
        level=1,
        metrics_is_observed=True,
    )

    # disaggregation decode
    DECODE_PREPARE = RequestStageConfig(  # 解码准备阶段
        "decode_prepare",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_BOOTSTRAP = RequestStageConfig(  # 解码引导阶段
        "decode_bootstrap",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_WAITING = RequestStageConfig(  # 解码等待阶段
        "decode_waiting",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_TRANSFERRED = RequestStageConfig(  # 解码已传输阶段
        "decode_transferred",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_FAKE_OUTPUT = RequestStageConfig(  # 解码伪输出阶段
        "fake_output",
        level=3,
        metrics_is_observed=True,
    )
    DECODE_QUICK_FINISH = RequestStageConfig(  # 解码快速完成阶段
        "quick_finish",
        level=1,
        metrics_is_observed=True,
    )

    # speculative decode
    SPEC_DRAFT = RequestStageConfig(  # 推测解码草稿阶段
        "spec_draft",
        level=2,
    )

    SPEC_VERIFY = RequestStageConfig(  # 推测解码验证阶段
        "spec_verify",
        level=2,
    )

    SPEC_DRAFT_EXTEND = RequestStageConfig(  # 推测解码草稿扩展阶段
        "spec_draft_extend",
        level=3,
    )

    # CPU-side run batch
    RUN_BATCH_CPU = RequestStageConfig(  # CPU端运行批次阶段
        "run_batch_cpu",
        level=4,
    )

    # other
    ANONYMOUS = RequestStageConfig("")  # 匿名阶段


@dataclass
class ReqTimeStatsBase:  # 请求时间统计基类
    enable_metrics: bool = False  # 是否启用指标
    metrics_collector: Optional[  # 指标收集器
        Union[SchedulerMetricsCollector, TokenizerMetricsCollector]
    ] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = field(  # 追踪上下文
        default_factory=TraceNullContext
    )
    disagg_mode: DisaggregationMode = DisaggregationMode.NULL  # 分离模式
    diff_realtime_monotonic: float = 0.0  # 时间差

    @classmethod
    def new_from_obj(cls, obj: ReqTimeStatsBase, *args, **kwargs) -> "ReqTimeStatsBase":  # 从现有对象创建新实例
        calibrate_time_diff()  # 校准时间差
        new_obj = cls(*args, **kwargs)  # 创建新对象
        if obj is None:  # 如果源对象为None
            return new_obj  # 返回新对象
        for key, value in obj.__dict__.items():  # 遍历源对象属性
            if hasattr(new_obj, key):  # 如果新对象有该属性
                setattr(new_obj, key, value)  # 复制属性

        if new_obj.trace_ctx.tracing_enable:  # 如果追踪启用
            new_obj.trace_ctx.rebuild_thread_context()  # 重建线程上下文

        return new_obj  # 返回新对象

    def disagg_mode_str(self) -> str:  # 获取分离模式字符串
        if self.disagg_mode == DisaggregationMode.NULL:  # 统一模式
            return "unified"  # 返回unified
        elif self.disagg_mode == DisaggregationMode.DECODE:  # 解码模式
            return "decode"  # 返回decode
        elif self.disagg_mode == DisaggregationMode.PREFILL:  # 预填充模式
            return "prefill"  # 返回prefill
        else:  # 其他
            return "unknown"  # 返回unknown

    def set_metrics_collector(  # 设置指标收集器
        self, collector: Union[SchedulerMetricsCollector, TokenizerMetricsCollector]
    ):
        if collector:  # 如果收集器不为空
            self.enable_metrics = True  # 启用指标
            self.metrics_collector = collector  # 设置收集器

    def observe_per_stage_req_latency(self, stage: RequestStageConfig, latency: float):  # 观测每阶段请求延迟
        if self.enable_metrics and stage.metrics_is_observed:  # 如果启用指标且阶段需要观测
            self.metrics_collector.observe_per_stage_req_latency(  # 调用收集器观测
                stage.stage_name, latency
            )

    def init_trace_ctx(  # 初始化追踪上下文
        self,
        rid: str,  # 请求ID
        bootstrap_room: Optional[int],  # 引导房间号
        external_trace_header: Optional[Dict[str, str]] = None,  # 外部追踪头
    ):
        self.trace_ctx = TraceReqContext(  # 创建追踪请求上下文
            rid=rid,  # 请求ID
            bootstrap_room=bootstrap_room,  # 引导房间号
            role=self.disagg_mode_str(),  # 角色
            module_name="request",  # 模块名
            external_trace_header=external_trace_header,  # 外部追踪头
        )

        if not self.trace_ctx.tracing_enable:  # 如果追踪未启用
            self.trace_ctx = TraceNullContext()  # 使用空上下文

    def trace_slice(  # 追踪时间切片
        self,
        stage: RequestStageConfig,  # 请求阶段
        start_time: float,  # 开始时间
        end_time: float,  # 结束时间
        attrs: Optional[Dict] = None,  # 属性
    ):
        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            _slice = TraceSliceContext(  # 创建追踪切片上下文
                slice_name=stage.stage_name,  # 切片名称
                start_time_ns=convert_time_to_realtime_ns(start_time),  # 开始时间（纳秒）
                end_time_ns=convert_time_to_realtime_ns(end_time),  # 结束时间（纳秒）
                level=stage.level,  # 层级
                attrs=attrs,  # 属性
            )
            self.trace_ctx.trace_slice(_slice)  # 追踪切片

    def __getstate__(self) -> object:  # 序列化状态（用于跨进程传输）
        # The object is propagated to other processes via serialization and deserialization methods,
        # requiring the metric collector to be reconfigured.
        return {
            "disagg_mode": self.disagg_mode,  # 分离模式
            "enable_metrics": False,  # 禁用指标（需要重新配置）
            "trace_ctx": self.trace_ctx,  # 追踪上下文
            "diff_realtime_monotonic": global_diff_realtime_monotonic,  # 时间差
        }

    def __setstate__(self, state: object):  # 反序列化状态
        for key in state.keys():  # 遍历状态键
            if key.endswith("time"):  # 如果是时间字段
                state[key] = convert_time_cross_thread(  # 跨线程转换时间
                    state[key],
                    state["diff_realtime_monotonic"],
                    global_diff_realtime_monotonic,
                )
        self.__dict__.update(state)  # 更新对象状态


@dataclass
class APIServerReqTimeStats(ReqTimeStatsBase):  # API服务器请求时间统计
    # get by time.perf_counter()
    created_time: float = 0.0  # 创建时间
    finished_time: float = 0.0  # 完成时间
    first_token_time: float = 0.0  # 首令牌时间
    last_time: float = 0.0  # 最后时间
    tokenize_finish_time: float = 0.0  # 分词完成时间
    api_server_dispatch_time: float = 0.0  # API服务器分发时间
    api_server_dispatch_finish_time: float = 0.0  # API服务器分发完成时间
    response_sent_to_client_time: float = 0.0  # 响应发送到客户端时间

    def __getstate__(self) -> object:  # 序列化状态
        state = {}  # 状态字典
        # send to DP controller or Scheduler
        # If necessary, can propagate the timestamp here, for example:
        # state = {
        #    "created_time": self.created_time,
        #    "api_server_dispatch_time": self.api_server_dispatch_time,
        # }
        state.update(super().__getstate__())  # 更新父类状态
        return state  # 返回状态

    def set_created_time(self, ts=None):  # 设置创建时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.created_time = ts  # 设置创建时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.trace_req_start(convert_time_to_realtime_ns(ts))  # 追踪请求开始

    def set_finished_time(self, ts=None):  # 设置完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.finished_time = ts  # 设置完成时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.trace_req_finish(convert_time_to_realtime_ns(ts))  # 追踪请求完成

    def set_first_token_time(self, ts=None):  # 设置首令牌时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.first_token_time = ts  # 设置首令牌时间
        self.last_time = ts  # 更新最后时间

    def set_last_time(self, ts=None):  # 设置最后时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.last_time = ts  # 设置最后时间

    def set_tokenize_finish_time(self, ts=None):  # 设置分词完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.tokenize_finish_time = ts  # 设置分词完成时间

        stage = RequestStage.TOKENIZE  # 获取分词阶段
        self.trace_slice(stage, self.created_time, ts)  # 追踪分词切片

    def set_api_server_dispatch_time(self, ts=None):  # 设置API服务器分发时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.api_server_dispatch_time = ts  # 设置分发时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.trace_slice_start(  # 追踪切片开始
                RequestStage.API_SERVER_DISPATCH.stage_name,
                RequestStage.API_SERVER_DISPATCH.level,
                convert_time_to_realtime_ns(ts),
            )

    def set_api_server_dispatch_finish_time(self, ts=None):  # 设置API服务器分发完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.api_server_dispatch_finish_time = ts  # 设置分发完成时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.trace_slice_end(  # 追踪切片结束
                RequestStage.API_SERVER_DISPATCH.stage_name,
                RequestStage.API_SERVER_DISPATCH.level,
                convert_time_to_realtime_ns(ts),
                thread_finish_flag=True,  # 线程完成标志
            )

    def set_response_sent_to_client_time(self, ts=None):  # 设置响应发送到客户端时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.response_sent_to_client_time = ts  # 设置时间

    def get_interval(self):  # 获取距上次的时间间隔
        return time.perf_counter() - self.last_time  # 返回间隔

    def get_first_token_latency(self):  # 获取首令牌延迟
        return self.first_token_time - self.created_time  # 返回延迟

    def get_e2e_latency(self):  # 获取端到端延迟
        return self.finished_time - self.created_time  # 返回延迟

    def get_decode_latency(self):  # 获取解码延迟
        return self.finished_time - self.first_token_time  # 返回延迟

    def get_response_sent_to_client_realtime(self):  # 获取响应发送到客户端的真实时间
        return convert_time_to_realtime(self.response_sent_to_client_time)  # 转换并返回

    def convert_to_output_meta_info(  # 转换为输出元信息
        self, scheduler_time_stats=None, completion_tokens=0
    ):
        meta_info = {}  # 元信息字典
        if self.created_time > 0.0:  # 如果创建时间有效
            meta_info["request_received_ts"] = convert_time_to_realtime(  # 请求接收时间戳
                self.created_time
            )
        if self.api_server_dispatch_finish_time > 0.0:  # 如果分发完成时间有效
            meta_info["api_server_dispatch_finish_ts"] = convert_time_to_realtime(  # 分发完成时间戳
                self.api_server_dispatch_finish_time
            )
        if self.response_sent_to_client_time > 0.0:  # 如果响应发送时间有效
            meta_info["response_sent_to_client_ts"] = convert_time_to_realtime(  # 响应发送时间戳
                self.response_sent_to_client_time
            )
        if self.finished_time > 0.0:  # 如果完成时间有效
            meta_info["request_finished_ts"] = convert_time_to_realtime(  # 请求完成时间戳
                self.finished_time
            )

        decode_latency = self.get_decode_latency()  # 获取解码延迟
        if decode_latency > 0.0 and completion_tokens > 1:  # 如果有效
            meta_info["decode_throughput"] = (completion_tokens - 1) / decode_latency  # 计算解码吞吐量
        return meta_info  # 返回元信息

    def convert_to_gen_ai_span_attrs(self):  # 转换为生成AI跨度属性
        span_attrs = {}  # 跨度属性字典
        if self.first_token_time and self.created_time:  # 如果首令牌时间和创建时间有效
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN] = (  # 首令牌延迟
                self.first_token_time - self.created_time
            )

        if self.finished_time and self.created_time:  # 如果完成时间和创建时间有效
            span_attrs[SpanAttributes.GEN_AI_LATENCY_E2E] = (  # 端到端延迟
                self.finished_time - self.created_time
            )

        if self.first_token_time and self.finished_time:  # 如果首令牌时间和完成时间有效
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE] = (  # 模型解码时间
                self.finished_time - self.first_token_time
            )

        if self.api_server_dispatch_finish_time and self.finished_time:  # 如果分发完成时间和完成时间有效
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE] = (  # 模型推理时间
                self.finished_time - self.api_server_dispatch_finish_time
            )

        if self.api_server_dispatch_finish_time and self.first_token_time:  # 如果分发完成时间和首令牌时间有效
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL] = (  # 模型预填充时间
                self.first_token_time - self.api_server_dispatch_finish_time
            )

        return span_attrs  # 返回跨度属性


@dataclass
class DPControllerReqTimeStats(ReqTimeStatsBase):  # DP控制器请求时间统计
    # propagated from tokenizer/grpc_server, get by time.perf_counter()
    created_time: float = 0.0  # 创建时间（从上层传播）
    api_server_dispatch_time: float = 0.0  # API服务器分发时间（从上层传播）

    # new timestamp, get by time.perf_counter()
    dpc_dispatch_time: float = 0.0  # DP控制器分发时间
    dpc_dispatch_finish_time: float = 0.0  # DP控制器分发完成时间

    def __getstate__(self) -> object:  # 序列化状态
        state = {}  # 状态字典
        # send to Scheduler
        # If necessary, can propagate the timestamp here, for example:
        # state = {
        #     "created_time": self.created_time,
        #     "api_server_dispatch_time": self.api_server_dispatch_time,
        #     "dpc_dispatch_time": self.dpc_dispatch_time,
        # }
        state.update(super().__getstate__())  # 更新父类状态
        return state  # 返回状态

    def set_dp_dispatch_time(self, ts=None):  # 设置DP控制器分发时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.dpc_dispatch_time = ts  # 设置分发时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.trace_slice_start(  # 追踪切片开始
                RequestStage.DPC_DISPATCH.stage_name,
                RequestStage.DPC_DISPATCH.level,
                convert_time_to_realtime_ns(ts),
            )

    def set_dp_dispatch_finish_time(self, ts=None):  # 设置DP控制器分发完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.dpc_dispatch_finish_time = ts  # 设置分发完成时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.trace_slice_end(  # 追踪切片结束
                RequestStage.DPC_DISPATCH.stage_name,
                RequestStage.DPC_DISPATCH.level,
                convert_time_to_realtime_ns(ts),
                thread_finish_flag=True,  # 线程完成标志
            )


@dataclass
class SchedulerReqTimeStats(ReqTimeStatsBase):  # 调度器请求时间统计
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    # Placeholder: not used currently
    # propagated from tokenizer/grpc_server or dp controller
    created_time: float = 0.0  # 创建时间（从上层传播）
    api_server_dispatch_time: float = 0.0  # API服务器分发时间（从上层传播）
    dpc_dispatch_time: float = 0.0  # DP控制器分发时间（从上层传播）

    # common, get by time.perf_counter()
    wait_queue_entry_time: float = 0.0  # 等待队列进入时间
    forward_entry_time: float = 0.0  # 前向进入时间
    prefill_finished_time: float = 0.0  # 预填充完成时间
    completion_time: float = 0.0  # 完成时间

    # prefill node, get by time.perf_counter()
    prefill_bootstrap_queue_entry_time: float = 0.0  # 预填充引导队列进入时间
    prefill_transfer_queue_entry_time: float = 0.0  # 预填充传输队列进入时间
    prefill_kv_transfer_finish_time: float = 0.0  # 预填充KV传输完成时间

    # decode node, get by time.perf_counter()
    decode_prealloc_queue_entry_time: float = 0.0  # 解码预分配队列进入时间
    decode_transfer_queue_entry_time: float = 0.0  # 解码传输队列进入时间
    decode_prebuilt_finish_time: float = 0.0  # 解码预构建完成时间

    # bootstrap sub-phase tracking (PD disagg)
    bootstrap_done_time: float = 0.0  # 引导完成时间

    # only for request tracing
    scheduler_recv_time: float = 0.0  # 调度器接收时间
    last_chunked_prefill_finish_time: float = 0.0  # 上次分块预填充完成时间
    last_decode_finish_time: float = 0.0  # 上次解码完成时间
    decode_ct: int = 0  # 解码计数
    last_decode_scheduled_time: float = 0.0  # 上次解码调度时间
    last_forward_entry_time: float = 0.0  # 上次前向进入时间
    last_prefill_finished_time: float = 0.0  # 上次预填充完成时间
    run_batch_cpu_start_time: float = 0.0  # CPU端运行批次开始时间

    # speculative decoding
    spec_draft_start_time: float = 0.0  # 推测解码草稿开始时间
    spec_verify_start_time: float = 0.0  # 推测解码验证开始时间
    spec_draft_extend_start_time: float = 0.0  # 推测解码草稿扩展开始时间

    # other
    transfer_speed_gb_s: float = 0.0  # 传输速度
    transfer_total_mb: float = 0.0  # 传输总量

    # Number of prefill retries for this request
    prefill_retry_count: int = 0  # 预填充重试计数

    def __getstate__(self) -> object:  # 序列化状态
        # send to detokenizer/tokenizer
        if not self.enable_metrics:  # 如果未启用指标
            return {}  # 返回空字典

        state = {
            "wait_queue_entry_time": self.wait_queue_entry_time,  # 等待队列进入时间
            "forward_entry_time": self.forward_entry_time,  # 前向进入时间
            "prefill_finished_time": self.prefill_finished_time,  # 预填充完成时间
            "diff_realtime_monotonic": global_diff_realtime_monotonic,  # 时间差
        }
        return state  # 返回状态

    def set_scheduler_recv_time(self, ts=None):  # 设置调度器接收时间
        calibrate_time_diff()  # 校准时间差
        ts = ts or time.perf_counter()  # 获取当前时间
        self.scheduler_recv_time = ts  # 设置接收时间

    def set_spec_draft_start_time(self, ts=None):  # 设置推测解码草稿开始时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.spec_draft_start_time = ts  # 设置开始时间

    def set_spec_draft_end_time(self, ts=None):  # 设置推测解码草稿结束时间
        ts = ts or time.perf_counter()  # 获取当前时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            stage = RequestStage.SPEC_DRAFT  # 获取草稿阶段
            self.trace_slice(stage, self.spec_draft_start_time, ts)  # 追踪切片

    def set_spec_verify_start_time(self, ts=None):  # 设置推测解码验证开始时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.spec_verify_start_time = ts  # 设置开始时间

    def set_spec_verify_end_time(  # 设置推测解码验证结束时间
        self,
        ts=None,
        num_correct_drafts: int = 0,  # 正确草稿数
        # FIXME: backward-compat alias, remove in next release.
        accepted_tokens: Optional[int] = None,  # 接受令牌数（向后兼容）
    ):
        if accepted_tokens is not None:  # 如果提供了接受令牌数
            num_correct_drafts = accepted_tokens  # 使用接受令牌数
        ts = ts or time.perf_counter()  # 获取当前时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            stage = RequestStage.SPEC_VERIFY  # 获取验证阶段
            self.trace_slice(  # 追踪切片
                stage,
                self.spec_verify_start_time,
                ts,
                {
                    "num_correct_drafts": num_correct_drafts,  # 正确草稿数
                    # FIXME: backward-compat alias, remove in next release.
                    "accepted_tokens": num_correct_drafts,  # 接受令牌数（向后兼容）
                },
            )

    def set_spec_draft_extend_start_time(self, ts=None):  # 设置推测解码草稿扩展开始时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.spec_draft_extend_start_time = ts  # 设置开始时间

    def set_spec_draft_extend_end_time(self, ts=None):  # 设置推测解码草稿扩展结束时间
        ts = ts or time.perf_counter()  # 获取当前时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            stage = RequestStage.SPEC_DRAFT_EXTEND  # 获取草稿扩展阶段
            self.trace_slice(stage, self.spec_draft_extend_start_time, ts)  # 追踪切片

    def set_run_batch_cpu_start_time(self, ts=None, attrs=None):  # 设置CPU端运行批次开始时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.run_batch_cpu_start_time = ts  # 设置开始时间

    def set_run_batch_cpu_end_time(self, ts=None, attrs=None):  # 设置CPU端运行批次结束时间
        ts = ts or time.perf_counter()  # 获取当前时间
        if self.run_batch_cpu_start_time > 0.0:  # 如果开始时间有效
            self.trace_slice(
                RequestStage.RUN_BATCH_CPU, self.run_batch_cpu_start_time, ts, attrs  # 追踪切片
            )
            self.run_batch_cpu_start_time = 0.0  # 重置开始时间

    def set_retract_time(self, ts=None):  # 设置撤回时间
        ts = ts or time.perf_counter()  # 获取当前时间
        # retract
        self.last_forward_entry_time = 0.0  # 重置前向进入时间
        self.last_prefill_finished_time = 0.0  # 重置预填充完成时间
        self.last_chunked_prefill_finish_time = 0.0  # 重置分块预填充完成时间
        self.last_decode_finish_time = 0.0  # 重置解码完成时间
        self.last_decode_scheduled_time = 0.0  # 重置解码调度时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.trace_event("retract", 1, convert_time_to_realtime_ns(ts))  # 追踪撤回事件

    def set_wait_queue_entry_time(self, ts=None):  # 设置等待队列进入时间
        ts = ts or time.perf_counter()  # 获取当前时间
        if self.wait_queue_entry_time == 0.0:  # 如果首次进入等待队列
            if self.enable_metrics or self.trace_ctx.tracing_enable:  # 如果需要记录
                if self.disagg_mode == DisaggregationMode.PREFILL:  # 预填充模式
                    stage = RequestStage.PREFILL_BOOTSTRAP  # 使用预填充引导阶段
                    slice_start_time = self.prefill_bootstrap_queue_entry_time  # 引导队列进入时间
                elif self.disagg_mode == DisaggregationMode.DECODE:  # 解码模式
                    stage = RequestStage.DECODE_TRANSFERRED  # 使用解码已传输阶段
                    slice_start_time = self.decode_transfer_queue_entry_time  # 传输队列进入时间
                else:  # 统一模式
                    stage = RequestStage.REQUEST_PROCESS  # 使用请求处理阶段
                    slice_start_time = self.scheduler_recv_time  # 调度器接收时间

                self.observe_per_stage_req_latency(stage, ts - slice_start_time)  # 观测阶段延迟
                self.trace_slice(stage, slice_start_time, ts)  # 追踪切片
        else:  # 重复进入等待队列（被撤回）
            self.set_retract_time(ts)  # 设置撤回时间

        self.wait_queue_entry_time = ts  # 更新等待队列进入时间

    def set_forward_entry_time(self, ts=None):  # 设置前向进入时间
        ts = ts or time.perf_counter()  # 获取当前时间
        if self.forward_entry_time == 0.0:  # 如果首次进入前向
            self.forward_entry_time = ts  # 设置前向进入时间
            self.last_forward_entry_time = ts  # 更新上次前向进入时间

            if self.enable_metrics:  # 如果启用指标
                self.metrics_collector.observe_queue_time(self.get_queueing_time())  # 观测队列时间

            if self.enable_metrics or self.trace_ctx.tracing_enable:  # 如果需要记录
                if self.disagg_mode == DisaggregationMode.DECODE:  # 解码模式
                    stage = RequestStage.DECODE_WAITING  # 使用解码等待阶段
                else:  # 其他模式
                    stage = RequestStage.PREFILL_WAITING  # 使用预填充等待阶段
                slice_start_time = self.wait_queue_entry_time  # 等待队列进入时间

                self.observe_per_stage_req_latency(stage, ts - slice_start_time)  # 观测阶段延迟
                self.trace_slice(stage, slice_start_time, ts)  # 追踪切片

                if self.disagg_mode == DisaggregationMode.DECODE:  # 解码模式
                    self.trace_ctx.trace_slice_start(  # 追踪解码前向切片开始
                        RequestStage.DECODE_FORWARD.stage_name,
                        RequestStage.DECODE_FORWARD.level,
                        convert_time_to_realtime_ns(ts),
                    )
                else:  # 其他模式
                    self.trace_ctx.trace_slice_start(  # 追踪预填充前向切片开始
                        RequestStage.PREFILL_FORWARD.stage_name,
                        RequestStage.PREFILL_FORWARD.level,
                        convert_time_to_realtime_ns(ts),
                    )
        elif self.last_forward_entry_time == 0.0:  # 如果有前向进入但无上次记录
            self.last_forward_entry_time = ts  # 更新上次前向进入时间

    def set_last_chunked_prefill_finish_time(self, ts=None):  # 设置上次分块预填充完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        last_time = self.last_chunked_prefill_finish_time  # 获取上次时间
        self.last_chunked_prefill_finish_time = ts  # 更新时间

        if last_time == 0.0:  # 如果没有上次记录
            last_time = self.last_forward_entry_time  # 使用上次前向进入时间

        stage = RequestStage.PREFILL_CHUNKED_FORWARD  # 获取分块预填充阶段
        self.observe_per_stage_req_latency(stage, ts - last_time)  # 观测延迟
        self.trace_slice(stage, last_time, ts)  # 追踪切片

    def set_prefill_finished_time(self, ts=None):  # 设置预填充完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        if self.prefill_finished_time == 0.0:  # 如果首次完成预填充
            self.prefill_finished_time = ts  # 设置预填充完成时间
            self.last_prefill_finished_time = ts  # 更新上次预填充完成时间

            stage = RequestStage.PREFILL_FORWARD  # 获取预填充前向阶段
            self.observe_per_stage_req_latency(stage, ts - self.last_forward_entry_time)  # 观测延迟

            if self.trace_ctx.tracing_enable:  # 如果追踪启用
                if self.last_chunked_prefill_finish_time > 0:  # 如果有分块预填充记录
                    self.trace_slice(  # 追踪分块预填充切片
                        RequestStage.PREFILL_CHUNKED_FORWARD,
                        self.last_chunked_prefill_finish_time,
                        ts,
                    )

                self.trace_ctx.trace_slice_end(  # 追踪预填充前向切片结束
                    stage.stage_name, stage.level, convert_time_to_realtime_ns(ts)
                )
                if (  # 如果统一模式且有解码调度记录
                    self.disagg_mode == DisaggregationMode.NULL
                    and self.last_decode_scheduled_time > 0
                ):
                    self.trace_ctx.trace_slice_start(  # 追踪解码前向切片开始
                        RequestStage.DECODE_FORWARD.stage_name,
                        RequestStage.DECODE_FORWARD.level,
                        convert_time_to_realtime_ns(ts),
                    )
        elif self.last_prefill_finished_time == 0.0:  # 如果有完成但无上次记录（撤回后）
            # retract
            self.last_prefill_finished_time = ts  # 更新上次预填充完成时间
            if self.last_chunked_prefill_finish_time > 0:  # 如果有分块预填充记录
                self.trace_slice(  # 追踪分块预填充切片
                    RequestStage.PREFILL_CHUNKED_FORWARD,
                    self.last_chunked_prefill_finish_time,
                    ts,
                )
            else:  # 否则
                self.trace_slice(  # 追踪预填充前向切片
                    RequestStage.PREFILL_FORWARD, self.last_forward_entry_time, ts
                )

    def set_last_decode_finish_time(self, ts=None):  # 设置上次解码完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        last_time = self.last_decode_finish_time  # 获取上次时间
        self.last_decode_finish_time = ts  # 更新时间

        if self.enable_metrics or self.trace_ctx.tracing_enable:  # 如果需要记录
            if last_time == 0.0:  # 如果没有上次记录
                if self.disagg_mode == DisaggregationMode.DECODE:  # 解码模式
                    last_time = self.decode_prebuilt_finish_time  # 使用预构建完成时间
                else:  # 其他模式
                    if (  # 如果解码调度时间早于预填充完成时间
                        self.last_decode_scheduled_time
                        < self.last_prefill_finished_time
                    ):
                        last_time = self.last_prefill_finished_time  # 使用预填充完成时间
                    else:  # 否则
                        last_time = self.last_decode_scheduled_time  # 使用解码调度时间
            stage = RequestStage.DECODE_LOOP  # 获取解码循环阶段
            self.observe_per_stage_req_latency(stage, ts - last_time)  # 观测延迟
            attrs = {"decode_ct": self.decode_ct}  # 创建属性
            self.trace_slice(stage, last_time, ts, attrs)  # 追踪切片
            self.decode_ct += 1  # 递增解码计数

    def set_last_scheduled_time(self, forward_mode: ForwardMode, ts=None, attrs=None):  # 设置上次调度时间
        ts = ts or time.perf_counter()  # 获取当前时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            if (  # 如果统一模式、解码模式、首次解码调度且有预填充记录
                self.disagg_mode == DisaggregationMode.NULL
                and forward_mode.is_decode()
                and self.last_decode_scheduled_time == 0.0
                and self.last_prefill_finished_time > 0
            ):
                self.trace_slice(  # 追踪解码等待切片
                    RequestStage.DECODE_WAITING, self.last_prefill_finished_time, ts
                )
                self.trace_ctx.trace_slice_start(  # 追踪解码前向切片开始
                    RequestStage.DECODE_FORWARD.stage_name,
                    RequestStage.DECODE_FORWARD.level,
                    convert_time_to_realtime_ns(ts),
                )
                self.last_decode_finish_time = ts  # 设置解码完成时间

            self.trace_ctx.trace_event(  # 追踪调度事件
                "schedule", 3, convert_time_to_realtime_ns(ts), attrs
            )

        if forward_mode.is_decode():  # 如果是解码模式
            self.last_decode_scheduled_time = ts  # 更新解码调度时间

    def set_completion_time(self, ts=None):  # 设置完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.completion_time = ts  # 设置完成时间

        if self.trace_ctx.tracing_enable:  # 如果追踪启用
            self.trace_ctx.abort()  # 中止追踪

    def compute_and_observe_kv_transfer_metrics(  # 计算并观测KV传输指标
        self,
        transfer_metric: KVTransferMetric,  # KV传输指标
    ) -> Optional[dict]:
        """Compute KV transfer metrics and observe them via the metrics collector.

        Returns a dict with latency_ms, total_mb, speed_gb_s if computable, else None.
        """
        result = {}  # 结果字典
        if transfer_metric.transfer_total_bytes is None:  # 如果传输总字节数为None
            return result if result else None  # 返回结果或None

        # Transfer latency, size, and speed
        if transfer_metric.transfer_latency_s is not None:  # 如果传输延迟有效
            transfer_latency_s = transfer_metric.transfer_latency_s  # 使用指标中的延迟
        else:  # 否则从时间戳计算
            if self.prefill_transfer_queue_entry_time <= 0 or self.completion_time <= 0:  # 检查时间戳
                return result if result else None  # 无法计算
            # Note: This only capture the last chunk time
            transfer_latency_s = (  # 计算传输延迟
                self.completion_time - self.prefill_transfer_queue_entry_time
            )

        if transfer_latency_s > 0:  # 如果延迟大于0
            latency_ms = transfer_latency_s * 1000  # 转换为毫秒

            total_bytes = transfer_metric.transfer_total_bytes  # 获取总字节数
            total_mb = total_bytes / (1024 * 1024)  # 转换为兆字节
            self.transfer_total_mb = total_mb  # 保存

            speed_gb_s = 0.0  # 传输速度初始化
            if transfer_latency_s > 0:  # 如果延迟大于0
                speed_gb_s = (total_mb / 1024) / transfer_latency_s  # 计算速度
                self.transfer_speed_gb_s = speed_gb_s  # 保存

            result["latency_ms"] = latency_ms  # 记录延迟
            result["total_mb"] = total_mb  # 记录总量
            result["speed_gb_s"] = speed_gb_s  # 记录速度

            if self.enable_metrics:  # 如果启用指标
                self.metrics_collector.observe_kv_transfer_metrics(  # 观测KV传输指标
                    latency_ms=latency_ms,
                    total_mb=total_mb,
                    speed_gb_s=speed_gb_s,
                )

        # Bootstrap and alloc durations
        if (  # 如果可以计算引导和分配时间
            self.prefill_bootstrap_queue_entry_time > 0
            and self.bootstrap_done_time > 0
            and self.wait_queue_entry_time > 0
        ):
            bootstrap_ms = (  # 计算引导时间
                self.bootstrap_done_time - self.prefill_bootstrap_queue_entry_time
            ) * 1000
            alloc_ms = (self.wait_queue_entry_time - self.bootstrap_done_time) * 1000  # 计算分配时间

            result["bootstrap_ms"] = bootstrap_ms  # 记录引导时间
            result["alloc_ms"] = alloc_ms  # 记录分配时间

            if self.enable_metrics:  # 如果启用指标
                self.metrics_collector.observe_kv_transfer_bootstrap(  # 观测KV传输引导时间
                    bootstrap_ms=bootstrap_ms,
                    alloc_ms=alloc_ms,
                )

        return result if result else None  # 返回结果或None

    def set_quick_finish_time(self, ts=None):  # 设置快速完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.set_completion_time(ts)  # 设置完成时间
        self.forward_entry_time = ts  # 设置前向进入时间

    def set_prefill_bootstrap_queue_entry_time(self, ts=None):  # 设置预填充引导队列进入时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.prefill_bootstrap_queue_entry_time = ts  # 设置进入时间

        stage = RequestStage.PREFILL_PREPARE  # 获取预填充准备阶段
        self.observe_per_stage_req_latency(stage, ts - self.scheduler_recv_time)  # 观测延迟
        self.trace_slice(stage, self.scheduler_recv_time, ts)  # 追踪切片

    def set_prefill_transfer_queue_entry_time(self, ts=None):  # 设置预填充传输队列进入时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.prefill_transfer_queue_entry_time = ts  # 设置进入时间

    def set_prefill_kv_transfer_finish_time(self, ts=None):  # 设置预填充KV传输完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.prefill_kv_transfer_finish_time = ts  # 设置完成时间

        stage = RequestStage.PREFILL_TRANSFER_KV_CACHE  # 获取KV缓存传输阶段
        self.observe_per_stage_req_latency(  # 观测延迟
            stage, ts - self.prefill_transfer_queue_entry_time
        )
        self.trace_slice(stage, self.prefill_transfer_queue_entry_time, ts)  # 追踪切片

    def set_decode_prealloc_queue_entry_time(self, ts=None):  # 设置解码预分配队列进入时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.decode_prealloc_queue_entry_time = ts  # 设置进入时间

        stage = RequestStage.DECODE_PREPARE  # 获取解码准备阶段
        self.observe_per_stage_req_latency(stage, ts - self.scheduler_recv_time)  # 观测延迟
        self.trace_slice(stage, self.scheduler_recv_time, ts)  # 追踪切片

    def set_decode_transfer_queue_entry_time(self, ts=None):  # 设置解码传输队列进入时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.decode_transfer_queue_entry_time = ts  # 设置进入时间

        stage = RequestStage.DECODE_BOOTSTRAP  # 获取解码引导阶段
        self.observe_per_stage_req_latency(  # 观测延迟
            stage, ts - self.decode_prealloc_queue_entry_time
        )
        self.trace_slice(stage, self.decode_prealloc_queue_entry_time, ts)  # 追踪切片

        if self.enable_metrics and self.bootstrap_done_time > 0:  # 如果启用指标且有引导完成时间
            bootstrap_ms = (  # 计算引导时间
                self.bootstrap_done_time - self.decode_prealloc_queue_entry_time
            ) * 1000
            alloc_ms = (ts - self.bootstrap_done_time) * 1000  # 计算分配时间
            self.metrics_collector.observe_kv_transfer_bootstrap(  # 观测引导时间
                bootstrap_ms=bootstrap_ms,
                alloc_ms=alloc_ms,
            )

    def set_bootstrap_done_time(self, ts=None):  # 设置引导完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        if self.bootstrap_done_time == 0.0:  # 如果尚未设置
            self.bootstrap_done_time = ts  # 设置引导完成时间

    def set_decode_prebuilt_finish_time(self, ts=None):  # 设置解码预构建完成时间
        ts = ts or time.perf_counter()  # 获取当前时间
        self.decode_prebuilt_finish_time = ts  # 设置完成时间

        stage = RequestStage.DECODE_FAKE_OUTPUT  # 获取伪输出阶段
        self.observe_per_stage_req_latency(stage, ts - self.last_forward_entry_time)  # 观测延迟
        self.trace_slice(stage, self.last_forward_entry_time, ts)  # 追踪切片

    def get_queueing_time(self) -> float:  # 获取队列等待时间
        return self.forward_entry_time - self.wait_queue_entry_time  # 返回等待时间

    def convert_to_duration(self) -> str:  # 转换为持续时间字符串
        if self.disagg_mode == DisaggregationMode.NULL:  # 统一模式
            queue_duration = self.duration_between(  # 计算队列持续时间
                self.wait_queue_entry_time, self.forward_entry_time
            )
            forward_duration = self.duration_between(  # 计算前向持续时间
                self.forward_entry_time, self.completion_time
            )

            if SGLANG_TEST_REQUEST_TIME_STATS:  # 如果启用测试
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, entry_time={self.format_wallclock(self.wait_queue_entry_time)}"  # 返回格式化字符串
        elif self.disagg_mode == DisaggregationMode.PREFILL:  # 预填充模式
            bootstrap_queue_duration = self.duration_between(  # 计算引导队列持续时间
                self.prefill_bootstrap_queue_entry_time, self.wait_queue_entry_time
            )
            queue_duration = self.duration_between(  # 计算队列持续时间
                self.wait_queue_entry_time, self.forward_entry_time
            )
            forward_duration = self.duration_between(  # 计算前向持续时间
                self.forward_entry_time, self.completion_time
            )

            if SGLANG_TEST_REQUEST_TIME_STATS:  # 如果启用测试
                if self.wait_queue_entry_time > 0:  # 如果有等待队列时间
                    assert (
                        bootstrap_queue_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"bootstrap_queue_duration={bootstrap_queue_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            # Break down bootstrap_queue_duration into sub-phases
            if self.bootstrap_done_time > 0:  # 如果有引导完成时间
                bootstrap_duration = self.duration_between(  # 计算引导时间
                    self.prefill_bootstrap_queue_entry_time, self.bootstrap_done_time
                )
                alloc_wait_duration = self.duration_between(  # 计算分配等待时间
                    self.bootstrap_done_time, self.wait_queue_entry_time
                )
                if SGLANG_TEST_REQUEST_TIME_STATS:  # 如果启用测试
                    assert (
                        bootstrap_duration >= 0 and alloc_wait_duration >= 0
                    ), f"bootstrap_duration={bootstrap_duration} < 0 or alloc_wait_duration={alloc_wait_duration} < 0"
                bootstrap_fields = (  # 构建引导字段
                    f"bootstrap_duration={self.format_duration(bootstrap_duration)}, "
                    f"alloc_wait_duration={self.format_duration(alloc_wait_duration)}, "
                )
            else:  # 没有引导完成时间
                bootstrap_fields = f"bootstrap_queue_duration={self.format_duration(bootstrap_queue_duration)}, "  # 使用引导队列持续时间

            return (  # 返回格式化字符串
                f"{bootstrap_fields}"
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"entry_time={self.format_wallclock(self.prefill_bootstrap_queue_entry_time)}, "
                f"transfer_speed={self.transfer_speed_gb_s:.2f} GB/s, "
                f"transfer_total={self.transfer_total_mb:.2f} MB, "
                f"#retries={self.prefill_retry_count}"
            )
        elif self.disagg_mode == DisaggregationMode.DECODE:  # 解码模式
            prealloc_duration = self.duration_between(  # 计算预分配持续时间
                self.decode_prealloc_queue_entry_time,
                self.decode_transfer_queue_entry_time,
            )
            transfer_duration = self.duration_between(  # 计算传输持续时间
                self.decode_transfer_queue_entry_time,
                self.wait_queue_entry_time,
            )
            queue_duration = self.duration_between(  # 计算队列持续时间
                self.wait_queue_entry_time,
                self.forward_entry_time,
            )
            forward_duration = self.duration_between(  # 计算前向持续时间
                self.forward_entry_time,
                self.completion_time,
            )

            if SGLANG_TEST_REQUEST_TIME_STATS:  # 如果启用测试
                if self.wait_queue_entry_time > 0:  # 如果有等待队列时间
                    assert (
                        prealloc_duration >= 0
                        and transfer_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0. {self=}"

            # Break down prealloc_duration into sub-phases
            if self.bootstrap_done_time > 0:  # 如果有引导完成时间
                bootstrap_duration = self.duration_between(  # 计算引导时间
                    self.decode_prealloc_queue_entry_time, self.bootstrap_done_time
                )
                alloc_wait_duration = self.duration_between(  # 计算分配等待时间
                    self.bootstrap_done_time, self.decode_transfer_queue_entry_time
                )
                if SGLANG_TEST_REQUEST_TIME_STATS:  # 如果启用测试
                    assert (
                        bootstrap_duration >= 0 and alloc_wait_duration >= 0
                    ), f"bootstrap_duration={bootstrap_duration} < 0 or alloc_wait_duration={alloc_wait_duration} < 0"
                prealloc_fields = (  # 构建预分配字段
                    f"bootstrap_duration={self.format_duration(bootstrap_duration)}, "
                    f"alloc_wait_duration={self.format_duration(alloc_wait_duration)}, "
                )
            else:  # 没有引导完成时间
                prealloc_fields = f"prealloc_queue_duration={self.format_duration(prealloc_duration)}, "  # 使用预分配队列持续时间

            return (  # 返回格式化字符串
                f"{prealloc_fields}"
                f"transfer_duration={self.format_duration(transfer_duration)}, "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"entry_time={self.format_wallclock(self.decode_prealloc_queue_entry_time)}"
            )
        else:  # 未知模式
            return "Unknown Time Stats"  # 返回未知

    def convert_to_output_meta_info(self):  # 转换为输出元信息
        meta_data = {}  # 元数据字典
        if self.forward_entry_time > 0.0:  # 如果前向进入时间有效
            meta_data["forward_entry_time"] = convert_time_to_realtime(  # 前向进入时间
                self.forward_entry_time
            )
        if self.prefill_finished_time > 0.0:  # 如果预填充完成时间有效
            meta_data["prefill_finished_time"] = convert_time_to_realtime(  # 预填充完成时间
                self.prefill_finished_time
            )
        meta_data.update(  # 更新队列时间
            {
                "queue_time": self.get_queueing_time(),  # 队列等待时间
            }
        )
        return meta_data  # 返回元数据

    def format_duration(self, duration: float) -> str:  # 格式化持续时间为毫秒字符串
        return f"{duration * 1e3:.2f}ms"  # 返回格式化字符串

    def duration_between(self, start: float, end: float) -> float:  # 计算两个时间点之间的持续时间
        if start <= 0 or end <= 0:  # 如果时间无效
            return 0.0  # 返回0
        return end - start  # 返回持续时间

    @staticmethod
    def format_wallclock(perf_counter_time: float) -> str:  # 格式化单调时间为真实时间字符串
        return f"{convert_time_to_realtime(perf_counter_time):.3f}"  # 返回格式化字符串


def set_schedule_time_batch(batch: ScheduleBatch):  # 设置调度时间批次（仅用于追踪）
    # only for tracing
    if not get_global_tracing_enabled():  # 如果未启用追踪
        return  # 直接返回

    ts = time.perf_counter()  # 获取当前时间
    bid = uuid.uuid4().hex[:8]  # 生成批次ID
    _attrs = {"bid": bid, "batch_size": len(batch.reqs)}  # 创建属性
    if batch.forward_mode.is_decode():  # 如果是解码模式
        _attrs["forward_mode"] = "decode"  # 设置前向模式为decode
    elif batch.forward_mode.is_prefill():  # 如果是预填充模式
        _attrs["forward_mode"] = "prefill"  # 设置前向模式为prefill
    elif batch.forward_mode.is_prebuilt():  # 如果是预构建模式
        _attrs["forward_mode"] = "prebuilt"  # 设置前向模式为prebuilt

    for req in batch.reqs:  # 遍历批次中的请求
        req.time_stats.set_last_scheduled_time(batch.forward_mode, ts, _attrs)  # 设置调度时间


def set_time_batch(  # 设置时间批次
    reqs: List[Any],  # 请求列表
    set_func: str,  # 设置函数名
    trace_only: bool = False,  # 是否仅追踪
    attrs: Optional[Dict[str, Any]] = None,  # 属性
):
    if reqs is None or len(reqs) == 0:  # 如果请求列表为空
        return  # 直接返回
    if trace_only and not get_global_tracing_enabled():  # 如果仅追踪且未启用
        return  # 直接返回

    ts = time.perf_counter()  # 获取当前时间
    for req in reqs:  # 遍历请求
        method = getattr(req.time_stats, set_func)  # 获取设置方法
        if attrs is None:  # 如果没有属性
            method(ts)  # 调用方法
        else:  # 如果有属性
            method(ts, attrs)  # 带属性调用方法
