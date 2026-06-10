# Prometheus指标收集器模块
# 本模块提供调度器、分词器、存储和Radix缓存的Prometheus指标收集功能
# 包含队列计数、内存池使用率、推测解码、PD分离、语法处理等指标

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
"""Utilities for Prometheus Metrics Collection."""

from __future__ import annotations  # 启用延迟类型注解求值

import dataclasses  # 导入数据类工具
import logging  # 导入日志模块
import os  # 导入操作系统模块
import time  # 导入时间模块
from collections import Counter  # 导入计数器
from dataclasses import dataclass, field  # 导入数据类装饰器和字段
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union  # 导入类型提示

from sglang.srt.disaggregation.utils import DisaggregationMode  # 导入分离模式
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # 导入前向模式
from sglang.srt.observability.utils import exponential_buckets, generate_buckets  # 导入桶生成工具
from sglang.srt.server_args import ServerArgs  # 导入服务器参数
from sglang.srt.utils import get_bool_env_var  # 导入布尔环境变量工具
from sglang.srt.utils.gauge_histogram import GaugeHistogram  # 导入仪表直方图

if TYPE_CHECKING:  # 类型检查时导入
    from prometheus_client import Gauge  # 导入Prometheus仪表

    from sglang.srt.managers.schedule_batch import Req  # 导入请求类

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")  # 是否测试请求时间统计

logger = logging.getLogger(__name__)  # 创建日志记录器


@dataclass
class QueueCount:  # 队列计数类，包含总数和可选的按优先级分解
    """Holds both the total count and optional per-priority breakdown for a queue."""

    total: int = 0  # 总数
    by_priority: Optional[Dict[int, int]] = None  # 按优先级分解的计数

    @classmethod
    def from_reqs(cls, reqs: List[Req], enable_priority_scheduling: bool = False):  # 从请求列表创建队列计数
        # NOTE: If requests have priority=None (no --default-priority-value set),
        # Counter will produce {None: N}, resulting in priority="None" Prometheus labels.
        # Set --default-priority-value when enabling priority scheduling to avoid this.
        by_priority = (  # 按优先级计算计数
            dict(Counter(req.priority for req in reqs))  # 统计每个优先级的请求数
            if enable_priority_scheduling  # 仅在启用优先级调度时
            else None  # 否则不分解
        )
        return cls(total=len(reqs), by_priority=by_priority)  # 创建并返回队列计数


@dataclass
class SchedulerStats:  # 调度器统计数据类
    # Basics
    num_running_reqs: QueueCount = field(default_factory=QueueCount)  # 运行中请求数
    num_queue_reqs: QueueCount = field(default_factory=QueueCount)  # 队列中请求数
    num_grammar_queue_reqs: int = 0  # 语法队列中请求数
    gen_throughput: float = 0.0  # 生成吞吐量
    cache_hit_rate: float = 0.0  # 缓存命中率
    decode_sum_seq_lens: int = 0  # 解码序列长度总和

    # Memory pool usage ratios (0.0–1.0).
    # Each pool tracks: used = total - available - evictable, usage = used / total.
    #
    # token_usage:      max(full, swa, mamba) — the bottleneck across all pools.
    #                   FIXME: misleadingly named "token_usage"; rename requires API deprecation.
    # full_token_usage: full-attention KV cache pool usage (always active).
    # swa_token_usage:  sliding-window attention KV cache pool usage (hybrid SWA models only, e.g. Gemma2).
    # mamba_usage:      Mamba SSM state pool usage (hybrid SSM models only, e.g. Jamba).
    token_usage: float = 0.0  # 令牌使用率（所有池的最大值）
    full_token_usage: float = 0.0  # 全注意力KV缓存池使用率
    swa_token_usage: float = 0.0  # 滑动窗口注意力KV缓存池使用率
    mamba_usage: float = 0.0  # Mamba SSM状态池使用率

    # Absolute token counts for the full-attention KV cache pool.
    # Invariant: kv_available_tokens + kv_evictable_tokens + kv_used_tokens <= max_total_num_tokens
    # (the gap accounts for protected/session-held tokens not exposed here).
    # max_total_num_tokens is emitted once at startup via emit_constants.
    #
    # kv_available_tokens:  free (unallocated) slots in the pool.
    # kv_evictable_tokens:  slots holding radix-cached KV data that can be evicted for new requests.
    # kv_used_tokens:       actively used slots (locked by running requests). Equals full_num_used.
    # num_used_tokens:      max(full_num_used, swa_num_used) for hybrid-SWA models, else full_num_used.
    #                       Does NOT include the mamba pool.
    num_used_tokens: int = 0  # 已使用令牌数
    kv_available_tokens: int = 0  # KV缓存可用令牌数
    kv_evictable_tokens: int = 0  # KV缓存可驱逐令牌数
    kv_used_tokens: int = 0  # KV缓存已使用令牌数

    swa_available_tokens: int = 0  # SWA可用令牌数
    swa_evictable_tokens: int = 0  # SWA可驱逐令牌数
    swa_used_tokens: int = 0  # SWA已使用令牌数
    mamba_available_tokens: int = 0  # Mamba可用状态数
    mamba_evictable_tokens: int = 0  # Mamba可驱逐状态数
    mamba_used_tokens: int = 0  # Mamba已使用状态数

    # Speculative decoding
    spec_accept_length: float = 0.0  # 推测解码接受长度
    spec_accept_rate: float = 0.0  # 推测解码接受率
    # Adaptive speculative decoding (currently active tier).
    spec_num_steps: int = 0  # 推测解码步数
    spec_num_draft_tokens: int = 0  # 推测解码草稿令牌数

    # Retract
    num_retracted_reqs: int = 0  # 被撤回的请求数
    num_paused_reqs: int = 0  # 被暂停的请求数

    # PD disaggregation
    num_prefill_bootstrap_queue_reqs: QueueCount = field(default_factory=QueueCount)  # 预填充引导队列请求数
    num_prefill_inflight_queue_reqs: QueueCount = field(default_factory=QueueCount)  # 预填充飞行中队列请求数
    num_decode_prealloc_queue_reqs: QueueCount = field(default_factory=QueueCount)  # 解码预分配队列请求数
    num_decode_transfer_queue_reqs: QueueCount = field(default_factory=QueueCount)  # 解码传输队列请求数
    kv_transfer_speed_gb_s: float = 0.0  # KV传输速度
    kv_transfer_latency_ms: float = 0.0  # KV传输延迟
    pending_prealloc_token_usage: float = 0.0  # 待预分配令牌使用率

    # Utilization
    utilization: float = 0.0  # 利用率
    fwd_occupancy: float = float("nan")  # 前向传播GPU占用率

    # Scheduler policy
    new_token_ratio: float = 0.0  # 新令牌比率

    # CUDA graph
    is_cuda_graph: int = 0  # 是否使用CUDA图

    # LoRA pool metrics
    lora_pool_slots_used: int = 0  # LoRA池已使用槽位数
    lora_pool_slots_total: int = 0  # LoRA池总槽位数
    lora_pool_utilization: float = 0.0  # LoRA池利用率

    # HiCache metrics
    hicache_host_used_tokens: int = 0  # 主机KV缓存已使用令牌数
    hicache_host_total_tokens: int = 0  # 主机KV缓存总令牌数

    # Streaming session metrics
    num_streaming_sessions: int = 0  # 流式会话数
    streaming_session_held_tokens: int = 0  # 流式会话持有令牌数

    # Routing key metrics
    num_unique_running_routing_keys: int = 0  # 唯一运行路由键数
    routing_key_running_req_counts: List[int] = field(default_factory=list)  # 路由键运行请求计数
    routing_key_all_req_counts: List[int] = field(default_factory=list)  # 路由键全部请求计数


ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS = [1, 2, 3, 5, 7, 10, 20, 50, 100, 200]  # 路由键请求计数桶边界


def compute_routing_key_stats(routing_keys: List[Optional[str]]) -> tuple:  # 计算路由键统计信息
    """Returns (num_unique_keys, per_key_counts)."""  # 返回唯一键数和每个键的计数
    from collections import Counter  # 导入计数器

    key_counts = Counter(k for k in routing_keys if k is not None)  # 统计非None键的计数
    return len(key_counts), list(key_counts.values())  # 返回唯一键数和计数列表


@dataclass
class DPCooperationInfo:  # DP协作信息
    # Users can derive that, except for cases with idle, num_decode_ranks=world_size-num_prefill_ranks
    # We do not provide `num_decode_ranks` to avoid cardinality explosion.
    num_prefill_ranks: int  # 预填充排名数

    @staticmethod
    def create(forward_modes: List[int]):  # 从前向模式列表创建DP协作信息
        return DPCooperationInfo(
            # Count ranks that are doing any extend-like work.
            # With overlap scheduling, prefill can appear as MIXED rather than EXTEND.
            num_prefill_ranks=sum(  # 统计执行扩展操作的排名数
                1 for mode in forward_modes if ForwardMode(mode).is_extend()  # 检查是否为扩展模式
            ),
        )

    def to_labels(self):  # 转换为标签字典
        return dataclasses.asdict(self)  # 使用asdict转换


# Role keys used by ServerArgs.stat_loggers to look up collector overrides.
# Embedded-use callers (e.g. Ray Serve LLM) pass {"scheduler": MyClass, ...} on
# ServerArgs and the five collector instantiation sites pick the right class.
STAT_LOGGER_ROLE_SCHEDULER = "scheduler"  # 调度器角色键
STAT_LOGGER_ROLE_TOKENIZER = "tokenizer"  # 分词器角色键
STAT_LOGGER_ROLE_STORAGE = "storage"  # 存储角色键
STAT_LOGGER_ROLE_RADIX_CACHE = "radix_cache"  # Radix缓存角色键
STAT_LOGGER_ROLE_EXPERT_DISPATCH = "expert_dispatch"  # 专家分发角色键


def resolve_collector_class(  # 解析收集器类
    server_args: Optional["ServerArgs"], role: str, default_cls: type  # 服务器参数、角色、默认类
) -> type:
    """Return the subclass registered for `role` on `server_args.stat_loggers`,
    or `default_cls` if none is registered. Tolerates `server_args=None` and
    `stat_loggers=None`."""
    if server_args is None:  # 如果服务器参数为None
        return default_cls  # 返回默认类
    stat_loggers = getattr(server_args, "stat_loggers", None)  # 获取统计日志器
    if not stat_loggers:  # 如果没有统计日志器
        return default_cls  # 返回默认类
    return stat_loggers.get(role, default_cls)  # 返回指定角色的类或默认类


class _StatLoggerDIMixin:  # 统计日志器依赖注入混入类
    """Shared DI override hooks for all *MetricsCollector classes.

    Subclasses (e.g. a Ray-backed wrapper) replace these class attributes with
    classes that mirror the prometheus_client API but emit through a different
    backend. ``None`` keeps the prometheus_client default.
    """

    _counter_cls = None  # 计数器类覆盖
    _gauge_cls = None  # 仪表类覆盖
    _histogram_cls = None  # 直方图类覆盖
    _summary_cls = None  # 摘要类覆盖


@dataclass(kw_only=True, frozen=True, slots=True)
class SchedulerMetricsCollectorContext:  # 调度器指标收集器上下文
    enable_metrics: bool  # 是否启用指标
    is_stats_logging_rank: bool  # 是否为统计日志排名
    current_scheduler_metrics_enabled: bool  # 当前调度器指标是否启用
    enable_kv_cache_events: bool  # 是否启用KV缓存事件
    collector: Optional["SchedulerMetricsCollector"]  # 指标收集器实例


class SchedulerMetricsCollector(_StatLoggerDIMixin):  # 调度器指标收集器

    def __init__(  # 初始化调度器指标收集器
        self,
        labels: Dict[str, str],  # 标签字典
        enable_lora: bool = False,  # 是否启用LoRA
        enable_hierarchical_cache: bool = False,  # 是否启用分层缓存
        enable_streaming_session: bool = False,  # 是否启用流式会话
        server_args: Optional["ServerArgs"] = None,  # 服务器参数
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter as _PromCounter  # 导入Prometheus计数器
        from prometheus_client import Gauge as _PromGauge  # 导入Prometheus仪表
        from prometheus_client import Histogram as _PromHistogram  # 导入Prometheus直方图
        from prometheus_client import Summary as _PromSummary  # 导入Prometheus摘要

        Counter = self._counter_cls or _PromCounter  # 使用覆盖或默认计数器类
        Gauge = self._gauge_cls or _PromGauge  # 使用覆盖或默认仪表类
        Histogram = self._histogram_cls or _PromHistogram  # 使用覆盖或默认直方图类
        Summary = self._summary_cls or _PromSummary  # 使用覆盖或默认摘要类

        self.labels = labels  # 保存标签
        self.enable_lora = enable_lora  # 保存LoRA启用状态
        self.enable_hierarchical_cache = enable_hierarchical_cache  # 保存分层缓存启用状态
        self.enable_streaming_session = enable_streaming_session  # 保存流式会话启用状态
        self.last_log_time = time.perf_counter()  # 记录上次日志时间
        self._known_priorities: Set[int] = set()  # 已知优先级集合

        # =================================================================
        # Basics
        # =================================================================
        self.num_running_reqs = Gauge(  # 运行中请求数仪表
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_queue_reqs = Gauge(  # 队列中请求数仪表
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_grammar_queue_reqs = Gauge(  # 语法队列请求数仪表
            name="sglang:num_grammar_queue_reqs",
            documentation="The number of requests in the grammar waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.gen_throughput = Gauge(  # 生成吞吐量仪表
            name="sglang:gen_throughput",
            documentation="The generation throughput (token/s).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cache_hit_rate = Gauge(  # 缓存命中率仪表
            name="sglang:cache_hit_rate",
            documentation="The prefix cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.decode_sum_seq_lens = Gauge(  # 解码序列长度总和仪表
            name="sglang:decode_sum_seq_lens",
            documentation="The sum of all sequence lengths in decode.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Memory pool usage ratios
        # =================================================================
        self.token_usage = Gauge(  # 令牌使用率仪表
            name="sglang:token_usage",
            documentation="The token usage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.full_token_usage = Gauge(  # 全注意力令牌使用率仪表
            name="sglang:full_token_usage",
            documentation="The token usage for full attention layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_token_usage = Gauge(  # SWA令牌使用率仪表
            name="sglang:swa_token_usage",
            documentation="The token usage for SWA layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_usage = Gauge(  # Mamba使用率仪表
            name="sglang:mamba_usage",
            documentation="The token usage for Mamba layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Absolute token counts
        # =================================================================
        self.num_used_tokens = Gauge(  # 已使用令牌数仪表
            name="sglang:num_used_tokens",
            documentation="The number of used tokens.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_available_tokens = Gauge(  # KV可用令牌数仪表
            name="sglang:kv_available_tokens",
            documentation="Number of free token slots in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_evictable_tokens = Gauge(  # KV可驱逐令牌数仪表
            name="sglang:kv_evictable_tokens",
            documentation="Number of evictable (radix-cached) token slots in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_used_tokens = Gauge(  # KV已使用令牌数仪表
            name="sglang:kv_used_tokens",
            documentation="Number of actively used token slots in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_available_tokens = Gauge(  # SWA可用令牌数仪表
            name="sglang:swa_available_tokens",
            documentation="Number of free token slots in the SWA pool (hybrid-SWA only).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_evictable_tokens = Gauge(  # SWA可驱逐令牌数仪表
            name="sglang:swa_evictable_tokens",
            documentation="Number of evictable (radix-cached) token slots in the SWA pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_used_tokens = Gauge(  # SWA已使用令牌数仪表
            name="sglang:swa_used_tokens",
            documentation="Number of actively used token slots in the SWA pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_available_tokens = Gauge(  # Mamba可用状态数仪表
            name="sglang:mamba_available_tokens",
            documentation="Number of free state slots in the mamba SSM pool (hybrid-SSM only).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_evictable_tokens = Gauge(  # Mamba可驱逐状态数仪表
            name="sglang:mamba_evictable_tokens",
            documentation="Number of evictable (radix-cached) state slots in the mamba SSM pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_used_tokens = Gauge(  # Mamba已使用状态数仪表
            name="sglang:mamba_used_tokens",
            documentation="Number of actively used state slots in the mamba SSM pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Speculative decoding
        # =================================================================
        self.spec_accept_length = Gauge(  # 推测解码接受长度仪表
            name="sglang:spec_accept_length",
            documentation="Mean acceptance length of speculative decoding (accepted drafts + bonus token per forward).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.spec_accept_rate = Gauge(  # 推测解码接受率仪表
            name="sglang:spec_accept_rate",
            documentation="Speculative acceptance rate (`accepted drafts / proposed drafts` in batch).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.spec_num_steps = Gauge(  # 推测解码步数仪表
            name="sglang:spec_num_steps",
            documentation="Currently active speculative_num_steps.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.spec_num_draft_tokens = Gauge(  # 推测解码草稿令牌数仪表
            name="sglang:spec_num_draft_tokens",
            documentation="Currently active speculative_num_draft_tokens (decouples from steps under topk>1).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Retract
        # =================================================================
        # TODO maybe remove this old gauge in favor of the new counter
        self.num_retracted_reqs = Gauge(  # 被撤回请求数仪表
            name="sglang:num_retracted_reqs",
            documentation="The number of retracted requests.",
            labelnames=labels.keys(),
        )
        self.num_retracted_reqs_total = Counter(  # 被撤回请求总数计数器
            # The name is `requests` instead of `reqs` to avoid dup name error
            name="sglang:num_retracted_requests_total",
            documentation="Total number of retracted requests.",
            labelnames=labels.keys(),
        )
        self.num_retracted_input_tokens_total = Counter(  # 被撤回输入令牌总数计数器
            name="sglang:num_retracted_input_tokens_total",
            documentation="Total number of retracted input tokens.",
            labelnames=labels.keys(),
        )
        self.num_retracted_output_tokens_total = Counter(  # 被撤回输出令牌总数计数器
            name="sglang:num_retracted_output_tokens_total",
            documentation="Total number of retracted output tokens.",
            labelnames=labels.keys(),
        )
        self.num_paused_reqs = Gauge(  # 被暂停请求数仪表
            name="sglang:num_paused_reqs",
            documentation="The number of paused requests by async weight sync.",
            labelnames=labels.keys(),
        )

        # =================================================================
        # PD disaggregation
        # =================================================================
        self.num_prefill_bootstrap_queue_reqs = Gauge(  # 预填充引导队列请求数仪表
            name="sglang:num_prefill_bootstrap_queue_reqs",
            documentation="The number of requests in the prefill bootstrap queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_prefill_inflight_queue_reqs = Gauge(  # 预填充飞行中队列请求数仪表
            name="sglang:num_prefill_inflight_queue_reqs",
            documentation="The number of requests in the prefill inflight queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_prealloc_queue_reqs = Gauge(  # 解码预分配队列请求数仪表
            name="sglang:num_decode_prealloc_queue_reqs",
            documentation="The number of requests in the decode prealloc queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_transfer_queue_reqs = Gauge(  # 解码传输队列请求数仪表
            name="sglang:num_decode_transfer_queue_reqs",
            documentation="The number of requests in the decode transfer queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_transfer_speed_gb_s = Histogram(  # KV传输速度直方图
            name="sglang:kv_transfer_speed_gb_s",
            documentation="Histogram of KV cache transfer speed in GB/s.",
            labelnames=labels.keys(),
            buckets=(0.1, 0.5, 1, 5, 10, 25, 50, 100, 200, 400),
        )
        self.kv_transfer_latency_ms = Histogram(  # KV传输延迟直方图
            name="sglang:kv_transfer_latency_ms",
            documentation="Histogram of KV cache transfer latency in ms.",
            labelnames=labels.keys(),
            buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
        )
        self.pending_prealloc_token_usage = Gauge(  # 待预分配令牌使用率仪表
            name="sglang:pending_prealloc_token_usage",
            documentation="The token usage for pending preallocated tokens (not preallocated yet).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_bootstrap_failed_reqs = Counter(  # 引导失败请求数计数器
            name="sglang:num_bootstrap_failed_reqs_total",
            documentation="The number of bootstrap failed requests.",
            labelnames=labels.keys(),
        )
        self.num_transfer_failed_reqs = Counter(  # 传输失败请求数计数器
            name="sglang:num_transfer_failed_reqs_total",
            documentation="The number of transfer failed requests.",
            labelnames=labels.keys(),
        )
        self.num_prefill_retries_total = Counter(  # 预填充重试总数计数器
            name="sglang:num_prefill_retries_total",
            documentation="Total number of prefill retries.",
            labelnames=labels.keys(),
        )
        self.kv_transfer_bootstrap_ms = Histogram(  # KV传输引导时间直方图
            name="sglang:kv_transfer_bootstrap_ms",
            documentation="Histogram of KV transfer bootstrap time in ms.",
            labelnames=labels.keys(),
            buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
        )
        self.kv_transfer_alloc_ms = Histogram(  # KV传输分配等待时间直方图
            name="sglang:kv_transfer_alloc_ms",
            documentation="Histogram of KV transfer allocation waiting time in ms.",
            labelnames=labels.keys(),
            buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
        )
        self.kv_transfer_total_mb = Histogram(  # KV传输总量直方图
            name="sglang:kv_transfer_total_mb",
            documentation="Histogram of KV cache transfer size in MB.",
            labelnames=labels.keys(),
            buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
        )

        # =================================================================
        # Utilization
        # =================================================================
        self.utilization = Gauge(  # 利用率仪表
            name="sglang:utilization",
            documentation="The utilization.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.fwd_occupancy = Gauge(  # 前向传播占用率仪表
            name="sglang:fwd_occupancy",
            documentation="Forward pass GPU occupancy percentage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Scheduler policy
        # =================================================================
        self.new_token_ratio = Gauge(  # 新令牌比率仪表
            name="sglang:new_token_ratio",
            documentation="The new token ratio.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # CUDA graph
        # =================================================================
        # TODO maybe remove this old gauge in favor of the new counter
        self.is_cuda_graph = Gauge(  # CUDA图使用仪表
            name="sglang:is_cuda_graph",
            documentation="Whether the batch is using CUDA graph.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cuda_graph_passes_total = Counter(  # CUDA图通过总数计数器
            name="sglang:cuda_graph_passes_total",
            documentation="Total number of forward passes categorized by CUDA graph.",
            labelnames=list(labels.keys()) + ["mode"],
        )

        # =================================================================
        # LoRA pool metrics (only created when LoRA is enabled)
        # =================================================================
        if self.enable_lora:  # 仅在启用LoRA时创建
            self.lora_pool_slots_used = Gauge(  # LoRA池已使用槽位数仪表
                name="sglang:lora_pool_slots_used",
                documentation="Number of LoRA adapter slots currently occupied in GPU memory.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.lora_pool_slots_total = Gauge(  # LoRA池总槽位数仪表
                name="sglang:lora_pool_slots_total",
                documentation="Total number of LoRA adapter slots available (max_loras_per_batch).",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.lora_pool_utilization = Gauge(  # LoRA池利用率仪表
                name="sglang:lora_pool_utilization",
                documentation="LoRA pool utilization ratio (used/total). 1.0 means pool is full.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )

        # =================================================================
        # HiCache metrics (only created when hierarchical cache is enabled)
        # =================================================================
        if self.enable_hierarchical_cache:  # 仅在启用分层缓存时创建
            self.hicache_host_used_tokens = Gauge(  # 主机KV缓存已使用令牌数仪表
                name="sglang:hicache_host_used_tokens",
                documentation="Number of tokens currently used in the host KV cache.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.hicache_host_total_tokens = Gauge(  # 主机KV缓存总令牌数仪表
                name="sglang:hicache_host_total_tokens",
                documentation="Total capacity of the host KV cache in tokens.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )

        # =================================================================
        # Streaming session metrics (only created when streaming sessions are enabled)
        # =================================================================
        if self.enable_streaming_session:  # 仅在启用流式会话时创建
            self.num_streaming_sessions = Gauge(  # 流式会话数仪表
                name="sglang:num_streaming_sessions",
                documentation="The number of streaming sessions.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.streaming_session_held_tokens = Gauge(  # 流式会话持有令牌数仪表
                name="sglang:streaming_session_held_tokens",
                documentation="The number of KV tokens currently held by streaming session slots.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )

        # =================================================================
        # Routing key metrics
        # =================================================================
        self.num_unique_running_routing_keys = Gauge(  # 唯一运行路由键数仪表
            name="sglang:num_unique_running_routing_keys",
            documentation="Number of unique routing keys in running batch.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.routing_key_running_req_count = GaugeHistogram(  # 路由键运行请求计数仪表直方图
            name="sglang:routing_key_running_req_count",
            documentation="Distribution of routing keys by running request count (gt < count <= le).",
            labelnames=list(labels.keys()),
            bucket_bounds=ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
        )
        self.routing_key_all_req_count = GaugeHistogram(  # 路由键全部请求计数仪表直方图
            name="sglang:routing_key_all_req_count",
            documentation="Distribution of routing keys by running+waiting request count (gt < count <= le).",
            labelnames=list(labels.keys()),
            bucket_bounds=ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
        )

        # =================================================================
        # Request latency
        # =================================================================
        self.queue_time = Histogram(  # 队列等待时间直方图
            name="sglang:queue_time_seconds",
            documentation="Histogram of queueing time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.000,
                0.001,
                0.005,
                0.010,
                0.050,
                0.100,
                0.200,
                0.500,
                1,
                2,
                3,
                4,
                5,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1200,
                1400,
                1600,
                1800,
                2000,
                2500,
                3000,
            ],
        )
        self.per_stage_req_latency_seconds = Histogram(  # 每阶段请求延迟直方图
            name="sglang:per_stage_req_latency_seconds",
            documentation="The latency of each stage of requests.",
            # captures latency in range [1ms - ~1191s]
            buckets=exponential_buckets(start=0.001, width=1.62, length=30),
            labelnames=list(labels.keys()) + ["stage"],
        )

        # =================================================================
        # Grammar
        # =================================================================
        self.grammar_compilation_time = Histogram(  # 语法编译时间直方图
            name="sglang:grammar_compilation_time_seconds",
            documentation="Histogram of grammar compilation time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.0,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                90,
                120,
                240,
            ],
        )
        self.num_grammar_cache_hit = Counter(  # 语法缓存命中数计数器
            name="sglang:num_grammar_cache_hit_total",
            documentation="Number of grammar cache hits.",
            labelnames=labels.keys(),
        )
        self.num_grammar_aborted = Counter(  # 语法中止数计数器
            name="sglang:num_grammar_aborted_total",
            documentation="Number of grammar aborted requests.",
            labelnames=labels.keys(),
        )
        self.num_grammar_timeout = Counter(  # 语法超时数计数器
            name="sglang:num_grammar_timeout_total",
            documentation="Number of grammar timeouts.",
            labelnames=labels.keys(),
        )
        self.num_grammar_total = Counter(  # 语法请求总数计数器
            name="sglang:num_grammar_total",
            documentation="Number of the total grammar requests.",
            labelnames=labels.keys(),
        )
        self.grammar_schema_count = Histogram(  # 语法模式计数直方图
            name="sglang:grammar_schema_count",
            documentation="Histogram of grammar schema count.",
            labelnames=labels.keys(),
            buckets=[
                0,
                1,
                2,
                5,
                10,
                20,
                30,
                40,
                60,
                80,
                100,
                120,
                140,
                160,
                180,
                200,
                300,
                400,
                500,
                700,
                1000,
            ],
        )
        self.grammar_ebnf_size = Histogram(  # 语法EBNF大小直方图
            name="sglang:grammar_ebnf_size",
            documentation="Histogram of grammar EBNF size.",
            labelnames=labels.keys(),
            buckets=[
                0,
                50,
                100,
                200,
                300,
                500,
                1000,
                2000,
                3000,
                5000,
                10000,
                20000,
                30000,
                50000,
                100000,
            ],
        )

        tree_traversal_time_buckets = [  # 树遍历时间桶
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
            15,
            30,
            60,
            90,
            120,
            240,
        ]
        self.grammar_tree_traversal_time_avg = Histogram(  # 平均语法树遍历时间直方图
            name="sglang:grammar_tree_traversal_time_avg",
            documentation="Histogram of average grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )
        self.grammar_tree_traversal_time_max = Histogram(  # 最大语法树遍历时间直方图
            name="sglang:grammar_tree_traversal_time_max",
            documentation="Histogram of max grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )

        # =================================================================
        # Execution
        # =================================================================
        if (  # 仅在EP排名0且启用EPLB均衡指标时创建
            labels["moe_ep_rank"] == 0
        ) and envs.SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC.get():
            self.eplb_balancedness = Summary(  # EPLB均衡度摘要
                name="sglang:eplb_balancedness",
                documentation="Balancedness of MoE in expert parallelism.",
                labelnames=list(labels.keys()) + ["forward_mode"],
            )

        self.realtime_tokens_total = Counter(  # 实时令牌总数计数器
            name="sglang:realtime_tokens_total",
            documentation=(
                "Total number of tokens processed (updated on each log interval). "
                "mode: prefill_compute, prefill_cache, decode."
            ),
            labelnames=list(labels.keys()) + ["mode"],
        )
        self.forward_execution_seconds_total = Counter(  # 前向执行时间总数计数器
            name="sglang:forward_execution_seconds_total",
            documentation=(
                "Total time that GPU is busy executing model forward passes. "
                "Refer to ForwardMode for category labels."
            ),
            labelnames=list(labels.keys()) + ["category"],
        )
        self.estimated_flops_per_gpu_total = Counter(  # 估算每GPU FLOP总数计数器
            name="sglang:estimated_flops_per_gpu_total",
            documentation=(
                "Estimated number of floating point operations per GPU "
                "(for Model FLOPs Utilization calculations)."
            ),
            labelnames=labels.keys(),
        )
        self.estimated_read_bytes_per_gpu_total = Counter(  # 估算每GPU读取字节总数计数器
            name="sglang:estimated_read_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes read from memory per GPU "
                "(for Model FLOPs Utilization calculations)."
            ),
            labelnames=labels.keys(),
        )
        self.estimated_write_bytes_per_gpu_total = Counter(  # 估算每GPU写入字节总数计数器
            name="sglang:estimated_write_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes written to memory per GPU "
                "(for Model FLOPs Utilization calculations)."
            ),
            labelnames=labels.keys(),
        )

        self.dp_cooperation_realtime_tokens_total = Counter(  # DP协作实时令牌总数计数器
            name="sglang:dp_cooperation_realtime_tokens_total",
            documentation=(
                "Total number of tokens processed with labels about DP cooperation. "
                "mode: prefill_compute, prefill_cache, decode."
            ),
            labelnames=list(labels.keys()) + ["mode", "num_prefill_ranks"],
        )
        self.dp_cooperation_forward_execution_seconds_total = Counter(  # DP协作前向执行时间总数计数器
            name="sglang:dp_cooperation_forward_execution_seconds_total",
            documentation=(
                "Total time that GPU is busy executing model forward passes, "
                "with labels about DP cooperation. "
                "Refer to ForwardMode for category labels."
            ),
            labelnames=list(labels.keys()) + ["category", "num_prefill_ranks"],
        )

        # =================================================================
        # Prefill delayer
        # =================================================================
        max_delay = server_args.prefill_delayer_max_delay_passes  # 获取预填充延迟器最大延迟通过数
        self.prefill_delayer_wait_forward_passes = Histogram(  # 预填充延迟器等待前向通过数直方图
            name="sglang:prefill_delayer_wait_forward_passes",
            documentation="Histogram of forward passes waited by prefill delayer.",
            labelnames=labels.keys(),
            buckets=sorted(
                set(
                    x
                    for x in (
                        server_args.prefill_delayer_forward_passes_buckets
                        or [5, 20, 50, 100, 200]
                    )
                    if x < max_delay
                )
                # Need bucket "<=0" for zero-delay cases, and "max_delay-1" to distinguish "max_delay" timeout passes
                | {0, max_delay - 1}
            ),
        )
        self.prefill_delayer_wait_seconds = Histogram(  # 预填充延迟器等待秒数直方图
            name="sglang:prefill_delayer_wait_seconds",
            documentation="Histogram of wait time in seconds by prefill delayer.",
            labelnames=labels.keys(),
            buckets=sorted(
                set(
                    server_args.prefill_delayer_wait_seconds_buckets
                    or [1, 2, 5, 10, 20, 50, 100, 200, 500]
                )
                # Need bucket "<=0" for zero-delay cases
                | {0}
            ),
        )
        self.prefill_delayer_outcomes_total = Counter(  # 预填充延迟器结果总数计数器
            name="sglang:prefill_delayer_outcomes_total",
            documentation="Prefill delayer outcome counts.",
            labelnames=[
                *labels.keys(),
                "input_estimation",
                "output_allow",
                "output_reason",
                "actual_execution",
            ],
        )

        # =================================================================
        # Constants (set once at startup via emit_constants)
        # =================================================================
        self.max_total_num_tokens = Gauge(  # 最大总令牌数仪表
            name="sglang:max_total_num_tokens",
            documentation="Maximum total number of tokens in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.max_running_requests_under_SLO = Gauge(  # SLO下最大运行请求数仪表
            name="sglang:max_running_requests_under_SLO",
            documentation="The maximum number of running requests under SLO.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.engine_startup_time = Gauge(  # 引擎启动时间仪表
            name="sglang:engine_startup_time",
            documentation="The time taken for the engine to start up.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.engine_load_weights_time = Gauge(  # 引擎权重加载时间仪表
            name="sglang:engine_load_weights_time",
            documentation="The time taken for the engine to load weights.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.page_size = Gauge(  # 页面大小仪表
            name="sglang:page_size",
            documentation="KV cache page size in tokens.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_pages = Gauge(  # 页面数仪表
            name="sglang:num_pages",
            documentation="Number of KV cache pages.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.context_len = Gauge(  # 上下文长度仪表
            name="sglang:context_len",
            documentation="Maximum context length.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.startup_available_gpu_memory_gb = Gauge(  # 启动时可用GPU内存仪表
            name="sglang:startup_available_gpu_memory_gb",
            documentation="Available GPU memory in GB at startup.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

    @classmethod
    def init_new(  # 创建新的调度器指标收集器上下文
        cls,
        *,
        server_args: "ServerArgs",  # 服务器参数
        ps: Any,  # 并行策略
        tp_rank: int,  # 张量并行排名
        pp_rank: int,  # 流水线并行排名
        dp_rank: Optional[int],  # 数据并行排名
        enable_priority_scheduling: bool,  # 是否启用优先级调度
        enable_lora: bool,  # 是否启用LoRA
        enable_hierarchical_cache: bool,  # 是否启用分层缓存
    ) -> "SchedulerMetricsCollectorContext":
        enable_metrics = server_args.enable_metrics  # 获取是否启用指标
        is_stats_logging_rank = ps.attn_tp_rank == 0  # 是否为统计日志排名
        current_scheduler_metrics_enabled = enable_metrics and (  # 当前调度器指标是否启用
            is_stats_logging_rank or server_args.enable_metrics_for_all_schedulers
        )
        enable_kv_cache_events = bool(  # 是否启用KV缓存事件
            server_args.kv_events_config
            and ps.attn_tp_rank == 0
            and ps.attn_cp_rank == 0
        )
        collector: Optional["SchedulerMetricsCollector"] = None  # 收集器初始化
        if enable_metrics:  # 如果启用指标
            engine_type = DisaggregationMode.to_engine_type(  # 获取引擎类型
                server_args.disaggregation_mode
            )
            labels = {  # 构建标签字典
                "model_name": server_args.served_model_name,  # 模型名称
                "engine_type": engine_type,  # 引擎类型
                "tp_rank": tp_rank,  # 张量并行排名
                "pp_rank": pp_rank,  # 流水线并行排名
                "moe_ep_rank": ps.moe_ep_rank,  # MoE专家并行排名
            }
            if enable_priority_scheduling:  # 如果启用优先级调度
                labels["priority"] = ""  # 添加优先级标签
            if dp_rank is not None:  # 如果有数据并行排名
                labels["dp_rank"] = dp_rank  # 添加数据并行标签
            if server_args.extra_metric_labels:  # 如果有额外指标标签
                labels.update(server_args.extra_metric_labels)  # 更新标签
            scheduler_collector_cls = resolve_collector_class(  # 解析收集器类
                server_args, STAT_LOGGER_ROLE_SCHEDULER, cls
            )
            collector = scheduler_collector_cls(  # 创建收集器实例
                labels=labels,
                enable_lora=enable_lora,
                enable_hierarchical_cache=enable_hierarchical_cache,
                enable_streaming_session=server_args.enable_streaming_session,
                server_args=server_args,
            )
        return SchedulerMetricsCollectorContext(  # 返回收集器上下文
            enable_metrics=enable_metrics,
            is_stats_logging_rank=is_stats_logging_rank,
            current_scheduler_metrics_enabled=current_scheduler_metrics_enabled,
            enable_kv_cache_events=enable_kv_cache_events,
            collector=collector,
        )

    def _log_gauge(self, gauge: Gauge, data: Union[int, float]) -> None:  # 记录仪表值
        # Convenience function for logging a scalar to gauge.
        gauge.labels(**self.labels).set(data)  # 设置仪表值

    def _log_gauge_queue_count(self, gauge: Gauge, data: QueueCount) -> None:  # 记录队列计数到仪表
        # Log a QueueCount to gauge: total under default labels, per-priority breakdown under priority="<int>".
        # NOTE: When priority scheduling is enabled, the total is recorded under
        # priority="" (the default label value). Per-priority breakdowns are recorded
        # with priority="<int>". Grafana queries should use priority="" for totals.
        gauge.labels(**self.labels).set(data.total)  # 记录总数
        if data.by_priority is not None:  # 如果有按优先级分解
            self._known_priorities.update(data.by_priority.keys())  # 更新已知优先级
            for priority in self._known_priorities:  # 遍历已知优先级
                value = data.by_priority.get(priority, 0)  # 获取值
                labels = dict(self.labels)  # 复制标签
                labels["priority"] = str(priority)  # 设置优先级标签
                gauge.labels(**labels).set(value)  # 记录值

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:  # 记录直方图值
        histogram.labels(**self.labels).observe(data)  # 观测数据

    def increment_bootstrap_failed_reqs(self) -> None:  # 增加引导失败请求数
        self.num_bootstrap_failed_reqs.labels(**self.labels).inc(1)  # 递增1

    def increment_transfer_failed_reqs(self) -> None:  # 增加传输失败请求数
        self.num_transfer_failed_reqs.labels(**self.labels).inc(1)  # 递增1

    def increment_prefill_retries(self, count: int) -> None:  # 增加预填充重试数
        if count > 0:  # 如果计数大于0
            self.num_prefill_retries_total.labels(**self.labels).inc(count)  # 递增

    def observe_kv_transfer_metrics(  # 观测KV传输指标
        self,
        latency_ms: float,  # 延迟（毫秒）
        total_mb: float,  # 总量（兆字节）
        speed_gb_s: float,  # 速度（吉字节/秒）
    ) -> None:
        self._log_histogram(self.kv_transfer_latency_ms, latency_ms)  # 记录延迟
        self._log_histogram(self.kv_transfer_total_mb, total_mb)  # 记录总量
        self._log_histogram(self.kv_transfer_speed_gb_s, speed_gb_s)  # 记录速度

    def observe_kv_transfer_bootstrap(  # 观测KV传输引导时间
        self,
        bootstrap_ms: float,  # 引导时间（毫秒）
        alloc_ms: float,  # 分配等待时间（毫秒）
    ) -> None:
        self._log_histogram(self.kv_transfer_bootstrap_ms, bootstrap_ms)  # 记录引导时间
        self._log_histogram(self.kv_transfer_alloc_ms, alloc_ms)  # 记录分配时间

    def observe_per_stage_req_latency(self, stage: str, latency: float) -> None:  # 观测每阶段请求延迟
        labels_with_stage = {**self.labels, "stage": stage}  # 添加阶段标签
        self.per_stage_req_latency_seconds.labels(**labels_with_stage).observe(latency)  # 记录延迟

    def observe_queue_time(self, latency: float) -> None:  # 观测队列等待时间
        self._log_histogram(self.queue_time, latency)  # 记录延迟

    def observe_prefill_delayer_outcome(  # 观测预填充延迟器结果
        self,
        forward_passes: int,  # 等待的前向通过数
        wait_seconds: float,  # 等待秒数
        input_estimation: str,  # 输入估算
        output_allow: bool,  # 是否允许输出
        output_reason: str,  # 输出原因
        actual_execution: bool,  # 是否实际执行
    ) -> None:
        if output_allow and actual_execution:  # 如果允许且实际执行
            self._log_histogram(  # 记录等待前向通过数
                self.prefill_delayer_wait_forward_passes, forward_passes
            )
            self._log_histogram(self.prefill_delayer_wait_seconds, wait_seconds)  # 记录等待秒数

        self.prefill_delayer_outcomes_total.labels(  # 记录结果
            **self.labels,
            input_estimation=input_estimation,
            output_allow=str(output_allow).lower(),
            output_reason=output_reason,
            actual_execution=str(actual_execution).lower(),
        ).inc(1)  # 递增1

    def increment_retracted_reqs(  # 增加被撤回请求数
        self,
        num_retracted_reqs: int,  # 被撤回的请求数
        num_retracted_input_tokens: int,  # 被撤回的输入令牌数
        num_retracted_output_tokens: int,  # 被撤回的输出令牌数
    ) -> None:
        self.num_retracted_reqs_total.labels(**self.labels).inc(num_retracted_reqs)  # 递增请求数
        self.num_retracted_input_tokens_total.labels(**self.labels).inc(  # 递增输入令牌数
            num_retracted_input_tokens
        )
        self.num_retracted_output_tokens_total.labels(**self.labels).inc(  # 递增输出令牌数
            num_retracted_output_tokens
        )

    def increment_decode_cuda_graph_pass(self, value: bool) -> None:  # 增加解码CUDA图通过数
        mode = "decode_cuda_graph" if value else "decode_none"  # 确定模式
        self.cuda_graph_passes_total.labels(**self.labels, mode=mode).inc(1)  # 递增

    def increment_prefill_cuda_graph_pass(self, value: bool) -> None:  # 增加预填充CUDA图通过数
        mode = "prefill_cuda_graph" if value else "prefill_none"  # 确定模式
        self.cuda_graph_passes_total.labels(**self.labels, mode=mode).inc(1)  # 递增

    def increment_eplb_balancedness(  # 增加EPLB均衡度
        self, forward_mode: str, balancedness: float
    ) -> None:
        self.eplb_balancedness.labels(**self.labels, forward_mode=forward_mode).observe(  # 观测均衡度
            balancedness
        )

    def increment_realtime_tokens(  # 增加实时令牌数
        self,
        dp_cooperation_info: Optional[DPCooperationInfo],  # DP协作信息
        prefill_compute_tokens=0,  # 预填充计算令牌数
        prefill_cache_tokens=0,  # 预填充缓存令牌数
        decode_tokens=0,  # 解码令牌数
    ):
        for mode, delta in [  # 遍历每种模式
            ("prefill_compute", prefill_compute_tokens),
            ("prefill_cache", prefill_cache_tokens),
            ("decode", decode_tokens),
        ]:
            if delta == 0:  # 如果增量为0
                continue  # 跳过
            self.realtime_tokens_total.labels(**self.labels, mode=mode).inc(delta)  # 递增
            if dp_cooperation_info is not None:  # 如果有DP协作信息
                self.dp_cooperation_realtime_tokens_total.labels(  # 递增DP协作令牌数
                    **self.labels,
                    mode=mode,
                    **dp_cooperation_info.to_labels(),
                ).inc(delta)

    def increment_forward_execution_seconds(  # 增加前向执行时间
        self,
        category: str,  # 类别
        t: float,  # 时间
        dp_cooperation_info: Optional[DPCooperationInfo] = None,  # DP协作信息
    ):
        self.forward_execution_seconds_total.labels(  # 递增前向执行时间
            **self.labels, category=category
        ).inc(t)
        if dp_cooperation_info is not None:  # 如果有DP协作信息
            self.dp_cooperation_forward_execution_seconds_total.labels(  # 递增DP协作前向执行时间
                **self.labels,
                category=category,
                **dp_cooperation_info.to_labels(),
            ).inc(t)

    def increment_estimated_perf(  # 增加估算性能
        self,
        num_flops_per_gpu: float = 0.0,  # 每GPU FLOP数
        num_read_bytes_per_gpu: float = 0.0,  # 每GPU读取字节数
        num_write_bytes_per_gpu: float = 0.0,  # 每GPU写入字节数
    ) -> None:
        if num_flops_per_gpu > 0:  # 如果有FLOP数
            self.estimated_flops_per_gpu_total.labels(**self.labels).inc(  # 递增
                num_flops_per_gpu
            )
        if num_read_bytes_per_gpu > 0:  # 如果有读取字节数
            self.estimated_read_bytes_per_gpu_total.labels(**self.labels).inc(  # 递增
                num_read_bytes_per_gpu
            )
        if num_write_bytes_per_gpu > 0:  # 如果有写入字节数
            self.estimated_write_bytes_per_gpu_total.labels(**self.labels).inc(  # 递增
                num_write_bytes_per_gpu
            )

    def log_stats(self, stats: SchedulerStats) -> None:  # 记录调度器统计信息
        # Basics
        self._log_gauge_queue_count(self.num_running_reqs, stats.num_running_reqs)  # 运行中请求数
        self._log_gauge_queue_count(self.num_queue_reqs, stats.num_queue_reqs)  # 队列请求数
        self._log_gauge(self.num_grammar_queue_reqs, stats.num_grammar_queue_reqs)  # 语法队列请求数
        self._log_gauge(self.gen_throughput, stats.gen_throughput)  # 生成吞吐量
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)  # 缓存命中率
        self._log_gauge(self.decode_sum_seq_lens, stats.decode_sum_seq_lens)  # 解码序列长度总和

        # Memory pool usage ratios
        self._log_gauge(self.token_usage, stats.token_usage)  # 令牌使用率
        self._log_gauge(self.full_token_usage, stats.full_token_usage)  # 全注意力令牌使用率
        self._log_gauge(self.swa_token_usage, stats.swa_token_usage)  # SWA令牌使用率
        self._log_gauge(self.mamba_usage, stats.mamba_usage)  # Mamba使用率

        # Absolute token counts
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)  # 已使用令牌数
        self._log_gauge(self.kv_available_tokens, stats.kv_available_tokens)  # KV可用令牌数
        self._log_gauge(self.kv_evictable_tokens, stats.kv_evictable_tokens)  # KV可驱逐令牌数
        self._log_gauge(self.kv_used_tokens, stats.kv_used_tokens)  # KV已使用令牌数
        self._log_gauge(self.swa_available_tokens, stats.swa_available_tokens)  # SWA可用令牌数
        self._log_gauge(self.swa_evictable_tokens, stats.swa_evictable_tokens)  # SWA可驱逐令牌数
        self._log_gauge(self.swa_used_tokens, stats.swa_used_tokens)  # SWA已使用令牌数
        self._log_gauge(self.mamba_available_tokens, stats.mamba_available_tokens)  # Mamba可用状态数
        self._log_gauge(self.mamba_evictable_tokens, stats.mamba_evictable_tokens)  # Mamba可驱逐状态数
        self._log_gauge(self.mamba_used_tokens, stats.mamba_used_tokens)  # Mamba已使用状态数

        # Speculative decoding
        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)  # 推测解码接受长度
        self._log_gauge(self.spec_accept_rate, stats.spec_accept_rate)  # 推测解码接受率
        self._log_gauge(self.spec_num_steps, stats.spec_num_steps)  # 推测解码步数
        self._log_gauge(self.spec_num_draft_tokens, stats.spec_num_draft_tokens)  # 推测解码草稿令牌数

        # Retract
        self._log_gauge(self.num_retracted_reqs, stats.num_retracted_reqs)  # 被撤回请求数
        self._log_gauge(self.num_paused_reqs, stats.num_paused_reqs)  # 被暂停请求数

        # PD disaggregation
        self._log_gauge_queue_count(  # 预填充引导队列请求数
            self.num_prefill_bootstrap_queue_reqs,
            stats.num_prefill_bootstrap_queue_reqs,
        )
        self._log_gauge_queue_count(  # 预填充飞行中队列请求数
            self.num_prefill_inflight_queue_reqs, stats.num_prefill_inflight_queue_reqs
        )
        self._log_gauge_queue_count(  # 解码预分配队列请求数
            self.num_decode_prealloc_queue_reqs, stats.num_decode_prealloc_queue_reqs
        )
        self._log_gauge_queue_count(  # 解码传输队列请求数
            self.num_decode_transfer_queue_reqs, stats.num_decode_transfer_queue_reqs
        )
        self._log_gauge(  # 待预分配令牌使用率
            self.pending_prealloc_token_usage, stats.pending_prealloc_token_usage
        )

        # Utilization
        self._log_gauge(self.utilization, stats.utilization)  # 利用率
        self._log_gauge(self.fwd_occupancy, stats.fwd_occupancy)  # 前向传播占用率

        # Scheduler policy
        self._log_gauge(self.new_token_ratio, stats.new_token_ratio)  # 新令牌比率

        # CUDA graph
        self._log_gauge(self.is_cuda_graph, stats.is_cuda_graph)  # CUDA图使用

        # LoRA pool metrics
        if self.enable_lora:  # 如果启用LoRA
            self._log_gauge(self.lora_pool_slots_used, stats.lora_pool_slots_used)  # LoRA已使用槽位
            self._log_gauge(self.lora_pool_slots_total, stats.lora_pool_slots_total)  # LoRA总槽位
            self._log_gauge(self.lora_pool_utilization, stats.lora_pool_utilization)  # LoRA利用率

        # HiCache metrics
        if self.enable_hierarchical_cache:  # 如果启用分层缓存
            self._log_gauge(  # 主机KV缓存已使用令牌数
                self.hicache_host_used_tokens, stats.hicache_host_used_tokens
            )
            self._log_gauge(  # 主机KV缓存总令牌数
                self.hicache_host_total_tokens, stats.hicache_host_total_tokens
            )

        # Streaming session metrics
        if self.enable_streaming_session:  # 如果启用流式会话
            self._log_gauge(self.num_streaming_sessions, stats.num_streaming_sessions)  # 流式会话数
            self._log_gauge(  # 流式会话持有令牌数
                self.streaming_session_held_tokens, stats.streaming_session_held_tokens
            )

        # Routing key metrics
        self._log_gauge(  # 唯一运行路由键数
            self.num_unique_running_routing_keys, stats.num_unique_running_routing_keys
        )
        self.routing_key_running_req_count.set_by_current_observations(  # 路由键运行请求计数
            self.labels, stats.routing_key_running_req_counts
        )
        self.routing_key_all_req_count.set_by_current_observations(  # 路由键全部请求计数
            self.labels, stats.routing_key_all_req_counts
        )

        self.last_log_time = time.perf_counter()  # 更新上次日志时间

    def log_grammar_stats(self, grammar_stats) -> None:  # 记录语法统计信息
        if grammar_stats.compilation_time is not None:  # 如果有编译时间
            self._log_histogram(
                self.grammar_compilation_time, grammar_stats.compilation_time
            )
        if grammar_stats.schema_count is not None:  # 如果有模式计数
            self._log_histogram(self.grammar_schema_count, grammar_stats.schema_count)
        if grammar_stats.ebnf_size is not None:  # 如果有EBNF大小
            self._log_histogram(self.grammar_ebnf_size, grammar_stats.ebnf_size)
        tree_times = grammar_stats.tree_traversal_time  # 获取树遍历时间
        if tree_times:  # 如果有树遍历时间
            max_time = max(tree_times)  # 计算最大时间
            avg_time = sum(tree_times) / len(tree_times)  # 计算平均时间
            self._log_histogram(self.grammar_tree_traversal_time_max, max_time)  # 记录最大时间
            self._log_histogram(self.grammar_tree_traversal_time_avg, avg_time)  # 记录平均时间
        if grammar_stats.is_cache_hit:  # 如果缓存命中
            self.num_grammar_cache_hit.labels(**self.labels).inc(1)  # 递增缓存命中数
        if grammar_stats.is_grammar_aborted:  # 如果语法中止
            self.num_grammar_aborted.labels(**self.labels).inc(1)  # 递增中止数
        if grammar_stats.num_timeout > 0:  # 如果有超时
            self.num_grammar_timeout.labels(**self.labels).inc(  # 递增超时数
                grammar_stats.num_timeout
            )
        self.num_grammar_total.labels(**self.labels).inc(1)  # 递增总数

    def emit_constants(  # 发射常量指标（启动时设置一次）
        self,
        max_total_num_tokens: int,  # 最大总令牌数
        max_running_requests_under_SLO: Optional[int],  # SLO下最大运行请求数
        engine_startup_time: float,  # 引擎启动时间
        engine_load_weights_time: float,  # 引擎权重加载时间
        page_size: int,  # 页面大小
        num_pages: int,  # 页面数
        context_len: int,  # 上下文长度
        startup_available_gpu_memory_gb: float,  # 启动时可用GPU内存
    ) -> None:
        self._log_gauge(self.max_total_num_tokens, max_total_num_tokens)  # 记录最大总令牌数
        if max_running_requests_under_SLO is not None:  # 如果有SLO值
            self._log_gauge(
                self.max_running_requests_under_SLO, max_running_requests_under_SLO
            )
        self._log_gauge(self.engine_startup_time, engine_startup_time)  # 记录启动时间
        self._log_gauge(self.engine_load_weights_time, engine_load_weights_time)  # 记录加载时间
        self._log_gauge(self.page_size, page_size)  # 记录页面大小
        self._log_gauge(self.num_pages, num_pages)  # 记录页面数
        self._log_gauge(self.context_len, context_len)  # 记录上下文长度
        self._log_gauge(  # 记录启动时可用GPU内存
            self.startup_available_gpu_memory_gb, startup_available_gpu_memory_gb
        )


class TokenizerMetricsCollector(_StatLoggerDIMixin):  # 分词器指标收集器
    def __init__(  # 初始化分词器指标收集器
        self,
        server_args: Optional[ServerArgs] = None,  # 服务器参数
        labels: Dict[str, str] = None,  # 标签字典
        bucket_time_to_first_token: Optional[List[float]] = None,  # 首令牌时间桶
        bucket_inter_token_latency: Optional[List[float]] = None,  # 令牌间延迟桶
        bucket_e2e_request_latency: Optional[List[float]] = None,  # 端到端延迟桶
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter as _PromCounter  # 导入计数器
        from prometheus_client import Histogram as _PromHistogram  # 导入直方图

        Counter = self._counter_cls or _PromCounter  # 使用覆盖或默认类
        Histogram = self._histogram_cls or _PromHistogram  # 使用覆盖或默认类

        self.labels = labels or {}  # 保存标签

        self.prompt_tokens_total = Counter(  # 提示令牌总数计数器
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )
        self.generation_tokens_total = Counter(  # 生成令牌总数计数器
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )
        self.spec_verify_calls_total = Counter(  # 推测解码验证调用总数计数器
            name="sglang:spec_verify_calls_total",
            documentation="Number of speculative decoding verification calls.",
            labelnames=labels.keys(),
        )

        default_bucket_prompt_tokens = [  # 默认提示令牌桶
            100,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            12500,
            15000,
            17500,
            20000,
            22500,
            25000,
            27500,
            30000,
            35000,
            40000,
            60000,
            80000,
            100000,
            200000,
            300000,
            400000,
            600000,
            800000,
            1000000,
            1100000,
        ]
        self.prompt_tokens_histogram = Histogram(  # 提示令牌直方图
            name="sglang:prompt_tokens_histogram",
            documentation="Histogram of prompt token length.",
            labelnames=labels.keys(),
            buckets=generate_buckets(
                server_args.prompt_tokens_buckets, default_bucket_prompt_tokens
            ),
        )
        self.uncached_prompt_tokens_histogram = Histogram(  # 未缓存提示令牌直方图
            name="sglang:uncached_prompt_tokens_histogram",
            documentation="Histogram of uncached (compute) prompt token length.",
            labelnames=labels.keys(),
            buckets=generate_buckets(
                server_args.prompt_tokens_buckets, default_bucket_prompt_tokens
            ),
        )
        self.generation_tokens_histogram = Histogram(  # 生成令牌直方图
            name="sglang:generation_tokens_histogram",
            documentation="Histogram of generation token length.",
            labelnames=labels.keys(),
            buckets=generate_buckets(
                server_args.generation_tokens_buckets,
                default_bucket_prompt_tokens,
            ),
        )

        self.cached_tokens_total = Counter(  # 缓存令牌总数计数器
            name="sglang:cached_tokens_total",
            documentation="Number of cached prompt tokens by source (device/host/storage).",
            labelnames=list(labels.keys()) + ["cache_source"],
        )

        self.num_requests_total = Counter(  # 请求总数计数器
            name="sglang:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labels.keys(),
        )

        self.get_loads_duration_seconds = Histogram(  # /v1/loads请求耗时直方图
            name="sglang:get_loads_duration_seconds",
            documentation="Time spent serving /v1/loads requests (seconds).",
            labelnames=labels.keys(),
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
        )

        self.num_so_requests_total = Counter(  # 结构化输出请求总数计数器
            name="sglang:num_so_requests_total",
            documentation="Number of structured output requests processed.",
            labelnames=labels.keys(),
        )

        self.num_aborted_requests_total = Counter(  # 中止请求总数计数器
            name="sglang:num_aborted_requests_total",
            documentation="Number of requests aborted.",
            labelnames=labels.keys(),
        )

        if bucket_time_to_first_token is None:  # 如果未指定首令牌时间桶
            bucket_time_to_first_token = [  # 使用默认桶
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
            ]

        if bucket_e2e_request_latency is None:  # 如果未指定端到端延迟桶
            bucket_e2e_request_latency = [  # 使用默认桶
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
                600,
                1200,
                1800,
                2400,
            ]

        if bucket_inter_token_latency is None:  # 如果未指定令牌间延迟桶
            bucket_inter_token_latency = [  # 使用默认桶
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.060,
                0.080,
                0.100,
                0.200,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
                4.000,
                6.000,
                8.000,
            ]

        self.histogram_time_to_first_token = Histogram(  # 首令牌时间直方图
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_time_to_first_token,
        )

        self.histogram_inter_token_latency = Histogram(  # 令牌间延迟直方图
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_inter_token_latency,
        )

        self.histogram_e2e_request_latency = Histogram(  # 端到端请求延迟直方图
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=bucket_e2e_request_latency,
        )

    def observe_one_finished_request(  # 观测一个完成的请求
        self,
        labels: Dict[str, str],  # 标签字典
        prompt_tokens: int,  # 提示令牌数
        generation_tokens: int,  # 生成令牌数
        cached_tokens: int,  # 缓存令牌数
        e2e_latency: float,  # 端到端延迟
        has_grammar: bool,  # 是否有语法约束
        cached_tokens_details: Optional[Dict[str, Any]] = None,  # 缓存令牌详情
        spec_verify_ct: int = 0,  # 推测解码验证次数
    ):
        self.prompt_tokens_total.labels(**labels).inc(prompt_tokens)  # 递增提示令牌数
        self.generation_tokens_total.labels(**labels).inc(generation_tokens)  # 递增生成令牌数
        if spec_verify_ct > 0:  # 如果有推测解码验证
            self.spec_verify_calls_total.labels(**labels).inc(spec_verify_ct)  # 递增验证调用数

        # Report cached tokens with detailed source breakdown
        if cached_tokens > 0:  # 如果有缓存令牌
            if cached_tokens_details:  # 如果有详细缓存信息
                # Report by cache source (device/host, and storage if L3 enabled)
                def report_cache_source(source: str, value: int):  # 报告缓存来源
                    if value > 0:  # 如果值大于0
                        source_labels = {**labels, "cache_source": source}  # 添加来源标签
                        self.cached_tokens_total.labels(**source_labels).inc(value)  # 递增

                report_cache_source("device", cached_tokens_details.get("device", 0))  # 报告设备缓存
                report_cache_source("host", cached_tokens_details.get("host", 0))  # 报告主机缓存

                # Storage fields are only present when L3 storage backend is enabled
                if "storage" in cached_tokens_details:  # 如果有存储缓存
                    storage_tokens = cached_tokens_details.get("storage", 0)  # 获取存储令牌数
                    if storage_tokens > 0:  # 如果大于0
                        backend = (  # 获取后端类型
                            cached_tokens_details.get("storage_backend") or "unknown"
                        )
                        report_cache_source(f"storage_{backend}", storage_tokens)  # 报告存储缓存
            else:  # 没有详细信息
                # Fallback for backward compatibility
                labels_total = {**labels, "cache_source": "total"}  # 使用total来源
                self.cached_tokens_total.labels(**labels_total).inc(cached_tokens)  # 递增总数

        self.num_requests_total.labels(**labels).inc(1)  # 递增请求总数
        if has_grammar:  # 如果有语法约束
            self.num_so_requests_total.labels(**labels).inc(1)  # 递增结构化输出请求数
        self.histogram_e2e_request_latency.labels(**labels).observe(float(e2e_latency))  # 观测端到端延迟
        self.prompt_tokens_histogram.labels(**labels).observe(float(prompt_tokens))  # 观测提示令牌数
        self.uncached_prompt_tokens_histogram.labels(**labels).observe(  # 观测未缓存提示令牌数
            float(prompt_tokens - cached_tokens)
        )
        self.generation_tokens_histogram.labels(**labels).observe(  # 观测生成令牌数
            float(generation_tokens)
        )

    def observe_time_to_first_token(self, labels: Dict[str, str], value: float):  # 观测首令牌时间
        self.histogram_time_to_first_token.labels(**labels).observe(value)  # 观测值

    def check_time_to_first_token_straggler(self, value: float) -> bool:  # 检查首令牌时间是否为离群值
        his = self.histogram_time_to_first_token.labels(**self.labels)  # 获取直方图
        total_observations = sum(bucket._value for bucket in his._buckets)  # 计算总观测数
        if total_observations < 100:  # 如果观测数太少
            return False  # 不判断
        p99_threshold = total_observations * 0.99  # 计算P99阈值
        cumulative_count = 0  # 累计计数
        for i, bucket in enumerate(his._buckets):  # 遍历桶
            cumulative_count += bucket._value  # 累加
            if cumulative_count > p99_threshold:  # 如果超过P99
                return value >= his._upper_bounds[i]  # 返回是否为离群值
        return False  # 不是离群值

    def observe_inter_token_latency(  # 观测令牌间延迟
        self, labels: Dict[str, str], internval: float, num_new_tokens: int
    ):
        adjusted_interval = internval / num_new_tokens  # 计算调整后的间隔

        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency.labels(**labels)  # 获取直方图
        his._sum.inc(internval)  # 递增总和

        for i, bound in enumerate(his._upper_bounds):  # 遍历上界
            if adjusted_interval <= bound:  # 如果在桶内
                his._buckets[i].inc(num_new_tokens)  # 递增桶计数
                break  # 跳出

    def observe_one_aborted_request(self, labels: Dict[str, str]):  # 观测一个中止的请求
        self.num_aborted_requests_total.labels(**labels).inc(1)  # 递增中止请求数


@dataclass
class StorageMetrics:  # 存储指标
    prefetch_pgs: List[int] = field(default_factory=list)  # 预取页数列表
    backup_pgs: List[int] = field(default_factory=list)  # 备份页数列表
    prefetch_bandwidth: List[float] = field(default_factory=list)  # 预取带宽列表
    backup_bandwidth: List[float] = field(default_factory=list)  # 备份带宽列表


class StorageMetricsCollector(_StatLoggerDIMixin):  # 存储指标收集器
    def __init__(  # 初始化存储指标收集器
        self,
        labels: Dict[str, str],  # 标签字典
    ):
        from prometheus_client import Counter as _PromCounter  # 导入计数器
        from prometheus_client import Histogram as _PromHistogram  # 导入直方图

        Counter = self._counter_cls or _PromCounter  # 使用覆盖或默认类
        Histogram = self._histogram_cls or _PromHistogram  # 使用覆盖或默认类

        self.labels = labels  # 保存标签

        self.prefetched_tokens_total = Counter(  # 预取令牌总数计数器
            name="sglang:prefetched_tokens_total",
            documentation="Number of prefetched prompt tokens.",
            labelnames=labels.keys(),
        )

        self.backuped_tokens_total = Counter(  # 备份令牌总数计数器
            name="sglang:backuped_tokens_total",
            documentation="Number of backuped tokens.",
            labelnames=labels.keys(),
        )

        bucket_io = [  # IO桶
            1,
            5,
            10,
            50,
            100,
        ]

        bucket_bandwidth = [  # 带宽桶
            0.1,
            0.5,
            1,
            5,
            10,
            50,
            100,
        ]

        self.histogram_prefetch_pgs = Histogram(  # 预取页数直方图
            name="sglang:prefetch_pgs",
            documentation="Histogram of prefetch pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_backup_pgs = Histogram(  # 备份页数直方图
            name="sglang:backup_pgs",
            documentation="Histogram of backup pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_prefetch_bandwidth = Histogram(  # 预取带宽直方图
            name="sglang:prefetch_bandwidth",
            documentation="Histogram of prefetch bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

        self.histogram_backup_bandwidth = Histogram(  # 备份带宽直方图
            name="sglang:backup_bandwidth",
            documentation="Histogram of backup bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

    def log_prefetched_tokens(self, prefetched_tokens: int):  # 记录预取令牌数
        if prefetched_tokens > 0:  # 如果大于0
            self.prefetched_tokens_total.labels(**self.labels).inc(prefetched_tokens)  # 递增

    def log_backuped_tokens(self, backuped_tokens: int):  # 记录备份令牌数
        if backuped_tokens > 0:  # 如果大于0
            self.backuped_tokens_total.labels(**self.labels).inc(backuped_tokens)  # 递增

    def _log_histogram(self, histogram, data: Union[int, float]):  # 记录直方图值
        histogram.labels(**self.labels).observe(data)  # 观测数据

    def log_storage_metrics(self, storage_metrics: Optional[StorageMetrics] = None):  # 记录存储指标
        if storage_metrics is None:  # 如果没有指标
            return  # 直接返回

        assert isinstance(storage_metrics, StorageMetrics)  # 断言类型

        for v in storage_metrics.prefetch_pgs:  # 遍历预取页数
            self._log_histogram(self.histogram_prefetch_pgs, v)  # 记录
        for v in storage_metrics.backup_pgs:  # 遍历备份页数
            self._log_histogram(self.histogram_backup_pgs, v)  # 记录
        for v in storage_metrics.prefetch_bandwidth:  # 遍历预取带宽
            self._log_histogram(self.histogram_prefetch_bandwidth, v)  # 记录
        for v in storage_metrics.backup_bandwidth:  # 遍历备份带宽
            self._log_histogram(self.histogram_backup_bandwidth, v)  # 记录


class ExpertDispatchCollector(_StatLoggerDIMixin):  # 专家分发收集器
    def __init__(self, ep_size: int) -> None:  # 初始化专家分发收集器
        from prometheus_client import Histogram as _PromHistogram  # 导入直方图

        Histogram = self._histogram_cls or _PromHistogram  # 使用覆盖或默认类

        ep_size_buckets = [i for i in range(ep_size)]  # 创建EP大小桶
        self.eplb_gpu_physical_count = Histogram(  # EPLB GPU物理专家计数直方图
            name="sglang:eplb_gpu_physical_count",
            documentation="The selected count of physical experts on each layer and GPU rank.",
            labelnames={"layer"},
            buckets=ep_size_buckets,
        )


class RadixCacheMetricsCollector(_StatLoggerDIMixin):  # Radix缓存指标收集器
    def __init__(  # 初始化Radix缓存指标收集器
        self,
        labels: Dict[str, str],  # 标签字典
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter as _PromCounter  # 导入计数器
        from prometheus_client import Histogram as _PromHistogram  # 导入直方图

        Counter = self._counter_cls or _PromCounter  # 使用覆盖或默认类
        Histogram = self._histogram_cls or _PromHistogram  # 使用覆盖或默认类

        self.labels = labels  # 保存标签

        bucket_eviction_duration = get_histogram_conf_from_env(  # 从环境变量获取驱逐时间桶
            "SGLANG_BUCKET_EVICTION_DURATION"
        )
        if bucket_eviction_duration is None:  # 如果未设置
            bucket_eviction_duration = [  # 使用默认桶
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
            ]
        bucket_load_back_duration = get_histogram_conf_from_env(  # 从环境变量获取加载回时间桶
            "SGLANG_BUCKET_LOAD_BACK_DURATION"
        )
        if bucket_load_back_duration is None:  # 如果未设置
            bucket_load_back_duration = [  # 使用默认桶
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
            ]
        self.eviction_duration_seconds = Histogram(  # 驱逐耗时直方图
            name="sglang:eviction_duration_seconds",
            documentation="Time taken to evict memory from GPU to CPU in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_eviction_duration,
        )

        self.eviction_num_tokens = Counter(  # 驱逐令牌数计数器
            name="sglang:evicted_tokens_total",
            documentation="The number of tokens evicted from GPU to CPU.",
            labelnames=labels.keys(),
        )

        self.load_back_duration_seconds = Histogram(  # 加载回耗时直方图
            name="sglang:load_back_duration_seconds",
            documentation="Time taken to load memory from CPU to GPU in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_load_back_duration,
        )

        self.load_back_num_tokens = Counter(  # 加载回令牌数计数器
            name="sglang:load_back_tokens_total",
            documentation="The number of tokens loaded from CPU to GPU.",
            labelnames=labels.keys(),
        )

    def increment_eviction_num_tokens(self, num_tokens: int) -> None:  # 增加驱逐令牌数
        self.eviction_num_tokens.labels(**self.labels).inc(num_tokens)  # 递增

    def increment_load_back_num_tokens(self, num_tokens: int) -> None:  # 增加加载回令牌数
        self.load_back_num_tokens.labels(**self.labels).inc(num_tokens)  # 递增

    def observe_eviction_duration(self, duration_seconds: float) -> None:  # 观测驱逐耗时
        self.eviction_duration_seconds.labels(**self.labels).observe(duration_seconds)  # 观测

    def observe_load_back_duration(self, duration_seconds: float) -> None:  # 观测加载回耗时
        self.load_back_duration_seconds.labels(**self.labels).observe(duration_seconds)  # 观测


def get_histogram_conf_from_env(env_var_name: str) -> Optional[List[float]]:  # 从环境变量获取直方图配置
    """
    Get the histogram configuration from the environment variable.
    env value should be like "0.1,0.2,0.5,1,2"
    """
    if env_var_name not in os.environ:  # 如果环境变量不存在
        return None  # 返回None
    # if the env var is not set or empty, return None
    env_var_value = os.environ[env_var_name]  # 获取环境变量值
    if not env_var_value:  # 如果为空
        return None  # 返回None
    return [float(x) for x in env_var_value.split(",")]  # 解析并返回浮点数列表
