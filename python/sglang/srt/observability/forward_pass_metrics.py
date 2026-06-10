# 前向传播指标模块
# 本模块提供调度器每次迭代的前向传播遥测指标
# 通过ZMQ PUB发布每迭代调度指标，供外部消费者实时观测
# 使用msgspec.Struct进行零拷贝序列化

"""
Forward pass metrics for per-iteration scheduler telemetry.

Emits per-iteration scheduling metrics over ZMQ PUB so that external
consumers can observe scheduler behavior in real time without polling
Prometheus.

Uses msgspec.Struct for zero-copy serialization.

Data flow::

    Scheduler process:
        SchedulerMetricsMixin._emit_forward_pass_metrics()
          -> _FpmPublisherThread -> ZMQ PUB (localhost)

    External consumer:
        ZMQ SUB -> deserialize ForwardPassMetrics

"""

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
import queue  # 导入队列模块
import threading  # 导入线程模块
import time  # 导入时间模块
from itertools import count  # 导入计数器

import msgspec  # 导入msgspec序列化库

# Schema version. Must match the consumer (Dynamo's ForwardPassMetrics).
# Bump when the schema changes incompatibly.
FPM_VERSION: int = 1  # 模式版本号，必须与消费者匹配

logger = logging.getLogger(__name__)  # 创建日志记录器


class WelfordAccumulator:  # Welford在线算法，用于计算计数/总和/总体方差
    """Welford's online algorithm for count / total / population-variance.

    Numerically stable single-pass computation.
    """

    __slots__ = ("count", "total", "_mean", "_m2")  # 定义实例属性槽

    def __init__(self) -> None:  # 初始化Welford累加器
        self.count = 0  # 计数
        self.total = 0  # 总和
        self._mean = 0.0  # 均值
        self._m2 = 0.0  # M2统计量

    def add(self, v: int) -> None:  # 添加一个值到累加器
        self.count += 1  # 增加计数
        self.total += v  # 累加总和
        delta = v - self._mean  # 计算差值
        self._mean += delta / self.count  # 更新均值
        delta2 = v - self._mean  # 计算新差值
        self._m2 += delta * delta2  # 更新M2

    def variance(self) -> float:  # 计算总体方差
        if self.count == 0:  # 如果没有数据
            return 0.0  # 返回0
        return self._m2 / self.count  # 返回方差


class ScheduledRequestMetrics(  # 本次迭代中已调度请求的指标
    msgspec.Struct,
    frozen=True,  # 不可变
    gc=False,  # 禁用垃圾回收
):
    """Metrics for requests scheduled in this iteration."""

    num_prefill_requests: int = 0  # 预填充请求数
    sum_prefill_tokens: int = 0  # 预填充标记总和
    var_prefill_length: float = 0.0  # 预填充长度方差
    sum_prefill_kv_tokens: int = 0  # 预填充KV标记总和
    num_decode_requests: int = 0  # 解码请求数
    sum_decode_kv_tokens: int = 0  # 解码KV标记总和
    var_decode_kv_tokens: float = 0.0  # 解码KV标记方差


class QueuedRequestMetrics(  # 队列中等待请求的指标
    msgspec.Struct,
    frozen=True,  # 不可变
    gc=False,  # 禁用垃圾回收
):
    """Metrics for requests waiting in the queue."""

    num_prefill_requests: int = 0  # 预填充请求数
    sum_prefill_tokens: int = 0  # 预填充标记总和
    var_prefill_length: float = 0.0  # 预填充长度方差
    num_decode_requests: int = 0  # 解码请求数
    sum_decode_kv_tokens: int = 0  # 解码KV标记总和
    var_decode_kv_tokens: float = 0.0  # 解码KV标记方差


class ForwardPassMetrics(  # 每次迭代的前向传播指标
    msgspec.Struct,
    frozen=True,  # 不可变
    gc=False,  # 禁用垃圾回收
):
    """Per-iteration metrics emitted by the scheduler.

    One message per scheduler iteration (one per forward pass).
    ``wall_time`` is the iteration duration in seconds.
    An idle heartbeat (all zeros, wall_time=0) is emitted when the
    engine transitions from active to idle.

    Field order must match Dynamo's ``ForwardPassMetrics`` in
    ``dynamo.common.forward_pass_metrics`` — msgspec uses positional
    encoding so any mismatch silently corrupts data.
    """

    version: int = FPM_VERSION  # 版本号
    worker_id: str = ""  # 工作节点ID
    dp_rank: int = 0  # 数据并行排名
    counter_id: int = 0  # 计数器ID
    wall_time: float = 0.0  # 迭代耗时（秒）
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()  # 已调度请求指标
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()  # 队列请求指标


_encoder = msgspec.msgpack.Encoder()  # 创建msgpack编码器
_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)  # 创建msgpack解码器


def encode(metrics: ForwardPassMetrics) -> bytes:  # 编码前向传播指标为字节
    return _encoder.encode(metrics)  # 返回编码后的字节


def decode(data: bytes) -> ForwardPassMetrics:  # 从字节解码前向传播指标
    return _decoder.decode(data)  # 返回解码后的指标


class _FpmPublisherThread:  # 后台线程，序列化并通过ZMQ发送前向传播指标
    """Background thread that serializes and sends ForwardPassMetrics over ZMQ.

    Also emits periodic heartbeats when idle.
    """

    SHUTDOWN_TIMEOUT: float = 1.0  # 关闭超时时间
    HEARTBEAT_INTERVAL: float = 1.0  # 心跳间隔

    def __init__(  # 初始化发布线程
        self,
        endpoint: str,  # ZMQ端点地址
        worker_id: str,  # 工作节点ID
        dp_rank: int,  # 数据并行排名
        max_queue_size: int = 10_000,  # 最大队列大小
    ) -> None:
        import zmq  # 导入ZMQ

        self._queue: queue.Queue[ForwardPassMetrics | None] = queue.Queue(  # 创建指标队列
            maxsize=max_queue_size  # 设置最大队列大小
        )
        self._seq = count()  # 创建计数器
        self._worker_id = worker_id  # 保存工作节点ID
        self._dp_rank = dp_rank  # 保存数据并行排名

        self._ctx = zmq.Context()  # 创建ZMQ上下文
        self._pub = self._ctx.socket(zmq.PUB)  # 创建PUB套接字
        self._pub.bind(endpoint)  # 绑定端点
        self._zmq = zmq  # 保存ZMQ引用

        self._running = True  # 运行标志
        self._thread = threading.Thread(  # 创建线程
            target=self._run, daemon=True, name="fpm-zmq-publisher"  # 守护线程
        )
        self._thread.start()  # 启动线程

    def publish(self, metrics: ForwardPassMetrics) -> None:  # 发布指标到队列
        if not self._running:  # 如果未运行
            return  # 直接返回
        try:  # 尝试入队
            self._queue.put_nowait(metrics)  # 非阻塞入队
        except queue.Full:  # 队列满则丢弃
            pass  # 忽略

    def shutdown(self) -> None:  # 关闭发布线程
        self._running = False  # 设置运行标志为False
        try:  # 尝试发送关闭信号
            self._queue.put_nowait(None)  # 入队None作为关闭信号
        except queue.Full:  # 队列满则忽略
            pass  # 忽略
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)  # 等待线程结束
        try:  # 尝试关闭ZMQ
            self._pub.close(linger=0)  # 关闭PUB套接字
            self._ctx.term()  # 终止ZMQ上下文
        except Exception:  # 捕获异常
            pass  # 忽略

    def _run(self) -> None:  # 线程运行主循环
        zmq = self._zmq  # 获取ZMQ引用
        topic = b""  # 空主题
        last_publish = time.monotonic()  # 记录上次发布时间

        while self._running or not self._queue.empty():  # 运行中或队列非空
            try:  # 尝试获取指标
                metrics = self._queue.get(timeout=self.HEARTBEAT_INTERVAL)  # 带超时获取
                if metrics is None:  # 如果收到关闭信号
                    break  # 跳出循环
            except queue.Empty:  # 队列为空
                if time.monotonic() - last_publish >= self.HEARTBEAT_INTERVAL:  # 检查是否需要心跳
                    metrics = ForwardPassMetrics(  # 创建心跳指标
                        worker_id=self._worker_id,  # 工作节点ID
                        dp_rank=self._dp_rank,  # 数据并行排名
                    )
                else:  # 不需要心跳
                    continue  # 继续等待

            try:  # 尝试发送
                seq = next(self._seq)  # 获取序列号
                metrics = msgspec.structs.replace(metrics, counter_id=seq)  # 设置计数器ID
                payload = encode(metrics)  # 编码指标
                seq_bytes = seq.to_bytes(8, "big")  # 序列号转字节
                self._pub.send_multipart((topic, seq_bytes, payload), flags=zmq.NOBLOCK)  # 非阻塞发送
                last_publish = time.monotonic()  # 更新发布时间
            except zmq.Again:  # ZMQ暂时不可用
                pass  # 忽略
            except Exception:  # 其他异常
                logger.warning("FPM publisher send failed", exc_info=True)  # 记录警告
