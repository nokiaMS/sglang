# 通信器模块：提供基于 ZeroMQ 的扇出（Fan-Out）请求分发与响应收集原语。
# 支持两种并发模式：排队模式（queueing）和观察模式（watching），
# 用于在调度器与多个工作进程之间进行一对多的请求-响应通信。

from __future__ import annotations

import asyncio
import copy
from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar

import zmq

T = TypeVar("T")  # 泛型类型变量，用于参数化通信器中传输的对象类型


class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.
    """

    # 扇出通信器：将一个请求发送给 fan_out 个接收者，并等待收集所有响应。
    # 排队模式下并发调用者按 FIFO 顺序依次执行；观察模式下并发调用者共享
    # 同一个进行中的请求，全部获得相同结果。任意时刻最多只有一个请求在飞行中。

    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        # 初始化扇出通信器
        self._sender = sender  # 用于发送请求的 ZMQ 套接字
        self._fan_out = fan_out  # 扇出数量，即需要收集的响应数
        self._mode = mode  # 通信模式："queueing"（排队）或 "watching"（观察）
        self._result_event: Optional[asyncio.Event] = None  # 用于通知所有响应已收集完成的事件
        self._result_values: Optional[List[T]] = None  # 存储已收集到的响应值列表
        self._ready_queue: Deque[asyncio.Event] = deque()  # 排队模式下等待执行的调用者事件队列

        assert mode in ["queueing", "watching"]  # 模式必须是排队或观察之一

    async def queueing_call(self, obj: T):
        # 排队模式调用：请求被串行化执行，并发调用者在 FIFO 队列中等待
        ready_event = asyncio.Event()
        # 如果当前有正在进行的请求或队列中已有等待者，则加入等待队列
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()  # 等待轮到自己执行
            # 被唤醒后，确保共享状态已被清理
            assert self._result_event is None
            assert self._result_values is None

        # 如果传入对象不为 None，则通过 ZMQ 套接字发送请求
        if obj is not None:
            self._sender.send_pyobj(obj)

        # 初始化结果收集状态，等待所有响应到达
        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()  # 阻塞直到所有 fan_out 个响应都收集完毕

        # 取出结果并清理共享状态
        result_values = self._result_values
        self._result_event = self._result_values = None

        # 如果队列中还有等待者，唤醒下一个
        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj):
        # 观察模式调用：并发调用者共享同一个进行中的请求，全部获得相同结果
        if self._result_event is None:
            # 当前没有进行中的请求，创建新的请求
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            # 如果传入对象不为 None，则通过 ZMQ 套接字发送请求
            if obj is not None:
                self._sender.send_pyobj(obj)

        # Capture local refs before await -- after event fires, the first
        # awakened coroutine clears shared state; later awaiters use local refs.
        # 在 await 之前捕获本地引用——事件触发后，第一个被唤醒的协程会清理共享状态；
        # 后续的等待者使用本地引用来获取结果。
        values = self._result_values
        event = self._result_event
        await event.wait()  # 等待所有响应到达

        # 深拷贝结果，因为多个调用者共享同一份数据
        result_values = copy.deepcopy(values)
        # 只有第一个被唤醒的协程（即 event 是当前共享事件的那个）负责清理共享状态
        if self._result_event is event:
            self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj):
        # 根据配置的模式分发到对应的调用方法
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        # 处理收到的单个响应：将其追加到结果列表中
        self._result_values.append(recv_obj)
        # 当收集到的响应数量等于扇出数量时，设置事件通知等待者
        if len(self._result_values) == self._fan_out:
            self._result_event.set()

    @staticmethod
    def merge_results(results):
        # 合并多个工作进程的结果：检查是否全部成功，并拼接所有消息
        all_success = all([r.success for r in results])  # 所有结果都成功才算成功
        all_message = [r.message for r in results]  # 收集所有结果的消息
        all_message = " | ".join(all_message)  # 用管道符拼接所有消息
        return all_success, all_message
