# 调度器输入阻塞器模块
# 实现调度器输入请求的阻塞与解除阻塞机制，支持全局同步屏障

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
import logging  # 日志库
from contextlib import contextmanager  # 上下文管理器装饰器
from enum import Enum, auto  # 枚举类型
from typing import Any, List, Optional  # 类型提示

from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType  # 阻塞请求输入和类型
from sglang.srt.utils.poll_based_barrier import PollBasedBarrier  # 基于轮询的屏障

logger = logging.getLogger(__name__)  # 获取日志记录器


class SchedulerInputBlocker:  # 调度器输入阻塞器，控制请求的阻塞与放行
    def __init__(self, noop: bool):  # 初始化输入阻塞器
        self._state = _State.UNBLOCKED  # 初始状态为未阻塞
        self._pending_reqs = []  # 待处理的请求列表
        self._noop = noop  # 是否为空操作模式（不阻塞）
        self._global_unblock_barrier = PollBasedBarrier(noop=noop)  # 全局解除阻塞屏障

    def handle(self, recv_reqs: Optional[List[Any]]):  # 处理接收到的请求列表
        assert (recv_reqs is None) == self._noop  # 断言：空操作模式时recv_reqs为None

        if not self._noop:  # 如果不是空操作模式
            output_reqs = []  # 输出请求列表
            for recv_req in recv_reqs:  # 遍历接收到的请求
                output_reqs += self._handle_recv_req(recv_req)  # 处理每个请求并累加输出

        global_arrived_unblock_barrier = (  # 检查全局解除阻塞屏障是否到达
            self._global_unblock_barrier.poll_global_arrived()
        )
        if (  # 如果当前在屏障等待状态且全局屏障已到达
            self._state == _State.GLOBAL_UNBLOCK_BARRIER
            and global_arrived_unblock_barrier
        ):
            output_reqs += self._handle_arrive_unblock_barrier()  # 处理到达解除阻塞屏障

        if not self._noop:  # 如果不是空操作模式
            return output_reqs  # 返回输出请求列表

    def _handle_recv_req(self, recv_req):  # 处理单个接收到的请求
        if isinstance(recv_req, BlockReqInput):  # 如果是阻塞请求
            if recv_req.type == BlockReqType.BLOCK:  # 如果是阻塞类型
                self._execute_block_req()  # 执行阻塞
                return []  # 返回空列表
            elif recv_req.type == BlockReqType.UNBLOCK:  # 如果是解除阻塞类型
                self._execute_unblock_req()  # 执行解除阻塞
                return []  # 返回空列表
            else:  # 其他类型
                raise NotImplementedError(f"{recv_req=}")  # 抛出未实现错误
        else:  # 如果是普通请求
            if self._state == _State.UNBLOCKED:  # 如果当前未阻塞
                return [recv_req]  # 直接放行请求
            else:  # 如果当前已阻塞
                self._pending_reqs.append(recv_req)  # 加入待处理列表
                return []  # 返回空列表

    def _execute_block_req(self):  # 执行阻塞请求
        logger.info("Handle block req")  # 记录处理阻塞请求
        self._change_state(original=_State.UNBLOCKED, target=_State.BLOCKED)  # 从未阻塞切换到阻塞

    def _execute_unblock_req(self):  # 执行解除阻塞请求
        logger.info("Handle unblock req")  # 记录处理解除阻塞请求
        self._change_state(  # 从阻塞切换到全局解除阻塞屏障
            original=_State.BLOCKED, target=_State.GLOBAL_UNBLOCK_BARRIER
        )
        self._global_unblock_barrier.local_arrive()  # 本地到达屏障

    def _handle_arrive_unblock_barrier(self):  # 处理到达全局解除阻塞屏障
        logger.info(f"Arrived at unblock barrier ({len(self._pending_reqs)=})")  # 记录到达屏障
        self._change_state(  # 从全局解除阻塞屏障切换到未阻塞
            original=_State.GLOBAL_UNBLOCK_BARRIER, target=_State.UNBLOCKED
        )
        output_reqs = [*self._pending_reqs]  # 复制待处理请求列表
        self._pending_reqs.clear()  # 清空待处理列表
        return output_reqs  # 返回待处理请求

    def _change_state(self, original: "_State", target: "_State"):  # 切换阻塞器状态
        assert self._state == original, f"{self._state=} {original=} {target=}"  # 断言当前状态与期望一致
        self._state = target  # 设置新状态


class _State(Enum):  # 阻塞器状态枚举
    UNBLOCKED = auto()  # 未阻塞
    BLOCKED = auto()  # 已阻塞
    GLOBAL_UNBLOCK_BARRIER = auto()  # 全局解除阻塞屏障等待中


@contextmanager
def input_blocker_guard_region(send_to_scheduler):  # 输入阻塞器保护区上下文管理器，在区内阻塞输入，退出时解除阻塞
    send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.BLOCK))  # 进入时发送阻塞请求
    try:  # 尝试执行
        yield  # 执行保护区代码
    finally:  # 无论如何都执行
        send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.UNBLOCK))  # 退出时发送解除阻塞请求
