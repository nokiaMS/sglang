# 缓存刷新包装器
# 封装缓存刷新操作，支持带超时的延迟刷新机制。
# 当服务器繁忙时，可以延迟刷新操作直到空闲，避免中断正在处理的请求。

import logging
import time
from typing import Callable, Optional, Tuple

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
)


class SchedulerFlushWrapper:
    """调度器缓存刷新包装器，支持立即刷新和带超时的延迟刷新"""

    def __init__(
        self,
        *,
        flush_cache: Callable[[], bool],
        is_fully_idle: Callable[[], bool],
        ipc_channels: SchedulerIpcChannels,
    ) -> None:
        self._flush_cache = flush_cache
        self._is_fully_idle = is_fully_idle
        self._ipc_channels = ipc_channels
        self._pending: Optional[Tuple[FlushCacheReqInput, float]] = None

    def handle(self, recv_req: FlushCacheReqInput) -> Optional[FlushCacheReqOutput]:
        """处理缓存刷新请求，支持无超时立即刷新和有超时的延迟刷新"""
        # 已有延迟刷新在等待中，拒绝新请求
        if self._pending is not None:
            return FlushCacheReqOutput(
                success=False,
                message="Another flush_cache is already in progress.",
            )

        timeout_s = float(recv_req.timeout_s or 0.0)
        # 无超时：立即刷新
        if timeout_s <= 0.0:
            return FlushCacheReqOutput(success=self._flush_cache())

        # 服务器已空闲：立即刷新
        if self._is_fully_idle():
            return FlushCacheReqOutput(success=self._flush_cache())

        # 服务器繁忙：记录为延迟刷新，等待空闲
        self._pending = (recv_req, time.monotonic() + timeout_s)
        return None

    def check_pending(self) -> None:
        """检查是否有延迟刷新请求等待处理，在空闲时执行或超时时取消"""
        if self._pending is None:
            return

        pending_req, deadline = self._pending

        # 服务器空闲：执行延迟刷新
        if self._is_fully_idle():
            success = self._flush_cache()
            self._pending = None
            self._ipc_channels.send_to_tokenizer.send_output(
                FlushCacheReqOutput(success=success), pending_req
            )
            return

        # 超时：取消延迟刷新并通知失败
        if time.monotonic() >= deadline:
            logging.warning(
                "Deferred flush_cache timed out while waiting for idle state."
            )
            self._pending = None
            self._ipc_channels.send_to_tokenizer.send_output(
                FlushCacheReqOutput(
                    success=False, message="Timed out waiting for idle state."
                ),
                pending_req,
            )
