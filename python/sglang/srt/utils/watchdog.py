# 本文件实现了看门狗(Watchdog)机制，用于监控 SGLang 服务中调度器等关键组件的活跃状态。
# 当组件在指定超时时间内未更新计数器（即"喂狗"），看门狗将触发超时处理：
# - 软看门狗(soft)：仅打印错误日志和堆栈信息
# - 硬看门狗：向父进程发送 SIGQUIT 信号以强制退出
# 此外还包含子进程看门狗(SubprocessWatchdog)，用于检测子进程崩溃并触发清理。

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import Process
from typing import Callable, List, Optional

import psutil

from sglang.srt.utils.common import pyspy_dump_schedulers

logger = logging.getLogger(__name__)


class Watchdog:
    """看门狗基类，提供工厂方法创建实际的看门狗实例。
    如果未设置超时时间，则返回空操作看门狗(_WatchdogNoop)；
    否则返回实际看门狗(_WatchdogReal)。"""

    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        """工厂方法：根据参数创建看门狗实例。
        :param debug_name: 调试名称，用于日志标识
        :param watchdog_timeout: 超时时间（秒），None 表示禁用看门狗
        :param soft: 是否为软看门狗（仅打印日志，不发送信号）
        :param test_stuck_time: 测试用的故意卡住时间，仅当软看门狗启用时可用
        """
        if watchdog_timeout is None:
            assert (
                test_stuck_time == 0
            ), f"stuck tester can be enabled only if soft watchdog is enabled."
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
        )

    def feed(self):
        """喂狗：更新计数器，表示组件仍然活跃。"""
        pass

    @contextmanager
    def disable(self):
        """上下文管理器：临时禁用看门狗超时检测。"""
        yield


class _WatchdogReal(Watchdog):
    """实际看门狗实现，维护一个计数器，每次 feed 递增。
    后台线程定期检查计数器是否变化，超时未变化则触发超时处理。"""

    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
    ):
        self._counter = 0  # 喂狗计数器
        self._active = True  # 看门狗是否活跃
        self._test_stuck_time = test_stuck_time  # 测试卡住时间
        self._test_stuck_triggered = False  # 测试卡住是否已触发
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
        )
        logger.info(f"Watchdog {self._raw.debug_name} initialized.")
        if self._test_stuck_time > 0:
            logger.info(
                f"Watchdog {self._raw.debug_name} is configured to use {test_stuck_time=}."
            )

    def feed(self):
        """喂狗：递增计数器。如果配置了测试卡住时间，首次喂狗时会故意阻塞指定时间。"""
        # Only trigger the test stuck behavior once to avoid blocking server
        # startup health checks while still testing watchdog timeout detection
        if self._test_stuck_time > 0 and not self._test_stuck_triggered:
            self._test_stuck_triggered = True
            logger.info(
                f"Watchdog {self._raw.debug_name} start deliberately stuck for {self._test_stuck_time}s"
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                f"Watchdog {self._raw.debug_name} end deliberately stuck for {self._test_stuck_time}s"
            )

        self._counter += 1

    @contextmanager
    def disable(self):
        """临时禁用看门狗超时检测，用于执行可能较长时间的初始化等操作。"""
        assert self._active
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True


class _WatchdogNoop(Watchdog):
    """空操作看门狗，所有方法均为空实现，用于未启用看门狗时避免额外开销。"""
    pass


class WatchdogRaw:
    """底层看门狗实现，在后台守护线程中定期检查计数器是否更新。
    如果在超时时间内计数器未变化，则触发超时处理。"""

    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
    ):
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info

        # 获取父进程引用，硬看门狗超时时向其发送信号
        self.parent_process = psutil.Process().parent()
        # 启动后台监控线程
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        """看门狗后台线程主循环，持续执行超时检测。"""
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            logger.error(
                f"{self.debug_name} watchdog thread crashed: {e}", exc_info=True
            )

    def _watchdog_once(self):
        """执行一次超时检测：等待计数器在超时时间内变化，否则触发超时处理。"""
        watchdog_last_counter = 0
        watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    # 计数器未变化，检查是否超时
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    # 计数器已变化，重置超时计时
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            # 以超时时间的一半为间隔轮询
            time.sleep(self.watchdog_timeout / 2)

        # 超时处理：输出调试信息
        if self.dump_info is not None and (info_msg := self.dump_info()):
            logger.error(f"{self.debug_name} debug info:\n{info_msg}")

        # 使用 py-spy 导出调度器堆栈
        pyspy_dump_schedulers()
        logger.error(
            f"{self.debug_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            # Wait for some time so that the parent process can print the error.
            time.sleep(5)
            # 向父进程发送 SIGQUIT 信号触发清理
            self.parent_process.send_signal(signal.SIGQUIT)


class SubprocessWatchdog:
    """Monitors subprocess liveness and triggers SIGQUIT when a crash is detected.

    When a subprocess crashes (e.g., NCCL timeout causing C++ std::terminate()),
    Python exception handlers never run, leaving the main process as a zombie
    service. This watchdog polls subprocess liveness in a daemon thread and
    sends SIGQUIT to trigger proper cleanup.

    子进程看门狗：监控子进程存活状态，当检测到子进程崩溃时向自身发送 SIGQUIT 信号
    触发清理。用于处理 NCCL 超时等导致 C++ 层直接终止而 Python 异常处理不执行的情况。

    See: https://github.com/sgl-project/sglang/issues/18421
    """

    def __init__(
        self,
        processes: List[Process],
        process_names: Optional[List[str]] = None,
        interval: float = 1.0,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval  # 轮询间隔（秒）
        self._stop_event = threading.Event()  # 停止事件标志
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动子进程监控线程。"""
        if self._thread is not None or not self._processes:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="subprocess-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        """停止子进程监控线程。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _monitor_loop(self) -> None:
        """监控循环：定期检查所有子进程的存活状态。"""
        try:
            while not self._stop_event.wait(self._interval):
                if self._check_processes():
                    return
        except Exception as e:
            logger.error(f"SubprocessWatchdog thread crashed: {e}", exc_info=True)

    def _check_processes(self) -> bool:
        """检查所有子进程是否存活，如果发现崩溃则发送 SIGQUIT 信号。
        返回 True 表示检测到崩溃。"""
        for proc, name in zip(self._processes, self._names):
            # 进程存活或正常退出（exitcode==0）则跳过
            if proc.is_alive() or proc.exitcode == 0:
                continue

            logger.error(
                f"Subprocess {name} (pid={proc.pid}) crashed "
                f"with exit code {proc.exitcode}. "
                f"Triggering SIGQUIT for cleanup..."
            )
            # 向当前进程发送 SIGQUIT 信号触发清理
            os.kill(os.getpid(), signal.SIGQUIT)
            return True
        return False
