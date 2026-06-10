# 文件名: test_scheduler_flush_cache.py - 调度器刷新缓存
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)

register_cpu_ci(est_time=14, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-b-test-cpu")


# TestSchedulerFlushCache类
class TestSchedulerFlushCache(unittest.TestCase):

    # TestSchedulerFlushCache类的内部方法_new_scheduler
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ipc_channels = MagicMock()
        scheduler.flush_cache = MagicMock(return_value=True)
        scheduler.is_fully_idle = MagicMock(return_value=False)
        scheduler.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=scheduler.flush_cache,
            is_fully_idle=scheduler.is_fully_idle,
            ipc_channels=scheduler.ipc_channels,
        )
        return scheduler

    # TestSchedulerFlushCache类的测试immediateflushnotimeout
    def test_immediate_flush_no_timeout(self):
        """No timeout → flush immediately regardless of idle state."""
        scheduler = self._new_scheduler()
        scheduler.flush_cache.return_value = False

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=None))

        self.assertFalse(output.success)  # 断言为假
        scheduler.flush_cache.assert_called_once()

    # TestSchedulerFlushCache类的测试immediateflushwhenidle
    def test_immediate_flush_when_idle(self):
        """Positive timeout but already idle → flush immediately."""
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=5.0))

        self.assertTrue(output.success)  # 断言为真
        scheduler.flush_cache.assert_called_once()

    # TestSchedulerFlushCache类的测试deferswhenbusy
    def test_defers_when_busy(self):
        """Positive timeout + busy → defers, returns None."""
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=3.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=10.0,
        ):
            output = scheduler.flush_wrapper.handle(req)

        self.assertIsNone(output)  # 断言为None
        pending_req, deadline = scheduler.flush_wrapper._pending
        self.assertIs(pending_req, req)  # 断言是同一对象
        self.assertEqual(deadline, 13.0)  # 断言相等

    # TestSchedulerFlushCache类的测试rejectswhenalreadypending
    def test_rejects_when_already_pending(self):
        """Any new request is rejected while another is pending."""
        scheduler = self._new_scheduler()
        scheduler.flush_wrapper._pending = (FlushCacheReqInput(timeout_s=10.0), 999.0)

        for timeout in [None, 5.0]:
            output = scheduler.flush_wrapper.handle(
                FlushCacheReqInput(timeout_s=timeout)
            )
            self.assertFalse(output.success)  # 断言为假
            self.assertIn("already in progress", output.message)  # 断言包含

        scheduler.flush_cache.assert_not_called()

    # TestSchedulerFlushCache类的测试pendingflushcompletesonidle
    def test_pending_flush_completes_on_idle(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 111.0)

        scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)  # 断言为None
        scheduler.flush_cache.assert_called_once()
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertTrue(out.success)  # 断言为真

    # TestSchedulerFlushCache类的测试pendingflushexpiresontimeout
    def test_pending_flush_expires_on_timeout(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 99.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)  # 断言为None
        scheduler.flush_cache.assert_not_called()
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertFalse(out.success)  # 断言为假

    # TestSchedulerFlushCache类的测试pendingflushsurvivesbeforedeadline
    def test_pending_flush_survives_before_deadline(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=5.0)
        scheduler.flush_wrapper._pending = (req, 101.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNotNone(scheduler.flush_wrapper._pending)  # 断言不为None
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
