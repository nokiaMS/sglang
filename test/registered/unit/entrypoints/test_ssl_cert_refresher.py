# 文件名: test_ssl_cert_refresher.py - SSL证书刷新器
import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from sglang.srt.entrypoints.ssl_utils import SSLCertRefresher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=14, suite="base-a-test-cpu")


# 内部方法_make_temp_pem
def _make_temp_pem(content: bytes) -> str:
    """Create a temporary PEM file and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name


# TestSSLCertRefresher类
class TestSSLCertRefresher(CustomTestCase):
    """Tests for the SSLCertRefresher class."""

    def setUp(self):
        super().setUp()
        self._temp_files: list[str] = []

    # TestSSLCertRefresher类的测试清理
    def tearDown(self):
        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        super().tearDown()

    # TestSSLCertRefresher类的内部方法_track
    def _track(self, path: str) -> str:
        """Register a temp file for cleanup."""
        self._temp_files.append(path)
        return path

    # TestSSLCertRefresher类的内部方法_run_async
    def _run_async(self, coro):
        """Helper to run an async coroutine in tests."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # TestSSLCertRefresher类的测试reloadcertkeyonfilechange
    def test_reload_cert_key_on_file_change(self):
        """SSLCertRefresher calls load_cert_chain when cert/key files change."""
        mock_ctx = MagicMock()
        cert_path = self._track(_make_temp_pem(b"CERT_V1"))
        key_path = self._track(_make_temp_pem(b"KEY_V1"))

        async def _test():
            refresher = SSLCertRefresher(mock_ctx, key_path, cert_path)
            await asyncio.sleep(0.3)

            with open(cert_path, "w") as f:
                f.write("CERT_V2")

            await asyncio.sleep(1.5)
            refresher.stop()
            return mock_ctx

        result_ctx = self._run_async(_test())
        result_ctx.load_cert_chain.assert_called_with(cert_path, key_path)

    # TestSSLCertRefresher类的测试reloadcaonfilechange
    def test_reload_ca_on_file_change(self):
        """SSLCertRefresher calls load_verify_locations when CA file changes."""
        mock_ctx = MagicMock()
        cert_path = self._track(_make_temp_pem(b"CERT"))
        key_path = self._track(_make_temp_pem(b"KEY"))
        ca_path = self._track(_make_temp_pem(b"CA_V1"))

        async def _test():
            refresher = SSLCertRefresher(mock_ctx, key_path, cert_path, ca_path)
            await asyncio.sleep(0.3)

            with open(ca_path, "w") as f:
                f.write("CA_V2")

            await asyncio.sleep(1.5)
            refresher.stop()
            return mock_ctx

        result_ctx = self._run_async(_test())
        result_ctx.load_verify_locations.assert_called_with(ca_path)

    # TestSSLCertRefresher类的测试stopcancelstasks
    def test_stop_cancels_tasks(self):
        """Calling stop() prevents further reloads."""
        mock_ctx = MagicMock()
        cert_path = self._track(_make_temp_pem(b"CERT"))
        key_path = self._track(_make_temp_pem(b"KEY"))

        async def _test():
            refresher = SSLCertRefresher(mock_ctx, key_path, cert_path)
            await asyncio.sleep(0.2)

            refresher.stop()

            with open(cert_path, "w") as f:
                f.write("CERT_AFTER_STOP")

            await asyncio.sleep(1.0)
            return mock_ctx

        result_ctx = self._run_async(_test())
        result_ctx.load_cert_chain.assert_not_called()

    # TestSSLCertRefresher类的测试nocawatcherwhencanotprovided
    def test_no_ca_watcher_when_ca_not_provided(self):
        """No CA watcher task is created when ca_path is None."""
        mock_ctx = MagicMock()
        cert_path = self._track(_make_temp_pem(b"CERT"))
        key_path = self._track(_make_temp_pem(b"KEY"))

        async def _test():
            refresher = SSLCertRefresher(mock_ctx, key_path, cert_path)
            self.assertEqual(len(refresher._tasks), 1)  # 断言相等
            refresher.stop()

        self._run_async(_test())

    # TestSSLCertRefresher类的测试cawatchercreatedwhencaprovided
    def test_ca_watcher_created_when_ca_provided(self):
        """A CA watcher task is created when ca_path is provided."""
        mock_ctx = MagicMock()
        cert_path = self._track(_make_temp_pem(b"CERT"))
        key_path = self._track(_make_temp_pem(b"KEY"))
        ca_path = self._track(_make_temp_pem(b"CA"))

        async def _test():
            refresher = SSLCertRefresher(mock_ctx, key_path, cert_path, ca_path)
            self.assertEqual(len(refresher._tasks), 2)  # 断言相等
            refresher.stop()

        self._run_async(_test())

    # TestSSLCertRefresher类的测试reloaderrordoesnotcrash
    def test_reload_error_does_not_crash(self):
        """A reload error is logged but doesn't crash the watcher."""
        mock_ctx = MagicMock()
        mock_ctx.load_cert_chain.side_effect = Exception("bad cert")
        cert_path = self._track(_make_temp_pem(b"CERT"))
        key_path = self._track(_make_temp_pem(b"KEY"))

        async def _test():
            refresher = SSLCertRefresher(mock_ctx, key_path, cert_path)
            await asyncio.sleep(0.3)

            with open(cert_path, "w") as f:
                f.write("BAD_CERT")

            await asyncio.sleep(1.5)

            for task in refresher._tasks:
                self.assertFalse(task.done())  # 断言为假

            refresher.stop()

        self._run_async(_test())


if __name__ == "__main__":
    unittest.main()
