# 文件名: test_request_metrics_exporter.py - 请求指标导出器
"""Unit tests for request_metrics_exporter.py — no server, no model loading."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")

import asyncio
import json
import os
import shutil
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX

# ── Test helper classes (local only, never injected into sys.modules) ──


@dataclass

# _GenerateReqInput类
class _GenerateReqInput:
    rid: Optional[str] = None
    text: Optional[str] = None
    image_data: Optional[Any] = None
    sampling_params: Optional[Dict] = None


@dataclass

# _EmbeddingReqInput类
class _EmbeddingReqInput:
    rid: Optional[str] = None
    text: Optional[str] = None
    image_data: Optional[Any] = None
    input_ids: Optional[List[int]] = None


# _ServerArgs类
class _ServerArgs:

    # _ServerArgs类的初始化
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Deferred import of the module-under-test ──
# request_metrics_exporter.py imports io_struct and server_args at module level.
# We use patch.dict to temporarily provide lightweight stubs so the import
# succeeds without pulling in heavy transitive deps (torch, triton, …).
# The patch is started in setUpModule and stopped in tearDownModule,
# so sys.modules is never modified during pytest collection.

_patcher = None

# Module-under-test symbols, populated by setUpModule
FileRequestMetricsExporter = None
RequestMetricsExporter = None
RequestMetricsExporterManager = None
create_request_metrics_exporters = None
_ConcreteExporter = None


# setUpModule
def setUpModule():
    global _patcher
    global FileRequestMetricsExporter, RequestMetricsExporter
    global RequestMetricsExporterManager, create_request_metrics_exporters
    global _ConcreteExporter

    stub_modules = {}
    for name in (
        "sglang.srt.managers",
        "sglang.srt.managers.io_struct",
        "sglang.srt.server_args",
    ):
        if name not in __import__("sys").modules:
            stub_modules[name] = types.ModuleType(name)

    if stub_modules:
        if "sglang.srt.managers.io_struct" in stub_modules:
            stub_modules["sglang.srt.managers.io_struct"].GenerateReqInput = (
                _GenerateReqInput
            )
            stub_modules["sglang.srt.managers.io_struct"].EmbeddingReqInput = (
                _EmbeddingReqInput
            )
        if "sglang.srt.server_args" in stub_modules:
            stub_modules["sglang.srt.server_args"].ServerArgs = _ServerArgs

        _patcher = patch.dict("sys.modules", stub_modules)
        _patcher.start()

    import sglang.srt.observability.request_metrics_exporter as _mod

    FileRequestMetricsExporter = _mod.FileRequestMetricsExporter
    RequestMetricsExporter = _mod.RequestMetricsExporter
    RequestMetricsExporterManager = _mod.RequestMetricsExporterManager
    create_request_metrics_exporters = _mod.create_request_metrics_exporters

    # ConcreteExporter类
    class ConcreteExporter(RequestMetricsExporter):
        """Minimal concrete subclass for testing base class methods."""

        async def write_record(self, obj, out_dict):
            pass

    _ConcreteExporter = ConcreteExporter


# tearDownModule
def tearDownModule():
    if _patcher is not None:
        _patcher.stop()


# ── Helpers ──


def _make_server_args(tmp_dir, enabled=True):
    return _ServerArgs(
        export_metrics_to_file=enabled,
        export_metrics_to_file_dir=tmp_dir,
    )


# TestFormatOutputData类
class TestFormatOutputData(unittest.TestCase):

    # TestFormatOutputData类的测试basicformatting
    def test_basic_formatting(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names=None
        )

        obj = _GenerateReqInput(
            rid="req-1", text="hello", sampling_params={"temp": 0.5}
        )
        out_dict = {"meta_info": {"latency": 1.5, "tokens": 10}}

        result = exporter._format_output_data(obj, out_dict)

        params = json.loads(result["request_parameters"])
        self.assertEqual(params["rid"], "req-1")  # 断言相等
        self.assertEqual(params["text"], "hello")  # 断言相等
        self.assertIn("latency", result)  # 断言包含
        self.assertIn("tokens", result)  # 断言包含

    # TestFormatOutputData类的测试excludesalwaysexcludefields
    def test_excludes_always_exclude_fields(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names=None
        )

        obj = _GenerateReqInput(rid="req-1", image_data="should_be_excluded")
        result = exporter._format_output_data(obj, {})

        params = json.loads(result["request_parameters"])
        self.assertNotIn("image_data", params)  # 断言不包含

    # TestFormatOutputData类的测试excludesobjskipnames
    def test_excludes_obj_skip_names(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names={"text"}, out_skip_names=None
        )

        obj = _GenerateReqInput(rid="req-1", text="skip_me")
        result = exporter._format_output_data(obj, {})

        params = json.loads(result["request_parameters"])
        self.assertNotIn("text", params)  # 断言不包含
        self.assertIn("rid", params)  # 断言包含

    # TestFormatOutputData类的测试excludesnonevalues
    def test_excludes_none_values(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names=None
        )

        obj = _GenerateReqInput(rid="req-1", text=None)
        result = exporter._format_output_data(obj, {})

        params = json.loads(result["request_parameters"])
        self.assertNotIn("text", params)  # 断言不包含

    # TestFormatOutputData类的测试filtersoutskipnames
    def test_filters_out_skip_names(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names={"secret"}
        )

        obj = _GenerateReqInput(rid="req-1")
        out_dict = {"meta_info": {"latency": 1.5, "secret": "hidden"}}
        result = exporter._format_output_data(obj, out_dict)

        self.assertIn("latency", result)  # 断言包含
        self.assertNotIn("secret", result)  # 断言不包含


# TestFileRequestMetricsExporter类
class TestFileRequestMetricsExporter(unittest.TestCase):

    # TestFileRequestMetricsExporter类的测试初始化设置
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    # TestFileRequestMetricsExporter类的测试清理
    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # TestFileRequestMetricsExporter类的内部方法_make_exporter
    def _make_exporter(self):
        return FileRequestMetricsExporter(_make_server_args(self.tmp_dir), None, None)

    # TestFileRequestMetricsExporter类的测试initcreatesdirectory
    def test_init_creates_directory(self):
        sub_dir = os.path.join(self.tmp_dir, "nested", "dir")
        FileRequestMetricsExporter(_make_server_args(sub_dir), None, None)
        self.assertTrue(os.path.isdir(sub_dir))  # 断言为真

    # TestFileRequestMetricsExporter类的测试ensurefilehandleropensfile
    def test_ensure_file_handler_opens_file(self):
        exporter = self._make_exporter()
        exporter._ensure_file_handler("20240101_12")
        self.assertIsNotNone(exporter._current_file_handler)  # 断言不为None
        self.assertEqual(exporter._current_hour_suffix, "20240101_12")  # 断言相等
        exporter.close()

    # TestFileRequestMetricsExporter类的测试ensurefilehandlerrotates
    def test_ensure_file_handler_rotates(self):
        exporter = self._make_exporter()
        exporter._ensure_file_handler("20240101_12")
        first_handler = exporter._current_file_handler
        exporter._ensure_file_handler("20240101_13")
        self.assertTrue(first_handler.closed)  # 断言为真
        self.assertEqual(exporter._current_hour_suffix, "20240101_13")  # 断言相等
        exporter.close()

    # TestFileRequestMetricsExporter类的测试ensurefilehandlercloseerror
    def test_ensure_file_handler_close_error(self):
        """Previous handler close failure is logged but doesn't prevent rotation."""
        exporter = self._make_exporter()
        mock_handler = MagicMock()
        mock_handler.close.side_effect = OSError("disk error")
        exporter._current_file_handler = mock_handler
        exporter._current_hour_suffix = "old"

        exporter._ensure_file_handler("new")
        self.assertEqual(exporter._current_hour_suffix, "new")  # 断言相等
        exporter.close()

    # TestFileRequestMetricsExporter类的测试ensurefilehandleropenerror
    def test_ensure_file_handler_open_error(self):
        exporter = self._make_exporter()
        with patch("builtins.open", side_effect=OSError("permission denied")):
            with self.assertRaises(OSError):  # 断言抛出异常
                exporter._ensure_file_handler("20240101_12")
        self.assertIsNone(exporter._current_file_handler)  # 断言为None
        self.assertIsNone(exporter._current_hour_suffix)  # 断言为None

    # TestFileRequestMetricsExporter类的测试close
    def test_close(self):
        exporter = self._make_exporter()
        exporter._ensure_file_handler("20240101_12")
        exporter.close()
        self.assertIsNone(exporter._current_file_handler)  # 断言为None
        self.assertIsNone(exporter._current_hour_suffix)  # 断言为None

    # TestFileRequestMetricsExporter类的测试closenoopwhennohandler
    def test_close_noop_when_no_handler(self):
        exporter = self._make_exporter()
        exporter.close()  # should not raise

    # TestFileRequestMetricsExporter类的测试closeerror
    def test_close_error(self):
        """Close failure is logged but state is still reset."""
        exporter = self._make_exporter()
        mock_handler = MagicMock()
        mock_handler.close.side_effect = OSError("disk error")
        exporter._current_file_handler = mock_handler
        exporter._current_hour_suffix = "old"

        exporter.close()
        self.assertIsNone(exporter._current_file_handler)  # 断言为None
        self.assertIsNone(exporter._current_hour_suffix)  # 断言为None

    # TestFileRequestMetricsExporter类的测试writerecord
    def test_write_record(self):
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid="req-1", text="hello")
        out_dict = {"meta_info": {"latency": 1.5}}

        asyncio.run(exporter.write_record(obj, out_dict))

        # Find the written file
        files = os.listdir(self.tmp_dir)
        self.assertEqual(len(files), 1)  # 断言相等
        with open(os.path.join(self.tmp_dir, files[0])) as f:
            record = json.loads(f.readline())
        self.assertIn("request_parameters", record)  # 断言包含
        self.assertAlmostEqual(record["latency"], 1.5)  # 断言近似相等
        exporter.close()

    # TestFileRequestMetricsExporter类的测试writerecordskipshealthcheck
    def test_write_record_skips_health_check(self):
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid=f"{HEALTH_CHECK_RID_PREFIX}_123", text="ping")
        asyncio.run(exporter.write_record(obj, {}))

        files = os.listdir(self.tmp_dir)
        self.assertEqual(len(files), 0)  # 断言相等

    # TestFileRequestMetricsExporter类的测试writerecordhandlernone
    def test_write_record_handler_none(self):
        """If file handler is None after ensure, write_record returns early."""
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid="req-1")

        with patch.object(exporter, "_ensure_file_handler"):
            exporter._current_file_handler = None
            asyncio.run(exporter.write_record(obj, {}))
        # No crash, no file written

    def test_write_record_exception(self):
        """Exceptions during write are caught and logged."""
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid="req-1")

        with patch.object(
            exporter, "_ensure_file_handler", side_effect=RuntimeError("boom")
        ):
            asyncio.run(exporter.write_record(obj, {}))
        # Should not raise


class TestRequestMetricsExporterManager(unittest.TestCase):

    # TestRequestMetricsExporterManager类的测试初始化设置
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    # TestRequestMetricsExporterManager类的测试清理
    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # TestRequestMetricsExporterManager类的测试noexporters
    def test_no_exporters(self):
        server_args = _make_server_args(self.tmp_dir, enabled=False)
        manager = RequestMetricsExporterManager(server_args)
        self.assertFalse(manager.exporter_enabled())  # 断言为假

    # TestRequestMetricsExporterManager类的测试withfileexporter
    def test_with_file_exporter(self):
        server_args = _make_server_args(self.tmp_dir, enabled=True)
        manager = RequestMetricsExporterManager(server_args)
        self.assertTrue(manager.exporter_enabled())  # 断言为真

    # TestRequestMetricsExporterManager类的测试writerecorddelegates
    def test_write_record_delegates(self):
        server_args = _make_server_args(self.tmp_dir, enabled=True)
        manager = RequestMetricsExporterManager(server_args)

        obj = _GenerateReqInput(rid="req-1", text="hello")
        out_dict = {"meta_info": {"latency": 1.0}}
        asyncio.run(manager.write_record(obj, out_dict))

        files = os.listdir(self.tmp_dir)
        self.assertEqual(len(files), 1)  # 断言相等


# TestCreateExporters类
class TestCreateExporters(unittest.TestCase):

    # TestCreateExporters类的测试初始化设置
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    # TestCreateExporters类的测试清理
    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # TestCreateExporters类的测试disabled
    def test_disabled(self):
        server_args = _make_server_args(self.tmp_dir, enabled=False)
        exporters = create_request_metrics_exporters(server_args)
        self.assertEqual(len(exporters), 0)  # 断言相等

    # TestCreateExporters类的测试enabled
    def test_enabled(self):
        server_args = _make_server_args(self.tmp_dir, enabled=True)
        exporters = create_request_metrics_exporters(server_args)
        self.assertEqual(len(exporters), 1)  # 断言相等
        self.assertIsInstance(exporters[0], FileRequestMetricsExporter)


if __name__ == "__main__":
    unittest.main()
