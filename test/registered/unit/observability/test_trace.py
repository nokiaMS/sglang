# 文件名: test_trace.py - 追踪
"""Unit tests for trace.py — no server, no model loading."""

import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import threading
import unittest
from unittest.mock import patch

import sglang.srt.observability.trace as mod
from sglang.srt.observability.trace import (
    SpanAttributes,
    TraceCustomIdGenerator,
    TraceEvent,
    TraceNullContext,
    TraceReqContext,
    TraceSliceContext,
    TraceThreadContext,
    TraceThreadInfo,
    extract_trace_headers,
    get_global_tracing_enabled,
    process_tracing_init,
    set_global_trace_level,
    trace_set_thread_info,
)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    from sglang.srt.observability.trace import get_otlp_span_exporter

    _has_otel = True
except ImportError:
    _has_otel = False

# Access the private module-level function (avoid name mangling inside classes).
_get_host_id = getattr(mod, "__get_host_id")


# TestTraceFunctions类
class TestTraceFunctions(unittest.TestCase):

    # TestTraceFunctions类的测试extracttraceheaders
    def test_extract_trace_headers(self):
        headers = {"traceparent": "abc", "tracestate": "xyz", "other": "skip"}
        result = extract_trace_headers(headers)
        self.assertEqual(result, {"traceparent": "abc", "tracestate": "xyz"})  # 断言相等

    # TestTraceFunctions类的测试extracttraceheadersmissing
    def test_extract_trace_headers_missing(self):
        self.assertEqual(extract_trace_headers({}), {})  # 断言相等

    # TestTraceFunctions类的测试setglobaltracelevel
    def test_set_global_trace_level(self):
        orig = mod.global_trace_level
        set_global_trace_level(5)
        self.assertEqual(mod.global_trace_level, 5)  # 断言相等
        mod.global_trace_level = orig

    # TestTraceFunctions类的测试globaltracelevelenvvar
    def test_global_trace_level_env_var(self):
        import importlib

        with patch.dict(os.environ, {"SGLANG_TRACE_LEVEL": "2"}):
            importlib.reload(mod)
            self.assertEqual(mod.global_trace_level, 2)  # 断言相等
        importlib.reload(mod)  # restore default (SGLANG_TRACE_LEVEL unset → 3)
        self.assertEqual(mod.global_trace_level, 3)  # 断言相等

    # TestTraceFunctions类的测试getglobaltracingenabled
    def test_get_global_tracing_enabled(self):
        self.assertEqual(get_global_tracing_enabled(), mod.opentelemetry_initialized)  # 断言相等

    # TestTraceFunctions类的测试getcurtimens
    def test_get_cur_time_ns(self):
        ts = mod.get_cur_time_ns()
        self.assertIsInstance(ts, int)
        self.assertGreater(ts, 0)  # 断言大于


# TestDataclasses类
class TestDataclasses(unittest.TestCase):

    # TestDataclasses类的测试tracethreadinfo
    def test_trace_thread_info(self):
        info = TraceThreadInfo("host", 123, "label", 0, 1, 0)
        self.assertEqual(info.thread_label, "label")  # 断言相等

    # TestDataclasses类的测试traceevent
    def test_trace_event(self):
        evt = TraceEvent("name", 100, {"k": "v"})
        self.assertEqual(evt.event_name, "name")  # 断言相等

    # TestDataclasses类的测试traceslicecontext
    def test_trace_slice_context(self):
        s = TraceSliceContext("slice", 100, end_time_ns=200, level=2, attrs={"a": 1})
        self.assertEqual(s.slice_name, "slice")  # 断言相等

    # TestDataclasses类的测试tracethreadcontext
    def test_trace_thread_context(self):
        info = TraceThreadInfo("h", 1, "l", 0, 0, 0)
        ctx = TraceThreadContext(thread_info=info, cur_slice_stack=[])
        self.assertEqual(len(ctx.cur_slice_stack), 0)  # 断言相等


# TestTraceNullContext类
class TestTraceNullContext(unittest.TestCase):

    # TestTraceNullContext类的测试nullobjectpattern
    def test_null_object_pattern(self):
        ctx = TraceNullContext()
        self.assertFalse(ctx.tracing_enable)  # 断言为假
        # Any attribute access returns self
        self.assertIs(ctx.some_method, ctx)  # 断言是同一对象
        # Callable returns self
        self.assertIs(ctx("arg1", key="val"), ctx)  # 断言是同一对象
        # Chaining works
        self.assertIs(ctx.foo.bar.baz(1, 2, 3), ctx)  # 断言是同一对象


# TestSpanAttributes类
class TestSpanAttributes(unittest.TestCase):

    # TestSpanAttributes类的测试constantsexist
    def test_constants_exist(self):
        self.assertEqual(SpanAttributes.GEN_AI_LATENCY_E2E, "gen_ai.latency.e2e")  # 断言相等
        self.assertIsInstance(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS, str)


# TestTraceCustomIdGenerator类
class TestTraceCustomIdGenerator(unittest.TestCase):

    # TestTraceCustomIdGenerator类的测试generatesnonzeroids
    def test_generates_nonzero_ids(self):
        gen = TraceCustomIdGenerator()
        trace_id = gen.generate_trace_id()
        span_id = gen.generate_span_id()
        self.assertIsInstance(trace_id, int)
        self.assertIsInstance(span_id, int)


# __get_host_id
class TestGetHostId(unittest.TestCase):

    # TestGetHostId类的测试frommachineidfile
    def test_from_machine_id_file(self):
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                unittest.mock.mock_open(read_data="abc123\n"),
            ),
        ):
            self.assertEqual(_get_host_id(), "abc123")  # 断言相等

    # TestGetHostId类的测试frommachineidfileerror
    def test_from_machine_id_file_error(self):
        """Falls back to MAC address when file read fails."""
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", side_effect=IOError("read error")),
        ):
            result = _get_host_id()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)  # 断言大于

    # TestGetHostId类的测试frommacaddress
    def test_from_mac_address(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("uuid.getnode", return_value=0x112233445566),
        ):
            result = _get_host_id()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)  # 断言大于

    # TestGetHostId类的测试unknownfallback
    def test_unknown_fallback(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("uuid.getnode", return_value=0),
        ):
            self.assertEqual(_get_host_id(), "unknown")  # 断言相等


@unittest.skipUnless(_has_otel, "opentelemetry not installed")

# TestGetOtlpSpanExporter类
class TestGetOtlpSpanExporter(unittest.TestCase):

    # TestGetOtlpSpanExporter类的测试grpcdefault
    def test_grpc_default(self):

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", None)
            exporter = get_otlp_span_exporter("localhost:4317")
        self.assertIsNotNone(exporter)  # 断言不为None

    # TestGetOtlpSpanExporter类的测试httpprotobuf
    def test_http_protobuf(self):

        with patch.dict(
            os.environ, {"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"}
        ):
            exporter = get_otlp_span_exporter("http://localhost:4318/v1/traces")
        self.assertIsNotNone(exporter)  # 断言不为None

    # TestGetOtlpSpanExporter类的测试invalidprotocol
    def test_invalid_protocol(self):

        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "invalid"}):
            with self.assertRaises(ValueError):  # 断言抛出异常
                get_otlp_span_exporter("localhost:4317")


# TestProcessTracingInit类
class TestProcessTracingInit(unittest.TestCase):

    # TestProcessTracingInit类的测试raiseswithoutotel
    def test_raises_without_otel(self):

        orig = mod.opentelemetry_imported
        mod.opentelemetry_imported = False
        try:
            with self.assertRaises(RuntimeError):  # 断言抛出异常
                process_tracing_init("localhost:4317", "test")
        finally:
            mod.opentelemetry_imported = orig


# TestTraceReqContextDisabled类
class TestTraceReqContextDisabled(unittest.TestCase):

    # TestTraceReqContextDisabled类的测试初始化设置
    def setUp(self):
        self.orig = mod.opentelemetry_initialized
        mod.opentelemetry_initialized = False

    # TestTraceReqContextDisabled类的测试清理
    def tearDown(self):
        mod.opentelemetry_initialized = self.orig

    # TestTraceReqContextDisabled类的测试initdisabled
    def test_init_disabled(self):
        ctx = TraceReqContext(rid="req-1")
        self.assertFalse(ctx.tracing_enable)  # 断言为假
        self.assertFalse(ctx.is_tracing_enabled())  # 断言为假

    # TestTraceReqContextDisabled类的测试allmethodsnoop
    def test_all_methods_noop(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start()
        ctx.trace_req_finish()
        ctx.trace_slice_start("s", 1)
        ctx.trace_slice_end("s", 1)
        ctx.trace_slice(TraceSliceContext("s", 100))
        ctx.trace_event("e", 1)
        ctx.trace_set_root_attrs({"k": "v"})
        ctx.trace_set_thread_attrs({"k": "v"})
        ctx.abort()
        ctx.rebuild_thread_context()

    # TestTraceReqContextDisabled类的测试getstatedisabled
    def test_getstate_disabled(self):
        ctx = TraceReqContext(rid="req-1")
        state = ctx.__getstate__()
        self.assertEqual(state, {"tracing_enable": False})  # 断言相等

    # TestTraceReqContextDisabled类的测试setstatedisabled
    def test_setstate_disabled(self):
        ctx = TraceReqContext.__new__(TraceReqContext)
        ctx.__setstate__({"tracing_enable": True, "is_copy": False})
        # opentelemetry_initialized is False → tracing forced off
        self.assertFalse(ctx.tracing_enable)  # 断言为假

    # TestTraceReqContextDisabled类的测试tracesetthreadinfodisabled
    def test_trace_set_thread_info_disabled(self):
        trace_set_thread_info("test_label")
        # Should not register anything


@unittest.skipUnless(_has_otel, "opentelemetry not installed")

# TestTraceReqContextEnabled类
class TestTraceReqContextEnabled(unittest.TestCase):

    # TestTraceReqContextEnabled类的测试初始化设置
    def setUp(self):

        self.orig_initialized = mod.opentelemetry_initialized
        self.orig_tracer = mod.tracer
        self.orig_threads = mod.threads_info.copy()
        self.orig_level = mod.global_trace_level

        self.provider = TracerProvider()
        otel_trace.set_tracer_provider(self.provider)
        mod.opentelemetry_initialized = True
        mod.tracer = otel_trace.get_tracer("test")
        mod.global_trace_level = 3

    # TestTraceReqContextEnabled类的测试清理
    def tearDown(self):
        mod.opentelemetry_initialized = self.orig_initialized
        mod.tracer = self.orig_tracer
        mod.threads_info.clear()
        mod.threads_info.update(self.orig_threads)
        mod.global_trace_level = self.orig_level

    # TestTraceReqContextEnabled类的测试tracesetthreadinfo
    def test_trace_set_thread_info(self):
        trace_set_thread_info("scheduler", tp_rank=0, dp_rank=0)

        pid = threading.get_native_id()
        self.assertIn(pid, mod.threads_info)  # 断言包含
        self.assertEqual(mod.threads_info[pid].thread_label, "scheduler")  # 断言相等

        # Second call for same thread is a no-op
        trace_set_thread_info("different_label")
        self.assertEqual(mod.threads_info[pid].thread_label, "scheduler")  # 断言相等

    # TestTraceReqContextEnabled类的测试fulllifecycle
    def test_full_lifecycle(self):
        """Start → slice_start → slice_end → finish."""
        ctx = TraceReqContext(rid="req-1", role="unified", module_name="test")
        self.assertTrue(ctx.tracing_enable)  # 断言为真

        ctx.trace_req_start(ts=1000)
        self.assertEqual(ctx.start_time_ns, 1000)  # 断言相等
        self.assertIsNotNone(ctx.root_span)  # 断言不为None
        self.assertIsNotNone(ctx.thread_context)  # 断言不为None

        ctx.trace_slice_start("prefill", level=1, ts=2000)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 1)  # 断言相等

        ctx.trace_slice_end("prefill", level=1, ts=3000)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)  # 断言相等
        self.assertIsNotNone(ctx.last_span_context)  # 断言不为None

        ctx.trace_req_finish(ts=4000, attrs={"tokens": 42})
        self.assertIsNone(ctx.root_span)  # 断言为None

    # TestTraceReqContextEnabled类的测试tracereqstartwithbootstraproom
    def test_trace_req_start_with_bootstrap_room(self):
        ctx = TraceReqContext(rid="req-1", bootstrap_room=0xFF, role="prefill")
        ctx.trace_req_start(ts=1000)
        self.assertIsNotNone(ctx.root_span)  # 断言不为None
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试tracereqfinishwithoutstart
    def test_trace_req_finish_without_start(self):
        """finish without start is a no-op."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.root_span = None
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试traceslicecombined
    def test_trace_slice_combined(self):
        """trace_slice() creates and ends a span in one call."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        s = TraceSliceContext(
            "decode",
            2000,
            end_time_ns=3000,
            level=1,
            attrs={"key": "val"},
            events=[TraceEvent("evt", 2500, {"e": 1})],
        )
        ctx.trace_slice(s)
        self.assertIsNotNone(ctx.last_span_context)  # 断言不为None
        ctx.trace_req_finish(ts=4000)

    # TestTraceReqContextEnabled类的测试traceslicewitheventscache
    def test_trace_slice_with_events_cache(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        # Add events to cache
        ctx.trace_event("schedule", level=1, ts=1500, attrs={"bid": "x"})
        self.assertEqual(len(ctx.events_cache), 1)  # 断言相等

        # trace_slice_start + trace_slice_end flushes matching events
        ctx.trace_slice_start("prefill", level=1, ts=1200)
        ctx.trace_slice_end("prefill", level=1, ts=2000)
        self.assertEqual(len(ctx.events_cache), 0)  # 断言相等

        ctx.trace_req_finish(ts=3000)

    # TestTraceReqContextEnabled类的测试traceslicecombinedwitheventscache
    def test_trace_slice_combined_with_events_cache(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_event("evt", level=1, ts=1500)
        s = TraceSliceContext("decode", 1200, end_time_ns=2000, level=1)
        ctx.trace_slice(s)
        self.assertEqual(len(ctx.events_cache), 0)  # 断言相等
        ctx.trace_req_finish(ts=3000)

    # TestTraceReqContextEnabled类的测试traceeventnoattrs
    def test_trace_event_no_attrs(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_event("evt", level=1, ts=1500, attrs=None)
        self.assertEqual(ctx.events_cache[0].attrs, {})  # 断言相等
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试tracesliceendemptystack
    def test_trace_slice_end_empty_stack(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        # End without start → warning, no crash
        ctx.trace_slice_end("missing", level=1, ts=2000)
        ctx.trace_req_finish(ts=3000)

    # TestTraceReqContextEnabled类的测试tracesliceendnamemismatch
    def test_trace_slice_end_name_mismatch(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("prefill", level=1, ts=1500)
        # Mismatched name → warning, slice popped
        ctx.trace_slice_end("wrong_name", level=1, ts=2000)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)  # 断言相等
        ctx.trace_req_finish(ts=3000)

    # TestTraceReqContextEnabled类的测试tracesliceendwithattrsandthreadfinish
    def test_trace_slice_end_with_attrs_and_thread_finish(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("dispatch", level=2, ts=1500)
        ctx.trace_slice_end(
            "dispatch",
            level=2,
            ts=2000,
            attrs={"key": "val"},
            thread_finish_flag=True,
        )
        # thread_finish_flag triggers abort → thread_context is None
        self.assertIsNone(ctx.thread_context)  # 断言为None

    # TestTraceReqContextEnabled类的测试traceslicecombinedwiththreadfinish
    def test_trace_slice_combined_with_thread_finish(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        s = TraceSliceContext("dispatch", 1500, end_time_ns=2000, level=2)
        ctx.trace_slice(s, thread_finish_flag=True)
        self.assertIsNone(ctx.thread_context)  # 断言为None

    # TestTraceReqContextEnabled类的测试nestedslices
    def test_nested_slices(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("outer", level=1, ts=1500)
        ctx.trace_slice_start("inner", level=2, ts=1600)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 2)  # 断言相等
        ctx.trace_slice_end("inner", level=2, ts=1800)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 1)  # 断言相等
        ctx.trace_slice_end("outer", level=1, ts=2000)
        ctx.trace_req_finish(ts=3000)

    # TestTraceReqContextEnabled类的测试nestedslicewithlastspancontext
    def test_nested_slice_with_last_span_context(self):
        """trace_slice uses last_span_context when slice stack is empty."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        # First slice sets last_span_context
        ctx.trace_slice_start("s1", level=1, ts=1500)
        ctx.trace_slice_end("s1", level=1, ts=2000)
        self.assertIsNotNone(ctx.last_span_context)  # 断言不为None

        # Second slice uses last_span_context as link
        ctx.trace_slice_start("s2", level=1, ts=2500)
        ctx.trace_slice_end("s2", level=1, ts=3000)

        # trace_slice also uses last_span_context
        s = TraceSliceContext("s3", 3500, end_time_ns=4000, level=1)
        ctx.trace_slice(s)

        ctx.trace_req_finish(ts=5000)

    # TestTraceReqContextEnabled类的测试tracesetrootattrs
    def test_trace_set_root_attrs(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_set_root_attrs({"model": "llama"})
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试tracesetrootattrsnospan
    def test_trace_set_root_attrs_no_span(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.root_span = None
        ctx.trace_set_root_attrs({"model": "llama"})  # no crash

    # TestTraceReqContextEnabled类的测试tracesetthreadattrs
    def test_trace_set_thread_attrs(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_set_thread_attrs({"batch_size": 32})
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试abortwithunclosedslices
    def test_abort_with_unclosed_slices(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("s1", level=1, ts=1500)
        ctx.trace_slice_start("s2", level=2, ts=1600)
        ctx.abort(ts=2000)
        self.assertIsNone(ctx.thread_context)  # 断言为None

    # TestTraceReqContextEnabled类的测试abortwitheventscache
    def test_abort_with_events_cache(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_event("evt", level=1, ts=1500)
        ctx.abort(ts=2000)
        self.assertEqual(len(ctx.events_cache), 0)  # 断言相等

    # TestTraceReqContextEnabled类的测试abortwithabortinfodict
    def test_abort_with_abort_info_dict(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.abort(ts=2000, abort_info={"reason": "cancelled"})
        self.assertIsNone(ctx.thread_context)  # 断言为None

    # TestTraceReqContextEnabled类的测试abortwithbasefinishreason
    def test_abort_with_base_finish_reason(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        abort_obj = FINISH_LENGTH(length=10)
        ctx.abort(ts=2000, abort_info=abort_obj)
        self.assertIsNone(ctx.thread_context)  # 断言为None

    # TestTraceReqContextEnabled类的测试checkfastreturnbylevel
    def test_check_fast_return_by_level(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_level = 1  # instance-level, set at init from global
        # Level 2 > trace_level 1 → fast return
        ctx.trace_slice_start("s", level=2, ts=1500)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)  # 断言相等
        ctx.trace_level = 3
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试rebuildthreadcontext
    def test_rebuild_thread_context(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        old_tc = ctx.thread_context
        ctx.rebuild_thread_context(ts=1500)
        self.assertIsNot(ctx.thread_context, old_tc)  # 断言不是同一对象
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试getstateenabled
    def test_getstate_enabled(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        state = ctx.__getstate__()
        self.assertTrue(state["tracing_enable"])  # 断言为真
        self.assertEqual(state["rid"], "req-1")  # 断言相等
        self.assertIn("root_span_context", state)  # 断言包含
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试getstatenorootcontext
    def test_getstate_no_root_context(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.root_span_context = None
        state = ctx.__getstate__()
        self.assertFalse(state["tracing_enable"])  # 断言为假
        ctx.root_span_context = True  # prevent __del__ issues
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试getstatewithslicestack
    def test_getstate_with_slice_stack(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("s1", level=1, ts=1500)
        state = ctx.__getstate__()
        self.assertIn("last_span_context", state)  # 断言包含
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试setstateenabled
    def test_setstate_enabled(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        state = ctx.__getstate__()
        ctx.trace_req_finish(ts=2000)

        ctx2 = TraceReqContext.__new__(TraceReqContext)
        ctx2.__setstate__(state)
        self.assertTrue(ctx2.tracing_enable)  # 断言为真
        self.assertTrue(ctx2.is_copy)  # 断言为真
        self.assertIsNotNone(ctx2.root_span_context)  # 断言不为None

    # TestTraceReqContextEnabled类的测试threadcontextwithtprank
    def test_thread_context_with_tp_rank(self):
        """Covers tp_rank branch in __create_thread_context."""

        pid = threading.get_native_id()
        mod.threads_info[pid] = TraceThreadInfo(
            "host", pid, "sched", tp_rank=0, dp_rank=0, pp_rank=0
        )
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        self.assertIsNotNone(ctx.thread_context)  # 断言不为None
        ctx.trace_req_finish(ts=2000)

    # TestTraceReqContextEnabled类的测试setstatewithlastspancontext
    def test_setstate_with_last_span_context(self):
        """Covers __setstate__ path where last_span_context is truthy."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("s1", level=1, ts=1500)
        ctx.trace_slice_end("s1", level=1, ts=2000)
        state = ctx.__getstate__()
        ctx.trace_req_finish(ts=3000)

        self.assertIsNotNone(state.get("last_span_context"))  # 断言不为None
        ctx2 = TraceReqContext.__new__(TraceReqContext)
        ctx2.__setstate__(state)
        self.assertIsNotNone(ctx2.last_span_context)  # 断言不为None

    # TestTraceReqContextEnabled类的测试eventscachepartialmatch
    def test_events_cache_partial_match(self):
        """Events outside the slice time range stay in cache."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_event("early", level=1, ts=500)
        ctx.trace_event("inside", level=1, ts=1500)
        ctx.trace_event("late", level=1, ts=5000)

        ctx.trace_slice_start("s", level=1, ts=1200)
        ctx.trace_slice_end("s", level=1, ts=2000)
        # "early" (500 < 1200) and "late" (5000 >= 2000) stay in cache
        self.assertEqual(len(ctx.events_cache), 2)  # 断言相等
        ctx.trace_req_finish(ts=6000)

    # TestTraceReqContextEnabled类的测试traceslicecombinedeventspartialmatch
    def test_trace_slice_combined_events_partial_match(self):
        """Events outside slice range stay in cache for trace_slice method."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_event("early", level=1, ts=500)
        ctx.trace_event("inside", level=1, ts=1500)

        s = TraceSliceContext("s", 1200, end_time_ns=2000, level=1)
        ctx.trace_slice(s)
        self.assertEqual(len(ctx.events_cache), 1)  # "early" stays  # 断言相等
        ctx.trace_req_finish(ts=3000)

    # TestTraceReqContextEnabled类的测试traceslicenestedparent
    def test_trace_slice_nested_parent(self):
        """trace_slice with parent from slice stack (not thread_span)."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_slice_start("outer", level=1, ts=1500)
        s = TraceSliceContext("inner", 1600, end_time_ns=1800, level=2)
        ctx.trace_slice(s)
        ctx.trace_slice_end("outer", level=1, ts=2000)
        ctx.trace_req_finish(ts=3000)

    # TestTraceReqContextEnabled类的测试deltriggersabort
    def test_del_triggers_abort(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        # __del__ calls abort
        ctx.__del__()
        self.assertIsNone(ctx.thread_context)  # 断言为None


if __name__ == "__main__":
    unittest.main()
