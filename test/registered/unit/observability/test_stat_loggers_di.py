# 文件名: test_stat_loggers_di.py - 统计日志器依赖注入
"""Unit tests for class-level DI on the five *MetricsCollector classes via
ServerArgs.stat_loggers — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import prometheus_client

from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_EXPERT_DISPATCH,
    STAT_LOGGER_ROLE_RADIX_CACHE,
    STAT_LOGGER_ROLE_SCHEDULER,
    STAT_LOGGER_ROLE_STORAGE,
    STAT_LOGGER_ROLE_TOKENIZER,
    ExpertDispatchCollector,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    resolve_collector_class,
)


# _StubArgs类
class _StubArgs:
    """Minimal ServerArgs stand-in. Avoids triggering heavy ServerArgs import chain."""

    def __init__(self, stat_loggers=None):
        self.stat_loggers = stat_loggers


# ── _gauge_cls / _counter_cls / _histogram_cls / _summary_cls override surface ──


class TestCollectorClassAttrs(unittest.TestCase):
    """All five collectors expose four DI hook class attrs, all defaulting to None
    so the existing prometheus_client backend is used unchanged."""

    # TestCollectorClassAttrs类的测试schedulercollectorattrsdefaultnone
    def test_scheduler_collector_attrs_default_none(self):
        self.assertIsNone(SchedulerMetricsCollector._counter_cls)  # 断言为None
        self.assertIsNone(SchedulerMetricsCollector._gauge_cls)  # 断言为None
        self.assertIsNone(SchedulerMetricsCollector._histogram_cls)  # 断言为None
        self.assertIsNone(SchedulerMetricsCollector._summary_cls)  # 断言为None

    # TestCollectorClassAttrs类的测试tokenizercollectorattrsdefaultnone
    def test_tokenizer_collector_attrs_default_none(self):
        self.assertIsNone(TokenizerMetricsCollector._counter_cls)  # 断言为None
        self.assertIsNone(TokenizerMetricsCollector._histogram_cls)  # 断言为None

    # TestCollectorClassAttrs类的测试storagecollectorattrsdefaultnone
    def test_storage_collector_attrs_default_none(self):
        self.assertIsNone(StorageMetricsCollector._counter_cls)  # 断言为None
        self.assertIsNone(StorageMetricsCollector._histogram_cls)  # 断言为None

    # TestCollectorClassAttrs类的测试expertdispatchcollectorattrsdefaultnone
    def test_expert_dispatch_collector_attrs_default_none(self):
        self.assertIsNone(ExpertDispatchCollector._histogram_cls)  # 断言为None

    # TestCollectorClassAttrs类的测试radixcachecollectorattrsdefaultnone
    def test_radix_cache_collector_attrs_default_none(self):
        self.assertIsNone(RadixCacheMetricsCollector._counter_cls)  # 断言为None
        self.assertIsNone(RadixCacheMetricsCollector._histogram_cls)  # 断言为None


# ── resolve_collector_class helper ──


class TestResolveCollectorClass(unittest.TestCase):

    # TestResolveCollectorClass类的测试returnsdefaultwhenserverargsnone
    def test_returns_default_when_server_args_none(self):
        cls = resolve_collector_class(None, "scheduler", SchedulerMetricsCollector)
        self.assertIs(cls, SchedulerMetricsCollector)  # 断言是同一对象

    # TestResolveCollectorClass类的测试returnsdefaultwhenstatloggersnone
    def test_returns_default_when_stat_loggers_none(self):
        cls = resolve_collector_class(
            _StubArgs(stat_loggers=None), "scheduler", SchedulerMetricsCollector
        )
        self.assertIs(cls, SchedulerMetricsCollector)  # 断言是同一对象

    # TestResolveCollectorClass类的测试returnsdefaultwhenstatloggersempty
    def test_returns_default_when_stat_loggers_empty(self):
        cls = resolve_collector_class(
            _StubArgs(stat_loggers={}), "scheduler", SchedulerMetricsCollector
        )
        self.assertIs(cls, SchedulerMetricsCollector)  # 断言是同一对象

    # TestResolveCollectorClass类的测试returnsdefaultwhenrolemissing
    def test_returns_default_when_role_missing(self):
        # Different role registered. Default still wins for "scheduler".
        class MyTokenizer(TokenizerMetricsCollector):
            pass

        cls = resolve_collector_class(
            _StubArgs(stat_loggers={"tokenizer": MyTokenizer}),
            "scheduler",
            SchedulerMetricsCollector,
        )
        self.assertIs(cls, SchedulerMetricsCollector)  # 断言是同一对象

    # TestResolveCollectorClass类的测试returnssubclasswhenroleregistered
    def test_returns_subclass_when_role_registered(self):

        # MyScheduler类
        class MyScheduler(SchedulerMetricsCollector):
            pass

        cls = resolve_collector_class(
            _StubArgs(stat_loggers={"scheduler": MyScheduler}),
            "scheduler",
            SchedulerMetricsCollector,
        )
        self.assertIs(cls, MyScheduler)  # 断言是同一对象

    # TestResolveCollectorClass类的测试roleconstantsmatchcollectorkeys
    def test_role_constants_match_collector_keys(self):
        """The exported role constants must be the exact strings the
        instantiation sites use to look up subclasses."""
        self.assertEqual(STAT_LOGGER_ROLE_SCHEDULER, "scheduler")  # 断言相等
        self.assertEqual(STAT_LOGGER_ROLE_TOKENIZER, "tokenizer")  # 断言相等
        self.assertEqual(STAT_LOGGER_ROLE_STORAGE, "storage")  # 断言相等
        self.assertEqual(STAT_LOGGER_ROLE_RADIX_CACHE, "radix_cache")  # 断言相等
        self.assertEqual(STAT_LOGGER_ROLE_EXPERT_DISPATCH, "expert_dispatch")  # 断言相等


# ── DI swap behavior — actually instantiate with a custom backend ──


class _RecordingGauge:
    """Test double that mirrors prometheus_client.Gauge constructor signature.
    Records every instantiation so the test can assert the override took effect."""

    instances = []

    # _RecordingGauge类的初始化
    def __init__(self, *args, **kwargs):
        type(self).instances.append((args, kwargs))

    # _RecordingGauge类的labels
    def labels(self, **kwargs):
        return self

    # _RecordingGauge类的set
    def set(self, value):
        pass

    # _RecordingGauge类的inc
    def inc(self, amount=1):
        pass


# _RecordingCounter类
class _RecordingCounter(_RecordingGauge):
    pass


# _RecordingHistogram类
class _RecordingHistogram(_RecordingGauge):

    # _RecordingHistogram类的observe
    def observe(self, value):
        pass


# _RecordingSummary类
class _RecordingSummary(_RecordingGauge):

    # _RecordingSummary类的observe
    def observe(self, value):
        pass


# TestDISwap类
class TestDISwap(unittest.TestCase):
    """Subclasses that set the DI hooks at class level cause the collector to
    instantiate the test doubles instead of prometheus_client classes."""

    # TestDISwap类的测试初始化设置
    def setUp(self):
        _RecordingGauge.instances = []
        _RecordingCounter.instances = []
        _RecordingHistogram.instances = []
        _RecordingSummary.instances = []

    # TestDISwap类的测试radixcachediswap
    def test_radix_cache_di_swap(self):
        """Smallest collector (4 metrics, Counter + Histogram) — verifies the
        DI shim flows through both class types."""

        # RaySwapRadixCache类
        class RaySwapRadixCache(RadixCacheMetricsCollector):
            _counter_cls = _RecordingCounter
            _histogram_cls = _RecordingHistogram

        labels = {"cache_type": "test"}
        RaySwapRadixCache(labels=labels)

        # 4 instruments total in RadixCacheMetricsCollector:
        # eviction_duration_seconds (H), eviction_num_tokens (C),
        # load_back_duration_seconds (H), load_back_num_tokens (C).
        self.assertEqual(len(_RecordingCounter.instances), 2)  # 断言相等
        self.assertEqual(len(_RecordingHistogram.instances), 2)  # 断言相等

    # TestDISwap类的测试expertdispatchdiswap
    def test_expert_dispatch_di_swap(self):
        """Smallest collector (1 Histogram metric)."""

        class RaySwapExpert(ExpertDispatchCollector):
            _histogram_cls = _RecordingHistogram

        RaySwapExpert(ep_size=4)
        self.assertEqual(len(_RecordingHistogram.instances), 1)  # 断言相等

    # TestDISwap类的测试defaultpathusesprometheusclient
    def test_default_path_uses_prometheus_client(self):
        """Without any subclass override, the collector instantiates the real
        prometheus_client classes — the existing behavior is unchanged."""
        labels = {"cache_type": "test_default"}
        collector = RadixCacheMetricsCollector(labels=labels)
        # The instruments must be real prometheus_client objects, not test doubles.
        self.assertIsInstance(collector.eviction_num_tokens, prometheus_client.Counter)
        self.assertIsInstance(
            collector.eviction_duration_seconds, prometheus_client.Histogram
        )


if __name__ == "__main__":
    unittest.main()
