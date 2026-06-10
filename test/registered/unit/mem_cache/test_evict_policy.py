# 文件名: test_evict_policy.py - 驱逐策略
"""Unit tests for evict_policy.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import (
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)


# 内部方法_make_node
def _make_node(**kwargs):
    node = MagicMock()
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.hit_count = kwargs.get("hit_count", 0)
    node.creation_time = kwargs.get("creation_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    return node


# TestLRUStrategy类
class TestLRUStrategy(unittest.TestCase):

    # TestLRUStrategy类的测试初始化设置
    def setUp(self):
        self.strategy = LRUStrategy()

    # TestLRUStrategy类的测试priorityislastaccesstime
    def test_priority_is_last_access_time(self):
        node = _make_node(last_access_time=42.0)
        self.assertEqual(self.strategy.get_priority(node), 42.0)  # 断言相等

    # TestLRUStrategy类的测试olderaccessevictedfirst
    def test_older_access_evicted_first(self):
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


# TestLFUStrategy类
class TestLFUStrategy(unittest.TestCase):

    # TestLFUStrategy类的测试初始化设置
    def setUp(self):
        self.strategy = LFUStrategy()

    # TestLFUStrategy类的测试priorityishitcountandtime
    def test_priority_is_hit_count_and_time(self):
        node = _make_node(hit_count=5, last_access_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), (5, 3.0))  # 断言相等

    # TestLFUStrategy类的测试lowerhitcountevictedfirst
    def test_lower_hit_count_evicted_first(self):
        cold = _make_node(hit_count=1, last_access_time=10.0)
        hot = _make_node(hit_count=100, last_access_time=1.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )

    # TestLFUStrategy类的测试samehitcountolderaccessevictedfirst
    def test_same_hit_count_older_access_evicted_first(self):
        old = _make_node(hit_count=3, last_access_time=1.0)
        new = _make_node(hit_count=3, last_access_time=10.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


# TestFIFOStrategy类
class TestFIFOStrategy(unittest.TestCase):

    # TestFIFOStrategy类的测试初始化设置
    def setUp(self):
        self.strategy = FIFOStrategy()

    # TestFIFOStrategy类的测试priorityiscreationtime
    def test_priority_is_creation_time(self):
        node = _make_node(creation_time=7.0)
        self.assertEqual(self.strategy.get_priority(node), 7.0)  # 断言相等

    # TestFIFOStrategy类的测试earliercreatedevictedfirst
    def test_earlier_created_evicted_first(self):
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(first), self.strategy.get_priority(second)
        )


# TestMRUStrategy类
class TestMRUStrategy(unittest.TestCase):

    # TestMRUStrategy类的测试初始化设置
    def setUp(self):
        self.strategy = MRUStrategy()

    # TestMRUStrategy类的测试priorityisnegatedaccesstime
    def test_priority_is_negated_access_time(self):
        node = _make_node(last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), -5.0)  # 断言相等

    # TestMRUStrategy类的测试mostrecentlyusedevictedfirst
    def test_most_recently_used_evicted_first(self):
        """MRU evicts the most recently accessed node first (lowest priority value)."""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(new), self.strategy.get_priority(old)
        )


# TestFILOStrategy类
class TestFILOStrategy(unittest.TestCase):

    # TestFILOStrategy类的测试初始化设置
    def setUp(self):
        self.strategy = FILOStrategy()

    # TestFILOStrategy类的测试priorityisnegatedcreationtime
    def test_priority_is_negated_creation_time(self):
        node = _make_node(creation_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), -3.0)  # 断言相等

    # TestFILOStrategy类的测试lastcreatedevictedfirst
    def test_last_created_evicted_first(self):
        """FILO evicts the most recently created node first."""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(second), self.strategy.get_priority(first)
        )


# TestPriorityStrategy类
class TestPriorityStrategy(unittest.TestCase):

    # TestPriorityStrategy类的测试初始化设置
    def setUp(self):
        self.strategy = PriorityStrategy()

    # TestPriorityStrategy类的测试priorityistuple
    def test_priority_is_tuple(self):
        node = _make_node(priority=2, last_access_time=4.0)
        self.assertEqual(self.strategy.get_priority(node), (2, 4.0))  # 断言相等

    # TestPriorityStrategy类的测试lowerpriorityevictedfirst
    def test_lower_priority_evicted_first(self):
        low = _make_node(priority=1, last_access_time=10.0)
        high = _make_node(priority=5, last_access_time=1.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(low), self.strategy.get_priority(high)
        )

    # TestPriorityStrategy类的测试samepriorityolderaccessevictedfirst
    def test_same_priority_older_access_evicted_first(self):
        old = _make_node(priority=3, last_access_time=1.0)
        new = _make_node(priority=3, last_access_time=10.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


# TestSLRUStrategy类
class TestSLRUStrategy(unittest.TestCase):

    # TestSLRUStrategy类的测试初始化设置
    def setUp(self):
        self.strategy = SLRUStrategy(protected_threshold=2)

    # TestSLRUStrategy类的测试probationarysegment
    def test_probationary_segment(self):
        node = _make_node(hit_count=1, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))  # 断言相等

    # TestSLRUStrategy类的测试protectedsegment
    def test_protected_segment(self):
        node = _make_node(hit_count=2, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))  # 断言相等

    # TestSLRUStrategy类的测试highlyaccessedisprotected
    def test_highly_accessed_is_protected(self):
        node = _make_node(hit_count=100, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))  # 断言相等

    # TestSLRUStrategy类的测试probationaryevictedbeforeprotected
    def test_probationary_evicted_before_protected(self):
        prob = _make_node(hit_count=1, last_access_time=10.0)
        prot = _make_node(hit_count=5, last_access_time=1.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(prob), self.strategy.get_priority(prot)
        )

    # TestSLRUStrategy类的测试samesegmentolderaccessevictedfirst
    def test_same_segment_older_access_evicted_first(self):
        old = _make_node(hit_count=0, last_access_time=1.0)
        new = _make_node(hit_count=0, last_access_time=10.0)
        self.assertLess(  # 断言小于
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    # TestSLRUStrategy类的测试customthreshold
    def test_custom_threshold(self):
        strategy = SLRUStrategy(protected_threshold=5)
        below = _make_node(hit_count=4, last_access_time=1.0)
        at = _make_node(hit_count=5, last_access_time=1.0)
        self.assertEqual(strategy.get_priority(below), (0, 1.0))  # 断言相等
        self.assertEqual(strategy.get_priority(at), (1, 1.0))  # 断言相等

    # TestSLRUStrategy类的测试defaultthresholdis2
    def test_default_threshold_is_2(self):
        default = SLRUStrategy()
        self.assertEqual(default.protected_threshold, 2)  # 断言相等


# TestEvictionOrdering类
class TestEvictionOrdering(unittest.TestCase):
    """Integration-style test: sort a list of nodes by eviction priority."""

    def test_lru_ordering(self):
        strategy = LRUStrategy()
        nodes = [
            _make_node(last_access_time=5.0),
            _make_node(last_access_time=1.0),
            _make_node(last_access_time=3.0),
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        times = [n.last_access_time for n in eviction_order]
        self.assertEqual(times, [1.0, 3.0, 5.0])  # 断言相等

    # TestEvictionOrdering类的测试slruordering
    def test_slru_ordering(self):
        strategy = SLRUStrategy(protected_threshold=2)
        nodes = [
            _make_node(hit_count=5, last_access_time=1.0),  # protected, old
            _make_node(hit_count=0, last_access_time=10.0),  # probationary, new
            _make_node(hit_count=0, last_access_time=2.0),  # probationary, old
            _make_node(hit_count=3, last_access_time=8.0),  # protected, new
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected = [
            (0, 2.0),  # probationary old
            (0, 10.0),  # probationary new
            (1, 1.0),  # protected old
            (1, 8.0),  # protected new
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected)  # 断言相等


if __name__ == "__main__":
    unittest.main()
