# 文件名: test_disaggregation_wire.py - 解聚通信线协议
import unittest

import numpy as np

from sglang.srt.disaggregation.common.utils import (
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# TestDisaggregationWire类
class TestDisaggregationWire(unittest.TestCase):

    # TestDisaggregationWire类的测试intlistsroundtrip
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)  # 断言相等

    # TestDisaggregationWire类的测试packacceptsndarray
    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])  # 断言相等

    # TestDisaggregationWire类的测试emptyouterlist
    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")  # 断言相等
        self.assertEqual(unpack_int_lists(b"", "Q"), [])  # 断言相等

    # TestDisaggregationWire类的测试emptyinnerlist
    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])  # 断言相等

    # TestDisaggregationWire类的测试listofbuffersroundtrip
    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)  # 断言相等


if __name__ == "__main__":
    unittest.main()
