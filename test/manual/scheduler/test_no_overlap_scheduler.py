# 文件名: test_no_overlap_scheduler.py - 无重叠调度器测试 - 验证禁用重叠调度时radix cache和chunked prefill的组合效果
"""
Usage:
python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
python3 test_overlap_schedule.py
"""

import unittest

from sglang.test.test_utils import CustomTestCase, run_mmlu_test


class TestOverlapSchedule(CustomTestCase):
    # 测试no radix attention chunked prefill
    def test_no_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=32, disable_overlap=True
        )

    # 测试no radix attention no chunked prefill
    def test_no_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    # 测试radix attention chunked prefill
    def test_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=32, disable_overlap=True
        )

    # 测试radix attention no chunked prefill
    def test_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
