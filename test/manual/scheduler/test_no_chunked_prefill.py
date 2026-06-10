# 文件名: test_no_chunked_prefill.py - 无分块预填充测试 - 验证禁用chunked prefill模式下的MMLU准确性和服务吞吐量
import unittest

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
    run_mmlu_test,
)


class TestNoChunkedPrefill(CustomTestCase):

    # 测试no chunked prefill
    def test_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, enable_mixed_chunk=False, chunked_prefill_size=-1
        )

    # 测试no chunked prefill without radix cache
    def test_no_chunked_prefill_without_radix_cache(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache", "--chunked-prefill-size", "-1"],
        )

        assert res["completed"] == 10


if __name__ == "__main__":
    unittest.main()
