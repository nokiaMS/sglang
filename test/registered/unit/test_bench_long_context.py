# 文件名: test_bench_long_context.py - 长上下文基准
"""Unit test for benchmark/hicache/bench_long_context.py.

Guards against the regression where ContextWorkloadGenerator.__init__ replaces
WorkloadGenerator.__init__ entirely but forgets to set attributes the inherited
request_sender/handle_request methods need (e.g. self.request_func).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
HICACHE_DIR = REPO_ROOT / "benchmark" / "hicache"
if str(HICACHE_DIR) not in sys.path:
    sys.path.insert(0, str(HICACHE_DIR))

import bench_long_context  # noqa: E402

from sglang.test.kits.cache_hit_kit import async_request_sglang_generate  # noqa: E402


# 内部方法_build_args
def _build_args(dataset_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        host="localhost",
        port=30000,
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        distribution="poisson",
        request_rate=1.0,
        dataset_path=dataset_path,
        num_clients=2,
        max_parallel=2,
        log_file="performance_metrics.jsonl",
        tag="",
    )


# 内部方法_fake_dataset
def _fake_dataset() -> dict:
    return {
        "contexts": ["ctx-zero ", "ctx-one "],
        "queries": [
            {"context": 0, "question": "q0", "reference_answer": "a0"},
            {"context": 1, "question": "q1", "reference_answer": "a1"},
        ],
    }


# TestContextWorkloadGeneratorInit类
class TestContextWorkloadGeneratorInit(CustomTestCase):
    """Verify ContextWorkloadGenerator wires up everything its inherited
    request_sender/handle_request/run methods rely on."""

    # TestContextWorkloadGeneratorInit类的测试初始化设置
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(_fake_dataset(), self._tmp)
        self._tmp.close()
        self.dataset_path = self._tmp.name

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer.return_value = {"input_ids": [5, 6]}

        self._tok_patch = patch.object(
            bench_long_context, "get_tokenizer", return_value=mock_tokenizer
        )
        self._tok_patch.start()

    # TestContextWorkloadGeneratorInit类的测试清理
    def tearDown(self):
        self._tok_patch.stop()
        Path(self.dataset_path).unlink(missing_ok=True)

    # TestContextWorkloadGeneratorInit类的测试requestfuncisset
    def test_request_func_is_set(self):
        """The bug we're guarding against: request_func not being set caused
        AttributeError as soon as the request_sender thread fired."""
        gen = bench_long_context.ContextWorkloadGenerator(
            _build_args(self.dataset_path)
        )
        self.assertTrue(callable(getattr(gen, "request_func", None)))  # 断言为真
        self.assertIs(gen.request_func, async_request_sglang_generate)  # 断言是同一对象

    # TestContextWorkloadGeneratorInit类的测试inheritsworkloadgeneratorcontract
    def test_inherits_workload_generator_contract(self):
        """All attributes WorkloadGenerator's run-time methods touch must exist."""
        gen = bench_long_context.ContextWorkloadGenerator(
            _build_args(self.dataset_path)
        )

        # handle_request (bench_multiturn.py) reads these
        for attr in ("request_func", "url", "pbar", "response_queue", "finished_time"):
            self.assertTrue(hasattr(gen, attr), f"missing attribute: {attr}")  # 断言为真

        # request_sender reads these
        for attr in (
            "sent_requests",
            "completed_requests",
            "max_parallel",
            "ready_queue",
            "distribution",
            "request_rate",
        ):
            self.assertTrue(hasattr(gen, attr), f"missing attribute: {attr}")  # 断言为真

        # run() reads these
        for attr in ("performance_metrics", "enable_round_barrier"):
            self.assertTrue(hasattr(gen, attr), f"missing attribute: {attr}")  # 断言为真

    # TestContextWorkloadGeneratorInit类的测试urltargetssglanggenerateendpoint
    def test_url_targets_sglang_generate_endpoint(self):
        gen = bench_long_context.ContextWorkloadGenerator(
            _build_args(self.dataset_path)
        )
        self.assertEqual(gen.url, "http://localhost:30000/generate")  # 断言相等

    # TestContextWorkloadGeneratorInit类的测试readyqueuesizematchesdataset
    def test_ready_queue_size_matches_dataset(self):
        gen = bench_long_context.ContextWorkloadGenerator(
            _build_args(self.dataset_path)
        )
        # 2 queries in fake dataset, num_clients=2 → 2 init requests
        self.assertEqual(len(gen.ready_queue.requests), 2)  # 断言相等


if __name__ == "__main__":
    unittest.main()
