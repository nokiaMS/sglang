# 文件名: test_piecewise_cuda_graph_support_1_gpu_archived.py - 归档：单GPU分段CUDA图支持测试（InternVL2.5）
"""Archived test classes split out of test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu_archived.py`.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)


# CI Registration
class TestPiecewiseCudaGraphInternVL25(CustomTestCase):
    """Test piecewise CUDA graph with InternVL2.5-8B model"""

    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "OpenGVLab/InternVL2_5-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enforce-piecewise-cuda-graph",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    # 类级别清理，关闭服务器或清理资源
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)  # 终止服务器进程

    # 测试gsm8k accuracy功能
    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)  # 运行评估
        print(f"GSM8K Accuracy: {metrics['score']:.3f}")

        # Baseline (no piecewise CUDA graph): 0.571 — this eval uses 5-shot
        # concatenated text via chat API, which scores lower than reported
        # benchmarks (~77.8%) that use proper CoT chat format. The threshold
        # is set 5% below observed to catch catastrophic regressions.
        self.assertGreaterEqual(metrics["score"], 0.54)  # 断言精度大于等于阈值


if __name__ == "__main__":
    unittest.main()
