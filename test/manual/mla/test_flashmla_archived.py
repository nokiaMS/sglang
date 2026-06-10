# 文件名: test_flashmla_archived.py - 归档：FlashMLA注意力后端与MTP推测解码测试
"""Archived test classes split out of test/registered/mla/test_flashmla.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/mla/test_flashmla_archived.py`.
"""

"""
Usage:
python3 test/registered/mla/test_flashmla.py
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


# FlashMLA attention backend tests with MTP speculative decoding
class TestFlashMLAAttnBackend(unittest.TestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:  # 检查CUDA可用性
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "2",
                    "--attention-backend",
                    "flashmla",
                ]
            )
        # Use longer timeout for DeepGEMM JIT compilation which can take 10-20 minutes
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
            other_args=other_args,
        )

    @classmethod
    # 类级别清理，关闭服务器或清理资源
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)  # 终止服务器进程

    # 测试gsm8k功能
    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)  # 运行评估
        print(metrics)

        self.assertGreater(metrics["score"], 0.60)  # 断言精度大于阈值


if __name__ == "__main__":
    unittest.main()
