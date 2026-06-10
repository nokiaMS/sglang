# 文件名: test_mla_tp.py - MLA张量并行测试 - 验证MLA模型在TP模式下的推理功能
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDeepseekTP2(CustomTestCase):
    @classmethod
    # setUpClass
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:
            other_args.extend(
                ["--tp", "2", "--enable-torch-compile", "--cuda-graph-max-bs", "2"]
            )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    # tearDownClass
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # 测试gsm8k
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
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.62)

    # 测试gsm8k bs1
    def test_gsm8k_bs1(self):
        # test torch compile accuracy for bs=1
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=10,
            num_threads=1,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.62)


if __name__ == "__main__":
    unittest.main()
