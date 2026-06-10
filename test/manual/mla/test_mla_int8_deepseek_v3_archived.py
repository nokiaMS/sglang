# 文件名: test_mla_int8_deepseek_v3_archived.py - 归档：DeepSeek-V3 INT8量化（通道/块级）MLA测试
"""Archived test classes split out of test/registered/mla/test_mla_int8_deepseek_v3.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/mla/test_mla_int8_deepseek_v3_archived.py`.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


# DeepSeek-V3 INT8 quantization tests (channel and block INT8)
class TestMLADeepseekV3ChannelInt8(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-channel-int8-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:  # 检查CUDA可用性
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "16",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "2",
                ]
            )
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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

        self.assertGreaterEqual(metrics["score"], 0.61)  # 断言精度大于等于阈值


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestMLADeepseekV3BlockInt8(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-block-int8-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code"]
        if torch.cuda.is_available() and torch.version.cuda:  # 检查CUDA可用性
            other_args.extend(
                [
                    "--cuda-graph-max-bs",
                    "16",
                    "--enable-torch-compile",
                    "--torch-compile-max-bs",
                    "2",
                ]
            )
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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

        self.assertGreater(metrics["score"], 0.62)  # 断言精度大于阈值


if __name__ == "__main__":
    unittest.main()
