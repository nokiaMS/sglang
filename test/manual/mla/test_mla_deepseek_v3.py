# 文件名: test_mla_deepseek_v3.py - 测试DeepSeek-V3 MLA注意力（含FA3 FP8 KV缓存与MTP推测解码）
import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import is_cuda, is_hip, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


class TestMLADeepseekV3(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code", "--chunked-prefill-size", "256"]
        if is_cuda():
            other_args.extend(["--enable-torch-compile", "--cuda-graph-max-bs", "2"])
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

        self.assertGreater(metrics["score"], 0.60)  # 断言精度大于阈值


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestMLADeepseekV3DisableFusedFunc(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        os.environ["SGLANG_CI_DISABLE_MOE_FUSED_FUNC"] = "1"  # 设置环境变量
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code", "--chunked-prefill-size", "256"]
        if is_cuda():
            other_args.extend(["--cuda-graph-max-bs", "2"])
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


@unittest.skipIf(is_hip(), "FA is not available.")
class TestMLADeepseekV3Fa3Fp8Kvcache(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--chunked-prefill-size",
            "256",
            "--kv-cache-dtype",
            "fp8_e4m3",
        ]
        if is_cuda():
            other_args.extend(
                [
                    "--attention-backend",
                    "fa3",
                    "--mem-fraction-static",
                    "0.8",
                    "--cuda-graph-max-bs",
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

        self.assertGreater(metrics["score"], 0.60)  # 断言精度大于阈值


class TestDeepseekV3MTP(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "lmsys/sglang-ci-dsv3-test"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--cuda-graph-max-bs",
            "2",
            "--disable-radix",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "1",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "4",
        ]
        # This test runs first (alphabetically) and needs longer timeout for
        # DeepGEMM JIT compilation which is required for DeepSeek-V3's FP8 MoE layers
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
        requests.get(self.base_url + "/flush_cache")  # 发送GET请求

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

        server_info = requests.get(self.base_url + "/server_info")  # 发送GET请求
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.5)  # 断言精度大于阈值


if __name__ == "__main__":
    unittest.main()
