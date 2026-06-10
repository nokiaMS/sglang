# 文件名: test_unsloth_models.py - 测试Unsloth Phi-4系列模型（含BNB 4-bit量化）
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestUnslothPhi4(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "unsloth/phi-4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
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
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.78)  # 断言精度大于阈值


class TestUnslothPhi4Bnb4bit(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "unsloth/phi-4-bnb-4bit"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-format",
                "bitsandbytes",
            ],
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
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.75)  # 断言精度大于阈值


class TestUnslothPhi4UnslothBnb4bit(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "unsloth/phi-4-unsloth-bnb-4bit"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-format",
                "bitsandbytes",
            ],
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
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.75)  # 断言精度大于阈值


class TestUnslothPhi4MiniInstruct(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "unsloth/Phi-4-mini-instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
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
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.65)  # 断言精度大于阈值


class TestUnslothPhi4MiniBnb4bit(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "unsloth/Phi-4-mini-instruct-bnb-4bit"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-format",
                "bitsandbytes",
            ],
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
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.6)  # 断言精度大于阈值


class TestUnslothPhi4MiniUnslothBnb4bit(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-format",
                "bitsandbytes",
            ],
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
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.6)  # 断言精度大于阈值


if __name__ == "__main__":
    unittest.main()
