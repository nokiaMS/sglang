# 文件名: test_falcon_h1_models.py - 测试Falcon-H1模型（含TP4和非门控RMS变体）
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestFalconH1(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "tiiuae/Falcon-H1-0.5B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tensor-parallel-size",
                "1",
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
        self.assertGreater(metrics["score"], 0.74)  # 断言精度大于阈值


class TestFalconH1TP4(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "tiiuae/Falcon-H1-0.5B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tensor-parallel-size",
                "4",
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
        self.assertGreater(metrics["score"], 0.74)  # 断言精度大于阈值


class TestFalconH1NoGatedRMS(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "tiiuae/Falcon-H1-1.5B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tensor-parallel-size",
                "1",
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
        self.assertGreater(metrics["score"], 0.74)  # 断言精度大于阈值


class TestFalconH1NoGatedTP4(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "tiiuae/Falcon-H1-1.5B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tensor-parallel-size",
                "4",
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
        self.assertGreater(metrics["score"], 0.74)  # 断言精度大于阈值
