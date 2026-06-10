# 文件名: test_grok_models.py - 测试Grok模型（使用dummy权重验证吞吐量）
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


class TestGrok(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = "lmzheng/grok-1"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(  # 启动推理服务器
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-format",
                "dummy",
                "--json-model-override-args",
                '{"num_hidden_layers": 2}',
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
            num_examples=64,
            num_threads=128,
        )
        metrics = run_eval(args)  # 运行评估
        print(f"{metrics=}")

        # It is dummy weights so we only assert the output throughput instead of accuracy.
        self.assertGreater(metrics["output_throughput"], 1000)  # 断言精度大于阈值


if __name__ == "__main__":
    unittest.main()
