# 文件名: test_pp_single_node_extra.py - 流水线并行(PP)单节点额外精度测试，包含多种模型
"""
Usage:
python3 -m unittest test_pp_single_node_extra.TestQwenVLPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node_extra.TestQwenPPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestQwenPPTieWeightsAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestQwenMoePPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestQwen35PPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestGLM41VPPAccuracy.test_mmmu
"""

import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
    DEFAULT_MODEL_NAME_FOR_TEST_VL_PP,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
)

# 注册CI配置：CUDA和AMD平台
register_cuda_ci(est_time=350, stage="extra-b", runner_config="4-gpu-h100")
register_amd_ci(est_time=350, suite="stage-c-test-4-gpu-amd")


@unittest.skipIf(
    is_in_amd_ci(),
    "VLM PP accuracy too low on AMD (0.48-0.50 with both aiter and triton)",
    # AMD上VLM PP精度过低（aiter和triton均为0.48-0.50）
)
class TestQwenVLPPAccuracy(unittest.TestCase):
    """Qwen视觉语言模型PP精度测试"""

    @classmethod
    def setUpClass(cls):
        """启动Qwen VLM PP服务器"""
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_VL_PP
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST_VL_PP,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                4,  # 4级流水线并行
                "--chunked-prefill-size",
                8192,  # 分块预填充大小
                "--enable-multimodal",  # 启用多模态
            ],
        )

    def test_gsm8k(self):
        """测试GSM8K数学推理精度"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,  # 128线程
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreaterEqual(metrics["score"], 0.65)  # 精度阈值65%
        # Wait a little bit so that the memory check happens.
        # 等待一段时间以确保内存检查完成
        time.sleep(4)

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        kill_process_tree(cls.process.pid)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")  # 跳过以减少CI执行时间
    def test_mmmu(self):
        """测试MMMU多模态理解精度"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.26)


class TestQwenPPAccuracy(unittest.TestCase):
    """Qwen模型PP精度一致性测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23334"  # different ports to avoid conflicts
        # 不同端口以避免冲突
        cls.model_name = "Qwen/Qwen3-8B"  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        """运行指定PP大小的GSM8K测试"""
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,  # 流水线并行大小
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")  # 跳过以减少CI执行时间
    def test_pp_consistency(self):
        """测试PP与基线模型的精度一致性"""
        baseline = self.run_gsm8k_test(pp_size=1)  # 基线：无PP
        pp_metrics = self.run_gsm8k_test(pp_size=2)  # PP=2

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.02,  # 允许2%的精度下降
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


@unittest.skipIf(is_in_amd_ci(), "PP consistency too flaky on AMD 4-GPU runners")  # AMD 4-GPU上PP一致性不稳定
class TestQwenPPTieWeightsAccuracy(unittest.TestCase):
    """Qwen模型（共享词嵌入权重）PP精度一致性测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23335"  # different ports to avoid conflicts
        # 不同端口以避免冲突
        cls.model_name = (
            "Qwen/Qwen3-0.6B"  # qwen3 < 8B all have tie_word_embeddings = True
            # qwen3 < 8B 都有 tie_word_embeddings = True
        )

    def run_gsm8k_test(self, pp_size):
        """运行指定PP大小的GSM8K测试"""
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        """测试共享词嵌入权重模型的PP精度一致性"""
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.38)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.02,  # 允许2%的精度下降
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


class TestQwenMoePPAccuracy(unittest.TestCase):
    """Qwen MoE模型PP精度一致性测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23336"  # different ports to avoid conflicts
        # 不同端口以避免冲突
        cls.model_name = "Qwen/Qwen3-30B-A3B"

    def run_gsm8k_test(self, pp_size):
        """运行指定PP大小的GSM8K测试"""
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        """测试MoE模型的PP精度一致性"""
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.02,  # 允许2%的精度下降
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


@unittest.skipIf(
    is_in_ci(), "Qwen35 PP consistency too flaky on H100 and AMD 4-GPU runners"
    # Qwen35 PP一致性在H100和AMD 4-GPU上不稳定
)
class TestQwen35PPAccuracy(unittest.TestCase):
    """Qwen3.5模型PP精度一致性测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23337"  # different ports to avoid conflicts
        # 不同端口以避免冲突
        cls.model_name = (
            "Qwen/Qwen3.5-35B-A3B"
        )

    def run_gsm8k_test(self, tp_size, pp_size):
        """运行指定TP和PP大小的GSM8K测试"""
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                tp_size,  # 张量并行大小
                "--pp-size",
                pp_size,  # 流水线并行大小
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        """测试Qwen3.5模型的PP精度一致性"""
        baseline = self.run_gsm8k_test(tp_size=2, pp_size=1)  # 基线：TP=2, 无PP
        pp_metrics = self.run_gsm8k_test(tp_size=1, pp_size=2)  # TP=1, PP=2

        print(f"[Qwen35 PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.83)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.05,  # 允许5%的精度下降
            msg=(
                f"PP accuracy dropped more than 5% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


@unittest.skipIf(
    is_in_ci(), "Skipping GLM41V PP accuracy test before it gets more stable"
    # 跳过GLM41V PP精度测试，等待更稳定
)
class TestGLM41VPPAccuracy(unittest.TestCase):
    """GLM-4.1V视觉语言模型PP精度测试"""

    @classmethod
    def setUpClass(cls):
        """启动GLM-4.1V PP服务器"""
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                2,  # 2级流水线并行
                "--chunked-prefill-size",
                8192,
                "--enable-multimodal",
                "--reasoning-parser",
                "glm45",  # GLM4.5推理解析器
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        kill_process_tree(cls.process.pid)

    def test_mmmu(self):
        """测试MMMU多模态理解精度"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
            response_answer_regex=r"<\|begin_of_box\|>(.*)<\|end_of_box\|>",  # 答案提取正则
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.45)  # 精度阈值45%


if __name__ == "__main__":
    unittest.main()
