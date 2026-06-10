# 文件名: test_pp_single_node.py - 流水线并行(PP)单节点精度和功能测试，包含DP Attention、Gemma4、混合分块等测试
"""
Usage:
python3 -m unittest test_pp_single_node.TestPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestDPAttentionDP2PP2.test_gsm8k
python3 -m unittest test_pp_single_node.TestGemma4PPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestGemma4PPAccuracy.test_mmmu
python3 -m unittest test_pp_single_node.TestGemma4PLEPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestPPMixedChunk.test_gsm8k
python3 -m unittest test_pp_single_node.TestFixedBugs.test_chunked_prefill_with_small_bs
"""

import time
import unittest
from types import SimpleNamespace

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PLE_PP,
    DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PP,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch_server,
)

# 注册CI配置
register_cuda_ci(est_time=500, stage="base-c", runner_config="4-gpu-h100")
register_amd_ci(est_time=500, suite="stage-c-test-4-gpu-amd")


class TestPPAccuracy(unittest.TestCase):
    """基础PP精度测试"""

    @classmethod
    def setUpClass(cls):
        """启动TP=2 PP=2服务器"""
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                2,
                "--pp-size",
                2,  # 2级流水线并行
                "--chunked-prefill-size",
                256,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """测试GSM8K数学推理精度"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_amd_ci():
            # AMD triton backend produces slightly lower accuracy than FA3 on NVIDIA
            # AMD triton后端精度略低于NVIDIA FA3
            self.assertGreater(metrics["score"], 0.70)
        else:
            self.assertGreater(metrics["score"], 0.74)
        # Wait a little bit so that the memory check happens.
        # 等待一段时间以确保内存检查完成
        time.sleep(4)

    def test_logprob(self):
        """测试对数概率返回功能"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
                "return_logprob": True,  # 返回对数概率
                "top_logprobs_num": 5,  # 返回top-5对数概率
                "logprob_start_len": 0,  # 从第0个token开始
            },
        )
        response_json = response.json()
        input_token_logprobs = response_json["meta_info"]["input_token_logprobs"]  # 输入token对数概率
        output_token_logprobs = response_json["meta_info"]["output_token_logprobs"]  # 输出token对数概率
        output_top_logprobs = response_json["meta_info"]["output_top_logprobs"]  # 输出top对数概率

        assert len(input_token_logprobs) == 6
        assert len(output_token_logprobs) == 16
        assert len(output_top_logprobs) == 16


@unittest.skipIf(is_in_amd_ci(), "MLA model with DP attention not yet supported on AMD")  # AMD尚未支持MLA模型的DP注意力
class TestDPAttentionDP2PP2(CustomTestCase):
    """DP Attention + PP测试（DP=2, PP=2）"""

    @classmethod
    def setUpClass(cls):
        """启动DP=2 PP=2服务器"""
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--pp-size",
                "2",
                "--enable-dp-attention",  # 启用DP注意力
                "--dp",
                "2",  # 数据并行度2
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """测试GSM8K数学推理精度"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.8)


@unittest.skipIf(
    is_in_amd_ci(),
    "Gemma4 PP not yet validated on AMD",  # Gemma4 PP尚未在AMD上验证
)
class TestGemma4PPAccuracy(unittest.TestCase):
    """End-to-end PP=2 accuracy gate for Gemma4 multimodal.
    Gemma4多模态PP=2端到端精度测试

    Gemma4 has full-attention layers with head_dim=512 (FA's max is 256), so
    sglang auto-selects the triton attention backend; no manual flag needed.
    The 26B BF16 model splits to ~26 GB per stage under PP=2, well within an
    H100's 80 GB.
    Gemma4有head_dim=512的全注意力层（FA最大为256），因此sglang自动选择
    triton注意力后端，无需手动标志。26B BF16模型在PP=2下每阶段约26GB，
    在H100的80GB范围内。
    """

    @classmethod
    def setUpClass(cls):
        """启动Gemma4 PP=2服务器"""
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PP
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                2,
                "--trust-remote-code",
                "--enable-multimodal",  # 启用多模态
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """测试GSM8K数学推理精度（使用Chat API）"""
        # Gemma4 is instruction-tuned and doesn't follow few-shot completion
        # prompts well — use the chat API (default in run_eval), which scores
        # ~0.98 on this model vs ~0.44 with api="completion".
        # Gemma4是指令微调模型，不适合few-shot补全提示——使用Chat API，
        # 得分约0.98，而api="completion"仅约0.44。
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        # Chat-API baseline ~0.98; gate well below to absorb sample-noise
        # without missing a real PP-routing regression (pre-PP-fix the model
        # produced garbage outputs scoring ≈ 0).
        # Chat API基线约0.98；设置较低阈值以吸收采样噪声，同时不会漏检
        # 真正的PP路由回归（PP修复前模型输出垃圾，得分约0）。
        self.assertGreaterEqual(metrics["score"], 0.90)
        # Wait a little bit so that the memory check happens.
        # 等待一段时间以确保内存检查完成
        time.sleep(4)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")  # 跳过以减少CI执行时间
    def test_mmmu(self):
        """测试MMMU多模态理解精度"""
        # Multimodal accuracy gate covering the vision_tower → embed_vision
        # (first rank) → PP-proxy handoff → LM tail (last rank) chain.
        # Measured 0.71 on 200 examples; full eval (~900 questions) takes
        # ~5-7 min on H100 so this is manual-only.
        # 多模态精度测试，覆盖vision_tower → embed_vision（第一rank）→
        # PP代理传递 → LM尾部（最后rank）链路。
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # Measured 0.72 on this setup; published Gemma-4-26B MMMU lies in
        # 0.69-0.73.  Gate 0.65 leaves ~5 SE of headroom (SE on 900 binary
        # samples ≈ 0.015) while still catching mid-grade vision/PP
        # regressions, not just complete breakage.
        self.assertGreater(metrics["score"], 0.65)


@unittest.skipIf(
    is_in_amd_ci(),
    "Gemma4 PP not yet validated on AMD",  # Gemma4 PP尚未在AMD上验证
)
class TestGemma4PLEPPAccuracy(unittest.TestCase):
    """PP=2 coverage for Gemma4 PLE variants (per_layer_inputs proxy path).
    Gemma4 PLE变体的PP=2测试（per_layer_inputs代理路径）

    26B-A4B has ``hidden_size_per_layer_input=0`` so the default Gemma4 PP
    test never crosses the PLE branch.  Cuda graph + PLE corrupts outputs
    (the runner's hardcoded ``{hidden_states, residual}`` PP-proxy schema
    drops ``per_layer_inputs``), so this test pins the eager configuration.
    26B-A4B的hidden_size_per_layer_input=0，因此默认Gemma4 PP测试不经过PLE分支。
    CUDA图+PLE会损坏输出（运行器的硬编码PP代理模式丢弃per_layer_inputs），
    因此此测试固定使用eager配置。
    """

    @classmethod
    def setUpClass(cls):
        """启动Gemma4 PLE PP=2服务器（禁用CUDA图）"""
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PLE_PP
        cls.base_url = "http://127.0.0.1:23339"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                2,
                "--trust-remote-code",
                "--enable-multimodal",
                # Required for PLE under PP — see Gemma4TextModel guard.
                # PLE在PP下的必需参数——参见Gemma4TextModel守卫
                "--disable-cuda-graph",  # 禁用CUDA图
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """测试GSM8K数学推理精度（eager路径）"""
        # Eager-path baseline ~0.92; gate 0.80 catches PLE breakage
        # (corruption collapses score to ~0).
        # Eager路径基线约0.92；阈值0.80可捕获PLE损坏（损坏时得分约0）
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=100,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], 0.80)
        time.sleep(4)


class TestPPMixedChunk(CustomTestCase):
    """PP混合分块测试"""

    @classmethod
    def setUpClass(cls):
        """启动TP=2 PP=2混合分块服务器"""
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = "http://127.0.0.1:23338"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                2,
                "--pp-size",
                2,
                "--chunked-prefill-size",
                256,
                "--enable-mixed-chunk",  # 启用混合分块
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """测试GSM8K数学推理精度"""
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
        print(f"{metrics=}")

        if is_in_amd_ci():
            # AMD triton backend produces slightly lower accuracy than FA3 on NVIDIA
            # AMD triton后端精度略低于NVIDIA FA3
            self.assertGreater(metrics["score"], 0.70)
        else:
            self.assertGreater(metrics["score"], 0.74)
        # Wait a little bit so that the memory check happens.
        # 等待一段时间以确保内存检查完成
        time.sleep(4)


class TestFixedBugs(unittest.TestCase):
    """已修复Bug的回归测试"""

    def test_chunked_prefill_with_small_bs(self):
        """测试小批量大小下的分块预填充（回归测试）"""
        model = DEFAULT_MODEL_NAME_FOR_TEST
        server_args = ServerArgs(model_path=model)
        bench_args = OneBatchBenchArgs(
            batch_size=(1,),
            input_len=(1,),  # 极短输入
            output_len=(1,),  # 极短输出
            base_url=DEFAULT_URL_FOR_TEST,
        )
        other_server_args = [
            "--tp-size",
            2,
            "--pp-size",
            2,
            "--chunked-prefill-size",
            256,
            "--max-running-requests",
            2,  # 最大并发请求数2
        ]
        run_bench_one_batch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            server_args,
            bench_args,
            other_server_args,
        )


if __name__ == "__main__":
    unittest.main()
