# 文件名: test_pcg_glm5_fp4.py - GLM-5 NVFP4模型分段CUDA图预填充测试
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# 注册CI配置
register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-b200")

GLM5_FP4_MODEL = "nvidia/GLM-5-NVFP4"  # GLM-5 FP4量化模型路径


class TestPCGGlm5Fp4(CustomTestCase):
    """PCG prefill on GLM-5-NVFP4 (DSA model, TP=4, B200).
    GLM-5 NVFP4模型上的分段CUDA图预填充测试（DSA模型，TP=4，B200）

    GLM-5 uses GlmMoeDsaForCausalLM (DSA attention). This test verifies that
    piecewise CUDA graph works correctly after the DSA indexer was updated to
    cache k_fp8/k_scale for PCG-compatible prefill.
    GLM-5使用GlmMoeDsaForCausalLM（DSA注意力机制）。此测试验证在DSA索引器
    更新为缓存k_fp8/k_scale以兼容PCG预填充后，分段CUDA图是否能正常工作。
    """

    @classmethod
    def setUpClass(cls):
        """启动GLM-5 FP4模型服务器"""
        cls.model = GLM5_FP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",  # 4路张量并行
                "--trust-remote-code",
                "--reasoning-parser",
                "glm45",  # GLM4.5推理解析器
                "--tool-call-parser",
                "glm47",  # GLM4.7工具调用解析器
                "--quantization",
                "modelopt_fp4",  # ModelOpt FP4量化
                "--disable-flashinfer-autotune",  # 禁用FlashInfer自动调优
                "--enforce-piecewise-cuda-graph",  # 强制分段CUDA图
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',  # 启用64线程加载
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
            num_examples=200,  # 200个测试样本
            num_threads=200,  # 200线程
            max_tokens=4096,  # 最大4096个token
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.92)  # 精度阈值92%


if __name__ == "__main__":
    unittest.main()
