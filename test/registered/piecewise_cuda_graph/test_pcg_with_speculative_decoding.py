# 文件名: test_pcg_with_speculative_decoding.py - 分段CUDA图与EAGLE3推测解码共存测试
"""Test piecewise CUDA graph coexisting with speculative decoding (EAGLE3).

PCG handles prefill/extend path while speculative decoding (EAGLE3) uses
decode CUDA graphs. This test verifies they don't interfere with each
other. MTP / STANDALONE / NGRAM variants moved to the sibling file
test_pcg_with_speculative_decoding_extra.py.
PCG处理预填充/扩展路径，而推测解码(EAGLE3)使用解码CUDA图。
此测试验证它们互不干扰。MTP/STANDALONE/NGRAM变体已移至同级文件。
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.pcg_spec_fixture import PCGSpecBase

# 注册CI配置
register_cuda_ci(est_time=531, stage="base-b", runner_config="2-gpu-large")


class TestPCGWithEAGLE3(PCGSpecBase, unittest.TestCase):
    """PCG + EAGLE3 on Qwen3-30B-A3B-Instruct-2507.
    PCG + EAGLE3在Qwen3-30B-A3B-Instruct-2507上的测试"""

    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    server_args = [
        "--tp",
        "2",  # 2路张量并行
        "--trust-remote-code",
        "--enforce-piecewise-cuda-graph",  # 强制分段CUDA图
        "--mem-fraction-static",
        "0.6",  # 静态内存分配比例
        "--speculative-algorithm",
        "EAGLE3",  # EAGLE3推测解码算法
        "--speculative-draft-model-path",
        "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",  # EAGLE3草稿模型路径
        "--speculative-num-steps",
        "5",  # 推测步数
        "--speculative-eagle-topk",
        "4",  # top-k值
        "--speculative-num-draft-tokens",
        "8",  # 草稿token数
    ]
    timeout_mult = 3  # 超时倍数
    server_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}  # 允许覆盖更长上下文长度
    accuracy_threshold = 0.75  # 精度阈值


if __name__ == "__main__":
    unittest.main()
