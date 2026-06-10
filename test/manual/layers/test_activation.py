# 文件名: test_activation.py - 测试激活函数实现（GeluAndMul、QuickGELU）
import itertools
import unittest

import torch

from sglang.srt.layers.activation import GeluAndMul, QuickGELU
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

_is_hip = is_hip()


class TestGeluAndMul(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 2048]
    D = [512, 4096, 5120, 13824]
    SEEDS = [0]

    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        if not torch.cuda.is_available():  # 检查CUDA可用性
            raise unittest.SkipTest("CUDA is not available")  # 跳过测试
        torch.set_default_device("cuda")

    # 运行GeluAndMul单参数组合测试
    def _run_gelu_and_mul_test(self, num_tokens, d, dtype, seed):
        torch.manual_seed(seed)  # 设置随机种子

        layer = GeluAndMul().to(dtype=dtype)
        x = torch.randn(num_tokens, 2 * d, dtype=dtype)

        with torch.inference_mode():  # 推理模式（禁用梯度计算）
            ref_out = layer.forward_native(x)
            out = layer.forward_cuda(x)

        if dtype == torch.bfloat16:
            atol = rtol = 1e-2
        else:
            atol = rtol = 1e-3

        self.assertTrue(torch.allclose(out, ref_out, atol=atol, rtol=rtol))  # 断言条件为真

    # 测试gelu and mul功能
    def test_gelu_and_mul(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.D,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                d=params[1],
                dtype=params[2],
                seed=params[3],
            ):
                self._run_gelu_and_mul_test(*params)


class TestQuickGELU(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 2048]  # batch = sequence length
    DIMS = [512, 4096, 5120, 13824]  # all multiples of 16 bytes
    SEEDS = [0]

    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        if not torch.cuda.is_available():  # 检查CUDA可用性
            raise unittest.SkipTest("CUDA is not available")  # 跳过测试
        torch.set_default_device("cuda")

    # 运行QuickGELU单参数组合测试
    def _run_gelu_quick_test(self, n_tok: int, dim: int, dtype: torch.dtype, seed: int):
        torch.manual_seed(seed)  # 设置随机种子

        layer = QuickGELU().to(dtype=dtype)

        x = torch.randn(n_tok, dim, dtype=dtype, device="cuda")

        with torch.inference_mode():  # 推理模式（禁用梯度计算）
            ref = layer.forward_native(x)  # x * sigmoid(1.702 * x), fp32 math
            if _is_hip:
                out = layer.forward_hip(x)  # 128-bit vectorised kernel from sgl-kernel
            else:
                out = layer.forward_cuda(x)

        tol = 1e-2 if dtype is torch.bfloat16 else 1e-3
        self.assertTrue(  # 断言条件为真
            torch.allclose(out, ref, atol=tol, rtol=tol),  # 验证张量近似相等
            msg=f"Mismatch @ B={n_tok}, D={dim}, dtype={dtype}",
        )
        print(f"Match @ B={n_tok}, D={dim}, dtype={dtype}")

    # 测试quick gelu功能
    def test_quick_gelu(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.DIMS, self.DTYPES, self.SEEDS
        ):
            with self.subTest(
                num_tokens=params[0],
                dim=params[1],
                dtype=params[2],
                seed=params[3],
            ):
                self._run_gelu_quick_test(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
