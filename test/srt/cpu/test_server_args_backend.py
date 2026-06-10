# 文件名: test_server_args_backend.py - 测试ServerArgs在CPU后端下的默认注意力后端选择逻辑
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs


class TestServerArgsCPUBackend(unittest.TestCase):
    def _make_server_args(self, attention_backend=None):
        # 创建一个未初始化的ServerArgs实例并设置CPU相关字段
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.device = "cpu"
        server_args.attention_backend = attention_backend
        server_args.sampling_backend = None
        return server_args

    @patch("sglang.srt.server_args.is_host_cpu_arm64", return_value=True)
    def test_arm_cpu_defaults_to_torch_native(self, _mock_is_arm64):
        # 测试ARM CPU默认使用torch_native注意力后端
        server_args = self._make_server_args()

        ServerArgs._handle_cpu_backends(server_args)

        self.assertEqual(server_args.attention_backend, "torch_native")
        self.assertEqual(server_args.sampling_backend, "pytorch")

    @patch("sglang.srt.server_args.is_host_cpu_arm64", return_value=False)
    def test_x86_cpu_defaults_to_intel_amx(self, _mock_is_arm64):
        # 测试x86 CPU默认使用intel_amx注意力后端
        server_args = self._make_server_args()

        ServerArgs._handle_cpu_backends(server_args)

        self.assertEqual(server_args.attention_backend, "intel_amx")
        self.assertEqual(server_args.sampling_backend, "pytorch")


if __name__ == "__main__":
    unittest.main()
