# 文件名: test_lora_tuning_config.py - 测试LoRA CSGMV调优配置的文件名生成与加载逻辑
"""Unit tests for LoRA CSGMV tuning config loading."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.lora.triton_ops.lora_tuning_config import (
    DEFAULT_EXPAND_CONFIG,
    DEFAULT_SHRINK_CONFIG,
    get_lora_config_file_name,
    get_lora_configs,
    get_lora_expand_config,
    get_lora_shrink_config,
)

_MODULE = "sglang.srt.lora.triton_ops.lora_tuning_config"

# Shared fixture
_TUNED_CONFIGS = {
    32: {"BLOCK_N": 32, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3},
    128: {"BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2},
}


class TestLoraConfigFileName(unittest.TestCase):
    @patch(f"{_MODULE}.get_device_name", return_value="NVIDIA H100")
    # 测试includes all params功能
    def test_includes_all_params(self, _):
        name = get_lora_config_file_name("shrink", K=1024, R=64, S=3)
        self.assertEqual(name, "lora_shrink,K=1024,R=64,S=3,device=NVIDIA_H100.json")  # 断言值相等

    @patch(f"{_MODULE}.get_device_name", return_value="GPU")
    # 测试different slices different filenames功能
    def test_different_slices_different_filenames(self, _):
        s1 = get_lora_config_file_name("shrink", 1024, 64, S=1)
        s3 = get_lora_config_file_name("shrink", 1024, 64, S=3)
        self.assertNotEqual(s1, s3)  # 断言值不相等


class TestLoraConfigLoading(unittest.TestCase):
    # 测试方法级别初始化
    def setUp(self):
        get_lora_configs.cache_clear()
        self.tmpdir = tempfile.mkdtemp()

    # 内部方法：write config
    def _write_config(self, triton_ver_dir, filename, data):
        d = os.path.join(self.tmpdir, "csgmv_configs", triton_ver_dir)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, filename), "w") as f:
            json.dump(data, f)

    @patch(f"{_MODULE}.get_device_name", return_value="TestGPU")
    @patch(f"{_MODULE}.triton")
    def test_load_and_fallback(self, mock_triton, _):
        """Loads exact version, falls back to other version, returns None if missing."""
        config_data = {"32": {"BLOCK_N": 32, "BLOCK_K": 128}}
        self._write_config(
            "triton_3_5_1",
            "lora_shrink,K=1024,R=64,S=3,device=TestGPU.json",
            config_data,
        )

        # Exact match
        mock_triton.__version__ = "3.5.1"
        with patch.dict(os.environ, {"SGLANG_LORA_CONFIG_DIR": self.tmpdir}):
            result = get_lora_configs("shrink", 1024, 64, 3)
        self.assertEqual(result[32]["BLOCK_N"], 32)  # 断言值相等

        # Fallback from newer version
        get_lora_configs.cache_clear()
        mock_triton.__version__ = "3.6.0"
        with patch.dict(os.environ, {"SGLANG_LORA_CONFIG_DIR": self.tmpdir}):
            result = get_lora_configs("shrink", 1024, 64, 3)
        self.assertIsNotNone(result)  # 断言值不为None

        # Missing config returns None
        get_lora_configs.cache_clear()
        with patch.dict(os.environ, {"SGLANG_LORA_CONFIG_DIR": self.tmpdir}):
            self.assertIsNone(get_lora_configs("shrink", 9999, 64, 1))


class TestConfigSelection(unittest.TestCase):
    """Test exact match, nearest-neighbor, and default fallback for both kernels."""

    KERNELS = [
        (get_lora_shrink_config, DEFAULT_SHRINK_CONFIG),
        (get_lora_expand_config, DEFAULT_EXPAND_CONFIG),
    ]

    # 测试方法级别初始化
    def setUp(self):
        get_lora_configs.cache_clear()
        from sglang.srt.lora.triton_ops import lora_tuning_config

        lora_tuning_config._logged_configs.clear()

    # 测试defaults when no config功能
    def test_defaults_when_no_config(self):
        for get_fn, default in self.KERNELS:
            with self.subTest(fn=get_fn.__name__):
                with patch(f"{_MODULE}.get_lora_configs", return_value=None):
                    config = get_fn(K=1024, R=64, num_slices=1, chunk_size=32)
                self.assertEqual(config, default)  # 断言值相等

    # 测试exact and nearest neighbor功能
    def test_exact_and_nearest_neighbor(self):
        for get_fn, _ in self.KERNELS:
            with self.subTest(fn=get_fn.__name__):
                with patch(f"{_MODULE}.get_lora_configs", return_value=_TUNED_CONFIGS):
                    # Exact match for chunk_size=32
                    self.assertEqual(  # 断言值相等
                        get_fn(K=1024, R=64, num_slices=1, chunk_size=32)["BLOCK_N"], 32
                    )
                    # Nearest neighbor: 100 is closer to 128
                    self.assertEqual(  # 断言值相等
                        get_fn(K=1024, R=64, num_slices=1, chunk_size=100)["BLOCK_N"],
                        64,
                    )


if __name__ == "__main__":
    unittest.main()
