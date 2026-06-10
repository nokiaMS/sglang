# 文件名: test_crusoe_backend.py - Crusoe后端测试 - 验证Crusoe后端的推理功能
"""
Manual tests for the Crusoe managed inference backend.

Requires CRUSOE_API_KEY to be set in the environment.

Run all tests:
    python3 -m unittest test/manual/test_crusoe_backend.py

Run a single test:
    python3 -m unittest test_crusoe_backend.TestCrusoeBackend.test_mt_bench
"""

import unittest

from sglang import Crusoe, set_default_backend
from sglang.test.test_programs import (
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_stream,
)
from sglang.test.test_utils import CustomTestCase

# Default model available on Crusoe managed inference.
DEFAULT_CRUSOE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


class TestCrusoeBackend(CustomTestCase):
    backend = None

    @classmethod
    # setUpClass
    def setUpClass(cls):
        cls.backend = Crusoe(DEFAULT_CRUSOE_MODEL)

    # setUp
    def setUp(self):
        set_default_backend(self.backend)

    # 测试mt bench
    def test_mt_bench(self):
        test_mt_bench()

    # 测试stream
    def test_stream(self):
        test_stream()

    # 测试parallel decoding
    def test_parallel_decoding(self):
        test_parallel_decoding()

    # 测试parallel encoding
    def test_parallel_encoding(self):
        test_parallel_encoding()


class TestCrusoeBackendInit(CustomTestCase):
    """Unit tests for Crusoe backend initialisation — no network required."""

    # 测试raises without api key
    def test_raises_without_api_key(self):
        import os

        key = os.environ.pop("CRUSOE_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                Crusoe(DEFAULT_CRUSOE_MODEL, api_key=None)
        finally:
            if key is not None:
                os.environ["CRUSOE_API_KEY"] = key

    # 测试accepts explicit api key
    def test_accepts_explicit_api_key(self):
        backend = Crusoe(DEFAULT_CRUSOE_MODEL, api_key="test-key")
        self.assertIsNotNone(backend)

    # 测试custom base url
    def test_custom_base_url(self):
        backend = Crusoe(
            DEFAULT_CRUSOE_MODEL,
            api_key="test-key",
            base_url="https://managed-inference-api-proxy.crusoecloud.com/v1/",
        )
        self.assertIsNotNone(backend)


if __name__ == "__main__":
    unittest.main()
