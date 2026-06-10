# 文件名: test_linear_attn_model_registry.py - 线性注意力模型注册表
"""Unit tests for srt/configs/linear_attn_model_registry.py"""

import unittest

from sglang.srt.configs.linear_attn_model_registry import (
    _LINEAR_ATTN_MODEL_REGISTRY,
    LinearAttnModelSpec,
    get_linear_attn_config,
    get_linear_attn_spec_by_arch,
    import_backend_class,
    register_linear_attn_model,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


# Dummy config classes for testing
class FakeLinearAttnConfig:
    full_attention_layer_ids = [0, 2, 4]


# FakeVLMWrapperConfig类
class FakeVLMWrapperConfig:
    """Simulates a VLM wrapper that has get_text_config()."""

    def __init__(self):
        self._text_config = FakeLinearAttnConfig()

    # FakeVLMWrapperConfig类的get_text_config
    def get_text_config(self):
        return self._text_config


# AnotherConfig类
class AnotherConfig:
    pass


# TestLinearAttnModelRegistry类
class TestLinearAttnModelRegistry(CustomTestCase):

    # TestLinearAttnModelRegistry类的测试初始化设置
    def setUp(self):
        # Save and clear the global registry between tests
        self._saved_registry = list(_LINEAR_ATTN_MODEL_REGISTRY)
        _LINEAR_ATTN_MODEL_REGISTRY.clear()

    # TestLinearAttnModelRegistry类的测试清理
    def tearDown(self):
        _LINEAR_ATTN_MODEL_REGISTRY.clear()
        _LINEAR_ATTN_MODEL_REGISTRY.extend(self._saved_registry)

    # TestLinearAttnModelRegistry类的内部方法_make_spec
    def _make_spec(self, **overrides):
        defaults = dict(
            config_class=FakeLinearAttnConfig,
            backend_class_name="sglang.srt.layers.attention.triton_backend.TritonAttnBackend",
            arch_names=["FakeModelForCausalLM"],
        )
        defaults.update(overrides)
        return LinearAttnModelSpec(**defaults)

    # TestLinearAttnModelRegistry类的测试registerandlookupbyconfig
    def test_register_and_lookup_by_config(self):
        spec = self._make_spec()
        register_linear_attn_model(spec)

        hf_config = FakeLinearAttnConfig()
        result = get_linear_attn_config(hf_config)
        self.assertIsNotNone(result)  # 断言不为None
        self.assertIs(result[0], spec)  # 断言是同一对象
        self.assertIs(result[1], hf_config)  # 断言是同一对象

    # TestLinearAttnModelRegistry类的测试lookupnomatch
    def test_lookup_no_match(self):
        spec = self._make_spec()
        register_linear_attn_model(spec)

        result = get_linear_attn_config(AnotherConfig())
        self.assertIsNone(result)  # 断言为None

    # TestLinearAttnModelRegistry类的测试lookupemptyregistry
    def test_lookup_empty_registry(self):
        result = get_linear_attn_config(FakeLinearAttnConfig())
        self.assertIsNone(result)  # 断言为None

    # TestLinearAttnModelRegistry类的测试unwraptextconfig
    def test_unwrap_text_config(self):
        spec = self._make_spec(unwrap_text_config=True)
        register_linear_attn_model(spec)

        vlm_config = FakeVLMWrapperConfig()
        result = get_linear_attn_config(vlm_config)
        self.assertIsNotNone(result)  # 断言不为None
        self.assertIs(result[0], spec)  # 断言是同一对象
        # The resolved config should be the inner text config
        self.assertIsInstance(result[1], FakeLinearAttnConfig)
        self.assertIs(result[1], vlm_config._text_config)  # 断言是同一对象

    # TestLinearAttnModelRegistry类的测试unwraptextconfignomatch
    def test_unwrap_text_config_no_match(self):
        """unwrap_text_config=False should not call get_text_config()."""
        spec = self._make_spec(unwrap_text_config=False)
        register_linear_attn_model(spec)

        vlm_config = FakeVLMWrapperConfig()
        # VLM wrapper itself is not a FakeLinearAttnConfig, so no match
        result = get_linear_attn_config(vlm_config)
        self.assertIsNone(result)  # 断言为None

    # TestLinearAttnModelRegistry类的测试lookupbyarch
    def test_lookup_by_arch(self):
        spec = self._make_spec(arch_names=["AlphaForCausalLM", "BetaForCausalLM"])
        register_linear_attn_model(spec)

        self.assertIs(get_linear_attn_spec_by_arch("AlphaForCausalLM"), spec)  # 断言是同一对象
        self.assertIs(get_linear_attn_spec_by_arch("BetaForCausalLM"), spec)  # 断言是同一对象
        self.assertIsNone(get_linear_attn_spec_by_arch("GammaForCausalLM"))  # 断言为None

    # TestLinearAttnModelRegistry类的测试lookupbyarchemptyregistry
    def test_lookup_by_arch_empty_registry(self):
        self.assertIsNone(get_linear_attn_spec_by_arch("AnyArch"))  # 断言为None

    # TestLinearAttnModelRegistry类的测试multipleregistrations
    def test_multiple_registrations(self):
        spec_a = self._make_spec(
            config_class=FakeLinearAttnConfig,
            arch_names=["AlphaForCausalLM"],
        )
        spec_b = self._make_spec(
            config_class=AnotherConfig,
            arch_names=["BetaForCausalLM"],
        )
        register_linear_attn_model(spec_a)
        register_linear_attn_model(spec_b)

        # Config-based lookup
        self.assertIs(get_linear_attn_config(FakeLinearAttnConfig())[0], spec_a)  # 断言是同一对象
        self.assertIs(get_linear_attn_config(AnotherConfig())[0], spec_b)  # 断言是同一对象

        # Arch-based lookup
        self.assertIs(get_linear_attn_spec_by_arch("AlphaForCausalLM"), spec_a)  # 断言是同一对象
        self.assertIs(get_linear_attn_spec_by_arch("BetaForCausalLM"), spec_b)  # 断言是同一对象

    # TestLinearAttnModelRegistry类的测试firstmatchwins
    def test_first_match_wins(self):
        """When two specs match the same config class, the first registered wins."""
        spec1 = self._make_spec(backend_class_name="pkg.Backend1")
        spec2 = self._make_spec(backend_class_name="pkg.Backend2")
        register_linear_attn_model(spec1)
        register_linear_attn_model(spec2)

        result = get_linear_attn_config(FakeLinearAttnConfig())
        self.assertIs(result[0], spec1)  # 断言是同一对象

    # TestLinearAttnModelRegistry类的测试importbackendclass
    def test_import_backend_class(self):
        # Import a real stdlib class to verify the mechanism
        cls = import_backend_class("collections.OrderedDict")
        from collections import OrderedDict

        self.assertIs(cls, OrderedDict)  # 断言是同一对象

    # TestLinearAttnModelRegistry类的测试specdefaults
    def test_spec_defaults(self):
        spec = LinearAttnModelSpec(
            config_class=FakeLinearAttnConfig,
            backend_class_name="pkg.mod.Cls",
        )
        self.assertEqual(spec.arch_names, [])  # 断言相等
        self.assertTrue(spec.uses_mamba_radix_cache)  # 断言为真
        self.assertTrue(spec.support_mamba_cache)  # 断言为真
        self.assertFalse(spec.support_mamba_cache_extra_buffer)  # 断言为假
        self.assertFalse(spec.unwrap_text_config)  # 断言为假


if __name__ == "__main__":
    unittest.main()
