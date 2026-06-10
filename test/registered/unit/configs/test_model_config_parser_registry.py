# 文件名: test_model_config_parser_registry.py - 模型配置解析器注册表
"""Unit tests for srt/configs/model_config_parser_registry.py"""

import unittest

from transformers import PretrainedConfig

from sglang.srt.configs.model_config_parser_registry import (
    _MODEL_CONFIG_PARSER_REGISTRY,
    ModelConfigParserBase,
    get_model_config_parser,
    register_model_config_parser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


# _FakeParser类
class _FakeParser(ModelConfigParserBase):

    # _FakeParser类的parse
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


# _AnotherFakeParser类
class _AnotherFakeParser(ModelConfigParserBase):

    # _AnotherFakeParser类的parse
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


# TestModelConfigParserRegistry类
class TestModelConfigParserRegistry(CustomTestCase):

    # TestModelConfigParserRegistry类的测试初始化设置
    def setUp(self):
        self._saved_registry = dict(_MODEL_CONFIG_PARSER_REGISTRY)
        _MODEL_CONFIG_PARSER_REGISTRY.clear()

    # TestModelConfigParserRegistry类的测试清理
    def tearDown(self):
        _MODEL_CONFIG_PARSER_REGISTRY.clear()
        _MODEL_CONFIG_PARSER_REGISTRY.update(self._saved_registry)

    # TestModelConfigParserRegistry类的测试registerthengetroundtrip
    def test_register_then_get_roundtrip(self):
        register_model_config_parser("fake")(_FakeParser)
        self.assertIsInstance(get_model_config_parser("fake"), _FakeParser)

    # TestModelConfigParserRegistry类的测试registerrejectsnonsubclass
    def test_register_rejects_non_subclass(self):

        # NotAParser类
        class NotAParser:
            pass

        with self.assertRaises(ValueError) as ctx:  # 断言抛出异常
            register_model_config_parser("bad")(NotAParser)
        self.assertIn("ModelConfigParserBase", str(ctx.exception))  # 断言包含

    # TestModelConfigParserRegistry类的测试unknownnameraiseswithregisteredlist
    def test_unknown_name_raises_with_registered_list(self):
        register_model_config_parser("fake")(_FakeParser)
        register_model_config_parser("another")(_AnotherFakeParser)
        with self.assertRaises(ValueError) as ctx:  # 断言抛出异常
            get_model_config_parser("does-not-exist")
        msg = str(ctx.exception)
        self.assertIn("does-not-exist", msg)  # 断言包含
        self.assertIn("another", msg)  # 断言包含
        self.assertIn("fake", msg)  # 断言包含


if __name__ == "__main__":
    unittest.main()
