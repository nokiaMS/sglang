# 文件名: test_llava.py - LLaVA视觉模型
import unittest
from unittest.mock import patch

from sglang.srt.models.llava import AutoModel, LlavaForConditionalGeneration
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-b-test-cpu")


# PixtralVisionConfig类
class PixtralVisionConfig:
    pass


# VoxtralRealtimeTextConfig类
class VoxtralRealtimeTextConfig:
    pass


# GoodConfig类
class GoodConfig:
    pass


# PixtralVisionModel类
class PixtralVisionModel:
    pass


# GoodArch类
class GoodArch:
    pass


# FakeMapping类
class FakeMapping:

    # FakeMapping类的初始化
    def __init__(self, voxtral_error):
        self.voxtral_error = voxtral_error

    # FakeMapping类的keys
    def keys(self):
        return [VoxtralRealtimeTextConfig, PixtralVisionConfig, GoodConfig]

    # FakeMapping类的get
    def get(self, config_cls, default=None):
        if config_cls is VoxtralRealtimeTextConfig:
            raise self.voxtral_error  # 抛出异常
        if config_cls is PixtralVisionConfig:
            return (PixtralVisionModel,)
        if config_cls is GoodConfig:
            return GoodArch
        return default


KNOWN_VOXTRAL_ERROR = ValueError(
    "Could not find VoxtralRealtimeTextModel neither in "
    "<module 'transformers.models.voxtral_realtime'> nor in "
    "<module 'transformers'>!"
)


# TestLlavaForConditionalGeneration类
class TestLlavaForConditionalGeneration(CustomTestCase):

    # TestLlavaForConditionalGeneration类的测试初始化设置
    def setUp(self):
        LlavaForConditionalGeneration._config_cls_name_to_arch_name_mapping.cache_clear()

    # TestLlavaForConditionalGeneration类的内部方法_build_mapping
    def _build_mapping(self, mapping):
        with patch.object(AutoModel, "_model_mapping", mapping):
            llava_model = object.__new__(LlavaForConditionalGeneration)
            return llava_model._config_cls_name_to_arch_name_mapping(AutoModel)

    @patch("sglang.srt.models.llava.logger.warning")

    # TestLlavaForConditionalGeneration类的测试skipknownbrokenvoxtralautomodelmappingentry
    def test_skip_known_broken_voxtral_automodel_mapping_entry(self, mock_warning):
        mapping = self._build_mapping(FakeMapping(KNOWN_VOXTRAL_ERROR))

        self.assertEqual(mapping[GoodConfig.__name__], GoodArch.__name__)  # 断言相等
        self.assertEqual(  # 断言相等
            mapping[PixtralVisionConfig.__name__], (PixtralVisionModel.__name__,)
        )
        self.assertNotIn(VoxtralRealtimeTextConfig.__name__, mapping)  # 断言不包含

        mock_warning.assert_called_once()
        self.assertEqual(  # 断言相等
            mock_warning.call_args.args,
            (
                "Skipping broken %s mapping for config %s: %s",
                AutoModel.__name__,
                VoxtralRealtimeTextConfig.__name__,
                unittest.mock.ANY,
            ),
        )

    # TestLlavaForConditionalGeneration类的测试othervoxtralmappingfailuresstillraise
    def test_other_voxtral_mapping_failures_still_raise(self):
        with self.assertRaisesRegex(ValueError, "some other failure"):
            self._build_mapping(FakeMapping(ValueError("some other failure")))


if __name__ == "__main__":
    unittest.main()
