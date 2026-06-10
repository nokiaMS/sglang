# 文件名: test_runai_model_streamer_loader.py - RunAI模型流加载器
import sys
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


# _FakeModel类
class _FakeModel:

    # _FakeModel类的eval
    def eval(self):
        return self


# TestRunaiModelStreamerLoader类
class TestRunaiModelStreamerLoader(CustomTestCase):

    # TestRunaiModelStreamerLoader类的测试passesquantconfigtomodelinit
    def test_passes_quant_config_to_model_init(self):
        quant_config = object()
        fake_model = _FakeModel()

        with (
            patch.object(
                loader_mod,
                "_get_quantization_config",
                return_value=quant_config,
            ),
            patch.object(loader_mod, "_initialize_model") as mock_initialize_model,
            patch.object(
                loader_mod.DefaultModelLoader,
                "load_weights_and_postprocess",
            ) as mock_load_weights,
        ):
            mock_initialize_model.return_value = fake_model
            runai_loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={},
                )
            )
            model_config = cast(
                ModelConfig,
                SimpleNamespace(dtype=torch.float16, modelopt_quant=False),
            )

            model = runai_loader.load_model(
                model_config=model_config,
                device_config=DeviceConfig("cpu"),
            )

        self.assertIs(model, fake_model)  # 断言是同一对象
        self.assertIs(mock_load_weights.call_args.args[0], fake_model)  # 断言是同一对象
        self.assertIs(mock_initialize_model.call_args.args[2], quant_config)  # 断言是同一对象

    # TestRunaiModelStreamerLoader类的测试marksstreamertensors
    def test_marks_streamer_tensors(self):
        source_tensor = torch.tensor([1], dtype=torch.int32)

        # FakeStreamer类
        class FakeStreamer:

            # FakeStreamer类的特殊方法__enter__
            def __enter__(self):
                return self

            # FakeStreamer类的特殊方法__exit__
            def __exit__(self, *_args):
                pass

            # FakeStreamer类的stream_files
            def stream_files(self, *_args, **_kwargs):
                self.files_to_tensors_metadata = {0: [object()]}

            # FakeStreamer类的get_tensors
            def get_tensors(self):
                yield "weight", source_tensor

        with patch.dict(
            sys.modules,
            {"runai_model_streamer": SimpleNamespace(SafetensorsStreamer=FakeStreamer)},
        ):
            weights = list(
                weight_utils.runai_safetensors_weights_iterator(["model.safetensors"])
            )

        self.assertEqual(weights[0][0], "weight")  # 断言相等
        self.assertTrue(getattr(weights[0][1], weight_utils.RUNAI_STREAMER_TENSOR_ATTR))  # 断言为真

    # TestRunaiModelStreamerLoader类的测试deepseekcloneonlyclonesmarkedtensors
    def test_deepseek_clone_only_clones_marked_tensors(self):
        unmarked = torch.tensor([1], dtype=torch.int32)

        self.assertIs(  # 断言是同一对象
            deepseek_weight_loader._clone_if_runai_streamed_tensor(unmarked),
            unmarked,
        )

        marked = torch.tensor([1], dtype=torch.int32)
        setattr(marked, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

        cloned = deepseek_weight_loader._clone_if_runai_streamed_tensor(marked)

        self.assertIsNot(cloned, marked)  # 断言不是同一对象
        marked.fill_(2)
        self.assertEqual(cloned.item(), 1)  # 断言相等

    # TestRunaiModelStreamerLoader类的测试getmodelloaderusesrunaiforprequantizedmodelopt
    def test_get_model_loader_uses_runai_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.RUNAI_STREAMER,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RunaiModelStreamerLoader)

    # TestRunaiModelStreamerLoader类的测试getmodelloaderusesremoteinstanceforprequantizedmodelopt
    def test_get_model_loader_uses_remote_instance_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.REMOTE_INSTANCE,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RemoteInstanceModelLoader)


if __name__ == "__main__":
    unittest.main()
