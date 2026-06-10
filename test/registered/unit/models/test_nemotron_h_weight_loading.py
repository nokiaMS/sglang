# 文件名: test_nemotron_h_weight_loading.py - Nemotron-H权重加载
"""
Unit tests for NemotronHForCausalLM.load_weights.

Regression test for Nemotron-H expert scale checkpoint tensors that map to
parameters absent from the current runtime model.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.nemotron_h import NemotronHForCausalLM


# _FakePPGroup类
class _FakePPGroup:
    is_first_rank = True
    is_last_rank = True


# _FakeParam类
class _FakeParam:

    # _FakeParam类的初始化
    def __init__(self):
        self.loaded = None

    # _FakeParam类的weight_loader
    def weight_loader(
        self, param, loaded_weight, name, *, shard_id=None, expert_id=None
    ):
        self.loaded = (param, loaded_weight, name, shard_id, expert_id)


# TestNemotronHWeightLoading类
class TestNemotronHWeightLoading(unittest.TestCase):

    # TestNemotronHWeightLoading类的内部方法_make_minimal_model
    def _make_minimal_model(self, named_parameters=()):
        model = object.__new__(NemotronHForCausalLM)
        model.config = SimpleNamespace(n_routed_experts=2, max_n_routed_experts=2)
        model.model = SimpleNamespace()
        model.pp_group = _FakePPGroup()
        model.remap_prefix = {}
        model.remap_substr = {}
        model.stacked_params_mapping = []
        model.named_parameters = lambda: iter(named_parameters)
        return model

    # TestNemotronHWeightLoading类的测试expertinputscalewithouttargetparameterisskipped
    def test_expert_input_scale_without_target_parameter_is_skipped(self):
        """Expert scale weights absent from params_dict should not raise KeyError."""
        model = self._make_minimal_model()
        weights = [
            (
                "model.layers.1.mixer.experts.0.down_proj.input_scale",
                torch.ones(1),
            )
        ]

        model.load_weights(weights)

    # TestNemotronHWeightLoading类的测试expertweightwithtargetparameterisloaded
    def test_expert_weight_with_target_parameter_is_loaded(self):
        param = _FakeParam()
        model = self._make_minimal_model(
            [("model.layers.1.mixer.experts.w2_weight", param)]
        )
        loaded_weight = torch.ones(1)
        weights = [
            (
                "model.layers.1.mixer.experts.0.down_proj.weight",
                loaded_weight,
            )
        ]

        model.load_weights(weights)

        self.assertEqual(  # 断言相等
            param.loaded,
            (
                param,
                loaded_weight,
                "model.layers.1.mixer.experts.w2_weight",
                "w2",
                0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
