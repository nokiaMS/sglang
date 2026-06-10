# 文件名: test_adaptive_spec_params.py - 自适应推测参数
import json
import tempfile
import unittest

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


# TestAdaptiveSpeculativeParams类
class TestAdaptiveSpeculativeParams(unittest.TestCase):

    # TestAdaptiveSpeculativeParams类的内部方法_make_params_from_config
    def _make_params_from_config(self, initial_steps: int, config: dict[str, object]):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(config, f)
            f.flush()
            return AdaptiveSpeculativeParams(
                initial_steps=initial_steps, cfg_path=f.name
            )

    # TestAdaptiveSpeculativeParams类的测试paramsloadsconfigpath
    def test_params_loads_config_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "candidate_steps": [1, 5],
                    "ema_alpha": 0.75,
                    "warmup_batches": 2,
                },
                f,
            )
            f.flush()

            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

        self.assertEqual(params.candidate_steps, [1, 3, 5])  # 断言相等
        self.assertEqual(params.ema_alpha, 0.75)  # 断言相等
        self.assertEqual(params.warmup_batches, 2)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试initialstepsaddedtocandidateswhenmissing
    def test_initial_steps_added_to_candidates_when_missing(self):
        params = self._make_params_from_config(2, {"candidate_steps": [1, 3, 7]})

        self.assertEqual(params.candidate_steps, [1, 2, 3, 7])  # 断言相等
        self.assertEqual(params.current_steps, 2)  # 断言相等
        self.assertEqual(params.ema_accept_len, 1.0)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试updaterespectswarmupandinterval
    def test_update_respects_warmup_and_interval(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 2,
            },
        )

        self.assertFalse(params.update([0, 0]))  # 断言为假
        self.assertEqual(params.current_steps, 3)  # 断言相等

        self.assertFalse(params.update([0, 0]))  # 断言为假
        self.assertEqual(params.current_steps, 3)  # 断言相等

        self.assertTrue(params.update([0, 0]))  # 断言为真
        self.assertEqual(params.current_steps, 1)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试emptybatchesdonotconsumewarmuporshiftsteps
    def test_empty_batches_do_not_consume_warmup_or_shift_steps(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 1,
            },
        )

        self.assertFalse(params.update([]))  # 断言为假
        self.assertEqual(params.current_steps, 3)  # 断言相等
        self.assertEqual(params.ema_accept_len, 2.0)  # 断言相等

        self.assertFalse(params.update([0, 0]))  # 断言为假
        self.assertEqual(params.current_steps, 3)  # 断言相等

        self.assertTrue(params.update([0, 0]))  # 断言为真
        self.assertEqual(params.current_steps, 1)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试updatescalesupacrosscandidates
    def test_update_scales_up_across_candidates(self):
        params = self._make_params_from_config(
            1,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([1, 1]))  # 断言为真
        self.assertEqual(params.current_steps, 3)  # 断言相等

        self.assertTrue(params.update([3, 3]))  # 断言为真
        self.assertEqual(params.current_steps, 7)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试updatecanscaledownacrosscandidatesinonerecompute
    def test_update_can_scale_down_across_candidates_in_one_recompute(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

        self.assertTrue(params.update([0, 0]))  # 断言为真
        self.assertEqual(params.current_steps, 1)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试exactrisethresholddoesnotupshift
    def test_exact_rise_threshold_does_not_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertFalse(params.update([2, 3]))  # 断言为假
        self.assertEqual(params.current_steps, 3)  # 断言相等
        self.assertEqual(params.ema_accept_len, 2.5)  # 断言相等

        self.assertTrue(params.update([3, 3]))  # 断言为真
        self.assertEqual(params.current_steps, 7)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试exactdropthresholddoesdownshift
    def test_exact_drop_threshold_does_downshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
                "up_hysteresis": 0.5,
            },
        )

        self.assertTrue(params.update([0, 1]))  # 断言为真
        self.assertEqual(params.current_steps, 1)  # 断言相等
        self.assertEqual(params.ema_accept_len, 0.5)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试hysteresiscanpreventprematureupshift
    def test_hysteresis_can_prevent_premature_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.75,
            },
        )

        self.assertFalse(params.update([3, 3]))  # 断言为假
        self.assertEqual(params.current_steps, 3)  # 断言相等

        self.assertTrue(params.update([4, 4]))  # 断言为真
        self.assertEqual(params.current_steps, 7)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试downhysteresiscanpreventprematuredownshift
    def test_down_hysteresis_can_prevent_premature_downshift(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": -0.75,
            },
        )

        self.assertFalse(params.update([2, 2]))  # 断言为假
        self.assertEqual(params.current_steps, 7)  # 断言相等

        self.assertTrue(params.update([1, 1]))  # 断言为真
        self.assertEqual(params.current_steps, 3)  # 断言相等

    # TestAdaptiveSpeculativeParams类的测试multibatchsequencecanrampupthenbackdown
    def test_multi_batch_sequence_can_ramp_up_then_back_down(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 0.5,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([4, 4]))  # 断言为真
        self.assertEqual(params.current_steps, 7)  # 断言相等
        self.assertEqual(params.ema_accept_len, 3.0)  # 断言相等

        self.assertTrue(params.update([0, 0]))  # 断言为真
        self.assertEqual(params.current_steps, 3)  # 断言相等
        self.assertEqual(params.ema_accept_len, 1.5)  # 断言相等

        self.assertFalse(params.update([0, 0]))  # 断言为假
        self.assertEqual(params.current_steps, 3)  # 断言相等
        self.assertEqual(params.ema_accept_len, 0.75)  # 断言相等

        self.assertTrue(params.update([0, 0]))  # 断言为真
        self.assertEqual(params.current_steps, 1)  # 断言相等
        self.assertEqual(params.ema_accept_len, 0.375)  # 断言相等


if __name__ == "__main__":
    unittest.main()
