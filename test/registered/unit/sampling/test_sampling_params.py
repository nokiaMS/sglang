# 文件名: test_sampling_params.py - 采样参数
"""Unit tests for srt/sampling/sampling_params.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.sampling.sampling_params import (
    MAX_LEN,
    TOP_K_ALL,
    SamplingParams,
    get_max_seq_length,
)
from sglang.test.test_utils import CustomTestCase


# TestSamplingParamsInit类
class TestSamplingParamsInit(CustomTestCase):

    # TestSamplingParamsInit类的测试zerotemperaturebecomesgreedy
    def test_zero_temperature_becomes_greedy(self):
        """Test greedy conversion when temperature is 0."""
        sp = SamplingParams(temperature=0.0)
        self.assertEqual(sp.top_k, 1)  # 断言相等
        self.assertEqual(sp.temperature, 1.0)  # 断言相等

    # TestSamplingParamsInit类的测试nearzerotemperaturebecomesgreedy
    def test_near_zero_temperature_becomes_greedy(self):
        """Test greedy conversion when temperature is near zero (1e-7)."""
        sp = SamplingParams(temperature=1e-7)
        self.assertEqual(sp.top_k, 1)  # 断言相等
        self.assertEqual(sp.temperature, 1.0)  # 断言相等

    # TestSamplingParamsInit类的测试temperatureatepsboundarynotgreedy
    def test_temperature_at_eps_boundary_not_greedy(self):
        """Test that temperature exactly at 1e-6 does not trigger greedy (strict <)."""
        sp = SamplingParams(temperature=1e-6)
        self.assertEqual(sp.temperature, 1e-6)  # 断言相等
        # top_k should remain at TOP_K_ALL (from -1 default)
        self.assertEqual(sp.top_k, TOP_K_ALL)  # 断言相等

    # TestSamplingParamsInit类的测试negativetemperaturenotmodified
    def test_negative_temperature_not_modified(self):
        """Test that __init__ preserves negative temperature (rejected by verify instead)."""
        sp = SamplingParams(temperature=-1.0)
        self.assertEqual(sp.temperature, -1.0)  # 断言相等

    # TestSamplingParamsInit类的测试topkminusonebecomestopkall
    def test_top_k_minus_one_becomes_top_k_all(self):
        """Test that top_k=-1 is converted to TOP_K_ALL (whole vocabulary)."""
        sp = SamplingParams(top_k=-1)
        self.assertEqual(sp.top_k, TOP_K_ALL)  # 断言相等

    # TestSamplingParamsInit类的测试positivetopkpreserved
    def test_positive_top_k_preserved(self):
        """Test that explicit positive top_k is kept as-is."""
        sp = SamplingParams(top_k=50)
        self.assertEqual(sp.top_k, 50)  # 断言相等

    # TestSamplingParamsInit类的测试stoptokenidsstoredasset
    def test_stop_token_ids_stored_as_set(self):
        """Test that stop_token_ids list is converted to set."""
        sp = SamplingParams(stop_token_ids=[1, 2, 3])
        self.assertIsInstance(sp.stop_token_ids, set)
        self.assertEqual(sp.stop_token_ids, {1, 2, 3})  # 断言相等

    # TestSamplingParamsInit类的测试stoptokenidsnonestaysnone
    def test_stop_token_ids_none_stays_none(self):
        """Test that None stop_token_ids stays None."""
        sp = SamplingParams(stop_token_ids=None)
        self.assertIsNone(sp.stop_token_ids)  # 断言为None

    # TestSamplingParamsInit类的测试emptystoptokenidsbecomesnone
    def test_empty_stop_token_ids_becomes_none(self):
        """Test that empty list is treated as None (falsy in Python)."""
        sp = SamplingParams(stop_token_ids=[])
        self.assertIsNone(sp.stop_token_ids)  # 断言为None


# TestSamplingParamsVerify类
class TestSamplingParamsVerify(CustomTestCase):

    VOCAB_SIZE = 32000

    # TestSamplingParamsVerify类的内部方法_make
    def _make(self, **kwargs):
        """Helper: create SamplingParams with safe defaults, override with kwargs."""
        defaults = dict(temperature=1.0, top_p=1.0, top_k=10, min_p=0.0)
        defaults.update(kwargs)
        return SamplingParams(**defaults)

    # TestSamplingParamsVerify类的测试validparamspass
    def test_valid_params_pass(self):
        """Default valid params should pass verify() without raising."""
        sp = self._make()
        sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试negativetemperatureraises
    def test_negative_temperature_raises(self):
        """Test that verify() rejects negative temperature (must be >= 0)."""
        sp = self._make(temperature=-0.5)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # --- top_p ---
    def test_top_p_negative_raises(self):
        """Test that verify() rejects negative top_p (valid range is (0, 1])."""
        sp = self._make(top_p=-0.5)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试toppzeroraises
    def test_top_p_zero_raises(self):
        """Test that verify() rejects top_p=0 (not in (0, 1])."""
        sp = self._make(top_p=0.0)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试toppaboveoneraises
    def test_top_p_above_one_raises(self):
        """Test that verify() rejects top_p > 1.0."""
        sp = self._make(top_p=1.1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试toppexactlyoneisvalid
    def test_top_p_exactly_one_is_valid(self):
        """Test that top_p=1.0 is accepted (inclusive upper bound)."""
        sp = self._make(top_p=1.0)
        sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试toppsmallpositiveisvalid
    def test_top_p_small_positive_is_valid(self):
        """Test that a small positive top_p (0.01) is accepted."""
        sp = self._make(top_p=0.01)
        sp.verify(self.VOCAB_SIZE)

    # --- min_p ---
    def test_min_p_negative_raises(self):
        """Test that verify() rejects negative min_p (valid range is [0, 1])."""
        sp = self._make(min_p=-0.1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试minpaboveoneraises
    def test_min_p_above_one_raises(self):
        """Test that verify() rejects min_p > 1.0."""
        sp = self._make(min_p=1.01)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试minpboundariesvalid
    def test_min_p_boundaries_valid(self):
        """Test that both 0.0 and 1.0 are accepted."""
        self._make(min_p=0.0).verify(self.VOCAB_SIZE)
        self._make(min_p=1.0).verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试topkzeroraises
    def test_top_k_zero_raises(self):
        """Test that verify() rejects top_k=0 (must be >=1 or -1 for all)."""
        sp = self._make()
        sp.top_k = 0  # bypass __init__ conversion
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试topknegativeraises
    def test_top_k_negative_raises(self):
        """Test that top_k=-2 is rejected (__init__ only converts -1)."""
        sp = self._make()
        sp.top_k = -2  # bypass __init__ conversion
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # --- frequency_penalty ---
    def test_frequency_penalty_below_minus_two_raises(self):
        """Test that verify() rejects frequency_penalty < -2.0."""
        sp = self._make(frequency_penalty=-2.1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试frequencypenaltyabovetworaises
    def test_frequency_penalty_above_two_raises(self):
        """Test that verify() rejects frequency_penalty > 2.0."""
        sp = self._make(frequency_penalty=2.1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试frequencypenaltyboundariesvalid
    def test_frequency_penalty_boundaries_valid(self):
        """Test that both -2.0 and 2.0 are accepted."""
        self._make(frequency_penalty=-2.0).verify(self.VOCAB_SIZE)
        self._make(frequency_penalty=2.0).verify(self.VOCAB_SIZE)

    # --- presence_penalty ---
    def test_presence_penalty_out_of_range_raises(self):
        """Test that verify() rejects presence_penalty outside [-2, 2]."""
        sp = self._make(presence_penalty=2.5)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # --- repetition_penalty ---
    def test_repetition_penalty_negative_raises(self):
        """Test that verify() rejects negative repetition_penalty (valid range is (0, 2])."""
        sp = self._make(repetition_penalty=-0.1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试repetitionpenaltyzeroraises
    def test_repetition_penalty_zero_raises(self):
        """Test that verify() rejects repetition_penalty=0.

        A value of 0 makes the sampling kernel divide logits by 0, producing
        inf/NaN in the probability tensor and crashing every TP rank with a
        device-side assert.
        """
        sp = self._make(repetition_penalty=0.0)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试repetitionpenaltyabovetworaises
    def test_repetition_penalty_above_two_raises(self):
        """Test that verify() rejects repetition_penalty > 2.0."""
        sp = self._make(repetition_penalty=2.1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试repetitionpenaltyboundarytwovalid
    def test_repetition_penalty_boundary_two_valid(self):
        """Test that the upper boundary value 2.0 is accepted."""
        self._make(repetition_penalty=2.0).verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试repetitionpenaltysmallpositivevalid
    def test_repetition_penalty_small_positive_valid(self):
        """Test that a small positive repetition_penalty (e.g. 1e-3) is accepted."""
        self._make(repetition_penalty=1e-3).verify(self.VOCAB_SIZE)

    # --- min_new_tokens / max_new_tokens ---
    def test_negative_min_new_tokens_raises(self):
        """Test that verify() rejects negative min_new_tokens."""
        sp = self._make(min_new_tokens=-1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试negativemaxnewtokensraises
    def test_negative_max_new_tokens_raises(self):
        """Test that verify() rejects negative max_new_tokens."""
        sp = self._make(max_new_tokens=-1)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试minexceedsmaxnewtokensraises
    def test_min_exceeds_max_new_tokens_raises(self):
        """Test that verify() rejects min_new_tokens > max_new_tokens."""
        sp = self._make(min_new_tokens=100, max_new_tokens=50)
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试minequalsmaxnewtokensvalid
    def test_min_equals_max_new_tokens_valid(self):
        """Test that min_new_tokens == max_new_tokens is accepted."""
        sp = self._make(min_new_tokens=10, max_new_tokens=10)
        sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试maxnewtokensnoneskipsvalidation
    def test_max_new_tokens_none_skips_validation(self):
        """Test that max_new_tokens=None skips the min<=max check."""
        sp = self._make(min_new_tokens=9999, max_new_tokens=None)
        sp.verify(self.VOCAB_SIZE)  # should not raise

    # --- logit_bias ---
    def test_logit_bias_token_exceeds_vocab_raises(self):
        """Test that verify() rejects logit_bias with token_id >= vocab_size."""
        sp = self._make(logit_bias={"99999": 1.0})
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试logitbiasnegativetokenraises
    def test_logit_bias_negative_token_raises(self):
        """Test that verify() rejects logit_bias with negative token_id."""
        sp = self._make(logit_bias={"-1": 1.0})
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试logitbiasvalidtokens
    def test_logit_bias_valid_tokens(self):
        """Test that logit_bias with token_ids within [0, vocab_size) is accepted."""
        sp = self._make(logit_bias={"0": 1.0, "31999": -0.5})
        sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试multiplegrammarsraises
    def test_multiple_grammars_raises(self):
        """Test that verify() rejects setting both json_schema and regex (mutually exclusive)."""
        sp = self._make(json_schema='{"type":"object"}', regex="abc")
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试singlegrammarvalid
    def test_single_grammar_valid(self):
        """Test that setting only one grammar type is accepted."""
        sp = self._make(json_schema='{"type":"object"}')
        sp.verify(self.VOCAB_SIZE)

    # TestSamplingParamsVerify类的测试allthreegrammarssetraises
    def test_all_three_grammars_set_raises(self):
        """Test that verify() rejects setting json_schema, regex, and ebnf together."""
        sp = self._make(json_schema="{}", regex="a", ebnf="rule")
        with self.assertRaises(ValueError):  # 断言抛出异常
            sp.verify(self.VOCAB_SIZE)


# TestSamplingParamsNormalize类
class TestSamplingParamsNormalize(CustomTestCase):

    # TestSamplingParamsNormalize类的测试nonestopstrsbecomesemptylist
    def test_none_stop_strs_becomes_empty_list(self):
        """Test that normalize() converts None stop to empty list with max_len=0."""
        sp = SamplingParams(stop=None)
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_strs, [])  # 断言相等
        self.assertEqual(sp.stop_str_max_len, 0)  # 断言相等

    # TestSamplingParamsNormalize类的测试stringstopstrwrappedinlist
    def test_string_stop_str_wrapped_in_list(self):
        """Test that normalize() wraps a single stop string into a list."""
        sp = SamplingParams(stop="<|end|>")
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_strs, ["<|end|>"])  # 断言相等

    # TestSamplingParamsNormalize类的测试liststopstrsunchanged
    def test_list_stop_strs_unchanged(self):
        """Test that normalize() preserves a list of stop strings as-is."""
        sp = SamplingParams(stop=["stop1", "stop2"])
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_strs, ["stop1", "stop2"])  # 断言相等

    # TestSamplingParamsNormalize类的测试stopstrmaxlenwithouttokenizer
    def test_stop_str_max_len_without_tokenizer(self):
        """Test that without a tokenizer, max_len is the raw string character count."""
        sp = SamplingParams(stop=["ab", "cdef"])
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_str_max_len, 4)  # len("cdef")  # 断言相等

    # TestSamplingParamsNormalize类的测试stopstrmaxlenwithtokenizer
    def test_stop_str_max_len_with_tokenizer(self):
        """Test that with a tokenizer, max_len counts encoded token IDs."""
        tokenizer = MagicMock()
        # "hello" encodes to 2 tokens, "world!!" to 3 tokens
        tokenizer.encode.side_effect = lambda s, add_special_tokens=False: {
            "hello": [101, 102],
            "world!!": [201, 202, 203],
        }[s]
        sp = SamplingParams(stop=["hello", "world!!"])
        sp.normalize(tokenizer=tokenizer)
        self.assertEqual(sp.stop_str_max_len, 3)  # 断言相等

    # TestSamplingParamsNormalize类的测试nonestopregexbecomesemptylist
    def test_none_stop_regex_becomes_empty_list(self):
        """Test that normalize() converts None stop_regex to empty list with max_len=0."""
        sp = SamplingParams(stop_regex=None)
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_regex_strs, [])  # 断言相等
        self.assertEqual(sp.stop_regex_max_len, 0)  # 断言相等

    # TestSamplingParamsNormalize类的测试stringstopregexwrappedinlist
    def test_string_stop_regex_wrapped_in_list(self):
        """Test that normalize() wraps a single stop_regex string into a list."""
        sp = SamplingParams(stop_regex=r"\d+")
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_regex_strs, [r"\d+"])  # 断言相等

    # TestSamplingParamsNormalize类的测试stopregexmaxlencomputed
    def test_stop_regex_max_len_computed(self):
        """Test that bounded regex computes a finite max length."""
        sp = SamplingParams(stop_regex=r"[a-z]{3}")
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_regex_max_len, 3)  # 断言相等


# TestRegexMaxLength类
class TestRegexMaxLength(CustomTestCase):

    # TestRegexMaxLength类的测试literalstring
    def test_literal_string(self):
        """Test that plain string 'abc' gives max length 3."""
        self.assertEqual(get_max_seq_length("abc"), 3)  # 断言相等

    # TestRegexMaxLength类的测试characterclass
    def test_character_class(self):
        """Test that character class '[a-z]' gives max length 1."""
        self.assertEqual(get_max_seq_length("[a-z]"), 1)  # 断言相等

    # TestRegexMaxLength类的测试dotany
    def test_dot_any(self):
        """Test that dot wildcard '.' gives max length 1."""
        self.assertEqual(get_max_seq_length("."), 1)  # 断言相等

    # TestRegexMaxLength类的测试unboundedstar
    def test_unbounded_star(self):
        """Test that 'a*' (zero or more, no upper bound) returns MAX_LEN."""
        result = get_max_seq_length("a*")
        self.assertEqual(result, MAX_LEN)  # 断言相等

    # TestRegexMaxLength类的测试unboundedplus
    def test_unbounded_plus(self):
        """Test that 'a+' (one or more, no upper bound) returns MAX_LEN."""
        result = get_max_seq_length("a+")
        self.assertEqual(result, MAX_LEN)  # 断言相等

    # TestRegexMaxLength类的测试boundedrepeat
    def test_bounded_repeat(self):
        """Test that exact repeat 'a{5}' gives max length 5."""
        self.assertEqual(get_max_seq_length("a{5}"), 5)  # 断言相等

    # TestRegexMaxLength类的测试boundedrangerepeat
    def test_bounded_range_repeat(self):
        """Test that range repeat 'a{2,4}' uses upper bound, giving max length 4."""
        self.assertEqual(get_max_seq_length("a{2,4}"), 4)  # 断言相等

    # TestRegexMaxLength类的测试branchtakesmax
    def test_branch_takes_max(self):
        """Test that alternation 'abc|de' takes the longer branch: max(3, 2) = 3."""
        self.assertEqual(get_max_seq_length("abc|de"), 3)  # 断言相等

    # TestRegexMaxLength类的测试subpatterngroup
    def test_subpattern_group(self):
        """Test that capturing group '(abc)' gives max length 3 from inner content."""
        self.assertEqual(get_max_seq_length("(abc)"), 3)  # 断言相等

    # TestRegexMaxLength类的测试zerowidthassertionsignored
    def test_zero_width_assertions_ignored(self):
        """Test that anchors ^ and $ in '^abc$' add 0, giving max length 3."""
        self.assertEqual(get_max_seq_length("^abc$"), 3)  # 断言相等

    # TestRegexMaxLength类的测试complexpattern
    def test_complex_pattern(self):
        """Test combined pattern '(foo|bar)\\d{2}': branch(3) + repeat(2) = 5."""
        self.assertEqual(get_max_seq_length(r"(foo|bar)\d{2}"), 5)  # 断言相等

    # TestRegexMaxLength类的测试nestedgroups
    def test_nested_groups(self):
        """Test that nested groups '((ab))' correctly recurse to give max length 2."""
        self.assertEqual(get_max_seq_length("((ab))"), 2)  # 断言相等

    # TestRegexMaxLength类的测试questionmarkoptional
    def test_question_mark_optional(self):
        """Test that optional 'a?' (equivalent to a{0,1}) gives max length 1."""
        self.assertEqual(get_max_seq_length("a?"), 1)  # 断言相等

    # TestRegexMaxLength类的测试mixedunboundedandbounded
    def test_mixed_unbounded_and_bounded(self):
        """Test that 'ab+c{3}' gives >= MAX_LEN because b+ is unbounded."""
        result = get_max_seq_length("ab+c{3}")
        self.assertGreaterEqual(result, MAX_LEN)

    # TestRegexMaxLength类的测试emptyregex
    def test_empty_regex(self):
        """Test that empty regex gives max length 0 (no tokens to match)."""
        self.assertEqual(get_max_seq_length(""), 0)  # 断言相等

    # TestRegexMaxLength类的测试lookaheadtriggersunhandledtoken
    def test_lookahead_triggers_unhandled_token(self):
        """Test that lookahead (?=a) hits the unhandled-token fallback (MAX_LEN)."""
        result = get_max_seq_length("(?=a)b")
        self.assertGreaterEqual(result, MAX_LEN)

    # TestRegexMaxLength类的测试lookbehindtriggersunhandledtoken
    def test_lookbehind_triggers_unhandled_token(self):
        """Test that lookbehind (?<=x) hits the unhandled-token fallback (MAX_LEN)."""
        result = get_max_seq_length("(?<=x)y")
        self.assertGreaterEqual(result, MAX_LEN)


if __name__ == "__main__":
    unittest.main()
