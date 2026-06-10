# 文件名: test_whisper_adapter.py - Whisper适配器
"""Unit tests for the Whisper transcription adapter.

Focused on ``WhisperAdapter.parse_fused_output`` — a pure static method
that parses the fused auto-detect output into ``(language, user_visible_text)``.
``visible=None`` means "forced prefix not yet locatable; streaming callers
should keep buffering, non-streaming callers should fall back to a
best-effort scrub".
"""

import re
import unittest
from typing import Any

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
    WHISPER_AUTODETECT_REGEX,
    WHISPER_AUTODETECT_TS_REGEX,
    WHISPER_LANG_TOKEN_CODES,
    WhisperAdapter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# TestWhisperParseFusedOutput类
class TestWhisperParseFusedOutput(CustomTestCase):
    """parse_fused_output: (language, visible) where visible=None means defer."""

    def test_happy_english(self):
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|> Hello world"
        )
        self.assertEqual((lang, visible), ("en", "Hello world"))  # 断言相等

    # TestWhisperParseFusedOutput类的测试happynonenglish
    def test_happy_non_english(self):
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|zh|><|transcribe|><|notimestamps|>你好世界"
        )
        self.assertEqual((lang, visible), ("zh", "你好世界"))

    # TestWhisperParseFusedOutput类的测试missinglanguageprefixdefers
    def test_missing_language_prefix_defers(self):
        # Partial prefix or raw untagged text — streaming callers should
        # keep buffering; non-streaming callers fall back to best-effort.
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output("raw untagged output"), (None, None)
        )
        self.assertEqual(WhisperAdapter.parse_fused_output(""), (None, None))  # 断言相等

    # TestWhisperParseFusedOutput类的测试missingsentineldefers
    def test_missing_sentinel_defers(self):
        # Reviewer's repro: <|zh|> Hi — language tag in but no sentinel.
        self.assertEqual(WhisperAdapter.parse_fused_output("<|zh|> Hi"), (None, None))  # 断言相等

    # TestWhisperParseFusedOutput类的测试truncatedaftertranscribedefers
    def test_truncated_after_transcribe_defers(self):
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output("<|en|><|transcribe|>"), (None, None)
        )

    # TestWhisperParseFusedOutput类的测试unsupportedlanguagecodedefers
    def test_unsupported_language_code_defers(self):
        # FSM regex only allows ISO639_1_SUPPORTED_LANGS. A bypassed-FSM
        # <|xx|> must not leak through as a valid detection.
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output("<|xx|><|transcribe|><|notimestamps|>hi"),
            (None, None),
        )

    # TestWhisperParseFusedOutput类的测试malformedprefixwithouttranscribedefers
    def test_malformed_prefix_without_transcribe_defers(self):
        # The parse must match the exact 3-token forced prefix, not
        # "lang tag + sentinel somewhere". A bypassed-FSM string that
        # skips <|transcribe|> must not parse as a valid detection.
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output("<|en|>junk<|notimestamps|>text"),
            (None, None),
        )
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output("<|en|><|0.00|> text"),
            (None, None),
        )

    # TestWhisperParseFusedOutput类的测试sentinelinbutwhitespaceonlyreturnsemptyvisible
    def test_sentinel_in_but_whitespace_only_returns_empty_visible(self):
        # Prefix arrived at a chunk boundary before the first word. The
        # .strip() collapses to "" so streaming callers see no delta yet;
        # the language is still reported as soon as the sentinel lands.
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output("<|en|><|transcribe|><|notimestamps|>"),
            ("en", ""),
        )
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output("<|en|><|transcribe|><|notimestamps|>  "),
            ("en", ""),
        )

    # TestWhisperParseFusedOutput类的测试trailingendoftextscrubbed
    def test_trailing_endoftext_scrubbed(self):
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|> Hello world<|endoftext|>"
        )
        self.assertEqual((lang, visible), ("en", "Hello world"))  # 断言相等

    # TestWhisperParseFusedOutput类的测试embeddedtimestamptokensscrubbed
    def test_embedded_timestamp_tokens_scrubbed(self):
        # Defensive: in the ts variant Whisper's tokenizer normally
        # decodes <|X.XX|> tokens to "" so they never reach this path,
        # but if a future tokenizer leaks them through they must be
        # scrubbed from the user-visible text. verbose_json segment
        # timing comes from _parse_segments over output_ids on a
        # separate path.
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|0.00|> Hello<|5.00|> world<|10.00|><|endoftext|>",
            ts_variant=True,
        )
        self.assertEqual((lang, visible), ("en", "Hello world"))  # 断言相等

    # TestWhisperParseFusedOutput类的测试tsvariantrealisticdecodedtext
    def test_ts_variant_realistic_decoded_text(self):
        # Real Whisper tokenizer decodes every <|X.XX|> timestamp token
        # (id 50365+) to "" even with skip_special_tokens=False, so for
        # the ts variant the cumulative text is just <|en|><|transcribe|>
        # followed directly by the BPE-decoded transcription. Asserts
        # that the parser handles this shape — without ts_variant=True
        # it would (correctly) defer because <|notimestamps|> is missing.
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|> Hello world<|endoftext|>", ts_variant=True
        )
        self.assertEqual((lang, visible), ("en", "Hello world"))  # 断言相等
        # Same input under non-ts contract correctly defers.
        self.assertEqual(  # 断言相等
            WhisperAdapter.parse_fused_output(
                "<|en|><|transcribe|> Hello world<|endoftext|>"
            ),
            (None, None),
        )

    # TestWhisperParseFusedOutput类的测试visiblegrowsmonotonicallyacrosssnapshots
    def test_visible_grows_monotonically_across_snapshots(self):
        # Streaming property: cumulative text produces cumulative visible.
        snapshots = [
            "<|en|><|transcribe|>",
            "<|en|><|transcribe|><|notimestamps|>",
            "<|en|><|transcribe|><|notimestamps|> Hello",
            "<|en|><|transcribe|><|notimestamps|> Hello world",
            "<|en|><|transcribe|><|notimestamps|> Hello world<|endoftext|>",
        ]
        visibles = [WhisperAdapter.parse_fused_output(s)[1] for s in snapshots]
        # (None, "", "Hello", "Hello world", "Hello world")
        self.assertEqual(visibles, [None, "", "Hello", "Hello world", "Hello world"])  # 断言相等
        # Every non-None entry is a prefix of the next non-None entry.
        real = [v for v in visibles if v is not None]
        for a, b in zip(real, real[1:]):
            self.assertTrue(b.startswith(a), f"monotonicity broken: {a!r} -> {b!r}")  # 断言为真


# TestWhisperLangTokenCoverage类
class TestWhisperLangTokenCoverage(CustomTestCase):
    """The FSM regex must cover every Whisper language token, not just the
    narrower ISO639_1_SUPPORTED_LANGS set used for input validation."""

    # TestWhisperLangTokenCoverage类的测试threelettercodesparse
    def test_three_letter_codes_parse(self):
        # yue (Cantonese, v3), haw (Hawaiian), jw (Javanese, two-letter but
        # missing from ISO639_1_SUPPORTED_LANGS) — reviewer's flagged examples.
        for code in ("yue", "haw", "jw"):
            with self.subTest(lang=code):
                lang, visible = WhisperAdapter.parse_fused_output(
                    f"<|{code}|><|transcribe|><|notimestamps|> Hi"
                )
                self.assertEqual(lang, code)  # 断言相等
                self.assertEqual(visible, "Hi")  # 断言相等

    # TestWhisperLangTokenCoverage类的测试knownwhisperlangsinallowlist
    def test_known_whisper_langs_in_allowlist(self):
        # Spot-check: codes the reviewer named + common 3-letter tokens.
        for code in ("yue", "haw", "jw", "su", "ba", "tt", "ln", "lo"):
            self.assertIn(code, WHISPER_LANG_TOKEN_CODES)  # 断言包含

    # TestWhisperLangTokenCoverage类的测试fsmregexincludesthreeletteralternatives
    def test_fsm_regex_includes_three_letter_alternatives(self):
        # Defensive: the regex alternation must spell out the 3-letter codes
        # so xgrammar's FSM admits the <|yue|> / <|haw|> single-token path.
        for code in ("yue", "haw"):
            self.assertIn(re.escape(code), WHISPER_AUTODETECT_REGEX)  # 断言包含
            self.assertIn(re.escape(code), WHISPER_AUTODETECT_TS_REGEX)  # 断言包含

    # TestWhisperLangTokenCoverage类的测试autodetectcodesroundtripthroughinputvalidator
    def test_autodetect_codes_round_trip_through_input_validator(self):
        # A code returned by fused autodetect must be accepted as
        # ``language=`` on a follow-up request. Before the fix,
        # ``normalize_language_to_code("yue")`` raised ValueError even
        # though verbose_json could report ``"yue"`` from the same server.
        from sglang.srt.multimodal.processors.whisper import (
            normalize_language_to_code,
        )

        for code in ("yue", "haw", "jw", "ba", "su", "tt"):
            with self.subTest(lang=code):
                self.assertEqual(normalize_language_to_code(code), code)  # 断言相等

    # TestWhisperLangTokenCoverage类的测试unknownlanguagetokenidraisescleanerror
    def test_unknown_language_token_id_raises_clean_error(self):
        # Some Whisper codes (yue, v3-only) aren't in older checkpoints'
        # vocabs. The explicit-language path must raise a clean ValueError
        # in that case instead of silently feeding the unk token into the
        # decoder and producing garbage. Mocks cover both "returns None"
        # and "returns unk_token_id" tokenizer behaviors.
        from unittest.mock import Mock

        from sglang.srt.multimodal.processors.whisper import WhisperProcessor

        proc = WhisperProcessor.__new__(WhisperProcessor)
        # Tokenizer where <|yue|> is not in the vocab → returns unk_id.
        tok = Mock()
        tok.convert_tokens_to_ids = Mock(return_value=100)  # arbitrary unk
        tok.unk_token_id = 100
        proc._tokenizer = tok
        with self.assertRaises(ValueError) as ctx:  # 断言抛出异常
            proc._get_language_token_id("yue")
        self.assertIn("yue", str(ctx.exception))  # 断言包含

        # Known code (English) on the same tokenizer still works.
        tok.convert_tokens_to_ids = Mock(return_value=50259)  # <|en|>
        self.assertEqual(proc._get_language_token_id("en"), 50259)  # 断言相等

        # Some tokenizers return None for unknown tokens instead of unk_id.
        tok2 = Mock()
        tok2.convert_tokens_to_ids = Mock(return_value=None)
        tok2.unk_token_id = 100
        proc._tokenizer = tok2
        with self.assertRaises(ValueError):  # 断言抛出异常
            proc._get_language_token_id("yue")


# TestWhisperStripSpecialTokens类
class TestWhisperStripSpecialTokens(CustomTestCase):
    """Fallback scrub used when parse_fused_output defers."""

    def test_strips_all_whisper_specials(self):
        self.assertEqual(  # 断言相等
            WhisperAdapter.strip_special_tokens(
                "<|en|><|transcribe|><|0.00|>hi<|5.00|>world<|endoftext|>"
            ),
            "hiworld",
        )

    # TestWhisperStripSpecialTokens类的测试identityonplaintext
    def test_identity_on_plain_text(self):
        self.assertEqual(  # 断言相等
            WhisperAdapter.strip_special_tokens("plain text"), "plain text"
        )
        self.assertEqual(WhisperAdapter.strip_special_tokens(""), "")  # 断言相等

    # TestWhisperStripSpecialTokens类的测试preservesspokenanglebracketsequences
    def test_preserves_spoken_angle_bracket_sequences(self):
        # The scrub must only remove actual Whisper special-token literals
        # (lang / control / <|X.XX|> timestamps), not arbitrary ``<|...|>``
        # patterns that can appear in transcribed speech (someone reading a
        # token name aloud, an AI-safety demo, code dictation, etc.).
        self.assertEqual(  # 断言相等
            WhisperAdapter.strip_special_tokens("the token <|foo|> is unused"),
            "the token <|foo|> is unused",
        )
        # Real specials still scrubbed even when interleaved with bogus ones.
        self.assertEqual(  # 断言相等
            WhisperAdapter.strip_special_tokens(
                "<|en|>hello <|foo|> world<|endoftext|>"
            ),
            "hello <|foo|> world",
        )

    # TestWhisperStripSpecialTokens类的测试parsepreservesspokenanglebracketsequences
    def test_parse_preserves_spoken_angle_bracket_sequences(self):
        # Same for the per-chunk scrub inside parse_fused_output.
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|> look at <|foo|><|endoftext|>"
        )
        self.assertEqual((lang, visible), ("en", "look at <|foo|>"))  # 断言相等


# TestWhisperBuildFusedAutodetectParams类
class TestWhisperBuildFusedAutodetectParams(CustomTestCase):
    """build_fused_autodetect_params picks the right regex + propagates ts param."""

    def _request(self, **kwargs: Any) -> TranscriptionRequest:
        base: dict[str, Any] = dict(model="whisper", temperature=0.0)
        base.update(kwargs)
        return TranscriptionRequest(**base)

    # TestWhisperBuildFusedAutodetectParams类的测试notimestampsusesnotimestampsregex
    def test_no_timestamps_uses_notimestamps_regex(self):
        params = WhisperAdapter().build_fused_autodetect_params(self._request())
        self.assertEqual(params["regex"], WHISPER_AUTODETECT_REGEX)  # 断言相等
        self.assertNotIn("timestamp_granularities", params)  # 断言不包含

    # TestWhisperBuildFusedAutodetectParams类的测试timestampsusestsregexandpropagatesgranularities
    def test_timestamps_uses_ts_regex_and_propagates_granularities(self):
        req = self._request(timestamp_granularities=["segment"])
        params = WhisperAdapter().build_fused_autodetect_params(req)
        self.assertEqual(params["regex"], WHISPER_AUTODETECT_TS_REGEX)  # 断言相等
        self.assertEqual(params["timestamp_granularities"], ["segment"])  # 断言相等

    # TestWhisperBuildFusedAutodetectParams类的测试emptytimestampslistusesnotimestampsregex
    def test_empty_timestamps_list_uses_notimestamps_regex(self):
        # Empty list is falsy — treat as "no timestamps requested".
        req = self._request(timestamp_granularities=[])
        params = WhisperAdapter().build_fused_autodetect_params(req)
        self.assertEqual(params["regex"], WHISPER_AUTODETECT_REGEX)  # 断言相等
        self.assertNotIn("timestamp_granularities", params)  # 断言不包含

    # TestWhisperBuildFusedAutodetectParams类的测试spacesbetweenspecialtokensisfalse
    def test_spaces_between_special_tokens_is_false(self):
        # parse_fused_output assumes a zero-space forced prefix. Slow
        # Whisper tokenizers otherwise insert a space between adjacent
        # special tokens, which would silently break the parse path.
        for req in (
            self._request(),
            self._request(timestamp_granularities=["segment"]),
        ):
            params = WhisperAdapter().build_fused_autodetect_params(req)
            self.assertIs(params["spaces_between_special_tokens"], False)  # 断言是同一对象

    # TestWhisperBuildFusedAutodetectParams类的测试fusedparamssurvivesamplingparamsconstruction
    def test_fused_params_survive_sampling_params_construction(self):
        # Regression: the multimodal processor's fused branch used to skip
        # popping `timestamp_granularities`, leaking the key into
        # SamplingParams(**kwargs) → TypeError on any language=None +
        # timestamp_granularities request. Mirrors what the processor does
        # before constructing SamplingParams.
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = self._request(timestamp_granularities=["segment"])
        params = WhisperAdapter().build_fused_autodetect_params(req)
        # Fields the processor pops before SamplingParams(**kwargs).
        params.pop("_detect_language", None)
        params.pop("timestamp_granularities", None)
        SamplingParams(**params)


if __name__ == "__main__":
    unittest.main()
