# 文件名: __init__.py - 自动基准测试测试
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.auto_benchmark_lib import build_candidates, build_server_candidates


# create_lightweight_tokenizer
def create_lightweight_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab.update({f"tok_{i}": i + 4 for i in range(4096)})

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    return hf_tokenizer


# AutoBenchmarkTestCase类
class AutoBenchmarkTestCase(unittest.TestCase):

    # AutoBenchmarkTestCase类的测试初始化设置
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)
        self.tokenizer = create_lightweight_tokenizer()
        self.tokenizer_dir = self.tmpdir_path / "tok"
        self.tokenizer.save_pretrained(self.tokenizer_dir)

    # AutoBenchmarkTestCase类的测试清理
    def tearDown(self):
        self.tmpdir.cleanup()

    # AutoBenchmarkTestCase类的内部方法_write_autobench_jsonl
    def _write_autobench_jsonl(self) -> str:
        rows = [
            {"prompt": "tok_1 tok_2 tok_3", "output_len": 32},
            {
                "messages": [{"role": "user", "content": "tok_4 tok_5"}],
                "output_len": 24,
                "extra_request_body": {"temperature": 0.0},
            },
            {
                "system": "tok_6",
                "content": ["tok_7 tok_8", "tok_9", "tok_10 tok_11"],
                "output_len": 16,
            },
        ]
        path = self.tmpdir_path / "sample.autobench.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    # AutoBenchmarkTestCase类的内部方法_write_sharegpt_json
    def _write_sharegpt_json(self) -> str:
        rows = [
            {
                "conversations": [
                    {"value": "tok_1 tok_2 tok_3"},
                    {"value": "tok_4 tok_5"},
                ]
            },
            {
                "conversations": [
                    {"value": "tok_6 tok_7"},
                    {"value": "tok_8 tok_9 tok_10"},
                ]
            },
        ]
        path = self.tmpdir_path / "sharegpt.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        return str(path)

    # AutoBenchmarkTestCase类的内部方法_build_candidates_for_capability
    def _build_candidates_for_capability(
        self,
        base_flags,
        search_space,
        *,
        tier,
        max_candidates=None,
        capability=None,
    ):
        with mock.patch(
            "sglang.auto_benchmark_lib.detect_current_cuda_capability",
            return_value=capability,
        ):
            return build_candidates(
                base_flags,
                search_space,
                tier=tier,
                max_candidates=max_candidates,
            )

    # AutoBenchmarkTestCase类的内部方法_build_server_candidates_for_capability
    def _build_server_candidates_for_capability(
        self,
        server_cfg,
        *,
        tier=2,
        max_candidates=None,
        capability=None,
    ):
        with mock.patch(
            "sglang.auto_benchmark_lib.detect_current_cuda_capability",
            return_value=capability,
        ):
            return build_server_candidates(
                server_cfg,
                tier=tier,
                max_candidates=max_candidates,
            )

    @staticmethod

    # AutoBenchmarkTestCase类的内部方法_trial_record
    def _trial_record(
        request_rate,
        *,
        candidate_id=0,
        max_concurrency=None,
        server_flags=None,
        output_throughput=1.0,
        mean_ttft_ms=1.0,
        mean_tpot_ms=1.0,
    ):
        return {
            "stage": "base",
            "candidate_id": candidate_id,
            "requested_qps": request_rate,
            "max_concurrency": max_concurrency,
            "server_flags": dict(server_flags or {"model_path": "/model"}),
            "sla_passed": True,
            "metrics": {
                "output_throughput": output_throughput,
                "mean_ttft_ms": mean_ttft_ms,
                "mean_tpot_ms": mean_tpot_ms,
            },
        }

    # AutoBenchmarkTestCase类的内部方法_make_run_trial_side_effect
    def _make_run_trial_side_effect(
        self,
        calls,
        *,
        output_throughput=1.0,
        mean_ttft_ms=1.0,
        mean_tpot_ms=1.0,
    ):

        # fake_run_trial
        def fake_run_trial(**kwargs):
            calls.append(kwargs["request_rate"])
            return self._trial_record(
                kwargs["request_rate"],
                candidate_id=kwargs["candidate_id"],
                max_concurrency=kwargs["max_concurrency"],
                server_flags=kwargs["server_flags"],
                output_throughput=output_throughput,
                mean_ttft_ms=mean_ttft_ms,
                mean_tpot_ms=mean_tpot_ms,
            )

        return fake_run_trial

    # AutoBenchmarkTestCase类的内部方法_run_candidate_kwargs
    def _run_candidate_kwargs(self, benchmark_cfg, **overrides):
        kwargs = {
            "stage_name": "base",
            "candidate_id": 0,
            "server_cfg": {"host": "127.0.0.1", "port": 30000},
            "benchmark_cfg": benchmark_cfg,
            "dataset_summary": {"num_requests": 1},
            "backend": "sglang-oai",
            "dataset_path": str(self.tmpdir_path / "fake.jsonl"),
            "tokenizer_path": str(self.tokenizer_dir),
            "server_flags": {"model_path": "/model"},
            "output_dir": str(self.tmpdir_path),
        }
        kwargs.update(overrides)
        return kwargs
