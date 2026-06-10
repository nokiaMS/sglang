# 文件名: test_spec_ngram_fa3.py - NGram推测解码FA3测试 - 使用FA3注意力后端验证NGram推测解码的GSM8K准确性
"""NGRAM speculative-decoding test, FA3 attention-backend variant.

Backend: `--attention-backend fa3`.
Not registered in any CI suite -- runnable manually only.
"""

import unittest

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase


class TestNgramSpeculativeDecodingBase(NgramServerBase, GSM8KMixin):
    attention_backend = "fa3"


if __name__ == "__main__":
    unittest.main()
