# 文件名: test_utils.py - 工具函数
"""
Unit tests for sglang.srt.constrained.utils.

Test Coverage:
- is_legacy_structural_tag: legacy format detection, new format detection,
  missing fields, edge cases with assertion errors.

Usage:
    python -m pytest test_utils.py -v
"""

import unittest

from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


# TestIsLegacyStructuralTag类
class TestIsLegacyStructuralTag(unittest.TestCase):
    """Test is_legacy_structural_tag function."""

    def test_legacy_format_returns_true(self):
        obj = {
            "structures": [{"begin": "<tool>", "end": "</tool>"}],
            "triggers": ["<tool>"],
        }
        self.assertTrue(is_legacy_structural_tag(obj))  # 断言为真

    # TestIsLegacyStructuralTag类的测试legacyformatemptylists
    def test_legacy_format_empty_lists(self):
        obj = {"structures": [], "triggers": []}
        self.assertTrue(is_legacy_structural_tag(obj))  # 断言为真

    # TestIsLegacyStructuralTag类的测试newformatreturnsfalse
    def test_new_format_returns_false(self):
        obj = {"format": {"type": "json_schema", "schema": {}}}
        self.assertFalse(is_legacy_structural_tag(obj))  # 断言为假

    # TestIsLegacyStructuralTag类的测试newformatemptyformat
    def test_new_format_empty_format(self):
        obj = {"format": {}}
        self.assertFalse(is_legacy_structural_tag(obj))  # 断言为假

    # TestIsLegacyStructuralTag类的测试legacymissingtriggersraises
    def test_legacy_missing_triggers_raises(self):
        """Legacy format requires both 'structures' and 'triggers'."""
        obj = {"structures": [{"begin": "<tool>", "end": "</tool>"}]}
        with self.assertRaises(AssertionError):  # 断言抛出异常
            is_legacy_structural_tag(obj)

    # TestIsLegacyStructuralTag类的测试newformatmissingformatraises
    def test_new_format_missing_format_raises(self):
        """New format (no 'structures') requires 'format' key."""
        obj = {"other_key": "value"}
        with self.assertRaises(AssertionError):  # 断言抛出异常
            is_legacy_structural_tag(obj)

    # TestIsLegacyStructuralTag类的测试emptydictraises
    def test_empty_dict_raises(self):
        with self.assertRaises(AssertionError):  # 断言抛出异常
            is_legacy_structural_tag({})

    # TestIsLegacyStructuralTag类的测试structuresnoneusesnewformatpath
    def test_structures_none_uses_new_format_path(self):
        """Explicitly None 'structures' should fall to new format check."""
        obj = {"structures": None, "format": {"type": "json_schema"}}
        self.assertFalse(is_legacy_structural_tag(obj))  # 断言为假

    # TestIsLegacyStructuralTag类的测试bothkeyspresentlegacywins
    def test_both_keys_present_legacy_wins(self):
        """When both 'structures' and 'format' present, 'structures' takes priority."""
        obj = {
            "structures": [{"begin": "<tool>"}],
            "triggers": ["<tool>"],
            "format": {"type": "json_schema"},
        }
        self.assertTrue(is_legacy_structural_tag(obj))  # 断言为真


if __name__ == "__main__":
    unittest.main()
