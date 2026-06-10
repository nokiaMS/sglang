# 文件名: test_field_validators.py - 字段验证器
import unittest

from sglang.srt.utils.field_validators import (
    validate_list_i64_1d,
    validate_optional_list_i64_1d_2d,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# TestValidateListI64_1d类
class TestValidateListI64_1d(CustomTestCase):
    """`validate_list_i64_1d` rejects anything that array('q', v) can't accept."""

    def test_accept_int_list(self):
        v = [1, 2, 3]
        self.assertIs(validate_list_i64_1d(v), v)  # 断言是同一对象

    # TestValidateListI64_1d类的测试acceptemptylist
    def test_accept_empty_list(self):
        v = []
        self.assertIs(validate_list_i64_1d(v), v)  # 断言是同一对象

    # TestValidateListI64_1d类的测试acceptint64boundaries
    def test_accept_int64_boundaries(self):
        for v in ([-(2**63)], [2**63 - 1], [0]):
            self.assertIs(validate_list_i64_1d(v), v)  # 断言是同一对象

    # TestValidateListI64_1d类的测试rejectnone
    def test_reject_none(self):
        with self.assertRaisesRegex(ValueError, "must not be None"):
            validate_list_i64_1d(None)

    # TestValidateListI64_1d类的测试rejectnonlist
    def test_reject_non_list(self):
        for v in ((1, 2, 3), "abc", {1: 2}, 42, 3.14):
            with self.subTest(v=v):
                with self.assertRaisesRegex(ValueError, "must be list"):
                    validate_list_i64_1d(v)

    # TestValidateListI64_1d类的测试rejectnonintfirstelement
    def test_reject_non_int_first_element(self):
        for v in ([1.5, 2, 3], ["a", "b"], [None, 1]):
            with self.subTest(v=v):
                with self.assertRaisesRegex(ValueError, "elements must be int"):
                    validate_list_i64_1d(v)

    # TestValidateListI64_1d类的测试rejectoverflowint64
    def test_reject_overflow_int64(self):
        # 2**63 overflows signed int64.
        with self.assertRaisesRegex(ValueError, "non-int64 element"):
            validate_list_i64_1d([0, 2**63])

    # TestValidateListI64_1d类的测试rejectnonintlaterelement
    def test_reject_non_int_later_element(self):
        # First-element fast path passes but C loop rejects later float.
        with self.assertRaisesRegex(ValueError, "non-int64 element"):
            validate_list_i64_1d([1, 2, 3.5])


# TestValidateOptionalListI64_1d_2d类
class TestValidateOptionalListI64_1d_2d(CustomTestCase):
    """`validate_optional_list_i64_1d_2d` accepts None | [] | list[int] | list[list[int]]."""

    def test_accept_none(self):
        self.assertIsNone(validate_optional_list_i64_1d_2d(None))  # 断言为None

    # TestValidateOptionalListI64_1d_2d类的测试acceptemptylist
    def test_accept_empty_list(self):
        v = []
        self.assertIs(validate_optional_list_i64_1d_2d(v), v)  # 断言是同一对象

    # TestValidateOptionalListI64_1d_2d类的测试accept1dintlist
    def test_accept_1d_int_list(self):
        v = [1, 2, 3]
        self.assertIs(validate_optional_list_i64_1d_2d(v), v)  # 断言是同一对象

    # TestValidateOptionalListI64_1d_2d类的测试accept2dintlist
    def test_accept_2d_int_list(self):
        v = [[1, 2], [3, 4, 5]]
        self.assertIs(validate_optional_list_i64_1d_2d(v), v)  # 断言是同一对象

    # TestValidateOptionalListI64_1d_2d类的测试accept2dwithemptyrow
    def test_accept_2d_with_empty_row(self):
        v = [[], [1, 2]]
        self.assertIs(validate_optional_list_i64_1d_2d(v), v)  # 断言是同一对象

    # TestValidateOptionalListI64_1d_2d类的测试rejectnonlisttoplevel
    def test_reject_non_list_top_level(self):
        for v in ((1, 2), "abc", 42, 3.14, {1: 2}):
            with self.subTest(v=v):
                with self.assertRaisesRegex(ValueError, "must be list or null"):
                    validate_optional_list_i64_1d_2d(v)

    # TestValidateOptionalListI64_1d_2d类的测试rejectmixedfirstelementtype
    def test_reject_mixed_first_element_type(self):
        with self.assertRaisesRegex(ValueError, "elements must be int or list"):
            validate_optional_list_i64_1d_2d([1.5, 2.5])

    # TestValidateOptionalListI64_1d_2d类的测试rejectoverflowin1d
    def test_reject_overflow_in_1d(self):
        with self.assertRaisesRegex(ValueError, "non-int64 element"):
            validate_optional_list_i64_1d_2d([0, 2**63])

    # TestValidateOptionalListI64_1d_2d类的测试rejectbadrowin2dreportsindex
    def test_reject_bad_row_in_2d_reports_index(self):
        with self.assertRaisesRegex(ValueError, "row 1:"):
            validate_optional_list_i64_1d_2d([[1, 2], [3, "x"]])

    # TestValidateOptionalListI64_1d_2d类的测试rejectoverflowin2dreportsindex
    def test_reject_overflow_in_2d_reports_index(self):
        with self.assertRaisesRegex(ValueError, "row 0:.*non-int64"):
            validate_optional_list_i64_1d_2d([[2**63], [1, 2]])


if __name__ == "__main__":
    unittest.main()
