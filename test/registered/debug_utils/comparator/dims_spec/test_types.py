# 文件名: test_types.py - 类型定义测试
import sys

import pytest

from sglang.srt.debug_utils.comparator.dims_spec import (
    BATCH_DIM_NAME,
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=1, suite="base-b-test-cpu")


class TestDimConstants:
    # 测试tokendimname
    def test_token_dim_name(self) -> None:
        assert TOKEN_DIM_NAME == "t"

    # 测试batchdimname
    def test_batch_dim_name(self) -> None:
        assert BATCH_DIM_NAME == "b"

    # 测试seqdimname
    def test_seq_dim_name(self) -> None:
        assert SEQ_DIM_NAME == "s"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
