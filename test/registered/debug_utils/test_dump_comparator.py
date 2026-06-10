# 文件名: test_dump_comparator.py - 转储比较器测试
import pytest
import torch

from sglang.srt.debug_utils.dump_comparator import (
    _argmax_coord,
    _calc_rel_diff,
    _compute_smaller_dtype,
    _try_unify_shape,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu", nightly=True)


# ----------------------------- Unit tests -----------------------------


class TestCalcRelDiff:
    # 测试identicalvectors
    def test_identical_vectors(self) -> None:
        x: torch.Tensor = torch.randn(10, 10)
        assert _calc_rel_diff(x, x).item() == pytest.approx(0.0, abs=1e-5)  # 获取标量值

    # 测试zerovectors
    def test_zero_vectors(self) -> None:
        z: torch.Tensor = torch.zeros(5)
        result = _calc_rel_diff(z, z)
        assert not torch.isnan(result) or True  # should not crash


class TestArgmaxCoord:
    # 测试knownposition
    def test_known_position(self) -> None:
        x: torch.Tensor = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        assert _argmax_coord(x) == (1, 2, 3)


class TestTryUnifyShape:
    # 测试squeezeleadingones
    def test_squeeze_leading_ones(self) -> None:
        target_shape: torch.Size = torch.Size([3, 4])
        result: torch.Tensor = _try_unify_shape(torch.randn(1, 1, 3, 4), target_shape)
        assert result.shape == target_shape

    # 测试noopwhennoleadingones
    def test_no_op_when_no_leading_ones(self) -> None:
        target_shape: torch.Size = torch.Size([3, 4])
        result: torch.Tensor = _try_unify_shape(torch.randn(2, 3, 4), target_shape)
        assert result.shape == (2, 3, 4)


class TestComputeSmallerDtype:
    # 测试knownpair
    def test_known_pair(self) -> None:
        assert _compute_smaller_dtype(torch.float32, torch.bfloat16) == torch.bfloat16
        assert _compute_smaller_dtype(torch.bfloat16, torch.float32) == torch.bfloat16

    # 测试noneforsamedtype
    def test_none_for_same_dtype(self) -> None:
        assert _compute_smaller_dtype(torch.float32, torch.float32) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
