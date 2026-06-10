# 文件名: test_tensor_naming.py - 张量命名测试
import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.dims_spec import (
    DimSpec,
    apply_dim_names,
    find_dim_index,
    get_dim_names,
    parse_dims,
    resolve_dim_by_name,
    without_dim_names,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=1, suite="base-b-test-cpu")


class TestFindDimIndex:
    # 测试found
    def test_found(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d").dims
        assert find_dim_index(specs, "s") == 1

    # 测试notfound
    def test_not_found(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d").dims
        assert find_dim_index(specs, "t") is None

    # 测试firstdim
    def test_first_dim(self) -> None:
        specs: list[DimSpec] = parse_dims("t h d").dims
        assert find_dim_index(specs, "t") == 0

    # 测试lastdim
    def test_last_dim(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d").dims
        assert find_dim_index(specs, "d") == 3

    # 测试withmodifiers
    def test_with_modifiers(self) -> None:
        specs: list[DimSpec] = parse_dims("b s[cp:zigzag] h[tp] d").dims
        assert find_dim_index(specs, "h") == 2

    # 测试emptylist
    def test_empty_list(self) -> None:
        assert find_dim_index([], "t") is None


class TestResolveDimByName:
    # 测试resolvefound
    def test_resolve_found(self) -> None:
        tensor: torch.Tensor = apply_dim_names(torch.randn(2, 3, 4), ["b", "s", "h"])
        assert resolve_dim_by_name(tensor, "b") == 0
        assert resolve_dim_by_name(tensor, "s") == 1
        assert resolve_dim_by_name(tensor, "h") == 2

    # 测试resolvenotfoundraises
    def test_resolve_not_found_raises(self) -> None:
        tensor: torch.Tensor = apply_dim_names(torch.randn(2, 3), ["b", "s"])
        with pytest.raises(ValueError, match="not in tensor names"):
            resolve_dim_by_name(tensor, "h")

    # 测试resolveunnamedraises
    def test_resolve_unnamed_raises(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3)
        with pytest.raises(ValueError, match="no names"):
            resolve_dim_by_name(tensor, "b")


class TestApplyDimNames:
    # 测试apply
    def test_apply(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3, 4)
        named: torch.Tensor = apply_dim_names(tensor, ["b", "s", "h"])
        assert get_dim_names(named) == ("b", "s", "h")
        assert named.shape == (2, 3, 4)

    # 测试applypreservesdata
    def test_apply_preserves_data(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3)
        named: torch.Tensor = apply_dim_names(tensor, ["x", "y"])
        assert torch.equal(without_dim_names(named), tensor)

    # 测试ndimmismatchgivesclearerror
    def test_ndim_mismatch_gives_clear_error(self) -> None:
        tensor: torch.Tensor = torch.randn(10, 1, 128)
        with pytest.raises(
            ValueError,
            match=r"dims metadata mismatch.*3 dims.*shape \[10, 1, 128\].*2 names \['t', 'num_experts'\].*fix the dims string",
        ):
            apply_dim_names(tensor, ["t", "num_experts"])


class TestStripDimNames:
    # 测试strip
    def test_strip(self) -> None:
        tensor: torch.Tensor = apply_dim_names(torch.randn(2, 3), ["a", "b"])
        stripped: torch.Tensor = without_dim_names(tensor)
        assert get_dim_names(stripped) == (None, None)

    # 测试stripalreadyunnamed
    def test_strip_already_unnamed(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3)
        stripped: torch.Tensor = without_dim_names(tensor)
        assert get_dim_names(stripped) == (None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
