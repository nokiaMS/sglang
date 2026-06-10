# 文件名: test_gdn_prefill_cutedsl.py - GDN预填充CuteDSL测试
"""Correctness test for the SM100 CuTe DSL GDN prefill kernel.

Ported from vLLM PR https://github.com/vllm-project/vllm/pull/43273.
Validates ``chunk_gated_delta_rule_cutedsl`` against the
``fused_recurrent_gated_delta_rule`` Triton reference.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

# CuteDSL prefill kernel only exists on Blackwell. Single-GPU kernel-unit
# suite is the right slot (matches existing jit_kernel test_*.py pattern).
register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-b200")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
    pytest.skip(
        "GDN CuteDSL prefill requires CUDA SM10x (Blackwell).",
        allow_module_level=True,
    )

from sglang.srt.layers.attention.fla.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.layers.attention.fla.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.linear.kernels.gdn_blackwell import (  # noqa: E402
    chunk_gated_delta_rule_cutedsl,
    prepare_metadata_cutedsl,
)


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
# 测试gdnchunkcutedslcorrectness
def test_gdn_chunk_cutedsl_correctness(num_seqs: int, state_dtype: torch.dtype):
    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to(device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())  # 获取标量值

    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128
    dtype = torch.bfloat16

    q = torch.randn(
        1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
    q = F.normalize(q.float(), p=2, dim=-1).to(dtype)  # 转换为单精度
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)  # 转换为单精度
    a = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    b = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)

    # Match upstream FLA GatedDeltaNet synthetic init.
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)  # 转换为单精度
    )
    beta = torch.sigmoid(b.float())  # 转换为单精度
    initial_state = (
        torch.randn(
            num_seqs,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )

    # Metadata kernel matches the FLA reference helpers.
    chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, total_tokens)
    torch.cuda.synchronize()  # 同步CUDA操作

    expected_indices = prepare_chunk_indices(cu_seqlens, 64)
    expected_offsets = prepare_chunk_offsets(cu_seqlens, 64)
    total_chunks = int(expected_offsets[-1].item())  # 获取标量值

    torch.testing.assert_close(chunk_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(chunk_indices[:total_chunks], expected_indices)

    # Reference: token-by-token recurrent kernel returns (o, final_state).
    # Recurrent path needs float32 state, so cast initial_state for the call.
    ref_o, ref_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.to(torch.float32),
        output_final_state=True,
        cu_seqlens=cu_seqlens.to(torch.int64),
        use_qk_l2norm_in_kernel=False,
    )

    actual_core_attn_out = torch.empty(
        total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
    actual_o, actual_state = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        core_attn_out=actual_core_attn_out,
    )
    torch.cuda.synchronize()  # 同步CUDA操作

    o_error = (actual_o.float() - ref_o.float()).abs()  # 转换为单精度
    state_error = (
        actual_state.float() - ref_state.to(actual_state.dtype).float()  # 转换为单精度
    ).abs()
    assert o_error.max().item() < 2e-3  # 获取标量值
    assert o_error.mean().item() < 6e-5  # 获取标量值
    assert state_error.max().item() < 2e-2  # 获取标量值
    assert state_error.mean().item() < 6e-4  # 获取标量值
    core_attn_out_error = (
        actual_core_attn_out.float() - actual_o.squeeze(0).float()  # 转换为单精度
    ).abs()
    assert core_attn_out_error.max().item() == 0  # 获取标量值

    no_buffer_o, no_buffer_state = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    torch.cuda.synchronize()  # 同步CUDA操作

    no_buffer_o_error = (no_buffer_o.float() - ref_o.float()).abs()  # 转换为单精度
    no_buffer_state_error = (
        no_buffer_state.float() - ref_state.to(no_buffer_state.dtype).float()  # 转换为单精度
    ).abs()
    buffer_o_error = (no_buffer_o.float() - actual_o.float()).abs()  # 转换为单精度
    buffer_state_error = (
        no_buffer_state.float() - actual_state.to(no_buffer_state.dtype).float()  # 转换为单精度
    ).abs()
    assert no_buffer_o_error.max().item() < 2e-3  # 获取标量值
    assert no_buffer_o_error.mean().item() < 6e-5  # 获取标量值
    assert no_buffer_state_error.max().item() < 2e-2  # 获取标量值
    assert no_buffer_state_error.mean().item() < 6e-4  # 获取标量值
    assert buffer_o_error.max().item() == 0  # 获取标量值
    assert buffer_state_error.max().item() == 0  # 获取标量值


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
