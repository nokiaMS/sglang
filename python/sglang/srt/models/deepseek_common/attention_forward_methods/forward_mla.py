# DeepSeek MLA（多潜在注意力）前向计算核心实现
# 本文件实现了 DeepSeek V2/V3 模型中 MLA 的前向计算逻辑，包括：
# - forward_absorb_prepare: 准备阶段，计算 Q/K/V 的潜在表示，应用 LayerNorm、
#   RoPE、吸收 kv_b_proj 权重到 Q 中等操作
# - forward_absorb_core: 核心注意力计算阶段，执行吸收式 MLA 或标准 MLA 的
#   注意力计算，包括多种平台（CUDA/HIP/CPU/NPU/MUSA）的 BMM 实现
# 本文件支持多种硬件后端和量化格式（FP8、MXFP4、AWQ 等）。

# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.piecewise_cuda_graph_manager import is_in_piecewise_cuda_graph
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.lora.deepseek_mla_correction import (
    apply_q_correction as apply_kv_b_lora_q_correction,
)
from sglang.srt.lora.deepseek_mla_correction import (
    apply_v_correction as apply_kv_b_lora_v_correction,
)
from sglang.srt.lora.deepseek_mla_correction import (
    is_kv_b_lora_active,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_cpu,
    _is_cublas_ge_129,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _is_musa,
    _use_aiter,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.state_capturer.indexer_topk import (
    maybe_capture_indexer_topk,
)
from sglang.srt.utils import BumpAllocator

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _is_cuda:
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

    from sglang.srt.utils.custom_op import register_custom_op

    # TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
    # Wrap bmm_fp8 as a custom op so torch.compile does not trace into
    # torch.cuda.current_blas_handle() (which returns a non-Tensor).
    # 将 bmm_fp8 包装为自定义算子，避免 torch.compile 追踪到
    # torch.cuda.current_blas_handle()（返回非 Tensor 对象）
    @register_custom_op(mutates_args=["out"])
    def _bmm_fp8_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

    # bmm_fp8 的包装函数，支持自动分配输出张量
    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)
        return out


if _use_aiter:
    # aiter ROCm/aiter#2958 renamed the public `fused_qk_rmsnorm` in
    # `aiter.ops.fused_qk_norm_rope_cache_quant` to a private `_fused_qk_rmsnorm`
    # and introduced a unified entry point in `aiter.ops.fused_qk_rmsnorm_group_quant`
    # with a different (in-place, kwarg-only, no-return) signature. Probe for the
    # new symbol first so SGLang works with both pre- and post-#2958 aiter without
    # requiring the docker pin to be bumped atomically.
    # AITER 库接口变更兼容处理：优先尝试新版 API，回退到旧版 API
    try:
        from aiter.ops.enum import QuantType as _AiterQuantType
        from aiter.ops.fused_qk_rmsnorm_group_quant import (
            fused_qk_rmsnorm as _aiter_fused_qk_rmsnorm_unified,
        )

        # 新版 AITER 的融合 QK RMSNorm 接口（签名不同，需要适配）
        def fused_qk_rmsnorm_bf16(q, q_weight, q_eps, k, k_weight, k_eps):
            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            _aiter_fused_qk_rmsnorm_unified(
                q_out_quantized=q_out,
                k_out=k_out,
                q=q,
                q_weight=q_weight,
                q_epsilon=q_eps,
                k=k,
                k_weight=k_weight,
                k_epsilon=k_eps,
                quant_type=_AiterQuantType.No,
            )
            return q_out, k_out

    except ImportError:
        # 旧版 AITER 的融合 QK RMSNorm 接口
        from aiter.ops.fused_qk_norm_rope_cache_quant import (
            fused_qk_rmsnorm as fused_qk_rmsnorm_bf16,
        )

    # AITER FP8 分组量化批量 GEMM 内核
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
if _use_aiter_gfx95:
    # gfx95 平台的融合 FP8 量化内核
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    # gfx95 平台的 MXFP4 量化工具
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    # gfx95 平台的融合 QK RoPE + 拼接 + 缓存 MLA 内核
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla


# DeepSeek MLA 前向计算混入类
# 提供 MLA 注意力的准备阶段和核心计算阶段，可被 DeepseekV2AttentionMLA 继承使用
class DeepseekMLAForwardMixin:
    # 初始化 MLA 前向计算相关参数
    def init_mla_forward(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )

    # MLA 前向计算准备阶段
    # 计算 Q/K/V 的潜在表示，应用 LayerNorm、RoPE、吸收 kv_b_proj 权重到 Q 中，
    # 并处理上下文并行（CP）和 DSA indexer 等特殊情况
    def forward_absorb_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            # 从 TP 上下文中获取 QKV 潜在向量，并拆分为 q 和 latent_cache
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            # k_nope 为潜在缓存中的非 RoPE 部分
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            # Q 和 K 的 LayerNorm 重叠执行（使用替代流）
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                # gfx95 平台 MXFP4 量化路径：融合 RMSNorm + MXFP4 量化
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    # gfx95 平台 FP8 量化路径：融合 RMSNorm + FP8 分组量化
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_dsa:
                            # DSA 模式需要保留未量化的 q_lora 给 indexer 使用
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    # AITER 平台融合 QK RMSNorm 路径
                    elif _use_aiter:
                        q, k_nope = fused_qk_rmsnorm_bf16(
                            q,
                            self.q_a_layernorm.weight,
                            self.q_a_layernorm.variance_epsilon,
                            k_nope,
                            self.kv_a_layernorm.weight,
                            self.kv_a_layernorm.variance_epsilon,
                        )
                    # 默认路径：分别对 Q 和 K 执行 LayerNorm
                    else:
                        q = self.q_a_layernorm(q)
                        k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            # DSA 模式下 indexer 需要 q_lora
            if self.use_dsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            # 解码阶段重叠执行 q_b_proj 投影和 indexer topk 选择
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                # 执行 indexer topk 选择或复用上一层的 topk 结果
                if not self.skip_topk or prev_topk_indices is None:
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    # skip_topk reuses prev layer's indices; mirror into this
                    # layer's slot so the captured buffer matches what's used.
                    topk_indices = maybe_capture_indexer_topk(
                        self.layer_id, prev_topk_indices
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                # 非重叠路径：顺序执行 q_b_proj 和 indexer
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    if not self.skip_topk or prev_topk_indices is None:
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = maybe_capture_indexer_topk(
                            self.layer_id, prev_topk_indices
                        )
        else:
            # 无 LoRA 路径：直接使用 q_proj 和 kv_a_proj_with_mqa
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        # 将 Q 拆分为非 RoPE 部分（q_nope）和 RoPE 部分（q_pe）
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # k_pe 为潜在缓存中的 RoPE 部分
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        # ============ 吸收 kv_b_proj 权重到 Q 的非 RoPE 部分 ============
        # 计算 q_nope @ w_kc，即 Q 的非 RoPE 部分与 KV 投影权重中的 K 部分相乘

        # DeepGEMM 分块 GEMM 路径（FP8 + 分块缩放）
        if self.use_deep_gemm_bmm:
            (
                q_nope_val,
                q_nope_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            # ROCm 平台路径
            # gfx95 + MXFP4 权重路径
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                # gfx95 FP8 权重路径 或 CUDA 图模式下的 FP8 FNUZ 路径
                if (_use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn) or (
                    get_is_capture_mode() and self.w_kc.dtype == torch.float8_e4m3fnuz
                ):
                    # fp8 Triton kernel: always on gfx950,
                    # cudagraph-only on gfx942 (hides launch overhead)
                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    # 默认 ROCm 路径：BF16 BMM + 权重缩放
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        # CUDA 平台 FP8 权重路径
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                # CPU 平台：转 BF16 后做 BMM
                q_nope_out = torch.bmm(
                    q_nope.to(torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(torch.bfloat16) * self.w_scale,
                )
            else:
                # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                # 修复 cuBLAS 12.9 下 BumpAllocator 导致的 bmm_fp8 错误
                q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                    q_nope.transpose(0, 1),
                    (
                        # cuBLAS >= 12.9 时使用普通内存分配，否则使用 BumpAllocator
                        torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                q_nope_out = bmm_fp8(
                    q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
                )
        else:
            # 默认路径：BF16 BMM
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)
        # 应用 LoRA 修正（如果 kv_b_proj 上有活跃的 LoRA）
        if is_kv_b_lora_active(self):
            q_nope_out = apply_kv_b_lora_q_correction(self, q_nope, q_nope_out)

        skip_rope_for_dsa_tilelang_fused = self._skip_rope_for_dsa_tilelang_fused()
        skip_rope_for_aiter_fused_mla = self._skip_rope_for_aiter_fused_mla()
        # 应用旋转位置编码（RoPE）
        # 跳过条件：无 rotary_emb / TRT-LLM MLA 融合 RoPE / DSA TileLang 融合 / AITER 融合 MLA
        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not skip_rope_for_dsa_tilelang_fused)
            and (not skip_rope_for_aiter_fused_mla)
            and (not _use_aiter or not _is_gfx95_supported or self.use_dsa)
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        # 上下文并行模式：重建 KV 缓存以支持 allgather + rearrange
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    # MLA 前向计算核心阶段
    # 执行吸收式 MLA 或标准 MLA 的注意力计算，包括 Q/K 拼接、
    # 注意力核心计算、V 投影（吸收 w_vc 权重）等
    def forward_absorb_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):
        save_kv_cache = True

        # 支持 absorbed core attention 的后端路径
        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            # DSA TileLang 融合路径：融合 RoPE + 拼接 + 缓存为一步操作
            if self._skip_rope_for_dsa_tilelang_fused() and self.rotary_emb is not None:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )
                # 融合执行：Q/K RoPE + 拼接 + 写入 KV 缓存
                q_cat, _, k_pe_fused, _ = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )
                save_kv_cache = False
                # On decode, pass q_cat directly to attn_mqa with q_rope=None so
                # dsa_backend.forward_decode reuses q_cat as a zero-copy view
                # (`q.contiguous().view(...)`) instead of running the
                # redundant `concat_mla_absorb_q_general(q_nope_fused, q_pe_fused)`
                # that would otherwise rebuild a tensor byte-identical to q_cat.
                # On ROCm tilelang decode, this eliminates the
                # `CatArrayBatchedCopy<OpaqueType<1u>, ...>` kernel that used to
                # fire once per layer per decode step (~2.6 us / layer saved).
                # Prefill keeps the split form because dsa_backend.forward_extend
                # asserts `q_rope is not None`.
                # 解码模式：直接将 q_cat 传给 attn_mqa，避免冗余的拼接操作
                # 预填充模式：保持拆分形式，因为 dsa_backend.forward_extend 要求 q_rope 非空
                if forward_batch.forward_mode.is_decode_or_idle():
                    if llama_4_scaling is not None:
                        # llama_4_scaling applies only to the q_nope portion;
                        # mutate in place via the slice view of q_cat.
                        # LLaMA-4 缩放仅应用于 q_nope 部分，通过切片视图原地修改
                        q_cat[..., : self.kv_lora_rank] *= llama_4_scaling
                    attn_output = self.attn_mqa(
                        q_cat,
                        None,
                        None,
                        forward_batch,
                        q_rope=None,
                        k_rope=k_pe_fused,
                        save_kv_cache=save_kv_cache,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
                else:
                    # 预填充模式：从 q_cat 中拆分出 q_nope_fused 和 q_pe_fused
                    q_nope_fused = q_cat[..., : self.kv_lora_rank]
                    q_pe_fused = q_cat[..., self.kv_lora_rank :]
                    if llama_4_scaling is not None:
                        q_nope_fused *= llama_4_scaling
                    attn_output = self.attn_mqa(
                        q_nope_fused,
                        None,
                        None,
                        forward_batch,
                        q_rope=q_pe_fused,
                        k_rope=k_pe_fused,
                        save_kv_cache=save_kv_cache,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
            else:
                # 非融合路径：直接使用 q_nope_out 和 q_pe/k_pe
                extra_args = {}
                # TRT-LLM MLA 融合 RoPE 路径
                if self._fuse_rope_for_trtllm_mla(forward_batch):
                    extra_args = {
                        "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                        "is_neox": self.rotary_emb.is_neox_style,
                        "llama_4_scaling": llama_4_scaling,
                    }
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
        else:
            # 非 absorbed core attention 后端路径（如 aiter 非 gfx95 等）
            # 需要 Q/K 完整拼接后再送入注意力计算
            if _use_aiter_gfx95:
                # gfx95 融合路径：融合 RoPE + 拼接 + 缓存
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                # 默认路径：拼接 Q 和 K 的非 RoPE 和 RoPE 部分
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        # 将注意力输出重塑为 (tokens, heads, kv_lora_rank) 形状
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        # ============ V 投影：吸收 w_vc 权重 ============
        # 计算 attn_output @ w_vc，将注意力输出投影到 V 空间

        # DeepGEMM 分块 GEMM 路径
        if self.use_deep_gemm_bmm:
            (
                attn_output_val,
                attn_output_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = per_token_group_quant_mla_deep_gemm_masked_fp8(
                attn_output.transpose(0, 1)
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            # ROCm 平台路径
            # gfx95 + MXFP4 权重路径
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                B_heads, M_batch = x.shape[0], x.shape[1]
                N_vdim = self.w_vc.shape[2]
                # Allocate in (batch, heads, dim) so the post-GEMM
                # transpose+flatten is a free view instead of a copy.
                # 以 (batch, heads, dim) 布局分配内存，使后续转置+展平为零开销的视图操作
                _bmm_buf = torch.empty(
                    M_batch,
                    B_heads,
                    N_vdim,
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                attn_bmm_output = _bmm_buf.transpose(0, 1)
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                _bmm_buf = None
                # gfx95 FP8 权重路径
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    # 默认 ROCm 路径：BF16 BMM + 权重缩放
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            # gfx95 后处理：根据 o_proj 权重类型选择量化/展平方式
            if _bmm_buf is not None:
                # _bmm_buf is already (batch, heads, dim) contiguous
                # _bmm_buf 已经是 (batch, heads, dim) 连续布局
                if self.o_proj.weight.dtype == torch.uint8:
                    # MXFP4 量化展平
                    attn_bmm_output = fused_flatten_mxfp4_quant(_bmm_buf)
                elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                    # FP8 分组量化展平
                    attn_bmm_output = fused_flatten_fp8_group_quant(
                        _bmm_buf, group_size=128, dtype_quant=torch.float8_e4m3fn
                    )
                else:
                    # BF16 直接展平
                    attn_bmm_output = _bmm_buf.flatten(1, 2)
            elif self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
                )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        # CUDA 平台 FP8 权重路径
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                # CPU 平台：转 BF16 后做 BMM
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                # FP8 量化后做 bmm_fp8
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        # cuBLAS >= 12.9 时使用普通内存分配
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        # MUSA 平台路径：转 BF16 后做 BMM
        elif _is_musa:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1), self.w_vc
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            # 默认 CUDA BF16 路径
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                # torch dynamo 要求使用 out= 参数的算子输出为非连续张量
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                # 预分配输出张量，使用 out= 参数避免额外内存分配
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        # 应用 LoRA V 修正（如果 kv_b_proj 上有活跃的 LoRA）
        if is_kv_b_lora_active(self):
            attn_bmm_output = apply_kv_b_lora_v_correction(
                self, attn_output, attn_bmm_output
            )
        # 输出投影
        output, _ = self.o_proj(attn_bmm_output)

        # 处理 indexer topk 结果传递
        if self.next_skip_topk is None:
            return output

        # Return topk_indices for the next layer when enabling index cache
        # 启用索引缓存时，将 topk_indices 传递给下一层
        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices

    # 判断是否应为 TRT-LLM MLA 解码的 FP8 路径融合 RoPE
    def _fuse_rope_for_trtllm_mla(
        self: DeepseekV2AttentionMLA, forward_batch: ForwardBatch
    ) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        # DSA/NSA 后端：当解码或预填充后端为 trtllm 且 KV 缓存为 fp8_e4m3 时融合
        if self.current_attention_backend in ("dsa", "nsa"):
            return (
                get_global_server_args().dsa_decode_backend == "trtllm"
                or get_global_server_args().dsa_prefill_backend == "trtllm"
            ) and get_attn_backend().kv_cache_dtype == torch.float8_e4m3fn

        # TRT-LLM MLA / TokenSpeed MLA / CuteDSL MLA 后端：
        # 解码或目标验证模式下，且数据类型为 fp8_e4m3 时融合
        return (
            self.current_attention_backend
            in ("trtllm_mla", "tokenspeed_mla", "cutedsl_mla")
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and get_attn_backend().data_type == torch.float8_e4m3fn
        )

    # 判断是否应在 gfx95 平台上使用 TileLang DSA 融合路径来跳过 RoPE
    def _skip_rope_for_dsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
        """
        Check if we should skip rope and use fused rope+cache path for TileLang DSA on gfx95.
        """
        server_args = get_global_server_args()
        return (
            _use_aiter_gfx95
            and self.current_attention_backend in ("dsa", "nsa")
            and (
                server_args.dsa_decode_backend == "tilelang"
                or server_args.dsa_prefill_backend == "tilelang"
            )
        )

    # 判断是否应在 AITER 融合 MLA 路径中跳过 RoPE
    # 当运行在 gfx95 平台且后端不在 FORWARD_ABSORB_CORE_ATTENTION_BACKENDS 中时，
    # 跳过 prepare 阶段的 RoPE，由 forward_absorb_core 中的融合内核处理
    def _skip_rope_for_aiter_fused_mla(self: DeepseekV2AttentionMLA) -> bool:
        """
        Skip rope in prepare and let the fused kernel in forward_absorb_core handle it,
        when running aiter-backend MLA on gfx95 (i.e., the `else` branch in forward_absorb_core
        that calls fused_qk_rope_cat_and_cache_mla).
        """
        return (
            _use_aiter_gfx95
            and self.current_attention_backend
            not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        )
