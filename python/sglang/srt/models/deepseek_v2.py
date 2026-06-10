# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
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

# 中文文件说明：
# 本文件实现了 DeepSeek V2/V3/V3.2 系列模型的推理-only 实现。
# 主要组件包括：
# - DeepseekV2MLP: 前馈网络（MLP）层，包含 gate_up_proj 和 down_proj
# - MoEGate: 混合专家（MoE）的门控网络，计算路由 logits
# - DeepseekV2MoE: 混合专家层，支持共享专家融合、DeepEP分发等
# - DeepseekV2AttentionMLA: 多头潜在注意力（Multi-head Latent Attention）层，
#   支持 MHA/MLA/DSA 等多种注意力模式
# - DeepseekV2DecoderLayer: Transformer 解码器层，组合自注意力和 MLP/MoE
# - DeepseekV2Model: 主模型结构，包含 embedding、多层解码器和 RMSNorm
# - DeepseekV2ForCausalLM: 因果语言模型，添加 lm_head 和 logits 处理
# 适配自 vLLM 项目的 DeepSeek V2 实现。

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.jit_kernel.dsv4 import (
    silu_and_mul_clamp,
    silu_and_mul_contig_post_quant,
)
from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.srt.batch_overlap.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.configs.model_config import (
    compute_mla_mscale_scaling,
    get_dsa_index_head_dim,
    get_dsa_index_n_heads,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.distributed import (
    divide,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_dsa_cp import DSACPLayerCommunicator
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_skip_post_experts_all_reduce,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
    is_deepep_class_backend,
    is_sbo_enabled,
    is_tbo_enabled,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    maybe_fuse_routed_scale_and_shared_add,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.cp_utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_prefill_context_parallel_enabled,
    mla_use_prefill_cp,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
    DeepseekMHAForwardMixin,
    DeepseekMLACpuForwardMixin,
    DeepseekMLAForwardMixin,
    DeepseekMLARocmForwardMixin,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.utils import (
    _device_sm,
    _get_llama_4_scaling,
    _is_cpu,
    _is_cpu_amx_available,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _is_musa,
    _is_npu,
    _is_xpu,
    _use_aiter,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    make_layers,
    use_intel_amx_backend,
)
from sglang.srt.utils.custom_op import register_custom_op

if _use_aiter:
    from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm

if _use_aiter_gfx95:
    from sglang.srt.layers.rocm_linear_utils import (
        get_dsv3_gemm_output_zero_allocator_size,
    )

if _use_aiter:
    pass

# 根据不同硬件平台导入对应的融合内核
if _is_cuda:
    from flashinfer.gemm import mm_M1_16_K7168_N256 as _raw_dsv3_router_gemm
    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
elif _is_npu:
    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
        forward_mha_core_npu,
        forward_mha_prepare_npu,
        forward_mla_core_npu,
        forward_mla_prepare_npu,
    )
elif _is_musa:
    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
else:
    pass

logger = logging.getLogger(__name__)


# DeepSeek V2 的前馈网络（MLP）层
# 包含 gate_up_proj（门控和上投影合并）和 down_proj（下投影）
class DeepseekV2MLP(nn.Module):
    # 初始化 MLP 层
    # hidden_size: 隐藏层维度, intermediate_size: 中间层维度, hidden_act: 激活函数类型
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        swiglu_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.swiglu_limit = swiglu_limit

        # gate_up_proj: 合并的门控投影和上投影层，输出维度为 [intermediate_size * 2]
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # down_proj: 下投影层，从 intermediate_size 映射回 hidden_size
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # 对于打包权重（如AMX），将 weight_packed 别名为 weight
        if not hasattr(self.gate_up_proj, "weight") and hasattr(
            self.gate_up_proj, "weight_packed"
        ):
            self.gate_up_proj.weight = self.gate_up_proj.weight_packed
        if not hasattr(self.down_proj, "weight") and hasattr(
            self.down_proj, "weight_packed"
        ):
            self.down_proj.weight = self.down_proj.weight_packed
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        # SiLU 激活函数 + 乘法门控（用于 SwiGLU）
        self.act_fn = SiluAndMul()
        # AMD 平台上是否使用融合的 clamp+激活+乘法 内核
        self.use_fused_clamp_act_mul = (
            _is_hip and envs.SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL.get()
        )
        self._fused_clamp_fp8_checked = False
        self._fused_clamp_use_fp8 = False

    # MLP 前向传播
    # 支持多种优化路径：NVFP4 融合、DeepGEMM FP8 融合、ROCm 融合 clamp、普通路径
    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ):
        # 当 tp_size=1 且输入为空时，直接返回
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        # 路径1：NVFP4 GEMM+SwiGLU 融合路径
        if (
            getattr(self, "_enable_nvfp4_gemm_swiglu_fusion", False)
            and self.swiglu_limit is None
            and not isinstance(x, tuple)
        ):
            from flashinfer import fp4_quantize

            from sglang.srt.layers.quantization.nvfp4_gemm_swiglu_nvfp4_quant import (
                nvfp4_gemm_swiglu_nvfp4_quant,
            )

            x_fp4, x_scale = fp4_quantize(
                x, self.gate_up_proj.input_scale_inv, enable_pdl=True
            )
            out_fp4, out_scale = nvfp4_gemm_swiglu_nvfp4_quant(
                x_fp4,
                x_scale,
                self.gate_up_proj.weight_swiglu_interleaved,
                self.gate_up_proj.weight_scale_swiglu_interleaved,
                self.gate_up_proj.alpha,
                self.down_proj.input_scale_inv,
                enable_pdl=True,
            )
            out, _ = self.down_proj(
                (out_fp4, out_scale),
                skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
            )
            return out

        # 路径2：使用预分配的零初始化输出缓冲区（用于小批量 FP8 GEMM）
        if (
            gemm_output_zero_allocator is not None
            and x.shape[0] <= 256
            and self.gate_up_proj.weight.dtype == torch.uint8
        ):
            y = gemm_output_zero_allocator.allocate(
                x.shape[0] * self.gate_up_proj.output_size_per_partition
            ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
            x = (x, None, y)

        # 计算 gate_up 投影
        gate_up, _ = self.gate_up_proj(x)
        # Fast path: fused silu+clamp+fp8_quant+deepgemm when conditions met.
        # Only valid when down_proj does NOT need an all-reduce and its weights
        # are fp8 (uint8 storage with weight_scale_inv).
        # 路径3：融合 SiLU+Clamp+FP8量化+DeepGEMM 的快速路径
        if (
            self.swiglu_limit is not None
            and not self.down_proj.reduce_results
            and self.down_proj.weight.dtype == torch.uint8
            and hasattr(self.down_proj, "weight_scale_inv")
        ):
            M, N = gate_up.shape
            down_input_fp8 = gate_up.new_empty((M, N // 2), dtype=torch.float8_e4m3fn)
            scale_block_size = 128
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(M, N // 2),
                device=gate_up.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            # 执行融合的 SiLU+乘法+Clamp+FP8量化
            silu_and_mul_contig_post_quant(
                input=gate_up,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=float(self.swiglu_limit),
            )
            down_output = gate_up.new_empty(
                (M, self.down_proj.output_size), dtype=torch.bfloat16
            )
            # 使用 DeepGEMM 执行 FP8 矩阵乘法
            deep_gemm_wrapper.gemm_nt_f8f8bf16(
                (down_input_fp8, down_input_scale),
                (self.down_proj.weight, self.down_proj.weight_scale_inv),
                down_output,
            )
            return down_output

        # 路径4：AMD 平台融合 clamp+激活+乘法
        if self.use_fused_clamp_act_mul and self.swiglu_limit is not None:
            from aiter.ops.triton.fusions.fused_clamp_act_mul import (
                fused_clamp_act_mul,
            )

            if not self._fused_clamp_fp8_checked:
                from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

                qm = getattr(self.down_proj, "quant_method", None)
                self._fused_clamp_use_fp8 = (
                    isinstance(qm, Fp8LinearMethod) and qm.block_quant
                )
                self._fused_clamp_fp8_checked = True

            if self._fused_clamp_use_fp8:
                from aiter import dtypes

                # 融合 clamp+激活+乘法 并输出 FP8 量化结果
                x_fp8, x_scale = fused_clamp_act_mul(
                    gate_up,
                    swiglu_limit=self.swiglu_limit,
                    activation="silu",
                    dtype_quant=dtypes.fp8,
                    transpose_scale=False,
                )
                x = (x_fp8, x_scale)
            else:
                x = fused_clamp_act_mul(
                    gate_up,
                    swiglu_limit=self.swiglu_limit,
                    activation="silu",
                )

        # Fallback: fused silu+clamp kernel (still faster than unfused)
        # 回退路径：融合 SiLU+Clamp 内核（比未融合版本更快）
        elif self.swiglu_limit is not None:
            M, N = gate_up.shape
            x = gate_up.new_empty((M, N // 2))
            silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))
        else:
            # 标准路径：仅使用 SiLU+Mul 激活函数（无 clamp）
            x = self.act_fn(gate_up)
        # 下投影
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x


# MoE 门控网络
# 计算路由 logits，决定每个 token 应该被分发到哪些专家
class MoEGate(nn.Module):
    # 初始化 MoE 门控
    # config: 模型配置, is_nextn: 是否为 next-n 预测层, is_hash_moe: 是否使用哈希 MoE
    # is_deepseek_v4: 是否为 DeepSeek V4 模型
    def __init__(
        self,
        config,
        quant_config,
        prefix: str = "",
        is_nextn: bool = False,
        is_hash_moe: bool = False,
        is_deepseek_v4: bool = False,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.is_deepseek_v4 = is_deepseek_v4
        # 门控权重：(n_routed_experts, hidden_size)
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )

        # noaux_tc 方法需要 correction_bias（修正偏置）
        if config.topk_method == "noaux_tc" and not is_hash_moe:
            correction_bias_dtype = torch.float32
            if quant_config is not None:
                if (
                    quant_config.get_name() == "modelopt_fp4"
                    and get_moe_runner_backend().is_flashinfer_trtllm()
                ):
                    correction_bias_dtype = torch.bfloat16
                elif _use_aiter and quant_config.get_name() in (
                    "fp8",
                    "compressed_tensors",
                    "quark",
                ):
                    correction_bias_dtype = torch.bfloat16
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)
            )
        else:
            self.e_score_correction_bias = None
        # CPU AMX 平台使用打包权重方法
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp

    # 门控前向传播：计算路由 logits
    # 根据不同硬件平台和配置选择不同的 GEMM 实现
    def forward(
        self,
        hidden_states,
        gemm_output_zero_allocator: BumpAllocator = None,
        forward_batch: ForwardBatch = None,
    ):
        # Intel AMX 平台使用打包权重的线性计算
        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states,
                self.weight,
                None,  # bias
                True,  # is_vnni
            )

        # 确定性推理模式使用标准 F.linear
        if get_global_server_args().enable_deterministic_inference:
            return F.linear(hidden_states, self.weight, None)

        # DSA/MLA 启用 prefill CP 时使用标准线性计算
        if (
            not self.is_deepseek_v4
            and forward_batch is not None
            and (
                dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)
                or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp)
            )
        ):
            logits = F.linear(hidden_states, self.weight, None)
        else:
            # NOTE: For some unknown reason, router_gemm seems degrade accept length.
            # CUDA 平台的小批量专用快速路由 GEMM
            if (
                _is_cuda
                and hidden_states.shape[0] <= 16
                and hidden_states.shape[1] == 7168
                and (self.weight.shape[0] == 256 or self.weight.shape[0] == 384)
                and _device_sm >= 90
            ):
                if _device_sm in [100, 103] and self.weight.shape[0] == 256:
                    # TODO: will check the dtype to be bf16
                    # router gemm output float32
                    # Blackwell 架构专用路由 GEMM，输出 float32
                    logits = torch.empty(
                        hidden_states.shape[0],
                        self.weight.shape[0],
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    flashinfer_dsv3_router_gemm(logits, hidden_states, self.weight)
                else:
                    # 非Blackwell架构使用 sgl_kernel 路由 GEMM
                    logits = dsv3_router_gemm(
                        hidden_states, self.weight, out_dtype=torch.float32
                    )

            elif _use_aiter:
                # AMD 平台使用 aiter 路由 GEMM
                logits = aiter_dsv3_router_gemm(hidden_states, self.weight)
            else:
                if self.is_deepseek_v4:
                    from sglang.jit_kernel.dsv4 import linear_bf16_fp32

                    # V4 使用 JIT 编译的 bf16->fp32 线性计算
                    logits = linear_bf16_fp32(hidden_states, self.weight)
                else:
                    # After testing, we may use the faster code in `if deepseek v4` branch
                    # 其他平台回退到标准 PyTorch 线性计算
                    logits = F.linear(hidden_states, self.weight, None)

        return logits


# DeepSeek V2 混合专家（MoE）层
# 支持多种 MoE 后端：标准融合 MoE、DeepEP、Mooncake、NIXL 等
# 支持共享专家融合、SBO/TBO 重叠计算等优化
class DeepseekV2MoE(nn.Module):

    # 初始化 MoE 层
    # config: 模型配置, layer_id: 层ID, is_nextn: 是否为 next-n 预测层
    # is_deepseek_v4: 是否为 V4 模型
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
        is_deepseek_v4: bool = False,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        n_shared_experts = (
            0 if config.n_shared_experts is None else int(config.n_shared_experts)
        )
        _fusion_disabled = get_global_server_args().disable_shared_experts_fusion

        # num_fused_shared_experts drives weight remapping in deepseek_weight_loader:
        # mlp.shared_experts → mlp.experts.256 when > 0.
        # 融合的共享专家数量，驱动权重加载时的重映射
        self.num_fused_shared_experts = 0 if _fusion_disabled else n_shared_experts

        # DeepEP shared expert fusion: shared expert is fused into the same MoE kernel
        # as a local expert at the home EP rank. Expert layout is expanded from 256
        # routed to 256+EP_size (e.g. 272 for EP=16). TopK handles interleaving.
        # DeepEP 共享专家融合：共享专家被融合到 MoE 内核中作为本地专家
        _is_deepep_fusion = (
            is_deepep_class_backend() and self.num_fused_shared_experts > 0
        )

        if _is_deepep_fusion:
            # 256 routed + EP_size shared slots = 272 experts total (for EP=16)
            # DeepEP 融合模式：路由专家数 + EP大小 的共享专家槽位
            num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
            top_k_for_moe = config.num_experts_per_tok + 1  # 8 routed + 1 shared
            # Interleaving for DeepEP dispatch is handled by TopK internally.
        else:
            # 非DeepEP模式：路由专家数 + 融合的共享专家数
            num_experts_for_moe = (
                config.n_routed_experts + self.num_fused_shared_experts
            )
            top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # 初始化门控网络
        self.gate = MoEGate(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
            is_hash_moe=self.is_hash,
            is_deepseek_v4=is_deepseek_v4,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )

        # scaling factor for fused shared experts on AMD-platform.
        # DeepEP doesn't need this: shared expert is only computed on home rank
        # (not all-reduced), so no 1/ep_size correction is needed.
        # 融合共享专家的缩放因子（非 DeepEP 模式下 EP 需要除以 ep_size）
        fused_shared_experts_scaling_factor = None
        if (
            self.moe_ep_size > 1
            and self.num_fused_shared_experts > 0
            and not _is_deepep_fusion
        ):
            # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
            # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
            # with fused_shared_experts
            # EP模式下每个GPU都会得到共享专家的输出，需要除以 ep_size 保持等价性
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        # 初始化专家层（根据量化配置选择不同的 MoE 实现）
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=num_experts_for_moe
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=top_k_for_moe,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            swiglu_limit=getattr(config, "swiglu_limit", None),
            prefix=add_prefix("experts", prefix),
        )

        # 初始化 TopK 选择器（哈希MoE 或 标准分组TopK）
        if self.is_hash and not (is_nextn and is_deepseek_v4):
            # 哈希 MoE 使用 HashTopK
            self.topk = HashTopK(
                topk=config.num_experts_per_tok + self.num_fused_shared_experts,
                num_experts=config.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                vocab_size=config.vocab_size,
                scoring_func=config.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            )
        else:
            # Default: grouped noaux_tc top-k. Covers V3/V3.2/GLM-5/Glm4MoeLite.
            # 默认：分组 noaux_tc TopK，覆盖 V3/V3.2/GLM-5/Glm4MoeLite
            topk_kwargs = dict(
                top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
                layer_id=self.layer_id,
                renormalize=config.norm_topk_prob,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                num_fused_shared_experts=self.num_fused_shared_experts,
                topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
                quant_config=quant_config,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
                fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
                # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
                # and requires the output format to be standard (except trtllm). We use quant_config to determine the output format.
                output_format=(
                    TopKOutputFormat.STANDARD
                    if (quant_config is None)
                    and (not get_moe_runner_backend().is_flashinfer_trtllm())
                    else None
                ),
            )
            # DSV4 override: ungrouped sqrtsoftplus + fp4 expert layout flag.
            # V4 覆盖：非分组 sqrtsoftplus + FP4 专家布局标志
            if is_deepseek_v4:
                topk_kwargs.update(
                    use_grouped_topk=False,
                    scoring_func=config.scoring_func,
                    is_fp4_experts=getattr(quant_config, "is_fp4_experts", False),
                    apply_routed_scaling_factor_on_output=(
                        True
                        if _use_aiter
                        else self.experts.should_fuse_routed_scaling_factor_in_topk
                    ),
                )
            self.topk = TopK(**topk_kwargs)

        # 共享专家相关标志
        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        self.shared_experts_weight_block_size = None
        self._shared_expert_tp1 = False
        # Shared experts: skip when fused into MoE kernel (self.num_fused_shared_experts > 0)
        # or when DeepEP fusion is enabled (shared expert is local slot 16 in FusedMoE, no separate MLP).
        # 当共享专家被融合到 MoE 内核中或 DeepEP 融合启用时，跳过独立共享专家
        if (
            config.n_shared_experts is not None
            and config.n_shared_experts > 0
            and self.num_fused_shared_experts == 0
            and not _is_deepep_fusion
        ):
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # Disable TP for shared experts for A2A/FP4 allgather paths, or when
            # explicitly requested for DSV4 checkpoints whose shared scales are
            # not divisible by the global TP size.
            # 对共享专家禁用 TP（用于 A2A/FP4 allgather 路径或显式请求）
            _shared_expert_use_tp1 = (
                get_moe_a2a_backend().is_deepep()
                or get_moe_a2a_backend().is_mooncake()
                or get_moe_a2a_backend().is_nixl()
                or get_moe_a2a_backend().is_mori()
                or get_moe_a2a_backend().is_ascend_fuseep()
                or get_moe_a2a_backend().is_flashinfer()
                or get_moe_a2a_backend().is_megamoe()
                or should_use_flashinfer_cutlass_moe_fp4_allgather()
                or envs.SGLANG_SHARED_EXPERT_TP1.get()
            )
            # 创建独立的共享专家 MLP
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                prefix=add_prefix("shared_experts", prefix),
                **(dict(tp_rank=0, tp_size=1) if _shared_expert_use_tp1 else {}),
            )
            # Flags must be set before weight load so
            # process_weights_after_loading sees them and builds the
            # [Up, Gate]-interleaved weight + scale.
            # 在权重加载前设置 NVFP4 融合标志
            from sglang.srt.layers.quantization.modelopt_quant import (
                ModelOptFp4LinearMethod,
            )
            from sglang.srt.utils.common import is_sm100_supported

            fc1_n = self.shared_experts.gate_up_proj.output_size_per_partition
            if (
                envs.SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION.get()
                and is_sm100_supported()
                and isinstance(
                    self.shared_experts.gate_up_proj.quant_method,
                    ModelOptFp4LinearMethod,
                )
                and isinstance(
                    self.shared_experts.down_proj.quant_method,
                    ModelOptFp4LinearMethod,
                )
                and fc1_n % 128 == 0
                and get_global_server_args().disable_piecewise_cuda_graph
            ):
                self.shared_experts.gate_up_proj._interleave_for_swiglu_fusion = True
                self.shared_experts._enable_nvfp4_gemm_swiglu_fusion = True
                self.shared_experts.down_proj._accepts_prequantized_fp4 = True
            self._shared_expert_tp1 = _shared_expert_use_tp1
            # 检查共享专家的量化类型
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            ) and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() in {
                "awq",
                "awq_marlin",
                "moe_wna16",
            }
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )
            if self.shared_experts_is_fp8:
                if (
                    _use_aiter
                    and config.quantization_config.get("quant_method")
                    == "compressed-tensors"
                ):
                    # For compressed-tensors ptpc model, don't need to check the weight_block_size
                    pass
                else:
                    # 验证 gate_up_proj 和 down_proj 的权重块大小一致
                    assert (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                        == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
                    )
                    self.shared_experts_weight_block_size = (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                    )

        self.top_k = config.num_experts_per_tok

        # 配置 EP（专家并行）相关参数
        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_nixl()
            or get_moe_a2a_backend().is_mori()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        # 标记是否启用 A2A（all-to-all）MoE
        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_nixl()
            or get_moe_a2a_backend().is_mori()
            or get_moe_a2a_backend().is_ascend_fuseep()
            or get_moe_a2a_backend().is_flashinfer()
        )
        # SBO（单批量重叠）模式下是否将共享专家融合到 SBO 内部
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    # 获取 MoE 权重参数列表（用于专家并行负载均衡等）
    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    # MoE 前向传播入口
    # 根据配置选择不同的前向路径：MegaMoE、标准双流、标准、DeepEP
    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe

        # 检查是否使用 MegaMoE（动态 MoE 路由策略）
        if should_use_mega_moe(self, hidden_states):
            return forward_mega_moe(
                self,
                hidden_states,
                forward_batch,
                input_ids_global=input_ids_global,
            )

        if not self._enable_a2a_moe:
            # 非 A2A 模式：根据条件选择双流或标准前向
            if (
                self.alt_stream is not None
                and self.num_fused_shared_experts == 0
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()
                and not (
                    get_global_server_args().enable_torch_compile
                    and hidden_states.shape[0]
                    <= get_global_server_args().torch_compile_max_bs
                    * (get_global_server_args().speculative_num_draft_tokens or 1)
                )
            ):
                # 使用双流（alt_stream）并行执行共享专家和路由专家
                return self.forward_normal_dual_stream(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids,
                    input_ids_global=input_ids_global,
                )
            else:
                return self.forward_normal(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids,
                    input_ids_global=input_ids_global,
                )
        else:
            # A2A 模式：使用 DeepEP 前向路径
            return self.forward_deepep(
                hidden_states, forward_batch, input_ids_global=input_ids_global
            )

    # 标准双流前向：共享专家和路由专家在两个 CUDA 流上并行执行
    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        # 在主流上计算共享专家
        shared_output = self._forward_shared_experts(
            hidden_states, gemm_output_zero_allocator
        )
        server_args = get_global_server_args()
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
            if server_args.enable_eplb
            else None
        )
        # 在 alt_stream 上并行执行路由专家计算
        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
            final_hidden_states = self.experts(hidden_states, topk_output)
            if (
                not _is_cuda
                and not _is_musa
                and not _use_aiter
                or isinstance(self.experts.quant_method, KTEPWrapperMethod)
            ):
                # 非 CUDA/MUSA 平台需要手动应用 routed_scaling_factor
                final_hidden_states *= self.routed_scaling_factor

        # 等待 alt_stream 完成
        current_stream.wait_stream(self.alt_stream)

        # 融合 routed scaling 和共享专家加法
        final_hidden_states = maybe_fuse_routed_scale_and_shared_add(
            self.experts,
            final_hidden_states,
            None if self._shared_expert_tp1 else shared_output,
            self.routed_scaling_factor,
        )

        # TP 模式下的 all-reduce
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
        # avoid summing the same shared output once per TP rank.
        # TP1 共享专家是复制的，在 all-reduce 之后添加，避免每个 TP rank 重复累加
        if self._shared_expert_tp1:
            final_hidden_states += shared_output
        return final_hidden_states

    # 标准前向路径（单流，无重叠）
    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # CPU AMX 平台使用专用前向路径
        if hasattr(self, "shared_experts") and use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ):
            return self.forward_cpu(hidden_states, should_allreduce_fusion)
        server_args = get_global_server_args()
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
            if server_args.enable_eplb
            else None
        )
        # 是否延迟共享专家计算（当 MoE runner 非原地操作时）
        defer_shared = not self.experts.moe_runner_config.inplace
        if hidden_states.shape[0] > 0:
            if not defer_shared and not self._fuse_shared_experts_inside_sbo:
                # 先计算共享专家
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator
                )
            # router_logits: (num_tokens, n_experts)
            # 计算路由 logits 和 TopK
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        # SBO 模式下通过钩子将共享专家融合到 SBO 流程中
        if self._fuse_shared_experts_inside_sbo:
            shared_output = None

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output
                self.alt_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.alt_stream):
                    # 在 alt_stream 上计算共享专家
                    shared_output = self._forward_shared_experts(
                        hidden_states, gemm_output_zero_allocator
                    )

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                nonlocal shared_output
                # 等待 alt_stream 上的共享专家计算完成
                torch.cuda.current_stream().wait_stream(self.alt_stream)
                post_combine_hook_handle.remove()

            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        # 执行路由专家计算
        final_hidden_states = self.experts(
            hidden_states,
            topk_output,
        )
        if (
            not _is_cuda
            and not _is_musa
            and not _is_xpu
            and not _use_aiter
            or isinstance(self.experts.quant_method, KTEPWrapperMethod)
        ):
            # fused in biased_grouped_topk so we can skip here
            # 非 CUDA/MUSA/XPU 平台需要手动应用 routed_scaling_factor
            final_hidden_states *= self.routed_scaling_factor

        # 延迟计算共享专家
        if (
            defer_shared
            and hidden_states.shape[0] > 0
            and not self._fuse_shared_experts_inside_sbo
        ):
            shared_output = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator
            )

        # 融合 routed scaling 和共享专家加法
        final_hidden_states = maybe_fuse_routed_scale_and_shared_add(
            self.experts,
            final_hidden_states,
            None if self._shared_expert_tp1 else shared_output,
            self.routed_scaling_factor,
        )

        # TP 模式下的 all-reduce
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
        # avoid summing the same shared output once per TP rank.
        # TP1 共享专家是复制的，在 all-reduce 之后添加
        if shared_output is not None and self._shared_expert_tp1:
            final_hidden_states += shared_output
        return final_hidden_states

    # CPU 平台前向路径（使用 Intel AMX 指令集优化）
    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        # router_logits: (num_tokens, n_experts)
        # 计算路由 logits 和 TopK
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        # 执行路由专家计算
        fused_experts_out = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        assert use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ) == use_intel_amx_backend(self.shared_experts.down_proj)
        # [Note] inplace should be False in fused_experts.
        # If inplace is True in fused_experts (self.experts), hidden_states will be changed after fused_experts
        # While hidden_states is still needed in shared_expert.
        # 使用 CPU 专用融合内核同时计算共享专家和路由专家的混合输出
        final_hidden_states = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states,
            self.shared_experts.gate_up_proj.weight,
            self.shared_experts.down_proj.weight,
            fused_experts_out,
            self.routed_scaling_factor,
            True,  # inplace
            self.shared_experts_is_int8,  # use_int8_w8a8
            self.shared_experts_is_fp8,  # use_fp8_w8a16
            (
                self.shared_experts.gate_up_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.gate_up_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w1_scale
            (
                self.shared_experts.down_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.down_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w2_scale
            (
                self.shared_experts_weight_block_size
                if self.shared_experts_is_fp8
                else None
            ),  # block_size
            True,  # is_vnni
        )
        # TP 模式下的 all-reduce
        if self.tp_size > 1 and not should_allreduce_fusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    # DeepEP 前向路径：使用 all-to-all 通信的专家并行
    # 支持 SBO（单批量重叠）和 TBO（双批量重叠）优化
    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shared_output = None
        sbo_enabled_flag = self._fuse_shared_experts_inside_sbo and not self.is_nextn
        # SBO 分发阶段重叠标志
        sbo_overlap_dispatch_flag = (
            sbo_enabled_flag and SboFlags.enable_dispatch_shared_one_stream_overlap()
        )
        # SBO 合并阶段重叠标志
        sbo_overlap_combine_flag = (
            sbo_enabled_flag and SboFlags.enable_combine_shared_two_stream_overlap()
        )

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            # 计算路由 logits
            router_logits = self.gate(hidden_states, forward_batch=forward_batch)
            # 非 SBO 模式下在 alt_stream 上异步计算共享专家
            if not sbo_enabled_flag and self.num_fused_shared_experts == 0:
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_output = self._forward_shared_experts(hidden_states)
                        shared_output.record_stream(self.alt_stream)
                        shared_event = self.alt_stream.record_event()
                else:
                    shared_output = self._forward_shared_experts(hidden_states)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
                **topk_kwargs,
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        # SBO 分发阶段重叠：共享专家与 DeepEP dispatch 重叠执行
        if sbo_overlap_dispatch_flag:
            shared_output = None

            def _deepep_dispatch_hook(dispatcher: BaseDispatcher):
                nonlocal shared_output
                # 在 dispatch 阶段计算共享专家
                shared_output = self._forward_shared_experts(hidden_states)
                for handle in deepep_dispatch_hook_handle:
                    handle.remove()

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):
                # 计算 combine 和 down gemm 的重叠参数
                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                post_dispatch_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                # 清理重叠参数
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            assert isinstance(self.experts.dispatcher, MaybeTboDeepEPDispatcher)
            deepep_dispatch_hook_handle = (
                self.experts.dispatcher.register_deepep_dispatch_hook(
                    _deepep_dispatch_hook
                )
            )
            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        # SBO 合并阶段重叠：共享专家与 DeepEP combine 重叠执行
        elif sbo_overlap_combine_flag:
            shared_output = None

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output

                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()

                # TODO reduce sm for non-deepgemm
                # 限制 DeepGEMM 使用的 SM 数量，为共享专家计算留出资源
                with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                    dispatcher.meta_overlap_args["compute_num_sms"]
                ):
                    shared_output = self._forward_shared_experts(hidden_states)

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )
        # Blackwell 平台：共享专家在 alt_stream 上与 DeepEP combine 重叠
        elif envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get():
            # On GB200: Shared experts overlapped on alt_stream, down gemm overlapped with DeepEP Combine

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):
                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()
                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        # 执行专家计算（包含分发和合并）
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        # 等待 alt_stream 上的共享专家计算完成
        if (
            hidden_states.shape[0] > 0
            and not sbo_enabled_flag
            and self.num_fused_shared_experts == 0
            and self.alt_stream is not None
        ):
            torch.cuda.current_stream().wait_event(shared_event)

        # 合并共享专家和路由专家的输出
        if shared_output is not None:
            x = shared_output
            # aiter moe call will handle routed_scaling_factor in the function
            # so add _use_aiter condition to eliminate to use self.routed_scaling_factor in add_ call
            # aiter MoE 已在内部处理 routed_scaling_factor，无需额外乘
            if self.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter:
                x.add_(final_hidden_states)
            else:
                # 非 aiter 模式需要乘以 routed_scaling_factor 后再加
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not (
                self.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter
            ):
                # 无共享专家时，直接乘以 routed_scaling_factor
                final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states

    # 前向计算共享专家（当共享专家未被融合到 MoE 内核时调用）
    def _forward_shared_experts(
        self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None
    ):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(
                hidden_states, gemm_output_zero_allocator=gemm_output_zero_allocator
            )
        else:
            return None

    # TBO 操作：计算门控路由 logits
    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    # TBO 操作：计算共享专家
    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if (self.num_fused_shared_experts == 0) and is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)
        else:
            state.shared_output = None

    # TBO 操作：选择专家（TopK 选择）
    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input

        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    # TBO 操作：分发阶段 A（将 token 发送到对应专家）
    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    # TBO 操作：分发阶段 B（接收来自其他 rank 的 token）
    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    # TBO 操作：执行专家核心计算
    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    # TBO 操作：合并阶段 A（将专家输出发送回原 rank）
    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    # TBO 操作：合并阶段 B（接收来自其他 rank 的专家输出并合并）
    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    # TBO 操作：输出处理（合并共享专家和路由专家输出）
    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if get_moe_a2a_backend().is_mori():
            num_tokens = state.pop("num_tokens")
            # 截取有效 token 数量（去除填充）
            final_hidden_states = final_hidden_states[:num_tokens]

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            if _use_aiter:
                # aiter 已在 TopK 内部处理 routed_scaling_factor
                x.add_(final_hidden_states)
            else:
                # 非 aiter 模式需要乘以 routed_scaling_factor 后再加
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        elif _use_aiter:
            # fused in aiter_biased_grouped_topk so we can skip here
            pass
        else:
            # 无共享专家时，直接乘以 routed_scaling_factor
            final_hidden_states *= self.routed_scaling_factor

        state.hidden_states_mlp_output = final_hidden_states


# DeepSeek V2 多头潜在注意力（MLA）层
# 支持多种注意力模式：MHA（多头注意力）、MLA（多头潜在注意力）、
# DSA（DeepSeek 注意力，V3.2 新增）、NPU 专用模式等
class DeepseekV2AttentionMLA(
    nn.Module,
    DeepseekMHAForwardMixin,
    DeepseekMLAForwardMixin,
    DeepseekMLARocmForwardMixin,
    DeepseekMLACpuForwardMixin,
):

    # 初始化 MLA 注意力层
    # qk_nope_head_dim: 不使用 RoPE 的 Q/K 头维度
    # qk_rope_head_dim: 使用 RoPE 的 Q/K 头维度
    # q_lora_rank: Q 的 LoRA 秩（MLA 压缩）
    # kv_lora_rank: KV 的 LoRA 秩（MLA 压缩）
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        # QK 总头维度 = 不使用 RoPE 的部分 + 使用 RoPE 的部分
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.quant_config = quant_config
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        if self.dsa_enable_prefill_cp:
            assert self.use_dsa, "CP currently only supports deepseek v3.2 model"
        # cp reuses the attn_tp comm group but needs to duplicate the weights;
        # store cp_size whenever either CP flavor is active so rebuild_cp_kv_cache
        # and the FA3 MLA wrapper can reach it on the dense MLA path too.
        # CP（上下文并行）复用 attn_tp 通信组但需要复制权重
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        # 本地注意力头数（按 TP 切分）
        self.num_local_heads = num_heads // attn_tp_size
        # 注意力缩放因子
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_global_server_args().kv_cache_dtype

        # NOTE modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        # 设置 rope_scaling 类型为 deepseek_yarn（DeepSeek 专用 YaRN 缩放）
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # For tensor parallel attention
        # 张量并行注意力层的投影矩阵初始化
        if self.q_lora_rank is not None:
            # MLA 模式：融合的 QKV a 投影（Q LoRA + KV LoRA + Q RoPE 部分）
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            # Q 的 a 层 LayerNorm（LoRA 压缩后）
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            # Q 的 b 投影（从 LoRA 秩恢复到完整头维度）
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=self._get_q_b_proj_quant_config(quant_config),
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            # 非 LoRA 模式：独立的 Q 投影和 KV 投影
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        # DSA（DeepSeek Sparse Attention）相关配置
        self.skip_topk = None
        self.next_skip_topk = None
        if self.use_dsa:
            # 初始化 DSA Indexer（稀疏注意力索引器）
            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
            self.indexer = Indexer(
                hidden_size=hidden_size,
                index_n_heads=get_dsa_index_n_heads(config),
                index_head_dim=get_dsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_dsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
            )
            # Refer: https://arxiv.org/abs/2603.12201 for more details.
            # skip_topk: when True, this layer will skip computation and reuse previous layer's topk indices.
            # next_skip_topk: when True, the next layer will skip computation and reuse this layer's topk indices.
            # skip_topk: 当为 True 时，此层跳过计算并复用上一层的 topk 索引
            # next_skip_topk: 当为 True 时，下一层将跳过计算并复用此层的 topk 索引
            if is_nextn:
                self.skip_topk = False
                self.next_skip_topk = False
            else:
                self.index_topk_freq = getattr(config, "index_topk_freq", 1)
                self.index_topk_pattern = getattr(config, "index_topk_pattern", None)
                if self.index_topk_pattern is None:
                    # 按频率决定是否跳过 topk 计算
                    self.skip_topk = max(layer_id - 1, 0) % self.index_topk_freq != 0
                    self.next_skip_topk = layer_id % self.index_topk_freq != 0
                else:
                    # 按模式决定是否跳过 topk 计算
                    self.skip_topk = self.index_topk_pattern[layer_id] == "S"
                    if layer_id < len(self.index_topk_pattern) - 1:
                        self.next_skip_topk = (
                            self.index_topk_pattern[layer_id + 1] == "S"
                        )
                    else:
                        self.next_skip_topk = False

        # KV 的 b 投影（从 LoRA 秩恢复到完整 KV 头维度）
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        # 输出投影
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # KV a 层 LayerNorm（LoRA 压缩后）
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # 初始化旋转位置编码（RoPE）
        if not skip_rope:
            is_neox_style = not getattr(config, "rope_interleave", True)
            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                device=get_global_server_args().device,
            )

            if rope_scaling and rope_scaling.get("apply_yarn_scaling", True):
                # 应用 YaRN 缩放到注意力缩放因子
                self.scaling = compute_mla_mscale_scaling(rope_scaling, self.scaling)
        else:
            self.rotary_emb = None
        self.use_deepseek_yarn_rope = rope_scaling is not None

        # MQA 注意力：使用压缩的 KV（kv_lora_rank + qk_rope_head_dim），1个KV头
        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        # MHA 注意力：使用完整的 KV 头，num_local_heads 个 KV 头
        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.alt_stream = alt_stream
        # 将 kv_b_proj 绑定到 attn_mha 上，用于 MHA 路径
        self.attn_mha.kv_b_proj = None

        # 吸收后的 KV 投影权重缓存
        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.current_attention_backend = (
            None  # Attention backend used by current forward batch
        )

        self.has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
        # 检查是否为打包权重（AWQ等量化格式）
        self.is_packed_weight = (
            self.has_fused_proj
            and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
            and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )
        # 是否使用最小延迟融合 A GEMM（针对小批量优化的专用内核）
        self.use_min_latency_fused_a_gemm = (
            self.has_fused_proj
            and not self.is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.bfloat16
            and self.fused_qkv_a_proj_with_mqa.weight.shape[0] == 2112
            and self.fused_qkv_a_proj_with_mqa.weight.shape[1] == 7168
            and _is_cuda
            and 90 <= _device_sm < 120
        )

        # 初始化各种注意力前向方法（来自 Mixin 类）
        self.init_mha_forward()
        self.init_mla_forward()
        self.init_mla_fused_rope_rocm_forward()
        self.init_mla_fused_rope_cpu_forward()

    # 根据前向批次信息分派注意力前向方法
    # 根据解码/预填充/投机等模式选择不同的注意力后端
    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        # Determine attention backend used by current forward batch
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = get_global_server_args().decode_attention_backend
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            # Use the specified backend for speculative operations (both verify and draft extend)
            # 投机解码模式：根据配置选择解码或预填充注意力后端
            if get_global_server_args().speculative_attention_mode == "decode":
                attention_backend = get_global_server_args().decode_attention_backend
            else:  # default to prefill
                attention_backend = get_global_server_args().prefill_attention_backend
        else:
            attention_backend = get_global_server_args().prefill_attention_backend
        self.current_attention_backend = attention_backend

        handler = AttentionBackendRegistry.get_handler(attention_backend)
        return handler(self, forward_batch)

    # TBO 操作：注意力准备阶段
    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
        )

    # TBO 操作：注意力核心计算
    def op_core(self, state):
        result = self.forward_core(state.pop("attn_intermediate_state"))
        # forward_core may return (hidden_states, topk_indices) for DSA models
        # with index cache enabled. In the TBO path, topk_indices is not
        # propagated between layers, so we discard it here.
        # DSA 模型可能返回 (hidden_states, topk_indices) 元组
        if isinstance(result, tuple):
            state.hidden_states_after_attn = result[0]
        else:
            state.hidden_states_after_attn = result

    # 注意力层前向传播入口
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        layer_scatter_modes: LayerScatterModes = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            layer_scatter_modes=layer_scatter_modes,
            llama_4_scaling=llama_4_scaling,
            prev_topk_indices=prev_topk_indices,
        )
        return self.forward_core(s)

    # 注意力准备阶段：根据前向方法分派到不同的准备函数
    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        layer_scatter_modes: LayerScatterModes = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        # 延迟绑定 kv_b_proj 到 attn_mha
        if self.attn_mha.kv_b_proj is None:
            self.attn_mha.kv_b_proj = self.kv_b_proj

        # when hidden_states is a tuple of tensors, the tuple will include quantized weight and scale tensor
        # 空输入的快速路径
        if isinstance(hidden_states, tuple):
            if (
                not get_attn_tp_context().input_scattered
                and hidden_states[0].shape[0] == 0
            ):
                assert (
                    not self.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states[0]
        else:
            if (
                not get_attn_tp_context().input_scattered
                and hidden_states.shape[0] == 0
            ):
                assert (
                    not self.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states, None, forward_batch, None

        # 根据注意力前向方法分派到对应的准备函数
        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)
        if attn_forward_method == AttnForwardMethod.MHA:
            # 标准 MHA 路径
            inner_state = self.forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            # MHA 分块 KV 路径
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            # MHA 一次性路径
            inner_state = self.forward_normal_one_shot_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA:
            # MLA 吸收路径
            inner_state = self.forward_absorb_prepare(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                llama_4_scaling,
                prev_topk_indices,
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_ROCM:
            # AMD ROCm 融合 MLA+RoPE 路径
            inner_state = self.forward_absorb_fused_mla_rope_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            # CPU 融合 MLA+RoPE 路径
            inner_state = self.forward_absorb_fused_mla_rope_cpu_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            # NPU MHA 路径
            inner_state = forward_mha_prepare_npu(
                self,
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                layer_scatter_modes,
            )
        elif attn_forward_method == AttnForwardMethod.MLA_NPU:
            # NPU MLA 路径
            inner_state = forward_mla_prepare_npu(
                self,
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                layer_scatter_modes,
            )
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            # NPU DSA 路径
            inner_state = forward_dsa_prepare_npu(
                self,
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                layer_scatter_modes,
                prev_topk_indices,
            )
        else:
            raise NotImplementedError
        return None, attn_forward_method, forward_batch, inner_state

    # 注意力核心计算：根据前向方法分派到不同的核心计算函数
    def forward_core(self, intermediate_state):
        hidden_states, attn_forward_method, forward_batch, inner_state = (
            intermediate_state
        )
        if inner_state is None:
            return hidden_states

        if attn_forward_method == AttnForwardMethod.MHA:
            return self.forward_normal_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            return self.forward_normal_one_shot_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA:
            return self.forward_absorb_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_ROCM:
            return self.forward_absorb_fused_mla_rope_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            return self.forward_absorb_fused_mla_rope_cpu_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            return forward_mha_core_npu(self, *inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_NPU:
            return forward_mla_core_npu(self, *inner_state)
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            return forward_dsa_core_npu(self, *inner_state)
        else:
            raise NotImplementedError

    # 准备 QKV 潜在表示
    # 使用融合的 QKV a 投影计算潜在 QKV，支持小批量优化 GEMM
    def prepare_qkv_latent(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        assert self.q_lora_rank is not None
        # When the module is wrapped with LoRA, the fused GEMM fast-path would
        # bypass the adapter because it reads weight.T directly.
        # 当模块被 LoRA 包装时，融合 GEMM 快速路径会绕过适配器
        lora_active = getattr(self.fused_qkv_a_proj_with_mqa, "set_lora", False)
        if (
            (not isinstance(hidden_states, tuple))
            and hidden_states.shape[0] >= 1
            and hidden_states.shape[0] <= 16
            and self.use_min_latency_fused_a_gemm
            and not lora_active
        ):
            # 小批量专用低延迟融合 A GEMM
            qkv_latent = dsv3_fused_a_gemm(
                hidden_states, self.fused_qkv_a_proj_with_mqa.weight.T
            )
        else:
            qkv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        return qkv_latent

    # 重建上下文并行（CP）的 KV 缓存
    # 执行 all-gather + 重排，将各 CP rank 的 KV 缓存合并
    def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):
        # support allgather+rerrange
        # 写入压缩的 KV 潜在缓存
        latent_cache[..., : self.kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., self.kv_lora_rank :] = k_pe.squeeze(1)
        # CP all-gather + 重排
        latent_cache_output = cp_all_gather_rerange_output(
            latent_cache.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
        # 从重排后的缓存中提取 k_nope 和 k_pe
        k_nope = latent_cache_output[..., : self.kv_lora_rank].unsqueeze(1)
        k_pe = latent_cache_output[..., self.kv_lora_rank :].unsqueeze(1)
        return k_nope, k_pe

    # 获取 q_b_proj 的量化配置
    # 当启用 NVFP4 检查点 FP8 GEMM 时，使用 FP8 配置
    @staticmethod
    def _get_q_b_proj_quant_config(quant_config):
        if envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get():
            # refer to real DeepSeek V3 quant config
            # 参考 DeepSeek V3 的真实量化配置
            return Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:
            return quant_config


# DeepSeek V2 解码器层
# 包含自注意力（MLA）和 MLP/MoE，以及层归一化和残差连接
class DeepseekV2DecoderLayer(nn.Module):

    # 初始化解码器层
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        # 解析 RoPE 参数
        if hasattr(config, "rope_parameters"):
            rope_theta = config.rope_parameters["rope_theta"]
            assert rope_theta is not None, f"rope_theta not found in config: {config}"
            rope_type = config.rope_parameters.get("rope_type")
            rope_scaling = config.rope_parameters if rope_type != "default" else None
        else:
            rope_theta = config.rope_theta
            rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        # 初始化自注意力层（MLA）
        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            is_nextn=is_nextn,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )
        if not hasattr(config, "q_lora_rank") and envs.SGLANG_USE_AG_AFTER_QLORA.get():
            raise ValueError(
                "SGLANG_USE_AG_AFTER_QLORA only supports the model with q_lora_rank"
            )

        # 判断当前层是否为稀疏层（使用 MoE）
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        # 初始化层的 scatter 模式（用于 DP/TP 通信优化）
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        # 根据是否为稀疏层选择 MLP 或 MoE
        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
                dsa_enable_prefill_cp=dsa_enable_prefill_cp,
                mla_enable_prefill_cp=mla_enable_prefill_cp,
            )
        else:
            # 密集层使用标准 MLP
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )

        # 输入层归一化（Pre-Norm）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 注意力后层归一化
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # 检测 GFX95 平台的量化格式
        self._gfx95_quant_format = self._detect_gfx95_quant_format()

        # 初始化层通信器（处理 TP/DP/CP 的通信和 scatter/gather）
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            # DSACPLayerCommunicator is flavor-agnostic; its internal gates
            # read both dsa_use_prefill_cp and mla_use_prefill_cp. The rename
            # to CPLayerCommunicator is deferred to a cleanup PR.
            # CP 模式使用专用的层通信器
            self.layer_communicator = DSACPLayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )
        else:
            # 标准层通信器
            self.layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )

    # 检测 AMD GFX95 平台的量化格式
    def _detect_gfx95_quant_format(self) -> str:
        if not _is_gfx95_supported:
            return ""
        weight = getattr(
            getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None), "weight", None
        )
        if weight is None:
            return ""
        if weight.dtype == torch.uint8:
            return "mxfp4"
        if weight.dtype == getattr(torch, "float8_e4m3fn", None):
            return "fp8"
        return ""

    # 判断指定层是否为稀疏层（使用 MoE 而非标准 MLP）
    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    # 解码器层前向传播
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        gemm_output_zero_allocator: BumpAllocator = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states_orig = hidden_states
        # 准备注意力输入（层归一化 + scatter）
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            getattr(self, "_gfx95_quant_format", ""),
        )

        # 自注意力计算
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            llama_4_scaling=llama_4_scaling,
            layer_scatter_modes=self.layer_scatter_modes,
            prev_topk_indices=prev_topk_indices,
        )
        # DSA 模式可能返回 topk_indices
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        get_attn_tp_context().clear_attn_inputs()

        # 准备 MLP 输入（层归一化 + scatter）
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # 判断是否需要将 MLP all-reduce 与下一层融合
        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        # DP 填充模式下可使用 reduce-scatter 替代 all-reduce
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if isinstance(self.mlp, DeepseekV2MLP):
            gemm_output_zero_allocator = None

        # 非 inplace MoE 需要使用输出缓冲区上下文
        if (
            isinstance(self.mlp, DeepseekV2MoE)
            and not self.mlp.experts.moe_runner_config.inplace
            and not torch.compiler.is_compiling()
        ):
            from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

            _mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
        else:
            _mlp_ctx = nullcontext()

        with _mlp_ctx:
            # MLP/MoE 计算
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                should_allreduce_fusion,
                use_reduce_scatter,
                gemm_output_zero_allocator,
            )

        # 标记需要 all-reduce 融合
        if (
            not (self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp)
            and should_allreduce_fusion
        ):
            hidden_states._sglang_needs_allreduce_fusion = True

        # 层后处理（all-reduce + residual + gather）
        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual, topk_indices

    # TBO 操作：通信准备注意力输入
    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        if get_moe_a2a_backend().is_mori():
            state.num_tokens = hidden_states.shape[0]
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    # TBO 操作：通信准备 MLP 输入
    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    # TBO 操作：通信层后处理
    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


# DeepSeek V2 主模型结构
# 包含 embedding 层、多层解码器和最终的 RMSNorm
class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    # 初始化主模型
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()
        # DSA/MLA 上下文并行配置
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = (
            is_prefill_context_parallel_enabled() and not is_deepseek_dsa(config)
        )
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
        else:
            self.cp_size = None

        # 词嵌入层（仅在第一个 PP rank 上创建）
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # 创建备用 CUDA 流（用于重叠计算）
        self.alt_stream = (
            torch.cuda.Stream()
            if (
                _is_cuda
                or _is_musa
                or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
            )
            else None
        )

        # 创建解码器层（支持 PP 分片和专家卸载）
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
                dsa_enable_prefill_cp=self.dsa_enable_prefill_cp,
                mla_enable_prefill_cp=self.mla_enable_prefill_cp,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
            offloader_kwargs=dict(
                submodule_accessor=lambda layer: (
                    layer.mlp.experts
                    if isinstance(layer.mlp, DeepseekV2MoE)
                    else layer.mlp
                ),
                whitelist_param_names_creator=lambda module: (
                    [
                        "w13_weight",
                        "w2_weight",
                        # only for nvfp4
                        *(
                            [
                                "w13_blockscale_swizzled",
                                "w2_blockscale_swizzled",
                            ]
                            if hasattr(module, "w13_blockscale_swizzled")
                            else []
                        ),
                    ]
                    if isinstance(module, FusedMoE)
                    else []
                ),
            ),
        )
        # 最终的 RMSNorm（仅在最后一个 PP rank 上创建）
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # GFX95 平台的 GEMM 输出零分配器大小
        self.gemm_output_zero_allocator_size = 0
        if (
            _use_aiter_gfx95
            and config.n_routed_experts == 256
            and self.embed_tokens.embedding_dim == 7168
        ):
            num_moe_layers = sum(
                [
                    1
                    for i in range(len(self.layers))
                    if isinstance(self.layers[i].mlp, DeepseekV2MoE)
                ]
            )

            allocate_size = 0
            for i in range(len(self.layers)):
                if isinstance(self.layers[i].mlp, DeepseekV2MoE):
                    # tp_size = get_tensor_model_parallel_world_size()
                    is_a2a_moe = is_deepep_class_backend()
                    tp_size = (
                        1 if is_a2a_moe else get_tensor_model_parallel_world_size()
                    )
                    intermediate_size = (
                        config.moe_intermediate_size * config.n_shared_experts
                    )
                    share_expert_output_size_per_partition = divide(
                        intermediate_size * 2, tp_size
                    )
                    allocate_size = share_expert_output_size_per_partition
                    break

            self.gemm_output_zero_allocator_size = (
                get_dsv3_gemm_output_zero_allocator_size(
                    config.n_routed_experts,
                    num_moe_layers,
                    allocate_size,
                    self.embed_tokens.embedding_dim,
                )
            )
        # 需要捕获辅助隐藏状态的层列表
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        # llama_4_scaling: for supporting Mistral-Large-3 model
        # llama_4_scaling：用于支持 Mistral-Large-3 模型
        self.llama_4_scaling_config = getattr(config, "llama_4_scaling", None)

    # 获取输入嵌入层
    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    # 主模型前向传播
    # 处理输入嵌入、多层解码器、PP 代理张量、CP 分片等
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
        # PP 第一个 rank 处理输入嵌入
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            # 非第一个 rank 从 PP 代理张量获取隐藏状态
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
        device = hidden_states.device
        # 创建零分配器（用于注意力计算中的零填充）
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )

        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )

        # CP 模式：分片和重建数据
        if dsa_use_prefill_cp(
            forward_batch, self.dsa_enable_prefill_cp
        ) or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        # llama_4_scaling: for supporting Mistral-Large-3 model
        # Compute llama 4 scaling once per forward pass if enabled
        # 计算 llama 4 缩放因子（用于 Mistral-Large-3 模型）
        llama_4_scaling: Optional[torch.Tensor] = None
        if self.llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=self.llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=self.llama_4_scaling_config["beta"],
                positions=positions,
            )

        # TBO 模式下调整正常层的范围
        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        if forward_batch.can_run_tbo:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0
        aux_hidden_states = []
        topk_indices = None
        # 逐层执行解码器前向传播
        for i in range(normal_start_layer, normal_end_layer):
            # NOTE: torch dynamo does not support graph break in context manager
            ctx = (
                nullcontext()
                if not get_global_server_args().disable_piecewise_cuda_graph
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                if i in self.layers_to_capture:
                    # 捕获辅助隐藏状态（用于投机解码等）
                    if self.enable_a2a_moe and i > self.first_k_dense_replace:
                        aux_hidden_state = get_attention_tp_group().all_gather(
                            hidden_states + residual, dim=0
                        )
                        aux_hidden_states.append(aux_hidden_state)
                    else:
                        aux_hidden_states.append(hidden_states + residual)
                layer = self.layers[i]
                hidden_states, residual, topk_indices = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    llama_4_scaling,
                    prev_topk_indices=topk_indices,
                )

        # TBO 模式：对 MoE 层使用两批量重叠
        if normal_end_layer != self.end_layer:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
            )

        # 非最后一个 PP rank 返回代理张量
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            # 最后一个 PP rank 应用最终 RMSNorm
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        # CP 模式：all-gather + 重排
        if self.pp_group.is_last_rank and (
            dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)
            or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp)
        ):
            # allgather + rerrange
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


# DeepSeek V2 因果语言模型
# 包含主模型和语言模型头（lm_head），支持权重加载和投机解码
class DeepseekV2ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    # for quark model load
    packed_modules_mapping = {}

    # 初始化因果语言模型
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # for quark model load
        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        # 当 q_lora_rank 不为 None 时，融合 q_a_proj 和 kv_a_proj_with_mqa
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        # Quant configs like Quark may rely on the model to provide fused-module
        # mappings so exclusion checks can unfuse derived names back to the
        # checkpoint's source layer names.
        # Quark 等量化配置依赖模型提供的融合模块映射
        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # 确定融合的共享专家数量
        self.determine_num_fused_shared_experts()
        self.use_dsa = is_deepseek_dsa(config)
        self.model = DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        # 语言模型头（仅在最后一个 PP rank 上创建）
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                # 权重共享模式：lm_head 复用 embed_tokens
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            # 非最后一个 PP rank 使用占位层
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)

        # 懒加载的每层路由专家权重（用于 EPLB 等场景）
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )
        self.capture_aux_hidden_states = False

        # CP 配置
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = (
            is_prefill_context_parallel_enabled() and not is_deepseek_dsa(config)
        )
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_rank = get_attention_cp_rank()
            self.cp_size = get_attention_cp_size()
        else:
            self.cp_rank = self.cp_size = None

        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        # 初始化注意力 TP 上下文
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_dsa(config))

    # 获取每层路由专家的权重
    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    # 确定融合的共享专家数量
    # 根据模型配置、硬件能力和运行时设置判断是否可以融合共享专家
    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        server_args = get_global_server_args()

        if server_args.disable_shared_experts_fusion:
            return

        disable_reason = None
        if server_args.enforce_shared_experts_fusion:
            pass
        elif is_sbo_enabled() or is_tbo_enabled():
            disable_reason = "SBO/TBO enabled: incompatible with fusing shared expert into MoE kernel."
        elif is_deepep_class_backend():
            disable_reason = "DeepEP: fusion off by default (use --enforce-shared-experts-fusion to enable)."
        elif (
            self.config.architectures[0] != architecture
            # Allow-list of n_routed_experts values that have been validated
            # for shared-experts fusion under this code path. Currently:
            #   256 -> DeepSeek-V3 / R1
            #   384 -> Kimi-K2.5, only when the checkpoint is Quark MXFP4
            #          (amd/Kimi-K2.5-MXFP4); the standard
            #          moonshotai/Kimi-K2.5 (compressed-tensors) checkpoint
            #          stores the shared expert loose and is NOT pre-fused,
            #          so the fused path silently mis-loads it.
            # 允许融合共享专家的 n_routed_experts 值白名单
            or self.config.n_routed_experts not in (256, 384)
            or self.config.n_shared_experts != 1
            or (
                self.config.n_routed_experts == 384
                and (
                    self.quant_config is None or self.quant_config.get_name() != "quark"
                )
            )
        ):
            disable_reason = "Config does not support fused shared expert(s)."
        elif (
            (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0))
            and (not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4))
            and (not _is_musa or torch.musa.get_device_capability("musa") < (3, 1))
        ):
            # 硬件能力不足
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
                "or MT-platform with capability >= 31 can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            # EP 模式下硬件能力不足
            disable_reason = (
                "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) "
                "can use shared experts fusion optimization under expert parallelism."
            )
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."

        if disable_reason is not None:
            server_args.disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        # 启用共享专家融合
        self.num_fused_shared_experts = self.config.n_shared_experts

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    # 因果语言模型前向传播
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # Minor fix for multi-modal model: input_ids is None
        # 多模态模型修复：input_ids 可能为 None
        len_input_ids = (
            input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
        )
        # 准备 CP 元数据
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(
                len_input_ids, self.cp_size, self.use_dsa, forward_batch
            ):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len_input_ids,
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
        elif self.mla_enable_prefill_cp:
            if can_cp_split(len_input_ids, self.cp_size, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len_input_ids,
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )

        # 执行主模型前向传播
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        # 最后一个 PP rank 计算 logits
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    # 起始层（PP 分片）
    @property
    def start_layer(self):
        return self.model.start_layer

    # 结束层（PP 分片）
    @property
    def end_layer(self):
        return self.model.end_layer

    # 加载模型权重
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)

    # 获取嵌入层和语言模型头权重
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    # 设置嵌入层和语言模型头权重（用于权重热更新）
    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 获取专家位置配置（用于 EPLB）
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    # 设置 EAGLE3 投机解码需要捕获的层
    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            # 捕获第2层、中间层和倒数第3层的隐藏状态
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # TODO (Qiaolin-Yu): check if other draft models need similar layer id
            # adjustment
            if layer_ids and layer_ids[0] == 1:
                self.model.layers_to_capture = [val + 1 for val in layer_ids]
            else:
                self.model.layers_to_capture = list(layer_ids)

    # 设置 DFLASH 投机解码需要捕获的层
    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


# DeepSeek V3 因果语言模型（继承自 V2）
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


# DeepSeek V3.2 因果语言模型（继承自 V2）
class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass


# 注册自定义操作：FlashInfer DeepSeek V3 路由 GEMM
# 使用 FlashInfer 的专用 GEMM 内核计算路由 logits
@register_custom_op(
    op_name="flashinfer_dsv3_router_gemm",
    mutates_args=[],
    fake_impl=lambda logits, hidden_states, weight: None,
)
def flashinfer_dsv3_router_gemm(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    _raw_dsv3_router_gemm(
        hidden_states,
        weight.t(),
        logits,
        launch_with_pdl=True,
    )


# 模型入口类列表，SGLang 框架通过此列表注册支持的模型架构
EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
