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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

# 本文件实现了 Qwen2MoE（Qwen2 混合专家）模型的推理代码，包括稀疏 MoE 层、注意力层、
# 解码器层和因果语言模型，支持共享专家融合、DeepEP 后端、双流并行和 EAGLE3 推测解码。

import logging
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_moe_data_parallel_world_size,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_prefill_context_parallel_enabled,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    make_layers,
    use_intel_amx_backend,
)

if is_npu():
    from sglang.srt.hardware_backend.npu.cmo import (
        shared_expert_on_independent_stream,
        wait_share_stream,
    )

from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用 AMD AITER 后端


def can_fuse_shared_expert(
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether the shared expert may be fused as an extra MoE expert (Qwen3.5 + Aiter).

    Caller must still gate on ``support_shared_expert_fusion`` and ``_use_aiter``.
    """

    # 判断共享专家是否可以融合为 MoE 的额外专家（适用于 Qwen3.5 + Aiter）
    if (
        get_global_server_args().disable_shared_experts_fusion is True
        or getattr(config, "shared_expert_intermediate_size", 0) <= 0
        or config.shared_expert_intermediate_size != config.moe_intermediate_size
        or get_moe_a2a_backend().is_deepep()
    ):
        return False

    # If the shared expert is excluded from quantization (stored as FP32 in the
    # checkpoint), fusing it into the quantized MoE weight tensor requires online
    # quantization which is not supported. Disable fusion in this case.
    if quant_config is not None:
        exclude_layers = getattr(quant_config, "exclude_layers", [])
        if any(
            "shared_expert" in layer
            and "shared_expert_gate" not in layer
            and not layer.startswith("mtp.")
            for layer in exclude_layers
        ):
            return False

    return True


class Qwen2MoeMLP(nn.Module):
    """Qwen2MoE 的标准 MLP 层，使用 SiLU 激活函数和门控机制"""

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
    ) -> None:
        super().__init__()
        # 门控和上投影合并为一个线性层
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # 下投影层
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
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU 激活函数，将门控和上投影的输出分割并相乘

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)  # 计算门控和上投影
        x = self.act_fn(gate_up)  # 应用 SiLU 激活
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )  # 下投影，可选跳过全归约
        return x


class Qwen2MoeSparseMoeBlock(nn.Module):
    """Qwen2MoE 的稀疏混合专家（MoE）模块，支持共享专家、共享专家融合和 DeepEP 后端"""

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
        is_nextn: bool = False,
        support_shared_expert_fusion: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.alt_stream = alt_stream  # 备用 CUDA 流，用于双流并行
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        self.num_experts = config.num_experts
        self.num_shared_experts = 0
        self.num_fused_shared_experts = 0
        if hasattr(config, "n_shared_experts"):
            # config defines the number of shared experts
            self.num_shared_experts = config.n_shared_experts
        elif (
            hasattr(config, "shared_expert_intermediate_size")
            and config.shared_expert_intermediate_size > 0
        ):
            # n_shared_experts is not defined, but shared_expert_intermediate_size is defined, so we use 1 as the number of shared experts
            self.num_shared_experts = 1

        self.enable_shared_expert_fusion = False  # default to False
        if _use_aiter:
            # enable shared expert fusion when use aiter
            self.enable_shared_expert_fusion = (
                support_shared_expert_fusion
                and can_fuse_shared_expert(config, quant_config)
            )
        if self.enable_shared_expert_fusion:
            self.num_fused_shared_experts = self.num_shared_experts  # 融合的共享专家数

        # Top-K 选择器
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
        )

        # MoE 专家层（使用动态获取的实现类）
        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=(
                config.num_experts_per_tok
                if not self.enable_shared_expert_fusion
                else config.num_experts_per_tok + self.num_fused_shared_experts
            ),
            num_experts=(
                config.num_experts + get_global_server_args().ep_num_redundant_experts
                if not self.enable_shared_expert_fusion
                else config.num_experts
                + get_global_server_args().ep_num_redundant_experts
                + self.num_fused_shared_experts
            ),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

        # 路由门控网络
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        # When enable_shared_expert_fusion, the shared expert runs inside the MoE kernel
        # (via _append_shared_to_topk_output); a separate shared_expert MLP would
        # double-count. If fusion is off (num_fused_shared_experts == 0), keep shared_expert.
        # 当共享专家融合启用时，共享专家在 MoE 内核中运行，不再需要单独的 MLP
        if (
            config.shared_expert_intermediate_size > 0
            and not self.enable_shared_expert_fusion
        ):
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if (
                        get_moe_a2a_backend().is_deepep()
                        or get_moe_a2a_backend().is_flashinfer()
                    )
                    else {}
                ),
            )
        else:
            self.shared_expert = None
        # 共享专家门控：控制共享专家的输出权重
        if _is_cpu and _is_cpu_amx_available:
            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=add_prefix("shared_expert_gate", prefix),
            )
        else:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.num_experts + get_global_server_args().ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok
        self.is_nextn = is_nextn

    def get_moe_weights(self):
        """获取 MoE 专家的权重参数，用于专家并行负载均衡"""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def _get_shared_expert_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return sigmoid(shared_expert_gate) for fused shared expert weights."""
        # 计算共享专家门控的 sigmoid 权重，用于融合共享专家
        if not self.enable_shared_expert_fusion or self.shared_expert_gate is None:
            return None
        shared_out = self.shared_expert_gate(hidden_states)
        shared_logits = shared_out[0] if isinstance(shared_out, tuple) else shared_out
        return F.sigmoid(shared_logits)

    def _append_shared_to_topk_output(
        self,
        topk_output: StandardTopKOutput,
        hidden_states: torch.Tensor,
    ) -> StandardTopKOutput:
        """Append shared expert ids and weights to topk output before fused MoE."""
        # 将共享专家的 ID 和权重追加到 Top-K 输出中，以便在融合 MoE 中一起执行
        if not self.enable_shared_expert_fusion:
            return topk_output
        shared_weights = self._get_shared_expert_weights(hidden_states)
        if shared_weights is None:
            return topk_output

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            fused_append_shared_experts_with_weights,
        )

        fused_topk_ids, fused_topk_weights = fused_append_shared_experts_with_weights(
            topk_output.topk_ids,
            topk_output.topk_weights,
            shared_weights,
            self.num_fused_shared_experts,
            N=self.num_experts,
        )
        return StandardTopKOutput(
            topk_weights=fused_topk_weights,
            topk_ids=fused_topk_ids,
            router_logits=topk_output.router_logits,
        )

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        """前向计算共享专家的输出，并应用门控权重"""
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                if use_intel_amx_backend(self.shared_expert_gate):
                    # Intel AMX 后端的融合线性-sigmoid-乘法操作
                    shared_output = torch.ops.sgl_kernel.fused_linear_sigmoid_mul(
                        hidden_states,
                        self.shared_expert_gate.weight,
                        self.shared_expert_gate.bias,
                        True,
                        shared_output,
                    )
                else:
                    # 标准路径：sigmoid 门控 * 共享专家输出
                    shared_output = (
                        F.sigmoid(self.shared_expert_gate(hidden_states))
                        * shared_output
                    )

        return shared_output

    def _forward_deepep(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        """使用 DeepEP 后端的 MoE 前向计算"""
        enable_dual_stream = (
            is_npu()
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and forward_batch.forward_mode.is_cuda_graph()
        )
        shared_output = None
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)  # 计算路由门控
            if enable_dual_stream:
                shared_output = shared_expert_on_independent_stream(
                    hidden_states.clone(), self._forward_shared_experts
                )
            else:
                shared_output = self._forward_shared_experts(hidden_states)  # 计算共享专家
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=(
                    ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    )
                    if not self.is_nextn
                    else None
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )  # 执行专家计算
        if enable_dual_stream:
            wait_share_stream()  # 等待共享专家流完成

        if shared_output is not None:
            final_hidden_states.add_(shared_output)  # 加上共享专家输出

        return final_hidden_states

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        """前向计算路由器和稀疏专家"""
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)  # 计算路由门控
        topk_output = self.topk(hidden_states, router_logits)  # Top-K 选择
        if self.enable_shared_expert_fusion and TopKOutputChecker.format_is_standard(
            topk_output
        ):
            topk_output = self._append_shared_to_topk_output(topk_output, hidden_states)  # 追加融合共享专家
        return self.experts(hidden_states, topk_output)  # 执行专家计算

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """使用双流并行计算路由专家和共享专家"""
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states.clone())  # 在主流上计算共享专家

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)  # 在备用流上计算路由专家

        current_stream.wait_stream(self.alt_stream)  # 等待备用流完成

        return router_output, shared_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        """MoE 块的前向传播入口"""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if get_moe_a2a_backend().is_deepep():
            return self._forward_deepep(hidden_states, forward_batch)  # DeepEP 后端路径

        if hidden_states.shape[0] == 0:
            # M=0 guard for idle DP ranks: skip shared_experts and gate
            # (which crash on empty tensors in FP4 GEMM), but still call
            # self.experts() to participate in alltoall collective.
            # 空张量保护：跳过共享专家和门控，但仍调用 experts 参与集合通信
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(hidden_states, topk_output)
        elif self.alt_stream is not None and get_is_capture_mode():
            # 双流并行路径
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            # 标准路径：依次计算共享专家和路由专家
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states)

        if shared_output is not None:
            final_hidden_states += shared_output  # 加上共享专家输出
        if (
            self.tp_size > 1
            and not should_skip_post_experts_all_reduce(
                is_tp_path=True,
                use_reduce_scatter=use_reduce_scatter,
                should_allreduce_fusion=should_allreduce_fusion,
            )
            and not get_moe_a2a_backend().is_flashinfer()
        ):
            # 跨张量并行 rank 做全归约
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        # Debug removed - was causing issues during CUDA graph capture

        return final_hidden_states.view(num_tokens, hidden_dim)  # 恢复原始形状


class Qwen2MoeAttention(nn.Module):
    """Qwen2MoE 注意力层，使用旋转位置编码和分组查询注意力，支持 DP 注意力"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # DP 注意力的张量并行配置
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size  # 每个 TP 分片的注意力头数
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 每个 TP 分片的 KV 头数
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # QKV 投影层
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        # 输出投影层（不执行全归约，由 LayerCommunicator 统一处理）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算 QKV 投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割 Q/K/V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    """Qwen2MoE 解码器层，包含注意力和 MoE 前馈网络，支持 DP 注意力和 reduce-scatter"""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.start_layer = start_layer
        rope_theta, rope_scaling = get_rope_config(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        qkv_bias = getattr(config, "qkv_bias", True)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qkv_bias=qkv_bias,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Qwen2MoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True  # 所有层都是稀疏 MoE 层
        is_previous_layer_sparse = True
        is_next_layer_sparse = True

        # 层散射模式，控制 DP/TP 之间的数据分发
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            # 稀疏 MoE 层
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            # 稠密 MLP 层（Qwen2MoE 当前未使用此分支）
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # 层通信器，统一处理 DP/TP 之间的归约和散射
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 准备注意力输入，处理 DP/TP 数据分发
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                **kwargs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # 准备 MLP 输入
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)  # MoE 前馈计算

        # 后处理层输出，处理 DP/TP 数据收集
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Qwen2MoeModel(nn.Module):
    """Qwen2MoE 模型主体，包含词嵌入、多个解码器层和最终归一化，支持流水线并行和上下文并行"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.moe_dp_size = get_moe_data_parallel_world_size()  # MoE 数据并行大小
        self.attn_cp_size = get_attn_context_model_parallel_world_size()  # 注意力上下文并行大小

        # 流水线并行第一 rank 负责词嵌入
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2MoeDecoderLayer
        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        # 构建解码器层，按流水线并行分配层范围
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                start_layer=pp_start_layer,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        # 流水线并行最后 rank 负责最终归一化
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []  # 需要捕获中间输出的层（用于 EAGLE3 推测解码）

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        """设置需要捕获中间输出的层，用于 EAGLE3 推测解码"""
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)  # 词嵌入查找
            else:
                hidden_states = input_embeds
            residual = None
        else:
            # 非第一 rank 从流水线代理张量中获取隐藏状态
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # 上下文并行：拆分和重建数据
        if (
            is_prefill_context_parallel_enabled()
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        aux_hidden_states = []  # 辅助隐藏状态，用于 EAGLE3
        if forward_batch.can_run_tbo:
            # 两批次重叠（TBO）模式
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            # 标准逐层前向
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if not get_global_server_args().disable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states
                            if getattr(layer, "_is_layer_to_capture", False)
                            else None
                        ),
                    )

        if not self.pp_group.is_last_rank:
            # 非最后 rank 返回代理张量用于流水线通信
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化

        # 上下文并行：全收集和重排输出
        if (
            self.pp_group.is_last_rank
            and is_prefill_context_parallel_enabled()
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.attn_cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states  # 返回辅助隐藏状态用于 EAGLE3


class Qwen2MoeForCausalLM(nn.Module):
    """Qwen2MoE 因果语言模型，包含模型主体和语言模型头，支持 EAGLE3 推测解码"""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None  # CUDA 备用流
        self.model = Qwen2MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        # 语言模型头
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        """前向传播：计算隐藏状态并生成 logits"""
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states  # 解包辅助隐藏状态
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        """分段预填充前向传播，将预填充过程拆分为多个区间以支持流水线并行"""
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        """获取流水线并行的起始层索引"""
        return self.model.start_layer

    @property
    def end_layer(self):
        """获取流水线并行的结束层索引"""
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理堆叠参数和 MoE 专家权重的映射"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            # 跳过不属于当前流水线阶段的权重
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue  # 跳过专家中的堆叠参数，由下方 expert_params_mapping 处理
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)  # 加载堆叠权重
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )  # 加载专家权重
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)  # 默认方式加载权重
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置配置，用于 EPLB（专家并行负载均衡）"""
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        """设置 EAGLE3 推测解码需要捕获的层"""
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture(
                [
                    2,
                    num_layers // 2,
                    num_layers - 3,
                ]
            )  # Specific layers for EAGLE3 support
        else:
            self.model.set_eagle3_layers_to_capture([val + 1 for val in layer_ids])


EntryClass = Qwen2MoeForCausalLM
