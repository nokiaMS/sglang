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
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/models/llama4.py
"""Inference-only LLaMA model compatible with HuggingFace weights."""

# LLaMA4 模型实现：支持混合专家（MoE）架构、共享专家与路由专家并行计算、
# 注意力温度调节（NoPE 层）以及 QK 归一化等特性的推理专用模型。

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Llama4TextConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.models.llama import LlamaForCausalLM, LlamaMLP
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.utils import (
    add_prefix,
    fast_topk,
    get_compiler_backend,
    is_cuda,
    is_npu,
    make_layers,
)
from sglang.srt.utils.common import get_current_device_stream_fast

_is_cuda = is_cuda()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


# LLaMA4 混合专家（MoE）模块：包含路由专家和共享专家
class Llama4MoE(nn.Module):

    # 自定义路由函数：使用 sigmoid 激活而非 softmax 对路由分数进行归一化
    @torch.compile(dynamic=True, backend=get_compiler_backend())
    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 使用快速 topk 选取前 k 个专家
        router_scores_aK, router_indices_aK = fast_topk(gating_output, topk, dim=-1)
        # 使用 sigmoid 激活函数替代传统的 softmax 归一化
        router_scores_aK = torch.sigmoid(router_scores_aK.float()).to(
            hidden_states.dtype
        )
        return (
            router_scores_aK.view(-1).reshape(router_scores_aK.shape),
            router_indices_aK.to(torch.int32),
        )

    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        # 每个 token 选择的专家数量
        self.top_k = config.num_experts_per_tok
        self.device_module = torch.get_device_module()

        intermediate_size_moe = config.intermediate_size
        # 路由器：将隐藏状态映射到各专家的概率分布
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("router", prefix),
        )

        # TopK 选择模块：选取 top-k 个专家
        self.topk = TopK(
            top_k=self.top_k,
            renormalize=False,
            custom_routing_function=Llama4MoE.custom_routing_function,
        )

        # 路由专家集合：多个独立的 MLP 专家
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            layer_id=layer_id,
            reduce_results=False,  # 不在此处做 all-reduce，由外部处理
            quant_config=quant_config,
            apply_router_weight_on_input=True,
            prefix=add_prefix("experts", prefix),
        )

        # 共享专家：所有 token 都会经过的 MLP
        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            prefix=add_prefix("shared_expert", prefix),
            reduce_results=False,  # We need to do scatter before reduce
        )

    # MoE 前向传播：合并共享专家和路由专家的输出
    def forward(
        self,
        hidden_states,
        forward_batch: ForwardBatch,
        use_reduce_scatter: bool = False,
    ):
        shared_out, routed_out = self._forward_core(
            hidden_states, forward_batch.forward_mode
        )

        # 共享专家输出 + 路由专家输出
        out_aD = routed_out + shared_out

        # 张量并行时执行 all-reduce 同步
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
        ):
            out_aD = tensor_model_parallel_all_reduce(out_aD)

        return out_aD

    # 根据设备类型选择核心前向计算策略
    def _forward_core(self, hidden_states, forward_mode: ForwardMode):
        if _is_cuda:
            # CUDA 上使用共享专家与路由专家重叠计算以提升性能
            return self._forward_core_shared_routed_overlap(hidden_states)
        else:
            return self._forward_core_normal(hidden_states)

    # 普通前向计算：共享专家和路由专家顺序执行
    def _forward_core_normal(self, hidden_states):
        # router_scores: [num_tokens, num_experts]
        router_logits, _ = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_out = self.experts(hidden_states, topk_output)
        return shared_out, routed_out

    # 共享专家与路由专家重叠计算：在 CUDA 上利用备用流实现并行
    def _forward_core_shared_routed_overlap(self, hidden_states):
        alt_stream = _get_or_create_alt_stream(self.device_module)

        alt_stream.wait_stream(get_current_device_stream_fast())

        # 在主流上计算共享专家
        shared_out = self.shared_expert(hidden_states)

        # 在备用流上计算路由专家（与共享专家并行）
        with self.device_module.stream(alt_stream):
            # router_scores: [num_tokens, num_experts]
            router_logits, _ = self.router(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            routed_out = self.experts(hidden_states, topk_output)
        # 等待备用流完成
        get_current_device_stream_fast().wait_stream(alt_stream)

        return shared_out, routed_out


# 备用 CUDA 流，用于共享专家和路由专家的重叠计算
_alt_stream = None


# 获取或创建备用 CUDA 流（单例模式）
def _get_or_create_alt_stream(device_module):
    global _alt_stream
    if _alt_stream is None:
        _alt_stream = device_module.Stream()
    return _alt_stream


# LLaMA4 注意力模块：支持 RoPE、QK 归一化、注意力温度调节
class Llama4Attention(nn.Module):

    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        # 每4层中有一层不使用 RoPE（NoPE 层）
        self.use_rope = (layer_id + 1) % 4 != 0
        # 仅在使用 RoPE 的层上启用 QK 归一化
        self.use_qk_norm = config.use_qk_norm and self.use_rope

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        # 当前张量并行分片上的注意力头数
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        # 当前分片上的 KV 头数
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = config.head_dim
        # Q 和 KV 的维度大小
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放因子
        self.scaling = self.head_dim**-0.5
        # 注意力温度调节相关参数
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        # GQA 分组重复次数
        self.n_rep = self.num_heads // self.num_kv_heads
        # QK 归一化层（可选，仅在使用 QK 归一化时创建）
        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
            )
            if self.use_qk_norm
            else None
        )

        # 量化配置：允许对 QKV 和 O 投影分别设置是否跳过量化的权重
        qkv_quant_config = quant_config
        o_quant_config = quant_config
        if quant_config and hasattr(quant_config, "ignore") and quant_config.ignore:
            if add_prefix("q_proj", prefix) in quant_config.ignore:
                qkv_quant_config = None
            if add_prefix("o_proj", prefix) in quant_config.ignore:
                o_quant_config = None

        # QKV 融合投影层
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=qkv_quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        # 输出投影层
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=o_quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
        )
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type in ["llama", "llama4"]:
            is_neox_style = False

        # 旋转位置编码（仅在使用 RoPE 的层上创建）
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_theta),
                rope_scaling=rope_scaling if rope_scaling != "default" else None,
                is_neox_style=is_neox_style,
            )
            if self.use_rope
            else None
        )

        # 注意力计算模块
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
        )

    # 计算注意力温度缩放因子（用于 NoPE 层的温度调节）
    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return attn_scale.unsqueeze(-1)

    # 编译优化的注意力缩放乘法
    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _mul_attn_scale(self, positions, q):
        attn_scale = self._get_attn_scale(positions)
        return (q * attn_scale).to(q.dtype)

    # 注意力前向传播
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 计算 QKV 投影
        qkv, _ = self.qkv_proj(hidden_states)

        # 将 QKV 分割为 QK 和 V（QK 仍需进一步分割以应用 RoPE）
        qk, v = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)

        # 应用旋转位置编码
        if self.rotary_emb is not None:
            q_view, k_view = qk.split([self.q_size, self.kv_size], dim=-1)
            q_out_unused, k_out_unused = self.rotary_emb(positions, q_view, k_view)
            if _is_npu:
                qk = torch.cat([q_out_unused, k_out_unused], dim=-1)
            del q_view, k_view, q_out_unused, k_out_unused

        # 应用 QK 归一化
        if self.qk_norm is not None and _is_cuda:
            # Strided in-place fused QK RMSNorm reads/writes the qkv buffer
            # directly via the split q/k views, so the reshape-to-(N, head_dim)
            # copy is no longer needed. The remaining redundant copy
            # (`q.contiguous()` inside the attention backend) is unrelated.
            q, k = qk.split([self.q_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.qk_norm,
                k_norm=self.qk_norm,
                head_dim=self.head_dim,
            )
        else:
            if self.qk_norm is not None:
                # NPU/other: qk has been rebuilt via torch.cat after RoPE, so
                # this reshape is a free view; keep the previous path.
                qk = qk.reshape(-1, self.head_dim).contiguous().bfloat16()
                qk = self.qk_norm(qk).to(torch.bfloat16)
                qk = qk.reshape(-1, self.q_size + self.kv_size)
            q, k = qk.split([self.q_size, self.kv_size], dim=-1)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        # 对 NoPE 层应用注意力温度调节，在不影响短上下文的前提下适配超长上下文
        if self.attn_temperature_tuning and not self.use_rope:
            q = self._mul_attn_scale(positions=positions, q=q)

        # 计算注意力输出
        attn_output = self.attn(q, k, v, forward_batch)
        # 输出投影
        output, _ = self.o_proj(attn_output)
        return output


# LLaMA4 解码器层：包含自注意力和 MoE/MLP 前馈网络
class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        max_position_embeddings = config.max_position_embeddings
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # 自注意力子层
        self.self_attn = Llama4Attention(
            config=config,
            layer_id=layer_id,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            prefix=add_prefix("self_attn", prefix),
        )
        self.config = config
        # 判断当前层及相邻层是否为 MoE 层
        is_moe_layer = self._is_moe_layer(layer_id)
        is_previous_moe_layer = self._is_moe_layer(layer_id - 1)
        is_next_moe_layer = self._is_moe_layer(layer_id + 1)

        # 根据是否为 MoE 层选择不同的前馈网络
        if is_moe_layer:
            self.feed_forward = Llama4MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        # 层归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # 层散射模式：控制 DP 注意力和 MoE 之间的数据分发
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_moe_layer,
            is_previous_layer_sparse=is_previous_moe_layer,
            is_next_layer_sparse=is_next_moe_layer,
        )

        # 层通信器：处理注意力前后的数据分发和归约
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    # 判断指定层是否为 MoE 层
    def _is_moe_layer(self, layer_id: int) -> bool:
        if self.config.interleave_moe_layer_step == 0:
            return self.config.num_local_experts > 0
        return (layer_id + 1) % self.config.interleave_moe_layer_step == 0

    # 获取当前层的中间层维度大小
    def get_intermediate_size(self) -> int:
        if isinstance(self.feed_forward, Llama4MoE):
            return self.config.intermediate_size
        else:
            return self.config.intermediate_size_mlp

    # 解码器层前向传播
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 注意力前的数据准备（层归一化 + scatter）
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        # 当存在有效 token 时计算自注意力
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # MLP 前的数据准备（层归一化 + scatter）
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        # 判断是否使用 reduce-scatter 替代 all-reduce（用于 DP + padding 场景）
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        # Fully Connected
        # 前馈网络计算
        hidden_states = self.feed_forward(
            hidden_states, forward_batch, use_reduce_scatter
        )
        # 层后处理（残差连接 + gather）
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


# LLaMA4 模型主体：包含词嵌入、多层解码器和最终层归一化
class Llama4Model(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # 词嵌入层
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
            use_attn_tp_group=is_dp_attention_enabled(),
        )
        # 解码器层列表
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Llama4DecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )

        # 最终层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 需要捕获隐藏状态的层索引列表
        self.layers_to_capture = []

    # 模型前向传播
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # 计算词嵌入或使用输入嵌入
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        aux_hidden_states = []
        # 逐层前向传播
        for i in range(len(self.layers)):
            # 捕获指定层的隐藏状态（用于推测解码等）
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        # 最终层归一化（非空闲模式时）
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


# LLaMA4 因果语言模型：继承自 LlamaForCausalLM，使用 Llama4Model 作为模型主体
class Llama4ForCausalLM(LlamaForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 获取解码器层列表
    def get_layers(self):
        return self.model.layers

    # 初始化 Llama4 模型主体
    def _init_model(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return Llama4Model(config, quant_config=quant_config, prefix=prefix)


EntryClass = [Llama4ForCausalLM]
