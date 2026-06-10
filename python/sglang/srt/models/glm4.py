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

# Modeling from:
# ./llama.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4/modular_glm4.py
"""Inference-only GLM-4-0414 model compatible with HuggingFace weights."""

# 本文件实现了 GLM-4-0414 模型的推理专用版本，兼容 HuggingFace 权重格式。
# 主要包含以下组件：
# - Glm4MLP: 前馈神经网络（MLP）层，使用 SiLU 激活函数和门控机制
# - Glm4Attention: 多头注意力层，支持 GQA（分组查询注意力）和部分旋转位置编码
# - Glm4DecoderLayer: 单个 Transformer 解码器层，包含自注意力、MLP 和多个 RMSNorm
# - Glm4Model: GLM-4 模型主体，由嵌入层、多个解码器层和最终归一化层组成
# - Glm4ForCausalLM: 因果语言模型，在模型主体基础上增加语言模型头和词嵌入

import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config

Glm4Config = None

logger = logging.getLogger(__name__)


class Glm4MLP(nn.Module):
    """GLM-4 的前馈神经网络（MLP）模块，采用门控线性单元（GLU）结构。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        # 门控投影和上投影合并为一个线性层，输出维度为 intermediate_size * 2
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        # 下投影层，将中间维度映射回隐藏维度
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        # SiLU 激活函数并与门控值相乘（SiluAndMul 将 gate 和 up 分成两半，分别做 SiLU 和乘法）
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
    ):
        # gate_up_proj 输出同时包含门控值和上投影值
        gate_up, _ = self.gate_up_proj(x)
        # SiluAndMul: 对前半部分做 SiLU 激活，再与后半部分逐元素相乘
        x = self.act_fn(gate_up)
        # 下投影，映射回原始隐藏维度
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter,
        )
        return x


class Glm4Attention(nn.Module):
    """GLM-4 的多头注意力模块，支持 GQA（分组查询注意力）和部分旋转位置编码（Partial RoPE）。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 131072,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        partial_rotary_factor: float = 0.5,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        # 每个张量并行rank分配的注意力头数
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        # 每个张量并行rank实际使用的 KV 头数
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            # 默认头维度 = 隐藏维度 / 注意力头数
            self.head_dim = hidden_size // self.total_num_heads
        # Q 和 KV 的总维度大小
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放因子，防止点积值过大
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        # 部分旋转位置编码因子，仅对部分维度应用 RoPE（GLM-4 默认为 0.5，即一半维度）
        self.partial_rotary_factor = partial_rotary_factor

        # QKV 合并投影层
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        # 输出投影层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # 旋转位置编码，使用非 Neox 风格（即 GPT-J 风格的旋转方式）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
            partial_rotary_factor=partial_rotary_factor,
            is_neox_style=False,
        )
        # Radix 注意力实现，支持前缀缓存等高效推理特性
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
        # 计算 QKV 投影
        qkv, _ = self.qkv_proj(hidden_states)
        # 按 Q、K、V 的维度切分
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 对 Q 和 K 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 执行注意力计算
        attn_output = self.attn(q, k, v, forward_batch)
        # 输出投影
        output, _ = self.o_proj(attn_output)
        return output


class Glm4DecoderLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    # GLM-4 单个解码器层，包含自注意力、MLP 以及四个 RMSNorm 层。
    # 与标准 LLaMA 架构不同，GLM-4 在自注意力和 MLP 之后各有一个额外的 Post-Norm。

    def __init__(
        self,
        config: Glm4Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # 获取 RoPE 配置（theta 和缩放参数）
        rope_theta, rope_scaling = get_rope_config(config)
        partial_rotary_factor = (rope_scaling or {}).get("partial_rotary_factor")
        if partial_rotary_factor is None:
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        bias = getattr(config, "attention_bias", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        # 自注意力子模块
        self.self_attn = Glm4Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            partial_rotary_factor=partial_rotary_factor,
            bias=bias,
            prefix=add_prefix("self_attn", prefix),
        )

        # MLP
        self.mlp = Glm4MLP(
            config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        # 输入层归一化（在自注意力之前）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 自注意力后的层归一化（用于残差连接，在 MLP 之前）
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # 自注意力输出的后归一化（GLM-4 特有，在自注意力输出上）
        self.post_self_attn_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # MLP 输出的后归一化（GLM-4 特有，在 MLP 输出上）
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            # 第一层没有残差，直接保存输入作为残差
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # input_layernorm 同时进行归一化和残差更新
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 执行自注意力计算
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        # 自注意力输出后归一化（GLM-4 特有）
        hidden_states = self.post_self_attn_layernorm(hidden_states)

        # Fully Connected
        # 在进入 MLP 前进行残差归一化
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 执行 MLP 前馈计算
        hidden_states = self.mlp(hidden_states)
        # MLP 输出后归一化（GLM-4 特有）
        hidden_states = self.post_mlp_layernorm(hidden_states)

        return hidden_states, residual


class Glm4Model(nn.Module):
    """GLM-4 模型主体，由词嵌入层、多个解码器层和最终归一化层组成。支持流水线并行（PP）。"""

    def __init__(
        self,
        config: Glm4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Glm4DecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        # 仅第一个 PP rank 需要嵌入层
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Glm4DecoderLayer
        decoder_layer_type = decoder_layer_type or Glm4DecoderLayer
        # 构建解码器层列表，支持流水线并行分层
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        # 仅最后一个 PP rank 需要最终归一化层
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        # 需要捕获中间层隐藏状态的层索引列表（用于 EAGLE3 推测解码）
        self.layers_to_capture = []

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入词嵌入层。"""
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        # 第一个 PP rank：将输入 token ID 转换为嵌入向量
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            # 非第一个 PP rank：从前一阶段的代理张量中获取隐藏状态和残差
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # 收集辅助隐藏状态（用于 EAGLE3）
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                # 记录当前层的隐藏状态（残差连接前）
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        # 非最后一个 PP rank：将隐藏状态和残差传递给下一阶段
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            # 最后一个 PP rank：应用最终归一化
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        # 如果没有需要捕获的中间层状态，直接返回最终隐藏状态
        if len(aux_hidden_states) == 0:
            return hidden_states

        # 同时返回最终隐藏状态和辅助隐藏状态（用于 EAGLE3）
        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """加载 KV 缓存的量化缩放因子，用于 FP8 等量化推理场景。"""
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                # 设置 K 和 V 的缩放因子
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling factor attribute!"
                )


class Glm4ForCausalLM(nn.Module):
    """GLM-4 因果语言模型，在模型主体基础上增加语言模型头（lm_head）用于词表预测。"""

    def __init__(
        self,
        config: Glm4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        # 构建模型主体
        self.model = Glm4Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        # 根据流水线并行 rank 决定语言模型头的处理方式
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                # 单卡且权重绑定时，lm_head 直接复用嵌入层权重
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            # 非最后一个 PP rank 使用占位层
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        # 流水线并行时的权重绑定：将嵌入层权重从第一个 rank 传送到最后一个 rank
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            else:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        # 池化层，用于嵌入类任务，取最后一个 token 的表示并归一化
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # For EAGLE3 support
        # 是否捕获辅助隐藏状态（用于 EAGLE3 推测解码）
        self.capture_aux_hidden_states = False

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """根据输入 token ID 获取对应的嵌入向量。"""
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入词嵌入层。"""
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # 前向传播：获取模型主体的隐藏状态
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        # 如果启用 EAGLE3，从模型输出中解包辅助隐藏状态
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                # 生成模式：通过 logits 处理器计算词表概率分布
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                # 嵌入模式：通过池化层获取向量表示
                return self.pooler(hidden_states, forward_batch)
        else:
            # 非最后一个 PP rank，直接返回隐藏状态供下一阶段使用
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
        """分段预填充前向传播，将预填充计算拆分为多个区间分别执行，降低峰值显存。"""
        start, end = split_interval
        # embed
        # 第一个区间：执行词嵌入
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        # 执行指定区间内的解码器层
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            # 最后一个区间：应用最终归一化
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            # 计算 logits
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            # 非最后一个区间：暂无最终结果
            result = None

        return result

    @property
    def start_layer(self):
        """当前 PP rank 负责的起始解码器层索引。"""
        return self.model.start_layer

    @property
    def end_layer(self):
        """当前 PP rank 负责的结束解码器层索引（不包含）。"""
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，支持合并投影（如 QKV 合并、gate_up 合并）和量化兼容。"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            # 跳过不属于当前 PP rank 的层权重
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            # 跳过旋转位置编码的逆频率参数和投影器权重
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            # 权重绑定时跳过 lm_head 权重（除非是多 PP 的最后一个 rank）
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    # 从权重中找到嵌入层权重用于绑定
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue

            # 处理需要合并的参数（如 QKV、gate_up）
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                # 将分片权重加载到合并参数的对应位置
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # 处理普通参数（无需合并）
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        """获取嵌入层权重和语言模型头权重，用于推测解码等场景。"""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """设置嵌入层权重和语言模型头权重，用于推测解码等场景。"""
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        # 清空 CUDA 缓存并同步，确保旧权重释放
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """加载 KV 缓存的量化缩放因子，委托给模型主体处理。"""
        self.model.load_kv_cache_scales(quantization_param_path)


# 模型入口类，供 SGLang 框架自动发现和注册
EntryClass = [Glm4ForCausalLM]
