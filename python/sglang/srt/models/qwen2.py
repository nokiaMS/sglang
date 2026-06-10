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

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""

# 本文件实现了 Qwen2 模型的推理专用代码，兼容 HuggingFace 权重格式。
# Qwen2 是基于 Transformer 的因果语言模型，支持张量并行(TP)和流水线并行(PP)。
# 主要包含以下组件：
#   - Qwen2MLP: 前馈网络(FFN)层，使用 SiLU 激活函数和门控机制
#   - Qwen2Attention: 多头注意力层，支持 GQA（分组查询注意力）和旋转位置编码(RoPE)
#   - Qwen2DecoderLayer: Transformer 解码器层，包含自注意力和前馈网络
#   - Qwen2Model: Qwen2 模型主体，由嵌入层、多层解码器和层归一化组成
#   - Qwen2ForCausalLM: 因果语言模型，在 Qwen2Model 基础上添加语言模型头

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_pp_indices,
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
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config

# Qwen2 配置类，延迟导入，将在运行时由模型加载器设置
Qwen2Config = None


logger = logging.getLogger(__name__)


# Qwen2 的前馈网络(MLP)层
# 采用门控线性单元(Gated Linear Unit)结构，包含 gate_up_proj 和 down_proj 两个线性层
# 激活函数使用 SiLU（Sigmoid Linear Unit）
class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # 门控和上投影合并为一个线性层，输出维度为 intermediate_size * 2
        # gate_proj 和 up_proj 合并后通过 SiLUAndMul 拆分并应用门控激活
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
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        # SiLU 激活函数，同时完成门控和乘法操作
        self.act_fn = SiluAndMul()

    # MLP 前向传播：gate_up_proj -> SiLU门控激活 -> down_proj
    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch = None,
    ) -> torch.Tensor:
        # 强化学习训练模式下，将输入转为 bfloat16 精度
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()

        # gate_up_proj 输出包含 gate 和 up 两部分，SiLuAndMul 对其分别应用 SiLU 和恒等变换后相乘
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        # down_proj 使用 RowParallelLinear 进行行并行计算，需要 forward_batch 做通信同步
        x, _ = self.down_proj(x, forward_batch=forward_batch)
        return x


# Qwen2 的多头注意力层
# 支持 GQA（分组查询注意力）和旋转位置编码(RoPE)
# 兼容双块注意力(dual chunk attention)配置
class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # 获取张量并行世界大小，用于切分注意力头
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        # Q 头数必须能被 TP 大小整除
        assert self.total_num_heads % tp_size == 0
        # 每个 TP 秩分到的 Q 头数
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
        # 每个 TP 秩分到的 KV 头数，最少为 1（不足时复制）
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # 每个注意力头的维度
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads
        # Q 和 KV 的总维度（在当前 TP 秩上）
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放因子，即 1/sqrt(head_dim)
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # QKV 投影层，将隐藏状态映射为 Q、K、V，Qwen2 使用 bias=True
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        # 输出投影层，将注意力输出映射回隐藏维度
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # 旋转位置编码(RoPE)初始化
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        # RadixAttention 是 SGLang 的高效注意力实现，支持 RadixTree KV 缓存
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    # 注意力前向传播：QKV投影 -> 旋转位置编码 -> 注意力计算 -> 输出投影
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 计算 Q、K、V 投影
        qkv, _ = self.qkv_proj(hidden_states)
        # 将 QKV 拆分为 Q、K、V 三个张量
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 对 Q 和 K 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 执行注意力计算
        attn_output = self.attn(q, k, v, forward_batch)
        # 输出投影
        output, _ = self.o_proj(attn_output)
        return output


# Qwen2 Transformer 解码器层
# 包含自注意力子层、前馈网络子层，以及对应的 RMSNorm 归一化和残差连接
class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.start_layer = start_layer
        # 从配置中获取 RoPE 参数
        rope_theta, rope_scaling = get_rope_config(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        # 自注意力子层
        self.self_attn = Qwen2Attention(
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
            prefix=add_prefix("self_attn", prefix),
        )
        # 前馈网络子层
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        # 自注意力前的 RMS 层归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 前馈网络前的 RMS 层归一化
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    # 解码器层前向传播
    # 采用 Pre-Norm 结构：LayerNorm -> Self-Attention -> 残差连接 -> LayerNorm -> MLP -> 残差连接
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # 第一层没有残差，直接将 hidden_states 作为残差
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 融合 RMSNorm 和残差加法，减少一次显存读写
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        # 注意力输出做 RMSNorm 和残差加法，然后送入 MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Qwen2 模型主体
# 由词嵌入层、多层 Transformer 解码器和最终层归一化组成
# 支持流水线并行(PP)，非首尾秩的嵌入层和归一化层用 PPMissingLayer 占位
class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        # 获取流水线并行组信息
        self.pp_group = get_pp_group()

        # 只有流水线并行的第一个秩需要词嵌入层
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                # 强化学习模式下使用 float32 精度
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )
        else:
            # 非首秩使用占位层，避免重复存储嵌入权重
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        # 获取当前 PP 秩对应的起始层索引
        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        # 构建解码器层列表，支持流水线并行分层
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
        # 只有流水线并行的最后一个秩需要最终层归一化
        if self.pp_group.is_last_rank:
            # 强化学习模式下使用 float32 精度进行归一化计算
            norm_kwargs = (
                dict(
                    weight_dtype=torch.float32,
                    cast_x_before_out_mul=True,
                    override_orig_dtype=torch.float32,
                    fp32_residual=True,
                )
                if get_global_server_args().rl_on_policy_target is not None
                else {}
            )
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
            )
        else:
            # 非末秩使用占位层
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        # EAGLE3 推测解码需要捕获中间层隐藏状态
        self.layers_to_capture = []

    # 获取输入嵌入向量，支持嵌入缩放（scale_emb）
    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(input_ids)

    # 返回词嵌入层
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    # 模型主体前向传播
    # 流程：词嵌入 -> 多层解码器 -> 层归一化
    # 支持流水线并行中间结果传递
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:

        # 首秩：从 input_ids 获取词嵌入或直接使用提供的 input_embeds
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            # 非首秩：从上游 PP 秩接收中间结果
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # 收集辅助隐藏状态（用于 EAGLE3 推测解码）
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            # 捕获指定层的隐藏状态用于 EAGLE3
            if i in self.layers_to_capture:
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
        # 非末秩：将中间结果打包传递给下游 PP 秩
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            # 末秩：应用最终层归一化
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        # 没有辅助隐藏状态时直接返回，否则一起返回
        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    # 加载 KV 缓存的量化缩放因子（用于 FP8 KV 缓存）
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            # 跳过被 PP 截断的层（Identity 层）
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                # 设置 K 和 V 缓存的缩放因子
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )


# Qwen2 因果语言模型
# 在 Qwen2Model 基础上添加语言模型头(lm_head)，用于生成词表上的 logits
# 支持词嵌入权重绑定(tie_word_embeddings)和 BitsAndBytes 量化
class Qwen2ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    # BitsAndBytes 量化的目标模块列表
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # BitsAndBytes 量化参数的堆叠映射关系
    # 将分散的 q/k/v/gate/up 投影映射到合并后的 qkv_proj/gate_up_proj
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        # 构建 Qwen2 模型主体
        self.model = Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        # 末秩才需要语言模型头，其他秩使用占位层
        if self.pp_group.is_last_rank:
            # 单卡且权重绑定时，lm_head 复用嵌入层权重
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                # 否则使用独立的并行语言模型头
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # logits 后处理器，处理采样温度、top-k/top-p 等
        self.logits_processor = LogitsProcessor(config)
        # 池化器，用于嵌入模型（如文本分类），取最后一个 token 的表示并归一化
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # For EAGLE3 support
        # EAGLE3 推测解码标志，为 True 时前向传播会返回中间层隐藏状态
        self.capture_aux_hidden_states = False

    # 获取输入嵌入向量
    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    # 返回词嵌入层
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    # 因果语言模型前向传播
    # 流程：模型主体 -> logits 处理 或 嵌入池化
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
        # 通过模型主体获取隐藏状态
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        # EAGLE3 模式下解包隐藏状态和辅助隐藏状态
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                # 正常生成模式：通过 lm_head 计算 logits
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                # 嵌入模式：通过池化器获取文本表示
                return self.pooler(hidden_states, forward_batch)
        else:
            # 非末秩返回中间结果供 PP 传递
            return hidden_states

    # 分段预填充前向传播
    # 将 prefill 阶段按层拆分为多个区间，支持更细粒度的调度
    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        # 在起始层执行词嵌入
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        # 逐层执行解码器计算
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
            # 在最后一层执行层归一化
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
            # 中间分段不产生最终结果
            result = None

        return result

    # 当前 PP 秩对应的起始层索引
    @property
    def start_layer(self):
        return self.model.start_layer

    # 当前 PP 秩对应的结束层索引
    @property
    def end_layer(self):
        return self.model.end_layer

    # 加载模型权重
    # 处理权重名映射（如 q_proj/k_proj/v_proj 合并为 qkv_proj）
    # 支持流水线并行裁剪、词嵌入权重绑定、GPTQ 额外偏置跳过等
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # 堆叠参数映射表：将 HuggingFace 中的分散权重名映射到合并后的参数名
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # 获取层索引，跳过不属于当前 PP 秩的层
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            # 词嵌入权重绑定时，将 embed_tokens 的权重同步到 lm_head
            if name == "model.embed_tokens.weight":
                if (
                    not hasattr(self, "pp_group") or self.pp_group.is_last_rank
                ) and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            # 跳过旋转位置编码的缓存张量（在运行时重新计算）
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # 跳过视觉编码器权重（Qwen2-VL 等多模态模型的视觉部分）
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            # 处理堆叠参数（qkv_proj 和 gate_up_proj）
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # 将分散的权重名替换为合并后的参数名
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                # GPTQ 模型可能有额外的偏置项，不在当前参数中则跳过
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                # 使用 weight_loader 加载对应的 shard
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # 非堆叠参数：直接加载
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    # 权重名在模型中找不到时发出警告
                    logger.warning(f"Parameter {name} not found in params_dict")

    # 获取词嵌入和语言模型头的权重（用于动态加载/切换）
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    # 设置词嵌入和语言模型头的权重（用于动态加载/切换）
    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        # 清理 GPU 显存并同步，确保旧权重被释放
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 加载 KV 缓存的量化缩放因子
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # 设置 EAGLE3 推测解码需要捕获的层
    # EAGLE3 需要中间层的隐藏状态作为推测解码的输入
    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        # 只有末秩才需要捕获辅助隐藏状态
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            # 默认捕获第 2 层、中间层和倒数第 3 层
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            # 用户指定的层 ID（+1 是因为 0 层是嵌入层）
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


# 模型入口类，SGLang 通过此名称查找和加载模型
EntryClass = Qwen2ForCausalLM
