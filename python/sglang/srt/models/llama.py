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
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

# 本文件实现了仅推理用的 LLaMA 模型，兼容 HuggingFace 权重格式。
# 包含 LLaMA 模型的核心组件：MLP层、注意力层、解码器层、模型主体及因果语言模型。
# 同时支持 Phi3、InternLM3、IQuestCoder 等基于 LLaMA 架构的变体模型。
# 支持张量并行(TP)、流水线并行(PP)、数据并行(DP)等分布式推理策略。

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
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
    maybe_remap_kv_scale_name,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu, is_xpu, make_layers
from sglang.utils import get_exception_traceback

# 检测当前运行设备类型
_is_cuda = is_cuda()
_is_xpu = is_xpu()

logger = logging.getLogger(__name__)
_is_npu = is_npu()

if _is_npu:
    # NPU 设备使用融合的 QKV 分割、RMSNorm 和 RoPE 算子
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope


# LLaMA 模型的 MLP（前馈网络）层，实现 gate-up 投影 + SiLU 激活 + down 投影
class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_dp_attention_reduce: bool = False,
    ) -> None:
        super().__init__()
        # gate_up_proj 将隐藏状态投影到 2 倍中间维度（gate 和 up 拼接）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # down_proj 将中间维度投影回隐藏维度
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_dp_attention_reduce=use_dp_attention_reduce,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        # SiLU 激活函数并与门控相乘（SiLUAndMul）
        self.act_fn = SiluAndMul()

    # MLP 前向传播：输入 -> gate_up 投影 -> SiLU 激活 -> down 投影 -> 输出
    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
    ):
        # gate_up_proj 输出拼接了 gate 和 up 两部分
        gate_up, _ = self.gate_up_proj(x)
        # SiLUAndMul: 对 gate 部分应用 SiLU 后与 up 部分逐元素相乘
        x = self.act_fn(gate_up)
        # down_proj 将结果投影回隐藏维度
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter,
        )
        return x


# LLaMA 模型的注意力层，实现多头注意力机制（支持 GQA）
class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.start_layer = start_layer
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        # 注意力头数必须能被 TP 大小整除
        assert self.total_num_heads % tp_size == 0
        # 每个 TP rank 分到的查询头数
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
        # 每个 TP rank 分到的 KV 头数，至少为 1
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        # 每个注意力头的维度，默认为 hidden_size / num_heads
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        # 部分旋转因子，用于旋转位置编码中只对部分维度做旋转
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        # Q 和 KV 的维度大小
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放因子，等于 1/sqrt(head_dim)
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # QKV 联合线性投影，支持张量并行
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
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # 旋转位置编码（RoPE）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        # 基数树注意力机制，用于高效的 KV 缓存管理
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    # 原生的 QKV 准备流程：投影 + 分割 + 应用 RoPE
    def forward_prepare_native(self, positions, hidden_states):
        # QKV 联合投影
        qkv, _ = self.qkv_proj(hidden_states)
        # 将 QKV 拆分为 Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 对 Q 和 K 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    # NPU 设备上的 QKV 准备流程，使用融合算子加速
    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        # 在起始层预计算 cos/sin 缓存
        if self.attn.layer_id == self.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        # 使用 NPU 融合算子同时完成 QKV 分割、RMSNorm 和 RoPE
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            is_neox_style=self.rotary_emb.is_neox_style,
        )
        return q, k, v

    # 注意力层前向传播：根据设备类型选择 QKV 准备方式，然后执行注意力计算
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 非 NPU 设备，或 NPU 上 extend 模式使用原生流程
        if (
            not _is_npu
            or not hasattr(self.rotary_emb, "get_cos_sin_with_position")
            or forward_batch.forward_mode.is_extend()
        ):
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            # NPU 设备上的 decode 模式使用融合算子
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # 执行注意力计算
        attn_output = self.attn(q, k, v, forward_batch)
        # 输出投影
        output, _ = self.o_proj(attn_output)
        return output


# LLaMA 解码器层，包含自注意力 + MLP + 残差连接 + LayerNorm
class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # 解析 RoPE 参数，支持 rope_parameters（如 Gemma 等模型）和传统 rope_theta/rope_scaling
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        # 如果有 rope_scaling 且配置中指定了原始最大位置编码数，则传入
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied with attention_bias
        # Support internlm/internlm-7b with bias
        # 支持带偏置项的注意力（如 Qwen2.5、InternLM 等变体）
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        # 自注意力子层
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=attention_bias,
        )
        # MLP（前馈网络）子层
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        # 自注意力前的 RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP 前的 RMSNorm（后注意力层归一化）
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    # 解码器层前向传播：Pre-Norm 架构，残差连接在归一化之前
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # 第一个子层没有残差，直接赋值
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 融合 RMSNorm 和残差加法
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 自注意力计算
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        # 后注意力归一化 + 残差
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # MLP 前馈计算
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# LLaMA 模型主体，由嵌入层 + 多个解码器层 + 最终归一化层组成
class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # 获取流水线并行组信息
        self.pp_group = get_pp_group()
        # 仅在 PP 第一个 rank 上创建词嵌入层，其余 rank 跳过以节省显存
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # 获取当前 PP rank 负责的起始层索引
        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        # 按流水线并行分配解码器层
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LlamaDecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                start_layer=pp_start_layer,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        # 仅在 PP 最后一个 rank 上创建最终归一化层
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        # 需要捕获辅助隐藏状态的层列表（用于推测解码等）
        self.layers_to_capture = []

    # LLaMA 模型前向传播：嵌入 -> 解码器层循环 -> 归一化
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        # PP 第一个 rank：从 input_ids 生成嵌入
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            # 非第一个 PP rank：从代理张量中恢复隐藏状态和残差
            assert pp_proxy_tensors is not None
            # FIXME(@ying): reduce the number of proxy tensors by not fusing layer norms
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            deferred_norm = None

        # 收集辅助隐藏状态（用于推测解码）
        aux_hidden_states = []
        # 逐层执行解码器层
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                # 捕获当前层的残差连接结果作为辅助隐藏状态
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        # 非 PP 最后一个 rank：返回代理张量给下一个 PP stage
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            # PP 最后一个 rank：执行最终 RMSNorm
            hidden_states, _ = self.norm(hidden_states, residual)

        # 无辅助隐藏状态时直接返回
        if len(aux_hidden_states) == 0:
            return hidden_states

        # 返回最终隐藏状态和辅助隐藏状态
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
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            # 设置 KV 缓存的缩放因子
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings from the model."""
        return self.embed_tokens


# LLaMA 因果语言模型，封装模型主体 + LM Head + logits 处理
class LlamaForCausalLM(nn.Module):
    # BitandBytes specific attributes
    # BitsAndBytes 量化目标模块列表
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    # 张量并行中按列维度分割的权重模块
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    # BitsAndBytes 堆叠参数映射：将分散的 q/k/v 和 gate/up 映射到合并后的参数
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        # 初始化 LLaMA 模型主体
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        # 如果词嵌入和 LM Head 权重共享，则直接复用嵌入权重
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            # 否则单独创建 LM Head
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        # logits 处理器
        self.logits_processor = LogitsProcessor(config)
        # 池化器，用于嵌入任务（取最后一个 token 的隐藏状态并归一化）
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # 堆叠参数映射：将分散的权重名映射到合并后的参数名
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # 是否捕获辅助隐藏状态（用于推测解码如 EAGLE）
        self.capture_aux_hidden_states = False

    # 初始化模型主体，子类可覆盖以使用不同的模型类
    def _init_model(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return LlamaModel(config, quant_config=quant_config, prefix=prefix)

    # 因果语言模型前向传播：模型主体 -> logits 处理 / 池化
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        # 执行模型主体前向传播
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # 解包辅助隐藏状态
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        # PP 最后一个 rank 执行 logits 处理或池化
        if self.pp_group.is_last_rank:
            if not get_embedding:
                # 生成 logits
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                # 嵌入模式：池化得到句子嵌入
                return self.pooler(hidden_states, forward_batch)
        else:
            # 非最后一个 PP rank 直接返回隐藏状态
            return hidden_states

    # 分段预填充前向传播：将 prefill 拆分为多个区间逐段执行，降低峰值显存
    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval
        # embed
        # 在起始位置执行嵌入查找
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        # 逐层执行解码器层
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        # 在最后一段执行归一化和 logits 处理
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
            # 非最后一段返回 None
            result = None

        return result

    # 当前 PP rank 负责的起始层索引
    @property
    def start_layer(self):
        return self.model.start_layer

    # 当前 PP rank 负责的结束层索引
    @property
    def end_layer(self):
        return self.model.end_layer

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    # 根据权重名获取对应的模块名（处理堆叠参数映射）
    def get_module_name_from_weight_name(self, name):
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    # 获取模型参数数量
    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    # 加载模型权重，处理堆叠参数映射和各种特殊情况
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # 兼容性重命名：activation_scale -> input_scale
            if name.endswith(".activation_scale"):
                name = name.replace(".activation_scale", ".input_scale")
            # 兼容性重命名：weight_scale_inv -> weight_scale
            if name.endswith(".weight_scale_inv"):
                name = name.replace(".weight_scale_inv", ".weight_scale")

            # 跳过不属于当前 PP rank 的层权重
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
            # 跳过 RoPE 逆频率和投影器权重
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # 跳过视觉塔中不在参数字典里的权重
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            # 词嵌入和 LM Head 共享权重时跳过 lm_head.weight
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            # 处理 FP8 KV 缓存缩放因子的名称重映射
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # 处理堆叠参数（qkv_proj, gate_up_proj）
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # 将分散的权重名替换为合并后的参数名
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                # 使用 weight_loader 加载堆叠参数的分片
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading kv_scale from ckpts towards new design.
                # 跳过旧版 KV 缓存缩放因子
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    # 加载非堆叠参数
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    # 根据参数名获取权重值（主要用于单元测试）
    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        try:
            # 词嵌入和 LM Head 共享权重时，返回嵌入权重
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                logger.info(
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."
                )
                return (
                    self.model.embed_tokens.weight.cpu()
                    .to(torch.float32)
                    .numpy()
                    .tolist()[:truncate_size]
                )

            # 处理堆叠参数映射
            mapped_name = name
            mapped_shard_id = None
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    mapped_shard_id = shard_id
                    break
            params_dict = dict(self.named_parameters())
            param = params_dict[mapped_name]
            if mapped_shard_id is not None:
                # 处理 QKV 堆叠参数的分片提取
                if mapped_shard_id in ["q", "k", "v"]:
                    num_heads = self.config.num_attention_heads // tp_size
                    num_kv_heads = self.config.num_key_value_heads // tp_size
                    head_dim = (
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    # 根据 shard 类型计算偏移和大小
                    if mapped_shard_id == "q":
                        offset = 0
                        size = num_heads * head_dim
                    elif mapped_shard_id == "k":
                        offset = num_heads * head_dim
                        size = num_kv_heads * head_dim
                    elif mapped_shard_id == "v":
                        offset = (num_heads + num_kv_heads) * head_dim
                        size = num_kv_heads * head_dim
                    # 按偏移和大小截取对应分片
                    weight = param.data.narrow(0, offset, size)
                # 处理 gate/up 堆叠参数的分片提取
                elif mapped_shard_id in [0, 1]:
                    intermediate_size = self.config.intermediate_size
                    slice_size = intermediate_size // tp_size
                    if mapped_shard_id == 0:  # gate_proj
                        offset = 0
                        size = slice_size
                    elif mapped_shard_id == 1:  # up_proj
                        offset = slice_size
                        size = slice_size

                    weight = param.data.narrow(0, offset, size)
                else:
                    weight = param.data
            else:
                weight = param.data
            # 对于行并行权重（o_proj, down_proj），需要 all-gather 收集完整权重
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered_weights, weight)
                weight = torch.cat(gathered_weights, dim=1)
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]

        except Exception:
            logger.error(
                f"Error getting weights by name {name} in LlamaForCausalLM: {get_exception_traceback()}"
            )
            return None

    # 获取嵌入权重和 LM Head 权重
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    # 设置嵌入权重和 LM Head 权重，并清理 GPU 缓存
    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        # 清理 GPU 显存缓存
        if _is_xpu:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # 获取嵌入层权重
    def get_embed(self):
        return self.model.embed_tokens.weight

    # 设置嵌入层权重，并清理 GPU 缓存
    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        # EAGLE3 推测解码中，若草稿模型和目标模型隐藏维度不同则不能共享嵌入
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        if _is_xpu:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # 加载 KV 缓存缩放因子
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # 设置 EAGLE3 推测解码需要捕获辅助隐藏状态的层
    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            # 默认捕获第 2 层、中间层和倒数第 3 层
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            # 加 1 是因为 sglang 中第 i 层使用第 i-1 层的输出作为辅助隐藏状态
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    # 设置 DFLASH 推测解码需要捕获辅助隐藏状态的层
    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


# Phi3 模型，复用 LLaMA 架构
class Phi3ForCausalLM(LlamaForCausalLM):
    pass


# InternLM3 模型，复用 LLaMA 架构
class InternLM3ForCausalLM(LlamaForCausalLM):
    pass


# IQuestCoder 模型，复用 LLaMA 架构
class IQuestCoderForCausalLM(LlamaForCausalLM):
    pass


# 模型入口类列表，框架根据模型类型自动选择对应的类
EntryClass = [
    LlamaForCausalLM,
    Phi3ForCausalLM,
    InternLM3ForCausalLM,
    IQuestCoderForCausalLM,
]
