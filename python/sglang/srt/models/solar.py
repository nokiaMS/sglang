# Solar 模型推理实现文件
# 本文件实现了Solar模型（基于Llama架构的深度上扩模型）
# 支持骨干跳跃连接(BSKCN)实现深度上扩机制
# 包含MLP、注意力、解码层、模型和权重加载等核心组件

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/solar.py
from collections.abc import Iterable  # 导入可迭代类型
from typing import Any, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size  # 导入分布式
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank  # 导入TP秩
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU激活
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化
from sglang.srt.layers.linear import (  # 导入线性层
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入逻辑处理器
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer  # 导入PP缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次
from sglang.srt.model_loader.weight_utils import (  # 导入权重工具
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.utils import add_prefix, make_layers  # 导入工具函数
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入RoPE配置


class SolarMLP(nn.Module):
    """Solar MLP模块"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        bias: bool = False,  # 偏置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化Solar MLP"""
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """MLP前向传播"""
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class SolarAttention(nn.Module):
    """Solar注意力模块"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 头数
        num_kv_heads: int,  # KV头数
        rope_theta: float = 10000,  # RoPE基础频率
        rope_scaling: Optional[dict[str, Any]] = None,  # RoPE缩放
        max_position_embeddings: int = 8192,  # 最大位置编码
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        bias: bool = False,  # 偏置
        prefix: str = "",  # 前缀
        layer_id: int = 0,  # 层ID
    ) -> None:
        """初始化Solar注意力"""
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(config, "head_dim", None)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        hidden_states: torch.Tensor,  # 隐藏状态
    ) -> torch.Tensor:
        """Solar注意力前向传播"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class SolarDecoderLayer(nn.Module):
    """Solar解码器层"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化Solar解码器层"""
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta, rope_scaling = get_rope_config(config)

        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = SolarAttention(
            config=config,
            layer_id=layer_id,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = SolarMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solar解码器层前向传播"""
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SolarModel(nn.Module):
    """Solar模型，支持深度上扩机制"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化Solar模型"""
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SolarDecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输入嵌入"""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        """Solar模型前向传播，支持BSKCN深度上扩"""
        if self.pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None

            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # Depth up-scaling mechanism: caches hidden states and residuals from intermediate layers and interpolates them with the states of later layers.
        # `bskcn` stands for "backbone skip connection".
        bskcn_h_1 = None  # 骨干跳跃连接隐藏状态1
        bskcn_h_2 = None  # 骨干跳跃连接隐藏状态2
        bskcn_r_1 = None  # 骨干跳跃连接残差1
        bskcn_r_2 = None  # 骨干跳跃连接残差2
        bskcn_tv = self.config.bskcn_tv[0] if self.training else self.config.bskcn_tv[1]  # 插值权重

        for i in range(self.start_layer, self.end_layer):
            if i in self.config.bskcn_1:  # 缓存点1
                bskcn_h_1 = hidden_states.clone()
                bskcn_r_1 = residual.clone() if residual is not None else None
            if i in self.config.bskcn_2:  # 缓存点2
                bskcn_h_2 = hidden_states.clone()
                bskcn_r_2 = residual.clone() if residual is not None else None
            if i in self.config.bskcn_3:  # 插值点1
                hidden_states = bskcn_h_1 * bskcn_tv + hidden_states * (1 - bskcn_tv)
                if bskcn_r_1 is not None and residual is not None:
                    residual = bskcn_r_1 * bskcn_tv + residual * (1 - bskcn_tv)
            if i in self.config.bskcn_4:  # 插值点2
                hidden_states = bskcn_h_2 * bskcn_tv + hidden_states * (1 - bskcn_tv)
                if bskcn_r_2 is not None and residual is not None:
                    residual = bskcn_r_2 * bskcn_tv + residual * (1 - bskcn_tv)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if not self.pp_group().is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """加载KV缓存缩放因子"""
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
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )


class SolarForCausalLM(nn.Module):
    """Solar因果语言模型"""

    packed_modules_mapping = {
        "qkv_proj": [
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
        ],
        "gate_up_proj": [
            ("gate_proj", 0),
            ("up_proj", 1),
        ],
    }

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化Solar因果语言模型"""
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = SolarModel(
            config=config,
            quant_config=self.quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
            )
            if config.tie_word_embeddings and self.pp_group.is_first_rank:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, config.vocab_size, logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
    ) -> Union[torch.Tensor, LogitsProcessorOutput]:
        """Solar因果语言模型前向传播"""
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=inputs_embeds,
        )

        if self.pp_group().is_last_rank:
            logits = self.logits_processor(self.lm_head, hidden_states, forward_batch)
            return logits

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """加载模型权重"""
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            is_packed = False
            for packed_name, sources in self.packed_modules_mapping.items():
                for src_name, shard_id in sources:
                    if src_name in name:

                        model_param_name = name.replace(src_name, packed_name)

                        if model_param_name in params_dict:
                            param = params_dict[model_param_name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight, shard_id)
                            is_packed = True
                            break
                if is_packed:
                    break

            if is_packed:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = SolarForCausalLM  # 模型注册入口类
