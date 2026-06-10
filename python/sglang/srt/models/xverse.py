# XVERSE因果语言模型实现
# 本文件实现了仅推理的XVERSE模型，兼容HuggingFace权重。
# 包含MLP、注意力层、解码器层和完整模型。

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
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/xverse.py#L1
"""Inference-only XVERSE model compatible with HuggingFace weights."""  # 仅推理的XVERSE模型，兼容HuggingFace权重

from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块
from transformers import LlamaConfig  # 导入LLaMA配置

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入TP世界大小获取函数
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU和乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.model_runner import ForwardBatch  # 导入前向批次
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入RoPE配置工具


class XverseMLP(nn.Module):
    """XVERSE模型的MLP模块"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控上投影合并层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小列表
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # 下投影层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
        )
        if hidden_act != "silu":  # 仅支持SiLU
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU和乘法激活

    def forward(self, x):
        """MLP前向传播"""
        gate_up, _ = self.gate_up_proj(x)  # 通过门控上投影
        x = self.act_fn(gate_up)  # 应用激活
        x, _ = self.down_proj(x)  # 通过下投影
        return x  # 返回输出


class XverseAttention(nn.Module):
    """XVERSE模型的注意力模块"""

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID
        rope_theta: float = 10000,  # RoPE theta
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放
        rope_is_neox_style: bool = True,  # 是否Neox风格RoPE
        max_position_embeddings: int = 8192,  # 最大位置编码
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        self.total_num_heads = num_heads  # 总头数
        assert self.total_num_heads % tp_size == 0  # 断言可整除
        self.num_heads = self.total_num_heads // tp_size  # TP后头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0  # 断言可整除
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0  # 断言可整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # TP后KV头数
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(  # 获取头维度
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim  # Q大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 保存RoPE theta
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置编码

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,  # 不使用偏置
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(  # 基数注意力
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
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """注意力前向传播"""
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分离Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力
        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class XverseDecoderLayer(nn.Module):
    """XVERSE解码器层"""

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置
        if rope_scaling is not None and getattr(  # 如果有缩放和原始最大位置
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 设置原始最大位置
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)  # RoPE风格
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置
        num_kv_heads = getattr(  # KV头数
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.self_attn = XverseAttention(  # 自注意力层
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = XverseMLP(  # MLP层
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播"""
        # Self Attention
        if residual is None:  # 无残差
            residual = hidden_states  # 设置残差
            hidden_states = self.input_layernorm(hidden_states)  # 归一化
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 带残差归一化
        hidden_states = self.self_attn(  # 通过自注意力
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP
        return hidden_states, residual  # 返回隐藏状态和残差


class XverseModel(nn.Module):
    """XVERSE模型主体"""

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                XverseDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> torch.Tensor:
        """模型主体前向传播"""
        if input_embeds is None:  # 无输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层
        else:
            hidden_states = input_embeds  # 使用输入嵌入
        residual = None  # 初始化残差
        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
            # print(f"layer[{i}].hidden_states: {hidden_states}")
        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化
        return hidden_states  # 返回隐藏状态


class XverseForCausalLM(nn.Module):
    """XVERSE因果语言模型"""

    def __init__(
        self,
        config: LlamaConfig,  # LLaMA配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = XverseModel(  # 创建模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 并行LM头
            config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> torch.Tensor:
        """因果语言模型前向传播"""
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型
        return self.logits_processor(  # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], name=None, loaded_weight=None
    ):
        """加载模型权重"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),  # Q投影
            ("qkv_proj", "k_proj", "k"),  # K投影
            ("qkv_proj", "v_proj", "v"),  # V投影
            ("gate_up_proj", "gate_proj", 0),  # 门控投影
            ("gate_up_proj", "up_proj", 1),  # 上投影
        ]
        params_dict = dict(self.named_parameters())  # 参数字典

        def load_weights_per_param(name, loaded_weight):  # 单参数加载函数
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转频率和投影器
                return
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                return
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射
                if weight_name not in name:  # 不匹配
                    continue
                name = name.replace(weight_name, param_name)  # 替换名称
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ偏置
                    continue
                if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过视觉塔
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 非堆叠参数
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ偏置
                    return
                if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过视觉塔
                    return
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取加载器
                weight_loader(param, loaded_weight)  # 加载权重

        if name is None or loaded_weight is None:  # 批量加载
            for name, loaded_weight in weights:  # 遍历权重
                load_weights_per_param(name, loaded_weight)  # 加载单个权重
        else:  # 单个加载
            load_weights_per_param(name, loaded_weight)  # 加载单个权重


EntryClass = XverseForCausalLM  # 入口类
