# GPT-J模型推理实现文件
# 本文件实现了仅用于推理的GPT-J模型，兼容HuggingFace权重格式
# 包含GPT-J注意力层、MLP层、Transformer块及因果语言模型等核心组件

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023-2025 SGLang Team
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
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gpt_j.py
"""Inference-only GPT-J model compatible with HuggingFace weights."""  # 仅推理用的GPT-J模型，兼容HuggingFace权重

from typing import Iterable, Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import GPTJConfig  # 导入GPT-J配置类

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取工具
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称
)
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class GPTJAttention(nn.Module):  # GPT-J注意力层类

    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: GPTJConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        total_num_heads = config.num_attention_heads  # 总注意力头数
        hidden_size = config.hidden_size  # 隐藏层大小
        head_dim = hidden_size // total_num_heads  # 每个头的维度

        self.qkv_proj = QKVParallelLinear(  # QKV投影线性层
            hidden_size,  # 输入维度
            head_dim,  # 每个头的维度
            total_num_heads,  # 总头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.out_proj = RowParallelLinear(  # 输出投影线性层（行并行）
            hidden_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("out_proj", prefix),  # 参数前缀
        )

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        assert total_num_heads % tensor_model_parallel_world_size == 0  # 断言总头数可被并行世界大小整除
        num_heads = total_num_heads // tensor_model_parallel_world_size  # 每个并行分片的头数

        scaling = head_dim**-0.5  # 缩放因子
        assert getattr(config, "rotary", True)  # 断言使用旋转位置编码
        assert config.rotary_dim % 2 == 0  # 断言旋转维度为偶数
        rope_theta = getattr(config, "rope_theta", 10000)  # RoPE基准角度，默认10000
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入数，默认8192
        self.rotary_emb = get_rope(  # 旋转位置编码
            head_dim,  # 头维度
            rotary_dim=config.rotary_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基准角度
            is_neox_style=False,  # 非Neox风格（GPT-J风格）
        )
        self.attn = RadixAttention(  # 基数注意力实现
            num_heads,  # 注意力头数
            head_dim,  # 每个头的维度
            scaling=scaling,  # 缩放因子
            num_kv_heads=num_heads,  # KV头数（与Q头数相同）
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影层
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # 分割为Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        attn_output, _ = self.out_proj(attn_output)  # 通过输出投影层
        return attn_output  # 返回注意力输出


class GPTJMLP(nn.Module):  # GPT-J MLP层类

    def __init__(  # 初始化函数
        self,
        intermediate_size: int,  # 中间层大小
        config: GPTJConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        hidden_size = config.n_embd  # 隐藏层大小
        self.fc_in = ColumnParallelLinear(  # 上投影线性层（列并行）
            hidden_size,  # 输入维度
            intermediate_size,  # 输出维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc_in", prefix),  # 参数前缀
        )
        self.fc_out = RowParallelLinear(  # 下投影线性层（行并行）
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc_out", prefix),  # 参数前缀
        )

        self.act = get_act_fn(config.activation_function)  # 获取激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        hidden_states, _ = self.fc_in(hidden_states)  # 上投影
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states, _ = self.fc_out(hidden_states)  # 下投影
        return hidden_states  # 返回MLP输出


class GPTJBlock(nn.Module):  # GPT-J Transformer块类

    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: GPTJConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        inner_dim = 4 * config.n_embd if config.n_inner is None else config.n_inner  # 中间维度，默认4倍隐藏维度
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)  # LayerNorm层
        self.attn = GPTJAttention(  # 注意力层
            layer_id,  # 层ID
            config,  # 模型配置
            quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )
        self.mlp = GPTJMLP(  # MLP层
            inner_dim,  # 中间维度
            config,  # 模型配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.ln_1(hidden_states)  # LayerNorm
        attn_output = self.attn(  # 注意力计算
            positions=positions,  # 位置
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 批次信息
        )
        mlp_output = self.mlp(hidden_states)  # MLP计算
        hidden_states = attn_output + mlp_output + residual  # 注意力+MLP+残差（并行结构）
        return hidden_states  # 返回块输出


class GPTJModel(nn.Module):  # GPT-J模型类

    def __init__(  # 初始化函数
        self,
        config: GPTJConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        embed_dim = config.n_embd  # 嵌入维度
        self.wte = VocabParallelEmbedding(  # 词嵌入层（词表并行）
            config.vocab_size,  # 词汇表大小
            embed_dim,  # 嵌入维度
        )
        self.h = nn.ModuleList(  # Transformer块列表
            [
                GPTJBlock(  # 每个Transformer块
                    i,  # 层索引
                    config,  # 模型配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"h.{i}", prefix),  # 参数前缀
                )
                for i in range(config.n_layer)  # 遍历所有层
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)  # 最终LayerNorm层

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:  # 获取输入嵌入
        return self.wte(input_ids)  # 通过词嵌入层获取嵌入

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 可选的输入嵌入
    ) -> torch.Tensor:
        if inputs_embeds is not None:  # 如果提供了输入嵌入
            hidden_states = inputs_embeds  # 直接使用输入嵌入
        else:  # 否则通过词嵌入层计算
            hidden_states = self.get_input_embeddings(input_ids)  # 获取词嵌入

        for layer in self.h:  # 遍历所有Transformer块
            hidden_states = layer(positions, hidden_states, forward_batch)  # 通过当前层
        hidden_states = self.ln_f(hidden_states)  # 最终LayerNorm
        return hidden_states  # 返回隐藏状态


class GPTJForCausalLM(nn.Module):  # GPT-J因果语言模型类

    def __init__(  # 初始化函数
        self,
        config: GPTJConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        assert not config.tie_word_embeddings  # 断言不共享词嵌入权重
        self.quant_config = quant_config  # 保存量化配置
        self.transformer = GPTJModel(  # GPT-J模型主体
            config,  # 模型配置
            quant_config,  # 量化配置
            prefix=add_prefix("transformer", prefix),  # 参数前缀
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词汇表大小
            config.n_embd,  # 嵌入维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        inputs_embeds: Optional[torch.Tensor] = None,  # 可选的输入嵌入
    ) -> torch.Tensor:
        hidden_states = self.transformer(  # 通过Transformer获取隐藏状态
            input_ids, positions, forward_batch, inputs_embeds  # 传入输入ID、位置、批次信息和嵌入
        )
        return self.logits_processor(  # 通过logits处理器获取logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 传入输入ID、隐藏状态、语言模型头和批次信息
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q投影映射
            ("qkv_proj", "k_proj", "k"),  # K投影映射
            ("qkv_proj", "v_proj", "v"),  # V投影映射
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "attn.bias" in name or "attn.masked_bias" in name:  # 跳过注意力偏置和掩码偏置
                continue  # 继续

            if self.quant_config is not None and (  # 如果有量化配置
                scale_name := self.quant_config.get_cache_scale(name)  # 获取缓存缩放名称
            ):
                # Loading kv cache quantization scales  # 加载KV缓存量化缩放因子
                param = params_dict[scale_name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                loaded_weight = (  # 处理权重维度
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]  # 0维直接使用，否则取第一个元素
                )
                weight_loader(param, loaded_weight)  # 加载权重
                continue  # 继续下一个权重

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在名称中
                    continue  # 继续
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 继续
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 按分片加载权重
                break  # 跳出内层循环
            else:  # 如果没有匹配的堆叠参数映射
                name = maybe_remap_kv_scale_name(name, params_dict)  # 可能重映射KV缩放名称
                if name is None:  # 如果名称为None
                    continue  # 继续
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 继续
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = GPTJForCausalLM  # 入口类为GPTJForCausalLM
