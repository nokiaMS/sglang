# GPTBigCode模型推理实现文件
# 本文件实现了仅用于推理的GPTBigCode模型，兼容HuggingFace权重格式
# 包含GPTBigCode注意力层、MLP层、Transformer块及因果语言模型等核心组件

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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/07eb6f19f3b0ee9f7adf6eb689607028aa40bfd5/vllm/model_executor/models/gpt_bigcode.py
"""Inference-only GPTBigCode model compatible with HuggingFace weights."""  # 仅推理用的GPTBigCode模型，兼容HuggingFace权重

from typing import Iterable, Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import GPTBigCodeConfig  # 导入GPTBigCode配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取工具
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class GPTBigCodeAttention(nn.Module):  # GPTBigCode注意力层类

    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: GPTBigCodeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        total_num_heads = config.num_attention_heads  # 总注意力头数
        self.tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        assert total_num_heads % self.tensor_model_parallel_world_size == 0  # 断言总头数可被并行世界大小整除
        self.num_heads = total_num_heads // self.tensor_model_parallel_world_size  # 每个并行分片的注意力头数
        self.head_dim = self.hidden_size // total_num_heads  # 每个头的维度
        self.scale = self.head_dim**-0.5  # 缩放因子，头维度的-0.5次方

        self.multi_query = config.multi_query  # 是否使用多查询（Multi-Query Attention）
        if self.multi_query:  # 如果使用多查询注意力
            total_num_kv_heads = 1  # KV头数为1
            self.num_kv_heads = 1  # 当前分片的KV头数为1
        else:  # 否则使用标准多头注意力
            total_num_kv_heads = total_num_heads  # KV头数等于总注意力头数
            self.num_kv_heads = self.num_heads  # 当前分片的KV头数等于当前头数
        self.kv_dim = self.head_dim * self.num_kv_heads  # KV维度 = 头维度 * KV头数
        self.c_attn = QKVParallelLinear(  # QKV投影线性层
            self.hidden_size,  # 输入隐藏维度
            self.head_dim,  # 每个头的维度
            total_num_heads,  # 总Q头数
            total_num_kv_heads,  # 总KV头数
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_attn", prefix),  # 参数前缀
        )

        self.c_proj = RowParallelLinear(  # 输出投影线性层（行并行）
            self.hidden_size,  # 输入维度
            self.hidden_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_proj", prefix),  # 参数前缀
        )
        self.attn = RadixAttention(  # 基数注意力实现
            self.num_heads,  # 注意力头数
            self.head_dim,  # 每个头的维度
            scaling=self.scale,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

    def forward(  # 前向传播函数
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)  # 通过QKV投影层获取QKV
        q, k, v = qkv.split(  # 按维度分割QKV
            [
                self.hidden_size // self.tensor_model_parallel_world_size,  # Q的维度
                self.kv_dim,  # K的维度
                self.kv_dim,  # V的维度
            ],
            dim=-1,  # 在最后一个维度上分割
        )
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        attn_output, _ = self.c_proj(attn_output)  # 通过输出投影层
        return attn_output  # 返回注意力输出


class GPTBigMLP(nn.Module):  # GPTBigCode MLP层类

    def __init__(  # 初始化函数
        self,
        intermediate_size: int,  # 中间层大小
        config: GPTBigCodeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        hidden_size = config.hidden_size  # 隐藏层大小
        self.c_fc = ColumnParallelLinear(  # 上投影线性层（列并行）
            hidden_size,  # 输入维度
            intermediate_size,  # 输出维度（中间层大小）
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_fc", prefix),  # 参数前缀
        )
        self.c_proj = RowParallelLinear(  # 下投影线性层（行并行）
            intermediate_size,  # 输入维度（中间层大小）
            hidden_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_proj", prefix),  # 参数前缀
        )
        self.act = get_act_fn(  # 获取激活函数
            config.activation_function, quant_config, intermediate_size  # 根据配置获取激活函数
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        hidden_states, _ = self.c_fc(hidden_states)  # 上投影
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states, _ = self.c_proj(hidden_states)  # 下投影
        return hidden_states  # 返回MLP输出


class GPTBigCodeBlock(nn.Module):  # GPTBigCode Transformer块类

    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: GPTBigCodeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        hidden_size = config.hidden_size  # 隐藏层大小
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size  # 中间维度，默认为4倍隐藏维度

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第一层LayerNorm
        self.attn = GPTBigCodeAttention(  # 注意力层
            layer_id, config, quant_config, prefix=add_prefix("attn", prefix)  # 传入层ID、配置和量化配置
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第二层LayerNorm
        self.mlp = GPTBigMLP(  # MLP层
            inner_dim, config, quant_config, prefix=add_prefix("mlp", prefix)  # 传入中间维度、配置和量化配置
        )

    def forward(  # 前向传播函数
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.ln_1(hidden_states)  # 第一个LayerNorm
        attn_output = self.attn(  # 注意力计算
            hidden_states=hidden_states, forward_batch=forward_batch  # 传入隐藏状态和批次信息
        )
        # residual connection  # 残差连接
        hidden_states = attn_output + residual  # 注意力输出加残差

        residual = hidden_states  # 保存新的残差
        hidden_states = self.ln_2(hidden_states)  # 第二个LayerNorm
        feed_forward_hidden_states = self.mlp(hidden_states)  # MLP前馈计算
        # residual connection  # 残差连接
        hidden_states = residual + feed_forward_hidden_states  # MLP输出加残差
        return hidden_states  # 返回块输出


class GPTBigCodeModel(nn.Module):  # GPTBigCode模型类

    def __init__(  # 初始化函数
        self,
        config: GPTBigCodeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        assert not config.add_cross_attention  # 断言不使用交叉注意力

        self.embed_dim = config.hidden_size  # 嵌入维度
        lora_vocab = 0  # LoRA词汇表大小，暂设为0
        self.vocab_size = config.vocab_size + lora_vocab  # 总词汇表大小
        self.wte = VocabParallelEmbedding(  # 词嵌入层（词表并行）
            self.vocab_size,  # 词汇表大小
            self.embed_dim,  # 嵌入维度
            org_num_embeddings=config.vocab_size,  # 原始嵌入数量
            prefix=add_prefix("wte", prefix),  # 参数前缀
        )
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)  # 位置嵌入层
        self.h = nn.ModuleList(  # Transformer块列表
            [
                GPTBigCodeBlock(  # 每个Transformer块
                    i, config, quant_config, prefix=add_prefix(f"h.{i}", prefix)  # 传入层索引、配置和量化配置
                )
                for i in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)  # 最终LayerNorm层

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        position_ids: torch.Tensor,  # 位置ID张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)  # 词嵌入
        position_embeds = self.wpe(position_ids)  # 位置嵌入
        hidden_states = inputs_embeds + position_embeds  # 词嵌入加位置嵌入

        for i in range(len(self.h)):  # 遍历所有Transformer块
            layer = self.h[i]  # 获取当前层
            hidden_states = layer(hidden_states, forward_batch)  # 通过当前层

        hidden_states = self.ln_f(hidden_states)  # 最终LayerNorm
        return hidden_states  # 返回隐藏状态


class GPTBigCodeForCausalLM(nn.Module):  # GPTBigCode因果语言模型类
    packed_modules_mapping = {"c_attn": ["c_attn"]}  # 打包模块映射

    supported_lora_modules = ["c_fc", "c_proj", "wte", "c_attn"]  # 支持的LoRA模块列表

    embedding_modules = {  # 嵌入模块映射
        "wte": "input_embeddings",  # 词嵌入 -> 输入嵌入
        "lm_head": "output_embeddings",  # 语言模型头 -> 输出嵌入
    }

    embedding_padding_modules = []  # 嵌入填充模块列表

    def __init__(  # 初始化函数
        self,
        config: GPTBigCodeConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        prefix: str = "",  # 参数前缀，默认为空字符串
    ):
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        self.quant_config = quant_config  # 保存量化配置
        self.transformer = GPTBigCodeModel(  # GPTBigCode模型主体
            config, quant_config, prefix=add_prefix("transformer", prefix)  # 传入配置和量化配置
        )
        self.lm_head = self.transformer.wte  # 语言模型头（与词嵌入共享权重）
        self.unpadded_vocab_size = config.vocab_size  # 未填充的词汇表大小
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()  # 禁用梯度计算装饰器
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)  # 通过Transformer获取隐藏状态
        return self.logits_processor(  # 通过logits处理器获取logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 传入输入ID、隐藏状态、语言模型头和批次信息
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "lm_head.weight" in name:  # 跳过lm_head权重（与wte共享）
                continue  # 继续
            if ".attn.bias" in name:  # 如果是注意力偏置
                # Skip attention mask.  # 跳过注意力掩码
                # NOTE: "c_attn.bias" should not be skipped.  # 注意："c_attn.bias"不应被跳过
                continue  # 继续
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            # TODO (@robertgshaw2-neuralmagic): move to fp8 linear method  # TODO: 迁移到fp8线性方法
            if "c_attn.input_scale" in name or "c_attn.weight_scale" in name:  # 如果是c_attn的缩放权重
                weight_loader(param, loaded_weight, "q")  # 加载Q的缩放权重
                weight_loader(param, loaded_weight, "k")  # 加载K的缩放权重
                weight_loader(param, loaded_weight, "v")  # 加载V的缩放权重
            else:  # 否则正常加载
                weight_loader(param, loaded_weight)  # 使用权重加载器加载权重


EntryClass = GPTBigCodeForCausalLM  # 入口类为GPTBigCodeForCausalLM
