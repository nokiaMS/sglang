# GPT-2模型推理实现文件
# 本文件实现了仅用于推理的GPT-2模型，兼容HuggingFace权重格式
# 包含GPT-2注意力层、MLP层、Transformer块及因果语言模型等核心组件

# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
# Copyright 2023 The vLLM team.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Inference-only GPT-2 model compatible with HuggingFace weights."""  # 仅推理用的GPT-2模型，兼容HuggingFace权重

from typing import Iterable, Optional, Tuple, Type  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import GPT2Config  # 导入GPT-2配置类

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import NewGELU  # 导入NewGELU激活函数
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


class GPT2Attention(nn.Module):  # GPT-2注意力层类

    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: GPT2Config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        total_num_heads = config.num_attention_heads  # 总注意力头数
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()  # 张量并行世界大小
        assert total_num_heads % tensor_model_parallel_world_size == 0  # 断言总头数可被并行世界大小整除
        self.num_heads = total_num_heads // tensor_model_parallel_world_size  # 每个并行分片的头数
        self.head_dim = self.hidden_size // total_num_heads  # 每个头的维度
        self.scale = self.head_dim**-0.5  # 缩放因子

        self.c_attn = QKVParallelLinear(  # QKV投影线性层
            self.hidden_size,  # 输入维度
            self.head_dim,  # 每个头的维度
            total_num_heads,  # 总头数
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
            num_kv_heads=total_num_heads,  # KV头数等于总头数（标准多头注意力）
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
        )

    def forward(  # 前向传播函数
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)  # 通过QKV投影层
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # 分割为Q、K、V
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        attn_output, _ = self.c_proj(attn_output)  # 通过输出投影层
        return attn_output  # 返回注意力输出


class GPT2MLP(nn.Module):  # GPT-2 MLP层类

    def __init__(  # 初始化函数
        self,
        intermediate_size: int,  # 中间层大小
        config: GPT2Config,  # 模型配置
        act_layer: Type[nn.Module] = NewGELU,  # 激活函数层类型，默认NewGELU
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ):
        super().__init__()  # 调用父类初始化
        hidden_size = config.hidden_size  # 隐藏层大小
        self.c_fc = ColumnParallelLinear(  # 上投影线性层（列并行）
            hidden_size,  # 输入维度
            intermediate_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_fc", prefix),  # 参数前缀
        )
        self.c_proj = RowParallelLinear(  # 下投影线性层（行并行）
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("c_proj", prefix),  # 参数前缀
        )
        self.act = act_layer()  # 实例化激活函数层

    def forward(  # 前向传播函数
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态张量
    ) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)  # 上投影
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states, _ = self.c_proj(hidden_states)  # 下投影
        return hidden_states  # 返回MLP输出


class GPT2Block(nn.Module):  # GPT-2 Transformer块类

    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: GPT2Config,  # 模型配置
        act_layer: Type[nn.Module] = NewGELU,  # 激活函数层类型，默认NewGELU
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ):
        super().__init__()  # 调用父类初始化
        hidden_size = config.hidden_size  # 隐藏层大小
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size  # 中间维度，默认4倍隐藏维度

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第一层LayerNorm
        self.attn = GPT2Attention(  # 注意力层
            layer_id, config, quant_config, prefix=add_prefix("attn", prefix)  # 传入层ID、配置和量化配置
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第二层LayerNorm
        self.mlp = GPT2MLP(  # MLP层
            inner_dim,  # 中间维度
            config,  # 模型配置
            act_layer=act_layer,  # 激活函数层
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )

    def forward(  # 前向传播函数
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.ln_1(hidden_states)  # 第一个LayerNorm
        attn_output = self.attn(  # 注意力计算
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 批次信息
        )
        # residual connection  # 残差连接
        hidden_states = attn_output + residual  # 注意力输出加残差

        residual = hidden_states  # 保存新的残差
        hidden_states = self.ln_2(hidden_states)  # 第二个LayerNorm
        feed_forward_hidden_states = self.mlp(hidden_states)  # MLP前馈计算
        # residual connection  # 残差连接
        hidden_states = residual + feed_forward_hidden_states  # MLP输出加残差
        return hidden_states  # 返回块输出


class GPT2Model(nn.Module):  # GPT-2模型类

    def __init__(  # 初始化函数
        self,
        config: GPT2Config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        assert not config.add_cross_attention  # 断言不使用交叉注意力
        assert not config.scale_attn_by_inverse_layer_idx  # 断言不按层索引逆缩放注意力
        assert not config.reorder_and_upcast_attn  # 断言不重排序和上转型注意力
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)  # 词嵌入层
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)  # 位置嵌入层
        self.h = nn.ModuleList(  # Transformer块列表
            [
                GPT2Block(  # 每个Transformer块
                    i,  # 层索引
                    config,  # 模型配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"h.{i}", prefix),  # 参数前缀
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


class GPT2LMHeadModel(nn.Module):  # GPT-2因果语言模型类

    def __init__(  # 初始化函数
        self,
        config: GPT2Config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.transformer = GPT2Model(  # GPT-2模型主体
            config, quant_config, prefix=add_prefix("transformer", prefix)  # 传入配置和量化配置
        )
        self.lm_head = self.transformer.wte  # 语言模型头（与词嵌入共享权重）

        self.logits_processor = LogitsProcessor(config)  # logits处理器

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
            if "lm_head.weight" in name:  # 如果是lm_head权重
                # GPT-2 ties the weights of the embedding layer and the final  # GPT-2将嵌入层和最终线性层的权重绑定
                # linear layer.  # 线性层权重共享
                continue  # 跳过
            if ".attn.bias" in name or ".attn.masked_bias" in name:  # 如果是注意力偏置或掩码偏置
                # Skip attention mask.  # 跳过注意力掩码
                # NOTE: "c_attn.bias" should not be skipped.  # 注意："c_attn.bias"不应被跳过
                continue  # 跳过
            if not name.startswith("transformer."):  # 如果名称不以"transformer."开头
                name = "transformer." + name  # 添加"transformer."前缀

            param = params_dict[name]  # 获取参数
            # The HF's GPT-2 implementation uses Conv1D instead of Linear.  # HuggingFace的GPT-2实现使用Conv1D而非Linear
            # Because of this, we need to transpose the weights.  # 因此需要转置权重
            # Note(zhuohan): the logic below might break quantized models.  # 注意(zhuohan)：以下逻辑可能会破坏量化模型
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:  # 遍历Conv1D权重名称
                if conv1d_weight_name not in name:  # 如果名称不包含Conv1D权重名
                    continue  # 继续
                if not name.endswith(".weight"):  # 如果不是权重参数
                    continue  # 继续
                loaded_weight = loaded_weight.t()  # 转置权重
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重


EntryClass = GPT2LMHeadModel  # 入口类为GPT2LMHeadModel
