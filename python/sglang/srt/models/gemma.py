# Google Gemma模型推理实现文件
# 本文件实现了Google Gemma模型的推理逻辑，仅用于推理，兼容HuggingFace权重格式
# 主要包含：GELU门控MLP、注意力层、解码器层、模型主体和因果语言模型
# Gemma模型特点：使用GELU激活函数、嵌入缩放、RMSNorm权重加1处理

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明

# Copyright 2023-2024 SGLang Team  # SGLang团队版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache 2.0许可证授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache 2.0许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不附带任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 请参阅许可证获取管理权限和
# limitations under the License.  # 限制的具体条款
# ==============================================================================  # 分隔线

# Adapted from:  # 适配自：
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/gemma.py#L1  # vLLM的Gemma模型实现
"""Inference-only Gemma model compatible with HuggingFace weights."""  # 仅推理的Gemma模型，兼容HuggingFace权重

from typing import Iterable, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch
from torch import nn  # 从PyTorch导入神经网络模块
from transformers import PretrainedConfig  # 从transformers导入预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入张量并行大小获取
from sglang.srt.layers.activation import GeluAndMul  # 导入GELU与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 从线性层模块导入
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class GemmaMLP(nn.Module):  # Gemma的MLP模块
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层维度大小
        intermediate_size: int,  # 中间层维度大小
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # gate和up的合并列并行线性层
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度（gate和up各一份）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # down行并行线性层
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
        )
        self.act_fn = GeluAndMul("none")  # GELU与乘法激活函数（近似方式为"none"即精确GELU）

    def forward(self, x):  # 前向传播函数
        gate_up, _ = self.gate_up_proj(x)  # 通过gate_up投影
        x = self.act_fn(gate_up)  # 应用GELU激活函数和门控
        x, _ = self.down_proj(x)  # 通过down投影
        return x  # 返回输出


class GemmaAttention(nn.Module):  # Gemma注意力模块
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层维度
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        head_dim: int,  # 头维度
        layer_id: int = 0,  # 层ID，默认为0
        max_position_embeddings: int = 8192,  # 最大位置编码数，默认8192
        rope_theta: float = 10000,  # 旋转位置编码基数，默认10000
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层维度
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 确保头数能被并行大小整除
        self.num_heads = self.total_num_heads // tp_size  # 每个并行秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于并行大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此进行分区
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头分配到多个张量并行GPU上
            assert self.total_num_kv_heads % tp_size == 0  # 确保KV头数能被并行大小整除
        else:  # 否则KV头数小于并行大小
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此进行复制
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头复制到多个张量并行GPU上
            assert tp_size % self.total_num_kv_heads == 0  # 确保并行大小能被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个并行秩的KV头数
        self.head_dim = head_dim  # 保存头维度
        self.q_size = self.num_heads * self.head_dim  # Q维度大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV维度大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 旋转位置编码基数

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 输出行并行线性投影
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=self.rope_theta,  # 基数
            is_neox_style=True,  # 使用Neox风格
        )
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割为Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力
        output, _ = self.o_proj(attn_output)  # 通过输出投影
        return output  # 返回输出


class GemmaDecoderLayer(nn.Module):  # Gemma解码器层
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int = 0,  # 层ID，默认为0
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.self_attn = GemmaAttention(  # 自注意力模块
            hidden_size=self.hidden_size,  # 隐藏层维度
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            head_dim=config.head_dim,  # 头维度
            layer_id=layer_id,  # 层ID
            max_position_embeddings=config.max_position_embeddings,  # 最大位置编码数
            rope_theta=config.rope_parameters["rope_theta"],  # 旋转位置编码基数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )
        self.mlp = GemmaMLP(  # MLP模块
            hidden_size=self.hidden_size,  # 隐藏层维度
            intermediate_size=config.intermediate_size,  # 中间层维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention  # 自注意力计算
        if residual is None:  # 如果没有残差（第一层）
            residual = hidden_states  # 初始化残差为隐藏状态
            hidden_states = self.input_layernorm(hidden_states)  # 对隐藏状态做层归一化
        else:  # 如果有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 融合层归一化和残差连接
        hidden_states = self.self_attn(  # 通过自注意力模块
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次信息
        )

        # Fully Connected  # 全连接层
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后层归一化和残差更新
        hidden_states = self.mlp(hidden_states)  # 通过MLP
        return hidden_states, residual  # 返回隐藏状态和残差


class GemmaModel(nn.Module):  # Gemma模型主体
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                GemmaDecoderLayer(  # 每个解码器层
                    config,  # 配置
                    i,  # 层ID
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 参数前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历所有层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，默认为None
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态
        else:  # 否则使用提供的嵌入
            hidden_states = input_embeds  # 使用输入嵌入

        # Normalize the embedding by sqrt(hidden_size)  # 通过sqrt(hidden_size)归一化嵌入
        hidden_states *= self.config.hidden_size**0.5  # 乘以sqrt(hidden_size)进行缩放

        residual = None  # 初始化残差为None
        for i in range(len(self.layers)):  # 遍历所有层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过当前层
                positions,  # 位置编码
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次信息
                residual,  # 残差
            )
        hidden_states, _ = self.norm(hidden_states, residual)  # 通过最终层归一化
        return hidden_states  # 返回隐藏状态


class GemmaForCausalLM(nn.Module):  # Gemma因果语言模型
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [  # QKV投影
            "q_proj",  # Q投影
            "k_proj",  # K投影
            "v_proj",  # V投影
        ],
        "gate_up_proj": [  # gate_up投影
            "gate_proj",  # gate投影
            "up_proj",  # up投影
        ],
    }

    # LoRA specific attributes  # LoRA特定属性
    supported_lora_modules = [  # 支持LoRA的模块
        "qkv_proj",  # QKV投影
        "o_proj",  # 输出投影
        "gate_up_proj",  # gate_up投影
        "down_proj",  # down投影
    ]
    # Gemma does not apply LoRA to the embedding layer.  # Gemma不在嵌入层应用LoRA
    embedding_modules = {}  # 嵌入模块映射为空
    embedding_padding_modules = []  # 嵌入填充模块为空

    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = GemmaModel(  # 模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型主体获取隐藏状态
        return self.logits_processor(  # 通过logits处理器（Gemma的lm_head与embed_tokens共享）
            input_ids, hidden_states, self.model.embed_tokens, forward_batch  # 使用embed_tokens作为lm_head
        )

    @torch.no_grad()  # 禁用梯度计算
    def forward_split_prefill(  # 分割预填充前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        split_interval: Tuple[int, int],  # [start, end) 0-based  # 分割区间，左闭右开，从0开始
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ):
        start, end = split_interval  # 获取分割区间的起始和结束
        # embed  # 嵌入
        if start == 0:  # 如果从第0层开始
            if input_embeds is None:  # 如果没有提供输入嵌入
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)  # 通过词嵌入层
            else:  # 否则使用提供的嵌入
                forward_batch.hidden_states = input_embeds  # 使用输入嵌入

            # Normalize the embedding by sqrt(hidden_size)  # 通过sqrt(hidden_size)归一化嵌入
            forward_batch.hidden_states *= self.model.config.hidden_size**0.5  # 乘以sqrt(hidden_size)缩放

        # decoder layer  # 解码器层
        for i in range(start, end):  # 遍历分割区间内的层
            layer = self.model.layers[i]  # 获取当前层
            forward_batch.hidden_states, forward_batch.residual = layer(  # 通过当前层
                positions,  # 位置编码
                forward_batch.hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次信息
                forward_batch.residual,  # 残差
            )

        if end == self.model.config.num_hidden_layers:  # 如果到达最后一层
            # norm  # 层归一化
            forward_batch.hidden_states, _ = self.model.norm(  # 通过最终层归一化
                forward_batch.hidden_states, forward_batch.residual
            )

            # logits process  # logits处理
            result = self.logits_processor(  # 通过logits处理器
                input_ids,  # 输入ID
                forward_batch.hidden_states,  # 隐藏状态
                self.model.embed_tokens,  # 使用embed_tokens作为lm_head
                forward_batch,  # 前向批次信息
            )
        else:  # 否则未到达最后一层
            result = None  # 结果为None

        return result  # 返回结果

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # QKV投影中Q的映射
            ("qkv_proj", "k_proj", "k"),  # QKV投影中K的映射
            ("qkv_proj", "v_proj", "v"),  # QKV投影中V的映射
            ("gate_up_proj", "gate_proj", 0),  # gate_up投影中gate的映射
            ("gate_up_proj", "up_proj", 1),  # gate_up投影中up的映射
        ]
        params_dict = dict(self.named_parameters())  # 参数名字典
        loaded_params = set()  # 已加载参数集合
        for name, loaded_weight in weights:  # 遍历所有权重
            for param_name, shard_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if shard_name not in name:  # 如果分片名不在权重名中
                    continue  # 跳过
                name = name.replace(shard_name, param_name)  # 替换分片名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果堆叠参数映射中没有匹配
                # lm_head is not used in vllm as it is tied with embed_token.  # vLLM中lm_head未使用，因为与embed_token绑定
                # To prevent errors, skip loading lm_head.weight.  # 为防止错误，跳过lm_head.weight
                if "lm_head.weight" in name:  # 如果是lm_head权重
                    continue  # 跳过
                # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                    continue  # 跳过
                # GemmaRMSNorm is different from Llama's in that it multiplies  # GemmaRMSNorm与Llama的不同之处在于它乘以
                # (1 + weight) to the output, instead of just weight.  # (1 + weight)而不仅仅是weight
                if "norm.weight" in name:  # 如果是归一化权重
                    loaded_weight += 1.0  # 权重加1（GemmaRMSNorm特性）
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重
            loaded_params.add(name)  # 将参数名添加到已加载集合


EntryClass = GemmaForCausalLM  # 模型入口类
