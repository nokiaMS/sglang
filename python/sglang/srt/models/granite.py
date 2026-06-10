# Granite模型推理实现文件
# 本文件实现了仅用于推理的Granite模型，兼容HuggingFace权重格式
# 包含Granite MLP层、注意力层、解码器层、模型主体及因果语言模型等核心组件
# 支持残差乘数缩放、嵌入乘数缩放、logit缩放及词嵌入权重共享等特性

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
"""Inference-only Granite model compatible with HuggingFace weights."""  # 仅推理用的Granite模型，兼容HuggingFace权重

import logging  # 导入日志模块
from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import GraniteConfig  # 导入Granite配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器和输出
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层和池化类型
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具
from sglang.utils import get_exception_traceback  # 导入异常回溯获取工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class GraniteMLP(nn.Module):  # Granite MLP层类
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 激活函数名称
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控上投影线性层（合并列并行）
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度（gate和up各一个中间层大小）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.down_proj = RowParallelLinear(  # 下投影线性层（行并行）
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
        )
        if hidden_act != "silu":  # 如果激活函数不是SiLU
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. "  # 不支持的激活函数
                "Only silu is supported for now."  # 目前只支持silu
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数

    def forward(self, x):  # 前向传播函数
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影
        x = self.act_fn(gate_up)  # 激活函数（SiLU和乘法）
        x, _ = self.down_proj(x)  # 下投影
        return x  # 返回MLP输出


class GraniteAttention(nn.Module):  # Granite注意力层类
    def __init__(  # 初始化函数
        self,
        config: GraniteConfig,  # 模型配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID，默认0
        rope_theta: float = 10000,  # RoPE基准角度，默认10000
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置，默认None
        rope_is_neox_style: bool = True,  # RoPE是否为Neox风格，默认True
        max_position_embeddings: int = 8192,  # 最大位置嵌入数，默认8192
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言总头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 当前分片的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此我们进行分区
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头分配到多个张量并行GPU上
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除
        else:  # 否则KV头数小于TP大小
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此我们进行复制
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头复制到多个张量并行GPU上
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 当前分片的KV头数
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo  # MistralConfig有一个Mistral-Nemo引入的可选head_dim
        self.head_dim = getattr(  # 获取头维度
            config, "head_dim", self.hidden_size // self.total_num_heads  # 默认为隐藏大小除以总头数
        )
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = config.attention_multiplier  # 缩放因子（来自配置的注意力乘数）
        self.rope_theta = rope_theta  # RoPE基准角度
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数

        self.qkv_proj = QKVParallelLinear(  # QKV投影线性层
            hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
        )
        self.o_proj = RowParallelLinear(  # 输出投影线性层（行并行）
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基准角度
            rope_scaling=rope_scaling,  # RoPE缩放配置
            is_neox_style=rope_is_neox_style,  # 是否为Neox风格
        )
        self.attn = RadixAttention(  # 基数注意力实现
            self.num_heads,  # 注意力头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影层
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割为Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 通过输出投影层
        return output  # 返回输出


class GraniteDecoderLayer(nn.Module):  # Granite解码器层类
    def __init__(  # 初始化函数
        self,
        config: GraniteConfig,  # 模型配置
        layer_id: int = 0,  # 层ID，默认0
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.residual_multiplier = config.residual_multiplier  # 残差乘数
        rope_theta = config.rope_parameters["rope_theta"]  # RoPE基准角度
        rope_scaling = config.rope_parameters  # RoPE缩放配置
        if rope_scaling is not None and getattr(  # 如果有RoPE缩放且有原始最大位置嵌入
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 设置原始最大位置嵌入
                config.original_max_position_embeddings  # 从配置获取
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)  # RoPE是否为Neox风格，默认True
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入数
        self.self_attn = GraniteAttention(  # 自注意力层
            config=config,  # 模型配置
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE基准角度
            rope_scaling=rope_scaling,  # RoPE缩放配置
            rope_is_neox_style=rope_is_neox_style,  # RoPE风格
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )
        self.mlp = GraniteMLP(  # MLP层
            hidden_size=self.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            hidden_act=config.hidden_act,  # 激活函数名称
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差张量，可选
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention  # 自注意力
        if residual is None:  # 如果没有残差（第一层）
            residual = hidden_states  # 残差等于隐藏状态
            hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        else:  # 否则
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 输入层归一化（带残差）
        hidden_states = (  # 计算注意力输出
            self.self_attn(  # 自注意力计算
                positions=positions,  # 位置
                hidden_states=hidden_states,  # 隐藏状态
                forward_batch=forward_batch,  # 批次信息
            )
            * self.residual_multiplier  # 乘以残差乘数
        )  # multiplier for Maximal Update Parameterization  # Maximal Update Parameterization的乘数

        # Fully Connected  # 全连接层（MLP）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后层归一化
        hidden_states = self.mlp(hidden_states) * self.residual_multiplier  # MLP输出乘以残差乘数
        return hidden_states, residual  # 返回隐藏状态和残差


class GraniteModel(nn.Module):  # Granite模型类
    def __init__(  # 初始化函数
        self,
        config: GraniteConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词汇表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size, config.hidden_size  # 词汇表大小和隐藏层大小
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                GraniteDecoderLayer(  # 每个解码器层
                    config,  # 模型配置
                    i,  # 层索引
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 参数前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终RMS归一化层

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，默认None
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入
        else:  # 否则
            hidden_states = input_embeds  # 使用提供的输入嵌入
        residual = None  # 残差初始化为None
        hidden_states *= self.config.embedding_multiplier  # 乘以嵌入乘数缩放
        for i in range(len(self.layers)):  # 遍历所有解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 执行当前层前向传播
                positions,  # 位置
                hidden_states,  # 隐藏状态
                forward_batch,  # 批次信息
                residual,  # 残差
            )
        hidden_states, _ = self.norm(hidden_states, residual)  # 最终RMS归一化
        return hidden_states  # 返回隐藏状态


class GraniteForCausalLM(nn.Module):  # Granite因果语言模型类
    def __init__(  # 初始化函数
        self,
        config: GraniteConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认None
        prefix: str = "",  # 参数前缀，默认空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = GraniteModel(  # Granite模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)  # 传入配置、量化配置和前缀
        )
        # If tie_word_embeddings == True, then input and output embeddings are  # 如果tie_word_embeddings为True，则输入和输出嵌入是
        # the same tensor. Enforce during object creation so that weights will  # 同一张量。在对象创建时强制执行，以便权重将
        # load correctly even if the LM head weights don't have a separate entry  # 即使LM头权重在状态字典中没有单独条目也能正确加载
        # in the state dict.  # 在状态字典中
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,  # 词汇表大小
            config.hidden_size,  # 隐藏层大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 参数前缀
        )
        if self.config.tie_word_embeddings:  # 如果共享词嵌入权重
            self.lm_head.tie_weights(self.model.embed_tokens)  # 绑定LM头权重与词嵌入权重

        # Granite logit scaling factors are applied via division, but  # Granite的logit缩放因子通过除法应用，但
        # LogitsProcessor expects a multiplicative factor.  # LogitsProcessor期望一个乘法因子
        if hasattr(config, "logits_scaling"):  # 如果配置有logits缩放
            logit_scale = 1.0 / config.logits_scaling  # 取倒数作为乘法因子
        else:  # 否则
            logit_scale = None  # 设为None
        self.logits_processor = LogitsProcessor(config, logit_scale=logit_scale)  # logits处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 池化层（取最后一个token，归一化）
        self.stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
            (".gate_up_proj", ".gate_proj", 0),  # gate投影映射
            (".gate_up_proj", ".up_proj", 1),  # up投影映射
        ]

    @torch.no_grad()  # 禁用梯度计算装饰器
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，默认None
        get_embedding: bool = False,  # 是否获取嵌入，默认False
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型获取隐藏状态
        if not get_embedding:  # 如果不需要获取嵌入
            logits_processor_output: LogitsProcessorOutput = self.logits_processor(  # 通过logits处理器获取logits
                input_ids, hidden_states, self.lm_head, forward_batch  # 传入输入ID、隐藏状态、语言模型头和批次信息
            )
            return logits_processor_output  # 返回logits处理器输出
        else:  # 否则需要获取嵌入
            return self.pooler(hidden_states, forward_batch)  # 通过池化层获取嵌入

    def get_module_name_from_weight_name(self, name):  # 根据权重名获取模块名
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:  # 遍历堆叠参数映射
            if weight_name in name:  # 如果权重名在名称中
                return (  # 返回模块名和分片数
                    name.replace(weight_name, param_name)[: -len(".weight")],  # 替换并去掉".weight"后缀
                    num_shard,  # 分片数
                )
        return name[: -len(".weight")], 1  # 默认返回去掉".weight"的名称和分片数1

    def get_num_params(self):  # 获取参数数量
        params_dict = dict(self.named_parameters())  # 获取参数字典
        return len(params_dict)  # 返回参数数量

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
            (".gate_up_proj", ".gate_proj", 0),  # gate投影映射
            (".gate_up_proj", ".up_proj", 1),  # up投影映射
        ]

        params_dict = dict(self.named_parameters())  # 获取参数字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转位置编码逆频率和投影器
                continue  # 继续
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的cos和sin
                # Models trained using ColossalAI may include these tensors in  # 使用ColossalAI训练的模型可能在检查点中包含这些张量
                # the checkpoint. Skip them.  # 跳过它们
                continue  # 继续
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过不在参数字典中的视觉塔权重
                continue  # 继续
            if "lm_head.weight" in name and self.config.tie_word_embeddings:  # 如果是lm_head权重且共享嵌入
                # Input and output embeddings are tied, so the output embeddings  # 输入和输出嵌入共享，因此输出嵌入
                # may not be present in the checkpoint. We assume that the input  # 可能在检查点中不存在。我们假设输入
                # embeddings are always present in the checkpoint.  # 嵌入始终存在于检查点中
                continue  # 跳过

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
                # This block only runs if the preceding for loop doesn't find  # 此块仅在前面的for循环没找到
                # a match for `name` in `stacked_params_mapping`.  # 在stacked_params_mapping中匹配的名称时运行

                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 继续
                # Skip loading kv_scale from ckpts towards new design.  # 跳过从检查点加载kv_scale（新设计不需要）
                if name.endswith(".kv_scale") and name not in params_dict:  # 如果是kv_scale且不在参数字典中
                    continue  # 继续
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重

    def get_weights_by_name(  # 根据名称获取权重
        self, name: str, truncate_size: int = 100, tp_size: int = 1  # 名称、截断大小和TP大小
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.  # 根据名称获取参数权重，类似于HuggingFace的get_parameter

        Only used for unit test with an unoptimized performance.  # 仅用于单元测试，性能未优化
        For optimized performance, please use torch.save and torch.load.  # 优化性能请使用torch.save和torch.load
        """
        try:  # 尝试获取权重
            if name == "lm_head.weight" and self.config.tie_word_embeddings:  # 如果是lm_head权重且共享嵌入
                logger.info(  # 记录信息日志
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."  # 此模型词嵌入已绑定，返回embed_tokens.weight作为lm_head.weight
                )
                return (  # 返回词嵌入权重
                    self.model.embed_tokens.weight.cpu()  # 移到CPU
                    .to(torch.float32)  # 转为float32
                    .numpy()  # 转为numpy数组
                    .tolist()[:truncate_size]  # 转为列表并截断
                )

            mapped_name = name  # 映射后的名称
            mapped_shard_id = None  # 映射后的分片ID
            for param_name, weight_name, shard_id in self.stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name in name:  # 如果权重名在名称中
                    mapped_name = name.replace(weight_name, param_name)  # 替换权重名
                    mapped_shard_id = shard_id  # 设置分片ID
                    break  # 跳出循环
            params_dict = dict(self.named_parameters())  # 获取参数字典
            param = params_dict[mapped_name]  # 获取参数
            if mapped_shard_id is not None:  # 如果有分片ID
                if mapped_shard_id in ["q", "k", "v"]:  # 如果是QKV分片
                    num_heads = self.config.num_attention_heads // tp_size  # 每个TP分片的头数
                    num_kv_heads = self.config.num_key_value_heads // tp_size  # 每个TP分片的KV头数
                    head_dim = (  # 头维度
                        self.config.hidden_size // self.config.num_attention_heads  # 隐藏大小除以头数
                    )
                    if mapped_shard_id == "q":  # Q分片
                        offset = 0  # 偏移为0
                        size = num_heads * head_dim  # 大小为头数乘以头维度
                    elif mapped_shard_id == "k":  # K分片
                        offset = num_heads * head_dim  # 偏移为Q大小
                        size = num_kv_heads * head_dim  # 大小为KV头数乘以头维度
                    elif mapped_shard_id == "v":  # V分片
                        offset = (num_heads + num_kv_heads) * head_dim  # 偏移为Q+K大小
                        size = num_kv_heads * head_dim  # 大小为KV头数乘以头维度
                    weight = param.data.narrow(0, offset, size)  # 窄化获取对应分片
                elif mapped_shard_id in [0, 1]:  # 如果是gate或up分片
                    intermediate_size = self.config.intermediate_size  # 中间层大小
                    slice_size = intermediate_size // tp_size  # 每个TP分片的切片大小
                    if mapped_shard_id == 0:  # gate_proj  # gate投影
                        offset = 0  # 偏移为0
                        size = slice_size  # 大小为切片大小
                    elif mapped_shard_id == 1:  # up_proj  # up投影
                        offset = slice_size  # 偏移为切片大小
                        size = slice_size  # 大小为切片大小

                    weight = param.data.narrow(0, offset, size)  # 窄化获取对应分片
                else:  # 其他情况
                    weight = param.data  # 使用完整参数数据
            else:  # 没有分片ID
                weight = param.data  # 使用完整参数数据
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):  # 如果TP大小>1且是行并行层
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]  # 创建收集张量列表
                torch.distributed.all_gather(gathered_weights, weight)  # 全收集权重
                weight = torch.cat(gathered_weights, dim=1)  # 在第1维拼接
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]  # 返回截断后的权重列表

        except Exception:  # 捕获异常
            logger.error(  # 记录错误日志
                f"Error getting weights by name {name} in GraniteForCausalLM: {get_exception_traceback()}"  # 获取权重出错信息
            )
            return None  # 返回None


EntryClass = [GraniteForCausalLM]  # 入口类为GraniteForCausalLM列表
