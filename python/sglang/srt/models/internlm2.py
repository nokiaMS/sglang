# InternLM2 因果语言模型实现
# 本文件实现了 InternLM2 模型的完整推理框架，包括 MLP、注意力机制、
# 解码器层、模型主体和因果语言模型，支持张量并行和量化。

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

# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/7f62077af5159c625fe3ad1c812e6c1a2b93ba3b/vllm/model_executor/models/internlm2.py
# 改编自 vLLM 项目的 InternLM2 实现

from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import SiluAndMul  # 导入 SiLU 与乘法激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化层
from sglang.srt.layers.linear import (  # 导入线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV 并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批处理信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class InternLM2MLP(nn.Module):  # InternLM2 MLP 类，实现前馈神经网络
    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏层维度
        intermediate_size: int,  # 中间层维度
        hidden_act: str,  # 激活函数名称
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 合并的 gate 和 up 投影层
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度为中间维度的两倍（gate 和 up 各一）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
        )
        self.w2 = RowParallelLinear(  # 下投影层（行并行）
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("w2", prefix),  # 参数前缀
        )
        if hidden_act != "silu":  # 如果激活函数不是 SiLU
            raise ValueError(  # 抛出异常
                f"Unsupported activation: {hidden_act}. "  # 不支持的激活函数
                "Only silu is supported for now."  # 目前仅支持 SiLU
            )
        self.act_fn = SiluAndMul()  # 创建 SiLU 与乘法激活函数

    def forward(self, x):  # 前向传播方法
        gate_up, _ = self.gate_up_proj(x)  # 通过 gate_up 投影层
        x = self.act_fn(gate_up)  # 应用 SiLU 激活并乘以 gate
        x, _ = self.w2(x)  # 通过下投影层
        return x  # 返回输出


class InternLM2Attention(nn.Module):  # InternLM2 注意力类，实现分组查询注意力机制
    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏层维度
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV 头数
        rope_theta: float = 10000,  # 旋转位置编码的基础频率
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE 缩放配置，可选
        max_position_embeddings: int = 8192,  # 最大位置编码数
        layer_id: int = 0,  # 层 ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层维度
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 确保头数能被并行度整除
        self.num_heads = self.total_num_heads // tp_size  # 每个 GPU 的注意力头数
        self.total_num_kv_heads = num_kv_heads  # 总 KV 头数
        if self.total_num_kv_heads >= tp_size:  # 如果 KV 头数大于等于并行度
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            # KV 头数大于 TP 大小，因此在多个张量并行 GPU 之间分配 KV 头
            assert self.total_num_kv_heads % tp_size == 0  # 确保 KV 头数能被并行度整除
        else:  # 否则
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            # KV 头数小于 TP 大小，因此在多个张量并行 GPU 之间复制 KV 头
            assert tp_size % self.total_num_kv_heads == 0  # 确保并行度能被 KV 头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个 GPU 的 KV 头数，至少为 1
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q 的大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV 的大小
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.rope_theta = rope_theta  # 旋转位置编码的基础频率
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码数

        self.wqkv = QKVParallelLinear(  # QKV 合并投影层
            hidden_size,  # 输入维度
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # Q 的总头数
            self.total_num_kv_heads,  # KV 的总头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("wqkv", prefix),  # 参数前缀
        )
        self.wo = RowParallelLinear(  # 输出投影层
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("wo", prefix),  # 参数前缀
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基础频率
            rope_scaling=rope_scaling,  # 缩放配置
        )
        self.attn = RadixAttention(  # 基数注意力层
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            self.num_kv_heads,  # KV 头数
            layer_id,  # 层 ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
        )

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批处理信息
    ) -> torch.Tensor:  # 返回注意力输出
        qkv, _ = self.wqkv(hidden_states)  # 通过 QKV 投影层
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分为 Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 通过注意力层计算
        output, _ = self.wo(attn_output)  # 通过输出投影层
        return output  # 返回输出


class InternLMDecoderLayer(nn.Module):  # InternLM 解码器层类，包含注意力和前馈网络
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int = 0,  # 层 ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层维度
        rope_theta = getattr(config, "rope_theta", 10000)  # 获取旋转位置编码基础频率，默认 10000
        rope_scaling = getattr(config, "rope_scaling", None)  # 获取 RoPE 缩放配置，默认 None
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 获取最大位置编码数，默认 8192
        self.attention = InternLM2Attention(  # 注意力层
            hidden_size=self.hidden_size,  # 隐藏层维度
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV 头数
            rope_theta=rope_theta,  # 旋转位置编码基础频率
            rope_scaling=rope_scaling,  # RoPE 缩放配置
            max_position_embeddings=max_position_embeddings,  # 最大位置编码数
            layer_id=layer_id,  # 层 ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attention", prefix),  # 参数前缀
        )
        self.feed_forward = InternLM2MLP(  # 前馈网络层
            hidden_size=self.hidden_size,  # 隐藏层维度
            intermediate_size=config.intermediate_size,  # 中间层维度
            hidden_act=config.hidden_act,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("feed_forward", prefix),  # 参数前缀
        )
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力归一化层
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 前馈网络归一化层

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批处理信息
        residual: Optional[torch.Tensor],  # 残差，可选
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回隐藏状态和残差的元组
        # Self Attention  # 自注意力
        if residual is None:  # 如果没有残差
            residual = hidden_states  # 保存当前隐藏状态作为残差
            hidden_states = self.attention_norm(hidden_states)  # 对隐藏状态进行归一化
        else:  # 否则
            hidden_states, residual = self.attention_norm(hidden_states, residual)  # 同时归一化和更新残差
        hidden_states = self.attention(  # 通过注意力层
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批处理信息
        )

        # Fully Connected  # 全连接层
        hidden_states, residual = self.ffn_norm(hidden_states, residual)  # 通过前馈网络归一化层
        hidden_states = self.feed_forward(hidden_states)  # 通过前馈网络
        return hidden_states, residual  # 返回隐藏状态和残差


class InternLM2Model(nn.Module):  # InternLM2 模型主体类
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存模型配置
        self.padding_idx = config.pad_token_id  # 填充 token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.tok_embeddings = VocabParallelEmbedding(  # 词表并行嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏层维度
            prefix=add_prefix("tok_embeddings", prefix),  # 参数前缀
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                InternLMDecoderLayer(  # 每个解码器层
                    config, i, quant_config, prefix=add_prefix(f"layers.{i}", prefix)  # 传入配置、层 ID 和量化配置
                )
                for i in range(config.num_hidden_layers)  # 根据配置的层数创建
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批处理信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
    ) -> torch.Tensor:  # 返回隐藏状态
        if input_embeds is None:  # 如果没有提供输入嵌入
            hidden_states = self.tok_embeddings(input_ids)  # 通过词表嵌入层获取隐藏状态
        else:  # 否则
            hidden_states = input_embeds  # 直接使用提供的输入嵌入
        residual = None  # 初始化残差为 None
        for i in range(len(self.layers)):  # 遍历每个解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 通过解码器层
                positions,  # 位置编码
                hidden_states,  # 隐藏状态
                forward_batch,  # 前向批处理信息
                residual,  # 残差
            )
        hidden_states, _ = self.norm(hidden_states, residual)  # 通过最终归一化层
        return hidden_states  # 返回隐藏状态


class InternLM2ForCausalLM(nn.Module):  # InternLM2 因果语言模型类
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存模型配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = InternLM2Model(  # 创建模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置和量化配置
        )
        self.output = ParallelLMHead(  # 输出语言模型头
            config.vocab_size, config.hidden_size, prefix=add_prefix("output", prefix)  # 词表大小和隐藏层维度
        )
        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.model.tok_embeddings  # 返回词表嵌入层

    @torch.no_grad()  # 禁用梯度计算，用于推理
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批处理信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
    ) -> torch.Tensor:  # 返回 logits
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型主体获取隐藏状态
        return self.logits_processor(  # 通过 logits 处理器计算并返回 logits
            input_ids, hidden_states, self.output, forward_batch  # 传入输入 ID、隐藏状态、输出头和批处理信息
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射表，用于合并门控和上投影
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片 ID)
            ("gate_up_proj", "w1", 0),  # gate_up 投影中的 w1（gate）部分
            ("gate_up_proj", "w3", 1),  # gate_up 投影中的 w3（up）部分
        ]
        params_dict = dict(self.named_parameters())  # 将模型参数转为字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转嵌入的逆频率
                continue  # 跳过，不需要加载
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不匹配
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.
                # 跳过 GPTQ 模型中的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是额外偏置
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break  # 跳出内层循环
            else:  # 如果没有匹配堆叠参数映射
                # Skip loading extra bias for GPTQ models.
                # 跳过 GPTQ 模型中的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是额外偏置
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                if "wqkv" in name:  # 如果是 QKV 合并权重
                    config = self.config  # 获取模型配置
                    kv_groups = config.num_attention_heads // config.num_key_value_heads  # 计算 KV 组数
                    head_dim = config.hidden_size // config.num_attention_heads  # 计算每个头的维度
                    loaded_weight = loaded_weight.view(  # 重塑权重形状
                        -1, 2 + kv_groups, head_dim, loaded_weight.shape[-1]  # 分离 Q、K、V 维度
                    )
                    wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1], dim=1)  # 拆分为 Q、K、V 权重
                    wq = wq.reshape(-1, wq.shape[-1])  # 重塑 Q 权重
                    wk = wk.reshape(-1, wk.shape[-1])  # 重塑 K 权重
                    wv = wv.reshape(-1, wv.shape[-1])  # 重塑 V 权重
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(param, wq, "q")  # 加载 Q 权重
                    weight_loader(param, wk, "k")  # 加载 K 权重
                    weight_loader(param, wv, "v")  # 加载 V 权重
                else:  # 其他权重
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认使用 default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重


EntryClass = InternLM2ForCausalLM  # 模型入口类，用于框架自动发现和注册模型
