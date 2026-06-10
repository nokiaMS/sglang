# OLMoE 混合专家因果语言模型实现
# 该文件实现了推理专用的 OLMoE 模型，兼容 HuggingFace 权重格式，
# 使用混合专家（MoE）架构替代传统 MLP，支持张量并行。

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
# https://github.com/vllm-project/vllm/pull/7922

"""Inference-only OLMoE model compatible with HuggingFace weights."""

from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示 # 导入类型提示

import torch  # 导入 PyTorch # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块 # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置 # 导入预训练配置基类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入张量并行工具 # 导入获取张量并行世界大小的函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化 # 导入 RMS 归一化层
from sglang.srt.layers.linear import (  # 导入线性层 # 导入各种并行线性层
    QKVParallelLinear,  # QKV 并行线性层 # QKV 并行线性层
    ReplicatedLinear,  # 复制线性层 # 复制线性层
    RowParallelLinear,  # 行并行线性层 # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器 # 导入 logits 处理器
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合 MoE # 导入融合 MoE
from sglang.srt.layers.moe.topk import TopK  # 导入 TopK 选择 # 导入 TopK 路由选择
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力 # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码 # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入 # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头 # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入 # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器 # 导入默认权重加载器
from sglang.srt.utils import add_prefix, make_layers, print_warning_once  # 导入工具函数 # 导入工具函数


class OlmoeMoE(nn.Module):
    """A tensor-parallel MoE implementation for Olmoe that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """
    """OLMoE 混合专家模块，将每个专家的权重在所有并行进程间分片"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        num_experts: int,  # 专家数量 # 专家数量
        top_k: int,  # Top-K 选择数 # Top-K 路由选择数
        hidden_size: int,  # 隐藏层大小 # 隐藏层维度
        intermediate_size: int,  # 中间层大小 # 中间层维度
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型 # 参数数据类型
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        tp_size: Optional[int] = None,  # 张量并行大小 # 张量并行大小
        layer_id: int = 0,  # 层 ID # 层 ID
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小 # 保存隐藏层维度

        # Gate always runs at half / full precision for now. # 门控始终以半精度/全精度运行
        self.gate = ReplicatedLinear(  # 门控网络 # 创建门控网络
            hidden_size,  # 输入大小 # 输入维度
            num_experts,  # 输出大小（专家数） # 输出维度（专家数量）
            bias=False,  # 无偏置 # 无偏置
            quant_config=None,  # 不量化门控 # 不对门控进行量化
            prefix=add_prefix("gate", prefix),  # 前缀 # 参数名前缀
        )

        self.topk = TopK(  # TopK 路由 # 创建 TopK 路由选择器
            top_k=top_k,  # K 值 # Top-K 值
            renormalize=False,  # 不重新归一化 # 不重新归一化
        )

        self.experts = FusedMoE(  # 融合专家 # 创建融合 MoE 专家
            num_experts=num_experts,  # 专家数量 # 专家数量
            hidden_size=hidden_size,  # 隐藏层大小 # 隐藏层维度
            intermediate_size=intermediate_size,  # 中间层大小 # 中间层维度
            reduce_results=True,  # 归约结果 # 归约结果
            quant_config=quant_config,  # 量化配置 # 量化配置
            layer_id=layer_id,  # 层 ID # 层 ID
            prefix=add_prefix("experts", prefix),  # 前缀 # 参数名前缀
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播 # 前向传播方法
        # NOTE: hidden_states can have either 1D or 2D shape. # 注意: hidden_states 可以是 1D 或 2D
        orig_shape = hidden_states.shape  # 保存原始形状 # 保存原始形状
        hidden_states = hidden_states.view(-1, self.hidden_size)  # 重塑为 2D # 重塑为 2D
        # router_logits: (num_tokens, n_experts) # 路由 logits: (令牌数, 专家数)
        router_logits, _ = self.gate(hidden_states)  # 门控前向 # 计算路由 logits
        topk_output = self.topk(hidden_states, router_logits)  # TopK 选择 # 计算 TopK 路由
        final_hidden_states = self.experts(hidden_states, topk_output)  # 专家前向 # 计算专家输出
        return final_hidden_states.view(orig_shape)  # 恢复原始形状 # 恢复原始形状


class OlmoeAttention(nn.Module):
    """OLMoE 注意力模块，支持 GQA 和 QK 归一化"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        layer_id: int,  # 层 ID # 层 ID
        hidden_size: int,  # 隐藏层大小 # 隐藏层维度
        num_heads: int,  # 注意力头数 # 注意力头数
        num_kv_heads: int,  # KV 头数 # KV 头数
        rope_theta: float = 10000,  # RoPE theta # 旋转位置编码 theta
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE 缩放 # 旋转位置编码缩放
        max_position_embeddings: int = 4096,  # 最大位置编码数 # 最大位置编码数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小 # 保存隐藏层维度
        tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小 # 张量并行世界大小
        self.total_num_heads = num_heads  # 总注意力头数 # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被并行度整除 # 断言头数可被并行度整除
        self.num_heads = self.total_num_heads // tp_size  # 每个并行的头数 # 每个并行进程的头数
        self.total_num_kv_heads = num_kv_heads  # 总 KV 头数 # 总 KV 头数
        if self.total_num_kv_heads >= tp_size:  # KV 头数大于等于并行度 # 如果 KV 头数大于等于并行度
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0  # 断言可整除 # 断言可整除
        else:  # KV 头数小于并行度 # 如果 KV 头数小于并行度
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0  # 断言可整除 # 断言可整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个并行的 KV 头数 # 每个并行进程的 KV 头数
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度 # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q 大小 # Q 的大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV 大小 # KV 的大小
        self.scaling = self.head_dim**-0.5  # 缩放因子 # 注意力缩放因子
        self.rope_theta = rope_theta  # RoPE theta # 旋转位置编码的 theta 参数
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码数 # 最大位置编码数

        self.qkv_proj = QKVParallelLinear(  # QKV 投影 # QKV 并行线性投影
            hidden_size,  # 输入大小 # 输入维度
            self.head_dim,  # 头维度 # 每个头的维度
            self.total_num_heads,  # 总头数 # 总注意力头数
            self.total_num_kv_heads,  # 总 KV 头数 # 总 KV 头数
            bias=False,  # 无偏置 # 无偏置
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 前缀 # 参数名前缀
        )
        self.q_norm = RMSNorm(hidden_size, eps=1e-5)  # Q 归一化 # Q 归一化层
        self.k_norm = RMSNorm(hidden_size, eps=1e-5)  # K 归一化 # K 归一化层
        self.o_proj = RowParallelLinear(  # 输出投影 # 行并行输出投影
            self.total_num_heads * self.head_dim,  # 输入大小 # 输入维度
            hidden_size,  # 输出大小 # 输出维度
            bias=False,  # 无偏置 # 无偏置
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 前缀 # 参数名前缀
        )

        self.rotary_emb = get_rope(  # 旋转位置编码 # 获取旋转位置编码
            self.head_dim,  # 头维度 # 每个头的维度
            rotary_dim=self.head_dim,  # 旋转维度 # 旋转维度
            max_position=max_position_embeddings,  # 最大位置 # 最大位置数
            base=rope_theta,  # 基数 # theta 基数
            rope_scaling=rope_scaling,  # 缩放配置 # 缩放配置
            is_neox_style=True,  # Neox 风格 # 使用 Neox 风格
        )
        self.attn = RadixAttention(  # 基数注意力 # 创建基数注意力
            self.num_heads,  # 头数 # 头数
            self.head_dim,  # 头维度 # 每个头的维度
            self.scaling,  # 缩放因子 # 缩放因子
            layer_id=layer_id,  # 层 ID # 层 ID
            num_kv_heads=self.num_kv_heads,  # KV 头数 # KV 头数
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("attn", prefix),  # 前缀 # 参数名前缀
        )

    def forward(  # 前向传播 # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码 # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
    ) -> torch.Tensor:  # 返回张量 # 返回注意力输出
        qkv, _ = self.qkv_proj(hidden_states)  # QKV 投影 # 计算 QKV 投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割 QKV # 分割为 Q、K、V
        q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())  # QK 归一化 # 应用 QK 归一化
        q, k = self.rotary_emb(positions, q, k)  # 旋转位置编码 # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力 # 计算注意力
        output, _ = self.o_proj(attn_output)  # 输出投影 # 输出投影
        return output  # 返回输出 # 返回输出


class OlmoeDecoderLayer(nn.Module):
    """OLMoE 解码器层，包含注意力和 MoE 子层"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置 # 模型配置
        layer_id: int = 0,  # 层 ID # 层 ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小 # 隐藏层维度
        rope_theta = config.rope_parameters["rope_theta"]  # RoPE theta # 旋转位置编码的 theta 参数
        rope_scaling = config.rope_parameters  # RoPE 缩放 # 旋转位置编码缩放配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 4096)  # 最大位置编码数 # 最大位置编码数

        self.self_attn = OlmoeAttention(  # 自注意力 # 创建 OLMoE 注意力层
            layer_id,  # 层 ID # 层 ID
            hidden_size=self.hidden_size,  # 隐藏层大小 # 隐藏层维度
            num_heads=config.num_attention_heads,  # 注意力头数 # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV 头数 # KV 头数
            rope_theta=rope_theta,  # RoPE theta # 旋转位置编码 theta
            rope_scaling=rope_scaling,  # RoPE 缩放 # 旋转位置编码缩放
            max_position_embeddings=max_position_embeddings,  # 最大位置编码数 # 最大位置编码数
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 前缀 # 参数名前缀
        )

        self.mlp = OlmoeMoE(  # MoE # 创建 OLMoE 混合专家层
            num_experts=config.num_experts,  # 专家数量 # 专家数量
            top_k=config.num_experts_per_tok,  # Top-K # 每个令牌选择的专家数
            hidden_size=config.hidden_size,  # 隐藏层大小 # 隐藏层维度
            intermediate_size=config.intermediate_size,  # 中间层大小 # 中间层维度
            layer_id=layer_id,  # 层 ID # 层 ID
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("mlp", prefix),  # 前缀 # 参数名前缀
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)  # 输入层归一化 # 输入层 RMS 归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)  # 注意力后层归一化 # 注意力后 RMS 归一化

    def forward(  # 前向传播 # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码 # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差 # 残差张量
    ) -> torch.Tensor:  # 返回隐藏状态和残差 # 返回隐藏状态和残差
        # Self Attention # 自注意力
        if residual is None:  # 如果没有残差 # 如果没有残差
            residual = hidden_states  # 保存残差 # 保存残差连接
            hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化 # 应用输入层归一化
        else:  # 否则 # 有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 归一化并更新残差 # 归一化并更新残差

        hidden_states = self.self_attn(  # 自注意力前向 # 计算自注意力
            positions=positions,  # 位置编码 # 位置编码
            hidden_states=hidden_states,  # 隐藏状态 # 隐藏状态
            forward_batch=forward_batch,  # 前向批次 # 前向批次信息
        )

        # Fully Connected # 全连接（MoE）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后归一化 # 应用注意力后归一化
        hidden_states = self.mlp(hidden_states)  # MoE 前向 # 计算 MoE
        return hidden_states, residual  # 返回隐藏状态和残差 # 返回隐藏状态和残差


class OlmoeModel(nn.Module):
    """OLMoE 模型主体，包含嵌入层、多个解码器层和最终归一化"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置 # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.padding_idx = config.pad_token_id  # 填充索引 # 填充标记 ID
        self.vocab_size = config.vocab_size  # 词表大小 # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入 # 词表并行嵌入层
            config.vocab_size,  # 词表大小 # 词表大小
            config.hidden_size,  # 隐藏层大小 # 隐藏层维度
            prefix=add_prefix("embed_tokens", prefix),  # 前缀 # 参数名前缀
        )
        self.layers = make_layers(  # 构建解码器层 # 构建解码器层列表
            config.num_hidden_layers,  # 隐藏层数量 # 隐藏层数量
            lambda idx, prefix: OlmoeDecoderLayer(  # 创建解码器层 # 创建 OLMoE 解码器层
                config=config,  # 配置 # 模型配置
                quant_config=quant_config,  # 量化配置 # 量化配置
                layer_id=idx,  # 层 ID # 层 ID
                prefix=prefix,  # 前缀 # 参数名前缀
            ),
            prefix=add_prefix("layers", prefix),  # 前缀 # 参数名前缀
        )
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)  # 最终归一化 # 最终 RMS 归一化

    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 输入嵌入（可选）
    ) -> torch.Tensor:  # 返回隐藏状态 # 返回隐藏状态
        if input_embeds is None:  # 如果没有提供嵌入 # 如果没有提供嵌入
            hidden_states = self.embed_tokens(input_ids)  # 词嵌入 # 通过词嵌入层获取嵌入
        else:  # 否则 # 使用提供的嵌入
            hidden_states = input_embeds  # 使用输入嵌入 # 使用输入嵌入
        residual = None  # 初始化残差 # 初始化残差
        for i in range(len(self.layers)):  # 遍历层 # 遍历所有解码器层
            layer = self.layers[i]  # 获取层 # 获取当前层
            hidden_states, residual = layer(  # 解码器层前向 # 解码器层前向传播
                positions, hidden_states, forward_batch, residual  # 位置、隐藏状态、批次、残差 # 传入位置、隐藏状态、批次和残差
            )
        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化 # 应用最终归一化
        return hidden_states  # 返回隐藏状态 # 返回隐藏状态


class OlmoeForCausalLM(nn.Module):
    """OLMoE 因果语言模型"""

    fall_back_to_pt_during_load = False  # 加载时不回退到 PyTorch # 加载权重时不回退到 PyTorch

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置 # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.config = config  # 保存配置 # 保存模型配置
        self.quant_config = quant_config  # 保存量化配置 # 保存量化配置
        self.model = OlmoeModel(  # 模型主体 # 创建 OLMoE 模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 配置和量化 # 传入配置和量化配置
        )
        self.lm_head = ParallelLMHead(  # 语言模型头 # 创建并行语言模型头
            config.vocab_size,  # 词表大小 # 词表大小
            config.hidden_size,  # 隐藏层大小 # 隐藏层维度
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 前缀 # 参数名前缀
        )
        self.logits_processor = LogitsProcessor(config)  # logits 处理器 # 创建 logits 处理器

    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 输入嵌入（可选）
    ) -> torch.Tensor:  # 返回张量 # 返回 logits
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 模型前向 # 模型前向传播
        return self.logits_processor(  # logits 处理 # 通过 logits 处理器计算 logits
            input_ids, hidden_states, self.lm_head, forward_batch  # 输入参数 # 输入参数
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重 # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射 # 堆叠参数映射
            # (param_name, shard_name, shard_id) # (参数名, 分片名, 分片 ID)
            ("qkv_proj", "q_proj", "q"),  # Q 投影 # Q 投影映射
            ("qkv_proj", "k_proj", "k"),  # K 投影 # K 投影映射
            ("qkv_proj", "v_proj", "v"),  # V 投影 # V 投影映射
            ("gate_up_proj", "gate_proj", 0),  # 门控投影 # 门控投影映射
            ("gate_up_proj", "up_proj", 1),  # 上投影 # 上投影映射
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id) # 专家参数映射
        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射 # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj",  # 检查点门控投影名 # 检查点中门控投影的名称
            ckpt_down_proj_name="down_proj",  # 检查点下投影名 # 检查点中下投影的名称
            ckpt_up_proj_name="up_proj",  # 检查点上投影名 # 检查点中上投影的名称
            num_experts=self.config.num_experts,  # 专家数量 # 专家数量
        )

        params_dict = dict(self.named_parameters())  # 参数字典 # 获取模型参数字典
        for name, loaded_weight in weights:  # 遍历权重 # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码的逆频率 # 跳过旋转位置编码的逆频率
                continue  # 继续 # 跳过
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射 # 遍历堆叠参数映射
                # Skip non-stacked layers and experts (experts handled below). # 跳过非堆叠层和专家（专家在下面处理）
                if weight_name not in name:  # 如果权重名不在参数名中 # 如果权重名不在参数名中
                    continue  # 继续 # 跳过
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:  # 如果是专家权重则跳过 # 如果是专家权重则跳过（在下面处理）
                    continue  # 继续 # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名 # 替换权重名
                # Skip loading extra bias for GPTQ models. # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 跳过不存在的偏置
                    continue  # 继续 # 跳过
                if name not in params_dict:  # 如果参数不存在 # 如果参数不存在
                    continue  # 继续 # 跳过

                param = params_dict[name]  # 获取参数 # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器 # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重 # 加载权重分片
                break  # 跳出内层循环 # 跳出内层循环
            else:  # 非堆叠参数 # 非堆叠参数
                for mapping in expert_params_mapping:  # 遍历专家参数映射 # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射 # 解包映射元组
                    if weight_name not in name:  # 如果权重名不在参数名中 # 如果权重名不在参数名中
                        continue  # 继续 # 跳过
                    name = name.replace(weight_name, param_name)  # 替换权重名 # 替换权重名
                    param = params_dict[name]  # 获取参数 # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器 # 获取权重加载器
                    weight_loader(  # 加载权重 # 加载专家权重
                        param,  # 参数 # 参数
                        loaded_weight,  # 加载的权重 # 加载的权重
                        name,  # 参数名 # 参数名
                        shard_id=shard_id,  # 分片 ID # 分片 ID
                        expert_id=expert_id,  # 专家 ID # 专家 ID
                    )
                    break  # 跳出内层循环 # 跳出内层循环
                else:  # 非专家参数 # 非专家参数
                    # Skip loading extra bias for GPTQ models. # 跳过 GPTQ 模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 跳过不存在的偏置
                        continue  # 继续 # 跳过
                    # Remapping the name of FP8 kv-scale. # 重映射 FP8 KV 缩放的名称
                    if name.endswith("kv_scale"):  # 如果是 KV 缩放 # 如果是 KV 缩放参数
                        remapped_kv_scale_name = name.replace(  # 重映射名称 # 重映射名称
                            ".kv_scale", ".attn.kv_scale"  # 替换后缀 # 替换后缀
                        )
                        if remapped_kv_scale_name not in params_dict:  # 如果重映射后的名称不存在 # 如果重映射后的名称不存在
                            print_warning_once(  # 打印警告 # 打印一次性警告
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded."
                            )
                            continue  # 继续 # 跳过
                        else:  # 否则 # 重映射成功
                            name = remapped_kv_scale_name  # 使用重映射后的名称 # 使用重映射后的名称

                    param = params_dict[name]  # 获取参数 # 获取参数
                    weight_loader = getattr(  # 获取权重加载器 # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认使用标准加载器 # 默认使用标准权重加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重 # 加载权重


EntryClass = OlmoeForCausalLM  # 入口类 # 模型入口类
