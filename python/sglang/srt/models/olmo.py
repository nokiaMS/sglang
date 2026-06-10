# OLMo 因果语言模型实现
# 该文件实现了推理专用的 OLMo 模型，兼容 HuggingFace 权重格式，
# 包含注意力层、MLP 层、解码器层和完整的因果语言模型。

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
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/olmo.py#L1
"""Inference-only OLMo model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple  # 导入类型提示 # 导入类型提示

import torch  # 导入 PyTorch # 导入 PyTorch 框架
from torch import nn  # 导入神经网络模块 # 导入神经网络模块
from transformers import OlmoConfig  # 导入 OLMo 配置 # 导入 OLMo 配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入张量并行工具 # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import SiluAndMul  # 导入 SiLU 激活 # 导入 SiLU 和乘法激活函数
from sglang.srt.layers.linear import (  # 导入线性层 # 导入各种并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层 # 合并列并行线性层
    QKVParallelLinear,  # QKV 并行线性层 # QKV 并行线性层
    RowParallelLinear,  # 行并行线性层 # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器 # 导入 logits 处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力 # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码 # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入 # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头 # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入 # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器 # 导入默认权重加载器
from sglang.srt.utils import add_prefix, make_layers  # 导入工具函数 # 导入前缀添加和层构建工具


class OlmoAttention(nn.Module):
    """
    This is the attention block where the output is computed as
    ``Attention(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """
    """OLMo 注意力模块，输出计算为 Attention(LN(x))"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: OlmoConfig,  # 模型配置 # OLMo 模型配置
        layer_id: int = 0,  # 层 ID # 层 ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.config = config  # 保存配置 # 保存模型配置
        self.hidden_size = config.hidden_size  # 隐藏层大小 # 隐藏层维度
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()  # 张量并行世界大小 # 获取张量并行世界大小
        self.total_num_heads = config.num_attention_heads  # 总注意力头数 # 总注意力头数

        assert self.hidden_size % self.total_num_heads == 0  # 断言隐藏大小可被头数整除 # 断言隐藏大小可被头数整除
        assert self.total_num_heads % tensor_model_parallel_world_size == 0  # 断言头数可被并行度整除 # 断言头数可被并行度整除

        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size  # 每个并行的头数 # 每个并行进程的头数
        self.head_dim = self.hidden_size // self.total_num_heads  # 每个头的维度 # 每个头的维度
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置编码数 # 最大位置编码数
        self.rope_theta = config.rope_parameters["rope_theta"]  # RoPE theta # 旋转位置编码的 theta 参数
        self.clip_qkv = config.clip_qkv  # QKV 裁剪值 # QKV 裁剪值

        # Attention input projection. Projects x -> (q, k, v) # 注意力输入投影，将 x 投影为 (q, k, v)
        self.qkv_proj = QKVParallelLinear(  # QKV 投影 # QKV 并行线性投影
            self.hidden_size,  # 输入大小 # 输入维度
            self.head_dim,  # 头维度 # 每个头的维度
            self.total_num_heads,  # 总头数 # 总注意力头数
            bias=config.attention_bias,  # 偏置 # 是否使用偏置
            prefix=add_prefix("qkv_proj", prefix),  # 前缀 # 参数名前缀
        )

        # Rotary embeddings. # 旋转位置编码
        self.rotary_emb = get_rope(  # 获取旋转位置编码 # 获取旋转位置编码
            self.head_dim,  # 头维度 # 每个头的维度
            rotary_dim=self.head_dim,  # 旋转维度 # 旋转维度
            max_position=self.max_position_embeddings,  # 最大位置 # 最大位置数
            base=self.rope_theta,  # 基数 # theta 基数
        )
        self.scaling = self.head_dim**-0.5  # 缩放因子 # 注意力缩放因子
        self.attn = RadixAttention(  # 基数注意力 # 创建基数注意力
            self.num_heads,  # 头数 # 头数
            self.head_dim,  # 头维度 # 每个头的维度
            self.scaling,  # 缩放因子 # 缩放因子
            num_kv_heads=self.num_heads,  # KV 头数 # KV 头数（与查询头数相同）
            layer_id=layer_id,  # 层 ID # 层 ID
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("attn", prefix),  # 前缀 # 参数名前缀
        )

        # Attention output projection. # 注意力输出投影
        self.o_proj = RowParallelLinear(  # 输出投影 # 行并行线性投影
            self.hidden_size,  # 输入大小 # 输入维度
            self.hidden_size,  # 输出大小 # 输出维度
            bias=config.attention_bias,  # 偏置 # 是否使用偏置
            prefix=add_prefix("o_proj", prefix),  # 前缀 # 参数名前缀
        )

    def forward(  # 前向传播 # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码 # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
    ) -> torch.Tensor:  # 返回张量 # 返回注意力输出
        qkv, _ = self.qkv_proj(hidden_states)  # QKV 投影 # 计算 QKV 投影
        if self.clip_qkv is not None:  # 如果需要裁剪 QKV # 如果需要裁剪 QKV
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)  # 裁剪 QKV # 裁剪 QKV 值
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # 分割 QKV # 分割为 Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码 # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力 # 计算注意力
        output, _ = self.o_proj(attn_output)  # 输出投影 # 输出投影
        return output  # 返回输出 # 返回输出


class OlmoMLP(nn.Module):
    """
    This is the MLP block where the output is computed as
    ``MLP(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """
    """OLMo MLP 模块，包含门控投影和下投影"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: OlmoConfig,  # 模型配置 # OLMo 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.config = config  # 保存配置 # 保存模型配置
        self.hidden_size = config.hidden_size  # 隐藏层大小 # 隐藏层维度
        self.intermediate_size = config.intermediate_size  # 中间层大小 # 中间层维度

        # Feed-forward input projection. # 前馈输入投影
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控上投影 # 合并列并行门控上投影
            self.hidden_size,  # 输入大小 # 输入维度
            [self.intermediate_size] * 2,  # 输出大小（门控和上投影） # 门控和上投影的输出维度
            bias=False,  # 无偏置 # 无偏置
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 前缀 # 参数名前缀
        )

        # Activation function. # 激活函数
        self.act_fn = SiluAndMul()  # SiLU 和乘法激活 # SiLU 和乘法组合激活函数

        # Feed-forward output projection. # 前馈输出投影
        self.down_proj = RowParallelLinear(  # 下投影 # 行并行下投影
            self.intermediate_size,  # 输入大小 # 输入维度
            self.hidden_size,  # 输出大小 # 输出维度
            bias=False,  # 无偏置 # 无偏置
            quant_config=quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 前缀 # 参数名前缀
        )

    def forward(  # 前向传播 # 前向传播方法
        self,
        x: torch.Tensor,  # 输入张量 # 输入张量
    ) -> torch.Tensor:  # 返回张量 # 返回 MLP 输出
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影 # 计算门控和上投影
        x = self.act_fn(gate_up)  # 激活函数 # 应用 SiLU 和乘法激活
        x, _ = self.down_proj(x)  # 下投影 # 计算下投影
        return x  # 返回输出 # 返回输出


class OlmoDecoderLayer(nn.Module):
    """
    This is a typical transformer block where the output is
    computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """
    """OLMo 解码器层，包含注意力和 MLP 子层"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: OlmoConfig,  # 模型配置 # OLMo 模型配置
        layer_id: int = 0,  # 层 ID # 层 ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        # Attention block. # 注意力块
        self.self_attn = OlmoAttention(  # 自注意力 # 创建 OLMo 注意力层
            config,  # 配置 # 模型配置
            layer_id,  # 层 ID # 层 ID
            quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 前缀 # 参数名前缀
        )

        # MLP block. # MLP 块
        self.mlp = OlmoMLP(  # MLP # 创建 OLMo MLP 层
            config,  # 配置 # 模型配置
            quant_config,  # 量化配置 # 量化配置
            prefix=add_prefix("mlp", prefix),  # 前缀 # 参数名前缀
        )

        # LayerNorm # 层归一化
        self.input_layernorm = nn.LayerNorm(  # 输入层归一化 # 输入层归一化
            config.hidden_size, elementwise_affine=False, bias=False  # 无仿射参数和偏置 # 无仿射参数和偏置
        )
        self.post_attention_layernorm = nn.LayerNorm(  # 注意力后层归一化 # 注意力后层归一化
            config.hidden_size, elementwise_affine=False, bias=False  # 无仿射参数和偏置 # 无仿射参数和偏置
        )

    def forward(  # 前向传播 # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码 # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:  # 返回隐藏状态和可选元组 # 返回隐藏状态
        # Attention block. # 注意力块
        residual = hidden_states  # 保存残差 # 保存残差连接
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化 # 应用输入层归一化
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)  # 自注意力 # 计算自注意力
        hidden_states = hidden_states + residual  # 残差连接 # 添加残差连接

        # MLP block. # MLP 块
        residual = hidden_states  # 保存残差 # 保存残差连接
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化 # 应用注意力后层归一化
        hidden_states = self.mlp(hidden_states)  # MLP # 计算 MLP
        hidden_states = residual + hidden_states  # 残差连接 # 添加残差连接
        return hidden_states  # 返回隐藏状态 # 返回隐藏状态


class OlmoModel(nn.Module):
    """OLMo 模型主体，包含嵌入层、多个解码器层和最终归一化"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: OlmoConfig,  # 模型配置 # OLMo 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.config = config  # 保存配置 # 保存模型配置

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入 # 词表并行嵌入层
            config.vocab_size,  # 词表大小 # 词表大小
            config.hidden_size,  # 隐藏层大小 # 隐藏层维度
            prefix=add_prefix("embed_tokens", prefix),  # 前缀 # 参数名前缀
        )
        self.layers = make_layers(  # 构建解码器层 # 构建解码器层列表
            config.num_hidden_layers,  # 隐藏层数量 # 隐藏层数量
            lambda idx, prefix: OlmoDecoderLayer(  # 创建解码器层 # 创建 OLMo 解码器层
                layer_id=idx,  # 层 ID # 层 ID
                config=config,  # 配置 # 模型配置
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=prefix,  # 前缀 # 参数名前缀
            ),
            prefix=add_prefix("layers", prefix),  # 前缀 # 参数名前缀
        )
        self.norm = nn.LayerNorm(  # 最终层归一化 # 最终层归一化
            config.hidden_size, elementwise_affine=False, bias=False  # 无仿射参数和偏置 # 无仿射参数和偏置
        )

    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 输入嵌入（可选）
    ) -> torch.Tensor:  # 返回隐藏状态 # 返回隐藏状态
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        # Get embeddings of input. # 获取输入嵌入
        # shape: (batch_size, seq_len, d_model) # 形状: (batch_size, seq_len, d_model)

        if input_embeds is None:  # 如果没有提供嵌入 # 如果没有提供嵌入
            hidden_states = self.embed_tokens(input_ids)  # 词嵌入 # 通过词嵌入层获取嵌入
        else:  # 否则 # 使用提供的嵌入
            hidden_states = input_embeds  # 使用输入嵌入 # 使用输入嵌入

        # Apply blocks one-by-one. # 逐层应用解码器
        for layer_id, decoder_layer in enumerate(self.layers):  # 遍历层 # 遍历所有解码器层
            # shape: (batch_size, seq_len, d_model) # 形状: (batch_size, seq_len, d_model)
            hidden_states = decoder_layer(  # 解码器层前向 # 解码器层前向传播
                positions,  # 位置编码 # 位置编码
                hidden_states,  # 隐藏状态 # 隐藏状态
                forward_batch,  # 前向批次 # 前向批次信息
            )

        # Apply final layer norm. # 应用最终层归一化
        # shape: (batch_size, seq_len or 1, d_model) # 形状: (batch_size, seq_len or 1, d_model)
        hidden_states = self.norm(hidden_states)  # 最终归一化 # 应用最终层归一化
        return hidden_states  # 返回隐藏状态 # 返回隐藏状态


class OlmoForCausalLM(nn.Module):
    """
    Extremely barebones HF model wrapper.
    """
    """OLMo 因果语言模型，极其简化的 HuggingFace 模型封装"""

    def __init__(  # 初始化方法 # 初始化方法
        self,
        config: OlmoConfig,  # 模型配置 # OLMo 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 量化配置
        prefix: str = "",  # 前缀 # 参数名前缀
    ):
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.config = config  # 保存配置 # 保存模型配置
        self.model = OlmoModel(config, quant_config, prefix=add_prefix("model", prefix))  # 模型主体 # 创建 OLMo 模型主体
        if config.tie_word_embeddings:  # 如果绑定词嵌入 # 如果绑定词嵌入权重
            self.lm_head = self.model.embed_tokens  # 语言模型头共享嵌入 # 语言模型头共享嵌入权重
        else:  # 否则 # 不绑定
            self.unpadded_vocab_size = config.vocab_size  # 未填充的词表大小 # 未填充的词表大小
            self.lm_head = ParallelLMHead(  # 并行语言模型头 # 创建并行语言模型头
                self.unpadded_vocab_size,  # 词表大小 # 词表大小
                config.hidden_size,  # 隐藏层大小 # 隐藏层维度
                org_num_embeddings=config.vocab_size,  # 原始嵌入数 # 原始嵌入数
                quant_config=quant_config,  # 量化配置 # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 前缀 # 参数名前缀
            )
        self.logits_processor = LogitsProcessor(config)  # logits 处理器 # 创建 logits 处理器

    @torch.no_grad()  # 禁用梯度 # 禁用梯度计算
    def forward(  # 前向传播 # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 ID # 输入标记 ID
        positions: torch.Tensor,  # 位置编码 # 位置编码
        forward_batch: ForwardBatch,  # 前向批次 # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 输入嵌入（可选）
    ) -> torch.Tensor:  # 返回张量 # 返回 logits
        hidden_states = self.model(  # 模型前向 # 模型前向传播
            input_ids=input_ids,  # 输入 ID # 输入标记 ID
            positions=positions,  # 位置编码 # 位置编码
            forward_batch=forward_batch,  # 前向批次 # 前向批次信息
            input_embeds=input_embeds,  # 输入嵌入 # 输入嵌入
        )
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
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 参数字典 # 获取模型参数字典
        for name, loaded_weight in weights:  # 遍历权重 # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码的逆频率 # 跳过旋转位置编码的逆频率
                continue  # 继续 # 跳过
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的余弦/正弦 # 跳过缓存的余弦/正弦值
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue  # 继续 # 跳过
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的 lm_head 权重 # 跳过绑定的语言模型头权重
                continue  # 继续 # 跳过
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠映射 # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中 # 如果权重名不在参数名中
                    continue  # 继续 # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名 # 替换权重名
                # Skip loading extra bias for GPTQ models. # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 跳过不存在的偏置
                    continue  # 继续 # 跳过
                param = params_dict[name]  # 获取参数 # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器 # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重 # 加载权重分片
                break  # 跳出内层循环 # 跳出内层循环
            else:  # 非堆叠参数 # 非堆叠参数
                # Skip loading extra bias for GPTQ models. # 跳过 GPTQ 模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 跳过不存在的偏置
                    continue  # 继续 # 跳过
                param = params_dict[name]  # 获取参数 # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器 # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重 # 加载权重


EntryClass = OlmoForCausalLM  # 入口类 # 模型入口类
