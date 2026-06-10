# MiniCPM模型：基于Llama架构的轻量级因果语言模型实现
# 本文件实现了MiniCPM模型，包括MLP、注意力、解码器层和完整模型
# 支持缩放嵌入、缩放深度残差连接和专家混合(MoE)权重加载

# Copyright 2023-2024 SGLang Team  # SGLang团队版权声明 # SGLang团队版权
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache 2.0许可证授权 # Apache 2.0许可证
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件 # 不得违反许可证使用
# You may obtain a copy of the License at  # 可在以下地址获取许可证副本 # 获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证链接 # 许可证URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意 # 法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 根据许可证分发的软件按"原样"提供 # 按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证 # 无任何保证
# See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限的特定语言 # 查看许可证
# limitations under the License.  # 和限制。 # 了解限制
# ==============================================================================  # 分隔线 # 分隔线
"""Inference-only MiniCPM model compatible with HuggingFace weights."""  # 仅推理的MiniCPM模型，兼容HuggingFace权重。 # 仅推理的MiniCPM模型，兼容HuggingFace权重

import math  # 导入数学库 # 导入数学工具库
from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示 # 导入类型提示工具

import torch  # 导入PyTorch库 # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块 # 导入PyTorch神经网络模块

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入TP世界大小 # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU激活函数 # 导入SiLU与乘法组合激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化 # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入线性层 # 导入并行线性层组件
    MergedColumnParallelLinear,  # 合并列并行线性层 # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层 # QKV并行线性层
    RowParallelLinear,  # 行并行线性层 # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器 # 导入logits后处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置 # 导入量化基础配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力 # 导入基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码 # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入 # 导入词表并行嵌入组件
    ParallelLMHead,  # 并行语言模型头 # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入 # 词表并行嵌入层
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向传播批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器 # 导入默认权重加载工具
from sglang.srt.utils import add_prefix  # 导入前缀工具 # 导入前缀添加工具
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入RoPE配置工具 # 导入HuggingFace旋转位置编码配置工具


class MiniCPMMLP(nn.Module):  # MiniCPM MLP模块 # MiniCPM的多层感知机模块
    def __init__(  # 初始化方法 # 初始化函数
        self,
        hidden_size: int,  # 隐藏层大小 # 隐藏层维度
        intermediate_size: int,  # 中间层大小 # 中间层维度
        hidden_act: str,  # 隐藏层激活函数 # 隐藏层激活函数名称
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用nn.Module的初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控上投影 # 创建门控和上投影的合并线性层
            hidden_size,
            [intermediate_size] * 2,  # 两个中间层大小 # 门控和上投影各一个中间层
            bias=False,  # 无偏置 # 不使用偏置
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(  # 下投影 # 创建行并行下投影线性层
            intermediate_size,
            hidden_size,
            bias=False,  # 无偏置 # 不使用偏置
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":  # 检查激活函数 # 验证激活函数是否为silu
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # 创建SiLU激活函数 # 创建SiLU与乘法组合激活函数

    def forward(self, x):  # 前向传播方法 # 前向传播函数
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影 # 通过门控上投影层
        x = self.act_fn(gate_up)  # 激活函数 # 应用SiLU激活和门控乘法
        x, _ = self.down_proj(x)  # 下投影 # 通过下投影层
        return x  # 返回输出 # 返回MLP输出


class MiniCPMAttention(nn.Module):  # MiniCPM注意力模块 # MiniCPM的注意力机制模块
    def __init__(  # 初始化方法 # 初始化函数
        self,
        hidden_size: int,  # 隐藏层大小 # 隐藏层维度
        num_heads: int,  # 注意力头数 # 注意力头数量
        num_kv_heads: int,  # KV头数 # 键值头数量
        layer_id: int = 0,  # 层ID # 层索引
        rope_theta: float = 10000,  # RoPE基数 # 旋转位置编码的基数
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置 # 旋转位置编码缩放配置
        max_position_embeddings: int = 8192,  # 最大位置嵌入 # 最大位置编码数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用nn.Module的初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小 # 存储隐藏层维度
        tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小 # 获取张量并行世界大小
        self.total_num_heads = num_heads  # 保存总头数 # 存储总注意力头数
        assert self.total_num_heads % tp_size == 0  # 断言头数可被TP大小整除 # 验证头数可被TP大小整除
        self.num_heads = self.total_num_heads // tp_size  # 计算每卡头数 # 计算每个TP秩的注意力头数
        self.total_num_kv_heads = num_kv_heads  # 保存总KV头数 # 存储总KV头数
        if self.total_num_kv_heads >= tp_size:  # 如果KV头数大于等于TP大小 # 判断KV头数是否大于等于TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此我们分区
            # the KV heads across multiple tensor parallel GPUs.  # KV头在多个张量并行GPU之间分区。 # KV头数大于TP大小，在GPU间分区
            assert self.total_num_kv_heads % tp_size == 0  # 断言KV头数可被TP大小整除 # 验证KV头数可被TP大小整除
        else:
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此我们复制
            # the KV heads across multiple tensor parallel GPUs.  # KV头在多个张量并行GPU之间复制。 # KV头数小于TP大小，在GPU间复制
            assert tp_size % self.total_num_kv_heads == 0  # 断言TP大小可被KV头数整除 # 验证TP大小可被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 计算每卡KV头数 # 计算每个TP秩的KV头数
        self.head_dim = hidden_size // self.total_num_heads  # 计算头维度 # 计算每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q大小 # 查询向量总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV大小 # 键值向量总大小
        self.scaling = self.head_dim**-0.5  # 缩放因子 # 注意力缩放因子
        self.rope_theta = rope_theta  # 保存RoPE基数 # 存储旋转位置编码基数
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置嵌入 # 存储最大位置编码数

        self.qkv_proj = QKVParallelLinear(  # QKV投影 # 创建QKV并行线性投影
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,  # 无偏置 # 不使用偏置
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 输出投影 # 创建行并行输出投影
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,  # 无偏置 # 不使用偏置
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(  # 旋转位置编码 # 创建旋转位置编码
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(  # 基数注意力 # 创建基数注意力层
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(  # 前向传播方法 # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置ID # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向传播批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影 # 通过QKV投影层
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割QKV # 将QKV分割为查询、键、值
        orig_dtype = q.dtype  # 保存原始数据类型 # 保存查询的原始数据类型
        q, k = q.float(), k.float()  # 转换为float32 # 将查询和键转换为float32
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码 # 应用旋转位置编码
        q, k = q.to(orig_dtype), k.to(orig_dtype)  # 恢复原始数据类型 # 恢复为原始数据类型
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力 # 通过基数注意力层计算
        output, _ = self.o_proj(attn_output)  # 输出投影 # 通过输出投影层
        return output  # 返回输出 # 返回注意力输出


class MiniCPMDecoderLayer(nn.Module):  # MiniCPM解码器层 # MiniCPM的解码器层模块
    def __init__(  # 初始化方法 # 初始化函数
        self,
        config,  # 模型配置 # 模型配置对象
        layer_id: int = 0,  # 层ID # 层索引
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用nn.Module的初始化
        self.config = config  # 保存配置 # 存储模型配置
        self.hidden_size = config.hidden_size  # 保存隐藏层大小 # 存储隐藏层维度
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置 # 获取旋转位置编码配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置嵌入 # 获取最大位置编码数
        self.self_attn = MiniCPMAttention(  # 自注意力 # 创建MiniCPM注意力层
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = MiniCPMMLP(  # MLP # 创建MiniCPM MLP层
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化 # 创建输入RMS归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化 # 创建注意力后的RMS归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 前向传播方法 # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置ID # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态 # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向传播批次信息
        residual: Optional[torch.Tensor],  # 残差 # 可选的残差张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention  # 自注意力 # 自注意力计算
        residual = hidden_states  # 保存残差 # 保存输入作为残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化 # 应用输入层归一化
        hidden_states = self.self_attn(  # 自注意力计算 # 通过自注意力层
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states * (  # 带缩放的残差连接 # 带深度缩放的残差连接
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        # Fully Connected  # 全连接 # 全连接层计算
        residual = hidden_states  # 保存残差 # 保存输入作为残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化 # 应用注意力后归一化
        hidden_states = self.mlp(hidden_states)  # MLP计算 # 通过MLP层
        hidden_states = residual + hidden_states * (  # 带缩放的残差连接 # 带深度缩放的残差连接
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
        )

        return hidden_states, None  # 返回隐藏状态和无残差 # 返回隐藏状态和None


class MiniCPMModel(nn.Module):  # MiniCPM模型 # MiniCPM模型主体
    def __init__(  # 初始化方法 # 初始化函数
        self,
        config,  # 模型配置 # 模型配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用nn.Module的初始化
        self.config = config  # 保存配置 # 存储模型配置
        self.padding_idx = config.pad_token_id  # 填充索引 # 存储填充token ID
        self.vocab_size = config.vocab_size  # 词表大小 # 存储词表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层 # 创建词表并行嵌入层
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,  # 原始嵌入数 # 原始嵌入数量
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(  # 解码器层列表 # 创建解码器层列表
            [
                MiniCPMDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)  # 遍历层数 # 为每层创建解码器
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化 # 创建最终RMS归一化

    def forward(  # 前向传播方法 # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入ID # 输入token ID张量
        positions: torch.Tensor,  # 位置ID # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向传播批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 可选的输入嵌入
    ) -> torch.Tensor:
        if input_embeds is None:  # 如果没有提供嵌入 # 判断是否使用输入嵌入
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb  # 获取嵌入并缩放 # 通过嵌入层获取隐藏状态并缩放
        else:
            hidden_states = input_embeds  # 使用提供的嵌入 # 使用传入的嵌入
        residual = None  # 初始残差为None # 初始残差为None

        for i in range(len(self.layers)):  # 遍历所有层 # 遍历解码器层
            layer = self.layers[i]  # 获取当前层 # 获取第i层
            hidden_states, residual = layer(  # 通过当前层 # 通过解码器层处理
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states = self.norm(hidden_states)  # 最终归一化 # 应用最终归一化
        return hidden_states  # 返回隐藏状态 # 返回归一化后的隐藏状态


class MiniCPMForCausalLM(nn.Module):  # MiniCPM因果语言模型 # MiniCPM因果语言模型，用于文本生成
    def __init__(  # 初始化方法 # 初始化函数
        self,
        config,  # 模型配置 # 模型配置对象
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置 # 可选的量化配置
        prefix: str = "",  # 参数前缀 # 参数名称前缀
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用nn.Module的初始化
        self.config = config  # 保存配置 # 存储模型配置

        self.num_experts = getattr(self.config, "num_experts", 0)  # 专家数 # 获取MoE专家数量
        self.quant_config = quant_config  # 保存量化配置 # 存储量化配置
        self.model = MiniCPMModel(  # 创建MiniCPM模型 # 创建MiniCPM模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)  # 已注释的语言模型头 # 已注释的语言模型头
        if not self.config.tie_word_embeddings:  # 如果不绑定词嵌入 # 判断是否绑定输入输出词嵌入
            self.lm_head = ParallelLMHead(  # 创建独立的语言模型头 # 创建并行的语言模型头
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,  # 原始嵌入数 # 原始嵌入数量
                prefix=add_prefix("lm_head", prefix),
            )

        self.scale_width = self.config.hidden_size / self.config.dim_model_base  # 宽度缩放因子 # 计算宽度缩放因子

        self.logits_processor = LogitsProcessor(config)  # 创建logits处理器 # 创建logits后处理器

    @torch.no_grad()  # 禁用梯度计算 # 装饰器：禁用梯度计算
    def forward(  # 前向传播方法 # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入ID # 输入token ID张量
        positions: torch.Tensor,  # 位置ID # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次 # 前向传播批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入 # 可选的输入嵌入
    ) -> torch.Tensor:
        if input_embeds is not None:  # 如果提供了嵌入 # 判断是否使用输入嵌入
            input_embeds = input_embeds * self.config.scale_emb  # 缩放嵌入 # 对输入嵌入进行缩放
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 获取隐藏状态 # 通过模型获取隐藏状态
        hidden_states = hidden_states / self.scale_width  # 宽度缩放 # 对隐藏状态进行宽度缩放
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入 # 判断是否绑定输入输出词嵌入
            lm_head = self.model.embed_tokens  # 使用嵌入层作为语言模型头 # 使用嵌入层作为语言模型头
        else:
            lm_head = self.lm_head  # 使用独立的语言模型头 # 使用独立的语言模型头
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)  # 返回logits # 通过logits处理器计算并返回

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法 # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射 # 需要堆叠的参数映射列表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID) # 参数名、分片名和分片ID的映射
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = [  # 专家参数映射 # MoE专家参数映射列表
            # (param_name, weight_name, expert_id)  # (参数名, 权重名, 专家ID) # 参数名、权重名和专家ID的映射
            (
                "ws" if weight_name in ["w1", "w3"] else "w2s",  # 权重名映射 # w1/w3映射为ws，w2映射为w2s
                f"experts.{expert_id}.{weight_name}.weight",
                expert_id,
            )
            for expert_id in range(self.num_experts)  # 遍历专家 # 为每个专家创建映射
            for weight_name in ["w1", "w2", "w3"]  # 遍历权重名 # 为每种权重创建映射
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典 # 将模型参数转为字典
        for name, loaded_weight in weights:  # 遍历权重 # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入频率 # 跳过旋转嵌入频率
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的旋转嵌入 # 跳过旋转嵌入的缓存值
                # Models trained using ColossalAI may include these tensors in  # 使用ColossalAI训练的模型可能包含这些张量
                # the checkpoint. Skip them.  # 在检查点中。跳过它们。 # 跳过ColossalAI训练模型中的缓存张量
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过绑定的lm_head权重 # 如果词嵌入绑定则跳过lm_head权重
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射 # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中 # 检查权重名是否匹配
                    continue
                name = name.replace(weight_name, param_name)  # 替换权重名 # 将分片名替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载。 # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 如果偏置不在参数字典中则跳过
                    continue
                param = params_dict[name]  # 获取参数 # 从字典中获取参数
                weight_loader = param.weight_loader  # 获取权重加载器 # 获取参数的权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重 # 使用权重加载器加载权重
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:  # 遍历专家参数映射 # 遍历MoE专家参数映射
                    if weight_name not in name:  # 如果权重名不在参数名中 # 检查权重名是否匹配
                        continue
                    name = name.replace(weight_name, param_name)  # 替换权重名 # 将原始权重名替换为参数名
                    param = params_dict[name]  # 获取参数 # 从字典中获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器 # 获取参数的权重加载器
                    weight_loader(  # 加载权重 # 使用权重加载器加载权重
                        param, loaded_weight, weight_name, expert_id=expert_id
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载。 # 跳过GPTQ模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置 # 如果偏置不在参数字典中则跳过
                        continue
                    param = params_dict[name]  # 获取参数 # 从字典中获取参数
                    weight_loader = getattr(  # 获取权重加载器 # 获取权重加载器或使用默认
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重 # 使用权重加载器加载权重


EntryClass = MiniCPMForCausalLM  # 入口类 # 模型注册入口类
