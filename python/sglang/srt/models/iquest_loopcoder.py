# IQuest LoopCoder 因果语言模型
# 实现了循环注意力机制（Loop Attention），支持多轮循环推理
# 通过门控投影（Gate Projection）在全局注意力和局部注意力之间进行动态混合
# 兼容 HuggingFace 权重的推理专用实现
# Copyright 2023-2024 SGLang Team  # SGLang 团队版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据 Apache 2.0 许可证授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可以在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache 2.0 许可证链接
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 参见许可证以了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================
"""Inference-only LoopCoder model compatible with HuggingFace weights."""  # 仅推理的 LoopCoder 模型，兼容 HuggingFace 权重

import logging  # 导入日志记录模块
from typing import Iterable, Optional, Tuple  # 导入类型提示工具

import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入 PyTorch 神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 导入获取张量并行世界大小的函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入 RMS 归一化层
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV 并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力模块
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 并行词表嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.llama import LlamaMLP as LoopCoderMLP  # 导入 Llama MLP 作为 LoopCoder MLP
from sglang.srt.utils import add_prefix, make_layers  # 导入前缀添加和层创建工具函数
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入 RoPE 配置获取函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class LoopGateProjection(nn.Module):  # 循环门控投影模块
    """Gate projection for mixed attention in Loop 2+.  # 循环 2+ 中混合注意力的门控投影

    Computes: g = sigmoid(linear(Q)) for each head independently.  # 计算方式：g = sigmoid(linear(Q))，每个头独立计算
    This gate determines how much to use Loop1's KV (global) vs current loop's KV (local).  # 该门控决定使用循环1的 KV（全局）与当前循环的 KV（局部）的比例

    Supports tensor parallelism: each GPU handles a subset of heads.  # 支持张量并行：每个 GPU 处理一部分头
    The weight matrix has shape [num_heads, head_dim] and is split along the head dimension.  # 权重矩阵形状为 [头数, 头维度]，沿头维度分割
    """

    def __init__(  # 初始化方法
        self,
        total_num_heads: int,  # 总注意力头数
        head_dim: int,  # 每个头的维度
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空
    ):
        super().__init__()  # 调用父类初始化
        self.total_num_heads = total_num_heads  # 保存总头数
        self.head_dim = head_dim  # 保存头维度
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        assert self.total_num_heads % tp_size == 0  # 断言总头数能被并行大小整除
        self.num_heads = self.total_num_heads // tp_size  # 当前分片的头数

        self.gate_proj = ColumnParallelLinear(  # 创建门控投影线性层（列并行）
            head_dim,  # 输入维度为头维度
            self.total_num_heads,  # 输出维度为总头数
            bias=True,  # 使用偏置
            gather_output=False,  # 不收集输出（保持并行分布）
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_proj", prefix),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        """Compute gate values from query tensor.  # 从查询张量计算门控值

        Args:  # 参数
            query: [num_heads, num_tokens, head_dim]  # 查询张量，形状为 [头数, token数, 头维度]
                where num_heads is the number of heads on this TP rank  # 其中 num_heads 是当前 TP 排名的头数
                and num_tokens = batch * seq_len  # num_tokens = 批次大小 * 序列长度

        Returns:  # 返回
            gate: [num_tokens, num_heads * head_dim] (flattened format matching q shape)  # 门控值，形状为 [token数, 头数*头维度]（展平格式匹配 q 形状）
        """
        num_heads, num_tokens, head_dim = query.shape  # 解析查询张量形状

        assert (  # 断言头数匹配
            num_heads == self.num_heads
        ), f"Expected {self.num_heads} heads, got {num_heads}"

        query_flat = query.reshape(-1, head_dim)  # 展平查询张量

        gate_logits_flat, _ = self.gate_proj(query_flat)  # 计算门控 logits

        gate_logits = gate_logits_flat.reshape(num_heads, num_tokens, self.num_heads)  # 重塑门控 logits

        # Extract diagonal: each head h's query should use output column h  # 提取对角线：每个头 h 的查询应使用输出列 h
        gate_logits = torch.diagonal(gate_logits, dim1=0, dim2=2)  # 提取对角线元素
        gate_logits = gate_logits.transpose(0, 1)  # 转置
        gate_logits = gate_logits.unsqueeze(-1)  # 增加最后一维

        # Apply sigmoid  # 应用 sigmoid 激活函数
        gate = torch.sigmoid(gate_logits)

        # Expand and reshape to match q shape: [num_tokens, num_heads * head_dim]  # 扩展并重塑以匹配 q 形状：[token数, 头数*头维度]
        gate = gate.transpose(0, 1)  # 转置
        gate = gate.expand(-1, -1, head_dim)  # 扩展到头维度
        gate = gate.reshape(num_tokens, num_heads * head_dim)  # 重塑为展平格式

        return gate  # 返回门控值


class LoopCoderAttention(nn.Module):  # LoopCoder 注意力模块
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV 头数（用于 GQA）
        layer_id: int = 0,  # 层 ID，默认为 0
        max_position: int = 4096 * 32,  # 最大位置数，默认 131072
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层 ID
        self.hidden_size = hidden_size  # 保存隐藏层大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = num_heads  # 保存总头数
        assert self.total_num_heads % tp_size == 0  # 断言总头数能被并行大小整除
        self.num_heads = self.total_num_heads // tp_size  # 当前分片的头数
        self.total_num_kv_heads = num_kv_heads  # 保存总 KV 头数
        if self.total_num_kv_heads >= tp_size:  # 如果 KV 头数大于等于并行大小
            assert self.total_num_kv_heads % tp_size == 0  # 断言 KV 头数能被并行大小整除
        else:  # 否则
            assert tp_size % self.total_num_kv_heads == 0  # 断言并行大小能被 KV 头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 当前分片的 KV 头数
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q 的大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV 的大小
        self.scaling = self.head_dim**-0.5  # 缩放因子

        # Get loop_num from config, default to 2 if not specified  # 从配置获取循环数，未指定则默认为 2
        self.loop_num = getattr(config, "loop_num", 2)  # 获取循环次数
        self.loop_window_size = getattr(config, "loop_window_size", 64)  # 获取循环窗口大小

        self.qkv_proj = QKVParallelLinear(  # 创建 QKV 投影层
            hidden_size,  # 输入维度
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # 总 Q 头数
            self.total_num_kv_heads,  # 总 KV 头数
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 创建输出投影层
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),
        )

        rope_theta, rope_scaling = get_rope_config(config)  # 获取 RoPE 配置
        max_position_embeddings = getattr(  # 获取最大位置嵌入数
            config, "max_position_embeddings", max_position
        )

        self.rotary_emb = get_rope(  # 创建旋转位置编码实例
            self.head_dim,
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置数
            base=rope_theta,  # 基础频率
            rope_scaling=rope_scaling,  # 缩放配置
        )

        # Create attention instances for each loop  # 为每个循环创建注意力实例
        # Loop 0: global attention without sliding window for full context  # 循环 0：无滑动窗口的全局注意力，用于完整上下文
        # Loop 1+: local attention with sliding window for recent tokens  # 循环 1+：带滑动窗口的局部注意力，用于近期 token
        # Each loop needs a unique layer_id to avoid KV cache conflicts  # 每个循环需要唯一的 layer_id 以避免 KV 缓存冲突
        self.attn = nn.ModuleList()  # 创建注意力模块列表
        total_layers = getattr(config, "num_hidden_layers", 24)  # 获取总层数，默认 24
        for loop_idx in range(self.loop_num):  # 遍历每个循环
            sliding_window = -1 if loop_idx == 0 else self.loop_window_size  # 循环 0 使用全局注意力，其他使用滑动窗口
            # Use unique layer_id for each loop: loop_idx * total_layers + layer_id  # 为每个循环使用唯一的 layer_id：循环索引 * 总层数 + 层 ID
            # This ensures each loop has its own KV cache space  # 这确保每个循环有自己的 KV 缓存空间
            unique_layer_id = loop_idx * total_layers + layer_id  # 计算唯一的层 ID

            self.attn.append(  # 添加注意力实例
                RadixAttention(
                    self.num_heads,  # 头数
                    self.head_dim,  # 头维度
                    self.scaling,  # 缩放因子
                    num_kv_heads=self.num_kv_heads,  # KV 头数
                    layer_id=unique_layer_id,  # Unique layer_id for each loop  # 每个循环的唯一 layer_id
                    sliding_window_size=sliding_window,  # 滑动窗口大小
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"attn.{loop_idx}", prefix),
                )
            )

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        loop_idx: int,  # 当前循环索引
        gate_proj: Optional[LoopGateProjection] = None,  # 门控投影，可选（循环 1+ 需要）
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 计算 QKV 投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割 Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码

        if loop_idx == 0:  # 如果是第一个循环
            # First loop: standard global attention, save KV to cache  # 第一个循环：标准全局注意力，保存 KV 到缓存
            attn_output = self.attn[0](q, k, v, forward_batch)  # 计算全局注意力输出
        else:  # 循环 2+
            # Loop 2+: mixed attention with learned gating  # 循环 2+：使用学习门控的混合注意力
            # Global attention: read from Loop 0's KV cache without updating (save_kv_cache=False)  # 全局注意力：从循环 0 的 KV 缓存读取，不更新（save_kv_cache=False）
            # This provides full context information  # 这提供完整上下文信息
            # Pass k=None, v=None to read from KV cache instead of recomputing  # 传入 k=None, v=None 从 KV 缓存读取而非重新计算
            global_attn_output = self.attn[0](  # 计算全局注意力输出
                q, None, None, forward_batch, save_kv_cache=False
            )

            # Local attention: use current loop's KV with sliding window  # 局部注意力：使用当前循环的 KV 和滑动窗口
            # This focuses on recent tokens within the window  # 聚焦于窗口内的近期 token
            local_attn_output = self.attn[loop_idx](q, k, v, forward_batch)  # 计算局部注意力输出

            # Compute gating weights using query-dependent projection  # 使用查询依赖投影计算门控权重
            assert gate_proj is not None, "gate_proj must be provided for loop_idx > 0"  # 断言门控投影不为空
            num_tokens = q.shape[0]  # 获取 token 数量
            q_reshaped = q.view(num_tokens, self.num_heads, self.head_dim).transpose(  # 重塑查询张量
                0, 1
            )
            gate = gate_proj(q_reshaped)  # 计算门控值

            # Mix global and local attention outputs with learned gate  # 用学习的门控混合全局和局部注意力输出
            # gate controls the balance between global context and local focus  # 门控控制全局上下文和局部聚焦之间的平衡
            attn_output = global_attn_output * gate + local_attn_output * (1 - gate)  # 加权混合

        output, _ = self.o_proj(attn_output)  # 通过输出投影层
        return output  # 返回输出


class LoopCoderDecoderLayer(nn.Module):  # LoopCoder 解码器层
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int = 0,  # 层 ID，默认为 0
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏层大小
        self.layer_id = layer_id  # 保存层 ID

        self.self_attn = LoopCoderAttention(  # 创建自注意力模块
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            max_position=getattr(config, "max_position_embeddings", 4096 * 32),  # 最大位置数
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = LoopCoderMLP(  # 创建 MLP 模块
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        loop_idx: int,  # 当前循环索引
        gate_proj: Optional[LoopGateProjection] = None,  # 门控投影，可选
    ) -> torch.Tensor:
        # Self Attention  # 自注意力
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        hidden_states = self.self_attn(  # 通过自注意力层
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            loop_idx=loop_idx,
            gate_proj=gate_proj,
        )
        hidden_states = hidden_states + residual  # 残差连接

        # MLP  # 多层感知机
        residual = hidden_states  # 保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后层归一化
        hidden_states = self.mlp(hidden_states)  # 通过 MLP
        hidden_states = hidden_states + residual  # 残差连接

        return hidden_states  # 返回隐藏状态


class IQuestLoopCoderModel(nn.Module):  # IQuest LoopCoder 模型
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.vocab_size = config.vocab_size  # 保存词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 创建词嵌入层
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 嵌入维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.loop_num = getattr(self.config, "loop_num", 2)  # 获取循环次数
        self.window_size = getattr(self.config, "loop_window_size", 64)  # 获取窗口大小

        # Gate projections for Loop 2+ (one per layer)  # 循环 2+ 的门控投影（每层一个）
        head_dim = config.hidden_size // config.num_attention_heads  # 计算头维度
        gate_projections = make_layers(  # 创建门控投影层
            config.num_hidden_layers,
            lambda idx, prefix: LoopGateProjection(  # 每层创建一个门控投影
                total_num_heads=config.num_attention_heads,
                head_dim=head_dim,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("gate_projections", prefix),
        )
        if isinstance(gate_projections, tuple):  # 如果返回了元组（包含流水线并行信息）
            self.start_layer, self.end_layer, self.gate_projections = gate_projections
        else:  # 否则
            self.start_layer, self.end_layer = 0, config.num_hidden_layers  # 使用全部层
            self.gate_projections = gate_projections

        layers = make_layers(  # 创建解码器层
            config.num_hidden_layers,
            lambda idx, prefix: LoopCoderDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        if isinstance(layers, tuple):  # 如果返回了元组（包含流水线并行信息）
            self.start_layer, self.end_layer, self.layers = layers
        else:  # 否则
            self.start_layer, self.end_layer = 0, config.num_hidden_layers  # 使用全部层
            self.layers = layers

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID 张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入张量，可选
    ) -> torch.Tensor:
        if input_embeds is not None:  # 如果提供了输入嵌入
            hidden_states = input_embeds  # 直接使用输入嵌入
        else:  # 否则
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入

        # Multi-loop forward pass  # 多循环前向传播
        for loop_idx in range(self.loop_num):  # 遍历每个循环
            for layer_idx in range(self.start_layer, self.end_layer):  # 遍历每层
                layer = self.layers[layer_idx]  # 获取当前层
                # Get gate_proj for this layer (only for loop_idx > 0)  # 获取该层的门控投影（仅用于 loop_idx > 0）
                gate_proj = self.gate_projections[layer_idx] if loop_idx > 0 else None  # 循环 0 不需要门控投影
                hidden_states = layer(  # 通过当前层
                    positions, hidden_states, forward_batch, loop_idx, gate_proj
                )

        hidden_states = self.norm(hidden_states)  # 应用最终归一化
        return hidden_states  # 返回隐藏状态


class IQuestLoopCoderForCausalLM(nn.Module):  # IQuest LoopCoder 因果语言模型
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = IQuestLoopCoderModel(  # 创建 LoopCoder 模型
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings:  # 如果绑定词嵌入和语言模型头
            self.lm_head = self.model.embed_tokens  # 共享嵌入权重
        else:  # 否则
            self.lm_head = ParallelLMHead(  # 创建独立的语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层大小
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID 张量
        positions: torch.Tensor,  # 位置张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入张量，可选
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 通过模型获取隐藏状态
        return self.logits_processor(  # 通过 logits 处理器计算输出
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射
            ("qkv_proj", "q_proj", "q"),  # QKV 投影堆叠
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),  # gate 和 up 投影堆叠
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())  # 创建参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入的逆频率
                continue

            # Handle gate_projections weights  # 处理门控投影权重
            if name.startswith("gate_projections."):  # 如果是门控投影权重
                if name.endswith(".weight"):  # 如果是权重
                    sglang_name = name.replace(".weight", ".gate_proj.weight")  # 映射到 SGLang 内部名称
                elif name.endswith(".bias"):  # 如果是偏置
                    sglang_name = name.replace(".bias", ".gate_proj.bias")  # 映射到 SGLang 内部名称
                else:  # 其他类型跳过
                    continue

                if sglang_name in params_dict:  # 如果映射后的名称在参数字典中
                    param = params_dict[sglang_name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                continue  # 继续处理下一个权重

            # Handle stacked parameters  # 处理堆叠参数
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                name = name.replace(weight_name, param_name)  # 替换权重名称
                if name.endswith(".bias") and name not in params_dict:  # 跳过不在参数字典中的偏置
                    continue
                if name in params_dict:  # 如果参数名在参数字典中
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 如果堆叠参数映射中没有匹配
                # Handle regular parameters  # 处理常规参数
                if name.endswith(".bias") and name not in params_dict:  # 跳过不在参数字典中的偏置
                    continue
                if name in params_dict:  # 如果参数名在参数字典中
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重


# Entry class for model registration  # 模型注册入口类
EntryClass = IQuestLoopCoderForCausalLM
