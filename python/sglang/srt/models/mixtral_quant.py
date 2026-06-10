# 量化Mixtral模型推理实现 - 仅推理模式的Mixtral稀疏专家混合模型，支持量化配置
# 本文件实现了Mixtral模型的量化版本，包含MLP专家层、MoE路由、注意力机制和权重加载逻辑

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
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral_quant.py#L1
"""Inference-only Mixtral model."""  # 仅推理的Mixtral模型

from typing import Iterable, Optional, Tuple  # 导入类型提示

import numpy as np  # 导入numpy用于数组操作
import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入神经网络函数模块
from torch import nn  # 导入神经网络模块
from transformers import MixtralConfig  # 导入Mixtral配置类

from sglang.srt.distributed import (  # 导入分布式相关函数
    get_tensor_model_parallel_rank,  # 获取当前张量并行秩
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 导入线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class MixtralMLP(nn.Module):
    """Mixtral模型的MLP专家层，包含三个线性变换和SiLU激活函数"""

    def __init__(
        self,
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.num_experts = num_experts  # 保存专家数量
        self.ffn_dim = intermediate_size  # 保存前馈网络维度
        self.hidden_dim = hidden_size  # 保存隐藏维度

        self.w1 = ReplicatedLinear(  # w1线性层，将隐藏维度映射到中间维度
            self.hidden_dim,
            self.ffn_dim,
            bias=False,  # 不使用偏置
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )
        self.w2 = ReplicatedLinear(  # w2线性层，将中间维度映射回隐藏维度
            self.ffn_dim,
            self.hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )
        self.w3 = ReplicatedLinear(  # w3线性层，门控投影，将隐藏维度映射到中间维度
            self.hidden_dim,
            self.ffn_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w3", prefix),
        )

        # TODO: Use vllm's SiluAndMul  # 待办：使用vllm的SiluAndMul融合算子
        self.act_fn = nn.SiLU()  # SiLU激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP前向传播：SwiGLU结构 w2(act(w1(x)) * w3(x))"""
        w1_out, _ = self.w1(hidden_states)  # w1线性变换
        w1_out = self.act_fn(w1_out)  # 对w1输出应用激活函数
        w3_out, _ = self.w3(hidden_states)  # w3线性变换（门控）
        current_hidden_states = w1_out * w3_out  # 逐元素相乘实现门控
        current_hidden_states, _ = self.w2(current_hidden_states)  # w2线性变换得到输出
        return current_hidden_states  # 返回当前专家的输出


class MixtralMoE(nn.Module):
    """Mixtral稀疏专家混合层，实现Top-K路由和专家并行"""

    def __init__(
        self,
        config: MixtralConfig,  # Mixtral配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.rank = get_tensor_model_parallel_rank()  # 获取当前张量并行秩
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        self.num_total_experts = config.num_local_experts  # 专家总数
        self.top_k = config.num_experts_per_tok  # 每个token选择的Top-K专家数
        if self.tp_size > self.num_total_experts:  # 张量并行大小不能超过专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}."
            )
        # Split experts equally between ranks  # 在各并行秩之间均分专家
        self.expert_indices = np.array_split(  # 按并行秩划分专家索引
            range(self.num_total_experts), self.tp_size
        )[self.rank].tolist()  # 获取当前秩负责的专家索引
        if not self.expert_indices:  # 如果没有分配到专家则报错
            raise ValueError(f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList(  # 创建专家模块列表
            [
                (
                    MixtralMLP(  # 当前秩负责的专家，创建MLP实例
                        self.num_total_experts,
                        config.hidden_size,
                        config.intermediate_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"experts.{idx}", prefix),
                    )
                    if idx in self.expert_indices  # 仅创建当前秩负责的专家
                    else None  # 其他专家设为None，不占内存
                )
                for idx in range(self.num_total_experts)  # 遍历所有专家索引
            ]
        )
        self.gate = ReplicatedLinear(  # 路由门控线性层
            config.hidden_size,
            self.num_total_experts,
            bias=False,
            quant_config=None,  # 门控不使用量化
            prefix=add_prefix("gate", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE前向传播：计算路由权重，选择Top-K专家，加权求和"""
        router_logits, _ = self.gate(hidden_states)  # 计算路由logits

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # Softmax得到路由概率
        routing_weights, selected_experts = torch.topk(  # 选择Top-K专家
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)  # 归一化路由权重

        final_hidden_states = None  # 初始化最终隐藏状态
        for expert_idx in self.expert_indices:  # 遍历当前秩负责的专家
            expert_layer = self.experts[expert_idx]  # 获取专家层
            expert_mask = selected_experts == expert_idx  # 创建专家掩码
            expert_weights = (routing_weights * expert_mask).sum(dim=-1, keepdim=True)  # 计算专家权重

            current_hidden_states = expert_layer(hidden_states).mul_(expert_weights)  # 加权专家输出
            if final_hidden_states is None:  # 第一个专家直接赋值
                final_hidden_states = current_hidden_states
            else:  # 后续专家累加
                final_hidden_states.add_(current_hidden_states)

        return tensor_model_parallel_all_reduce(final_hidden_states)  # 跨张量并行秩全归约


class MixtralAttention(nn.Module):
    """Mixtral注意力层，支持GQA和旋转位置编码"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID
        max_position: int = 4096 * 32,  # 最大位置编码长度
        rope_theta: float = 10000,  # RoPE基频
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏大小
        tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % tp_size == 0  # 头数必须能被并行大小整除
        self.num_heads = self.total_num_heads // tp_size  # 当前秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= tp_size:  # KV头数大于等于TP大小时
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，按TP划分KV头
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:  # KV头数小于TP大小时
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，复制KV头
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 当前秩的KV头数
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q的总大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.rope_theta = rope_theta  # 保存RoPE基频

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,  # 使用Neox风格的RoPE
        )
        self.attn = RadixAttention(  # 基数注意力实现
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        """注意力前向传播：QKV投影 -> RoPE -> 注意力计算 -> 输出投影"""
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割QKV
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回注意力输出


class MixtralDecoderLayer(nn.Module):
    """Mixtral解码器层，包含自注意力和稀疏MoE前馈网络"""

    def __init__(
        self,
        config: MixtralConfig,  # Mixtral配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 保存隐藏大小
        # Requires transformers > 4.32.0  # 需要transformers版本大于4.32.0
        rope_theta = config.rope_parameters["rope_theta"]  # 从配置中获取RoPE基频
        self.self_attn = MixtralAttention(  # 自注意力层
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.block_sparse_moe = MixtralMoE(  # 稀疏MoE层
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("block_sparse_moe", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差连接
    ) -> torch.Tensor:
        """解码器层前向传播：自注意力 -> 残差 -> MoE -> 残差"""
        # Self Attention  # 自注意力
        if residual is None:  # 第一层没有残差
            residual = hidden_states  # 保存输入作为残差
            hidden_states = self.input_layernorm(hidden_states)  # 对隐藏状态做归一化
        else:  # 后续层有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 归一化并更新残差
        hidden_states = self.self_attn(  # 自注意力计算
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected  # 全连接（MoE）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后归一化
        hidden_states = self.block_sparse_moe(hidden_states)  # 稀疏MoE前向传播
        return hidden_states, residual  # 返回隐藏状态和残差


class MixtralModel(nn.Module):
    """Mixtral模型主体，包含词嵌入、解码器层堆叠和最终归一化"""

    def __init__(
        self,
        config: MixtralConfig,  # Mixtral配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.padding_idx = config.pad_token_id  # 填充token索引
        self.vocab_size = config.vocab_size  # 词表大小

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                MixtralDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)  # 按配置创建所有解码器层
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        """模型前向传播：嵌入 -> 解码器层 -> 归一化"""
        if input_embeds is None:  # 没有提供输入嵌入时
            hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取嵌入
        else:  # 提供了输入嵌入时
            hidden_states = input_embeds  # 直接使用输入嵌入
        residual = None  # 初始化残差为None
        for i in range(len(self.layers)):  # 遍历所有解码器层
            layer = self.layers[i]  # 获取当前层
            hidden_states, residual = layer(  # 前向传播
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)  # 最终归一化，融合残差
        return hidden_states  # 返回最终隐藏状态


class QuantMixtralForCausalLM(nn.Module):
    """量化Mixtral因果语言模型，包含模型主体、语言模型头和logits处理"""

    def __init__(
        self,
        config: MixtralConfig,  # Mixtral配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        self.model = MixtralModel(  # Mixtral模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        """因果语言模型前向传播：模型主体 -> logits处理"""
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 模型前向传播
        return self.logits_processor(  # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，支持QKV堆叠和专家并行"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())  # 参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入的逆频率
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 检查堆叠参数
                if weight_name not in name:  # 不匹配则继续
                    continue
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:  # 参数不存在则跳过
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 按分片加载权重
                break
            else:  # 非堆叠参数
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.  # 跳过未分配给当前工作进程的专家
                if "block_sparse_moe.experts." in name and name not in params_dict:
                    continue
                if name not in params_dict:  # 参数不存在则跳过
                    continue
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = QuantMixtralForCausalLM  # 入口类，用于模型注册
