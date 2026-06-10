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
# https://github.com/vllm-project/vllm/blob/14f91fe67c2342f2fe859dc6a5c40810df0e1c61/vllm/model_executor/models/deepseek.py
"""Inference-only Deepseek model."""

# 本文件实现了 DeepSeek 大语言模型的推理-only 架构，包括：
# - DeepseekMLP: 前馈神经网络（FFN）模块
# - DeepseekMoE: 混合专家（Mixture of Experts）模块
# - DeepseekAttention: 多头注意力模块（支持 GQA）
# - DeepseekDecoderLayer: 解码器层
# - DeepseekModel: DeepSeek 模型主体
# - DeepseekForCausalLM: 用于因果语言建模的 DeepSeek 模型

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu, is_npu
from sglang.srt.utils.hf_transformers_utils import get_rope_config

# 检测当前硬件平台及特性
_is_cpu_amx_available = cpu_has_amx_support()  # CPU 是否支持 AMX 指令集
_is_cpu = is_cpu()  # 是否运行在 CPU 上
_is_npu = is_npu()  # 是否运行在 NPU 上

if _is_cpu and _is_cpu_amx_available:
    import sgl_kernel  # noqa: F401  # CPU AMX 模式下需要 sgl_kernel 加速库

if _is_npu:
    # NPU 平台使用专用的融合 MoE 实现
    from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
        fused_moe_npu as fused_moe,
    )
else:
    # 默认使用 Triton 实现的融合 MoE
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe


class DeepseekMLP(nn.Module):
    """DeepSeek 前馈神经网络（MLP）模块，使用门控 SiLU 激活函数。
    结构为 gate_up_proj -> SiLU_AND_Mul -> down_proj。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # 门控投影和上投影合并为一个线性层，输出维度为 intermediate_size * 2
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        # 下投影层，将中间维度映射回隐藏维度
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU 门控激活函数：对 gate 部分做 silu，再与 up 部分逐元素相乘

    def forward(self, x):
        """前向传播：gate_up_proj -> SiLUAndMul -> down_proj"""
        gate_up, _ = self.gate_up_proj(x)  # 计算门控和上投影的合并输出
        x = self.act_fn(gate_up)  # 应用 SiLU 门控激活：silu(gate) * up
        x, _ = self.down_proj(x)  # 下投影回隐藏维度
        return x


class DeepseekMoE(nn.Module):
    """DeepSeek 混合专家（Mixture of Experts）模块。
    包含路由专家（routed experts）和可选的共享专家（shared experts），
    通过门控网络（router）选择 top-k 个专家进行计算。"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()  # 当前张量并行秩
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行世界大小
        self.n_routed_experts = config.n_routed_experts  # 路由专家数量
        self.top_k = config.num_experts_per_tok  # 每个 token 选择的专家数
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )
        # TopK 选择器，用于从路由 logits 中选出 top-k 专家
        self.topk = TopK(
            top_k=self.top_k,
            renormalize=config.norm_topk_prob,
        )
        # 创建所有路由专家的 MLP 模块
        self.experts = nn.ModuleList(
            [
                DeepseekMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,  # MoE 中各专家不做 all-reduce，最后统一做
                    prefix=add_prefix(f"{idx}.experts", prefix),
                )
                for idx in range(self.n_routed_experts)
            ]
        )
        self.pack_params()  # 将专家参数打包为连续张量以提升融合 MoE 计算效率

        # 路由门控网络，输出每个专家的 logits
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            quant_config=None,  # 门控网络不进行量化
            prefix=add_prefix("gate", prefix),
        )

        # 共享专家：每个 token 都会经过，不参与路由选择
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )

    def pack_params(self):
        """将所有专家的 gate_up_proj 和 down_proj 权重打包为连续张量，
        以便在融合 MoE kernel 中高效访问。"""
        w1 = []
        w2 = []
        for expert in self.experts:
            w1.append(expert.gate_up_proj.weight)  # 收集各专家的 gate_up 权重
            w2.append(expert.down_proj.weight)  # 收集各专家的 down 权重
        # 将 w1 列表中的张量展平合并为一个连续存储的张量
        self.w1 = torch._utils._flatten_dense_tensors(w1)
        # 再拆分回来，确保原始参数共享同一块底层存储
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)
        for data, param in zip(w1s, w1):
            param.data = data
        # 重塑为 (num_experts, out_features, in_features) 形状
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)

        # 同样处理 down_proj 权重
        self.w2 = torch._utils._flatten_dense_tensors(w2)
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)
        for data, param in zip(w2s, w2):
            param.data = data

        self.w2 = self.w2.view(len(w2), *w2s[0].shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE 前向传播：路由选择 + 专家计算 + 共享专家 + all-reduce"""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # 计算共享专家输出（如果存在）
        if self.config.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)  # 计算路由 logits
        topk_output = self.topk(hidden_states, router_logits)  # 选出 top-k 专家
        # 根据硬件平台选择不同的融合 MoE 实现
        if _is_cpu and _is_cpu_amx_available:
            # CPU AMX 加速路径
            topk_weights, topk_ids, _ = topk_output
            final_hidden_states = torch.ops.sgl_kernel.fused_experts_cpu(
                hidden_states,
                self.w1,
                self.w2,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                0,  # CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                None,  # w1_bias
                None,  # w2_bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni
            )
        else:
            # GPU Triton 融合 MoE 路径
            final_hidden_states = fused_moe(
                hidden_states,
                w1=self.w1,
                w2=self.w2,
                topk_output=topk_output,
                moe_runner_config=MoeRunnerConfig(inplace=True),
            )
        # 将路由专家输出与共享专家输出相加
        if self.config.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output
        # 张量并行 all-reduce，汇总各 GPU 上的部分结果
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class DeepseekAttention(nn.Module):
    """DeepSeek 多头注意力模块，支持分组查询注意力（GQA）和旋转位置编码（RoPE）。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 每个 TP 秩分到的 Q 头数
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            # KV 头数 >= TP 大小：将 KV 头在多个 GPU 间切分
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            # KV 头数 < TP 大小：在多个 GPU 间复制 KV 头
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个 TP 秩分到的 KV 头数
        self.head_dim = hidden_size // self.total_num_heads  # 每个头的维度
        self.q_size = self.num_heads * self.head_dim  # Q 的总维度
        self.kv_size = self.num_kv_heads * self.head_dim  # K/V 的总维度
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # QKV 合并投影层
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        # 输出投影层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # 旋转位置编码（RoPE）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # 注意力计算模块（使用 RadixAttention 实现）
        self.attn = RadixAttention(
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
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """注意力前向传播：QKV 投影 -> RoPE -> 注意力计算 -> 输出投影"""
        qkv, _ = self.qkv_proj(hidden_states)  # 计算 Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分 Q、K、V
        q, k = self.rotary_emb(positions, q, k)  # 对 Q、K 应用旋转位置编码
        attn_output = self.attn(q, k, v, forward_batch)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output


class DeepseekDecoderLayer(nn.Module):
    """DeepSeek 解码器层，包含自注意力、MLP/MoE 和两层 RMSNorm。"""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta, rope_scaling = get_rope_config(config)  # 获取 RoPE 配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = DeepseekAttention(
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
        # 根据层 ID 决定使用 MoE 还是普通 MLP
        # 条件：配置了路由专家 且 当前层 >= first_k_dense_replace 且 当前层是 MoE 频率层
        if (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            # 前几层或非 MoE 频率层使用普通 MLP
            self.mlp = DeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力前的 LayerNorm
        self.post_attention_layernorm = RMSNorm(  # MLP 前的 LayerNorm
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """解码器层前向传播：Pre-Norm 架构，残差连接在 LayerNorm 之外。"""
        # Self Attention
        if residual is None:
            residual = hidden_states  # 第一层没有残差，直接保存
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # Pre-Norm + 残差
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # Post-Attn Norm + 残差
        hidden_states = self.mlp(hidden_states)  # MLP 或 MoE 计算
        return hidden_states, residual


class DeepseekModel(nn.Module):
    """DeepSeek 模型主体，由词嵌入层、多层解码器和最终 RMSNorm 组成。"""

    fall_back_to_pt_during_load = False  # 加载权重时不回退到 PyTorch 默认加载方式

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 词嵌入层，将 token ID 映射为隐藏状态向量
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        # 多层解码器
        self.layers = nn.ModuleList(
            [
                DeepseekDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终的 RMSNorm

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """模型主体前向传播：词嵌入 -> 解码器层 -> 最终 RMSNorm"""
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)  # 从 token ID 获取词嵌入
        else:
            hidden_states = input_embeds  # 直接使用输入嵌入

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        # 最后一层需要对残差和隐藏状态做 RMSNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekForCausalLM(nn.Module):
    """DeepSeek 因果语言模型，在 DeepseekModel 基础上添加语言模型头（lm_head）用于生成 logits。"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 语言模型头，将隐藏状态映射为词表 logits
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)  # logits 后处理器

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入词嵌入层"""
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """因果语言模型前向传播：模型主体 -> logits 处理"""
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理堆叠参数（QKV 合并、gate_up 合并）和专家并行。"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # QKV 合并映射：将独立的 q/k/v 投影合并为一个 qkv_proj
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # gate_up 合并映射：将独立的 gate/up 投影合并为一个 gate_up_proj
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue  # 跳过 RoPE 的 inv_freq，由模型自行计算
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                # 跳过不属于当前 worker 的专家参数（专家并行时）
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)  # 按分片 ID 加载堆叠参数
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                # 跳过不属于当前 worker 的专家参数
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)  # 普通参数加载


EntryClass = DeepseekForCausalLM  # 模型入口类，供框架自动发现和注册
