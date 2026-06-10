# Laguna模型推理实现文件
# 本文件实现了poolside/Laguna-XS.2模型的推理专用版本
# 包含MLP、MoE门控、MoE混合专家、注意力机制、解码器层、模型主体及因果语言模型等组件
# 支持张量并行、流水线并行、数据并行注意力及多种量化配置

# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Inference-only Laguna (poolside/Laguna-XS.2) model."""

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 日志模块
from collections.abc import Iterable  # 可迭代类型
from typing import Any, Dict, Optional, Tuple, Union  # 类型提示

import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数式接口
from torch import nn  # 神经网络模块

from sglang.srt.configs.laguna import LagunaConfig  # Laguna模型配置类
from sglang.srt.distributed import (  # 分布式通信相关
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活与乘法融合层
from sglang.srt.layers.communicator import (  # 层通信器
    LayerCommunicator,  # 层通信器类
    LayerScatterModes,  # 层散射模式
)
from sglang.srt.layers.dp_attention import (  # 数据并行注意力相关
    get_attention_tp_rank,  # 获取注意力张量并行秩
    get_attention_tp_size,  # 获取注意力张量并行大小
    is_dp_attention_enabled,  # 判断是否启用数据并行注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 均方根归一化层
from sglang.srt.layers.linear import (  # 并行线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce  # 判断是否跳过专家后全归约
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 获取MoE实现类
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 融合MoE Triton实现
from sglang.srt.layers.moe.topk import TopK  # Top-K选择器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力层
from sglang.srt.layers.rotary_embedding import get_rope  # 获取旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 流水线缺失层及层ID获取
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.models.utils import apply_qk_norm  # 应用QK归一化
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.utils import LazyValue, add_prefix, make_layers  # 惰性值、前缀添加、层创建

logger = logging.getLogger(__name__)  # 获取当前模块日志器


class LagunaMLP(nn.Module):
    """Laguna模型的密集MLP层，包含门控投影、上投影和下投影"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        if hidden_act != "silu":  # 仅支持silu激活
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并的并行线性层
            hidden_size,
            [intermediate_size] * 2,  # 两个中间层大小
            bias=False,  # 无偏置
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(  # 下投影行并行线性层
            intermediate_size,
            hidden_size,
            bias=False,  # 无偏置
            quant_config=quant_config,
            reduce_results=reduce_results,  # 是否归约结果
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()  # SiLU激活与乘法融合函数

    def forward(
        self,
        x: torch.Tensor,  # 输入张量
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """MLP前向传播：门控上投影 -> 激活 -> 下投影"""
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影
        x = self.act_fn(gate_up)  # 应用SiLU激活和门控乘法
        # Skip the in-block reduce when LayerCommunicator will fuse it or when
        # the next layer expects reduce-scatter — otherwise we'd double-reduce.
        x, _ = self.down_proj(  # 下投影
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,  # 跳过全归约以避免重复归约
        )
        return x  # 返回MLP输出


class LagunaMoEGate(nn.Module):
    """Laguna MoE路由门控层，计算每个token到各专家的logits"""

    def __init__(
        self,
        config: LagunaConfig,  # 模型配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.weight = nn.Parameter(  # 门控权重参数
            torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        )
        # Released checkpoint stores this under `mlp.experts.e_score_correction_bias`
        # (load_weights remaps it) but every value is 0.0; zero-init keeps us
        # correct if a future checkpoint omits the tensor entirely.
        self.e_score_correction_bias = nn.Parameter(  # 专家分数修正偏置，零初始化
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,  # 不需要梯度
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """门控前向传播：计算路由logits"""
        return F.linear(hidden_states.to(torch.float32), self.weight, None)  # 线性变换得到路由logits


class LagunaMoE(nn.Module):
    """Laguna混合专家层，包含共享专家、路由专家和Top-K选择"""

    def __init__(
        self,
        config: LagunaConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.routed_scaling_factor = config.moe_routed_scaling_factor  # 路由缩放因子
        self.router_logit_softcapping = getattr(  # 路由logit软上限
            config, "moe_router_logit_softcapping", 0.0
        )

        if self.tp_size > config.num_experts:  # 张量并行大小不能超过专家数
            raise ValueError(
                f"TP size {self.tp_size} > num_experts {config.num_experts}."
            )

        self.gate = LagunaMoEGate(config, prefix=add_prefix("gate", prefix))  # 路由门控

        self.experts = get_moe_impl_class(quant_config)(  # 专家网络实现
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,  # 加上冗余专家数
            top_k=config.num_experts_per_tok,  # 每个token选择的专家数
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
            quant_config=quant_config,
            reduce_results=False,  # 不在专家内部归约
            apply_router_weight_on_input=bool(config.moe_apply_router_weight_on_input),  # 是否在输入上应用路由权重
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(  # Top-K选择器
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=True,  # 重新归一化权重
            use_grouped_topk=False,  # 不使用分组Top-K
            scoring_func="sigmoid",  # 使用sigmoid评分函数
            correction_bias=self.gate.e_score_correction_bias,  # 修正偏置
        )

        # HF safetensors key is singular `shared_expert.…`; mirror so the
        # default loader picks it up without remapping.
        self.shared_expert = LagunaMLP(  # 共享专家
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,  # 共享专家中间层大小
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,  # 不归约，后面手动归约
            prefix=add_prefix("shared_expert", prefix),
        )

    def get_moe_weights(self):
        """获取MoE专家的权重数据列表"""
        return [x.data for x in self.experts.parameters()]  # 返回专家参数数据

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """MoE前向传播：共享专家 + 路由专家，支持软上限和路由缩放"""
        if hidden_states.shape[0] == 0:  # 空输入直接返回
            return hidden_states

        shared_out = self.shared_expert(hidden_states)  # 共享专家输出

        router_logits = self.gate(hidden_states)  # 计算路由logits
        if self.router_logit_softcapping > 0.0:  # 应用logit软上限
            cap = self.router_logit_softcapping  # 上限值
            router_logits = torch.tanh(router_logits / cap) * cap  # tanh软上限变换
        topk_output = self.topk(hidden_states, router_logits)  # Top-K选择
        routed_out = self.experts(hidden_states, topk_output)  # 路由专家输出

        # Non-grouped TopK doesn't honor apply_routed_scaling_factor_on_output,
        # so scale routed manually before adding the unscaled shared expert.
        if self.routed_scaling_factor != 1.0:  # 手动应用路由缩放因子
            routed_out = routed_out * self.routed_scaling_factor
        final = routed_out + shared_out  # 合并路由输出和共享输出

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 张量并行时执行全归约
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final = tensor_model_parallel_all_reduce(final)  # 执行张量并行全归约
        return final  # 返回最终输出


class LagunaAttention(nn.Module):
    """Laguna注意力层，包含QKV投影、QK归一化、旋转位置编码、softplus门控和输出投影"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        head_dim: int,  # 头维度
        layer_id: int,  # 层ID
        rms_norm_eps: float,  # RMS归一化epsilon
        rope_theta: float,  # RoPE基准频率
        rope_scaling: Optional[Dict[str, Any]],  # RoPE缩放配置
        partial_rotary_factor: float,  # 部分旋转因子
        max_position_embeddings: int,  # 最大位置编码数
        attention_bias: bool,  # 是否使用注意力偏置
        sliding_window_size: int,  # 滑动窗口大小
        layer_type: str,  # 层类型（滑动注意力或全注意力）
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.head_dim = head_dim  # 保存头维度
        self.layer_id = layer_id  # 保存层ID

        attn_tp_rank = get_attention_tp_rank()  # 获取注意力张量并行秩
        attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小

        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % attn_tp_size == 0  # 头数必须能被TP大小整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 每个TP秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= attn_tp_size:  # KV头数大于等于TP大小时
            assert self.total_num_kv_heads % attn_tp_size == 0  # KV头数必须能被TP大小整除
        else:  # KV头数小于TP大小时
            assert attn_tp_size % self.total_num_kv_heads == 0  # TP大小必须能被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 每个TP秩的KV头数
        self.q_size = self.num_heads * self.head_dim  # Q的总维度
        self.kv_size = self.num_kv_heads * self.head_dim  # KV的总维度
        self.scaling = self.head_dim**-0.5  # 缩放因子

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 输出投影行并行线性层
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,  # 不自动归约
            prefix=add_prefix("o_proj", prefix),
        )

        # Per-head softplus gate (`gating=True` in HF). Shard like Q so the
        # local output dim matches `num_heads`.
        self.g_proj = ColumnParallelLinear(  # 逐头softplus门控投影
            hidden_size,
            self.total_num_heads,  # 输出维度等于头数
            bias=False,
            gather_output=False,  # 不收集输出
            quant_config=None,  # 不量化门控
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("g_proj", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)  # Q归一化
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)  # K归一化

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
        )

        assert layer_type in {"sliding_attention", "full_attention"}  # 层类型必须是两种之一
        use_sliding = layer_type == "sliding_attention"  # 是否使用滑动窗口
        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=sliding_window_size if use_sliding else -1,  # 滑动窗口大小
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """注意力前向传播：QKV投影 -> QK归一化 -> RoPE -> 注意力计算 -> 门控 -> 输出投影"""
        if hidden_states.shape[0] == 0:  # 空输入直接返回
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分离Q、K、V

        q, k = apply_qk_norm(  # 应用QK归一化
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
        )
        q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码

        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力

        gate, _ = self.g_proj(hidden_states)  # 计算门控值
        gate = F.softplus(gate.float()).to(attn_output.dtype)  # 应用softplus激活
        attn_output = attn_output.view(-1, self.num_heads, self.head_dim)  # 重塑为多头形状
        attn_output = attn_output * gate.view(-1, self.num_heads, 1)  # 逐头门控相乘
        attn_output = attn_output.reshape(-1, self.num_heads * self.head_dim)  # 展平多头维度

        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回注意力输出


class LagunaDecoderLayer(nn.Module):
    """Laguna解码器层，包含自注意力和MLP/MoE，支持层通信和散射模式"""

    def __init__(
        self,
        config: LagunaConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.layer_id = layer_id  # 保存层ID
        self.hidden_size = config.hidden_size  # 保存隐藏层大小

        layer_types = config.layer_types  # 各层类型列表
        layer_type = layer_types[layer_id]  # 当前层类型
        is_swa = layer_type == "sliding_attention"  # 是否为滑动窗口注意力

        layer_num_heads = config.num_attention_heads_per_layer[layer_id]  # 当前层注意力头数

        if is_swa:  # 滑动窗口注意力层的RoPE参数
            rope_theta = config.swa_rope_theta
            rope_scaling = config.swa_rope_scaling
            partial_rotary_factor = config.swa_partial_rotary_factor
        else:  # 全注意力层的RoPE参数
            rope_theta = config.rope_theta
            rope_scaling = config.full_rope_scaling
            partial_rotary_factor = config.partial_rotary_factor

        self.self_attn = LagunaAttention(  # 自注意力层
            hidden_size=self.hidden_size,
            num_heads=layer_num_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_id=layer_id,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            attention_bias=config.attention_bias,
            # SGLang's window is exclusive; HF's `sliding_window` is inclusive.
            sliding_window_size=config.sliding_window - 1,  # 滑动窗口大小（SGLang使用不包含右端）
            layer_type=layer_type,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        mlp_types = config.mlp_layer_types  # MLP层类型列表
        self.is_layer_sparse = mlp_types[layer_id] == "sparse"  # 当前层是否为稀疏MoE
        is_previous_layer_sparse = layer_id > 0 and mlp_types[layer_id - 1] == "sparse"  # 前一层是否稀疏
        is_next_layer_sparse = (  # 后一层是否稀疏
            layer_id + 1 < config.num_hidden_layers
            and mlp_types[layer_id + 1] == "sparse"
        )

        if self.is_layer_sparse:  # 稀疏层使用MoE
            self.mlp = LagunaMoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:  # 密集层使用标准MLP
            self.mlp = LagunaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=True,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 初始化层散射模式
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,  # 允许reduce-scatter
            is_last_layer=(layer_id == config.num_hidden_layers - 1),  # 是否为最后一层
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差连接
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播：注意力准备 -> 自注意力 -> MLP准备 -> MLP/MoE -> 后处理"""
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力输入
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:  # 非空输入时计算注意力
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP输入
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (  # 是否融合MLP全归约与下一层
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(  # 是否使用reduce-scatter
            forward_batch
        )

        hidden_states = self.mlp(  # MLP/MoE计算
            hidden_states,
            forward_batch=forward_batch,
            should_allreduce_fusion=should_allreduce_fusion,
            use_reduce_scatter=use_reduce_scatter,
        )

        if should_allreduce_fusion:  # 标记需要融合全归约
            hidden_states._sglang_needs_allreduce_fusion = True
        else:  # 正常后处理
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual  # 返回隐藏状态和残差


class LagunaModel(nn.Module):
    """Laguna模型主体，包含嵌入层、多层解码器和最终归一化"""

    def __init__(
        self,
        config: LagunaConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        decoder_layer_type: type = LagunaDecoderLayer,  # 解码器层类型
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = getattr(config, "pad_token_id", None)  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if self.pp_group.is_first_rank:  # 第一个秩初始化嵌入层
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:  # 非第一个秩使用缺失层
            self.embed_tokens = PPMissingLayer()

        decoder_layer_type = decoder_layer_type or LagunaDecoderLayer  # 使用默认解码器层类型
        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行大小
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:  # 最后一个秩初始化最终归一化
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:  # 非最后一个秩使用缺失层
            self.norm = PPMissingLayer(return_tuple=True)

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.embed_tokens  # 返回嵌入层

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:
        """模型主体前向传播：嵌入 -> 解码器层 -> 归一化"""
        if self.pp_group.is_first_rank:  # 第一个秩处理嵌入
            if input_embeds is None:  # 无预计算嵌入时从token ID获取
                hidden_states = self.embed_tokens(input_ids)
            else:  # 使用预计算嵌入
                hidden_states = input_embeds
            residual = None  # 初始无残差
        else:  # 非第一个秩从代理张量获取
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]  # 获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 获取残差

        for i in range(self.start_layer, self.end_layer):  # 遍历所有层
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:  # 非最后一个秩返回代理张量
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:  # 非空时应用最终归一化
            if residual is None:  # 无残差时直接归一化
                hidden_states = self.norm(hidden_states)
            else:  # 有残差时融合归一化
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states  # 返回隐藏状态


class LagunaForCausalLM(nn.Module):
    """Laguna因果语言模型，包含模型主体和语言模型头"""

    fall_back_to_pt_during_load = False  # 加载权重时不回退到PyTorch
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: LagunaConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.model = LagunaModel(  # 模型主体
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        if self.pp_group.is_last_rank:  # 最后一个秩初始化语言模型头
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否启用DP LM头
            )
        else:  # 非最后一个秩使用缺失层
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)  # logits处理器

        # Only walk this rank's local layers — out-of-range entries can be PPMissingLayer.
        self._routed_experts_weights_of_layer = LazyValue(  # 惰性加载路由专家权重
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, LagunaMoE)  # 仅MoE层
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        """获取各层路由专家的权重"""
        return self._routed_experts_weights_of_layer.value  # 返回惰性值

    @property
    def start_layer(self):
        """获取起始层索引"""
        return self.model.start_layer

    @property
    def end_layer(self):
        """获取结束层索引"""
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> torch.Tensor:
        """因果语言模型前向传播：模型主体 -> logits处理"""
        hidden_states = self.model(  # 模型主体前向传播
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:  # 最后一个秩计算logits
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states  # 非最后一个秩返回隐藏状态

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.model.embed_tokens  # 返回嵌入层

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，支持堆叠参数、专家参数和权重名重映射"""
        stacked_params_mapping = [  # 堆叠参数映射
            ("qkv_proj", "q_proj", "q"),  # QKV合并
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),  # 门控上投影合并
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())  # 参数字典

        # (layer, expert, shard) tuples that hit the per-expert loader,
        # cross-checked against `expected` below to fail on dropped weights.
        loaded_expert_shards: set[Tuple[int, int, str]] = set()  # 已加载的专家分片集合
        moe_layer_ids = [  # MoE层ID列表
            i
            for i, mt in enumerate(self.config.mlp_layer_types)
            if mt == "sparse" and self.start_layer <= i < self.end_layer
        ]

        for name, loaded_weight in weights:  # 遍历所有权重
            layer_id = get_layer_id(name)  # 获取层ID
            if layer_id is not None and (  # 跳过非当前流水线阶段的层
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码频率
                continue

            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 跳过共享权重
                continue

            # HF stores the router correction bias under the experts namespace;
            # our parameter lives on the gate. Remap before dispatch.
            if name.endswith("mlp.experts.e_score_correction_bias"):  # 重映射修正偏置名称
                name = name.replace(
                    "mlp.experts.e_score_correction_bias",
                    "mlp.gate.e_score_correction_bias",
                )

            # Stacked dense (QKV / gate_up). The `mlp.experts.` guard stops
            # `up_proj` substring from false-matching `experts.{i}.up_proj.weight`.
            matched_stacked = False  # 是否匹配到堆叠参数
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:  # 名称不包含权重名则跳过
                    continue
                if "mlp.experts." in name:  # 跳过专家参数（单独处理）
                    continue
                name_mapped = name.replace(weight_name, param_name)  # 替换为参数名
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:  # 跳过不存在的偏置
                    continue
                if name_mapped not in params_dict:  # 跳过不存在的参数
                    continue
                param = params_dict[name_mapped]  # 获取参数
                param.weight_loader(param, loaded_weight, shard_id)  # 加载权重分片
                matched_stacked = True  # 标记为已匹配
                break
            if matched_stacked:  # 已处理堆叠参数则继续
                continue

            matched_expert = False  # 是否匹配到专家参数
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:  # 名称不包含权重名则跳过
                    continue
                name_mapped = name.replace(weight_name, param_name)  # 替换为参数名
                if name_mapped not in params_dict:  # 跳过不存在的参数
                    continue
                param = params_dict[name_mapped]  # 获取参数
                param.weight_loader(  # 加载专家权重
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                if layer_id is not None:  # 记录已加载的专家分片
                    loaded_expert_shards.add((layer_id, expert_id, shard_id))
                matched_expert = True  # 标记为已匹配
                break
            if matched_expert:  # 已处理专家参数则继续
                continue

            if name.endswith(".bias") and name not in params_dict:  # 跳过不存在的偏置
                continue
            if name not in params_dict:  # 参数不存在时发出警告
                logger.warning("Parameter %s not found in params_dict", name)
                continue
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            weight_loader(param, loaded_weight)  # 加载权重

        # If any routed-expert tensor was silently dropped (e.g. a future
        # checkpoint renaming `gate_proj`, or a ckpt-vs-mapping shape mismatch),
        # fail loud here instead of generating garbage.
        expected = {  # 期望加载的所有专家分片
            (layer_id, expert_id, shard_id)
            for layer_id in moe_layer_ids
            for expert_id in range(self.config.num_experts)
            for shard_id in ("w1", "w2", "w3")
        }
        missing = expected - loaded_expert_shards  # 未加载的专家分片
        if missing:  # 有未加载的专家分片时抛出异常
            sample = sorted(missing)[:5]
            raise RuntimeError(
                f"{len(missing)} routed-expert tensors were not loaded "
                f"(sample: {sample}). Expected {len(expected)} (layers={moe_layer_ids}, "
                f"num_experts={self.config.num_experts}, shards=3)."
            )

    def get_embed_and_head(self):
        """获取嵌入权重和语言模型头权重"""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """设置嵌入权重和语言模型头权重"""
        del self.model.embed_tokens.weight  # 删除旧嵌入权重
        del self.lm_head.weight  # 删除旧LM头权重
        self.model.embed_tokens.weight = embed  # 设置新嵌入权重
        self.lm_head.weight = head  # 设置新LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA


EntryClass = LagunaForCausalLM  # 入口类
