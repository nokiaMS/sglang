# LLaDA2-MoE模型推理实现文件
# 本文件实现了LLaDA2混合专家模型的推理专用版本
# 包含MLP、MoE门控、稀疏MoE块、注意力、解码器块、模型主体及因果语言模型等组件
# 支持共享专家、DeepEP分发、数据并行注意力、QK归一化等特性

# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""SGLang LLaDA2MoeModelLM model."""

import logging  # 日志模块
from typing import Iterable, Optional, Tuple, Union  # 类型提示

import torch  # PyTorch核心库
import torch.nn.functional as F  # PyTorch函数式接口
from torch import nn  # 神经网络模块
from transformers import PretrainedConfig  # 预训练配置基类

from sglang.srt.distributed import (  # 分布式通信相关
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    parallel_state,  # 并行状态
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 专家位置模型配置
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo  # 专家位置分发信息
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活与乘法融合层
from sglang.srt.layers.communicator import (  # 层通信器
    LayerCommunicator,  # 层通信器类
    LayerScatterModes,  # 层散射模式
    enable_moe_dense_fully_dp,  # 启用MoE密集全DP
)
from sglang.srt.layers.dp_attention import (  # 数据并行注意力相关
    get_attention_dp_size,  # 获取注意力DP大小
    get_attention_tp_rank,  # 获取注意力TP秩
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 均方根归一化层
from sglang.srt.layers.linear import (  # 并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.moe import (  # MoE相关
    get_deepep_mode,  # 获取DeepEP模式
    get_moe_a2a_backend,  # 获取MoE全到全后端
    should_skip_post_experts_all_reduce,  # 是否跳过专家后全归约
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 获取MoE实现类
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 融合MoE Triton实现
from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher  # DeepEP分发器
from sglang.srt.layers.moe.topk import TopK  # Top-K选择器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention  # 注意力类型和基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 获取旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer  # 流水线缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 获取是否捕获模式
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.models.utils import (  # 模型工具函数
    apply_qk_norm,  # 应用QK归一化
    create_fused_set_kv_buffer_arg,  # 创建融合设置KV缓冲区参数
    enable_fused_set_kv_buffer,  # 启用融合设置KV缓冲区
)
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.utils import (  # 工具函数
    add_prefix,
    is_cuda,
    is_non_idle_and_non_empty,  # 是否非空闲且非空
    is_npu,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 获取RoPE配置

LoraConfig = None  # LoRA配置占位
logger = logging.getLogger(__name__)  # 获取当前模块日志器
_is_cuda = is_cuda()  # 是否为CUDA设备
_is_npu = is_npu()  # 是否为NPU设备


class LLaDA2MoeMLP(nn.Module):
    """LLaDA2 MoE模型的密集MLP层"""

    def __init__(
        self,
        intermediate_size: int,  # 中间层大小
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: Optional[bool] = True,  # 是否归约结果
        prefix: str = "",  # 参数前缀
        tp_rank: Optional[int] = None,  # 张量并行秩
        tp_size: Optional[int] = None,  # 张量并行大小
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.tp_size = tp_size  # 保存TP大小

        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并
            config.hidden_size,
            [intermediate_size] * 2,  # 两个中间层大小
            bias=config.use_bias,  # 是否使用偏置
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(  # 下投影
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,  # 是否使用偏置
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        if config.hidden_act != "silu":  # 仅支持silu激活
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()  # SiLU激活与乘法融合函数

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """密集MLP前向传播：门控上投影 -> SiLU激活 -> 下投影"""
        if (self.tp_size == 1) and hidden_states.shape[0] == 0:  # 单TP且空输入直接返回
            return hidden_states

        gate_up, _ = self.gate_up_proj(hidden_states)  # 门控上投影
        hidden_states = self.act_fn(gate_up)  # SiLU激活和门控乘法
        hidden_states, _ = self.down_proj(  # 下投影
            hidden_states, skip_all_reduce=use_reduce_scatter  # 是否跳过全归约
        )
        return hidden_states  # 返回MLP输出


class LLaDA2MoeGate(nn.Module):
    """LLaDA2 MoE路由门控层，计算每个token到各专家的logits"""

    def __init__(
        self,
        config,  # 模型配置
        params_dtype: Optional[torch.dtype] = None,  # 参数数据类型
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        if params_dtype is None:  # 默认数据类型
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype  # 保存参数数据类型
        self.weight = nn.Parameter(  # 门控权重参数
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):  # 启用专家偏置
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:  # 不使用专家偏置
            self.expert_bias = None

    def forward(self, hidden_states):
        """门控前向传播：计算路由logits"""
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(  # 线性变换
            hidden_states.dtype  # 转回原始数据类型
        )
        return logits  # 返回路由logits


class LLaDA2MoeSparseMoeBlock(nn.Module):
    """LLaDA2稀疏MoE块，支持sigmoid/softmax路由、共享专家、DeepEP分发和双流计算"""

    def __init__(
        self,
        layer_id: int,  # 层ID
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层ID
        self.alt_stream = alt_stream  # 保存备用流
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化Top-K概率
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.num_shared_experts = config.num_shared_experts  # 共享专家数
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)  # 路由缩放因子
        self.score_function = getattr(config, "score_function", None)  # 评分函数

        # fused_topk_npu() conducting norm before scale with routed_scaling_factor by default
        # norm_topk_prob=True will renorm the routed_scaling_factor thus need to keep norm_topk_prob=False
        if _is_npu:  # NPU上不归一化Top-K概率
            self.norm_topk_prob = False

        if config.hidden_act != "silu":  # 仅支持silu激活
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)  # 路由器数据类型
        if router_dtype is None:  # 未指定
            self.router_dtype = None
        elif router_dtype == "fp32":  # FP32
            self.router_dtype = torch.float32
        else:  # BF16
            self.router_dtype = torch.bfloat16

        # TODO global_server_args.ep_num_redundant_experts is used for eplb, not supported now
        assert get_global_server_args().ep_num_redundant_experts == 0  # 当前不支持冗余专家
        # check group topk
        self.num_expert_group = getattr(config, "n_group", 0)  # 专家分组数
        self.topk_group = getattr(config, "topk_group", 0)  # 每组Top-K
        if self.num_expert_group > 0 or self.topk_group > 0:  # 使用分组Top-K
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:  # 不使用分组Top-K
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = (  # 总专家数
            config.num_experts + get_global_server_args().ep_num_redundant_experts
        )

        self.gate = LLaDA2MoeGate(  # 路由门控
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = (  # 修正偏置
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:  # 校验评分函数与偏置组合
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"

        self.topk = TopK(  # Top-K选择器
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            # num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        self.experts = get_moe_impl_class(quant_config)(  # 专家网络实现
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )
        # shared expert
        if config.num_shared_experts is not None:  # 共享专家
            if hasattr(config, "moe_shared_expert_intermediate_size"):  # 共享专家中间层大小
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:  # 使用MoE中间层大小
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts  # 乘以共享专家数
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = LLaDA2MoeMLP(  # 共享专家MLP
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,  # 不归约
                prefix=add_prefix("shared_experts", prefix),
                **(  # DeepEP模式下禁用TP
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        # dispatcher
        if get_moe_a2a_backend().is_deepep():  # DeepEP分发器
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()  # EP大小等于TP大小

            self.deepep_dispatcher = DeepEPDispatcher(  # DeepEP分发器
                group=parallel_state.get_tp_group().device_group,  # 设备组
                router_topk=self.top_k,  # 路由Top-K
                permute_fusion=True,  # 排列融合
                num_experts=self.num_experts,  # 专家数
                num_local_experts=config.num_experts // self.tp_size,  # 本地专家数
                hidden_size=config.hidden_size,  # 隐藏层大小
                params_dtype=config.torch_dtype,  # 参数数据类型
                deepep_mode=get_deepep_mode(),  # DeepEP模式
                async_finish=True,  # 异步完成
                return_recv_hook=True,  # 返回接收钩子
            )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """稀疏MoE前向传播，根据后端选择正常或DeepEP模式"""
        if not get_moe_a2a_backend().is_deepep():  # 非DeepEP模式
            return self.forward_normal(hidden_states, use_reduce_scatter)
        else:  # DeepEP模式
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        """获取MoE专家的权重数据列表"""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]  # 排除修正偏置
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        """计算共享专家输出"""
        shared_output = None  # 初始化共享输出
        if self.num_shared_experts > 0:  # 有共享专家时计算
            shared_output = self.shared_experts(hidden_states)
        return shared_output  # 返回共享输出

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        """计算路由专家输出"""
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)  # 计算路由logits
        topk_output = self.topk(hidden_states, router_logits)  # Top-K选择
        return self.experts(hidden_states, topk_output)  # 返回路由专家输出

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """双流计算模式：共享专家和路由专家并行执行"""
        current_stream = torch.cuda.current_stream()  # 当前CUDA流
        self.alt_stream.wait_stream(current_stream)  # 等待当前流完成
        shared_output = self._forward_shared_experts(hidden_states.clone())  # 克隆输入计算共享专家

        with torch.cuda.stream(self.alt_stream):  # 在备用流上计算路由专家
            router_output = self._forward_router_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)  # 等待备用流完成

        return router_output, shared_output  # 返回路由输出和共享输出

    def forward_normal(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """正常模式MoE前向传播"""
        num_tokens, hidden_size = hidden_states.shape  # 获取形状
        hidden_states = hidden_states.view(-1, hidden_size)  # 展平

        if (  # 使用双流计算
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(  # 双流计算
                hidden_states
            )
        else:  # 顺序计算
            shared_output = self._forward_shared_experts(hidden_states)  # 共享专家
            final_hidden_states = self._forward_router_experts(hidden_states)  # 路由专家

        if self.num_shared_experts > 0:  # 合并共享和路由输出
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 张量并行全归约
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 全归约
        return final_hidden_states.view(num_tokens, hidden_size)  # 恢复形状并返回

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch  # 输入隐藏状态，前向批次
    ) -> torch.Tensor:
        """DeepEP模式MoE前向传播"""
        shared_output = None  # 共享输出初始化
        forward_mode = forward_batch.forward_mode  # 前向模式
        if is_non_idle_and_non_empty(forward_mode, hidden_states):  # 非空闲且非空
            router_logits = self.gate(hidden_states)  # 计算路由logits
            if self.num_shared_experts > 0:  # 计算共享专家
                shared_output = self.shared_experts(hidden_states)

            topk_output = self.topk(  # Top-K选择
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,  # 非填充token数
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(  # 专家位置分发信息
                    layer_id=self.layer_id,
                ),
            )
        else:  # 空闲或空输入
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空Top-K输出

        final_hidden_states = self.experts(  # 专家计算
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:  # 合并共享输出
            final_hidden_states += shared_output
        return final_hidden_states  # 返回最终输出


class LLaDA2MoeAttention(nn.Module):
    """LLaDA2 MoE注意力层，支持QK归一化、部分旋转和编码器类型注意力"""

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        self.total_kv_heads = config.num_key_value_heads  # 总KV头数
        self.dp_size = get_attention_dp_size()  # 注意力DP大小
        attn_tp_rank = get_attention_tp_rank()  # 注意力TP秩
        attn_tp_size = get_attention_tp_size()  # 注意力TP大小

        assert self.total_num_heads % attn_tp_size == 0  # 头数必须能被TP大小整除
        if self.total_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_kv_heads % attn_tp_size == 0  # KV头数必须能被TP大小整除
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_kv_heads == 0  # TP大小必须能被KV头数整除
        assert self.total_num_heads >= self.total_kv_heads  # Q头数必须大于等于KV头数

        self.num_heads = self.total_num_heads // attn_tp_size  # 每个TP秩的头数
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)  # 头维度
        self.q_size = self.head_dim * self.num_heads  # Q的总维度

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size)  # 每个TP秩的KV头数
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)  # KV的总维度

        self.scale = self.head_dim**-0.5  # 缩放因子

        self.use_qk_norm = getattr(config, "use_qk_norm", True)  # 是否使用QK归一化

        self.query_key_value = QKVParallelLinear(  # QKV并行线性投影
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),  # 是否使用偏置
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.use_qk_norm:  # QK归一化
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q层归一化
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K层归一化

        self.dense = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,  # 是否使用偏置
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if hasattr(config, "partial_rotary_factor"):  # 部分旋转因子
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):  # 旋转维度
            self.rotary_dim = config.rotary_dim
        else:  # 默认全旋转
            self.rotary_dim = self.head_dim
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置
        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,  # 编码器类型注意力
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream  # 保存备用流

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """注意力前向传播：QKV投影 -> QK归一化 -> RoPE -> 注意力计算 -> 输出投影"""
        if hidden_states.shape[0] == 0:  # 空输入直接返回
            return hidden_states
        qkv, _ = self.query_key_value(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分离Q、K、V
        if self.use_qk_norm:  # 应用QK归一化
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.query_layernorm,
                k_norm=self.key_layernorm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,  # 备用流
            )
        can_fuse_set_kv = (  # 是否可以融合设置KV缓冲区
            self.head_dim == self.rotary_emb.rotary_dim  # 头维度等于旋转维度
            and enable_fused_set_kv_buffer(forward_batch)  # 启用融合设置KV
        )
        q, k = self.rotary_emb(  # 应用旋转位置编码
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(  # 融合设置KV参数
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if can_fuse_set_kv  # 可以融合时创建参数
                else None  # 否则为None
            ),
        )
        context_layer = self.attn(  # 计算注意力
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not can_fuse_set_kv,  # 不能融合时保存KV缓存
        )
        attn_output, _ = self.dense(context_layer)  # 输出投影
        return attn_output  # 返回注意力输出


class LLaDA2MoeBlock(nn.Module):
    """LLaDA2 MoE解码器块，包含注意力、MLP/MoE和层通信器"""

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ):
        super().__init__()  # 调用父类初始化
        hidden_size = config.hidden_size  # 隐藏层大小

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.dp_size = get_attention_dp_size()  # 注意力DP大小
        self.attention = LLaDA2MoeAttention(  # 注意力层
            config,
            layer_id,
            quant_config,
            reduce_results=False,  # 不自动归约
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id  # 保存层ID
        self.attn_tp_size = get_attention_tp_size()  # 注意力TP大小
        self.attn_tp_rank = get_attention_tp_rank()  # 注意力TP秩

        self.is_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id)  # 是否为稀疏层
        is_previous_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id - 1)  # 前一层是否稀疏
        is_next_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id + 1)  # 后一层是否稀疏

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 初始化层散射模式
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1  # 是否为最后一层

        if self.is_layer_sparse:  # 稀疏层使用MoE
            self.mlp = LLaDA2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:  # 密集层使用MLP
            if enable_moe_dense_fully_dp():  # 启用MoE密集全DP时
                mlp_tp_rank, mlp_tp_size = 0, 1  # 禁用TP
            else:  # 正常TP
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = LLaDA2MoeMLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)  # 注意力后归一化

        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,  # 允许reduce-scatter
        )

    def _is_layer_sparse(self, config: PretrainedConfig, layer_id: int) -> bool:
        """判断指定层是否为稀疏MoE层"""
        return (
            config.num_experts is not None and layer_id >= config.first_k_dense_replace  # 在前k个密集层之后为稀疏
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差连接
    ) -> torch.Tensor:
        """解码器块前向传播：注意力准备 -> 注意力 -> MLP准备 -> MLP/MoE -> 后处理"""
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力输入
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        hidden_states = self.attention(  # 注意力计算
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP输入
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(  # 是否使用reduce-scatter
            forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)  # MLP/MoE计算

        hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        return hidden_states, residual  # 返回隐藏状态和残差


class LLaDA2MoeModel(nn.Module):
    """LLaDA2 MoE模型主体，包含嵌入层、解码器块和最终归一化"""

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.vocab_size = config.vocab_size  # 词表大小
        self.embed_dim = config.hidden_size  # 嵌入维度
        if self.pp_group.is_first_rank:  # 第一个秩初始化嵌入层
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("word_embeddings", prefix),
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:  # 非第一个秩使用缺失层
            self.word_embeddings = PPMissingLayer()

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)  # 嵌入dropout

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,
            lambda idx, prefix: LLaDA2MoeBlock(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行大小
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:  # 最后一个秩初始化最终归一化
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:  # 非最后一个秩使用缺失层
            self.norm = PPMissingLayer(return_tuple=True)

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
                hidden_states = self.word_embeddings(input_ids)
            else:  # 使用预计算嵌入
                hidden_states = input_embeds
            residual = None  # 初始无残差
        else:  # 非第一个秩从代理张量获取
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]  # 获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 获取残差

        for i in range(self.start_layer, self.end_layer):  # 遍历所有层
            with get_global_expert_distribution_recorder().with_current_layer(i):  # 记录专家分布
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
        if not self.pp_group.is_last_rank:  # 非最后一个秩返回代理张量
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:  # 最后一个秩应用归一化
            if not forward_batch.forward_mode.is_idle():  # 非空闲模式
                if residual is None:  # 无残差时直接归一化
                    hidden_states = self.norm(hidden_states)
                else:  # 有残差时融合归一化
                    hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states  # 返回隐藏状态


class LLaDA2MoeModelLM(nn.Module):
    """LLaDA2 MoE因果语言模型，包含模型主体和语言模型头"""

    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        alt_stream = torch.cuda.Stream() if _is_cuda else None  # CUDA设备上创建备用流

        self.model = LLaDA2MoeModel(  # 模型主体
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""),
        )

        if config.tie_word_embeddings:  # 共享词嵌入和LM头权重
            self.lm_head = self.model.word_embeddings
        else:  # 独立LM头
            # TODO something wrong with ParallelLMHead with DP attention enabled
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否启用DP LM头
            )
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)  # logits处理器（返回全logits）

    @property
    def start_layer(self):
        """获取起始层索引"""
        return self.model.start_layer

    @property
    def end_layer(self):
        """获取结束层索引"""
        return self.model.end_layer

    def get_embed_and_head(self):
        """Used by the eagle_worker."""
        """获取嵌入权重和语言模型头权重（供eagle_worker使用）"""
        return self.model.word_embeddings.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        """设置嵌入权重和语言模型头权重（供eagle_worker使用）"""
        del self.model.word_embeddings.weight  # 删除旧嵌入权重
        del self.lm_head.weight  # 删除旧LM头权重
        self.model.word_embeddings.weight = embed  # 设置新嵌入权重
        self.lm_head.weight = head  # 设置新LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA

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
        else:  # 非最后一个秩返回隐藏状态
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，支持堆叠参数和专家参数"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),  # 门控投影合并
            ("gate_up_proj", "up_proj", 1),  # 上投影合并
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())  # 参数字典
        for name, loaded_weight in weights:  # 遍历所有权重
            if (  # 跳过不需要的权重
                ("v_head" in name)  # 跳过v_head
                or ("inv_freq" in name)  # 跳过旋转频率
                or (self.config.tie_word_embeddings and "lm_head" in name)  # 跳过共享权重
            ):
                continue

            if (  # 对lm_head权重进行归一化
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                import torch.nn.functional as F

                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)  # L2归一化

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 处理堆叠参数
                if weight_name not in name:  # 名称不包含权重名则跳过
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:  # 跳过专家参数（单独处理）
                    continue
                name = name.replace(weight_name, param_name)  # 替换为参数名
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ模型额外偏置
                    continue
                if name not in params_dict:  # 跳过不存在的参数
                    continue

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重分片
                break
            else:  # 处理专家参数
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:  # 名称不包含权重名则跳过
                        continue
                    name = name.replace(weight_name, param_name)  # 替换为参数名
                    if name not in params_dict:  # 跳过不存在的参数
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载专家权重
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:  # 处理常规权重
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ模型额外偏置
                        continue
                    if name not in params_dict:  # 跳过不存在的参数
                        continue

                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重

        self.routed_experts_weights_of_layer = {  # 保存路由专家权重
            layer_id: layer.mlp.get_moe_weights()
            for layer_id, layer in enumerate(self.model.layers)
            if not isinstance(layer, PPMissingLayer)  # 跳过缺失层
            and isinstance(layer.mlp, LLaDA2MoeSparseMoeBlock)  # 仅MoE层
        }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置的模型配置"""
        num_groups = getattr(config, "n_group", 0)  # 专家分组数
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,  # 层数
            num_logical_experts=config.num_experts,  # 逻辑专家数
            num_groups=None if num_groups == 0 else num_groups,  # 分组数
        )


EntryClass = LLaDA2MoeModelLM  # 入口类
