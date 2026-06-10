# EXAONE-MoE模型推理实现文件
# 本文件实现了基于混合专家(MoE)架构的EXAONE模型，仅用于推理，兼容HuggingFace权重格式
# 主要包含：MLP层、稀疏MoE块、注意力层、解码器层、模型主体和因果语言模型

# Copyright 2025 The LG AI Research Team  # LG AI研究团队版权声明
# Copyright 2023-2024 SGLang Team  # SGLang团队版权声明
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache 2.0许可证授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache 2.0许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 依许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不附带任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 请参阅许可证获取管理权限和
# limitations under the License.  # 限制的具体条款
# ==============================================================================  # 分隔线

# Adapted from the vLLM version of EXAONE-MoE model  # 从vLLM版本的EXAONE-MoE模型适配而来
"""Inference-only ExaoneMoE model compatible with HuggingFace weights."""  # 仅推理的ExaoneMoE模型，兼容HuggingFace权重

import logging  # 导入日志模块
from collections.abc import Iterable  # 从collections.abc导入Iterable类型
from typing import Any, Dict, Optional, Tuple, Union  # 导入类型注解

import torch  # 导入PyTorch
from torch import nn  # 从PyTorch导入神经网络模块
from transformers import PretrainedConfig  # 从transformers导入预训练配置类

from sglang.srt.distributed import (  # 从分布式模块导入
    get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_world_size,  # 获取张量模型并行世界大小
    tensor_model_parallel_all_reduce,  # 张量模型并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入全局专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置模型配置
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo  # 导入专家位置调度信息
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.dp_attention import (  # 从数据并行注意力模块导入
    get_attention_tp_rank,  # 获取注意力张量并行秩
    get_attention_tp_size,  # 获取注意力张量并行大小
    is_dp_attention_enabled,  # 判断是否启用数据并行注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import (  # 从线性层模块导入
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 导入logits处理器及其输出
from sglang.srt.layers.moe import (  # 从MoE模块导入
    get_moe_a2a_backend,  # 获取MoE全对全通信后端
    should_skip_post_experts_all_reduce,  # 判断是否跳过专家后全归约
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类获取函数
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # 导入融合MoE Triton实现
from sglang.srt.layers.moe.topk import TopK  # 导入TopK路由选择
from sglang.srt.layers.moe.utils import RoutingMethodType  # 导入路由方法类型枚举
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码获取函数
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 从词表并行嵌入模块导入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入层
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 导入CUDA图捕获模式判断
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和流水线代理张量
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取
from sglang.srt.utils import LazyValue, add_prefix, is_cuda, make_layers  # 导入惰性值、前缀添加、CUDA判断、层创建工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_is_cuda = is_cuda()  # 判断当前是否为CUDA环境


class ExaoneMoEMLP(nn.Module):  # ExaoneMoE的MLP（多层感知机）模块
    def __init__(  # 初始化函数
        self,
        hidden_size: int,  # 隐藏层维度大小
        intermediate_size: int,  # 中间层维度大小
        hidden_act: str,  # 隐藏层激活函数名称
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，默认为None
        reduce_results: bool = True,  # 是否归约结果，默认为True
        prefix: str = "",  # 参数前缀，默认为空
        tp_rank: Optional[int] = None,  # 张量并行秩，默认为None
        tp_size: Optional[int] = None,  # 张量并行大小，默认为None
    ) -> None:
        super().__init__()  # 调用父类初始化
        gateup_quant_config = quant_config  # gate_up投影的量化配置
        down_quant_config = quant_config  # down投影的量化配置
        if quant_config and hasattr(quant_config, "ignore") and quant_config.ignore:  # 如果量化配置存在且有忽略列表
            if add_prefix("gate_proj", prefix) in quant_config.ignore:  # 如果gate_proj在忽略列表中
                gateup_quant_config = None  # gate_up投影不使用量化
            if add_prefix("down_proj", prefix) in quant_config.ignore:  # 如果down_proj在忽略列表中
                down_quant_config = None  # down投影不使用量化

        self.gate_up_proj = MergedColumnParallelLinear(  # gate和up的合并列并行线性层
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度（gate和up各一份）
            bias=False,  # 不使用偏置
            quant_config=gateup_quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 参数前缀
            tp_rank=tp_rank,  # 张量并行秩
            tp_size=tp_size,  # 张量并行大小
        )
        self.down_proj = RowParallelLinear(  # down行并行线性层
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 不使用偏置
            quant_config=down_quant_config,  # 量化配置
            reduce_results=reduce_results,  # 是否归约结果
            prefix=add_prefix("down_proj", prefix),  # 参数前缀
            tp_rank=tp_rank,  # 张量并行秩
            tp_size=tp_size,  # 张量并行大小
        )
        if hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. "  # 不支持的激活函数
                "Only silu is supported for now."  # 目前仅支持silu
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数

    def forward(  # 前向传播函数
        self,
        x,  # 输入张量
        forward_batch=None,  # 前向批次信息
        should_allreduce_fusion: bool = False,  # 是否融合全归约，默认为False
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter，默认为False
    ):
        gate_up, _ = self.gate_up_proj(x)  # 通过gate_up投影并获取输出
        x = self.act_fn(gate_up)  # 应用SiLU激活函数和门控
        x, _ = self.down_proj(  # 通过down投影
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,  # 是否跳过全归约
        )
        return x  # 返回输出


class ExaoneMoESparseMoEBlock(nn.Module):  # ExaoneMoE稀疏MoE块
    def __init__(  # 初始化函数
        self,
        layer_id: int,  # 层ID
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流，用于双流并行
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行大小
        self.moe_ep_size = get_moe_expert_parallel_world_size()  # 获取MoE专家并行大小
        self.layer_id = layer_id  # 保存层ID
        self.routed_scaling_factor = config.routed_scaling_factor  # 路由缩放因子
        self.alt_stream = alt_stream  # 保存备用CUDA流

        self.n_routed_experts = config.num_experts  # 路由专家数量

        if self.tp_size > config.num_experts:  # 如果张量并行大小大于专家数量
            raise ValueError(  # 抛出值错误
                f"Tensor parallel size {self.tp_size} is greater than "  # 张量并行大小大于
                f"the number of experts {config.num_experts}."  # 专家数量
            )

        self.gate = ReplicatedLinear(  # 路由门控线性层
            config.hidden_size,  # 输入维度
            config.num_experts,  # 输出维度（专家数量）
            bias=False,  # 不使用偏置
            quant_config=None,  # 不使用量化
            prefix=add_prefix("gate", prefix),  # 参数前缀
        )

        self.e_score_correction_bias = nn.Parameter(  # 专家分数校正偏置参数
            torch.empty(config.num_experts, dtype=torch.float32)  # 大小为专家数量的float32张量
        )

        self.experts = get_moe_impl_class(quant_config)(  # 获取MoE实现类并实例化
            num_experts=config.num_experts  # 专家数量
            + get_global_server_args().ep_num_redundant_experts,  # 加上冗余专家数量
            top_k=config.num_experts_per_tok,  # 每个token选择的专家数
            hidden_size=config.hidden_size,  # 隐藏层维度
            intermediate_size=config.moe_intermediate_size,  # MoE中间层维度
            layer_id=self.layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("experts", prefix),  # 参数前缀
            routing_method_type=RoutingMethodType.RenormalizeNaive,  # 路由方法类型为重归一化朴素
        )

        self.topk = TopK(  # TopK路由选择
            top_k=config.num_experts_per_tok,  # 每个token选择的专家数
            renormalize=config.norm_topk_prob,  # 是否重归一化
            use_grouped_topk=True,  # 使用分组TopK
            num_expert_group=config.n_group,  # 专家分组数
            topk_group=config.topk_group,  # 每组选择的专家数
            correction_bias=self.e_score_correction_bias,  # 校正偏置
            routed_scaling_factor=self.routed_scaling_factor,  # 路由缩放因子
            apply_routed_scaling_factor_on_output=True,  # 在输出上应用路由缩放因子
            scoring_func="sigmoid",  # 评分函数为sigmoid
        )

        if config.num_shared_experts is not None:  # 如果存在共享专家
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts  # 共享专家中间层维度
            self.shared_experts = ExaoneMoEMLP(  # 共享专家MLP
                hidden_size=config.hidden_size,  # 隐藏层维度
                intermediate_size=intermediate_size,  # 中间层维度
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                reduce_results=False,  # 不归约结果
                prefix=add_prefix("shared_experts", prefix),  # 参数前缀
                **(  # 如果使用DeepEP后端则设置tp_rank和tp_size
                    dict(tp_rank=0, tp_size=1)  # tp_rank=0, tp_size=1
                    if get_moe_a2a_backend().is_deepep()  # 如果是DeepEP后端
                    else {}  # 否则不添加额外参数
                ),
            )

        if get_moe_a2a_backend().is_deepep():  # 如果使用DeepEP后端
            self.ep_size = get_moe_expert_parallel_world_size()  # 获取专家并行大小
            self.num_experts = (  # 专家总数
                config.num_experts + get_global_server_args().ep_num_redundant_experts  # 包括冗余专家
            )
            self.top_k = config.num_experts_per_tok  # 每个token选择的专家数

    def get_moe_weights(self):  # 获取MoE权重
        return [  # 返回权重列表
            x.data  # 参数数据
            for name, x in self.experts.named_parameters()  # 遍历专家的所有命名参数
            if name not in ["correction_bias"]  # 排除校正偏置
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播共享专家
        shared_output = self.shared_experts(hidden_states)  # 通过共享专家MLP
        return shared_output  # 返回共享专家输出

    def _forward_deepep(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):  # DeepEP后端前向传播
        shared_output = None  # 初始化共享输出为None
        if hidden_states.shape[0] > 0:  # 如果有token输入
            router_logits, _ = self.gate(hidden_states)  # 通过门控获取路由logits
            shared_output = self._forward_shared_experts(hidden_states)  # 获取共享专家输出
            topk_output = self.topk(  # 获取TopK路由输出
                hidden_states,  # 隐藏状态
                router_logits,  # 路由logits
                num_token_non_padded=forward_batch.num_token_non_padded,  # 非填充token数
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(  # 专家位置调度信息
                    layer_id=self.layer_id,  # 层ID
                ),
            )
        else:  # 如果没有token输入
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 获取空的TopK输出
        final_hidden_states = self.experts(  # 通过路由专家
            hidden_states=hidden_states,  # 隐藏状态
            topk_output=topk_output,  # TopK路由输出
        )

        if shared_output is not None:  # 如果共享输出不为空
            final_hidden_states.add_(shared_output)  # 将共享输出加到最终隐藏状态上

        return final_hidden_states  # 返回最终隐藏状态

    def _forward_router_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播路由专家
        router_logits, _ = self.gate(hidden_states)  # 通过门控获取路由logits
        topk_output = self.topk(hidden_states, router_logits)  # 获取TopK路由输出
        return self.experts(hidden_states, topk_output)  # 通过路由专家并返回

    def forward_normal_dual_stream(  # 使用双流并行的前向传播
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()  # 获取当前CUDA流
        self.alt_stream.wait_stream(current_stream)  # 备用流等待当前流

        shared_output = self._forward_shared_experts(hidden_states.clone())  # 在当前流计算共享专家（克隆输入避免竞争）

        with torch.cuda.stream(self.alt_stream):  # 在备用流上执行
            router_output = self._forward_router_experts(hidden_states)  # 计算路由专家

        current_stream.wait_stream(self.alt_stream)  # 当前流等待备用流完成

        return router_output, shared_output  # 返回路由专家输出和共享专家输出

    def forward(  # 前向传播主函数
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次信息
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape  # 获取token数和隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑隐藏状态形状

        if get_moe_a2a_backend().is_deepep():  # 如果使用DeepEP后端
            return self._forward_deepep(hidden_states, forward_batch)  # 使用DeepEP前向传播

        if (  # 如果满足双流并行条件
            self.alt_stream is not None  # 备用流存在
            and hidden_states.shape[0] > 0  # 有token输入
            and get_is_capture_mode()  # 处于CUDA图捕获模式
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(  # 使用双流并行
                hidden_states
            )
        else:  # 否则顺序执行
            shared_output = self._forward_shared_experts(hidden_states)  # 先计算共享专家
            final_hidden_states = self._forward_router_experts(hidden_states)  # 再计算路由专家

        if shared_output is not None:  # 如果共享输出不为空
            final_hidden_states = final_hidden_states + shared_output  # 合并路由专家和共享专家输出
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 如果需要全归约
            is_tp_path=True,  # 是张量并行路径
            use_reduce_scatter=use_reduce_scatter,  # 是否使用reduce-scatter
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 执行张量并行全归约

        return final_hidden_states.view(num_tokens, hidden_dim)  # 返回重塑后的隐藏状态


class ExaoneMoEAttention(nn.Module):  # ExaoneMoE注意力模块
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        hidden_size: int,  # 隐藏层维度
        num_heads: int,  # 注意力头数
        num_kv_heads: int,  # KV头数
        layer_id: int = 0,  # 层ID，默认为0
        rope_theta: float = 1000000,  # 旋转位置编码基数，默认1000000
        rope_scaling: Optional[Dict[str, Any]] = None,  # 旋转位置编码缩放配置
        rope_is_neox_style: bool = True,  # 是否使用Neox风格的RoPE，默认为True
        max_position_embeddings: int = 8192,  # 最大位置编码数，默认8192
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        bias: bool = False,  # 是否使用偏置，默认为False
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 保存隐藏层维度
        attn_tp_rank = get_attention_tp_rank()  # 获取注意力张量并行秩
        attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小

        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % attn_tp_size == 0  # 确保头数能被并行大小整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 每个并行秩的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= attn_tp_size:  # 如果KV头数大于等于并行大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此进行分区
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头分配到多个张量并行GPU上
            assert self.total_num_kv_heads % attn_tp_size == 0  # 确保KV头数能被并行大小整除
        else:  # 否则KV头数小于并行大小
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此进行复制
            # the KV heads across multiple tensor parallel GPUs.  # 将KV头复制到多个张量并行GPU上
            assert attn_tp_size % self.total_num_kv_heads == 0  # 确保并行大小能被KV头数整除
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 每个并行秩的KV头数
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo  # MistralConfig有一个可选的head_dim，由Mistral-Nemo引入
        self.head_dim = getattr(  # 头维度
            config, "head_dim", self.hidden_size // self.total_num_heads  # 从配置获取或根据隐藏维度计算
        )
        self.q_size = self.num_heads * self.head_dim  # Q维度大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV维度大小
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码数

        qkv_quant_config = quant_config  # QKV投影的量化配置
        o_quant_config = quant_config  # 输出投影的量化配置
        if quant_config and hasattr(quant_config, "ignore") and quant_config.ignore:  # 如果量化配置有忽略列表
            if add_prefix("q_proj", prefix) in quant_config.ignore:  # 如果q_proj在忽略列表中
                qkv_quant_config = None  # QKV投影不使用量化
            if add_prefix("o_proj", prefix) in quant_config.ignore:  # 如果o_proj在忽略列表中
                o_quant_config = None  # 输出投影不使用量化

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性投影
            hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            bias=bias,  # 是否使用偏置
            quant_config=qkv_quant_config,  # 量化配置
            prefix=add_prefix("qkv_proj", prefix),  # 参数前缀
            tp_rank=attn_tp_rank,  # 注意力张量并行秩
            tp_size=attn_tp_size,  # 注意力张量并行大小
        )
        self.o_proj = RowParallelLinear(  # 输出行并行线性投影
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=bias,  # 是否使用偏置
            quant_config=o_quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 参数前缀
            tp_rank=attn_tp_rank,  # 注意力张量并行秩
            tp_size=attn_tp_size,  # 注意力张量并行大小
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q的RMS归一化
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K的RMS归一化

        if quant_config is not None and quant_config.get_name() == "gguf":  # 如果使用GGUF量化
            rope_is_neox_style = False  # 不使用Neox风格的RoPE

        self.sliding_window = config.layer_types[layer_id] == "sliding_attention"  # 是否使用滑动窗口注意力

        # apply rotary embeddings to every layer in full attention models  # 在全注意力模型中对每层应用旋转位置编码
        self.apply_rope_all_layers = "sliding_attention" not in config.layer_types  # 如果没有滑动窗口层，则对所有层应用RoPE

        self.rotary_emb = get_rope(  # 获取旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # 基数
            rope_scaling=rope_scaling,  # 缩放配置
            is_neox_style=rope_is_neox_style,  # 是否Neox风格
        )

        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            prefix=add_prefix("attn", prefix),  # 参数前缀
            sliding_window_size=(  # 滑动窗口大小
                config.sliding_window if self.sliding_window else None  # 如果使用滑动窗口则设置大小，否则为None
            ),
        )
        self.layer_id = layer_id  # 保存层ID

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 分割为Q、K、V

        q = q.reshape(-1, self.head_dim)  # 重塑Q为二维
        q = self.q_norm(q)  # 对Q进行RMS归一化
        q = q.reshape(-1, self.num_heads * self.head_dim)  # 重塑Q回原始形状

        k = k.reshape(-1, self.head_dim)  # 重塑K为二维
        k = self.k_norm(k)  # 对K进行RMS归一化
        k = k.reshape(-1, self.num_kv_heads * self.head_dim)  # 重塑K回原始形状

        if self.sliding_window or self.apply_rope_all_layers:  # 如果使用滑动窗口或对所有层应用RoPE
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码

        attn_output = self.attn(q, k, v, forward_batch)  # 计算注意力
        output, _ = self.o_proj(attn_output)  # 通过输出投影

        return output  # 返回输出


class ExaoneMoEDecoderLayer(nn.Module):  # ExaoneMoE解码器层
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int = 0,  # 层ID，默认为0
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.config = config  # 保存配置
        rope_theta = getattr(config, "rope_theta", 1000000)  # 获取RoPE基数，默认1000000
        rope_scaling = getattr(config, "rope_scaling", None)  # 获取RoPE缩放配置
        if rope_scaling is not None and getattr(  # 如果RoPE缩放配置存在且配置中有原始最大位置编码
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (  # 设置原始最大位置编码
                config.original_max_position_embeddings  # 从配置获取
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)  # 获取是否Neox风格RoPE
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)  # 获取最大位置编码数

        attention_bias = getattr(config, "attention_bias", False) or getattr(  # 获取是否使用注意力偏置
            config, "bias", False
        )
        self.attn_tp_size = get_attention_tp_size()  # 获取注意力张量并行大小
        self.attn_tp_rank = get_attention_tp_rank()  # 获取注意力张量并行秩

        self.self_attn = ExaoneMoEAttention(  # 自注意力模块
            config=config,  # 预训练配置
            hidden_size=self.hidden_size,  # 隐藏层维度
            num_heads=config.num_attention_heads,  # 注意力头数
            num_kv_heads=config.num_key_value_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            rope_theta=rope_theta,  # RoPE基数
            rope_scaling=rope_scaling,  # RoPE缩放配置
            rope_is_neox_style=rope_is_neox_style,  # 是否Neox风格RoPE
            max_position_embeddings=max_position_embeddings,  # 最大位置编码数
            quant_config=quant_config,  # 量化配置
            bias=attention_bias,  # 是否使用偏置
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )

        if config.is_moe_layer[layer_id]:  # 如果当前层是MoE层
            self.mlp = ExaoneMoESparseMoEBlock(  # 使用稀疏MoE块
                layer_id=layer_id,  # 层ID
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                alt_stream=alt_stream,  # 备用CUDA流
                prefix=add_prefix("mlp", prefix),  # 参数前缀
            )
        else:  # 否则是密集层
            self.mlp = ExaoneMoEMLP(  # 使用普通MLP
                hidden_size=self.hidden_size,  # 隐藏层维度
                intermediate_size=config.intermediate_size,  # 中间层维度
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 参数前缀
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(  # 前向传播函数
        self,
        positions: torch.Tensor,  # 位置编码张量
        hidden_states: torch.Tensor,  # 隐藏状态张量
        forward_batch: ForwardBatch,  # 前向批次信息
        residual: Optional[torch.Tensor],  # 残差张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:  # 如果没有残差（第一层）
            residual = hidden_states  # 初始化残差为隐藏状态
            hidden_states = self.input_layernorm(hidden_states)  # 对隐藏状态做层归一化
        else:  # 如果有残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 融合层归一化和残差连接

        # Self Attention  # 自注意力计算
        hidden_states = self.self_attn(  # 通过自注意力模块
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次信息
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # 注意力后层归一化和残差更新
        # Fully Connected  # 全连接层
        hidden_states = self.mlp(hidden_states)  # 通过MLP或MoE

        return hidden_states, residual  # 返回隐藏状态和残差


class ExaoneMoEModel(nn.Module):  # ExaoneMoE模型主体
    fall_back_to_pt_during_load = False  # 加载时是否回退到PyTorch

    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层维度
                enable_tp=not is_dp_attention_enabled(),  # 是否启用张量并行
            )
        else:  # 否则不是第一个秩
            self.embed_tokens = PPMissingLayer()  # 使用缺失层占位

        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 层数
            lambda idx, prefix: ExaoneMoEDecoderLayer(  # 层创建函数
                layer_id=idx,  # 层ID
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 参数前缀
                alt_stream=alt_stream,  # 备用CUDA流
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行秩
            pp_size=self.pp_group.world_size,  # 流水线并行大小
            prefix=add_prefix("layers", prefix),  # 参数前缀
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终层归一化
        else:  # 否则不是最后一个秩
            self.norm = PPMissingLayer(return_tuple=True)  # 使用缺失层占位

        # for EAGLE3 support  # 用于EAGLE3推测解码支持
        self.layers_to_capture = []  # 需要捕获的层列表

    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，默认为None
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:  # 如果是流水线并行的第一个秩
            if input_embeds is None:  # 如果没有提供输入嵌入
                hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层获取隐藏状态
            else:  # 否则使用提供的嵌入
                hidden_states = input_embeds  # 使用输入嵌入作为隐藏状态
            residual = None  # 初始化残差为None
        else:  # 否则不是第一个秩
            assert pp_proxy_tensors is not None  # 确保代理张量不为空
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理张量获取残差

        aux_hidden_states = []  # 辅助隐藏状态列表（用于EAGLE3）
        for i in range(self.start_layer, self.end_layer):  # 遍历所有层
            with get_global_expert_distribution_recorder().with_current_layer(i):  # 记录当前层
                if i in self.layers_to_capture:  # 如果当前层需要捕获
                    aux_hidden_states.append(hidden_states + residual)  # 添加辅助隐藏状态
                layer = self.layers[i]  # 获取当前层
                hidden_states, residual = layer(  # 通过当前层
                    positions, hidden_states, forward_batch, residual
                )
        if not self.pp_group.is_last_rank:  # 如果不是最后一个秩
            return PPProxyTensors(  # 返回代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:  # 否则是最后一个秩
            if hidden_states.shape[0] != 0:  # 如果有token输出
                if residual is None:  # 如果没有残差
                    hidden_states = self.norm(hidden_states)  # 对隐藏状态做层归一化
                else:  # 如果有残差
                    hidden_states, _ = self.norm(hidden_states, residual)  # 融合层归一化和残差
        if len(aux_hidden_states) == 0:  # 如果没有辅助隐藏状态
            return hidden_states  # 只返回隐藏状态

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态


class ExaoneMoEForCausalLM(nn.Module):  # ExaoneMoE因果语言模型
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置
        alt_stream = torch.cuda.Stream() if _is_cuda else None  # 如果是CUDA则创建备用流
        self.model = ExaoneMoEModel(  # 模型主体
            config,  # 预训练配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 参数前缀
            alt_stream=alt_stream,  # 备用CUDA流
        )
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入权重
            self.lm_head = self.model.embed_tokens  # 语言模型头共享词嵌入
        else:  # 否则不绑定
            self.lm_head = ParallelLMHead(  # 并行语言模型头
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏层维度
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("lm_head", prefix),  # 参数前缀
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力张量并行组
            )
        self.logits_processor = LogitsProcessor(config)  # logits处理器
        # For EAGLE3 support  # 用于EAGLE3推测解码支持
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

        self._routed_experts_weights_of_layer = LazyValue(  # 惰性值：每层的路由专家权重
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()  # 获取MoE权重
                for layer_id in range(self.start_layer, self.end_layer)  # 遍历所有层
                if isinstance(self.model.layers[layer_id].mlp, ExaoneMoESparseMoEBlock)  # 仅MoE层
            }
        )

    @property  # 属性装饰器
    def routed_experts_weights_of_layer(self):  # 获取路由专家权重
        return self._routed_experts_weights_of_layer.value  # 返回惰性值的计算结果

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播函数
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(  # 通过模型主体获取隐藏状态
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,  # 代理张量
        )

        aux_hidden_states = None  # 初始化辅助隐藏状态
        if self.capture_aux_hidden_states:  # 如果需要捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包隐藏状态和辅助隐藏状态

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            return self.logits_processor(  # 通过logits处理器
                input_ids,  # 输入ID
                hidden_states,  # 隐藏状态
                self.lm_head,  # 语言模型头
                forward_batch,  # 前向批次信息
                aux_hidden_states,  # 辅助隐藏状态
            )
        else:  # 否则不是最后一个秩
            return hidden_states  # 直接返回隐藏状态

    @torch.no_grad()  # 禁用梯度计算
    def forward_split_prefill(  # 分割预填充前向传播
        self,
        input_ids: torch.Tensor,  # 输入token ID张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        split_interval: Tuple[int, int],  # [start, end) 0-based  # 分割区间，左闭右开，从0开始
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ):
        start, end = split_interval  # 获取分割区间的起始和结束
        # embed  # 嵌入
        if start == 0:  # 如果从第0层开始
            if input_embeds is None:  # 如果没有提供输入嵌入
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)  # 通过词嵌入层
            else:  # 否则使用提供的嵌入
                forward_batch.hidden_states = input_embeds  # 使用输入嵌入
        # decoder layer  # 解码器层
        for i in range(start, end):  # 遍历分割区间内的层
            layer = self.model.layers[i]  # 获取当前层
            forward_batch.hidden_states, forward_batch.residual = layer(  # 通过当前层
                positions,  # 位置编码
                forward_batch.hidden_states,  # 隐藏状态
                forward_batch,  # 前向批次信息
                forward_batch.residual,  # 残差
            )

        if end == self.model.config.num_hidden_layers:  # 如果到达最后一层
            # norm  # 层归一化
            hidden_states, _ = self.model.norm(  # 通过最终层归一化
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states  # 更新隐藏状态
            # logits process  # logits处理
            result = self.logits_processor(  # 通过logits处理器
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:  # 否则未到达最后一层
            result = None  # 结果为None

        return result  # 返回结果

    @property  # 属性装饰器
    def start_layer(self):  # 获取起始层索引
        return self.model.start_layer  # 返回模型的起始层

    @property  # 属性装饰器
    def end_layer(self):  # 获取结束层索引
        return self.model.end_layer  # 返回模型的结束层

    def get_embed_and_head(self):  # 获取词嵌入和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和LM头权重

    def set_embed_and_head(self, embed, head):  # 设置词嵌入和语言模型头权重
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重
        del self.lm_head.weight  # 删除旧的LM头权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重
        self.lm_head.weight = head  # 设置新的LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA操作

    def load_weights(  # 加载权重函数
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False  # 权重迭代器，是否为MTP模型
    ):
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # QKV投影中Q的映射
            ("qkv_proj", "k_proj", "k"),  # QKV投影中K的映射
            ("qkv_proj", "v_proj", "v"),  # QKV投影中V的映射
            ("gate_up_proj", "gate_proj", 0),  # gate_up投影中gate的映射
            ("gate_up_proj", "up_proj", 1),  # gate_up投影中up的映射
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 专家参数映射
            ckpt_gate_proj_name="gate_proj",  # 检查点中gate投影的名称
            ckpt_down_proj_name="down_proj",  # 检查点中down投影的名称
            ckpt_up_proj_name="up_proj",  # 检查点中up投影的名称
            num_experts=self.config.num_experts,  # 专家数量
        )

        params_dict = dict(self.named_parameters())  # 参数名字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if is_mtp:  # 如果是MTP模型
                if "mtp" not in name:  # 如果权重名不包含"mtp"
                    continue  # 跳过
                if name in [  # 如果是特定的MTP权重
                    "mtp.fc.weight",  # MTP全连接权重
                    "mtp.pre_fc_norm_embedding.weight",  # MTP嵌入预归一化权重
                    "mtp.pre_fc_norm_hidden.weight",  # MTP隐藏预归一化权重
                ]:
                    name = name.replace("mtp.", "")  # 去掉"mtp."前缀
                else:  # 其他MTP权重
                    name = name.replace("mtp", "model")  # 将"mtp"替换为"model"

            if not is_mtp and "mtp" in name:  # 如果不是MTP模型但权重名包含"mtp"
                continue  # 跳过

            if "rotary_emb.inv_freq" in name or "projector" in name:  # 如果是旋转嵌入频率或投影器
                continue  # 跳过
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 如果是缓存的余弦/正弦值
                # Models trained using ColossalAI may include these tensors in  # 使用ColossalAI训练的模型可能在检查点中包含这些张量
                # the checkpoint. Skip them.  # 跳过它们
                continue  # 跳过
            if name.startswith("model.vision_tower") and name not in params_dict:  # 如果是视觉塔但不在参数字典中
                continue  # 跳过

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果分片名不在权重名中
                    continue  # 跳过
                if "mlp.experts" in name:  # 如果是MoE专家的权重
                    continue  # 跳过（由专家参数映射处理）
                name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                    continue  # 跳过
                if name not in params_dict:  # 如果参数名不在字典中
                    continue  # 跳过

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果堆叠参数映射中没有匹配
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果分片名不在权重名中
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载权重
                        param,
                        loaded_weight,
                        name,  # 参数名
                        expert_id=expert_id,  # 专家ID
                        shard_id=shard_id,  # 分片ID
                    )
                    break  # 跳出内层循环
                else:  # 如果专家参数映射中也没有匹配
                    # Skip loading extra bias for GPTQ models.  # 跳过加载GPTQ模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置但不在参数字典中
                        continue  # 跳过

                    if name not in params_dict:  # 如果参数名不在字典中
                        continue  # 跳过

                    if name in params_dict.keys():  # 如果参数名在字典中
                        param = params_dict[name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader  # 默认使用default_weight_loader
                        )
                        weight_loader(param, loaded_weight)  # 加载权重
                    else:  # 否则参数未找到
                        logger.warning(f"Parameter {name} not found in params_dict")  # 记录警告

    @classmethod  # 类方法装饰器
    def get_model_config_for_expert_location(cls, config):  # 获取专家位置的模型配置
        return ModelConfigForExpertLocation(  # 返回专家位置模型配置
            num_layers=config.num_hidden_layers,  # 隐藏层数
            num_logical_experts=config.num_experts,  # 逻辑专家数
            num_groups=None,  # 分组数为None
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):  # 设置EAGLE3需要捕获的层
        if not get_pp_group().is_last_rank:  # 如果不是流水线并行的最后一个秩
            return  # 直接返回

        self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
        if layer_ids is None:  # 如果没有指定层ID
            num_layers = self.config.num_hidden_layers  # 获取隐藏层数
            self.model.layers_to_capture = [  # 设置默认捕获层
                2,  # 第2层
                num_layers // 2,  # 中间层
                num_layers - 3,  # 倒数第3层
            ]  # Specific layers for EAGLE3 support  # EAGLE3支持的特定层
        else:  # 否则使用指定的层ID
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 加1偏移（跳过嵌入层）


EntryClass = ExaoneMoEForCausalLM  # 模型入口类
