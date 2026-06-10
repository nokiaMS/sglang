# Longcat-Flash模型推理实现文件
# 本文件实现了Longcat-Flash模型的推理专用版本
# 采用DeepSeek V2风格的MLA注意力机制和混合专家架构
# 包含MLP、路由器、MoE、解码器层、模型主体及因果语言模型等组件
# 支持N-gram嵌入、零专家、fp8/int8量化、AWQ量化和DeepGEMM加速

# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import concurrent.futures  # 并发执行
import logging  # 日志模块
from typing import Iterable, List, Optional, Tuple  # 类型提示

import torch  # PyTorch核心库
from torch import nn  # 神经网络模块

from sglang.srt.configs import LongcatFlashConfig  # Longcat-Flash配置类
from sglang.srt.distributed import (  # 分布式通信相关
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 专家位置模型配置
from sglang.srt.layers import deep_gemm_wrapper  # DeepGEMM包装器
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活与乘法融合层
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes  # 层通信器
from sglang.srt.layers.dp_attention import (  # 数据并行注意力相关
    get_attention_tp_rank,  # 获取注意力TP秩
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 均方根归一化层
from sglang.srt.layers.linear import (  # 并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # logits处理器
from sglang.srt.layers.moe.ep_moe.kernels import zero_experts_compute_triton  # 零专家Triton计算
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class  # DeepEP MoE和实现类获取
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 融合MoE Triton实现
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK  # 标准Top-K输出和Top-K选择器
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert  # 过滤MoE权重参数
from sglang.srt.layers.n_gram_embedding import NgramEmbedding  # N-gram嵌入层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置基类
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 是否为FP8 FNUZ格式
from sglang.srt.layers.quantization.fp8_utils import (  # FP8量化工具
    block_quant_dequant,  # 块量化反量化
    block_quant_to_tensor_quant,  # 块量化到张量量化
    channel_quant_to_tensor_quant,  # 通道量化到张量量化
    normalize_e4m3fn_to_e4m3fnuz,  # E4M3FN到E4M3FNUZ归一化
    requant_weight_ue8m0_inplace,  # 就地重量化权重为UE8M0
)
from sglang.srt.layers.quantization.int8_utils import (  # INT8量化工具
    block_dequant as int8_block_dequant,  # INT8块反量化
)
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息
from sglang.srt.model_loader.utils import (  # 模型加载工具
    maybe_executor_submit,  # 可能提交到线程池
    should_async_load,  # 是否异步加载
    should_deepgemm_weight_requant_ue8m0,  # 是否需要DeepGEMM权重重量化
)
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA  # DeepSeek V2 MLA注意力
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.utils import (  # 工具函数
    BumpAllocator,  # 凸包分配器
    add_prefix,
    bind_or_assign,  # 绑定或赋值
    cpu_has_amx_support,  # CPU是否支持AMX
    get_bool_env_var,  # 获取布尔环境变量
    get_device_sm,  # 获取设备计算能力
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

_is_hip = is_hip()  # 是否为HIP设备
_is_cuda = is_cuda()  # 是否为CUDA设备
_is_npu = is_npu()  # 是否为NPU设备
_is_fp8_fnuz = is_fp8_fnuz()  # 是否为FP8 FNUZ格式
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITer
_is_cpu_amx_available = cpu_has_amx_support()  # CPU AMX是否可用
_is_cpu = is_cpu()  # 是否为CPU设备
_device_sm = get_device_sm()  # 设备计算能力版本

if _is_cuda:  # CUDA平台
    from sgl_kernel import awq_dequantize  # AWQ反量化
elif _is_cpu and _is_cpu_amx_available:  # CPU AMX平台
    pass
elif _is_hip:  # HIP平台
    from sglang.srt.layers.quantization.awq.awq_triton import (
        awq_dequantize_triton as awq_dequantize,  # AWQ Triton反量化
    )
else:  # 其他平台
    pass

logger = logging.getLogger(__name__)  # 获取当前模块日志器


class LongcatFlashMLP(nn.Module):
    """Longcat-Flash密集MLP层"""

    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        reduce_results: bool = False,  # 是否归约结果
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控和上投影合并
            hidden_size,
            [intermediate_size] * 2,  # 两个中间层大小
            bias=False,  # 无偏置
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(  # 下投影
            intermediate_size,
            hidden_size,
            bias=False,  # 无偏置
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":  # 仅支持silu激活
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU激活与乘法融合函数

    def forward(
        self,
        x,  # 输入张量
    ):
        """MLP前向传播：门控上投影 -> SiLU激活 -> 下投影"""
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影
        x = self.act_fn(gate_up)  # SiLU激活和门控乘法
        x, _ = self.down_proj(x)  # 下投影
        return x  # 返回MLP输出


class LongcatFlashRouter(nn.Module):
    """Longcat-Flash MoE路由器，计算每个token到各专家的logits"""

    def __init__(
        self,
        config,  # 模型配置
        zero_expert_num=0,  # 零专家数
        rounter_params_dtype=torch.float32,  # 路由器参数数据类型
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.n_routed_experts = config.n_routed_experts  # 路由专家数
        self.n_routed_experts = self.n_routed_experts + zero_expert_num  # 加上零专家数
        self.rounter_params_dtype = rounter_params_dtype  # 路由器参数数据类型
        self.classifier = ReplicatedLinear(  # 分类器（复制线性层）
            config.hidden_size,
            self.n_routed_experts,  # 输出维度等于专家数
            bias=config.router_bias,  # 是否使用偏置
            params_dtype=rounter_params_dtype,
            quant_config=None,  # 路由器不量化
            prefix=add_prefix("classifier", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(  # 专家分数修正偏置
            torch.zeros((self.n_routed_experts), dtype=rounter_params_dtype)
        )

    def forward(self, hidden_states):
        """路由器前向传播：计算路由logits"""
        logits, _ = self.classifier(hidden_states.to(self.rounter_params_dtype))  # 分类得到logits
        return logits  # 返回路由logits


class LongcatFlashMoE(nn.Module):
    """Longcat-Flash混合专家层，包含路由器、Top-K选择、零专家和专家网络"""

    def __init__(
        self,
        config: LongcatFlashConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.layer_id = layer_id  # 保存层ID
        self.routed_scaling_factor = config.routed_scaling_factor  # 路由缩放因子
        self.num_experts = config.n_routed_experts  # 专家数
        self.top_k = config.moe_topk  # Top-K值
        self.zero_expert_num = config.zero_expert_num  # 零专家数
        self.zero_expert_type = config.zero_expert_type  # 零专家类型

        if config.rounter_params_dtype == "float32":  # 路由器参数数据类型
            self.rounter_params_dtype = torch.float32
        else:  # BF16
            self.rounter_params_dtype = torch.bfloat16

        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小

        if self.tp_size > config.n_routed_experts:  # TP大小不能超过专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":  # 仅支持silu激活
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.router = LongcatFlashRouter(  # 路由器
            config=self.config,
            zero_expert_num=self.zero_expert_num,
            rounter_params_dtype=self.rounter_params_dtype,
            prefix=add_prefix("router", prefix),
        )

        self.topk = TopK(  # Top-K选择器
            top_k=self.top_k,
            renormalize=False,  # 不重新归一化
            use_grouped_topk=False,  # 不使用分组Top-K
            correction_bias=self.router.e_score_correction_bias.data,  # 修正偏置
            layer_id=layer_id,
        )
        self.topk.forward = self.topk.forward_native  # 使用原生Top-K前向

        self.experts = get_moe_impl_class(quant_config)(  # 专家网络实现
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE前向传播：路由 -> Top-K选择 -> 专家计算 -> 零专家 -> 缩放 -> 全归约"""
        num_tokens, hidden_dim = hidden_states.shape  # 获取形状
        hidden_states = hidden_states.view(-1, hidden_dim)  # 展平

        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(hidden_states)  # 计算路由logits
        topk_weights, topk_idx, _ = self.topk(  # Top-K选择
            hidden_states,
            router_logits,
        )
        if self.zero_expert_type is not None:  # 计算零专家结果
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_idx,
                expert_scales=topk_weights,
                num_experts=self.num_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )
        topk_output = StandardTopKOutput(topk_weights, topk_idx, _)  # 构建标准Top-K输出

        final_hidden_states = self.experts(hidden_states, topk_output)  # 专家计算
        final_hidden_states *= self.routed_scaling_factor  # 应用路由缩放

        if self.zero_expert_type is not None and hidden_states.shape[0] > 0:  # 添加零专家结果
            final_hidden_states += zero_expert_result.to(final_hidden_states.device)

        if self.tp_size > 1:  # 张量并行时执行全归约
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)  # 恢复形状并返回

    def get_moe_weights(self):
        """获取MoE专家的权重数据列表"""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]  # 排除修正偏置
            and filter_moe_weight_param_global_expert(  # 过滤全局专家权重
                name, x, self.experts.num_local_experts
            )
        ]


class LongcatFlashDecoderLayer(nn.Module):
    """Longcat-Flash解码器层，包含两个MLA注意力、两个密集MLP和一个MoE"""

    def __init__(
        self,
        config: LongcatFlashConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.layer_id = layer_id  # 保存层ID
        self.alt_stream = alt_stream  # 保存备用流
        self.self_attn = nn.ModuleList(  # 两个MLA注意力层
            [
                DeepseekV2AttentionMLA(
                    config=config,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=config.rope_parameters["rope_theta"],
                    rope_scaling=None,
                    max_position_embeddings=config.max_position_embeddings,
                    quant_config=(
                        None
                        if "self_attn" in getattr(config, "disable_quant_module", [])  # 禁用量化模块
                        else quant_config
                    ),
                    layer_id=layer_id * 2 + i,  # 层ID翻倍加偏移
                    reduce_results=False,  # 不自动归约
                    prefix=add_prefix(f"self_attn.{i}", prefix),
                    alt_stream=self.alt_stream,
                )
                for i in range(2)  # 两个注意力层
            ]
        )

        self.input_layernorm = nn.ModuleList(  # 两个输入层归一化
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(  # 两个注意力后归一化
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )

        self.mlps = nn.ModuleList(  # 两个密集MLP
            [
                LongcatFlashMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=(
                        None
                        if "mlps" in getattr(config, "disable_quant_module", [])  # 禁用量化模块
                        else quant_config
                    ),
                    prefix=add_prefix(f"mlps.{i}", prefix),
                )
                for i in range(2)  # 两个MLP
            ]
        )

        self.mlp = LongcatFlashMoE(  # MoE层
            layer_id=self.layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.attn_tp_size = get_attention_tp_size()  # 注意力TP大小
        self.attn_tp_rank = get_attention_tp_rank()  # 注意力TP秩

        self.mlp_layer_scatter_modes = [  # MLP层散射模式
            LayerScatterModes.init_new(
                layer_id=self.layer_id * 2 + i,
                num_layers=config.num_hidden_layers,
                is_layer_sparse=False,  # 非稀疏
                is_previous_layer_sparse=False,
                # TODO: Check if the following is correct.
                is_next_layer_sparse=False,
            )
            for i in range(2)  # 两个MLP散射模式
        ]
        self.mlp_layer_communicator = [  # MLP层通信器
            LayerCommunicator(
                layer_scatter_modes=self.mlp_layer_scatter_modes[i],
                input_layernorm=self.input_layernorm[i],
                post_attention_layernorm=self.post_attention_layernorm[i],
                qkv_latent_func=self.self_attn[i].prepare_qkv_latent,  # QKV潜在函数
            )
            for i in range(2)  # 两个MLP通信器
        ]

        self.moe_layer_scatter_modes = LayerScatterModes.init_new(  # MoE层散射模式
            layer_id=self.layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=True,  # 稀疏
            is_previous_layer_sparse=True,
            # TODO: Check if the following is correct.
            is_next_layer_sparse=True,
        )
        self.moe_layer_communicator = LayerCommunicator(  # MoE层通信器
            layer_scatter_modes=self.moe_layer_scatter_modes,
            input_layernorm=self.input_layernorm[0],
            post_attention_layernorm=self.post_attention_layernorm[0],
            qkv_latent_func=self.self_attn[0].prepare_qkv_latent,  # QKV潜在函数
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引
        hidden_states: torch.Tensor,  # 输入隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差连接
        zero_allocator: BumpAllocator,  # 零分配器
    ) -> torch.Tensor:
        """解码器层前向传播：第一个注意力 -> MoE -> 两个MLP+注意力 -> 合并"""
        # first_attn
        hidden_states, residual = self.moe_layer_communicator.prepare_attn(  # 准备注意力输入
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:  # 非空输入时计算第一个注意力
            hidden_states = self.self_attn[0](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        # moe
        hidden_states, residual = self.moe_layer_communicator.prepare_mlp(  # 准备MoE输入
            hidden_states, residual, forward_batch
        )
        moe_hidden_states = hidden_states.clone()  # 克隆用于MoE分支
        moe_residual = residual.clone()  # 克隆MoE残差
        moe_hidden_states = self.mlp(moe_hidden_states)  # MoE计算
        moe_hidden_states, moe_residual = self.moe_layer_communicator.postprocess_layer(  # MoE后处理
            moe_hidden_states, moe_residual, forward_batch
        )

        hidden_states, residual = self.forward_mlp(  # MLP分支计算
            hidden_states, positions, residual, forward_batch, zero_allocator
        )

        hidden_states = moe_hidden_states + hidden_states  # 合并MoE和MLP分支
        return hidden_states, residual  # 返回隐藏状态和残差

    def forward_mlp(
        self, hidden_states, positions, residual, forward_batch, zero_allocator
    ):
        """MLP分支前向传播：第一个MLP -> 第二个注意力 -> 第二个MLP"""
        # first_mlp
        hidden_states = self.mlps[0](hidden_states)  # 第一个MLP
        # TP all_reduce
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)  # 张量并行全归约

        # second_attn
        hidden_states, residual = self.mlp_layer_communicator[1].prepare_attn(  # 准备第二个注意力
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:  # 非空输入时计算第二个注意力
            hidden_states = self.self_attn[1](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        # second_mlp
        hidden_states, residual = self.mlp_layer_communicator[1].prepare_mlp(  # 准备第二个MLP
            hidden_states, residual, forward_batch
        )
        hidden_states = self.mlps[1](hidden_states)  # 第二个MLP
        # TP all_reduce
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)  # 张量并行全归约

        hidden_states, residual = self.mlp_layer_communicator[1].postprocess_layer(  # 后处理
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual  # 返回隐藏状态和残差


class LongcatFlashModel(nn.Module):
    """Longcat-Flash模型主体，包含嵌入层、解码器层和最终归一化"""

    fall_back_to_pt_during_load = False  # 加载权重时不回退到PyTorch

    def __init__(
        self,
        config: LongcatFlashConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.vocab_size = config.vocab_size  # 词表大小

        if config.use_ngram_embedding:  # 使用N-gram嵌入
            self.use_ngram_embedding = True
            self.embed_tokens = NgramEmbedding(  # N-gram嵌入层
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                over_embedding_m=config.ngram_embedding_m,  # M参数
                over_embedding_k=config.ngram_embedding_k,  # K参数
                over_embedding_n=config.ngram_embedding_n,  # N参数
            )
        else:  # 使用标准嵌入
            self.use_ngram_embedding = False
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )

        self.alt_stream = torch.cuda.Stream()  # 备用CUDA流
        self.layers = nn.ModuleList(  # 解码器层列表
            [
                LongcatFlashDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)  # 每层一个解码器
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化
        self.layers_to_capture = []  # 需要捕获的层列表

    def get_input_embeddings(self) -> torch.Tensor:
        """获取输入嵌入层"""
        return self.embed_tokens  # 返回嵌入层

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        """模型主体前向传播：嵌入 -> 解码器层 -> 归一化"""
        total_num_layers = len(self.layers)  # 总层数
        device = input_embeds.device if input_embeds is not None else input_ids.device  # 设备
        zero_allocator = BumpAllocator(  # 零分配器
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),  # 缓冲区大小
            dtype=torch.float32,
            device=device,
        )
        if input_embeds is None:  # 无预计算嵌入
            if self.use_ngram_embedding:  # 使用N-gram嵌入
                hidden_states = self.embed_tokens(input_ids, forward_batch)
            else:  # 使用标准嵌入
                hidden_states = self.embed_tokens(input_ids)
        else:  # 使用预计算嵌入
            hidden_states = input_embeds

        residual = None  # 初始无残差

        aux_hidden_states = []  # 辅助隐藏状态列表
        for i in range(total_num_layers):  # 遍历所有层
            if i in self.layers_to_capture:  # 捕获指定层的隐藏状态
                aux_hidden_states.append(hidden_states + residual)
            with get_global_expert_distribution_recorder().with_current_layer(i):  # 记录专家分布
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual, zero_allocator
                )

        if hidden_states.shape[0] != 0:  # 非空时应用最终归一化
            if residual is None:  # 无残差时直接归一化
                hidden_states = self.norm(hidden_states)
            else:  # 有残差时融合归一化
                hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:  # 无辅助状态
            return hidden_states

        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助状态


class LongcatFlashForCausalLM(nn.Module):
    """Longcat-Flash因果语言模型，包含模型主体和语言模型头"""

    # for quark model load
    packed_modules_mapping = {}  # 打包模块映射（Quark模型）

    def __init__(
        self,
        config: LongcatFlashConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        # for quark model load
        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        self.fuse_qkv_a_proj = (  # 是否融合q_a和kv_a投影
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:  # 添加融合映射
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # TP大小
        self.quant_config = quant_config  # 保存量化配置
        self.model = LongcatFlashModel(  # 模型主体
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.use_ngram_embedding = config.use_ngram_embedding  # 是否使用N-gram嵌入
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否启用DP LM头
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.model.embed_tokens  # 返回嵌入层

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入（可选）
    ) -> torch.Tensor:
        """因果语言模型前向传播：模型主体 -> logits处理"""
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)  # 模型前向

        aux_hidden_states = None  # 辅助隐藏状态
        if self.capture_aux_hidden_states:  # 捕获辅助状态
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(  # 返回logits
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def post_load_weights(self, weight_names=None):
        """权重加载后处理：分解kv_b_proj权重、处理量化缩放"""

        # Perform post-processing after loading weights
        if weight_names is None:  # 未指定权重名则处理所有层
            layer_ids = range(self.config.num_hidden_layers)
        else:  # 仅处理包含kv_b_proj的层
            layer_ids = set()
            for name in weight_names:
                if "kv_b_proj" in name:  # 包含kv_b_proj的权重
                    layer_id = int(name.split(".")[2])  # 提取层ID
                    if layer_id < self.config.num_hidden_layers:
                        layer_ids.add(layer_id)

        for layer_id in layer_ids:  # 遍历需要处理的层
            for i in range(2):  # 每层有两个注意力
                self_attn = self.model.layers[layer_id].self_attn[i]
                if hasattr(self_attn.kv_b_proj, "qweight"):  # AWQ量化
                    # AWQ compatible
                    if _is_cuda or _is_hip:  # CUDA或HIP平台
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                        ).T
                    else:  # 其他平台
                        w = awq_dequantize(
                            self_attn.kv_b_proj.qweight,
                            self_attn.kv_b_proj.scales,
                            self_attn.kv_b_proj.qzeros,
                            0,
                            0,
                            0,
                        ).T
                else:  # 非量化权重
                    w = self_attn.kv_b_proj.weight
                use_deep_gemm_bmm = False  # 是否使用DeepGEMM BMM

                if w.dtype in (  # FP8权重处理
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    if (  # 块量化
                        hasattr(self.quant_config, "weight_block_size")
                        and self.quant_config.weight_block_size is not None
                    ):
                        weight_block_size = self.quant_config.weight_block_size
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        if _is_fp8_fnuz:  # FNUZ格式归一化
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                input_scale=None,
                            )
                        else:  # 标准FP8格式
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv

                        if (  # CUDA + 128x128块大小
                            _is_cuda
                            and weight_block_size[0] == 128
                            and weight_block_size[1] == 128
                        ):
                            if (  # DeepGEMM BMM
                                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                                and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL
                                and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                            ):
                                block_scale = weight_scale  # 块缩放
                                use_deep_gemm_bmm = True
                            else:  # 块量化反量化
                                w = block_quant_dequant(
                                    weight,
                                    weight_scale,
                                    weight_block_size,
                                    torch.bfloat16,
                                )
                        else:  # 其他块大小
                            w, scale = block_quant_to_tensor_quant(
                                weight, weight_scale, weight_block_size
                            )
                            self_attn.w_scale = scale
                    else:  # 通道量化
                        if _is_fp8_fnuz:  # FNUZ格式归一化
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale,
                                input_scale=None,
                            )
                        else:  # 标准FP8格式
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale

                        w, scale = channel_quant_to_tensor_quant(weight, weight_scale)  # 通道量化到张量
                        self_attn.w_scale = scale

                if w.dtype == torch.int8:  # INT8权重处理
                    if hasattr(self.quant_config, "weight_block_size"):  # 块级INT8
                        # block-wise int8 need it
                        weight_block_size = self.quant_config.weight_block_size
                        if weight_block_size is not None:
                            assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv
                            w = int8_block_dequant(  # INT8块反量化
                                weight, weight_scale, weight_block_size
                            ).to(torch.bfloat16)
                    else:  # 通道级INT8
                        # channel-wise int8 need it
                        w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                            torch.bfloat16
                        )

                w_kc, w_vc = w.unflatten(  # 分离k和v分量
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                if not use_deep_gemm_bmm:  # 非DeepGEMM BMM模式
                    self_attn.w_kc = bind_or_assign(  # 绑定k权重
                        self_attn.w_kc,
                        w_kc.transpose(1, 2).contiguous().transpose(1, 2),
                    )
                    self_attn.w_vc = bind_or_assign(  # 绑定v权重
                        self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
                    )
                    if (  # 绑定权重缩放
                        hasattr(self_attn.kv_b_proj, "weight_scale")
                        and self_attn.w_scale is None
                    ):
                        self_attn.w_scale = bind_or_assign(
                            self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                        )
                        if _is_hip:  # HIP平台缩放2倍
                            self_attn.w_scale *= 2.0
                else:  # DeepGEMM BMM模式
                    num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]  # K分块数
                    num_tiles_n = self_attn.v_head_dim // weight_block_size[0]  # N分块数
                    ws_kc, ws_vc = block_scale.unflatten(  # 分离k和v缩放
                        0, (-1, (num_tiles_k + num_tiles_n))
                    ).split([num_tiles_k, num_tiles_n], dim=1)
                    self_attn.w_scale_k = bind_or_assign(  # 绑定k缩放
                        self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                    )
                    self_attn.w_scale_v = bind_or_assign(  # 绑定v缩放
                        self_attn.w_scale_v, ws_vc.contiguous()
                    )
                    self_attn.w_kc = bind_or_assign(  # 绑定k权重
                        self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                    )
                    self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())  # 绑定v权重
                    self_attn.use_deep_gemm_bmm = True  # 启用DeepGEMM BMM

                if self.config.mla_scale_q_lora:  # 缩放Q LoRA归一化权重
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank
                    ) ** 0.5
                if self.config.mla_scale_kv_lora:  # 缩放KV LoRA归一化权重
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.kv_lora_rank
                    ) ** 0.5

        # TODO(linguoyuan) EPMoE not support DEEPGEMM_BLACKWELL, DeepEP needs to be supported in the future
        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 = False  # 禁用DeepGEMM UE8M0缩放

        if should_deepgemm_weight_requant_ue8m0(  # 需要DeepGEMM权重重量化
            weight_block_size=getattr(self.quant_config, "weight_block_size", None)
        ):
            self._weight_requant_ue8m0()  # 执行重量化

    def _weight_requant_ue8m0(self):
        """将权重重量化为UE8M0格式以配合DeepGEMM"""
        weight_block_size = self.quant_config.weight_block_size  # 权重块大小

        for layer_id in range(self.config.num_hidden_layers):  # 遍历所有层
            layer = self.model.layers[layer_id]
            for i in range(2):  # 每层两个注意力
                self_attn = layer.self_attn[i]
                module_list = [  # 需要重量化的模块列表
                    self_attn.kv_b_proj,
                    self_attn.o_proj,
                ]

                if self.config.q_lora_rank is not None:  # 有LoRA时
                    module_list.append(self_attn.fused_qkv_a_proj_with_mqa)
                    module_list.append(self_attn.q_b_proj)
                else:  # 无LoRA时
                    module_list.append(self_attn.kv_a_proj_with_mqa)
                    module_list.append(self_attn.q_proj)

                for module in module_list:  # 重量化注意力模块
                    if hasattr(module, "weight_scale_inv"):
                        requant_weight_ue8m0_inplace(
                            module.weight, module.weight_scale_inv, weight_block_size
                        )

                mlp = layer.mlps[i]  # MLP模块
                assert isinstance(mlp, LongcatFlashMLP)
                for module in [  # 重量化MLP模块
                    mlp.gate_up_proj,
                    mlp.down_proj,
                ]:
                    if hasattr(module, "weight_scale_inv"):
                        requant_weight_ue8m0_inplace(
                            module.weight, module.weight_scale_inv, weight_block_size
                        )

        for layer_id in range(self.config.num_hidden_layers):  # 重量化DeepEP MoE专家
            experts = layer.mlp.experts
            if isinstance(experts, DeepEPMoE):  # DeepEP MoE
                for w in [
                    (experts.w13_weight, experts.w13_weight_scale_inv),
                    (experts.w2_weight, experts.w2_weight_scale_inv),
                ]:
                    requant_weight_ue8m0_inplace(w[0], w[1], weight_block_size)  # 重量化

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，支持堆叠参数、专家参数、QKV融合和异步加载"""

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
            num_experts=self.config.n_routed_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (  # 是否融合QKV a投影
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None  # 缓存的a投影权重
        with concurrent.futures.ThreadPoolExecutor() as executor:  # 线程池
            futures = []  # 异步任务列表
            params_dict = dict(self.named_parameters())  # 参数字典
            weight_names = []  # 权重名列表
            for name, loaded_weight in weights:  # 遍历所有权重
                use_async_loading = should_async_load(loaded_weight)  # 是否异步加载
                if "mtp" in name:  # 跳过MTP权重
                    continue
                if self.use_ngram_embedding:  # N-gram嵌入权重处理
                    if ".embed_tokens." in name:  # 词嵌入权重
                        name = "model.embed_tokens.word_embeder.weight"
                    if ".ngram_embeddings" in name:  # N-gram权重
                        self.model.embed_tokens.load_weight(None, name, loaded_weight)
                        continue
                weight_names.append(name)  # 记录权重名
                if "rotary_emb.inv_freq" in name:  # 跳过旋转频率
                    continue
                for param_name, weight_name, shard_id in stacked_params_mapping:  # 处理堆叠参数
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:  # 名称不包含权重名则跳过
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:  # 跳过专家参数
                        continue
                    name = name.replace(weight_name, param_name)  # 替换为参数名
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                        continue
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    maybe_executor_submit(  # 提交加载任务
                        executor=executor,
                        futures=futures,
                        use_async=use_async_loading,
                        func=weight_loader,
                        func_args=(param, loaded_weight, shard_id),
                    )
                    break
                else:  # 处理专家参数
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:  # 名称不包含权重名则跳过
                            continue
                        name = name.replace(weight_name, param_name)  # 替换为参数名
                        param = params_dict[name]  # 获取参数
                        weight_loader = param.weight_loader  # 获取权重加载器
                        maybe_executor_submit(  # 提交加载任务
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, name),
                            func_kwargs={
                                "shard_id": shard_id,
                                "expert_id": expert_id,
                            },
                        )
                        break
                    else:  # 处理常规权重
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:  # 跳过GPTQ额外偏置
                            continue
                        if fuse_qkv_a_proj and (  # QKV a投影融合
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = loaded_weight  # 缓存a投影权重
                            q_a_proj_name = (  # q_a投影名
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (  # kv_a投影名
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (  # 两个a投影都已缓存
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]  # 获取q_a权重
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]  # 获取kv_a权重
                                cat_dim = 0  # 拼接维度
                                if self.quant_config is not None and (  # AWQ量化时沿维度1拼接
                                    self.quant_config.get_name() == "awq"
                                    or self.quant_config.get_name() == "awq_marlin"
                                    or self.quant_config.get_name() == "moe_wna16"
                                ):
                                    cat_dim = 1
                                fused_weight = torch.cat(  # 拼接融合权重
                                    [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                )
                                param_name = (  # 融合参数名
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                param = params_dict[param_name]  # 获取参数

                                weight_loader = getattr(  # 获取权重加载器
                                    param, "weight_loader", default_weight_loader
                                )
                                maybe_executor_submit(  # 提交融合权重加载任务
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, fused_weight),
                                )
                                cached_a_proj.pop(q_a_proj_name)  # 移除已加载的缓存
                                cached_a_proj.pop(kv_a_proj_name)
                        else:  # 常规权重加载
                            if (  # modelopt的KV缩放重命名
                                "k_scale" in name or "v_scale" in name
                            ) and name not in params_dict:
                                # modelopt attn kv scale is named differently
                                for scale in ["k_scale", "v_scale"]:
                                    if scale in name:
                                        name = name.replace(
                                            f"{scale[0]}_proj", "attn_mqa"
                                        )
                                        break
                            if name not in params_dict:  # 参数不存在
                                # modelopt ckpt contains not needed weights for MTP module:
                                # model.decoder.self_attn.attn_mqa.v_scale and
                                # model.decoder.self_attn.attn_mqa.k_scale
                                logger.warning(f"{name} not found in params_dict.")  # 发出警告
                                continue
                            param = params_dict[name]  # 获取参数
                            weight_loader = getattr(  # 获取权重加载器
                                param, "weight_loader", default_weight_loader
                            )
                            maybe_executor_submit(  # 提交加载任务
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(param, loaded_weight),
                            )

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):  # 等待所有任务完成
                future.result()

        self.post_load_weights(weight_names=weight_names)  # 权重加载后处理

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

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置的模型配置"""
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,  # 层数
            num_logical_experts=config.n_routed_experts,  # 逻辑专家数
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        """设置EAGLE3需要捕获隐藏状态的层"""
        if layer_ids is None:  # 未指定层ID
            self.capture_aux_hidden_states = True  # 启用辅助状态捕获
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]  # 默认捕获层
        else:  # 指定了层ID
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 偏移1


EntryClass = [LongcatFlashForCausalLM]  # 入口类列表
