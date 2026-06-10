# GLM-4.7-Flash MoE模型推理实现
# 本文件实现了GLM-4.7-Flash（轻量版MoE）模型的推理逻辑，
# 采用MLA（多潜在注意力）机制和DeepseekV2风格的注意力，
# 包含MLP、门控、稀疏MoE块、解码器层、模型和因果LM等核心组件，
# 支持张量并行、专家并行、流水线并行和量化。

# Copyright 2026-2027 SGLang Team
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

"""Inference-only GLM-4.7-Flash model compatible with HuggingFace weights."""  # 仅推理的GLM-4.7-Flash模型，兼容HuggingFace权重

import logging  # 导入日志模块
import re  # 导入正则表达式模块
from typing import Iterable, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入函数式神经网络接口
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags  # 导入单批次重叠标志
from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo  # 导入可能的双批次重叠前向传播
from sglang.srt.distributed import (  # 导入分布式通信模块
    get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    parallel_state,  # 并行状态
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入NCCL分配器
    use_symmetric_memory,  # 使用对称内存
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入全局专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置模型配置
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo  # 导入专家位置调度信息
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活函数
from sglang.srt.layers.communicator import (  # 导入层通信器
    LayerCommunicator,  # 层通信器
    LayerScatterModes,  # 层散射模式
    enable_moe_dense_fully_dp,  # 启用MoE密集全数据并行
    get_attn_tp_context,  # 获取注意力TP上下文
)
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力
    is_allocation_symmetric,  # 判断分配是否对称
    is_dp_attention_enabled,  # 判断是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化层
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear  # 导入并行线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe import (  # 导入MoE相关模块
    get_moe_a2a_backend,  # 获取MoE全互连后端
    should_skip_post_experts_all_reduce,  # 判断是否跳过专家后的全归约
    should_use_flashinfer_cutlass_moe_fp4_allgather,  # 判断是否使用FlashInfer Cutlass MoE FP4全收集
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类获取函数
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE Triton层
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod  # 导入KT EP包装方法
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat  # 导入TopK选择和输出格式
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert  # 导入MoE权重参数过滤函数
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入层
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 导入CUDA图捕获模式判断
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (  # 导入Deepseek权重加载器混入
    DeepseekV2WeightLoaderMixin,  # DeepseekV2权重加载器混入类
)
from sglang.srt.models.deepseek_common.utils import _is_cuda, _use_aiter  # 导入平台检测工具
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA  # 导入DeepseekV2 MLA注意力
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils import (  # 导入工具函数
    BumpAllocator,  # 凸起分配器
    LazyValue,  # 延迟值
    add_prefix,  # 添加前缀
    is_non_idle_and_non_empty,  # 判断是否非空闲且非空
    log_info_on_rank0,  # 在rank0上记录信息
    make_layers,  # 创建层
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入RoPE配置获取函数

logger = logging.getLogger(__name__)  # 创建日志记录器


class Glm4MoeLiteMLP(nn.Module):
    """GLM-4.7-Flash MoE模型的密集MLP层，使用SiLU激活函数。"""

    def __init__(  # 初始化方法
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 隐藏层激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        reduce_results: bool = True,  # 是否归约结果
        prefix: str = "",  # 参数前缀
        tp_rank: Optional[int] = None,  # 张量并行rank，可选
        tp_size: Optional[int] = None,  # 张量并行大小，可选
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.tp_size = tp_size  # 张量并行大小

        self.gate_up_proj = MergedColumnParallelLinear(  # 合并的gate-up并行线性层
            hidden_size,  # 输入大小
            [intermediate_size] * 2,  # 输出大小为中间大小的两倍（gate和up）
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 添加前缀
            tp_rank=tp_rank,  # 张量并行rank
            tp_size=tp_size,  # 张量并行大小
        )
        self.down_proj = RowParallelLinear(  # 行并行下投影层
            intermediate_size,  # 输入大小
            hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            reduce_results=reduce_results,  # 是否归约结果
            prefix=add_prefix("down_proj", prefix),  # 添加前缀
            tp_rank=tp_rank,  # 张量并行rank
            tp_size=tp_size,  # 张量并行大小
        )
        if hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."  # 不支持的激活函数，目前仅支持silu
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数

    def forward(  # 前向传播方法
        self,
        x,  # 输入张量
        forward_batch=None,  # 前向批次，可选
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:  # 如果单卡且输入为空
            return x  # 直接返回

        gate_up, _ = self.gate_up_proj(x)  # 通过gate_up投影
        x = self.act_fn(gate_up)  # SiLU激活并乘以up部分
        x, _ = self.down_proj(  # 通过下投影
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter  # 是否跳过全归约
        )
        return x  # 返回输出


class Glm4MoeLiteGate(nn.Module):
    """GLM-4.7-Flash MoE门控网络，使用FP32精度计算路由logits。"""

    def __init__(  # 初始化方法
        self,
        config,  # 模型配置
        prefix: str = "",  # 参数前缀
        is_nextn: bool = False,  # 是否为Next-N推测解码层
    ):
        super().__init__()  # 调用父类初始化
        self.is_nextn = is_nextn  # 是否为Next-N层
        self.weight = nn.Parameter(  # 门控权重参数
            torch.empty((config.n_routed_experts, config.hidden_size))  # 形状为[路由专家数, 隐藏大小]
        )
        self.e_score_correction_bias = nn.Parameter(  # 专家分数校正偏置
            torch.empty((config.n_routed_experts), dtype=torch.float32)  # 形状为[路由专家数]，FP32精度
        )
        # GLM requires FP32 gate projection; cache to avoid per-forward cast.  # GLM要求FP32门控投影；缓存以避免每次前向传播时转换
        # FIXME: if gate weight is updated at runtime (e.g. expert rebalancing), _weight_fp32 must be invalidated.  # FIXME：如果运行时更新门控权重（如专家重平衡），必须使_weight_fp32失效
        self.register_buffer("_weight_fp32", None, persistent=False)  # 注册FP32权重缓存缓冲区

    def forward(self, hidden_states):  # 前向传播方法
        if self._weight_fp32 is None:  # 如果FP32缓存为空
            self._weight_fp32 = self.weight.data.to(torch.float32)  # 将权重转为FP32并缓存
        logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)  # FP32精度的线性投影
        return logits  # 返回路由logits


class Glm4MoeLiteSparseMoeBlock(nn.Module):
    """GLM-4.7-Flash稀疏MoE块，包含路由门控、专家网络和共享专家。"""

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流，可选
        is_nextn: bool = False,  # 是否为Next-N层
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.routed_scaling_factor = config.routed_scaling_factor  # 路由专家缩放因子
        self.n_shared_experts = config.n_shared_experts  # 共享专家数量
        self.num_fused_shared_experts = (  # 融合共享专家数量
            0  # 0
            if get_global_server_args().disable_shared_experts_fusion  # 如果禁用共享专家融合
            else config.n_shared_experts  # 否则等于共享专家数量
        )
        self.config = config  # 保存配置
        self.layer_id = layer_id  # 层ID
        self.alt_stream = alt_stream  # 备用CUDA流
        self.is_nextn = is_nextn  # 是否为Next-N层

        if self.tp_size > config.n_routed_experts:  # 如果TP大小大于路由专家数
            raise ValueError(  # 抛出值错误
                f"Tensor parallel size {self.tp_size} is greater than "  # 张量并行大小大于
                f"the number of experts {config.n_routed_experts}."  # 专家数量
            )

        if config.hidden_act != "silu":  # 如果激活函数不是silu
            raise ValueError(  # 抛出值错误
                f"Unsupported activation: {config.hidden_act}. "  # 不支持的激活函数
                "Only silu is supported for now."  # 目前仅支持silu
            )

        self.gate = Glm4MoeLiteGate(  # MoE门控网络
            config=config, prefix=add_prefix("gate", prefix), is_nextn=is_nextn  # 传入配置、前缀和Next-N标志
        )

        self.experts = get_moe_impl_class(quant_config)(  # 获取MoE实现类并实例化
            num_experts=config.n_routed_experts  # 总专家数=路由专家数
            + self.num_fused_shared_experts  # + 融合共享专家数
            + get_global_server_args().ep_num_redundant_experts,  # + 冗余专家数
            num_fused_shared_experts=self.num_fused_shared_experts,  # 融合共享专家数
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,  # top-k=每token专家数+融合共享专家数
            hidden_size=config.hidden_size,  # 隐藏大小
            intermediate_size=config.moe_intermediate_size,  # MoE中间层大小
            layer_id=self.layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            routed_scaling_factor=self.routed_scaling_factor,  # 路由缩放因子
            prefix=add_prefix("experts", prefix),  # 添加前缀
        )

        self.topk = TopK(  # Top-K选择器
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,  # top-k值
            layer_id=self.layer_id,  # 层ID
            renormalize=config.norm_topk_prob,  # 是否重归一化top-k概率
            use_grouped_topk=True,  # 使用分组top-k
            num_expert_group=config.n_group,  # 专家分组数
            num_fused_shared_experts=self.num_fused_shared_experts,  # 融合共享专家数
            topk_group=config.topk_group,  # 每组选择的专家数
            correction_bias=self.gate.e_score_correction_bias,  # 校正偏置
            quant_config=quant_config,  # 量化配置
            routed_scaling_factor=self.routed_scaling_factor,  # 路由缩放因子
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,  # 是否在输出上应用路由缩放因子
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized  # 某些FP4 MoE后端要求跳过输出格式，但MTP层未量化
            # and requires the output format to be standard. We use quant_config to determine the output format.  # 且要求标准输出格式。使用quant_config确定输出格式
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,  # 无量化时使用标准输出格式
        )

        self.shared_experts_is_int8 = False  # 共享专家是否为INT8
        self.shared_experts_is_fp8 = False  # 共享专家是否为FP8
        self.shared_experts_weight_block_size = None  # 共享专家权重块大小
        self._shared_expert_tp1 = False  # 共享专家是否使用TP=1
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:  # 如果有共享专家且未融合
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts  # 共享专家中间层大小
            # disable tp for shared experts when enable deepep moe, or with fp4 allgather  # 启用deepep MoE或FP4全收集时禁用共享专家的TP
            self.shared_experts = Glm4MoeLiteMLP(  # 共享专家MLP
                hidden_size=config.hidden_size,  # 隐藏大小
                intermediate_size=intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                reduce_results=False,  # 不归约结果
                prefix=add_prefix("shared_experts", prefix),  # 添加前缀
                **(  # 条件参数
                    dict(tp_rank=0, tp_size=1)  # TP=1
                    if get_moe_a2a_backend().is_deepep()  # 如果是DeepEP
                    or get_moe_a2a_backend().is_mooncake()  # 或Mooncake
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()  # 或使用FP4全收集
                    else {}  # 否则不添加额外参数
                ),
            )
            is_packed_weight = hasattr(  # 检查是否为打包权重
                self.shared_experts.gate_up_proj.quant_method, "quant_config"  # 量化方法是否有quant_config属性
            )
            self.shared_experts_is_int8 = (  # 判断共享专家是否为INT8
                not is_packed_weight  # 非打包权重
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8  # 且权重为INT8
            )
            self.shared_experts_is_fp8 = (  # 判断共享专家是否为FP8
                not is_packed_weight  # 非打包权重
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn  # 且权重为FP8
            )

        self.top_k = config.num_experts_per_tok  # 每token选择的专家数

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():  # 如果使用DeepEP或Mooncake后端
            # TODO: we will support tp < ep in the future  # TODO：未来将支持TP < EP
            self.ep_size = get_moe_expert_parallel_world_size()  # 专家并行大小
            self.num_experts = (  # 专家总数
                config.n_routed_experts  # 路由专家数
                + get_global_server_args().ep_num_redundant_experts  # + 冗余专家数
            )
            self.renormalize = config.norm_topk_prob  # 是否重归一化
            self.topk_group = config.topk_group  # 每组top-k
            self.num_expert_group = config.n_group  # 专家分组数
            self.correction_bias = (  # 校正偏置
                self.gate.e_score_correction_bias.data  # 门控校正偏置数据
                if self.gate.e_score_correction_bias is not None  # 如果存在
                else None  # 否则为None
            )

        self._enable_a2a_moe = (  # 是否启用全互连MoE
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()  # DeepEP或Mooncake
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()  # 是否在SBO内融合共享专家

    def get_moe_weights(self):  # 获取MoE权重
        return [  # 返回权重列表
            x.data  # 权重数据
            for name, x in self.experts.named_parameters()  # 遍历专家参数
            if name not in ["correction_bias"]  # 排除校正偏置
            and filter_moe_weight_param_global_expert(  # 过滤全局专家权重参数
                name, x, self.experts.num_local_experts  # 传入名称、参数和本地专家数
            )
        ]

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次，可选
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        if not self._enable_a2a_moe:  # 如果未启用全互连MoE
            if (  # 如果
                self.alt_stream is not None  # 有备用CUDA流
                and self.num_fused_shared_experts == 0  # 未融合共享专家
                and hidden_states.shape[0] > 0  # 有有效token
                and get_is_capture_mode()  # 处于CUDA图捕获模式
            ):
                return self.forward_normal_dual_stream(  # 使用双流前向传播
                    hidden_states, should_allreduce_fusion, use_reduce_scatter  # 传入参数
                )
            else:  # 否则
                return self.forward_normal(  # 使用普通前向传播
                    hidden_states, should_allreduce_fusion, use_reduce_scatter  # 传入参数
                )
        else:  # 启用全互连MoE
            return self.forward_deepep(hidden_states, forward_batch)  # 使用DeepEP前向传播

    def forward_normal_dual_stream(  # 双流普通前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()  # 获取当前CUDA流
        self.alt_stream.wait_stream(current_stream)  # 等待当前流完成
        shared_output = self._forward_shared_experts(hidden_states)  # 计算共享专家输出

        with torch.cuda.stream(self.alt_stream):  # 在备用流上执行
            # router_logits: (num_tokens, n_experts)  # 路由logits：(token数, 专家数)
            router_logits = self.gate(hidden_states)  # 计算路由logits
            topk_output = self.topk(hidden_states, router_logits)  # Top-K选择
            final_hidden_states = self.experts(hidden_states, topk_output)  # 路由专家计算
            if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):  # 非CUDA或使用KT EP包装
                final_hidden_states *= self.routed_scaling_factor  # 乘以路由缩放因子

        current_stream.wait_stream(self.alt_stream)  # 等待备用流完成
        final_hidden_states += shared_output  # 加上共享专家输出
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 如果TP>1且不跳过全归约
            is_tp_path=True,  # TP路径
            use_reduce_scatter=use_reduce_scatter,  # 是否使用reduce-scatter
            should_allreduce_fusion=should_allreduce_fusion,  # 是否融合全归约
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 张量并行全归约
        return final_hidden_states  # 返回最终隐藏状态

    def forward_normal(  # 普通前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:  # 如果有有效token
            shared_output = self._forward_shared_experts(hidden_states)  # 计算共享专家输出
            # router_logits: (num_tokens, n_experts)  # 路由logits：(token数, 专家数)
            router_logits = self.gate(hidden_states)  # 计算路由logits
            topk_output = self.topk(hidden_states, router_logits)  # Top-K选择
        else:  # 无有效token
            shared_output = None  # 共享输出为None
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空Top-K输出

        final_hidden_states = self.experts(hidden_states, topk_output)  # 路由专家计算
        if not _is_cuda and not _use_aiter:  # 非CUDA且非AITER
            final_hidden_states *= self.routed_scaling_factor  # 乘以路由缩放因子
        if shared_output is not None:  # 如果有共享专家输出
            with use_symmetric_memory(  # 使用对称内存
                parallel_state.get_tp_group(), disabled=not is_allocation_symmetric()  # 分配不对称时禁用
            ):
                final_hidden_states_out = torch.empty_like(final_hidden_states)  # 分配输出缓冲区
            torch.add(final_hidden_states, shared_output, out=final_hidden_states_out)  # 原地加法
            final_hidden_states = final_hidden_states_out  # 更新最终隐藏状态
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 如果TP>1且不跳过全归约
            is_tp_path=True,  # TP路径
            use_reduce_scatter=use_reduce_scatter,  # 是否使用reduce-scatter
            should_allreduce_fusion=should_allreduce_fusion,  # 是否融合全归约
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 张量并行全归约
        return final_hidden_states  # 返回最终隐藏状态

    def forward_deepep(  # DeepEP前向传播方法
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch  # 隐藏状态、前向批次
    ) -> torch.Tensor:
        shared_output = None  # 初始化共享输出
        if hidden_states.shape[0] > 0:  # 如果有有效token
            # router_logits: (num_tokens, n_experts)  # 路由logits：(token数, 专家数)
            router_logits = self.gate(hidden_states)  # 计算路由logits
            shared_output = self._forward_shared_experts(hidden_states)  # 计算共享专家输出
            topk_output = self.topk(  # Top-K选择
                hidden_states,  # 隐藏状态
                router_logits,  # 路由logits
                num_token_non_padded=forward_batch.num_token_non_padded,  # 非填充token数
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(  # 专家位置调度信息
                    layer_id=self.layer_id,  # 层ID
                ),
            )
        else:  # 无有效token
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空Top-K输出

        final_hidden_states = self.experts(  # 路由专家计算
            hidden_states=hidden_states,  # 隐藏状态
            topk_output=topk_output,  # Top-K输出
        )

        if shared_output is not None:  # 如果有共享专家输出
            x = shared_output  # 取共享输出
            if self.experts.should_fuse_routed_scaling_factor_in_topk:  # 如果在top-k中融合路由缩放因子
                x.add_(final_hidden_states)  # 直接加
            else:  # 否则
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)  # 带缩放因子的加法
            final_hidden_states = x  # 更新最终隐藏状态
        else:  # 无共享专家输出
            if not self.experts.should_fuse_routed_scaling_factor_in_topk:  # 如果未在top-k中融合路由缩放因子
                final_hidden_states *= self.routed_scaling_factor  # 乘以路由缩放因子

        return final_hidden_states  # 返回最终隐藏状态

    def _forward_shared_experts(self, hidden_states: torch.Tensor):  # 共享专家前向传播
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):  # 有token且未融合共享专家
            return self.shared_experts(hidden_states)  # 通过共享专家
        else:  # 否则
            return None  # 返回None

    def op_gate(self, state):  # 操作：门控计算
        if is_non_idle_and_non_empty(  # 如果非空闲且非空
            state.forward_batch.forward_mode, state.hidden_states_mlp_input  # 前向模式和MLP输入
        ):
            # router_logits: (num_tokens, n_experts)  # 路由logits：(token数, 专家数)
            state.router_logits = self.gate(state.hidden_states_mlp_input)  # 计算路由logits
        else:  # 否则
            state.router_logits = None  # 路由logits为None

    def op_shared_experts(self, state):  # 操作：共享专家计算
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")  # 弹出MLP输入
        if (self.num_fused_shared_experts == 0) and is_non_idle_and_non_empty(  # 未融合共享专家且非空闲非空
            state.forward_batch.forward_mode, hidden_states_mlp_input  # 前向模式和输入
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)  # 计算共享专家输出
        else:  # 否则
            state.shared_output = None  # 共享输出为None

    def op_select_experts(self, state):  # 操作：专家选择
        router_logits = state.pop("router_logits")  # 弹出路由logits
        hidden_states = state.hidden_states_mlp_input  # 获取MLP输入隐藏状态

        if router_logits is not None:  # 如果有路由logits
            with get_global_expert_distribution_recorder().with_current_layer(  # 记录当前层
                self.layer_id  # 层ID
            ):
                state.topk_output = self.topk(  # Top-K选择
                    hidden_states=hidden_states,  # 隐藏状态
                    router_logits=router_logits,  # 路由logits
                    num_token_non_padded=state.forward_batch.num_token_non_padded,  # 非填充token数
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(  # 专家位置调度信息
                        layer_id=self.layer_id,  # 层ID
                    ),
                )
        else:  # 无路由logits
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空Top-K输出

    def op_dispatch_a(self, state):  # 操作：调度阶段A
        if self.ep_size > 1:  # 如果EP>1
            self.experts.dispatcher.dispatch_a(  # 执行调度A
                hidden_states=state.hidden_states_mlp_input,  # 隐藏状态
                topk_output=state.pop("topk_output"),  # 弹出Top-K输出
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )

    def op_dispatch_b(self, state):  # 操作：调度阶段B
        if self.ep_size > 1:  # 如果EP>1
            with get_global_expert_distribution_recorder().with_current_layer(  # 记录当前层
                self.layer_id  # 层ID
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(  # 执行调度B
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
                )

    def op_experts(self, state):  # 操作：专家计算
        state.combine_input = self.experts.run_moe_core(  # 运行MoE核心计算
            dispatch_output=state.dispatch_output,  # 调度输出
        )

    def op_combine_a(self, state):  # 操作：合并阶段A
        if self.ep_size > 1:  # 如果EP>1
            self.experts.dispatcher.combine_a(  # 执行合并A
                combine_input=state.pop("combine_input"),  # 弹出合并输入
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )
            state.pop("dispatch_output")  # 弹出调度输出

    def op_combine_b(self, state):  # 操作：合并阶段B
        if self.ep_size > 1:  # 如果EP>1
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(  # 执行合并B
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )

    def op_output(self, state):  # 操作：输出处理
        final_hidden_states = state.pop("hidden_states_after_combine")  # 弹出合并后的隐藏状态

        if get_moe_a2a_backend().is_mori():  # 如果使用Mori后端
            num_tokens = state.pop("num_tokens")  # 弹出token数
            final_hidden_states = final_hidden_states[:num_tokens]  # 截取有效token

        if (shared_output := state.pop("shared_output")) is not None:  # 如果有共享输出
            x = shared_output  # 取共享输出
            if _use_aiter:  # 如果使用AITER
                x.add_(final_hidden_states)  # 直接加
            else:  # 否则
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)  # 带缩放因子加法
            final_hidden_states = x  # 更新最终隐藏状态
        elif _use_aiter:  # 使用AITER但无共享输出
            # fused in aiter_biased_grouped_topk so we can skip here  # 已在aiter_biased_grouped_topk中融合，此处跳过
            pass  # 跳过
        else:  # 非AITER且无共享输出
            final_hidden_states *= self.routed_scaling_factor  # 乘以路由缩放因子

        state.hidden_states_mlp_output = final_hidden_states  # 保存MLP输出


class Glm4MoeLiteDecoderLayer(nn.Module):
    """GLM-4.7-Flash MoE解码器层，包含MLA注意力和MoE/密集MLP。"""

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        is_nextn: bool = False,  # 是否为Next-N层
        prefix: str = "",  # 参数前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 备用CUDA流，可选
    ) -> None:

        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.config = config  # 保存配置
        rope_theta, rope_scaling = get_rope_config(config)  # 获取RoPE配置
        max_position_embeddings = getattr(config, "max_position_embeddings", 202752)  # 最大位置嵌入数
        self.layer_id = layer_id  # 层ID
        self.is_nextn = is_nextn  # 是否为Next-N层

        self.self_attn = DeepseekV2AttentionMLA(  # DeepseekV2风格的多潜在注意力
            config=config,  # 配置
            hidden_size=config.hidden_size,  # 隐藏大小
            num_heads=config.num_attention_heads,  # 注意力头数
            qk_nope_head_dim=config.qk_nope_head_dim,  # QK非旋转头维度
            qk_rope_head_dim=config.qk_rope_head_dim,  # QK旋转头维度
            v_head_dim=config.v_head_dim,  # V头维度
            q_lora_rank=config.q_lora_rank,  # Q LoRA秩
            kv_lora_rank=config.kv_lora_rank,  # KV LoRA秩
            rope_theta=rope_theta,  # RoPE theta
            rope_scaling=rope_scaling,  # RoPE缩放
            max_position_embeddings=max_position_embeddings,  # 最大位置嵌入
            quant_config=quant_config,  # 量化配置
            reduce_results=False,  # 不归约结果
            layer_id=layer_id,  # 层ID
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)  # 判断是否为稀疏层
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)  # 前一层是否稀疏
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)  # 后一层是否稀疏

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 初始化层散射模式
            layer_id=layer_id,  # 层ID
            num_layers=1 if is_nextn else config.num_hidden_layers,  # 总层数
            is_layer_sparse=self.is_layer_sparse,  # 是否稀疏层
            is_previous_layer_sparse=is_previous_layer_sparse,  # 前一层是否稀疏
            is_next_layer_sparse=is_next_layer_sparse,  # 后一层是否稀疏
        )

        if self.is_layer_sparse:  # 如果是稀疏层
            self.mlp = Glm4MoeLiteSparseMoeBlock(  # 使用稀疏MoE块
                config=config,  # 配置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 添加前缀
                layer_id=self.layer_id,  # 层ID
                alt_stream=alt_stream,  # 备用CUDA流
                is_nextn=is_nextn,  # 是否为Next-N层
            )
        else:  # 密集层
            if enable_moe_dense_fully_dp():  # 如果启用MoE密集全DP
                mlp_tp_rank, mlp_tp_size = 0, 1  # TP=1
            else:  # 否则
                mlp_tp_rank, mlp_tp_size = None, None  # 使用默认TP
            self.mlp = Glm4MoeLiteMLP(  # 使用密集MLP
                hidden_size=config.hidden_size,  # 隐藏大小
                intermediate_size=config.intermediate_size,  # 中间层大小
                hidden_act=config.hidden_act,  # 激活函数
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("mlp", prefix),  # 添加前缀
                tp_rank=mlp_tp_rank,  # TP rank
                tp_size=mlp_tp_size,  # TP大小
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后归一化
            config.hidden_size, eps=config.rms_norm_eps  # 隐藏大小和epsilon
        )

        self._gfx95_quant_format = self._detect_gfx95_quant_format()  # 检测GFX95量化格式

        self.layer_communicator = LayerCommunicator(  # 层通信器
            layer_scatter_modes=self.layer_scatter_modes,  # 层散射模式
            input_layernorm=self.input_layernorm,  # 输入归一化
            post_attention_layernorm=self.post_attention_layernorm,  # 注意力后归一化
            allow_reduce_scatter=True,  # 允许reduce-scatter
            is_last_layer=(  # 是否为最后一层
                is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)  # Next-N层或最后一个隐藏层
            ),
            qkv_latent_func=self.self_attn.prepare_qkv_latent,  # QKV潜在函数
        )

    def _detect_gfx95_quant_format(self) -> str:  # 检测GFX95量化格式
        from sglang.srt.models.deepseek_common.utils import _is_gfx95_supported  # 导入GFX95支持检测

        if not _is_gfx95_supported:  # 如果不支持GFX95
            return ""  # 返回空字符串
        weight = getattr(  # 获取fused_qkv_a_proj_with_mqa的权重
            getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None), "weight", None  # 尝试获取权重属性
        )
        if weight is None:  # 如果权重不存在
            return ""  # 返回空字符串
        if weight.dtype == torch.uint8:  # 如果是uint8类型
            return "mxfp4"  # 返回MXFP4格式
        if weight.dtype == getattr(torch, "float8_e4m3fn", None):  # 如果是FP8类型
            return "fp8"  # 返回FP8格式
        return ""  # 返回空字符串

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:  # 判断是否为稀疏层
        return is_nextn or (  # Next-N层或
            self.config.n_routed_experts is not None  # 配置了路由专家
            and layer_id >= self.config.first_k_dense_replace  # 层ID大于等于首个替换层
            and layer_id % self.config.moe_layer_freq == 0  # 且层ID是MoE层频率的倍数
        )

    def forward(  # 前向传播方法
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差，可选
        zero_allocator: BumpAllocator,  # 零分配器
    ) -> torch.Tensor:
        hidden_states, residual = self.layer_communicator.prepare_attn(  # 准备注意力输入
            hidden_states,  # 隐藏状态
            residual,  # 残差
            forward_batch,  # 前向批次
            getattr(self, "_gfx95_quant_format", ""),  # GFX95量化格式
        )

        hidden_states = self.self_attn(  # 通过MLA注意力层
            positions=positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
            zero_allocator=zero_allocator,  # 零分配器
            layer_scatter_modes=self.layer_scatter_modes,  # 层散射模式
        )
        if isinstance(hidden_states, tuple):  # 如果输出是元组
            hidden_states = hidden_states[0]  # 取第一个元素
        get_attn_tp_context().clear_attn_inputs()  # 清除注意力TP上下文的输入

        hidden_states, residual = self.layer_communicator.prepare_mlp(  # 准备MLP输入
            hidden_states, residual, forward_batch  # 传入隐藏状态、残差和批次
        )

        should_allreduce_fusion = (  # 是否融合全归约
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(  # 判断是否与下一层融合
                forward_batch  # 前向批次
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.  # 对于带填充的DP，可使用reduce-scatter代替all-reduce
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(  # 判断是否使用reduce-scatter
            forward_batch  # 前向批次
        )

        hidden_states = self.mlp(  # 通过MLP/MoE层
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter  # 传入参数
        )

        if should_allreduce_fusion:  # 如果需要融合全归约
            hidden_states._sglang_needs_allreduce_fusion = True  # 标记需要融合
        else:  # 否则
            hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理层
                hidden_states, residual, forward_batch  # 传入隐藏状态、残差和批次
            )

        return hidden_states, residual  # 返回隐藏状态和残差

    def op_comm_prepare_attn(  # 操作：准备注意力通信
        self,
        state,  # 状态对象
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差，可选
        zero_allocator: BumpAllocator,  # 零分配器
        tbo_subbatch_index: Optional[int] = None,  # TBO子批次索引，可选
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (  # 通信后注意力前隐藏状态和残差
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)  # 准备注意力
        )
        if get_moe_a2a_backend().is_mori():  # 如果使用Mori后端
            state.num_tokens = hidden_states.shape[0]  # 记录token数
        state.update(  # 更新状态
            dict(
                forward_batch=forward_batch,  # 前向批次
                positions=positions,  # 位置编码
                zero_allocator=zero_allocator,  # 零分配器
                tbo_subbatch_index=tbo_subbatch_index,  # TBO子批次索引
            )
        )

    def op_comm_prepare_mlp(self, state):  # 操作：准备MLP通信
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (  # MLP输入和残差
            self.layer_communicator.prepare_mlp(  # 准备MLP
                state.pop("hidden_states_after_attn"),  # 弹出注意力后的隐藏状态
                state.pop("residual_after_input_ln"),  # 弹出输入归一化后的残差
                state.forward_batch,  # 前向批次
            )
        )

    def op_comm_postprocess_layer(self, state):  # 操作：层后处理通信
        hidden_states, residual = self.layer_communicator.postprocess_layer(  # 后处理层
            state.pop("hidden_states_mlp_output"),  # 弹出MLP输出
            state.pop("residual_after_comm_pre_mlp"),  # 弹出MLP前的残差
            state.forward_batch,  # 前向批次
        )

        output = dict(  # 输出字典
            positions=state.positions,  # 位置编码
            hidden_states=hidden_states,  # 隐藏状态
            residual=residual,  # 残差
            forward_batch=state.forward_batch,  # 前向批次
            zero_allocator=state.zero_allocator,  # 零分配器
            tbo_subbatch_index=state.tbo_subbatch_index,  # TBO子批次索引
        )

        state.clear(  # 清除状态
            expect_keys={  # 期望保留的键
                "positions",  # 位置
                "forward_batch",  # 前向批次
                "zero_allocator",  # 零分配器
                "tbo_subbatch_index",  # TBO子批次索引
            }
        )
        return output  # 返回输出


class Glm4MoeLiteModel(nn.Module):
    """GLM-4.7-Flash MoE模型主体，包含词嵌入、解码器层堆叠和归一化。"""
    fall_back_to_pt_during_load = False  # 加载时不回退到PyTorch

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.padding_id = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.first_k_dense_replace = config.first_k_dense_replace  # 前k个密集层
        self.pp_group = get_pp_group()  # 获取流水线并行组

        if self.pp_group.is_first_rank:  # 如果是第一个rank
            self.embed_tokens = VocabParallelEmbedding(  # 词表并行嵌入
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏大小
                use_attn_tp_group=is_dp_attention_enabled(),  # 是否使用注意力TP组
            )
        else:  # 否则
            self.embed_tokens = PPMissingLayer()  # 流水线并行缺失层占位

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None  # CUDA平台创建备用流
        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,  # 层数
            lambda idx, prefix: Glm4MoeLiteDecoderLayer(  # 解码器层工厂函数
                config=config,  # 配置
                layer_id=idx,  # 层ID
                quant_config=quant_config,  # 量化配置
                prefix=prefix,  # 前缀
                alt_stream=self.alt_stream,  # 备用流
            ),
            pp_rank=self.pp_group.rank_in_group,  # 流水线并行rank
            pp_size=self.pp_group.world_size,  # 流水线并行世界大小
            prefix=add_prefix("layers", prefix),  # 添加前缀
        )
        if self.pp_group.is_last_rank:  # 如果是最后一个rank
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层
        else:  # 否则
            self.norm = PPMissingLayer(return_tuple=True)  # 流水线并行缺失层
        self.layers_to_capture = []  # 需要捕获辅助隐藏状态的层列表

    def get_input_embeddings(self) -> torch.Tensor:  # 获取输入嵌入
        return self.embed_tokens  # 返回词嵌入层

    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，可选
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer  # 本rank负责的总层数
        if self.pp_group.is_first_rank:  # 如果是第一个rank
            if input_embeds is None:  # 如果没有提供嵌入
                hidden_states = self.embed_tokens(input_ids)  # 通过词嵌入层
            else:  # 否则
                hidden_states = input_embeds  # 直接使用输入嵌入
            residual = None  # 初始化残差为None
        else:  # 非第一个rank
            assert pp_proxy_tensors is not None  # 确保有代理张量
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理张量获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理张量获取残差
        device = hidden_states.device  # 获取设备
        zero_allocator = BumpAllocator(  # 创建凸起分配器，用于MLA零分配
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),  # 缓冲区大小
            dtype=torch.float32,  # 数据类型
            device=device,  # 设备
        )

        normal_start_layer = self.start_layer  # 正常前向起始层
        normal_end_layer = self.end_layer  # 正常前向结束层
        if forward_batch.can_run_tbo:  # 如果可以运行双批次重叠
            if (  # 如果
                self.first_k_dense_replace > normal_start_layer  # 首个稀疏层在起始层之后
                and self.first_k_dense_replace < normal_end_layer  # 首个稀疏层在结束层之前
            ):
                normal_end_layer = self.first_k_dense_replace  # 正常前向只到首个稀疏层
            elif self.first_k_dense_replace < normal_start_layer:  # 首个稀疏层在起始层之前
                normal_end_layer = normal_start_layer = 0  # 全部用TBO
        aux_hidden_states = []  # 辅助隐藏状态列表
        for i in range(normal_start_layer, normal_end_layer):  # 遍历正常前向层
            with get_global_expert_distribution_recorder().with_current_layer(i):  # 记录当前层
                if i in self.layers_to_capture:  # 如果需要捕获此层
                    aux_hidden_states.append(hidden_states + residual)  # 添加辅助隐藏状态
                layer = self.layers[i]  # 获取层
                hidden_states, residual = layer(  # 前向传播
                    positions,  # 位置编码
                    hidden_states,  # 隐藏状态
                    forward_batch,  # 前向批次
                    residual,  # 残差
                    zero_allocator,  # 零分配器
                )

        if normal_end_layer != self.end_layer:  # 如果有TBO层
            hidden_states, residual = model_forward_maybe_tbo(  # 双批次重叠前向
                layers=self.layers[normal_end_layer : self.end_layer],  # TBO层
                enable_tbo=True,  # 启用TBO
                positions=positions,  # 位置编码
                forward_batch=forward_batch,  # 前向批次
                hidden_states=hidden_states,  # 隐藏状态
                residual=residual,  # 残差
                input_data_scatter_mode=self.layers[  # 输入数据散射模式
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,  # 零分配器
            )

        if not self.pp_group.is_last_rank:  # 如果不是最后一个rank
            return PPProxyTensors(  # 返回流水线代理张量
                {
                    "hidden_states": hidden_states,  # 隐藏状态
                    "residual": residual,  # 残差
                }
            )
        else:  # 最后一个rank
            if not forward_batch.forward_mode.is_idle():  # 如果不是空闲模式
                if residual is None:  # 如果没有残差
                    hidden_states = self.norm(hidden_states)  # 仅归一化
                else:  # 有残差
                    hidden_states, _ = self.norm(hidden_states, residual)  # 带残差的归一化

        if len(aux_hidden_states) == 0:  # 如果没有辅助隐藏状态
            return hidden_states  # 返回隐藏状态
        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助隐藏状态


class Glm4MoeLiteForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    """GLM-4.7-Flash MoE因果语言模型，整合模型主体和语言模型头。"""
    # for quark model load  # 用于quark模型加载
    packed_modules_mapping = {}  # 打包模块映射

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 预训练配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        config.moe_layer_freq = 1  # 设置MoE层频率为1（每层都是MoE）
        self.config = config  # 保存配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.quant_config = quant_config  # 量化配置
        self.pp_group = get_pp_group()  # 流水线并行组
        self.determine_num_fused_shared_experts("Glm4MoeLiteForCausalLM")  # 确定融合共享专家数
        self.model = Glm4MoeLiteModel(  # 创建模型主体
            config, quant_config, prefix=add_prefix("model", prefix)  # 传入配置和前缀
        )
        self.lm_head = ParallelLMHead(  # 并行语言模型头
            config.vocab_size,  # 词表大小
            config.hidden_size,  # 隐藏大小
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("lm_head", prefix),  # 添加前缀
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力TP组
        )
        self.logits_processor = LogitsProcessor(config)  # logits处理器

        self._routed_experts_weights_of_layer = LazyValue(  # 延迟计算的路由专家权重
            lambda: {  # 字典推导
                layer_id: layer.mlp.get_moe_weights()  # 获取每层MoE权重
                for layer_id, layer in enumerate(self.model.layers)  # 遍历所有层
                if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock)  # 仅MoE层
            }
        )
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    @property  # 属性装饰器
    def routed_experts_weights_of_layer(self):  # 路由专家权重属性
        return self._routed_experts_weights_of_layer.value  # 返回延迟计算的值

    def determine_num_fused_shared_experts(  # 确定融合共享专家数量
        self, architecture: str = "Glm4MoeLiteForCausalLM"  # 架构名称
    ):
        self.num_fused_shared_experts = 0  # 初始化融合共享专家数为0
        if get_global_server_args().disable_shared_experts_fusion:  # 如果禁用共享专家融合
            return  # 返回

        disable_reason = None  # 禁用原因
        if (  # 如果
            not _is_cuda  # 非CUDA
            or torch.cuda.get_device_capability("cuda") < (8, 0)  # 计算能力<8.0
            or self.config.architectures[0] != architecture  # 架构不匹配
            or self.config.n_shared_experts != 1  # 共享专家数不为1
        ):
            disable_reason = "Only GLM-4.5 or GLM-4.6 on NV-platform with capability >= 80 can use shared experts fusion optimization."  # 仅NV平台能力>=80的GLM-4.5/4.6可使用共享专家融合
        elif get_moe_expert_parallel_world_size() > 1:  # 专家并行>1
            disable_reason = "GLM-4.5 or GLM-4.6 cannot use shared experts fusion optimization under expert parallelism."  # 专家并行下不能使用共享专家融合

        if disable_reason is not None:  # 如果有禁用原因
            get_global_server_args().disable_shared_experts_fusion = True  # 全局禁用共享专家融合
            self.num_fused_shared_experts = 0  # 融合共享专家数为0
            log_info_on_rank0(  # 在rank0上记录
                logger,  # 日志记录器
                f"{disable_reason} Shared experts fusion optimization is disabled.",  # 禁用原因和提示
            )
            return  # 返回

        self.num_fused_shared_experts = self.config.n_shared_experts  # 设置融合共享专家数

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入
        return self.model.embed_tokens  # 返回模型词嵌入层

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线代理张量，可选
    ) -> torch.Tensor:
        with get_attn_tp_context().maybe_input_scattered(forward_batch):  # 可能输入散射的注意力TP上下文
            hidden_states = self.model(  # 通过模型主体
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors  # 传入所有参数
            )
        aux_hidden_states = None  # 辅助隐藏状态
        if self.capture_aux_hidden_states:  # 如果需要捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 解包

        if self.pp_group.is_last_rank:  # 如果是最后一个rank
            return self.logits_processor(  # 通过logits处理器
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states  # 传入参数
            )
        else:  # 否则
            return hidden_states  # 返回隐藏状态

    @property  # 属性装饰器
    def start_layer(self):  # 起始层属性
        return self.model.start_layer  # 返回模型的起始层

    @property  # 属性装饰器
    def end_layer(self):  # 结束层属性
        return self.model.end_layer  # 返回模型的结束层

    def get_embed_and_head(self):  # 获取嵌入和语言模型头权重
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回词嵌入权重和LM头权重

    def set_embed_and_head(self, embed, head):  # 设置嵌入和语言模型头权重
        del self.model.embed_tokens.weight  # 删除旧词嵌入权重
        del self.lm_head.weight  # 删除旧LM头权重
        self.model.embed_tokens.weight = embed  # 设置新词嵌入权重
        self.lm_head.weight = head  # 设置新LM头权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA

    @classmethod  # 类方法
    def get_model_config_for_expert_location(cls, config):  # 获取专家位置的模型配置
        return ModelConfigForExpertLocation(  # 返回专家位置模型配置
            num_layers=config.num_hidden_layers,  # 隐藏层数
            num_logical_experts=config.n_routed_experts,  # 逻辑专家数
            num_groups=config.n_group,  # 分组数
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):  # 设置EAGLE3捕获层
        if not self.pp_group.is_last_rank:  # 如果不是最后一个rank
            return  # 返回

        if layer_ids is None:  # 如果未指定层ID
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            num_layers = self.config.num_hidden_layers  # 总层数
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]  # 默认捕获第2、中间、倒数第3层
        else:  # 指定了层ID
            self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
            # TODO (Qiaolin-Yu): check if other draft models need similar layer id  # TODO：检查其他草稿模型是否需要类似的层ID
            # adjustment  # 调整
            if layer_ids and layer_ids[0] == 1:  # 如果第一个层ID为1
                self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 所有层ID加1
            else:  # 否则
                self.model.layers_to_capture = list(layer_ids)  # 直接使用指定层ID

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):  # 设置DFLASH捕获层
        if not self.pp_group.is_last_rank:  # 如果不是最后一个rank
            return  # 返回

        if layer_ids is None:  # 如果未指定层ID
            raise ValueError(  # 抛出值错误
                "DFLASH requires explicit layer_ids for aux hidden capture."  # DFLASH需要明确的层ID来捕获辅助隐藏状态
            )

        self.capture_aux_hidden_states = True  # 启用辅助隐藏状态捕获
        self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 所有层ID加1

    def load_weights(  # 加载权重方法
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],  # 权重迭代器
        is_nextn=False,  # 是否为Next-N模式
        params_dict=None,  # 参数字典，可选
    ):
        if is_nextn:  # 如果是Next-N模式
            if hasattr(self.config, "num_nextn_predict_layers"):  # 如果配置中有Next-N预测层数
                num_nextn_layers = self.config.num_nextn_predict_layers  # 获取层数
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"  # 仅支持1个Next-N层
                # compatible with old design  # 兼容旧设计
                nextn_layer_id = (  # Next-N层ID
                    0  # 如果只有1个隐藏层
                    if self.config.num_hidden_layers == 1  # 单层
                    else self.config.num_hidden_layers  # 否则为最后一层
                )
            else:  # 配置中没有Next-N预测层数
                raise ValueError("num_nextn_predict_layers is not in the config")  # 抛出值错误

        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # Q投影
            ("qkv_proj", "k_proj", "k"),  # K投影
            ("qkv_proj", "v_proj", "v"),  # V投影
            ("gate_up_proj", "gate_proj", 0),  # gate投影
            ("gate_up_proj", "up_proj", 1),  # up投影
        ]

        if self.num_fused_shared_experts > 0:  # 如果有融合共享专家
            assert self.num_fused_shared_experts == 1  # 确保只有1个

            def iter_weights_with_fused_shared_experts(  # 迭代融合共享专家权重的生成器
                weights: Iterable[Tuple[str, torch.Tensor]],  # 权重迭代器
            ) -> Iterable[Tuple[str, torch.Tensor]]:

                pattern = re.compile(  # 编译正则表达式
                    r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(.+)$"  # 匹配shared_experts权重
                )
                for name, weight in weights:  # 遍历权重
                    match = pattern.match(name)  # 匹配名称
                    if match:  # 如果匹配
                        layer_id = int(match.group(1))  # 获取层ID
                        suffix = match.group(2)  # 获取后缀
                        name = f"model.layers.{layer_id}.mlp.experts.{self.config.n_routed_experts}.{suffix}"  # 重映射到专家层
                    yield name, weight  # 生成权重

            weights = iter_weights_with_fused_shared_experts(weights)  # 使用生成器

        # Params for weights, fp8 weight scales, fp8 activation scales  # 权重、FP8权重缩放、FP8激活缩放的参数
        # (param_name, weight_name, expert_id, shard_id)  # (参数名, 权重名, 专家ID, 分片ID)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj",  # 检查点gate投影名
            ckpt_down_proj_name="down_proj",  # 检查点down投影名
            ckpt_up_proj_name="up_proj",  # 检查点up投影名
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,  # 总专家数
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None  # 当q_lora_rank不为None时，沿输出维度融合q_a_proj和kv_a_proj_with_mqa
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (  # 判断是否需要融合QKV_a投影
            self.config.q_lora_rank is not None  # q_lora_rank不为None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None  # 缓存a投影权重

        if is_nextn:  # Next-N模式
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"  # Next-N层前缀
            nextn_spec_weight_names = [  # Next-N特定权重名
                "shared_head.norm",  # 共享头归一化
                "eh_proj",  # 嵌入隐藏投影
                "enorm",  # 嵌入归一化
                "hnorm",  # 隐藏归一化
            ]
        else:  # 非Next-N模式
            nextn_layer_prefix = None  # 无前缀
            nextn_spec_weight_names = []  # 空列表

        if params_dict is None:  # 如果未提供参数字典
            params_dict = dict(self.named_parameters())  # 获取模型参数字典

        weight_names = []  # 权重名称列表
        for name, loaded_weight in weights:  # 遍历权重
            weight_names.append(name)  # 记录权重名

            if not is_nextn:  # 非Next-N模式
                if hasattr(self.config, "num_nextn_predict_layers"):  # 如果配置中有Next-N预测层数
                    num_nextn_layers = self.config.num_nextn_predict_layers  # 获取层数
                    if num_nextn_layers > 0 and name.startswith("model.layers"):  # 有Next-N层且名称以model.layers开头
                        name_list = name.split(".")  # 分割名称
                        if (  # 如果
                            len(name_list) >= 3  # 至少3层
                            and int(name_list[2]) >= self.config.num_hidden_layers  # 层ID>=隐藏层数
                        ):
                            continue  # 跳过Next-N层
            else:  # Next-N模式
                if nextn_layer_prefix and not name.startswith(nextn_layer_prefix):  # 非Next-N层权重
                    continue  # 跳过

                if nextn_layer_prefix is not None:  # mtp  # MTP（多token预测）
                    # Use shared head and embed weights from target model  # 使用目标模型的共享头和嵌入权重
                    if "shared_head.head" in name or "embed_tokens" in name:  # 共享头或嵌入
                        continue  # 跳过

                    is_decoder = True  # 标记为解码器权重
                    # For nextn specific weights  # Next-N特定权重
                    for weight_name in nextn_spec_weight_names:  # 遍历特定权重名
                        if weight_name in name:  # 如果名称包含
                            name = name.replace(nextn_layer_prefix, "model")  # 替换前缀
                            is_decoder = False  # 非解码器权重
                            break  # 跳出
                    # For decoder layer weights  # 解码器层权重
                    if is_decoder:  # 如果是解码器权重
                        name = name.replace(nextn_layer_prefix, "model.decoder")  # 替换为model.decoder

            if "rotary_emb.inv_freq" in name:  # 跳过旋转位置编码逆频率
                continue  # 跳过
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                # Skip non-stacked layers and experts (experts handled below).  # 跳过非堆叠层和专家（专家在下面处理）
                if weight_name not in name:  # 权重名不在名称中
                    continue  # 跳过
                # We have mlp.experts[0].gate_proj in the checkpoint.  # 检查点中有mlp.experts[0].gate_proj
                # Since we handle the experts below in expert_params_mapping,  # 因为在expert_params_mapping中处理专家
                # we need to skip here BEFORE we update the name, otherwise  # 需要在更新名称之前跳过，否则
                # name will be updated to mlp.experts[0].gate_up_proj, which  # 名称会更新为mlp.experts[0].gate_up_proj
                # will then be updated below in expert_params_mapping  # 然后在expert_params_mapping中再次更新
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.  # 变为mlp.experts[0].gate_gate_up_proj，导致加载失败
                if "mlp.experts" in name:  # 如果是专家权重
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:  # 偏置不在参数字典中
                    continue  # 跳过
                if name not in params_dict:  # 参数不在字典中
                    continue  # 跳过

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重分片
                break  # 跳出循环
            else:  # 非堆叠参数
                # Track if this is an expert weight to enable early skipping  # 跟踪是否为专家权重以启用早期跳过
                is_expert_weight = False  # 初始化专家权重标志

                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 权重名不在名称中
                        continue  # 跳过

                    # Mark as expert weight regardless of whether we can process it  # 无论是否能处理，都标记为专家权重
                    is_expert_weight = True  # 标记为专家权重

                    name = name.replace(weight_name, param_name)  # 替换权重名
                    if name not in params_dict:  # 参数不在字典中
                        # Expert weight not on this rank, will be skipped below  # 专家权重不在本rank，将跳过
                        continue  # 跳过

                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载权重
                        param,  # 参数
                        loaded_weight,  # 加载的权重
                        name,  # 名称
                        shard_id=shard_id,  # 分片ID
                        expert_id=expert_id,  # 专家ID
                    )
                    break  # 跳出循环
                else:  # 非专家权重
                    if is_expert_weight:  # 如果是专家权重但不在本rank
                        # This is an expert weight but not mapped to this rank, skip all remaining processing  # 这是专家权重但未映射到本rank，跳过所有后续处理
                        continue  # 跳过

                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 偏置不在参数字典中
                        continue  # 跳过

                    # GLM NOTE: for MLA  # GLM注意：用于MLA
                    if fuse_qkv_a_proj and (  # 如果需要融合QKV_a投影
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name  # 且是q_a_proj或kv_a_proj_with_mqa
                    ):
                        cached_a_proj[name] = loaded_weight  # 缓存a投影权重
                        q_a_proj_name = (  # Q a投影名称
                            name  # 当前名称
                            if "q_a_proj" in name  # 如果包含q_a_proj
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")  # 否则替换
                        )
                        kv_a_proj_name = (  # KV a投影名称
                            name  # 当前名称
                            if "kv_a_proj_with_mqa" in name  # 如果包含kv_a_proj_with_mqa
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")  # 否则替换
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter  # 当q_a_proj和kv_a_proj_with_mqa都缓存后，加载融合权重到参数
                        if (  # 如果
                            q_a_proj_name in cached_a_proj  # Q a投影已缓存
                            and kv_a_proj_name in cached_a_proj  # KV a投影已缓存
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]  # 获取Q a投影权重
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]  # 获取KV a投影权重
                            fused_weight = torch.cat(  # 融合权重
                                [q_a_proj_weight, kv_a_proj_weight], dim=0  # 沿输出维度拼接
                            )
                            param_name = (  # 融合参数名
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")  # 替换Q a投影
                                if "q_a_proj" in name  # 如果包含q_a_proj
                                else name.replace(  # 否则替换KV a投影
                                    "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"  # 替换
                                )
                            )
                            if param_name not in params_dict:  # 如果融合参数不在字典中
                                continue  # 跳过
                            param = params_dict[param_name]  # 获取融合参数

                            weight_loader = getattr(  # 获取权重加载器
                                param, "weight_loader", default_weight_loader  # 默认使用default_weight_loader
                            )
                            weight_loader(param, fused_weight)  # 加载融合权重
                            cached_a_proj.pop(q_a_proj_name)  # 弹出Q a投影缓存
                            cached_a_proj.pop(kv_a_proj_name)  # 弹出KV a投影缓存
                    else:  # 非MLA投影权重
                        if (  # 如果
                            "k_scale" in name or "v_scale" in name  # 包含k_scale或v_scale
                        ) and name not in params_dict:  # 且不在参数字典中
                            # modelopt attn kv scale is named differently  # modelopt注意力KV缩放命名不同
                            if any(scale in name for scale in ["k_scale", "v_scale"]):  # 如果包含缩放名
                                name = name.replace("_proj", "attn_mqa")  # 替换_proj为attn_mqa
                            else:  # 否则
                                logger.warning(  # 记录警告
                                    f"Unknown scale found in checkpoint: {name}"  # 检查点中发现未知缩放
                                )

                    if name not in params_dict:  # 如果名称不在参数字典中
                        continue  # 跳过

                    if name in params_dict.keys():  # 如果名称在参数字典中
                        param = params_dict[name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader  # 默认使用default_weight_loader
                        )
                        weight_loader(param, loaded_weight)  # 加载权重
                    else:  # 否则
                        logger.warning(f"Parameter {name} not found in params_dict")  # 记录警告

        # DeepseekV2AttentionMLA.forward_* expects post_load_weights() to populate  # DeepseekV2AttentionMLA的forward_*需要post_load_weights()填充
        # per-layer packed weights like `w_kc`/`w_vc` (used during CUDA graph capture).  # 每层打包权重如w_kc/w_vc（CUDA图捕获时使用）
        # GLM-4.7-Flash configs not set `config.mla`, but this model always uses  # GLM-4.7-Flash配置未设置config.mla，但此模型始终使用
        # DeepseekV2AttentionMLA, so we must run the post-load processing.  # DeepseekV2AttentionMLA，因此必须运行后加载处理
        # Use weight_names=None to ensure we always process all layers. Some checkpoints /  # 使用weight_names=None确保始终处理所有层。某些检查点/
        # naming schemes may not include "kv_b_proj" in `weight_names`, but `w_kc`/`w_vc`  # 命名方案可能不包含"kv_b_proj"，但w_kc/w_vc
        # are still required by DeepseekV2AttentionMLA at runtime.  # 在运行时仍被DeepseekV2AttentionMLA需要
        self.post_load_weights(is_nextn=is_nextn, weight_names=None)  # 执行后加载权重处理


EntryClass = [Glm4MoeLiteForCausalLM]  # 入口类列表
