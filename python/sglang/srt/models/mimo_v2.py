# MiMo V2混合专家模型实现
# 该模块实现了MiMo V2模型，包括MoE（混合专家）层、注意力层、解码器层和因果语言模型
# 支持特性：张量并行、专家并行、流水线并行、DP注意力、滑动窗口注意力、TBO（两批次重叠）
# 多模态支持：视觉编码器（MiMoVisionTransformer）和音频编码器（MiMoAudioEncoder）
# 核心组件：MiMoV2MLP、MoEGate、MiMoV2MoE、MiMoV2Attention、MiMoV2DecoderLayer、
# MiMoV2Model、MiMoV2ForCausalLM

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

import logging  # 导入日志模块
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union  # 导入类型注解

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch函数式接口
from torch import nn  # 导入PyTorch神经网络模块

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo  # 导入TBO前向传播
from sglang.srt.configs.model_config import get_mimo_v2_fused_qkv_expected_tp_size  # 导入融合QKV期望TP大小
from sglang.srt.distributed import (  # 导入分布式工具
    get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
    tensor_model_parallel_all_reduce,  # 张量并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 导入专家位置模型配置
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo  # 导入专家位置分发信息
from sglang.srt.layers.activation import SiluAndMul  # 导入SiLU与乘法激活
from sglang.srt.layers.communicator import (  # 导入层通信器
    LayerCommunicator,  # 层通信器
    LayerScatterModes,  # 层散射模式
    ScatterMode,  # 散射模式
    enable_moe_dense_fully_dp,  # 启用MoE密集全DP
)
from sglang.srt.layers.dp_attention import (  # 导入DP注意力工具
    get_attention_tp_rank,  # 获取注意力TP秩
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 检查DP注意力是否启用
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化
from sglang.srt.layers.linear import (  # 导入并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.moe import (  # 导入MoE工具
    get_moe_a2a_backend,  # 获取MoE全互连后端
    get_moe_runner_backend,  # 获取MoE运行器后端
    should_skip_post_experts_all_reduce,  # 检查是否跳过专家后全归约
)
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class  # 导入DeepEP MoE层
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat  # 导入TopK选择器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入层工具
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行LM头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态token填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    kv_cache_scales_loader,  # KV缓存缩放加载器
)
from sglang.srt.models.mimo_audio import MiMoAudioEncoder, MiMoAudioEncoderConfig  # 导入MiMo音频编码器
from sglang.srt.models.mimo_vl import MiMoVisionTransformer, MiMoVLVisionConfig  # 导入MiMo视觉Transformer
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import (  # 导入工具函数
    LazyValue,  # 懒加载值
    add_prefix,  # 添加前缀
    is_non_idle_and_non_empty,  # 检查非空闲且非空
    make_layers,  # 创建层
)

MiMoV2Config = None  # MiMo V2配置占位，由外部设置

logger = logging.getLogger(__name__)  # 获取日志记录器


def load_mimo_v2_qkv_proj_weight(
    name, param, loaded_weight, expected_fused_tp_size: Optional[int] = None
):
    """加载MiMo V2融合QKV投影权重，支持分片和融合格式"""
    if loaded_weight.shape == param.shape:  # 形状匹配，直接加载
        # The checkpoint already stores this rank's qkv_proj shard.  # 检查点已存储该rank的qkv_proj分片
        default_weight_loader(param, loaded_weight)
        return

    if loaded_weight.ndim != param.ndim or loaded_weight.shape[1:] != param.shape[1:]:  # 维度不匹配
        raise ValueError(
            f"qkv_proj weight {name}: unexpected shape {tuple(loaded_weight.shape)}; "
            f"expected sharded {tuple(param.shape)}"
        )

    tp_size = get_attention_tp_size()  # 注意力TP大小
    tp_rank = get_attention_tp_rank()  # 注意力TP秩
    if expected_fused_tp_size is not None and tp_size != expected_fused_tp_size:  # TP大小不匹配
        raise ValueError(
            f"MiMoV2 fused qkv_proj checkpoint is TP={expected_fused_tp_size}-"
            f"interleaved; got attention tp_size={tp_size} while loading {name}."
        )

    fused_shape = (param.shape[0] * tp_size, *param.shape[1:])  # 融合后的形状
    if tuple(loaded_weight.shape) != fused_shape:  # 形状不匹配
        raise ValueError(
            f"qkv_proj weight {name}: unexpected shape {tuple(loaded_weight.shape)}; "
            f"expected fused {fused_shape} or sharded {tuple(param.shape)}"
        )

    default_weight_loader(param, loaded_weight.chunk(tp_size, dim=0)[tp_rank])  # 分片加载


class MiMoV2MLP(nn.Module):
    """MiMo V2 MLP层：门控上投影+SiLU激活+下投影"""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.tp_size = tp_size  # 张量并行大小

        self.gate_up_proj = MergedColumnParallelLinear(  # 门控+上投影合并线性层
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度：门控和上投影各一个
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(  # 下投影线性层
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            reduce_results=reduce_results,  # 是否归约结果
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":  # 检查激活函数
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()  # SiLU与乘法激活函数

    def forward(
        self,
        x,
        forward_batch: ForwardBatch = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        """MLP前向传播"""
        if (self.tp_size == 1) and x.shape[0] == 0:  # 空输入直接返回
            return x

        gate_up, _ = self.gate_up_proj(x)  # 门控上投影
        x = self.act_fn(gate_up)  # SiLU激活+门控
        x, _ = self.down_proj(  # 下投影
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


class MoEGate(nn.Module):
    """MoE门控网络：计算路由logits"""
    def __init__(
        self,
        config,
        quant_config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()  # 调用父类初始化
        self.is_nextn = is_nextn  # 是否是NextN层
        self.dtype = torch.float32  # 门控计算使用float32
        self.weight = nn.Parameter(  # 门控权重
            torch.empty((config.n_routed_experts, config.hidden_size), dtype=self.dtype)
        )
        if config.topk_method == "noaux_tc":  # 无辅助TC方法
            correction_bias_dtype = (  # 校正偏置数据类型
                torch.bfloat16
                if quant_config is not None
                and quant_config.get_name() == "modelopt_fp4"
                and get_moe_runner_backend().is_flashinfer_trtllm()
                else self.dtype
            )
            self.e_score_correction_bias = nn.Parameter(  # 专家分数校正偏置
                torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)
            )
        else:  # 不使用校正偏置
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        """门控前向传播：计算路由logits"""
        logits = F.linear(hidden_states.to(self.dtype), self.weight, None)  # 线性变换

        return logits


class MiMoV2MoE(nn.Module):
    """MiMo V2混合专家层：支持普通和DeepEP模式"""

    def __init__(
        self,
        config: MiMoV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行大小

        self.config = config  # 保存配置
        self.layer_id = layer_id  # 层ID

        if self.tp_size > config.n_routed_experts:  # TP大小不能超过专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":  # 检查激活函数
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(  # MoE门控
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
        )

        experts_type = get_moe_impl_class(quant_config)  # 获取专家实现类
        self.experts = experts_type(  # 专家层
            num_experts=config.n_routed_experts
            + get_global_server_args().ep_num_redundant_experts,  # 含冗余专家
            top_k=config.num_experts_per_tok,  # 每token选择专家数
            hidden_size=config.hidden_size,  # 隐藏维度
            intermediate_size=config.moe_intermediate_size,  # MoE中间维度
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=1.0,  # 路由缩放因子
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(  # TopK选择器
            top_k=config.num_experts_per_tok,  # TopK值
            renormalize=config.norm_topk_prob,  # 是否归一化
            use_grouped_topk=True,  # 使用分组TopK
            num_expert_group=config.n_group,  # 专家组数
            topk_group=config.topk_group,  # 每组TopK
            correction_bias=self.gate.e_score_correction_bias,  # 校正偏置
            quant_config=quant_config,
            routed_scaling_factor=1.0,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,  # 是否在TopK输出上应用路由缩放
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized  # 某些Fp4 MoE后端需要绕过输出格式，但MTP层未量化
            # and requires the output format to be standard. We use quant_config to determine the output format.  # 并且需要标准输出格式。使用quant_config确定输出格式。
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,  # 输出格式
        )

        # todo : implement tbo forward needed  # 待办：实现TBO前向传播
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():  # DeepEP或Mooncake后端
            # TODO: we will support tp < ep in the future  # 待办：未来将支持TP < EP
            self.ep_size = get_moe_expert_parallel_world_size()  # 专家并行大小
            self.num_experts = (  # 专家数（含冗余）
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob  # 是否归一化
            self.topk_group = config.topk_group  # 每组TopK
            self.num_expert_group = config.n_group  # 专家组数
            self.correction_bias = (  # 校正偏置
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (  # 是否启用全互连MoE
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        )

    def get_moe_weights(self):
        """获取MoE权重"""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        """MoE前向传播：根据后端选择普通或DeepEP模式"""
        if not self._enable_a2a_moe:  # 普通模式
            return self.forward_normal(
                hidden_states,
                should_allreduce_fusion,
                use_reduce_scatter,
            )
        else:  # DeepEP模式
            return self.forward_deepep(hidden_states, forward_batch)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        """普通MoE前向传播"""
        if hidden_states.shape[0] > 0:  # 有token
            # router_logits: (num_tokens, n_experts)  # 路由logits：(token数, 专家数)
            router_logits = self.gate(hidden_states)  # 计算路由logits
            topk_output = self.topk(hidden_states, router_logits)  # TopK选择
        else:  # 空输入
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空TopK输出

        final_hidden_states = self.experts(hidden_states, topk_output)  # 专家计算

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 需要全归约
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 张量并行全归约

        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """DeepEP MoE前向传播"""
        if hidden_states.shape[0] > 0:  # 有token
            # router_logits: (num_tokens, n_experts)  # 路由logits：(token数, 专家数)
            router_logits = self.gate(hidden_states)  # 计算路由logits
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,  # 非填充token数
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),  # 专家位置分发信息
            )
        else:  # 空输入
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空TopK输出

        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )  # 专家计算

        return final_hidden_states

    def op_gate(self, state):
        """TBO操作：计算路由logits"""
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):  # 非空闲且有数据
            # router_logits: (num_tokens, n_experts)  # 路由logits：(token数, 专家数)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:  # 空闲或无数据
            state.router_logits = None

    def op_select_experts(self, state):
        """TBO操作：选择专家"""
        router_logits = state.pop("router_logits")  # 获取路由logits
        hidden_states = state.hidden_states_mlp_input  # 获取隐藏状态
        if router_logits is not None:  # 有路由logits
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):  # 记录专家分布
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:  # 无路由logits
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        """TBO操作：分发阶段A"""
        if self.ep_size > 1:  # 多专家并行
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        """TBO操作：分发阶段B"""
        if self.ep_size > 1:  # 多专家并行
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):  # 记录专家分布
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        """TBO操作：专家计算"""
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        """TBO操作：合并阶段A"""
        if self.ep_size > 1:  # 多专家并行
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")  # 移除分发输出

    def op_combine_b(self, state):
        """TBO操作：合并阶段B"""
        if self.ep_size > 1:  # 多专家并行
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        """TBO操作：输出"""
        state.hidden_states_mlp_output = state.pop("hidden_states_after_combine")


class MiMoV2Attention(nn.Module):
    """MiMo V2注意力层：支持GQA、滑动窗口、部分旋转、注意力汇聚偏置"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        v_scale: Optional[float] = None,
        sliding_window_size: int = -1,  # if is -1 ,normal attention,else ,window attention  # -1为普通注意力，否则为窗口注意力
        attention_bias: bool = False,
        attention_sink_bias: bool = False,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        partial_rotary_factor: float = 1.0,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 隐藏维度

        attn_tp_rank = get_attention_tp_rank()  # 注意力TP秩
        attn_tp_size = get_attention_tp_size()  # 注意力TP大小

        self.total_num_heads = num_heads  # 总注意力头数
        assert self.total_num_heads % attn_tp_size == 0  # 头数必须能被TP大小整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 当前TP组的头数
        self.total_num_kv_heads = num_kv_heads  # 总KV头数
        if self.total_num_kv_heads >= attn_tp_size:  # KV头数>=TP大小
            # Number of KV heads is greater than TP size, so we partition  # KV头数大于TP大小，因此分区
            # the KV heads across multiple tensor parallel GPUs.  # KV头分布在多个张量并行GPU上
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:  # KV头数<TP大小
            # Number of KV heads is less than TP size, so we replicate  # KV头数小于TP大小，因此复制
            # the KV heads across multiple tensor parallel GPUs.  # KV头在多个张量并行GPU上复制
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 当前TP组的KV头数
        self.head_dim = head_dim  # 头维度
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim  # V头维度

        self.q_size = self.num_heads * self.head_dim  # Q大小
        self.k_size = self.num_kv_heads * self.head_dim  # K大小
        self.v_size = self.num_kv_heads * self.v_head_dim  # V大小

        self.v_scale = v_scale  # V缩放因子

        self.scaling = self.head_dim**-0.5  # 注意力缩放因子

        self.qkv_proj = QKVParallelLinear(  # QKV并行线性层
            hidden_size,  # 输入维度
            self.head_dim,  # 头维度
            self.total_num_heads,  # 总Q头数
            self.total_num_kv_heads,  # 总KV头数
            v_head_size=self.v_head_dim,  # V头大小
            bias=attention_bias,  # 注意力偏置

            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # TP秩
            tp_size=attn_tp_size,  # TP大小
            prefix=add_prefix("qkv_proj", prefix),
            skip_block_quant_check=True,  # 跳过块量化检查
        )

        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.v_head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # TP秩
            tp_size=attn_tp_size,  # TP大小
            reduce_results=False,  # 不自动归约
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(  # 旋转位置编码
            self.head_dim,  # 头维度
            rotary_dim=self.head_dim,  # 旋转维度
            max_position=max_position_embeddings,  # 最大位置
            base=rope_theta,  # RoPE基频
            rope_scaling=rope_scaling,  # RoPE缩放
            partial_rotary_factor=partial_rotary_factor,  # 部分旋转因子
        )

        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,  # V头维度
            sliding_window_size=sliding_window_size,  # if is -1 ,normal attention,else ,window attention  # 滑动窗口大小
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.attention_sink_bias = (  # 注意力汇聚偏置
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if attention_sink_bias
            else None
        )

    def op_prepare(self, state):
        """TBO操作：注意力准备阶段"""
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        """TBO操作：注意力核心计算"""
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """注意力准备阶段：QKV投影+旋转位置编码"""
        if hidden_states.shape[0] == 0:  # 空输入
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)  # 拆分QKV

        q, k = self.rotary_emb(positions, q, k)  # 旋转位置编码
        if self.v_scale is not None:  # V缩放
            v = v * self.v_scale

        inner_state = q, k, v, forward_batch  # 内部状态
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        """注意力核心计算：注意力+输出投影"""
        hidden_states, forward_batch, inner_state = intermediate_state  # 解包中间状态
        if inner_state is None:  # 空状态
            return hidden_states
        attn_output = self.attn(
            *inner_state,
            sinks=self.attention_sink_bias,  # 注意力汇聚偏置
        )
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """注意力前向传播"""
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)  # 拆分QKV

        # [t, h, dr]  # 时间步，头数，旋转维度
        q, k = self.rotary_emb(positions, q, k)  # 旋转位置编码
        # [t, h, d]  # 时间步，头数，头维度

        if self.v_scale is not None:  # V缩放
            v = v * self.v_scale
        attn_output = self.attn(q, k, v, forward_batch, sinks=self.attention_sink_bias)  # 注意力计算
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output


class MiMoV2DecoderLayer(nn.Module):
    """MiMo V2解码器层：注意力+MLP/MoE，支持滑动窗口和MoE混合"""
    def __init__(
        self,
        config: MiMoV2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.hidden_size = config.hidden_size  # 隐藏维度
        self.layer_id = layer_id  # 层ID

        rope_theta = getattr(config, "rope_theta", 10000)  # RoPE基频
        rope_scaling = getattr(config, "rope_scaling", None)  # RoPE缩放
        # In v5, rope_scaling is a property alias for rope_parameters and returns  # 在v5中，rope_scaling是rope_parameters的属性别名
        # a standardized dict even when there's no actual scaling.  Treat the  # 即使没有实际缩放也返回标准化字典
        # "default" (no-op) type as None so factory.py uses plain RotaryEmbedding.  # 将"default"（无操作）类型视为None，使factory.py使用普通RotaryEmbedding
        if (
            isinstance(rope_scaling, dict)
            and rope_scaling.get("rope_type") == "default"
        ):  # 默认类型视为无缩放
            rope_scaling = None
        max_position_embeddings = getattr(  # 最大位置嵌入
            config,
            "context_len",
            getattr(config, "max_position_embeddings", 32768),
        )

        if self.is_swa_layer():  # 滑动窗口注意力层
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,  # SWA头数
                num_kv_heads=config.swa_num_key_value_heads,  # SWA KV头数
                head_dim=config.swa_head_dim,  # SWA头维度
                v_head_dim=getattr(config, "swa_v_head_dim", None),  # SWA V头维度
                v_scale=getattr(config, "attention_value_scale", None),  # 注意力值缩放
                sliding_window_size=config.sliding_window_size,  # 滑动窗口大小
                attention_bias=config.attention_bias,
                attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),  # SWA注意力汇聚偏置
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),  # SWA RoPE基频
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=add_prefix("self_attn", prefix),
            )
        else:  # 全注意力层
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=getattr(config, "attention_value_scale", None),
                sliding_window_size=-1,  # normal attention  # 普通注意力
                attention_bias=config.attention_bias,
                attention_sink_bias=getattr(
                    config, "add_full_attention_sink_bias", False
                ),  # 全注意力汇聚偏置
                layer_id=layer_id,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=add_prefix("self_attn", prefix),
            )

        self.is_layer_sparse = self.is_moe_layer(layer_id)  # 当前层是否是MoE层
        is_previous_layer_sparse = self.is_moe_layer(layer_id - 1)  # 前一层是否是MoE层
        is_next_layer_sparse = self.is_moe_layer(layer_id + 1)  # 后一层是否是MoE层

        if self.is_layer_sparse:  # MoE层
            self.mlp = MiMoV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=layer_id,
            )
        else:  # 密集MLP层
            if enable_moe_dense_fully_dp():  # MoE密集全DP模式
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:  # 正常模式
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = MiMoV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.layernorm_epsilon
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(  # 层散射模式
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
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1),  # 是否最后一层
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播"""
        # Self Attention  # 自注意力
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )  # 准备注意力输入

        if hidden_states.shape[0] != 0:  # 有token
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )  # 自注意力计算

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )  # 准备MLP输入

        should_allreduce_fusion = (  # 是否融合MLP全归约与下一层
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.  # 对于带填充的DP，可以使用reduce-scatter代替all-reduce
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )  # 是否使用reduce-scatter

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )  # MLP/MoE计算

        if should_allreduce_fusion:  # 融合全归约
            hidden_states._sglang_needs_allreduce_fusion = True
        else:  # 正常后处理
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        """判断指定层是否是MoE层"""
        return (
            hasattr(self.config, "moe_layer_freq")
            and 0 <= layer_idx < len(self.config.moe_layer_freq)
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

    def is_swa_layer(self) -> bool:
        """判断当前层是否是滑动窗口注意力层"""
        return self.config.hybrid_layer_pattern[self.layer_id] == 1

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        """TBO操作：准备注意力通信"""
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )  # 准备注意力
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )  # 更新状态

    def op_comm_prepare_mlp(self, state):
        """TBO操作：准备MLP通信"""
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_comm_postprocess_layer(self, state):
        """TBO操作：层后处理通信"""
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )  # 清除已消费的状态
        return output


class MiMoV2Model(nn.Module):
    """MiMo V2模型主体：嵌入+解码器层堆叠+归一化"""
    def __init__(
        self,
        config: MiMoV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = MiMoV2DecoderLayer,
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.padding_idx = getattr(config, "pad_token_id", None)  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.pp_group = get_pp_group()  # 流水线并行组

        if self.pp_group.is_first_rank:  # 第一个PP rank
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,  # 词表大小
                config.hidden_size,  # 隐藏维度
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),  # 使用注意力TP组
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:  # 非第一个PP rank
            self.embed_tokens = PPMissingLayer()  # 占位层

        # Use the provided decoder layer type or default to MiMoV2DecoderLayer  # 使用提供的解码器层类型或默认MiMoV2DecoderLayer
        decoder_layer_type = decoder_layer_type or MiMoV2DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(  # 创建解码器层
            config.num_hidden_layers,
            layer_fn=lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,  # PP rank
            pp_size=self.pp_group.world_size,  # PP世界大小
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:  # 最后一个PP rank
            self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)  # 最终归一化
        else:  # 非最后一个PP rank
            self.norm = PPMissingLayer(return_tuple=True)  # 占位层

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输入嵌入"""
        if hasattr(self.config, "scale_emb"):  # 是否缩放嵌入
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:  # 不缩放
            return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        """模型前向传播"""
        if self.pp_group.is_first_rank:  # 第一个PP rank
            if input_embeds is None:  # 没有预计算的嵌入
                hidden_states = self.embed_tokens(input_ids)  # 词嵌入
            else:  # 使用预计算的嵌入
                hidden_states = input_embeds
            residual = None
        else:  # 非第一个PP rank
            assert pp_proxy_tensors is not None  # 必须有PP代理张量
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理获取残差

        if forward_batch.can_run_tbo:  # 可以运行TBO
            tbo_start_layer = self.start_layer  # TBO起始层
            tbo_end_layer = self.end_layer  # TBO结束层

            # skip first layer for TBO when starting from layer 0  # 从第0层开始时跳过第一层用于TBO
            if self.start_layer == 0:
                layer = self.layers[0]  # 第一层
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual
                )
                tbo_start_layer = tbo_start_layer + 1  # 调整TBO起始层

            hidden_states, residual = model_forward_maybe_tbo(  # TBO前向传播
                layers=self.layers[tbo_start_layer:tbo_end_layer],
                enable_tbo=True,
                input_data_scatter_mode=(
                    ScatterMode.model_input_output()
                    if tbo_start_layer == self.start_layer
                    else self.layers[
                        tbo_start_layer - 1
                    ].layer_scatter_modes.layer_output_mode
                ),  # 散射模式
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:  # 非TBO模式
            for i in range(self.start_layer, self.end_layer):  # 逐层前向传播
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )

        hidden_states_before_norm = None  # 归一化前的隐藏状态
        if not self.pp_group.is_last_rank:  # 非最后一个PP rank
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )  # 返回PP代理张量
        else:  # 最后一个PP rank
            if hidden_states.shape[0] > 0:  # 有token
                if forward_batch.return_hidden_states_before_norm:  # 需要返回归一化前的隐藏状态
                    hidden_states_before_norm = (
                        hidden_states if residual is None else hidden_states + residual
                    )
                if residual is None:  # 无残差
                    hidden_states = self.norm(hidden_states)
                else:  # 有残差
                    hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states, hidden_states_before_norm  # 返回隐藏状态和归一化前状态

    # If this function is called, it should always initialize KV cache scale  # 如果调用此函数，应始终初始化KV缓存缩放因子
    # factors (or else raise an exception). Thus, handled exceptions should  # 因子（否则抛出异常）。因此，已处理的异常应
    # make sure to leave KV cache scale factors in a known good (dummy) state  # 确保将KV缓存缩放因子保持在已知良好（虚拟）状态
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """加载KV缓存缩放因子"""
        attn_tp_rank = get_attention_tp_rank()  # 注意力TP秩
        attn_tp_size = get_attention_tp_size()  # 注意力TP大小
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            attn_tp_rank,
            attn_tp_size,
            self.config.num_hidden_layers,  # 层数
            self.config.__class__.model_type,  # 模型类型
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):  # 非占位层
                layer_self_attn = self.layers[layer_idx].self_attn  # 获取自注意力层
            if hasattr(layer_self_attn.attn, "k_scale"):  # 有K缩放属性
                layer_self_attn.attn.k_scale = scaling_factor  # 设置K缩放
                layer_self_attn.attn.v_scale = scaling_factor  # 设置V缩放
            else:  # 无缩放属性
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )


class MiMoV2ForCausalLM(nn.Module):
    """MiMo V2因果语言模型：支持纯语言和多模态（视觉+音频）"""
    # BitandBytes specific attributes  # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射
        # shard_name, weight_name, index  # 分片名，权重名，索引
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Prefixes for weight routing in encoder_only/language_only modes  # encoder_only/language_only模式的权重路由前缀
    _LANGUAGE_WEIGHT_PREFIXES = ("model.", "lm_head.")  # 语言权重前缀
    _VISION_AUDIO_WEIGHT_PREFIXES = ("visual.", "vision_model.", "audio_")  # 视觉/音频权重前缀
    _VISION_AUDIO_WEIGHT_SUBSTRING = "speech_embeddings"  # 视觉/音频权重子串

    def __init__(
        self,
        config: MiMoV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.pp_group = get_pp_group()  # 流水线并行组
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 量化配置
        self._encoder_processor = None  # lazy-created in preprocess_mm_for_encoder  # 懒创建的编码器预处理器

        if not self.config.encoder_only:  # 非纯编码器模式
            self.model = MiMoV2Model(
                config, quant_config=quant_config, prefix=add_prefix("model", prefix)
            )

            if self.pp_group.is_last_rank:  # 最后一个PP rank
                self.lm_head = ParallelLMHead(
                    config.vocab_size,  # 词表大小
                    config.hidden_size,  # 隐藏维度
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # DP LM头
                )
            else:  # 非最后一个PP rank
                self.lm_head = PPMissingLayer()  # 占位层
        else:  # 纯编码器模式
            self.model = None
            self.lm_head = None

        self.logits_processor = (
            LogitsProcessor(config) if not self.config.encoder_only else None
        )  # logits处理器

        vision_config = getattr(config, "vision_config", None)  # 视觉配置
        audio_config = getattr(config, "audio_config", None)  # 音频配置
        self._is_multimodal = vision_config is not None and audio_config is not None  # 是否多模态
        # Always build vision/audio encoders so P can fall back to local  # 始终构建视觉/音频编码器，以便P可以回退到本地
        # encoding when the EPD encoder is unreachable.  # 当EPD编码器不可达时进行编码
        if self._is_multimodal:  # 多模态模式
            if hasattr(vision_config, "to_dict"):  # 转换为字典
                vision_config = vision_config.to_dict()
            if hasattr(audio_config, "to_dict"):  # 转换为字典
                audio_config = audio_config.to_dict()

            self.visual = MiMoVisionTransformer(  # 视觉Transformer
                MiMoVLVisionConfig.from_dict(vision_config),
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=None,
                prefix=add_prefix("visual", prefix),
            )
            self.audio_config = MiMoAudioEncoderConfig(**audio_config)  # 音频编码器配置
            self.audio_encoder = MiMoAudioEncoder(self.audio_config)  # 音频编码器

        self._routed_experts_weights_of_layer = LazyValue(  # 懒加载的路由专家权重
            lambda: (
                {
                    layer_id: layer.mlp.get_moe_weights()
                    for layer_id, layer in enumerate(self.model.layers)
                    if isinstance(layer.mlp, MiMoV2MoE)
                }
                if self.model is not None
                else {}
            )
        )

    @property
    def routed_experts_weights_of_layer(self):
        """获取路由专家权重"""
        return self._routed_experts_weights_of_layer.value

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输入嵌入"""
        assert (
            self.model is not None
        ), "get_input_embedding() is not available in encoder_only mode"
        return self.model.get_input_embedding(input_ids)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """填充输入ID以适配多模态token"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def preprocess_mm_for_encoder(self, mm_data, modality, config):
        """预处理多模态数据用于编码器"""
        if self._encoder_processor is None:  # 懒创建处理器
            from sglang.srt.multimodal.processors.mimo_v2 import MiMoProcessor

            self._encoder_processor = MiMoProcessor.from_hf_config(
                self.config, mm_config=config
            )
        return self._encoder_processor.preprocess_for_encoder(mm_data, modality)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取图像特征"""
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )  # 拼接像素值
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)  # 拼接网格维度
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        return self.visual(pixel_values, grid_thw=image_grid_thw)  # 视觉编码

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取视频特征"""
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )  # 拼接像素值
        video_grid_thw = torch.cat([item.video_grid_thw for item in items], dim=0)  # 拼接网格维度
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        return self.visual(pixel_values, grid_thw=video_grid_thw)  # 视觉编码

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取音频特征"""
        return self.audio_encoder.get_audio_feature(items)

    @torch.inference_mode()  # 推理模式
    def encode_video_audio(self, mm_inputs: Dict) -> Optional[torch.Tensor]:
        """编码视频中的音频轨道"""
        # EPD-side hook: encode audio tracks pulled from videos and trim to the  # EPD侧钩子：编码从视频提取的音频轨道并裁剪到
        # interleaved per-video segments produced by MiMoProcessor (segment  # MiMoProcessor产生的交错每视频段（段
        # starts / lens / per_video_num_units). Returns None if there is no  # 起始/长度/每视频单元数）。如果没有
        # audio to encode. The server passes the result through to the receiver  # 音频需要编码则返回None。服务器将结果传递给接收方
        # under aux_data["video_audio_embedding"].  # 在aux_data["video_audio_embedding"]下
        import numpy as np  # 导入NumPy

        audio_features = mm_inputs.get("video_audio_features")  # 获取视频音频特征
        if not audio_features:  # 无音频特征
            return None

        def _as_tensor(data):  # 转换为张量
            if isinstance(data, torch.Tensor):  # 已是张量
                return data
            if isinstance(data, np.ndarray):  # NumPy数组
                return torch.tensor(data)
            if isinstance(data, list) and data and isinstance(data[0], np.ndarray):  # NumPy数组列表
                return torch.tensor(np.array(data))
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):  # 数值列表
                return torch.tensor(data)
            return data

        audio_feature_lens = mm_inputs["video_audio_feature_lens"]  # 音频特征长度
        audio_item = MultimodalDataItem.from_dict(
            {
                "modality": Modality.AUDIO,  # 音频模态
                "feature": _as_tensor(audio_features),  # 音频特征
            }
        )
        audio_item.set("audio_feature_lens", _as_tensor(audio_feature_lens))  # 设置特征长度

        audio_embedding = self.get_audio_feature([audio_item]).cpu()  # 获取音频嵌入
        if audio_embedding.ndim != 2:  # 确保是二维
            audio_embedding = audio_embedding.reshape(-1, audio_embedding.shape[-1])

        segment_lens_flat = mm_inputs["video_audio_segment_lens_flat"]  # 段长度
        segment_starts_flat = mm_inputs["video_audio_segment_starts_flat"]  # 段起始位置
        per_video_num_units = mm_inputs["video_audio_per_video_num_units"]  # 每视频单元数
        per_video_audio_token_lens = (  # 每视频音频token长度
            audio_feature_lens.tolist()
            if hasattr(audio_feature_lens, "tolist")
            else list(audio_feature_lens)
        )

        trimmed_chunks = []  # 裁剪后的块
        emb_offset = 0  # 嵌入偏移
        unit_idx = 0  # 单元索引
        audio_video_idx = 0  # 音频视频索引
        for num_units in per_video_num_units:  # 遍历每个视频
            if num_units <= 0:  # 无单元
                continue
            vid_audio_len = per_video_audio_token_lens[audio_video_idx]  # 视频音频长度
            for _ in range(num_units):  # 遍历每个单元
                start = segment_starts_flat[unit_idx]  # 段起始
                seg_len = segment_lens_flat[unit_idx]  # 段长度
                trimmed_chunks.append(
                    audio_embedding[emb_offset + start : emb_offset + start + seg_len]
                )  # 裁剪嵌入
                unit_idx += 1
            emb_offset += vid_audio_len  # 更新偏移
            audio_video_idx += 1  # 更新索引

        return (
            torch.cat(trimmed_chunks, dim=0) if trimmed_chunks else audio_embedding[:0]
        )  # 返回裁剪后的嵌入

    def get_input_embeddings(self) -> Optional[nn.Embedding]:
        """获取输入嵌入层"""
        return self.model.embed_tokens if self.model is not None else None

    @torch.no_grad()  # 禁用梯度计算
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        """因果语言模型前向传播"""
        assert (
            not self.config.encoder_only
        ), "forward() should not be called in encoder_only mode"  # 纯编码器模式不能调用forward

        if self._is_multimodal:  # 多模态模式
            hidden_states, hidden_states_before_norm = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:  # 纯语言模式
            hidden_states, hidden_states_before_norm = self.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        if self.pp_group.is_last_rank:  # 最后一个PP rank
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                hidden_states_before_norm=hidden_states_before_norm,
            )  # 计算logits
        else:  # 非最后一个PP rank
            return hidden_states

    @property
    def start_layer(self):
        """获取起始层索引"""
        return self.model.start_layer if self.model is not None else 0

    @property
    def end_layer(self):
        """获取结束层索引"""
        return self.model.end_layer if self.model is not None else 0

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        stacked_params_mapping_vit = [  # ViT堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)  # 专家参数映射
        expert_params_mapping = DeepEPMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())  # 参数字典
        skipped_mtp_weights = False  # 是否跳过MTP权重

        def _is_vision_audio_weight(name):  # 判断是否是视觉/音频权重
            return (
                name.startswith(self._VISION_AUDIO_WEIGHT_PREFIXES)
                or self._VISION_AUDIO_WEIGHT_SUBSTRING in name
            )

        for name, loaded_weight in weights:  # 遍历权重
            if not self._is_multimodal and _is_vision_audio_weight(name):  # 非多模态跳过视觉/音频权重
                continue

            if self.config.encoder_only and name.startswith(  # 纯编码器模式跳过语言权重
                self._LANGUAGE_WEIGHT_PREFIXES
            ):
                continue

            if self._is_multimodal and "audio" in name:  # 多模态音频权重
                if "projection" in name:  # 投影层权重
                    if (
                        "audio_encoder.audio_projection" in name
                        and "audio_encoder.projection" not in name
                    ):  # 重命名投影层
                        name = name.replace(
                            "audio_encoder.audio_projection", "audio_encoder.projection"
                        )
                    elif (
                        "audio_projection" in name
                        and "audio_encoder.projection" not in name
                    ):  # 重命名投影层
                        name = name.replace(
                            "audio_projection", "audio_encoder.projection"
                        )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    continue

                if "input_local_transformer" in name:  # 输入局部Transformer权重
                    if (
                        "audio_input_local_transformer" in name
                        and "audio_encoder.input_local_transformer" not in name
                    ):  # 重命名
                        name = name.replace(
                            "audio_input_local_transformer",
                            "audio_encoder.input_local_transformer",
                        )
                    if name not in params_dict:  # 参数不存在
                        logger.warning(
                            f"Parameter {name} not found in params_dict, skipping"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    continue

            if self._is_multimodal and "speech_embeddings" in name:  # 语音嵌入权重
                if (
                    "speech_embeddings" in name
                    and "audio_encoder.speech_embeddings" not in name
                ):  # 重命名
                    name = name.replace(
                        "speech_embeddings", "audio_encoder.speech_embeddings"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight[: param.shape[0], :])  # 截取匹配大小
                continue

            if self._is_multimodal and "visual" in name:  # 视觉权重
                name = name.replace("vision_model.", "")  # 移除vision_model前缀
                name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换注意力前缀
                match_stacked_vit = False
                for param_name, weight_name, shard_id in stacked_params_mapping_vit:  # 匹配堆叠参数
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:
                        match_stacked_vit = True
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    match_stacked_vit = True
                    break
                if match_stacked_vit:  # 匹配到ViT堆叠参数
                    continue
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

                if name.endswith("patch_embed.proj.weight"):  # 补丁嵌入投影权重
                    patch_embed = self.get_submodule(name.rsplit(".", 2)[0])
                    if hasattr(patch_embed, "sync_proj_weight_linear_format"):  # 同步格式
                        patch_embed.sync_proj_weight_linear_format()
                continue

            layer_id = get_layer_id(name)  # 获取层ID
            if (  # 跳过不在当前PP范围内的层
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:  # 跳过旋转嵌入频率和投影器
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存的cos/sin
                # Models trained using ColossalAI may include these tensors in  # 使用ColossalAI训练的模型可能包含这些张量
                # the checkpoint. Skip them.  # 在检查点中。跳过它们。
                continue

            if self.config.tie_word_embeddings and "lm_head.weight" in name:  # 权重绑定
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:  # PP模式
                    # Handle pp weight tying here  # 在此处理PP权重绑定
                    # find the embed_tokens.weight in the weights  # 在权重中查找embed_tokens.weight
                    embed_token_weights = next(

                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:  # 非PP模式
                    continue

            if "mtp" in name:  # MTP权重
                if not skipped_mtp_weights:
                    logger.info(
                        "Skipping draft-only MiMo-V2 MTP weights while loading the "
                        "target model; MiMoV2MTP loads these weights in the draft "
                        "model runner."
                    )  # 跳过MTP权重提示
                    skipped_mtp_weights = True
                continue

            # Support fused qkv_proj checkpoint (Pro format)  # 支持融合qkv_proj检查点（Pro格式）
            if "qkv_proj" in name:  # 融合QKV权重
                if name in params_dict:
                    param = params_dict[name]
                    expected_fused_tp_size = get_mimo_v2_fused_qkv_expected_tp_size(
                        self.config
                    )
                    load_mimo_v2_qkv_proj_weight(
                        name, param, loaded_weight, expected_fused_tp_size
                    )  # 加载融合QKV权重
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 匹配堆叠参数
                if (
                    "compression_attention" in name
                    or "hybrid_softmax_attention" in name
                    or "compressed_softmax_attn" in name
                ):  # 跳过压缩注意力
                    continue
                if weight_name not in name:  # 不匹配
                    continue
                if ("mlp.experts." in name) and name not in params_dict:  # 专家参数不存在
                    continue

                name = name.replace(weight_name, param_name)  # 替换名称
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 未匹配堆叠参数
                for mapping in expert_params_mapping:  # 匹配专家参数
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )  # 加载专家权重
                    break
                else:  # 未匹配专家参数
                    # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if name in params_dict.keys():  # 参数存在
                        param = params_dict[name]
                        if "attention_sink_bias" in name:  # 注意力汇聚偏置
                            start = get_attention_tp_rank() * param.numel()  # 起始位置
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )  # 按TP秩截取
                        else:  # 普通权重
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:  # 参数不存在
                        logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        """获取嵌入权重和LM头权重"""
        assert (
            self.model is not None and self.lm_head is not None
        ), "get_embed_and_head() is not available in encoder_only mode"
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """设置嵌入权重和LM头权重"""
        assert (
            self.model is not None and self.lm_head is not None
        ), "set_embed_and_head() is not available in encoder_only mode"
        del self.model.embed_tokens.weight  # 删除旧权重
        del self.lm_head.weight  # 删除旧权重
        self.model.embed_tokens.weight = embed  # 设置新权重
        self.lm_head.weight = head  # 设置新权重
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """加载KV缓存缩放因子"""
        if self.model is not None:
            self.model.load_kv_cache_scales(quantization_param_path)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置模型配置"""
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,  # 层数
            num_logical_experts=getattr(config, "n_routed_experts", 1),  # 逻辑专家数
            num_groups=getattr(config, "n_group", None),  # 分组数
        )


# Keep the old Flash architecture name loadable while new configs use MiMoV2ForCausalLM.  # 保持旧的Flash架构名称可加载，而新配置使用MiMoV2ForCausalLM
class MiMoV2FlashForCausalLM(MiMoV2ForCausalLM):
    """MiMo V2 Flash因果语言模型（兼容旧名称）"""
    pass


EntryClass = [MiMoV2ForCausalLM, MiMoV2FlashForCausalLM]  # 入口类列表
