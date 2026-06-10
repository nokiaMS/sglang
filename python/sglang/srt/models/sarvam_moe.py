# Sarvam MoE 模型推理实现文件
# 本文件实现了Sarvam MoE混合专家模型，支持MLA和MHA两种注意力模式
# 包含SarvamMLAForCausalLM (105B) 和 SarvamMoEForCausalLM (30B) 两个模型
# 支持张量并行、专家并行、FP8量化和双流并行计算

"""Inference-only Sarvam MoE models for SGLang.
- SarvamMLAForCausalLM (105B)
- SarvamMoEForCausalLM (30B)
"""

import math  # 导入数学模块
from enum import IntEnum, auto  # 导入枚举类型
from typing import Any, Dict, Iterable, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入函数式接口
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.distributed import (  # 导入分布式模块
    get_pp_group,  # 获取PP组
    get_tensor_model_parallel_world_size,  # 获取TP世界大小
    tensor_model_parallel_all_reduce,  # TP全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 专家分布记录器
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation  # 专家位置配置
from sglang.srt.layers.activation import SiluAndMul  # SiLU激活函数
from sglang.srt.layers.attention.utils import concat_and_cast_mha_k_triton  # MHA K拼接和类型转换
from sglang.srt.layers.communicator import (  # 层通信器
    LayerCommunicator,  # 层通信器
    LayerScatterModes,  # 层散射模式
    enable_moe_dense_fully_dp,  # 启用MoE密集全DP
)
from sglang.srt.layers.dp_attention import (  # DP注意力
    get_attention_tp_rank,  # 获取注意力TP秩
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # RMS归一化
from sglang.srt.layers.linear import (  # 线性层
    ColumnParallelLinear,  # 列并行线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput  # 逻辑处理器
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce  # MoE全归约跳过判断
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # MoE实现类
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 融合MoE
from sglang.srt.layers.moe.topk import TopK  # TopK路由
from sglang.srt.layers.moe.utils import RoutingMethodType  # 路由方法类型
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 旋转位置编码
from sglang.srt.layers.utils import get_layer_id  # 层ID获取
from sglang.srt.layers.vocab_parallel_embedding import (  # 词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # CUDA图捕获模式
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 前向批次
from sglang.srt.model_executor.forward_context import (  # 前向上下文
    get_attn_backend,  # 获取注意力后端
    get_token_to_kv_pool,  # 获取KV池
)
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 默认权重加载器
from sglang.srt.models.bailing_moe import BailingMoEForCausalLM  # 百灵MoE模型
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,  # DeepSeek MHA前向混入
)
from sglang.srt.server_args import get_global_server_args  # 全局服务器参数
from sglang.srt.utils import (  # 工具函数
    BumpAllocator,  # 凸包分配器
    add_prefix,  # 添加前缀
    bind_or_assign,  # 绑定或赋值
    is_cuda,  # 是否CUDA
    is_nvidia_cublas_version_ge_12_9,  # cuBLAS版本判断
    make_layers,  # 创建层
    next_power_of_2,  # 下一个2的幂
)

_is_cuda = is_cuda()  # 是否CUDA
_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()  # cuBLAS版本>=12.9

if _is_cuda:  # CUDA平台导入
    try:
        from sgl_kernel import bmm_fp8, merge_state_v2  # sgl_kernel算子

        from sglang.jit_kernel.concat_mla import concat_mla_k  # MLA K拼接
        from sglang.srt.layers.quantization.fp8_kernel import per_tensor_quant_mla_fp8  # FP8量化

        _has_fp8_support = True  # 支持FP8
        _has_concat_mla_k = True  # 支持MLA K拼接
    except ImportError:
        _has_fp8_support = False  # 不支持FP8
        _has_concat_mla_k = False  # 不支持MLA K拼接
        bmm_fp8 = None
        concat_mla_k = None
        merge_state_v2 = None
        per_tensor_quant_mla_fp8 = None
else:  # 非CUDA平台
    _has_fp8_support = False
    _has_concat_mla_k = False
    bmm_fp8 = None
    concat_mla_k = None
    merge_state_v2 = None
    per_tensor_quant_mla_fp8 = None


class AttnForwardMethod(IntEnum):
    """注意力前向方法枚举"""
    MLA_SEPARATE_ROPE = auto()  # MLA分离RoPE模式
    MLA_CONCAT_ROPE = auto()  # MLA拼接RoPE模式
    MHA_PREFILL = auto()  # MHA预填充模式


SEPARATE_ROPE_BACKENDS = frozenset(
    ["fa3", "flashinfer", "dsa", "nsa", "cutlass_mla", "trtllm_mla"]
    # "nsa" is a deprecated alias for "dsa"
)  # 支持分离RoPE的后端集合
CONCAT_ROPE_BACKENDS = frozenset(["flashmla", "triton"])  # 支持拼接RoPE的后端集合


class AttentionBackendRegistry:
    """注意力后端注册表，管理不同后端的注意力前向方法"""
    _handlers = {}

    @classmethod
    def register(cls, backend_name: str, handler_func):
        """注册注意力后端处理器"""
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name: str):
        """获取指定后端的处理器"""
        return cls._handlers.get(backend_name, cls._default_handler)

    @classmethod
    def _default_handler(cls, attn, forward_batch) -> AttnForwardMethod:
        """默认处理器，返回MLA拼接RoPE模式"""
        return AttnForwardMethod.MLA_CONCAT_ROPE

    @classmethod
    def get_forward_method(
        cls, backend_name: str, attn, forward_batch
    ) -> AttnForwardMethod:
        """获取指定后端的注意力前向方法"""
        handler = cls.get_handler(backend_name)
        return handler(attn, forward_batch)


def _handle_separate_rope_backend(attn, forward_batch) -> AttnForwardMethod:
    """处理分离RoPE后端，返回MLA分离RoPE方法"""
    return AttnForwardMethod.MLA_SEPARATE_ROPE


def _handle_concat_rope_backend(attn, forward_batch) -> AttnForwardMethod:
    """处理拼接RoPE后端，返回MLA拼接RoPE方法"""
    return AttnForwardMethod.MLA_CONCAT_ROPE


for backend in SEPARATE_ROPE_BACKENDS:
    AttentionBackendRegistry.register(backend, _handle_separate_rope_backend)  # 注册分离RoPE后端
for backend in CONCAT_ROPE_BACKENDS:
    AttentionBackendRegistry.register(backend, _handle_concat_rope_backend)  # 注册拼接RoPE后端


def get_attn_forward_method(server_args, forward_batch) -> AttnForwardMethod:
    """根据服务器参数和前向批次获取注意力前向方法"""
    is_decode = forward_batch.forward_mode.is_decode_or_idle()  # 是否解码或空闲
    if is_decode:
        backend = server_args.decode_attention_backend or server_args.attention_backend  # 解码后端
    else:
        backend = server_args.prefill_attention_backend or server_args.attention_backend  # 预填充后端
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and backend == "fa3"
        ):
            return AttnForwardMethod.MHA_PREFILL  # FA3扩展模式使用MHA预填充
    return AttentionBackendRegistry.get_forward_method(backend, None, forward_batch)  # 从注册表获取


class SarvamMoEMLP(nn.Module):
    """Sarvam MoE的MLP模块"""
    def __init__(
        self,
        hidden_size: int,  # 隐藏层大小
        intermediate_size: int,  # 中间层大小
        hidden_act: str,  # 激活函数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        reduce_results: bool = True,  # 是否归约结果
        tp_rank: Optional[int] = None,  # TP秩
        tp_size: Optional[int] = None,  # TP大小
    ) -> None:
        """初始化MLP，配置门控上投影和下投影"""
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()  # SiLU激活函数

    def forward(
        self,
        x,
        forward_batch: ForwardBatch = None,  # 前向批次
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ):
        """MLP前向传播：门控上投影、激活、下投影"""
        if x.shape[0] == 0:  # 空输入
            return x
        gate_up, _ = self.gate_up_proj(x)  # 门控上投影
        x = self.act_fn(gate_up)  # 激活
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )  # 下投影
        return x


class SarvamMoESparseMoeBlock(nn.Module):
    """Sarvam MoE稀疏MoE块，支持共享专家和路由专家"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 替代CUDA流
    ):
        """初始化稀疏MoE块，配置路由、专家和共享专家"""
        super().__init__()
        self.config = config  # 保存配置
        self.layer_id = layer_id  # 层ID
        self.tp_size = get_tensor_model_parallel_world_size()  # TP大小
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 2.5)  # 路由缩放因子
        self.score_function = getattr(config, "score_function", "sigmoid")  # 评分函数
        self.n_group = getattr(config, "n_group", None)  # 分组数
        self.topk_group = getattr(config, "topk_group", None)  # 分组Top-K
        self.alt_stream = alt_stream  # 替代流

        dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        router_dtype_cfg = getattr(config, "router_dtype", "fp32")
        self.router_dtype = dtype_map.get(router_dtype_cfg, None)  # 路由器数据类型

        if self.tp_size > config.num_experts:  # TP不能超过专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )  # 评分修正偏置

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            use_grouped_topk=self.n_group is not None and self.topk_group is not None,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
            renormalize=True,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
            scoring_func=self.score_function,
            correction_bias=self.e_score_correction_bias,
            quant_config=quant_config,
            layer_id=layer_id,
        )  # TopK路由

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.Renormalize,
        )  # 路由专家

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )  # 门控线性层

        if (
            getattr(config, "num_shared_experts", None)
            and config.num_shared_experts > 0
        ):
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            if enable_moe_dense_fully_dp():
                shared_tp_rank, shared_tp_size = 0, 1
            else:
                shared_tp_rank, shared_tp_size = None, None
            self.shared_experts = SarvamMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                reduce_results=False,
                tp_rank=shared_tp_rank,
                tp_size=shared_tp_size,
            )  # 共享专家
        else:
            self.shared_experts = None  # 无共享专家

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
        gemm_output_zero_allocator: Optional[BumpAllocator] = None,  # GEMM输出零分配器
    ) -> torch.Tensor:
        """MoE前向传播，根据条件选择双流或普通模式"""
        del gemm_output_zero_allocator

        if (
            self.shared_experts is not None
            and self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            return self.forward_normal_dual_stream(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )  # 双流模式
        else:
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )  # 普通模式

    def get_moe_weights(self):
        """获取MoE专家权重"""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """共享专家前向传播"""
        return self.shared_experts(hidden_states)

    def _forward_router_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """路由专家前向传播"""
        if self.router_dtype is not None:
            router_logits = F.linear(
                hidden_states.to(self.router_dtype),
                self.gate.weight.to(self.router_dtype),
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """双流MoE前向传播，共享专家和路由专家并行执行"""
        num_tokens, hidden_dim = hidden_states.shape
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_out = self._forward_shared_experts(hidden_states)  # 共享专家
        with torch.cuda.stream(self.alt_stream):
            final_hidden_states = self._forward_router_experts(hidden_states)  # 路由专家
            if self.routed_scaling_factor != 1.0:
                final_hidden_states = final_hidden_states * self.routed_scaling_factor
        current_stream.wait_stream(self.alt_stream)
        final_hidden_states = final_hidden_states + shared_out  # 合并
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # TP全归约
        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """普通MoE前向传播"""
        if hidden_states.shape[0] == 0:  # 空输入
            return hidden_states

        num_tokens, hidden_dim = hidden_states.shape
        identity = (
            hidden_states.clone() if self.shared_experts is not None else hidden_states
        )

        if self.router_dtype is not None:
            router_logits = F.linear(
                hidden_states.to(self.router_dtype),
                self.gate.weight.to(self.router_dtype),
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)

        if self.shared_experts is not None:  # 有共享专家
            shared_out = self.shared_experts(identity)
            if self.routed_scaling_factor != 1.0:
                shared_out.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            else:
                shared_out.add_(final_hidden_states)
            final_hidden_states = shared_out
        elif self.routed_scaling_factor != 1.0:
            final_hidden_states = final_hidden_states * self.routed_scaling_factor

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class SarvamMoEMLAAttention(nn.Module):
    """Sarvam MoE的MLA注意力模块，支持MLA和MHA两种模式"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        hidden_size: int,  # 隐藏层大小
        num_heads: int,  # 头数
        layer_id: int = 0,  # 层ID
        rope_theta: float = 10000,  # RoPE基础频率
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放
        max_position_embeddings: int = 8192,  # 最大位置编码
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 替代CUDA流
    ) -> None:
        """初始化MLA注意力模块"""
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.quant_config = quant_config

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.qk_nope_head_dim = config.qk_nope_head_dim  # QK非RoPE维度
        self.qk_rope_head_dim = config.qk_rope_head_dim  # QK RoPE维度
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # QK总维度
        self.v_head_dim = config.v_head_dim  # V维度
        self.q_lora_rank = getattr(config, "q_lora_rank", None)  # Q LoRA秩
        self.kv_lora_rank = config.kv_lora_rank  # KV LoRA秩

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_global_server_args().kv_cache_dtype

        self._server_args = None
        self.current_attention_backend = None

        if self.q_lora_rank is None:  # 无LoRA秩时直接Q投影
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )
        else:  # 有LoRA秩时使用Qa+Qb投影
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_a_proj", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)  # KV归一化
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
        )

        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if rope_scaling and rope_scaling["type"] == "deepseek_yarn":
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 1.0)
            scaling_factor = rope_scaling.get("factor", 1.0)
            mscale = self.yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )  # MQA注意力

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )  # MHA注意力

        self.w_kc = None  # K吸收权重
        self.w_vc = None  # V吸收权重
        self.w_scale = None  # 权重缩放

    def yarn_get_mscale(self, scale: float = 1, mscale: float = 1) -> float:
        """计算YaRN缩放的mscale值"""
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _concat_and_cast_mha_k(
        self,
        k_nope: torch.Tensor,  # K非RoPE部分
        k_pe: torch.Tensor,  # K RoPE部分
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """拼接和类型转换MHA的K张量"""
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)

        if (
            _is_cuda
            and _has_concat_mla_k
            and (self.num_local_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):  # 使用专用拼接内核
            k = k_nope.new_empty(*k_shape)
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
            return k

        if (
            _is_cuda
            and next_power_of_2(self.num_local_heads) == self.num_local_heads
            and next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim
            and next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim
        ):  # 使用Triton拼接内核
            if (
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = get_token_to_kv_pool().dtype
            else:
                attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
            return k

        k = k_nope.new_empty(*k_shape)  # 朴素拼接
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        return k

    def _set_current_attention_backend(self, forward_batch: ForwardBatch) -> None:
        """设置当前注意力后端"""
        if self._server_args is None:
            self._server_args = get_global_server_args()
        if forward_batch.forward_mode.is_decode_or_idle():
            self.current_attention_backend = (
                self._server_args.decode_attention_backend
                or self._server_args.attention_backend
            )
        else:
            self.current_attention_backend = (
                self._server_args.prefill_attention_backend
                or self._server_args.attention_backend
            )

    def _maybe_fp8_bmm(
        self,
        x_bmk: torch.Tensor,  # 输入
        w_bkn: torch.Tensor,  # 权重
        zero_allocator: Optional[BumpAllocator] = None,  # 零分配器
    ) -> torch.Tensor:
        """可选的FP8批量矩阵乘法"""
        if (
            _has_fp8_support
            and w_bkn is not None
            and w_bkn.dtype == torch.float8_e4m3fn
        ):  # FP8路径
            x_val, x_scale = per_tensor_quant_mla_fp8(
                x_bmk,
                (
                    torch.zeros((1,), dtype=torch.float32, device=x_bmk.device)
                    if _is_cublas_ge_129
                    else (
                        zero_allocator.allocate(1)
                        if zero_allocator
                        else torch.zeros((1,), dtype=torch.float32, device=x_bmk.device)
                    )
                ),
            )
            w_scale = self.w_scale if self.w_scale is not None else 1.0
            return bmm_fp8(x_val, w_bkn, x_scale, w_scale, torch.bfloat16)

        return torch.bmm(x_bmk, w_bkn)  # 普通批量矩阵乘法

    def _run_mha_prefill(
        self,
        positions: torch.Tensor,  # 位置
        q: torch.Tensor,  # 查询
        q_pe: torch.Tensor,  # 查询RoPE
        k_nope: torch.Tensor,  # K非RoPE
        k_pe: torch.Tensor,  # K RoPE
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """运行MHA预填充模式"""
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)  # 应用RoPE
        q[..., self.qk_nope_head_dim :] = q_pe

        get_token_to_kv_pool().set_mla_kv_buffer(
            self.attn_mha,
            forward_batch.out_cache_loc,
            k_nope,
            k_pe,
        )  # 设置MLA KV缓冲区

        kv_a = k_nope.squeeze(1)
        kv_expanded, _ = self.kv_b_proj(kv_a)  # KV扩展投影
        kv_expanded = kv_expanded.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_expanded = kv_expanded[..., : self.qk_nope_head_dim]
        v = kv_expanded[..., self.qk_nope_head_dim :]

        k = self._concat_and_cast_mha_k(k_nope_expanded, k_pe, forward_batch)  # 拼接K

        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )

        self._set_current_attention_backend(forward_batch)
        can_use_prefix_cache = not self._server_args.disable_radix_cache
        do_prefix_merge = has_extend_prefix and can_use_prefix_cache

        if do_prefix_merge and forward_batch.num_prefix_chunks is None:  # 准备分块前缀缓存
            if hasattr(forward_batch, "prepare_chunked_prefix_cache_info"):
                forward_batch.prepare_chunked_prefix_cache_info(q.device)
            else:
                forward_batch.num_prefix_chunks = 0
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):
                get_attn_backend().init_mha_chunk_metadata(forward_batch)

        forward_batch.set_attn_attend_prefix_cache(False)
        forward_batch.mha_return_lse = do_prefix_merge
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)  # MHA注意力

        if do_prefix_merge and merge_state_v2 is not None:  # 合并前缀缓存
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        forward_batch.set_attn_attend_prefix_cache(None)

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,  # 查询
        accum_output: torch.Tensor,  # 累积输出
        accum_lse: torch.Tensor,  # 累积LSE
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """分块前缀注意力MHA合并"""
        return DeepseekMHAForwardMixin._chunked_prefix_attn_mha(
            self, q, accum_output, accum_lse, forward_batch
        )

    def _get_mla_kv_buffer(
        self,
        kv_indices: torch.Tensor,  # KV索引
        dst_dtype: torch.dtype,  # 目标数据类型
        forward_batch: ForwardBatch,  # 前向批次
    ):
        """获取MLA KV缓冲区"""
        return DeepseekMHAForwardMixin._get_mla_kv_buffer(
            self, kv_indices, dst_dtype, forward_batch
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        zero_allocator: Optional[BumpAllocator] = None,  # 零分配器
        llama_4_scaling: Optional[torch.Tensor] = None,  # Llama4缩放
    ) -> torch.Tensor:
        """MLA注意力前向传播"""
        del llama_4_scaling
        if hidden_states.shape[0] == 0:
            return hidden_states

        if self.q_lora_rank is None:  # 无LoRA秩
            q, _ = self.q_proj(hidden_states)
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
        else:  # 有LoRA秩
            q_a, _ = self.q_a_proj(hidden_states)
            q_a = self.q_a_layernorm(q_a)
            q, _ = self.q_b_proj(q_a)
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self._server_args is None:
            self._server_args = get_global_server_args()
        self._set_current_attention_backend(forward_batch)

        forward_method = get_attn_forward_method(self._server_args, forward_batch)

        if forward_method == AttnForwardMethod.MHA_PREFILL:
            return self._run_mha_prefill(
                positions=positions,
                q=q,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                forward_batch=forward_batch,
            )

        if self.alt_stream is not None and get_is_capture_mode():  # 双流并行
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with torch.cuda.stream(self.alt_stream):
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)  # RoPE在替代流

            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )  # 吸收在主流
            q_nope_out = q_nope_out.transpose(0, 1)

            current_stream.wait_stream(self.alt_stream)
        else:  # 单流
            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )
            q_nope_out = q_nope_out.transpose(0, 1)

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if forward_method == AttnForwardMethod.MLA_SEPARATE_ROPE:  # 分离RoPE
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
            )
        elif forward_method == AttnForwardMethod.MLA_CONCAT_ROPE:  # 拼接RoPE
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
            )
        else:
            raise ValueError(f"Unknown forward method: {forward_method}")
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = self._maybe_fp8_bmm(
            attn_output.transpose(0, 1), self.w_vc, zero_allocator
        )  # V吸收
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        output, _ = self.o_proj(attn_bmm_output)
        return output

    def forward_prepare(
        self,
        positions: torch.Tensor,  # 位置
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        zero_allocator: Optional[BumpAllocator] = None,  # 零分配器
        llama_4_scaling: Optional[torch.Tensor] = None,  # Llama4缩放
    ) -> Tuple[Optional[torch.Tensor], ForwardBatch, Optional[Tuple]]:
        """注意力准备阶段，分离QKV计算和注意力核心"""
        del llama_4_scaling
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None

        if self.q_lora_rank is None:
            # Dual-stream parallel Q and KV projections
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
                q, _ = self.q_proj(hidden_states)
                current_stream.wait_stream(self.alt_stream)
            else:
                q, _ = self.q_proj(hidden_states)
                latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
        else:
            # For q_lora_rank path, overlap q_a_proj with kv_a_proj
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
                q_a, _ = self.q_a_proj(hidden_states)
                current_stream.wait_stream(self.alt_stream)
            else:
                q_a, _ = self.q_a_proj(hidden_states)
                latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            q_a = self.q_a_layernorm(q_a)
            q, _ = self.q_b_proj(q_a)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self._server_args is None:
            self._server_args = get_global_server_args()
        self._set_current_attention_backend(forward_batch)
        forward_method = get_attn_forward_method(self._server_args, forward_batch)

        if forward_method == AttnForwardMethod.MHA_PREFILL:
            output = self._run_mha_prefill(
                positions=positions,
                q=q,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                forward_batch=forward_batch,
            )
            return output, forward_batch, None

        # Parallel Absorption + RoPE on separate streams
        # - Stream 1 (main): Absorption (q_nope @ w_kc)
        # - Stream 2 (alt): RoPE (q_pe, k_pe)
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            # RoPE on alt stream
            with torch.cuda.stream(self.alt_stream):
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

            # Absorption on main stream (runs in parallel with RoPE)
            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )
            q_nope_out = q_nope_out.transpose(0, 1)

            current_stream.wait_stream(self.alt_stream)
        else:
            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )
            q_nope_out = q_nope_out.transpose(0, 1)

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        inner_state = (q_nope_out, k_nope, q_pe, k_pe, forward_batch, zero_allocator)
        return None, forward_batch, inner_state

    def forward_core(
        self,
        intermediate_state: Tuple[
            Optional[torch.Tensor], ForwardBatch, Optional[Tuple]
        ],
    ) -> torch.Tensor:
        """注意力核心计算阶段"""
        hidden_states, forward_batch, inner_state = intermediate_state

        if inner_state is None:
            return hidden_states

        q_nope_out, k_nope, q_pe, k_pe, forward_batch, zero_allocator = inner_state

        if self._server_args is None:
            self._server_args = get_global_server_args()
        self._set_current_attention_backend(forward_batch)

        forward_method = get_attn_forward_method(self._server_args, forward_batch)

        if forward_method == AttnForwardMethod.MLA_SEPARATE_ROPE:
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
            )
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = self._maybe_fp8_bmm(
            attn_output.transpose(0, 1), self.w_vc, zero_allocator
        )
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        output, _ = self.o_proj(attn_bmm_output)
        return output

    def prepare_qkv_latent(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """准备QKV潜在缓存"""
        del forward_batch
        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        return latent_cache


class SarvamMoEMLADecoderLayer(nn.Module):
    """Sarvam MoE MLA解码器层"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
        alt_stream: Optional[torch.cuda.Stream] = None,  # 替代CUDA流
    ) -> None:
        """初始化解码器层，配置注意力和MoE/MLP"""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        if hasattr(config, "rope_parameters"):  # RoPE参数
            rope_theta = config.rope_parameters.get("rope_theta")
            rope_type = config.rope_parameters.get("rope_type")
            rope_scaling = config.rope_parameters if rope_type != "default" else None
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = SarvamMoEMLAAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        first_k_dense = getattr(config, "first_k_dense_replace", 1)  # 前K层使用密集层
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)  # MoE层频率
        has_moe = getattr(config, "num_experts", None) is not None  # 是否有MoE
        self.is_layer_sparse = (
            has_moe
            and layer_id >= first_k_dense
            and (layer_id - first_k_dense) % moe_layer_freq == 0
        )
        is_previous_layer_sparse = (
            has_moe
            and layer_id > 0
            and (layer_id - 1) >= first_k_dense
            and (layer_id - 1 - first_k_dense) % moe_layer_freq == 0
        )
        is_next_layer_sparse = (
            has_moe
            and layer_id < config.num_hidden_layers - 1
            and (layer_id + 1) >= first_k_dense
            and (layer_id + 1 - first_k_dense) % moe_layer_freq == 0
        )

        if self.is_layer_sparse:
            self.mlp = SarvamMoESparseMoeBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = SarvamMoEMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=False,
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_tp_size = get_attention_tp_size()
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            qkv_latent_func=self.self_attn.prepare_qkv_latent,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码器层前向传播"""
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )
        if (
            not self.is_layer_sparse
            and self.attn_tp_size > 1
            and not use_reduce_scatter
            and not should_allreduce_fusion
        ):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual


class SarvamMLAModel(nn.Module):
    """Sarvam MLA模型"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化Sarvam MLA模型"""
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = nn.Identity()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SarvamMoEMLADecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> torch.Tensor:
        """Sarvam MLA模型前向传播"""
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class SarvamMLAForCausalLM(nn.Module):
    """Sarvam MLA因果语言模型"""
    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化Sarvam MLA因果语言模型"""
        super().__init__()
        self._remap_config(config)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = SarvamMLAModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @staticmethod
    def _remap_config(config: PretrainedConfig) -> None:
        """重新映射配置，设置默认值"""
        defaults = {
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "n_group": 1,
            "topk_group": 1,
            "router_dtype": "fp32",
            "routed_scaling_factor": 2.5,
            "score_function": "sigmoid",
            "norm_topk_prob": True,
            "topk_method": "noaux_tc",
        }
        for attr, default in defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, default)

    @property
    def start_layer(self):
        """起始层"""
        return self.model.start_layer

    @property
    def end_layer(self):
        """结束层"""
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入"""
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> LogitsProcessorOutput:
        """Sarvam MLA因果语言模型前向传播"""
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        split_interval: Tuple[int, int],  # 分割区间
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> Optional[LogitsProcessorOutput]:
        """分块预填充前向传播"""
        start, end = split_interval
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
            forward_batch.residual = None

        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            if forward_batch.residual is None:
                hidden_states = self.model.norm(forward_batch.hidden_states)
            else:
                hidden_states, _ = self.model.norm(
                    forward_batch.hidden_states, forward_batch.residual
                )
            forward_batch.hidden_states = hidden_states
            return self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        return None

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置配置"""
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=getattr(config, "n_group", None),
        )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ) -> None:
        """加载模型权重"""
        del is_nextn
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            if ".mlp.gate.e_score_correction_bias" in name:
                name = name.replace(
                    ".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"
                )

            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                is_stacked = True
                break
            if is_stacked:
                continue

            is_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    loaded_weight,
                    mapped_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                is_expert = True
                break
            if is_expert:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self._set_mla_wkc_wvc()  # 设置MLA吸收权重
        if not hasattr(self, "routed_experts_weights_of_layer"):
            self.routed_experts_weights_of_layer = {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, SarvamMoESparseMoeBlock)
            }

    def _set_mla_wkc_wvc(self) -> None:
        """设置MLA的K吸收和V吸收权重"""
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            if not hasattr(self_attn, "kv_b_proj") or self_attn.kv_b_proj is None:
                continue

            w = self_attn.kv_b_proj.weight.data
            weight_scale = None
            if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):  # FP8权重
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.kv_b_proj.weight_scale is not None
                ):
                    weight_scale = self_attn.kv_b_proj.weight_scale
                elif (
                    hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    and self_attn.kv_b_proj.weight_scale_inv is not None
                ):
                    weight_scale = self_attn.kv_b_proj.weight_scale_inv
                elif (
                    hasattr(self_attn.kv_b_proj, "scale")
                    and self_attn.kv_b_proj.scale is not None
                ):
                    weight_scale = self_attn.kv_b_proj.scale

            w_reshaped = w.unflatten(
                0,
                (
                    self_attn.num_local_heads,
                    self_attn.qk_nope_head_dim + self_attn.v_head_dim,
                ),
            )  # 重塑权重
            w_kc, w_vc = w_reshaped.split(
                [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1
            )  # 分离K和V吸收权重
            self_attn.w_kc = bind_or_assign(
                self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            )  # 设置K吸收权重
            self_attn.w_vc = bind_or_assign(
                self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
            )  # 设置V吸收权重
            if weight_scale is not None:
                self_attn.w_scale = weight_scale  # 设置权重缩放


class SarvamMoEForCausalLM(BailingMoEForCausalLM):
    """Sarvam MoE因果语言模型，继承自百灵MoE"""
    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次
        split_interval: Tuple[int, int],  # 分割区间
        input_embeds: torch.Tensor = None,  # 输入嵌入
    ) -> Optional[LogitsProcessorOutput]:
        """分块预填充前向传播"""
        start, end = split_interval

        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.word_embeddings(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
            forward_batch.residual = None

        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            if forward_batch.residual is None:
                hidden_states = self.model.norm(forward_batch.hidden_states)
            else:
                hidden_states, _ = self.model.norm(
                    forward_batch.hidden_states, forward_batch.residual
                )
            forward_batch.hidden_states = hidden_states

            return self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )

        return None


EntryClass = [SarvamMLAForCausalLM, SarvamMoEForCausalLM]  # 模型注册入口类列表
