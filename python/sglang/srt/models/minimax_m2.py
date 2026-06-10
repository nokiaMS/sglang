# MiniMax M2 模型推理实现文件
# 本文件实现了仅推理的 MiniMax M2 模型，兼容 HuggingFace 权重格式
# 包含 RMSNorm TP、QK归一化、MoE专家混合、注意力机制等核心组件
# 支持张量并行(TP)、专家并行(EP)、流水线并行(PP)和双批次重叠(TBO)

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

# Adapted from DeepSeek and Mixtral implementation
"""Inference-only MiniMax M2 model compatible with HuggingFace weights."""

import logging  # 导入日志模块
from contextlib import nullcontext  # 导入空上下文管理器
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.jit_kernel.all_reduce import (  # 导入融合并行QK归一化相关
    fused_parallel_qknorm,  # 融合并行QK归一化
    get_fused_parallel_qknorm_max_occupancy,  # 获取融合并行QK归一化最大占用率
)
from sglang.kernel_api_logging import debug_kernel_api  # 导入内核API调试
from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo  # 导入双批次重叠模型前向
from sglang.srt.distributed import (  # 导入分布式相关
    get_moe_expert_parallel_world_size,  # 获取MoE专家并行世界大小
    get_pp_group,  # 获取流水线并行组
    get_tensor_model_parallel_world_size,  # 获取张量模型并行世界大小
    tensor_model_parallel_all_reduce,  # 张量模型并行全归约
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 导入专家分布记录器
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo  # 导入专家位置调度信息
from sglang.srt.layers.communicator import (  # 导入层通信器
    LayerCommunicator,  # 层通信器
    LayerScatterModes,  # 层散射模式
    ScatterMode,  # 散射模式
)
from sglang.srt.layers.dp_attention import (  # 导入DP注意力相关
    attn_tp_all_reduce,  # 注意力TP全归约
    get_attention_tp_group,  # 获取注意力TP组
    get_attention_tp_rank,  # 获取注意力TP秩
    get_attention_tp_size,  # 获取注意力TP大小
    is_dp_attention_enabled,  # 是否启用DP注意力
)
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化
from sglang.srt.layers.linear import (  # 导入线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入逻辑处理器
from sglang.srt.layers.moe import (  # 导入MoE相关
    get_moe_a2a_backend,  # 获取MoE全对全后端
    should_skip_post_experts_all_reduce,  # 是否跳过专家后全归约
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class  # 导入MoE实现类获取
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合MoE Triton层
from sglang.srt.layers.moe.topk import TopK  # 导入TopK路由
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id  # 导入PP缺失层和层ID获取
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    maybe_remap_kv_scale_name,  # 可能重映射KV缩放名称
    narrow_padded_param_and_loaded_weight,  # 缩窄填充参数和加载权重
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数

# get_bool_env_var is defined in sglang.srt.utils.common, not sglang.srt.distributed.
# Importing from the wrong module causes this file to fail import, which prevents the
# native MiniMaxM2ForCausalLM from registering in ModelRegistry. The fallback to the
# transformers wrapper then crashes on config.rope_parameters (transformers v5 issue).
# Other files (custom_all_reduce.py, hf_transformers_utils.py) also use sglang.srt.utils.
from sglang.srt.utils import (  # 导入工具函数
    BumpAllocator,  # 凸包分配器
    add_prefix,  # 添加前缀
    cpu_has_amx_support,  # CPU是否支持AMX
    get_bool_env_var,  # 获取布尔环境变量
    get_compiler_backend,  # 获取编译器后端
    is_cpu,  # 是否CPU
    is_cuda,  # 是否CUDA
    is_non_idle_and_non_empty,  # 是否非空闲且非空
    is_npu,  # 是否NPU
    make_layers,  # 创建层
)
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册
from sglang.srt.utils.hf_transformers_utils import get_rope_config  # 导入RoPE配置获取

logger = logging.getLogger(__name__)  # 创建日志记录器
_is_cpu = is_cpu()  # 检测是否CPU
_is_amx_available = cpu_has_amx_support()  # 检测CPU是否支持AMX
_is_cuda = is_cuda()  # 检测是否CUDA
_is_npu = is_npu()  # 检测是否NPU

if _is_npu:  # 如果是NPU
    from sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope import split_qkv_tp_rmsnorm_rope  # 导入NPU融合内核


@triton.jit
def rmsnorm_sumsq_kernel_serial(
    x1_ptr,  # T* [B, D]  输入张量1指针
    x2_ptr,  # T* [B, D]  输入张量2指针
    stride_x1,  # int  x1的步长
    stride_x2,  # int  x2的步长
    sum_sq_ptr,  # float* [B]  平方和输出指针
    B,  # int  批次大小
    D1,  # int  x1的维度
    D2,  # int  x2的维度
    BLOCK_SIZE1: tl.constexpr,  # x1的块大小
    BLOCK_SIZE2: tl.constexpr,  # x2的块大小
):
    """计算两个输入张量的逐行平方和的Triton内核（串行版本）"""
    row_id = tl.program_id(0)  # 获取当前行ID
    x1_row = x1_ptr + row_id * stride_x1  # 计算x1行偏移
    x2_row = x2_ptr + row_id * stride_x2  # 计算x2行偏移

    offsets1 = tl.arange(0, BLOCK_SIZE1)  # x1的偏移量范围
    mask1 = offsets1 < D1  # x1的掩码
    offsets2 = tl.arange(0, BLOCK_SIZE2)  # x2的偏移量范围
    mask2 = offsets2 < D2  # x2的掩码

    x1 = tl.load(x1_row + offsets1, mask=mask1, other=0.0)  # 加载x1数据
    x2 = tl.load(x2_row + offsets2, mask=mask2, other=0.0)  # 加载x2数据

    x1_f32 = x1.to(tl.float32)  # 转换x1为float32
    sum_sq1 = tl.sum(x1_f32 * x1_f32, axis=0)  # 计算x1平方和

    x2_f32 = x2.to(tl.float32)  # 转换x2为float32
    sum_sq2 = tl.sum(x2_f32 * x2_f32, axis=0)  # 计算x2平方和

    tl.store(sum_sq_ptr + row_id, sum_sq1)  # 存储x1平方和
    tl.store(sum_sq_ptr + row_id + B, sum_sq2)  # 存储x2平方和


@triton.jit
def rmsnorm_apply_kernel_serial(
    x1_ptr,  # T* [B, D]  输入张量1指针
    x2_ptr,  # T* [B, D]  输入张量2指针
    w1_ptr,  # T* [D]  权重1指针
    w2_ptr,  # T* [D]  权重2指针
    sum_sq_ptr,  # float* [B]  平方和指针
    out1_ptr,  # T* [B, D]  输出1指针
    out2_ptr,  # T* [B, D]  输出2指针
    B,  # int  批次大小
    D1,  # int  x1的维度
    D2,  # int  x2的维度
    stride_x1,  # int  x1的步长
    stride_x2,  # int  x2的步长
    tp_world,  # int  张量并行世界大小
    eps,  # float  epsilon值
    BLOCK_SIZE1: tl.constexpr,  # x1的块大小
    BLOCK_SIZE2: tl.constexpr,  # x2的块大小
):
    """应用RMS归一化的Triton内核（串行版本），根据预计算的平方和进行归一化"""
    row_id = tl.program_id(0)  # 获取当前行ID
    x1_row = x1_ptr + row_id * stride_x1  # 计算x1行偏移
    x2_row = x2_ptr + row_id * stride_x2  # 计算x2行偏移
    out1_row = out1_ptr + row_id * stride_x1  # 计算输出1行偏移
    out2_row = out2_ptr + row_id * stride_x2  # 计算输出2行偏移

    sum_sq1 = tl.load(sum_sq_ptr + row_id)  # 加载x1平方和
    sum_sq2 = tl.load(sum_sq_ptr + row_id + B)  # 加载x2平方和
    inv_rms1 = tl.rsqrt(sum_sq1 / D1 / tp_world + eps)  # 计算x1逆RMS
    inv_rms2 = tl.rsqrt(sum_sq2 / D2 / tp_world + eps)  # 计算x2逆RMS

    offsets1 = tl.arange(0, BLOCK_SIZE1)  # x1的偏移量范围
    offsets2 = tl.arange(0, BLOCK_SIZE2)  # x2的偏移量范围

    mask1 = offsets1 < D1  # x1的掩码
    mask2 = offsets2 < D2  # x2的掩码

    x1 = tl.load(x1_row + offsets1, mask=mask1, other=0.0)  # 加载x1数据
    w1 = tl.load(w1_ptr + offsets1, mask=mask1, other=1.0)  # 加载w1权重
    x2 = tl.load(x2_row + offsets2, mask=mask2, other=0.0)  # 加载x2数据
    w2 = tl.load(w2_ptr + offsets2, mask=mask2, other=1.0)  # 加载w2权重

    out1 = (x1.to(tl.float32) * inv_rms1 * w1.to(tl.float32)).to(x1.dtype)  # 计算输出1
    out2 = (x2.to(tl.float32) * inv_rms2 * w2.to(tl.float32)).to(x2.dtype)  # 计算输出2
    tl.store(out1_row + offsets1, out1, mask=mask1)  # 存储输出1
    tl.store(out2_row + offsets2, out2, mask=mask2)  # 存储输出2


@debug_kernel_api
def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """串行计算两个张量的RMS平方和，用于后续归一化"""
    assert x1.is_cuda and x2.is_cuda  # 确保输入在CUDA上
    B, D1 = x1.shape  # 获取x1形状
    B2, D2 = x2.shape  # 获取x2形状
    assert B == B2  # 确保批次大小一致

    stride_x1 = x1.stride(0)  # 获取x1步长
    stride_x2 = x2.stride(0)  # 获取x2步长

    # We found that custom all-reduce `sglang::cross_device_reduce_1stage`
    # is much faster than the nccl all-reduce in torch.
    # However, `should_custom_ar` checks if the reduced buffer is 16-byte aligned.
    # RMSNormTP reduces a [B, 2] fp32 tensor, so we pad the total element count to
    # satisfy the alignment requirement.
    B_padded = (B + B2 + 3) // 4 * 4  # 填充批次大小以满足对齐要求

    sum_sq = torch.empty(B_padded, device=x1.device, dtype=torch.float32)  # 分配平方和缓冲区

    BLOCK_SIZE1 = triton.next_power_of_2(D1)  # 计算x1块大小
    BLOCK_SIZE2 = triton.next_power_of_2(D2)  # 计算x2块大小

    grid = (B,)  # 设置网格大小

    rmsnorm_sumsq_kernel_serial[grid](  # 启动平方和计算内核
        x1,
        x2,
        stride_x1,
        stride_x2,
        sum_sq,
        B,
        D1,
        D2,
        BLOCK_SIZE1,
        BLOCK_SIZE2,
    )
    return sum_sq  # 返回平方和


@debug_kernel_api
def rms_apply_serial(
    x1: torch.Tensor,  # 输入张量1
    x2: torch.Tensor,  # 输入张量2
    w1: torch.Tensor,  # 权重1
    w2: torch.Tensor,  # 权重2
    sum_sq: torch.Tensor,  # 平方和
    tp_world: int = 1,  # 张量并行世界大小
    eps: float = 1e-5,  # epsilon值
) -> torch.Tensor:
    """串行应用RMS归一化，根据预计算的平方和对两个张量进行归一化"""
    assert x1.is_cuda and x2.is_cuda and w1.is_cuda and w2.is_cuda and sum_sq.is_cuda  # 确保输入在CUDA上
    B, D1 = x1.shape  # 获取x1形状
    B2, D2 = x2.shape  # 获取x2形状
    assert B == B2  # 确保批次大小一致

    stride_x1 = x1.stride(0)  # 获取x1步长
    stride_x2 = x2.stride(0)  # 获取x2步长
    out1 = torch.empty(B, D1, device=x1.device, dtype=x1.dtype)  # 分配输出1缓冲区
    out2 = torch.empty(B, D2, device=x2.device, dtype=x2.dtype)  # 分配输出2缓冲区

    BLOCK_SIZE1 = triton.next_power_of_2(D1)  # 计算x1块大小
    BLOCK_SIZE2 = triton.next_power_of_2(D2)  # 计算x2块大小

    grid = (B,)  # 设置网格大小

    rmsnorm_apply_kernel_serial[grid](  # 启动RMS归一化应用内核
        x1,
        x2,
        w1,
        w2,
        sum_sq,
        out1,
        out2,
        B,
        D1,
        D2,
        stride_x1,
        stride_x2,
        tp_world,
        eps,
        BLOCK_SIZE1,
        BLOCK_SIZE2,
    )
    return out1, out2  # 返回归一化后的两个张量


class MiniMaxM2RMSNormTP(nn.Module):
    """RMSNorm with Tensor Parallel support for QK normalization."""
    """支持张量并行的RMS归一化层，用于QK归一化"""

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6) -> None:
        """初始化RMSNormTP层，配置TP分片和权重参数"""
        super().__init__()
        self.attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小
        self.attn_tp_rank = get_attention_tp_rank()  # 获取注意力TP秩

        # Align with QKVParallelLinear pattern
        if self.attn_tp_size >= num_heads:  # 如果TP大小大于等于头数
            assert (
                self.attn_tp_size % num_heads == 0
            ), f"attn_tp_size ({self.attn_tp_size}) must be divisible by num_heads ({num_heads})"
            self.num_heads = 1  # 每个分片1个头
            self.num_head_replicas = self.attn_tp_size // num_heads  # 头副本数
        else:  # 如果TP大小小于头数
            assert (
                num_heads % self.attn_tp_size == 0
            ), f"num_heads ({num_heads}) must be divisible by attn_tp_size ({self.attn_tp_size})"
            self.num_heads = num_heads // self.attn_tp_size  # 每个分片的头数
            self.num_head_replicas = 1  # 副本数为1

        self.head_dim = hidden_size // num_heads  # 计算头维度

        # Weight parameter is sharded across TP ranks
        self.weight = nn.Parameter(torch.ones(self.num_heads * self.head_dim))  # 初始化权重为1
        self.weight.weight_loader = self.weight_loader  # 设置自定义权重加载器
        self.variance_epsilon = eps  # 保存epsilon值

    def weight_loader(
        self,
        param: nn.Parameter,  # 参数
        loaded_weight: torch.Tensor,  # 加载的权重
    ) -> None:
        """Custom weight loader that handles TP sharding."""
        """自定义权重加载器，处理TP分片加载"""
        shard_id = self.attn_tp_rank // self.num_head_replicas  # 计算分片ID
        shard_size = param.data.shape[0]  # 获取分片大小

        if _is_cpu and _is_amx_available:  # 如果是CPU且支持AMX
            # Handle uneven TP sharding on CPU
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param.data,
                loaded_weight,
                0,  # param_data_start
                shard_id * shard_size,  # weight_start
                0,  # shard_axis
                shard_size,
            )
            param_data.copy_(loaded_weight)  # 复制权重
            return

        shard_end = (shard_id + 1) * shard_size  # 计算分片结束位置
        assert shard_end <= loaded_weight.shape[0], (  # 确保分片不越界
            f"Weight shard out of bounds: shard [{shard_id * shard_size}:{shard_end}] "
            f"exceeds loaded_weight size {loaded_weight.shape[0]} "
            f"(attn_tp_rank={self.attn_tp_rank}, num_head_replicas={self.num_head_replicas})"
        )
        shard = slice(shard_id * shard_size, shard_end)  # 创建分片切片
        param.data.copy_(loaded_weight[shard])  # 复制分片权重

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def forward(
        self,
        x: torch.Tensor,  # 输入张量
        residual: Optional[torch.Tensor] = None,  # 残差张量
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with TP-aware variance computation."""
        """带TP感知方差计算的前向传播"""
        assert residual is None, "RMSNormTP does not support residual connection."  # 不支持残差

        orig_dtype = x.dtype  # 保存原始数据类型
        x = x.to(torch.float32)  # 转换为float32

        # Compute variance across the full dimension (not just local shard)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)  # 计算方差

        if self.attn_tp_size > 1:  # 如果TP大小大于1
            # All-reduce variance across TP ranks to get global variance
            variance = attn_tp_all_reduce(variance) / self.attn_tp_size  # 跨TP秩全归约方差

        # Normalize and apply local weight shard
        x = x * torch.rsqrt(variance + self.variance_epsilon)  # 应用RMS归一化
        x = (x * self.weight).to(orig_dtype)  # 应用权重并恢复数据类型

        return x  # 返回归一化结果


@register_custom_op(mutates_args=["q", "k"])
def fused_tp_qknorm(
    counter: int,  # 计数器
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    q_weight: torch.Tensor,  # Q权重
    k_weight: torch.Tensor,  # K权重
    eps: float,  # epsilon值
) -> None:
    """融合TP QK归一化自定义算子，调用融合并行QK归一化内核"""
    return fused_parallel_qknorm(
        MiniMaxM2QKRMSNorm.COMM_MAP[counter].obj,  # 获取通信对象
        q,
        k,
        q_weight,
        k_weight,
        eps=eps,
    )


class MiniMaxM2QKRMSNorm:
    """MiniMax M2 QK RMS归一化实现，支持朴素和融合两种模式"""
    COUNTER = 0  # 计数器
    COMM_MAP: Dict[int, Any] = {}  # 通信映射表

    def __init__(
        self,
        q_norm: MiniMaxM2RMSNormTP,  # Q归一化层
        k_norm: MiniMaxM2RMSNormTP,  # K归一化层
    ) -> None:
        """初始化QK RMS归一化，选择朴素或融合实现"""
        assert q_norm.variance_epsilon == k_norm.variance_epsilon  # 确保epsilon一致
        self._q_norm = q_norm  # 保存Q归一化
        self._k_norm = k_norm  # 保存K归一化
        self._world_size = self._q_norm.attn_tp_size  # 保存世界大小
        self._eps = q_norm.variance_epsilon  # 保存epsilon
        use_fused_norm = get_bool_env_var("SGLANG_USE_FUSED_PARALLEL_QKNORM")  # 是否使用融合归一化

        self._forward_impl = self._forward_naive  # 默认使用朴素实现
        if self._world_size > 1 and _is_cuda and use_fused_norm:  # 如果TP>1且CUDA且启用融合
            occupancy = get_fused_parallel_qknorm_max_occupancy(
                q_norm.weight.dtype,
                self._world_size,
                # NOTE: we need full dimension
                q_dim=q_norm.weight.shape[0] * self._world_size,  # Q维度
                k_dim=k_norm.weight.shape[0] * self._world_size,  # K维度
            )
            counter = MiniMaxM2QKRMSNorm._get_comm(q_norm.weight.device, occupancy)  # 获取通信计数器
            if counter is not None:  # 如果可用
                self._counter = counter  # 保存计数器
                self._forward_impl = self._forward_fused  # 使用融合实现
        elif _is_cpu and _is_amx_available:  # 如果是CPU且支持AMX
            self._forward_impl = self._forward_cpu  # 使用CPU实现

    @lru_cache
    @staticmethod
    def _get_comm(device: torch.device, occupancy: int):
        """获取或创建用于融合QK归一化的通信对象"""
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        props = torch.cuda.get_device_properties(device)  # 获取设备属性
        # probe the maximum tokens for one prefill
        server_args = get_global_server_args()  # 获取服务器参数
        max_tokens = server_args.chunked_prefill_size  # 分块预填充大小
        if max_tokens is None:  # 如果未设置
            max_tokens = server_args.model_config.context_len  # 使用上下文长度
        max_tokens = max(max_tokens, server_args.max_prefill_tokens)  # 取最大值
        logger.info(f"[AR] Using CustomAllReduceV2 for MiniMaxM2 with {max_tokens = }")  # 记录日志
        ALIGN = 512  # 对齐大小
        # typically, this should not exceed 1M, since max_tokens is usually less than 16384
        max_size = ((8 * max_tokens + ALIGN - 1) // ALIGN) * ALIGN  # 计算最大尺寸
        comm = CustomAllReduceV2(
            group=get_attention_tp_group().cpu_group,  # CPU组
            device=device,  # 设备
            max_pull_size=0,  # 最大拉取大小
            max_pull_blocks=0,  # 最大拉取块数
            max_push_size=max_size,  # 最大推送大小
            max_push_blocks=props.multi_processor_count * occupancy,  # 最大推送块数
        )
        counter = MiniMaxM2QKRMSNorm.COUNTER  # 获取当前计数器
        MiniMaxM2QKRMSNorm.COUNTER += 1  # 递增计数器
        MiniMaxM2QKRMSNorm.COMM_MAP[counter] = comm  # 存储通信对象
        return counter if not comm.disabled else None  # 如果未禁用则返回计数器

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """QK RMS归一化的前向传播入口"""
        return self._forward_impl(q, k)  # 调用选定的实现

    def _forward_naive(self, q: torch.Tensor, k: torch.Tensor):
        """朴素QK RMS归一化实现，使用串行内核"""
        q, k = q.contiguous(), k.contiguous()  # 确保连续内存
        sum_sq = rms_sumsq_serial(q, k)  # 计算平方和
        if self._world_size > 1:  # 如果TP>1
            sum_sq = attn_tp_all_reduce(sum_sq)  # 全归约平方和
        return rms_apply_serial(  # 应用RMS归一化
            q,
            k,
            self._q_norm.weight,
            self._k_norm.weight,
            sum_sq,
            self._world_size,
            self._eps,
        )

    def _forward_fused(self, q: torch.Tensor, k: torch.Tensor):
        """融合QK RMS归一化实现，使用自定义融合内核"""
        fused_tp_qknorm(
            self._counter,
            q,
            k,
            self._q_norm.weight,
            self._k_norm.weight,
            self._eps,
        )
        return q, k  # 返回归一化后的Q和K

    def _forward_cpu(self, q: torch.Tensor, k: torch.Tensor):
        """CPU上的QK RMS归一化实现"""
        # TODO: add c++ kernel for cpu
        q = self._q_norm(q.contiguous())  # 对Q进行归一化
        k = self._k_norm(k.contiguous())  # 对K进行归一化
        return q, k  # 返回归一化后的Q和K


class MiniMaxM2MoE(nn.Module):
    """MiniMax MoE implementation using DeepEP for Expert Parallel support."""
    """MiniMax MoE实现，使用DeepEP支持专家并行"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化MiniMax MoE层，配置路由、专家和门控"""
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取TP大小
        if self.tp_size > config.num_local_experts:  # 如果TP大于专家数
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )
        self.use_routing_bias = getattr(config, "use_routing_bias", False)  # 是否使用路由偏置
        if self.use_routing_bias:  # 如果使用路由偏置
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)  # 创建偏置参数
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM2MoE.ebias_weight_loader  # 设置权重加载器
            )
        else:
            self.e_score_correction_bias = None  # 不使用偏置

        self.experts = get_moe_impl_class(quant_config)(  # 创建专家实现
            num_experts=config.num_local_experts
            + get_global_server_args().ep_num_redundant_experts,  # 专家数加冗余
            top_k=config.num_experts_per_tok,  # Top-K
            hidden_size=config.hidden_size,  # 隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("experts", prefix),  # 前缀
        )
        self.topk = TopK(  # 创建TopK路由
            top_k=config.num_experts_per_tok,  # Top-K值
            renormalize=True,  # 重归一化
            scoring_func=config.scoring_func,  # 评分函数
            correction_bias=self.e_score_correction_bias,  # 修正偏置
            routed_scaling_factor=1.0,  # 路由缩放因子
        )

        self.gate = ReplicatedLinear(  # 创建门控线性层
            config.hidden_size,  # 输入大小
            config.num_local_experts,  # 输出大小
            bias=False,  # 无偏置
            params_dtype=torch.float32,  # 参数数据类型
            quant_config=None,  # 无量化
            prefix=add_prefix("gate", prefix),  # 前缀
        )

        self.layer_id = layer_id  # 保存层ID

        if get_moe_a2a_backend().is_deepep():  # 如果使用DeepEP
            self.ep_size = get_moe_expert_parallel_world_size()  # 获取EP大小
            self.top_k = config.num_experts_per_tok  # 保存Top-K

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """加载专家偏置权重的静态方法"""
        assert param.size() == loaded_weight.size()  # 确保尺寸匹配
        param.data.copy_(loaded_weight.to(torch.float32))  # 复制权重

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: Optional[ForwardBatch] = None,  # 前向批次
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """MoE前向传播，根据后端选择普通或DeepEP模式"""
        if (
            not get_moe_a2a_backend().is_deepep()
            and not get_moe_a2a_backend().is_ascend_fuseep()
        ):
            return self.forward_normal(  # 普通模式
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        else:
            return self.forward_deepep(hidden_states, forward_batch)  # DeepEP模式

    def forward_normal(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        should_allreduce_fusion: bool = False,  # 是否融合全归约
        use_reduce_scatter: bool = False,  # 是否使用reduce-scatter
    ) -> torch.Tensor:
        """普通MoE前向传播，不使用DeepEP"""
        num_tokens, hidden_dim = hidden_states.shape  # 获取token数和隐藏维度
        hidden_states = hidden_states.view(-1, hidden_dim)  # 重塑形状

        if hidden_states.shape[0] > 0:  # 如果有token
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states.to(torch.float32))  # 计算路由logits
            topk_output = self.topk(hidden_states, router_logits)  # Top-K选择
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)  # 空Top-K输出

        final_hidden_states = self.experts(hidden_states, topk_output)  # 专家计算
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(  # 如果需要全归约
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # 全归约

        return final_hidden_states.view(num_tokens, hidden_dim)  # 返回结果

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """DeepEP模式MoE前向传播，支持专家并行"""
        if hidden_states.shape[0] > 0:  # 如果有token
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states.to(torch.float32))  # 计算路由logits
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,  # 非填充token数
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,  # 层ID
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(device=hidden_states.device)  # 空Top-K输出
        final_hidden_states = self.experts(
            hidden_states=hidden_states,  # 隐藏状态
            topk_output=topk_output,  # Top-K输出
        )

        return final_hidden_states  # 返回结果

    # TBO Operations for MiniMax MoE
    def op_gate(self, state):
        """Gate operation for TBO - compute router logits"""
        """TBO门控操作 - 计算路由logits"""
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):  # router_logits: (num_tokens, num_experts)
            state.router_logits, _ = self.gate(state.hidden_states_mlp_input)  # 计算路由logits
        else:
            state.router_logits = None  # 空闲时设为None

    def op_select_experts(self, state):
        """Expert selection operation for TBO"""
        """TBO专家选择操作"""
        router_logits = state.pop("router_logits")  # 弹出路由logits
        hidden_states = state.hidden_states_mlp_input  # 获取隐藏状态

        if router_logits is not None:  # 如果有路由logits
            ctx = (
                nullcontext()
                if not get_global_server_args().disable_piecewise_cuda_graph
                else get_global_expert_distribution_recorder().with_current_layer(
                    self.layer_id
                )
            )
            with ctx:  # 上下文管理
                state.topk_weights_local, state.topk_idx_local, _ = self.topk(
                    hidden_states=hidden_states,  # 隐藏状态
                    router_logits=router_logits,  # 路由logits
                    num_token_non_padded=state.forward_batch.num_token_non_padded,  # 非填充token数
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,  # 层ID
                    ),
                )
        else:
            state.topk_idx_local = torch.full(  # 创建全-1的索引张量
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            state.topk_weights_local = torch.empty(  # 创建空权重张量
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

    def op_dispatch_a(self, state):
        """Dispatch A operation for TBO - start async dispatch"""
        """TBO分发A操作 - 开始异步分发"""
        if self.ep_size > 1:  # 如果EP>1
            self.experts.deepep_dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),  # 隐藏状态
                topk_idx=state.pop("topk_idx_local"),  # Top-K索引
                topk_weights=state.pop("topk_weights_local"),  # Top-K权重
                forward_batch=state.forward_batch,  # 前向批次
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )

    def op_dispatch_b(self, state):
        """Dispatch B operation for TBO - complete async dispatch"""
        """TBO分发B操作 - 完成异步分发"""
        if self.ep_size > 1:  # 如果EP>1
            ctx = (
                nullcontext()
                if not get_global_server_args().disable_piecewise_cuda_graph
                else get_global_expert_distribution_recorder().with_current_layer(
                    self.layer_id
                )
            )
            with ctx:  # 上下文管理
                state.dispatch_output = self.experts.deepep_dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
                )

    def op_experts(self, state):
        """Expert computation for TBO"""
        """TBO专家计算操作"""
        state.hidden_states_experts_output = self.experts.moe_impl(
            dispatch_output=state.dispatch_output,  # 分发输出
        )

    def op_combine_a(self, state):
        """Combine A operation for TBO - start async combine"""
        """TBO合并A操作 - 开始异步合并"""
        if self.ep_size > 1:  # 如果EP>1
            self.experts.deepep_dispatcher.combine_a(
                hidden_states=state.pop("hidden_states_experts_output"),  # 专家输出
                topk_idx=state.dispatch_output.topk_idx,  # Top-K索引
                topk_weights=state.dispatch_output.topk_weights,  # Top-K权重
                forward_batch=state.forward_batch,  # 前向批次
                tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
            )
            state.pop("dispatch_output")  # 弹出分发输出

    def op_combine_b(self, state):
        """Combine B operation for TBO - complete async combine"""
        """TBO合并B操作 - 完成异步合并"""
        if self.ep_size > 1:  # 如果EP>1
            state.hidden_states_after_combine = (
                self.experts.deepep_dispatcher.combine_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),  # TBO子批次索引
                )
            )

    def op_output(self, state):
        """Output operation for TBO - final MLP output"""
        """TBO输出操作 - 最终MLP输出"""
        final_hidden_states = state.pop("hidden_states_after_combine")  # 弹出合并后状态
        # MiniMax doesn't have shared experts like DeepSeek, so no need to add them
        state.hidden_states_mlp_output = final_hidden_states  # 设置MLP输出


class MiniMaxM2Attention(nn.Module):
    """MiniMax Attention implementation with QK normalization and partial RoPE."""
    """MiniMax注意力实现，支持QK归一化和部分旋转位置编码"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int = 0,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化MiniMax注意力层，配置QKV投影、RoPE和QK归一化"""
        super().__init__()
        self.hidden_size = config.hidden_size  # 隐藏层大小

        # Use attention TP rank/size for dp-attention support
        attn_tp_rank = get_attention_tp_rank()  # 获取注意力TP秩
        attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小

        # Get dimensions from config
        self.total_num_heads = config.num_attention_heads  # 总注意力头数
        assert self.total_num_heads % attn_tp_size == 0  # 确保可被TP整除
        self.num_heads = self.total_num_heads // attn_tp_size  # 每个分片的头数
        self.total_num_kv_heads = config.num_key_value_heads  # 总KV头数

        if self.total_num_kv_heads >= attn_tp_size:  # KV头数>=TP大小时
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:  # KV头数<TP大小时
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)  # 每个分片KV头数

        # Use head_dim from config if available, otherwise calculate
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )  # 头维度
        self.q_size = self.num_heads * self.head_dim  # Q大小
        self.kv_size = self.num_kv_heads * self.head_dim  # KV大小
        self.scaling = self.head_dim**-0.5  # 缩放因子

        # RoPE settings - support partial RoPE
        # FIXME: minimax_m2 config use external config that not compatible with transformers v5
        self.rope_theta, self.rope_scaling = get_rope_config(config)  # 获取RoPE配置
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)  # 最大位置编码
        self.rotary_dim = getattr(
            config, "rotary_dim", self.head_dim
        )  # MiniMax uses rotary_dim=64  旋转维度

        # QK Normalization settings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)  # 是否使用QK归一化
        self.qk_norm_type = getattr(config, "qk_norm_type", "per_layer")  # QK归一化类型

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,  # 无偏置
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # TP秩
            tp_size=attn_tp_size,  # TP大小
            prefix=add_prefix("qkv_proj", prefix),  # 前缀
        )

        self.o_proj = RowParallelLinear(  # 输出投影
            self.total_num_heads * self.head_dim,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=False,  # 无偏置
            reduce_results=False,  # 不自动归约
            quant_config=quant_config,  # 量化配置
            tp_rank=attn_tp_rank,  # TP秩
            tp_size=attn_tp_size,  # TP大小
            prefix=add_prefix("o_proj", prefix),  # 前缀
        )

        # Setup RoPE with partial rotary dimension
        self.rotary_emb = get_rope(
            self.head_dim,  # 头维度
            rotary_dim=self.rotary_dim,  # Use partial rotary dimension  使用部分旋转维度
            max_position=self.max_position_embeddings,  # 最大位置
            base=self.rope_theta,  # 基础频率
            rope_scaling=self.rope_scaling,  # 缩放配置
        )

        # QK Normalization layers
        if self.use_qk_norm:  # 如果使用QK归一化
            if self.qk_norm_type == "per_layer":  # 按层归一化
                # Use RMSNormTP for proper tensor parallel support
                # Use total dimensions (before TP sharding) for correct normalization
                self.q_norm = MiniMaxM2RMSNormTP(
                    self.total_num_heads * self.head_dim,  # Q归一化维度
                    num_heads=self.total_num_heads,  # 头数
                    eps=config.rms_norm_eps,  # epsilon
                )
                self.k_norm = MiniMaxM2RMSNormTP(
                    self.total_num_kv_heads * self.head_dim,  # K归一化维度
                    num_heads=self.total_num_kv_heads,  # KV头数
                    eps=config.rms_norm_eps,  # epsilon
                )
                self.qk_norm_impl = MiniMaxM2QKRMSNorm(self.q_norm, self.k_norm)  # QK归一化实现
            else:
                raise ValueError(f"Unsupported qk_norm_type: {self.qk_norm_type}")  # 不支持的类型

        self.attn = RadixAttention(  # 基数注意力
            self.num_heads,  # 头数
            self.head_dim,  # 头维度
            self.scaling,  # 缩放因子
            num_kv_heads=self.num_kv_heads,  # KV头数
            layer_id=layer_id,  # 层ID
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 前缀
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ):
        """准备注意力计算的QKV，包括投影、QK归一化和RoPE"""
        if hidden_states.shape[0] == 0:  # 如果没有token
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states, forward_batch, None  # 返回空状态
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV
        if self.use_qk_norm:  # 如果使用QK归一化
            q, k = self.qk_norm_impl.forward(q, k)  # 应用QK归一化
        q, k = self.rotary_emb(positions, q, k)  # 应用RoPE
        inner_state = q, k, v, forward_batch  # 组装内部状态
        return None, forward_batch, inner_state  # 返回状态

    def forward_prepare_npu(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ):
        """NPU上的注意力准备，使用融合QKV拆分+RMSNorm+RoPE内核"""
        if hidden_states.shape[0] == 0:  # 如果没有token
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states, forward_batch, None  # 返回空状态
        qkv, _ = self.qkv_proj(hidden_states)  # QKV投影
        if self.use_qk_norm:  # 如果使用QK归一化
            cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, positions.flatten())  # 获取cos/sin缓存
            cos, sin = cos_sin.chunk(2, dim=-1)  # 拆分cos和sin
            q, k, v = split_qkv_tp_rmsnorm_rope(  # 使用NPU融合内核
                input=qkv,
                cos=cos,
                sin=sin,
                q_weight=self.q_norm.weight,  # Q归一化权重
                k_weight=self.k_norm.weight,  # K归一化权重
                q_hidden_size=self.q_size,  # Q大小
                kv_hidden_size=self.kv_size,  # KV大小
                head_dim=self.head_dim,  # 头维度
                rotary_dim=self.rotary_dim,  # 旋转维度
                eps=self.q_norm.variance_epsilon,  # epsilon
                tp_world=self.q_norm.attn_tp_size,  # TP大小
                tp_group=get_attention_tp_group().device_group,  # TP组
            )
        else:  # 不使用QK归一化
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 拆分QKV
            q, k = q.contiguous(), k.contiguous()  # 确保连续
            q, k = self.rotary_emb(positions, q, k)  # 应用RoPE

        inner_state = q, k, v, forward_batch  # 组装内部状态
        return None, forward_batch, inner_state  # 返回状态

    def forward_core(self, intermediate_state):
        """注意力核心计算，执行注意力前向和输出投影"""
        hidden_states, forward_batch, inner_state = intermediate_state  # 解包状态
        if inner_state is None:  # 如果无内部状态
            return hidden_states  # 返回原始隐藏状态
        attn_output = self.attn(*inner_state)  # 执行注意力计算
        output, _ = self.o_proj(attn_output)  # 输出投影
        return output  # 返回输出

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
    ) -> torch.Tensor:
        """MiniMax注意力前向传播，根据设备选择准备方法"""
        if not _is_npu:  # 非NPU
            s = self.forward_prepare(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:  # NPU
            s = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        return self.forward_core(s)  # 执行核心计算

    def op_prepare(self, state):
        """TBO准备操作 - 执行注意力准备阶段"""
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        """TBO核心操作 - 执行注意力核心计算"""
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )


class MiniMaxM2DecoderLayer(nn.Module):
    """MiniMax Decoder Layer implementation with MoE support."""
    """MiniMax解码器层实现，支持MoE"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化MiniMax解码器层，配置注意力和MoE"""
        super().__init__()
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.layer_id = layer_id  # 层ID

        # TBO support: All MiniMax layers are sparse (MoE)
        self.is_layer_sparse = True  # 所有层都是稀疏的(MoE)

        self.self_attn = MiniMaxM2Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.block_sparse_moe = MiniMaxM2MoE(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("block_sparse_moe", prefix),
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )  # 输入层归一化
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )  # 注意力后归一化

        is_previous_layer_sparse = True  # 前一层是稀疏的
        is_next_layer_sparse = True  # 下一层是稀疏的
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
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,  # 捕获的最后层输出
    ) -> torch.Tensor:
        """MiniMax解码器层前向传播，包含注意力和MoE"""
        # Self Attention
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )
        if not forward_batch.forward_mode.is_idle():  # 非空闲模式
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected (MLP or MoE)

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

        hidden_states = self.block_sparse_moe(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion:  # 如果融合全归约
            hidden_states._sglang_needs_allreduce_fusion = True  # 标记需要融合
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual  # 返回隐藏状态和残差

    # TBO Operations for MiniMax Decoder Layer
    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,  # 位置编码
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        residual: Optional[torch.Tensor],  # 残差
        zero_allocator: BumpAllocator,  # 零分配器
        tbo_subbatch_index: Optional[int] = None,  # TBO子批次索引
    ):
        """Communication prepare for attention - TBO operation"""
        """注意力通信准备 - TBO操作"""
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        """Communication prepare for MLP - TBO operation"""
        """MLP通信准备 - TBO操作"""
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_comm_postprocess_layer(self, state):
        """Communication postprocess for layer - TBO operation"""
        """层通信后处理 - TBO操作"""
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
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )
        return output  # 返回输出字典


class MiniMaxM2Model(nn.Module):
    """MiniMax Model implementation."""
    """MiniMax模型实现"""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化MiniMax模型，配置嵌入、层和归一化"""
        super().__init__()

        self.padding_idx = getattr(config, "pad_token_id", 0)  # 填充索引
        self.vocab_size = config.vocab_size  # 词表大小
        self.pp_group = get_pp_group()  # 获取PP组

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),
        )

        def layer_fn(idx, prefix: str) -> nn.Module:
            return MiniMaxM2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            layer_fn,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:  # 如果是最后一个PP秩
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化
        else:
            self.norm = PPMissingLayer(return_tuple=True)  # PP缺失层

        # For EAGLE3 support
        self.layers_to_capture = []  # 待捕获的层列表

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输入嵌入"""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> Union[torch.Tensor, PPProxyTensors, Tuple[torch.Tensor, list[torch.Tensor]]]:
        """MiniMax模型前向传播，支持PP和TBO"""
        if self.pp_group.is_first_rank:  # 第一个PP秩
            if input_embeds is None:
                hidden_states = self.get_input_embeddings(input_ids)  # 获取嵌入
            else:
                hidden_states = input_embeds  # 使用输入嵌入
            residual = None
        else:  # 非第一个PP秩
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]  # 从代理获取隐藏状态
            residual = pp_proxy_tensors["residual"]  # 从代理获取残差

        aux_hidden_states = []  # 辅助隐藏状态
        if forward_batch.can_run_tbo:  # 如果可以运行TBO
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:  # 非TBO模式
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if not get_global_server_args().disable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:  # 上下文管理
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions=positions,
                        forward_batch=forward_batch,
                        hidden_states=hidden_states,
                        residual=residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states if i in self.layers_to_capture else None
                        ),
                    )

        if not self.pp_group.is_last_rank:  # 非最后一个PP秩
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:  # 如果有token
            if residual is not None:  # 如果有残差
                hidden_states, _ = self.norm(hidden_states, residual)  # 带残差归一化
            else:
                hidden_states = self.norm(hidden_states)  # 无残差归一化

        if len(aux_hidden_states) == 0:  # 无辅助状态
            return hidden_states
        return hidden_states, aux_hidden_states  # 返回隐藏状态和辅助状态


class MiniMaxM2ForCausalLM(nn.Module):
    """MiniMax M2 model for causal language modeling."""
    """MiniMax M2因果语言模型"""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        """初始化MiniMax M2因果语言模型"""
        super().__init__()

        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        self.model = MiniMaxM2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        if get_pp_group().is_last_rank:  # 如果是最后一个PP秩
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)  # 逻辑处理器
        self.pp_group = get_pp_group()  # PP组

        # For EAGLE3
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取输入嵌入"""
        return self.model.get_input_embeddings(input_ids)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        """设置EAGLE3需要捕获的层"""
        if not get_pp_group().is_last_rank:  # 非最后秩不捕获
            return

        self.capture_aux_hidden_states = True  # 启用捕获
        if layer_ids is None:  # 使用默认层
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support  EAGLE3特定的层
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]  # 偏移层ID

    def get_embed_and_head(self):
        """获取嵌入权重和语言模型头权重"""
        return self.model.embed_tokens.weight, self.lm_head.weight

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        input_embeds: torch.Tensor = None,  # 输入嵌入
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # PP代理张量
    ) -> torch.Tensor:
        """MiniMax M2因果语言模型前向传播"""
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:  # 如果捕获辅助状态
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:  # 如果是最后PP秩
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights with proper mapping for MiniMax architecture."""
        """加载模型权重，支持MiniMax架构的权重映射"""

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        params_dict = dict(self.named_parameters())  # 参数字典
        loaded_params: Set[str] = set()  # 已加载参数集合
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转频率
                continue

            layer_id = get_layer_id(name)  # 获取层ID
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):  # 跳过不属于当前PP范围的层
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)  # 获取推测层索引
            if spec_layer is not None:
                continue  # skip spec decode layers for main model  跳过推测解码层

            _is_kv_scale = name.endswith(".k_scale") or name.endswith(".v_scale")  # 是否KV缩放

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # Skip kv cache scales - maybe_remap_kv_scale_name expects the
                # original checkpoint name (e.g. self_attn.k_proj.k_scale) to
                # remap it to self_attn.attn.k_scale. Renaming k_proj -> qkv_proj
                # here would break that pattern match.
                if _is_kv_scale:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name not in params_dict:
                    continue

                if name.endswith(".bias"):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:  # 非堆叠参数
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:  # 普通参数
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)  # 添加到已加载集合
        return loaded_params  # 返回已加载参数

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """获取专家位置配置的模型配置"""
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> Optional[int]:
    """根据权重名称获取推测解码层的索引"""
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None


# Entry class for model registration
EntryClass = MiniMaxM2ForCausalLM  # 模型注册入口类
