# Copyright 2023-2024 SGLang Team  # 版权所有 2023-2024 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版（"许可证"）授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的担保
# See the License for the specific language governing permissions and  # 参见许可证了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================  # ==============================================================================
# 层间通信器模块
# 实现张量并行(TP)和数据并行(DP)下的层间数据分发、聚合和通信逻辑，
# 包括ScatterMode枚举、LayerCommunicator、各种通信函数类等
import logging  # 导入日志模块
from contextlib import contextmanager  # 导入上下文管理器工具
from dataclasses import dataclass  # 导入数据类装饰器
from enum import Enum, auto  # 导入枚举类型
from functools import partial  # 导入偏函数工具
from typing import Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch库

from sglang.srt.distributed import (  # 导入分布式通信函数
    attention_tensor_model_parallel_all_reduce,  # 注意力TP全归约
    attention_tensor_model_parallel_quant_all_reduce,  # 注意力TP量化全归约
    get_tensor_model_parallel_rank,  # 获取TP rank
    get_tensor_model_parallel_world_size,  # 获取TP world size
    get_tp_group,  # 获取TP通信组
    moe_tensor_model_parallel_all_reduce,  # MoE TP全归约
    tensor_model_parallel_all_reduce,  # TP全归约
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入NCCL分配器
    use_symmetric_memory,  # 对称内存上下文管理器
)
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.attention.dsa.utils import (  # 导入DSA工具函数
    dsa_use_prefill_cp,  # DSA是否使用预填充CP
    is_dsa_enable_prefill_cp,  # DSA是否启用预填充CP
)
from sglang.srt.layers.dp_attention import (  # 导入DP注意力函数
    attn_tp_all_gather_into_tensor,  # 注意力TP全聚集到张量
    attn_tp_reduce_scatter_tensor,  # 注意力TP归约散射到张量
    dp_gather_partial,  # DP部分聚集
    dp_reduce_scatter_tensor,  # DP归约散射到张量
    dp_scatter,  # DP散射
    get_attention_cp_rank,  # 获取注意力CP rank
    get_attention_cp_size,  # 获取注意力CP大小
    get_attention_dp_size,  # 获取注意力DP大小
    get_attention_tp_group,  # 获取注意力TP通信组
    get_attention_tp_rank,  # 获取注意力TP rank
    get_attention_tp_size,  # 获取注意力TP大小
    get_dp_global_num_tokens,  # 获取DP全局token数
    get_global_dp_buffer,  # 获取全局DP缓冲区
    get_local_dp_buffer,  # 获取本地DP缓冲区
    get_moe_cp_rank,  # 获取MoE CP rank
    get_moe_cp_size,  # 获取MoE CP大小
    is_allocation_symmetric,  # 判断分配是否对称
    is_dp_attention_enabled,  # 判断DP注意力是否启用
    is_enable_moe_cp_allgather,  # 判断MoE CP全聚集是否启用
    moe_cp_all_gather_into_tensor,  # MoE CP全聚集到张量
)
from sglang.srt.layers.flashinfer_comm_fusion import is_flashinfer_allreduce_unavailable  # 导入flashinfer全归约不可用检测
from sglang.srt.layers.moe import (  # 导入MoE相关函数
    get_moe_a2a_backend,  # 获取MoE all-to-all后端
    should_use_dp_reduce_scatterv,  # 判断是否使用DP归约散射v
    should_use_flashinfer_cutlass_moe_fp4_allgather,  # 判断是否使用flashinfer cutlass MoE FP4全聚集
)
from sglang.srt.layers.utils.cp_utils import (  # 导入上下文并行工具
    is_mla_prefill_cp_enabled,  # MLA预填充CP是否启用
    mla_use_prefill_cp,  # MLA是否使用预填充CP
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 导入推测算法
from sglang.srt.utils import (  # 导入工具函数
    get_bool_env_var,  # 获取布尔环境变量
    is_cuda,  # CUDA检测
    is_flashinfer_available,  # flashinfer可用性检测
    is_gfx95_supported,  # gfx95支持检测
    is_hip,  # HIP(AMD ROCm)检测
    is_npu,  # NPU检测
    is_sm90_supported,  # SM90(Hopper)支持检测
    is_sm100_supported,  # SM100支持检测
)

_is_cuda = is_cuda()  # 检测当前是否为CUDA环境
_is_flashinfer_available = is_flashinfer_available()  # 检测flashinfer是否可用
_is_sm90_supported = _is_cuda and is_sm90_supported()  # 检测是否支持SM90
_is_sm100_supported = _is_cuda and is_sm100_supported()  # 检测是否支持SM100
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()  # 检测是否使用aiter（ROCm专用）
_is_gfx95_supported = is_gfx95_supported()  # 检测是否支持gfx95
_is_npu = is_npu()  # 检测是否为NPU环境
_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()  # 获取QLora后全聚集标志

if _use_aiter:  # 如果使用aiter
    from aiter.ops.rmsnorm import add_rmsnorm_quant as _aiter_add_rmsnorm_quant  # 导入aiter残差加RMSNorm量化
    from aiter.ops.rmsnorm import rmsnorm_quant as _aiter_rmsnorm_quant  # 导入aiter RMSNorm量化

    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype as _aiter_fp8_dtype  # 导入FP8数据类型

    if _is_gfx95_supported:  # 如果支持gfx95
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant  # 导入融合RMSNorm+FP8分组量化

        from sglang.srt.layers.quantization.rocm_mxfp4_utils import (  # 导入ROCm MX-FP4量化工具
            fused_rms_mxfp4_quant,  # 融合RMSNorm+MXFP4量化
        )
elif _is_npu:  # 如果是NPU环境
    from sglang.srt.hardware_backend.npu.cmo import prepare_weight_cache  # 导入NPU权重缓存准备函数


def _fused_rmsnorm_fp8_per_token_quant(  # 融合RMSNorm + FP8逐token量化函数
    hidden_states: torch.Tensor,  # 输入隐藏状态
    weight: torch.Tensor,  # RMSNorm权重
    epsilon: float,  # RMSNorm epsilon值
    residual: Optional[torch.Tensor] = None,  # 可选的残差张量
):
    """Fused (optional residual-add +) RMSNorm + FP8 per-token quantization.  # 融合（可选残差加+）RMSNorm + FP8逐token量化

    Only used with the aiter (ROCm) backend.  # 仅用于aiter（ROCm）后端

    Args:  # 参数
        residual: if provided, computes hidden_states + residual before RMSNorm  # 如果提供，在RMSNorm前计算hidden_states + residual
                  and returns updated residual_out as second element.  # 并将更新的residual_out作为第二个元素返回

    Returns:  # 返回值
        If residual is None:  (out_fp8, scale)  # 如果residual为None：(FP8输出, 缩放因子)
        If residual provided: ((out_fp8, scale), residual_out)  # 如果提供residual：((FP8输出, 缩放因子), 残差输出)
    """
    M, N = hidden_states.shape  # 获取隐藏状态形状
    out_fp8 = torch.empty((M, N), dtype=_aiter_fp8_dtype, device=hidden_states.device)  # 创建FP8输出张量
    scale = torch.empty(M, dtype=torch.float32, device=hidden_states.device)  # 创建缩放因子张量
    if residual is not None:  # 如果有残差
        residual_out = torch.empty_like(hidden_states)  # 创建残差输出张量
        _aiter_add_rmsnorm_quant(  # 调用aiter残差加RMSNorm量化核函数
            out_fp8,  # FP8输出
            hidden_states,  # 隐藏状态
            residual,  # 残差输入
            residual_out,  # 残差输出
            scale,  # 缩放因子
            weight,  # RMSNorm权重
            epsilon,  # epsilon值
            0,  # group_size=0 → per-token  # group_size=0 → 逐token量化
        )
        return (out_fp8, scale.unsqueeze(1)), residual_out  # 返回量化结果和残差输出
    else:  # 没有残差
        _aiter_rmsnorm_quant(  # 调用aiter RMSNorm量化核函数
            out_fp8,  # FP8输出
            hidden_states,  # 隐藏状态
            scale,  # 缩放因子
            weight,  # RMSNorm权重
            epsilon,  # epsilon值
            0,  # group_size=0 → per-token  # group_size=0 → 逐token量化
        )
        return (out_fp8, scale.unsqueeze(1))  # 返回量化结果


# TODO: According to the discussion in https://github.com/flashinfer-ai/flashinfer/issues/1223#issuecomment-3047256465  # TODO: 根据讨论
# We set the max token num to 128 for allreduce fusion with min-latency case(use_oneshot=True).  # 我们将全归约融合（最小延迟模式，use_oneshot=True）的最大token数设为128
FUSE_ALLREDUCE_MAX_BATCH_SIZE = 2048  # 全归约融合的最大批次大小


def apply_flashinfer_allreduce_fusion(batch_size: int):  # 判断是否应用flashinfer全归约融合
    return (  # 返回是否满足所有条件
        # NOTE: flashinfer 0.6.1 caused performance regression on sm100 for allreduce fusion  # 注意：flashinfer 0.6.1在sm100上导致全归约融合性能回退
        # Ref: https://github.com/sgl-project/sglang/issues/17237  # 参考: https://github.com/sgl-project/sglang/issues/17237
        (_is_sm90_supported or _is_sm100_supported)  # 支持SM90或SM100
        and _is_flashinfer_available  # flashinfer可用
        and batch_size > 0  # 批次大小大于0
        and batch_size <= FUSE_ALLREDUCE_MAX_BATCH_SIZE  # 批次大小不超过最大值
        and not is_dp_attention_enabled()  # DP注意力未启用
        and get_global_server_args().enable_flashinfer_allreduce_fusion  # 全局启用flashinfer全归约融合
        and not is_flashinfer_allreduce_unavailable()  # flashinfer全归约可用
    )


def apply_aiter_all_reduce_fusion(input_tensor: torch.Tensor):  # 判断是否应用aiter全归约融合
    n = input_tensor.shape[-1]  # 获取最后一维大小
    total_bytes = input_tensor.numel() * input_tensor.element_size()  # 计算总字节数
    # Aiter's should_custom_ar uses <= max_size/2 (64 MB); match that boundary.  # Aiter的should_custom_ar使用<= max_size/2（64 MB）；匹配该边界
    return (  # 返回是否满足所有条件
        _use_aiter  # 使用aiter
        and total_bytes > 0  # 总字节数大于0
        and n <= 16384  # 最后一维大小不超过16384
        and total_bytes <= 8 * 1024 * 8192  # 总字节数不超过64MB
        and get_tensor_model_parallel_world_size() != 6  # TP world size不为6
        and not is_dp_attention_enabled()  # DP注意力未启用
        and get_global_server_args().enable_aiter_allreduce_fusion  # 全局启用aiter全归约融合
    )


class ScatterMode(Enum):  # 数据散射模式枚举
    """
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d  # 假设TP=4，DP=2，启用DP注意力，系统处理序列a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively  # 模型输入/输出：四个rank分别为[ab, ab, cd, cd]
    SCATTERED: [a, b, c, d]  # 散射模式：每个rank只有自己的数据
    TP_ATTN_FULL: [ab, ab, cd, cd], i.e. all ranks inside a TP attn group have full data of the group  # TP注意力完整：TP注意力组内所有rank拥有完整组数据
    FULL: [abcd, abcd, abcd, abcd]  # 完整模式：所有rank拥有全部数据
    MOE_FULL: full within the MoE group (cp_per_moe CP chunks), used when moe_dp_size < attn_cp_size  # MoE完整：MoE组内完整（cp_per_moe个CP块），当moe_dp_size < attn_cp_size时使用
    """

    SCATTERED = auto()  # 散射模式
    TP_ATTN_FULL = auto()  # TP注意力完整模式
    FULL = auto()  # 完整模式
    MOE_FULL = auto()  # MoE完整模式

    @staticmethod
    def model_input_output():  # 获取模型前向传播输入输出的散射模式
        """The scatter mode for model forward pass input and output data  # 模型前向传播输入输出数据的散射模式"""
        if is_dsa_enable_prefill_cp() or is_mla_prefill_cp_enabled():  # 如果启用DSA或MLA预填充CP
            return ScatterMode.SCATTERED  # 返回散射模式

        return ScatterMode.TP_ATTN_FULL  # 返回TP注意力完整模式


class AttentionInputs:  # 注意力输入管理类，延迟获取隐藏状态和QKV潜变量
    def __init__(  # 初始化方法
        self,
        hidden_states: torch.Tensor,  # 本地隐藏状态
        forward_batch: ForwardBatch,  # 前向批次信息
        qkv_latent_func: Callable,  # QKV潜变量计算函数
    ):
        self.hidden_states_local = hidden_states  # 保存本地隐藏状态
        self.forward_batch = forward_batch  # 保存前向批次信息
        self.qkv_latent_func = qkv_latent_func  # 保存QKV潜变量函数
        self.hidden_states_ = None  # 缓存的完整隐藏状态
        self.qkv_latent_ = None  # 缓存的QKV潜变量

    def tp_all_gather_hidden_states(self, hidden_states, forward_batch):  # TP全聚集隐藏状态
        total_tokens = forward_batch.input_ids.shape[0]  # 获取总token数
        output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))  # 创建输出张量
        get_tp_group().all_gather_into_tensor(output, hidden_states)  # 执行TP全聚集
        return output  # 返回聚集后的隐藏状态

    def fetch_qkv_latent(self):  # 获取QKV潜变量（延迟计算）
        if self.qkv_latent_ is not None:  # 如果已缓存
            return self.qkv_latent_  # 返回缓存值
        assert self.qkv_latent_func is not None  # 断言QKV潜变量函数存在
        self.qkv_latent_ = self.qkv_latent_func(  # 计算QKV潜变量
            self.hidden_states_local, self.forward_batch  # 使用本地隐藏状态和批次信息
        )
        if get_attn_tp_context().input_scattered:  # 如果输入是散射的
            self.qkv_latent_ = self.tp_all_gather_hidden_states(  # 全聚集QKV潜变量
                self.qkv_latent_, self.forward_batch  # 
            )
        return self.qkv_latent_  # 返回QKV潜变量

    def fetch_hidden_states(self):  # 获取完整隐藏状态（延迟聚集）
        if self.hidden_states_ is not None:  # 如果已缓存
            return self.hidden_states_  # 返回缓存值
        self.hidden_states_ = self.hidden_states_local  # 从本地隐藏状态开始
        if get_attn_tp_context().input_scattered:  # 如果输入是散射的
            self.hidden_states_ = self.tp_all_gather_hidden_states(  # 全聚集隐藏状态
                self.hidden_states_, self.forward_batch  # 
            )
        return self.hidden_states_  # 返回完整隐藏状态


class AttnTpContext:  # 注意力TP上下文，管理输入散射模式
    def __init__(self):  # 初始化方法
        self.allow_input_scattered = False  # 是否允许输入散射
        self.input_scattered_ = False  # 当前是否处于输入散射状态
        self.attn_inputs_: Optional[AttentionInputs] = None  # 注意力输入缓存
        self.is_dsa = False  # 是否为DSA模式

    def init_context(self, q_lora_rank, is_dsa):  # 初始化上下文，判断是否启用输入散射
        self.is_dsa = is_dsa  # 保存DSA标志
        self.allow_input_scattered = (  # 判断是否允许输入散射
            get_global_server_args().enable_attn_tp_input_scattered  # 全局启用标志
            and (_is_cuda or _is_npu)  # CUDA或NPU环境
            and q_lora_rank is not None  # 有Q LoRA秩
            and not is_dsa  # 非DSA模式
            and get_tensor_model_parallel_world_size() > 1  # TP大小大于1
            and not is_dp_attention_enabled()  # DP注意力未启用
            and get_moe_a2a_backend().is_none()  # 无MoE all-to-all后端
            and not enable_moe_dense_fully_dp()  # 未启用MoE dense完全DP
            and get_global_server_args().disable_piecewise_cuda_graph  # 禁用分段CUDA图
            and get_global_server_args().speculative_algorithm != "EAGLE3"  # 非EAGLE3推测算法
        )
        if get_global_server_args().enable_attn_tp_input_scattered:  # 如果全局启用了输入散射
            if not self.allow_input_scattered:  # 但条件不满足
                logging.info(  # 记录信息
                    "attn_tp_input_scattered is not enabled while other conditions are not met"  # 输入散射未启用，其他条件不满足
                )
            else:  # 条件满足
                logging.info("attn_tp_input_scattered is enabled")  # 记录输入散射已启用

    def use_input_scattered(self, forward_batch: ForwardBatch):  # 判断当前批次是否使用输入散射
        return (  # 返回是否满足所有条件
            self.allow_input_scattered  # 允许输入散射
            and forward_batch.forward_mode.is_extend()  # 是扩展模式
            and not forward_batch.forward_mode.is_target_verify()  # 不是目标验证
            and not forward_batch.forward_mode.is_draft_extend()  # 不是草稿扩展
            and forward_batch.input_ids is not None  # 有输入ID
            and not forward_batch.can_run_tbo  # 不能运行TBO
        )

    @property
    def input_scattered(self):  # 输入散射状态属性
        return self.input_scattered_  # 返回当前散射状态

    def set_attn_inputs(self, attn_inputs: AttentionInputs):  # 设置注意力输入
        self.attn_inputs_ = attn_inputs  # 保存注意力输入

    def fetch_qkv_latent(self):  # 获取QKV潜变量
        assert self.attn_inputs_ is not None  # 断言注意力输入已设置
        return self.attn_inputs_.fetch_qkv_latent()  # 返回QKV潜变量

    def fetch_hidden_states(self):  # 获取完整隐藏状态
        assert self.attn_inputs_ is not None  # 断言注意力输入已设置
        return self.attn_inputs_.fetch_hidden_states()  # 返回完整隐藏状态

    def clear_attn_inputs(self) -> None:  # 清除注意力输入缓存
        self.attn_inputs_ = None  # 设为None

    @contextmanager
    def maybe_input_scattered(self, forward_batch: ForwardBatch):  # 上下文管理器，临时设置输入散射状态
        flag = self.use_input_scattered(forward_batch)  # 判断是否使用输入散射
        old_flag = self.input_scattered  # 保存旧状态
        self.input_scattered_ = flag  # 设置新状态
        yield  # 执行上下文
        self.input_scattered_ = old_flag  # 恢复旧状态
        self.attn_inputs_ = None  # 清除注意力输入


ATTN_TP_CONTEXT = AttnTpContext()  # 全局注意力TP上下文实例


def get_attn_tp_context():  # 获取全局注意力TP上下文
    return ATTN_TP_CONTEXT  # 返回全局实例


@dataclass
class _LayerModeComputationContext:  # 层模式计算上下文，用于确定各阶段的散射模式
    num_layers: int  # 总层数
    layer_id: int  # 当前层ID
    is_layer_sparse: bool  # 当前层是否为稀疏（MoE）层
    is_previous_layer_sparse: Optional[bool]  # 前一层是否为稀疏层
    is_next_layer_sparse: Optional[bool]  # 后一层是否为稀疏层

    def previous_layer(self):  # 获取前一层的计算上下文
        assert self.is_previous_layer_sparse is not None  # 断言前一层信息存在
        return _LayerModeComputationContext(  # 返回前一层的上下文
            num_layers=self.num_layers,  # 总层数
            layer_id=self.layer_id - 1,  # 层ID减1
            is_layer_sparse=self.is_previous_layer_sparse,  # 前一层的稀疏性
            is_previous_layer_sparse=None,  # 前前层信息不可用
            is_next_layer_sparse=self.is_layer_sparse,  # 当前层作为后一层
        )


@dataclass
class LayerScatterModes:  # 层散射模式数据类，定义层各阶段的散射模式
    layer_input_mode: ScatterMode  # 层输入模式
    attn_mode: ScatterMode  # 注意力模式
    # Can be further split into e.g. mlp_input_mode and mlp_output_mode if needed  # 如果需要可进一步拆分为mlp_input_mode和mlp_output_mode
    mlp_mode: ScatterMode  # MLP模式
    middle_residual_mode: ScatterMode  # 中间残差模式
    layer_output_mode: ScatterMode  # 层输出模式

    @classmethod
    def init_new(cls, **kwargs):  # 创建新的层散射模式实例
        context = _LayerModeComputationContext(**kwargs)  # 创建层模式计算上下文
        return cls(  # 返回新实例
            layer_input_mode=cls._compute_layer_input_mode(context),  # 计算层输入模式
            attn_mode=ScatterMode.TP_ATTN_FULL,  # 注意力模式固定为TP_ATTN_FULL
            mlp_mode=cls._compute_mlp_mode(context),  # 计算MLP模式
            middle_residual_mode=cls._compute_middle_residual_mode(context),  # 计算中间残差模式
            layer_output_mode=cls._compute_layer_output_mode(context),  # 计算层输出模式
        )

    @classmethod
    def _compute_layer_input_mode(cls, context: _LayerModeComputationContext):  # 计算层输入的散射模式
        if context.layer_id == 0:  # 如果是第一层
            return ScatterMode.model_input_output()  # 返回模型输入输出模式
        return cls._compute_layer_output_mode(context.previous_layer())  # 返回前一层的输出模式

    @classmethod
    def _compute_mlp_mode(cls, context: _LayerModeComputationContext):  # 计算MLP的散射模式
        if context.is_layer_sparse:  # 如果是稀疏（MoE）层
            if (  # 如果满足以下条件
                # Token dispatch/combine will be handled outside of LayerCommunicator for these modes.  # 对于这些模式，token分发/合并将在LayerCommunicator外部处理
                not get_moe_a2a_backend().is_none()  # 有MoE all-to-all后端
                or should_use_flashinfer_cutlass_moe_fp4_allgather()  # 或使用flashinfer cutlass MoE FP4全聚集
            ):
                return ScatterMode.SCATTERED  # 返回散射模式
            # DSA CP and MLA CP both don't support MOE_FULL yet; fall back to FULL.  # DSA CP和MLA CP都不支持MOE_FULL；回退到FULL
            if is_enable_moe_cp_allgather() and not (  # 如果启用MoE CP全聚集且非DSA/MLA CP
                is_dsa_enable_prefill_cp() or is_mla_prefill_cp_enabled()  # 
            ):
                return ScatterMode.MOE_FULL  # 返回MoE完整模式
            return ScatterMode.FULL  # 返回完整模式
        else:  # 密集层
            return (  # 
                ScatterMode.SCATTERED  # 散射模式
                if enable_moe_dense_fully_dp()  # 如果启用MoE dense完全DP
                else ScatterMode.FULL  # 否则完整模式
            )

    @classmethod
    def _should_gather_for_tbo(cls, context: _LayerModeComputationContext):  # 判断是否为TBO（两批次重叠）进行聚集
        return (  # 返回是否满足所有条件
            not context.is_layer_sparse  # 非稀疏层
            and context.is_next_layer_sparse  # 下一层是稀疏层
            and enable_moe_dense_fully_dp()  # 启用MoE dense完全DP
            and get_global_server_args().enable_two_batch_overlap  # 启用两批次重叠
        )

    @classmethod
    def _compute_middle_residual_mode(cls, context: _LayerModeComputationContext):  # 计算中间残差的散射模式
        mlp_mode = cls._compute_mlp_mode(context)  # 获取MLP模式
        if mlp_mode == ScatterMode.SCATTERED:  # 如果MLP是散射模式
            return ScatterMode.SCATTERED  # 残差也是散射模式
        if mlp_mode in (ScatterMode.FULL, ScatterMode.MOE_FULL):  # 如果MLP是完整或MoE完整模式
            return ScatterMode.TP_ATTN_FULL  # 残差是TP注意力完整模式
        raise NotImplementedError  # 其他情况未实现


    @classmethod
    def _compute_layer_output_mode(cls, context: _LayerModeComputationContext):  # 计算层输出的散射模式
        mlp_mode = cls._compute_mlp_mode(context)  # 获取MLP模式
        if context.layer_id == context.num_layers - 1:  # 如果是最后一层
            return ScatterMode.model_input_output()  # 返回模型输入输出模式
        if mlp_mode == ScatterMode.SCATTERED:  # 如果MLP是散射模式
            if cls._should_gather_for_tbo(context):  # 如果需要为TBO聚集
                return ScatterMode.TP_ATTN_FULL  # 返回TP注意力完整模式
            return ScatterMode.SCATTERED  # 返回散射模式
        if mlp_mode in (ScatterMode.FULL, ScatterMode.MOE_FULL):  # 如果MLP是完整或MoE完整模式
            return ScatterMode.TP_ATTN_FULL  # 返回TP注意力完整模式
        raise NotImplementedError  # 其他情况未实现



def enable_moe_dense_fully_dp():  # 判断是否启用MoE dense完全DP
    return get_global_server_args().moe_dense_tp_size == 1  # MoE dense TP大小为1时启用


class LayerCommunicator:  # 层通信器，管理层间数据分发和聚合
    def __init__(  # 初始化方法
        self,
        layer_scatter_modes: LayerScatterModes,  # 层散射模式
        input_layernorm: torch.nn.Module,  # 输入层归一化模块
        post_attention_layernorm: torch.nn.Module,  # 注意力后层归一化模块
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.  # 归约散射需要跳过MoE/MLP后模型代码中的全归约，因此仅对已实现的模型启用。对所有使用LayerCommunicator的模型完成实现后移除此标志
        allow_reduce_scatter: bool = False,  # 是否允许归约散射
        is_last_layer: bool = False,  # 是否是最后一层
        qkv_latent_func: Optional[Callable] = None,  # 可选的QKV潜变量函数
    ):
        self.layer_scatter_modes = layer_scatter_modes  # 保存层散射模式
        self.input_layernorm = input_layernorm  # 保存输入层归一化
        self.post_attention_layernorm = post_attention_layernorm  # 保存注意力后层归一化
        self.allow_reduce_scatter = allow_reduce_scatter  # 保存归约散射标志
        self.is_last_layer = is_last_layer  # 保存最后层标志
        self.qkv_latent_func = qkv_latent_func  # 保存QKV潜变量函数

        self._context = CommunicateContext.init_new()  # 初始化通信上下文
        self._post_init_communicate()  # 后初始化通信函数
        self._speculative_algo = SpeculativeAlgorithm.from_string(  # 解析推测算法
            get_global_server_args().speculative_algorithm  # 从全局参数获取
        )

    def _post_init_communicate(self):  # 后初始化，根据散射模式选择通信函数
        self._communicate_simple_fn = CommunicateSimpleFn.get_fn(  # 获取简单通信函数
            input_mode=self.layer_scatter_modes.layer_input_mode,  # 输入模式
            output_mode=self.layer_scatter_modes.attn_mode,  # 输出模式
            context=self._context,  # 通信上下文
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (  # 获取带全归约和层归一化的通信函数
            CommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.attn_mode,  # 隐藏状态输入模式
                residual_input_mode=self.layer_scatter_modes.layer_input_mode,  # 残差输入模式
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # 隐藏状态输出模式
                residual_output_mode=self.layer_scatter_modes.middle_residual_mode,  # 残差输出模式
                context=self._context,  # 通信上下文
            )
        )
        self._communicate_summable_tensor_pair_fn = (  # 获取可求和张量对的通信函数
            CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # 隐藏状态输入模式
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,  # 残差输入模式
                output_mode=self.layer_scatter_modes.layer_output_mode,  # 输出模式
                context=self._context,  # 通信上下文
            )
        )

    def prepare_attn_and_capture_last_layer_outputs(  # 准备注意力并捕获最后一层输出
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,  # 捕获的最后一层输出列表
        post_residual_addition: Optional[torch.Tensor] = None,  # 残差后加法
    ):
        hidden_states, residual = self.prepare_attn(  # 准备注意力
            hidden_states,  # 隐藏状态
            residual,  # 残差
            forward_batch,  # 前向批次
            post_residual_addition=post_residual_addition,  # 残差后加法
        )
        if captured_last_layer_outputs is not None:  # 如果需要捕获输出
            gathered_last_layer_output = self._communicate_simple_fn(  # 通信聚集残差
                hidden_states=residual,  # 使用残差作为输入
                forward_batch=forward_batch,  # 前向批次
                context=self._context,  # 通信上下文
            )
            if gathered_last_layer_output is residual:  # 如果通信未改变引用
                # Clone to avoid modifying the original residual by Custom RMSNorm inplace operation  # 克隆以避免自定义RMSNorm原地操作修改原始残差
                gathered_last_layer_output = residual.clone()  # 克隆残差
            captured_last_layer_outputs.append(gathered_last_layer_output)  # 添加到捕获列表
        return hidden_states, residual  # 返回隐藏状态和残差

    def prepare_attn(  # 准备注意力层的输入
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        quant_format: str = "",  # 量化格式
        post_residual_addition: Optional[torch.Tensor] = None,  # 残差后加法
    ):
        if get_attn_tp_context().input_scattered:  # 如果输入是散射的
            hidden_states, residual = self._tp_reduce_scatter(  # TP归约散射
                hidden_states,  # 隐藏状态
                residual,  # 残差
            )
        if hidden_states.shape[0] == 0:  # 如果没有token
            residual = hidden_states  # 残差设为空隐藏状态
        else:  # 有token
            if (  # 如果需要全归约融合
                residual is not None  # 有残差
                and hasattr(hidden_states, "_sglang_needs_allreduce_fusion")  # 有融合标志
                and hidden_states._sglang_needs_allreduce_fusion  # 标志为True
            ):
                if (  # 如果可以应用全归约融合
                    apply_aiter_all_reduce_fusion(hidden_states)  # aiter全归约融合
                    or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])  # flashinfer全归约融合
                ) and hasattr(self.input_layernorm, "forward_with_allreduce_fusion"):  # 层归一化支持融合
                    hidden_states, residual = (  # 使用融合的前向传播
                        self.input_layernorm.forward_with_allreduce_fusion(
                            hidden_states, residual, use_attn_tp_group=False  # 不使用注意力TP组
                        )
                    )
                else:  # 不能融合
                    hidden_states = moe_tensor_model_parallel_all_reduce(hidden_states)  # MoE TP全归约
                    hidden_states, residual = self.input_layernorm(  # 应用层归一化
                        hidden_states, residual  # 
                    )
            else:  # 不需要全归约融合
                if residual is None:  # 没有残差
                    residual = hidden_states  # 残差设为隐藏状态

                    if _use_aiter and _is_gfx95_supported and ("mxfp4" in quant_format):  # aiter + gfx95 + mxfp4
                        hidden_states, *_, _ = fused_rms_mxfp4_quant(  # 融合RMSNorm+MXFP4量化
                            hidden_states,  # 隐藏状态
                            self.input_layernorm.weight,  # 层归一化权重
                            self.input_layernorm.variance_epsilon,  # 方差epsilon
                            None, None, None, None,  # 其他参数为None
                        )
                    elif _use_aiter and _is_gfx95_supported and (quant_format == "fp8"):  # aiter + gfx95 + fp8
                        # aiter (ROCm gfx95) fused RMSNorm + FP8 group quant.  # aiter (ROCm gfx95) 融合RMSNorm + FP8分组量化
                        # When DSA is active, also preserve the unquantized bf16  # 当DSA激活时，同时保留未量化的bf16
                        # output as a 3-tuple (fp8, scale, bf16) so the DSA  # 输出为三元组(fp8, scale, bf16)，以便DSA
                        # indexer can skip redundant FP8 dequantization.  # 索引器可以跳过冗余的FP8反量化
                        _dsa_needs_bf16 = get_attn_tp_context().is_dsa  # 检查DSA是否需要bf16
                        hidden_states, _unq_bf16, _, _res = fused_rms_fp8_group_quant(  # 融合RMSNorm+FP8分组量化
                            hidden_states,  # 隐藏状态
                            self.input_layernorm.weight,  # 层归一化权重
                            self.input_layernorm.variance_epsilon,  # 方差epsilon
                            inp2=None, inp2_weight=None, inp2_epsilon=None,  # 第二输入相关参数
                            group_size=128,  # 分组大小
                            dtype_quant=torch.float8_e4m3fn,  # 量化数据类型
                            res1=None,  # 残差1
                            output_unquantized_inp1=_dsa_needs_bf16,  # 是否输出未量化的输入1
                        )
                        if _dsa_needs_bf16:  # 如果DSA需要bf16
                            hidden_states = (  # 打包为三元组
                                hidden_states[0],  # FP8输出
                                hidden_states[1],  # 缩放因子
                                _unq_bf16,  # 未量化的bf16
                            )

                    elif _use_aiter and (quant_format == "fp8_per_token"):  # aiter + fp8逐token量化
                        hidden_states = _fused_rmsnorm_fp8_per_token_quant(  # 融合RMSNorm+FP8逐token量化
                            hidden_states,  # 隐藏状态
                            self.input_layernorm.weight.data,  # 层归一化权重数据
                            self.input_layernorm.variance_epsilon,  # 方差epsilon
                        )

                    else:  # 其他情况
                        hidden_states = self.input_layernorm(hidden_states)  # 仅应用层归一化
                else:  # 有残差
                    if _use_aiter and _is_gfx95_supported and ("mxfp4" in quant_format):  # aiter + gfx95 + mxfp4
                        hidden_states, *_, residual = fused_rms_mxfp4_quant(  # 融合RMSNorm+MXFP4量化（带残差）
                            hidden_states,  # 隐藏状态
                            self.input_layernorm.weight,  # 层归一化权重
                            self.input_layernorm.variance_epsilon,  # 方差epsilon
                            None, None, None,  # 其他参数
                            residual,  # 残差
                        )
                    elif _use_aiter and _is_gfx95_supported and (quant_format == "fp8"):  # aiter + gfx95 + fp8
                        # aiter (ROCm gfx95) fused RMSNorm + FP8 group quant  # aiter (ROCm gfx95) 融合RMSNorm + FP8分组量化
                        # with residual addition. When DSA is active, pack  # 带残差加法。当DSA激活时，打包
                        # the unquantized bf16 as a 3-tuple (fp8, scale, bf16).  # 未量化的bf16为三元组(fp8, scale, bf16)
                        _dsa_needs_bf16 = get_attn_tp_context().is_dsa  # 检查DSA是否需要bf16
                        hidden_states, _unq_bf16, _, residual = (  # 融合量化（带残差）
                            fused_rms_fp8_group_quant(
                                hidden_states,  # 隐藏状态
                                self.input_layernorm.weight,  # 层归一化权重
                                self.input_layernorm.variance_epsilon,  # 方差epsilon
                                inp2=None, inp2_weight=None, inp2_epsilon=None,  # 第二输入相关参数
                                group_size=128,  # 分组大小
                                dtype_quant=torch.float8_e4m3fn,  # 量化数据类型
                                res1=residual,  # 残差
                                output_unquantized_inp1=_dsa_needs_bf16,  # 是否输出未量化的输入1
                            )
                        )
                        if _dsa_needs_bf16:  # 如果DSA需要bf16
                            hidden_states = (  # 打包为三元组
                                hidden_states[0],  # FP8输出
                                hidden_states[1],  # 缩放因子
                                _unq_bf16,  # 未量化的bf16
                            )
                    elif _use_aiter and (quant_format == "fp8_per_token"):  # aiter + fp8逐token量化
                        if post_residual_addition is not None:  # 如果有残差后加法
                            residual = residual + post_residual_addition  # 将加法累加到残差
                        hidden_states, residual = _fused_rmsnorm_fp8_per_token_quant(  # 融合RMSNorm+FP8逐token量化（带残差）
                            hidden_states,  # 隐藏状态
                            self.input_layernorm.weight.data,  # 层归一化权重数据
                            self.input_layernorm.variance_epsilon,  # 方差epsilon
                            residual=residual,  # 残差
                        )
                    else:  # 其他情况
                        hidden_states, residual = self.input_layernorm(  # 应用层归一化（带残差）
                            hidden_states,  # 隐藏状态
                            residual,  # 残差
                            post_residual_addition,  # 残差后加法
                        )

        hidden_states = self._communicate_simple_fn(  # 对隐藏状态执行简单通信
            hidden_states=hidden_states,  # 隐藏状态
            forward_batch=forward_batch,  # 前向批次
            context=self._context,  # 通信上下文
        )
        if self.qkv_latent_func is not None:  # 如果有QKV潜变量函数
            attn_inputs = AttentionInputs(  # 创建注意力输入
                hidden_states, forward_batch, self.qkv_latent_func  # 
            )
            get_attn_tp_context().set_attn_inputs(attn_inputs)  # 设置到上下文
        return hidden_states, residual  # 返回隐藏状态和残差

    def _tp_reduce_scatter(  # TP归约散射，将全量数据散射到各TP rank
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回散射后的隐藏状态和残差
        if hidden_states.shape[0] == 0:  # 如果没有token
            return hidden_states, hidden_states  # 返回空张量
        assert (  # 断言token数可被TP大小整除
            hidden_states.shape[0] % self._context.tp_size == 0
        ), f"Expected total tokens {hidden_states.shape[0]} % tp_size {self._context.tp_size} to be 0"  # token数不能被TP大小整除
        local_tokens = hidden_states.shape[0] // self._context.tp_size  # 计算本地token数
        output = hidden_states.new_empty(local_tokens, *hidden_states.shape[1:])  # 创建输出张量
        get_tp_group().reduce_scatter_tensor(output, hidden_states)  # 执行归约散射
        if residual is not None:  # 如果有残差
            residual = residual.tensor_split(self._context.tp_size)[  # 按TP大小拆分残差
                self._context.tp_rank  # 取当前rank的部分
            ]
        return output, residual  # 返回散射后的隐藏状态和残差

    def prepare_mlp(  # 准备MLP层的输入
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        cache=None,  # 可选的缓存
    ):
        if cache is not None:  # 如果有缓存
            self._context.cache = cache  # 设置到上下文

        return self._communicate_with_all_reduce_and_layer_norm_fn(  # 执行带全归约和层归一化的通信
            hidden_states=hidden_states,  # 隐藏状态
            residual=residual,  # 残差
            forward_batch=forward_batch,  # 前向批次
            layernorm=self.post_attention_layernorm,  # 注意力后层归一化
            context=self._context,  # 通信上下文
        )

    def postprocess_layer(  # 层后处理，合并隐藏状态和残差
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
    ):
        return self._communicate_summable_tensor_pair_fn(  # 执行可求和张量对通信
            hidden_states=hidden_states,  # 隐藏状态
            residual=residual,  # 残差
            forward_batch=forward_batch,  # 前向批次
            context=self._context,  # 通信上下文
            allow_reduce_scatter=self.allow_reduce_scatter,  # 归约散射标志
        )

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):  # 判断是否应使用归约散射
        if not self.allow_reduce_scatter:  # 如果不允许归约散射
            return False  # 返回False
        if (  # 如果通信函数是散射隐藏状态
            self._communicate_summable_tensor_pair_fn
            is CommunicateSummableTensorPairFn._scatter_hidden_states
        ):
            if should_use_dp_reduce_scatterv():  # 如果应使用DP归约散射v
                return True  # 返回True
            if forward_batch.dp_padding_mode.is_max_len():  # 如果DP填充模式是最大长度
                return True  # 返回True
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):  # DSA或MLA使用预填充CP
            return True  # 返回True
        if get_attn_tp_context().input_scattered and not self.is_last_layer:  # 输入散射且非最后层
            return True  # 返回True
        return False  # 其他情况返回False

    # NOTE: This function will cause torch recompilation  # 注意：此函数会导致torch重新编译
    def should_fuse_mlp_allreduce_with_next_layer(  # 判断是否应将MLP全归约与下一层融合
        self, forward_batch: ForwardBatch  # 前向批次
    ) -> bool:  # 返回布尔值
        # When MOE_FULL is active (moe_cp allgather), fusion must be disabled because  # 当MOE_FULL激活（moe_cp全聚集）时，必须禁用融合，因为
        # the fusion path skips postprocess_layer which contains the moe_cp scatter.  # 融合路径跳过了包含moe_cp散射的postprocess_layer
        # Without scatter, hidden_states remain at MOE_FULL size while residual is at  # 没有散射，隐藏状态保持MOE_FULL大小，而残差为
        # TP_ATTN_FULL size, causing a shape mismatch.  # TP_ATTN_FULL大小，导致形状不匹配
        if is_enable_moe_cp_allgather():  # 如果启用MoE CP全聚集
            return False  # 返回False

        if (  # 如果启用了DP注意力且使用Eagle推测算法
            is_dp_attention_enabled()
            and self._speculative_algo is not None
            and self._speculative_algo.is_eagle()
        ):
            return False  # 返回False

        if get_attn_tp_context().input_scattered:  # 如果输入是散射的
            return False  # 返回False

        batch_size = (  # 获取批次大小
            forward_batch.input_ids.shape[0]  # 从输入ID获取
            if hasattr(forward_batch, "input_ids")  # 如果有input_ids属性
            else 0  # 否则为0
        )

        # When mlp_mode is SCATTERED, the MLP runs on scattered data with no TP  # 当mlp_mode为SCATTERED时，MLP在散射数据上运行，无TP
        # all-reduce, so there is nothing to fuse with the next layer.  # 全归约，因此没有可与下一层融合的内容
        if self.layer_scatter_modes.mlp_mode == ScatterMode.SCATTERED:  # 如果MLP模式为散射
            return False  # 返回False

        return (  # 返回是否满足融合条件
            (
                apply_flashinfer_allreduce_fusion(batch_size)  # flashinfer全归约融合
                or (
                    _use_aiter  # 使用aiter
                    and batch_size > 0  # 批次大小大于0
                    and get_tensor_model_parallel_world_size() != 6  # TP大小不为6
                    and get_global_server_args().enable_aiter_allreduce_fusion  # 全局启用aiter全归约融合
                )
            )
            and (not self.is_last_layer)  # 不是最后一层
            and (self._context.tp_size > 1)  # TP大小大于1
        )


@dataclass
class CommunicateContext:  # 通信上下文数据类，存储通信所需的分组信息
    process_group_sizes: Dict[ScatterMode, int]  # 各散射模式对应的进程组大小
    attn_tp_rank: int  # 注意力TP rank
    attn_tp_size: int  # 注意力TP大小
    attn_dp_size: int  # 注意力DP大小
    attn_cp_rank: int  # 注意力CP rank
    attn_cp_size: int  # 注意力CP大小
    tp_size: int  # 全局TP大小
    cache = None  # 可选缓存
    tp_rank: int  # 全局TP rank

    def is_same_group_size(self, a: ScatterMode, b: ScatterMode):  # 判断两种散射模式的进程组大小是否相同
        return self.process_group_sizes[a] == self.process_group_sizes[b]  # 比较进程组大小

    @classmethod
    def init_new(cls):  # 创建新的通信上下文
        attn_tp_rank = get_attention_tp_rank()  # 获取注意力TP rank
        attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小
        attn_dp_size = get_attention_dp_size()  # 获取注意力DP大小
        attn_cp_size = get_attention_cp_size()  # 获取注意力CP大小
        attn_cp_rank = get_attention_cp_rank()  # 获取注意力CP rank
        tp_size = get_tensor_model_parallel_world_size()  # 获取全局TP大小
        tp_rank = get_tensor_model_parallel_rank()  # 获取全局TP rank
        moe_cp_size = get_moe_cp_size()  # 获取MoE CP大小
        process_group_sizes = {  # 计算各散射模式的进程组大小
            ScatterMode.SCATTERED: 1,  # 散射模式：大小为1
            ScatterMode.TP_ATTN_FULL: attn_tp_size,  # TP注意力完整：大小为注意力TP大小
            # TODO: support --moe-dense-tp-size > 1  # TODO: 支持--moe-dense-tp-size > 1
            # With context parallel enabled, we should exclude  # 启用上下文并行时，应排除
            # the attn_cp_size from the total tp_size  # 总tp_size中的attn_cp_size
            ScatterMode.FULL: tp_size // attn_cp_size,  # 完整模式：TP大小除以CP大小
            ScatterMode.MOE_FULL: tp_size // (attn_cp_size // moe_cp_size),  # MoE完整：TP大小除以(CP大小/MoE CP大小)
        }
        return cls(  # 返回新实例
            process_group_sizes=process_group_sizes,  # 进程组大小映射
            attn_tp_rank=attn_tp_rank,  # 注意力TP rank
            attn_tp_size=attn_tp_size,  # 注意力TP大小
            attn_dp_size=attn_dp_size,  # 注意力DP大小
            attn_cp_rank=attn_cp_rank,  # 注意力CP rank
            attn_cp_size=attn_cp_size,  # 注意力CP大小
            tp_size=tp_size,  # 全局TP大小
            tp_rank=tp_rank,  # 全局TP rank
        )


class CommunicateSimpleFn:  # 简单通信函数类，处理隐藏状态的散射/聚集
    @staticmethod
    def get_fn(  # 根据输入输出模式获取对应的通信函数
        input_mode: ScatterMode,  # 输入散射模式
        output_mode: ScatterMode,  # 输出散射模式
        context: CommunicateContext,  # 通信上下文
    ):
        if context.is_same_group_size(input_mode, output_mode):  # 如果输入输出组大小相同
            return CommunicateSimpleFn._trivial  # 返回平凡函数（无通信）

        if (input_mode == ScatterMode.SCATTERED) and (  # 如果输入为散射
            output_mode == ScatterMode.TP_ATTN_FULL  # 输出为TP注意力完整
        ):
            if _use_ag_after_qlora:  # 如果QLora后使用全聚集
                return CommunicateSimpleFn._trivial  # 返回平凡函数
            return CommunicateSimpleFn._scattered_to_tp_attn_full  # 返回散射到TP注意力完整函数

        raise NotImplementedError(f"{input_mode=} {output_mode=}")  # 其他模式未实现

    @staticmethod
    def _trivial(  # 平凡通信函数，直接返回输入
        hidden_states: torch.Tensor,  # 隐藏状态
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
    ) -> torch.Tensor:  # 返回隐藏状态
        return hidden_states  # 直接返回

    @staticmethod
    def _scattered_to_tp_attn_full(  # 从散射模式聚集到TP注意力完整模式
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # 隐藏状态（可能是元组）
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # 返回聚集后的隐藏状态
        if isinstance(hidden_states, tuple):  # 如果是元组（多张量情况）
            gathered_hidden_states = []  # 存储聚集后的张量
            for local_hidden_states in hidden_states:  # 遍历每个张量
                with use_symmetric_memory(  # 使用对称内存上下文
                    get_tp_group(),  # TP通信组
                    disabled=not is_allocation_symmetric(),  # 非对称分配时禁用
                ):
                    output = torch.empty(  # 创建输出张量
                        (
                            local_hidden_states.shape[0] * context.attn_tp_size,  # 聚集后的token数
                            *local_hidden_states.shape[1:],  # 其他维度不变
                        ),
                        dtype=local_hidden_states.dtype,  # 数据类型
                        device=local_hidden_states.device,  # 设备
                    )
                attn_tp_all_gather_into_tensor(  # 注意力TP全聚集
                    output,  # 输出
                    local_hidden_states,  # 本地数据
                )
                gathered_hidden_states.append(output)  # 添加到列表
            return tuple(gathered_hidden_states)  # 返回元组

        hidden_states, local_hidden_states = (  # 单张量情况
            get_local_dp_buffer(get_attention_tp_group()),  # 从DP缓冲区获取输出
            hidden_states,  # 本地数据
        )
        attn_tp_all_gather_into_tensor(  # 注意力TP全聚集
            hidden_states,  # 输出
            local_hidden_states,  # 本地数据
        )
        return hidden_states  # 返回聚集后的隐藏状态


class CommunicateWithAllReduceAndLayerNormFn:  # 带全归约和层归一化的通信函数类
    """Besides communication, needs to  # 除通信外，还需要
    1. All reduce in tp_attn_group on hidden_states  # 1. 在tp_attn_group中对隐藏状态执行全归约
    2. Apply layer norm  # 2. 应用层归一化
    """

    @staticmethod
    def get_fn(  # 根据输入输出模式获取对应的通信函数
        hidden_states_input_mode: ScatterMode,  # 隐藏状态输入模式
        residual_input_mode: ScatterMode,  # 残差输入模式
        hidden_states_output_mode: ScatterMode,  # 隐藏状态输出模式
        residual_output_mode: ScatterMode,  # 残差输出模式
        context: CommunicateContext,  # 通信上下文
    ):

        if (  # 如果所有组大小相同且attn_tp_size为1
            context.is_same_group_size(
                hidden_states_input_mode, hidden_states_output_mode
            )
            and context.is_same_group_size(residual_input_mode, residual_output_mode)
            and context.attn_tp_size == 1
        ):
            return CommunicateWithAllReduceAndLayerNormFn._simple  # 返回简单函数

        if (  # TP_ATTN_FULL -> FULL，残差SCATTERED/TP_ATTN_FULL -> TP_ATTN_FULL
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return partial(  # 返回部分应用函数
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,  # 传入残差输入模式
            )

        if (  # TP_ATTN_FULL -> MOE_FULL，残差SCATTERED/TP_ATTN_FULL -> TP_ATTN_FULL
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.MOE_FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return partial(  # 返回部分应用函数
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe,
                residual_input_mode=residual_input_mode,  # 传入残差输入模式
            )

        if (  # TP_ATTN_FULL -> SCATTERED，残差SCATTERED/TP_ATTN_FULL -> SCATTERED
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.SCATTERED)
            and (residual_output_mode == ScatterMode.SCATTERED)
        ):
            return partial(  # 返回部分应用函数
                CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,  # 传入残差输入模式
            )

        if (  # TP_ATTN_FULL -> TP_ATTN_FULL，残差SCATTERED/TP_ATTN_FULL -> TP_ATTN_FULL，attn_tp_size > 1
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
            and context.attn_tp_size > 1
        ):
            # Used when the dense MLP is tensor-parallelized along the  # 当密集MLP沿注意力TP组进行张量并行时使用
            # attention TP group (``moe_dense_tp_size > 1``): hidden states  # (moe_dense_tp_size > 1)：隐藏状态
            # need an all-reduce inside the attention TP group before the  # 需要在下一个层归一化前在注意力TP组内执行全归约
            # next layernorm, while staying in TP_ATTN_FULL on both sides.  # 同时两侧保持TP_ATTN_FULL
            return (
                CommunicateWithAllReduceAndLayerNormFn._tp_attn_all_reduce_and_layernorm
            )

        raise NotImplementedError(  # 其他模式未实现
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _simple(  # 简单模式，仅应用层归一化（无通信）
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        layernorm: torch.nn.Module,  # 层归一化模块
        context: CommunicateContext,  # 通信上下文
    ):
        # TODO move these `if shape != 0` into LayerNorm itself  # TODO: 将这些`if shape != 0`移入LayerNorm本身
        if hidden_states.shape[0] != 0:  # 如果有token
            hidden_states, residual = layernorm(hidden_states, residual)  # 应用层归一化
        return hidden_states, residual  # 返回隐藏状态和残差

    @staticmethod
    def _tp_attn_all_reduce_and_layernorm(  # 注意力TP组全归约后应用层归一化
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        layernorm: torch.nn.Module,  # 层归一化模块
        context: CommunicateContext,  # 通信上下文
    ):
        """All-reduce hidden states inside the attention TP group, then layernorm.  # 在注意力TP组内对隐藏状态全归约，然后层归一化

        Used when the dense MLP shares the attention TP group  # 当密集MLP共享注意力TP组时使用
        (``moe_dense_tp_size > 1``): both hidden states and residual stay in  # (moe_dense_tp_size > 1)：隐藏状态和残差保持在
        ``TP_ATTN_FULL`` across the boundary.  # 跨边界的TP_ATTN_FULL状态
        """
        hidden_states = get_attention_tp_group().all_reduce(hidden_states)  # 注意力TP组全归约
        if hidden_states.shape[0] != 0:  # 如果有token
            hidden_states, residual = layernorm(hidden_states, residual)  # 应用层归一化
        return hidden_states, residual  # 返回隐藏状态和残差

    @staticmethod
    def _gather_hidden_states_and_residual(  # 聚集隐藏状态和残差，应用全归约和层归一化
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        layernorm: torch.nn.Module,  # 层归一化模块
        context: CommunicateContext,  # 通信上下文
        *,  # 以下为关键字参数
        residual_input_mode,  # 残差输入模式
    ):
        if get_attn_tp_context().input_scattered:  # 如果输入是散射的
            return CommunicateWithAllReduceAndLayerNormFn._tp_all_reduce_with_scattered_residual(  # 使用散射残差的TP全归约
                hidden_states,  # 隐藏状态
                residual,  # 残差
                layernorm,  # 层归一化
                context,  # 通信上下文
            )

        if residual_input_mode == ScatterMode.SCATTERED and context.attn_tp_size > 1:  # 残差散射且TP大小>1
            residual, local_residual = (  # 准备残差聚集
                get_local_dp_buffer(get_attention_tp_group()),  # 从DP缓冲区获取输出
                residual,  # 本地残差
            )
            attn_tp_all_gather_into_tensor(residual, local_residual)  # 注意力TP全聚集残差
        if context.attn_dp_size != 1:  # 如果DP大小不为1
            # Perform layernorm on smaller data before comm. Only valid when attn_tp_size is 1 (tp_size == dp_size)  # 通信前对较小数据执行层归一化。仅在attn_tp_size为1（tp_size == dp_size）时有效
            use_layer_norm_before_gather = context.attn_tp_size == 1  # 判断是否在聚集前执行层归一化
            if use_layer_norm_before_gather and hidden_states.shape[0] != 0:  # 聚集前归一化且有token
                with use_symmetric_memory(  # 使用对称内存上下文
                    get_tp_group(),  # TP通信组
                    disabled=not is_allocation_symmetric(),  # 非对称分配时禁用
                ):
                    hidden_states, residual = layernorm(hidden_states, residual)  # 应用层归一化
            elif context.attn_tp_rank == 0:  # TP rank为0时
                hidden_states += residual  # 隐藏状态加上残差

            hidden_states, local_hidden_states = (  # 准备隐藏状态聚集
                get_global_dp_buffer(get_tp_group()),  # 从全局DP缓冲区获取输出
                hidden_states,  # 本地隐藏状态
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)  # DP部分聚集

            if not use_layer_norm_before_gather:  # 如果不在聚集前归一化
                dp_scatter(residual, hidden_states, forward_batch)  # DP散射残差
                if hidden_states.shape[0] != 0:  # 如果有token
                    hidden_states = layernorm(hidden_states)  # 应用层归一化（无残差）
        else:  # DP大小为1
            handled = False  # 是否已处理标志
            if (  # 如果可以应用全归约融合
                apply_aiter_all_reduce_fusion(hidden_states)  # aiter全归约融合
                or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])  # flashinfer全归约融合
            ) and hasattr(layernorm, "forward_with_allreduce_fusion"):  # 层归一化支持融合
                hidden_states, residual = layernorm.forward_with_allreduce_fusion(  # 融合全归约和层归一化
                    hidden_states, residual, use_attn_tp_group=True  # 使用注意力TP组
                )
                handled = True  # 标记已处理

            if not handled:  # 如果未处理
                quantize_communications = (  # 判断是否量化通信
                    not forward_batch.forward_mode.is_decode_or_idle()  # 非解码或空闲模式
                    and get_global_server_args().enable_quant_communications  # 全局启用量化通信
                )
                if quantize_communications:  # 如果量化通信
                    hidden_states = attention_tensor_model_parallel_quant_all_reduce(  # 量化全归约
                        hidden_states  # 隐藏状态
                    )
                else:  # 非量化通信
                    hidden_states = attention_tensor_model_parallel_all_reduce(  # 注意力TP全归约
                        hidden_states  # 隐藏状态
                    )
                if _is_npu and context.cache is not None:  # NPU环境且有缓存
                    _ = prepare_weight_cache(hidden_states, context.cache)  # 准备权重缓存
                hidden_states, residual = layernorm(hidden_states, residual)  # 应用层归一化
        return hidden_states, residual  # 返回隐藏状态和残差

    @staticmethod
    def _scatter_hidden_states_and_residual(  # 散射隐藏状态和残差
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        layernorm: torch.nn.Module,  # 层归一化模块
        context: CommunicateContext,  # 通信上下文
        *,  # 以下为关键字参数
        residual_input_mode,  # 残差输入模式
    ):
        input_hidden_states = hidden_states  # 保存原始隐藏状态引用
        hidden_states = hidden_states.tensor_split(context.attn_tp_size)[  # 按TP大小拆分
            context.attn_tp_rank  # 取当前rank的部分
        ]
        attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)  # 注意力TP归约散射
        if residual_input_mode == ScatterMode.TP_ATTN_FULL:  # 如果残差为TP注意力完整模式
            residual = residual.tensor_split(context.attn_tp_size)[context.attn_tp_rank]  # 拆分残差
        if hidden_states.shape[0] != 0:  # 如果有token
            hidden_states, residual = layernorm(hidden_states, residual)  # 应用层归一化
        return hidden_states, residual  # 返回隐藏状态和残差

    @staticmethod
    def _tp_all_reduce_with_scattered_residual(  # 带散射残差的TP全归约
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        layernorm: torch.nn.Module,  # 层归一化模块
        context: CommunicateContext,  # 通信上下文
    ):
        if hidden_states.shape[0] == 0:  # 如果没有token
            return hidden_states, hidden_states  # 返回空张量

        scattered_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]  # 按TP大小拆分取当前rank
        scattered_states += residual  # 加上残差
        residual = tensor_model_parallel_all_reduce(hidden_states)  # TP全归约作为残差
        hidden_states = layernorm(residual)  # 对残差应用层归一化
        return hidden_states, residual  # 返回隐藏状态和残差

    @staticmethod
    def _gather_hidden_states_and_residual_moe(  # MoE模式下聚集隐藏状态和残差
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch,  # 前向批次
        layernorm: torch.nn.Module,  # 层归一化模块
        context: CommunicateContext,  # 通信上下文
        *,  # 以下为关键字参数
        residual_input_mode,  # 残差输入模式
    ):
        """Allgather tokens for MoE when moe_dp_size < attn_cp_size.  # 当moe_dp_size < attn_cp_size时为MoE全聚集token

        Steps:  # 步骤
          1. Standard attn-TP all-reduce + optional DP allgather + layernorm (same as  # 1. 标准注意力TP全归约 + 可选DP全聚集 + 层归一化（同
             _gather_hidden_states_and_residual for the dp>1 case, or simple all-reduce  # _gather_hidden_states_and_residual的dp>1情况，或简单全归约
             + layernorm for dp==1).  # + 层归一化当dp==1）
          2. moe_cp allgather: gather tokens from cp_per_moe CP ranks so each rank holds  # 2. moe_cp全聚集：从cp_per_moe个CP rank聚集token，使每个rank持有
             all tokens for its MoE group.  # 其MoE组的所有token

        Residual is left at TP_ATTN_FULL throughout.  # 残差始终保持TP_ATTN_FULL
        """
        # Early return on empty tensor is safe for MOE_CP because:  # 空张量提前返回对MOE_CP是安全的，因为
        # - During CP extend: zigzag split guarantees all CP ranks have non-zero tokens,  # - CP扩展期间：zigzag拆分保证所有CP rank有非零token
        #   so no rank hits this path while others proceed to the allgather.  # 因此没有rank在其他rank继续全聚集时命中此路径
        # - During decode: moe_cp allgather is skipped (guarded by is_context_parallel_extend).  # - 解码期间：moe_cp全聚集被跳过（由is_context_parallel_extend保护）
        # - CUDA graph warmup: not applicable when --disable-piecewise-cuda-graph is used.  # - CUDA图预热：使用--disable-piecewise-cuda-graph时不适用
        if hidden_states.shape[0] == 0:  # 如果没有token
            return hidden_states, residual  # 提前返回

        # Step 1: Standard all-reduce/DP-allgather + layernorm (reuse existing logic).  # 步骤1：标准全归约/DP全聚集 + 层归一化（复用现有逻辑）
        hidden_states, residual = (
            CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual(
                hidden_states=hidden_states,  # 隐藏状态
                residual=residual,  # 残差
                forward_batch=forward_batch,  # 前向批次
                layernorm=layernorm,  # 层归一化
                context=context,  # 通信上下文
                residual_input_mode=residual_input_mode,  # 残差输入模式
            )
        )

        # Step 2: moe_cp allgather — gather across cp_per_moe CP ranks.  # 步骤2：moe_cp全聚集——从cp_per_moe个CP rank聚集
        # Only active during prefill (context-parallel extend); decode keeps existing path.  # 仅在预填充（上下文并行扩展）时激活；解码保持现有路径
        moe_cp_size = get_moe_cp_size()  # 获取MoE CP大小
        if (  # 如果满足MoE CP全聚集条件
            moe_cp_size > 1  # MoE CP大小大于1
            and hidden_states.shape[0] > 0  # 有token
            and forward_batch.forward_mode.is_context_parallel_extend()  # 是上下文并行扩展模式
            and forward_batch.attn_cp_metadata is not None  # 有CP元数据
        ):
            # Zigzag split can produce unequal token counts across CP ranks  # Zigzag拆分可能导致各CP rank的token数不等
            # (when seq_len % (cp_size * 2) != 0). NCCL allgather requires  # （当seq_len % (cp_size * 2) != 0时）。NCCL全聚集要求
            # equal input sizes, so pad to the max per-rank token count.  # 输入大小相等，因此填充到最大per-rank token数
            per_rank_tokens = forward_batch.attn_cp_metadata.per_rank_actual_token  # 获取每个rank的实际token数
            max_tokens = max(per_rank_tokens)  # 获取最大token数
            pad_size = max_tokens - hidden_states.shape[0]  # 计算填充大小
            if pad_size > 0:  # 如果需要填充
                hidden_states = torch.nn.functional.pad(  # 填充隐藏状态
                    hidden_states, [0, 0, 0, pad_size]  # 在最后一个维度后填充
                )

            output = torch.empty(  # 创建全聚集输出张量
                (max_tokens * moe_cp_size, hidden_states.shape[1]),  # 形状
                dtype=hidden_states.dtype,  # 数据类型
                device=hidden_states.device,  # 设备
            )
            moe_cp_all_gather_into_tensor(output, hidden_states)  # 执行MoE CP全聚集
            hidden_states = output  # 更新隐藏状态

        return hidden_states, residual  # 返回隐藏状态和残差


class CommunicateSummableTensorPairFn:  # 可求和张量对通信函数类
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""  # 如果需要，允许将(hidden_states, residual)设为(hidden_states + residual, None)

    @classmethod
    def execute(  # 执行通信
        cls,
        hidden_states_input_mode,  # 隐藏状态输入模式
        residual_input_mode,  # 残差输入模式
        output_mode,  # 输出模式
        context,  # 通信上下文
        **kwargs,  # 其他关键字参数
    ):
        return cls.get_fn(  # 获取并执行通信函数
            hidden_states_input_mode=hidden_states_input_mode,  # 隐藏状态输入模式
            residual_input_mode=residual_input_mode,  # 残差输入模式
            output_mode=output_mode,  # 输出模式
            context=context,  # 通信上下文
        )(context=context, **kwargs)  # 执行通信函数

    @staticmethod
    def get_fn(  # 根据输入输出模式获取对应的通信函数
        hidden_states_input_mode: ScatterMode,  # 隐藏状态输入模式
        residual_input_mode: ScatterMode,  # 残差输入模式
        output_mode: ScatterMode,  # 输出模式
        context: CommunicateContext,  # 通信上下文
    ):
        if context.is_same_group_size(  # 如果隐藏状态和输出组大小相同
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):  # 残差和输出组大小也相同
            return CommunicateSummableTensorPairFn._trivial  # 返回平凡函数

        if (  # FULL + TP_ATTN_FULL -> TP_ATTN_FULL
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return CommunicateSummableTensorPairFn._scatter_hidden_states  # 返回散射隐藏状态函数

        if (  # SCATTERED + SCATTERED -> TP_ATTN_FULL
            (hidden_states_input_mode == ScatterMode.SCATTERED)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return CommunicateSummableTensorPairFn._gather  # 返回聚集函数

        if (  # TP_ATTN_FULL + TP_ATTN_FULL -> SCATTERED
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return CommunicateSummableTensorPairFn._scatter  # 返回散射函数

        if (  # MOE_FULL + TP_ATTN_FULL -> TP_ATTN_FULL
            (hidden_states_input_mode == ScatterMode.MOE_FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return CommunicateSummableTensorPairFn._scatter_hidden_states_moe  # 返回MoE散射隐藏状态函数

        raise NotImplementedError(  # 其他模式未实现
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _trivial(  # 平凡函数，直接返回隐藏状态和残差
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
        **kwargs,  # 其他关键字参数
    ):
        return hidden_states, residual  # 直接返回

    @staticmethod
    def _scatter_hidden_states(  # 散射隐藏状态（FULL -> TP_ATTN_FULL）
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
        allow_reduce_scatter: bool = False,  # 是否允许归约散射
    ):
        if get_tensor_model_parallel_world_size() == get_attention_dp_size():  # 如果TP大小等于DP大小
            group = get_tp_group()  # 使用TP组
        else:  # 否则
            group = get_attention_tp_group()  # 使用注意力TP组
        hidden_states, global_hidden_states = (  # 准备散射
            get_local_dp_buffer(group),  # 从DP缓冲区获取输出
            hidden_states,  # 全局隐藏状态
        )
        if should_use_dp_reduce_scatterv():  # 如果应使用DP归约散射v
            get_tp_group().reduce_scatterv(  # 执行归约散射v
                global_hidden_states,  # 全局数据
                output=hidden_states,  # 输出
                sizes=get_dp_global_num_tokens(),  # 各rank的token数
            )
        elif allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():  # 允许归约散射且为最大长度模式
            dp_reduce_scatter_tensor(hidden_states, global_hidden_states)  # DP归约散射
        else:  # 其他情况
            dp_scatter(hidden_states, global_hidden_states, forward_batch)  # DP散射
        return hidden_states, residual  # 返回隐藏状态和残差

    @staticmethod
    def _gather(  # 聚集函数（SCATTERED -> TP_ATTN_FULL）
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
        **kwargs,  # 其他关键字参数
    ):
        hidden_states += residual  # 隐藏状态加上残差
        residual = None  # 残差设为None
        hidden_states, local_hidden_states = (  # 准备聚集
            get_local_dp_buffer(get_attention_tp_group()),  # 从DP缓冲区获取输出
            hidden_states,  # 本地隐藏状态
        )
        attn_tp_all_gather_into_tensor(  # 注意力TP全聚集
            hidden_states,  # 输出
            local_hidden_states,  # 本地数据
        )
        return hidden_states, residual  # 返回聚集后的隐藏状态和None残差

    @staticmethod
    def _scatter(  # 散射函数（TP_ATTN_FULL -> SCATTERED）
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
    ):
        assert residual is None, "not yet handled residual!=None"  # 断言残差为None
        tensor_list = list(hidden_states.tensor_split(context.attn_tp_size))  # 按TP大小拆分
        hidden_states = tensor_list[context.attn_tp_rank]  # 取当前rank的部分
        return hidden_states, residual  # 返回散射后的隐藏状态和残差

    @staticmethod
    def _scatter_hidden_states_moe(  # MoE模式下散射隐藏状态（MOE_FULL -> TP_ATTN_FULL）
        hidden_states: torch.Tensor,  # 隐藏状态
        residual: torch.Tensor,  # 残差
        forward_batch: ForwardBatch,  # 前向批次
        context: CommunicateContext,  # 通信上下文
        **kwargs,  # 其他关键字参数
    ):
        """Scatter MoE output back to TP_ATTN_FULL after MOE_FULL computation.  # MOE_FULL计算后将MoE输出散射回TP_ATTN_FULL

        After moe_tensor_model_parallel_all_reduce (which runs unconditionally since  # moe_tensor_model_parallel_all_reduce执行后（由于
        use_reduce_scatter=False for this path), all ranks in the moe_cp group hold the  # 此路径use_reduce_scatter=False，无条件运行），moe_cp组中的所有rank持有
        full MoE result for all cp_per_moe token chunks. We simply slice out this rank's  # 所有cp_per_moe token块的完整MoE结果。我们只需切出此rank的
        CP-local portion.  # CP本地部分

        If DP>1, further scatter back to the local DP slice.  # 如果DP>1，进一步散射回本地DP分片
        """
        # Only scatter back during prefill; decode was never allgathered so no-op.  # 仅在预填充时散射回；解码从未全聚集，因此无操作
        # Safe w.r.t. empty tensors: same reasoning as _gather_hidden_states_and_residual_moe  # 空张量安全性：与_gather_hidden_states_and_residual_moe相同理由
        # — CP extend always has non-zero tokens per rank, and decode skips this path.  # — CP扩展始终每个rank有非零token，解码跳过此路径
        moe_cp_size = get_moe_cp_size()  # 获取MoE CP大小
        if (  # 如果满足MoE CP散射条件
            moe_cp_size > 1  # MoE CP大小大于1
            and forward_batch.forward_mode.is_context_parallel_extend()  # 是上下文并行扩展模式
            and forward_batch.attn_cp_metadata is not None  # 有CP元数据
        ):
            moe_cp_rank = get_moe_cp_rank()  # 获取MoE CP rank
            # The allgather was padded to max_tokens_per_rank (equal chunks).  # 全聚集被填充到max_tokens_per_rank（等长块）
            # Extract this rank's actual (non-padded) tokens from its chunk.  # 从此rank的块中提取实际（非填充）token
            per_rank_tokens = forward_batch.attn_cp_metadata.per_rank_actual_token  # 获取每个rank的实际token数
            max_tokens_per_rank = max(per_rank_tokens)  # 获取最大per-rank token数
            actual_local_tokens = per_rank_tokens[moe_cp_rank]  # 获取当前rank的实际token数
            hidden_states = hidden_states.narrow(  # 窄化到当前rank的token范围
                0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens  # 从偏移开始，取实际token数
            ).contiguous()  # 确保内存连续

        # DP scatter (if DP attention is enabled)  # DP散射（如果DP注意力启用）
        if context.attn_dp_size > 1:  # 如果DP大小大于1
            if get_tensor_model_parallel_world_size() == get_attention_dp_size():  # TP大小等于DP大小
                group = get_tp_group()  # 使用TP组
            else:  # 否则
                group = get_attention_tp_group()  # 使用注意力TP组
            hidden_states_output, global_hidden_states = (  # 准备DP散射
                get_local_dp_buffer(group),  # 从DP缓冲区获取输出
                hidden_states,  # 全局隐藏状态
            )
            dp_scatter(hidden_states_output, global_hidden_states, forward_batch)  # DP散射
            hidden_states = hidden_states_output  # 更新隐藏状态

        return hidden_states, residual  # 返回隐藏状态和残差
