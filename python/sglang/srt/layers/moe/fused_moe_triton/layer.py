# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py
# 本文件实现了SGLang中MoE（混合专家）模型的核心融合层（FusedMoE）。
# 主要功能包括：专家权重的创建、加载（支持多种量化方案）、分发与合并（dispatch/combine）、
# 以及前向推理（forward）。该层将门控投影（gate_proj/w1）和上升投影（up_proj/w3）融合为w13，
# 下降投影（down_proj/w2）单独处理，支持张量并行和专家并行。

import logging
from enum import Enum
from functools import cached_property
from typing import List, Optional, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.compilation.piecewise_context_manager import (
    get_forward_context,
    is_in_piecewise_cuda_graph,
)
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    create_kt_config_from_server_args,
)
from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.moe.utils import RoutingMethodType, is_deepep_class_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMxInt4MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    print_info_once,
    round_up,
)
from sglang.srt.utils.custom_op import register_custom_op

_is_hip = is_hip()  # 是否为AMD HIP平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU是否支持AMX指令集
_is_cpu = is_cpu()  # 是否为CPU平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AMD AITER库


def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:
    """根据MoE运行配置创建对应的token分发器(dispatcher)。
    
    根据不同的all-to-all后端选择不同的分发器实现：
    - 无后端/MegaMoE/Ascend FuseEP: 使用标准分发器
    - DeepEP/Mooncake/Mori/NIXL: 使用DeepEP分发器（支持TBO）
    - FlashInfer: 使用FlashInfer分发器
    """
    a2a_backend = get_moe_a2a_backend()
    if (
        a2a_backend.is_none()
        or a2a_backend.is_megamoe()
        or a2a_backend.is_ascend_fuseep()
    ):
        # ascend_fuseep bypasses the dispatcher abstraction (see
        # forward_fuseep in hardware_backend/npu/moe/fuseep.py); a
        # StandardDispatcher is created but never invoked.
        # 标准分发器：不进行跨节点token通信，本地处理
        return StandardDispatcher(moe_runner_config)
    elif (
        a2a_backend.is_deepep()
        or a2a_backend.is_mooncake()
        or a2a_backend.is_mori()
        or a2a_backend.is_nixl()
    ):
        # DeepEP类分发器：支持低延迟和正常两种模式，支持TBO（两批次重叠）
        return MaybeTboDeepEPDispatcher(
            group=(
                get_tp_group().device_group
                if not a2a_backend.is_mori()
                else get_tp_group()
            ),
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
            deepep_mode=get_deepep_mode(),
            async_finish=True,
            return_recv_hook=True,
        )
    elif a2a_backend.is_flashinfer():
        # FlashInfer分发器：使用FlashInfer库进行token分发
        return FlashinferDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
        )
    else:
        raise NotImplementedError(f"Unsupported a2a backend: {a2a_backend}")


class FusedMoeWeightScaleSupported(Enum):
    """枚举类：定义MoE权重缩放支持的粒度类型。
    
    - TENSOR: 按张量粒度缩放（每个专家一个缩放值）
    - CHANNEL: 按通道粒度缩放（每个输出通道一个缩放值）
    - GROUP: 按分组粒度缩放（每组权重一个缩放值）
    - BLOCK: 按块粒度缩放（每个权重块一个缩放值）
    """
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.
    # FusedMoE层：MoE模型的核心融合层实现

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).
    # 本层包含MergedColumnParallel权重（gate_up_proj/w13）和RowParallelLinear权重（down_proj/w2）

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model  # 模型中专家的总数
        top_k: Number of experts selected for each token  # 每个token选择的专家数量
        hidden_size: Input hidden state size of the transformer  # Transformer输入隐藏状态维度
        intermediate_size: Intermediate size of the experts  # 专家中间层维度
        params_dtype: Data type for the parameters.  # 参数的数据类型
        reduce_results: Whether to apply all_reduce on the output of the layer  # 是否对输出进行all_reduce归约
        quant_config: Quantization configuration.  # 量化配置
        inplace: suggestion to compute inplace (modify input activation).  # 是否原地计算（修改输入激活）
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
        use_weight_loader_fused: bool = False,
        with_bias=False,
        routing_method_type: Optional[RoutingMethodType] = None,
        is_gated: bool = True,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.layer_id = layer_id  # 当前MoE层的ID
        self.top_k = top_k  # Top-K路由中每个token选择的专家数
        self.hidden_size = hidden_size  # 隐藏层维度大小
        self.num_experts = num_experts  # 专家总数
        self.num_fused_shared_experts = num_fused_shared_experts  # 融合的共享专家数量

        # 判断是否使用FlashInfer CUTLASS MoE后端
        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        # 获取专家并行和张量并行的rank和world_size
        self.moe_ep_size = get_moe_expert_parallel_world_size()  # 专家并行度
        self.moe_ep_rank = get_moe_expert_parallel_rank()  # 当前专家并行rank
        self.moe_tp_size = get_moe_tensor_parallel_world_size()  # 张量并行度
        self.moe_tp_rank = get_moe_tensor_parallel_rank()  # 当前张量并行rank

        # DeepEP: each rank has its own shared expert slot, so total shared
        # weight slots = num_fused_shared_experts * ep_size.
        # AMD/Standard: shared experts are global, slots = num_fused_shared_experts.
        # DeepEP模式下：每个rank有自己的共享专家槽位，总数 = 共享专家数 * EP大小
        # AMD/标准模式下：共享专家是全局的，槽位数 = 共享专家数
        if num_fused_shared_experts > 0 and is_deepep_class_backend():
            num_shared_slots = num_fused_shared_experts * self.moe_ep_size
        else:
            num_shared_slots = num_fused_shared_experts

        assert (num_experts - num_shared_slots) % self.moe_ep_size == 0
        self._num_global_routed = num_experts - num_shared_slots  # 全局路由专家数（不含共享专家）
        self._num_local_routed = self._num_global_routed // self.moe_ep_size  # 本地路由专家数
        self.num_local_experts = self._num_local_routed + num_fused_shared_experts  # 本地专家总数 = 本地路由 + 共享
        self._has_fused_shared = num_fused_shared_experts > 0  # 是否有融合共享专家

        assert intermediate_size % self.moe_tp_size == 0
        # 中间层维度按张量并行度切分
        self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
        self.reduce_results = reduce_results  # 是否对输出进行all_reduce
        self.use_presharded_weights = use_presharded_weights  # 是否使用预切分权重

        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernels()  # 是否使用Triton内核

        # 判断是否使用FlashInfer TRT-LLM MoE后端
        self.use_flashinfer_trtllm_moe = (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        )
        self.use_deep_gemm = get_moe_runner_backend().is_deep_gemm()  # 是否使用DeepGEMM后端

        # flashinfer_trtllm kernel requires intermediate_size to be a multiple of 128
        # Pad the intermediate_size_per_partition if necessary
        # FlashInfer TRT-LLM内核要求中间维度是128的倍数，需要时进行填充
        if (
            self.use_flashinfer_trtllm_moe
            and self.intermediate_size_per_partition % 128 != 0
        ):
            self.intermediate_size_per_partition = round_up(
                self.intermediate_size_per_partition, 128
            )

        self.quant_config = quant_config
        self.use_flashinfer_mxfp4_moe = get_moe_runner_backend().is_flashinfer_mxfp4()  # 是否使用MX FP4 MoE
        # TODO maybe we should remove this `if`, since `Mxfp4MoEMethod` does another round-up logic
        # MX FP4量化需要对hidden_size向上取整到256的倍数
        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.use_flashinfer_mxfp4_moe
        ):
            hidden_size = round_up(hidden_size, 256)
        self.hidden_size = hidden_size

        # 构建MoE运行配置对象
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
            swiglu_limit=swiglu_limit,
            is_gated=is_gated,
            routing_method_type=routing_method_type,
        )

        # 根据量化配置和KT配置选择量化方法
        self.quant_method: Optional[FusedMoEMethodBase] = None
        server_args = get_global_server_args()
        # 创建KT（Kernel Transfer）专家并行配置
        kt_config = create_kt_config_from_server_args(server_args, layer_id)
        if kt_config is not None:
            # 如果有KT配置，将GPU方法包装在KT方法中
            if quant_config is not None:
                gpu_method = quant_config.get_quant_method(self, prefix)
            else:
                gpu_method = UnquantizedFusedMoEMethod(self.use_triton_kernels)
            self.quant_method = KTEPWrapperMethod(gpu_method, kt_config)
        else:
            # 无KT配置：使用量化方法或未量化方法
            if quant_config is not None:
                self.quant_method = quant_config.get_quant_method(self, prefix)
            if self.quant_method is None:
                self.quant_method = UnquantizedFusedMoEMethod(
                    self.use_triton_kernels,
                    self.use_flashinfer_trtllm_moe,
                    self.use_deep_gemm,
                )

        # 创建专家权重（w13权重、w2权重及对应的缩放因子等）
        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=(
                self.weight_loader
                if not use_weight_loader_fused
                else self.weight_loader_fused
            ),
            with_bias=with_bias,
            moe_intermediate_size=intermediate_size,
        )

        # 创建MoE运行器并初始化token分发器
        self.quant_method.create_moe_runner(self, self.moe_runner_config)
        self.dispatcher = create_moe_dispatcher(self.moe_runner_config)
        self._use_ascend_fuseep = get_moe_a2a_backend().is_ascend_fuseep()  # 是否使用昇腾FuseEP

        # FlashInfer TRT-LLM MoE后端不支持原地计算
        if (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            or get_moe_runner_backend().is_flashinfer_trtllm()
        ):
            if self.moe_runner_config.inplace:
                print_info_once(
                    "Setting inplace to False for FlashInfer TRTLLM MoE backend."
                )
            self.moe_runner_config.inplace = False

        # 判断是否应在TopK阶段融合routed_scaling_factor
        # 当使用ModelOpt NV FP4、FP8+Cutlass/FlashInfer TRT-LLM路由后端、
        # 或未量化+FlashInfer TRT-LLM路由后端时，需要在TopK中融合缩放因子
        self.should_fuse_routed_scaling_factor_in_topk = (
            isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
            or (
                isinstance(self.quant_method, Fp8MoEMethod)
                and (
                    get_moe_runner_backend().is_cutlass()
                    or get_moe_runner_backend().is_flashinfer_trtllm_routed()
                )
            )
            or (
                isinstance(self.quant_method, UnquantizedFusedMoEMethod)
                and get_moe_runner_backend().is_flashinfer_trtllm_routed()
            )
        )

        self.routing_method_type = routing_method_type  # 路由方法类型

        # overlap args
        # 重叠计算参数，用于GEMM计算与通信的重叠
        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        if self.quant_method is not None and hasattr(self.quant_method, "runner"):
            self.runner = self.quant_method.runner  # MoE运行器实例

    @cached_property
    def use_padded_loading(self) -> bool:
        """判断是否需要使用填充加载方式。
        
        以下情况需要使用填充加载：
        1. CPU平台（始终需要）
        2. GPU + FlashInfer TRT-LLM填充（中间维度已填充到128的倍数）
        3. GPU + AITER填充
        """
        # This handles the case where the loaded weights are smaller than the padded expert_data
        # Use narrow_padded_param_and_loaded_weight for:
        # 1. CPU (always)
        # 2. GPU with flashinfer_trtllm padding (when intermediate_size is padded to 128)
        # 3. GPU with Aiter padding
        aiter_padded = (
            _use_aiter
            and hasattr(self, "w2_weight")
            and getattr(self.w2_weight, "weight_padded", False)
        )

        return _is_cpu or self.use_flashinfer_trtllm_moe or aiter_padded

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        """加载每个张量粒度的权重缩放因子。
        
        对于w1/w3（门控/上升投影），将缩放值存储在对应索引位置。
        对于w2（下降投影），直接存储缩放值。
        """
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            # w1和w3的缩放因子需要分别保存，因为后续可能需要重新量化
            idx = 0 if shard_id == "w1" else 1  # w1在索引0，w3在索引1
            if self.moe_runner_config.is_gated:
                param_data[expert_id][idx] = loaded_weight
            else:
                param_data[expert_id] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            # w2是行并行（RowParallel），缩放因子直接存储
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        """加载模型权重或分组权重缩放因子。
        
        根据shard_id将权重分发到w2或w13的加载方法中。
        """
        # Load grouped weight scales for group quantization
        # or model weights
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
        elif shard_id in ("w1", "w3", "w13"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        """加载每个通道粒度的权重缩放因子。
        
        w2的缩放因子直接拷贝，w1/w3的缩放因子通过_load_w13加载。
        """
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        """加载w13权重（w1门控投影 + w3上升投影的融合权重）。
        
        处理张量并行的切分、填充加载、以及w1/w3在w13中的排列顺序。
        对于gated模型，w13的前半部分是w1，后半部分是w3。
        """
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        assert shard_id in {"w1", "w3", "w13"}

        if is_bias:
            # if this weight is a bias, the last dimension must be the sharded dimension
            # 偏置项的最后一个维度是切分维度
            shard_dim = -1

        if shard_id in {"w1", "w3"} and self.moe_runner_config.is_gated:
            # non-fused version
            # 非融合版本：w1和w3各占w13的一半
            shard_size = expert_data.shape[shard_dim] // 2
        elif shard_id in {"w13"} or (
            shard_id in {"w1", "w3"} and not self.moe_runner_config.is_gated
        ):
            # fused version
            # 融合版本：w13全部用于当前shard
            shard_size = expert_data.shape[shard_dim]
        else:
            raise NotImplementedError

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        # w3, up_proj: Load into second logical weight of w13.
        # trtllm cutlass kernel assumes differently
        # 确定w1/w3在w13中的起始位置
        # 某些量化方法（如TRT-LLM CUTLASS）要求w1和w3的顺序互换
        switch_w13 = getattr(self.quant_method, "load_up_proj_weight_first", False)
        if (
            (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3")
        ) and self.moe_runner_config.is_gated:
            start = shard_size  # w3（或交换后的w1）从后半部分开始
        else:
            start = 0  # w1（或交换后的w3）从前半部分开始

        if self.use_padded_loading:
            # 填充加载模式：使用narrow_padded_param_and_loaded_weight处理填充后的参数
            if _is_cpu and is_bias:
                shard_dim = 1
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                start,
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            # 非填充加载模式：直接切片加载
            if not self.use_presharded_weights:
                if not is_bias and self.use_triton_kernels:
                    # do not transpose for bias
                    # Triton内核需要转置权重（偏置不需要）
                    loaded_weight = loaded_weight.transpose(-2, -1)
                # 按张量并行rank切分加载的权重
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

            # 按起始位置切分专家数据
            expert_data = expert_data.narrow(shard_dim, start, shard_size)
        expert_data.copy_(loaded_weight)  # 将加载的权重拷贝到专家数据中

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        """Load w2 weights for down projection.
        # 加载w2权重（下降投影权重）

        Args:
            expert_data: The expert data tensor to load into  # 要加载到的专家数据张量
            shard_dim: The dimension to shard along  # 切分维度
            shard_id: The shard ID (must be "w2")  # 切片ID（必须为"w2"）
            loaded_weight: The weight tensor to load from  # 要加载的权重张量
            tp_rank: The tensor parallel rank  # 张量并行rank
        """
        if not isinstance(expert_data, torch.Tensor) or not isinstance(
            loaded_weight, torch.Tensor
        ):
            raise ValueError("expert_data and loaded_weight must be torch.Tensor")

        if (
            self.quant_config is not None
            and "modelopt" in self.quant_config.get_name()
            and (expert_data.dim() != 2 or loaded_weight.dim() != 2)
        ):
            raise ValueError(
                f"Expected 2D tensors, got expert_data shape {expert_data.shape} and loaded_weight shape {loaded_weight.shape}"
            )

        if shard_id != "w2":
            raise ValueError(f"shard_id must be 'w2', got {shard_id}")

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        # w2是行并行（RowParallel），在输入维度上进行TP切分
        if is_bias:
            # this expert_data is a bias, not weight,
            # for w2_weight_bias in TP, it does not need to be sharded
            # 偏置项不需要TP切分
            shard_size = expert_data.shape[-1]
        else:
            # this parameter is a weight matrix
            # for w2 in TP, it shards the input_features, i.e., shard_dim=2
            # w2权重矩阵：在输入特征维度（shard_dim=2）上进行TP切分
            shard_size = expert_data.shape[shard_dim]

        if self.use_padded_loading:
            # 填充加载模式
            if _is_cpu and is_bias:
                shard_dim = 1
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                0,  # param_data_start  # w2的起始位置始终为0
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            # 非填充加载模式
            if not is_bias and not self.use_presharded_weights:
                if self.use_triton_kernels:
                    loaded_weight = loaded_weight.transpose(-2, -1)  # Triton内核需要转置
                # 按张量并行rank切分权重
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)  # 将权重拷贝到专家数据中

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        """加载单个值（如input_scale），直接按专家ID存储。"""
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        """加载分组索引（g_idx），用于分组量化。
        
        w2的g_idx通过_load_w2加载，w1/w3的g_idx直接拷贝。
        """
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """将全局专家ID映射为本地专家ID。
        
        根据当前EP rank计算本地路由专家的范围，
        如果专家ID不在本地路由范围内，则检查是否为共享专家。
        返回-1表示该专家不属于当前rank。
        """
        start_idx = self.moe_ep_rank * self._num_local_routed  # 本地路由专家的起始索引
        end_idx = start_idx + self._num_local_routed  # 本地路由专家的结束索引
        if start_idx <= expert_id < end_idx:
            # 该专家属于本rank的路由专家
            return expert_id - start_idx
        elif self._has_fused_shared and expert_id >= self._num_global_routed:
            # 该专家是共享专家，映射到本地路由专家之后的槽位
            return expert_id - self._num_global_routed + self._num_local_routed
        else:
            return -1  # 该专家不属于当前rank

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: Optional[int],
    ) -> None:
        """权重加载器：将检查点中的权重加载到模型参数中。
        
        支持多种权重类型的加载，包括：
        - 模型权重（weight）
        - 缩放因子（scale/input_scale/weight_scale）
        - 分组索引（g_idx）
        - 偏置（bias）
        
        处理全局专家到本地专家的映射，以及EPLB（专家位置负载均衡）场景。
        """
        # if expert_id is None, then
        # all the experts are loaded at the same time
        # expert_id为None时，所有专家同时加载（用于mxfp4静态配置）
        if (
            not expert_id
            and self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        # 处理EPLB（专家位置负载均衡）场景
        global_expert_location_metadata = get_global_expert_location_metadata()
        if global_expert_location_metadata is None:
            # 无EPLB：进行全局到本地的专家ID映射
            if not getattr(param, "_sglang_require_global_experts", False):
                expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
                if expert_id == -1:
                    return  # 该专家不属于当前rank，跳过

            self._weight_loader_impl(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return

        # 有EPLB：需要将逻辑专家ID映射为物理专家ID
        require_global_experts = getattr(param, "_sglang_require_global_experts", False)
        shared_expert_id = (
            expert_id - global_expert_location_metadata.num_logical_experts
            if self._has_fused_shared and expert_id is not None
            else -1
        )
        if 0 <= shared_expert_id < self.num_fused_shared_experts:
            # Checkpoint shared experts start after logical routed experts, while
            # local fused MoE weights store them after physical routed experts.
            # 共享专家：检查点中的共享专家排在逻辑路由专家之后，
            # 而本地融合MoE权重中共享专家排在物理路由专家之后
            if require_global_experts and is_deepep_class_backend():
                physical_expert_ids = [
                    rank * self.num_local_experts
                    + self._num_local_routed
                    + shared_expert_id
                    for rank in range(self.moe_ep_size)
                ]
            else:
                physical_expert_ids = [self._num_global_routed + shared_expert_id]
        else:
            # 路由专家：通过EPLB元数据进行逻辑到物理的映射
            physical_expert_ids = (
                global_expert_location_metadata.logical_to_all_physical(
                    self.layer_id, expert_id, require_global_experts
                )
            )

        # 对每个物理专家ID执行权重加载
        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        """物理权重加载器：处理物理专家ID到本地专家ID的映射。
        
        对于KT EP包装方法，还检查GPU专家数量限制。
        """
        # WARN: This makes the `expert_id` mean "local" and "global" in different cases
        # 注意：此方法中expert_id可能代表"本地"或"全局"专家ID
        if not getattr(param, "_sglang_require_global_experts", False):
            expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if expert_id < 0 or expert_id >= self.num_local_experts:
                return  # 专家不属于当前rank，跳过

        # KT EP方法：检查GPU专家数量限制
        if isinstance(
            self.quant_method,
            KTEPWrapperMethod,
        ):
            if self.quant_method.num_gpu_experts != -1:
                if expert_id >= self.quant_method.num_gpu_experts:
                    return  # 超出GPU专家数量限制，跳过

        self._weight_loader_impl(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )

    def _load_gguf_weight(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        expert_id: int,
        tp_rank: int,
    ) -> bool:
        """Handle GGUF weight loading.
        # 处理GGUF格式的权重加载

        Args:
            param: The parameter to load the weight into.  # 目标参数
            loaded_weight: The weight tensor to load.  # 源权重张量
            shard_id: The shard ID (w1, w2, or w3).  # 切片ID
            expert_id: The expert ID.  # 专家ID
            tp_rank: The tensor parallel rank.  # 张量并行rank

        Returns:
            True if the weight was handled as a GGUF weight, False otherwise.
            # 如果权重作为GGUF权重处理则返回True，否则返回False
        """
        is_gguf_weight = getattr(param, "is_gguf_weight", False)  # 是否为GGUF权重
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)  # 是否为GGUF权重类型

        if is_gguf_weight_type:
            # Store weight type for this expert
            # 存储此专家的权重类型
            param.weight_type = loaded_weight.item()
            return True

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            if self.moe_tp_size > 1:
                # 在TP>1时，对输出维度进行切分
                if shard_id in ["w1", "w3", "w2"] and output_dim == 0:
                    shard_size = loaded_weight.size(0) // self.moe_tp_size
                    start_idx = tp_rank * shard_size
                    loaded_weight = loaded_weight.narrow(
                        0, start_idx, shard_size
                    ).clone()

            # Store in data_container with expert/shard info
            # 将权重按专家ID和切片ID存储到expert_data_map中
            if not hasattr(param, "expert_data_map"):
                param.expert_data_map = {}

            key = (expert_id, shard_id)
            param.expert_data_map[key] = loaded_weight
            param.data_container.append(loaded_weight)
            return True

        return False

    def _weight_loader_impl(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        """权重加载的核心实现：根据权重名称和类型分发到不同的加载方法。
        
        处理的权重类型包括：
        - GGUF权重
        - 模型权重（weight）
        - 输入缩放因子（input_scale）
        - 分组索引（g_idx）
        - 权重缩放因子（scale/weight_scale）
        - 偏置（bias）
        """
        tp_rank = self.moe_tp_rank

        # Special case for GGUF weights
        # 特殊情况：GGUF权重
        if self._load_gguf_weight(param, loaded_weight, shard_id, expert_id, tp_rank):
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        # compressed-tensors格式的打包权重存储时是转置的，需要转置回来
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        if method.__class__.__name__ == "KTEPWrapperMethod":
            method = method.gpu_method  # 获取KT包装内的GPU方法

        # For flashinfer TRT-LLM BF16 path, process_weights_after_loading reshapes
        # expert weights into block layout. During weight update, we must restore
        # canonical load-time shapes before copying checkpoint tensors.
        # FlashInfer TRT-LLM BF16路径：加载时需要恢复标准形状
        if isinstance(method, UnquantizedFusedMoEMethod):
            method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer=self,
                param=param,
                weight_name=weight_name,
            )

        # compressed-tensors的WNA16格式权重需要转置
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MarlinMoE",
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Flashinfer assumes w31 format for w13_weight. Same for the scales.
        # FlashInfer假设w13权重为w31格式（w3在前，w1在后），需要交换shard_id
        if self.use_flashinfer_trtllm_moe and (
            isinstance(method, ModelOptNvFp4FusedMoEMethod)
            or isinstance(method, Fp8MoEMethod)
            or isinstance(method, UnquantizedFusedMoEMethod)
            or isinstance(method, CompressedTensorsMxInt4MoE)
        ):
            shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]  # 交换w1和w3

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        # 根据shard_id获取切分维度：w1/w3在输出维度（dim=0）切分，w2在输入维度（dim=1）切分
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]  # 获取当前专家的数据

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        # 判断权重是否已转置：某些量化方案（GPTQ、compressed-tensors）要求转置存储
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if self.use_triton_kernels:
            is_transposed = True  # Triton内核要求转置
        if is_transposed:
            shard_dim = int(not shard_dim)  # 转置时翻转切分维度

        # Case input scale: input_scale loading is only supported for fp8
        # 情况1：输入缩放因子（input_scale），仅FP8量化支持
        if "input_scale" in weight_name:
            # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust input_scale for e4m3fnuz (AMD)
            # AMD平台上INT4-FP8混合量化需要调整input_scale
            if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                loaded_weight = loaded_weight * 2.0

            # this is needed for compressed-tensors only
            # compressed-tensors需要将缩放因子移到参数所在设备
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                (
                    "compressed" in method.__class__.__name__.lower()
                    or "w4afp8" in self.quant_config.get_name()
                )
                and (param.data[expert_id] != 1).any()
                and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any()
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        # 情况2：分组索引（g_idx），用于分组量化
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        # 情况3：ModelOpt量化方法的权重缩放
        if "ModelOpt" in method.__class__.__name__:
            # Determine per-tensor weight scale patterns based on variant
            is_fp4_variant = isinstance(method, ModelOptNvFp4FusedMoEMethod)

            # FP4 uses "weight_scale_2" for per-tensor, FP8 uses "weight_scale" for per-tensor
            # FP4变体使用"weight_scale_2"表示每张量缩放，FP8使用"weight_scale"
            per_tensor_conditions = (
                "weight_scale_2" in weight_name
                if is_fp4_variant
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name

            if per_tensor_conditions:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            return

        # Case weight scales and zero_points
        # 情况4：权重缩放因子和零点（scale/zero/offset）
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust INT4 column-wise scaling number to e4m3fnuz (AMD)
                # AMD平台INT4-FP8混合量化需要调整通道缩放
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 0.5

                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                # 分组/块粒度缩放：使用与模型权重相同的加载方式
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust FP8 per-tensor scaling number for e4m3fnuz (AMD)
                # AMD平台INT4-FP8混合量化需要调整张量缩放
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 2.0

                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        # 情况5：权重形状信息（weight_shape），仅compressed-tensors需要
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        # 情况6：模型权重（weight），最常见的情况
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        # 情况7：偏置（bias），仅modelslim量化方法支持
        if (
            "bias" in weight_name
            and self.quant_config.quant_description["quant_method"] == "modelslim"
        ):
            self._load_per_channel_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def weight_loader_fused(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
    ) -> None:
        """融合权重加载器：一次性加载w13或w2的融合权重。
        
        与weight_loader不同，此方法加载的是已经融合好的w13权重
        （gate_proj + up_proj已拼接），不需要按专家逐个加载。
        """
        tp_rank = self.moe_tp_rank

        # mxfp4静态配置的特殊处理：所有专家同时加载
        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            elif "scale" in weight_name:
                param.data.copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO: check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        # compressed-tensors的WNA16格式权重需要转置
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            else loaded_weight
        )

        if shard_id not in ("w13", "w2"):
            raise ValueError(f"shard_id must be ['w13','w2'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        # 融合权重的切分维度映射：w13和w2有不同的默认和转置切分维度
        SHARD_ID_TO_SHARDED_DIM = {"w13": 1, "w2": 2}  # w13在dim=1切分，w2在dim=2切分
        SHARD_ID_TO_SHARDED_DIM_TRANSPOSE = {"w13": 2, "w2": 1}  # 转置后的切分维度

        expert_data = param.data
        is_bias = expert_data.dim() == 2  # 2维张量表示偏置

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)

        if self.use_triton_kernels:
            is_transposed = True
        shard_dim = (
            SHARD_ID_TO_SHARDED_DIM[shard_id]
            if not is_transposed
            else SHARD_ID_TO_SHARDED_DIM_TRANSPOSE[shard_id]
        )

        # Case model weights
        # 融合权重只处理模型权重
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
            return
        else:
            logging.warning(
                f"Unsupported weight_name {weight_name} for FusedMoE weight_loader_fused. Nothing is loaded."
            )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        """FusedMoE的前向传播入口。
        
        处理三种情况：
        1. 昇腾FuseEP：使用专门的forward_fuseep实现
        2. CUDA图分段模式：使用自定义算子实现（避免图断点）
        3. 普通模式：直接调用forward_impl
        """
        if self._use_ascend_fuseep:
            # 昇腾NPU的FuseEP实现
            from sglang.srt.hardware_backend.npu.moe.fuseep import forward_fuseep

            return forward_fuseep(self, hidden_states, topk_output)
        if is_in_piecewise_cuda_graph():
            # CUDA图分段模式：使用自定义算子以避免图断点
            if TopKOutputChecker.format_is_standard(topk_output):
                return moe_forward_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.topk_weights,
                    topk_output.topk_ids,
                    topk_output.router_logits,
                    self.layer_id,
                )
            elif TopKOutputChecker.format_is_bypassed(topk_output):
                return fused_moe_bypassed_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.router_logits,
                    topk_output.topk_config.top_k,
                    topk_output.topk_config.topk_group,
                    topk_output.topk_config.num_expert_group,
                    topk_output.topk_config.correction_bias,
                    topk_output.topk_config.renormalize,
                    self.layer_id,
                )
            else:
                # Make sure there is torch lib op registration for the whole moe layer
                # 确保整个MoE层有torch库算子注册
                return self.forward_impl(hidden_states, topk_output)
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        """FusedMoE前向传播的核心实现。
        
        执行流程：
        1. dispatch: 将token分发到对应的专家
        2. run_moe_core: 在专家上执行计算
        3. combine: 将专家输出合并回原始token顺序
        4. 可选的all_reduce: 在TP/EP组内归约结果
        """
        origin_hidden_states_dim = hidden_states.shape[-1]  # 保存原始隐藏维度（可能因填充而不同）
        assert self.quant_method is not None

        # 步骤1：分发token到对应专家
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )

        # 步骤2：在专家上执行GEMM计算
        combine_input = self.run_moe_core(
            dispatch_output=dispatch_output,
        )

        # 步骤3：合并专家输出
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            final_hidden_states = self.dispatcher.combine(combine_input=combine_input)

            # TODO: should we add some conditions here?
            # 裁剪填充维度，恢复到原始隐藏维度
            final_hidden_states = final_hidden_states[
                ..., :origin_hidden_states_dim
            ].contiguous()

        # 步骤4：如果需要且TP/EP度大于1，执行all_reduce归约
        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:
        """运行MoE核心计算：调用量化方法的apply函数执行专家GEMM。"""
        # TODO: consider using symmetric memory
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        """创建专家参数映射表，将检查点中的权重名称映射到模型参数。
        
        返回格式：(param_name, weight_name, expert_id, shard_id)
        gate_proj和up_proj映射到w13前缀，down_proj映射到w2前缀。
        """
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    @classmethod
    def make_expert_params_mapping_fused(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
    ):
        """创建融合权重的参数映射表（w13和w2已融合，不需要按专家逐个映射）。"""
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
        ]

    @classmethod
    def make_expert_params_mapping_fused_mxfp4(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
        ckpt_gate_up_proj_scale_name: str,
        ckpt_down_proj_scale_name: str,
    ):
        """创建MX FP4融合权重的参数映射表，包含权重、偏置和缩放因子。"""
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
            (
                "experts.w13_weight_scale",
                f"experts.{ckpt_gate_up_proj_scale_name}",
                "w13",
            ),
            ("experts.w2_weight_scale", f"experts.{ckpt_down_proj_scale_name}", "w2"),
        ]

    @classmethod
    def make_expert_input_scale_params_mapping(
        cls,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        """创建输入缩放因子的参数映射表（用于FP8量化）。"""
        # (param_name, weight_name, expert_id, shard_id)
        return [
            (
                "experts.w13_" if shard_id in ["w1", "w3"] else "experts.w2_",
                f"experts.{expert_id}.{shard_id}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id in ["w1", "w2", "w3"]
        ]

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        """设置GEMM计算与通信的重叠参数，用于提升吞吐量。"""
        if hasattr(self, "runner"):
            self.runner.set_overlap_args(down_gemm_overlap_args, meta_overlap_args)
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = down_gemm_overlap_args
            self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        """清除重叠计算参数。"""
        if hasattr(self, "runner"):
            self.runner.clear_overlap_args()
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = None
            self.meta_overlap_args = None

    def materialize_gguf_weights(self) -> None:
        """Process weights after loading, especially for GGUF quantization.
        # 加载后处理权重，专门用于GGUF量化格式

        This materializes GGUF UninitializedParameters from their data_containers.
        # 将GGUF的UninitializedParameters从data_containers中物化为实际张量
        """

        for name, param in list(self.named_parameters()):
            is_gguf_weight = getattr(param, "is_gguf_weight", False)

            if is_gguf_weight and isinstance(param, UninitializedParameter):
                data_container = getattr(param, "data_container", [])
                expert_data_map = getattr(param, "expert_data_map", {})
                tensor_shape = getattr(param, "tensor_shape", None)

                if data_container and tensor_shape:
                    # Determine the structure from expert_data_map
                    num_experts = tensor_shape[0]

                    # Collect weights by expert
                    # 按专家ID收集权重
                    expert_weights = {}
                    for (expert_id, shard_id), weight in expert_data_map.items():
                        if expert_id not in expert_weights:
                            expert_weights[expert_id] = {}
                        expert_weights[expert_id][shard_id] = weight

                    # Build the full tensor
                    # 构建完整张量
                    if "w13" in name:
                        # w13 is gate+up fused
                        # w13是gate+up融合权重，按专家拼接w1和w3
                        weight_list = []
                        for e in range(num_experts):
                            if e in expert_weights:
                                w1 = expert_weights[e].get("w1")
                                w3 = expert_weights[e].get("w3")

                                if w1 is not None and w3 is not None:
                                    fused = torch.cat([w1, w3], dim=0)  # 拼接w1和w3
                                    weight_list.append(fused)

                        if weight_list:
                            stacked = torch.stack(weight_list, dim=0)
                            param.materialize(stacked.shape, dtype=stacked.dtype)
                            param.data.copy_(stacked)
                    elif "w2" in name:
                        # w2 is down projection
                        # w2是下降投影权重，直接堆叠
                        weight_list = []
                        for e in range(num_experts):
                            if e in expert_weights and "w2" in expert_weights[e]:
                                w2_weight = expert_weights[e]["w2"]
                                weight_list.append(w2_weight)

                        if weight_list:
                            stacked = torch.stack(weight_list, dim=0)
                            param.materialize(stacked.shape, dtype=stacked.dtype)
                            param.data.copy_(stacked)


@register_custom_op(out_shape="hidden_states")
def moe_forward_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """CUDA图分段模式下的MoE前向传播实现（标准TopK输出格式）。
    
    作为自定义算子注册，用于在CUDA图中避免图断点。
    """
    # only standard topk output is supported for piecewise cuda graph
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
    )
    forward_context = get_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]  # 根据layer_id获取对应的MoE层
    return moe_layer.forward_impl(hidden_states, topk_output)


@register_custom_op(out_shape="hidden_states")
def fused_moe_bypassed_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    topk_group: Optional[int],
    num_expert_group: Optional[int],
    correction_bias: Optional[torch.Tensor],
    renormalize: bool,
    layer_id: int,
) -> torch.Tensor:
    """CUDA图分段模式下的MoE前向传播实现（旁路TopK输出格式）。
    
    TopK计算在CUDA图内部完成，不需要外部传入topk_weights和topk_ids。
    作为自定义算子注册，用于在CUDA图中避免图断点。
    """
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=top_k,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            correction_bias=correction_bias,
            renormalize=renormalize,
        ),
    )
    forward_context = get_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]  # 根据layer_id获取对应的MoE层
    return moe_layer.forward_impl(hidden_states, topk_output)
