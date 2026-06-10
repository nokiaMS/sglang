# TopK专家选择模块
# 本模块实现了MoE（混合专家）模型中的TopK专家选择逻辑，
# 支持多种选择策略（标准TopK、分组TopK、带偏置的分组TopK等），
# 支持多种平台（CUDA、HIP/AMD、CPU、XPU、NPU、MUSA），
# 支持多种量化精度（BF16、FP8、FP4/MXFP4），
# 并提供了TopK输出格式的抽象（Standard、TritonKernel、Bypassed）。

# Copyright 2024 SGLang Team
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

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 日志模块
import math  # 数学模块
from dataclasses import dataclass  # 数据类装饰器
from enum import IntEnum, auto  # 整数枚举和自动值生成
from typing import (  # 类型注解
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数式接口

try:
    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx  # Triton内核路由相关类型
    from triton_kernels.tensor import make_ragged_tensor_metadata  # 创建不规则张量元数据
    from triton_kernels.topk import topk as triton_kernels_topk  # Triton内核TopK函数

    def routing(  # Triton内核路由函数：执行TopK选择并生成路由数据
        logits,  # 路由逻辑值
        n_expts_act,  # 激活的专家数
        sm_first=False,  # 是否先执行softmax
        expt_indx=None,  # 专家索引
        simulated_ep=1,  # 模拟的EP大小
        n_rows=None,  # 行数
    ):
        if simulated_ep != 1:  # 如果模拟EP不为1
            raise NotImplementedError(
                "simulated_ep routing is not supported with triton_kernels 3.6.0"
            )  # Triton内核3.6.0不支持模拟EP路由

        if sm_first:  # 如果先执行softmax
            logits = torch.softmax(logits, dim=-1)  # 对逻辑值执行softmax

        sparse_logits = triton_kernels_topk(  # 执行Triton内核TopK
            logits,
            n_expts_act,
            apply_softmax=not sm_first,  # 是否应用softmax
            y_indx=expt_indx,  # 专家索引
            n_rows=n_rows,  # 行数
        )
        dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx  # 按行排序的分发索引
        combine_indx = sparse_logits.mask_metadata.col_sorted_indx  # 按列排序的合并索引
        ragged_metadata = make_ragged_tensor_metadata(
            sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0]
        )  # 创建不规则张量元数据
        gate_scal = sparse_logits.vals.flatten()[combine_indx]  # 门控缩放值
        routing_data = RoutingData(  # 创建路由数据
            gate_scal,
            ragged_metadata.slice_sizes,
            logits.shape[-1],
            n_expts_act,
            ragged_metadata,
        )
        gather_indx = GatherIndx(combine_indx, dispatch_indx)  # 创建收集索引
        scatter_indx = ScatterIndx(dispatch_indx, combine_indx)  # 创建散射索引
        return routing_data, gather_indx, scatter_indx  # 返回路由数据、收集索引和散射索引

except ImportError:  # Triton内核不可用
    pass  # 跳过

from sglang.jit_kernel.dsv4 import mask_topk_ids  # JIT内核：掩码TopK ID
from sglang.srt.distributed import (  # 分布式通信相关
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)  # 对称内存上下文管理器
from sglang.srt.environ import envs  # 环境变量集合
from sglang.srt.eplb import expert_location_dispatch  # 专家位置分发
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 全局专家分布记录器
from sglang.srt.eplb.expert_location_dispatch import (  # 专家位置分发相关
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric  # 是否对称分配
from sglang.srt.layers.moe import get_moe_runner_backend  # 获取MoE运行器后端
from sglang.srt.layers.moe.utils import is_deepep_class_backend  # 是否为DeepEP类后端
from sglang.srt.layers.utils import MultiPlatformOp  # 多平台操作基类
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer  # 全局专家捕获器
from sglang.srt.utils import (  # 通用工具函数
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists  # 注册伪实现工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.quantization import QuantizationConfig


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
_is_cuda = is_cuda()  # 是否为CUDA平台
_is_hip = is_hip()  # 是否为HIP平台
_is_cpu = is_cpu()  # 是否为CPU平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU是否支持AMX指令集
_is_xpu = is_xpu()  # 是否为XPU平台
_is_npu = is_npu()  # 是否为NPU平台
_is_xpu = is_xpu()  # 是否为XPU平台（重复检查）
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITer（仅HIP平台）
_is_musa = is_musa()  # 是否为MUSA平台

if _is_cuda:  # CUDA平台特定导入
    from sgl_kernel import moe_fused_gate  # MoE融合门控内核

    try:
        from flashinfer.fused_moe import fused_topk_deepseek as _fused_topk_deepseek  # FlashInfer DeepSeek融合TopK

        from sglang.srt.utils.custom_op import register_custom_op  # 注册自定义操作

        @register_custom_op(  # 注册自定义操作：融合TopK DeepSeek
            op_name="fused_topk_deepseek",
            mutates_args=["topk_weights", "topk_ids"],
        )
        def fused_topk_deepseek(  # DeepSeek融合TopK函数
            gating_output: torch.Tensor,  # 门控输出
            correction_bias: torch.Tensor,  # 校正偏置
            num_expert_group: int,  # 专家组数
            topk_group: int,  # TopK组数
            topk: int,  # TopK值
            scaling_factor: float,  # 缩放因子
            topk_weights: torch.Tensor,  # TopK权重输出
            topk_ids: torch.Tensor,  # TopK ID输出
            renormalize: bool,  # 是否重新归一化
        ) -> None:
            _fused_topk_deepseek(
                gating_output,
                correction_bias,
                num_expert_group,
                topk_group,
                topk,
                scaling_factor,
                topk_weights,
                topk_ids,
                renormalize,
            )  # 调用FlashInfer融合TopK实现

    except ImportError:  # FlashInfer不可用
        fused_topk_deepseek = None  # DeepSeek融合TopK不可用

    try:
        from sgl_kernel import kimi_k2_moe_fused_gate  # Kimi K2 MoE融合门控内核
    except ImportError as e:
        pass  # 导入失败则跳过

if _is_cuda or _is_hip or _is_xpu:  # CUDA/HIP/XPU平台特定导入
    from sgl_kernel import topk_softmax  # TopK softmax内核

    try:
        from sgl_kernel import topk_sigmoid  # TopK sigmoid内核
    except ImportError:
        pass  # 导入失败则跳过
if _use_aiter:  # AITer特定导入
    try:
        from aiter import biased_grouped_topk as aiter_biased_grouped_topk  # AITer带偏置分组TopK
        from aiter.fused_moe import fused_topk as aiter_fused_topk  # AITer融合TopK
    except ImportError:
        raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")  # AITer导入失败
if _is_musa:  # MUSA平台特定导入
    try:
        from mate import moe_fused_gate  # MATE MoE融合门控内核
    except ImportError as e:
        raise ImportError("mate is required for the biased grouped topk.")  # MATE导入失败

    from sglang.srt.hardware_backend.musa.kernels.topk import topk_sigmoid, topk_softmax  # MUSA TopK内核

# -------------------------------- TopKConfig ---------------------------------------


@dataclass
class TopKConfig:  # TopK配置数据类
    top_k: int  # 每个令牌选择的TopK专家数
    use_grouped_topk: bool = False  # 是否使用分组TopK，默认False
    topk_group: Optional[int] = None  # 每个组选择的专家数，可选
    num_expert_group: Optional[int] = None  # 专家组数，可选
    renormalize: bool = True  # 是否重新归一化权重，默认True
    num_fused_shared_experts: int = 0  # 融合共享专家数，默认0
    custom_routing_function: Optional[Callable] = None  # 自定义路由函数，可选
    correction_bias: Optional[torch.Tensor] = None  # 校正偏置，可选
    torch_native: bool = False  # 是否使用PyTorch原生实现，默认False
    routed_scaling_factor: Optional[float] = None  # 路由专家缩放因子，可选
    apply_routed_scaling_factor_on_output: bool = False  # 是否在输出上应用路由缩放因子，默认False
    fused_shared_experts_scaling_factor: Optional[float] = None  # 融合共享专家缩放因子，可选
    output_format: Optional[TopKOutputFormat] = None  # 输出格式，可选
    scoring_func: str = "softmax"  # 评分函数，默认softmax


# -------------------------------- TopKOutput ---------------------------------------


class TopKOutputChecker:  # TopK输出格式检查器

    @staticmethod
    def format_is_standard(topk_output: TopKOutput) -> TypeGuard[StandardTopKOutput]:  # 检查是否为标准格式
        return isinstance(topk_output, StandardTopKOutput)

    @staticmethod
    def format_is_triton_kernels(  # 检查是否为Triton内核格式
        topk_output: TopKOutput,
    ) -> TypeGuard[TritonKernelTopKOutput]:
        return isinstance(topk_output, TritonKernelTopKOutput)

    @staticmethod
    def format_is_bypassed(topk_output: TopKOutput) -> TypeGuard[BypassedTopKOutput]:  # 检查是否为旁路格式
        return isinstance(topk_output, BypassedTopKOutput)


class TopKOutputFormat(IntEnum):  # TopK输出格式枚举
    STANDARD = auto()  # 标准格式
    TRITON_KERNEL = auto()  # Triton内核格式
    BYPASSED = auto()  # 旁路格式


@runtime_checkable
class TopKOutput(Protocol):  # TopK输出协议（接口）
    """Protocol for top-k outputs in different formats."""  # 不同格式的TopK输出协议。 # 不同格式的TopK输出协议。

    @property
    def format(self) -> TopKOutputFormat:  # 返回输出格式
        """The format of the output."""  # 输出的格式。 # 输出的格式
        ...


class StandardTopKOutput(NamedTuple):  # 标准TopK输出的具名元组
    """Standard top-k output format."""  # 标准TopK输出格式。 # 标准TopK输出格式。

    topk_weights: torch.Tensor  # TopK权重
    topk_ids: torch.Tensor  # TopK专家ID
    router_logits: torch.Tensor  # 路由器逻辑值

    @property
    def format(self) -> TopKOutputFormat:  # 返回输出格式
        return TopKOutputFormat.STANDARD  # 标准格式


class TritonKernelTopKOutput(NamedTuple):  # Triton内核TopK输出的具名元组
    """Triton kernel top-k output format."""  # Triton内核TopK输出格式。 # Triton内核TopK输出格式。

    routing_data: RoutingData  # 路由数据
    gather_indx: GatherIndx  # 收集索引
    scatter_indx: ScatterIndx  # 散射索引

    @property
    def format(self) -> TopKOutputFormat:  # 返回输出格式
        return TopKOutputFormat.TRITON_KERNEL  # Triton内核格式


class BypassedTopKOutput(NamedTuple):  # 旁路TopK输出的具名元组
    """Bypassed top-k output format."""  # 旁路TopK输出格式。 # 旁路TopK输出格式。

    hidden_states: torch.Tensor  # 隐藏状态
    router_logits: torch.Tensor  # 路由器逻辑值
    topk_config: TopKConfig  # TopK配置
    num_token_non_padded: Optional[torch.Tensor] = None  # 非填充令牌数，可选
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None  # 专家位置分发信息，可选

    @property
    def format(self) -> TopKOutputFormat:  # 返回输出格式
        return TopKOutputFormat.BYPASSED  # 旁路格式

    def to_standard(self, layer_id: Optional[int] = None) -> "StandardTopKOutput":  # 将旁路输出转换为标准输出
        """Materialize routing tensors. Used by MoE kernels that need explicit
        topk_ids / topk_weights rather than doing routing internally."""  # 物化路由张量。用于需要显式topk_ids/topk_weights而非内部路由的MoE内核。 # 物化路由张量，用于需要显式topk_ids/topk_weights的MoE内核
        return select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            topk_config=self.topk_config,
            layer_id=layer_id,
            num_token_non_padded=self.num_token_non_padded,
            expert_location_dispatch_info=self.expert_location_dispatch_info,
        )


def _make_round_robin_expert_ids(  # 生成轮询式专家ID（用于基准测试）
    num_tokens: int,  # 令牌数
    topk: int,  # TopK值
    num_experts: int,  # 专家总数
    *,  # 仅限关键字参数
    device: torch.device,  # 设备
    dtype: torch.dtype,  # 数据类型
    layer_id: Optional[int] = None,  # 层ID，可选
) -> torch.Tensor:
    if topk == 0:  # 如果TopK为0
        return torch.empty((num_tokens, 0), device=device, dtype=dtype)  # 返回空张量

    step = max(num_experts // topk, 1)  # 计算步长
    layer_offset = 0 if layer_id is None else layer_id  # 层偏移量
    offsets = torch.arange(num_tokens, device=device, dtype=dtype).unsqueeze(1)  # 令牌偏移量
    steps = torch.arange(topk, device=device, dtype=dtype).unsqueeze(0) * step  # 步骤偏移量
    return (offsets + layer_offset + steps) % num_experts  # 返回轮询专家ID


# -------------------------------- TopK ---------------------------------------


class TopK(MultiPlatformOp):  # TopK操作类，继承自MultiPlatformOp
    """
    Parameters:
    --top_k: The all number of top experts selected per token, including the fused shared expert(s).
    --num_fused_shared_experts: num of shared experts, can be activate both in TP or EP mode.
    --routed_scaling_factor: the scaling factor for routed experts in topk_weights.
    --fused_shared_experts_scaling_factor: scaling factor for fused shared experts on AMD-platform.
    """
    # 参数说明：
    # --top_k: 每个令牌选择的TopK专家总数（包括融合共享专家） # 每个令牌选择的TopK专家总数
    # --num_fused_shared_experts: 共享专家数，在TP或EP模式下均可激活 # 共享专家数
    # --routed_scaling_factor: 路由专家的缩放因子 # 路由专家缩放因子
    # --fused_shared_experts_scaling_factor: AMD平台上融合共享专家的缩放因子 # AMD平台融合共享专家缩放因子

    def __init__(  # 初始化TopK操作
        self,
        top_k: int,  # TopK值
        *,  # 仅限关键字参数
        layer_id: Optional[int] = None,  # 层ID，可选
        use_grouped_topk: bool = False,  # 是否使用分组TopK，默认False
        topk_group: Optional[int] = None,  # 每个组选择的专家数，可选
        num_expert_group: Optional[int] = None,  # 专家组数，可选
        renormalize: bool = True,  # 是否重新归一化，默认True
        num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
        custom_routing_function: Optional[Callable] = None,  # 自定义路由函数，可选
        scoring_func: str = "softmax",  # 评分函数，默认softmax
        correction_bias: Optional[torch.Tensor] = None,  # 校正偏置，可选
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
        apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，默认False
        output_format: Optional[TopKOutputFormat] = None,  # 输出格式，可选
        fused_shared_experts_scaling_factor: Optional[float] = None,  # 融合共享专家缩放因子，可选
        is_fp4_experts: bool = False,  # 是否为FP4专家，默认False
    ):
        # NOTE: scoring_func is not used for now, but we keep it for future use
        # see https://github.com/sgl-project/sglang/pull/4505 for more details
        # 注意：scoring_func目前未使用，但保留以备将来使用 # 注意：scoring_func目前未使用，保留以备将来使用
        super().__init__()  # 调用父类初始化

        if use_grouped_topk:  # 如果使用分组TopK
            assert num_expert_group is not None and topk_group is not None  # 断言专家组数和TopK组数已设置

        self.layer_id = layer_id  # 保存层ID
        from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数

        self.enable_deepep_waterfill = (  # 是否启用DeepEP水填充
            num_fused_shared_experts > 0
            and get_global_server_args().enable_deepep_waterfill
        )

        self.deepep_waterfill_balancer = None  # DeepEP水填充均衡器，初始为None
        if self.enable_deepep_waterfill:  # 如果启用DeepEP水填充
            # TODO(ch-wan): Refactor shared-expert fusion and routed TopK fusion.
            # TODO(ch-wan): 重构共享专家融合和路由TopK融合。 # TODO: 重构共享专家融合和路由TopK融合
            top_k -= num_fused_shared_experts  # 从TopK中减去共享专家数
            num_fused_shared_experts = 0  # 重置共享专家数为0
            output_format = TopKOutputFormat.STANDARD  # 输出格式设为标准

        # flashinfer_mxfp4 backend only: True -> STANDARD (Mxfp4FlashinferTrtllmMoEMethod
        # consumes), False -> BYPASSED (flashinfer's own mxfp4 kernel). No-op otherwise.
        # 仅flashinfer_mxfp4后端：True -> STANDARD（Mxfp4FlashinferTrtllmMoEMethod消费），False -> BYPASSED（flashinfer自己的mxfp4内核）。其他情况无操作。 # 仅flashinfer_mxfp4后端：True转为STANDARD，False转为BYPASSED
        self.is_fp4_experts = is_fp4_experts  # 保存FP4专家标志
        self.topk_config = TopKConfig(  # 创建TopK配置
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            output_format=output_format,
            scoring_func=scoring_func,
        )

    def _apply_deepep_waterfill(  # 应用DeepEP水填充均衡
        self, topk_output: TopKOutput, num_tokens: int  # TopK输出和令牌数
    ) -> TopKOutput:
        if self.enable_deepep_waterfill and self.deepep_waterfill_balancer is None:  # 如果启用水填充但均衡器未设置
            raise RuntimeError(
                "DeepEP waterfill TopK must be prepared by ModelRunner before forward."
            )  # DeepEP水填充必须在forward之前由ModelRunner准备
        if self.deepep_waterfill_balancer is None:  # 如果均衡器为None
            return topk_output  # 直接返回原输出
        assert TopKOutputChecker.format_is_standard(topk_output)  # 断言输出格式为标准格式
        return self.deepep_waterfill_balancer.expand_topk(topk_output, num_tokens)  # 扩展TopK输出

    def forward_native(  # 原生前向传播（使用PyTorch原生TopK实现）
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        router_logits: torch.Tensor,  # 路由器逻辑值
        *,  # 仅限关键字参数
        num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
    ) -> TopKOutput:
        self.topk_config.torch_native = True  # 设置为PyTorch原生实现
        topk_output = select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )  # 调用select_experts选择专家
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])  # 应用水填充并返回

    def forward_cuda(  # CUDA前向传播
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        router_logits: torch.Tensor,  # 路由器逻辑值
        *,  # 仅限关键字参数
        num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
    ) -> TopKOutput:
        if self.topk_config.output_format is not None:  # 如果指定了输出格式
            output_format = self.topk_config.output_format  # 使用指定格式
        elif get_moe_runner_backend().is_triton_kernels():  # 如果后端为Triton内核
            output_format = TopKOutputFormat.TRITON_KERNEL  # 使用Triton内核格式
        elif get_moe_runner_backend().is_flashinfer_trtllm() or (
            get_moe_runner_backend().is_flashinfer_mxfp4() and not self.is_fp4_experts
        ):  # 如果后端为flashinfer_trtllm或flashinfer_mxfp4（非FP4专家）
            output_format = TopKOutputFormat.BYPASSED  # 使用旁路格式
        else:
            output_format = TopKOutputFormat.STANDARD  # 使用标准格式

        if output_format == TopKOutputFormat.TRITON_KERNEL:  # 如果使用Triton内核格式
            # renormalize=True is equivalent to sm_first=False
            # renormalize=True等价于sm_first=False # renormalize=True等价于sm_first=False
            routing_data, gather_idx, scatter_idx = routing(
                router_logits,
                self.topk_config.top_k,
                sm_first=not self.topk_config.renormalize,
            )  # 执行Triton路由
            return TritonKernelTopKOutput(routing_data, gather_idx, scatter_idx)  # 返回Triton内核输出
        elif output_format == TopKOutputFormat.BYPASSED:  # 如果使用旁路格式
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )  # 返回旁路输出
        else:  # 使用标准格式
            self.topk_config.torch_native = False  # 不使用PyTorch原生实现
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):  # 使用对称内存上下文
                topk_output = select_experts(
                    hidden_states=hidden_states,
                    layer_id=self.layer_id,
                    router_logits=router_logits,
                    topk_config=self.topk_config,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=expert_location_dispatch_info,
                )  # 调用select_experts选择专家
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])  # 应用水填充并返回

    def forward_cpu(  # CPU前向传播
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        router_logits: torch.Tensor,  # 路由器逻辑值
        *,  # 仅限关键字参数
        num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
    ) -> TopKOutput:
        topk_output = select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )  # 调用select_experts选择专家
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])  # 应用水填充并返回

    def forward_npu(  # NPU前向传播
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        router_logits: torch.Tensor,  # 路由器逻辑值
        *,  # 仅限关键字参数
        num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
    ) -> TopKOutput:

        from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu  # 导入NPU融合TopK

        return fused_topk_npu(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            layer_id=self.layer_id,
        )  # 调用NPU融合TopK并返回

    def empty_topk_output(self, device: torch.device) -> TopKOutput:  # 创建空的TopK输出
        topk = self.topk_config.top_k - self.topk_config.num_fused_shared_experts  # 计算路由TopK值
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):  # 使用对称内存上下文
            topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)  # 创建空的权重张量
            topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)  # 创建空的ID张量
        # FIXME: router_logits should be of size (0, num_experts)
        # 修复：router_logits的大小应为(0, num_experts) # 修复：router_logits大小应为(0, num_experts)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)  # 创建空的路由逻辑值张量
        topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)  # 创建标准TopK输出
        if self.topk_config.num_fused_shared_experts > 0 and is_deepep_class_backend():  # 如果有融合共享专家且为DeepEP后端
            n = self.topk_config.num_fused_shared_experts  # 获取共享专家数
            topk_output = topk_output._replace(
                topk_ids=topk_output.topk_ids.new_empty(
                    (0, topk_output.topk_ids.shape[-1] + n)
                ),  # 扩展TopK ID
                topk_weights=topk_output.topk_weights.new_empty(
                    (0, topk_output.topk_weights.shape[-1] + n)
                ),  # 扩展TopK权重
            )
        return self._apply_deepep_waterfill(topk_output, 0)  # 应用水填充并返回

    def forward_xpu(  # XPU前向传播
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        router_logits: torch.Tensor,  # 路由器逻辑值
        *,  # 仅限关键字参数
        num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
    ) -> TopKOutput:
        self.topk_config.torch_native = True  # 默认使用PyTorch原生实现
        # [NOTE] XPU device support for topk kernels
        #   - support 'topk_softmax' and 'topk_sigmoid'
        #   - support up to 8 top-k and 256 experts
        # [注意] XPU设备对TopK内核的支持 # XPU设备对TopK内核的支持
        #   - 支持'topk_softmax'和'topk_sigmoid' # - 支持topk_softmax和topk_sigmoid
        #   - 支持最多8个TopK和256个专家 # - 支持最多8个TopK和256个专家
        self.topk_config.torch_native = not (
            self.topk_config.top_k <= 8 and router_logits.shape[1] <= 256
        )  # 如果TopK<=8且专家数<=256则使用内核实现

        return select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )  # 调用select_experts选择专家并返回


# ------------------------------- TopK implementation -------------------------------------


def fused_topk_torch_native(  # PyTorch原生融合TopK实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    correction_bias: torch.Tensor = None,  # 校正偏置，默认None
    scoring_func: str = "softmax",  # 评分函数，默认softmax
):
    def scoring_func_impl(gating_output: torch.Tensor) -> torch.Tensor:  # 评分函数实现
        if scoring_func == "softmax":  # softmax评分
            return gating_output.softmax(dim=-1)
        elif scoring_func == "sigmoid":  # sigmoid评分
            return gating_output.sigmoid()
        else:
            raise ValueError(f"Invalid scoring function: {scoring_func}")  # 无效的评分函数

    if correction_bias is not None:  # 如果有校正偏置
        n_routed_experts = gating_output.shape[-1]  # 路由专家数
        scores = scoring_func_impl(gating_output)  # 计算分数
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)  # 加上校正偏置
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]  # 选择TopK
        topk_weights = scores.gather(1, topk_ids)  # 收集TopK权重
    else:  # 没有校正偏置
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"  # 断言令牌数匹配
        M, _ = hidden_states.shape  # 获取令牌数
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )  # 预分配权重张量
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)  # 预分配ID张量
        topk_weights = scoring_func_impl(gating_output.float())  # 计算分数
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)  # 选择TopK

    if renormalize:  # 如果需要重新归一化
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # 重新归一化权重
    return topk_weights, topk_ids  # 返回权重和ID


def fused_topk_softmax_torch_raw_logits(  # 基于原始逻辑值的融合TopK softmax实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"  # 断言令牌数匹配

    _, topk_ids = torch.topk(gating_output, k=topk, dim=-1, sorted=False)  # 基于原始逻辑值选择TopK
    logits = gating_output.float()  # 转为float32
    topk_weights = logits.gather(1, topk_ids)  # 收集TopK逻辑值
    if renormalize:  # 如果需要重新归一化
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)  # 对TopK逻辑值执行softmax

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)  # 返回权重和ID


def fused_topk_cpu(  # CPU融合TopK实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    correction_bias: torch.Tensor = None,  # 校正偏置，默认None
    scoring_func: str = "softmax",  # 评分函数，默认softmax
):
    # TODO: add c++ kernel for cpu
    # The topk_softmax_cpu kernel only handles vanilla softmax scoring with no
    # correction bias. Fall back to the torch-native impl for the rest
    # (e.g. MiniMax sets both correction_bias and scoring_func).
    # TODO: 为CPU添加C++内核 # TODO: 为CPU添加C++内核
    # topk_softmax_cpu内核仅处理无校正偏置的普通softmax评分。其余情况回退到PyTorch原生实现（如MiniMax同时设置correction_bias和scoring_func）。 # topk_softmax_cpu内核仅处理无校正偏置的softmax评分，其余回退到PyTorch原生实现
    if correction_bias is not None or scoring_func != "softmax":  # 如果有校正偏置或评分函数非softmax
        return fused_topk_torch_native(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )  # 回退到PyTorch原生实现

    topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )  # 调用CPU TopK softmax内核
    return topk_weights, topk_ids  # 返回权重和ID


def apply_topk_weights_cpu(need_apply, topk_weights, inputs):  # 在CPU上应用TopK权重
    if not need_apply:  # 如果不需要应用
        return inputs, topk_weights  # 直接返回

    # TODO: fuse below processing in fused_experts_cpu kernel
    # TODO: 将以下处理融合到fused_experts_cpu内核中 # TODO: 将以下处理融合到fused_experts_cpu内核中
    inputs = inputs * topk_weights.to(inputs.dtype)  # 将权重应用到输入上
    topk_weights = torch.ones_like(
        topk_weights, dtype=torch.float32
    )  # clear topk_weights as already applied # 清除topk_weights，因为已经应用了 # 清除topk_weights因为已经应用

    return inputs, topk_weights  # 返回加权后的输入和全1权重


def fused_topk(  # 融合TopK实现（自动选择最优内核）
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    correction_bias: Optional[torch.Tensor] = None,  # 校正偏置，可选
    scoring_func: str = "softmax",  # 评分函数，默认softmax
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言令牌数匹配

    M, _ = hidden_states.shape  # 获取令牌数

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )  # 预分配权重张量
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)  # 预分配ID张量

    if scoring_func == "softmax":  # softmax评分函数
        if _use_aiter:  # 如果使用AITer

            # Use fused_topk instead of topk_softmax to auto dispatch to the correct kernel
            # 使用fused_topk而非topk_softmax以自动分发到正确的内核 # 使用fused_topk而非topk_softmax以自动分发到正确的内核
            topk_weights, topk_ids = aiter_fused_topk(
                hidden_states,
                gating_output,
                topk,
                renormalize,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )  # 调用AITer融合TopK
        else:  # 不使用AITer
            topk_softmax(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
            )  # 调用sgl_kernel TopK softmax
    elif scoring_func == "sigmoid":  # sigmoid评分函数
        if _use_aiter and correction_bias is not None:  # 如果使用AITer且有校正偏置
            aiter_biased_grouped_topk(
                gating_output,
                correction_bias.to(dtype=gating_output.dtype),
                topk_weights,
                topk_ids,
                num_expert_group=1,
                topk_group=1,
                need_renorm=renormalize,
            )  # 调用AITer带偏置分组TopK
        else:  # 不使用AITer或无校正偏置
            topk_sigmoid(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                correction_bias,
            )  # 调用sgl_kernel TopK sigmoid
    else:
        raise ValueError(f"Invalid scoring function: {scoring_func}")  # 无效的评分函数

    return topk_weights, topk_ids  # 返回权重和ID


# This is used by the Deepseek V2/V3/R1 series models
# 此函数用于Deepseek V2/V3/R1系列模型 # 此函数用于DeepSeek V2/V3/R1系列模型
@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def grouped_topk_gpu(  # GPU分组TopK实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    num_expert_group: Optional[int] = None,  # 专家组数，可选
    topk_group: Optional[int] = None,  # TopK组数，可选
    num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言令牌数匹配

    scores = torch.softmax(gating_output, dim=-1)  # 计算softmax分数
    num_token = scores.shape[0]  # 令牌数
    num_experts = scores.shape[1]  # 专家数
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group] # 每组的最大分数
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group] # 选择TopK组
    group_mask = torch.zeros_like(group_scores)  # [n, n_group] # 创建组掩码
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group] # 填充组掩码
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e] # 扩展为分数掩码
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e] # 应用分数掩码
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )  # 从掩码分数中选择TopK
    if num_fused_shared_experts:  # 如果有融合共享专家
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )  # 设置共享专家ID
        if routed_scaling_factor is not None:  # 如果有路由缩放因子
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )  # 设置共享专家权重

    if renormalize:  # 如果需要重新归一化
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )  # 计算权重和
        topk_weights = topk_weights / topk_weights_sum  # 重新归一化
        if apply_routed_scaling_factor_on_output:  # 如果在输出上应用路由缩放因子
            topk_weights *= routed_scaling_factor  # 应用缩放因子

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)  # 转换数据类型

    return topk_weights, topk_ids  # 返回权重和ID


def grouped_topk_cpu(  # CPU分组TopK实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    num_expert_group: Optional[int] = None,  # 专家组数，可选
    topk_group: Optional[int] = None,  # TopK组数，可选
    num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):
    assert not apply_routed_scaling_factor_on_output  # CPU不支持在输出上应用路由缩放因子
    return torch.ops.sgl_kernel.grouped_topk_cpu(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        # num_token_non_padded must be None since it is not supported in kernel
        # num_token_non_padded必须为None，因为内核不支持 # num_token_non_padded必须为None因为内核不支持
        num_token_non_padded=None,
    )  # 调用CPU分组TopK内核


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def kimi_k2_biased_topk_impl(  # Kimi K2带偏置TopK实现（num_expert_group=1的优化版本）
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    correction_bias: torch.Tensor,  # 校正偏置
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):
    """
    Optimized version for num_expert_group=1 case (e.g., Kimi K2 with 384 experts).
    Simplifies the grouped topk logic by removing unnecessary group masking operations.
    Note: This function assumes num_fused_shared_experts=0.
    """
    # num_expert_group=1情况的优化版本（如384专家的Kimi K2）。 # num_expert_group=1的优化版本
    # 通过移除不必要的组掩码操作简化分组TopK逻辑。 # 移除不必要的组掩码操作简化逻辑
    # 注意：此函数假设num_fused_shared_experts=0。 # 假设num_fused_shared_experts=0
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言令牌数匹配

    scores = gating_output.sigmoid()  # 使用sigmoid计算分数
    num_token = scores.shape[0]  # 令牌数

    # When num_expert_group=1, no need for group masking
    # Directly compute scores with correction bias
    # 当num_expert_group=1时，不需要组掩码，直接计算带校正偏置的分数 # 当num_expert_group=1时不需要组掩码，直接计算带校正偏置的分数
    tmp_scores = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    # Directly select topk experts (no need to sort since num_fused_shared_experts=0)
    # 直接选择TopK专家（无需排序因为num_fused_shared_experts=0） # 直接选择TopK专家，无需排序
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)  # 收集TopK权重

    if renormalize:  # 如果需要重新归一化
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)  # 计算权重和
        topk_weights = topk_weights / topk_weights_sum  # 重新归一化
        if apply_routed_scaling_factor_on_output:  # 如果在输出上应用路由缩放因子
            topk_weights *= routed_scaling_factor  # 应用缩放因子

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)  # 转换数据类型
    return topk_weights, topk_ids  # 返回权重和ID


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_topk_impl(  # 带偏置TopK实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    correction_bias: torch.Tensor,  # 校正偏置
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    scoring_func: str = "sigmoid",  # 评分函数，默认sigmoid
    num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言令牌数匹配

    if scoring_func == "sigmoid":  # sigmoid评分函数
        scores = gating_output.sigmoid()  # 使用sigmoid计算分数
    elif scoring_func == "sqrtsoftplus":  # sqrtsoftplus评分函数
        scores = torch.nn.functional.softplus(gating_output).sqrt()  # 使用sqrtsoftplus计算分数

    num_token = scores.shape[0]  # 令牌数
    num_experts = scores.shape[1]  # 专家数

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)  # 加上校正偏置
    _, topk_ids = torch.topk(
        scores_for_choice,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )  # 选择TopK
    topk_weights = scores.gather(1, topk_ids)  # 收集TopK权重

    if num_fused_shared_experts:  # 如果有融合共享专家
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )  # 设置共享专家ID
        if routed_scaling_factor is not None:  # 如果有路由缩放因子
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )  # 设置共享专家权重

    if renormalize:  # 如果需要重新归一化
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )  # 计算权重和
        topk_weights = topk_weights / topk_weights_sum  # 重新归一化
        if apply_routed_scaling_factor_on_output:  # 如果在输出上应用路由缩放因子
            topk_weights *= routed_scaling_factor  # 应用缩放因子

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)  # 转换数据类型
    return topk_weights, topk_ids  # 返回权重和ID


def biased_topk_jit_kernel_impl(  # 带偏置TopK的JIT内核实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    correction_bias: torch.Tensor,  # 校正偏置
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    scoring_func: str = "sigmoid",  # 评分函数，默认sigmoid
    num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言令牌数匹配

    if _use_aiter and scoring_func == "sqrtsoftplus" and num_fused_shared_experts == 0:  # AITer + sqrtsoftplus + 无共享专家
        from aiter import topk_gating  # 导入AITer topk_gating

        num_tokens = gating_output.shape[0]  # 令牌数
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )  # 预分配权重张量
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )  # 预分配ID张量

        topk_gating(
            topk_weights,
            topk_ids,
            gating_output,
            correction_bias,
            renormalize,
            routed_scaling_factor,
            score_func="sqrtsoftplus",
        )  # 调用AITer topk_gating

        return topk_weights, topk_ids  # 返回权重和ID

    else:  # 其他情况使用JIT内核
        from sglang.jit_kernel.moe_fused_gate import moe_fused_gate  # 导入JIT MoE融合门控

        topk_weights, topk_ids = moe_fused_gate(
            gating_output,
            correction_bias,
            topk=topk,
            scoring_func=scoring_func,
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )  # 调用JIT MoE融合门控
        topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(
            torch.int32
        )  # 转换数据类型
        return topk_weights, topk_ids  # 返回权重和ID


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_grouped_topk_impl(  # 带偏置分组TopK实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    correction_bias: torch.Tensor,  # 校正偏置
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    num_expert_group: Optional[int] = None,  # 专家组数，可选
    topk_group: Optional[int] = None,  # TopK组数，可选
    num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 断言令牌数匹配

    scores = gating_output.sigmoid()  # 使用sigmoid计算分数
    num_token = scores.shape[0]  # 令牌数
    num_experts = scores.shape[1]  # 专家数
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)  # 加上校正偏置
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group] # 每组前2名的分数和
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group] # 选择TopK组
    group_mask = torch.zeros_like(group_scores)  # [n, n_group] # 创建组掩码
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group] # 填充组掩码
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e] # 扩展为分数掩码
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e] # 应用分数掩码
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )  # 从掩码分数中选择TopK
    topk_weights = scores.gather(1, topk_ids)  # 收集TopK权重

    if num_fused_shared_experts:  # 如果有融合共享专家
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )  # 设置共享专家ID
        if routed_scaling_factor is not None:  # 如果有路由缩放因子
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )  # 设置共享专家权重

    if renormalize:  # 如果需要重新归一化
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )  # 计算权重和
        topk_weights = topk_weights / topk_weights_sum  # 重新归一化
        if apply_routed_scaling_factor_on_output:  # 如果在输出上应用路由缩放因子
            topk_weights *= routed_scaling_factor  # 应用缩放因子

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)  # 转换数据类型

    return topk_weights, topk_ids  # 返回权重和ID


def is_power_of_two(n):  # 判断一个数是否为2的幂
    return n > 0 and math.log2(n).is_integer()  # 大于0且log2为整数


def _mask_topk_ids_padded_region(  # 掩码TopK ID的填充区域
    topk_ids: torch.Tensor,  # TopK ID
    num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
) -> None:
    if num_token_non_padded is None:  # 如果没有非填充令牌数
        return  # 直接返回
    # TODO: let the kernel support other dtypes
    # TODO: 让内核支持其他数据类型 # TODO: 让内核支持其他数据类型
    if _is_cuda and topk_ids.dtype == torch.int32:  # CUDA平台且int32类型
        mask_topk_ids(topk_ids, num_token_non_padded)  # 调用JIT内核掩码
    else:  # 其他情况
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)  # 创建索引
        topk_ids[indices >= num_token_non_padded, :] = -1  # 将填充区域的ID设为-1


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _biased_grouped_topk_postprocess(  # 带偏置分组TopK后处理：逻辑ID到物理ID的转换和填充掩码
    topk_ids, expert_location_dispatch_info, num_token_non_padded
):
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)  # 逻辑ID转物理ID
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)  # 掩码填充区域
    return topk_ids  # 返回处理后的TopK ID


def biased_grouped_topk_gpu(  # GPU带偏置分组TopK实现（根据条件选择最优路径）
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    correction_bias: torch.Tensor,  # 校正偏置
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    num_expert_group: Optional[int] = None,  # 专家组数，可选
    topk_group: Optional[int] = None,  # TopK组数，可选
    num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):

    num_tokens = gating_output.shape[0]  # 令牌数
    num_experts = gating_output.shape[1]  # 专家数
    experts_per_group = (
        num_experts // num_expert_group if num_expert_group else num_experts
    )  # 每组专家数

    # topk for routed experts only (shared experts are appended separately below)
    # 仅路由专家的TopK（共享专家在下面单独追加） # 仅路由专家的TopK（共享专家在下面单独追加）
    topk_routed = topk - num_fused_shared_experts  # 路由TopK值
    if (
        _is_cuda
        and fused_topk_deepseek is not None
        and is_power_of_two(num_experts)
        # flashinfer constraints (applied to routed experts only)
        and topk_routed <= 8
        and topk_group <= num_expert_group
        and topk_group * num_expert_group >= topk_routed
        and (
            (experts_per_group <= 32 and experts_per_group * topk_group <= 128)
            if num_expert_group > 1
            else num_experts <= 384
        )
    ):  # FlashInfer DeepSeek融合TopK路径条件
        # flashinfer约束（仅应用于路由专家） # FlashInfer约束条件
        # Pre-allocate output tensors (flashinfer mutates them in-place)
        # 预分配输出张量（flashinfer就地修改） # 预分配输出张量（flashinfer就地修改）
        topk_weights = torch.empty(
            (num_tokens, topk_routed), dtype=torch.float32, device=gating_output.device
        )  # 预分配权重张量
        topk_ids = torch.empty(
            (num_tokens, topk_routed), dtype=torch.int32, device=gating_output.device
        )  # 预分配ID张量

        # flashinfer always applies the scaling_factor internally
        # flashinfer总是在内部应用缩放因子 # flashinfer总是在内部应用缩放因子
        scaling_factor = 1.0  # 缩放因子默认1.0
        if routed_scaling_factor is not None and apply_routed_scaling_factor_on_output:  # 如果有路由缩放因子且在输出上应用
            scaling_factor = routed_scaling_factor  # 使用路由缩放因子

        # flashinfer's fused_topk_deepseek # FlashInfer的融合TopK DeepSeek
        fused_topk_deepseek(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk_routed,
            scaling_factor,
            topk_weights,
            topk_ids,
            True,
        )  # 调用FlashInfer融合TopK DeepSeek

        if num_fused_shared_experts > 0:  # 如果有融合共享专家
            # Append shared expert columns: ID = num_experts (first shared slot),
            # weight = sum(routed) / scaling_factor (matching biased_grouped_topk_impl).
            # DeepEP fusion will overwrite both in _remap_topk_ids_for_deepep_fusion.
            # 追加共享专家列：ID = num_experts（第一个共享槽位）， # 追加共享专家列
            # 权重 = sum(routed) / scaling_factor（匹配biased_grouped_topk_impl）。 # 权重 = sum(routed) / scaling_factor
            # DeepEP融合将在_remap_topk_ids_for_deepep_fusion中覆盖两者。 # DeepEP融合将覆盖两者
            topk_ids = F.pad(topk_ids, (0, num_fused_shared_experts), value=num_experts)  # 填充共享专家ID
            topk_weights = F.pad(topk_weights, (0, num_fused_shared_experts))  # 填充共享专家权重
            if routed_scaling_factor is not None:  # 如果有路由缩放因子
                topk_weights[:, topk_routed:] = (
                    topk_weights[:, :topk_routed].sum(dim=-1, keepdim=True)
                    / routed_scaling_factor
                )  # 设置共享专家权重

        return topk_weights, topk_ids  # 返回权重和ID

    elif (
        _is_cuda
        # moe_fused_gate kernel ensures that num_experts/num_expert_group does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32, we can remove this assertion.
        and experts_per_group <= 32
        and is_power_of_two(num_experts)
    ):  # sgl_kernel moe_fused_gate路径条件
        # moe_fused_gate内核确保num_experts/num_expert_group不超过MAX_VPT=32。当内核能处理MAX_VPT>32时可移除此断言。 # moe_fused_gate内核确保每组专家不超过32
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            apply_routed_scaling_factor_on_output,
        )  # 调用sgl_kernel MoE融合门控

        return topk_weights, topk_ids  # 返回权重和ID

    elif _use_aiter:  # AITer路径
        assert not apply_routed_scaling_factor_on_output, "Not implemented"  # AITer不支持在输出上应用路由缩放因子
        token = gating_output.shape[0]  # 令牌数
        device = gating_output.device  # 设备
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch: hidden_states.shape[0] = {hidden_states.shape[0]}, gating_output.shape[0] = {gating_output.shape[0]}"  # 断言令牌数匹配
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)  # 预分配权重张量
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)  # 预分配ID张量
        aiter_biased_grouped_topk(
            gating_output,
            correction_bias.to(dtype=gating_output.dtype),
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
        )  # 调用AITer带偏置分组TopK
        return topk_weights, topk_ids  # 返回权重和ID
    elif _is_musa and (
        gating_output.shape[1] // num_expert_group <= 32
        or (num_expert_group == 1 and gating_output.shape[1] in {160, 256, 384})
    ):  # MUSA路径条件
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            True,
            apply_routed_scaling_factor_on_output,
        )  # 调用MATE MoE融合门控
    else:  # 其他路径
        # Use optimized path for Kimi K2 (384 experts with num_expert_group=1)
        # 使用Kimi K2的优化路径（384专家，num_expert_group=1） # 使用Kimi K2的优化路径（384专家，num_expert_group=1）
        num_experts = gating_output.shape[1]  # 专家数
        if _is_cuda and num_experts == 384 and num_expert_group == 1:  # CUDA + 384专家 + 单组
            return kimi_k2_moe_fused_gate(
                gating_output.to(dtype=torch.float32),
                correction_bias,
                topk=topk,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )  # 调用Kimi K2融合门控
        elif (
            _is_cuda
            and num_expert_group == 1
            and topk_group == 1
            and num_fused_shared_experts == 0
            and num_experts <= 512
            and topk <= 8
        ):  # CUDA + 单组 + TopK组为1 + 无共享专家 + 专家<=512 + TopK<=8
            from sglang.jit_kernel.grouped_topk import grouped_topk as jit_grouped_topk  # 导入JIT分组TopK

            scaling = (
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            )  # 缩放因子
            if not apply_routed_scaling_factor_on_output:  # 如果不在输出上应用
                scaling = 1.0  # 缩放因子设为1.0
            return jit_grouped_topk(
                gating_output.to(dtype=torch.float32),
                correction_bias.to(dtype=torch.float32),
                num_expert_group,
                topk_group,
                topk,
                renormalize,
                scaling,
            )  # 调用JIT分组TopK
        else:  # 最终回退路径
            return biased_grouped_topk_impl(
                hidden_states,
                gating_output,
                correction_bias,
                topk,
                renormalize,
                num_expert_group,
                topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )  # 调用带偏置分组TopK实现


def biased_grouped_topk_cpu(  # CPU带偏置分组TopK实现
    hidden_states: torch.Tensor,  # 隐藏状态
    gating_output: torch.Tensor,  # 门控输出
    correction_bias: torch.Tensor,  # 校正偏置
    topk: int,  # TopK值
    renormalize: bool,  # 是否重新归一化
    num_expert_group: Optional[int] = None,  # 专家组数，可选
    topk_group: Optional[int] = None,  # TopK组数，可选
    compiled: bool = True,  # 是否编译，默认True
    num_fused_shared_experts: int = 0,  # 融合共享专家数，默认0
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子，可选
    apply_routed_scaling_factor_on_output: Optional[bool] = False,  # 是否在输出上应用路由缩放因子，可选
):
    return torch.ops.sgl_kernel.biased_grouped_topk_cpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor if apply_routed_scaling_factor_on_output else None,
        # num_token_non_padded must be None since it is not supported in kernel
        # num_token_non_padded必须为None，因为内核不支持 # num_token_non_padded必须为None因为内核不支持
        num_token_non_padded=None,
    )  # 调用CPU带偏置分组TopK内核


if _is_cpu and _is_cpu_amx_available:  # CPU + AMX可用
    biased_grouped_topk = biased_grouped_topk_cpu  # 使用CPU实现
    grouped_topk = grouped_topk_cpu  # 使用CPU实现
    fused_topk_native = fused_topk_cpu  # 使用CPU实现
    fused_topk = fused_topk_cpu  # 使用CPU实现
else:  # 其他平台
    biased_grouped_topk = biased_grouped_topk_gpu  # 使用GPU实现
    grouped_topk = grouped_topk_gpu  # 使用GPU实现
    fused_topk_native = fused_topk_torch_native  # 使用PyTorch原生实现


def _remap_topk_for_deepep(  # 将TopK输出重新映射为DeepEP交错专家布局
    topk_ids: torch.Tensor,  # TopK专家ID
    topk_weights: torch.Tensor,  # TopK权重
    num_fused_shared_experts: int,  # 融合共享专家数
    num_physical_routed_experts: int,  # 物理路由专家数
    topk_config: TopKConfig,  # TopK配置
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap TopK output to DeepEP interleaved expert layout.

    DeepEP dispatch needs each rank's shared expert at a unique ID so tokens
    route to the correct rank. The layout interleaves shared slots among
    routed experts: [routed_0..L-1, shared, routed_L..2L-1, shared, ...].

    Routed IDs:  e -> e + e // num_local_routed
    Shared IDs:  ep_rank * num_local_experts + num_local_routed
    Shared weight: 1 / routed_scaling_factor (compensates post-MoE scaling)
    """
    # 将TopK输出重新映射为DeepEP交错专家布局。 # 将TopK输出重新映射为DeepEP交错专家布局
    #
    # DeepEP分发需要每个rank的共享专家有唯一ID，以便令牌路由到正确的rank。 # DeepEP分发需要每个rank的共享专家有唯一ID
    # 布局在路由专家间交错共享槽位：[routed_0..L-1, shared, routed_L..2L-1, shared, ...]。 # 布局在路由专家间交错共享槽位
    #
    # 路由ID: e -> e + e // num_local_routed # 路由ID映射公式
    # 共享ID: ep_rank * num_local_experts + num_local_routed # 共享ID映射公式
    # 共享权重: 1 / routed_scaling_factor（补偿MoE后缩放） # 共享权重公式
    if topk_ids.shape[0] == 0:  # 如果没有令牌
        return topk_ids, topk_weights  # 直接返回

    ep_size = get_moe_expert_parallel_world_size()  # 获取EP世界大小
    ep_rank = get_moe_expert_parallel_rank()  # 获取EP rank
    # Static EPLB may add redundant physical experts. At this point routed
    # topk_ids have already been remapped from logical to physical ids, so the
    # DeepEP interleaved layout must use the physical routed count.
    # 静态EPLB可能添加冗余物理专家。此时路由topk_ids已从逻辑ID重新映射为物理ID，因此DeepEP交错布局必须使用物理路由计数。 # 静态EPLB可能添加冗余物理专家，DeepEP交错布局必须使用物理路由计数
    num_local_routed = num_physical_routed_experts // ep_size  # 每个rank的本地路由专家数
    num_local_experts = num_local_routed + num_fused_shared_experts  # 每个rank的本地专家总数

    # Remap routed IDs: insert gaps for shared expert slots (single fused op)
    # 重新映射路由ID：为共享专家槽位插入间隔（单次融合操作） # 重新映射路由ID：为共享专家槽位插入间隔
    routed = topk_ids[:, :-num_fused_shared_experts]  # 获取路由专家ID
    topk_ids[:, :-num_fused_shared_experts] = routed + routed // num_local_routed  # 重新映射路由ID

    # Set shared expert IDs to route to home rank (vectorized)
    # 设置共享专家ID以路由到本rank（向量化） # 设置共享专家ID以路由到本rank（向量化）
    topk_ids[:, -num_fused_shared_experts:] = (
        ep_rank * num_local_experts
        + num_local_routed
        + torch.arange(num_fused_shared_experts, device=topk_ids.device)
    )

    # Override shared weight: 1/routed_scaling_factor so net contribution = 1.0
    # 覆盖共享权重：1/routed_scaling_factor，使净贡献=1.0 # 覆盖共享权重使净贡献为1.0
    routed_scaling_factor = topk_config.routed_scaling_factor
    if routed_scaling_factor is not None and routed_scaling_factor != 0:  # 如果有路由缩放因子
        topk_weights[:, -num_fused_shared_experts:] = 1.0 / routed_scaling_factor  # 设置共享专家权重

    return topk_ids, topk_weights  # 返回重新映射后的ID和权重


def _post_process_topk_ids(  # TopK ID后处理：逻辑到物理映射、专家捕获、共享专家追加
    topk_ids: torch.Tensor,  # TopK专家ID
    topk_weights: torch.Tensor,  # TopK权重
    topk_config: TopKConfig,  # TopK配置
    router_logits: torch.Tensor,  # 路由器逻辑值
    layer_id: int,  # 层ID
    num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
) -> torch.Tensor:
    num_fused_shared_experts = topk_config.num_fused_shared_experts  # 获取融合共享专家数
    fused_shared_experts_scaling_factor = (
        topk_config.fused_shared_experts_scaling_factor
    )  # 获取融合共享专家缩放因子
    if (cap := get_global_experts_capturer()) is not None:  # 如果有专家捕获器
        cap.capture(
            layer_id=layer_id,
            topk_indices=topk_ids,
        )  # 捕获TopK索引
    if _is_cuda:  # CUDA平台
        # When shared experts are fused (appended as extra columns in topk_ids),
        # EPLB dispatch must only remap the routed expert columns.
        # The shared expert column (value = n_routed_experts) would be out-of-bounds
        # for the logical-to-physical dispatch table.
        # 当共享专家被融合（作为额外列追加到topk_ids中）时， # 当共享专家被融合追加到topk_ids时
        # EPLB分发必须仅重新映射路由专家列。 # EPLB分发必须仅重新映射路由专家列
        # 共享专家列（值 = n_routed_experts）会超出逻辑到物理分发表的范围。 # 共享专家列会超出逻辑到物理分发表的范围
        if num_fused_shared_experts > 0 and is_deepep_class_backend():  # 如果有共享专家且为DeepEP后端
            shared_cols = topk_ids[:, -num_fused_shared_experts:]  # 获取共享专家列
            routed_cols = topk_ids[:, :-num_fused_shared_experts]  # 获取路由专家列
            routed_cols = _biased_grouped_topk_postprocess(
                routed_cols, expert_location_dispatch_info, num_token_non_padded
            )  # 对路由专家列执行后处理
            topk_ids = torch.cat([routed_cols, shared_cols], dim=-1)  # 合并路由和共享专家列
        else:  # 无共享专家或非DeepEP后端
            topk_ids = _biased_grouped_topk_postprocess(
                topk_ids, expert_location_dispatch_info, num_token_non_padded
            )  # 对所有TopK ID执行后处理

    if num_fused_shared_experts > 0 and _use_aiter:  # 如果有共享专家且使用AITer
        M, N = router_logits.shape  # 获取形状
        scale_factor = (
            1.0
            if fused_shared_experts_scaling_factor is None
            else fused_shared_experts_scaling_factor
        )  # 确定缩放因子

        # Lazy import to avoid circular-import issues
        # 延迟导入以避免循环导入问题 # 延迟导入以避免循环导入问题
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            fused_append_shared_experts,
        )  # 导入融合追加共享专家函数

        topk_ids, topk_weights = fused_append_shared_experts(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            scale_factor,
            N,  # base id for shared experts # 共享专家的基础ID
        )  # 追加共享专家

    # DeepEP: remap to interleaved expert layout where each rank's shared
    # expert has a unique ID for dispatch routing.
    # DeepEP：重新映射为交错专家布局，其中每个rank的共享专家有唯一ID用于分发路由。 # DeepEP：重新映射为交错专家布局
    if num_fused_shared_experts > 0 and is_deepep_class_backend():  # 如果有共享专家且为DeepEP后端
        num_physical_routed_experts = (
            expert_location_dispatch_info.num_physical_experts
            if expert_location_dispatch_info is not None
            else router_logits.shape[1]
        )  # 获取物理路由专家数
        topk_ids, topk_weights = _remap_topk_for_deepep(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            num_physical_routed_experts,
            topk_config,
        )  # 重新映射为DeepEP布局

    return topk_ids, topk_weights  # 返回处理后的ID和权重


def select_experts(  # 选择专家：根据路由逻辑值和TopK配置选择专家
    hidden_states: torch.Tensor,  # 隐藏状态
    router_logits: torch.Tensor,  # 路由器逻辑值
    topk_config: TopKConfig,  # TopK配置
    *,  # 仅限关键字参数
    layer_id: Optional[int] = None,  # 层ID，可选
    num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充令牌数，可选
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置分发信息，可选
) -> StandardTopKOutput:
    top_k = topk_config.top_k  # 获取TopK值
    use_grouped_topk = topk_config.use_grouped_topk  # 是否使用分组TopK
    topk_group = topk_config.topk_group  # TopK组数
    num_expert_group = topk_config.num_expert_group  # 专家组数
    renormalize = topk_config.renormalize  # 是否重新归一化
    num_fused_shared_experts = topk_config.num_fused_shared_experts  # 融合共享专家数
    custom_routing_function = topk_config.custom_routing_function  # 自定义路由函数
    correction_bias = topk_config.correction_bias  # 校正偏置
    torch_native = topk_config.torch_native  # 是否使用PyTorch原生实现
    routed_scaling_factor = topk_config.routed_scaling_factor  # 路由缩放因子
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )  # 是否在输出上应用路由缩放因子

    scoring_func = topk_config.scoring_func  # 评分函数

    (
        router_logits,
        correction_bias,
    ) = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )  # 转换select_experts的输入

    # DeepSeek V2/V3/R1 series models use grouped_top_k
    # remove num_fused_shared_experts from grouped_topk/biased_grouped_topk
    # DeepSeek V2/V3/R1系列模型使用分组TopK # DeepSeek V2/V3/R1系列模型使用分组TopK
    # 从grouped_topk/biased_grouped_topk中移除num_fused_shared_experts # 从grouped_topk/biased_grouped_topk中移除num_fused_shared_experts
    num_routed_topk = top_k - num_fused_shared_experts  # 计算路由TopK值
    if use_grouped_topk:  # 如果使用分组TopK
        assert topk_group is not None  # 断言TopK组数已设置
        assert num_expert_group is not None  # 断言专家组数已设置
        if correction_bias is None:  # 无校正偏置
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )  # 调用分组TopK
        else:  # 有校正偏置
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )  # 调用带偏置分组TopK
    elif torch_native and custom_routing_function is None:  # PyTorch原生 + 无自定义路由
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"  # 原生TopK不支持num_token_non_padded
        assert expert_location_dispatch_info is None  # 原生TopK不支持专家位置分发
        assert not apply_routed_scaling_factor_on_output, "Not implemented"  # 原生TopK不支持输出上应用缩放因子
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )  # 调用原生TopK
    elif custom_routing_function is None:  # 无自定义路由函数
        if scoring_func != "sqrtsoftplus":  # 非sqrtsoftplus评分函数
            assert not apply_routed_scaling_factor_on_output, "Not implemented"  # 非sqrtsoftplus不支持输出上应用缩放因子

        if scoring_func == "sqrtsoftplus":  # sqrtsoftplus评分函数
            _biased_topk = (
                biased_topk_jit_kernel_impl
                if envs.SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK.get()
                else biased_topk_impl
            )  # 根据环境变量选择JIT内核或编译实现

            topk_weights, topk_ids = _biased_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )  # 调用选定的带偏置TopK实现
        elif (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            and scoring_func == "softmax"
            and correction_bias is None
        ):  # flashinfer_trtllm_routed + softmax + 无偏置
            # flashinfer_trtllm_routed uses raw-logits topk
            # flashinfer_trtllm_routed使用原始逻辑值TopK # flashinfer_trtllm_routed使用原始逻辑值TopK
            topk_weights, topk_ids = fused_topk_softmax_torch_raw_logits(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
            )  # 调用原始逻辑值TopK softmax
        else:  # 其他情况
            # Qwen3MOE uses fused_topk
            # Qwen3MOE使用融合TopK # Qwen3MOE使用融合TopK
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                correction_bias=correction_bias,
                scoring_func=scoring_func,
            )  # 调用融合TopK
    else:  # 有自定义路由函数
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"  # 自定义路由不支持num_token_non_padded
        assert expert_location_dispatch_info is None  # 自定义路由不支持专家位置分发
        assert not apply_routed_scaling_factor_on_output, "Not implemented"  # 自定义路由不支持输出上应用缩放因子
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
        )  # 调用自定义路由函数

    simulate_uniform_experts = envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()  # 是否模拟均匀专家
    simulate_round_robin_experts = envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()  # 是否模拟轮询专家
    if simulate_uniform_experts and simulate_round_robin_experts:  # 两者不能同时启用
        raise ValueError(
            "SGLANG_SIMULATE_UNIFORM_EXPERTS and "
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS are mutually exclusive"
        )  # 模拟均匀专家和模拟轮询专家互斥

    if simulate_uniform_experts:  # 如果模拟均匀专家
        # Benchmark-only: override gating with random-offset uniform expert assignment
        # to avoid expert imbalance from dummy/random weights. Do NOT use in production.
        # 仅用于基准测试：用随机偏移的均匀专家分配覆盖门控， # 仅用于基准测试
        # 以避免虚拟/随机权重导致的专家不平衡。不要在生产环境中使用。 # 用随机偏移均匀分配覆盖门控，不要在生产环境使用
        num_tokens, k = topk_ids.shape  # 获取形状
        num_experts = router_logits.shape[1]  # 专家数
        if k > 0:  # 如果有TopK
            offsets = torch.randint(
                0, num_experts, (num_tokens, 1), device=topk_ids.device
            )  # 生成随机偏移
            steps = torch.arange(k, device=topk_ids.device).unsqueeze(0)  # 步骤
            step = max(num_experts // k, 1)  # 步长
            topk_ids = ((offsets + steps * step) % num_experts).to(topk_ids.dtype)  # 计算均匀分配的专家ID
            topk_weights = torch.ones_like(topk_weights) / k  # 设置均匀权重
    elif simulate_round_robin_experts:  # 如果模拟轮询专家
        # Benchmark-only: override gating with deterministic expert assignment
        # to avoid routing noise from dummy/random weights. Do NOT use in production.
        # 仅用于基准测试：用确定性专家分配覆盖门控， # 仅用于基准测试
        # 以避免虚拟/随机权重导致的路由噪声。不要在生产环境中使用。 # 用确定性分配覆盖门控，不要在生产环境使用
        num_tokens, k = topk_ids.shape  # 获取形状
        num_experts = router_logits.shape[1]  # 专家数
        topk_ids = _make_round_robin_expert_ids(
            num_tokens,
            k,
            num_experts,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
            layer_id=layer_id,
        )  # 生成轮询专家ID
        if k > 0:  # 如果有TopK
            topk_weights = torch.full_like(topk_weights, 1.0 / k)  # 设置均匀权重

    topk_ids, topk_weights = _post_process_topk_ids(  # 执行TopK ID后处理
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=topk_config,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)  # 记录专家选择分布

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)  # 返回标准TopK输出


# Register fake implementations for torch.compile support
# 注册伪实现以支持torch.compile # 注册伪实现以支持torch.compile
if _is_cuda:  # CUDA平台

    @torch.library.register_fake("sgl_kernel::moe_fused_gate")  # 注册moe_fused_gate的伪实现
    def _moe_fused_gate(  # moe_fused_gate的伪实现（用于torch.compile）
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=0,
        apply_routed_scaling_factor_on_output=False,
    ):
        num_rows = input_tensor.shape[0]  # 行数
        topk_weights = torch.empty(
            (num_rows, topk), dtype=torch.float32, device=input_tensor.device
        )  # 创建空权重张量
        topk_ids = torch.empty(
            (num_rows, topk), dtype=torch.int32, device=input_tensor.device
        )  # 创建空ID张量
        return topk_weights, topk_ids  # 返回空张量

    @register_fake_if_exists("sgl_kernel::kimi_k2_moe_fused_gate")  # 注册kimi_k2_moe_fused_gate的伪实现
    def _kimi_k2_moe_fused_gate(  # kimi_k2_moe_fused_gate的伪实现
        input_tensor,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    ):
        num_rows = input_tensor.shape[0]  # 行数
        topk_weights = input_tensor.new_empty(
            num_rows,
            topk,
            dtype=torch.float32,
        )  # 创建空权重张量
        topk_ids = input_tensor.new_empty(
            num_rows,
            topk,
            dtype=torch.int32,
        )  # 创建空ID张量
        return topk_weights, topk_ids  # 返回空张量
