# 标准令牌分发器模块
# 本模块实现了标准模式下的MoE令牌分发与合并逻辑，
# 支持专家并行(EP)的本地专家映射、FlashInfer Cutlass FP4 All-Gather路径，
# 以及AITer加速的专家掩码处理。

from __future__ import annotations  # 启用延迟类型注解求值

from typing import TYPE_CHECKING, NamedTuple, Optional  # 类型注解

import torch  # PyTorch深度学习框架

from sglang.srt.distributed import (  # 分布式通信相关
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)  # 对称内存上下文管理器
from sglang.srt.layers.dp_attention import (  # 数据并行注意力相关
    get_dp_global_num_tokens,
    get_local_dp_buffer,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # MoE运行器配置
from sglang.srt.layers.moe.token_dispatcher.base import (  # 分发器基类及相关类型
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput, TopKOutputChecker  # TopK输出类型
from sglang.srt.layers.moe.utils import (  # MoE工具函数
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.utils.common import (  # 通用工具函数
    get_bool_env_var,
    get_device,
    is_hip,
)

_is_hip = is_hip()  # 是否为AMD HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITer加速库（仅HIP平台）

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.topk import TopKOutput


try:
    from flashinfer import (
        nvfp4_block_scale_interleave as nvfp4_block_scale_interleave_flashinfer,
    )  # FlashInfer FP4块缩放交错函数

    from sglang.srt.layers.quantization.modelopt_quant import (
        fp4_quantize as fp4_quantize_flashinfer,
    )  # FlashInfer FP4量化函数
except ImportError:
    fp4_quantize_flashinfer = None  # FlashInfer FP4量化不可用
    nvfp4_block_scale_interleave_flashinfer = None  # FlashInfer FP4块缩放交错不可用


class StandardDispatchOutput(NamedTuple):  # 标准分发输出的具名元组
    """Standard dispatch output."""  # 标准分发输出。 # 标准分发输出。

    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: Optional[torch.Tensor]  # 隐藏状态缩放因子（可选）
    topk_output: TopKOutput  # TopK选择结果

    @property
    def format(self) -> DispatchOutputFormat:  # 返回分发输出格式
        return DispatchOutputFormat.STANDARD  # 返回标准格式


assert isinstance(StandardDispatchOutput, DispatchOutput)  # 验证StandardDispatchOutput是DispatchOutput的子类


class StandardCombineInput(NamedTuple):  # 标准合并输入的具名元组
    """Standard combine input."""  # 标准合并输入。 # 标准合并输入。

    hidden_states: torch.Tensor  # 隐藏状态张量

    @property
    def format(self) -> CombineInputFormat:  # 返回合并输入格式
        return CombineInputFormat.STANDARD  # 返回标准格式


assert isinstance(StandardCombineInput, CombineInput)  # 验证StandardCombineInput是CombineInput的子类


class StandardDispatcher(BaseDispatcher):  # 标准分发器，继承自BaseDispatcher

    def __init__(self, moe_runner_config: MoeRunnerConfig):  # 初始化标准分发器
        super().__init__()  # 调用父类初始化
        self.moe_ep_size = get_moe_expert_parallel_world_size()  # 获取专家并行世界大小
        backend = get_moe_runner_backend()  # 获取MoE运行器后端
        self.enable_flashinfer_cutlass_moe = backend.is_flashinfer_cutlass()  # 是否启用FlashInfer Cutlass MoE
        self.enable_flashinfer_mxfp4_moe = backend.is_flashinfer_mxfp4()  # 是否启用FlashInfer MXFP4 MoE
        self.enable_flashinfer_trtllm_routed_moe = backend.is_flashinfer_trtllm_routed()  # 是否启用FlashInfer TRT-LLM路由MoE
        # Skip local expert mapping when the backend handles EP with global expert IDs:
        # - cutlass / cutedsl / trtllm_routed handle EP internally
        # - mxfp4 dispatcher mapping is already global
        # 当后端使用全局专家ID处理EP时跳过本地专家映射： # 当后端使用全局专家ID处理EP时跳过本地专家映射
        # - cutlass/cutedsl/trtllm_routed内部处理EP # - cutlass/cutedsl/trtllm_routed内部处理EP
        # - mxfp4分发器映射已经是全局的 # - mxfp4分发器映射已经是全局的
        self.skip_local_expert_mapping = (
            backend.is_flashinfer_cutlass()
            or backend.is_flashinfer_cutedsl()
            or backend.is_flashinfer_trtllm()
            or backend.is_flashinfer_trtllm_routed()
            or self.enable_flashinfer_mxfp4_moe
        )  # 判断是否跳过本地专家映射
        self.num_experts = moe_runner_config.num_experts  # 保存专家总数
        self.num_local_experts = moe_runner_config.num_local_experts  # 保存本地专家数
        self.num_local_shared_experts = moe_runner_config.num_fused_shared_experts  # 保存本地共享专家数
        self.num_local_routed_experts = (
            self.num_local_experts - self.num_local_shared_experts
        )  # 计算本地路由专家数
        self.moe_ep_rank = get_moe_expert_parallel_rank()  # 获取当前EP rank
        self.local_expert_mapping = None  # 本地专家映射表，延迟初始化
        self.expert_mask_gpu = None  # GPU上的专家掩码

    def dispatch(  # 分发方法：执行令牌到专家的分发
        self, hidden_states: torch.Tensor, topk_output: TopKOutput  # 输入隐藏状态和TopK结果
    ) -> StandardDispatchOutput:

        if should_use_flashinfer_cutlass_moe_fp4_allgather():  # 如果使用FlashInfer Cutlass FP4 All-Gather路径
            # all-gather fp4 hidden states # All-Gather FP4隐藏状态
            if (
                fp4_quantize_flashinfer is None
                or nvfp4_block_scale_interleave_flashinfer is None
            ):  # 如果FP4量化函数不可用
                raise RuntimeError(
                    "FlashInfer fp4_quantize and nvfp4_block_scale_interleave "
                    "are required for the flashinfer_cutlass FP4 all-gather "
                    "path."
                )  # FlashInfer FP4量化函数不可用
            global_scale = self.quant_config.get("input_global_scale", None)  # 获取全局缩放因子
            assert global_scale is not None, "input_global_scale is not set"  # 断言全局缩放因子已设置
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids  # 解包TopK输出

            # Quantize before comm, swizzle after.
            # 通信前量化，之后交错。 # 通信前量化，之后交错
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):  # 使用对称内存上下文
                if hidden_states.shape[0] > 0:  # 如果有令牌
                    x, x_sf = fp4_quantize_flashinfer(
                        hidden_states, global_scale, is_sf_swizzled_layout=False
                    )  # 执行FP4量化
                else:  # 没有令牌
                    x_col = hidden_states.shape[1]  # 获取隐藏维度
                    x = torch.zeros(
                        0, x_col // 2, dtype=torch.uint8, device=hidden_states.device
                    )  # 创建空的FP4张量
                    x_sf = torch.zeros(
                        0, x_col // 16, dtype=torch.uint8, device=hidden_states.device
                    )  # 创建空的缩放因子
            topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(
                [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens()
            )  # 执行All-GatherV操作
            # TODO: fuse into cutlass moe
            # TODO: 融合到cutlass moe中 # TODO: 融合到cutlass moe中
            x_sf = nvfp4_block_scale_interleave_flashinfer(x_sf)  # 执行FP4块缩放交错

            hidden_states = x  # 更新隐藏状态
            hidden_states_scale = x_sf  # 更新隐藏状态缩放因子
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=topk_output.router_logits,  # never tested # 未测试
            )  # 创建标准TopK输出
        else:  # 不使用FP4 All-Gather路径
            hidden_states = hidden_states  # 保持隐藏状态不变
            hidden_states_scale = None  # 缩放因子为None

        if (
            self.moe_ep_size > 1
            and not self.skip_local_expert_mapping
            and TopKOutputChecker.format_is_standard(topk_output)
        ):  # 如果EP大小>1且不跳过本地专家映射且TopK格式为标准格式
            if self.local_expert_mapping is None:  # 如果本地专家映射尚未初始化
                device = get_device()  # 获取设备
                self.local_expert_mapping = torch.full(
                    (self.num_experts,), -1, dtype=torch.int32, device=device
                )  # 创建专家映射表，初始值为-1（表示未映射）
                self.local_expert_mapping[
                    self.moe_ep_rank
                    * self.num_local_routed_experts : (self.moe_ep_rank + 1)
                    * self.num_local_routed_experts
                ] = torch.arange(
                    0, self.num_local_routed_experts, dtype=torch.int32, device=device
                )  # 设置本地路由专家的映射

                if self.num_local_shared_experts > 0:  # 如果有共享专家
                    self.local_expert_mapping[-self.num_local_shared_experts :] = (
                        torch.arange(
                            self.num_local_routed_experts,
                            self.num_local_routed_experts
                            + self.num_local_shared_experts,
                            dtype=torch.int32,
                            device="cpu",
                        )
                    )  # 设置共享专家的映射

        if self.local_expert_mapping is not None and not self.skip_local_expert_mapping:  # 如果有本地专家映射且不跳过
            if _use_aiter:  # 如果使用AITer
                self.expert_mask_gpu = (
                    (
                        (self.local_expert_mapping >= 0)
                        & (self.local_expert_mapping < self.num_local_experts)
                    )
                    .to(torch.int32)
                    .to(device="cuda")
                )  # 创建专家掩码GPU张量
            else:  # 不使用AITer
                if TopKOutputChecker.format_is_standard(topk_output):  # 如果TopK格式为标准格式
                    topk_output = topk_output._replace(
                        topk_ids=self.local_expert_mapping[topk_output.topk_ids]
                    )  # 使用映射表替换TopK ID
                elif TopKOutputChecker.format_is_triton_kernels(topk_output):  # 如果TopK格式为Triton内核格式
                    raise NotImplementedError()  # Triton内核格式不支持本地专家映射

        return StandardDispatchOutput(  # 构造并返回标准分发输出
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_output=topk_output,
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:  # 合并方法：执行专家输出到令牌的合并
        (hidden_states,) = combine_input  # 解包合并输入
        if should_use_flashinfer_cutlass_moe_fp4_allgather():  # 如果使用FlashInfer Cutlass FP4 All-Gather路径
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )  # 获取本地DP缓冲区和全局隐藏状态
            get_tp_group().reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )  # 执行Reduce-ScatterV操作
        return hidden_states  # 返回合并后的隐藏状态
