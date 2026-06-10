# FlashInfer token分发器模块
# 实现基于FlashInfer的MoE专家并行token分发和合并，支持NVFP4量化
from __future__ import annotations  # 启用延迟注解求值

import logging  # 日志模块
from typing import NamedTuple, Optional  # 类型注解工具

import torch  # PyTorch深度学习框架

from sglang.kernel_api_logging import debug_kernel_api  # 内核API调试日志装饰器
from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens  # 获取DP全局token数
from sglang.srt.layers.moe.token_dispatcher import (  # 基础分发器相关类型
    BaseDispatcher,  # 分发器基类
    CombineInput,  # 合并输入协议
    CombineInputFormat,  # 合并输入格式枚举
    DispatchOutput,  # 分发输出协议
    DispatchOutputFormat,  # 分发输出格式枚举
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (  # FlashInfer通信工具
    TorchDistributedCommBackend,  # 基于torch.distributed的通信后端
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput  # TopK输出类型
from sglang.srt.layers.moe.utils import get_moe_runner_backend  # 获取MoE运行器后端
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 推测算法
from sglang.srt.utils import get_int_env_var  # 获取整数环境变量

try:  # 尝试导入FlashInfer库
    from flashinfer import nvfp4_block_scale_interleave  # NVFP4块缩放交织函数
    from flashinfer.comm import MoeAlltoAll, moe_a2a_get_workspace_size_per_rank  # MoE全对全通信
    from flashinfer.comm.mapping import Mapping  # 通信映射
    from flashinfer.comm.mnnvl import MnnvlConfig  # MNNVL配置

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize  # FP4量化工具

    use_flashinfer = True  # 标记FlashInfer可用
except ImportError:  # FlashInfer不可用
    use_flashinfer = False  # 标记FlashInfer不可用

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()  # 获取MoE NVFP4分发配置


class FlashinferDispatchOutput(NamedTuple):  # FlashInfer分发输出
    """Flashinfer EP dispatch output."""  # FlashInfer EP分发输出
    # FlashInfer EP分发输出

    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: Optional[torch.Tensor]  # 隐藏状态缩放因子（FP4/NVFP4时使用）
    topk_output: StandardTopKOutput  # 标准TopK输出
    # Provide an output tensor to fused_moe so it writes directly to our buffer
    # 提供输出张量给fused_moe，使其直接写入我们的缓冲区
    # 提供输出张量给fused_moe，使其直接写入我们的缓冲区
    moe_output: Optional[torch.Tensor] = None  # MoE输出张量（可选）

    @property
    def format(self) -> DispatchOutputFormat:  # 输出格式属性
        return DispatchOutputFormat.FLASHINFER  # 返回FlashInfer格式


assert isinstance(FlashinferDispatchOutput, DispatchOutput)  # 验证输出符合DispatchOutput协议


class FlashinferCombineInput(NamedTuple):  # FlashInfer合并输入
    """Flashinfer combine input."""  # FlashInfer合并输入
    # FlashInfer合并输入

    hidden_states: torch.Tensor  # 隐藏状态张量

    @property
    def format(self) -> CombineInputFormat:  # 输入格式属性
        return CombineInputFormat.FLASHINFER  # 返回FlashInfer格式


assert isinstance(FlashinferCombineInput, CombineInput)  # 验证输入符合CombineInput协议


class FlashinferDispatcher(BaseDispatcher):  # FlashInfer分发器，实现基于FlashInfer A2A后端的token分发与合并
    """Main dispatcher class for Flashinfer A2A backend."""  # FlashInfer A2A后端的主分发器类
    # FlashInfer A2A后端的主分发器类

    def __init__(  # 初始化FlashInfer分发器
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由TopK值
        num_experts: int = None,  # 专家总数
        num_local_experts: int = None,  # Unused # 本地专家数（未使用）
        # 本地专家数（未使用）
        hidden_size: int = None,  # 隐藏层大小
        params_dtype: torch.dtype = None,  # Unused # 参数数据类型（未使用）
        # 参数数据类型（未使用）
    ):
        super().__init__()  # 调用父类初始化
        if not use_flashinfer:  # 若FlashInfer不可用
            raise ImportError(  # 抛出导入错误
                "Flashinfer is not installed or does not support A2A. "
                "Please install the appropriate version of Flashinfer."
            )

        self.ep_size = group.size()  # 专家并行大小
        self.ep_rank = group.rank()  # 当前rank在专家并行组中的排名
        self.router_topk = router_topk  # 路由TopK值
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_experts = num_experts  # 专家总数
        self.num_local_experts = num_local_experts  # 本地专家数

        # TODO: Can other moe runners use payload_in_workspace too?
        # TODO: 其他MoE运行器也能使用payload_in_workspace吗？
        # TODO: 其他MoE运行器也能使用payload_in_workspace吗？
        self.payload_in_workspace = get_moe_runner_backend().is_flashinfer_cutlass()  # 是否在工作空间中传递payload

        # TODO: Can this be a server arg and shared with deepep/mooncakeep?
        # FlashInfer sizes the workspace from the maximum dispatched tokens per
        # EP rank. See FlashInfer's moe_a2a_get_workspace_size_per_rank(),
        # which reserves ep_size * max_num_tokens * payload bytes, and the C++
        # dispatch op's epSize * runtimeMaxTokensPerRank payload buffer.
        # TODO: 这能成为服务器参数并与deepep/mooncakeep共享吗？
        # FlashInfer根据每个EP rank的最大分发token数来确定工作空间大小。
        # 参见FlashInfer的moe_a2a_get_workspace_size_per_rank()，
        # 它预留ep_size * max_num_tokens * payload字节，以及C++
        # dispatch操作的epSize * runtimeMaxTokensPerRank payload缓冲区。
        # TODO: 这能成为服务器参数并与deepep/mooncakeep共享吗？
        # FlashInfer根据每个EP rank的最大分发token数来确定工作空间大小。
        # 参见FlashInfer的moe_a2a_get_workspace_size_per_rank()，
        # 它预留ep_size * max_num_tokens * payload字节，以及C++
        # dispatch操作的epSize * runtimeMaxTokensPerRank payload缓冲区。
        self.max_num_tokens = get_int_env_var(  # 每个rank最大分发token数
            "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096  # 默认4096
        )

        # Calculate workspace size. For eagle mode, use the larger workspace size since nextn layer will be unquantized.
        # 计算工作空间大小。对于eagle模式，使用更大的工作空间大小，因为nextn层将不被量化。
        # 计算工作空间大小。对于eagle模式，使用更大的工作空间大小，因为nextn层将不被量化。
        speculative_algo = SpeculativeAlgorithm.from_string(  # 获取推测算法
            get_global_server_args().speculative_algorithm
        )
        if MOE_NVFP4_DISPATCH and not speculative_algo.is_eagle():  # NVFP4分发且非eagle模式
            total_dispatch_payload_size_per_token = (  # 每个token的分发payload总大小
                hidden_size // 2  # nvfp4 hidden states # nvfp4隐藏状态
                + hidden_size // 16  # fp8 scaling factors # fp8缩放因子
                + self.router_topk * 4  # int32 topks ids # int32的topk ID
                + self.router_topk * 4  # float32 topk weights # float32的topk权重
            )
        else:  # 非NVFP4或eagle模式
            total_dispatch_payload_size_per_token = (  # 每个token的分发payload总大小
                hidden_size * 2  # bf16 hidden states # bf16隐藏状态
                + self.router_topk * 4  # int32 topks ids # int32的topk ID
                + self.router_topk * 4  # float32 topk weights # float32的topk权重
            )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states # 每个token的合并payload大小（bf16隐藏状态）
        # 每个token的合并payload大小（bf16隐藏状态）
        self.workspace_size = moe_a2a_get_workspace_size_per_rank(  # 计算每个rank的工作空间大小
            ep_size=self.ep_size,  # 专家并行大小
            max_num_tokens=self.max_num_tokens,  # 最大token数
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,  # 分发payload大小
            combine_payload_size_per_token=combine_payload_size_per_token,  # 合并payload大小
        )

        self.mapping = Mapping(  # 创建FlashInfer通信映射
            rank=self.ep_rank,  # 当前rank
            tp_size=self.ep_size,  # 张量并行大小
            moe_ep_size=self.ep_size,  # MoE专家并行大小
            world_size=self.ep_size,  # 全局进程数
            gpus_per_node=torch.cuda.device_count(),  # 每个节点的GPU数
            pp_size=1,  # 流水线并行大小
            cp_size=1,  # 上下文并行大小
        )
        self.moe_a2a = MoeAlltoAll(  # 创建MoE全对全通信对象
            mapping=self.mapping,  # 通信映射
            max_num_tokens=self.max_num_tokens,  # 最大token数
            top_k=self.router_topk,  # TopK值
            num_experts=self.num_experts,  # 专家数量
            workspace_size_per_rank=self.workspace_size,  # 每个rank的工作空间大小
            mnnvl_config=MnnvlConfig(comm_backend=TorchDistributedCommBackend(group)),  # MNNVL配置，使用torch.distributed后端
        )

        self.dummy_topk_ids = torch.full(  # 虚拟TopK ID（全为num_experts，表示无效专家）
            (1, self.router_topk), self.num_experts, dtype=torch.int32, device="cuda"
        )
        self.dummy_topk_ids_current_rank = torch.full(  # 虚拟TopK ID（指向当前rank的本地专家）
            (1, self.router_topk),
            self.ep_rank * self.num_local_experts,  # 当前rank的专家起始ID
            dtype=torch.int32,
            device="cuda",
        )
        self.dummy_topk_weights = torch.zeros(  # 虚拟TopK权重（全零）
            (1, self.router_topk), dtype=torch.float32, device="cuda"
        )

    @debug_kernel_api  # 内核API调试装饰器
    def dispatch(  # 分发方法：将token分发到各专家
        self, hidden_states: torch.Tensor, topk_output: TopKOutput  # 隐藏状态和TopK输出
    ) -> FlashinferDispatchOutput:
        output_dtype = hidden_states.dtype  # 保存原始数据类型
        x = hidden_states  # 工作副本
        x_sf = None  # 缩放因子初始化为None
        topk_ids = topk_output.topk_ids  # TopK专家ID
        topk_weights = topk_output.topk_weights  # TopK权重

        self.has_dummy_token = x.shape[0] == 0  # 检查是否为空token（需要虚拟token）
        if self.has_dummy_token:  # 若需要虚拟token
            x = hidden_states.new_zeros((1, self.hidden_size))  # 创建一个零向量
            topk_ids = self.dummy_topk_ids  # 使用虚拟TopK ID
            topk_weights = self.dummy_topk_weights  # 使用虚拟TopK权重

        global_scale = self.quant_config.get("input_global_scale", None)  # 获取全局缩放因子
        if global_scale is not None:  # 若存在全局缩放因子
            x, x_sf = fp4_quantize(x, global_scale, is_sf_swizzled_layout=False)  # FP4量化

        payloads = []  # payload列表
        payloads.append(x)  # 添加隐藏状态
        if x_sf is not None:  # 若存在缩放因子
            payloads.append(x_sf)  # 添加缩放因子
            expert_id_payload_index = 2  # 专家ID在payload中的索引为2
        else:  # 无缩放因子
            expert_id_payload_index = 1  # 专家ID在payload中的索引为1
        payloads.append(topk_ids)  # 添加TopK ID
        payloads.append(topk_weights)  # 添加TopK权重

        dp_global = get_dp_global_num_tokens()  # 获取DP全局token数
        if dp_global is not None and len(dp_global) > 1:  # DP注意力：多个DP rank有不同token数
            # DP attention: multiple DP ranks with different token counts.
            # Use the max across ranks so the A2A workspace fits the fattest.
            # DP注意力：多个DP rank有不同的token数。
            # 使用各rank中的最大值，以确保A2A工作空间能容纳最胖的rank。
            # DP注意力：多个DP rank有不同的token数。
            # 使用各rank中的最大值，以确保A2A工作空间能容纳最胖的rank。
            self.runtime_max_tokens_per_rank = max(dp_global)  # 取最大token数
        else:  # dp_size=1或SP模式
            # dp_size=1 or SP: use the actual input tensor size (post-scatter
            # in SP mode, full batch otherwise).  Avoids the pre-scatter
            # scheduler count which can exceed the workspace cap.
            # dp_size=1或SP：使用实际输入张量大小（SP模式下为scatter后的大小，
            # 否则为完整批次）。避免使用预scatter调度器计数，该计数可能超出工作空间上限。
            # dp_size=1或SP：使用实际输入张量大小（SP模式下为scatter后的大小，
            # 否则为完整批次）。避免使用预scatter调度器计数，该计数可能超出工作空间上限。
            self.runtime_max_tokens_per_rank = x.shape[0]  # 使用实际输入大小
        if self.has_dummy_token:  # 若使用虚拟token
            self.runtime_max_tokens_per_rank = max(self.runtime_max_tokens_per_rank, 1)  # 至少为1

        # Passing topk_ids + invalid_token_expert_id triggers the sanitize step
        # inside moe_a2a. The recv buffer has shape
        # [ep_size, max_tokens_per_rank, ...], so any rank below max leaves
        # padding slots whose expert_id would otherwise route to a real expert
        # and waste downstream MoE compute. Sanitizing the padding to a
        # sentinel id is structural, not optional.
        # 传入topk_ids + invalid_token_expert_id会触发moe_a2a内部的清理步骤。
        # 接收缓冲区的形状为[ep_size, max_tokens_per_rank, ...]，
        # 因此任何token数低于最大值的rank会留下填充槽位，
        # 其expert_id本会路由到真实专家并浪费下游MoE计算。
        # 将填充清理为哨兵ID是结构性的，不是可选的。
        # 传入topk_ids + invalid_token_expert_id会触发moe_a2a内部的清理步骤。
        # 接收缓冲区的形状为[ep_size, max_tokens_per_rank, ...]，
        # 因此任何token数低于最大值的rank会留下填充槽位，
        # 其expert_id本会路由到真实专家并浪费下游MoE计算。
        # 将填充清理为哨兵ID是结构性的，不是可选的。
        recv_tensors = self.moe_a2a.dispatch(  # 执行FlashInfer全对全分发
            self.dummy_topk_ids_current_rank if self.has_dummy_token else topk_ids,  # 使用虚拟或真实TopK ID
            payloads,  # payload列表
            self.runtime_max_tokens_per_rank,  # 运行时最大token数
            invalid_token_expert_id=self.num_experts,  # 无效token的专家ID（哨兵值）
            expert_id_payload_index=expert_id_payload_index,  # 专家ID在payload中的索引
        )
        if x_sf is not None:  # 若存在缩放因子（FP4量化）
            x_recv, x_sf_recv, topk_ids_recv, topk_weights_recv = recv_tensors  # 解包接收到的张量
            x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])  # 重塑缩放因子形状
            # TODO: fuse interleave into cutlass moe # TODO: 将交织操作融合到cutlass moe中
            # TODO: 将交织操作融合到cutlass moe中
            if get_moe_runner_backend().is_flashinfer_cutlass():  # 使用flashinfer cutlass后端
                x_sf = nvfp4_block_scale_interleave(x_sf)  # NVFP4块缩放交织
        else:  # 无缩放因子
            x_recv, topk_ids_recv, topk_weights_recv = recv_tensors  # 解包接收到的张量
        x = x_recv.view(-1, x_recv.shape[-1])  # 重塑隐藏状态形状
        topk_ids = topk_ids_recv.view(-1, topk_ids_recv.shape[-1])  # 重塑TopK ID形状
        topk_weights = topk_weights_recv.view(-1, topk_weights_recv.shape[-1])  # 重塑TopK权重形状

        # Provide an output tensor to fused_moe so it writes directly to our buffer
        # 提供输出张量给fused_moe，使其直接写入我们的缓冲区
        # 提供输出张量给fused_moe，使其直接写入我们的缓冲区
        moe_output = None  # MoE输出初始化为None
        if self.payload_in_workspace:  # 若在工作空间中传递payload
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(  # 获取工作空间中的合并payload张量
                self.runtime_max_tokens_per_rank, self.hidden_size, output_dtype  # 运行时最大token数、隐藏层大小、数据类型
            ).view(-1, self.hidden_size)  # 重塑为一维
        return FlashinferDispatchOutput(  # 返回FlashInfer分发输出
            x,
            x_sf,
            StandardTopKOutput(topk_weights, topk_ids, topk_output.router_logits),  # 标准TopK输出
            moe_output,  # MoE输出张量
        )

    @debug_kernel_api  # 内核API调试装饰器
    def combine(self, combine_input: FlashinferCombineInput) -> torch.Tensor:  # 合并方法：将各专家结果合并回原始token
        hidden_states = combine_input.hidden_states  # 获取隐藏状态
        output_hidden_size = hidden_states.shape[-1]  # 获取输出隐藏层大小
        hidden_states = self.moe_a2a.combine(  # 执行FlashInfer全对全合并
            hidden_states.view(  # 重塑隐藏状态为3D形状
                self.ep_size, self.runtime_max_tokens_per_rank, output_hidden_size  # [ep_size, max_tokens, hidden_size]
            ),
            self.runtime_max_tokens_per_rank,  # 运行时最大token数
            payload_in_workspace=self.payload_in_workspace,  # 是否在工作空间中传递payload
        )

        if self.has_dummy_token:  # 若使用了虚拟token
            hidden_states = hidden_states[1:, :]  # 去掉虚拟token行

        del self.runtime_max_tokens_per_rank  # 删除运行时变量
        del self.has_dummy_token  # 删除虚拟token标志
        return hidden_states  # 返回合并后的隐藏状态
