# Nixl EP令牌分发器模块
# 本模块实现了基于Nixl框架的专家并行(EP)令牌分发与合并逻辑，
# 目前仅支持低延迟(Low Latency)模式，支持FP8量化分发，
# 使用RDMA进行节点间通信，通过TCPStore进行rank间协调。

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 日志模块
from enum import Enum, auto  # 枚举类型和自动值生成
from typing import Optional  # 可选类型

import torch  # PyTorch深度学习框架
import torch.distributed as dist  # PyTorch分布式通信模块

from sglang.srt.distributed.utils import get_global_tcp_store  # 获取全局TCPStore
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager  # 弹性EP状态管理器
from sglang.srt.environ import envs  # 环境变量集合
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 获取全局专家分布记录器
from sglang.srt.layers import deep_gemm_wrapper  # DeepGEMM包装器
from sglang.srt.layers.dp_attention import get_is_extend_in_batch  # 获取当前批次是否为扩展阶段
from sglang.srt.layers.moe.token_dispatcher.base import (  # 分发器基类及相关类型
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (  # DeepEP分发/合并输出类型
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)
from sglang.srt.layers.moe.topk import TopKOutput  # TopK选择输出类型
from sglang.srt.layers.moe.utils import DeepEPMode  # DeepEP模式枚举

try:
    from nixl_ep import Buffer  # 尝试导入Nixl EP Buffer

    use_nixl = True  # Nixl EP可用
except ImportError:
    use_nixl = False  # Nixl EP不可用

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

NixlEPDispatchOutput = DeepEPLLDispatchOutput  # Nixl EP分发输出类型，复用DeepEP低延迟分发输出
NixlEPCombineInput = DeepEPLLCombineInput  # Nixl EP合并输入类型，复用DeepEP低延迟合并输入


class NixlEPBuffer:  # Nixl EP缓冲区单例管理类，用于缓存Nixl EP Buffer
    _buffer = None  # 缓冲区实例
    _hidden_size: Optional[int] = None  # 隐藏层大小
    _num_max_dispatch_tokens_per_rank: Optional[int] = None  # 每个rank最大分发令牌数
    _num_experts: Optional[int] = None  # 专家总数
    _num_local_experts: Optional[int] = None  # 本地专家数

    @classmethod
    def get_nixl_buffer(  # 获取或创建Nixl EP缓冲区（懒加载单例模式）
        cls,
        group: dist.ProcessGroup,  # 进程组
        hidden_size: int,  # 隐藏层大小
        deepep_mode: DeepEPMode,  # DeepEP模式
        num_max_dispatch_tokens_per_rank: int = -1,  # 每个rank最大分发令牌数，默认-1表示未设置
        num_experts: int = -1,  # 专家总数，默认-1表示未设置
        num_local_experts: int = -1,  # 本地专家数，默认-1表示未设置
    ):
        if cls._buffer is not None:  # 如果缓冲区已存在，直接返回
            return cls._buffer

        cls._hidden_size = hidden_size  # 记录隐藏层大小
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank  # 记录每个rank最大分发令牌数
        cls._num_experts = num_experts  # 记录专家总数
        cls._num_local_experts = num_local_experts  # 记录本地专家数

        num_rdma_bytes = 0  # RDMA字节数初始化为0
        if deepep_mode.enable_normal():  # 如果启用了普通模式
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")  # Nixl EP尚不支持普通模式
        if deepep_mode.enable_low_latency():  # 如果启用了低延迟模式
            assert num_max_dispatch_tokens_per_rank != -1  # 断言最大分发令牌数已设置
            assert num_experts != -1 and num_experts % group.size() == 0  # 断言专家数已设置且能被进程数整除
            num_rdma_bytes = Buffer.get_rdma_size_hint(  # 获取RDMA缓冲区大小提示
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                group.size(),
                num_experts,
            )

        rank = dist.get_rank(group)  # 获取当前rank
        world_size = dist.get_world_size(group)  # 获取世界大小

        # Get the global TCPStore for coordination
        # 获取全局TCPStore用于协调 # 获取全局TCPStore用于协调
        tcp_store = get_global_tcp_store()
        if tcp_store is None:  # 如果TCPStore未初始化
            raise RuntimeError(
                "Global TCPStore is not initialized. "
                "Make sure init_distributed_environment was called before using NIXL EP."
            )  # 全局TCPStore未初始化

        logger.info(
            f"Using NIXL EP (world_size={world_size}, rank={rank}, "
            f"num_experts={cls._num_experts}, num_experts_per_rank={cls._num_local_experts}) "
        )  # 记录Nixl EP使用信息

        cls._buffer = Buffer(  # 创建Buffer实例
            rank=rank,
            tcp_store_group=tcp_store,
        )

        cls._buffer.update_memory_buffers(  # 更新内存缓冲区
            num_ranks=world_size,
            num_experts_per_rank=cls._num_local_experts,
            num_rdma_bytes=num_rdma_bytes,
        )
        all_ranks = list(range(world_size))  # 生成所有rank的列表
        cls._buffer.connect_ranks(all_ranks)  # 连接所有rank

        return cls._buffer  # 返回缓冲区实例

    @classmethod
    def clean_buffer(cls):  # 清理缓冲区
        cls._buffer.clean_buffer(
            cls._num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )


class _NixlEPDispatcherImplBase:  # Nixl EP分发器实现基类
    def __init__(  # 初始化Nixl EP分发器实现基类
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由器TopK值
        permute_fusion: bool,  # 是否启用排列融合
        num_experts: int,  # 专家总数
        num_local_experts: int,  # 本地专家数
        hidden_size: int,  # 隐藏层大小
        params_dtype: torch.dtype,  # 参数数据类型
        deepep_mode: DeepEPMode,  # DeepEP模式
    ):
        if not use_nixl:  # 如果Nixl EP不可用
            raise ImportError(
                "NixlEP is not installed. Please install NixlEP package from "
                "https://github.com/ai-dynamo/nixl."
            )  # NixlEP未安装

        self.group = group  # 保存进程组引用
        self.router_topk = router_topk  # 保存路由器TopK值
        self.permute_fusion = permute_fusion  # 保存排列融合标志
        self.num_experts = num_experts  # 保存专家总数
        self.num_local_experts = num_local_experts  # 保存本地专家数
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.params_dtype = params_dtype  # 保存参数数据类型
        self.deepep_mode = deepep_mode  # 保存DeepEP模式

        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )  # 从环境变量获取每个rank最大分发令牌数
        # NixlEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        # NixlEP节点间低延迟分发使用FINISHED_SUM_TAG=1024，逻辑要求从一个rank发送到另一个rank的令牌数小于该值 # NixlEP节点间低延迟分发使用FINISHED_SUM_TAG=1024，要求从一个rank发送到另一个rank的令牌数小于该值
        assert self.num_max_dispatch_tokens_per_rank <= 1024  # 断言最大分发令牌数不超过1024
        elastic_state = ElasticEPStateManager.instance()  # 获取弹性EP状态管理器实例
        self.active_ranks = (
            elastic_state.active_ranks if elastic_state is not None else None
        )  # 获取活跃的rank列表
        self._mask_buffer = (
            torch.zeros_like(self.active_ranks)
            if self.active_ranks is not None
            else None
        )  # 创建掩码缓冲区

        self.handle = None  # 通信句柄，初始为None
        self.quant_config = None  # 量化配置，初始为None
        self.overlap_args = None  # 重叠参数，初始为None
        self.meta_overlap_args = None  # 元数据重叠参数，初始为None

    def set_quant_config(self, quant_config: dict) -> None:  # 设置量化配置
        self.quant_config = quant_config  # 保存量化配置

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args) -> None:  # 设置重叠参数
        self.overlap_args = combine_overlap_args  # 保存合并重叠参数
        self.meta_overlap_args = meta_overlap_args  # 保存元数据重叠参数

    def dispatch_a(  # 分发阶段A（抽象方法，子类实现）
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        raise NotImplementedError  # 子类必须实现

    def dispatch_b(self, *args, **kwargs):  # 分发阶段B（抽象方法，子类实现）
        raise NotImplementedError  # 子类必须实现

    def combine_a(  # 合并阶段A（抽象方法，子类实现）
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
    ):
        raise NotImplementedError  # 子类必须实现

    def combine_b(self, *args, **kwargs):  # 合并阶段B（抽象方法，子类实现）
        raise NotImplementedError  # 子类必须实现

    def _get_buffer(self):  # 获取缓冲区（抽象方法，子类实现）
        raise NotImplementedError  # 子类必须实现


class _NixlEPDispatcherImpl(_NixlEPDispatcherImplBase):  # Nixl EP分发器具体实现类（低延迟模式）
    def __init__(self, return_recv_hook: bool, **kwargs):  # 初始化Nixl EP分发器实现
        super().__init__(**kwargs)  # 调用父类初始化

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/ai-dynamo/nixl
        """
        # num_max_dispatch_tokens_per_rank: 解码引擎中的实际批大小应小于256
        self.return_recv_hook = return_recv_hook  # 是否使用接收钩子而非事件等待
        self.device_module = torch.get_device_module()  # 获取设备模块

    def dispatch_a(  # 分发阶段A：启动异步分发操作
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        buffer = self._get_buffer()  # 获取EP缓冲区
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids  # 解包TopK输出
        topk_ids = topk_ids.to(torch.int64)  # 将TopK ID转换为int64类型
        expected_m = (  # 计算预期的接收令牌数
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(  # 执行分发核心逻辑
            hidden_states,
            topk_ids,
        )
        return (  # 返回分发中间状态
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(  # 分发阶段B：等待分发完成并构造输出
        self,
        hidden_states,  # 隐藏状态
        topk_ids,  # TopK专家ID
        topk_weights,  # TopK专家权重
        masked_m,  # 掩码后的令牌计数
        expected_m,  # 预期令牌数
        event,  # CUDA事件
        hook,  # 接收钩子
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()  # 等待异步操作完成

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(  # 记录低延迟分发的专家分布
            masked_m
        )

        if isinstance(hidden_states, tuple):  # 如果隐藏状态是元组（含缩放因子）
            hidden_states, hidden_states_scale = hidden_states  # 解包隐藏状态和缩放因子
        else:
            hidden_states_scale = None  # 否则缩放因子为None

        nixl_output = NixlEPDispatchOutput(  # 构造Nixl EP分发输出
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return nixl_output  # 返回分发输出

    def _dispatch_core(  # 分发核心逻辑：调用Nixl Buffer执行实际的All-to-All分发
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_idx: torch.Tensor,  # TopK专家索引
    ):
        use_fp8 = not envs.SGLANG_NIXL_EP_BF16_DISPATCH.get()  # 默认使用FP8，除非环境变量指定BF16

        buffer = self._get_buffer()  # 获取EP缓冲区
        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
            buffer.dispatch(
                hidden_states,
                topk_idx,
                self.num_max_dispatch_tokens_per_rank,  # 每个rank最大分发令牌数
                self.num_experts,  # 专家总数
                use_fp8=use_fp8,  # 是否使用FP8量化
                async_finish=not self.return_recv_hook,  # 是否异步完成
                return_recv_hook=self.return_recv_hook,  # 是否使用接收钩子
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,  # 是否使用round_scale（Blackwell平台）
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,  # 是否使用ue8m0缩放格式（Blackwell平台）
            )
        )
        return packed_recv_hidden, self.packed_recv_count, event, hook  # 返回接收的隐藏状态、令牌计数、事件和钩子

    def combine_a(  # 合并阶段A：启动异步合并操作
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
    ):
        hidden_states, event, hook = self._combine_core(  # 执行合并核心逻辑
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook  # 返回合并中间状态

    def combine_b(self, hidden_states, event, hook):  # 合并阶段B：等待合并完成并返回结果
        hook() if self.return_recv_hook else event.current_stream_wait()  # 等待异步操作完成
        return hidden_states  # 返回合并后的隐藏状态

    def _combine_core(  # 合并核心逻辑：调用Nixl Buffer执行实际的All-to-All合并
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
    ):
        buffer = self._get_buffer()  # 获取EP缓冲区

        combined_hidden_states, event, hook = buffer.combine(  # 执行合并操作
            x=hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            handle=self.handle,  # 分发时返回的通信句柄
            async_finish=not self.return_recv_hook,  # 是否异步完成
            return_recv_hook=self.return_recv_hook,  # 是否使用接收钩子
        )
        if self._mask_buffer is not None:  # 如果有掩码缓冲区
            buffer.query_mask_buffer(self._mask_buffer)  # 查询掩码缓冲区
            self.active_ranks.copy_(1 - self._mask_buffer)  # 更新活跃rank列表

        self.packed_recv_count = self.handle = None  # 清空令牌计数和通信句柄
        return combined_hidden_states, event, hook  # 返回合并结果、事件和钩子

    def _get_buffer(self):  # 获取Nixl EP缓冲区实例
        return NixlEPBuffer.get_nixl_buffer(  # 通过NixlEPBuffer单例获取缓冲区
            self.group,
            self.hidden_size,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_local_experts,
        )


class _Stage(Enum):  # 分发/合并阶段枚举，用于状态机管理
    INITIAL = auto()  # 初始状态
    AFTER_DISPATCH_A = auto()  # 分发阶段A完成后
    AFTER_DISPATCH_B = auto()  # 分发阶段B完成后
    AFTER_COMBINE_A = auto()  # 合并阶段A完成后


class NixlEPDispatcher(BaseDispatcher):  # Nixl EP分发器，继承自BaseDispatcher
    def __init__(  # 初始化Nixl EP分发器
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由器TopK值
        permute_fusion: bool = False,  # 是否启用排列融合，默认False
        num_experts: int = None,  # 专家总数
        num_local_experts: int = None,  # 本地专家数
        hidden_size: int = None,  # 隐藏层大小
        params_dtype: torch.dtype = None,  # 参数数据类型
        deepep_mode: DeepEPMode = DeepEPMode.LOW_LATENCY,  # DeepEP模式，默认LOW_LATENCY
        async_finish: bool = False,  # 是否异步完成（未使用）
        return_recv_hook: bool = False,  # 是否使用接收钩子，默认False
    ):
        self.deepep_mode = deepep_mode  # 保存DeepEP模式

        common_kwargs = dict(  # 公共参数字典
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():  # 如果启用低延迟模式
            self._low_latency_dispatcher = _NixlEPDispatcherImpl(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )  # 创建低延迟分发器实现
        if self.deepep_mode.enable_normal():  # 如果启用普通模式
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")  # Nixl EP尚不支持普通模式

        self._stage = _Stage.INITIAL  # 初始化阶段状态为INITIAL

    def dispatch(  # 同步分发方法：依次执行dispatch_a和dispatch_b
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states=hidden_states, topk_output=topk_output)  # 执行分发阶段A
        ret = self.dispatch_b()  # 执行分发阶段B
        return ret  # 返回分发输出

    def dispatch_a(  # 分发阶段A：更新状态并调用内部实现的dispatch_a
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)  # 更新阶段状态
        inner_state = self._get_impl().dispatch_a(  # 调用内部实现的dispatch_a
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state  # 保存分发中间状态

    def dispatch_b(self):  # 分发阶段B：更新状态并调用内部实现的dispatch_b
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)  # 更新阶段状态
        inner_state = self._dispatch_intermediate_state  # 获取分发中间状态
        del self._dispatch_intermediate_state  # 删除中间状态引用
        return self._get_impl().dispatch_b(*inner_state)  # 调用内部实现的dispatch_b并返回结果

    def combine(  # 同步合并方法：依次执行combine_a和combine_b
        self,
        combine_input: CombineInput,  # 合并输入
    ) -> torch.Tensor:
        self.combine_a(combine_input)  # 执行合并阶段A
        ret = self.combine_b()  # 执行合并阶段B
        return ret  # 返回合并结果

    def combine_a(  # 合并阶段A：更新状态并调用内部实现的combine_a
        self,
        combine_input: CombineInput,  # 合并输入
    ):
        hidden_states, topk_ids, topk_weights = combine_input  # 解包合并输入
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)  # 更新阶段状态
        inner_state = self._get_impl().combine_a(  # 调用内部实现的combine_a
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state  # 保存合并中间状态

    def combine_b(self):  # 合并阶段B：更新状态并调用内部实现的combine_b
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)  # 更新阶段状态回初始
        inner_state = self._combine_intermediate_state  # 获取合并中间状态
        del self._combine_intermediate_state  # 删除中间状态引用
        return self._get_impl().combine_b(*inner_state)  # 调用内部实现的combine_b并返回结果

    def _get_impl(self) -> _NixlEPDispatcherImplBase:  # 根据当前状态获取对应的分发器实现
        is_extend_in_batch = get_is_extend_in_batch()  # 获取当前是否为扩展阶段
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)  # 解析实际的DeepEP模式
        if resolved_deepep_mode == DeepEPMode.NORMAL:  # 如果解析为普通模式
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")  # Nixl EP尚不支持普通模式
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:  # 如果解析为低延迟模式
            return self._low_latency_dispatcher  # 返回低延迟分发器实现
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")  # 无效的DeepEP模式

    def set_quant_config(self, quant_config: dict):  # 设置量化配置
        super().set_quant_config(quant_config)  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 如果启用低延迟模式
            self._low_latency_dispatcher.set_quant_config(quant_config)  # 设置低延迟分发器的量化配置

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args):  # 设置重叠参数
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)  # 调用父类方法
        if self.deepep_mode.enable_low_latency():  # 如果启用低延迟模式
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )  # 设置低延迟分发器的重叠参数

    def _update_stage(self, old_stage, new_stage):  # 更新阶段状态，确保状态转换合法
        assert self._stage == old_stage  # 断言当前阶段等于期望的旧阶段
        self._stage = new_stage  # 更新为新阶段
