# Mooncake EP令牌分发器模块
# 本模块实现了基于Mooncake框架的专家并行(EP)令牌分发与合并逻辑，
# 支持低延迟模式下的异步dispatch/combine操作，使用分阶段(stage)状态机
# 管理分发和合并的执行流程。

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 日志模块
from dataclasses import dataclass  # 数据类装饰器
from enum import Enum, auto  # 枚举类型和自动值生成
from typing import NamedTuple, Optional  # 具名元组和可选类型

import torch  # PyTorch深度学习框架
import torch.distributed as dist  # PyTorch分布式通信模块

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager  # 弹性EP状态管理器
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # 获取全局专家分布记录器
from sglang.srt.layers.dp_attention import get_is_extend_in_batch  # 获取当前批次是否为扩展阶段
from sglang.srt.layers.moe.token_dispatcher.base import (  # 分发器基类及相关类型
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput  # TopK选择输出类型
from sglang.srt.layers.moe.utils import DeepEPMode  # DeepEP模式枚举
from sglang.srt.utils import get_int_env_var  # 获取整数环境变量工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class MooncakeDispatchOutput(NamedTuple):  # Mooncake EP分发输出的具名元组
    """Mooncake EP dispatch output."""  # Mooncake EP分发输出。 # Mooncake EP分发输出。

    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: Optional[torch.Tensor]  # 隐藏状态的缩放因子（可选）
    topk_ids: torch.Tensor  # TopK选择的专家ID
    topk_weights: torch.Tensor  # TopK选择的专家权重
    masked_m: torch.Tensor  # 掩码后的令牌计数
    expected_m: int  # 预期的令牌数量

    @property
    def format(self) -> DispatchOutputFormat:  # 返回分发输出格式
        return DispatchOutputFormat.DEEPEP_LL  # 返回DeepEP低延迟格式


assert isinstance(MooncakeDispatchOutput, DispatchOutput)  # 验证MooncakeDispatchOutput是DispatchOutput的子类


class MooncakeCombineInput(NamedTuple):  # Mooncake EP合并输入的具名元组
    """Mooncake EP combine input."""  # Mooncake EP合并输入。 # Mooncake EP合并输入。

    pass  # 空实现，合并输入不携带额外字段

    @property
    def format(self) -> CombineInputFormat:  # 返回合并输入格式
        return CombineInputFormat.DEEPEP_LL  # 返回DeepEP低延迟格式


assert isinstance(MooncakeCombineInput, CombineInput)  # 验证MooncakeCombineInput是CombineInput的子类


class EPBuffer:  # EP缓冲区单例管理类，用于缓存Mooncake EP Buffer
    _buffer = None  # 缓冲区实例
    _hidden_size: Optional[int] = None  # 隐藏层大小
    _num_max_dispatch_tokens_per_rank: Optional[int] = None  # 每个rank最大分发令牌数
    _num_experts: Optional[int] = None  # 专家总数

    @classmethod
    def get_ep_buffer(  # 获取或创建EP缓冲区（懒加载单例模式）
        cls,
        group: dist.ProcessGroup,  # 进程组
        hidden_size: int,  # 隐藏层大小
        param_bytes: int,  # 参数字节数
        deepep_mode: DeepEPMode,  # DeepEP模式
        num_max_dispatch_tokens_per_rank: int = -1,  # 每个rank最大分发令牌数，默认-1表示未设置
        num_experts: int = -1,  # 专家总数，默认-1表示未设置
    ):
        if cls._buffer is not None:  # 如果缓冲区已存在，直接返回
            return cls._buffer

        # Lazy import Buffer to avoid creating CUDA context at module import time
        # 延迟导入Buffer以避免在模块导入时创建CUDA上下文 # 延迟导入Buffer以避免在模块导入时创建CUDA上下文
        from mooncake.mooncake_ep_buffer import Buffer

        cls._hidden_size = hidden_size  # 记录隐藏层大小
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank  # 记录每个rank最大分发令牌数
        cls._num_experts = num_experts  # 记录专家总数

        num_ep_buffer_bytes = 0  # EP缓冲区字节数初始化为0
        if deepep_mode.enable_normal():  # 如果启用了普通模式
            raise NotImplementedError(
                "Normal mode is not supported for Mooncake EP yet."
            )  # Mooncake EP尚不支持普通模式
        if deepep_mode.enable_low_latency():  # 如果启用了低延迟模式
            assert num_max_dispatch_tokens_per_rank != -1  # 断言最大分发令牌数已设置
            assert num_experts != -1 and num_experts % group.size() == 0  # 断言专家数已设置且能被进程数整除
            num_ep_buffer_bytes = Buffer.get_ep_buffer_size_hint(  # 获取EP缓冲区大小提示
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                group.size(),
                num_experts,
            )

        cls._buffer = Buffer(group, num_ep_buffer_bytes)  # 创建Buffer实例并缓存
        return cls._buffer  # 返回缓冲区实例


class _MooncakeEPDispatcherImpl:  # Mooncake EP分发器内部实现类
    def __init__(  # 初始化Mooncake EP分发器实现
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由器TopK值
        permute_fusion: bool,  # 是否启用排列融合
        num_experts: int,  # 专家总数
        num_local_experts: int,  # 本地专家数
        hidden_size: int,  # 隐藏层大小
        params_dtype: torch.dtype,  # 参数数据类型
        return_recv_hook: bool,  # 是否使用接收钩子而非事件等待
        deepep_mode: DeepEPMode,  # DeepEP模式
    ):
        try:
            from mooncake.mooncake_ep_buffer import Buffer  # noqa: F401 # 尝试导入Mooncake EP Buffer
        except ImportError:
            raise ImportError(
                "Mooncake EP is not installed. Please install Mooncake package at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "with EP support to run SGLang with Mooncake EP."
            )  # Mooncake EP未安装，请安装Mooncake包
        self.group = group  # 保存进程组引用
        self.router_topk = router_topk  # 保存路由器TopK值
        self.permute_fusion = permute_fusion  # 保存排列融合标志
        self.num_experts = num_experts  # 保存专家总数
        self.num_local_experts = num_local_experts  # 保存本地专家数
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.params_dtype = params_dtype  # 保存参数数据类型
        self.return_recv_hook = return_recv_hook  # 保存接收钩子标志
        self.deepep_mode = deepep_mode  # 保存DeepEP模式

        self.params_bytes = 2  # 参数字节数设为2
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(  # 从环境变量获取每个rank最大分发令牌数
            "SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128  # 默认值为128
        )
        # Mooncake EP dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        # Mooncake EP分发使用FINISHED_SUM_TAG=1024，逻辑要求从一个rank发送到另一个rank的令牌数小于该值 # Mooncake EP分发使用FINISHED_SUM_TAG=1024，逻辑要求从一个rank发送到另一个rank的令牌数小于该值
        assert self.num_max_dispatch_tokens_per_rank <= 1024  # 断言最大分发令牌数不超过1024

        self.first_execution = True  # 标记是否为首次执行
        self.timeout_us = 10000000  # 超时时间（微秒），默认10秒

        self.handle = None  # 通信句柄，初始为None

    def dispatch_a(  # 分发阶段A：启动异步分发操作
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ):
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights  # 解包TopK输出
        buffer = self._get_buffer()  # 获取EP缓冲区
        topk_ids = topk_ids.to(torch.int64)  # 将TopK ID转换为int64类型
        expected_m = (  # 计算预期的接收令牌数
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(  # 执行分发核心逻辑
            hidden_states,
            topk_ids,
            use_fp8=True,  # 启用FP8量化
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
        hook() if self.return_recv_hook else event.current_stream_wait()  # 如果使用接收钩子则调用钩子，否则等待事件完成

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(  # 记录低延迟分发的专家分布
            masked_m
        )

        if isinstance(hidden_states, tuple):  # 如果隐藏状态是元组（含缩放因子）
            hidden_states, hidden_states_scale = hidden_states  # 解包隐藏状态和缩放因子
        else:
            hidden_states_scale = None  # 否则缩放因子为None

        return MooncakeDispatchOutput(  # 构造并返回Mooncake分发输出
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )

    def _dispatch_core(  # 分发核心逻辑：调用Mooncake Buffer执行实际的All-to-All分发
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        use_fp8: bool = False,  # 是否使用FP8量化，默认为False
    ):
        buffer = self._get_buffer()  # 获取EP缓冲区
        active_ranks = ElasticEPStateManager.instance().active_ranks  # 获取活跃的rank列表
        packed_recv_hidden, packed_recv_count, self.handle, event, hook = (  # 执行分发操作
            buffer.dispatch(
                hidden_states,
                topk_ids,
                active_ranks,
                self.num_max_dispatch_tokens_per_rank,  # 每个rank最大分发令牌数
                self.num_experts,  # 专家总数
                -1 if self.first_execution else self.timeout_us,  # 首次执行无超时，否则使用设定超时
                use_fp8=use_fp8,  # FP8量化标志
                async_finish=not self.return_recv_hook,  # 是否异步完成
                return_recv_hook=self.return_recv_hook,  # 是否使用接收钩子
            )
        )
        return packed_recv_hidden, packed_recv_count, event, hook  # 返回接收的隐藏状态、令牌计数、事件和钩子

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
        return hidden_states, event, hook  # 返回合并后的隐藏状态、事件和钩子

    def combine_b(self, hidden_states, event, hook):  # 合并阶段B：等待合并完成并返回结果
        hook() if self.return_recv_hook else event.current_stream_wait()  # 等待异步操作完成
        return hidden_states  # 返回合并后的隐藏状态

    def _combine_core(  # 合并核心逻辑：调用Mooncake Buffer执行实际的All-to-All合并
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_ids: torch.Tensor,  # TopK专家ID
        topk_weights: torch.Tensor,  # TopK专家权重
    ):
        buffer = self._get_buffer()  # 获取EP缓冲区
        active_ranks = ElasticEPStateManager.instance().active_ranks  # 获取活跃的rank列表
        combined_hidden_states, event, hook = buffer.combine(  # 执行合并操作
            hidden_states,
            topk_ids,
            topk_weights,
            active_ranks,  # 活跃rank列表
            -1 if self.first_execution else self.timeout_us,  # 首次执行无超时，否则使用设定超时
            self.handle,  # 分发时返回的通信句柄
            async_finish=not self.return_recv_hook,  # 是否异步完成
            return_recv_hook=self.return_recv_hook,  # 是否使用接收钩子
        )
        self.first_execution = False  # 标记首次执行完成
        self.handle = None  # 清空通信句柄
        return combined_hidden_states, event, hook  # 返回合并结果、事件和钩子

    def _get_buffer(self):  # 获取EP缓冲区实例
        return EPBuffer.get_ep_buffer(  # 通过EPBuffer单例获取缓冲区
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


@dataclass
class _Stage(Enum):  # 分发/合并阶段枚举，用于状态机管理
    INITIAL = auto()  # 初始状态
    AFTER_DISPATCH_A = auto()  # 分发阶段A完成后
    AFTER_DISPATCH_B = auto()  # 分发阶段B完成后
    AFTER_COMBINE_A = auto()  # 合并阶段A完成后


class MooncakeEPDispatcher(BaseDispatcher):  # Mooncake EP分发器，继承自BaseDispatcher
    def __init__(  # 初始化Mooncake EP分发器
        self,
        group: torch.distributed.ProcessGroup,  # 进程组
        router_topk: int,  # 路由器TopK值
        permute_fusion: bool = False,  # 是否启用排列融合，默认False
        num_experts: int = None,  # 专家总数
        num_local_experts: int = None,  # 本地专家数
        hidden_size: int = None,  # 隐藏层大小
        params_dtype: torch.dtype = None,  # 参数数据类型
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,  # DeepEP模式，默认AUTO
        async_finish: bool = False,  # 是否异步完成（未使用）
        return_recv_hook: bool = False,  # 是否使用接收钩子，默认False
    ):
        super().__init__()  # 调用父类初始化

        self.deepep_mode = deepep_mode  # 保存DeepEP模式

        if self.deepep_mode.enable_low_latency():  # 如果启用了低延迟模式
            self._low_latency_dispatcher = _MooncakeEPDispatcherImpl(  # 创建低延迟分发器实现
                group=group,
                router_topk=router_topk,
                permute_fusion=permute_fusion,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                hidden_size=hidden_size,
                params_dtype=params_dtype,
                return_recv_hook=return_recv_hook,
                deepep_mode=deepep_mode,
            )
        if self.deepep_mode.enable_normal():  # 如果启用了普通模式
            raise NotImplementedError  # 普通模式尚未实现

        self._stage = _Stage.INITIAL  # 初始化阶段状态为INITIAL

    def dispatch(  # 同步分发方法：依次执行dispatch_a和dispatch_b
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        topk_output: TopKOutput,  # TopK选择结果
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)  # 执行分发阶段A
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

    def _get_impl(self) -> _MooncakeEPDispatcherImpl:  # 根据当前状态获取对应的分发器实现
        is_extend_in_batch = get_is_extend_in_batch()  # 获取当前是否为扩展阶段
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)  # 解析实际的DeepEP模式
        if resolved_deepep_mode == DeepEPMode.NORMAL:  # 如果解析为普通模式
            raise NotImplementedError  # 普通模式尚未实现
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:  # 如果解析为低延迟模式
            return self._low_latency_dispatcher  # 返回低延迟分发器实现
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")  # 无效的DeepEP模式

    def _update_stage(self, old_stage, new_stage):  # 更新阶段状态，确保状态转换合法
        assert self._stage == old_stage  # 断言当前阶段等于期望的旧阶段
        self._stage = new_stage  # 更新为新阶段
