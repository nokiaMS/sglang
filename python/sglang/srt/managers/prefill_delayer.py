# 预填充延迟器模块
# 通过延迟预填充请求来优化解码吞吐量，避免小批量预填充打断解码

import dataclasses  # 数据类工具
import logging  # 日志库
import time  # 时间库
from dataclasses import dataclass, field  # 数据类装饰器和字段
from typing import TYPE_CHECKING, NamedTuple, Optional  # 类型提示

import torch  # PyTorch库

from sglang.srt.environ import envs  # 环境变量
from sglang.srt.utils import get_bool_env_var  # 获取布尔环境变量

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector  # 调度器指标收集器

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")  # 是否启用预填充延迟器调试日志

logger = logging.getLogger(__name__)  # 获取日志记录器


@dataclass(frozen=True)  # 不可变数据类，表示延迟状态
class _State:
    delayed_count: int = 0  # 已延迟的次数
    start_time: float = field(default_factory=time.perf_counter)  # 延迟开始时间

    def bump_delayed_count(self) -> "_State":  # 增加延迟计数，返回新的不可变状态对象
        return dataclasses.replace(self, delayed_count=self.delayed_count + 1)  # 创建新状态，计数加1


class _NegotiateOutput(NamedTuple):  # 协商输出的命名元组
    next_state: Optional[_State]  # 下一个状态，None表示不需要延迟
    input_estimation: str  # 输入估算（"all"/"none"/"mixed"）
    output_allow: bool  # 是否允许预填充
    output_reason: str  # 允许/拒绝的原因
    num_prefillable: int  # 可预填充的DP rank数量
    num_token_watermark_force_allow: int  # 因低Token水位线强制允许的DP rank数量


class PrefillDelayer:  # 预填充延迟器，通过协商机制决定是否延迟预填充请求
    def __init__(  # 初始化预填充延迟器
        self,
        dp_size: int,  # 数据并行大小
        attn_tp_size: int,  # 注意力张量并行大小
        cpu_group,  # CPU通信组
        server_args,  # 服务器参数
        max_delay_passes: int,  # 最大延迟轮数
        token_usage_low_watermark: Optional[float],  # Token使用率低水位线
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,  # 指标收集器
        device: Optional["torch.device"] = "cpu",  # 计算设备
        device_group=None,  # 设备通信组
    ):
        self._max_delay_passes = max_delay_passes  # 最大延迟轮数
        self._token_usage_low_watermark = token_usage_low_watermark  # Token使用率低水位线
        # Queue-based trigger is opt-in: activates only when queue_min_ratio
        # is explicitly set. Additive with the slot-based trigger.
        # 基于队列的触发器是可选的：仅在显式设置queue_min_ratio时激活。与基于槽位的触发器叠加。
        self._queue_min_ratio = server_args.prefill_delayer_queue_min_ratio  # 队列最小比率
        # Fall back to 5000ms if unset; this is a local safety cap, not a
        # semantic default, so we don't surface it via ServerArgs.
        # 如果未设置则回退到5000ms；这是本地安全上限，不是语义默认值，因此不通过ServerArgs暴露。
        self._max_delay_ms = server_args.prefill_delayer_max_delay_ms  # 最大延迟毫秒数
        if self._max_delay_ms is None:  # 如果未设置
            self._max_delay_ms = 5000.0  # 默认5000毫秒
        self._queue_trigger_enabled = self._queue_min_ratio is not None  # 是否启用队列触发器
        logger.info(  # 记录初始化信息
            f"PrefillDelayer initialized with "
            f"max_delay_passes={self._max_delay_passes} "
            f"token_usage_low_watermark={self._token_usage_low_watermark} "
            f"queue_min_ratio={self._queue_min_ratio} "
            f"max_delay_ms={self._max_delay_ms} "
            f"queue_trigger_enabled={self._queue_trigger_enabled}"
        )
        self.dp_size = dp_size  # 数据并行大小
        self.enable_dp_attention = server_args.enable_dp_attention  # 是否启用数据并行注意力
        dp_size_dim = dp_size if self.enable_dp_attention else 1  # 数据并行维度

        # Mirror scheduler_dp_attn_mixin's NCCL all-gather path: when the
        # env flag is on (or overlap scheduling is disabled), ride the NCCL
        # device group on `device` instead of gloo on CPU.
        # 镜像scheduler_dp_attn_mixin的NCCL全收集路径：当环境标志开启（或重叠调度被禁用）时，
        # 使用`device`上的NCCL设备组而非CPU上的gloo。
        use_nccl = (  # 是否使用NCCL
            server_args.disable_overlap_schedule  # 禁用重叠调度
            or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()  # 或环境变量指定
        )
        if use_nccl:  # 如果使用NCCL
            assert (  # 断言设备组不为空
                device_group is not None
            ), "device_group is required when using NCCL for PrefillDelayer all-gather"
            self._gather_group = device_group  # 使用设备通信组
            self._gather_device = device  # 使用指定设备
        else:  # 如果不使用NCCL
            self._gather_group = cpu_group  # 使用CPU通信组
            self._gather_device = "cpu"  # 使用CPU设备

        # Fields packed per rank into the all-gather tensor: prefillable,
        # token_watermark_force_allow, running_batch, max_prefill_bs,
        # waiting_queue_len.
        # 每个rank打包到全收集张量中的字段：可预填充、Token水位线强制允许、运行批次、最大预填充批次大小、等待队列长度。
        self._global_info_buffer = torch.empty(  # 全局信息缓冲区
            (dp_size_dim, attn_tp_size, 5),  # 形状：[DP, TP, 5]
            dtype=torch.int64,  # 64位整数
            device=self._gather_device,  # 设备
        )

        self._metrics_collector = metrics_collector  # 指标收集器

        self._curr_state: Optional[_State] = None  # 当前状态，初始为None
        self.skip_first_delayer = True  # 是否跳过第一次延迟，初始为True

        assert (  # 断言重叠调度未被禁用
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."  # 使用PrefillDelayer时，disable_overlap_schedule必须为False

    def _negotiate_should_allow_prefill(  # 协商是否允许预填充（带状态的包装方法）
        self,
        local_prefillable: bool,  # 本地是否可预填充
        token_usage: float,  # Token使用率
        running_batch: int = 0,  # 运行中的批次数
        max_prefill_bs: int = 0,  # 最大预填充批次大小
        max_running_requests: int = 0,  # 最大运行请求数
        waiting_queue_len: int = 0,  # 等待队列长度
    ) -> _NegotiateOutput:  # 返回协商输出
        out = self._negotiate_should_allow_prefill_pure(  # 调用纯函数进行协商
            prev_state=self._curr_state,  # 上一个状态
            local_prefillable=local_prefillable,  # 本地可预填充标志
            token_usage=token_usage,  # Token使用率
            running_batch=running_batch,  # 运行批次数
            max_prefill_bs=max_prefill_bs,  # 最大预填充批次大小
            max_running_requests=max_running_requests,  # 最大运行请求数
            waiting_queue_len=waiting_queue_len,  # 等待队列长度
        )
        self._curr_state = out.next_state  # 更新当前状态
        return out  # 返回协商输出

    # (Almost) pure function, do not modify self state
    # （几乎）纯函数，不修改self状态
    def _negotiate_should_allow_prefill_pure(  # 纯函数：协商是否允许预填充
        self,
        prev_state: Optional[_State],  # 上一个状态
        local_prefillable: bool,  # 本地是否可预填充
        token_usage: float,  # Token使用率
        running_batch: int = 0,  # 运行中的批次数
        max_prefill_bs: int = 0,  # 最大预填充批次大小
        max_running_requests: int = 0,  # 最大运行请求数
        waiting_queue_len: int = 0,  # 等待队列长度
    ) -> _NegotiateOutput:  # 返回协商输出
        # Compute local states  计算本地状态
        local_token_watermark_force_allow = (  # 本地Token水位线强制允许
            local_prefillable  # 本地可预填充
            and ((x := self._token_usage_low_watermark) is not None)  # 且低水位线已设置
            and (token_usage < x)  # 且Token使用率低于低水位线
        )

        # Gather global states  收集全局状态
        tp0_info = self._gather_info(  # 收集信息
            local_prefillable=local_prefillable,  # 本地可预填充
            local_token_watermark_force_allow=local_token_watermark_force_allow,  # 本地Token水位线强制允许
            running_batch=running_batch,  # 运行批次数
            max_prefill_bs=max_prefill_bs,  # 最大预填充批次大小
            waiting_queue_len=waiting_queue_len,  # 等待队列长度
        )
        global_prefillable = tp0_info[:, 0]  # 全局可预填充状态
        global_token_watermark_force_allow = tp0_info[:, 1]  # 全局Token水位线强制允许状态
        global_running_batch = tp0_info[:, 2]  # 全局运行批次数
        global_max_prefill_bs = tp0_info[:, 3]  # 全局最大预填充批次大小
        global_waiting_queue_len = tp0_info[:, 4]  # 全局等待队列长度

        # Compute derived global states  计算派生的全局状态
        if global_prefillable.min().item() > 0:  # 如果所有DP rank都可预填充
            prefillable_status = "all"  # 全部可预填充
        elif global_prefillable.max().item() == 0:  # 如果所有DP rank都不可预填充
            prefillable_status = "none"  # 全部不可预填充
        else:  # 部分可预填充
            prefillable_status = "mixed"  # 混合状态
        global_exists_token_watermark_force_allow = (  # 是否存在因低水位线强制允许的DP rank
            global_token_watermark_force_allow.max().item() > 0
        )
        debug_info = dict(  # 调试信息字典
            input_estimation=prefillable_status,  # 输入估算
            num_prefillable=global_prefillable.sum().item(),  # 可预填充的DP rank数量
            num_token_watermark_force_allow=global_token_watermark_force_allow.sum().item(),  # 因低水位线强制允许的数量
        )

        # Compute outputs  计算输出
        if prefillable_status == "all":  # 如果全部可预填充
            # Safety valve: low KV usage means GPU is underutilized, skip
            # delay. Mirrors the check in the "mixed" branch.
            # 安全阀：低KV使用率意味着GPU未充分利用，跳过延迟。镜像"mixed"分支中的检查。
            if global_exists_token_watermark_force_allow:  # 如果存在低水位线强制允许
                return _NegotiateOutput(  # 返回允许预填充
                    next_state=None,  # 无需延迟状态
                    output_allow=True,  # 允许预填充
                    output_reason="token_watermark",  # 原因：Token水位线
                    **debug_info,  # 调试信息
                )

            if not self.enable_dp_attention:  # 如果未启用数据并行注意力
                max_running_requests = (  # 计算每个DP rank的最大运行请求数
                    max_running_requests + self.dp_size - 1
                ) // self.dp_size  # 向上取整除法

            global_running_batch_max = int(global_running_batch.max().item())  # 全局运行批次最大值
            global_max_prefill_bs_max = int(global_max_prefill_bs.max().item())  # 全局最大预填充批次大小最大值
            global_waiting_queue_max = int(global_waiting_queue_len.max().item())  # 全局等待队列长度最大值

            # Queue-based trigger: delay prefill until the waiting queue
            # reaches queue_min = min(running_req * ratio, max_prefill_bs),
            # capped by a wall-clock timeout to bound worst-case TTFT.
            # Targets workloads where decode requests finish one-at-a-time
            # and fragment prefill into many tiny batches.
            # 基于队列的触发器：延迟预填充直到等待队列达到
            # queue_min = min(running_req * ratio, max_prefill_bs)，
            # 通过墙上时钟超时限制最坏情况TTFT。
            # 针对解码请求逐个完成并将预填充碎片化为许多微小批次的工作负载。
            queue_condition = False  # 队列条件初始化为False
            if self._queue_trigger_enabled and global_running_batch_max > 0:  # 如果启用队列触发器且有运行批次
                queue_min_effective = min(  # 计算有效的队列最小值
                    int(global_running_batch_max * self._queue_min_ratio),  # 运行批次 × 比率
                    global_max_prefill_bs_max,  # 不超过最大预填充批次大小
                )
                queue_condition = (  # 队列条件：等待队列不足
                    queue_min_effective > 0  # 有效最小值大于0
                    and global_waiting_queue_max < queue_min_effective  # 且等待队列不足
                )
                if queue_condition and prev_state is not None:  # 如果满足队列条件且有前状态
                    elapsed_ms = (time.perf_counter() - prev_state.start_time) * 1000.0  # 计算已延迟毫秒数
                    if elapsed_ms >= self._max_delay_ms:  # 如果超过最大延迟时间
                        queue_condition = False  # 解除队列条件

            slot_condition = (  # 槽位条件：剩余槽位不足以容纳一个完整预填充批次
                max_running_requests - global_running_batch_max  # 剩余槽位数
                < global_max_prefill_bs_max  # 小于最大预填充批次大小
            )

            if slot_condition or queue_condition:  # 如果满足槽位条件或队列条件
                # When the "max_decode_bs - running_bs < max_prefill_bs" condition is met,
                # the first merge_batch causes the decoding to fail to reach the maximum batch size.
                # 当满足"max_decode_bs - running_bs < max_prefill_bs"条件时，
                # 第一次merge_batch会导致解码无法达到最大批次大小。
                if self.skip_first_delayer:  # 如果跳过第一次延迟
                    self.skip_first_delayer = False  # 标记已跳过
                    pass  # 直接通过
                else:  # 不跳过
                    next_state = prev_state or _State()  # 获取或创建新状态
                    next_state = next_state.bump_delayed_count()  # 增加延迟计数
                    return _NegotiateOutput(  # 返回延迟预填充
                        next_state=next_state,  # 延迟状态
                        output_allow=False,  # 不允许预填充
                        output_reason="delay",  # 原因：延迟
                        **debug_info,  # 调试信息
                    )
            exist_previous_wait = prev_state is not None  # 是否存在之前的等待
            return _NegotiateOutput(  # 返回允许预填充
                next_state=None,  # 无需延迟状态
                output_allow=True,  # 允许预填充
                output_reason="wait_success" if exist_previous_wait else "no_wait",  # 等待成功或无需等待
                **debug_info,  # 调试信息
            )
        elif prefillable_status == "none":  # 如果全部不可预填充
            return _NegotiateOutput(  # 返回允许（因为无影响）
                next_state=None,  # 无需延迟状态
                # It does not matter whether we allow or not, thus we allow for simplicity  允许与否无关紧要，因此简单起见选择允许
                output_allow=True,  # 允许预填充
                output_reason="",  # 原因为空
                **debug_info,  # 调试信息
            )
        elif prefillable_status == "mixed":  # 如果混合状态（部分可预填充）
            if global_exists_token_watermark_force_allow:  # 如果存在低水位线强制允许
                return _NegotiateOutput(  # 返回允许预填充
                    next_state=None,  # 无需延迟状态
                    output_allow=True,  # 允许预填充
                    output_reason="token_watermark",  # 原因：Token水位线
                    **debug_info,  # 调试信息
                )

            prev_delayed_count = prev_state.delayed_count if prev_state else 0  # 获取之前延迟次数
            if prev_delayed_count < self._max_delay_passes - 1:  # 如果未达到最大延迟轮数
                next_state = prev_state or _State()  # 获取或创建新状态
                next_state = next_state.bump_delayed_count()  # 增加延迟计数
                return _NegotiateOutput(  # 返回延迟预填充
                    next_state=next_state,  # 延迟状态
                    output_allow=False,  # 不允许预填充
                    output_reason="delay",  # 原因：延迟
                    **debug_info,  # 调试信息
                )
            else:  # 达到最大延迟轮数
                return _NegotiateOutput(  # 返回允许预填充（超时）
                    next_state=None,  # 无需延迟状态
                    output_allow=True,  # 允许预填充
                    output_reason="wait_timeout",  # 原因：等待超时
                    **debug_info,  # 调试信息
                )
        else:  # 其他未知状态
            raise NotImplementedError  # 抛出未实现错误

    def _gather_info(  # 收集各rank的信息并进行全收集同步
        self,
        local_prefillable: bool,  # 本地是否可预填充
        local_token_watermark_force_allow: bool,  # 本地Token水位线强制允许
        running_batch: int = 0,  # 运行中的批次数
        max_prefill_bs: int = 0,  # 最大预填充批次大小
        waiting_queue_len: int = 0,  # 等待队列长度
    ):
        local_info = torch.tensor(  # 构建本地信息张量
            [  # 五个字段
                int(local_prefillable),  # 是否可预填充
                int(local_token_watermark_force_allow),  # 是否因低水位线强制允许
                running_batch,  # 运行批次数
                max_prefill_bs,  # 最大预填充批次大小
                waiting_queue_len,  # 等待队列长度
            ],
            device=self._gather_device,  # 设备
            dtype=torch.int64,  # 64位整数
        )
        torch.distributed.all_gather_into_tensor(  # 全收集操作
            self._global_info_buffer.flatten(),  # 目标缓冲区（展平）
            local_info,  # 源张量
            group=self._gather_group,  # 通信组
        )
        tp0_info = self._global_info_buffer[:, 0, :]  # 取TP rank 0的数据
        return tp0_info  # 返回TP0信息


class PrefillDelayerSinglePassExecutor:  # 预填充延迟器单次执行器，确保每次调度只协商一次
    def __init__(self, prefill_delayer: PrefillDelayer, token_usage: float):  # 初始化单次执行器
        self._prefill_delayer = prefill_delayer  # 预填充延迟器实例
        self._token_usage = token_usage  # Token使用率
        self._result: Optional[_NegotiateOutput] = None  # 协商结果，初始为None

    @property
    def _called(self) -> bool:  # 是否已调用过协商
        return self._result is not None  # 结果不为None表示已调用

    def finalize(self, *, actual_prefill: bool):  # 完成单次执行，记录指标
        if not self._called:  # 如果未调用过协商
            self.negotiate_should_allow_prefill(local_prefillable=False)  # 以不可预填充调用一次

        _record_single_pass_result(  # 记录单次执行结果
            actual_execution=actual_prefill,  # 实际是否执行了预填充
            output=self._result,  # 协商输出
            metrics_collector=self._prefill_delayer._metrics_collector,  # 指标收集器
        )

    def negotiate_should_allow_prefill(  # 协商是否允许预填充（单次执行版本）
        self,
        local_prefillable: bool,  # 本地是否可预填充
        running_batch: int = 0,  # 运行中的批次数
        max_prefill_bs: int = 0,  # 最大预填充批次大小
        max_running_requests: int = 0,  # 最大运行请求数
        waiting_queue_len: int = 0,  # 等待队列长度
    ) -> bool:  # 返回是否允许预填充
        if not self._called:  # 如果尚未调用过
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(  # 调用协商
                local_prefillable=local_prefillable,  # 本地可预填充
                token_usage=self._token_usage,  # Token使用率
                running_batch=running_batch,  # 运行批次数
                max_prefill_bs=max_prefill_bs,  # 最大预填充批次大小
                max_running_requests=max_running_requests,  # 最大运行请求数
                waiting_queue_len=waiting_queue_len,  # 等待队列长度
            )
        return self._result.output_allow  # 返回是否允许预填充


def _record_single_pass_result(  # 记录单次执行结果到日志和指标收集器
    actual_execution: bool,  # 实际是否执行了预填充
    output: _NegotiateOutput,  # 协商输出
    metrics_collector: Optional["SchedulerMetricsCollector"],  # 指标收集器
) -> None:
    if _DEBUG_LOG:  # 如果启用调试日志
        if output.output_allow and (output.output_reason == "wait_timeout"):  # 如果因超时允许
            logger.info(  # 记录超时信息
                f"PrefillDelayer timeout thus not forbid prefill "
                f"(num_prefillable={output.num_prefillable}, "
                f"actual_execution={actual_execution})"
            )
        elif output.output_allow and (output.output_reason == "token_watermark"):  # 如果因低水位线允许
            logger.info(  # 记录低水位线信息
                f"PrefillDelayer force allow prefill due to low watermark. "
                f"(num_prefillable={output.num_prefillable}, "
                f"num_token_watermark_force_allow={output.num_token_watermark_force_allow}, "
                f"actual_execution={actual_execution})"
            )
        else:  # 其他情况
            assert output.output_reason in {  # 断言原因在已知集合中
                "",
                "wait_success",
                "no_wait",
                "delay",
            }

    if metrics_collector is not None:  # 如果有指标收集器
        if (s := output.next_state) is not None:  # 如果有延迟状态
            wait_seconds = time.perf_counter() - s.start_time  # 计算等待秒数
            forward_passes = s.delayed_count  # 获取延迟轮数
        else:  # 无延迟状态
            wait_seconds = forward_passes = 0  # 等待时间和延迟轮数均为0
        metrics_collector.observe_prefill_delayer_outcome(  # 观察预填充延迟器结果
            forward_passes=forward_passes,  # 延迟轮数
            wait_seconds=wait_seconds,  # 等待秒数
            input_estimation=output.input_estimation,  # 输入估算
            output_allow=output.output_allow,  # 是否允许预填充
            output_reason=output.output_reason,  # 允许/拒绝原因
            actual_execution=actual_execution,  # 实际是否执行
        )
