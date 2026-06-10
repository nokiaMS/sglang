# 本文件实现了 SGLang 的性能分析(Profiling)管理功能，支持多种性能分析器：
# - Torch Profiler：基于 torch.profiler 的 CPU/GPU 活动追踪
# - Memory Profiler：CUDA 内存快照记录
# - CUDA Runtime Profiler：cudaProfilerStart/Stop 控制
# - RPD Profiler：AMD ROCm 平台的性能分析
# 支持按推理阶段(prefill/decode)触发分析，通过 ProfileManager 统一管理。

import logging
import os
import time
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_npu

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    torch_npu._apply_patches(patches)

logger = logging.getLogger(__name__)


class ProfileManager:
    """性能分析管理器，统一管理性能分析的配置、启动和停止。
    通过 _StageBasedTrigger 按推理阶段(prefill/decode)自动触发分析的开始和结束。"""

    def __init__(self, ps: ParallelState, cpu_group):
        self.stage_based_trigger = _StageBasedTrigger(
            on_start=self._do_start,
            on_stop=self._do_stop,
        )
        self.ps = ps  # 并行状态信息
        self.cpu_group = cpu_group  # CPU 通信组，用于同步
        # 判断当前 rank 是否为节点内第一个 rank（只有第一个 rank 执行某些操作）
        self.first_rank_in_node = ps.gpu_id == get_global_server_args().base_gpu_id
        self.profiler_kwargs = None  # 性能分析器参数
        self.profiler = None  # 当前的性能分析器实例

    def step(self, forward_mode: ForwardMode):
        """在每个推理步骤中调用，根据前向模式更新阶段触发器。"""
        stage = _get_stage_from_forward_mode(forward_mode)
        if stage is None:
            return

        self.stage_based_trigger.step(stage=stage)

    def configure(
        self,
        *,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
        merge_profiles: bool,
        profile_prefix: str,
        profile_stages: Optional[List[str]] = None,
    ):
        """配置性能分析参数。目前仅支持按阶段(profile_by_stage)分析模式。"""
        # not supported yet
        assert start_step is None
        assert (
            profile_by_stage
        ), "only support profile_by_stage=true now"  # `false` can be easily supported
        assert not merge_profiles

        # 设置输出目录，默认使用环境变量或 /tmp
        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.profiler_kwargs = dict(
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            output_dir=output_dir,
            output_prefix=profile_prefix,
            profile_id=profile_id,
        )

        self.stage_based_trigger.configure(
            num_steps=num_steps,
            # 默认分析 prefill 和 decode 两个阶段
            interesting_stages=profile_stages or ["prefill", "decode"],
        )

        return ProfileReqOutput(success=True, message="Succeeded")

    def manual_start(self):
        """手动启动性能分析（暂未实现）。"""
        raise NotImplementedError("manually start is only supported yet")

    def manual_stop(self):
        """手动停止性能分析（暂未实现）。"""
        raise NotImplementedError("manually stop is only supported yet")

    def _do_start(self, stage: Optional[str] = None):
        """实际启动性能分析，根据配置创建对应的分析器实例。"""
        logger.info(
            f"Profiling starts{f' for {stage}' if stage else ''}. "
            f"Traces will be saved to: {self.profiler_kwargs['output_dir']} "
            f"(with profile id: {self.profiler_kwargs['profile_id']})",
        )

        assert self.profiler is None
        self.profiler = _ProfilerBase.create(
            **self.profiler_kwargs,
            ps=self.ps,
            cpu_group=self.cpu_group,
            first_rank_in_node=self.first_rank_in_node,
            output_suffix=f"-{stage}" if stage else "",
        )
        self.profiler.start()

    def _do_stop(self):
        """实际停止性能分析，导出追踪数据。"""
        logger.info("Stop profiling...")
        self.profiler.stop()
        logger.info(
            f"Profiling done. Traces are saved to: {self.profiler_kwargs['output_dir']}"
        )
        self.profiler = None


def _get_stage_from_forward_mode(forward_mode: ForwardMode):
    """将前向模式映射为性能分析阶段名称。
    prefill -> "prefill", decode -> "decode", idle -> None（跳过）。"""
    if forward_mode.is_prefill():
        return "prefill"
    elif forward_mode.is_decode():
        return "decode"
    elif forward_mode.is_idle():
        return None
    else:
        raise RuntimeError(f"unsupported profile stage: {forward_mode=}")


# ======================================== Stage related ==========================================


class _StageBasedTrigger:
    """基于推理阶段的触发器，控制性能分析的自动开始和结束。
    当进入感兴趣的阶段时自动开始分析，达到指定步数后或切换阶段时自动停止。"""

    @dataclass
    class _StageConfig:
        """阶段配置：记录需要分析的步数目标。"""
        target_count: int

    @dataclass
    class _RunningState:
        """运行状态：记录当前正在分析的阶段和已完成步数。"""
        curr_stage: str
        curr_count: int

    def __init__(self, on_start: Callable, on_stop: Callable):
        self.on_start = on_start  # 开始分析的回调
        self.on_stop = on_stop  # 停止分析的回调

        self.running_state: Optional[_StageBasedTrigger._RunningState] = None
        # When a stage is in the dict, it means it is being or should be executed
        # 当阶段在字典中时，表示该阶段正在或将要被执行分析
        self.stage_configs: Dict[str, _StageBasedTrigger._StageConfig] = {}

    def configure(self, num_steps: int, interesting_stages: List[str]):
        """配置需要分析的阶段和每个阶段的步数。"""
        assert self.running_state is None
        self.stage_configs = {
            stage: self._StageConfig(target_count=num_steps)
            for stage in interesting_stages
        }

    def step(self, stage: str):
        """在每个推理步骤中调用，更新计数器并根据条件触发开始/停止分析。"""
        # Incr counter
        # 递增当前运行阶段的计数器
        if (s := self.running_state) is not None:
            s.curr_count += 1

        # Maybe stop
        # 可能停止：当前阶段计数达到目标或阶段切换时停止分析
        if ((s := self.running_state) is not None) and (
            (s.curr_count > self.stage_configs[s.curr_stage].target_count)
            or (stage != s.curr_stage)
        ):
            del self.stage_configs[s.curr_stage]
            self.running_state = None
            self.on_stop()

        # Maybe start
        # 可能开始：当前无运行分析且阶段在配置中时开始分析
        if (self.running_state is None) and (stage in self.stage_configs):
            self.running_state = self._RunningState(
                curr_stage=stage,
                curr_count=0,
            )
            self.on_start(stage=stage)

        # Sanity check
        # 一致性检查：运行状态与阶段配置应保持同步
        assert (self.running_state is not None) == (stage in self.stage_configs)
        if (s := self.running_state) is not None:
            assert s.curr_stage == stage


# ======================================== Concrete profilers ==========================================


class _ProfilerBase(ABC):
    """性能分析器基类，提供工厂方法根据活动类型创建具体的分析器组合。"""

    @staticmethod
    def create(activities, with_stack, record_shapes, **kwargs):
        """工厂方法：根据活动类型列表创建对应的分析器实例。
        支持的活动类型：CPU/GPU（Torch Profiler）、MEM（内存分析）、
        CUDA_PROFILER（CUDA Runtime 分析）、RPD（ROCm 分析）。"""
        inners = []
        if ("CPU" in activities) or ("GPU" in activities):
            inners.append(
                _ProfilerTorch(
                    **kwargs,
                    activities=activities,
                    with_stack=with_stack,
                    record_shapes=record_shapes,
                )
            )
        if "MEM" in activities:
            inners.append(_ProfilerMemory(**kwargs))
        if "CUDA_PROFILER" in activities:
            inners.append(_ProfilerCudart(**kwargs))
        if "RPD" in activities:  # for ROCm
            inners.append(_ProfilerRPD(**kwargs))

        return _ProfilerList(inners)

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _ProfilerList(_ProfilerBase):
    """组合多个分析器的列表实现，同时启动/停止所有内部分析器。"""

    def __init__(self, inners: List[_ProfilerBase]):
        self.inners = inners

    def start(self):
        for inner in self.inners:
            inner.start()

    def stop(self):
        for inner in self.inners:
            inner.stop()


class _ProfilerConcreteBase(_ProfilerBase):
    """具体分析器的公共基类，包含输出路径、并行状态等通用参数。"""

    def __init__(
        self,
        output_dir: str,
        output_prefix: str,
        output_suffix: str,
        profile_id: str,
        ps: ParallelState,
        cpu_group,
        first_rank_in_node: bool,
    ):
        self.output_dir = output_dir  # 输出目录
        self.output_prefix = output_prefix  # 输出文件名前缀
        self.output_suffix = output_suffix  # 输出文件名后缀（如阶段名）
        self.profile_id = profile_id  # 分析标识
        self.ps = ps  # 并行状态
        self.cpu_group = cpu_group  # CPU 通信组
        self.first_rank_in_node = first_rank_in_node  # 是否为节点内第一个 rank


class _ProfilerTorch(_ProfilerConcreteBase):
    """基于 torch.profiler 的性能分析器，记录 CPU/GPU 活动，
    导出 Chrome Trace 格式的追踪文件。"""

    def __init__(self, with_stack: bool, record_shapes: bool, activities, **kwargs):
        super().__init__(**kwargs)
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.activities = activities

    def start(self):
        """启动 Torch Profiler，根据活动类型配置 CPU/GPU 追踪。"""
        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in self.activities if a in activity_map
        ]

        self.torch_profiler = torch.profiler.profile(
            activities=torchprof_activities,
            with_stack=self.with_stack if self.with_stack is not None else True,
            record_shapes=(
                self.record_shapes if self.record_shapes is not None else False
            ),
            # NPU 平台使用 tensorboard 追踪处理器
            on_trace_ready=(
                None
                if not _is_npu
                else torch_npu.profiler.tensorboard_trace_handler(self.output_dir)
            ),
        )
        self.torch_profiler.start()

    def stop(self):
        """停止 Torch Profiler 并导出 Chrome Trace 文件。"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.torch_profiler.stop()
        if not _is_npu:
            # Build filename with only non-zero ranks to maintain backward compatibility
            # 构建文件名：包含 profile_id 和 TP rank，仅在对应并行度>1时添加 DP/PP/EP rank
            filename_parts = [self.profile_id, f"TP-{self.ps.tp_rank}"]

            # Only add other ranks if parallelism is enabled (size > 1)
            if self.ps.dp_size > 1:
                filename_parts.append(f"DP-{self.ps.dp_rank}")
            if self.ps.pp_size > 1:
                filename_parts.append(f"PP-{self.ps.pp_rank}")
            if self.ps.moe_ep_size > 1:
                filename_parts.append(f"EP-{self.ps.moe_ep_rank}")

            filename = (
                (self.output_prefix + "-" if self.output_prefix else "")
                + "-".join(filename_parts)
                + self.output_suffix
                + ".trace.json.gz"
            )

            self.torch_profiler.export_chrome_trace(
                os.path.join(self.output_dir, filename)
            )
        # 所有 rank 同步，确保所有分析数据都已写入
        torch.distributed.barrier(self.cpu_group)

        # TODO: migrate `_merge_profile_traces`


class _ProfilerMemory(_ProfilerConcreteBase):
    """CUDA 内存分析器，记录内存分配历史并导出快照文件。"""

    def start(self):
        """启动内存历史记录。"""
        torch.cuda.memory._record_memory_history(max_entries=100000)

    def stop(self):
        """停止内存历史记录并导出快照到 pickle 文件。"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        memory_profile_path = os.path.join(
            self.output_dir,
            str(time.time())
            + f"-TP-{self.ps.tp_rank}-memory"
            + self.output_suffix
            + ".pickle",
        )
        torch.cuda.memory._dump_snapshot(memory_profile_path)
        # 停止内存历史记录
        torch.cuda.memory._record_memory_history(enabled=None)


class _ProfilerCudart(_ProfilerConcreteBase):
    """CUDA Runtime 性能分析器，通过 cudaProfilerStart/Stop 控制外部分析工具
    （如 Nsight Systems）的数据采集。仅由节点内第一个 rank 执行。"""

    def start(self):
        """启动 CUDA Runtime 性能分析。"""
        if self.first_rank_in_node:
            logger.info(f"Call cudaProfilerStart")
            torch.cuda.cudart().cudaProfilerStart()

    def stop(self):
        """停止 CUDA Runtime 性能分析。"""
        if self.first_rank_in_node:
            logger.info(f"Call cudaProfilerStop")
            torch.cuda.cudart().cudaProfilerStop()


class _ProfilerRPD(_ProfilerConcreteBase):
    """AMD ROCm 平台的 RPD 性能分析器，使用 rpdTracerControl 记录追踪数据，
    并转换为 Chrome Trace 格式。仅由 TP rank 0 执行转换。"""

    def start(self):
        """启动 RPD 性能分析，初始化追踪数据库和追踪器。"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        from rpdTracerControl import rpdTracerControl

        rpdTracerControl.skipCreate()

        self.rpd_profile_path = os.path.join(
            self.output_dir,
            "rpd-" + str(time.time()) + f"-TP-{self.ps.tp_rank}" + ".trace.json.gz",
        )

        # TP rank 0 初始化 RPD 数据库 schema
        if self.ps.tp_rank == 0:
            import sqlite3

            from rocpd.schema import RocpdSchema

            if os.path.exists("trace.rpd"):
                os.unlink("trace.rpd")
            schema = RocpdSchema()
            connection = sqlite3.connect("trace.rpd")
            schema.writeSchema(connection)
            connection.commit()
            del connection
        # 所有 rank 同步，确保数据库已初始化
        torch.distributed.barrier(self.cpu_group)

        # 启动 RPD 追踪器
        self.rpd_profiler = rpdTracerControl()
        self.rpd_profiler.setPythonTrace(True)
        self.rpd_profiler.start()
        self.rpd_profiler.rangePush("", "rpd profile range", "")

    def stop(self):
        """停止 RPD 追踪，刷新数据，并由 TP rank 0 将 RPD 追踪转换为 Chrome Trace 格式。"""
        self.rpd_profiler.rangePop()
        self.rpd_profiler.stop()
        self.rpd_profiler.flush()

        # 所有 rank 同步，确保追踪数据已刷新
        torch.distributed.barrier(self.cpu_group)
        # TP rank 0 执行格式转换
        if self.ps.tp_rank == 0:
            from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace

            rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
