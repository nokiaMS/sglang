# 调度器状态日志记录器，定期输出调度器的运行中队列和等待队列状态
# 通过JSON格式将调度器状态信息发送到配置的日志目标
from __future__ import annotations  # 启用延迟注解求值

import time  # 导入时间模块
from typing import TYPE_CHECKING, List, Optional  # 导入类型注解

import torch.distributed as dist  # 导入PyTorch分布式通信模块

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.utils.log_utils import create_log_targets, log_json  # 导入日志目标创建和JSON日志工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # 导入请求和调度批次类型


class SchedulerStatusLogger:  # 调度器状态日志记录器类
    def __init__(self, targets: List[str], dump_interval: float):  # 初始化日志记录器
        self.loggers = create_log_targets(targets=targets, name_prefix=__name__)  # 创建日志目标列表
        self.dump_interval = dump_interval  # 状态转储间隔（秒）
        self.last_dump_time = 0.0  # 上次转储时间戳
        self.rank = dist.get_rank() if dist.is_initialized() else 0  # 获取当前进程的分布式秩

    @staticmethod
    def maybe_create(enable_metrics: bool) -> Optional["SchedulerStatusLogger"]:  # 根据环境变量条件创建日志记录器
        target = envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.get()  # 获取日志目标配置
        if not target:  # 如果未配置日志目标
            return None  # 不创建日志记录器

        if not enable_metrics:  # 如果未启用指标
            raise ValueError(  # 抛出异常
                "SGLANG_LOG_SCHEDULER_STATUS_TARGET is set but --enable-metrics "
                "is not active. Status dumps require --enable-metrics to work."  # 状态转储需要启用指标
            )

        return SchedulerStatusLogger(  # 创建并返回日志记录器实例
            targets=[t.strip() for t in target.split(",") if t.strip()],  # 解析逗号分隔的日志目标
            dump_interval=envs.SGLANG_LOG_SCHEDULER_STATUS_INTERVAL.get(),  # 获取转储间隔配置
        )

    def maybe_dump(  # 条件性转储调度器状态（按间隔控制）
        self, running_batch: "ScheduleBatch", waiting_queue: List["Req"]  # 运行中批次和等待队列
    ) -> None:
        now = time.time()  # 获取当前时间
        if now - self.last_dump_time < self.dump_interval:  # 如果距上次转储未超过间隔
            return  # 跳过本次转储

        self.last_dump_time = now  # 更新上次转储时间
        log_json(  # 以JSON格式记录调度器状态
            self.loggers,  # 日志目标列表
            "scheduler.status",  # 日志事件名称
            {  # 状态数据字典
                "rank": self.rank,  # 当前进程的分布式秩
                "running_rids": [r.rid for r in running_batch.reqs],  # 运行中请求的ID列表
                "queued_rids": [r.rid for r in waiting_queue],  # 等待队列中请求的ID列表
            },
        )
