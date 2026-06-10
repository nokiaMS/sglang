# Ray Actor封装的SGLang调度器模块
# 本模块实现SchedulerActor，作为Ray Actor封装SGLang调度器，
# 每个Actor管理一个GPU并运行Scheduler+TpModelWorker堆栈。
# Copyright 2023-2024 SGLang Team
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
"""Ray actor wrapper for SGLang Scheduler."""  # SGLang调度器的Ray Actor封装

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from typing import TYPE_CHECKING, Any, Dict, Optional  # 导入类型注解

import ray  # 导入Ray

if TYPE_CHECKING:  # 类型检查时
    from sglang.srt.server_args import PortArgs, ServerArgs  # 导入服务器参数类


logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


@ray.remote
class SchedulerActor:  # 调度器Ray Actor
    """Ray actor wrapper for SGLang Scheduler.  # SGLang调度器的Ray Actor封装

    Each actor manages one GPU and runs the Scheduler + TpModelWorker stack.  # 每个Actor管理一个GPU并运行Scheduler+TpModelWorker堆栈
    Ray is used for process lifecycle; ZMQ handles request/response communication.  # Ray用于进程生命周期管理；ZMQ处理请求/响应通信
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dist_init_addr: Optional[str] = None,
    ):  # 初始化方法
        import dataclasses  # 导入数据类模块

        from sglang.srt.environ import envs  # 导入环境变量
        from sglang.srt.managers.scheduler import Scheduler, configure_scheduler_process  # 导入调度器和配置函数
        from sglang.srt.utils.numa_utils import (  # 导入NUMA工具函数
            get_numa_node_if_available,
            numa_bind_to_node,
        )

        # Override dist_init_addr if provided (for multi-node)  # 如果提供了dist_init_addr则覆盖（用于多节点）
        if dist_init_addr:  # 如果有分布式初始化地址
            server_args = dataclasses.replace(
                server_args, dist_init_addr=dist_init_addr
            )  # 替换dist_init_addr

        # Get actual GPU IDs from Ray runtime context  # 从Ray运行时上下文获取实际GPU ID
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()  # 获取加速器ID
        assigned_gpus = accelerator_ids.get("GPU", [])  # 获取分配的GPU列表

        if assigned_gpus:  # 如果Ray分配了GPU
            # Ray assigned specific GPU(s), use the first one  # Ray分配了特定GPU，使用第一个
            actual_gpu_id = int(assigned_gpus[0])  # 转换为整数
            logger.info(f"[TP{tp_rank}] Ray assigned GPU: {actual_gpu_id}")  # 记录日志
        else:  # Ray未分配GPU
            # Fallback to passed gpu_id  # 回退到传入的gpu_id
            actual_gpu_id = gpu_id  # 使用传入的GPU ID
            logger.info(f"[TP{tp_rank}] Using passed gpu_id: {gpu_id}")  # 记录日志

        # Configure worker (logging, process title, etc.)  # 配置worker（日志、进程标题等）
        dp_rank = configure_scheduler_process(
            server_args,
            actual_gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
        )  # 配置调度器进程

        # Ray actors can't use the numactl subprocess-wrapping approach  # Ray Actor无法使用numactl子进程包装方式
        # (SGLANG_NUMA_BIND_V2's normal path), so bind in-process via libnuma.  # SGLANG_NUMA_BIND_V2的常规路径，因此通过libnuma在进程内绑定
        # The V1 path inside configure_scheduler_process already handles  # configure_scheduler_process中的V1路径已处理
        # SGLANG_NUMA_BIND_V2=False.  # SGLANG_NUMA_BIND_V2=False的情况
        if envs.SGLANG_NUMA_BIND_V2.get():  # 如果启用NUMA绑定V2
            numa_node = get_numa_node_if_available(server_args, actual_gpu_id)  # 获取NUMA节点
            if numa_node is not None:  # 如果有可用NUMA节点
                numa_bind_to_node(numa_node)  # 绑定到NUMA节点
                logger.info(
                    f"[TP{tp_rank}] Bound to NUMA node {numa_node} for GPU {actual_gpu_id}"
                )  # 记录NUMA绑定日志

        # Create scheduler (loads model into GPU, initializes NCCL)  # 创建调度器（将模型加载到GPU，初始化NCCL）
        self.scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=actual_gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
        )  # 创建调度器实例

        self._tp_rank = tp_rank  # 保存TP rank
        self._pp_rank = pp_rank  # 保存PP rank

    def get_info(self) -> Dict[str, Any]:  # 获取调度器初始化信息
        """Return scheduler initialization info for handshake."""  # 返回调度器初始化信息用于握手
        return self.scheduler.get_init_info()  # 返回初始化信息

    def run_event_loop(self) -> None:  # 运行调度器事件循环
        """Run the scheduler's event loop. Blocks until shutdown."""  # 运行调度器的事件循环，阻塞直到关闭
        try:  # 尝试运行
            import torch  # 导入PyTorch

            # Need to set the GPU id for the event loop for nccl to work  # 需要为事件循环设置GPU ID以使NCCL工作
            torch.cuda.set_device(self.scheduler.ps.gpu_id)  # 设置CUDA设备
            self.scheduler.run_event_loop()  # 运行事件循环
        except Exception as e:  # 异常
            logger.error(f"Scheduler PP{self._pp_rank} TP{self._tp_rank} crashed: {e}")  # 记录错误
            raise  # 重新抛出异常
