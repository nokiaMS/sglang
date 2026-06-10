# Ray感知的数据并行控制器模块
# 本模块实现RayDataParallelController，使用Ray Actor而非mp.Process
# 来启动调度器进程，支持数据并行和DP注意力模式。
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
"""Ray-aware DataParallelController that launches SchedulerActors instead of mp.Process."""  # Ray感知的数据并行控制器，使用SchedulerActor替代mp.Process

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from typing import List, Optional  # 导入类型注解

import ray  # 导入Ray
import zmq  # 导入ZeroMQ

from sglang.srt.entrypoints.engine import _calculate_rank_ranges  # 导入排名范围计算函数
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info  # 导入DP注意力世界信息计算
from sglang.srt.managers.data_parallel_controller import DataParallelController  # 导入数据并行控制器基类
from sglang.srt.ray.engine import (  # 导入Ray引擎相关函数
    _compute_world_size,
    _create_scheduler_actor,
    _get_bundle_node_ip,
    _resolve_bundle_indices,
)
from sglang.srt.server_args import PortArgs, ServerArgs  # 导入服务器参数类
from sglang.srt.utils.network import bind_port, get_zmq_socket, get_zmq_socket_on_host  # 导入网络工具函数

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


class RayDataParallelController(DataParallelController):
    """DataParallelController that uses Ray actors for scheduler processes.  # 使用Ray Actor作为调度器进程的数据并行控制器

    Overrides the process-spawning methods to create SchedulerActor Ray actors
    instead of mp.Process. Runs in-process (not as a separate mp.Process) and
    reuses the parent's event_loop, dispatching, and ZMQ routing.  # 覆盖进程生成方法以创建SchedulerActor Ray Actor而非mp.Process，在进程内运行并复用父级事件循环
    """

    def __init__(  # 初始化方法
        self,
        server_args: ServerArgs,  # 服务器参数
        port_args: PortArgs,  # 端口参数
        placement_group,  # Ray放置组
        bundle_for_node: Optional[List[int]],  # 节点到bundle索引的映射
        rank0_node_ip: str,  # rank0节点IP
    ):
        # Set Ray-specific attributes BEFORE super().__init__() because the  # 在super().__init__()之前设置Ray特定属性
        # parent constructor calls launch_dp_schedulers / launch_dp_attention_schedulers  # 因为父类构造函数会调用launch_dp_schedulers / launch_dp_attention_schedulers
        # which we override, and those methods need these attributes.  # 这些方法需要这些属性
        self.pg = placement_group  # 保存放置组
        self.bundle_for_node = bundle_for_node  # 保存节点bundle映射
        self.rank0_node_ip = rank0_node_ip  # 保存rank0节点IP
        self.scheduler_actors: List = []  # 调度器Actor列表
        self.event_loop_refs: List = []  # 事件循环引用列表

        # super().__init__ will call our overridden launch methods via MRO.  # super().__init__通过MRO调用我们覆盖的启动方法
        # Pass run_scheduler_process_func=None since we don't spawn mp.Process.  # 传入run_scheduler_process_func=None因为不生成mp.Process
        super().__init__(server_args, port_args, run_scheduler_process_func=None)

    def launch_dp_schedulers(self, server_args: ServerArgs, port_args: PortArgs):
        """Override: launch Ray scheduler actors per DP rank."""  # 覆盖：为每个DP rank启动Ray调度器Actor
        sockets = []  # 用于持有NCCL端口的socket列表
        dp_port_args_list = []  # 每个DP rank的端口参数列表

        for dp_rank in range(server_args.dp_size):  # 遍历每个DP rank
            tmp_port_args = PortArgs.init_new(server_args)  # 初始化新的端口参数
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name  # 复用分词器IPC名称
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name  # 复用反分词器IPC名称
            tmp_port_args.instance_id = port_args.instance_id  # 复用实例ID

            # Hold NCCL port so the next DP rank gets a different one  # 占用NCCL端口以便下一个DP rank获得不同端口
            sockets.append(bind_port(tmp_port_args.nccl_port))  # 绑定并持有端口
            dp_port_args_list.append(tmp_port_args)  # 添加到列表

            # Create ZMQ PUSH socket for this DP rank (controller → scheduler)  # 为此DP rank创建ZMQ PUSH socket（控制器→调度器）
            if server_args.node_rank == 0:  # 如果是rank0节点
                self.workers[dp_rank] = get_zmq_socket(  # 创建ZMQ PUSH socket
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Release held ports before creating actors  # 在创建Actor之前释放占用的端口
        for sock in sockets:  # 遍历所有socket
            sock.close()  # 关闭socket

        # Create actors for each DP rank sequentially  # 为每个DP rank顺序创建Actor
        for dp_rank in range(server_args.dp_size):  # 遍历每个DP rank
            self._launch_ray_tp_group(server_args, dp_port_args_list[dp_rank], dp_rank)  # 启动Ray TP组

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        """Override: pre-allocate ports, skip broadcast, create Ray actors."""  # 覆盖：预分配端口，跳过广播，创建Ray Actor
        # Pre-allocate worker ports on the controller node, binding to the  # 在控制器节点上预分配worker端口
        # rank-0 node IP instead of tcp://* to avoid exposing unauthenticated  # 绑定到rank-0节点IP而非tcp://*以避免暴露未认证的
        # ZMQ sockets (CVE-2026-3060).  # ZMQ socket（CVE-2026-3060）
        worker_ports = []  # worker端口列表
        for dp_rank in range(server_args.dp_size):  # 遍历每个DP rank
            worker_port, worker_socket = get_zmq_socket_on_host(  # 在指定主机上创建ZMQ socket
                self.context, zmq.PUSH, host=self.rank0_node_ip
            )
            worker_ports.append(worker_port)  # 添加端口到列表
            self.workers[dp_rank] = worker_socket  # 保存socket
            logger.debug(f"Assigned port {worker_port} to worker {dp_rank}")  # 记录调试日志

        # Skip _broadcast_worker_ports — Ray creates all actors centrally,  # 跳过_broadcast_worker_ports——Ray集中创建所有Actor
        # so there's no need for the inter-node handshake protocol.  # 因此不需要节点间握手协议
        self._launch_ray_tp_group(
            server_args, port_args, dp_rank=None, worker_ports=worker_ports
        )  # 启动Ray TP组

    def _launch_ray_tp_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        """Create SchedulerActor Ray actors for one TP group (one DP rank).  # 为一个TP组（一个DP rank）创建SchedulerActor Ray Actor

        Args:
            dp_rank: DP rank for regular DP; None for DP attention (derived from tp_rank).  # 常规DP的DP rank；DP attention模式为None
            worker_ports: Pre-allocated ports for DP attention; None for regular DP.  # DP attention的预分配端口；常规DP为None
        """
        nnodes = server_args.nnodes  # 节点数量
        batch_start_idx = len(self.scheduler_actors)  # 当前批次的起始索引

        if self.server_args.placement_group is None:  # 如果没有自定义放置组
            for node_idx in range(nnodes):  # 遍历每个节点
                bundle_idx = self.bundle_for_node[node_idx]  # 获取此节点的bundle索引
                pp_range, tp_range, pp_per_node, tp_per_node = _calculate_rank_ranges(
                    nnodes, server_args.pp_size, server_args.tp_size, node_rank=node_idx
                )  # 计算PP和TP排名范围
                for pp_rank in pp_range:  # 遍历PP排名
                    for tp_rank in tp_range:  # 遍历TP排名
                        rank_port_args = port_args  # 默认使用传入的端口参数
                        actual_dp_rank = dp_rank  # 实际DP rank

                        local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                            tp_rank % tp_per_node
                        )  # 计算本地GPU索引

                        if server_args.enable_dp_attention:  # 如果启用DP注意力
                            _, _, actual_dp_rank, _ = compute_dp_attention_world_info(
                                server_args.enable_dp_attention,
                                tp_rank,
                                server_args.tp_size,
                                server_args.dp_size,
                                server_args.attn_cp_size,
                            )  # 计算DP注意力世界信息
                            rank_port_args = PortArgs.init_new(
                                server_args, actual_dp_rank, worker_ports
                            )  # 初始化新的端口参数
                            # All DP ranks share the same NCCL port (reuse TP group)  # 所有DP rank共享同一NCCL端口（复用TP组）
                            rank_port_args.nccl_port = port_args.nccl_port  # 复用NCCL端口
                            rank_port_args.instance_id = port_args.instance_id  # 复用实例ID
                            # The detokenizer and tokenizer bind using the  # 反分词器和分词器使用
                            # original port_args addresses (127.0.0.1 when  # 原始port_args地址绑定（当
                            # dist_init_addr is unset).  Scheduler actors must  # dist_init_addr未设置时为127.0.0.1）
                            # connect to the same addresses.  # 调度器Actor必须连接到相同地址
                            rank_port_args.detokenizer_ipc_name = (
                                port_args.detokenizer_ipc_name
                            )  # 复用反分词器IPC名称
                            rank_port_args.tokenizer_ipc_name = (
                                port_args.tokenizer_ipc_name
                            )  # 复用分词器IPC名称

                        dist_init_addr = (
                            f"{self.rank0_node_ip}:{rank_port_args.nccl_port}"
                        )  # 构建分布式初始化地址

                        actor = _create_scheduler_actor(  # 创建调度器Actor
                            pg=self.pg,
                            bundle_idx=bundle_idx,
                            gpu_id=local_gpu_idx,
                            server_args=server_args,
                            port_args=rank_port_args,
                            tp_rank=tp_rank,
                            pp_rank=pp_rank,
                            dp_rank=actual_dp_rank,
                            dist_init_addr=dist_init_addr,
                            rank0_node_ip=self.rank0_node_ip,
                        )
                        self.scheduler_actors.append(actor)  # 添加到Actor列表

        else:  # 自定义放置组模式
            world_size = _compute_world_size(server_args)  # 计算世界大小
            bundle_indices = _resolve_bundle_indices(self.pg, world_size)  # 解析bundle索引

            ranks_per_tp_group = server_args.tp_size * server_args.pp_size  # 每个TP组的rank数
            if dp_rank is not None:  # 常规DP模式
                start_rank = dp_rank * ranks_per_tp_group  # 起始rank
                end_rank = start_rank + ranks_per_tp_group  # 结束rank
                # Each DP group must use its own local rank-0's node IP for  # 每个DP组必须使用自己本地rank-0的节点IP
                # NCCL rendezvous, not the world rank-0's node IP.  # 进行NCCL会合，而非世界rank-0的节点IP
                local_rank0_bundle_idx = bundle_indices[start_rank]  # 本地rank-0的bundle索引
                local_rank0_node_ip = _get_bundle_node_ip(
                    self.pg, local_rank0_bundle_idx
                )  # 获取本地rank-0节点IP
            else:  # DP注意力模式
                start_rank = 0  # 起始rank为0
                end_rank = world_size  # 结束rank为世界大小
                local_rank0_node_ip = self.rank0_node_ip  # 使用rank0节点IP

            for global_rank in range(start_rank, end_rank):  # 遍历全局rank
                local_rank = global_rank % ranks_per_tp_group  # 计算本地rank
                pp_rank = local_rank // server_args.tp_size  # 计算PP rank
                tp_rank = local_rank % server_args.tp_size  # 计算TP rank
                rank_port_args = port_args  # 默认使用传入的端口参数
                actual_dp_rank = dp_rank  # 实际DP rank

                bundle_idx = bundle_indices[global_rank]  # 获取bundle索引

                if server_args.enable_dp_attention:  # 如果启用DP注意力
                    _, _, actual_dp_rank, _ = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )  # 计算DP注意力世界信息
                    rank_port_args = PortArgs.init_new(
                        server_args, actual_dp_rank, worker_ports
                    )  # 初始化新的端口参数
                    rank_port_args.nccl_port = port_args.nccl_port  # 复用NCCL端口
                    rank_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name  # 复用反分词器IPC
                    rank_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name  # 复用分词器IPC

                dist_init_addr = f"{local_rank0_node_ip}:{rank_port_args.nccl_port}"  # 构建分布式初始化地址

                actor = _create_scheduler_actor(  # 创建调度器Actor
                    pg=self.pg,
                    bundle_idx=bundle_idx,
                    gpu_id=0,  # Each bundle has exactly 1 GPU  # 每个bundle恰好有1个GPU
                    server_args=server_args,
                    port_args=rank_port_args,
                    tp_rank=tp_rank,
                    pp_rank=pp_rank,
                    dp_rank=actual_dp_rank,
                    dist_init_addr=dist_init_addr,
                    rank0_node_ip=local_rank0_node_ip,
                )
                self.scheduler_actors.append(actor)  # 添加到Actor列表

        # Wait for all actors created in this call to initialize  # 等待此调用中创建的所有Actor完成初始化
        batch_actors = self.scheduler_actors[batch_start_idx:]  # 获取此批次创建的Actor
        try:  # 尝试获取Actor信息
            scheduler_infos = ray.get(
                [actor.get_info.remote() for actor in batch_actors]
            )  # 获取所有Actor的初始化信息
        except ray.exceptions.RayActorError as e:  # Actor初始化失败
            for actor in self.scheduler_actors:  # 遍历所有Actor
                try:  # 尝试杀死Actor
                    ray.kill(actor)  # 杀死Actor
                except Exception:  # 杀死失败
                    logger.error(f"Failed to kill Ray scheduler actor: {actor}")  # 记录错误
            raise RuntimeError(f"Scheduler actor failed to initialize: {e}")  # 抛出运行时错误

        # Store init info from the first actor (same across all actors)  # 从第一个Actor存储初始化信息（所有Actor相同）
        if scheduler_infos:  # 如果有初始化信息
            self.max_total_num_tokens = scheduler_infos[0]["max_total_num_tokens"]  # 保存最大总token数
            self.max_req_input_len = scheduler_infos[0]["max_req_input_len"]  # 保存最大请求输入长度

        # Start event loops (non-blocking — runs until actor is killed)  # 启动事件循环（非阻塞——运行直到Actor被杀死）
        self.event_loop_refs.extend(
            [actor.run_event_loop.remote() for actor in batch_actors]
        )  # 扩展事件循环引用列表

    # Override launch_tensor_parallel_group to be a no-op since we don't use it.  # 覆盖launch_tensor_parallel_group为空操作，因为不使用它
    # The parent's launch_dp_schedulers/launch_dp_attention_schedulers call this,  # 父类的launch_dp_schedulers/launch_dp_attention_schedulers调用此方法
    # but our overrides call _launch_ray_tp_group instead.  # 但我们的覆盖调用_launch_ray_tp_group
    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):  # 覆盖为不可调用的方法
        raise RuntimeError(
            "RayDataParallelController should not call launch_tensor_parallel_group. "
            "Use _launch_ray_tp_group instead."
        )  # 抛出运行时错误
