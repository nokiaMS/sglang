# Ray引擎模块
# 本模块实现RayEngine，继承Engine基类，使用Ray Actor启动调度器进程，
# 支持自动和自定义放置组、多节点分布式部署等功能。
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
"""RayEngine - Engine subclass that launches schedulers as Ray actors."""  # RayEngine——使用Ray Actor启动调度器的Engine子类

from __future__ import annotations  # 启用延迟类型注解评估

import dataclasses  # 导入数据类模块
import logging  # 导入日志模块
import threading  # 导入线程模块
from typing import Callable, List, Optional  # 导入类型注解

import ray  # 导入Ray
from ray.util.placement_group import PlacementGroup  # 导入放置组类
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # 导入放置组调度策略

from sglang.srt.entrypoints.engine import (  # 导入引擎基类和相关函数
    Engine,
    SchedulerInitResult,
    _calculate_rank_ranges,
    _compute_parallelism_ranks,
)
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.ray.scheduler_actor import SchedulerActor  # 导入调度器Actor类
from sglang.srt.server_args import PortArgs, ServerArgs  # 导入服务器参数类

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


@dataclasses.dataclass
class RaySchedulerInitResult(SchedulerInitResult):  # Ray调度器初始化结果
    """SchedulerInitResult that also holds Ray actor handles for cleanup."""  # 同时持有Ray Actor句柄以便清理的调度器初始化结果

    scheduler_actors: list = dataclasses.field(default_factory=list)  # 调度器Actor句柄列表


def _find_engine_bundle(
    placement_group: PlacementGroup, nnodes: int
) -> tuple[int, str]:  # 查找Engine所在的放置组bundle
    """Find which placement group bundle is on the same node as the Engine.  # 查找与Engine位于同一节点的放置组bundle
    Rank0 scheduler must be co-located with the Engine. Returns (bundle_index, engine_ip).  # Rank0调度器必须与Engine共置，返回(bundle索引, Engine IP)
    """
    engine_ip = ray.util.get_node_ip_address()  # 获取Engine节点IP

    @ray.remote(num_cpus=0, num_gpus=0)  # 定义远程函数获取节点IP
    def get_node_ip():  # 获取节点IP
        return ray.util.get_node_ip_address()  # 返回节点IP

    bundle_ips = ray.get(  # 获取每个bundle的节点IP
        [
            get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=i,
                ),
            ).remote()
            for i in range(nnodes)  # 遍历每个节点
        ]
    )

    try:  # 尝试找到Engine节点
        return bundle_ips.index(engine_ip), engine_ip  # 返回bundle索引和Engine IP
    except ValueError:  # Engine节点不在任何bundle中
        raise RuntimeError(
            f"Engine node {engine_ip} not found in any placement group bundle {bundle_ips}. "
            f"Rank-0 scheduler must be co-located with the Engine."
        )  # 抛出运行时错误


def _get_bundle_node_ip(placement_group: PlacementGroup, bundle_idx: int) -> str:  # 获取指定bundle所在节点的IP
    """Get the IP address of the node where a specific bundle is located.  # 获取指定bundle所在节点的IP地址

    Args:
        placement_group: The placement group  # 放置组
        bundle_idx: Bundle index to query  # 要查询的bundle索引

    Returns:
        IP address of the node where the bundle is located.  # bundle所在节点的IP地址
    """

    @ray.remote(num_cpus=0, num_gpus=0)  # 定义远程函数获取节点IP
    def get_node_ip():  # 获取节点IP
        return ray.util.get_node_ip_address()  # 返回节点IP

    return ray.get(
        get_node_ip.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_idx,
            ),
        ).remote()
    )  # 返回bundle所在节点的IP


def _compute_world_size(server_args: ServerArgs) -> int:  # 计算世界大小（调度器Actor/GPU总数）
    """Compute world_size (total number of scheduler actors/GPUs needed).  # 计算world_size（所需的调度器Actor/GPU总数）

    Normal: dp_size * tp_size * pp_size; DP attention: tp_size * pp_size.  # 常规模式：dp_size * tp_size * pp_size；DP注意力模式：tp_size * pp_size
    """
    if server_args.enable_dp_attention:  # 如果启用DP注意力
        return server_args.tp_size * server_args.pp_size  # TP乘以PP
    return server_args.dp_size * server_args.tp_size * server_args.pp_size  # DP乘以TP乘以PP


def _resolve_bundle_indices(pg: PlacementGroup, world_size: int) -> List[int]:  # 解析自定义放置组的bundle索引
    """Resolve bundle indices for Custom PG mode.  # 解析自定义放置组模式的bundle索引

    Parses SGLANG_RAY_BUNDLE_INDICES env var if set; otherwise returns  # 解析SGLANG_RAY_BUNDLE_INDICES环境变量（如已设置）；否则返回
    sequential indices [0, 1, ..., world_size-1].  # 顺序索引[0, 1, ..., world_size-1]

    Args:
        pg: Placement group (used to get total_bundles count).  # 放置组（用于获取total_bundles数量）
        world_size: Number of bundle indices expected (pre-computed via _compute_world_size).  # 期望的bundle索引数量

    Returns:
        List of bundle indices of length world_size.  # 长度为world_size的bundle索引列表
    """
    total_bundles = len(pg.bundle_specs)  # 获取放置组中的bundle总数
    indices_str = envs.SGLANG_RAY_BUNDLE_INDICES.get()  # 获取环境变量
    if not indices_str:  # 如果未设置
        return list(range(world_size))  # 返回顺序索引

    indices = list(map(int, indices_str.split(",")))  # 解析逗号分隔的索引

    if len(indices) != world_size:  # 如果索引数量不匹配
        raise ValueError(
            f"SGLANG_RAY_BUNDLE_INDICES has {len(indices)} values, "
            f"expected {world_size}"
        )  # 抛出值错误

    if len(set(indices)) != len(indices):  # 如果有重复索引
        raise ValueError(f"SGLANG_RAY_BUNDLE_INDICES has duplicates: {indices}")  # 抛出值错误

    for idx in indices:  # 遍历每个索引
        if idx < 0 or idx >= total_bundles:  # 如果索引越界
            raise ValueError(f"Bundle index {idx} out of range [0, {total_bundles})")  # 抛出值错误

    return indices  # 返回索引列表


def _validate_custom_placement_group(pg: PlacementGroup, world_size: int) -> None:  # 验证自定义放置组
    """Validate custom placement group: 1 GPU per bundle, enough GPU bundles for world_size.  # 验证自定义放置组：每个bundle 1个GPU，GPU bundle数足够world_size

    Args:
        pg: User-provided placement group.  # 用户提供的放置组
        world_size: Number of GPU bundles required.  # 所需的GPU bundle数量
    """
    bundles = pg.bundle_specs  # 获取bundle规格列表
    gpu_bundle_count = 0  # GPU bundle计数
    for bundle in bundles:  # 遍历每个bundle
        gpu_count = bundle.get("GPU", 0)  # 获取GPU数量
        if gpu_count > 1:  # 如果超过1个GPU
            raise ValueError(
                "Custom placement group must have exactly 1 GPU per bundle. "
                f"Found bundle with {gpu_count} GPUs."
            )  # 抛出值错误
        if gpu_count > 0:  # 如果有GPU
            gpu_bundle_count += 1  # 增加计数

    if gpu_bundle_count < world_size:  # 如果GPU bundle不足
        raise ValueError(
            f"Custom placement group has {gpu_bundle_count} GPU bundles, "
            f"but needs {world_size} for world_size. "
            "Provide more bundles or reduce parallelism."
        )  # 抛出值错误


def _create_scheduler_actor(
    pg: PlacementGroup,
    bundle_idx: int,
    gpu_id: int,
    server_args: ServerArgs,
    port_args: PortArgs,
    tp_rank: int,
    pp_rank: int,
    dp_rank: int,
    dist_init_addr: str,
    rank0_node_ip: str,
) -> SchedulerActor:  # 创建调度器Actor
    """Create a SchedulerActor on the given placement group bundle.  # 在指定放置组bundle上创建调度器Actor

    Args:
        pg: Placement group to schedule actor onto.  # 用于调度Actor的放置组
        bundle_idx: Bundle index within the placement group.  # 放置组内的bundle索引
        gpu_id: GPU ID within the bundle (0 for custom PG, computed for auto PG).  # bundle内的GPU ID
        rank0_node_ip: IP of rank-0's node, used for NCCL rendezvous.  # rank-0节点IP，用于NCCL会合
        dist_init_addr: Distributed init address (tcp://rank0_node_ip:nccl_port).  # 分布式初始化地址
    """
    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
        server_args, tp_rank
    )  # 计算并行排名

    return SchedulerActor.options(  # 创建调度器Actor
        num_cpus=0,  # 不需要CPU
        num_gpus=1,  # 需要1个GPU
        name=(
            f"sglang_scheduler_node{rank0_node_ip}"
            f"_dp{dp_rank}_pp{pp_rank}_tp{tp_rank}"
            f"_pg{pg.id.hex()[:8]}_bundle{bundle_idx}"
        ),  # Actor名称
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_idx,
        ),  # 放置组调度策略
    ).remote(
        server_args=server_args,
        port_args=port_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        moe_ep_rank=moe_ep_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        dist_init_addr=dist_init_addr,
    )  # 返回Actor句柄


class RayEngine(Engine):  # Ray引擎类
    """Engine using Ray actors for scheduler processes."""  # 使用Ray Actor作为调度器进程的引擎

    def __init__(self, **kwargs):  # 初始化方法
        placement_group = kwargs.pop("placement_group", None)  # 弹出放置组参数
        if "log_level" not in kwargs:  # 如果未设置日志级别
            kwargs["log_level"] = "error"  # 默认错误级别
        server_args = ServerArgs(**kwargs)  # 创建服务器参数
        server_args.placement_group = placement_group  # 设置放置组
        super().__init__(server_args=server_args)  # 调用父类初始化

    def shutdown(self):  # 关闭引擎
        """Shutdown the engine — kill Ray scheduler actors then local processes."""  # 关闭引擎——杀死Ray调度器Actor然后关闭本地进程
        for actor in self._scheduler_init_result.scheduler_actors:  # 遍历所有Actor
            try:  # 尝试杀死
                ray.kill(actor)  # 杀死Actor
            except Exception:  # 杀死失败
                logger.error(f"Failed to kill Ray scheduler actor: {actor}")  # 记录错误
        super().shutdown()  # 调用父类关闭

    @classmethod
    def _launch_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> tuple[SchedulerInitResult, None]:  # 启动调度器进程
        """Launch schedulers as Ray actors.  # 使用Ray Actor启动调度器

        Returns:
            Tuple of (RaySchedulerInitResult, None).  # (RaySchedulerInitResult, None)元组
            scheduler_procs is None since Ray uses actors instead of mp.Process.  # scheduler_procs为None，因为Ray使用Actor而非mp.Process
        """
        pg = server_args.placement_group or ray.util.get_current_placement_group()  # 获取放置组
        if pg is None:  # 如果没有放置组
            from ray.util.placement_group import (
                placement_group as create_placement_group,
            )  # 导入放置组创建函数

            if server_args.enable_dp_attention:  # 如果启用DP注意力
                total_gpus = server_args.tp_size * server_args.pp_size  # GPU总数
            else:  # 常规模式
                total_gpus = (
                    server_args.dp_size * server_args.tp_size * server_args.pp_size
                )  # GPU总数

            nnodes = server_args.nnodes  # 节点数
            gpus_per_node = total_gpus // nnodes  # 每节点GPU数
            strategy = "STRICT_PACK" if nnodes == 1 else "SPREAD"  # 单节点紧凑放置，多节点分散放置

            logger.info(
                "No placement group detected. Auto-creating one with "
                f"{nnodes} bundle(s), {gpus_per_node} GPU(s)/bundle, "
                "placement group explicitly and schedule the Engine onto it."
            )  # 记录自动创建放置组信息

            pg = create_placement_group(
                [{"CPU": 1, "GPU": gpus_per_node}] * nnodes,
                strategy=strategy,
            )  # 创建放置组
            ray.get(pg.ready())  # 等待放置组就绪

        is_custom_pg = server_args.placement_group is not None  # 是否是自定义放置组
        nnodes = server_args.nnodes  # 节点数
        world_size = _compute_world_size(server_args)  # 世界大小

        if not is_custom_pg:  # 自动放置组模式
            engine_bundle, engine_ip = _find_engine_bundle(pg, nnodes)  # 查找Engine bundle
            bundle_for_node = [engine_bundle] + [
                i for i in range(nnodes) if i != engine_bundle
            ]  # 构建节点到bundle的映射
            rank0_node_ip = engine_ip  # rank0节点IP
        else:  # 自定义放置组模式
            try:  # 尝试验证放置组
                _validate_custom_placement_group(pg, world_size)  # 验证自定义放置组
            except ValueError as e:  # 验证失败
                logger.error(f"Custom placement group validation failed: {e}")  # 记录错误
                raise RuntimeError(
                    f"Custom placement group validation failed: {e}"
                ) from e  # 抛出运行时错误
            bundle_for_node = None  # 不需要节点bundle映射
            indices_str = envs.SGLANG_RAY_BUNDLE_INDICES.get()  # 获取bundle索引环境变量
            rank0_bundle_idx = int(indices_str.split(",")[0]) if indices_str else 0  # rank0 bundle索引
            rank0_node_ip = _get_bundle_node_ip(pg, rank0_bundle_idx)  # rank0节点IP

        if server_args.dp_size == 1:  # 非DP模式
            dist_init_addr = f"{rank0_node_ip}:{port_args.nccl_port}"  # 分布式初始化地址
            logger.info(f"dist_init_addr: {dist_init_addr}")  # 记录日志

            scheduler_actors = []  # 调度器Actor列表

            if not is_custom_pg:  # 自动放置组模式
                gpus_per_node = world_size // nnodes  # 每节点GPU数
                logger.info(
                    f"Ray cluster (auto PG): {nnodes} nodes, "
                    f"{gpus_per_node} GPUs/node, world_size={world_size}"
                )  # 记录集群信息

                for node_idx in range(nnodes):  # 遍历每个节点
                    bundle_idx = bundle_for_node[node_idx]  # 获取bundle索引
                    pp_range, tp_range, pp_per_node, tp_per_node = (
                        _calculate_rank_ranges(
                            nnodes,
                            server_args.pp_size,
                            server_args.tp_size,
                            node_rank=node_idx,
                        )
                    )  # 计算排名范围
                    for pp_rank in pp_range:  # 遍历PP排名
                        for tp_rank in tp_range:  # 遍历TP排名
                            local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                                tp_rank % tp_per_node
                            )  # 计算本地GPU索引

                            actor = _create_scheduler_actor(  # 创建调度器Actor
                                pg=pg,
                                bundle_idx=bundle_idx,
                                gpu_id=local_gpu_idx,
                                server_args=server_args,
                                port_args=port_args,
                                tp_rank=tp_rank,
                                pp_rank=pp_rank,
                                dp_rank=0,
                                dist_init_addr=dist_init_addr,
                                rank0_node_ip=rank0_node_ip,
                            )
                            scheduler_actors.append(actor)  # 添加到列表

            else:  # 自定义放置组模式
                try:  # 尝试解析bundle索引
                    bundle_indices = _resolve_bundle_indices(pg, world_size)
                except ValueError as e:  # 解析失败
                    logger.error(f"Failed to resolve bundle indices: {e}")  # 记录错误
                    raise RuntimeError(f"Failed to resolve bundle indices: {e}") from e  # 抛出运行时错误

                logger.info(
                    f"Ray cluster (custom PG): world_size={world_size}, "
                    f"bundle_indices={bundle_indices}"
                )  # 记录集群信息

                for rank in range(world_size):  # 遍历所有rank
                    pp_rank = rank // server_args.tp_size  # 计算PP rank
                    tp_rank = rank % server_args.tp_size  # 计算TP rank
                    bundle_idx = bundle_indices[rank]  # 获取bundle索引

                    actor = _create_scheduler_actor(  # 创建调度器Actor
                        pg=pg,
                        bundle_idx=bundle_idx,
                        gpu_id=0,  # Each bundle has exactly 1 GPU  # 每个bundle恰好有1个GPU
                        server_args=server_args,
                        port_args=port_args,
                        tp_rank=tp_rank,
                        pp_rank=pp_rank,
                        dp_rank=0,
                        dist_init_addr=dist_init_addr,
                        rank0_node_ip=rank0_node_ip,
                    )
                    scheduler_actors.append(actor)  # 添加到列表

            try:  # 尝试获取Actor信息
                scheduler_infos = ray.get(
                    [actor.get_info.remote() for actor in scheduler_actors]
                )  # 获取所有Actor的初始化信息
            except ray.exceptions.RayActorError as e:  # Actor初始化失败
                for actor in scheduler_actors:  # 遍历所有Actor
                    try:  # 尝试杀死
                        ray.kill(actor)  # 杀死Actor
                    except Exception:  # 杀死失败
                        logger.error(f"Failed to kill Ray scheduler actor: {actor}")  # 记录错误
                raise RuntimeError(f"Scheduler actor failed to initialize: {e}")  # 抛出运行时错误

            event_loop_refs = [
                actor.run_event_loop.remote() for actor in scheduler_actors
            ]  # 启动所有Actor的事件循环

            def wait_for_completion():  # 等待完成函数
                try:  # 尝试等待
                    ray.get(event_loop_refs)  # 等待事件循环结束
                except Exception as e:  # 等待失败
                    logger.error(f"Ray scheduler actor terminated with error: {e}")  # 记录错误

            return (
                RaySchedulerInitResult(
                    scheduler_infos=scheduler_infos,
                    wait_for_completion=wait_for_completion,
                    scheduler_actors=scheduler_actors,
                ),
                None,
            )  # 返回初始化结果和None
        else:  # DP模式
            # Launch the data parallel controller  # 启动数据并行控制器
            return (
                cls._launch_dp_scheduler_processes(
                    server_args,
                    port_args,
                    pg,
                    bundle_for_node,
                    rank0_node_ip,
                ),
                None,
            )  # 返回DP调度器初始化结果和None

    @classmethod
    def _launch_dp_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        pg,
        bundle_for_node: Optional[List[int]],
        rank0_node_ip: str,
    ) -> RaySchedulerInitResult:  # 通过RayDataParallelController启动DP调度器
        """Launch DP schedulers via RayDataParallelController."""  # 通过RayDataParallelController启动DP调度器
        from sglang.srt.ray.data_parallel_controller import (
            RayDataParallelController,
        )  # 导入Ray数据并行控制器

        if server_args.enable_dp_attention:  # 如果启用DP注意力
            # DP attention folds DP into TP — total GPUs = tp_size * pp_size  # DP注意力将DP折叠到TP中——总GPU数 = tp_size * pp_size
            total_gpus = server_args.tp_size * server_args.pp_size  # 计算总GPU数
        else:  # 常规DP模式
            total_gpus = server_args.dp_size * server_args.tp_size * server_args.pp_size  # 计算总GPU数
        gpus_per_node = total_gpus // server_args.nnodes  # 每节点GPU数
        logger.info(
            f"Ray DP cluster: {server_args.nnodes} nodes, "
            f"{gpus_per_node} GPUs/node, dp_size={server_args.dp_size}, "
            f"tp_size={server_args.tp_size}, pp_size={server_args.pp_size}, "
            f"enable_dp_attention={server_args.enable_dp_attention}"
        )  # 记录集群信息

        # Set dist_init_addr on server_args so PortArgs.init_new() can compute  # 在server_args上设置dist_init_addr以便PortArgs.init_new()计算
        # TCP addresses correctly (required for DP attention path).  # 正确的TCP地址（DP注意力路径需要）
        dp_server_args = dataclasses.replace(
            server_args,
            dist_init_addr=f"{rank0_node_ip}:{port_args.nccl_port}",
        )  # 创建DP模式的服务器参数
        # dataclasses.replace only copies declared fields; placement_group is  # dataclasses.replace仅复制声明字段；placement_group是
        # a dynamic attribute that must be manually appended after the rebuild.  # 动态属性，必须在重建后手动追加
        dp_server_args.placement_group = server_args.placement_group  # 追加放置组属性

        # Create the DP controller in-process. This blocks until all actors  # 在进程内创建DP控制器，阻塞直到所有Actor
        # are initialized and their event loops have started.  # 初始化完成且事件循环已启动
        controller = RayDataParallelController(
            dp_server_args, port_args, pg, bundle_for_node, rank0_node_ip
        )  # 创建DP控制器

        # Start the DP controller's event loop in a daemon thread.  # 在守护线程中启动DP控制器的事件循环
        # It routes requests from the tokenizer to per-DP-rank schedulers.  # 它将请求从分词器路由到每DP rank的调度器
        dp_thread = threading.Thread(
            target=controller.event_loop, daemon=True, name="dp_controller"
        )  # 创建守护线程
        dp_thread.start()  # 启动线程

        scheduler_infos = [
            {
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        ]  # 构建调度器信息

        event_loop_refs = controller.event_loop_refs  # 获取事件循环引用

        def wait_for_completion():  # 等待完成函数
            try:  # 尝试等待
                ray.get(event_loop_refs)  # 等待事件循环结束
            except Exception as e:  # 等待失败
                logger.error(f"Ray scheduler actor terminated with error: {e}")  # 记录错误

        return RaySchedulerInitResult(
            scheduler_infos=scheduler_infos,
            wait_for_completion=wait_for_completion,
            scheduler_actors=controller.scheduler_actors,
        )  # 返回初始化结果
