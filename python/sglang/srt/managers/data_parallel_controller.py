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
# 数据并行控制器：负责将请求分发到多个数据并行工作进程，支持多种负载均衡策略
# 包括轮询、跟随引导房间、按请求数和按token数的负载均衡方式。
# 同时支持DP Attention模式下的调度器启动和跨节点端口广播。
"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# 调度器进程PID参数键名
SCHEDULER_PIDS_ARG = "scheduler_pids"


class LoadBalanceMethod(Enum):
    """Load balance method."""
    # 负载均衡方法枚举

    ROUND_ROBIN = auto()
    # 轮询调度：依次将请求分配给各工作进程

    FOLLOW_BOOTSTRAP_ROOM = auto()
    # 跟随引导房间：根据请求的bootstrap_room字段取模分配

    TOTAL_REQUESTS = auto()
    # 按总请求数调度：将请求分配给当前请求数最少的工作进程

    TOTAL_TOKENS = auto()
    # 按总token数调度：将请求分配给当前token负载最低的工作进程

    @classmethod
    def from_str(cls, method: str):
        # 从字符串解析负载均衡方法
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    # 数据并行预算管理器：跟踪各DP rank的请求和token负载，用于负载均衡决策

    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        # 各DP rank的运行中请求数
        self.total_requests = [0] * dp_size
        # 各DP rank的总token数
        self.total_tokens = [0] * dp_size
        # 各DP rank上次快照时间戳，用于跳过过期数据
        self.last_timestamp = [0.0] * dp_size

    def update_budget(self, loads):
        """Update budget from shm snapshots, skipping stale reads."""
        # 从共享内存快照更新预算，跳过过期的读取
        for load in loads:
            if load.timestamp == self.last_timestamp[load.dp_rank]:
                continue
            self.last_timestamp[load.dp_rank] = load.timestamp
            self.total_requests[load.dp_rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self.total_tokens[load.dp_rank] = load.num_total_tokens

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        # 根据负载均衡方法选择目标DP rank
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            # 选择总请求数最少的rank
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            # 选择总token数最少的rank，token数相同时以请求数作为打破平局的依据
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        # 启发式地将目标worker的负载加1，避免突发请求全部落到同一rank
        self.total_requests[target_rank] += 1
        self.total_tokens[target_rank] += estimated_tokens
        return target_rank


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""
    # 数据并行控制器：核心调度器，将请求分发到多个数据并行工作进程

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        # 解析服务参数
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # Init inter-process communication
        # 初始化进程间通信（ZMQ上下文）
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            # 只有0号节点接收来自tokenizer的请求
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        # 分发方法：根据负载均衡策略选择对应的调度函数
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        # 仅TOTAL_REQUESTS和TOTAL_TOKENS策略需要在每次分发时刷新负载预算
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
        )

        # Load balance budget
        # 负载均衡预算初始化
        self.dp_budget = DPBudget(server_args.dp_size)
        self.load_snapshot_reader = create_load_snapshot_reader(
            server_args,
            port_args,
            caller="dp_controller",
        )
        self._last_refresh_time = 0.0

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        # 环境变量锁，保护修改CUDA_VISIBLE_DEVICES时的线程安全
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        # 启动数据并行工作进程
        self.scheduler_procs = []
        # 每个DP rank对应的ZMQ发送socket
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        # 各DP rank的存活状态
        self.status: List[bool] = [True] * server_args.dp_size

        if server_args.enable_dp_attention:
            # 启用DP Attention模式时的调度器启动
            self.launch_dp_attention_schedulers(server_args, port_args)
            # When local control broadcast is enabled, send control messages to
            # every DP group leader (attn_tp_rank=0) so each leader broadcasts
            # within its own attn_tp_group instead of the full tp_group.
            # Otherwise fall back to the original behaviour: send to only the
            # first leader, which then broadcasts over the full tp_group.
            # 启用本地控制广播时，控制消息步长为1（发给每个DP组leader），
            # 否则步长为tp_size（只发给第一个leader，由其广播）
            local_ctrl = server_args.enable_dp_attention_local_control_broadcast
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            # 普通DP模式启动调度器
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        # 初始化请求分发器
        self.init_dispatcher()

        # 创建软看门狗，用于检测控制器是否卡死
        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        # 向所有存活的工作进程广播消息
        for i, worker in enumerate(self.workers):
            if self.status[i]:
                worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        # 向每个TP组的第一个worker发送控制消息（由其广播到组内其他worker）
        for worker in self.workers[:: self.control_message_step]:
            worker.send_pyobj(obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        # 更新各DP rank的存活状态
        self.status = ranks.status

    def refresh_load_budget(self):
        # Throttle to at most once per 20ms.  When a burst of requests
        # arrives, dispatching_with_trace() calls this before every
        # dispatch.  Each call reads the latest scheduler snapshot and
        # overwrites the speculative +1 increments that DPBudget.dispatch()
        # added for previously dispatched requests in this burst.  Without
        # throttling, the budget resets to the (stale) scheduler-reported
        # value on every request, causing the entire burst to land on a
        # single DP rank.  The 20ms interval lets the burst complete
        # using speculative counters, then refreshes from the real
        # scheduler load for the next batch.
        # 限流：最多每20ms刷新一次负载预算，防止突发请求全部落到同一DP rank
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.02:
            return
        self._last_refresh_time = now
        # 从共享内存读取所有调度器的最新负载快照
        self.dp_budget.update_budget(self.load_snapshot_reader.read_all())

    def dispatching_with_trace(self, req: Req, refresh_load_budget: bool = True):
        # 带追踪信息的请求分发：记录分发时间并执行调度
        if refresh_load_budget and self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()

        req.time_stats = DPControllerReqTimeStats.new_from_obj(req.time_stats)
        # 记录DP分发开始时间
        req.time_stats.set_dp_dispatch_time()
        self.dispatching(req)
        # 记录DP分发结束时间
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        # 批量生成请求分发：先刷新负载预算，再逐个分发请求
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        # 批量嵌入请求分发：先刷新负载预算，再逐个分发请求
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def init_dispatcher(self):
        # 初始化基于类型的请求分发器，将不同类型的请求路由到对应的处理函数
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )
        # 未匹配的请求类型回退到控制消息发送
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        # 启动普通数据并行模式的调度器进程（非DP Attention模式）
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            # 为每个DP rank初始化端口参数
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            tmp_port_args.instance_id = port_args.instance_id

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            # 预先绑定端口，防止下一个DP worker获取到相同端口
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            # 为每个DP worker创建启动线程
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            # 计算下一个DP rank的起始GPU ID
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                # 0号节点创建到每个DP worker的ZMQ发送socket
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        # 在启动TP worker线程之前释放所有已绑定的端口socket
        for sock in sockets:
            sock.close()

        # Start all threads
        # 启动所有TP组启动线程，并等待就绪
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        # 在线程中启动张量并行组，设置就绪事件后保持线程存活
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        # 线程不能退出，否则scheduler中的kill_itself_when_parent_died会杀死调度器进程
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        # 从节点0向所有其他节点广播worker端口，确保多节点间端口一致
        # Determine the endpoint for inter-node communication
        # 确定节点间通信的端点地址
        if server_args.dist_init_addr is None:
            na = NetworkAddress(
                server_args.host or "127.0.0.1",
                server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(server_args.dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            # 节点0作为服务端广播worker端口
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            # 其他节点作为客户端接收worker端口
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        # 作为服务端广播worker端口给所有客户端节点
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                # 等待客户端握手
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                # 向客户端发送worker端口列表
                rep_socket.send_pyobj(worker_ports)
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            if self.server_args.elastic_ep_backend is None:
                # 非弹性EP模式：广播完成后关闭socket
                rep_socket.close()
            else:
                # 弹性EP模式：启动后台线程持续响应新加入节点的端口请求
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: List[int]):
        """
        Runs as a background thread to broadcast worker ports for recovered EP ranks
        """
        # 后台线程：为恢复的EP rank持续提供worker端口广播服务
        while True:
            # Wait for client handshake
            try:
                client_rank = rep_socket.recv().decode()
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            # Send worker ports to client
            rep_socket.send_pyobj(worker_ports)
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        # 作为客户端从节点0接收worker端口
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            # 发送包含本节点rank的握手消息
            req_socket.send(str(node_rank).encode())

            # Receive worker ports
            # 接收worker端口列表
            worker_ports = req_socket.recv_pyobj()
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            # 接收超时：10分钟内未收到节点0的响应
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        # 启动DP Attention模式的调度器，支持多节点间的端口协调
        if server_args.dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(server_args.dist_init_addr).host

        # Pre-allocate worker ports on node 0 to avoid conflicts
        # 在节点0上预分配worker端口，避免端口冲突
        worker_ports = []
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                worker_port, worker_socket = get_zmq_socket_on_host(
                    self.context, zmq.PUSH, host=bind_host
                )
                worker_ports.append(worker_port)
                self.workers[dp_rank] = worker_socket
                logger.debug(
                    "Assigned port %s to worker %s on host %s",
                    worker_port,
                    dp_rank,
                    bind_host,
                )

        # 广播端口到所有节点，然后启动TP组
        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        # 启动张量并行组：为每个PP rank和TP rank组合创建调度器进程
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        # 计算当前节点负责的PP rank范围
        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        # 计算当前节点负责的TP rank范围
        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        attn_cp_rank = 0
        moe_dp_rank = 0
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    # DP Attention有不同的分片逻辑
                    _, _, dp_rank, _ = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    # 为当前dp rank计算ZMQ端口
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    # 数据并行复用张量并行组，因此所有dp rank使用相同的nccl端口
                    rank_port_args.nccl_port = port_args.nccl_port
                    rank_port_args.instance_id = port_args.instance_id

                # 创建父子进程通信管道
                reader, writer = mp.Pipe(duplex=False)
                # 计算当前rank对应的GPU ID
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                # DP Attention模式下attn_dp_size等于dp_size，否则为1
                attn_dp_size = (
                    server_args.dp_size if server_args.enable_dp_attention else 1
                )

                # Parallelism hierarchy (outermost to innermost):
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                # 并行层次结构（从外到内）：
                # - Attention: 全局TP -> DP -> ATTN_CP -> ATTN_TP（最内层）
                # - MoE: 全局TP -> MOE_DP -> EP -> MOE_TP（最内层）
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                # 计算attention CP rank
                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
                # 计算MoE DP rank
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                # 计算MoE EP rank
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                # 在环境锁和设备ID重索引的保护下启动调度器子进程
                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                        ),
                    )
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        # 等待所有调度器进程完成模型加载
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        # 从第一个调度器获取最大总token数和最大请求输入长度
        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        # 检查请求是否已有外部指定的DP rank路由，如果有则直接发送到该rank
        if req.routed_dp_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.routed_dp_rank}")
            self.workers[req.routed_dp_rank].send_pyobj(req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        # 轮询调度器：按顺序将请求分配给各工作进程，跳过不可用的rank
        if self.maybe_external_dp_rank_routing(req):
            return

        while True:
            if self.status[self.round_robin_counter]:
                logger.debug(f"Choose worker {self.round_robin_counter}")
                self.workers[self.round_robin_counter].send_pyobj(req)
                # 移动到下一个worker
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
                break
            # 跳过不可用的worker
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def follow_bootstrap_room_scheduler(self, req: Req):
        # 跟随引导房间调度器：根据请求的bootstrap_room字段取模分配到worker
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        # 根据bootstrap_room取模确定目标rank
        target_rank = req.bootstrap_room % len(self.workers)
        self.workers[target_rank].send_pyobj(req)

    def total_requests_scheduler(self, req: Req):
        # 按总请求数调度：将请求分配给当前请求数最少的worker
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.workers[target_worker].send_pyobj(req)

    def total_tokens_scheduler(self, req: Req):
        # 按总token数调度：将请求分配给当前token负载最低的worker
        if self.maybe_external_dp_rank_routing(req):
            return
        # 估算当前请求的token数，用于负载均衡决策
        estimated_tokens = len(req.input_ids)
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
        )
        self.workers[target_worker].send_pyobj(req)

    def event_loop(self):
        # 事件循环：持续从tokenizer接收请求并分发到对应的worker
        while True:
            while True:
                # 喂看门狗，防止超时
                self.soft_watchdog.feed()
                try:
                    # 非阻塞方式接收请求
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                # 根据请求类型分发到对应的处理函数
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    # 数据并行控制器进程的入口函数：初始化控制器并运行事件循环
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    # 获取父进程引用，用于异常时发送信号
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        # 初始化分布式追踪
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        # 创建数据并行控制器实例
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        # 收集所有调度器进程的PID
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        # 通过管道向父进程报告就绪状态和配置信息
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        if server_args.node_rank == 0:
            # 只有0号节点运行事件循环接收请求
            controller.event_loop()
        for proc in controller.scheduler_procs:
            # 等待所有调度器进程结束
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        # 异常时获取完整traceback并通知父进程
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
