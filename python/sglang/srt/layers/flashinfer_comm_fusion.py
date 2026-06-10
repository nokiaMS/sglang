# FlashInfer 通信融合实现，提供 AllReduce + 残差 + RMSNorm 的融合操作，
# 包含工作区管理、预检检查、平台兼容性处理等功能。

import contextlib  # 导入上下文工具
import inspect  # 导入检查模块
import logging  # 导入日志模块
import platform  # 导入平台检测模块
from typing import Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch

from sglang.srt.distributed import (  # 导入分布式通信相关函数
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_moe_tp_group,
    get_tp_group,
)
from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.utils import (  # 导入工具函数
    ceil_align,
    get_cuda_driver_bindings,
    is_flashinfer_available,
)
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册

logger = logging.getLogger(__name__)  # 获取日志记录器

_flashinfer_comm = None  # FlashInfer 通信模块引用
_TorchDistBackend = None  # TorchDistBackend 类引用
_flashinfer_allreduce_unavailable = False  # FlashInfer AllReduce 是否不可用
_flashinfer_create_workspace_supports_group = False  # 创建工作区是否支持 group 参数
_flashinfer_create_workspace_supports_comm_backend = False  # 创建工作区是否支持 comm_backend 参数
_flashinfer_allreduce_supports_trigger_completion = False  # AllReduce 是否支持 trigger_completion_at_end 参数
_posix_transport_override_logged = False  # POSIX 传输覆盖日志是否已记录


def _should_force_posix_fd_transport() -> bool:  # 判断是否应强制使用 POSIX FD 传输
    force_posix_env = envs.SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT.get()  # 获取环境变量
    if force_posix_env is not None:  # 如果设置了环境变量
        return force_posix_env  # 返回环境变量值

    machine = platform.machine().lower()  # 获取机器架构
    if machine not in ("aarch64", "arm64"):  # 如果不是 ARM 架构
        return False  # 不强制

    if not torch.cuda.is_available():  # 如果 CUDA 不可用
        return False  # 不强制

    try:
        major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())  # 获取 GPU 计算能力
    except Exception as e:
        logger.debug("Failed to get CUDA device capability: %s", e)  # 记录调试日志
        return False  # 不强制

    return major == 10  # Blackwell (sm_10x) 平台需要强制 POSIX FD


@contextlib.contextmanager
def _flashinfer_posix_fd_transport_override_if_needed():  # 必要时临时覆盖 FlashInfer 的传输方式为 POSIX FD
    # TODO(mmangkad): Remove this temporary override once the
    # FlashInfer unified allreduce-fusion transport issue on
    # GB200/GB300 platforms is fixed and verified resolved.
    # TODO(mmangkad): 一旦 FlashInfer 统一 allreduce-fusion 传输在 GB200/GB300 平台上的问题修复并验证，移除此临时覆盖。
    global _posix_transport_override_logged  # 声明全局变量

    if not _should_force_posix_fd_transport():  # 如果不需要强制 POSIX FD
        yield  # 直接继续
        return

    try:
        import flashinfer.comm.mnnvl as flashinfer_mnnvl  # 尝试导入 FlashInfer MNNVL 通信模块
    except Exception as e:
        logger.debug(
            "Failed to import flashinfer.comm.mnnvl for transport override: %s", e
        )  # 记录调试日志
        yield  # 导入失败，直接继续
        return

    original_checker = getattr(flashinfer_mnnvl, "is_mnnvl_fabric_supported", None)  # 保存原始检查函数
    if original_checker is None:  # 如果没有检查函数
        yield  # 直接继续
        return

    if not _posix_transport_override_logged:  # 如果尚未记录日志
        logger.warning(
            "Applying FlashInfer transport workaround: forcing PosixFD "
            "symmetric-memory handle exchange on aarch64 + sm10x to avoid "
            "known data corruption with Fabric handle exchange on GB systems. "
            "Set SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT=0 to disable."
        )  # 记录警告日志
        _posix_transport_override_logged = True  # 标记已记录

    def _always_disable_fabric(_device_idx: int) -> bool:  # 始终禁用 Fabric 的替代函数
        return False

    flashinfer_mnnvl.is_mnnvl_fabric_supported = _always_disable_fabric  # 替换检查函数
    try:
        yield  # 执行上下文中的代码
    finally:
        flashinfer_mnnvl.is_mnnvl_fabric_supported = original_checker  # 恢复原始检查函数


if is_flashinfer_available():  # 如果 FlashInfer 可用
    try:
        import flashinfer.comm as comm  # 导入 FlashInfer 通信模块

        if hasattr(comm, "allreduce_fusion") and hasattr(  # 检查是否有融合 AllReduce API
            comm, "create_allreduce_fusion_workspace"
        ):
            _flashinfer_comm = comm  # 保存通信模块引用
            workspace_params = inspect.signature(  # 获取创建工作区函数签名
                comm.create_allreduce_fusion_workspace
            ).parameters
            allreduce_params = inspect.signature(comm.allreduce_fusion).parameters  # 获取 AllReduce 函数签名
            _flashinfer_create_workspace_supports_group = "group" in workspace_params  # 检查是否支持 group 参数
            _flashinfer_create_workspace_supports_comm_backend = (
                "comm_backend" in workspace_params  # 检查是否支持 comm_backend 参数
            )
            _flashinfer_allreduce_supports_trigger_completion = (
                "trigger_completion_at_end" in allreduce_params  # 检查是否支持 trigger_completion 参数
            )
        else:
            _flashinfer_allreduce_unavailable = True  # 标记不可用
            logger.warning(
                "flashinfer.comm unified allreduce_fusion API is not available, "
                "falling back to standard implementation"
            )  # 记录警告日志
    except ImportError:
        _flashinfer_allreduce_unavailable = True  # 标记不可用
        logger.warning(
            "flashinfer.comm is not available, falling back to standard "
            "implementation"
        )  # 记录警告日志

    try:
        from flashinfer.comm.mnnvl import TorchDistBackend  # 尝试导入 TorchDistBackend

        class _FixedTorchDistBackend(TorchDistBackend):  # 修复版 TorchDistBackend
            """Workaround for FlashInfer TorchDistBackend issues.

            1. bcast fix: TorchDistBackend.bcast passes the in-group rank
               directly as `src` to broadcast_object_list, which expects a
               global rank.
            2. Graph-capture fix: initialize with NCCL device_group (so
               the backend derives correct device_idx / GPU mapping), but
               broadcast via GLOO cpu_group (to avoid NCCL collectives
               that interfere with CUDA graph capture).
            """  # FlashInfer TorchDistBackend 问题的临时解决方案。1. bcast 修复：TorchDistBackend.bcast 将组内排名直接作为 src 传给 broadcast_object_list，但后者期望全局排名。2. Graph 捕获修复：用 NCCL device_group 初始化（获取正确的 device_idx/GPU 映射），但通过 GLOO cpu_group 广播（避免干扰 CUDA graph 捕获的 NCCL 集合操作）。

            def __init__(self, device_group, cpu_group):  # 初始化
                super().__init__(group=device_group)  # 用设备组初始化父类
                self._cpu_group = cpu_group  # 保存 CPU 组

            def bcast(self, data, root):  # 广播方法（修复版）
                import torch.distributed as dist  # 导入分布式模块

                group_ranks = dist.get_process_group_ranks(self._cpu_group)  # 获取组内全局排名
                global_root = group_ranks[root]  # 将组内排名转为全局排名
                object_list = [data]  # 包装数据
                dist.broadcast_object_list(
                    object_list, src=global_root, group=self._cpu_group  # 使用全局排名和 CPU 组广播
                )
                return object_list[0]  # 返回广播后的数据

        _TorchDistBackend = _FixedTorchDistBackend  # 使用修复版
    except ImportError:
        logger.debug(
            "flashinfer.comm.mnnvl.TorchDistBackend is not available, "
            "allreduce fusion will use the default process group"
        )  # 记录调试日志


def is_flashinfer_allreduce_unavailable() -> bool:  # 检查 FlashInfer AllReduce 是否不可用
    return _flashinfer_allreduce_unavailable


def _make_flashinfer_workspace_allocation_prop(cuda_driver):  # 创建 FlashInfer 工作区的内存分配属性
    if _should_force_posix_fd_transport():  # 如果需要强制 POSIX FD 传输
        handle_type = (  # 使用 POSIX 文件描述符类型
            cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
    else:
        from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported  # 导入 Fabric 支持检查

        if is_mnnvl_fabric_supported(torch.cuda.current_device()):  # 如果支持 NVLink Fabric
            handle_type = (  # 使用 Fabric 句柄类型
                cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        else:
            handle_type = (  # 使用 POSIX 文件描述符类型
                cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )

    prop = cuda_driver.CUmemAllocationProp()  # 创建分配属性
    prop.requestedHandleTypes = handle_type  # 设置句柄类型
    prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED  # 设置为固定内存
    prop.location = cuda_driver.CUmemLocation()  # 创建位置
    prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE  # 设置为设备位置
    prop.location.id = torch.cuda.current_device()  # 设置设备 ID
    prop.allocFlags.gpuDirectRDMACapable = 1  # 启用 GPU Direct RDMA
    return prop  # 返回分配属性


def _flashinfer_trtllm_workspace_allocation_sizes(  # 计算 FlashInfer TRTLLM 工作区的分配大小
    cuda_driver,
    prop,
    world_size: int,  # 世界大小
    max_token_num: int,  # 最大令牌数
    hidden_dim: int,  # 隐藏维度
    dtype: torch.dtype,  # 数据类型
) -> list[int]:
    """Mirror FlashInfer TRTLLM SymmDeviceMemory local allocation sizes."""  # 镜像 FlashInfer TRTLLM SymmDeviceMemory 本地分配大小。
    elem_size = 4 if dtype == torch.float32 else 2  # 计算每个元素的字节数
    buffer_size = world_size * max_token_num * hidden_dim * 2  # 缓冲区大小
    flag_size = world_size * 256 * 4  # 标志大小

    max_comm_size = 2147483647 & ~((1 << 21) - 1)  # 最大通信大小（2GB 对齐）
    lamport_comm_size = min(
        world_size * max_token_num * hidden_dim * elem_size,
        max_comm_size,
    )  # Lamport 通信大小
    lamport_buffer_size = lamport_comm_size * 3  # Lamport 缓冲区大小（3 倍）

    # trtllm_create_ipc_workspace_for_all_reduce_fusion rounds each logical
    # buffer to 2 MiB before passing it to SymmDeviceMemory.
    # trtllm_create_ipc_workspace_for_all_reduce_fusion 在传递给 SymmDeviceMemory 前将每个逻辑缓冲区舍入到 2 MiB。
    buffer_sizes = (
        ceil_align(size, 1 << 21)  # 对齐到 2 MiB
        for size in (buffer_size, flag_size, lamport_buffer_size)
    )

    signal_pad_size = 2048  # 信号填充大小
    allocation_sizes = []  # 分配大小列表
    for buffer_size in buffer_sizes:  # 遍历每个缓冲区大小
        err, alloc_granularity = cuda_driver.cuMemGetAllocationGranularity(  # 获取分配粒度
            prop,
            cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:  # 如果获取失败
            raise RuntimeError(
                "cuMemGetAllocationGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(buffer_size + signal_pad_size, alloc_granularity)  # 对齐分配大小

        mc_prop = cuda_driver.CUmulticastObjectProp()  # 创建多播属性
        mc_prop.numDevices = world_size  # 设置设备数
        mc_prop.size = allocation_size  # 设置大小
        mc_prop.handleTypes = prop.requestedHandleTypes  # 设置句柄类型

        err, mc_granularity = cuda_driver.cuMulticastGetGranularity(  # 获取多播粒度
            mc_prop,
            cuda_driver.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:  # 如果获取失败
            raise RuntimeError(
                "cuMulticastGetGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(allocation_size, mc_granularity)  # 按多播粒度对齐
        allocation_sizes.append(allocation_size)  # 添加到列表
    return allocation_sizes  # 返回分配大小列表


def _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop) -> bool:  # 探测 cuMemCreate 是否能成功
    handles = []  # 句柄列表
    try:
        for allocation_size in allocation_sizes:  # 遍历每个分配大小
            err, handle = cuda_driver.cuMemCreate(allocation_size, prop, 0)  # 尝试创建内存
            if err != cuda_driver.CUresult.CUDA_SUCCESS:  # 如果失败
                return False  # 返回失败
            handles.append(handle)  # 添加到句柄列表
        return True  # 全部成功
    finally:
        for handle in reversed(handles):  # 逆序释放句柄
            cuda_driver.cuMemRelease(handle)


def _preflight_check_workspace_memory(  # 预检工作区内存是否足够
    world_size: int,  # 世界大小
    max_token_num: int,  # 最大令牌数
    hidden_dim: int,  # 隐藏维度
    dtype: torch.dtype,  # 数据类型
    cpu_group: Optional["torch.distributed.ProcessGroup"] = None,  # 可选的 CPU 进程组
) -> bool:
    """Collectively decide whether to enter FlashInfer workspace creation.

    FlashInfer TRTLLM workspaces allocate several SymmDeviceMemory buffers and
    then exchange handles across ranks. If one rank fails local cuMemCreate and
    exits while peers enter handle exchange, peers can hang until the watchdog
    aborts. Probe the same handle type and allocation sequence first, then vote
    on a CPU group so all ranks proceed or skip together.
    """  # 集体决定是否进入 FlashInfer 工作区创建。FlashInfer TRTLLM 工作区分配多个 SymmDeviceMemory 缓冲区，然后跨排名交换句柄。如果某个排名的本地 cuMemCreate 失败并退出，而其他排名进入句柄交换，则其他排名可能挂起直到看门狗终止。先探测相同的句柄类型和分配序列，然后在 CPU 组上投票，使所有排名一起继续或跳过。
    import torch.distributed as dist  # 导入分布式模块

    group = cpu_group  # 使用提供的 CPU 组
    if group is None:  # 如果没有提供
        tp_group = get_tp_group()  # 获取 TP 组
        if tp_group.world_size <= 1:  # 如果只有 1 个进程
            return True  # 不需要预检
        group = tp_group.cpu_group  # 使用 TP 组的 CPU 组

    allocation_sizes = []  # 分配大小列表
    try:
        cuda_driver = get_cuda_driver_bindings()  # 获取 CUDA 驱动绑定
        prop = _make_flashinfer_workspace_allocation_prop(cuda_driver)  # 创建分配属性
        allocation_sizes = _flashinfer_trtllm_workspace_allocation_sizes(  # 计算分配大小
            cuda_driver,
            prop,
            world_size,
            max_token_num,
            hidden_dim,
            dtype,
        )
        local_ok = _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop)  # 探测本地是否可以创建
    except Exception as e:
        logger.warning(
            "FlashInfer workspace preflight probe failed (%s). "
            "Skipping allreduce fusion.",
            e,
        )  # 记录警告日志
        local_ok = False  # 本地探测失败

    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32)  # 创建投票标志
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)  # 跨排名进行 AND 投票

    logger.debug(
        "FlashInfer workspace preflight [rank %s]: probe=%.2f GB, "
        "local_probe=%s, vote=%s",
        dist.get_rank(group=group),
        sum(allocation_sizes) / 1e9,
        "OK" if local_ok else "FAIL",
        "PROCEED" if flag.item() == 1 else "SKIP",
    )  # 记录调试日志
    if flag.item() == 0:  # 如果有任何一个排名失败
        logger.warning(
            "FlashInfer workspace preflight: cuMemCreate probe failed on at "
            "least one rank. Skipping allreduce fusion to avoid cross-rank "
            "desync inside the flashinfer collective."
        )  # 记录警告日志
        return False  # 返回失败
    return True  # 返回成功


class FlashInferWorkspaceManager:  # FlashInfer 工作区管理器
    def __init__(self):  # 初始化
        self.workspace = None  # 工作区
        self.world_size = None  # 世界大小
        self.rank = None  # 排名
        self.group = None  # 进程组
        self.max_token_num = None  # 最大令牌数
        self.hidden_dim = None  # 隐藏维度
        self.dtype = None  # 数据类型
        self.initialized = False  # 是否已初始化

    def initialize(  # 初始化工作区
        self,
        world_size: int,  # 世界大小
        rank: int,  # 排名
        max_token_num: int,  # 最大令牌数
        hidden_dim: int,  # 隐藏维度
        dtype: torch.dtype,  # 数据类型
        use_oneshot: Optional[bool] = None,  # 是否使用 oneshot 模式
        device_group: Optional["torch.distributed.ProcessGroup"] = None,  # 设备进程组
        cpu_group: Optional["torch.distributed.ProcessGroup"] = None,  # CPU 进程组
    ):
        """Initialize workspace"""  # 初始化工作区
        if _flashinfer_comm is None:  # 如果 FlashInfer 通信模块不可用
            logger.warning(
                "FlashInfer comm not available, skipping workspace initialization"
            )  # 记录警告日志
            return

        self.cleanup()  # 先清理旧工作区

        global _flashinfer_allreduce_unavailable  # 声明全局变量
        if not _preflight_check_workspace_memory(  # 预检内存
            world_size=world_size,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            cpu_group=cpu_group,
        ):
            _flashinfer_allreduce_unavailable = True  # 标记不可用
            self.workspace = None  # 清空工作区
            self.initialized = False  # 标记未初始化
            return

        try:
            kwargs = dict(  # 构建参数
                backend="trtllm",  # 使用 TRTLLM 后端
                world_size=world_size,
                rank=rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                force_oneshot_support=bool(use_oneshot),  # 是否强制 oneshot 支持
            )
            create_workspace = _flashinfer_comm.create_allreduce_fusion_workspace  # 获取创建工作区函数
            if _flashinfer_create_workspace_supports_group:  # 如果支持 group 参数
                # Pin the symmetric-memory rendezvous to the actual subgroup.
                # Older FlashInfer releases only support comm_backend.
                # 将对称内存集合绑定到实际子组。旧版 FlashInfer 仅支持 comm_backend。
                kwargs["group"] = device_group
            if (  # 如果支持 comm_backend 且提供了进程组
                _TorchDistBackend is not None
                and _flashinfer_create_workspace_supports_comm_backend
                and device_group is not None
                and cpu_group is not None
            ):
                kwargs["comm_backend"] = _TorchDistBackend(  # 使用修复版后端
                    device_group=device_group, cpu_group=cpu_group
                )
            with _flashinfer_posix_fd_transport_override_if_needed():  # 必要时覆盖传输方式
                self.workspace = create_workspace(**kwargs)  # 创建工作区
        except Exception as e:
            _flashinfer_allreduce_unavailable = True  # 标记不可用
            logger.warning(
                f"Failed to initialize FlashInfer workspace: {e}. "
                "Disabling flashinfer allreduce fusion permanently."
            )  # 记录警告日志
            self.workspace = None  # 清空工作区
            self.initialized = False  # 标记未初始化
            return

        self.world_size = world_size  # 保存世界大小
        self.rank = rank  # 保存排名
        self.group = (device_group, cpu_group)  # 保存进程组
        self.max_token_num = max_token_num  # 保存最大令牌数
        self.hidden_dim = hidden_dim  # 保存隐藏维度
        self.dtype = dtype  # 保存数据类型
        self.initialized = True  # 标记已初始化

        backend = getattr(self.workspace, "backend", "unknown")  # 获取后端名称
        logger.info(
            f"FlashInfer workspace initialized for rank {rank}, "
            f"world_size {world_size}, backend {backend}, "
            f"max_token_num {max_token_num}, hidden_dim {hidden_dim}"
        )  # 记录信息日志

    def is_buffer_size_sufficient(  # 检查缓冲区大小是否足够
        self,
        token_num: int,  # 令牌数
        hidden_dim: int,  # 隐藏维度
        dtype: torch.dtype,  # 数据类型
        use_oneshot: Optional[bool] = None,  # 是否使用 oneshot 模式
    ) -> bool:
        if not self.initialized or self.workspace is None:  # 如果未初始化
            return False  # 不足够
        try:
            return self.workspace.is_buffer_size_sufficient(  # 调用工作区检查方法
                tp_size=self.world_size,
                num_tokens=token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                use_oneshot=use_oneshot,
            )
        except Exception as e:
            logger.debug(f"FlashInfer workspace size check failed: {e}")  # 记录调试日志
            return False  # 检查失败

    def cleanup(self):  # 清理工作区
        """Clean up workspace"""  # 清理工作区
        if self.workspace is not None:  # 如果工作区存在
            try:
                self.workspace.destroy()  # 销毁工作区
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")  # 记录警告日志
            finally:
                self.workspace = None  # 清空工作区
                self.initialized = False  # 标记未初始化
                self.world_size = None  # 清空世界大小
                self.rank = None  # 清空排名
                self.group = None  # 清空进程组
                self.max_token_num = None  # 清空最大令牌数
                self.hidden_dim = None  # 清空隐藏维度
                self.dtype = None  # 清空数据类型


_attn_tp_workspace_manager = FlashInferWorkspaceManager()  # 注意力 TP 工作区管理器
_moe_tp_workspace_manager = FlashInferWorkspaceManager()  # MoE TP 工作区管理器


def _get_workspace_manager(use_attn_tp_group: bool) -> FlashInferWorkspaceManager:  # 获取工作区管理器
    return (
        _attn_tp_workspace_manager if use_attn_tp_group else _moe_tp_workspace_manager  # 根据参数选择
    )


def _sync_allreduce_unavailable_across_tp():  # 跨 TP 排名同步 AllReduce 不可用状态
    """Synchronize _flashinfer_allreduce_unavailable across all TP ranks.

    If workspace initialization fails on any rank, all ranks must agree to
    disable fusion. Otherwise ranks diverge during CUDA graph capture: some
    use FlashInfer fusion (skipping custom allreduce), others fall back to
    standard allreduce (calling register_buffer collectives), causing a hang
    in register_graph_buffers.
    """  # 跨所有 TP 排名同步 _flashinfer_allreduce_unavailable。如果任何排名的工作区初始化失败，所有排名必须同意禁用融合。否则排名在 CUDA graph 捕获期间会分歧：一些使用 FlashInfer 融合（跳过自定义 allreduce），另一些回退到标准 allreduce（调用 register_buffer 集合操作），导致 register_graph_buffers 中挂起。
    global _flashinfer_allreduce_unavailable  # 声明全局变量
    try:
        import torch.distributed as dist  # 导入分布式模块

        tp_group = get_tp_group()  # 获取 TP 组
        if tp_group.world_size <= 1:  # 如果只有 1 个进程
            return  # 不需要同步
        flag = torch.tensor(
            [1 if _flashinfer_allreduce_unavailable else 0],
            dtype=torch.int32,
        )  # 创建标志
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=tp_group.cpu_group)  # 取最大值同步
        if flag.item() > 0 and not _flashinfer_allreduce_unavailable:  # 如果其他排名标记不可用
            _flashinfer_allreduce_unavailable = True  # 本地也标记不可用
            logger.warning(
                "FlashInfer allreduce fusion disabled globally because "
                "workspace initialization failed on at least one rank."
            )  # 记录警告日志
    except Exception as e:
        logger.debug(f"Failed to sync flashinfer unavailable flag: {e}")  # 记录调试日志


def ensure_workspace_initialized(  # 确保工作区已初始化
    max_token_num: int = 2048,  # 最大令牌数，默认 2048
    hidden_dim: int = 4096,  # 隐藏维度，默认 4096
    dtype: torch.dtype = torch.float16,  # 数据类型，默认 float16
    token_num: Optional[int] = None,  # 当前令牌数
    use_oneshot: Optional[bool] = None,  # 是否使用 oneshot 模式
    use_attn_tp_group: bool = True,  # 是否使用注意力 TP 组
):
    """Ensure workspace is initialized"""  # 确保工作区已初始化
    if _flashinfer_allreduce_unavailable:  # 如果 AllReduce 不可用
        return False

    if not is_flashinfer_available() or _flashinfer_comm is None:  # 如果 FlashInfer 不可用
        return False

    if use_attn_tp_group:  # 如果使用注意力 TP 组
        world_size = get_attn_tensor_model_parallel_world_size()  # 获取注意力 TP 大小
        rank = get_attn_tensor_model_parallel_rank()  # 获取注意力 TP 排名
        coordinator = get_attn_tp_group()  # 获取注意力 TP 协调器
    else:
        if get_moe_expert_parallel_world_size() > 1:  # 如果 MoE 专家并行大于 1
            world_size = get_moe_expert_parallel_world_size()  # 使用 EP 大小
            rank = get_moe_expert_parallel_rank()  # 使用 EP 排名
            coordinator = get_moe_ep_group()  # 使用 EP 组
        else:
            world_size = get_moe_tensor_parallel_world_size()  # 使用 MoE TP 大小
            rank = get_moe_tensor_parallel_rank()  # 使用 MoE TP 排名
            coordinator = get_moe_tp_group()  # 使用 MoE TP 组

    # Always pass the coordinator's groups: flashinfer >=0.6.10 reads the
    # rendezvous group from `group=...` (falling back to WORLD when None),
    # so leaving it None silently rendezvouses on WORLD and the kernel ends
    # up addressing the wrong peers in TP/EP/CP subgroup setups.
    # 始终传递协调器的组：flashinfer >=0.6.10 从 `group=...` 读取集合组（当 None 时回退到 WORLD），因此留 None 会静默地在 WORLD 上集合，内核最终在 TP/EP/CP 子组设置中寻址错误的对等方。
    device_group = coordinator.device_group  # 获取设备组
    cpu_group = coordinator.cpu_group  # 获取 CPU 组

    if world_size <= 1:  # 如果只有 1 个进程
        return False  # 不需要融合

    workspace_manager = _get_workspace_manager(use_attn_tp_group)  # 获取工作区管理器
    token_num = token_num or max_token_num  # 使用当前令牌数或最大令牌数
    group_key = (device_group, cpu_group)  # 创建组键

    if (  # 如果需要重新初始化
        not workspace_manager.initialized
        or workspace_manager.world_size != world_size
        or workspace_manager.rank != rank
        or workspace_manager.group != group_key
        or not workspace_manager.is_buffer_size_sufficient(
            token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            use_oneshot=use_oneshot,
        )
    ):
        workspace_manager.initialize(  # 重新初始化工作区
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            use_oneshot=use_oneshot,
            device_group=device_group,
            cpu_group=cpu_group,
        )

        _sync_allreduce_unavailable_across_tp()  # 同步不可用状态

    return workspace_manager.initialized  # 返回是否已初始化


def fake_flashinfer_allreduce_residual_rmsnorm(  # FlashInfer AllReduce+残差+RMSNorm 的假实现（用于 torch.compile）
    input_tensor: torch.Tensor,  # 输入张量
    residual: torch.Tensor,  # 残差张量
    weight: torch.Tensor,  # RMSNorm 权重
    eps: float = 1e-6,  # RMSNorm epsilon
    max_token_num: int = 16384,  # 最大令牌数
    use_oneshot: Optional[bool] = None,  # 是否使用 oneshot 模式
    trigger_completion_at_end: bool = False,  # 是否在结束时触发完成
    fp32_acc: bool = False,  # 是否使用 fp32 精度
    use_attn_tp_group: bool = True,  # 是否使用注意力 TP 组
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual)  # 分配残差输出
    norm_out = torch.empty_like(input_tensor)  # 分配归一化输出
    return norm_out, residual_out  # 返回空输出


@register_custom_op(  # 注册为自定义算子
    mutates_args=["input_tensor", "residual", "weight"],  # 标记被修改的参数
    fake_impl=fake_flashinfer_allreduce_residual_rmsnorm,  # 假实现
)
def flashinfer_allreduce_residual_rmsnorm(  # FlashInfer 融合 AllReduce + 残差 + RMSNorm
    input_tensor: torch.Tensor,  # 输入张量（需要 AllReduce）
    residual: torch.Tensor,  # 残差张量
    weight: torch.Tensor,  # RMSNorm 权重
    eps: float = 1e-6,  # RMSNorm epsilon
    max_token_num: int = 2048,  # 最大令牌数
    use_oneshot: Optional[bool] = None,  # 是否使用 oneshot 模式
    trigger_completion_at_end: bool = False,  # 是否在结束时触发完成
    fp32_acc: bool = False,  # 是否使用 fp32 精度
    use_attn_tp_group: bool = True,  # 是否使用注意力 TP 组
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use FlashInfer's fused allreduce + residual + RMS norm operation

    Args:
        input_tensor: Input tensor that needs allreduce
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision
        use_attn_tp_group: If True, use attention TP group; otherwise use MoE TP group

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """  # 使用 FlashInfer 的融合 AllReduce + 残差 + RMSNorm 操作。参数：input_tensor 为需要 AllReduce 的输入张量；residual 为残差张量；weight 为 RMSNorm 权重；eps 为 RMSNorm epsilon；max_token_num 为最大令牌数；use_oneshot 是否使用 oneshot 模式；trigger_completion_at_end 是否在结束时触发完成；fp32_acc 是否使用 fp32 精度；use_attn_tp_group 为 True 时使用注意力 TP 组，否则使用 MoE TP 组。返回：(归一化输出, 残差输出)。
    if not is_flashinfer_available() or _flashinfer_comm is None:  # 如果 FlashInfer 不可用
        logger.debug(
            "FlashInfer not available, falling back to standard implementation"
        )  # 记录调试日志
        return None, None  # 返回 None 表示应回退

    if use_attn_tp_group:  # 如果使用注意力 TP 组
        world_size = get_attn_tensor_model_parallel_world_size()  # 获取注意力 TP 大小
    else:
        # If MoE expert parallel world size > 1, use expert parallel group
        # Otherwise, use tensor parallel group
        # The two values cannot be larger than 1 at the same time
        # 如果 MoE 专家并行大小 > 1，使用专家并行组；否则使用张量并行组。这两个值不能同时大于 1。
        if get_moe_expert_parallel_world_size() > 1:  # 如果 MoE 专家并行大于 1
            world_size = get_moe_expert_parallel_world_size()  # 使用 EP 大小
        else:
            world_size = get_moe_tensor_parallel_world_size()  # 使用 TP 大小

    if world_size <= 1:  # 如果只有 1 个进程
        logger.debug("Single GPU, no need for allreduce fusion")  # 记录调试日志
        return None, None  # 不需要融合

    assert input_tensor.shape[0] <= max_token_num  # 断言令牌数不超过最大值
    if (
        not input_tensor.is_contiguous()
        or not residual.is_contiguous()
        or not weight.is_contiguous()
    ):  # 如果张量不连续
        logger.debug("Non-contiguous tensors, skipping FlashInfer allreduce fusion")  # 记录调试日志
        return None, None  # 跳过融合

    if not ensure_workspace_initialized(  # 确保工作区已初始化
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        dtype=input_tensor.dtype,
        token_num=input_tensor.shape[0],
        use_oneshot=use_oneshot,
        use_attn_tp_group=use_attn_tp_group,
    ):
        logger.debug("FlashInfer workspace not available")  # 记录调试日志
        return None, None  # 工作区不可用

    residual_out = torch.empty_like(residual)  # 分配残差输出
    norm_out = torch.empty_like(input_tensor)  # 分配归一化输出

    workspace_manager = _get_workspace_manager(use_attn_tp_group)  # 获取工作区管理器
    kwargs = dict(  # 构建 AllReduce 融合参数
        input=input_tensor,  # 输入张量
        workspace=workspace_manager.workspace,  # 工作区
        pattern=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,  # 融合模式：AllReduce+残差+RMSNorm
        launch_with_pdl=True,  # 启用 PDL 启动
        residual_out=residual_out,  # 残差输出
        norm_out=norm_out,  # 归一化输出
        residual_in=residual,  # 残差输入
        rms_gamma=weight,  # RMSNorm gamma 权重
        rms_eps=eps,  # RMSNorm epsilon
        use_oneshot=use_oneshot,  # oneshot 模式
        fp32_acc=fp32_acc,  # fp32 精度
    )
    if _flashinfer_allreduce_supports_trigger_completion:  # 如果支持 trigger_completion
        kwargs["trigger_completion_at_end"] = trigger_completion_at_end  # 传递参数
    _flashinfer_comm.allreduce_fusion(**kwargs)  # 执行融合 AllReduce

    return norm_out, residual_out  # 返回归一化输出和残差输出


def pre_initialize_workspaces(  # 预初始化工作区（在 CUDA Graph 捕获前调用）
    max_token_num: int,  # 最大令牌数
    hidden_dim: int,  # 隐藏维度
    dtype: torch.dtype,  # 数据类型
    use_oneshot: Optional[bool] = None,  # 是否使用 oneshot 模式
):
    """Pre-initialize flashinfer workspaces before CUDA graph capture.

    This must be called before graph capture to avoid collective operations
    (broadcasts, barriers) inside the graph capture context, which can
    deadlock with custom_all_reduce.register_graph_buffers.
    """  # 在 CUDA Graph 捕获前预初始化 FlashInfer 工作区。必须在 Graph 捕获前调用，以避免在 Graph 捕获上下文中的集合操作（广播、屏障）与 custom_all_reduce.register_graph_buffers 发生死锁。
    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:  # 如果不可用
        return  # 直接返回

    # Initialize MoE workspace
    # 初始化 MoE 工作区
    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=False,  # 使用 MoE TP 组
    )

    # Initialize attention workspace
    # 初始化注意力工作区
    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=True,  # 使用注意力 TP 组
    )


def cleanup_flashinfer_workspace():  # 清理 FlashInfer 工作区
    global _attn_tp_workspace_manager, _moe_tp_workspace_manager  # 声明全局变量
    if _attn_tp_workspace_manager is not None:  # 如果注意力 TP 工作区存在
        _attn_tp_workspace_manager.cleanup()  # 清理
    if (
        _moe_tp_workspace_manager is not None
        and _moe_tp_workspace_manager is not _attn_tp_workspace_manager
    ):  # 如果 MoE TP 工作区存在且不同于注意力 TP 工作区
        _moe_tp_workspace_manager.cleanup()  # 清理
