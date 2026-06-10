# NUMA（非统一内存访问）工具模块
# 提供NUMA节点查询、绑定和子进程配置功能
# 支持GPU与NUMA节点的亲和性自动配置，优化多GPU系统的内存访问性能
import ctypes  # 导入C类型库
import glob  # 导入文件模式匹配模块
import logging  # 导入日志模块
import math  # 导入数学模块
import multiprocessing  # 导入多进程模块
import os  # 导入操作系统模块
import random  # 导入随机数模块
import shutil  # 导入文件操作模块
import time  # 导入时间模块
from contextlib import contextmanager  # 导入上下文管理器装饰器
from pathlib import Path  # 导入路径处理模块
from typing import Optional  # 导入类型提示

import psutil  # 导入系统工具库
import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类
from sglang.srt.utils import is_cuda  # 导入CUDA检测函数

_is_cuda = is_cuda()  # 检测是否为CUDA环境

logger = logging.getLogger(__name__)  # 创建日志记录器


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):  # 配置子进程的NUMA绑定
    if envs.SGLANG_NUMA_BIND_V2.get():  # 如果启用了V2版NUMA绑定
        numa_node = get_numa_node_if_available(server_args, gpu_id)  # 获取GPU对应的NUMA节点
        if numa_node is not None:  # 如果找到了NUMA节点
            numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"  # 构建numactl参数
            executable, debug_str = _create_numactl_executable(  # 创建numactl包装脚本
                numactl_args=numactl_args
            )
            debug_str += (  # 添加调试信息
                f", logical_gpu_id={gpu_id}, "
                f"physical_gpu_id={_get_nvml_device_index(gpu_id)}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
            )
            with _mp_set_executable(executable=executable, debug_str=debug_str):  # 临时设置子进程可执行文件
                yield  # 执行上下文中的代码
                return
    yield  # 如果不需要NUMA绑定，直接执行


def _create_numactl_executable(numactl_args: str):  # 创建一个使用numactl的shell脚本包装器
    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())  # 获取当前Python解释器路径
    script = f'''#!/bin/sh
exec numactl {numactl_args} {old_executable} "$@"'''  # 创建使用numactl的脚本
    path = Path(  # 生成临时文件路径
        f"/tmp/sglang_temp_file_{time.time()}_{random.randrange(0, 10000000)}.sh"
    )
    path.write_text(script)  # 写入脚本内容
    path.chmod(0o777)  # 设置可执行权限
    return str(path), f"{script=}"  # 返回路径和调试信息


@contextmanager
def _mp_set_executable(executable: str, debug_str: str):  # 临时设置多进程的子进程可执行文件
    start_method = multiprocessing.get_start_method()  # 获取当前多进程启动方式
    assert start_method == "spawn", f"{start_method=}"  # 断言使用spawn方式

    old_executable = os.fsdecode(multiprocessing.spawn.get_executable())  # 保存原始可执行文件路径
    multiprocessing.spawn.set_executable(executable)  # 设置新的可执行文件
    logger.debug(f"mp.set_executable {old_executable} -> {executable} ({debug_str})")  # 记录调试日志
    try:
        yield  # 执行上下文中的代码
    finally:
        assert (  # 断言可执行文件未被其他代码修改
            os.fsdecode(multiprocessing.spawn.get_executable()) == executable
        ), f"{multiprocessing.spawn.get_executable()=}"
        multiprocessing.spawn.set_executable(old_executable)  # 恢复原始可执行文件
        logger.debug(f"mp.set_executable revert to {old_executable}")  # 记录恢复日志


def _get_nvml_device_index(device_id: int) -> int:  # 获取CUDA逻辑设备ID对应的NVML物理设备索引
    # _get_nvml_device_index is an internal PyTorch helper, so fall back to
    # device_id directly if the helper is unavailable.
    get_nvml_device_index = getattr(torch.cuda, "_get_nvml_device_index", None)  # 尝试获取PyTorch内部函数
    if get_nvml_device_index is None:  # 如果不可用
        logger.warning(
            "torch.cuda._get_nvml_device_index is unavailable; falling back to "
            f"device_id={device_id} as the NVML device index. This may select "
            "the wrong physical GPU when CUDA_VISIBLE_DEVICES reorders devices "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})."
        )
        return device_id  # 回退到使用逻辑设备ID
    return get_nvml_device_index(device_id)  # 返回NVML设备索引


def get_numa_node_if_available(server_args: ServerArgs, gpu_id: int) -> Optional[int]:  # 获取GPU对应的NUMA节点
    """
    Returns the NUMA node for the given GPU id. If it is not set in the server_args, it will try to query the NUMA node for the GPU.
    If the NUMA node is not available, has already been configured externally, or the user lacks permission to set NUMA affinity, it will return None.

    Args:
        server_args: The server arguments.
        gpu_id: The GPU id.

    Returns:
        The NUMA node for the given GPU id or None if it is not available.
    """
    if server_args.numa_node is not None:  # 如果服务器参数中指定了NUMA节点
        return server_args.numa_node[gpu_id]  # 返回指定的NUMA节点
    if _is_numa_available():  # 如果NUMA可用
        queried_numa_node = _query_numa_node_for_gpu(gpu_id)  # 查询GPU的NUMA节点
        if len(queried_numa_node) == 0:  # 如果没有查到
            return None  # 返回None
        if len(queried_numa_node) > 1:  # 如果有多个NUMA节点
            # get_numa_node_for_gpu could return multiple nodes, we use the first one for now.
            # I don't think there any hardware configs that would have more than one.
            logger.warning(
                f"Multiple NUMA nodes found for GPU {gpu_id}: {queried_numa_node}. Using the first one."
            )
        return queried_numa_node[0]  # 返回第一个NUMA节点
    return None  # NUMA不可用，返回None


def get_libnuma():  # 加载libnuma共享库
    libnuma = None  # 初始化为None

    for libnuma_so in ["libnuma.so", "libnuma.so.1"]:  # 遍历可能的库文件名
        try:
            libnuma = ctypes.CDLL(libnuma_so)  # 尝试加载共享库
        except OSError as e:  # 如果加载失败
            logger.debug(f"{e}")  # 记录调试日志
            libnuma = None  # 重置为None
        if libnuma is not None:  # 如果加载成功
            break  # 退出循环
    return libnuma  # 返回libnuma对象


def numa_bind_to_node(node: int):  # 将当前进程绑定到指定的NUMA节点
    libnuma = get_libnuma()  # 获取libnuma库

    if libnuma is None or libnuma.numa_available() < 0:  # 如果NUMA不可用
        logger.warning("numa not available on this system, skip bind action")  # 记录警告
    else:
        libnuma.numa_run_on_node(ctypes.c_int(node))  # 绑定CPU到指定NUMA节点
        libnuma.numa_set_preferred(ctypes.c_int(node))  # 设置内存分配优先节点


def _can_set_mempolicy() -> bool:  # 检查进程是否有权限使用NUMA内存策略系统调用
    """Check if the process has permission to use NUMA memory policy syscalls."""
    try:
        libnuma = get_libnuma()  # 获取libnuma库
        if libnuma is None or libnuma.numa_available() < 0:  # 如果NUMA不可用
            return False
        mode = ctypes.c_int()  # 创建C整数变量
        ret = libnuma.get_mempolicy(  # 调用get_mempolicy系统调用
            ctypes.byref(mode), None, ctypes.c_ulong(0), None, ctypes.c_ulong(0)
        )
        return ret == 0  # 返回调用是否成功
    except Exception:
        return False  # 出现异常，返回False


def _is_numa_available() -> bool:  # 检查NUMA是否可用且未被外部配置
    """
    Check if NUMA is available and not already configured externally.
    """
    if not _is_cuda:  # 如果不是CUDA环境
        return False

    # Check if this is a numa system.  # 检查是否是NUMA系统
    if not os.path.isdir("/sys/devices/system/node/node1"):  # 如果不存在node1目录
        return False  # 不是多节点NUMA系统

    # Check if affinity is already constrained  # 检查CPU亲和性是否已被限制
    pid = os.getpid()  # 获取当前进程ID
    process = psutil.Process(pid)  # 获取进程对象
    cpu_affinity = process.cpu_affinity()  # 获取当前CPU亲和性列表
    all_cpus = list(range(psutil.cpu_count()))  # 获取所有CPU核心列表
    constrained_affinity = cpu_affinity != all_cpus  # 判断亲和性是否被限制
    if constrained_affinity:  # 如果已被限制
        logger.warning(
            "NUMA affinity is already constrained for process, skipping NUMA node configuration for GPU. Remove your constraints to allow automatic configuration."
        )
        return False

    if not shutil.which("numactl") and envs.SGLANG_NUMA_BIND_V2.get():  # 如果没有numactl命令且启用了V2绑定
        logger.debug(
            "numactl command not found, skipping NUMA node configuration for GPU. Install numactl (e.g., apt-get install numactl) to enable automatic NUMA binding."
        )
        return False

    if not _can_set_mempolicy():  # 如果没有设置内存策略的权限
        logger.warning(
            "User lacks permission to set NUMA affinity, skipping NUMA node configuration for GPU. If using docker, try adding --cap-add SYS_NICE to your docker run command."
        )
        return False

    return True  # NUMA可用


def _query_numa_node_for_gpu(device_id: int):  # 查询GPU设备的NUMA节点亲和性列表
    """
    Get the NUMA node affinity list for a GPU device.

    Args:
        device_id: CUDA logical device index (post-CUDA_VISIBLE_DEVICES).
    Returns:
        List of NUMA node IDs that have affinity with the device.
    """
    try:
        import pynvml  # 导入NVML库
    except ModuleNotFoundError:
        logger.warning("pynvml not installed, skipping NUMA node configuration for GPU")  # 记录警告
        return []  # 返回空列表

    try:
        pynvml.nvmlInit()  # 初始化NVML

        # device_id is a CUDA logical index. Convert it to the corresponding
        # NVML index so reordered CUDA_VISIBLE_DEVICES maps to the right GPU.
        # _get_nvml_device_index takes CUDA_VISIBLE_DEVICES into account.
        nvml_device_id = _get_nvml_device_index(device_id)  # 转换为NVML设备索引
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_id)  # 获取设备句柄
        numa_node_count = len(glob.glob("/sys/devices/system/node/node[0-9]*"))  # 计算NUMA节点数量

        c_ulong_bits = ctypes.sizeof(ctypes.c_ulong) * 8  # 计算c_ulong的位数
        node_set_size = max(1, math.ceil(numa_node_count / c_ulong_bits))  # 计算位掩码数组大小
        node_set = pynvml.nvmlDeviceGetMemoryAffinity(  # 获取GPU内存亲和性位掩码
            handle,
            node_set_size,
            pynvml.NVML_AFFINITY_SCOPE_NODE,
        )

        # Decode the bitmask into a list of NUMA node IDs  # 将位掩码解码为NUMA节点ID列表
        numa_nodes = []  # 初始化节点列表
        for node_id in range(numa_node_count):  # 遍历每个可能的节点ID
            mask_array_index = node_id // c_ulong_bits  # 计算位掩码数组索引
            mask_bit_index = node_id % c_ulong_bits  # 计算位索引
            if node_set[mask_array_index] & (1 << mask_bit_index):  # 检查对应位是否设置
                numa_nodes.append(node_id)  # 添加到列表
        return numa_nodes  # 返回NUMA节点列表
    except pynvml.NVMLError as e:  # 如果NVML操作出错
        logger.warning(
            f"NVML error querying memory affinity for GPU {device_id}: {e}, skipping NUMA node configuration for GPU"
        )
        return []  # 返回空列表
    finally:
        try:
            pynvml.nvmlShutdown()  # 关闭NVML
        except Exception:
            pass  # Ignore shutdown errors  # 忽略关闭错误
