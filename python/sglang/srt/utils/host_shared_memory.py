# 主机共享内存管理模块
# 提供跨进程的共享内存分配与管理功能，支持CUDA主机内存注册
# 用于多GPU进程间的数据共享，如模型参数的共享存储
import logging  # 导入日志模块
from dataclasses import dataclass  # 导入数据类装饰器
from multiprocessing import shared_memory  # 导入共享内存模块
from pathlib import Path  # 导入路径处理模块
from typing import List, Optional  # 导入类型提示

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch

from sglang.srt.distributed.naive_distributed import get_naive_distributed  # 导入分布式通信工具
from sglang.srt.utils import check_cuda_result  # 导入CUDA结果检查工具

logger = logging.getLogger(__name__)  # 创建日志记录器


class HostSharedMemoryManager:  # 主机共享内存管理器，用于分配和管理跨进程共享内存
    def __init__(self, base_name: str):  # 初始化共享内存管理器，接收基础名称
        self._base_name = Path(base_name)  # 保存基础名称路径
        self._operation_index = 0  # 操作索引计数器
        self._records: List[_Record] = []  # 存储共享内存记录的列表

    def malloc(self, *, shape, dtype):  # 分配指定形状和数据类型的共享内存张量
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")  # 创建元设备上的空张量以计算字节数
        raw = self._malloc_raw(num_bytes=meta_tensor.nbytes)  # 分配原始共享内存
        return raw.view(dtype).view(*shape)  # 将原始内存视图转换为指定形状和张量类型

    def _malloc_raw(self, *, num_bytes: int) -> torch.Tensor:  # 分配指定字节数的原始共享内存
        import cuda.bindings.runtime as cuda_rt  # 导入CUDA运行时绑定

        self._operation_index += 1  # 递增操作索引
        shm_name = f"{self._base_name}_op{self._operation_index}"  # 生成共享内存名称

        # TODO handle dispose  # TODO: 处理共享内存的释放
        if get_naive_distributed().get_rank() == 0:  # 如果是0号进程
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)  # 创建共享内存

        get_naive_distributed().barrier()  # 进程同步屏障

        if get_naive_distributed().get_rank() != 0:  # 如果不是0号进程
            shm = shared_memory.SharedMemory(name=shm_name)  # 打开已创建的共享内存

        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)  # 将共享内存缓冲区映射为NumPy数组
        tensor = torch.from_numpy(np_array)  # 将NumPy数组转换为PyTorch张量

        check_cuda_result(  # 检查CUDA操作结果
            cuda_rt.cudaHostRegister(  # 注册主机内存到CUDA
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable  # 使用可移植标志注册
            )
        )

        get_naive_distributed().barrier()  # 进程同步屏障

        self._records.append(  # 记录共享内存信息
            _Record(
                shm=shm,  # 共享内存对象
                np_array=np_array,  # NumPy数组
                tensor=tensor,  # PyTorch张量
            )
        )
        return tensor  # 返回共享内存张量


@dataclass
class _Record:  # 共享内存记录数据类，保存共享内存相关信息
    shm: shared_memory.SharedMemory  # 共享内存对象
    np_array: np.ndarray  # NumPy数组
    tensor: torch.Tensor  # PyTorch张量


# Can have multi instances if needed  # 如果需要可以创建多个实例
_instance: Optional[HostSharedMemoryManager] = None  # 全局单例实例


def get_host_shared_memory_manager():  # 获取全局共享内存管理器实例
    assert _instance is not None  # 断言实例已初始化
    return _instance


def set_host_shared_memory_manager(instance: HostSharedMemoryManager):  # 设置全局共享内存管理器实例
    global _instance  # 声明使用全局变量
    assert _instance is None  # 断言实例尚未初始化
    _instance = instance  # 设置实例
