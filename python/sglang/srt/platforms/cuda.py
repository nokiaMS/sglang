"""CUDA device operations for the SRT platform layer."""
# 本文件实现了 SRT（SGLang Runtime）平台层的 CUDA 设备操作，
# 提供了 CUDA GPU 设备的内存查询、设备管理、分布式后端配置等功能，
# 是 SGLang 在 NVIDIA GPU 上运行的基础平台支持模块。

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


class CudaDeviceMixin(DeviceMixin):
    """CUDA implementation of the shared device operations."""
    # CUDA 设备混入类，继承自 DeviceMixin，
    # 实现了所有与 CUDA GPU 设备相关的共享操作接口。

    _enum: PlatformEnum = PlatformEnum.CUDA  # 平台枚举标识为 CUDA
    device_name: str = "cuda"  # 设备名称
    device_type: str = "cuda"  # 设备类型

    def get_device_total_memory(self, device_id: int = 0) -> int:
        # 获取指定 GPU 设备的总显存大小（字节）
        return int(torch.cuda.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        # 获取当前 PyTorch 在该设备上分配的最大内存量（字节）
        return float(torch.cuda.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        # 根据 local_rank 获取对应的 CUDA torch.device 对象
        return torch.device("cuda", local_rank)

    def set_device(self, device: "torch.device") -> None:
        # 设置当前线程使用的 CUDA 设备
        torch.cuda.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        # 获取指定 GPU 设备的型号名称，如 "NVIDIA A100-SXM4-80GB"
        return str(torch.cuda.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        # 获取指定 GPU 设备的 UUID 唯一标识符
        return str(torch.cuda.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # 获取指定 GPU 设备的计算能力版本（如 8.0、9.0 等）
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major, minor)

    def empty_cache(self) -> None:
        # 释放 PyTorch 缓存的未使用显存，交还给 GPU 可用池
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        # 同步所有 CUDA 流上的操作，等待所有 GPU 计算完成
        torch.cuda.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        # 获取指定 GPU 设备的可用显存信息
        # 返回元组：(空闲显存字节数, 总显存字节数)
        return torch.cuda.mem_get_info(device_id)

    def get_torch_distributed_backend_str(self) -> str:
        # 获取 PyTorch 分布式通信后端名称，CUDA 使用 NCCL
        return "nccl"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        # 设置所有随机种子以确保可复现性
        if seed is not None:
            super().seed_everything(seed)  # 调用父类设置 CPU 等随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有 CUDA GPU 设置随机种子


class CudaSRTPlatform(CudaDeviceMixin, SRTPlatform):
    """Default in-tree CUDA SRT platform."""
    # CUDA SRT 平台类，组合了 CudaDeviceMixin 和 SRTPlatform，
    # 是 SGLang 在 NVIDIA GPU 上运行的默认平台实现。

    def supports_fp8(self) -> bool:
        # 判断是否支持 FP8（8位浮点）精度计算，CUDA 平台支持
        return True

    def support_cuda_graph(self) -> bool:
        # 判断是否支持 CUDA Graph 特性，CUDA 平台支持
        # CUDA Graph 可以将多个 GPU 操作录制为图以减少 CPU 开销
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        # 判断是否支持分段 CUDA Graph 特性，CUDA 平台支持
        # 分段 CUDA Graph 允许将计算图分为多段分别录制和重放
        return True
