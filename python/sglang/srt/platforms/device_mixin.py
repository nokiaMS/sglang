# 设备混入模块
# 本模块提供SGLang平台的共享设备抽象，DeviceMixin提供通用的设备身份查询
# 和操作方法，供SRT（LLM推理）和多模态（扩散）平台层级共享使用。
"""
Shared device abstraction for SGLang platforms.

DeviceMixin provides the common device identity queries and operations
shared between the SRT (LLM inference) and Multimodal (diffusion)
platform hierarchies.  Concrete per-device mixins (e.g. MyDeviceMixin)
implement the abstract operations; subsystem-specific platforms
(SRTPlatform, MMPlatform) inherit DeviceMixin and add their own methods.

Hierarchy example (OOT plugin)::

    DeviceMixin
    ├── MyDeviceMixin(DeviceMixin)        # vendor-specific device operations  # 厂商特定的设备操作
    ├── SRTPlatform(DeviceMixin)          # + graph runner, KV pool, …  # 加上图运行器、KV池等
    │   └── MySRTPlatform(SRTPlatform, MyDeviceMixin)
    └── MMPlatform(DeviceMixin)           # + attention backend, VAE, …  # 加上注意力后端、VAE等
        └── MyMMPlatform(MMPlatform, MyDeviceMixin)

Method status annotations:  # 方法状态注解

- ``[Active]``  — SGLang core calls this method through ``current_platform``.  # SGLang核心通过current_platform调用此方法
  OOT implementations take effect immediately.  # OOT实现立即生效
- ``[Planned]`` — Reserved interface. SGLang core still uses hardcoded calls  # 保留接口，SGLang核心仍使用硬编码调用
  (e.g. ``torch.cuda.empty_cache()``). OOT implementations will NOT take  # 如torch.cuda.empty_cache()，OOT实现暂不生效
  effect until the core is migrated in a future PR.  # 直到核心在未来PR中迁移
"""

import enum  # 导入枚举模块
import random  # 导入随机数模块
from typing import NamedTuple, Optional  # 导入类型注解

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch


class PlatformEnum(enum.Enum):
    """Enumeration of known platform types.  # 已知平台类型的枚举

    Superset of both SRT and MM enums so that a single PlatformEnum can
    be shared across subsystems.  # SRT和MM枚举的超集，可在子系统间共享
    """

    CUDA = enum.auto()  # NVIDIA CUDA平台
    ROCM = enum.auto()  # AMD ROCm平台
    CPU = enum.auto()  # CPU平台
    XPU = enum.auto()  # Intel XPU平台
    MUSA = enum.auto()  # 摩尔线程MUSA平台
    NPU = enum.auto()  # 华为昇腾NPU平台
    TPU = enum.auto()  # Google TPU平台
    MPS = enum.auto()  # Apple MPS平台
    OOT = enum.auto()  # Out-of-tree (external plugin)  # 外部插件平台
    UNSPECIFIED = enum.auto()  # 未指定平台


class CpuArchEnum(enum.Enum):
    """CPU architecture enumeration."""  # CPU架构枚举

    X86 = enum.auto()  # x86架构
    ARM = enum.auto()  # ARM架构
    UNSPECIFIED = enum.auto()  # 未指定架构


class DeviceCapability(NamedTuple):
    """Device compute capability (major, minor).  # 设备计算能力（主版本，次版本）

    Uses NamedTuple for built-in comparison support:  # 使用NamedTuple以支持内建比较
    ``DeviceCapability(9, 0) >= DeviceCapability(8, 9)`` works naturally.  # 自然支持比较运算
    """

    major: int  # 主版本号
    minor: int  # 次版本号

    def as_version_str(self) -> str:  # 转换为版本字符串
        """将计算能力转换为版本字符串"""  # Convert capability to version string
        return f"{self.major}.{self.minor}"  # 返回"主.次"格式

    def to_int(self) -> int:  # 转换为整数
        """Express capability as ``<major><minor>`` (minor is single digit)."""  # 将计算能力表示为整数，次版本为单数字
        assert 0 <= self.minor < 10  # 断言次版本号在0-9范围内
        return self.major * 10 + self.minor  # 返回主版本*10+次版本


class DeviceMixin:
    """Mixin providing device identity queries and basic device operations.  # 提供设备身份查询和基本设备操作的混入类

    Class-level attributes (override in subclasses):  # 类级属性（在子类中覆盖）
        _enum:       PlatformEnum identifying this platform.  # 标识此平台的PlatformEnum
        device_name: Human-readable short name (e.g. "cuda", "npu").  # 人类可读的短名称
        device_type: ``torch.device`` type string (e.g. "cuda", "npu").  # torch.device类型字符串
    """

    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED  # 平台枚举，默认未指定
    device_name: str = "unknown"  # 设备名称，默认"unknown"
    device_type: str = "cpu"  # 设备类型，默认"cpu"

    # ------------------------------------------------------------------
    # Platform identity queries  # 平台身份查询
    # ------------------------------------------------------------------

    def is_cuda(self) -> bool:  # 判断是否为CUDA平台
        """判断当前平台是否为CUDA"""  # Check if current platform is CUDA
        return self._enum == PlatformEnum.CUDA  # 比较枚举值

    def is_rocm(self) -> bool:  # 判断是否为ROCm平台
        """判断当前平台是否为ROCm"""  # Check if current platform is ROCm
        return self._enum == PlatformEnum.ROCM  # 比较枚举值

    def is_cpu(self) -> bool:  # 判断是否为CPU平台
        """判断当前平台是否为CPU"""  # Check if current platform is CPU
        return self._enum == PlatformEnum.CPU  # 比较枚举值

    def is_xpu(self) -> bool:  # 判断是否为XPU平台
        """判断当前平台是否为XPU"""  # Check if current platform is XPU
        return self._enum == PlatformEnum.XPU  # 比较枚举值

    def is_musa(self) -> bool:  # 判断是否为MUSA平台
        """判断当前平台是否为MUSA"""  # Check if current platform is MUSA
        return self._enum == PlatformEnum.MUSA  # 比较枚举值

    def is_npu(self) -> bool:  # 判断是否为NPU平台
        """判断当前平台是否为NPU"""  # Check if current platform is NPU
        return self._enum == PlatformEnum.NPU  # 比较枚举值

    def is_tpu(self) -> bool:  # 判断是否为TPU平台
        """判断当前平台是否为TPU"""  # Check if current platform is TPU
        return self._enum == PlatformEnum.TPU  # 比较枚举值

    def is_mps(self) -> bool:  # 判断是否为MPS平台
        """判断当前平台是否为MPS"""  # Check if current platform is MPS
        return self._enum == PlatformEnum.MPS  # 比较枚举值

    def is_cuda_alike(self) -> bool:  # 判断是否为类CUDA平台
        """True for CUDA, ROCm, or MUSA (all expose CUDA-like APIs)."""  # CUDA、ROCm或MUSA返回True（都暴露类CUDA API）
        return self._enum in (  # 检查是否在类CUDA平台集合中
            PlatformEnum.CUDA,
            PlatformEnum.ROCM,
            PlatformEnum.MUSA,
        )

    def is_out_of_tree(self) -> bool:  # 判断是否为外部注册平台
        """True for externally-registered OOT platforms."""  # 外部注册的OOT平台返回True
        return self._enum == PlatformEnum.OOT  # 比较枚举值

    # ------------------------------------------------------------------
    # Active methods — core calls these through current_platform.  # 活跃方法——核心通过current_platform调用
    # OOT implementations take effect immediately.  # OOT实现立即生效
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:  # 获取设备总内存
        """[Active] Get total device memory in bytes."""  # [活跃] 获取设备总内存（字节）
        raise NotImplementedError  # 子类必须实现

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:  # 获取当前内存使用量
        """[Active] Get current peak memory usage in bytes."""  # [活跃] 获取当前峰值内存使用量（字节）
        raise NotImplementedError  # 子类必须实现

    # ------------------------------------------------------------------
    # Planned methods — reserved interface.  Core still uses hardcoded  # 计划方法——保留接口，核心仍使用硬编码调用
    # calls (e.g. torch.cuda.*).  OOT implementations will NOT take  # 如torch.cuda.*，OOT实现暂不生效
    # effect until the core is migrated in a future PR.  # 直到核心在未来PR中迁移
    # ------------------------------------------------------------------

    # ---- Device management ----  # 设备管理

    def get_device(self, local_rank: int) -> "torch.device":  # 获取设备对象
        """[Planned] Return ``torch.device`` for the given local rank."""  # [计划] 返回给定本地排名的torch.device
        raise NotImplementedError  # 子类必须实现

    def set_device(self, device: "torch.device") -> None:  # 设置当前设备
        """[Planned] Set the current device."""  # [计划] 设置当前设备
        raise NotImplementedError  # 子类必须实现

    def get_device_name(self, device_id: int = 0) -> str:  # 获取设备名称
        """[Planned] Get human-readable device name."""  # [计划] 获取人类可读的设备名称
        raise NotImplementedError  # 子类必须实现

    def get_device_uuid(self, device_id: int = 0) -> str:  # 获取设备UUID
        """[Planned] Get unique device identifier string."""  # [计划] 获取唯一设备标识符字符串
        raise NotImplementedError  # 子类必须实现

    def get_device_capability(self, device_id: int = 0) -> Optional["DeviceCapability"]:  # 获取设备计算能力
        """[Planned] Get device compute capability. None if N/A."""  # [计划] 获取设备计算能力，不可用时返回None
        raise NotImplementedError  # 子类必须实现

    def empty_cache(self) -> None:  # 释放缓存内存
        """[Planned] Release cached device memory. No-op for CPU-like platforms."""  # [计划] 释放缓存的设备内存，CPU类平台为空操作
        pass  # 默认空操作

    def synchronize(self) -> None:  # 同步设备操作
        """[Planned] Synchronize device operations. No-op for CPU-like platforms."""  # [计划] 同步设备操作，CPU类平台为空操作
        pass  # 默认空操作

    # ---- Memory ----  # 内存

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:  # 获取可用内存
        """[Planned] Return ``(free_bytes, total_bytes)``."""  # [计划] 返回(空闲字节数, 总字节数)
        raise NotImplementedError  # 子类必须实现

    # ---- Distributed ----  # 分布式

    def get_torch_distributed_backend_str(self) -> str:  # 获取分布式后端字符串
        """[Planned] Return the torch.distributed backend string (e.g. "nccl", "hccl")."""  # [计划] 返回torch.distributed后端字符串
        raise NotImplementedError  # 子类必须实现

    def get_communicator_class(self) -> type | None:  # 获取通信器类
        """[Planned] Return platform-specific communicator class, or None for default."""  # [计划] 返回平台特定的通信器类，默认返回None
        return None  # 默认返回None

    # ---- Misc ----  # 其他

    @classmethod
    def inference_mode(cls):  # 获取推理模式上下文管理器
        """[Planned] Return inference mode context manager."""  # [计划] 返回推理模式上下文管理器
        return torch.inference_mode(mode=True)  # 返回PyTorch推理模式

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:  # 设置随机种子
        """[Planned] Set random seeds for reproducibility across all libraries."""  # [计划] 设置所有库的随机种子以确保可重现性
        if seed is not None:  # 如果指定了种子
            random.seed(seed)  # 设置Python随机种子
            np.random.seed(seed)  # 设置NumPy随机种子
            torch.manual_seed(seed)  # 设置PyTorch随机种子

    def verify_quantization(self, quant: str) -> None:  # 验证量化方法
        """[Planned] Validate that a quantization method is supported. No-op by default."""  # [计划] 验证量化方法是否支持，默认空操作
        pass  # 默认空操作

    @classmethod
    def get_cpu_architecture(cls) -> "CpuArchEnum":  # 检测CPU架构
        """[Planned] Detect CPU architecture."""  # [计划] 检测CPU架构
        import platform as _platform  # 导入平台检测模块

        machine = _platform.machine().lower()  # 获取机器架构并转为小写
        if machine in ("x86_64", "amd64", "i386", "i686"):  # x86架构标识
            return CpuArchEnum.X86  # 返回x86架构
        elif machine in ("arm64", "aarch64"):  # ARM架构标识
            return CpuArchEnum.ARM  # 返回ARM架构
        return CpuArchEnum.UNSPECIFIED  # 返回未指定架构

    # ------------------------------------------------------------------
    # Dunder helpers  # 双下划线辅助方法
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # 字符串表示
        return f"{self.__class__.__name__}(device={self.device_name})"  # 返回类名和设备名
