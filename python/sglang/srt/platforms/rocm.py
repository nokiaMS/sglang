# ROCm设备操作模块
# 本模块为SRT平台层提供ROCm设备操作。由于PyTorch通过相同的torch.cuda.* API
# 暴露ROCm（HIP是二进制兼容层），RocmDeviceMixin继承CudaDeviceMixin的全部设备操作，
# 仅覆盖身份标识（_enum、device_name）。
"""ROCm device operations for the SRT platform layer.

PyTorch exposes ROCm through the same ``torch.cuda.*`` API surface as CUDA
(HIP is a binary shim, and ``torch.device("rocm")`` does not exist). So
``RocmDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).
"""

from sglang.srt.platforms.cuda import CudaDeviceMixin  # 导入CUDA设备混入类
from sglang.srt.platforms.device_mixin import PlatformEnum  # 导入平台枚举
from sglang.srt.platforms.interface import SRTPlatform  # 导入SRT平台基类


class RocmDeviceMixin(CudaDeviceMixin):  # ROCm设备混入类，继承CUDA设备混入
    """ROCm device ops — identical surface to CUDA via torch.cuda's HIP shim."""  # ROCm设备操作——通过torch.cuda的HIP兼容层与CUDA具有相同的接口

    _enum: PlatformEnum = PlatformEnum.ROCM  # 平台枚举为ROCM
    device_name: str = "rocm"  # 设备名称为"rocm"
    # device_type stays "cuda" — torch.device("cuda") is the only valid  # device_type保持"cuda"——torch.device("cuda")是HIP设备在PyTorch中唯一有效的设备类型字符串
    # device-type string for HIP devices in PyTorch.  # HIP设备在PyTorch中唯一的设备类型字符串


class RocmSRTPlatform(RocmDeviceMixin, SRTPlatform):  # ROCm SRT平台类
    """Default in-tree ROCm SRT platform.  # 默认的树内ROCm SRT平台

    Capability flags (supports_fp8, support_cuda_graph, support_piecewise_cuda_graph)
    keep the conservative SRTPlatform defaults rather than mirroring CudaSRTPlatform.  # 能力标志保持SRTPlatform的保守默认值而非镜像CudaSRTPlatform
    They are currently only consulted in OOT branches gated on is_out_of_tree(),  # 它们目前仅在is_out_of_tree()门控的OOT分支中被查询
    so the defaults are behaviorally inert for the in-tree ROCm path. A follow-up  # 因此默认值对树内ROCm路径是行为上无效的
    that migrates AMD-specific gating off legacy is_hip() should set these here.  # 后续将AMD特定的门控从旧版is_hip()迁移时应在此处设置这些值
    """
