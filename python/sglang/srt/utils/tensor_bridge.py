# Copied and adapted from: https://github.com/vllm-project/vllm-metal
# SPDX-License-Identifier: Apache-2.0
# MLX与PyTorch之间的张量桥接工具，支持Apple Silicon统一内存的零拷贝转换
# 提供MLX数组和PyTorch张量之间的双向转换，自动处理数据类型映射和MPS设备大小限制
"""Tensor bridge between MLX and PyTorch.  # MLX与PyTorch之间的张量桥接

Provides zero-copy conversion when possible using Apple Silicon's unified memory.  # 在可能的情况下使用Apple Silicon统一内存提供零拷贝转换
"""

from __future__ import annotations  # 启用延迟注解求值

import logging  # 导入日志记录模块
from functools import lru_cache  # 导入LRU缓存装饰器
from typing import TYPE_CHECKING, Literal  # 导入类型注解

import torch  # 导入PyTorch张量库

from sglang.srt.environ import envs  # 导入环境变量配置

if TYPE_CHECKING:  # 仅在类型检查时导入
    import mlx.core as mx  # 导入MLX核心模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_MLX_AVAILABLE: bool = False  # MLX是否可用的标志
try:  # 尝试导入MLX
    import mlx.core as mx  # noqa: F811  # 导入MLX核心模块

    _MLX_AVAILABLE = True  # 标记MLX可用
except ImportError:  # 如果导入失败
    pass  # 忽略


def is_mlx_available() -> bool:  # 检查MLX包是否可导入
    """Return True when the ``mlx`` package can be imported."""  # 当mlx包可导入时返回True
    return _MLX_AVAILABLE  # 返回MLX可用标志


@lru_cache(maxsize=1)  # LRU缓存，最多缓存1个结果
def use_mlx() -> bool:  # 检查用户是否选择使用MLX且MLX可导入
    """Return True when the user opted-in via ``SGLANG_USE_MLX=1`` **and** MLX is importable."""  # 当用户通过SGLANG_USE_MLX=1选择启用且MLX可导入时返回True
    return bool(envs.SGLANG_USE_MLX.get()) and _MLX_AVAILABLE  # 检查环境变量和MLX可用性


# MPS has a 4GB (2^32 bytes) limit for MPSTemporaryNDArray allocations.  # MPS对MPSTemporaryNDArray分配有4GB限制
# Metal may allocate multiple temporary buffers internally, so we use a  # Metal内部可能分配多个临时缓冲区，因此我们使用
# conservative threshold of 1GB to avoid hitting the limit.  # 保守的1GB阈值以避免达到限制
# See: https://github.com/anthropics/vllm-metal/issues/43  # 参见：vllm-metal问题#43
_MPS_SAFE_SIZE_BYTES = 1 << 30  # 1GB  # MPS安全大小阈值（1GB）

# MLX to PyTorch dtype mapping  # MLX到PyTorch数据类型映射
# TODO(perf): float64 is CPU-only in MLX (see ml-explore/mlx#1843).  # 待办：float64在MLX中仅支持CPU
# When the target device is GPU/MPS we should auto-downcast float64 → float32  # 当目标设备为GPU/MPS时应自动将float64降级为float32
# to avoid a runtime error; when the target is CPU we can keep float64.  # 以避免运行时错误；当目标为CPU时可保留float64
# For now float64 is omitted from the mapping so it hits the ValueError  # 目前float64未包含在映射中，因此会触发ValueError
# fallback in mlx_to_torch().  # 在mlx_to_torch()中的回退处理
MLX_TO_TORCH_DTYPE = (  # MLX到PyTorch数据类型映射字典
    {
        mx.float32: torch.float32,  # float32映射
        mx.float16: torch.float16,  # float16映射
        mx.bfloat16: torch.bfloat16,  # bfloat16映射
        mx.int32: torch.int32,  # int32映射
        mx.int64: torch.int64,  # int64映射
        mx.int16: torch.int16,  # int16映射
        mx.int8: torch.int8,  # int8映射
        mx.uint8: torch.uint8,  # uint8映射
        mx.bool_: torch.bool,  # bool映射
    }
    if _MLX_AVAILABLE  # 仅在MLX可用时构建映射
    else {}  # MLX不可用时使用空字典
)

# PyTorch to MLX dtype mapping  # PyTorch到MLX数据类型映射
TORCH_TO_MLX_DTYPE = {v: k for k, v in MLX_TO_TORCH_DTYPE.items()}  # 反转映射方向


def get_torch_device() -> torch.device:  # 获取Metal/MPS的PyTorch设备
    """Get the PyTorch device for Metal/MPS.  # 获取Metal/MPS的PyTorch设备

    Returns:  # 返回值
        torch.device for MPS if available, else CPU  # 如果MPS可用返回MPS设备，否则返回CPU设备
    """
    if torch.backends.mps.is_available():  # 如果MPS后端可用
        return torch.device("mps")  # 返回MPS设备
    return torch.device("cpu")  # 返回CPU设备


def _get_tensor_size_bytes(array: mx.array) -> int:  # 计算MLX数组的字节大小
    """Calculate the size of an MLX array in bytes.  # 计算MLX数组的字节大小

    Args:  # 参数
        array: MLX array  # MLX数组

    Returns:  # 返回值
        Size in bytes  # 字节大小
    """
    return array.size * array.dtype.size  # 元素数量乘以每个元素的字节大小


def _is_safe_for_mps(array: mx.array) -> bool:  # 检查数组是否可以安全传输到MPS而不会超出大小限制
    """Check if an array is safe to transfer to MPS without hitting size limits.  # 检查数组是否可安全传输到MPS而不会超出大小限制

    MPS has a 4GB limit for MPSTemporaryNDArray, but Metal may allocate  # MPS对MPSTemporaryNDArray有4GB限制，但Metal可能分配
    multiple temporary buffers internally. We use a conservative threshold.  # 多个内部临时缓冲区。我们使用保守阈值。

    Args:  # 参数
        array: MLX array to check  # 要检查的MLX数组

    Returns:  # 返回值
        True if safe to transfer to MPS, False if should stay on CPU  # 如果安全传输到MPS返回True，否则应留在CPU返回False
    """
    return _get_tensor_size_bytes(array) < _MPS_SAFE_SIZE_BYTES  # 比较数组大小与安全阈值


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:  # 将PyTorch张量转换为MLX数组
    """Convert PyTorch tensor to MLX array.  # 将PyTorch张量转换为MLX数组

    Uses numpy as an intermediate to enable zero-copy on unified memory.  # 使用numpy作为中间层以在统一内存上启用零拷贝

    Args:  # 参数
        tensor: PyTorch tensor (can be on any device)  # PyTorch张量（可在任何设备上）

    Returns:  # 返回值
        MLX array with the same data  # 具有相同数据的MLX数组
    """
    # Move to CPU if on MPS for numpy conversion  # 如果在MPS上，先移到CPU以进行numpy转换
    if tensor.device.type != "cpu":  # 如果张量不在CPU上
        tensor = tensor.cpu()  # 将张量移到CPU

    tensor = tensor.detach()  # 分离张量的计算图

    # Note: numpy does not support bfloat16.  # 注意：numpy不支持bfloat16
    if tensor.dtype == torch.bfloat16:  # 如果张量是bfloat16类型
        return mx.array(tensor)  # 直接通过MLX转换（绕过numpy）

    return mx.array(tensor.numpy())  # 通过numpy中间层转换


# TODO(perf): accept a list/batch of arrays and convert them in one pass  # 待办：接受数组/批次列表并在一次传递中转换
# to reduce the Python ↔ MLX round-trip overhead.  # 以减少Python与MLX之间的往返开销
def mlx_to_torch(  # 将MLX数组转换为PyTorch张量
    array: mx.array,  # MLX数组
    device: torch.device | Literal["mps", "cpu"] | None = None,  # 目标PyTorch设备
    already_contiguous: bool = False,  # 是否已知数组连续（跳过连续性检查）
) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor.  # 将MLX数组转换为PyTorch张量

    Uses numpy as an intermediate to enable zero-copy on unified memory.  # 使用numpy作为中间层以在统一内存上启用零拷贝

    Args:  # 参数
        array: MLX array  # MLX数组
        device: Target PyTorch device (default: MPS if available)  # 目标PyTorch设备（默认：MPS如果可用）
        already_contiguous: Skip contiguity check if array is known contiguous  # 如果已知数组连续则跳过连续性检查

    Returns:  # 返回值
        PyTorch tensor with the same data  # 具有相同数据的PyTorch张量
    """
    if device is None:  # 如果未指定设备
        device = get_torch_device()  # 使用默认设备
    elif isinstance(device, str):  # 如果设备是字符串
        device = torch.device(device)  # 转换为torch.device对象

    # Use memoryview for zero-copy conversion (bypasses numpy for bfloat16)  # 使用memoryview进行零拷贝转换（绕过numpy处理bfloat16）
    # reference: https://github.com/ml-explore/mlx/issues/403  # 参考：MLX问题#403
    torch_dtype = MLX_TO_TORCH_DTYPE.get(array.dtype)  # 查找MLX到PyTorch的数据类型映射
    if torch_dtype is not None:  # 如果找到了映射
        if already_contiguous:  # 如果已知数组连续
            # Fast path: skip contiguity check, single eval  # 快速路径：跳过连续性检查，单次求值
            mx.eval(array)  # 强制求值MLX数组
            buffer = memoryview(array)  # 获取数组的内存视图
        else:  # 需要检查连续性
            # MLX views / non-contiguous arrays expose a non-contiguous buffer (or  # MLX视图/非连续数组暴露非连续缓冲区
            # sometimes no usable buffer), which `torch.frombuffer` can't consume.  # 有时无可用的缓冲区，torch.frombuffer无法消费
            # Make contiguous first, then eval once  # 先确保连续，然后单次求值
            array = mx.contiguous(array)  # 使数组连续
            mx.eval(array)  # 强制求值
            buffer = memoryview(array)  # 获取数组的内存视图

        tensor = torch.frombuffer(buffer, dtype=torch_dtype).reshape(array.shape)  # 从缓冲区创建张量并重塑形状
    else:  # 如果未找到映射
        # Fallback to numpy path for unsupported dtypes  # 对不支持的数据类型回退到numpy路径
        raise ValueError(f"Unsupported MLX dtype: {array.dtype}")  # 抛出不支持的类型错误

    # Move to target device, but check for MPS size limits first  # 移到目标设备，但先检查MPS大小限制
    if device.type == "mps":  # 如果目标是MPS设备
        if _is_safe_for_mps(array):  # 如果数组大小安全
            tensor = tensor.to(device)  # 移到MPS设备
        else:  # 数组太大
            # Large tensor - keep on CPU to avoid MPS 4GB limit crash  # 大张量 - 保留在CPU以避免MPS 4GB限制崩溃
            # See: https://github.com/anthropics/vllm-metal/issues/43  # 参见：vllm-metal问题#43
            logger.debug(  # 记录调试信息
                "Tensor too large for MPS (%d bytes > %d limit), keeping on CPU",  # 张量对MPS太大，保留在CPU
                _get_tensor_size_bytes(array),  # 数组字节大小
                _MPS_SAFE_SIZE_BYTES,  # 安全阈值
            )
    elif device.type != "cpu":  # 如果目标是其他非CPU设备
        tensor = tensor.to(device)  # 移到目标设备

    return tensor  # 返回转换后的张量


def sync_mlx() -> None:  # 同步MLX操作
    """Synchronize MLX operations.  # 同步MLX操作

    Call this before converting MLX arrays to ensure all operations complete.  # 在转换MLX数组前调用以确保所有操作完成
    """
    # Prefer an explicit MLX barrier when available; otherwise force evaluation.  # 优先使用显式MLX屏障；否则强制求值
    # `mx.eval([])` is a no-op, so we evaluate a tiny scalar as a safe fallback.  # mx.eval([])是空操作，因此求值一个小标量作为安全回退
    try:
        mx.synchronize()  # 尝试使用MLX同步屏障
    except (AttributeError, TypeError):  # 如果不支持同步屏障
        mx.eval(mx.array(0, dtype=mx.int32))  # 求值一个小标量作为回退


def sync_torch() -> None:  # 同步PyTorch MPS操作
    """Synchronize PyTorch MPS operations.  # 同步PyTorch MPS操作

    Call this before converting PyTorch tensors to ensure all operations complete.  # 在转换PyTorch张量前调用以确保所有操作完成
    """
    if torch.backends.mps.is_available():  # 如果MPS后端可用
        torch.mps.synchronize()  # 同步MPS操作


__all__ = [  # 模块公开导出符号列表
    "is_mlx_available",  # MLX可用性检查函数
    "use_mlx",  # MLX使用检查函数
    "mlx_to_torch",  # MLX到PyTorch转换函数
    "torch_to_mlx",  # PyTorch到MLX转换函数
    "get_torch_device",  # 获取设备函数
]
