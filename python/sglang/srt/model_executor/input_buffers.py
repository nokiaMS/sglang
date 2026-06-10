# 本文件实现了前向推理输入缓冲区的共享机制。
# 通过复用已分配的 GPU 张量，避免每次前向推理时重复分配内存，
# 从而减少显存碎片和分配开销，提升推理性能。
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Dict

import torch

from sglang.srt.utils import is_npu

# 全局缓冲区池，按名称缓存已分配的张量，用于跨请求复用
_forward_input_buffer_pool: Dict[str, torch.Tensor] = {}


@dataclass
class ForwardInputBuffers:
    """前向推理输入缓冲区的数据类，支持张量缓冲区的跨请求共享复用。"""

    def _share_one_buffer(self, name: str, new_buffer: torch.Tensor) -> torch.Tensor:
        """尝试为指定名称的缓冲区复用已分配的张量。

        如果池中已有同名缓冲区且容量足够，则复用旧缓冲区，
        避免重复分配显存。返回的张量视图大小和步长与新缓冲区一致。

        Args:
            name: 缓冲区的唯一标识名称
            new_buffer: 当前请求需要的新缓冲区张量

        Returns:
            复用或新分配的张量，其视图大小和步长与 new_buffer 一致
        """

        buffer_size = new_buffer.size()  # 记录新缓冲区的大小
        buffer_stride = new_buffer.stride()  # 记录新缓冲区的步长

        old_buffer = _forward_input_buffer_pool.get(name, None)  # 从池中查找已有缓冲区
        if old_buffer is not None:
            assert (
                new_buffer.dtype == old_buffer.dtype
            ), f"Buffer {name} has different dtype than before."
            assert (
                new_buffer.device == old_buffer.device
            ), f"Buffer {name} has different device than before."
            # 如果旧缓冲区元素数不少于新缓冲区，则复用旧缓冲区以节省显存分配
            if old_buffer.numel() > new_buffer.numel():
                new_buffer = old_buffer

        # 将缓冲区存入池中供后续请求复用
        _forward_input_buffer_pool[name] = new_buffer
        # 返回以新缓冲区大小和步长为视图的张量，确保形状与原始请求一致
        return new_buffer.as_strided(buffer_size, buffer_stride)

    def share_buffers(self):
        """遍历当前数据类的所有字段，对张量类型的缓冲区执行共享复用。

        支持三种字段类型：torch.Tensor、dict（值为 torch.Tensor）、
        以及 dataclass（属性为 torch.Tensor）。
        """
        # disable share input buffer on npu due to accuracy issue
        # NPU 上禁用输入缓冲区共享，因为存在精度问题
        if is_npu():
            return

        # 遍历数据类的所有字段
        for f in fields(self):
            name = f.name  # 字段名称
            buffer = getattr(self, name)  # 获取字段值

            if buffer is None:
                continue  # 跳过空值字段

            # 如果字段值是 dataclass，则提取其属性字典
            if dataclasses.is_dataclass(buffer):
                buffer = vars(buffer)

            # 如果字段值是字典，对其中的每个张量值执行缓冲区共享
            if isinstance(buffer, dict):
                for sub_name, sub_buffer in buffer.items():
                    assert isinstance(
                        sub_buffer, torch.Tensor
                    ), f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                    # 为字典中的每个子张量尝试复用缓冲区
                    new_buffer = self._share_one_buffer(
                        f"{name}.{sub_name}", sub_buffer
                    )
                    buffer[sub_name] = new_buffer  # 用复用后的张量替换原值
            else:
                # 直接对张量字段执行缓冲区共享
                assert isinstance(
                    buffer, torch.Tensor
                ), f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
                new_buffer = self._share_one_buffer(name, buffer)
                setattr(self, name, new_buffer)  # 用复用后的张量替换原字段值
