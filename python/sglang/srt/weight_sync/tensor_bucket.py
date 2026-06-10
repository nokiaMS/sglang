# 张量桶模块
# 实现将多个张量展平为单个张量的桶结构，用于高效传输和处理
# 同时保留所有元数据以支持从展平张量重构原始张量

from dataclasses import dataclass  # 数据类装饰器
from typing import List, Tuple  # 类型注解

import torch  # PyTorch深度学习框架


@dataclass  # 数据类装饰器
class FlattenedTensorMetadata:  # 展平张量元数据，记录单个张量在桶中的位置信息
    """Metadata for a tensor in a flattened bucket"""
    # 展平桶中张量的元数据

    name: str  # 张量名称
    shape: torch.Size  # 张量原始形状
    dtype: torch.dtype  # 张量数据类型
    start_idx: int  # 在展平张量中的起始字节索引
    end_idx: int  # 在展平张量中的结束字节索引
    numel: int  # 字节元素数量


class FlattenedTensorBucket:  # 展平张量桶，将多个张量展平为单个张量
    """
    A bucket that flattens multiple tensors into a single tensor for efficient processing
    while preserving all metadata needed for reconstruction.
    """
    # 将多个张量展平为单个张量的桶，用于高效处理
    # 同时保留所有重构所需的元数据。

    # This field is solely for users of to check whether the class supports this feature
    # 此字段仅供用户检查该类是否支持此功能
    supports_multi_dtypes = True  # 支持多种数据类型

    def __init__(  # 初始化展平张量桶
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,  # 命名张量列表（用于创建新桶）
        flattened_tensor: torch.Tensor = None,  # 预展平的张量（用于重构）
        metadata: List[FlattenedTensorMetadata] = None,  # 预计算的元数据（用于重构）
    ):
        """
        Initialize a tensor bucket from a list of named tensors OR from pre-flattened data.
        Args:
            named_tensors: List of (name, tensor) tuples (for creating new bucket)
            flattened_tensor: Pre-flattened tensor (for reconstruction)
            metadata: Pre-computed metadata (for reconstruction)
        """
        # 从命名张量列表或预展平数据初始化张量桶。
        # 参数：
        #   named_tensors: (名称, 张量)元组列表（用于创建新桶）
        #   flattened_tensor: 预展平的张量（用于重构）
        #   metadata: 预计算的元数据（用于重构）
        if named_tensors is not None:  # 如果提供了命名张量列表
            # Create bucket from named tensors
            # 从命名张量创建桶
            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)  # 初始化元数据列表
            self.flattened_tensor: torch.Tensor = None  # 初始化展平张量为None

            if not named_tensors:  # 如果命名张量列表为空
                raise ValueError("Cannot create empty tensor bucket")  # 抛出异常：不能创建空桶

            # Collect metadata and flatten tensors
            # 收集元数据并展平张量
            current_idx = 0  # 当前字节索引
            flattened_tensors: List[torch.Tensor] = [None] * len(named_tensors)  # 初始化展平张量列表

            for i, (name, tensor) in enumerate(named_tensors):  # 遍历命名张量
                flattened = tensor.flatten().view(torch.uint8)  # 展平并按字节视图查看
                flattened_tensors[i] = flattened  # 存储展平后的张量

                # Store metadata
                # 存储元数据

                numel = flattened.numel()  # 获取字节元素数量
                metadata_obj = FlattenedTensorMetadata(  # 创建元数据对象
                    name=name,  # 张量名称
                    shape=tensor.shape,  # 原始形状
                    dtype=tensor.dtype,  # 数据类型
                    start_idx=current_idx,  # 起始索引
                    end_idx=current_idx + numel,  # 结束索引
                    numel=numel,  # 字节元素数量
                )
                self.metadata[i] = metadata_obj  # 存储元数据对象
                current_idx += numel  # 更新当前索引

            # Concatenate all flattened tensors
            # 拼接所有展平张量
            self.flattened_tensor = torch.cat(flattened_tensors, dim=0)  # 沿第0维拼接
        else:  # 从预展平数据初始化
            # Initialize from pre-flattened data
            if flattened_tensor is None or metadata is None:  # 如果缺少展平张量或元数据
                raise ValueError(  # 抛出异常
                    "Must provide either named_tensors or both flattened_tensor and metadata"  # 必须提供命名张量或展平张量和元数据
                )
            self.flattened_tensor = flattened_tensor  # 保存展平张量
            self.metadata = metadata  # 保存元数据

    def get_flattened_tensor(self) -> torch.Tensor:  # 获取包含所有桶张量的展平张量
        """Get the flattened tensor containing all bucket tensors"""
        # 获取包含所有桶张量的展平张量
        return self.flattened_tensor  # 返回展平张量

    def get_metadata(self) -> List[FlattenedTensorMetadata]:  # 获取桶中所有张量的元数据
        """Get metadata for all tensors in the bucket"""
        # 获取桶中所有张量的元数据
        return self.metadata  # 返回元数据列表

    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:  # 从展平张量重构原始张量
        """
        Reconstruct original tensors from flattened tensor with optimized performance.
        Uses memory-efficient operations to minimize allocations and copies.
        """
        # 从展平张量重构原始张量，优化性能。
        # 使用内存高效的操作来最小化分配和拷贝。
        # preallocate the result list
        # 预分配结果列表
        reconstructed = [None] * len(self.metadata)  # 预分配与元数据等长的列表

        for i, meta in enumerate(self.metadata):  # 遍历元数据
            tensor = (  # 从展平张量中切片重构
                self.flattened_tensor[meta.start_idx : meta.end_idx]  # 按字节范围切片
                .view(meta.dtype)  # 恢复原始数据类型
                .reshape(meta.shape)  # 恢复原始形状
            )

            reconstructed[i] = (meta.name, tensor)  # 存储名称和张量的元组

        return reconstructed  # 返回重构结果列表
