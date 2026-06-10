# Torch LoRA算子模块的初始化文件
# 根据是否使用CUDA Graph模式，自动选择Graph路径或Control路径执行LoRA Sgemm前向计算
# 导出sgemm_lora_a_embedding_fwd、sgemm_lora_a_fwd、sgemm_lora_b_fwd三个接口
from typing import Optional # 导入可选类型

import torch # 导入PyTorch库

from sglang.srt.lora.utils import LoRABatchInfo # 导入LoRA批处理信息

from .graph_lora_ops import ( # 导入CUDA Graph路径的LoRA算子
    sgemm_lora_a_embedding_graph_fwd, # Graph路径的LoRA A嵌入前向
    sgemm_lora_a_graph_fwd, # Graph路径的LoRA A Sgemm前向
    sgemm_lora_b_graph_fwd, # Graph路径的LoRA B Sgemm前向
)
from .lora_ops import sgemm_lora_a_embedding_fwd as sgemm_lora_a_embedding_control_fwd # 导入Control路径的LoRA A嵌入前向，重命名为control版本
from .lora_ops import sgemm_lora_a_fwd as sgemm_lora_a_control_fwd # 导入Control路径的LoRA A Sgemm前向，重命名为control版本
from .lora_ops import sgemm_lora_b_fwd as sgemm_lora_b_control_fwd # 导入Control路径的LoRA B Sgemm前向，重命名为control版本


def sgemm_lora_a_embedding_fwd( # LoRA A嵌入查找前向（根据CUDA Graph模式选择路径）
    inputs: torch.Tensor, # 输入token ID张量
    weights: torch.Tensor, # LoRA权重张量
    batch_info: LoRABatchInfo, # 批处理信息
    vocab_size: int, # 词表大小
) -> torch.Tensor: # 返回输出张量
    output: torch.Tensor # 声明输出张量
    if batch_info.use_cuda_graph: # 如果使用CUDA Graph模式
        output = sgemm_lora_a_embedding_graph_fwd( # 调用Graph路径的LoRA A嵌入前向
            inputs, # 输入token ID
            weights, # LoRA权重
            batch_info.weight_indices, # GPU上的权重索引
            batch_info.seg_lens, # GPU上的段长度
            batch_info.scalings, # GPU上的缩放因子
            vocab_size, # 词表大小
        )
    else: # 非CUDA Graph模式
        output = sgemm_lora_a_embedding_control_fwd( # 调用Control路径的LoRA A嵌入前向
            inputs, # 输入token ID
            weights, # LoRA权重
            batch_info.weight_indices_cpu, # CPU上的权重索引
            batch_info.seg_lens_cpu, # CPU上的段长度
            batch_info.lora_ranks_cpu, # CPU上的LoRA秩
            batch_info.scalings_cpu, # CPU上的缩放因子
            vocab_size, # 词表大小
        )
    return output # 返回输出张量


def sgemm_lora_a_fwd( # LoRA A矩阵Sgemm前向（根据CUDA Graph模式选择路径）
    inputs: torch.Tensor, # 输入张量
    weights: torch.Tensor, # LoRA A权重张量
    batch_info: LoRABatchInfo, # 批处理信息
    num_slices: int = 1, # 切片数量，默认为1
) -> torch.Tensor: # 返回输出张量
    output: torch.Tensor # 声明输出张量
    if batch_info.use_cuda_graph: # 如果使用CUDA Graph模式
        output = sgemm_lora_a_graph_fwd( # 调用Graph路径的LoRA A Sgemm前向
            inputs, # 输入张量
            weights, # LoRA A权重
            batch_info.weight_indices, # GPU上的权重索引
            batch_info.seg_lens, # GPU上的段长度
            batch_info.scalings, # GPU上的缩放因子
            num_slices, # 切片数量
        )
    else: # 非CUDA Graph模式
        output = sgemm_lora_a_control_fwd( # 调用Control路径的LoRA A Sgemm前向
            inputs, # 输入张量
            weights, # LoRA A权重
            batch_info.weight_indices_cpu, # CPU上的权重索引
            batch_info.seg_lens_cpu, # CPU上的段长度
            batch_info.lora_ranks_cpu, # CPU上的LoRA秩
            batch_info.scalings_cpu, # CPU上的缩放因子
            num_slices, # 切片数量
        )
    return output # 返回输出张量


def sgemm_lora_b_fwd( # LoRA B矩阵Sgemm前向（根据CUDA Graph模式选择路径）
    inputs: torch.Tensor, # 输入张量
    weights: torch.Tensor, # LoRA B权重张量
    batch_info: LoRABatchInfo, # 批处理信息
    slice_offsets: torch.Tensor, # 切片偏移量
    base_output: Optional[torch.Tensor] = None, # 基础输出张量（可选）
) -> torch.Tensor: # 返回输出张量
    output: torch.Tensor # 声明输出张量
    if batch_info.use_cuda_graph: # 如果使用CUDA Graph模式
        output = sgemm_lora_b_graph_fwd( # 调用Graph路径的LoRA B Sgemm前向
            inputs, # 输入张量
            weights, # LoRA B权重
            batch_info.weight_indices, # GPU上的权重索引
            batch_info.seg_lens, # GPU上的段长度
            slice_offsets, # 切片偏移量
            base_output, # 基础输出
        )
    else: # 非CUDA Graph模式
        output = sgemm_lora_b_control_fwd( # 调用Control路径的LoRA B Sgemm前向
            inputs, # 输入张量
            weights, # LoRA B权重
            batch_info.weight_indices_cpu, # CPU上的权重索引
            batch_info.seg_lens_cpu, # CPU上的段长度
            batch_info.lora_ranks_cpu, # CPU上的LoRA秩
            slice_offsets, # 切片偏移量
            base_output, # 基础输出
        )
    return output # 返回输出张量


__all__ = [ # 模块导出列表
    "sgemm_lora_a_embedding_fwd", # LoRA A嵌入查找前向
    "sgemm_lora_a_fwd", # LoRA A Sgemm前向
    "sgemm_lora_b_fwd", # LoRA B Sgemm前向
]