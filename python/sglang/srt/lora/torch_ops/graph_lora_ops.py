# 基于CUDA Graph的LoRA算子实现，提供LoRA A矩阵（Embedding和线性层）以及LoRA B矩阵的前向计算
# 适用于需要CUDA Graph捕获的场景，通过逐适配器循环实现批量LoRA计算
from typing import Optional  # 导入可选类型提示

import torch  # 导入PyTorch核心库
import torch.nn.functional as F  # 导入PyTorch函数式API


def sgemm_lora_a_embedding_graph_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    scaling_tensor: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:  # LoRA A矩阵Embedding层的前向传播（CUDA Graph版本），对每个适配器逐一计算Embedding查找并累加结果
    total_seq_len = inputs.shape[0]  # 获取总序列长度
    if weights.numel() == 0:  # 如果权重为空
        return torch.zeros(total_seq_len, 0, dtype=weights.dtype, device=weights.device)  # 返回空张量

    num_loras, max_rank, _ = weights.shape  # 获取LoRA数量、最大秩和词表维度

    output = torch.zeros(
        total_seq_len, max_rank, dtype=weights.dtype, device=weights.device
    )  # 初始化输出张量，形状为(总序列长度, 最大秩)

    for lora_idx in range(num_loras):  # 遍历每个LoRA适配器

        batch_token_mask = weight_indices[:total_seq_len] == lora_idx  # 创建当前适配器的token掩码

        x_seq = torch.where(batch_token_mask, inputs, 0)  # 将不属于当前适配器的token置零
        w_seq = weights[lora_idx]  # 获取当前适配器的权重

        output.add_(
            scaling_tensor[lora_idx]
            * torch.where(
                batch_token_mask.unsqueeze(1), F.embedding(x_seq, w_seq.t()), 0
            )
        )  # 计算Embedding查找结果，乘以缩放因子，并累加到输出中

    return output  # 返回最终输出


def sgemm_lora_a_graph_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
) -> torch.Tensor:  # LoRA A矩阵线性层的前向传播（CUDA Graph版本），对每个适配器逐一计算矩阵乘法并累加结果
    total_seq_len, input_dim = inputs.shape  # 获取总序列长度和输入维度
    if weights.numel() == 0:  # 如果权重为空
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)  # 返回空张量

    num_loras, weight_out_dim, _ = weights.shape  # 获取LoRA数量、权重输出维度和输入维度
    max_rank = weight_out_dim // num_slices  # 计算最大秩（权重输出维度除以切片数）

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=inputs.dtype, device=inputs.device
    )  # 初始化输出张量，形状为(总序列长度, 切片数*最大秩)

    for lora_idx in range(num_loras):  # 遍历每个LoRA适配器

        batch_token_mask = (weight_indices[:total_seq_len] == lora_idx).unsqueeze(1)  # 创建当前适配器的token掩码，增加一个维度用于广播

        x_seq = torch.where(batch_token_mask, inputs, 0)  # 将不属于当前适配器的token置零
        w_seq = weights[lora_idx]  # 获取当前适配器的权重

        output.add_(scaling_tensor[lora_idx] * torch.mm(x_seq, w_seq.t()))  # 计算矩阵乘法，乘以缩放因子，并累加到输出中

    return output  # 返回最终输出


def sgemm_lora_b_graph_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:  # LoRA B矩阵的前向传播（CUDA Graph版本），支持多切片输出，可累加到基础输出上
    total_seq_len, input_dim = inputs.shape  # 获取总序列长度和输入维度
    num_loras, weight_out_dim, _ = weights.shape  # 获取LoRA数量、权重输出维度和输入维度
    total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0  # 计算总输出维度

    if weights.numel() == 0:  # 如果权重为空
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )  # 返回空张量

    num_slices = len(slice_offsets) - 1  # 计算切片数量
    max_rank = input_dim // num_slices  # 计算最大秩（输入维度除以切片数）

    if base_output is not None:  # 如果提供了基础输出
        output = base_output  # 直接使用基础输出作为输出张量
    else:
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )  # 否则初始化为零张量

    for lora_idx in range(num_loras):  # 遍历每个LoRA适配器

        batch_token_mask = (weight_indices[:total_seq_len] == lora_idx).unsqueeze(1)  # 创建当前适配器的token掩码，增加一个维度用于广播
        inputs_masked = torch.where(batch_token_mask, inputs, 0)  # 将不属于当前适配器的输入置零

        for slice_idx in range(num_slices):  # 遍历每个切片
            slice_start_input = slice_idx * max_rank  # 计算当前切片在输入维度的起始位置
            slice_end_input = (slice_idx + 1) * max_rank  # 计算当前切片在输入维度的结束位置

            slice_start_output = slice_offsets[slice_idx]  # 获取当前切片在输出维度的起始偏移
            slice_end_output = slice_offsets[slice_idx + 1]  # 获取当前切片在输出维度的结束偏移

            x_slice = inputs_masked[..., slice_start_input:slice_end_input]  # 获取当前切片的输入
            w_slice = weights[
                lora_idx, slice_start_output:slice_end_output
            ]  # (slice_dim, max_rank) 获取当前切片的权重 # (slice_dim, max_rank) 获取当前切片的权重
            output[..., slice_start_output:slice_end_output].add_(
                torch.mm(x_slice, w_slice.t())
            )  # 计算矩阵乘法并累加到对应输出切片

    return output  # 返回最终输出
