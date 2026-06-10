# LoRA算子的PyTorch原生实现，提供LoRA A矩阵（Embedding和线性层）以及LoRA B矩阵的前向计算
# 通过逐序列段循环实现批量LoRA计算，适用于非CUDA Graph场景
from typing import Optional  # 导入可选类型提示

import torch  # 导入PyTorch核心库


def sgemm_lora_a_embedding_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:  # LoRA A矩阵Embedding层的前向传播，按序列段逐一计算Embedding查找
    total_seq_len = inputs.shape[0]  # 获取总序列长度
    if weights.numel() == 0:  # 如果权重为空
        return torch.zeros(total_seq_len, 0, dtype=weights.dtype, device=weights.device)  # 返回空张量

    num_loras, max_rank, _ = weights.shape  # 获取LoRA数量、最大秩和词表维度

    output = torch.zeros(
        total_seq_len, max_rank, dtype=weights.dtype, device=weights.device
    )  # 初始化输出张量，形状为(总序列长度, 最大秩)

    token_offset = 0  # 初始化token偏移量
    for lora_idx, seq_len in zip(weight_indices, seg_len_tensor):  # 遍历每个适配器及其对应的序列长度
        if seq_len == 0:  # 如果当前序列长度为0，跳过
            continue

        rank = lora_ranks[lora_idx]  # 获取当前适配器的LoRA秩
        if rank > 0:  # 如果秩大于0

            x_seq = inputs[token_offset : token_offset + seq_len]  # 获取当前序列段的输入
            w_seq = weights[lora_idx, :rank]  # 获取当前适配器的权重（仅取前rank行）

            result = torch.nn.functional.embedding(x_seq, w_seq.T)  # 执行Embedding查找
            output[token_offset : token_offset + seq_len, :rank] = (
                scaling_tensor[lora_idx].item() * result
            )  # 将查找结果乘以缩放因子后写入输出

        token_offset += seq_len  # 更新token偏移量

    return output  # 返回最终输出


def sgemm_lora_a_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
) -> torch.Tensor:  # LoRA A矩阵线性层的前向传播，按序列段逐一计算矩阵乘法
    total_seq_len, input_dim = inputs.shape  # 获取总序列长度和输入维度
    if weights.numel() == 0:  # 如果权重为空
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)  # 返回空张量

    num_loras, weight_out_dim, _ = weights.shape  # 获取LoRA数量、权重输出维度和输入维度
    max_rank = weight_out_dim // num_slices  # 计算最大秩（权重输出维度除以切片数）

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=inputs.dtype, device=inputs.device
    )  # 初始化输出张量，形状为(总序列长度, 切片数*最大秩)

    token_offset = 0  # 初始化token偏移量
    for lora_idx, seq_len in zip(weight_indices, seg_len_tensor):  # 遍历每个适配器及其对应的序列长度
        if seq_len == 0:  # 如果当前序列长度为0，跳过
            continue

        rank = lora_ranks[lora_idx]  # 获取当前适配器的LoRA秩
        if rank > 0:  # 如果秩大于0

            x_seq = inputs[token_offset : token_offset + seq_len]  # 获取当前序列段的输入
            w_seq = weights[lora_idx, : num_slices * rank]  # 获取当前适配器的权重（仅取前num_slices*rank行）

            output[token_offset : token_offset + seq_len, : num_slices * rank].addmm_(
                x_seq,
                w_seq.T,
                beta=0,
                alpha=scaling_tensor[lora_idx].item(),
            )  # 执行矩阵乘法并乘以缩放因子，写入输出

        token_offset += seq_len  # 更新token偏移量

    return output  # 返回最终输出


def sgemm_lora_b_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:  # LoRA B矩阵的前向传播，支持多切片输出，可累加到基础输出上
    total_seq_len, _ = inputs.shape  # 获取总序列长度
    num_loras, weight_out_dim, _ = weights.shape  # 获取LoRA数量、权重输出维度和输入维度
    total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0  # 计算总输出维度

    if weights.numel() == 0:  # 如果权重为空
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )  # 返回空张量

    num_slices = len(slice_offsets) - 1  # 计算切片数量

    if base_output is not None:  # 如果提供了基础输出
        output = base_output  # 直接使用基础输出作为输出张量
    else:
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )  # 否则初始化为零张量

    token_offset = 0  # 初始化token偏移量
    for lora_idx, seq_len in zip(weight_indices, seg_len_tensor):  # 遍历每个适配器及其对应的序列长度
        if seq_len == 0:  # 如果当前序列长度为0，跳过
            continue

        rank = lora_ranks[lora_idx]  # 获取当前适配器的LoRA秩
        if rank > 0:  # 如果秩大于0

            for slice_idx in range(num_slices):  # 遍历每个切片
                slice_start_input = slice_idx * rank  # 计算当前切片在输入维度的起始位置
                slice_end_input = (slice_idx + 1) * rank  # 计算当前切片在输入维度的结束位置

                slice_start_output = slice_offsets[slice_idx]  # 获取当前切片在输出维度的起始偏移
                slice_end_output = slice_offsets[slice_idx + 1]  # 获取当前切片在输出维度的结束偏移

                x_slice = inputs[
                    token_offset : token_offset + seq_len,
                    slice_start_input:slice_end_input,
                ]  # (seq_len, rank) 获取当前切片的输入 # (seq_len, rank) 获取当前切片的输入
                w_slice = weights[
                    lora_idx, slice_start_output:slice_end_output, :rank
                ]  # (slice_dim, rank) 获取当前切片的权重 # (slice_dim, rank) 获取当前切片的权重

                output[
                    token_offset : token_offset + seq_len,
                    slice_start_output:slice_end_output,
                ].addmm_(x_slice, w_slice.T)  # 执行矩阵乘法并累加到对应输出切片

        token_offset += seq_len  # 更新token偏移量

    return output  # 返回最终输出
