# 分块Embedding LoRA A前向计算的Triton实现
# 提供Triton kernel和对应的Python包装函数，用于高效地按分块方式执行LoRA A矩阵的Embedding查找
import torch  # 导入PyTorch核心库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批次信息工具类


@triton.jit  # Triton JIT编译装饰器
def _chunked_embedding_lora_a_kernel(
    # Pointers to tensors 张量指针
    input_ids,  # 输入token ID指针
    weights,  # LoRA权重指针
    output,  # 输出指针
    # Dimensions 维度参数
    vocab_size,  # 词表大小
    rank,  # LoRA秩
    num_loras,  # LoRA适配器数量
    # Strides 步长参数
    w_stride_0,  # stride for lora index 权重在LoRA索引维度上的步长
    w_stride_1,  # stride for rank 权重在秩维度上的步长
    w_stride_2,  # stride for vocab 权重在词表维度上的步长
    output_stride_0,  # 输出在第一个维度上的步长
    output_stride_1,  # 输出在第二个维度上的步长
    # Chunk info 分块信息
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引指针
    lora_ranks,  # LoRA秩数组指针
    num_segments,  # 段数量
    permutation,  # 排列映射指针
    # Meta-parameters 元参数
    BLOCK_RANK: tl.constexpr,  # 秩维度的块大小（编译时常量）
):
    """
    Embedding lookup for LoRA A weights without support for extra tokens.
    对LoRA A权重执行Embedding查找，不支持额外token

    Each program handles one chunk of tokens across rank dimension
    每个程序处理一个token分块在秩维度上的Embedding查找
    """
    chunk_idx = tl.program_id(axis=0)  # 获取当前程序的块索引
    # If chunk id is larger than actual number of chunks, skip 如果块索引超过实际块数，则跳过
    if chunk_idx >= num_segments:  # 如果块索引超出段数量
        return  # 跳过
    # Load LoRA adapter index for this segment, then look up the rank 加载当前段的LoRA适配器索引，然后查找对应的秩
    lora_index = tl.load(weight_indices + chunk_idx)  # 加载当前段的LoRA适配器索引
    rank_val = tl.load(lora_ranks + lora_index)  # 加载当前适配器的LoRA秩
    # If rank is 0, skip 如果秩为0，则跳过
    if rank_val == 0:  # 如果当前适配器的秩为0
        return  # 跳过
    # for each token in chunk, load embedding across rank dimension 对分块中的每个token，在秩维度上加载Embedding
    chunk_start = tl.load(seg_indptr + chunk_idx)  # 加载当前分块的起始位置
    chunk_end = tl.load(seg_indptr + chunk_idx + 1)  # 加载当前分块的结束位置
    for c in range(chunk_start, chunk_end):  # 遍历分块中的每个token
        s_index = tl.load(permutation + c)  # 通过排列映射获取物理索引
        # Load the token ID 加载token ID
        token_id = tl.load(input_ids + s_index)  # 加载当前token的ID
        # Process in chunks of BLOCK_RANK dimensions 按BLOCK_RANK大小的块在秩维度上处理
        num_blocks = tl.cdiv(rank_val, BLOCK_RANK)  # 计算需要的块数

        for block_id in range(num_blocks):  # 遍历每个块
            rank_offset = tl.arange(0, BLOCK_RANK) + block_id * BLOCK_RANK  # 计算当前块的秩偏移
            rank_mask = rank_offset < rank_val  # 创建秩掩码，防止越界

            # Use regular LoRA A weights 使用常规LoRA A权重
            # weights shape: (num_loras, rank, vocab_size) 权重形状：(LoRA数量, 秩, 词表大小)
            # We need to load weights[lora_index, rank_offset, token_id] 需要加载weights[lora_index, rank_offset, token_id]
            weight_ptr = (
                weights
                + lora_index * w_stride_0
                + rank_offset * w_stride_1
                + token_id * w_stride_2
            )  # 计算权重指针
            emb_values = tl.load(weight_ptr, mask=rank_mask, other=0.0)  # 加载Embedding值，越界处用0填充

            # Write to output 写入输出
            output_ptr = (
                output + s_index * output_stride_0 + rank_offset * output_stride_1
            )  # 计算输出指针
            tl.store(output_ptr, emb_values, mask=rank_mask)  # 将Embedding值存储到输出中


def chunked_embedding_lora_a_forward(
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    vocab_size: int,
) -> torch.Tensor:  # 分块Embedding LoRA A前向计算函数，每个程序处理属于同一适配器的一块Embedding查找工作
    """
    Chunked Forward pass for LoRA A embedding lookup; each program handles one chunk of embedding lookup work
    belonging to the same adapter
    分块LoRA A Embedding查找前向传播；每个程序处理属于同一适配器的一块Embedding查找工作

    Args:
    参数：
        input_ids: (s,) token IDs token ID张量
        weights: (num_loras, rank, vocab_size) LoRA A embedding weights LoRA A Embedding权重
        batch_info: LoRABatchInfo containing batch information 包含批次信息的LoRABatchInfo
        vocab_size: base vocabulary size 基础词表大小

    Returns:
    返回：
        output: (s, rank) embedded features Embedding特征输出
    """
    assert input_ids.is_contiguous()  # 断言输入ID是连续的
    assert weights.is_contiguous()  # 断言权重是连续的
    assert len(input_ids.shape) == 1  # 断言输入ID是一维的
    assert len(weights.shape) == 3  # 断言权重是三维的

    S = input_ids.shape[0]  # 获取总token数
    num_loras = weights.shape[0]  # 获取LoRA适配器数量
    rank = weights.shape[1]  # 获取LoRA秩

    # Block size for rank dimension 秩维度的块大小
    BLOCK_RANK = 128  # 设置秩维度的块大小为128
    num_segments = batch_info.num_segments  # 获取段数量
    # 1D Grid: one program per chunk of embedding lookup work 一维网格：每个Embedding查找工作块一个程序
    grid = (batch_info.bs if batch_info.use_cuda_graph else num_segments,)  # 根据是否使用CUDA Graph设置网格大小
    output = torch.zeros((S, rank), device=input_ids.device, dtype=weights.dtype)  # 初始化输出张量

    _chunked_embedding_lora_a_kernel[grid](  # 启动Triton kernel
        input_ids,  # 输入token ID
        weights,  # LoRA权重
        output,  # 输出张量
        vocab_size,  # 词表大小
        rank,  # LoRA秩
        num_loras,  # LoRA适配器数量
        weights.stride(0),  # 权重在第0维的步长
        weights.stride(1),  # 权重在第1维的步长
        weights.stride(2),  # 权重在第2维的步长
        output.stride(0),  # 输出在第0维的步长
        output.stride(1),  # 输出在第1维的步长
        batch_info.seg_indptr,  # 段索引指针
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩数组
        batch_info.num_segments,  # 段数量
        batch_info.permutation,  # 排列映射
        BLOCK_RANK,  # 秩维度的块大小
    )

    return output  # 返回最终输出
