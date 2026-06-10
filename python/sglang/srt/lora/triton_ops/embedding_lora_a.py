# LoRA嵌入层A矩阵查找的Triton内核实现
# 实现LoRA A嵌入权重的查找操作，支持额外token的嵌入处理

import torch  # 导入PyTorch张量库
import triton  # 导入Triton编译框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批次信息数据类


@triton.jit  # Triton JIT编译装饰器
def _embedding_lora_a_kernel(  # LoRA A嵌入查找内核函数
    # Pointers to tensors  # 张量指针
    input_ids,  # 输入token ID指针
    weights,  # LoRA A权重指针
    output,  # 输出张量指针
    extra_embeddings,  # 额外嵌入权重指针
    # Dimensions  # 维度参数
    vocab_size,  # 词汇表大小
    rank,  # LoRA秩
    num_loras,  # LoRA适配器数量
    # Strides  # 步长参数
    w_stride_0,  # stride for lora index  # 适配器索引维度的步长
    w_stride_1,  # stride for rank  # 秩维度的步长
    w_stride_2,  # stride for vocab  # 词汇表维度的步长
    output_stride_0,  # 输出第0维步长
    output_stride_1,  # 输出第1维步长
    extra_emb_stride_0,  # stride for lora index  # 额外嵌入适配器索引维度的步长
    extra_emb_stride_1,  # stride for token  # 额外嵌入token维度的步长
    extra_emb_stride_2,  # stride for hidden dim (= rank for extra embeddings)  # 额外嵌入隐藏维度的步长（等于秩）
    # Batch info  # 批次信息
    seg_lens,  # 段长度指针
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引指针
    lora_ranks,  # LoRA秩指针
    # Meta-parameters  # 元参数
    BLOCK_RANK: tl.constexpr,  # 秩维度的块大小
    HAS_EXTRA_EMBEDDINGS: tl.constexpr,  # 是否有额外嵌入的标志
):
    """
    Embedding lookup for LoRA A weights with support for extra tokens.
    LoRA A权重的嵌入查找，支持额外token。

    Each program handles one token across a block of rank dimensions.
    每个程序处理一个token在秩维度的一个块上的查找。
    Grid: (cdiv(max_len, 1), bs) - one program per token in each batch
    网格: (cdiv(max_len, 1), bs) - 每个批次中每个token一个程序
    """
    batch_id = tl.program_id(axis=1)  # 获取批次维度的程序ID
    token_idx = tl.program_id(axis=0)  # 获取token维度的程序ID

    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次对应的LoRA适配器索引
    rank_val = tl.load(lora_ranks + w_index)  # 加载当前适配器的LoRA秩

    # If rank is 0, skip
    # 如果秩为0，跳过
    if rank_val == 0:  # 秩为0时无需计算
        return  # 直接返回

    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    seg_len = tl.load(seg_lens + batch_id)  # 加载当前段的长度

    # Check if this token is within the segment
    # 检查当前token是否在段内
    if token_idx >= seg_len:  # token索引超出段长度
        return  # 超出范围则返回

    # Load the token ID
    # 加载token ID
    token_id = tl.load(input_ids + seg_start + token_idx)  # 从输入中加载当前token的ID

    # Process in chunks of BLOCK_RANK dimensions
    # 按BLOCK_RANK大小的块分块处理秩维度
    num_blocks = tl.cdiv(rank_val, BLOCK_RANK)  # 计算秩维度需要的块数

    for block_id in range(num_blocks):  # 遍历每个秩维度块
        rank_offset = tl.arange(0, BLOCK_RANK) + block_id * BLOCK_RANK  # 计算当前块的秩偏移量
        rank_mask = rank_offset < rank_val  # 计算秩维度的有效掩码

        # Check if this is an extra token
        # 检查是否为额外token
        is_extra_token = token_id >= vocab_size  # token ID超出词汇表大小则为额外token

        if HAS_EXTRA_EMBEDDINGS and is_extra_token:  # 如果有额外嵌入且当前token是额外token
            # Use extra embeddings
            # 使用额外嵌入
            extra_token_id = token_id - vocab_size  # 计算额外token的相对ID
            extra_emb_ptr = (  # 计算额外嵌入的指针位置
                extra_embeddings  # 额外嵌入基址
                + w_index * extra_emb_stride_0  # 适配器偏移
                + extra_token_id * extra_emb_stride_1  # token偏移
                + rank_offset * extra_emb_stride_2  # 秩维度偏移
            )
            emb_values = tl.load(extra_emb_ptr, mask=rank_mask, other=0.0)  # 加载额外嵌入值
        else:  # 普通token
            # Use regular LoRA A weights
            # 使用常规LoRA A权重
            # weights shape: (num_loras, rank, vocab_size)
            # 权重形状: (num_loras, rank, vocab_size)
            # We need to load weights[w_index, rank_offset, token_id]
            # 需要加载weights[w_index, rank_offset, token_id]
            token_id_clamped = tl.minimum(token_id, vocab_size - 1)  # 将token ID限制在词汇表范围内，防止越界
            weight_ptr = (  # 计算权重指针位置
                weights  # 权重基址
                + w_index * w_stride_0  # 适配器偏移
                + rank_offset * w_stride_1  # 秩维度偏移
                + token_id_clamped * w_stride_2  # 词汇表维度偏移
            )
            emb_values = tl.load(weight_ptr, mask=rank_mask, other=0.0)  # 加载嵌入值

        # Write to output
        # 写入输出
        output_ptr = (  # 计算输出指针位置
            output  # 输出基址
            + (seg_start + token_idx) * output_stride_0  # token行偏移
            + rank_offset * output_stride_1  # 秩维度偏移
        )
        tl.store(output_ptr, emb_values, mask=rank_mask)  # 将嵌入值存储到输出


def embedding_lora_a_fwd(  # LoRA A嵌入查找前向传播函数
    input_ids: torch.Tensor,  # 输入token ID张量
    weights: torch.Tensor,  # LoRA A权重张量
    batch_info: LoRABatchInfo,  # LoRA批次信息
    vocab_size: int,  # 基础词汇表大小
    extra_embeddings: torch.Tensor = None,  # 额外token嵌入权重，默认为None
) -> torch.Tensor:  # 返回嵌入输出张量
    """
    Forward pass for LoRA A embedding lookup.
    LoRA A嵌入查找的前向传播。

    Args:
        input_ids: (s,) token IDs
        input_ids: (s,) token ID张量
        weights: (num_loras, rank, vocab_size) LoRA A embedding weights
        weights: (num_loras, rank, vocab_size) LoRA A嵌入权重
        batch_info: LoRABatchInfo containing batch information
        batch_info: LoRABatchInfo，包含批次信息
        vocab_size: base vocabulary size
        vocab_size: 基础词汇表大小
        extra_embeddings: (num_loras, num_extra_tokens, rank) extra token embeddings
        extra_embeddings: (num_loras, num_extra_tokens, rank) 额外token嵌入

    Returns:
        output: (s, rank) embedded features
        output: (s, rank) 嵌入特征
    """
    assert input_ids.is_contiguous()  # 断言input_ids在内存中连续
    assert weights.is_contiguous()  # 断言weights在内存中连续
    assert len(input_ids.shape) == 1  # 断言input_ids是1维张量
    assert len(weights.shape) == 3  # 断言weights是3维张量

    S = input_ids.shape[0]  # 总序列长度
    num_loras = weights.shape[0]  # LoRA适配器数量
    rank = weights.shape[1]  # LoRA秩
    vocab_size_weights = weights.shape[2]  # 权重中的词汇表维度

    # Block size for rank dimension
    # 秩维度的块大小
    BLOCK_RANK = 128  # 设置秩维度的块大小为128

    has_extra_embeddings = extra_embeddings is not None  # 检查是否有额外嵌入

    if has_extra_embeddings:  # 如果有额外嵌入
        assert extra_embeddings.is_contiguous()  # 断言extra_embeddings在内存中连续
        extra_emb_stride = (  # 获取额外嵌入的步长
            extra_embeddings.stride(0),  # 适配器维度的步长
            extra_embeddings.stride(1),  # token维度的步长
            extra_embeddings.stride(2),  # 隐藏维度的步长
        )
    else:  # 没有额外嵌入
        # Create dummy tensor to satisfy Triton
        # 创建虚拟张量以满足Triton要求
        extra_embeddings = torch.empty(  # 创建空的虚拟张量
            (1, 1, 1), device=input_ids.device, dtype=weights.dtype  # 形状为(1,1,1)，与输入同设备和数据类型
        )
        extra_emb_stride = (1, 1, 1)  # 设置虚拟步长

    # Grid: one program per token in each batch segment
    # 网格：每个批次段中每个token一个程序
    grid = (  # 定义内核启动的网格大小
        batch_info.max_len,  # 最大序列长度（token维度）
        batch_info.bs,  # 批次数
    )

    output = torch.zeros((S, rank), device=input_ids.device, dtype=weights.dtype)  # 创建输出张量，初始化为零

    _embedding_lora_a_kernel[grid](  # 启动LoRA A嵌入查找内核
        input_ids,  # 输入token ID
        weights,  # LoRA A权重
        output,  # 输出张量
        extra_embeddings,  # 额外嵌入权重
        vocab_size,  # 词汇表大小
        rank,  # LoRA秩
        num_loras,  # 适配器数量
        weights.stride(0),  # 权重第0维步长
        weights.stride(1),  # 权重第1维步长
        weights.stride(2),  # 权重第2维步长
        output.stride(0),  # 输出第0维步长
        output.stride(1),  # 输出第1维步长
        extra_emb_stride[0],  # 额外嵌入第0维步长
        extra_emb_stride[1],  # 额外嵌入第1维步长
        extra_emb_stride[2],  # 额外嵌入第2维步长
        batch_info.seg_lens,  # 段长度
        batch_info.seg_indptr,  # 段索引
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        BLOCK_RANK,  # 秩维度的块大小
        has_extra_embeddings,  # 是否有额外嵌入的标志
    )

    return output  # 返回嵌入输出
