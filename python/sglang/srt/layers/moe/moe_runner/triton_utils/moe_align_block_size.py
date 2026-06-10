# MoE块大小对齐工具文件
# 本文件实现MoE推理中token按专家分组的块大小对齐操作，
# 确保每个专家处理的token数量能被block_size整除，
# 以适配块矩阵乘法的要求。

from __future__ import annotations  # 启用延迟类型注解评估

from typing import Tuple  # 导入元组类型提示

import torch  # 导入PyTorch深度学习框架
import triton  # 导入Triton GPU编程框架

from sglang.srt.utils import is_cuda, is_hip, is_musa, is_xpu  # 导入平台检测工具函数

_is_cuda = is_cuda()  # 检测当前是否为CUDA平台
_is_hip = is_hip()  # 检测当前是否为HIP(AMD ROCm)平台
_is_xpu = is_xpu()  # 检测当前是否为XPU(Intel)平台
_is_musa = is_musa()  # 检测当前是否为MUSA(摩尔线程)平台

if _is_cuda or _is_hip or _is_xpu or _is_musa:  # 如果是支持的GPU平台
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size  # 导入SGLang内核的MoE块对齐函数


def moe_align_block_size(  # MoE块大小对齐函数
    topk_ids: torch.Tensor, block_size: int, num_experts: int  # TopK专家ID、块大小、专家总数
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回排序token ID、专家ID、填充后token数
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.

    对齐各专家的token分布，使其与块矩阵乘法的块大小兼容。

    参数：
    - topk_ids: 形状为[total_tokens, top_k]的张量，表示每个token的TopK专家索引。
    - block_size: 块矩阵乘法中使用的块大小。
    - num_experts: 专家总数。

    返回：
    - sorted_token_ids: 按分配的专家排序的token索引张量。
    - expert_ids: 指示每个块对应专家索引的张量。
    - num_tokens_post_padded: 填充后的token总数，确保能被block_size整除。

    此函数对每个专家需要处理的token数量进行填充，使其能被block_size整除。
    填充确保在块矩阵乘法中维度正确对齐。

    示例：
    给定 topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]]，
    block_size = 4，num_experts = 4：
    - 初始有12个token（重复top_k次后）和4个专家，每个专家需处理3个token。
    - 由于block_size为4，为每个专家填充1个token。
    - 首先，展平topk_ids为 [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3]。
    - 然后为每个块追加填充token [12, 12, 12, 12]。
    - 按专家索引排序后，得到token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]。
        token 12是不存在的（填充），在后续矩阵乘法中被忽略。
    - 填充确保token总数能被block_size整除，以进行正确的块矩阵操作。
    """
    if topk_ids.numel() < num_experts + 1:  # 如果token总数小于专家数+1
        max_num_tokens_padded = topk_ids.numel() * block_size  # 最坏情况：每个token单独占一块
    else:  # 正常情况
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)  # 预估填充后的最大token数
    sorted_ids = torch.empty(  # 分配排序后token ID张量
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device  # 形状为1D，int32类型
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)  # 计算最大M维度块数
    expert_ids = torch.empty(  # 分配专家ID张量
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device  # 形状为1D，int32类型
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)  # 分配填充后token数张量

    # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
    # 在EP（专家并行）中，被过滤专家的expert_ids为-1。总共有num_experts + 1个ID。
    cumsum_buffer = torch.empty(  # 分配累积和缓冲区
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device  # 大小为专家数+2
    )

    sgl_moe_align_block_size(  # 调用SGLang内核的MoE块对齐函数
        topk_ids,  # TopK专家ID
        num_experts + 1,  # 专家数+1（包括过滤标记）
        block_size,  # 块大小
        sorted_ids,  # 排序后token ID（输出）
        expert_ids,  # 专家ID（输出）
        num_tokens_post_pad,  # 填充后token数（输出）
        cumsum_buffer,  # 累积和缓冲区
        True,  # 是否填充的标志
    )
    return sorted_ids, expert_ids, num_tokens_post_pad  # 返回排序token ID、专家ID和填充后token数
