# 文件说明：LoRA A矩阵投影的分段批量矩阵乘法Triton内核
# 本文件实现了LoRA A矩阵的分段批量矩阵乘法（SGEMM），
# 对输入激活值与LoRA A权重进行分段批量矩阵乘法，
# 输出结果为低秩中间表示，供LoRA B投影使用。

import torch  # 导入PyTorch张量库
import triton  # 导入Triton JIT编译框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions  # 导入token位置解析工具
from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批量信息数据类


@triton.jit  # Triton JIT编译装饰器
def _sgemm_lora_a_kernel(  # LoRA A矩阵的分段批量矩阵乘法内核
    # Pointers to matrices 矩阵指针
    x,  # 输入张量指针
    weights,  # 权重张量指针
    output,  # 输出张量指针
    # Matrix dimensions 矩阵维度
    N,  # stack_num * r  # 堆叠数乘以LoRA秩
    K,  # input_dim  # 输入维度
    stack_num,  # 堆叠数（qkv=3, gate_up=2, 其他=1）
    # Strides 步幅参数
    x_stride_0,  # x的第0维步幅
    x_stride_1,  # x的第1维步幅
    w_stride_0,  # weights的第0维步幅
    w_stride_1,  # weights的第1维步幅
    w_stride_2,  # weights的第2维步幅
    output_stride_0,  # output的第0维步幅
    output_stride_1,  # output的第1维步幅
    # Information on sequence lengths,ranks and weight id 序列长度、秩和权重ID信息
    seg_lens,  # 各段的长度
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引
    lora_ranks,  # LoRA秩
    sorted_token_ids,  # 排序后的token ID
    # Meta parameters 元参数
    SORTED_BY_ADAPTER: tl.constexpr,  # 是否按适配器排序
    BLOCK_S: tl.constexpr,  # 序列维度的块大小
    BLOCK_N: tl.constexpr,  # 输出维度的块大小
    BLOCK_K: tl.constexpr,  # 收缩维度的块大小
):
    """
    Computes a segmented batched matrix multiplication for the LoRA A matrix.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * stack_num]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA 'A' weights for all available adapters,
            with shape `(num_lora, N, K)`.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    # 当前块计算batch_id对应的序列，从x的第seg_start行开始，长度为seg_len
    batch_id = tl.program_id(axis=1)  # 获取批次ID（axis=1）
    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次的权重索引
    rank = tl.load(lora_ranks + w_index)  # 加载当前LoRA适配器的秩

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    # 如果秩为0，本内核不执行任何操作，因为输出总是平凡正确的
    if rank == 0:
        return  # 直接返回

    pid = tl.program_id(axis=0)  # 获取程序ID（axis=0）
    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    seg_len = tl.load(seg_lens + batch_id)  # 加载当前段的长度
    if seg_len == 0:  # 如果段长度为0
        return  # 直接返回

    # Adjust N (stack_num * max_rank) according to the specific LoRA adapter
    # 根据特定LoRA适配器调整N（stack_num * max_rank）
    N = tl.minimum(N, rank * stack_num)  # 限制输出维度

    # The tile in output matrix will have (pid_s, pid_n) as id
    # 输出矩阵中的分片将以(pid_s, pid_n)为ID
    num_pid_n = tl.cdiv(N, BLOCK_N)  # 计算N维度的程序数
    pid_s = pid // num_pid_n  # 计算序列维度的程序ID
    pid_n = pid % num_pid_n  # 计算输出维度的程序ID
    if pid_s * BLOCK_S >= seg_len:  # 如果序列维度超出段长度
        return  # 直接返回

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    # 为x和weights[batch_id]的第一个块创建指针
    # 指针将在K方向移动时前进并累加
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S  # 序列维度的偏移量
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 输出维度的偏移量
    k_offset = tl.arange(0, BLOCK_K)  # 收缩维度的偏移量
    s_physical = _resolve_token_positions(  # 解析物理token位置
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)  # 计算x的指针
    w_ptrs = (weights + w_index * w_stride_0) + (  # 计算权重指针
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    # 迭代计算输出矩阵中的块
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)  # 初始化部分和
    for k in range(0, tl.cdiv(K, BLOCK_K)):  # 遍历收缩维度的块
        x_tile = tl.load(  # 加载x分片
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),  # 掩码
            other=0.0,  # 掩码外填充0
        )
        w_tile = tl.load(  # 加载权重分片
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),  # 掩码
            other=0.0,  # 掩码外填充0
        )
        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

        x_ptrs += BLOCK_K * x_stride_1  # 推进x指针
        w_ptrs += BLOCK_K * w_stride_2  # 推进权重指针

    # Store result to output matrix
    # 将结果存储到输出矩阵
    partial_sum = partial_sum.to(x.dtype.element_ty)  # 转换回输入数据类型
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)  # 输出掩码
    output_ptr = output + (  # 计算输出指针
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    tl.store(output_ptr, partial_sum, mask=output_mask)  # 将结果写入输出


def sgemm_lora_a_fwd(  # LoRA A矩阵的前向计算：分段批量矩阵乘法
    x: torch.Tensor,  # 输入：(s, input_dim)
    weights: torch.Tensor,  # LoRA A权重：(num_lora, stack_num * r, input_dim)
    batch_info: LoRABatchInfo,  # LoRA批量信息
    stack_num: int = 1,  # 堆叠数（qkv=3, gate_up=2, 其他=1）
) -> torch.Tensor:  # 返回：(s, stack_num * r)
    # x: (s, input_dim)
    # weights: (num_lora, stack_num * r, input_dim)
    # output: (s, stack_num * r)
    # stack_num: run_qkv_lora: 3, run_gate_up_lora: 2
    # when called by run_qkv_lora, the weights.shape[-2] will be 3 * r
    # input_dim is much larger than r
    # stack_num: run_qkv_lora时为3, run_gate_up_lora时为2
    # 当被run_qkv_lora调用时，weights.shape[-2]将是3 * r
    # input_dim远大于r

    assert x.is_contiguous()  # 断言x是连续的
    assert weights.is_contiguous()  # 断言weights是连续的
    assert len(x.shape) == 2  # 断言x是2维的
    assert len(weights.shape) == 3  # 断言weights是3维的

    S = x.shape[0]  # 总token数
    R = weights.shape[-2]  # 输出维度（stack_num * rank）
    K = weights.shape[-1]  # 输入维度
    assert x.shape[-1] == K  # 断言x的最后一维等于K

    # Block shapes 块形状
    BLOCK_S = 16  # 序列维度的块大小
    BLOCK_K = 256  # 收缩维度的块大小（input_dim较大，使用大块）
    BLOCK_R = 16  # 输出维度的块大小（rank较小，使用小块）

    grid = (  # 计算网格大小
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(R, BLOCK_R),
        batch_info.bs,  # 批次大小
    )

    sorted_by_adapter = batch_info.permutation is not None  # 判断是否按适配器排序

    output = torch.empty((S, R), device=x.device, dtype=x.dtype)  # 分配输出张量
    _sgemm_lora_a_kernel[grid](  # 启动LoRA A内核
        x,  # 输入
        weights,  # 权重
        output,  # 输出
        R,  # 输出维度N
        K,  # 输入维度K
        stack_num,  # 堆叠数
        x.stride(0),  # x的第0维步幅
        x.stride(1),  # x的第1维步幅
        weights.stride(0),  # weights的第0维步幅
        weights.stride(1),  # weights的第1维步幅
        weights.stride(2),  # weights的第2维步幅
        output.stride(0),  # output的第0维步幅
        output.stride(1),  # output的第1维步幅
        batch_info.seg_lens,  # 各段长度
        batch_info.seg_indptr,  # 段索引指针
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        batch_info.permutation,  # 排序后的token ID
        sorted_by_adapter,  # 是否按适配器排序
        BLOCK_S,  # 序列维度块大小
        BLOCK_R,  # 输出维度块大小
        BLOCK_K,  # 收缩维度块大小
    )
    return output  # 返回输出张量
