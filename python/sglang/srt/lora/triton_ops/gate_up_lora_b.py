# LoRA gate_up投影B矩阵的Triton内核实现
# 将LoRA A投影的结果与LoRA B权重相乘，实现从低秩空间到输出维度的映射
# 将gate和up两个投影合并到单个内核中执行，提高计算效率

import torch  # 导入PyTorch张量库
import triton  # 导入Triton编译框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions  # 导入token位置解析函数
from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批次信息数据类


@triton.jit  # Triton JIT编译装饰器
def _gate_up_lora_b_kernel(  # gate_up LoRA B投影内核函数
    # Pointers to matrices  # 矩阵指针
    x,  # 输入张量指针（LoRA A投影结果）
    weights,  # LoRA B权重指针
    output,  # 输出张量指针
    # Parameters of size  # 大小参数
    K,  # K = R  # K维度等于LoRA秩
    output_dim,  # 输出维度大小
    # Strides  # 步长参数
    x_stride_0,  # 输入第0维步长
    x_stride_1,  # 输入第1维步长
    w_stride_0,  # 权重第0维步长
    w_stride_1,  # 权重第1维步长
    w_stride_2,  # 权重第2维步长
    output_stride_0,  # 输出第0维步长
    output_stride_1,  # 输出第1维步长
    # Information on sequence lengths,ranks and weight id  # 序列长度、秩和权重ID信息
    seg_lens,  # 段长度指针
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引指针
    lora_ranks,  # LoRA秩指针
    sorted_token_ids,  # 排序后的token ID指针
    # Meta parameters  # 元参数
    SORTED_BY_ADAPTER: tl.constexpr,  # 是否按适配器排序的标志
    BLOCK_S: tl.constexpr,  # 序列维度的块大小
    BLOCK_N: tl.constexpr,  # 输出维度的块大小
    BLOCK_K: tl.constexpr,  # K维度的块大小
    # For fused output scaling  # 用于融合输出缩放
    scalings,  # 缩放因子指针
):
    """
    This kernel packs 2 sgemms (gate/up) into a single kernel. The multiplication
    results are accumulated into the output tensor.
    该内核将2个SGEMM（gate/up）合并到单个内核中。乘法结果累加到输出张量中。

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).
    当序列的秩为0时，该内核本质上是空操作，遵循PyTorch中
    形状(m,0)和(0,n)两个矩阵的乘积为形状(m,n)的全零矩阵的约定。

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, 2 * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank.
        x (Tensor): 输入张量，即LoRA A投影的结果。
            形状: (s, 2 * K)，其中s是批次中所有序列长度之和，K是最大LoRA秩。
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, 2 * output_dim, K).
        weights (Tensor): 所有适配器的LoRA B权重。
            形状: (num_lora, 2 * output_dim, K)。
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, 2 * output_dim).
        output (Tensor): 存储结果的输出张量。
            形状: (s, 2 * output_dim)。
    """
    # output_dim >> K  # 输出维度远大于K（LoRA秩）

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # gate_up_id decides which of gate or up (0: gate, 1: up)
    # 当前块计算batch_id对应的序列，从x的seg_start行开始，长度为seg_len。
    # gate_up_id决定计算gate还是up投影（0: gate, 1: up）
    batch_id = tl.program_id(axis=2)  # 获取批次维度的程序ID
    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次对应的LoRA适配器索引
    rank = tl.load(lora_ranks + w_index)  # 加载当前适配器的LoRA秩

    # If rank is 0, this kernel is a no-op.
    # 如果秩为0，该内核为空操作。
    if rank == 0:  # LoRA秩为0
        return  # 直接返回，不执行计算

    gate_up_id = tl.program_id(axis=1)  # 获取gate/up维度的程序ID（0=gate, 1=up）
    pid = tl.program_id(axis=0)  # 获取主程序ID
    seg_len = tl.load(seg_lens + batch_id)  # 加载当前段的长度
    if seg_len == 0:  # 如果段长度为0
        return  # 直接返回
    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    n_start = gate_up_id * output_dim  # offset on output dim  # 输出维度上的偏移量
    scaling = tl.load(scalings + w_index)  # 加载当前适配器的缩放因子

    # Adjust K (rank) according to the specific LoRA adapter
    # 根据特定LoRA适配器调整K（秩）
    K = tl.minimum(K, rank)  # 取K和rank的最小值作为有效秩

    # The tile in output matrix will have (pid_s, pid_n) as id
    # 输出矩阵中的块将具有(pid_s, pid_n)作为标识
    num_pid_n = tl.cdiv(output_dim, BLOCK_N)  # N维度上的块数
    pid_s = pid // num_pid_n  # 序列维度的块ID
    pid_n = pid % num_pid_n  # 输出维度的块ID
    if pid_s * BLOCK_S >= seg_len:  # 如果当前块超出段长度
        return  # 直接返回

    # Create pointers for the first block of x and weights
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    # 为x和权重的第一个块创建指针
    # 指针将随着K方向的移动而前进并累加
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S  # 序列维度的偏移量
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 输出维度的偏移量
    k_offset = tl.arange(0, BLOCK_K)  # K维度的偏移量

    s_physical = _resolve_token_positions(  # 解析物理token位置
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER  # 传入排序token ID、段起始、偏移和排序标志
    )
    x_ptrs = (  # 计算输入x的指针位置
        x  # 输入基址
        + (gate_up_id * K) * x_stride_1  # gate/up在秩维度上的偏移
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)  # 行偏移*行步长 + K偏移*列步长
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (  # 计算权重指针位置，定位到对应适配器和gate/up偏移
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1  # K维偏移*K步长 + N维偏移*N步长
    )

    # Iterate to compute the block in output matrix
    # 迭代计算输出矩阵中的块
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)  # 初始化部分和为零
    for k in range(0, tl.cdiv(K, BLOCK_K)):  # 沿K维度迭代
        x_tile = tl.load(  # 加载输入x的一个小块
            x_ptrs,  # x的指针位置
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),  # 序列和K维掩码
            other=0.0,  # 越界填充0
        )
        w_tile = tl.load(  # 加载权重的一个小块
            w_ptrs,  # 权重的指针位置
            mask=(k_offset[:, None] < K - k * BLOCK_K)  # K维掩码
            & (n_offset[None, :] < output_dim),  # N维掩码
            other=0.0,  # 越界填充0
        )
        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

        x_ptrs += BLOCK_K * x_stride_1  # 移动x指针到下一个K块
        w_ptrs += BLOCK_K * w_stride_2  # 移动权重指针到下一个K块

    # Store result to output matrix
    # 将结果存储到输出矩阵
    partial_sum *= scaling  # 乘以缩放因子
    partial_sum = partial_sum.to(x.dtype.element_ty)  # 转换为输入数据类型
    output_ptr = (  # 计算输出指针位置
        output  # 输出基址
        + n_start * output_stride_1  # gate/up在输出维度上的偏移
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)  # 行偏移和列偏移
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < output_dim)  # 计算输出掩码
    partial_sum += tl.load(output_ptr, mask=output_mask)  # 将部分和与输出中已有值相加（累加到base_output）
    tl.store(output_ptr, partial_sum, mask=output_mask)  # 存储结果到输出


def gate_up_lora_b_fwd(  # gate_up LoRA B投影前向传播函数
    x: torch.Tensor,  # 输入张量（LoRA A投影结果）
    gate_up_lora_b: torch.Tensor,  # LoRA B权重张量
    batch_info: LoRABatchInfo,  # LoRA批次信息
    output_dim: int,  # 输出维度大小
    base_output: torch.Tensor = None,  # 基础输出张量（可选），用于累加
) -> torch.Tensor:  # 返回输出张量

    # x: (s, 2 * r)  # x: 输入，形状为(总序列长度, 2 * 秩)
    # gate_up_lora_b: (num_lora, 2 * output_dim, r)  # gate_up_lora_b: 权重，形状为(适配器数, 2*输出维度, 秩)
    # output: (s, 2 * output_dim)  # output: 输出，形状为(总序列长度, 2 * 输出维度)

    # Compute lora_output with shape (s, output_dim) as follows:
    # 计算形状为(s, output_dim)的lora_output如下：
    # lora_output[:, :output_dim] = sgemm(x[:, :r], gate_up_lora_b[:, :output_dim, :])
    # lora_output[:, :output_dim] = sgemm(x[:, :r], gate_up_lora_b[:, :output_dim, :])  # gate投影
    # lora_output[:, output_dim:]
    #      = sgemm(x[:, r:], gate_up_lora_b[:, output_dim:, :])
    # lora_output[:, output_dim:]
    #      = sgemm(x[:, r:], gate_up_lora_b[:, output_dim:, :])  # up投影

    # Get dims  # 获取维度
    s = x.shape[0]  # 总序列长度
    input_dim = x.shape[1]  # 输入维度
    r = gate_up_lora_b.shape[-1]  # LoRA秩
    assert input_dim == 2 * r  # 断言输入维度等于2倍秩

    BLOCK_S = 16  # 序列维度的块大小
    BLOCK_R = 16  # 秩维度的块大小
    BLOCK_OUT = 64  # 输出维度的块大小

    grid_b = (  # 定义内核启动的网格大小
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(output_dim, BLOCK_OUT),  # 序列块数 * 输出块数
        2,  # this dimension decides current block computes on gate or up proj  # 此维度决定当前块计算gate还是up投影
        batch_info.bs,  # 批次数
    )

    if base_output is None:  # 如果没有提供基础输出
        output = torch.zeros((s, 2 * output_dim), device=x.device, dtype=x.dtype)  # 创建全零输出张量
    else:  # 如果提供了基础输出
        output = base_output  # 直接使用基础输出（结果将累加到其中）

    sorted_by_adapter = batch_info.permutation is not None  # 检查是否按适配器排序
    _gate_up_lora_b_kernel[grid_b](  # 启动gate_up LoRA B投影内核
        x,  # 输入张量
        gate_up_lora_b,  # LoRA B权重
        output,  # 输出张量
        r,  # LoRA秩
        output_dim,  # 输出维度
        x.stride(0),  # 输入第0维步长
        x.stride(1),  # 输入第1维步长
        gate_up_lora_b.stride(0),  # 权重第0维步长
        gate_up_lora_b.stride(1),  # 权重第1维步长
        gate_up_lora_b.stride(2),  # 权重第2维步长
        output.stride(0),  # 输出第0维步长
        output.stride(1),  # 输出第1维步长
        batch_info.seg_lens,  # 段长度
        batch_info.seg_indptr,  # 段索引
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        batch_info.permutation,  # 排列映射
        sorted_by_adapter,  # 是否按适配器排序
        BLOCK_S,  # 序列维度块大小
        BLOCK_OUT,  # 输出维度块大小
        BLOCK_R,  # 秩维度块大小
        batch_info.scalings,  # 缩放因子
    )

    return output  # 返回输出张量
