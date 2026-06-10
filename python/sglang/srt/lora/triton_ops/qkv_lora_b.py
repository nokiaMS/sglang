# 文件说明：QKV LoRA B矩阵投影的Triton内核
# 本文件实现了将3个SGEMM（Q/K/V）打包到单个Triton内核中的计算，
# 对LoRA B权重矩阵与LoRA A投影的中间结果进行分段批量矩阵乘法，
# 并将结果累加到输出张量中。

import torch  # 导入PyTorch张量库
import triton  # 导入Triton JIT编译框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions  # 导入token位置解析工具
from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批量信息数据类


@triton.jit  # Triton JIT编译装饰器
def _qkv_lora_b_kernel(  # QKV LoRA B内核：将3个sgemm（q/k/v）打包到单个内核中
    # Pointers to matrices 矩阵指针
    x,  # 输入张量指针
    weights,  # 权重张量指针
    output,  # 输出张量指针
    # Parameters of size 尺寸参数
    K,  # K = R  # K等于LoRA秩
    max_qkv_out_dim,  # max(output_q_dim, output_kv_dim)  # 最大QKV输出维度
    # Strides 步幅参数
    x_stride_0,  # x的第0维步幅
    x_stride_1,  # x的第1维步幅
    w_stride_0,  # weights的第0维步幅
    w_stride_1,  # weights的第1维步幅
    w_stride_2,  # weights的第2维步幅
    output_stride_0,  # output的第0维步幅
    output_stride_1,  # output的第1维步幅
    # Information on sequence lengths and weight id 序列长度和权重ID信息
    seg_lens,  # 各段的长度
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引
    lora_ranks,  # LoRA秩
    # Offsets of q/k/v slice on output dimension 输出维度上q/k/v切片的偏移量
    n_offs,  # QKV在输出维度上的偏移量
    sorted_token_ids,  # 排序后的token ID
    # Meta parameters 元参数
    SORTED_BY_ADAPTER: tl.constexpr,  # 是否按适配器排序
    BLOCK_S: tl.constexpr,  # 序列维度的块大小
    BLOCK_N: tl.constexpr,  # 输出维度的块大小
    BLOCK_K: tl.constexpr,  # 收缩维度的块大小
    # For fused output scaling 用于融合输出缩放
    scalings,  # LoRA缩放因子
):
    """
    This kernel packs 3 sgemms (q/k/v) into a single kernel. The multiplication
    results are accumulated into the output tensor.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, 3 * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank. The second dimension is partitioned
            for Q, K, and V.
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, N_Q + 2 * N_KV, K).
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, N_Q + 2 * N_KV).
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    # 当前块计算batch_id对应的序列，从x的第seg_start行开始，长度为seg_len。
    # qkv_id决定计算q/k/v中的哪一个（0: q, 1: k, 2: v）
    batch_id = tl.program_id(axis=2)  # 获取批次ID（axis=2）
    w_index = tl.load(weight_indices + batch_id)  # 加载当前批次的权重索引
    rank = tl.load(lora_ranks + w_index)  # 加载当前LoRA适配器的秩

    # If rank is 0, this kernel is a no-op.
    # 如果秩为0，本内核不执行任何操作
    if rank == 0:
        return  # 直接返回

    qkv_id = tl.program_id(axis=1)  # 获取QKV ID（0: q, 1: k, 2: v）
    pid = tl.program_id(axis=0)  # 获取程序ID
    seg_len = tl.load(seg_lens + batch_id)  # 加载当前段的长度
    if seg_len == 0:  # 如果段长度为0
        return  # 直接返回
    seg_start = tl.load(seg_indptr + batch_id)  # 加载当前段的起始位置
    n_start = tl.load(n_offs + qkv_id)  # 加载当前QKV切片的起始偏移
    n_size = tl.load(n_offs + qkv_id + 1) - n_start  # 计算当前QKV切片的大小
    scaling = tl.load(scalings + w_index)  # 加载缩放因子
    # Adjust K (rank) according to the specific LoRA adapter
    # 根据特定LoRA适配器调整K（rank）
    K = tl.minimum(K, rank)  # 取最小值，限制收缩维度

    # The tile in output matrix will have (pid_s, pid_n) as id
    # 输出矩阵中的分片将以(pid_s, pid_n)为ID
    num_pid_n = tl.cdiv(max_qkv_out_dim, BLOCK_N)  # 计算N维度的程序数
    pid_s = pid // num_pid_n  # 计算序列维度的程序ID
    pid_n = pid % num_pid_n  # 计算输出维度的程序ID
    if pid_s * BLOCK_S >= seg_len:  # 如果序列维度超出段长度
        return  # 直接返回

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    # 为x和weights[batch_id][n_start: n_end][:]的第一个块创建指针
    # 指针将在K方向移动时前进并累加
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S  # 序列维度的偏移量
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 输出维度的偏移量
    k_offset = tl.arange(0, BLOCK_K)  # 收缩维度的偏移量

    s_physical = _resolve_token_positions(  # 解析物理token位置
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = (  # 计算x的指针
        x
        + (qkv_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (  # 计算权重指针
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
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < n_size),  # 掩码
            other=0.0,  # 掩码外填充0
        )
        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

        x_ptrs += BLOCK_K * x_stride_1  # 推进x指针
        w_ptrs += BLOCK_K * w_stride_2  # 推进权重指针

    # Store result to output matrix
    # 将结果存储到输出矩阵
    partial_sum *= scaling  # 应用缩放因子
    partial_sum = partial_sum.to(x.dtype.element_ty)  # 转换回输入数据类型
    output_ptr = (  # 计算输出指针
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < n_size)  # 输出掩码
    partial_sum += tl.load(output_ptr, mask=output_mask)  # 读取当前输出值并累加
    tl.store(output_ptr, partial_sum, mask=output_mask)  # 将结果写入输出


def qkv_lora_b_fwd(  # QKV LoRA B前向计算：对Q/K/V三个切片执行LoRA B投影并累加到输出
    x: torch.Tensor,  # 输入：(s, n_slices * r)，LoRA A投影的结果
    qkv_lora_b: torch.Tensor,  # LoRA B权重：(num_lora, output_dim_q + 2 * output_dim_kv, r)
    batch_info: LoRABatchInfo,  # LoRA批量信息
    output_offset: torch.Tensor,  # QKV在输出维度上的偏移量
    max_qkv_out_dim: int,  # max(output_dim_q, output_dim_kv)
    base_output: torch.Tensor = None,  # 基础输出，若提供则原地累加
    n_slices: int = 3,  # 切片数（Q/K/V为3）
) -> torch.Tensor:  # 返回输出张量

    # x: (s, n_slices * r)
    # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
    # output_offset = [0, output_dim_q, output_dim_q + output_dim_kv,
    #                     output_dim_q + 2 * output_dim_kv]  (length n_slices + 1)
    # max_qkv_out_dim = max(output_dim_q, output_dim_kv)
    # output: (s, output_dim_q + 2 * output_dim_kv)

    # Compute lora_output with shape (s, output_dim) as follows:
    # 计算形状为(s, output_dim)的lora_output，如下所示：
    # lora_output[:, :output_dim_q] = sgemm(x[:, :r], qkv_lora_b[:, :outptu_dim_q, :])
    # lora_output[:, output_dim_q: output_dim_q + output_dim_kv]
    #      = sgemm(x[:, r: 2 * r], qkv_lora_b[:, outptu_dim_q: output_dim_q + output_dim_kv, :])
    # lora_output[:, output_dim_q + output_dim_kv: ]
    #      = sgemm(x[:, 2 * r: , qkv_lora_b[:, output_dim_q + output_dim_kv: , :])

    # Get dims 获取维度
    s = x.shape[0]  # 总token数
    input_dim = x.shape[1]  # 输入维度
    r = qkv_lora_b.shape[-1]  # LoRA秩
    output_dim = qkv_lora_b.shape[-2]  # 输出维度
    assert input_dim == n_slices * r  # 断言：输入维度等于切片数乘以秩
    assert output_offset.shape[0] == n_slices + 1  # 断言：偏移量长度等于切片数加1

    BLOCK_S = 16  # 序列维度的块大小
    BLOCK_R = 16  # 收缩维度（rank）的块大小
    BLOCK_OUT = 64  # 输出维度的块大小

    grid_b = (  # 计算网格大小
        triton.cdiv(batch_info.max_len, BLOCK_S)
        * triton.cdiv(max_qkv_out_dim, BLOCK_OUT),
        n_slices,  # QKV三个切片
        batch_info.bs,  # 批次大小
    )

    if base_output is None:  # 如果没有提供基础输出
        output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)  # 创建零输出张量
    else:  # 如果提供了基础输出
        output = base_output  # 使用基础输出（原地累加）

    sorted_by_adapter = batch_info.permutation is not None  # 判断是否按适配器排序
    _qkv_lora_b_kernel[grid_b](  # 启动QKV LoRA B内核
        x,  # 输入
        qkv_lora_b,  # LoRA B权重
        output,  # 输出
        r,  # LoRA秩
        max_qkv_out_dim,  # 最大QKV输出维度
        x.stride(0),  # x的第0维步幅
        x.stride(1),  # x的第1维步幅
        qkv_lora_b.stride(0),  # weights的第0维步幅
        qkv_lora_b.stride(1),  # weights的第1维步幅
        qkv_lora_b.stride(2),  # weights的第2维步幅
        output.stride(0),  # output的第0维步幅
        output.stride(1),  # output的第1维步幅
        batch_info.seg_lens,  # 各段长度
        batch_info.seg_indptr,  # 段索引指针
        batch_info.weight_indices,  # 权重索引
        batch_info.lora_ranks,  # LoRA秩
        output_offset,  # QKV偏移量
        batch_info.permutation,  # 排序后的token ID
        sorted_by_adapter,  # 是否按适配器排序
        BLOCK_S,  # 序列维度块大小
        BLOCK_OUT,  # 输出维度块大小
        BLOCK_R,  # 收缩维度块大小
        batch_info.scalings,  # 缩放因子
    )

    return output  # 返回输出张量
