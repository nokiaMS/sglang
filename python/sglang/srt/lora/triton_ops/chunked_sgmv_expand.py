# 分块SGMV LoRA扩展前向计算的Triton实现
# 提供Triton kernel和对应的Python包装函数，用于高效地按分块方式执行LoRA B矩阵的扩展（expand）乘法
from typing import Optional  # 导入可选类型提示

import torch  # 导入PyTorch核心库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.triton_ops.lora_tuning_config import get_lora_expand_config  # 导入LoRA扩展配置获取函数
from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批次信息工具类
from sglang.srt.utils import cached_triton_kernel  # 导入Triton kernel缓存装饰器


@cached_triton_kernel(  # 缓存Triton kernel装饰器
    lambda _, kwargs: (kwargs["NUM_SLICES"], kwargs["BLOCK_M"], kwargs["OUTPUT_DIM"])  # 根据切片数、块M大小和输出维度作为缓存键
)
@triton.jit(do_not_specialize=["num_segs", "output_stride_0", "output_stride_1"])  # Triton JIT编译，不对指定参数特化
def _chunked_lora_expand_kernel(
    # Pointers to matrices 矩阵指针
    x,  # 输入张量指针
    weights,  # LoRA B权重指针
    output,  # 输出张量指针
    # Output strides may differ from OUTPUT_DIM when compact LoRA output is
    # accumulated into a wider base projection.
    # 当紧凑LoRA输出累加到更宽的基础投影时，输出步长可能与OUTPUT_DIM不同
    output_stride_0,  # 输出在第0维的步长
    output_stride_1,  # 输出在第1维的步长
    # Information on sequence lengths and weight id 序列长度和权重ID信息
    seg_indptr,  # 段索引指针
    weight_indices,  # 权重索引指针
    lora_ranks,  # LoRA秩数组指针
    permutation,  # 排列映射指针
    num_segs,  # 段数量
    # For fused output scaling 用于融合输出缩放
    scalings,  # 缩放因子数组指针
    # Offsets of q/k/v slice on output dimension 输出维度上Q/K/V切片的偏移
    slice_offsets,  # 切片偏移指针
    # Meta parameters 元参数
    NUM_SLICES: tl.constexpr,  # 切片数量（编译时常量）
    OUTPUT_DIM: tl.constexpr,  # 输出维度（编译时常量）
    MAX_RANK: tl.constexpr,  # K = R 最大秩（编译时常量）
    BLOCK_M: tl.constexpr,  # M维度的块大小（编译时常量）
    BLOCK_N: tl.constexpr,  # N维度的块大小（编译时常量）
    BLOCK_K: tl.constexpr,  # K维度的块大小（编译时常量）
):
    """
    Computes a chunked SGMV for LoRA expand operations.
    计算分块SGMV用于LoRA扩展操作

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).
    当序列的秩为0时，内核实际上是一个空操作，遵循PyTorch的约定，即形状为(m, 0)和(0, n)的
    两个矩阵的乘积是形状为(m, n)的全零矩阵

    Args:
    参数：
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, num_slices * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank.
            输入张量，即LoRA A投影的结果。形状：(s, num_slices * K)，
            其中s是批次中所有序列长度的总和，K是最大LoRA秩
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, output_dim, K).
            所有适配器的LoRA B权重。形状：(num_lora, output_dim, K)
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, output_dim) or a wider base output.
            存储结果的输出张量。形状：(s, output_dim)或更宽的基础输出
    """
    x_stride_0: tl.constexpr = NUM_SLICES * MAX_RANK  # 输入在第0维的步长（编译时常量）
    x_stride_1: tl.constexpr = 1  # 输入在第1维的步长（编译时常量）

    w_stride_0: tl.constexpr = OUTPUT_DIM * MAX_RANK  # 权重在LoRA维度上的步长（编译时常量）
    w_stride_1: tl.constexpr = MAX_RANK  # 权重在输出维度上的步长（编译时常量）
    w_stride_2: tl.constexpr = 1  # 权重在秩维度上的步长（编译时常量）

    pid_s = tl.program_id(axis=2)  # 获取当前程序在段维度上的ID
    if pid_s >= num_segs:  # 如果段ID超出段数量
        return  # 跳过

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    # 当前块计算batch_id对应的序列，从x的第seg_start行开始，长度为seg_len
    # qkv_id决定计算q、k、v中的哪一个（0: q, 1: k, 2: v）
    w_index = tl.load(weight_indices + pid_s)  # 加载当前段的权重索引
    cur_rank = tl.load(lora_ranks + w_index)  # 加载当前适配器的LoRA秩

    # If rank is 0, this kernel is a no-op. 如果秩为0，此内核为空操作
    if cur_rank == 0:  # 如果当前秩为0
        return  # 跳过

    seg_start = tl.load(seg_indptr + pid_s)  # 加载当前段的起始位置
    seg_end = tl.load(seg_indptr + pid_s + 1)  # 加载当前段的结束位置

    slice_id = tl.program_id(axis=1)  # 获取当前程序在切片维度上的ID
    slice_start = tl.load(slice_offsets + slice_id)  # 加载当前切片的输出起始偏移
    slice_end = tl.load(slice_offsets + slice_id + 1)  # 加载当前切片的输出结束偏移

    scaling = tl.load(scalings + w_index)  # 加载当前适配器的缩放因子
    # Adjust K (rank) according to the specific LoRA adapter 根据特定LoRA适配器调整K（秩）
    cur_rank = tl.minimum(MAX_RANK, cur_rank)  # 确保当前秩不超过最大秩

    # Map logical sequence index to physical index 将逻辑序列索引映射到物理索引
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start  # 计算逻辑行偏移
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )  # 通过排列映射获取物理行偏移

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    # 为x的第一个块和weights[batch_id][n_start: n_end][:]创建指针
    # 指针将在K方向上前进并累加
    pid_n = tl.program_id(axis=0)  # 获取当前程序在N维度上的ID
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start  # 计算N维度的偏移
    k_offset = tl.arange(0, BLOCK_K)  # 计算K维度的偏移

    x_ptrs = (
        x
        + slice_id * cur_rank * x_stride_1
        + (s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )  # 计算输入块的指针
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )  # 计算权重块的指针

    # Iterate to compute the block in output matrix 迭代计算输出矩阵中的块
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # 初始化部分和为零
    for k in range(0, tl.cdiv(cur_rank, BLOCK_K)):  # 在K维度上分块迭代
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < cur_rank - k * BLOCK_K),
            other=0.0,
        )  # 加载输入分块，越界处用0填充
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < cur_rank - k * BLOCK_K)
            & (n_offset[None, :] < slice_end),
            other=0.0,
        )  # 加载权重分块，越界处用0填充
        partial_sum += tl.dot(x_tile, w_tile)  # 计算分块矩阵乘法并累加

        x_ptrs += BLOCK_K * x_stride_1  # 前进输入指针到下一个K块
        w_ptrs += BLOCK_K * w_stride_2  # 前进权重指针到下一个K块

    # Store result to output matrix 将结果存储到输出矩阵
    partial_sum *= scaling  # 乘以缩放因子
    partial_sum = partial_sum.to(x.dtype.element_ty)  # 转换为输入的数据类型
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )  # 计算输出指针
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < slice_end
    )  # 创建输出掩码
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)  # 加载现有输出值并累加
    tl.store(output_ptr, partial_sum, mask=output_mask)  # 将结果存储到输出


def chunked_sgmv_lora_expand_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: Optional[torch.Tensor],
) -> torch.Tensor:  # 分块SGMV LoRA扩展前向计算函数，执行LoRA B矩阵与A矩阵输出的乘法

    # x: (s, slice_num * r) 输入，形状为(总token数, 切片数*秩)
    # weights: (num_lora, output_dim, r) LoRA B权重，形状为(适配器数, 输出维度, 秩)
    # slice_offsets: boundaries for different slices in the output dimension 输出维度中不同切片的边界
    # output: (s, output_dim) 输出，形状为(总token数, 输出维度)

    # Compute lora_output with shape (s, output_dim) as follows:
    # 计算形状为(s, output_dim)的lora_output，步骤如下：
    # For each slice i, accumulates:
    # 对每个切片i，累加：
    # lora_output[:, slice_offsets[i]:slice_offsets[i+1]] += scaling * sgemm(x[:, i*cur_rank:(i+1)*cur_rank], weights[:, slice_offsets[i]:slice_offsets[i+1], :])

    assert x.is_contiguous()  # 断言输入是连续的
    assert weights.is_contiguous()  # 断言权重是连续的
    assert len(x.shape) == 2  # 断言输入是二维的
    assert len(weights.shape) == 3  # 断言权重是三维的

    # Get dims 获取维度
    M = x.shape[0]  # 获取总token数
    input_dim = x.shape[1]  # 获取输入维度
    OUTPUT_DIM = weights.shape[1]  # 获取输出维度
    MAX_RANK = weights.shape[2]  # 获取最大秩
    num_slices = len(slice_offsets) - 1  # 计算切片数量
    assert input_dim == num_slices * MAX_RANK  # 断言输入维度等于切片数乘以最大秩

    # Block shapes — use auto-tuned config if available, else defaults 块形状——如有自动调优配置则使用，否则使用默认值
    BLOCK_M = batch_info.max_len  # 设置M维度的块大小为最大序列长度
    config = get_lora_expand_config(
        K=OUTPUT_DIM, R=MAX_RANK, num_slices=num_slices, chunk_size=BLOCK_M
    )  # 获取LoRA扩展配置
    BLOCK_K = config["BLOCK_K"]  # 从配置中获取K维度的块大小
    BLOCK_N = config["BLOCK_N"]  # 从配置中获取N维度的块大小

    num_segments = batch_info.num_segments  # 获取段数量

    grid = (
        triton.cdiv(max_slice_size, BLOCK_N),
        num_slices,  # number of slices in the input/output 输入/输出中的切片数量
        batch_info.bs if batch_info.use_cuda_graph else num_segments,
    )  # 设置Triton网格大小

    if base_output is None:  # 如果没有提供基础输出
        output = torch.zeros((M, OUTPUT_DIM), device=x.device, dtype=x.dtype)  # 初始化为零张量
    else:
        output = base_output  # 使用基础输出

    # Optional launch params from tuned config 从调优配置中获取可选的启动参数
    extra_kwargs = {}  # 初始化额外参数字典
    if "num_warps" in config:  # 如果配置中包含warp数
        extra_kwargs["num_warps"] = config["num_warps"]  # 设置warp数
    if "num_stages" in config:  # 如果配置中包含流水线级数
        extra_kwargs["num_stages"] = config["num_stages"]  # 设置流水线级数
    if "maxnreg" in config:  # 如果配置中包含最大寄存器数
        extra_kwargs["maxnreg"] = config["maxnreg"]  # 设置最大寄存器数

    _chunked_lora_expand_kernel[grid](  # 启动Triton kernel
        x=x,  # 输入张量
        weights=weights,  # LoRA B权重
        output=output,  # 输出张量
        output_stride_0=output.stride(0),  # 输出在第0维的步长
        output_stride_1=output.stride(1),  # 输出在第1维的步长
        seg_indptr=batch_info.seg_indptr,  # 段索引指针
        weight_indices=batch_info.weight_indices,  # 权重索引
        lora_ranks=batch_info.lora_ranks,  # LoRA秩数组
        permutation=batch_info.permutation,  # 排列映射
        num_segs=num_segments,  # 段数量
        scalings=batch_info.scalings,  # 缩放因子数组
        slice_offsets=slice_offsets,  # 切片偏移
        # constants 常量
        NUM_SLICES=num_slices,  # 切片数量
        OUTPUT_DIM=OUTPUT_DIM,  # 输出维度
        MAX_RANK=MAX_RANK,  # 最大秩
        BLOCK_M=BLOCK_M,  # M维度的块大小
        BLOCK_N=BLOCK_N,  # N维度的块大小
        BLOCK_K=BLOCK_K,  # K维度的块大小
        **extra_kwargs,  # 额外参数
    )

    return output  # 返回最终输出
