# 分块SGMV收缩操作的Triton内核实现，用于LoRA的A矩阵投影（降维）计算
# 将输入激活值与LoRA A权重矩阵相乘，实现从高维到低维的映射

import torch  # 导入PyTorch张量库
import triton  # 导入Triton编译框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.lora.triton_ops.lora_tuning_config import get_lora_shrink_config  # 导入LoRA收缩操作的自动调优配置获取函数
from sglang.srt.lora.utils import LoRABatchInfo  # 导入LoRA批次信息数据类
from sglang.srt.utils import cached_triton_kernel  # 导入缓存的Triton内核装饰器


@cached_triton_kernel(  # 使用缓存的Triton内核装饰器，避免重复编译
    lambda _, kwargs: (kwargs["K"], kwargs["NUM_SLICES"], kwargs["BLOCK_M"])  # 根据K、NUM_SLICES和BLOCK_M参数生成缓存键
)
@triton.jit(do_not_specialize=["num_segs"])  # Triton JIT编译，不对num_segs参数进行特化
def _chunked_lora_shrink_kernel(  # 分块LoRA收缩计算内核函数
    # Pointers to matrices  # 矩阵指针
    x,  # 输入激活值矩阵指针
    weights,  # LoRA权重矩阵指针
    output,  # 输出矩阵指针
    # Information on sequence lengths,ranks and weight id  # 序列长度、秩和权重ID信息
    seg_indptr,  # 段索引指针，指向每个段的起始位置
    weight_indices,  # 权重索引指针，指向每个段对应的LoRA适配器
    lora_ranks,  # LoRA秩指针，指向每个适配器的秩值
    permutation,  # 排列映射指针，将逻辑位置映射到物理位置
    num_segs,  # 段的数量
    # Meta parameters  # 元参数
    N: tl.constexpr,  # num_slices * r  # N维大小，等于切片数乘以秩
    K: tl.constexpr,  # input_dim  # K维大小，即输入维度
    NUM_SLICES: tl.constexpr,  # 切片数量
    BLOCK_M: tl.constexpr,  # M方向的块大小
    BLOCK_N: tl.constexpr,  # N方向的块大小
    BLOCK_K: tl.constexpr,  # K方向的块大小
):
    """
    Computes a chunked SGMV for LoRA shrink operations.
    计算LoRA收缩操作的分块SGMV（分段矩阵向量乘法）。

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * num_slices]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).
    该内核确保output[seg_start:seg_start + seg_len, :rank * num_slices]
    存储了输入`x`与对应序列的LoRA权重的乘积。这意味着当rank为0时，
    该内核本质上是一个空操作，因为output[seg_start:seg_start + seg_len, :0]是空切片，天然正确。

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        x (torch.Tensor): 输入激活张量，形状为`(s, K)`，其中`s`是批次中所有序列长度之和。
        weights (torch.Tensor): The LoRA A weights for all available adapters,
            with shape `(num_lora, N, K)` where N = num_slices * r.
        weights (torch.Tensor): 所有可用适配器的LoRA A权重，
            形状为`(num_lora, N, K)`，其中N = num_slices * r。
        output (torch.Tensor): The output tensor of shape `(s, N)`.
        output (torch.Tensor): 输出张量，形状为`(s, N)`。
    """
    x_stride_1: tl.constexpr = 1  # x矩阵第1维（列）的步长
    x_stride_0: tl.constexpr = K  # x矩阵第0维（行）的步长

    w_stride_0: tl.constexpr = N * K  # 权重矩阵第0维（适配器）的步长
    w_stride_1: tl.constexpr = K  # 权重矩阵第1维（N维）的步长
    w_stride_2: tl.constexpr = 1  # 权重矩阵第2维（K维）的步长

    output_stride_0: tl.constexpr = N  # 输出矩阵第0维（行）的步长
    output_stride_1: tl.constexpr = 1  # 输出矩阵第1维（列）的步长

    pid_s = tl.program_id(1)  # 获取段维度的程序ID
    if pid_s >= num_segs:  # 如果段ID超出段数量
        return  # 超出范围则直接返回

    pid_n = tl.program_id(0)  # 获取N维度的程序ID

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    # 当前块计算batch_id对应的序列，从x的seg_start行开始，长度为seg_len
    w_index = tl.load(weight_indices + pid_s)  # 加载当前段对应的LoRA适配器索引
    rank = tl.load(lora_ranks + w_index)  # 加载当前适配器的LoRA秩

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    # 如果rank为0，该内核变为空操作，因为输出始终是平凡的（空切片）。
    if rank == 0:  # 如果LoRA秩为0
        return  # 秩为0则无需计算，直接返回

    seg_start = tl.load(seg_indptr + pid_s)  # 加载当前段的起始行索引
    seg_end = tl.load(seg_indptr + pid_s + 1)  # 加载当前段的结束行索引

    # Adjust N dim according to the specific LoRA adapter
    # 根据特定LoRA适配器调整N维度
    cur_n = tl.minimum(N, rank * NUM_SLICES)  # 当前有效的N维度大小，取N和rank*切片数的最小值

    # Map logical sequence index to physical index
    # 将逻辑序列索引映射到物理索引
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start  # 计算逻辑行偏移量
    s_offset_physical = tl.load(  # 加载物理行偏移量
        permutation + s_offset_logical, mask=s_offset_logical < seg_end  # 使用掩码防止越界
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N  # 计算N维偏移量
    k_offset = tl.arange(0, BLOCK_K)  # 计算K维偏移量
    x_ptrs = x + (  # 计算输入矩阵x的指针位置
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1  # 行偏移*行步长 + 列偏移*列步长
    )
    w_ptrs = (weights + w_index * w_stride_0) + (  # 计算权重矩阵的指针位置，先定位到对应适配器
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1  # K维偏移*K步长 + N维偏移*N步长
    )

    # Iterate to compute the block in output matrix
    # 迭代计算输出矩阵中的块
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # 初始化部分和为零
    for k in range(0, tl.cdiv(K, BLOCK_K)):  # 沿K维度迭代，每次处理BLOCK_K个元素
        x_tile = tl.load(  # 加载输入矩阵的一个小块
            x_ptrs,  # x的指针位置
            mask=(s_offset_logical[:, None] < seg_end)  # 掩码：逻辑行偏移在段内
            & (k_offset[None, :] < K - k * BLOCK_K),  # 掩码：K维偏移未超出范围
            other=0.0,  # 越界位置填充0
        )
        w_tile = tl.load(  # 加载权重矩阵的一个小块
            w_ptrs,  # 权重的指针位置
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),  # 掩码：K和N维偏移均在范围内
            other=0.0,  # 越界位置填充0
        )
        partial_sum += tl.dot(x_tile, w_tile)  # 累加矩阵乘法结果

        x_ptrs += BLOCK_K * x_stride_1  # 移动x指针到下一个K块
        w_ptrs += BLOCK_K * w_stride_2  # 移动权重指针到下一个K块

    # Store result to output matrix
    # 将结果存储到输出矩阵
    partial_sum = partial_sum.to(x.dtype.element_ty)  # 将部分和转换为输入数据类型
    output_ptr = output + (  # 计算输出矩阵的指针位置
        s_offset_physical[:, None] * output_stride_0  # 行偏移*行步长
        + n_offset[None, :] * output_stride_1  # 列偏移*列步长
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)  # 计算输出掩码
    tl.store(output_ptr, partial_sum, mask=output_mask)  # 将部分和存储到输出矩阵


def chunked_sgmv_lora_shrink_forward(  # 分块SGMV LoRA收缩前向传播函数
    x: torch.Tensor,  # 输入激活张量
    weights: torch.Tensor,  # LoRA权重张量
    batch_info: LoRABatchInfo,  # LoRA批次信息
    num_slices: int,  # 切片数量
) -> torch.Tensor:  # 返回输出张量
    # x: (s, input_dim)  # x: 输入，形状为(总序列长度, 输入维度)
    # weights: (num_lora, num_slices * r, input_dim)  # weights: 权重，形状为(适配器数, 切片数*秩, 输入维度)
    # output: (s, num_slices * r)  # output: 输出，形状为(总序列长度, 切片数*秩)
    # num_slices: qkv=3, gate_up=2, others=1  # 切片数: qkv为3, gate_up为2, 其他为1
    # when called with multiple slices, the weights.shape[-2] will be num_slices * r  # 当使用多个切片调用时，weights.shape[-2]将是切片数*秩
    # input_dim is much larger than r  # 输入维度远大于秩

    assert x.is_contiguous()  # 断言x在内存中是连续的
    assert weights.is_contiguous()  # 断言weights在内存中是连续的
    assert len(x.shape) == 2  # 断言x是2维张量
    assert len(weights.shape) == 3  # 断言weights是3维张量

    # Block shapes — use auto-tuned config if available, else defaults
    # 块形状 — 如果有自动调优配置则使用，否则使用默认值
    BLOCK_M = batch_info.max_len  # M方向的块大小设为最大序列长度
    # weights shape is (num_lora, num_slices * rank, input_dim)
    # 权重形状为(num_lora, num_slices * rank, input_dim)
    MAX_RANK = weights.shape[1] // num_slices  # 计算最大LoRA秩 = 权重第1维 / 切片数
    config = get_lora_shrink_config(  # 获取LoRA收缩操作的自动调优配置
        K=weights.shape[2], R=MAX_RANK, num_slices=num_slices, chunk_size=BLOCK_M  # 传入K维度、最大秩、切片数和块大小
    )
    BLOCK_N = config["BLOCK_N"]  # 从配置中获取N方向的块大小
    BLOCK_K = config["BLOCK_K"]  # 从配置中获取K方向的块大小

    S = x.shape[0]  # 总序列长度
    N = weights.shape[1]  # 输出维度 = 切片数 * 秩
    K = weights.shape[2]  # 输入维度
    assert x.shape[-1] == K  # 断言x的最后一维等于K

    num_segments = batch_info.num_segments  # 获取段的数量
    grid = (  # 定义内核启动的网格大小
        triton.cdiv(N, BLOCK_N),  # N维度上的块数
        batch_info.bs if batch_info.use_cuda_graph else num_segments,  # 段维度：使用CUDA图时为批次数，否则为段数
    )

    # Optional launch params from tuned config
    # 来自调优配置的可选启动参数
    extra_kwargs = {}  # 额外的关键字参数字典
    if "num_warps" in config:  # 如果配置中包含num_warps
        extra_kwargs["num_warps"] = config["num_warps"]  # 设置warp数量
    if "num_stages" in config:  # 如果配置中包含num_stages
        extra_kwargs["num_stages"] = config["num_stages"]  # 设置流水线阶段数

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)  # 创建输出张量，形状为(S, N)
    _chunked_lora_shrink_kernel[grid](  # 启动分块LoRA收缩内核
        x=x,  # 输入激活值
        weights=weights,  # LoRA权重
        output=output,  # 输出张量
        seg_indptr=batch_info.seg_indptr,  # 段索引指针
        weight_indices=batch_info.weight_indices,  # 权重索引指针
        lora_ranks=batch_info.lora_ranks,  # LoRA秩指针
        permutation=batch_info.permutation,  # 排列映射指针
        num_segs=num_segments,  # 段的数量
        # constants  # 常量参数
        N=N,  # N维度大小
        K=K,  # K维度大小
        NUM_SLICES=num_slices,  # 切片数量
        BLOCK_M=BLOCK_M,  # M方向的块大小
        BLOCK_N=BLOCK_N,  # N方向的块大小
        BLOCK_K=BLOCK_K,  # K方向的块大小
        **extra_kwargs,  # 展开额外的启动参数
    )

    return output  # 返回输出张量
