# AWQ Triton内核实现模块
# 提供基于Triton的AWQ反量化和GEMM内核，以及纯PyTorch分解反量化函数
# 适配自vLLM项目的AWQ Triton实现

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq_triton.py  # 适配自vLLM项目的AWQ Triton实现

# SPDX-License-Identifier: Apache-2.0  # Apache 2.0许可证
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

import torch  # PyTorch深度学习框架
import triton  # Triton GPU编程框架
import triton.language as tl  # Triton语言模块

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]  # AWQ Triton支持的分组大小列表，-1表示每列为一个组


@triton.jit  # Triton JIT编译装饰器
def awq_dequantize_kernel(  # AWQ反量化Triton内核函数
    qweight_ptr,  # quantized matrix  # 量化权重矩阵指针
    scales_ptr,  # scales, per group  # 缩放因子指针，按组存储
    zeros_ptr,  # zeros, per group  # 零点指针，按组存储
    group_size,  # Should always be one of the supported group sizes  # 分组大小，应为支持的分组大小之一
    result_ptr,  # Output matrix  # 输出矩阵指针
    num_cols,  # input num cols in qweight  # qweight的输入列数
    num_rows,  # input num rows in qweight  # qweight的输入行数
    BLOCK_SIZE_X: tl.constexpr,  # X方向块大小（编译时常量）
    BLOCK_SIZE_Y: tl.constexpr,  # Y方向块大小（编译时常量）
):
    # Setup the pids.  # 设置程序ID
    pid_x = tl.program_id(axis=0)  # 获取X方向的程序ID
    pid_y = tl.program_id(axis=1)  # 获取Y方向的程序ID

    # Compute offsets and masks for qweight_ptr.  # 计算qweight_ptr的偏移量和掩码
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)  # Y方向偏移量
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)  # X方向偏移量
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]  # 二维偏移量（行优先）

    masks_y = offsets_y < num_rows  # Y方向边界掩码
    masks_x = offsets_x < num_cols  # X方向边界掩码

    masks = masks_y[:, None] & masks_x[None, :]  # 组合掩码

    # Compute offsets and masks for result output ptr.  # 计算结果输出指针的偏移量和掩码
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)  # 结果Y方向偏移量
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)  # 结果X方向偏移量（乘8因为每个int32解包为8个4位值）
    result_offsets = (  # 二维偏移量
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]  # 行优先，列数乘8
    )

    result_masks_y = result_offsets_y < num_rows  # 结果Y方向边界掩码
    result_masks_x = result_offsets_x < num_cols * 8  # 结果X方向边界掩码（乘8）
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]  # 组合结果掩码

    # Load the weights.  # 加载权重
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)  # 从全局内存加载量化权重
    iweights = tl.interleave(iweights, iweights)  # 交错复制，将每个元素重复并交错排列
    iweights = tl.interleave(iweights, iweights)  # 第二次交错，4倍扩展
    iweights = tl.interleave(iweights, iweights)  # 第三次交错，8倍扩展

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]  # 创建反向AWQ顺序张量：[0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.  # 将给定索引映射到正确的顺序
    reverse_awq_order_tensor = (  # 构建反向AWQ顺序张量
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]  # 2x4网格后展平为8元素
    ).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and  # 用此计算一组移位量，用于解包和
    # reorder the values in iweights and zeros.  # 重排iweights和zeros中的值
    shifts = reverse_awq_order_tensor * 4  # 每个位置乘4得到4位移位量
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))  # 广播到所需形状
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))  # 重塑为二维形状

    # Unpack and reorder: shift out the correct 4-bit value and mask.  # 解包并重排：右移提取正确的4位值并掩码
    iweights = (iweights >> shifts) & 0xF  # 右移后取低4位

    # Compute zero offsets and masks.  # 计算零点的偏移量和掩码
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)  # 零点Y方向偏移量（按组计算）
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)  # 零点X方向偏移量
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]  # 二维偏移量

    zero_masks_y = zero_offsets_y < num_rows // group_size  # 零点Y方向边界掩码
    zero_masks_x = zero_offsets_x < num_cols  # 零点X方向边界掩码
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]  # 组合零点掩码

    # Load the zeros.  # 加载零点
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)  # 从全局内存加载零点
    zeros = tl.interleave(zeros, zeros)  # 交错复制
    zeros = tl.interleave(zeros, zeros)  # 第二次交错
    zeros = tl.interleave(zeros, zeros)  # 第三次交错，8倍扩展
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))  # 广播到与iweights相同的形状

    # Unpack and reorder: shift out the correct 4-bit value and mask.  # 解包并重排：右移提取正确的4位零点值并掩码
    zeros = (zeros >> shifts) & 0xF  # 右移后取低4位

    # Compute scale offsets and masks.  # 计算缩放因子的偏移量和掩码
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)  # 缩放因子Y方向偏移量（按组计算）
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)  # 缩放因子X方向偏移量
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]  # 二维偏移量
    scale_masks_y = scale_offsets_y < num_rows // group_size  # 缩放因子Y方向边界掩码
    scale_masks_x = scale_offsets_x < num_cols * 8  # 缩放因子X方向边界掩码
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]  # 组合缩放因子掩码

    # Load the scales.  # 加载缩放因子
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)  # 从全局内存加载缩放因子
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))  # 广播到与iweights相同的形状

    # Dequantize.  # 反量化
    iweights = (iweights - zeros) * scales  # 反量化公式：(权重-零点)*缩放因子
    iweights = iweights.to(result_ptr.type.element_ty)  # 转换为输出指针的数据类型

    # Finally, store.  # 最后，存储结果
    tl.store(result_ptr + result_offsets, iweights, result_masks)  # 将反量化结果写回全局内存


@triton.jit  # Triton JIT编译装饰器
def awq_gemm_kernel(  # AWQ GEMM（通用矩阵乘法）Triton内核函数
    a_ptr,  # 输入矩阵A指针
    b_ptr,  # 量化权重矩阵B指针
    c_ptr,  # 输出矩阵C指针
    zeros_ptr,  # 零点指针
    scales_ptr,  # 缩放因子指针
    M,  # 矩阵A的行数
    N,  # 矩阵B的列数
    K,  # 矩阵A的列数/矩阵B的行数
    group_size,  # 量化分组大小
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小（编译时常量）
    BLOCK_SIZE_N: tl.constexpr,  # N方向块大小（编译时常量）
    BLOCK_SIZE_K: tl.constexpr,  # K方向块大小（编译时常量）
    SPLIT_K: tl.constexpr,  # K维度分割数（编译时常量），用于并行化
):
    pid = tl.program_id(axis=0)  # 获取程序ID
    pid_z = tl.program_id(1)  # 获取Z方向程序ID（对应SPLIT_K维度）

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.  # 注意：这在TRITON_INTERPRET=1模式下不工作，使用下面的替代
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N  # 计算N方向的程序数
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # 使用tl.cdiv计算N方向的程序数

    pid_m = pid // num_pid_n  # 计算M方向的程序ID
    pid_n = pid % num_pid_n  # 计算N方向的程序ID

    accumulator_dtype = c_ptr.type.element_ty  # 获取累加器数据类型

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.  # 注意：这在TRITON_INTERPRET=1模式下不工作，使用下面的替代
    # accumulator = tl.arange(0, BLOCK_SIZE_N)  # 以下为替代初始化方式
    # accumulator = tl.broadcast_to(accumulator[None, :],  # 广播到二维
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0  # 清零
    # accumulator = accumulator.to(accumulator_dtype)  # 转换类型
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)  # 初始化累加器为零

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]  # 创建反向AWQ顺序张量：[0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.  # 将给定索引映射到正确的顺序
    reverse_awq_order_tensor = (  # 构建反向AWQ顺序张量
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]  # 2x4网格后展平为8元素
    ).reshape(8)

    # Create the necessary shifts to use to unpack.  # 创建解包所需的移位量
    shifts = reverse_awq_order_tensor * 4  # 每个位置乘4得到4位移位量
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))  # 广播到所需形状
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))  # 重塑为二维形状

    # Offsets and masks.  # 偏移量和掩码
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 矩阵A的M方向偏移量
    masks_am = offsets_am < M  # 矩阵A的M方向边界掩码

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)  # 量化矩阵B的N方向偏移量（除8因为打包）
    masks_bn = offsets_bn < N // 8  # 量化矩阵B的N方向边界掩码

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)  # 零点的N方向偏移量
    masks_zn = offsets_zn < N // 8  # 零点的N方向边界掩码

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 缩放因子的N方向偏移量
    masks_sn = offsets_sn < N  # 缩放因子的N方向边界掩码

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # K方向偏移量
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]  # 矩阵A的二维偏移量
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]  # 矩阵B的二维偏移量

    a_ptrs = a_ptr + offsets_a  # 矩阵A的指针偏移
    b_ptrs = b_ptr + offsets_b  # 矩阵B的指针偏移

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv  # 注意：在TRITON_INTERPRET=1模式下使用此替代tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K  # K方向块偏移量
    # for k in range(0, (K + block_offset - 1) // (block_offset)):  # 遍历K方向的块
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):  # 遍历K方向的块数
        masks_k = offsets_k < K  # K方向边界掩码
        masks_a = masks_am[:, None] & masks_k[None, :]  # 矩阵A的组合掩码
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)  # 加载矩阵A的数据

        masks_b = masks_k[:, None] & masks_bn[None, :]  # 矩阵B的组合掩码
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)  # 加载量化矩阵B的数据
        b = tl.interleave(b, b)  # 交错复制
        b = tl.interleave(b, b)  # 第二次交错
        b = tl.interleave(b, b)  # 第三次交错，8倍扩展

        # Dequantize b.  # 反量化矩阵B
        offsets_szk = (  # 计算当前K分段的零点/缩放因子偏移量
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K  # 当前K分段的起始位置
        ) // group_size + tl.arange(0, 1)  # 除以group_size得到组索引
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]  # 零点二维偏移量
        masks_zk = offsets_szk < K // group_size  # 零点K方向边界掩码
        masks_z = masks_zk[:, None] & masks_zn[None, :]  # 零点组合掩码
        zeros_ptrs = zeros_ptr + offsets_z  # 零点指针偏移
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)  # 加载零点
        zeros = tl.interleave(zeros, zeros)  # 交错复制
        zeros = tl.interleave(zeros, zeros)  # 第二次交错
        zeros = tl.interleave(zeros, zeros)  # 第三次交错，8倍扩展
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))  # 广播到与B相同的形状

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]  # 缩放因子二维偏移量
        masks_sk = offsets_szk < K // group_size  # 缩放因子K方向边界掩码
        masks_s = masks_sk[:, None] & masks_sn[None, :]  # 缩放因子组合掩码
        scales_ptrs = scales_ptr + offsets_s  # 缩放因子指针偏移
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)  # 加载缩放因子
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))  # 广播到与B相同的形状

        b = (b >> shifts) & 0xF  # 解包4位权重值
        zeros = (zeros >> shifts) & 0xF  # 解包4位零点值
        b = (b - zeros) * scales  # 反量化：(权重-零点)*缩放因子
        b = b.to(c_ptr.type.element_ty)  # 转换为输出数据类型

        # Accumulate results.  # 累加结果
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)  # 矩阵乘法并累加

        offsets_k += BLOCK_SIZE_K * SPLIT_K  # 更新K方向偏移量
        a_ptrs += BLOCK_SIZE_K * SPLIT_K  # 更新矩阵A指针偏移
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)  # 更新矩阵B指针偏移

    c = accumulator.to(c_ptr.type.element_ty)  # 将累加器转换为输出数据类型
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 输出矩阵M方向偏移量
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 输出矩阵N方向偏移量
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]  # 输出指针偏移（包含SPLIT_K维度）
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  # 输出边界掩码
    tl.store(c_ptrs, c, mask=c_mask)  # 存储计算结果


# qweights - [K     , M // 8], int32  # 量化权重形状：[K, M//8]，int32类型
# scales   - [K // G, M     ], float16  # 缩放因子形状：[K//G, M]，float16类型
# zeros    - [K // G, M // 8], int32  # 零点形状：[K//G, M//8]，int32类型
def awq_dequantize_triton(  # 使用Triton内核的AWQ反量化函数
    qweight: torch.Tensor,  # 量化权重张量
    scales: torch.Tensor,  # 缩放因子张量
    zeros: torch.Tensor,  # 零点张量
    block_size_x: int = 32,  # X方向块大小，默认32
    block_size_y: int = 32,  # Y方向块大小，默认32
) -> torch.Tensor:  # 返回反量化后的权重张量
    K = qweight.shape[0]  # 权重行数（K维度）
    M = scales.shape[1]  # 缩放因子列数（M维度）
    group_size = qweight.shape[0] // scales.shape[0]  # 计算分组大小

    assert K > 0 and M > 0  # 断言K和M必须大于0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M  # 断言缩放因子形状正确
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8  # 断言零点形状正确
    assert group_size <= K  # 断言分组大小不超过K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K  # 断言分组大小在支持列表中或等于K

    # Result tensor:  # 结果张量：
    # number of rows = same as input tensor  # 行数与输入张量相同
    # number of cols = 8 x input tensor num cols  # 列数为输入张量列数的8倍
    result = torch.empty(  # 创建空的结果张量
        qweight.shape[0],  # 行数与qweight相同
        qweight.shape[1] * 8,  # 列数为qweight列数的8倍
        device=qweight.device,  # 设备与qweight相同
        dtype=scales.dtype,  # 数据类型与scales相同
    )

    Y = qweight.shape[0]  # num rows  # 行数
    X = qweight.shape[1]  # num cols  # 列数

    grid = lambda META: (  # 定义Triton内核的网格大小
        triton.cdiv(X, META["BLOCK_SIZE_X"]),  # X方向的网格大小
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),  # Y方向的网格大小
    )
    awq_dequantize_kernel[grid](  # 启动反量化内核
        qweight,  # 量化权重
        scales,  # 缩放因子
        zeros,  # 零点
        group_size,  # 分组大小
        result,  # 输出结果
        X,  # 列数
        Y,  # 行数
        BLOCK_SIZE_X=block_size_x,  # X方向块大小
        BLOCK_SIZE_Y=block_size_y,  # Y方向块大小
    )

    return result  # 返回反量化结果


# input   - [M, K]  # 输入矩阵形状：[M, K]
# qweight - [K, N // 8]  # 量化权重形状：[K, N//8]
# qzeros  - [K // G, N // 8]  # 量化零点形状：[K//G, N//8]
# scales  - [K // G, N]  # 缩放因子形状：[K//G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.  # split_k_iters - K维度的并行度，整数，2的幂
def awq_gemm_triton(  # 使用Triton内核的AWQ GEMM函数
    input: torch.Tensor,  # 输入矩阵
    qweight: torch.Tensor,  # 量化权重矩阵
    scales: torch.Tensor,  # 缩放因子矩阵
    qzeros: torch.Tensor,  # 量化零点矩阵
    split_k_iters: int,  # K维度分割迭代次数
    block_size_m: int = 32,  # M方向块大小，默认32
    block_size_n: int = 32,  # N方向块大小，默认32
    block_size_k: int = 32,  # K方向块大小，默认32
) -> torch.Tensor:  # 返回矩阵乘法结果张量
    M, K = input.shape  # 获取输入矩阵的M和K维度
    N = qweight.shape[1] * 8  # N维度为量化权重列数的8倍
    group_size = qweight.shape[0] // qzeros.shape[0]  # 计算分组大小

    assert N > 0 and K > 0 and M > 0  # 断言维度必须大于0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8  # 断言量化权重形状正确
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8  # 断言零点形状正确
    assert scales.shape[0] == K // group_size and scales.shape[1] == N  # 断言缩放因子形状正确
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0  # 断言split_k_iters是2的幂且不为0
    assert split_k_iters <= 32  # 断言split_k_iters不超过32
    assert group_size <= K  # 断言分组大小不超过K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K  # 断言分组大小在支持列表中或等于K

    grid = lambda META: (  # 定义Triton内核的网格大小
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),  # MN方向的网格大小
        split_k_iters,  # K方向分割数
    )

    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)  # 创建零结果张量，包含SPLIT_K维度

    # A = input, B = qweight, C = result  # A=输入，B=量化权重，C=结果
    # A = M x K, B = K x N, C = M x N  # 矩阵维度说明
    awq_gemm_kernel[grid](  # 启动GEMM内核
        input,  # 输入矩阵A
        qweight,  # 量化权重矩阵B
        result,  # 输出矩阵C
        qzeros,  # 量化零点
        scales,  # 缩放因子
        M,  # 矩阵A行数
        N,  # 矩阵B列数
        K,  # 矩阵A列数
        group_size,  # 分组大小
        BLOCK_SIZE_M=block_size_m,  # M方向块大小
        BLOCK_SIZE_N=block_size_n,  # N方向块大小
        BLOCK_SIZE_K=block_size_k,  # K方向块大小
        SPLIT_K=split_k_iters,  # K维度分割数
    )

    result = result.sum(0)  # 沿SPLIT_K维度求和，归约为最终结果

    return result  # 返回矩阵乘法结果


def awq_dequantize_decomposition(  # 基于PyTorch算子分解的AWQ反量化函数（不使用Triton）
    qweight: torch.Tensor,  # 量化权重张量
    scales: torch.Tensor,  # 缩放因子张量
    zeros: torch.Tensor,  # 零点张量
) -> torch.Tensor:  # 返回反量化后的权重张量
    qweight_tmp = qweight  # 临时量化权重变量
    qzeros_tmp = zeros  # 临时零点变量
    qweight_list = []  # 存储解包后的权重列表
    qzeros_list = []  # 存储解包后的零点列表
    shifts = [0, 4, 1, 5, 2, 6, 3, 7]  # AWQ反序移位表
    for i in range(0, 8):  # 遍历8个4位槽位
        shift_num = shifts[i] * 4  # 计算移位量
        qzeros_list.append((qzeros_tmp.reshape(-1, 1) >> shift_num) & 0xF)  # 解包零点的第i个4位值
        qweight_list.append((qweight_tmp.reshape(-1, 1) >> shift_num) & 0xF)  # 解包权重的第i个4位值
    qzeros_tmp = (  # 重组零点
        torch.cat(qzeros_list, dim=-1).reshape(qzeros_tmp.shape[0], -1).to(scales.dtype)  # 拼接后重塑并转换类型
    )
    qweight_tmp = (  # 重组权重
        torch.cat(qweight_list, dim=-1)  # 拼接解包后的权重
        .reshape(qweight_tmp.shape[0], -1)  # 重塑形状
        .to(scales.dtype)  # 转换为缩放因子的数据类型
    )
    res = (  # 计算反量化结果
        qweight_tmp.reshape(qzeros_tmp.shape[0], -1, qzeros_tmp.shape[1])  # 重塑权重为三维
        - qzeros_tmp.unsqueeze(1)  # 减去零点（广播）
    ) * scales.unsqueeze(1)  # 乘以缩放因子（广播）
    return res.reshape(qweight_tmp.shape[0], -1)  # 重塑并返回结果
