# 门控层归一化Triton内核实现
# 该模块实现了带门控机制的层归一化/RMS归一化的Triton GPU内核，
# 支持分组归一化、偏置项和门控分支，适配Mamba模型的需求。

# SPDX-License-Identifier: Apache-2.0  # SPDX许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX版权声明
# Copyright (c) 2024, Tri Dao.  # 版权声明
# Adapted from https://github.com/state-spaces/mamba/blob/60dadf2e0ee730ac337035d5533de10bc26e4847/mamba_ssm/ops/triton/layernorm_gated.py  # 改编自Mamba项目的门控层归一化实现

import torch  # 导入PyTorch库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言并简写为tl


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})  # 启发式：判断是否有偏置
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})  # 启发式：判断是否有门控分支Z
@triton.jit  # Triton JIT编译装饰器
def _layer_norm_fwd_1pass_kernel(  # 层归一化前向传播单趟内核函数
    X,  # pointer to the input  # 输入指针
    Y,  # pointer to the output  # 输出指针
    W,  # pointer to the weights  # 权重指针
    B,  # pointer to the biases  # 偏置指针
    Z,  # pointer to the other branch  # 门控分支指针
    Mean,  # pointer to the mean  # 均值指针
    Rstd,  # pointer to the 1/std  # 1/标准差指针
    stride_x_row: tl.int64,  # X的行步长
    stride_y_row: tl.int64,  # Y的行步长
    stride_z_row: tl.int64,  # Z的行步长
    M: tl.int64,  # number of rows in X  # X的行数
    N: tl.int64,  # number of columns in X  # X的列数
    eps,  # epsilon to avoid division by zero  # 防止除零的epsilon值
    BLOCK_N: tl.constexpr,  # 块大小N（编译时常量）
    HAS_BIAS: tl.constexpr,  # 是否有偏置（编译时常量）
    HAS_Z: tl.constexpr,  # 是否有门控分支（编译时常量）
    NORM_BEFORE_GATE: tl.constexpr,  # 是否在门控之前归一化（编译时常量）
    IS_RMS_NORM: tl.constexpr,  # 是否为RMS归一化（编译时常量）
):
    # Map the program id to the row of X and Y it should compute.  # 将程序ID映射到X和Y中需要计算的行
    row = tl.program_id(0)  # 获取第0轴程序ID作为行索引
    group = tl.program_id(1)  # 获取第1轴程序ID作为分组索引
    X += row * stride_x_row + group * N  # 计算输入X的基地址偏移
    Y += row * stride_y_row + group * N  # 计算输出Y的基地址偏移
    if HAS_Z:  # 如果有门控分支
        Z += row * stride_z_row + group * N  # 计算门控Z的基地址偏移
    if not IS_RMS_NORM:  # 如果不是RMS归一化（即LayerNorm）
        Mean += group * M  # 计算均值指针偏移
    Rstd += group * M  # 计算标准差倒数指针偏移
    W += group * N  # 计算权重指针偏移
    if HAS_BIAS:  # 如果有偏置
        B += group * N  # 计算偏置指针偏移
    # Compute mean and variance  # 计算均值和方差
    cols = tl.arange(0, BLOCK_N)  # 生成列索引范围
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)  # 加载输入数据并转为float32
    if HAS_Z and not NORM_BEFORE_GATE:  # 如果有门控且在门控之后归一化
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)  # 加载门控数据
        x *= z * tl.sigmoid(z)  # 应用SiLU门控：x = x * silu(z)
    if not IS_RMS_NORM:  # 如果是LayerNorm
        mean = tl.sum(x, axis=0) / N  # 计算均值
        tl.store(Mean + row, mean)  # 存储均值
        xbar = tl.where(cols < N, x - mean, 0.0)  # 计算中心化后的值
        var = tl.sum(xbar * xbar, axis=0) / N  # 计算方差
    else:  # 如果是RMS归一化
        xbar = tl.where(cols < N, x, 0.0)  # 不做中心化，直接使用原值
        var = tl.sum(xbar * xbar, axis=0) / N  # 计算均方值
    rstd = 1 / tl.sqrt(var + eps)  # 计算标准差倒数
    tl.store(Rstd + row, rstd)  # 存储标准差倒数
    # Normalize and apply linear transformation  # 归一化并应用线性变换
    mask = cols < N  # 创建列掩码
    w = tl.load(W + cols, mask=mask).to(tl.float32)  # 加载权重
    if HAS_BIAS:  # 如果有偏置
        b = tl.load(B + cols, mask=mask).to(tl.float32)  # 加载偏置
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd  # 归一化：LayerNorm减均值，RMSNorm不减
    y = x_hat * w + b if HAS_BIAS else x_hat * w  # 应用仿射变换：y = x_hat * w + b
    if HAS_Z and NORM_BEFORE_GATE:  # 如果有门控且在归一化之前门控
        z = tl.load(Z + cols, mask=mask).to(tl.float32)  # 加载门控数据
        y *= z * tl.sigmoid(z)  # 应用SiLU门控：y = y * silu(z)
    # Write output  # 写入输出
    tl.store(Y + cols, y, mask=mask)  # 将结果存储到输出指针


def _layer_norm_fwd(  # 层归一化前向传播主机函数
    x,  # 输入张量
    weight,  # 权重张量
    bias,  # 偏置张量
    eps,  # epsilon值
    z=None,  # 门控分支张量，默认为None
    out=None,  # 输出张量，默认为None
    group_size=None,  # 分组大小，默认为None
    norm_before_gate=True,  # 是否在门控之前归一化，默认为True
    is_rms_norm=False,  # 是否为RMS归一化，默认为False
):
    M, N = x.shape  # 获取输入的行数和列数
    if group_size is None:  # 如果未指定分组大小
        group_size = N  # 默认分组大小等于列数
    assert N % group_size == 0  # 断言列数能被分组大小整除
    ngroups = N // group_size  # 计算分组数
    assert x.stride(-1) == 1  # 断言输入在最内层维度上是连续的
    if z is not None:  # 如果有门控分支
        assert z.stride(-1) == 1  # 断言门控在最内层维度上是连续的
        assert z.shape == (M, N)  # 断言门控形状与输入相同
    assert weight.shape == (N,)  # 断言权重形状正确
    assert weight.stride(-1) == 1  # 断言权重在最内层维度上是连续的
    if bias is not None:  # 如果有偏置
        assert bias.stride(-1) == 1  # 断言偏置在最内层维度上是连续的
        assert bias.shape == (N,)  # 断言偏置形状正确
    # allocate output  # 分配输出
    if out is not None:  # 如果提供了输出张量
        assert out.shape == x.shape  # 断言输出形状与输入相同
    else:  # 如果未提供输出张量
        out = torch.empty_like(x)  # 创建与输入同形状的空张量
    assert out.stride(-1) == 1  # 断言输出在最内层维度上是连续的
    mean = (  # 分配均值缓冲区
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)  # 创建均值张量
        if not is_rms_norm  # 如果不是RMS归一化才需要均值
        else None  # RMS归一化不需要均值
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)  # 分配标准差倒数缓冲区
    # Less than 64KB per feature: enqueue fused kernel  # 每个特征少于64KB时：入队融合内核
    MAX_FUSED_SIZE = 65536 // x.element_size()  # 计算最大融合大小
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))  # 计算块大小
    if group_size > BLOCK_N:  # 如果分组大小超过块大小
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")  # 抛出不支持错误
    # heuristics for number of warps  # 线程束数量的启发式选择
    num_warps = min(max(BLOCK_N // 256, 1), 8)  # 根据块大小计算线程束数量，最少1最多8
    grid = (M, ngroups)  # 设置内核启动网格大小
    with torch.get_device_module(x.device).device(x.device.index):  # 在对应设备上执行
        _layer_norm_fwd_1pass_kernel[grid](  # 启动层归一化内核
            x,  # 输入
            out,  # 输出
            weight,  # 权重
            bias,  # 偏置
            z,  # 门控分支
            mean,  # 均值缓冲区
            rstd,  # 标准差倒数缓冲区
            x.stride(0),  # 输入行步长
            out.stride(0),  # 输出行步长
            z.stride(0) if z is not None else 0,  # 门控行步长
            M,  # 行数
            group_size,  # 分组大小
            eps,  # epsilon值
            BLOCK_N=BLOCK_N,  # 块大小
            NORM_BEFORE_GATE=norm_before_gate,  # 是否在门控前归一化
            IS_RMS_NORM=is_rms_norm,  # 是否RMS归一化
            num_warps=num_warps,  # 线程束数量
        )
    return out, mean, rstd  # 返回输出、均值和标准差倒数


def rms_norm_gated(  # RMS门控归一化函数
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True  # 参数：输入、权重、偏置、门控、epsilon、分组大小、门控顺序
):
    x_shape_og = x.shape  # 保存原始形状
    # reshape input data into 2D tensor  # 将输入数据重塑为2D张量
    x = x.reshape(-1, x.shape[-1])  # 重塑为(行数, 列数)的2D张量
    if x.stride(-1) != 1:  # 如果输入在最内层维度不连续
        x = x.contiguous()  # 转为连续张量
    if z is not None:  # 如果有门控分支
        assert z.shape == x_shape_og  # 断言门控形状与原始输入相同
        z = z.reshape(-1, z.shape[-1])  # 将门控重塑为2D张量
        if z.stride(-1) != 1:  # 如果门控在最内层维度不连续
            z = z.contiguous()  # 转为连续张量
    weight = weight.contiguous()  # 确保权重连续
    if bias is not None:  # 如果有偏置
        bias = bias.contiguous()  # 确保偏置连续
    y, _, _ = _layer_norm_fwd(  # 调用层归一化前向传播
        x,  # 输入
        weight,  # 权重
        bias,  # 偏置
        eps,  # epsilon值
        z=z,  # 门控分支
        group_size=group_size,  # 分组大小
        norm_before_gate=norm_before_gate,  # 门控顺序
        is_rms_norm=True,  # 使用RMS归一化
    )

    return y.reshape(x_shape_og)  # 将输出重塑回原始形状并返回
