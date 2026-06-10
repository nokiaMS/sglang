# DSV4 hc_head LM-head混合器的融合Triton内核，实现RMS归一化+线性+Sigmoid门控+加权和的融合计算
# SPDX-License-Identifier: Apache-2.0
"""Fused triton kernel for the DSV4 hc_head LM-head mixer.
DSV4 hc_head LM-head混合器的融合Triton内核。

Reference torch implementation (deepseek_v4.py DeepseekV4Model.hc_head):
参考PyTorch实现 (deepseek_v4.py DeepseekV4Model.hc_head):

    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)

Shapes (DSV4-Pro, hc_mult=4, hidden_size=7168 typical):
形状 (DSV4-Pro, hc_mult=4, hidden_size=7168 典型值):
    x      : (T, hc_mult, hidden_size)            bf16
    hc_fn  : (hc_mult, hc_mult * hidden_size)     fp32
    scale  : (1,)                                 fp32
    base   : (hc_mult,)                           fp32
    out y  : (T, hidden_size)                     bf16

This is a one-shot LM-head op (fires once per forward on the last PP rank), so
we use a 1-CTA-per-token design that does two passes over x without split-K.
这是一个一次性LM-head操作（在最后一个PP rank上每次前向传播执行一次），
因此我们采用1-CTA-per-token的设计，对x进行两轮遍历而不使用split-K。
"""

from __future__ import annotations  # 启用延迟注解评估

import torch  # 导入PyTorch核心库
import triton  # 导入Triton编译器
import triton.language as tl  # 导入Triton语言


@triton.jit  # Triton JIT编译装饰器
def _hc_head_kernel(  # 定义hc_head融合内核函数
    x_ptr,  # 输入张量x的指针
    fn_ptr,  # 混合函数权重指针
    scale_ptr,  # 缩放因子指针
    base_ptr,  # 偏置基底指针
    y_ptr,  # 输出张量y的指针
    hidden_size: tl.constexpr,  # 隐藏层大小（编译时常量）
    HC_MULT: tl.constexpr,  # HC乘数（编译时常量）
    K_TOTAL: tl.constexpr,  # K维度总大小（编译时常量）
    BLOCK_K: tl.constexpr,  # K维度块大小（编译时常量）
    BLOCK_D: tl.constexpr,  # D维度块大小（编译时常量）
    norm_eps: tl.constexpr,  # 归一化epsilon（编译时常量）
    hc_eps: tl.constexpr,  # HC epsilon（编译时常量）
):
    pid = tl.program_id(0).to(tl.int64)  # 获取当前程序的token索引

    # ---------- Pass 1: sum_sq over flattened K dim, plus hc_mult inner products ----------
    # ---------- 第一轮：在展平的K维度上计算平方和，以及hc_mult个内积 ----------
    sumsq = tl.zeros((), dtype=tl.float32)  # 初始化平方和累加器
    mix = tl.zeros((HC_MULT,), dtype=tl.float32)  # 初始化混合结果累加器

    x_row = x_ptr + pid * K_TOTAL  # 计算当前token的x行起始地址
    m_idx = tl.arange(0, HC_MULT)  # 创建HC乘数索引

    for k_off in tl.range(0, K_TOTAL, BLOCK_K):  # 按BLOCK_K分块遍历K维度
        k_offs = k_off + tl.arange(0, BLOCK_K)  # 计算K维度偏移
        k_mask = k_offs < K_TOTAL  # 创建边界掩码
        x_tile = tl.load(x_row + k_offs, mask=k_mask, other=0.0).to(tl.float32)  # 加载x数据块并转为float32

        sumsq += tl.sum(x_tile * x_tile, axis=0)  # 累加平方和

        fn_offs = m_idx[:, None] * K_TOTAL + k_offs[None, :]  # 计算函数权重的偏移
        fn_mask = (m_idx[:, None] < HC_MULT) & k_mask[None, :]  # 创建函数权重掩码
        fn_tile = tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0)  # 加载函数权重数据块
        mix += tl.sum(fn_tile * x_tile[None, :], axis=1)  # 计算内积并累加到混合结果

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + norm_eps)  # 计算RMS归一化的rsqrt值
    scale_v = tl.load(scale_ptr).to(tl.float32)  # 加载缩放因子
    base_v = tl.load(base_ptr + m_idx).to(tl.float32)  # 加载偏置基底

    # pre[m] = sigmoid(mix[m] * rsqrt * scale + base[m]) + hc_eps
    # pre[m] = sigmoid(mix[m] * rsqrt * scale + base[m]) + hc_eps
    pre = tl.sigmoid(mix * rsqrt * scale_v + base_v) + hc_eps  # 计算Sigmoid门控值并加上epsilon

    # ---------- Pass 2: y[d] = sum_m pre[m] * x[m, d]  for d in range(hidden_size) ----------
    # ---------- 第二轮：y[d] = sum_m pre[m] * x[m, d]，d从0到hidden_size-1 ----------
    y_row = y_ptr + pid * hidden_size  # 计算当前token的y行起始地址

    for d_off in tl.range(0, hidden_size, BLOCK_D):  # 按BLOCK_D分块遍历隐藏维度
        d_offs = d_off + tl.arange(0, BLOCK_D)  # 计算D维度偏移
        d_mask = d_offs < hidden_size  # 创建边界掩码

        x_offs = m_idx[:, None] * hidden_size + d_offs[None, :]  # 计算x的偏移索引
        x_mask = (m_idx[:, None] < HC_MULT) & d_mask[None, :]  # 创建x加载掩码
        x_block = tl.load(x_row + x_offs, mask=x_mask, other=0.0).to(tl.float32)  # 加载x数据块

        y_block = tl.sum(pre[:, None] * x_block, axis=0)  # 计算加权和：y = sum_m pre[m] * x[m, d]

        tl.store(y_row + d_offs, y_block.to(y_ptr.dtype.element_ty), mask=d_mask)  # 存储结果到输出张量


def fused_hc_head(  # 定义融合hc_head函数
    x: torch.Tensor,  # 输入张量，形状(T, hc_mult, hidden_size)
    hc_fn: torch.Tensor,  # 混合函数权重，形状(hc_mult, hc_mult * hidden_size)
    hc_scale: torch.Tensor,  # 缩放因子，形状(1,)
    hc_base: torch.Tensor,  # 偏置基底，形状(hc_mult,)
    norm_eps: float,  # RMS归一化epsilon
    hc_eps: float,  # 加法epsilon
) -> torch.Tensor:  # 返回输出张量
    """Fused (RMSNorm + Linear + Sigmoid-gate + weighted-sum) for the DSV4 hc_head.
    DSV4 hc_head的融合操作（RMS归一化 + 线性 + Sigmoid门控 + 加权和）。

    Args:
        x         : (T, hc_mult, hidden_size) bf16/fp16, must be contiguous
        x         : (T, hc_mult, hidden_size) bf16/fp16，必须连续
        hc_fn     : (hc_mult, hc_mult * hidden_size) fp32, contiguous
        hc_fn     : (hc_mult, hc_mult * hidden_size) fp32，连续
        hc_scale  : (1,) fp32 scalar
        hc_scale  : (1,) fp32 标量
        hc_base   : (hc_mult,) fp32
        hc_base   : (hc_mult,) fp32
        norm_eps  : RMS epsilon
        norm_eps  : RMS归一化epsilon
        hc_eps    : additive epsilon after sigmoid
        hc_eps    : sigmoid后的加法epsilon

    Returns:
        y : (T, hidden_size) same dtype as x
        y : (T, hidden_size) 与x相同的数据类型
    """
    assert x.is_contiguous(), "x must be contiguous"  # 断言x必须连续
    assert hc_fn.is_contiguous(), "hc_fn must be contiguous"  # 断言hc_fn必须连续
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32  # 断言缩放和基底为float32
    assert hc_fn.dtype == torch.float32  # 断言函数权重为float32
    assert x.dim() == 3, f"x must be 3D (T, hc_mult, hidden_size), got {x.shape}"  # 断言x为3维

    T, hc_mult, hidden_size = x.shape  # 解析x的形状
    assert hc_fn.shape == (hc_mult, hc_mult * hidden_size), (  # 断言hc_fn形状正确
        f"hc_fn shape {hc_fn.shape} does not match (hc_mult={hc_mult}, "
        f"hc_mult*hidden_size={hc_mult * hidden_size})"
    )
    assert hc_base.shape == (hc_mult,)  # 断言hc_base形状正确
    assert hc_scale.numel() == 1  # 断言hc_scale为标量

    y = torch.empty((T, hidden_size), dtype=x.dtype, device=x.device)  # 分配输出张量

    if T == 0:  # 如果没有token
        return y  # 直接返回空张量

    BLOCK_K = 512  # K维度块大小
    BLOCK_D = 512  # D维度块大小

    hc_mult_pow2 = max(1, triton.next_power_of_2(hc_mult))  # 计算大于等于hc_mult的最小2的幂

    grid = (T,)  # 设置内核网格大小为token数
    _hc_head_kernel[grid](  # 调用hc_head内核
        x,  # 输入张量
        hc_fn,  # 混合函数权重
        hc_scale,  # 缩放因子
        hc_base,  # 偏置基底
        y,  # 输出张量
        hidden_size=hidden_size,  # 隐藏层大小
        HC_MULT=hc_mult_pow2,  # HC乘数（2的幂）
        K_TOTAL=hc_mult * hidden_size,  # K维度总大小
        BLOCK_K=BLOCK_K,  # K维度块大小
        BLOCK_D=BLOCK_D,  # D维度块大小
        norm_eps=norm_eps,  # 归一化epsilon
        hc_eps=hc_eps,  # HC epsilon
        num_warps=4,  # 每个CTA使用的warp数
    )
    return y  # 返回输出张量
