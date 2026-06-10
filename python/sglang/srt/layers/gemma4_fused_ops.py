# Gemma4解码器层的融合Triton内核
# 本文件实现了Gemma4模型解码器层的融合操作，将标准RMSNorm + 残差加法
# （+ 可选标量乘法）融合到单次内核传递中，以减少内核启动开销。

"""Fused triton kernels for Gemma4 decoder layer operations.
Gemma4解码器层操作的融合Triton内核。

Fuses standard RMSNorm + residual-add (+ optional scalar multiply) into
将标准RMSNorm + 残差加法（+ 可选标量乘法）融合到
a single kernel pass to reduce kernel launch overhead.
单次内核传递中以减少内核启动开销。
"""

from typing import Optional  # 导入可选类型 # 导入可选类型

import torch  # 导入PyTorch库 # 导入PyTorch库
import triton  # 导入Triton库 # 导入Triton库
import triton.language as tl  # 导入Triton语言模块并简写为tl # 导入Triton语言模块并简写为tl


# Gemma RMSNorm + 残差 + 标量乘法融合内核
@triton.jit
def _gemma_rmsnorm_residual_kernel(  # Gemma RMSNorm+残差+标量内核
    X_ptr,  # 输入X指针 # 输入X指针
    W_ptr,  # 归一化权重指针 # 归一化权重指针
    Residual_ptr,  # 残差指针 # 残差指针
    Scalar_ptr,  # 标量指针 # 标量指针
    Out_ptr,  # 输出指针 # 输出指针
    stride_x,  # X的行步长 # X的行步长
    stride_r,  # 残差的行步长 # 残差的行步长
    stride_o,  # 输出的行步长 # 输出的行步长
    N,  # 列数（隐藏维度） # 列数（隐藏维度）
    eps,  # epsilon值 # epsilon值
    HAS_SCALAR: tl.constexpr,  # 是否有标量乘法的编译时常量 # 是否有标量乘法
    BLOCK_SIZE: tl.constexpr,  # 块大小的编译时常量 # 块大小
):
    """Fused kernel: out = rmsnorm(x, w) + residual [* scalar]
    融合内核：输出 = rmsnorm(x, w) + 残差 [* 标量]

    When HAS_SCALAR is True, also multiplies by a scalar loaded from Scalar_ptr.
    当HAS_SCALAR为True时，还会乘以从Scalar_ptr加载的标量。
    """
    row = tl.program_id(0)  # 获取行程序ID # 获取行程序ID
    cols = tl.arange(0, BLOCK_SIZE)  # 生成列索引 # 生成列索引
    mask = cols < N  # 生成列掩码 # 生成列掩码

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)  # 加载输入x # 加载输入x
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载权重w # 加载权重w
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(  # 加载残差r
        tl.float32  # 转换为float32 # 转换为float32
    )

    var = tl.sum(x * x, axis=0) / N  # 计算方差 # 计算方差
    rrms = tl.rsqrt(var + eps)  # 计算标准差倒数 # 计算标准差倒数
    out = x * rrms * w + r  # 计算归一化加残差 # 计算归一化加残差

    if HAS_SCALAR:  # 如果有标量乘法 # 如果有标量乘法
        scalar = tl.load(Scalar_ptr).to(tl.float32)  # 加载标量 # 加载标量
        out = out * scalar  # 乘以标量 # 乘以标量

    tl.store(Out_ptr + row * stride_o + cols, out.to(x.dtype), mask=mask)  # 存储输出 # 存储输出


# 融合(rmsnorm(x) + 残差) * 标量的Python包装函数
def gemma_rmsnorm_residual_scalar(  # Gemma RMSNorm+残差+标量函数
    x: torch.Tensor,  # 输入张量 # 输入张量
    weight: torch.Tensor,  # 归一化权重 # 归一化权重
    residual: torch.Tensor,  # 残差张量 # 残差张量
    scalar: torch.Tensor,  # 标量张量 # 标量张量
    eps: float = 1e-6,  # epsilon值，默认1e-6 # epsilon值
) -> torch.Tensor:  # 返回输出张量 # 返回输出张量
    """Fused (rmsnorm(x) + residual) * scalar.
    融合操作 (rmsnorm(x) + 残差) * 标量。
    """
    assert x.dim() == 2 and x.stride(-1) == 1, "Expected contiguous 2D input"  # 断言输入为连续2D张量 # 断言输入为连续2D张量
    M, N = x.shape  # 获取行数和列数 # 获取行数和列数
    BLOCK_SIZE = triton.next_power_of_2(N)  # 计算大于等于N的最小2的幂 # 计算大于等于N的最小2的幂
    out = torch.empty_like(x)  # 分配输出张量 # 分配输出张量

    _gemma_rmsnorm_residual_kernel[(M,)](  # 启动融合内核
        x,  # 输入x # 输入x
        weight,  # 权重 # 权重
        residual,  # 残差 # 残差
        scalar,  # 标量 # 标量
        out,  # 输出 # 输出
        x.stride(0),  # x行步长 # x行步长
        residual.stride(0),  # 残差行步长 # 残差行步长
        out.stride(0),  # 输出行步长 # 输出行步长
        N,  # 列数 # 列数
        eps,  # epsilon值 # epsilon值
        HAS_SCALAR=True,  # 启用标量乘法 # 启用标量乘法
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小 # 块大小
    )
    return out  # 返回输出 # 返回输出


# Gemma双重RMSNorm + 残差 + 标量乘法融合内核
@triton.jit
def _gemma_dual_rmsnorm_residual_kernel(  # Gemma双重RMSNorm+残差+标量内核
    X1_ptr,  # 第一个输入X1指针 # 第一个输入X1指针
    W1_ptr,  # 第一个归一化权重W1指针 # 第一个归一化权重W1指针
    X2_ptr,  # 第二个输入X2指针 # 第二个输入X2指针
    W2_ptr,  # 第二个归一化权重W2指针 # 第二个归一化权重W2指针
    W3_ptr,  # 第三层归一化权重W3指针 # 第三层归一化权重W3指针
    Residual_ptr,  # 残差指针 # 残差指针
    Scalar_ptr,  # 标量指针 # 标量指针
    Out_ptr,  # 输出指针 # 输出指针
    stride_x1,  # X1行步长 # X1行步长
    stride_x2,  # X2行步长 # X2行步长
    stride_r,  # 残差行步长 # 残差行步长
    stride_o,  # 输出行步长 # 输出行步长
    N,  # 列数 # 列数
    eps1,  # 第一次RMSNorm的epsilon # 第一次RMSNorm的epsilon
    eps2,  # 第二次RMSNorm的epsilon # 第二次RMSNorm的epsilon
    eps3,  # 第三次RMSNorm的epsilon # 第三次RMSNorm的epsilon
    BLOCK_SIZE: tl.constexpr,  # 块大小的编译时常量 # 块大小
):
    """Fused: out = (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar"""
    # 融合：输出 = (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + 残差) * 标量
    row = tl.program_id(0)  # 获取行程序ID # 获取行程序ID
    cols = tl.arange(0, BLOCK_SIZE)  # 生成列索引 # 生成列索引
    mask = cols < N  # 生成列掩码 # 生成列掩码

    x1 = tl.load(X1_ptr + row * stride_x1 + cols, mask=mask, other=0.0).to(tl.float32)  # 加载x1 # 加载x1
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载w1 # 加载w1
    x2 = tl.load(X2_ptr + row * stride_x2 + cols, mask=mask, other=0.0).to(tl.float32)  # 加载x2 # 加载x2
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载w2 # 加载w2
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载w3 # 加载w3
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(  # 加载残差
        tl.float32  # 转换为float32 # 转换为float32
    )

    var1 = tl.sum(x1 * x1, axis=0) / N  # 计算x1的方差 # 计算x1的方差
    norm1 = x1 * tl.rsqrt(var1 + eps1) * w1  # 对x1进行RMSNorm # 对x1进行RMSNorm

    var2 = tl.sum(x2 * x2, axis=0) / N  # 计算x2的方差 # 计算x2的方差
    norm2 = x2 * tl.rsqrt(var2 + eps2) * w2  # 对x2进行RMSNorm # 对x2进行RMSNorm

    combined = norm1 + norm2  # 合并两个归一化结果 # 合并两个归一化结果

    var3 = tl.sum(combined * combined, axis=0) / N  # 计算合并结果的方差 # 计算合并结果的方差
    norm3 = combined * tl.rsqrt(var3 + eps3) * w3  # 对合并结果进行RMSNorm # 对合并结果进行RMSNorm

    scalar = tl.load(Scalar_ptr).to(tl.float32)  # 加载标量 # 加载标量
    out = (norm3 + r) * scalar  # 加残差后乘以标量 # 加残差后乘以标量

    tl.store(Out_ptr + row * stride_o + cols, out.to(x1.dtype), mask=mask)  # 存储输出 # 存储输出


# Gemma QKV RMSNorm融合内核
@triton.jit
def _gemma_qkv_rmsnorm_kernel(  # Gemma QKV RMSNorm内核
    Q_ptr,  # Q张量指针 # Q张量指针
    K_ptr,  # K张量指针 # K张量指针
    V_ptr,  # V张量指针 # V张量指针
    Q_w_ptr,  # Q归一化权重指针 # Q归一化权重指针
    K_w_ptr,  # K归一化权重指针 # K归一化权重指针
    stride_q_m,  # Q的token步长 # Q的token步长
    stride_k_m,  # K的token步长 # K的token步长
    stride_v_m,  # V的token步长 # V的token步长
    NUM_Q_HEADS: tl.constexpr,  # Q头数量的编译时常量 # Q头数量
    NUM_KV_HEADS: tl.constexpr,  # KV头数量的编译时常量 # KV头数量
    HEAD_DIM: tl.constexpr,  # 头维度的编译时常量 # 头维度
    eps,  # epsilon值 # epsilon值
    HAS_KV: tl.constexpr,  # 是否有KV的编译时常量 # 是否有KV
    BLOCK: tl.constexpr,  # 块大小的编译时常量 # 块大小
):
    """Per-token fused RMSNorm of Q (with q_w), K (with k_w), V (no scale).
    逐token融合RMSNorm：Q（带q_w权重）、K（带k_w权重）、V（无缩放）。

    Layout assumption: each tensor's last dim packs (num_heads, head_dim) contiguously
    布局假设：每个张量的最后一维连续打包(num_heads, head_dim)
    so per-head offset is `h * HEAD_DIM`. The token (M) stride is taken from
    因此每个头的偏移是`h * HEAD_DIM`。token(M)步长取自
    stride_*_m so the kernel works on strided views (e.g. slices of a larger
    stride_*_m，因此内核可以处理跨步视图（例如，更大
    qkv buffer produced by `qkv.split`) without requiring `.contiguous()` copies.
    的qkv缓冲区中由`qkv.split`产生的切片），而无需`.contiguous()`拷贝。
    V uses `weight=ones` semantics so the multiply-by-weight is omitted.
    V使用`weight=ones`语义，因此省略了乘以权重的操作。
    """
    m = tl.program_id(0)  # 获取token程序ID # 获取token程序ID
    cols = tl.arange(0, BLOCK)  # 生成列索引 # 生成列索引
    mask = cols < HEAD_DIM  # 生成列掩码 # 生成列掩码

    qw = tl.load(Q_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载Q权重 # 加载Q权重

    # Q heads
    # Q头
    for h in tl.static_range(NUM_Q_HEADS):  # 遍历每个Q头 # 遍历每个Q头
        off = m * stride_q_m + h * HEAD_DIM + cols  # 计算偏移 # 计算偏移
        x = tl.load(Q_ptr + off, mask=mask, other=0.0).to(tl.float32)  # 加载Q头数据 # 加载Q头数据
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)  # 计算RMSNorm标准差倒数 # 计算RMSNorm标准差倒数
        out = x * rrms * qw  # 应用RMSNorm和权重 # 应用RMSNorm和权重
        tl.store(Q_ptr + off, out.to(Q_ptr.dtype.element_ty), mask=mask)  # 原地存储Q归一化结果 # 原地存储Q归一化结果

    if HAS_KV:  # 如果有KV # 如果有KV
        kw = tl.load(K_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)  # 加载K权重 # 加载K权重

        # K heads
        # K头
        for h in tl.static_range(NUM_KV_HEADS):  # 遍历每个K头 # 遍历每个K头
            off = m * stride_k_m + h * HEAD_DIM + cols  # 计算偏移 # 计算偏移
            x = tl.load(K_ptr + off, mask=mask, other=0.0).to(tl.float32)  # 加载K头数据 # 加载K头数据
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)  # 计算RMSNorm标准差倒数 # 计算RMSNorm标准差倒数
            out = x * rrms * kw  # 应用RMSNorm和权重 # 应用RMSNorm和权重
            tl.store(K_ptr + off, out.to(K_ptr.dtype.element_ty), mask=mask)  # 原地存储K归一化结果 # 原地存储K归一化结果

        # V heads (no scaling: V-norm uses weight=ones)
        # V头（无缩放：V归一化使用weight=ones）
        for h in tl.static_range(NUM_KV_HEADS):  # 遍历每个V头 # 遍历每个V头
            off = m * stride_v_m + h * HEAD_DIM + cols  # 计算偏移 # 计算偏移
            x = tl.load(V_ptr + off, mask=mask, other=0.0).to(tl.float32)  # 加载V头数据 # 加载V头数据
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)  # 计算RMSNorm标准差倒数 # 计算RMSNorm标准差倒数
            out = x * rrms  # 应用RMSNorm（无权重缩放） # 应用RMSNorm（无权重缩放）
            tl.store(V_ptr + off, out.to(V_ptr.dtype.element_ty), mask=mask)  # 原地存储V归一化结果 # 原地存储V归一化结果


# Gemma QKV RMSNorm的Python包装函数
def gemma_qkv_rmsnorm(  # Gemma QKV RMSNorm函数
    q: torch.Tensor,  # Q张量 # Q张量
    k: Optional[torch.Tensor],  # K张量（可选） # K张量
    v: Optional[torch.Tensor],  # V张量（可选） # V张量
    q_weight: torch.Tensor,  # Q归一化权重 # Q归一化权重
    k_weight: Optional[torch.Tensor],  # K归一化权重（可选） # K归一化权重
    num_q_heads: int,  # Q头数量 # Q头数量
    num_kv_heads: int,  # KV头数量 # KV头数量
    head_dim: int,  # 头维度 # 头维度
    eps: float = 1e-6,  # epsilon值，默认1e-6 # epsilon值
) -> None:  # 无返回值（原地修改） # 无返回值
    """In-place fused RMSNorm on Q, K, V for Gemma4 attention.
    Gemma4注意力的原地融合RMSNorm，作用于Q、K、V。

    All three norms compute `x * rsqrt(mean(x^2) + eps)` independently per head.
    三个归一化均独立按头计算 `x * rsqrt(mean(x^2) + eps)`。
    Q is scaled by `q_weight`, K by `k_weight`, V by 1 (Gemma4's V-norm has
    Q由`q_weight`缩放，K由`k_weight`缩放，V由1缩放（Gemma4的V归一化
    `with_scale=False`).
    的`with_scale=False`）。

    Inputs may be 2D `(M, num_heads * head_dim)` or strided views of a larger
    输入可以是2D的`(M, num_heads * head_dim)`或更大缓冲区的跨步视图
    buffer (such as q/k/v slices from `qkv.split`). The kernel uses the actual
    （例如从`qkv.split`得到的q/k/v切片）。内核使用实际的
    `stride(0)` so no `.contiguous()` copy is required. Within a token, the
    `stride(0)`，因此不需要`.contiguous()`拷贝。在一个token内，
    last dim must be contiguous so heads pack as `h * head_dim` offsets.
    最后一维必须是连续的，以便头按`h * head_dim`偏移打包。

    If k and v are both None (KV-shared layer), only Q is normalized.
    如果k和v都为None（KV共享层），则仅对Q进行归一化。
    """
    assert q.is_cuda  # 断言Q在CUDA上 # 断言Q在CUDA上
    assert q.stride(-1) == 1, "Q's last dim must be contiguous"  # 断言Q最后一维连续 # 断言Q最后一维连续
    assert q_weight.shape[-1] == head_dim  # 断言Q权重维度匹配 # 断言Q权重维度匹配
    M = q.shape[0] if q.dim() >= 2 else 1  # 获取token数量 # 获取token数量
    BLOCK = triton.next_power_of_2(head_dim)  # 计算大于等于head_dim的最小2的幂 # 计算大于等于head_dim的最小2的幂

    has_kv = k is not None and v is not None  # 判断是否有KV # 判断是否有KV
    if has_kv:  # 如果有KV # 如果有KV
        assert k.is_cuda and v.is_cuda  # 断言K和V在CUDA上 # 断言K和V在CUDA上
        assert k.stride(-1) == 1 and v.stride(-1) == 1  # 断言K和V最后一维连续 # 断言K和V最后一维连续
        assert k_weight is not None and k_weight.shape[-1] == head_dim  # 断言K权重维度匹配 # 断言K权重维度匹配

    _gemma_qkv_rmsnorm_kernel[(M,)](  # 启动QKV RMSNorm内核
        q,  # Q张量 # Q张量
        k if has_kv else q,  # K张量（无KV时用Q占位） # K张量
        v if has_kv else q,  # V张量（无KV时用Q占位） # V张量
        q_weight,  # Q权重 # Q权重
        k_weight if has_kv else q_weight,  # K权重（无KV时用Q权重占位） # K权重
        q.stride(0),  # Q token步长 # Q token步长
        k.stride(0) if has_kv else 0,  # K token步长 # K token步长
        v.stride(0) if has_kv else 0,  # V token步长 # V token步长
        NUM_Q_HEADS=num_q_heads,  # Q头数量 # Q头数量
        NUM_KV_HEADS=num_kv_heads if has_kv else 0,  # KV头数量 # KV头数量
        HEAD_DIM=head_dim,  # 头维度 # 头维度
        eps=eps,  # epsilon值 # epsilon值
        HAS_KV=has_kv,  # 是否有KV # 是否有KV
        BLOCK=BLOCK,  # 块大小 # 块大小
    )


# Gemma双重RMSNorm + 残差 + 标量乘法的Python包装函数
def gemma_dual_rmsnorm_residual_scalar(  # Gemma双重RMSNorm+残差+标量函数
    x1: torch.Tensor,  # 第一个输入张量 # 第一个输入张量
    weight1: torch.Tensor,  # 第一个归一化权重 # 第一个归一化权重
    x2: torch.Tensor,  # 第二个输入张量 # 第二个输入张量
    weight2: torch.Tensor,  # 第二个归一化权重 # 第二个归一化权重
    weight3: torch.Tensor,  # 第三层归一化权重 # 第三层归一化权重
    residual: torch.Tensor,  # 残差张量 # 残差张量
    scalar: torch.Tensor,  # 标量张量 # 标量张量
    eps1: float = 1e-6,  # 第一次RMSNorm的epsilon # 第一次RMSNorm的epsilon
    eps2: float = 1e-6,  # 第二次RMSNorm的epsilon # 第二次RMSNorm的epsilon
    eps3: float = 1e-6,  # 第三次RMSNorm的epsilon # 第三次RMSNorm的epsilon
) -> torch.Tensor:  # 返回输出张量 # 返回输出张量
    """Fused (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar."""
    # 融合操作 (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + 残差) * 标量
    assert x1.dim() == 2 and x1.stride(-1) == 1  # 断言x1为连续2D张量 # 断言x1为连续2D张量
    M, N = x1.shape  # 获取行数和列数 # 获取行数和列数
    BLOCK_SIZE = triton.next_power_of_2(N)  # 计算大于等于N的最小2的幂 # 计算大于等于N的最小2的幂
    out = torch.empty_like(x1)  # 分配输出张量 # 分配输出张量

    _gemma_dual_rmsnorm_residual_kernel[(M,)](  # 启动双重RMSNorm+残差内核
        x1,  # 第一个输入 # 第一个输入
        weight1,  # 第一个权重 # 第一个权重
        x2,  # 第二个输入 # 第二个输入
        weight2,  # 第二个权重 # 第二个权重
        weight3,  # 第三层权重 # 第三层权重
        residual,  # 残差 # 残差
        scalar,  # 标量 # 标量
        out,  # 输出 # 输出
        x1.stride(0),  # x1行步长 # x1行步长
        x2.stride(0),  # x2行步长 # x2行步长
        residual.stride(0),  # 残差行步长 # 残差行步长
        out.stride(0),  # 输出行步长 # 输出行步长
        N,  # 列数 # 列数
        eps1,  # 第一次RMSNorm的epsilon # 第一次RMSNorm的epsilon
        eps2,  # 第二次RMSNorm的epsilon # 第二次RMSNorm的epsilon
        eps3,  # 第三次RMSNorm的epsilon # 第三次RMSNorm的epsilon
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小 # 块大小
    )
    return out  # 返回输出 # 返回输出
