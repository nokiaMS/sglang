# DeepSeek V4 旋转位置编码（RoPE）实现，包含 Triton 内核和融合 RMSNorm+RoPE 操作。

import math  # 导入数学库
from functools import lru_cache  # 导入 LRU 缓存装饰器
from typing import Optional  # 导入可选类型

import torch  # 导入 PyTorch
import triton  # 导入 Triton
import triton.language as tl  # 导入 Triton 语言

try:
    import tilelang  # 尝试导入 tilelang

    tilelang.set_log_level("WARNING")  # 设置 tilelang 日志级别为警告
    pass_configs = {  # 配置 tilelang 编译选项
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # 禁用 warp 特化
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  # 禁用 TMA 降低
    }
except ImportError:
    pass  # tilelang 不可用时忽略

FP8 = "float8_e4m3"  # FP8 数据类型字符串
BF16 = "bfloat16"  # BF16 数据类型字符串
FP32 = "float32"  # FP32 数据类型字符串
INT32 = "int32"  # INT32 数据类型字符串


@lru_cache(2)  # 使用 LRU 缓存，最多缓存 2 个结果
def precompute_freqs_cis(  # 预计算旋转位置编码的复数频率
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:

    def find_correction_dim(num_rotations, dim, base, max_seq_len):  # 查找修正维度
        return (  # 返回修正维度值
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):  # 查找修正范围
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))  # 下界取整
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))  # 上界取整
        return max(low, 0), min(high, dim - 1)  # 限制在 [0, dim-1] 范围内

    def linear_ramp_factor(min, max, dim):  # 计算线性渐变因子
        if min == max:  # 避免除零
            max += 0.001  # 微调避免除零
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)  # 线性函数
        ramp_func = torch.clamp(linear_func, 0, 1)  # 裁剪到 [0, 1]
        return ramp_func  # 返回渐变因子

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # 计算基础频率
    if original_seq_len > 0:  # 如果有原始序列长度
        low, high = find_correction_range(  # 查找修正范围
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)  # 计算平滑因子
        freqs = freqs / factor * (1 - smooth) + freqs * smooth  # 应用 YaRN 风格的频率缩放

    t = torch.arange(seqlen)  # 生成位置索引
    freqs = torch.outer(t, freqs)  # 计算外积得到位置-频率矩阵
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 转换为复数形式 (cos + i*sin)
    return freqs_cis  # 返回复数频率


@triton.jit  # Triton JIT 编译的旋转位置编码内核
def apply_rotary_emb_triton_kernel(  # 应用旋转位置编码的 Triton 内核
    x_ptr,  # 输入张量指针
    freqs_ptr,  # 频率张量指针
    positions_ptr,  # 位置索引指针
    rope_dim,  # RoPE 维度
    stride_x_batch,  # 输入批次步幅
    stride_x_head,  # 输入头步幅
    stride_x_dim,  # 输入维度步幅
    stride_freq_pos,  # 频率位置步幅
    stride_freq_dim,  # 频率维度步幅
    USE_POS: tl.constexpr,  # 是否使用显式位置索引
    IS_INVERSE: tl.constexpr,  # 是否为逆旋转
    IS_3D: tl.constexpr,  # 输入是否为 3D 张量
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid_batch = tl.program_id(0)  # 获取批次维度的程序 ID
    pid_head = tl.program_id(1)  # 获取头维度的程序 ID
    pid_dim = tl.program_id(2)  # 获取维度维度的程序 ID

    if USE_POS:  # 如果使用显式位置索引
        position = tl.load(positions_ptr + pid_batch)  # 从位置数组加载位置
    else:
        position = pid_batch  # 否则位置等于批次索引

    if IS_3D:  # 如果是 3D 输入
        base_offset = pid_batch * stride_x_batch + pid_head * stride_x_head  # 计算基础偏移（含头维度）
    else:
        base_offset = pid_batch * stride_x_batch  # 计算基础偏移（不含头维度）

    offs_pair = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算对偏移量
    mask = offs_pair < (rope_dim // 2)  # 生成掩码，防止越界

    offs_x_real = base_offset + offs_pair * 2 * stride_x_dim  # 实部偏移
    offs_x_imag = base_offset + (offs_pair * 2 + 1) * stride_x_dim  # 虚部偏移

    x_real = tl.load(x_ptr + offs_x_real, mask=mask, other=0.0).to(tl.float32)  # 加载实部
    x_imag = tl.load(x_ptr + offs_x_imag, mask=mask, other=0.0).to(tl.float32)  # 加载虚部

    offs_freq_real = position * stride_freq_pos + offs_pair * 2 * stride_freq_dim  # 频率实部偏移
    offs_freq_imag = position * stride_freq_pos + (offs_pair * 2 + 1) * stride_freq_dim  # 频率虚部偏移

    freq_real = tl.load(freqs_ptr + offs_freq_real, mask=mask, other=0.0)  # 加载频率实部（cos）
    freq_imag = tl.load(freqs_ptr + offs_freq_imag, mask=mask, other=0.0)  # 加载频率虚部（sin）

    if IS_INVERSE:  # 如果是逆旋转
        out_real = x_real * freq_real + x_imag * freq_imag  # 逆旋转实部
        out_imag = x_imag * freq_real - x_real * freq_imag  # 逆旋转虚部
    else:  # 正向旋转
        out_real = x_real * freq_real - x_imag * freq_imag  # 旋转实部
        out_imag = x_real * freq_imag + x_imag * freq_real  # 旋转虚部

    tl.store(x_ptr + offs_x_real, out_real, mask=mask)  # 存储旋转后的实部
    tl.store(x_ptr + offs_x_imag, out_imag, mask=mask)  # 存储旋转后的虚部


def apply_rotary_emb_triton(  # 应用旋转位置编码的 Triton 实现
    x: torch.Tensor,  # 输入张量
    freqs_cis: torch.Tensor,  # 复数频率张量
    positions: Optional[torch.Tensor] = None,  # 可选的位置索引
    inverse: bool = False,  # 是否为逆旋转
) -> torch.Tensor:  # 返回旋转编码后的张量
    is_3d = x.ndim == 3  # 判断输入是否为 3D

    if is_3d:  # 如果是 3D 输入
        batch_size, n_heads, rope_dim = x.shape  # 解包形状
    else:
        batch_size, rope_dim = x.shape  # 解包形状
        n_heads = 1  # 头数设为 1

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)  # 将复数频率转为实数表示 [cos, sin]

    BLOCK_SIZE = 128  # 块大小

    num_blocks_dim = triton.cdiv(rope_dim // 2, BLOCK_SIZE)  # 计算维度方向的块数
    grid = (batch_size, n_heads if is_3d else 1, num_blocks_dim)  # 设置网格大小

    if positions is not None:  # 如果提供了显式位置
        assert positions.shape == (  # 断言位置形状正确
            batch_size,
        ), f"positions shape {positions.shape} != ({batch_size},)"

        apply_rotary_emb_triton_kernel[grid](  # 启动内核，使用显式位置
            x,
            freqs_real,
            positions,
            rope_dim,
            x.stride(0),  # 批次步幅
            x.stride(1) if is_3d else 0,  # 头步幅
            x.stride(-1),  # 维度步幅
            freqs_real.stride(0),  # 频率位置步幅
            freqs_real.stride(1),  # 频率维度步幅
            USE_POS=True,  # 使用显式位置
            IS_INVERSE=inverse,  # 是否逆旋转
            IS_3D=is_3d,  # 是否 3D
            BLOCK_SIZE=BLOCK_SIZE,  # 块大小
        )
    else:  # 未提供显式位置
        assert (
            freqs_real.shape[0] == batch_size
        ), f"freqs_cis batch size {freqs_real.shape[0]} != x batch size {batch_size}"  # 断言频率批次大小匹配

        apply_rotary_emb_triton_kernel[grid](  # 启动内核，使用隐式位置
            x,
            freqs_real,
            None,  # 无位置张量
            rope_dim,
            x.stride(0),  # 批次步幅
            x.stride(1) if is_3d else 0,  # 头步幅
            x.stride(-1),  # 维度步幅
            freqs_real.stride(0),  # 频率位置步幅
            freqs_real.stride(1),  # 频率维度步幅
            USE_POS=False,  # 不使用显式位置
            IS_INVERSE=inverse,  # 是否逆旋转
            IS_3D=is_3d,  # 是否 3D
            BLOCK_SIZE=BLOCK_SIZE,  # 块大小
        )

    return x  # 返回旋转编码后的张量


@triton.jit  # Triton JIT 编译的融合 RMSNorm + RoPE 内核
def _fused_norm_rope_kernel(  # 融合 RMSNorm 和 RoPE 的内核（原地操作）
    x_ptr,  # 输入/输出张量指针
    weight_ptr,  # RMSNorm 权重指针
    freqs_real_ptr,  # 频率实部指针
    positions_ptr,  # 位置索引指针
    eps,  # RMSNorm epsilon
    stride_x_row,  # 输入行步幅
    stride_freq_row,  # 频率行步幅
    HEAD_DIM: tl.constexpr,  # 头维度
    ROPE_DIM: tl.constexpr,  # RoPE 维度
    HEAD_BLOCK: tl.constexpr,  # 头维度块大小
    ROPE_PAIR_BLOCK: tl.constexpr,  # RoPE 对块大小
    HAS_WEIGHT: tl.constexpr,  # 是否有权重
    USE_POS: tl.constexpr,  # 是否使用显式位置
):
    # NOTE: avoids store-then-reload on the same kernel: rope-segment values
    # are loaded a 2nd time as (real, imag) pairs straight from the input,
    # rms_inv/weight applied in register, and all stores happen at the end.
    # 注意：避免同一内核中的先存后读：RoPE 段的值作为 (实部, 虚部) 对直接从输入第二次加载，rms_inv/weight 在寄存器中应用，所有存储在最后发生。
    pid = tl.program_id(0)  # 获取程序 ID
    base = pid.to(tl.int64) * stride_x_row  # 计算行基础偏移

    offs = tl.arange(0, HEAD_BLOCK)  # 生成头维度偏移
    mask = offs < HEAD_DIM  # 生成掩码
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)  # 加载输入数据

    sum_sq = tl.sum(x * x, axis=0)  # 计算平方和
    rms_inv = tl.rsqrt(sum_sq / HEAD_DIM + eps)  # 计算 RMS 逆

    if HAS_WEIGHT:  # 如果有权重
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)  # 加载权重
        x_normed = x * rms_inv * w  # 应用 RMSNorm 并乘以权重
    else:
        x_normed = x * rms_inv  # 仅应用 RMSNorm

    rope_start = HEAD_DIM - ROPE_DIM  # 计算 RoPE 起始位置

    pair_offs = tl.arange(0, ROPE_PAIR_BLOCK)  # 生成 RoPE 对偏移
    pair_mask = pair_offs < (ROPE_DIM // 2)  # 生成 RoPE 对掩码

    x_real = tl.load(  # 第二次加载：RoPE 段实部
        x_ptr + base + rope_start + 2 * pair_offs,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    x_imag = tl.load(  # 第二次加载：RoPE 段虚部
        x_ptr + base + rope_start + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    if HAS_WEIGHT:  # 如果有权重，对 RoPE 段也应用权重
        w_real = tl.load(  # 加载 RoPE 段实部权重
            weight_ptr + rope_start + 2 * pair_offs,
            mask=pair_mask,
            other=1.0,
        ).to(tl.float32)
        w_imag = tl.load(  # 加载 RoPE 段虚部权重
            weight_ptr + rope_start + 2 * pair_offs + 1,
            mask=pair_mask,
            other=1.0,
        ).to(tl.float32)
        x_real = x_real * rms_inv * w_real  # 归一化并加权实部
        x_imag = x_imag * rms_inv * w_imag  # 归一化并加权虚部
    else:
        x_real = x_real * rms_inv  # 仅归一化实部
        x_imag = x_imag * rms_inv  # 仅归一化虚部

    if USE_POS:  # 如果使用显式位置
        position = tl.load(positions_ptr + pid).to(tl.int64)  # 加载位置
    else:
        position = pid.to(tl.int64)  # 使用程序 ID 作为位置

    freq_base = position * stride_freq_row  # 计算频率基础偏移
    f_real = tl.load(  # 加载频率实部（cos）
        freqs_real_ptr + freq_base + 2 * pair_offs,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    f_imag = tl.load(  # 加载频率虚部（sin）
        freqs_real_ptr + freq_base + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    out_real = x_real * f_real - x_imag * f_imag  # 旋转后的实部
    out_imag = x_real * f_imag + x_imag * f_real  # 旋转后的虚部

    is_non_rope = offs < rope_start  # 判断是否为非 RoPE 区域
    tl.store(  # 存储非 RoPE 区域的归一化结果
        x_ptr + base + offs,
        x_normed.to(x_ptr.dtype.element_ty),
        mask=mask & is_non_rope,
    )
    tl.store(  # 存储 RoPE 区域旋转后的实部
        x_ptr + base + rope_start + 2 * pair_offs,
        out_real.to(x_ptr.dtype.element_ty),
        mask=pair_mask,
    )
    tl.store(  # 存储 RoPE 区域旋转后的虚部
        x_ptr + base + rope_start + 2 * pair_offs + 1,
        out_imag.to(x_ptr.dtype.element_ty),
        mask=pair_mask,
    )


@triton.jit  # Triton JIT 编译的融合 softmax 加权池化内核
def _fused_softmax_pool_kernel(  # 融合 softmax 加权求和内核
    kv_score_ptr,  # KV 和分数张量指针
    out_ptr,  # 输出张量指针
    stride_bs: tl.constexpr,  # 批次步幅
    stride_k: tl.constexpr,  # K 维度步幅
    K: tl.constexpr,  # K 的大小
    HEAD_DIM: tl.constexpr,  # 头维度
    HEAD_BLOCK: tl.constexpr,  # 头维度块大小
):
    pid = tl.program_id(0)  # 获取程序 ID
    base = pid * stride_bs  # 计算基础偏移

    offs = tl.arange(0, HEAD_BLOCK)  # 生成偏移
    mask = offs < HEAD_DIM  # 生成掩码

    max_val = tl.full([HEAD_BLOCK], float("-inf"), dtype=tl.float32)  # 初始化最大值为负无穷
    for k in range(K):  # 遍历 K 维度查找最大值
        s = tl.load(  # 加载分数
            kv_score_ptr + base + k * stride_k + HEAD_DIM + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        max_val = tl.maximum(max_val, s)  # 更新最大值

    sum_exp = tl.zeros([HEAD_BLOCK], dtype=tl.float32)  # 初始化指数和
    weighted = tl.zeros([HEAD_BLOCK], dtype=tl.float32)  # 初始化加权和
    for k in range(K):  # 遍历 K 维度计算加权求和
        s = tl.load(  # 加载分数
            kv_score_ptr + base + k * stride_k + HEAD_DIM + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        v = tl.load(  # 加载 KV 值
            kv_score_ptr + base + k * stride_k + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.exp(s - max_val)  # 计算指数权重（减去最大值防止溢出）
        sum_exp += w  # 累加指数和
        weighted += v * w  # 累加加权和

    result = weighted / sum_exp  # 计算 softmax 加权平均
    tl.store(  # 存储结果
        out_ptr + pid * HEAD_DIM + offs, result.to(out_ptr.dtype.element_ty), mask=mask
    )


def fused_softmax_pool_triton(  # 融合 softmax 加权求和的 Triton 实现
    kv_score: torch.Tensor,  # KV 和分数张量 [bs, K, 2*head_dim]
    head_dim: int,  # 头维度
) -> torch.Tensor:  # 返回 [bs, head_dim] 的输出
    """Fused softmax-weighted-sum: out = (kv * softmax(score, dim=1)).sum(dim=1).

    Replaces the generic cunn_SpatialSoftMaxForward + elementwise multiply + sum
    with a single Triton kernel.

    Args:
        kv_score: [bs, K, 2 * head_dim] where first head_dim is kv, second is score.
        head_dim: dimension of each of kv and score.
    Returns:
        output: [bs, head_dim]
    """  # 融合 softmax 加权求和：out = (kv * softmax(score, dim=1)).sum(dim=1)。用单个 Triton 内核替代通用的 cunn_SpatialSoftMaxForward + 逐元素乘法 + 求和。参数：kv_score 为 [bs, K, 2*head_dim]，前 head_dim 是 kv，后 head_dim 是 score；head_dim 是 kv 和 score 各自的维度。返回 [bs, head_dim] 的输出。
    assert kv_score.dim() == 3  # 断言输入为 3D
    bs, K, last = kv_score.shape  # 解包形状
    assert last == 2 * head_dim  # 断言最后一维大小为 2*head_dim
    assert kv_score.is_contiguous()  # 断言内存连续

    out = torch.empty(bs, head_dim, dtype=kv_score.dtype, device=kv_score.device)  # 分配输出张量
    if bs == 0:  # 如果批次为 0
        return out  # 直接返回空张量

    HEAD_BLOCK = triton.next_power_of_2(head_dim)  # 计算头维度的 2 的幂块大小
    grid = (bs,)  # 设置网格大小
    _fused_softmax_pool_kernel[grid](  # 启动融合内核
        kv_score,
        out,
        stride_bs=kv_score.stride(0),  # 批次步幅
        stride_k=kv_score.stride(1),  # K 维度步幅
        K=K,  # K 的大小
        HEAD_DIM=head_dim,  # 头维度
        HEAD_BLOCK=HEAD_BLOCK,  # 头维度块大小
    )
    return out  # 返回输出张量


def fused_norm_rope_inplace_triton(  # 融合 RMSNorm + RoPE 的原地 Triton 实现
    kv: torch.Tensor,  # 输入/输出张量 [M, head_dim]，原地修改
    weight: Optional[torch.Tensor],  # RMSNorm 权重 [head_dim] 或 None
    eps: float,  # RMSNorm epsilon
    freqs_cis: torch.Tensor,  # 复数频率张量
    positions: Optional[torch.Tensor] = None,  # 可选的位置索引 [M]
) -> None:  # 无返回值（原地操作）
    """Fused RMSNorm (over head_dim) + RoPE (on last rope_dim of head_dim), in-place.

    Equivalent to::

        kv = rms_normalize(kv, eps, weight)
        apply_rotary_emb_triton(kv[..., -rope_dim:], freqs_cis, positions=positions)

    Args:
        kv: [M, head_dim], any float dtype, contiguous along last dim. Modified in-place.
        weight: [head_dim] or None.
        eps: RMSNorm epsilon.
        freqs_cis: complex tensor.
            - If ``positions`` is None: shape [M, rope_dim // 2], one freq per token.
            - Else: shape [max_seq, rope_dim // 2], full table; indexed by ``positions``.
        positions: optional [M] int tensor, absolute positions to index into ``freqs_cis``.
    """  # 融合 RMSNorm（对 head_dim）+ RoPE（对 head_dim 的最后 rope_dim），原地操作。等价于：先 rms_normalize(kv, eps, weight)，再 apply_rotary_emb_triton(kv[..., -rope_dim:], freqs_cis, positions=positions)。参数：kv 为 [M, head_dim]，任意浮点类型，最后一维连续，原地修改；weight 为 [head_dim] 或 None；eps 为 RMSNorm epsilon；freqs_cis 为复数张量，若 positions 为 None 则形状为 [M, rope_dim//2]，否则为 [max_seq, rope_dim//2]；positions 为可选的 [M] 整数张量。
    assert kv.dim() == 2 and kv.stride(-1) == 1  # 断言输入为 2D 且最后一维连续
    M, head_dim = kv.shape  # 解包形状

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)  # 将复数频率转为实数表示
    rope_dim = freqs_real.shape[-1]  # 获取 RoPE 维度
    assert head_dim >= rope_dim and rope_dim % 2 == 0  # 断言 head_dim >= rope_dim 且 rope_dim 为偶数
    if weight is not None:  # 如果有权重
        assert weight.shape == (head_dim,)  # 断言权重形状正确
    if positions is None:  # 如果没有显式位置
        assert (
            freqs_real.shape[0] == M
        ), f"freqs_cis row count {freqs_real.shape[0]} != M={M}"  # 断言频率行数等于 M
    else:
        assert positions.shape == (M,) and positions.dim() == 1  # 断言位置形状正确

    if M == 0:  # 如果没有数据
        return  # 直接返回

    HEAD_BLOCK = triton.next_power_of_2(head_dim)  # 计算头维度的 2 的幂块大小
    ROPE_PAIR_BLOCK = max(triton.next_power_of_2(rope_dim // 2), 1)  # 计算 RoPE 对的 2 的幂块大小

    grid = (M,)  # 设置网格大小
    _fused_norm_rope_kernel[grid](  # 启动融合内核
        kv,
        weight,
        freqs_real,
        positions,
        eps,
        kv.stride(0),  # 行步幅
        freqs_real.stride(0),  # 频率行步幅
        HEAD_DIM=head_dim,  # 头维度
        ROPE_DIM=rope_dim,  # RoPE 维度
        HEAD_BLOCK=HEAD_BLOCK,  # 头维度块大小
        ROPE_PAIR_BLOCK=ROPE_PAIR_BLOCK,  # RoPE 对块大小
        HAS_WEIGHT=(weight is not None),  # 是否有权重
        USE_POS=(positions is not None),  # 是否使用显式位置
    )
