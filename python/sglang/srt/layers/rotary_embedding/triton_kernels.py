# Triton JIT内核模块
# 为多模态旋转位置编码(MRoPE)提供Triton JIT加速的融合内核实现
"""Triton JIT kernels for multimodal rotary positional embeddings."""  # 多模态旋转位置编码的Triton JIT内核

from __future__ import annotations  # 启用延迟注解评估

from typing import List  # 类型提示：列表类型

import torch  # PyTorch深度学习框架
import triton  # Triton编译器框架
import triton.language as tl  # Triton语言接口


@triton.jit  # Triton JIT装饰器，将函数编译为GPU内核
def _triton_mrope_forward_fused(  # MRoPE融合前向内核（Triton JIT实现）
    q_ptr,  # 查询张量指针
    k_ptr,  # 键张量指针
    cos_sin_cache_ptr,  # 余弦正弦缓存指针
    positions_ptr,  # 位置张量指针
    q_stride,  # 查询的步幅
    k_stride,  # 键的步幅
    positions_stride,  # 位置的步幅
    n_qh: tl.constexpr,  # 查询头数（编译时常量）
    n_kh: tl.constexpr,  # 键头数（编译时常量）
    hd: tl.constexpr,  # 头维度（编译时常量）
    rd: tl.constexpr,  # 旋转维度（编译时常量）
    pad_n_qh: tl.constexpr,  # 填充后的查询头数（编译时常量）
    pad_n_kh: tl.constexpr,  # 填充后的键头数（编译时常量）
    pad_hd: tl.constexpr,  # 填充后的头维度（编译时常量）
    mrope_section_t: tl.constexpr,  # MRoPE时间维度段大小（编译时常量）
    mrope_section_h: tl.constexpr,  # MRoPE高度维度段大小（编译时常量）
    mrope_section_w: tl.constexpr,  # MRoPE宽度维度段大小（编译时常量）
    is_interleaved: tl.constexpr,  # 是否交错布局（编译时常量）
    is_interleaved_glm: tl.constexpr,  # 是否GLM交错布局（编译时常量）
    is_neox_style: tl.constexpr,  # 是否NeoX风格（编译时常量）
    axis_map_ptr,  # 轴映射指针（用于GLM交错布局）
):
    pid = tl.program_id(0)  # 获取当前程序的ID（token索引）
    q_ptr = q_ptr + pid * q_stride  # 计算当前token的查询指针偏移
    k_ptr = k_ptr + pid * k_stride  # 计算当前token的键指针偏移
    half_rd = rd // 2  # 旋转维度的一半
    t = tl.load(positions_ptr + 0 * positions_stride + pid)  # 加载时间维度的位置索引
    h = tl.load(positions_ptr + 1 * positions_stride + pid)  # 加载高度维度的位置索引
    w = tl.load(positions_ptr + 2 * positions_stride + pid)  # 加载宽度维度的位置索引
    t_cos = cos_sin_cache_ptr + t * rd  # 计算时间维度的余弦缓存偏移
    h_cos = cos_sin_cache_ptr + h * rd  # 计算高度维度的余弦缓存偏移
    w_cos = cos_sin_cache_ptr + w * rd  # 计算宽度维度的余弦缓存偏移
    t_sin = t_cos + half_rd  # 计算时间维度的正弦缓存偏移（余弦之后）
    h_sin = h_cos + half_rd  # 计算高度维度的正弦缓存偏移（余弦之后）
    w_sin = w_cos + half_rd  # 计算宽度维度的正弦缓存偏移（余弦之后）
    cos_offsets = tl.arange(0, pad_hd // 2)  # 生成余弦/正弦值的偏移量序列
    if is_interleaved:  # 如果使用交错布局
        if is_interleaved_glm:  # 如果是GLM交错布局
            axes = tl.load(axis_map_ptr + cos_offsets, mask=cos_offsets < (pad_hd // 2))  # 加载轴映射
            t_mask = axes == 0  # 时间维度掩码
            h_mask = axes == 1  # 高度维度掩码
            w_mask = axes == 2  # 宽度维度掩码
        else:  # 非GLM交错布局
            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)  # 高度维度掩码（模3余1且在范围内）
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)  # 宽度维度掩码（模3余2且在范围内）
            t_mask = ~(h_mask | w_mask)  # 时间维度掩码（非高度且非宽度）
    else:  # 非交错布局（分区布局）
        t_end = mrope_section_t  # 时间段结束位置
        h_end = t_end + mrope_section_h  # 高度段结束位置
        t_mask = cos_offsets < mrope_section_t  # 时间维度掩码
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)  # 高度维度掩码
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)  # 宽度维度掩码
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)  # 加载时间维度的余弦值
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)  # 加载时间维度的正弦值
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)  # 加载高度维度的余弦值
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)  # 加载高度维度的正弦值
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)  # 加载宽度维度的余弦值
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)  # 加载宽度维度的正弦值
    cos_row = t_cos_row + h_cos_row + w_cos_row  # 合并三个维度的余弦值
    sin_row = t_sin_row + h_sin_row + w_sin_row  # 合并三个维度的正弦值
    if is_neox_style:  # 如果是NeoX风格
        fhq = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]  # 查询的前半部分偏移
        fhk = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]  # 键的前半部分偏移
        fqm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (  # 查询前半部分的有效掩码
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        fkm = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (  # 键前半部分的有效掩码
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        q1 = tl.load(q_ptr + fhq, mask=fqm, other=0).to(sin_row.dtype)  # 加载查询的前半部分
        k1 = tl.load(k_ptr + fhk, mask=fkm, other=0).to(sin_row.dtype)  # 加载键的前半部分
        shq = fhq + (rd // 2)  # 查询后半部分的偏移
        shk = fhk + (rd // 2)  # 键后半部分的偏移
        q2 = tl.load(q_ptr + shq, mask=fqm, other=0).to(sin_row.dtype)  # 加载查询的后半部分
        k2 = tl.load(k_ptr + shk, mask=fkm, other=0).to(sin_row.dtype)  # 加载键的后半部分
        tl.store(q_ptr + fhq, q1 * cos_row - q2 * sin_row, mask=fqm)  # 存储旋转后的查询前半部分
        tl.store(q_ptr + shq, q2 * cos_row + q1 * sin_row, mask=fqm)  # 存储旋转后的查询后半部分
        tl.store(k_ptr + fhk, k1 * cos_row - k2 * sin_row, mask=fkm)  # 存储旋转后的键前半部分
        tl.store(k_ptr + shk, k2 * cos_row + k1 * sin_row, mask=fkm)  # 存储旋转后的键后半部分
    else:  # GPT-J风格
        bq = tl.arange(0, pad_n_qh)[:, None] * hd  # 查询的块偏移
        bk = tl.arange(0, pad_n_kh)[:, None] * hd  # 键的块偏移
        ei = 2 * tl.arange(0, pad_hd // 2)[None, :]  # 偶数索引
        oi = ei + 1  # 奇数索引
        im = tl.arange(0, pad_hd // 2)[None, :] < (rd // 2)  # 旋转维度的有效掩码
        qm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & im  # 查询的有效掩码
        km = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & im  # 键的有效掩码
        qe = tl.load(q_ptr + bq + ei, mask=qm, other=0).to(sin_row.dtype)  # 加载查询的偶数位置
        qo = tl.load(q_ptr + bq + oi, mask=qm, other=0).to(sin_row.dtype)  # 加载查询的奇数位置
        ke = tl.load(k_ptr + bk + ei, mask=km, other=0).to(sin_row.dtype)  # 加载键的偶数位置
        ko = tl.load(k_ptr + bk + oi, mask=km, other=0).to(sin_row.dtype)  # 加载键的奇数位置
        tl.store(q_ptr + bq + ei, qe * cos_row - qo * sin_row, mask=qm)  # 存储旋转后的查询偶数位置
        tl.store(q_ptr + bq + oi, qo * cos_row + qe * sin_row, mask=qm)  # 存储旋转后的查询奇数位置
        tl.store(k_ptr + bk + ei, ke * cos_row - ko * sin_row, mask=km)  # 存储旋转后的键偶数位置
        tl.store(k_ptr + bk + oi, ko * cos_row + ke * sin_row, mask=km)  # 存储旋转后的键奇数位置


def triton_mrope_fused(  # MRoPE融合操作的入口函数
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    cos_sin_cache: torch.Tensor,  # 余弦正弦缓存
    positions: torch.Tensor,  # 位置张量
    mrope_section: List[int],  # MRoPE各维度段大小[t, h, w]
    head_size: int,  # 头维度大小
    rotary_dim: int,  # 旋转维度大小
    mrope_interleaved: bool,  # 是否使用交错布局
    mrope_interleaved_glm: bool,  # 是否使用GLM交错布局
    is_neox_style: bool,  # 是否使用NeoX风格
    axis_map: torch.Tensor,  # 轴映射张量
) -> None:  # 原地操作，无返回值
    num_tokens, n_q_dim = q.shape  # 获取token数和查询维度
    n_k_dim = k.shape[1]  # 获取键维度
    n_qh = n_q_dim // head_size  # 计算查询头数
    n_kh = n_k_dim // head_size  # 计算键头数
    pad_n_qh = triton.next_power_of_2(n_qh)  # 将查询头数填充到2的幂次
    pad_n_kh = triton.next_power_of_2(n_kh)  # 将键头数填充到2的幂次
    pad_hd = triton.next_power_of_2(head_size)  # 将头维度填充到2的幂次
    _triton_mrope_forward_fused[(num_tokens,)](  # 启动Triton内核，每个token一个程序
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),  # 查询的步幅
        k.stride(0),  # 键的步幅
        positions.stride(0),  # 位置的步幅
        n_qh,  # 查询头数
        n_kh,  # 键头数
        head_size,  # 头维度
        rotary_dim,  # 旋转维度
        pad_n_qh,  # 填充后的查询头数
        pad_n_kh,  # 填充后的键头数
        pad_hd,  # 填充后的头维度
        mrope_section[0],  # 时间维度段大小
        mrope_section[1],  # 高度维度段大小
        mrope_section[2],  # 宽度维度段大小
        mrope_interleaved,  # 交错布局标志
        mrope_interleaved_glm,  # GLM交错布局标志
        is_neox_style,  # NeoX风格标志
        axis_map,  # 轴映射
    )


@triton.jit  # Triton JIT装饰器，将函数编译为GPU内核
def _triton_ernie45_rope_qk_fused(  # Ernie4.5 RoPE融合内核（Triton JIT实现）
    q_ptr,  # 查询张量指针
    k_ptr,  # 键张量指针
    cos_sin_cache_ptr,  # 余弦正弦缓存指针
    positions_ptr,  # 位置张量指针
    q_stride0: tl.constexpr,  # 查询的步幅（编译时常量）
    k_stride0: tl.constexpr,  # 键的步幅（编译时常量）
    pos_stride0: tl.constexpr,  # 位置的步幅（编译时常量）
    n_qh: tl.constexpr,  # 查询头数（编译时常量）
    n_kh: tl.constexpr,  # 键头数（编译时常量）
    hd: tl.constexpr,  # 头维度（编译时常量）
    rd: tl.constexpr,  # 旋转维度（编译时常量）
    pad_n_qh: tl.constexpr,  # 填充后的查询头数（编译时常量）
    pad_n_kh: tl.constexpr,  # 填充后的键头数（编译时常量）
    pad_hd: tl.constexpr,  # 填充后的头维度（编译时常量）
    section_hw: tl.constexpr,  # 高度+宽度段大小（编译时常量）
    is_neox_style: tl.constexpr,  # 是否NeoX风格（编译时常量）
):
    pid = tl.program_id(0)  # 获取当前程序的ID（token索引）
    q_ptr = q_ptr + pid * q_stride0  # 计算当前token的查询指针偏移
    k_ptr = k_ptr + pid * k_stride0  # 计算当前token的键指针偏移
    half_rd = rd // 2  # 旋转维度的一半
    tpos = tl.load(positions_ptr + 0 * pos_stride0 + pid).to(tl.int32)  # 加载时间维度位置索引
    hpos = tl.load(positions_ptr + 1 * pos_stride0 + pid).to(tl.int32)  # 加载高度维度位置索引
    wpos = tl.load(positions_ptr + 2 * pos_stride0 + pid).to(tl.int32)  # 加载宽度维度位置索引
    ridx = tl.arange(0, pad_hd // 2)  # 生成旋转维度的索引序列
    rmask = ridx < half_rd  # 旋转维度的有效掩码
    use_hw = ridx < section_hw  # 判断是否使用高度+宽度维度
    use_h = (ridx & 1) == 0  # 判断是否使用高度维度（偶数索引为高度）
    pos = tl.where(use_hw, tl.where(use_h, hpos, wpos), tpos)  # 根据维度选择对应的位置索引
    cos = tl.load(cos_sin_cache_ptr + pos * rd + ridx, mask=rmask, other=0.0)  # 加载余弦值
    sin = tl.load(  # 加载正弦值
        cos_sin_cache_ptr + pos * rd + (ridx + half_rd), mask=rmask, other=0.0
    )
    if is_neox_style:  # 如果是NeoX风格
        qh = tl.arange(0, pad_n_qh)[:, None]  # 查询头索引
        kh = tl.arange(0, pad_n_kh)[:, None]  # 键头索引
        d = tl.arange(0, pad_hd // 2)[None, :]  # 维度索引
        qm = (qh < n_qh) & (d < half_rd)  # 查询的有效掩码
        km = (kh < n_kh) & (d < half_rd)  # 键的有效掩码
        qo0 = qh * hd + d  # 查询前半部分的偏移
        ko0 = kh * hd + d  # 键前半部分的偏移
        qo1 = qo0 + half_rd  # 查询后半部分的偏移
        ko1 = ko0 + half_rd  # 键后半部分的偏移
        q0 = tl.load(q_ptr + qo0, mask=qm, other=0.0).to(cos.dtype)  # 加载查询前半部分
        q1 = tl.load(q_ptr + qo1, mask=qm, other=0.0).to(cos.dtype)  # 加载查询后半部分
        k0 = tl.load(k_ptr + ko0, mask=km, other=0.0).to(cos.dtype)  # 加载键前半部分
        k1 = tl.load(k_ptr + ko1, mask=km, other=0.0).to(cos.dtype)  # 加载键后半部分
        cb = cos[None, :]  # 广播余弦值
        sb = sin[None, :]  # 广播正弦值
        tl.store(q_ptr + qo0, q0 * cb - q1 * sb, mask=qm)  # 存储旋转后的查询前半部分
        tl.store(q_ptr + qo1, q1 * cb + q0 * sb, mask=qm)  # 存储旋转后的查询后半部分
        tl.store(k_ptr + ko0, k0 * cb - k1 * sb, mask=km)  # 存储旋转后的键前半部分
        tl.store(k_ptr + ko1, k1 * cb + k0 * sb, mask=km)  # 存储旋转后的键后半部分
    else:  # GPT-J风格
        qh = tl.arange(0, pad_n_qh)[:, None]  # 查询头索引
        kh = tl.arange(0, pad_n_kh)[:, None]  # 键头索引
        p = tl.arange(0, pad_hd // 2)[None, :]  # 维度索引
        qm = (qh < n_qh) & (p < half_rd)  # 查询的有效掩码
        km = (kh < n_kh) & (p < half_rd)  # 键的有效掩码
        even = 2 * p  # 偶数索引
        odd = even + 1  # 奇数索引
        qe = tl.load(q_ptr + qh * hd + even, mask=qm, other=0.0).to(cos.dtype)  # 加载查询偶数位置
        qo = tl.load(q_ptr + qh * hd + odd, mask=qm, other=0.0).to(cos.dtype)  # 加载查询奇数位置
        ke = tl.load(k_ptr + kh * hd + even, mask=km, other=0.0).to(cos.dtype)  # 加载键偶数位置
        ko = tl.load(k_ptr + kh * hd + odd, mask=km, other=0.0).to(cos.dtype)  # 加载键奇数位置
        cb = cos[None, :]  # 广播余弦值
        sb = sin[None, :]  # 广播正弦值
        tl.store(q_ptr + qh * hd + even, qe * cb - qo * sb, mask=qm)  # 存储旋转后的查询偶数位置
        tl.store(q_ptr + qh * hd + odd, qo * cb + qe * sb, mask=qm)  # 存储旋转后的查询奇数位置
        tl.store(k_ptr + kh * hd + even, ke * cb - ko * sb, mask=km)  # 存储旋转后的键偶数位置
        tl.store(k_ptr + kh * hd + odd, ko * cb + ke * sb, mask=km)  # 存储旋转后的键奇数位置


def triton_ernie45_rope_fused_inplace(  # Ernie4.5 RoPE原地融合操作入口函数
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    cos_sin_cache: torch.Tensor,  # 余弦正弦缓存
    positions: torch.Tensor,  # 位置张量
    mrope_section: list,  # MRoPE各维度段大小[h, w, t]
    head_size: int,  # 头维度大小
    rotary_dim: int,  # 旋转维度大小
    is_neox_style: bool,  # 是否使用NeoX风格
) -> None:  # 原地操作，无返回值
    num_tokens = q.shape[0]  # 获取token数
    n_qh = q.shape[1] // head_size  # 计算查询头数
    n_kh = k.shape[1] // head_size  # 计算键头数
    rd = rotary_dim  # 旋转维度
    section_h, section_w, section_t = mrope_section  # 解包MRoPE段大小
    assert section_h == section_w, "Ernie4.5 layout assumes section_h == section_w"  # 断言高度段等于宽度段
    assert section_h + section_w + section_t == rd // 2  # 断言各段之和等于旋转维度的一半
    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.device != q.device:  # 如果缓存与查询的数据类型或设备不匹配
        cos_sin_cache = cos_sin_cache.to(device=q.device, dtype=q.dtype)  # 转换缓存的数据类型和设备
    pad_n_qh = triton.next_power_of_2(n_qh)  # 将查询头数填充到2的幂次
    pad_n_kh = triton.next_power_of_2(n_kh)  # 将键头数填充到2的幂次
    pad_hd = triton.next_power_of_2(head_size)  # 将头维度填充到2的幂次
    num_warps = 4 if (pad_n_qh * pad_hd) <= 8192 else 8  # 根据数据量选择线程束数量
    _triton_ernie45_rope_qk_fused[(num_tokens,)](  # 启动Triton内核，每个token一个程序
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),  # 查询的步幅
        k.stride(0),  # 键的步幅
        positions.stride(0),  # 位置的步幅
        n_qh=n_qh,  # 查询头数
        n_kh=n_kh,  # 键头数
        hd=head_size,  # 头维度
        rd=rd,  # 旋转维度
        pad_n_qh=pad_n_qh,  # 填充后的查询头数
        pad_n_kh=pad_n_kh,  # 填充后的键头数
        pad_hd=pad_hd,  # 填充后的头维度
        section_hw=section_h + section_w,  # 高度+宽度段大小
        is_neox_style=is_neox_style,  # NeoX风格标志
        num_warps=num_warps,  # 线程束数量
    )
