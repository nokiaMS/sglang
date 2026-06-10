# 融合Q RMSNorm + KV RMSNorm + RoPE + FP8量化 + 分页SWA存储的Triton内核
# 本文件实现了一个单一的Triton内核，替换了原来的两内核路径：
#   1. fused_reduce_qk_norm_rope_swa_write (归一化 + RoPE)
#   2. store_cache -> fused_store_cache (FP8量化 + 分页散射)
# 网格布局: (cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)
#   pid_h < num_local_heads: Q头程序（split-K归约 + 归一化 + RoPE）
#   pid_h == num_local_heads: KV程序（归一化 + RoPE + FP8量化nope + 分页散射）

"""Fused Q per-head RMSNorm + KV RMSNorm + RoPE + FP8 nope quant + paged SWA store.
融合Q逐头RMSNorm + KV RMSNorm + RoPE + FP8 nope量化 + 分页SWA存储。

Single Triton kernel replacing the 2-kernel path:
单一Triton内核替换两内核路径：
  1. fused_reduce_qk_norm_rope_swa_write (norm + RoPE)
  1. fused_reduce_qk_norm_rope_swa_write (归一化 + RoPE)
  2. store_cache -> fused_store_cache (FP8 quant + paged scatter)
  2. store_cache -> fused_store_cache (FP8量化 + 分页散射)

Grid: (cdiv(M, BLOCK_SIZE_M), num_local_heads + 1).
网格: (cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)。
  pid_h < num_local_heads: Q head programs (split-K reduce + norm + RoPE)
  pid_h < num_local_heads: Q头程序（split-K归约 + 归一化 + RoPE）
  pid_h == num_local_heads: KV program (norm + RoPE + FP8 quant nope + paged scatter)
  pid_h == num_local_heads: KV程序（归一化 + RoPE + FP8量化nope + 分页散射）
"""

from typing import Optional  # 导入可选类型 # 导入可选类型

import torch  # 导入PyTorch库 # 导入PyTorch库
import triton  # 导入Triton库 # 导入Triton库
import triton.language as tl  # 导入Triton语言模块并简写为tl # 导入Triton语言模块并简写为tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz  # 导入FP8 FNUZ格式检测函数 # 导入FP8 FNUZ格式检测函数

_fp8_fnuz = is_fp8_fnuz()  # 检测当前是否使用FP8 FNUZ格式 # 检测当前是否使用FP8 FNUZ格式


# ---------------------------------------------------------------------------
# Triton JIT helpers
# Triton JIT辅助函数
# ---------------------------------------------------------------------------


# 批量RMSNorm辅助函数
@triton.jit
def _batched_rmsnorm(row, weight, n_cols, epsilon):  # 批量RMSNorm：对行进行归一化
    row_norm = tl.sum(row * row, axis=-1)  # 计算行平方和 # 计算行平方和
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)  # 计算RMSNorm的归一化因子 # 计算RMSNorm的归一化因子
    if weight is not None:  # 如果有权重 # 如果有权重
        return row * norm_factor[:, None] * weight[None, :]  # 返回加权归一化结果 # 返回加权归一化结果
    return row * norm_factor[:, None]  # 返回无权重归一化结果 # 返回无权重归一化结果


# GPT-J风格旋转辅助函数
@triton.jit
def _gptj_rotate(x, mask, BM: tl.constexpr, BD: tl.constexpr, BDH: tl.constexpr):  # GPT-J风格旋转：对偶数/奇数位置取反并翻转
    x_rot = tl.where(mask, x, -x)  # 偶数位置保持，奇数位置取反 # 偶数位置保持，奇数位置取反
    x_rot = tl.reshape(x_rot, (BM, BDH, 2))  # 重塑为(BM, BDH, 2)以便翻转 # 重塑为(BM, BDH, 2)以便翻转
    x_rot = tl.flip(x_rot, 2)  # 在最后一维翻转，交换奇偶位置 # 在最后一维翻转，交换奇偶位置
    return tl.reshape(x_rot, (BM, BD))  # 重塑回(BM, BD)形状 # 重塑回(BM, BD)形状


# 批量RoPE辅助函数
@triton.jit
def _batched_rope(  # 批量旋转位置编码：应用cos/sin旋转
    x_pe, cos, sin, d_pe_offs, BM: tl.constexpr, BD: tl.constexpr, BDH: tl.constexpr  # 输入PE、cos、sin、偏移量及维度常量
):
    mask = (d_pe_offs % 2 == 0)[None, :]  # 生成偶数位置掩码 # 生成偶数位置掩码
    x_rot = _gptj_rotate(x_pe, mask, BM, BD, BDH)  # 对PE应用GPT-J旋转 # 对PE应用GPT-J旋转
    return x_pe * cos + x_rot * sin  # 应用旋转位置编码公式 # 应用旋转位置编码公式


# ---------------------------------------------------------------------------
# Main kernel
# 主内核
# ---------------------------------------------------------------------------


# 融合Q/K归一化 + RoPE + SWA存储的主内核
@triton.jit
def _fused_qk_norm_rope_store_kernel(  # 融合Q/K归一化+RoPE+存储内核
    q_in_ptr,  # Q输入指针 # Q输入指针
    q_out_ptr,  # Q输出指针 # Q输出指针
    kv_ptr,  # KV输入/输出指针（原地修改） # KV输入/输出指针
    q_norm_weight_ptr,  # Q归一化权重指针 # Q归一化权重指针
    kv_norm_weight_ptr,  # KV归一化权重指针 # KV归一化权重指针
    positions_ptr,  # 位置索引指针 # 位置索引指针
    cos_ptr,  # RoPE余弦缓存指针 # RoPE余弦缓存指针
    sin_ptr,  # RoPE正弦缓存指针 # RoPE正弦缓存指针
    swa_cache_ptr,  # SWA分页缓存指针 # SWA分页缓存指针
    swa_loc_ptr,  # SWA位置索引指针 # SWA位置索引指针
    M,  # token数量 # token数量
    q_in_splitk_stride,  # Q输入split-K步长 # Q输入split-K步长
    q_in_m_stride,  # Q输入token步长 # Q输入token步长
    q_in_d_stride,  # Q输入维度步长 # Q输入维度步长
    stride_qm,  # Q输出token步长 # Q输出token步长
    stride_qh,  # Q输出头步长 # Q输出头步长
    stride_qd,  # Q输出维度步长 # Q输出维度步长
    stride_kv_m,  # KV token步长 # KV token步长
    stride_kv_d,  # KV维度步长 # KV维度步长
    cos_stride_t,  # 余弦token步长 # 余弦token步长
    cos_stride_d,  # 余弦维度步长 # 余弦维度步长
    swa_cache_stride_page,  # SWA缓存页步长 # SWA缓存页步长
    q_eps,  # Q归一化epsilon # Q归一化epsilon
    kv_eps,  # KV归一化epsilon # KV归一化epsilon
    BLOCK_SIZE_M: tl.constexpr,  # M方向块大小 # M方向块大小
    HEAD_DIM: tl.constexpr,  # 头维度 # 头维度
    ROPE_DIM: tl.constexpr,  # RoPE维度 # RoPE维度
    NUM_LOCAL_HEADS: tl.constexpr,  # 本地Q头数量 # 本地Q头数量
    NUM_SPLITK: tl.constexpr,  # split-K数量 # split-K数量
    HAS_SWA_STORE: tl.constexpr,  # 是否有SWA存储 # 是否有SWA存储
    DIM_NOPE: tl.constexpr,  # nope维度大小 # nope维度大小
    TILE_SIZE: tl.constexpr,  # 分块大小 # 分块大小
    NUM_NOPE_TILES: tl.constexpr,  # nope分块数量 # nope分块数量
    FP8_MIN: tl.constexpr,  # FP8最小值 # FP8最小值
    FP8_MAX: tl.constexpr,  # FP8最大值 # FP8最大值
    BYTES_PER_TOKEN: tl.constexpr,  # 每token字节数 # 每token字节数
    SWA_PAGE_SIZE: tl.constexpr,  # SWA页大小 # SWA页大小
):
    pid_m = tl.program_id(0).to(tl.int64)  # 获取M方向程序ID # 获取M方向程序ID
    pid_h = tl.program_id(1).to(tl.int64)  # 获取头方向程序ID # 获取头方向程序ID
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM  # 计算nope维度 = 头维度 - RoPE维度 # 计算nope维度
    NUM_PE_CHUNKS: tl.constexpr = HEAD_DIM // ROPE_DIM  # PE分块数 = 头维度 / RoPE维度 # PE分块数

    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)  # 计算token偏移 # 计算token偏移
    m_mask = m_offs < M  # 生成token掩码 # 生成token掩码

    offs_d_full = tl.arange(0, HEAD_DIM)  # 全维度偏移 # 全维度偏移
    nope_d_mask = offs_d_full < NOPE_DIM  # nope维度掩码 # nope维度掩码

    d_pe_offs = tl.arange(0, ROPE_DIM).to(tl.int64)  # RoPE维度偏移 # RoPE维度偏移
    d_cos_offs = d_pe_offs // 2  # cos/sin维度偏移（每对共享一个cos/sin） # cos/sin维度偏移

    # ===== Q path =====
    # ===== Q路径 =====
    if pid_h < NUM_LOCAL_HEADS:  # 如果是Q头程序 # 如果是Q头程序
        head_id = pid_h.to(tl.int32)  # 获取头ID # 获取头ID
        offs_n = head_id * HEAD_DIM + offs_d_full  # 计算该头在全维度上的偏移 # 计算该头在全维度上的偏移

        splitk_offs = tl.arange(0, NUM_SPLITK).to(tl.int64)  # split-K偏移 # split-K偏移
        q_ptrs = (  # 计算Q输入指针
            q_in_ptr  # Q输入基地址 # Q输入基地址
            + splitk_offs[:, None, None] * q_in_splitk_stride  # split-K偏移 # split-K偏移
            + m_offs[None, :, None] * q_in_m_stride  # token偏移 # token偏移
            + offs_n[None, None, :] * q_in_d_stride  # 维度偏移 # 维度偏移
        )
        q_tile = tl.load(q_ptrs, mask=m_mask[None, :, None], other=0.0).to(tl.float32)  # 加载Q分块数据 # 加载Q分块数据
        q_acc = tl.sum(q_tile, axis=0)  # 在split-K维度上归约 # 在split-K维度上归约

        if q_norm_weight_ptr is not None:  # 如果Q归一化权重存在 # 如果Q归一化权重存在
            w_q = tl.load(q_norm_weight_ptr + offs_d_full).to(tl.float32)  # 加载Q权重 # 加载Q权重
        else:  # 否则 # 否则
            w_q = None  # Q权重设为None # Q权重设为None
        q_normed = _batched_rmsnorm(q_acc, w_q, HEAD_DIM, q_eps)  # 对Q进行RMSNorm # 对Q进行RMSNorm

        q_base = q_out_ptr + m_offs[:, None] * stride_qm + pid_h * stride_qh  # 计算Q输出基地址 # 计算Q输出基地址
        tl.store(  # 存储Q的nope部分（归一化后的非RoPE部分）
            q_base + offs_d_full[None, :] * stride_qd,  # Q输出地址 # Q输出地址
            q_normed.to(q_out_ptr.dtype.element_ty),  # 转换为目标数据类型 # 转换为目标数据类型
            mask=m_mask[:, None] & nope_d_mask[None, :],  # nope部分掩码 # nope部分掩码
        )

        q_pe = tl.where((offs_d_full >= NOPE_DIM)[None, :], q_normed, 0.0)  # 提取Q的PE部分 # 提取Q的PE部分
        q_pe = tl.reshape(q_pe, (BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM))  # 重塑为分块形状 # 重塑为分块形状
        q_pe = tl.sum(q_pe, axis=1)  # 对PE分块求和 # 对PE分块求和

        pos = tl.load(positions_ptr + m_offs, mask=m_mask, other=0)  # 加载位置索引 # 加载位置索引
        cos_o = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d  # 计算cos/sin偏移 # 计算cos/sin偏移
        cos = tl.load(cos_ptr + cos_o, mask=m_mask[:, None], other=0)  # 加载cos值 # 加载cos值
        sin = tl.load(sin_ptr + cos_o, mask=m_mask[:, None], other=0)  # 加载sin值 # 加载sin值

        q_pe = _batched_rope(  # 对Q的PE部分应用RoPE
            q_pe, cos, sin, d_pe_offs, BLOCK_SIZE_M, ROPE_DIM, ROPE_DIM // 2  # 传入PE数据、cos、sin及维度参数
        )
        tl.store(  # 存储Q的RoPE部分
            q_base + (NOPE_DIM + d_pe_offs[None, :]) * stride_qd,  # Q输出RoPE部分地址 # Q输出RoPE部分地址
            q_pe.to(q_out_ptr.dtype.element_ty),  # 转换为目标数据类型 # 转换为目标数据类型
            mask=m_mask[:, None],  # token掩码 # token掩码
        )
        return  # Q路径结束 # Q路径结束

    # ===== KV path =====
    # ===== KV路径 =====
    src_id = m_offs.to(tl.int32)  # 源token ID # 源token ID
    src_mask = m_mask  # 源掩码 # 源掩码

    pos = tl.load(positions_ptr + src_id, mask=src_mask, other=0)  # 加载位置索引 # 加载位置索引
    cos_o = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d  # 计算cos/sin偏移 # 计算cos/sin偏移
    cos = tl.load(cos_ptr + cos_o, mask=src_mask[:, None], other=0)  # 加载cos值 # 加载cos值
    sin = tl.load(sin_ptr + cos_o, mask=src_mask[:, None], other=0)  # 加载sin值 # 加载sin值

    kv_base = kv_ptr + src_id[:, None].to(tl.int64) * stride_kv_m  # 计算KV基地址 # 计算KV基地址
    kv_full_ptrs = kv_base + offs_d_full[None, :] * stride_kv_d  # 计算KV全维度指针 # 计算KV全维度指针

    kv_full = tl.load(kv_full_ptrs, mask=src_mask[:, None], other=0.0).to(tl.float32)  # 加载KV全维度数据 # 加载KV全维度数据

    if kv_norm_weight_ptr is not None:  # 如果KV归一化权重存在 # 如果KV归一化权重存在
        w_kv = tl.load(kv_norm_weight_ptr + offs_d_full).to(tl.float32)  # 加载KV权重 # 加载KV权重
    else:  # 否则 # 否则
        w_kv = None  # KV权重设为None # KV权重设为None
    kv_normed = _batched_rmsnorm(kv_full, w_kv, HEAD_DIM, kv_eps)  # 对KV进行RMSNorm # 对KV进行RMSNorm

    tl.store(  # 原地存储KV的nope部分
        kv_full_ptrs,  # KV nope部分地址 # KV nope部分地址
        kv_normed.to(kv_ptr.dtype.element_ty),  # 转换为目标数据类型 # 转换为目标数据类型
        mask=src_mask[:, None] & nope_d_mask[None, :],  # nope部分掩码 # nope部分掩码
    )

    kv_pe = tl.where((offs_d_full >= NOPE_DIM)[None, :], kv_normed, 0.0)  # 提取KV的PE部分 # 提取KV的PE部分
    kv_pe = tl.reshape(kv_pe, (BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM))  # 重塑为分块形状 # 重塑为分块形状
    kv_pe = tl.sum(kv_pe, axis=1)  # 对PE分块求和 # 对PE分块求和

    kv_pe = _batched_rope(  # 对KV的PE部分应用RoPE
        kv_pe, cos, sin, d_pe_offs, BLOCK_SIZE_M, ROPE_DIM, ROPE_DIM // 2  # 传入PE数据、cos、sin及维度参数
    )
    tl.store(  # 原地存储KV的RoPE部分
        kv_base + (NOPE_DIM + d_pe_offs[None, :]) * stride_kv_d,  # KV RoPE部分地址 # KV RoPE部分地址
        kv_pe.to(kv_ptr.dtype.element_ty),  # 转换为目标数据类型 # 转换为目标数据类型
        mask=src_mask[:, None],  # token掩码 # token掩码
    )

    # ===== Paged SWA store: FP8 quant nope + BF16 rope + scales =====
    # ===== 分页SWA存储：FP8量化nope + BF16 RoPE + 缩放因子 =====
    # Layout within a page (matches fused_store_flashmla_cache CUDA kernel):
    # 页内布局（与fused_store_flashmla_cache CUDA内核匹配）：
    #   Values region: [page_size tokens * 576 bytes/token]
    #   值区域：[页大小 * token数 * 576字节/token]
    #     Per token: 448 bytes FP8 nope + 128 bytes BF16 rope
    #     每token：448字节FP8 nope + 128字节BF16 rope
    #   Scales region: [page_size tokens * 8 bytes/token]
    #   缩放区域：[页大小 * token数 * 8字节/token]
    #     Per token: 7 scale bytes + 1 pad byte
    #     每token：7个缩放字节 + 1个填充字节
    # Total per page before padding: page_size * 584
    # 填充前每页总计：页大小 * 584
    VALUE_STRIDE: tl.constexpr = DIM_NOPE + ROPE_DIM * 2  # 值步长 = nope维度 + RoPE维度*2(BF16) # 值步长
    SCALE_BYTES: tl.constexpr = NUM_NOPE_TILES + 1  # 缩放字节数 = nope分块数 + 1(填充) # 缩放字节数

    if HAS_SWA_STORE:  # 如果需要SWA存储 # 如果需要SWA存储
        loc = tl.load(swa_loc_ptr + src_id, mask=src_mask, other=0)  # 加载SWA位置索引 # 加载SWA位置索引
        page_id = loc // SWA_PAGE_SIZE  # 计算页ID # 计算页ID
        page_off = loc % SWA_PAGE_SIZE  # 计算页内偏移 # 计算页内偏移
        page_base = page_id.to(tl.int64) * swa_cache_stride_page  # 计算页基地址 # 计算页基地址
        value_base = page_base + page_off.to(tl.int64) * VALUE_STRIDE  # 计算值基地址 # 计算值基地址
        scale_base = (  # 计算缩放基地址
            page_base  # 页基地址 # 页基地址
            + SWA_PAGE_SIZE * VALUE_STRIDE  # 偏移到缩放区域 # 偏移到缩放区域
            + page_off.to(tl.int64) * SCALE_BYTES  # 加上页内偏移 # 加上页内偏移
        )

        EPS: tl.constexpr = 1e-8  # 用于防止除零的小常数 # 用于防止除零的小常数
        nope_tile_offs = tl.arange(0, TILE_SIZE)  # nope分块偏移 # nope分块偏移

        for tile_i in tl.static_range(NUM_NOPE_TILES):  # 遍历nope分块 # 遍历nope分块
            tile_start = tile_i * TILE_SIZE  # 计算分块起始位置 # 计算分块起始位置
            tile_data = tl.load(  # 加载分块数据
                kv_ptr  # KV基地址 # KV基地址
                + src_id[:, None].to(tl.int64) * stride_kv_m  # token偏移 # token偏移
                + (tile_start + nope_tile_offs[None, :]) * stride_kv_d,  # 维度偏移 # 维度偏移
                mask=src_mask[:, None],  # token掩码 # token掩码
                other=0.0,  # 掩码外填充值 # 掩码外填充值
            ).to(tl.float32)  # 转换为float32 # 转换为float32

            abs_max = tl.max(tl.abs(tile_data), axis=-1)  # 计算绝对值最大值 # 计算绝对值最大值
            abs_max_c = tl.maximum(abs_max, EPS)  # 防止除零 # 防止除零
            scale_f = abs_max_c / FP8_MAX  # 计算缩放因子 # 计算缩放因子
            log2_s = tl.log2(scale_f)  # 取以2为底的对数 # 取以2为底的对数
            ceil_log2 = tl.math.ceil(log2_s)  # 向上取整 # 向上取整
            scale_pow2 = tl.exp2(ceil_log2)  # 计算缩放的2的幂次 # 计算缩放的2的幂次
            inv_scale = 1.0 / scale_pow2  # 计算逆缩放 # 计算逆缩放
            x_scaled = tile_data * inv_scale[:, None]  # 应用逆缩放 # 应用逆缩放
            x_fp8 = tl.clamp(x_scaled, FP8_MIN, FP8_MAX)  # 裁剪到FP8范围 # 裁剪到FP8范围

            x_fp8_cast = x_fp8.to(tl.float8e4nv)  # 转换为FP8 E4M3格式 # 转换为FP8 E4M3格式
            x_fp8_bytes = x_fp8_cast.to(tl.uint8, bitcast=True)  # 将FP8按位转为uint8 # 将FP8按位转为uint8
            fp8_byte_offs = value_base[:, None] + tile_start + nope_tile_offs[None, :]  # 计算FP8字节偏移 # 计算FP8字节偏移
            tl.store(  # 存储FP8量化后的nope数据
                swa_cache_ptr + fp8_byte_offs,  # FP8数据地址 # FP8数据地址
                x_fp8_bytes,  # FP8字节数据 # FP8字节数据
                mask=src_mask[:, None],  # token掩码 # token掩码
            )

            scale_uint8 = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)  # 将log2缩放因子编码为uint8 # 将log2缩放因子编码为uint8
            tl.store(  # 存储缩放因子
                swa_cache_ptr + scale_base + tile_i,  # 缩放因子地址 # 缩放因子地址
                scale_uint8,  # 缩放因子uint8值 # 缩放因子uint8值
                mask=src_mask,  # token掩码 # token掩码
            )

        rope_data = kv_pe.to(tl.bfloat16)  # 将RoPE数据转换为BF16 # 将RoPE数据转换为BF16
        rope_offs = tl.arange(0, ROPE_DIM)  # RoPE维度偏移 # RoPE维度偏移
        rope_byte_base = value_base[:, None] + DIM_NOPE + rope_offs[None, :] * 2  # 计算RoPE字节基地址 # 计算RoPE字节基地址
        rope_data_as_i16 = rope_data.to(tl.int16, bitcast=True)  # 将BF16按位转为int16 # 将BF16按位转为int16
        lo = (rope_data_as_i16 & 0xFF).to(tl.uint8)  # 提取低8位 # 提取低8位
        hi = ((rope_data_as_i16 >> 8) & 0xFF).to(tl.uint8)  # 提取高8位 # 提取高8位
        tl.store(swa_cache_ptr + rope_byte_base, lo, mask=src_mask[:, None])  # 存储低字节 # 存储低字节
        tl.store(swa_cache_ptr + rope_byte_base + 1, hi, mask=src_mask[:, None])  # 存储高字节 # 存储高字节


# ---------------------------------------------------------------------------
# Python wrapper
# Python包装函数
# ---------------------------------------------------------------------------


# 融合Q/K归一化 + RoPE + SWA存储的Python包装函数
def fused_qk_norm_rope_swa_store(  # 融合Q/K归一化+RoPE+SWA存储函数
    q: torch.Tensor,  # Q张量 [M, N] 或 [splitk, M, N] # Q张量
    kv: torch.Tensor,  # KV张量 [M, head_dim=512]，原地修改 # KV张量
    q_norm_weight: Optional[torch.Tensor],  # Q归一化权重 # Q归一化权重
    kv_norm_weight: Optional[torch.Tensor],  # KV归一化权重 # KV归一化权重
    q_rms_eps: float,  # Q归一化epsilon # Q归一化epsilon
    kv_rms_eps: float,  # KV归一化epsilon # KV归一化epsilon
    rope_head_dim: int,  # RoPE头维度 # RoPE头维度
    cos_cache: torch.Tensor,  # RoPE余弦缓存 # RoPE余弦缓存
    sin_cache: torch.Tensor,  # RoPE正弦缓存 # RoPE正弦缓存
    positions: torch.Tensor,  # 位置索引 # 位置索引
    swa_cache: Optional[torch.Tensor] = None,  # SWA分页KV池缓冲区 [num_pages, bytes_per_page] uint8 # SWA缓存
    swa_loc: Optional[torch.Tensor] = None,  # SWA位置索引 [M] int32 # SWA位置索引
    swa_page_size: int = 128,  # 每页token数（默认128） # SWA页大小
    q_out: Optional[torch.Tensor] = None,  # Q输出张量（可选） # Q输出张量
    dtype: torch.dtype = torch.bfloat16,  # 输出数据类型（默认BF16） # 输出数据类型
) -> torch.Tensor:  # 返回Q输出张量 # 返回Q输出张量
    """Fused Q norm + KV norm + RoPE + optional FP8 paged SWA store.
    融合Q归一化 + KV归一化 + RoPE + 可选FP8分页SWA存储。

    Args:
        q: [M, N] or [splitk, M, N] where N = num_local_heads * head_dim
        Q: [M, N] 或 [splitk, M, N]，其中N = 本地头数 * 头维度
        kv: [M, head_dim=512] mutated in-place (norm + RoPE)
        KV: [M, head_dim=512] 原地修改（归一化 + RoPE）
        swa_cache: paged SWA KV pool buffer [num_pages, bytes_per_page] uint8
        SWA缓存：分页SWA KV池缓冲区 [num_pages, bytes_per_page] uint8
        swa_loc: [M] int32 pre-translated paged indices
        SWA位置：[M] int32 预转换的分页索引
        swa_page_size: tokens per SWA page (default 128)
        SWA页大小：每页token数（默认128）
    """
    head_dim = kv.shape[1]  # 获取头维度 # 获取头维度

    if q.dim() == 3:  # 如果Q是3维（含split-K维度） # 如果Q是3维
        num_splitk, M, N = q.shape  # 解包3维形状 # 解包3维形状
        q_in_splitk_stride = q.stride(0)  # split-K步长 # split-K步长
        q_in_m_stride = q.stride(1)  # token步长 # token步长
        q_in_d_stride = q.stride(2)  # 维度步长 # 维度步长
    else:  # 否则Q是2维 # 否则Q是2维
        M, N = q.shape  # 解包2维形状 # 解包2维形状
        num_splitk = 1  # split-K数为1 # split-K数为1
        q_in_splitk_stride = 0  # split-K步长为0 # split-K步长为0
        q_in_m_stride = q.stride(0)  # token步长 # token步长
        q_in_d_stride = q.stride(1)  # 维度步长 # 维度步长

    num_local_heads = N // head_dim  # 计算本地Q头数量 # 计算本地Q头数量

    if q_out is None:  # 如果未提供Q输出 # 如果未提供Q输出
        q_out = torch.empty(  # 分配Q输出张量
            (M, num_local_heads, head_dim), dtype=dtype, device=q.device  # 形状为[M, 头数, 头维度]
        )

    HAS_SWA_STORE = swa_cache is not None and swa_loc is not None  # 判断是否需要SWA存储 # 判断是否需要SWA存储

    dim_nope = 448  # nope维度大小 # nope维度大小
    dim_rope = 64  # RoPE维度大小 # RoPE维度大小
    tile_size = 64  # 分块大小 # 分块大小
    num_nope_tiles = dim_nope // tile_size  # nope分块数量 # nope分块数量
    scale_pad = 1  # 缩放填充字节数 # 缩放填充字节数
    bytes_per_token = dim_nope + dim_rope * 2 + num_nope_tiles + scale_pad  # 每token字节数 # 每token字节数

    if _fp8_fnuz:  # 如果使用FNUZ格式 # 如果使用FNUZ格式
        fp8_info = torch.finfo(torch.float8_e4m3fnuz)  # 获取FNUZ格式信息 # 获取FNUZ格式信息
    else:  # 否则 # 否则
        fp8_info = torch.finfo(torch.float8_e4m3fn)  # 获取标准FP8格式信息 # 获取标准FP8格式信息

    BLOCK_SIZE_M = min(4, triton.next_power_of_2(M)) if M < 4 else 4  # 计算M方向块大小 # 计算M方向块大小
    num_warps = 4  # 设置warp数量为4 # 设置warp数量为4

    grid = (triton.cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)  # 计算网格大小 # 计算网格大小
    _fused_qk_norm_rope_store_kernel[grid](  # 启动融合内核
        q,  # Q输入 # Q输入
        q_out,  # Q输出 # Q输出
        kv,  # KV数据 # KV数据
        q_norm_weight,  # Q归一化权重 # Q归一化权重
        kv_norm_weight,  # KV归一化权重 # KV归一化权重
        positions,  # 位置索引 # 位置索引
        cos_cache,  # 余弦缓存 # 余弦缓存
        sin_cache,  # 正弦缓存 # 正弦缓存
        swa_cache if HAS_SWA_STORE else None,  # SWA缓存 # SWA缓存
        swa_loc if HAS_SWA_STORE else None,  # SWA位置索引 # SWA位置索引
        M,  # token数量 # token数量
        q_in_splitk_stride,  # Q输入split-K步长 # Q输入split-K步长
        q_in_m_stride,  # Q输入token步长 # Q输入token步长
        q_in_d_stride,  # Q输入维度步长 # Q输入维度步长
        q_out.stride(0),  # Q输出token步长 # Q输出token步长
        q_out.stride(1),  # Q输出头步长 # Q输出头步长
        q_out.stride(2),  # Q输出维度步长 # Q输出维度步长
        kv.stride(0),  # KV token步长 # KV token步长
        kv.stride(1),  # KV维度步长 # KV维度步长
        cos_cache.stride(0),  # 余弦token步长 # 余弦token步长
        cos_cache.stride(-1),  # 余弦维度步长 # 余弦维度步长
        swa_cache.stride(0) if HAS_SWA_STORE else 0,  # SWA缓存页步长 # SWA缓存页步长
        q_rms_eps,  # Q归一化epsilon # Q归一化epsilon
        kv_rms_eps,  # KV归一化epsilon # KV归一化epsilon
        BLOCK_SIZE_M=BLOCK_SIZE_M,  # M方向块大小 # M方向块大小
        HEAD_DIM=head_dim,  # 头维度 # 头维度
        ROPE_DIM=rope_head_dim,  # RoPE维度 # RoPE维度
        NUM_LOCAL_HEADS=num_local_heads,  # 本地Q头数量 # 本地Q头数量
        NUM_SPLITK=num_splitk,  # split-K数量 # split-K数量
        HAS_SWA_STORE=HAS_SWA_STORE,  # 是否有SWA存储 # 是否有SWA存储
        DIM_NOPE=dim_nope,  # nope维度大小 # nope维度大小
        TILE_SIZE=tile_size,  # 分块大小 # 分块大小
        NUM_NOPE_TILES=num_nope_tiles,  # nope分块数量 # nope分块数量
        FP8_MIN=fp8_info.min,  # FP8最小值 # FP8最小值
        FP8_MAX=fp8_info.max,  # FP8最大值 # FP8最大值
        BYTES_PER_TOKEN=bytes_per_token,  # 每token字节数 # 每token字节数
        SWA_PAGE_SIZE=swa_page_size,  # SWA页大小 # SWA页大小
        num_warps=num_warps,  # warp数量 # warp数量
    )
    return q_out  # 返回Q输出张量 # 返回Q输出张量
