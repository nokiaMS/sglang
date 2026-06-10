# DSV4 (d_qk=512) 的 Triton MLA 解码内核模块
# 本模块包含 DSV4 特定的 gather+反量化 内核以及 DSV4 稀疏注意力解码的主入口点
# 核心功能：从 FP8 量化的 KV 缓存中按稀疏索引收集数据并反量化为 BF16，
# 支持 main 和 extra 两种 KV 作用域的融合收集操作
"""
Triton MLA Decode Kernels for DSV4 (d_qk=512). # DSV4 (d_qk=512) 的 Triton MLA 解码内核

This module contains DSV4-specific gather+dequant kernels and the main # 本模块包含 DSV4 特定的 gather+反量化 内核以及
sparse attention decode entry point for DSV4. # DSV4 稀疏注意力解码的主入口点
"""

import os # 导入操作系统模块
from typing import Optional, Tuple # 导入类型提示

import torch # 导入 PyTorch
import triton # 导入 Triton
import triton.language as tl # 导入 Triton 语言

from .triton_mla_kernels_decode_common import ( # 从公共模块导入辅助函数
    _bucket_total_tokens, # 将 total_tokens 分桶
    _get_workload_size_category, # 获取工作负载大小类别
    compute_token_ranges, # 计算令牌范围（用于分块）
    run_chunked_attention_triton, # 运行分块注意力（Triton 版本）
    run_splitk_unified_attention, # 运行 Split-K 统一注意力
    run_unified_attention, # 运行统一注意力
    slice_kv_scope_for_tokens, # 为指定令牌切片 KV 作用域
)

# Enable Triton autotune cache persistence # 启用 Triton 自动调优缓存持久化
TRITON_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".triton_cache") # Triton 缓存目录路径
os.makedirs(TRITON_CACHE_DIR, exist_ok=True) # 创建缓存目录（如不存在）
os.environ.setdefault("TRITON_CACHE_DIR", TRITON_CACHE_DIR) # 设置环境变量默认值

# Constants for DSV4 layout # DSV4 布局的常量定义
DSV4_D_QK = 512 # QK 维度大小
DSV4_D_NOPE = 448 # 非旋转位置编码维度
DSV4_D_ROPE = 64 # 旋转位置编码维度
DSV4_TILE_SIZE = 64 # 每个 tile 的大小
DSV4_NUM_TILES = 7 # tile 数量（448/64=7）
DSV4_BYTES_PER_TOKEN_DATA = 576  # 448 nope + 128 rope # 每个令牌数据的字节数（nope + rope）
DSV4_BYTES_PER_TOKEN_SCALE = 8  # 7 scales + 1 padding # 每个令牌缩放因子的字节数（7个缩放 + 1个填充）

# Performance tuning thresholds (empirically determined) # 性能调优阈值（经验确定）
# These thresholds balance kernel launch overhead vs. computation efficiency # 这些阈值平衡了内核启动开销与计算效率
#
# DSV4_USE_FUSED_THRESHOLD: Use 1D fused kernel below this element count # DSV4_USE_FUSED_THRESHOLD：元素数低于此值时使用一维融合内核
#   Rationale: Single kernel launch reduces overhead for small/medium workloads # 理由：单次内核启动减少中小型工作负载的开销
#   Value 150K determined by benchmarking on typical production workloads # 值 150K 由典型生产工作负载的基准测试确定
DSV4_USE_FUSED_THRESHOLD = 150000 # 融合内核使用阈值（15万元素）
#
# DSV4_USE_FIXED_KERNEL_THRESHOLD: Use fixed BLOCK_TK=128 kernel below this # DSV4_USE_FIXED_KERNEL_THRESHOLD：低于此值时使用固定 BLOCK_TK=128 的内核
#   Rationale: Avoids autotune overhead for small workloads where fixed config # 理由：对于固定配置表现良好的小型工作负载，避免自动调优开销
#   performs well. Value 32K balances autotune benefit vs. overhead # 表现良好。值 32K 平衡了自动调优收益与开销
DSV4_USE_FIXED_KERNEL_THRESHOLD = 32768 # 固定内核使用阈值（32768元素）


# ============================================================================
# DSV4 Gather+Dequant Kernels - Optimized with Batched Scale Loading # DSV4 收集+反量化内核 - 批量缩放加载优化
# ============================================================================


@triton.autotune( # Triton 自动调优装饰器
    configs=[ # 配置列表
        # This is a pure memory-copy + FP8→BF16 dequant kernel. # 这是纯内存拷贝 + FP8→BF16 反量化内核
        # - BLOCK_TK controls how many (token×topk) pairs per block. # BLOCK_TK 控制每个块处理多少 (令牌×topk) 对
        # - Larger BLOCK_TK amortizes launch overhead but needs more warps. # 较大的 BLOCK_TK 分摊启动开销但需要更多 warp
        # - BLOCK_TK=128 is already validated as the fixed config for small workloads # BLOCK_TK=128 已验证为小型工作负载的固定配置
        #   (below DSV4_USE_FIXED_KERNEL_THRESHOLD = 32K elements). # （低于 DSV4_USE_FIXED_KERNEL_THRESHOLD = 32K 元素）
        # - BLOCK_TK=64/128: good for small/medium workloads (fewer warps, less overhead). # BLOCK_TK=64/128：适合中小型工作负载（warp 少，开销低）
        # - BLOCK_TK=256: better bandwidth utilization for large workloads. # BLOCK_TK=256：大型工作负载有更好的带宽利用率
        triton.Config({"BLOCK_TK": 64}, num_warps=4, num_stages=1), # BLOCK_TK=64，4个warp
        triton.Config({"BLOCK_TK": 128}, num_warps=4, num_stages=1), # BLOCK_TK=128，4个warp
        triton.Config({"BLOCK_TK": 256}, num_warps=8, num_stages=1), # BLOCK_TK=256，8个warp
    ],
    key=["total_tokens_bucket", "topk", "workload_size_cat"], # 自动调优键
)
@triton.jit # Triton JIT 编译装饰器
def _gather_dequant_dsv4_kernel( # DSV4 gather+反量化内核（带自动调优）
    KV_Cache, # KV 缓存指针
    Indices, # 索引指针
    TopkLength, # Topk 长度指针
    OutputKV, # 输出 KV 指针
    OutputMask, # 输出掩码指针
    total_tokens, # 总令牌数
    total_tokens_bucket, # 总令牌数（分桶后）
    topk, # topk 值
    num_blocks, # 块数量
    block_size, # 块大小
    workload_size_cat, # 工作负载大小类别
    k_offset, # K 维偏移量
    s_q, # 每批次的序列长度
    stride_kv_block, # KV 缓存块步长
    stride_idx_t, # 索引令牌步长
    stride_idx_k, # 索引 K 步长
    stride_out_t, # 输出令牌步长
    stride_out_k, # 输出 K 步长
    stride_out_d, # 输出维度步长
    stride_mask_t, # 掩码令牌步长
    stride_mask_k, # 掩码 K 步长
    BLOCK_TK: tl.constexpr, # 每块处理的 (令牌×topk) 对数（编译时常量）
    D_NOPE: tl.constexpr, # 非旋转维度大小（编译时常量）
    D_ROPE: tl.constexpr, # 旋转维度大小（编译时常量）
    BYTES_PER_TOKEN_DATA: tl.constexpr, # 每令牌数据字节数（编译时常量）
    BYTES_PER_TOKEN_SCALE: tl.constexpr, # 每令牌缩放字节数（编译时常量）
    TILE_SIZE: tl.constexpr, # tile 大小（编译时常量）
    HAS_TOPK_LENGTH: tl.constexpr, # 是否有 topk_length 参数（编译时常量）
):
    """Optimized gather + dequant kernel with batched scale loading.""" # 带批量缩放加载的优化 gather+反量化内核
    pid = tl.program_id(0) # 获取程序 ID
    num_tk = total_tokens * topk # 总的 (令牌×topk) 对数

    offs_tk = pid * BLOCK_TK + tl.arange(0, BLOCK_TK) # 计算本块处理的偏移量
    mask_tk = offs_tk < num_tk # 越界掩码

    t_idx = offs_tk // topk # 令牌索引
    k_idx = offs_tk % topk # topk 索引

    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k # 索引指针
    indices = tl.load(idx_ptrs, mask=mask_tk, other=-1) # 加载索引值

    is_invalid = indices == -1 # 判断是否为无效索引

    if HAS_TOPK_LENGTH: # 如果有 topk_length 参数
        batch_idx = t_idx // s_q # 计算批次索引
        topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk) # 加载每批次的 topk 长度
        is_invalid = is_invalid | (k_idx >= topk_len) # 超出 topk 长度的也是无效的

    mask_out_ptrs = ( # 输出掩码指针
        OutputMask + t_idx * stride_mask_t + (k_idx + k_offset) * stride_mask_k # 加上 K 偏移量
    )
    tl.store(mask_out_ptrs, is_invalid, mask=mask_tk) # 存储无效掩码

    valid_mask = mask_tk & ~is_invalid # 有效掩码
    indices_clamped = tl.maximum(indices, 0) # 将索引限制为非负值

    block_idx = indices_clamped // block_size # 计算块索引
    offset_in_block = indices_clamped % block_size # 计算块内偏移

    block_idx_64 = block_idx.to(tl.int64) # 转换为 int64
    offset_in_block_64 = offset_in_block.to(tl.int64) # 转换为 int64

    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block # KV 缓存块基地址

    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA # nope+rope 的偏移量
    scale_base_offset = ( # 缩放因子的基础偏移量
        block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE # 块数据之后紧跟缩放数据
    )

    t_idx_64 = t_idx.to(tl.int64) # 令牌索引转 int64
    k_idx_64 = k_idx.to(tl.int64) # topk 索引转 int64
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64) # 输出令牌步长转 int64
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64) # 输出 K 步长转 int64
    out_base_ptrs = ( # 输出基地址指针
        OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64 # 加上 K 偏移
    )

    # Load all 7 scales at once - each scale is at scale_base_offset + tile_idx # 一次加载所有 7 个缩放因子 - 每个缩放在 scale_base_offset + tile_idx
    scale_ptrs_0 = kv_block_base + scale_base_offset # 缩放因子0指针
    scale_ptrs_1 = kv_block_base + scale_base_offset + 1 # 缩放因子1指针
    scale_ptrs_2 = kv_block_base + scale_base_offset + 2 # 缩放因子2指针
    scale_ptrs_3 = kv_block_base + scale_base_offset + 3 # 缩放因子3指针
    scale_ptrs_4 = kv_block_base + scale_base_offset + 4 # 缩放因子4指针
    scale_ptrs_5 = kv_block_base + scale_base_offset + 5 # 缩放因子5指针
    scale_ptrs_6 = kv_block_base + scale_base_offset + 6 # 缩放因子6指针

    scale_uint8_0 = tl.load(scale_ptrs_0, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子0为 uint8
    scale_uint8_1 = tl.load(scale_ptrs_1, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子1为 uint8
    scale_uint8_2 = tl.load(scale_ptrs_2, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子2为 uint8
    scale_uint8_3 = tl.load(scale_ptrs_3, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子3为 uint8
    scale_uint8_4 = tl.load(scale_ptrs_4, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子4为 uint8
    scale_uint8_5 = tl.load(scale_ptrs_5, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子5为 uint8
    scale_uint8_6 = tl.load(scale_ptrs_6, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子6为 uint8

    # Convert all scales to bf16 and pre-compute 2D versions # 将所有缩放因子转换为 bf16 并预计算 2D 版本
    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子0：uint8→float32→exp2→bf16
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子1
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子2
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子3
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子4
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子5
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子6
    # Pre-compute 2D versions for tile processing # 预计算 2D 版本用于 tile 处理
    scale_2d_0 = scale_bf16_0[:, None] # 缩放因子0 扩展为2D
    scale_2d_1 = scale_bf16_1[:, None] # 缩放因子1 扩展为2D
    scale_2d_2 = scale_bf16_2[:, None] # 缩放因子2 扩展为2D
    scale_2d_3 = scale_bf16_3[:, None] # 缩放因子3 扩展为2D
    scale_2d_4 = scale_bf16_4[:, None] # 缩放因子4 扩展为2D
    scale_2d_5 = scale_bf16_5[:, None] # 缩放因子5 扩展为2D
    scale_2d_6 = scale_bf16_6[:, None] # 缩放因子6 扩展为2D

    offs_d = tl.arange(0, TILE_SIZE) # 维度偏移量

    # Pre-compute base pointers for optimization # 预计算基础指针以优化性能
    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None] # tile 基地址
    out_base = out_base_ptrs[:, None] # 输出基地址
    valid_mask_2d = valid_mask[:, None] # 2D 有效掩码
    is_invalid_2d = is_invalid[:, None] # 2D 无效标记
    mask_tk_2d = mask_tk[:, None] # 2D 越界掩码

    # Process tile 0 # 处理 tile 0
    nope_ptrs = tile_base + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_0 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + offs_d[None, :] * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 1 # 处理 tile 1
    tile_start_1 = TILE_SIZE # tile 1 起始偏移
    nope_ptrs = tile_base + tile_start_1 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_1 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_1 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 2 # 处理 tile 2
    tile_start_2 = 2 * TILE_SIZE # tile 2 起始偏移
    nope_ptrs = tile_base + tile_start_2 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_2 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_2 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 3 # 处理 tile 3
    tile_start_3 = 3 * TILE_SIZE # tile 3 起始偏移
    nope_ptrs = tile_base + tile_start_3 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_3 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_3 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 4 # 处理 tile 4
    tile_start_4 = 4 * TILE_SIZE # tile 4 起始偏移
    nope_ptrs = tile_base + tile_start_4 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_4 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_4 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 5 # 处理 tile 5
    tile_start_5 = 5 * TILE_SIZE # tile 5 起始偏移
    nope_ptrs = tile_base + tile_start_5 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_5 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_5 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 6 # 处理 tile 6
    tile_start_6 = 6 * TILE_SIZE # tile 6 起始偏移
    nope_ptrs = tile_base + tile_start_6 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_6 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_6 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process rope # 处理旋转位置编码数据
    offs_rope = tl.arange(0, D_ROPE) # RoPE 维度偏移
    rope_byte_start = D_NOPE # RoPE 数据在令牌中的起始字节位置

    rope_lo_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 # RoPE 低字节指针
    rope_hi_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 + 1 # RoPE 高字节指针

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16) # 加载低字节
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16) # 加载高字节

    rope_uint16 = rope_lo | (rope_hi << 8) # 合并为 uint16
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True) # 位转换为 BF16
    rope_bf16 = tl.where(is_invalid_2d, 0.0, rope_bf16) # 无效位置填零

    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, rope_bf16, mask=mask_tk_2d) # 存储 RoPE 结果


@triton.jit # Triton JIT 编译装饰器
def _gather_dequant_dsv4_kernel_fixed_128( # DSV4 gather+反量化内核（固定 BLOCK_TK=128）
    KV_Cache, # KV 缓存指针
    Indices, # 索引指针
    TopkLength, # Topk 长度指针
    OutputKV, # 输出 KV 指针
    OutputMask, # 输出掩码指针
    total_tokens, # 总令牌数
    total_tokens_bucket, # 总令牌数（分桶后）
    topk, # topk 值
    num_blocks, # 块数量
    block_size, # 块大小
    k_offset, # K 维偏移量
    s_q, # 每批次的序列长度
    stride_kv_block, # KV 缓存块步长
    stride_idx_t, # 索引令牌步长
    stride_idx_k, # 索引 K 步长
    stride_out_t, # 输出令牌步长
    stride_out_k, # 输出 K 步长
    stride_out_d, # 输出维度步长
    stride_mask_t, # 掩码令牌步长
    stride_mask_k, # 掩码 K 步长
    D_NOPE: tl.constexpr, # 非旋转维度大小（编译时常量）
    D_ROPE: tl.constexpr, # 旋转维度大小（编译时常量）
    BYTES_PER_TOKEN_DATA: tl.constexpr, # 每令牌数据字节数（编译时常量）
    BYTES_PER_TOKEN_SCALE: tl.constexpr, # 每令牌缩放字节数（编译时常量）
    TILE_SIZE: tl.constexpr, # tile 大小（编译时常量）
    HAS_TOPK_LENGTH: tl.constexpr, # 是否有 topk_length 参数（编译时常量）
):
    """Fixed-config gather kernel with BLOCK_TK=128 and batched scale loading.""" # 固定配置的 gather 内核，BLOCK_TK=128，批量缩放加载
    BLOCK_TK: tl.constexpr = 128 # 固定 BLOCK_TK 为 128
    pid = tl.program_id(0) # 获取程序 ID
    num_tk = total_tokens * topk # 总的 (令牌×topk) 对数

    offs_tk = pid * BLOCK_TK + tl.arange(0, BLOCK_TK) # 计算本块处理的偏移量
    mask_tk = offs_tk < num_tk # 越界掩码

    t_idx = offs_tk // topk # 令牌索引
    k_idx = offs_tk % topk # topk 索引

    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k # 索引指针
    indices = tl.load(idx_ptrs, mask=mask_tk, other=-1) # 加载索引值

    is_invalid = indices == -1 # 判断是否为无效索引

    if HAS_TOPK_LENGTH: # 如果有 topk_length 参数
        batch_idx = t_idx // s_q # 计算批次索引
        topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk) # 加载每批次的 topk 长度
        is_invalid = is_invalid | (k_idx >= topk_len) # 超出 topk 长度的也是无效的

    mask_out_ptrs = ( # 输出掩码指针
        OutputMask + t_idx * stride_mask_t + (k_idx + k_offset) * stride_mask_k # 加上 K 偏移量
    )
    tl.store(mask_out_ptrs, is_invalid, mask=mask_tk) # 存储无效掩码

    valid_mask = mask_tk & ~is_invalid # 有效掩码
    indices_clamped = tl.maximum(indices, 0) # 将索引限制为非负值

    block_idx = indices_clamped // block_size # 计算块索引
    offset_in_block = indices_clamped % block_size # 计算块内偏移

    block_idx_64 = block_idx.to(tl.int64) # 转换为 int64
    offset_in_block_64 = offset_in_block.to(tl.int64) # 转换为 int64

    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block # KV 缓存块基地址

    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA # nope+rope 的偏移量
    scale_base_offset = ( # 缩放因子的基础偏移量
        block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE # 块数据之后紧跟缩放数据
    )

    t_idx_64 = t_idx.to(tl.int64) # 令牌索引转 int64
    k_idx_64 = k_idx.to(tl.int64) # topk 索引转 int64
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64) # 输出令牌步长转 int64
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64) # 输出 K 步长转 int64
    out_base_ptrs = ( # 输出基地址指针
        OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64 # 加上 K 偏移
    )

    # Load all 7 scales at once # 一次加载所有 7 个缩放因子
    scale_ptrs_0 = kv_block_base + scale_base_offset # 缩放因子0指针
    scale_ptrs_1 = kv_block_base + scale_base_offset + 1 # 缩放因子1指针
    scale_ptrs_2 = kv_block_base + scale_base_offset + 2 # 缩放因子2指针
    scale_ptrs_3 = kv_block_base + scale_base_offset + 3 # 缩放因子3指针
    scale_ptrs_4 = kv_block_base + scale_base_offset + 4 # 缩放因子4指针
    scale_ptrs_5 = kv_block_base + scale_base_offset + 5 # 缩放因子5指针
    scale_ptrs_6 = kv_block_base + scale_base_offset + 6 # 缩放因子6指针

    scale_uint8_0 = tl.load(scale_ptrs_0, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子0为 uint8
    scale_uint8_1 = tl.load(scale_ptrs_1, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子1为 uint8
    scale_uint8_2 = tl.load(scale_ptrs_2, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子2为 uint8
    scale_uint8_3 = tl.load(scale_ptrs_3, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子3为 uint8
    scale_uint8_4 = tl.load(scale_ptrs_4, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子4为 uint8
    scale_uint8_5 = tl.load(scale_ptrs_5, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子5为 uint8
    scale_uint8_6 = tl.load(scale_ptrs_6, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子6为 uint8

    # Convert all scales to bf16 and pre-compute 2D versions # 将所有缩放因子转换为 bf16 并预计算 2D 版本
    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子0：uint8→float32→exp2→bf16
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子1
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子2
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子3
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子4
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子5
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子6
    # Pre-compute 2D versions for tile processing # 预计算 2D 版本用于 tile 处理
    scale_2d_0 = scale_bf16_0[:, None] # 缩放因子0 扩展为2D
    scale_2d_1 = scale_bf16_1[:, None] # 缩放因子1 扩展为2D
    scale_2d_2 = scale_bf16_2[:, None] # 缩放因子2 扩展为2D
    scale_2d_3 = scale_bf16_3[:, None] # 缩放因子3 扩展为2D
    scale_2d_4 = scale_bf16_4[:, None] # 缩放因子4 扩展为2D
    scale_2d_5 = scale_bf16_5[:, None] # 缩放因子5 扩展为2D
    scale_2d_6 = scale_bf16_6[:, None] # 缩放因子6 扩展为2D

    offs_d = tl.arange(0, TILE_SIZE) # 维度偏移量

    # Pre-compute base pointers for optimization # 预计算基础指针以优化性能
    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None] # tile 基地址
    out_base = out_base_ptrs[:, None] # 输出基地址
    valid_mask_2d = valid_mask[:, None] # 2D 有效掩码
    is_invalid_2d = is_invalid[:, None] # 2D 无效标记
    mask_tk_2d = mask_tk[:, None] # 2D 越界掩码

    # Process tile 0 # 处理 tile 0
    nope_ptrs = tile_base + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_0 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + offs_d[None, :] * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 1 # 处理 tile 1
    tile_start_1 = TILE_SIZE # tile 1 起始偏移
    nope_ptrs = tile_base + tile_start_1 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_1 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_1 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 2 # 处理 tile 2
    tile_start_2 = 2 * TILE_SIZE # tile 2 起始偏移
    nope_ptrs = tile_base + tile_start_2 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_2 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_2 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 3 # 处理 tile 3
    tile_start_3 = 3 * TILE_SIZE # tile 3 起始偏移
    nope_ptrs = tile_base + tile_start_3 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_3 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_3 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 4 # 处理 tile 4
    tile_start_4 = 4 * TILE_SIZE # tile 4 起始偏移
    nope_ptrs = tile_base + tile_start_4 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_4 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_4 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 5 # 处理 tile 5
    tile_start_5 = 5 * TILE_SIZE # tile 5 起始偏移
    nope_ptrs = tile_base + tile_start_5 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_5 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_5 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process tile 6 # 处理 tile 6
    tile_start_6 = 6 * TILE_SIZE # tile 6 起始偏移
    nope_ptrs = tile_base + tile_start_6 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_6 # 反量化：乘以缩放因子
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + (tile_start_6 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储反量化结果

    # Process rope # 处理旋转位置编码数据
    offs_rope = tl.arange(0, D_ROPE) # RoPE 维度偏移
    rope_byte_start = D_NOPE # RoPE 数据在令牌中的起始字节位置

    rope_lo_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 # RoPE 低字节指针
    rope_hi_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 + 1 # RoPE 高字节指针

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16) # 加载低字节
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16) # 加载高字节

    rope_uint16 = rope_lo | (rope_hi << 8) # 合并为 uint16
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True) # 位转换为 BF16
    rope_bf16 = tl.where(is_invalid_2d, 0.0, rope_bf16) # 无效位置填零

    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, rope_bf16, mask=mask_tk_2d) # 存储 RoPE 结果


# ============================================================================
# DSV4 Wrapper Functions # DSV4 包装函数
# ============================================================================


def gather_dequant_fp8_dsv4( # DSV4 gather+反量化统一入口函数
    kv_cache_quantized: torch.Tensor, # 量化后的 KV 缓存
    indices: torch.Tensor, # 稀疏索引
    block_size: int, # 块大小
    output_kv: torch.Tensor, # 输出 KV 张量
    output_mask: torch.Tensor, # 输出掩码张量
    k_offset: int = 0, # K 维偏移量，默认为0
    topk_length: Optional[torch.Tensor] = None, # 可选的每批次 topk 长度
    s_q: int = 1, # 每批次的序列长度，默认为1
) -> bool: # 返回是否成功
    """Unified DSV4 gather+dequant with optional topk_length mask.""" # 统一的 DSV4 gather+反量化，支持可选的 topk_length 掩码
    total_tokens, topk = indices.shape # 获取总令牌数和 topk 值
    num_blocks = kv_cache_quantized.shape[0] # 获取块数量

    kv_uint8 = kv_cache_quantized.view(torch.uint8) # 将量化缓存视为 uint8
    bytes_per_block = kv_uint8.shape[1] * kv_uint8.shape[2] * kv_uint8.shape[3] # 计算每块字节数
    kv_flat = kv_uint8.reshape(num_blocks, bytes_per_block) # 展平为 2D

    stride_kv_block = kv_uint8.stride(0) # 获取块步长
    workload_size_cat = _get_workload_size_category(total_tokens, topk) # 获取工作负载大小类别

    grid = lambda meta: (triton.cdiv(total_tokens * topk, meta["BLOCK_TK"]),) # 网格大小

    topk_length_tensor = topk_length if topk_length is not None else output_mask[:1, 0] # 使用 topk_length 或占位张量
    has_topk_length = topk_length is not None # 是否有 topk_length

    _gather_dequant_dsv4_kernel[grid]( # 启动自动调优的 gather+反量化内核
        kv_flat, # 展平的 KV 缓存
        indices, # 稀疏索引
        topk_length_tensor, # topk 长度张量
        output_kv, # 输出 KV
        output_mask, # 输出掩码
        total_tokens, # 总令牌数
        _bucket_total_tokens(total_tokens), # 分桶后的令牌数
        topk, # topk 值
        num_blocks, # 块数量
        block_size, # 块大小
        workload_size_cat, # 工作负载大小类别
        k_offset, # K 维偏移量
        s_q, # 每批次序列长度
        stride_kv_block, # KV 块步长
        indices.stride(0), # 索引令牌步长
        indices.stride(1), # 索引 K 步长
        output_kv.stride(0), # 输出令牌步长
        output_kv.stride(1), # 输出 K 步长
        output_kv.stride(2), # 输出维度步长
        output_mask.stride(0), # 掩码令牌步长
        output_mask.stride(1), # 掩码 K 步长
        D_NOPE=DSV4_D_NOPE, # 非旋转维度大小
        D_ROPE=DSV4_D_ROPE, # 旋转维度大小
        BYTES_PER_TOKEN_DATA=DSV4_BYTES_PER_TOKEN_DATA, # 每令牌数据字节数
        BYTES_PER_TOKEN_SCALE=DSV4_BYTES_PER_TOKEN_SCALE, # 每令牌缩放字节数
        TILE_SIZE=DSV4_TILE_SIZE, # tile 大小
        HAS_TOPK_LENGTH=has_topk_length, # 是否有 topk_length
    )
    return True # 返回成功


# ============================================================================
# DSV4 1D Grid Fused Gather+Dequant Kernel (Optimized - No Empty Blocks) # DSV4 一维网格融合 gather+反量化内核（优化 - 无空块）
# Single kernel launch with 1D grid: (num_main_pids + num_extra_pids,) # 单次内核启动，一维网格：(主进程数 + 额外进程数,)
# ============================================================================


@triton.jit # Triton JIT 编译装饰器
def _gather_dequant_dsv4_1d_fused_kernel( # DSV4 一维融合 gather+反量化内核
    # Main KV cache # 主 KV 缓存
    KV_Cache_Main, # 主 KV 缓存指针
    Indices_Main, # 主索引指针
    TopkLength_Main, # 主 TopkLength 指针
    # Extra KV cache # 额外 KV 缓存
    KV_Cache_Extra, # 额外 KV 缓存指针
    Indices_Extra, # 额外索引指针
    TopkLength_Extra, # 额外 TopkLength 指针
    # Output # 输出
    OutputKV, # 输出 KV 指针
    OutputMask, # 输出掩码指针
    # Dimensions # 维度
    total_tokens, # 总令牌数
    topk_main, # 主 topk 值
    topk_extra, # 额外 topk 值
    num_blocks_main, # 主块数量
    num_blocks_extra, # 额外块数量
    block_size_main, # 主块大小
    block_size_extra, # 额外块大小
    s_q, # 每批次的序列长度
    # Strides for main # 主缓存步长
    stride_kv_block_main, # 主 KV 块步长
    stride_idx_t_main, # 主索引令牌步长
    stride_idx_k_main, # 主索引 K 步长
    # Strides for extra # 额外缓存步长
    stride_kv_block_extra, # 额外 KV 块步长
    stride_idx_t_extra, # 额外索引令牌步长
    stride_idx_k_extra, # 额外索引 K 步长
    # Output strides # 输出步长
    stride_out_t, # 输出令牌步长
    stride_out_k, # 输出 K 步长
    stride_out_d, # 输出维度步长
    stride_mask_t, # 掩码令牌步长
    stride_mask_k, # 掩码 K 步长
    # Grid info # 网格信息
    num_main_pids, # 主进程数
    # Constexpr # 编译时常量
    BLOCK_TK: tl.constexpr, # 每块处理的 (令牌×topk) 对数
    D_NOPE: tl.constexpr, # 非旋转维度大小
    D_ROPE: tl.constexpr, # 旋转维度大小
    BYTES_PER_TOKEN_DATA: tl.constexpr, # 每令牌数据字节数
    BYTES_PER_TOKEN_SCALE: tl.constexpr, # 每令牌缩放字节数
    TILE_SIZE: tl.constexpr, # tile 大小
    HAS_TOPK_LENGTH_MAIN: tl.constexpr, # 主缓存是否有 topk_length
    HAS_TOPK_LENGTH_EXTRA: tl.constexpr, # 额外缓存是否有 topk_length
):
    """1D fused gather kernel - single launch, no empty blocks. # 一维融合 gather 内核 - 单次启动，无空块

    Grid: (num_main_pids + num_extra_pids,) # 网格：(主进程数 + 额外进程数,)
    - pid < num_main_pids: process main cache # pid < 主进程数：处理主缓存
    - pid >= num_main_pids: process extra cache # pid >= 主进程数：处理额外缓存

    This eliminates empty blocks when main/extra topk differ significantly. # 当主/额外 topk 差异较大时，消除空块
    """
    pid = tl.program_id(0) # 获取程序 ID

    # Determine if this is main or extra processing # 判断是处理主缓存还是额外缓存
    is_main_pid = pid < num_main_pids # 是否为主进程

    # Select parameters based on pid # 根据进程 ID 选择参数
    if is_main_pid: # 如果是主进程
        local_pid = pid # 本地进程 ID
        topk = topk_main # topk 值
        k_offset = 0 # K 偏移量为0
        num_tk = total_tokens * topk_main # 总 (令牌×topk) 对数
        KV_Cache = KV_Cache_Main # KV 缓存
        Indices = Indices_Main # 索引
        TopkLength = TopkLength_Main # TopkLength
        block_size = block_size_main # 块大小
        stride_kv_block = stride_kv_block_main # KV 块步长
        stride_idx_t = stride_idx_t_main # 索引令牌步长
        stride_idx_k = stride_idx_k_main # 索引 K 步长
    else: # 如果是额外进程
        local_pid = pid - num_main_pids # 本地进程 ID
        topk = topk_extra # topk 值
        k_offset = topk_main # K 偏移量为主 topk
        num_tk = total_tokens * topk_extra # 总 (令牌×topk) 对数
        KV_Cache = KV_Cache_Extra # KV 缓存
        Indices = Indices_Extra # 索引
        TopkLength = TopkLength_Extra # TopkLength
        block_size = block_size_extra # 块大小
        stride_kv_block = stride_kv_block_extra # KV 块步长
        stride_idx_t = stride_idx_t_extra # 索引令牌步长
        stride_idx_k = stride_idx_k_extra # 索引 K 步长

    # Compute element indices for this block # 计算本块的元素索引
    offs_tk = local_pid * BLOCK_TK + tl.arange(0, BLOCK_TK) # 计算偏移量
    mask_tk = offs_tk < num_tk # 越界掩码

    t_idx = offs_tk // topk # 令牌索引
    k_idx = offs_tk % topk # topk 索引

    # Load indices # 加载索引
    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k # 索引指针
    indices = tl.load(idx_ptrs, mask=mask_tk, other=-1) # 加载索引值

    is_invalid = indices == -1 # 判断是否为无效索引

    # Handle topk_length - need to handle both cases # 处理 topk_length - 需要处理两种情况
    batch_idx = t_idx // s_q # 批次索引
    if is_main_pid: # 如果是主进程
        if HAS_TOPK_LENGTH_MAIN: # 如果主缓存有 topk_length
            topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk) # 加载 topk 长度
            is_invalid = is_invalid | (k_idx >= topk_len) # 更新无效标记
    else: # 如果是额外进程
        if HAS_TOPK_LENGTH_EXTRA: # 如果额外缓存有 topk_length
            topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk) # 加载 topk 长度
            is_invalid = is_invalid | (k_idx >= topk_len) # 更新无效标记

    # Store mask # 存储掩码
    mask_out_ptrs = ( # 输出掩码指针
        OutputMask + t_idx * stride_mask_t + (k_idx + k_offset) * stride_mask_k # 加上 K 偏移量
    )
    tl.store(mask_out_ptrs, is_invalid, mask=mask_tk) # 存储无效掩码

    valid_mask = mask_tk & ~is_invalid # 有效掩码
    indices_clamped = tl.maximum(indices, 0) # 将索引限制为非负值

    block_idx = indices_clamped // block_size # 计算块索引
    offset_in_block = indices_clamped % block_size # 计算块内偏移

    block_idx_64 = block_idx.to(tl.int64) # 转换为 int64
    offset_in_block_64 = offset_in_block.to(tl.int64) # 转换为 int64

    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block # KV 缓存块基地址

    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA # nope+rope 的偏移量
    scale_base_offset = ( # 缩放因子的基础偏移量
        block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE # 块数据之后紧跟缩放数据
    )

    t_idx_64 = t_idx.to(tl.int64) # 令牌索引转 int64
    k_idx_64 = k_idx.to(tl.int64) # topk 索引转 int64
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64) # 输出令牌步长转 int64
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64) # 输出 K 步长转 int64
    out_base_ptrs = ( # 输出基地址指针
        OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64 # 加上 K 偏移
    )

    # Load all 7 scales # 一次加载所有 7 个缩放因子
    scale_ptrs_0 = kv_block_base + scale_base_offset # 缩放因子指针
    scale_uint8_0 = tl.load(scale_ptrs_0, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子0
    scale_uint8_1 = tl.load(scale_ptrs_0 + 1, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子1
    scale_uint8_2 = tl.load(scale_ptrs_0 + 2, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子2
    scale_uint8_3 = tl.load(scale_ptrs_0 + 3, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子3
    scale_uint8_4 = tl.load(scale_ptrs_0 + 4, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子4
    scale_uint8_5 = tl.load(scale_ptrs_0 + 5, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子5
    scale_uint8_6 = tl.load(scale_ptrs_0 + 6, mask=valid_mask, other=127).to(tl.uint8) # 加载缩放因子6

    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子0 转 bf16
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子1 转 bf16
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子2 转 bf16
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子3 转 bf16
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子4 转 bf16
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子5 转 bf16
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16) # 缩放因子6 转 bf16
    # Pre-compute 2D versions for tile processing # 预计算 2D 版本用于 tile 处理
    scale_2d_0 = scale_bf16_0[:, None] # 缩放因子0 扩展为2D
    scale_2d_1 = scale_bf16_1[:, None] # 缩放因子1 扩展为2D
    scale_2d_2 = scale_bf16_2[:, None] # 缩放因子2 扩展为2D
    scale_2d_3 = scale_bf16_3[:, None] # 缩放因子3 扩展为2D
    scale_2d_4 = scale_bf16_4[:, None] # 缩放因子4 扩展为2D
    scale_2d_5 = scale_bf16_5[:, None] # 缩放因子5 扩展为2D
    scale_2d_6 = scale_bf16_6[:, None] # 缩放因子6 扩展为2D

    offs_d = tl.arange(0, TILE_SIZE) # 维度偏移量

    # Pre-compute base pointers for optimization # 预计算基础指针以优化性能
    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None] # tile 基地址
    out_base = out_base_ptrs[:, None] # 输出基地址
    valid_mask_2d = valid_mask[:, None] # 2D 有效掩码
    is_invalid_2d = is_invalid[:, None] # 2D 无效标记
    mask_tk_2d = mask_tk[:, None] # 2D 越界掩码

    # Process tile 0 # 处理 tile 0
    nope_ptrs = tile_base + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载 uint8 数据
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换为 FP8
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 转换为 BF16
    dequant = nope_bf16 * scale_2d_0 # 反量化
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效位置填零
    out_ptrs = out_base + offs_d[None, :] * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储

    # Process tile 1 # 处理 tile 1
    tile_start_1 = TILE_SIZE # tile 1 起始偏移
    nope_ptrs = tile_base + tile_start_1 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 类型转换
    dequant = nope_bf16 * scale_2d_1 # 反量化
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效填零
    out_ptrs = out_base + (tile_start_1 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储

    # Process tile 2 # 处理 tile 2
    tile_start_2 = 2 * TILE_SIZE # tile 2 起始偏移
    nope_ptrs = tile_base + tile_start_2 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 类型转换
    dequant = nope_bf16 * scale_2d_2 # 反量化
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效填零
    out_ptrs = out_base + (tile_start_2 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储

    # Process tile 3 # 处理 tile 3
    tile_start_3 = 3 * TILE_SIZE # tile 3 起始偏移
    nope_ptrs = tile_base + tile_start_3 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 类型转换
    dequant = nope_bf16 * scale_2d_3 # 反量化
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效填零
    out_ptrs = out_base + (tile_start_3 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储

    # Process tile 4 # 处理 tile 4
    tile_start_4 = 4 * TILE_SIZE # tile 4 起始偏移
    nope_ptrs = tile_base + tile_start_4 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 类型转换
    dequant = nope_bf16 * scale_2d_4 # 反量化
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效填零
    out_ptrs = out_base + (tile_start_4 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储

    # Process tile 5 # 处理 tile 5
    tile_start_5 = 5 * TILE_SIZE # tile 5 起始偏移
    nope_ptrs = tile_base + tile_start_5 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 类型转换
    dequant = nope_bf16 * scale_2d_5 # 反量化
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效填零
    out_ptrs = out_base + (tile_start_5 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储

    # Process tile 6 # 处理 tile 6
    tile_start_6 = 6 * TILE_SIZE # tile 6 起始偏移
    nope_ptrs = tile_base + tile_start_6 + offs_d[None, :] # nope 数据指针
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0) # 加载
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True) # 位转换
    nope_bf16 = nope_fp8.to(tl.bfloat16) # 类型转换
    dequant = nope_bf16 * scale_2d_6 # 反量化
    dequant = tl.where(is_invalid_2d, 0.0, dequant) # 无效填零
    out_ptrs = out_base + (tile_start_6 + offs_d[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, dequant, mask=mask_tk_2d) # 存储

    # Process rope # 处理旋转位置编码
    offs_rope = tl.arange(0, D_ROPE) # RoPE 维度偏移
    rope_byte_start = D_NOPE # RoPE 起始字节
    rope_lo_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 # 低字节指针
    rope_hi_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 + 1 # 高字节指针
    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16) # 加载低字节
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16) # 加载高字节
    rope_uint16 = rope_lo | (rope_hi << 8) # 合并为 uint16
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True) # 位转换为 BF16
    rope_bf16 = tl.where(is_invalid_2d, 0.0, rope_bf16) # 无效填零
    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d # 输出指针
    tl.store(out_ptrs, rope_bf16, mask=mask_tk_2d) # 存储


def _prepare_kv_cache_flat(kv_cache): # 准备 KV 缓存的展平版本
    """Helper to prepare KV cache for gather operations. # 辅助函数：为 gather 操作准备 KV 缓存

    Returns: (kv_flat, num_blocks, stride_kv_block) # 返回：(展平的KV, 块数, 块步长)
    """
    kv_uint8 = kv_cache.view(torch.uint8) # 视为 uint8
    num_blocks = kv_cache.shape[0] # 块数量
    bytes_per_block = kv_uint8.shape[1] * kv_uint8.shape[2] * kv_uint8.shape[3] # 每块字节数
    kv_flat = kv_uint8.reshape(num_blocks, bytes_per_block) # 展平为 2D
    stride_kv_block = kv_uint8.stride(0) # 获取块步长
    return kv_flat, num_blocks, stride_kv_block # 返回展平结果


def _launch_gather_dequant_one_dsv4( # 启动单个 KV 缓存的 gather+反量化内核
    kv_flat, # 展平的 KV 缓存
    indices, # 稀疏索引
    topk_length_tensor, # topk 长度张量
    output_kv, # 输出 KV
    output_mask, # 输出掩码
    total_tokens, # 总令牌数
    topk, # topk 值
    num_blocks, # 块数量
    block_size, # 块大小
    k_offset, # K 维偏移量
    s_q, # 每批次序列长度
    stride_kv_block, # KV 块步长
    stride_idx_t, # 索引令牌步长
    stride_idx_k, # 索引 K 步长
    stride_out_t, # 输出令牌步长
    stride_out_k, # 输出 K 步长
    stride_out_d, # 输出维度步长
    stride_mask_t, # 掩码令牌步长
    stride_mask_k, # 掩码 K 步长
    has_topk_length, # 是否有 topk_length
):
    """Helper to launch gather+dequant kernel for one KV cache (main or extra). # 辅助函数：为一个 KV 缓存（主或额外）启动 gather+反量化内核

    This eliminates code duplication between main and extra kernel launches # 消除主和额外内核启动之间的代码重复
    in the two-kernel path of fused_gather_dequant_fp8_dsv4. # 在 fused_gather_dequant_fp8_dsv4 的双内核路径中
    """
    total_elements = total_tokens * topk # 总元素数

    if total_elements < DSV4_USE_FIXED_KERNEL_THRESHOLD: # 小负载：使用固定配置内核
        grid = (triton.cdiv(total_elements, 128),) # 网格大小
        _gather_dequant_dsv4_kernel_fixed_128[grid]( # 启动固定 BLOCK_TK=128 内核
            kv_flat, # 展平的 KV 缓存
            indices, # 索引
            topk_length_tensor, # topk 长度
            output_kv, # 输出 KV
            output_mask, # 输出掩码
            total_tokens, # 总令牌数
            _bucket_total_tokens(total_tokens), # 分桶令牌数
            topk, # topk 值
            num_blocks, # 块数量
            block_size, # 块大小
            k_offset, # K 偏移
            s_q, # 序列长度
            stride_kv_block, # KV 块步长
            stride_idx_t, # 索引令牌步长
            stride_idx_k, # 索引 K 步长
            stride_out_t, # 输出令牌步长
            stride_out_k, # 输出 K 步长
            stride_out_d, # 输出维度步长
            stride_mask_t, # 掩码令牌步长
            stride_mask_k, # 掩码 K 步长
            D_NOPE=DSV4_D_NOPE, # 非旋转维度
            D_ROPE=DSV4_D_ROPE, # 旋转维度
            BYTES_PER_TOKEN_DATA=DSV4_BYTES_PER_TOKEN_DATA, # 每令牌数据字节数
            BYTES_PER_TOKEN_SCALE=DSV4_BYTES_PER_TOKEN_SCALE, # 每令牌缩放字节数
            TILE_SIZE=DSV4_TILE_SIZE, # tile 大小
            HAS_TOPK_LENGTH=has_topk_length, # 是否有 topk_length
            num_warps=8, # 8个warp
            num_stages=2, # 2个阶段
        )
    else: # 大负载：使用自动调优内核
        workload_cat = _get_workload_size_category(total_tokens, topk) # 获取工作负载类别
        grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_TK"]),) # 动态网格大小
        _gather_dequant_dsv4_kernel[grid]( # 启动自动调优内核
            kv_flat, # 展平的 KV 缓存
            indices, # 索引
            topk_length_tensor, # topk 长度
            output_kv, # 输出 KV
            output_mask, # 输出掩码
            total_tokens, # 总令牌数
            _bucket_total_tokens(total_tokens), # 分桶令牌数
            topk, # topk 值
            num_blocks, # 块数量
            block_size, # 块大小
            workload_cat, # 工作负载类别
            k_offset, # K 偏移
            s_q, # 序列长度
            stride_kv_block, # KV 块步长
            stride_idx_t, # 索引令牌步长
            stride_idx_k, # 索引 K 步长
            stride_out_t, # 输出令牌步长
            stride_out_k, # 输出 K 步长
            stride_out_d, # 输出维度步长
            stride_mask_t, # 掩码令牌步长
            stride_mask_k, # 掩码 K 步长
            D_NOPE=DSV4_D_NOPE, # 非旋转维度
            D_ROPE=DSV4_D_ROPE, # 旋转维度
            BYTES_PER_TOKEN_DATA=DSV4_BYTES_PER_TOKEN_DATA, # 每令牌数据字节数
            BYTES_PER_TOKEN_SCALE=DSV4_BYTES_PER_TOKEN_SCALE, # 每令牌缩放字节数
            TILE_SIZE=DSV4_TILE_SIZE, # tile 大小
            HAS_TOPK_LENGTH=has_topk_length, # 是否有 topk_length
        )


def truly_fused_gather_dequant_fp8_dsv4( # DSV4 真正融合的 gather+反量化函数（一维网格）
    kv_cache_main, # 主 KV 缓存
    indices_main, # 主索引
    block_size_main, # 主块大小
    topk_length_main, # 主 topk 长度
    kv_cache_extra, # 额外 KV 缓存
    indices_extra, # 额外索引
    block_size_extra, # 额外块大小
    topk_length_extra, # 额外 topk 长度
    output_kv, # 输出 KV
    output_mask, # 输出掩码
    s_q=1, # 每批次序列长度
):
    """Truly fused DSV4 gather - single kernel launch with 1D grid (no empty blocks).""" # 真正融合的 DSV4 gather - 单次内核启动，一维网格（无空块）
    total_tokens, topk_main = indices_main.shape # 获取总令牌数和主 topk
    topk_extra = indices_extra.shape[1] # 获取额外 topk
    b = total_tokens // s_q  # batch size # 批次大小

    kv_flat_main, num_blocks_main, stride_kv_block_main = _prepare_kv_cache_flat( # 准备主缓存
        kv_cache_main # 主 KV 缓存
    )
    kv_flat_extra, num_blocks_extra, stride_kv_block_extra = _prepare_kv_cache_flat( # 准备额外缓存
        kv_cache_extra # 额外 KV 缓存
    )

    has_topk_length_main = topk_length_main is not None # 主缓存是否有 topk_length
    has_topk_length_extra = topk_length_extra is not None # 额外缓存是否有 topk_length

    # Always use int32 tensors for topk_length to avoid type mismatch in Triton # 始终使用 int32 张量作为 topk_length，避免 Triton 中的类型不匹配
    if has_topk_length_main: # 如果有主 topk_length
        topk_length_main_tensor = topk_length_main # 直接使用
    else: # 否则创建填充张量
        topk_length_main_tensor = torch.full( # 创建填充张量
            (b,), topk_main, dtype=torch.int32, device=indices_main.device # 填充为主 topk 值
        )

    if has_topk_length_extra: # 如果有额外 topk_length
        topk_length_extra_tensor = topk_length_extra # 直接使用
    else: # 否则创建填充张量
        topk_length_extra_tensor = torch.full( # 创建填充张量
            (b,), topk_extra, dtype=torch.int32, device=indices_extra.device # 填充为额外 topk 值
        )

    stride_idx_t_main, stride_idx_k_main = indices_main.stride(0), indices_main.stride( # 主索引步长
        1
    )
    stride_idx_t_extra, stride_idx_k_extra = indices_extra.stride( # 额外索引步长
        0
    ), indices_extra.stride(1)
    stride_out_t, stride_out_k, stride_out_d = ( # 输出步长
        output_kv.stride(0), # 令牌步长
        output_kv.stride(1), # K 步长
        output_kv.stride(2), # 维度步长
    )
    stride_mask_t, stride_mask_k = output_mask.stride(0), output_mask.stride(1) # 掩码步长

    BLOCK_TK = 128 # 固定 BLOCK_TK 为 128

    # Calculate grid sizes - 1D grid with exact number of needed blocks # 计算网格大小 - 一维网格，精确计算所需块数
    num_elements_main = total_tokens * topk_main # 主缓存总元素数
    num_elements_extra = total_tokens * topk_extra # 额外缓存总元素数
    num_main_pids = triton.cdiv(num_elements_main, BLOCK_TK) # 主进程数
    num_extra_pids = triton.cdiv(num_elements_extra, BLOCK_TK) # 额外进程数

    # 1D grid: (num_main_pids + num_extra_pids,) - no empty blocks! # 一维网格：(主进程数 + 额外进程数) - 无空块！
    grid = (num_main_pids + num_extra_pids,) # 网格大小

    _gather_dequant_dsv4_1d_fused_kernel[grid]( # 启动一维融合内核
        kv_flat_main, # 主展平缓存
        indices_main, # 主索引
        topk_length_main_tensor, # 主 topk 长度
        kv_flat_extra, # 额外展平缓存
        indices_extra, # 额外索引
        topk_length_extra_tensor, # 额外 topk 长度
        output_kv, # 输出 KV
        output_mask, # 输出掩码
        total_tokens, # 总令牌数
        topk_main, # 主 topk
        topk_extra, # 额外 topk
        num_blocks_main, # 主块数量
        num_blocks_extra, # 额外块数量
        block_size_main, # 主块大小
        block_size_extra, # 额外块大小
        s_q, # 序列长度
        stride_kv_block_main, # 主 KV 块步长
        stride_idx_t_main, # 主索引令牌步长
        stride_idx_k_main, # 主索引 K 步长
        stride_kv_block_extra, # 额外 KV 块步长
        stride_idx_t_extra, # 额外索引令牌步长
        stride_idx_k_extra, # 额外索引 K 步长
        stride_out_t, # 输出令牌步长
        stride_out_k, # 输出 K 步长
        stride_out_d, # 输出维度步长
        stride_mask_t, # 掩码令牌步长
        stride_mask_k, # 掩码 K 步长
        num_main_pids, # 主进程数
        BLOCK_TK=BLOCK_TK, # 每块元素数
        D_NOPE=DSV4_D_NOPE, # 非旋转维度
        D_ROPE=DSV4_D_ROPE, # 旋转维度
        BYTES_PER_TOKEN_DATA=DSV4_BYTES_PER_TOKEN_DATA, # 每令牌数据字节数
        BYTES_PER_TOKEN_SCALE=DSV4_BYTES_PER_TOKEN_SCALE, # 每令牌缩放字节数
        TILE_SIZE=DSV4_TILE_SIZE, # tile 大小
        HAS_TOPK_LENGTH_MAIN=has_topk_length_main, # 主是否有 topk_length
        HAS_TOPK_LENGTH_EXTRA=has_topk_length_extra, # 额外是否有 topk_length
        num_warps=8, # 8个warp
        num_stages=2, # 2个阶段
    )
    return True # 返回成功


def fused_gather_dequant_fp8_dsv4( # DSV4 融合 gather+反量化函数（自动选择融合或双内核路径）
    kv_cache_main, # 主 KV 缓存
    indices_main, # 主索引
    block_size_main, # 主块大小
    topk_length_main, # 主 topk 长度
    kv_cache_extra, # 额外 KV 缓存
    indices_extra, # 额外索引
    block_size_extra, # 额外块大小
    topk_length_extra, # 额外 topk 长度
    output_kv, # 输出 KV
    output_mask, # 输出掩码
    s_q=1, # 每批次序列长度
):
    """Fused DSV4 gather - uses 1D fused kernel for small workloads, two kernels for large.""" # 融合 DSV4 gather - 小负载使用一维融合内核，大负载使用双内核
    has_topk_length_main = topk_length_main is not None # 主缓存是否有 topk_length
    has_topk_length_extra = topk_length_extra is not None # 额外缓存是否有 topk_length

    total_tokens, topk_main = indices_main.shape # 获取总令牌数和主 topk
    topk_extra = indices_extra.shape[1] # 获取额外 topk
    total_elements = total_tokens * (topk_main + topk_extra) # 总元素数

    # Use fused 2D grid kernel only for small workloads where kernel launch overhead matters # 仅在小负载（内核启动开销重要）时使用融合 2D 网格内核
    # For large workloads, the two-kernel approach is more efficient # 大负载时双内核方法更高效
    USE_FUSED_THRESHOLD = DSV4_USE_FUSED_THRESHOLD # 使用融合内核的阈值

    # IMPORTANT: Disable fused kernel when topk_length settings differ between main and extra # 重要：当主和额外的 topk_length 设置不同时，禁用融合内核
    # The 1D fused kernel has issues with runtime conditional handling when # 一维融合内核在运行时条件处理时存在问题，当
    # HAS_TOPK_LENGTH_MAIN != HAS_TOPK_LENGTH_EXTRA, causing incorrect results in extra part. # HAS_TOPK_LENGTH_MAIN != HAS_TOPK_LENGTH_EXTRA 时，导致额外部分的结果不正确
    # Only use fused kernel when both have same topk_length setting. # 仅在两者具有相同的 topk_length 设置时使用融合内核
    topk_length_settings_match = has_topk_length_main == has_topk_length_extra # topk_length 设置是否匹配
    use_fused = total_elements < USE_FUSED_THRESHOLD and topk_length_settings_match # 是否使用融合内核

    if use_fused: # 使用融合内核
        return truly_fused_gather_dequant_fp8_dsv4( # 调用一维融合 gather 函数
            kv_cache_main, # 主 KV 缓存
            indices_main, # 主索引
            block_size_main, # 主块大小
            topk_length_main, # 主 topk 长度
            kv_cache_extra, # 额外 KV 缓存
            indices_extra, # 额外索引
            block_size_extra, # 额外块大小
            topk_length_extra, # 额外 topk 长度
            output_kv, # 输出 KV
            output_mask, # 输出掩码
            s_q, # 序列长度
        )

    # Use original two-kernel approach for large workloads # 大负载使用原始的双内核方法
    kv_flat_main, num_blocks_main, stride_kv_block_main = _prepare_kv_cache_flat( # 准备主缓存
        kv_cache_main # 主 KV 缓存
    )
    kv_flat_extra, num_blocks_extra, stride_kv_block_extra = _prepare_kv_cache_flat( # 准备额外缓存
        kv_cache_extra # 额外 KV 缓存
    )

    topk_length_main_tensor = ( # 主 topk 长度张量
        topk_length_main if has_topk_length_main else output_mask[:1, 0] # 使用实际值或占位
    )
    topk_length_extra_tensor = ( # 额外 topk 长度张量
        topk_length_extra if has_topk_length_extra else output_mask[:1, 0] # 使用实际值或占位
    )

    stride_idx_t_main, stride_idx_k_main = indices_main.stride(0), indices_main.stride( # 主索引步长
        1
    )
    stride_idx_t_extra, stride_idx_k_extra = indices_extra.stride( # 额外索引步长
        0
    ), indices_extra.stride(1)
    stride_out_t, stride_out_k, stride_out_d = ( # 输出步长
        output_kv.stride(0), # 令牌步长
        output_kv.stride(1), # K 步长
        output_kv.stride(2), # 维度步长
    )
    stride_mask_t, stride_mask_k = output_mask.stride(0), output_mask.stride(1) # 掩码步长

    # Launch main kernel # 启动主内核
    _launch_gather_dequant_one_dsv4( # 调用主缓存 gather+反量化
        kv_flat_main, # 主展平缓存
        indices_main, # 主索引
        topk_length_main_tensor, # 主 topk 长度
        output_kv, # 输出 KV
        output_mask, # 输出掩码
        total_tokens, # 总令牌数
        topk_main, # 主 topk
        num_blocks_main, # 主块数量
        block_size_main, # 主块大小
        0, # K 偏移为0
        s_q, # 序列长度
        stride_kv_block_main, # 主 KV 块步长
        stride_idx_t_main, # 主索引令牌步长
        stride_idx_k_main, # 主索引 K 步长
        stride_out_t, # 输出令牌步长
        stride_out_k, # 输出 K 步长
        stride_out_d, # 输出维度步长
        stride_mask_t, # 掩码令牌步长
        stride_mask_k, # 掩码 K 步长
        has_topk_length_main, # 主是否有 topk_length
    )

    # Launch extra kernel # 启动额外内核
    _launch_gather_dequant_one_dsv4( # 调用额外缓存 gather+反量化
        kv_flat_extra, # 额外展平缓存
        indices_extra, # 额外索引
        topk_length_extra_tensor, # 额外 topk 长度
        output_kv, # 输出 KV
        output_mask, # 输出掩码
        total_tokens, # 总令牌数
        topk_extra, # 额外 topk
        num_blocks_extra, # 额外块数量
        block_size_extra, # 额外块大小
        topk_main, # K 偏移为主 topk
        s_q, # 序列长度
        stride_kv_block_extra, # 额外 KV 块步长
        stride_idx_t_extra, # 额外索引令牌步长
        stride_idx_k_extra, # 额外索引 K 步长
        stride_out_t, # 输出令牌步长
        stride_out_k, # 输出 K 步长
        stride_out_d, # 输出维度步长
        stride_mask_t, # 掩码令牌步长
        stride_mask_k, # 掩码 K 步长
        has_topk_length_extra, # 额外是否有 topk_length
    )

    return True # 返回成功


def triton_sparse_attn_decode_dsv4( # DSV4 稀疏注意力解码主入口函数
    q: torch.Tensor, # 查询张量
    kv_scope, # KV 作用域
    extra_kv_scope, # 额外 KV 作用域
    sm_scale: float, # softmax 缩放因子
    d_v: int = 512, # 值维度，默认512
    attn_sink: Optional[torch.Tensor] = None, # 可选的注意力汇聚值
) -> Tuple[torch.Tensor, torch.Tensor]: # 返回输出和 LSE
    """Sparse attention decode for DSV4 (d_qk=512).""" # DSV4 (d_qk=512) 的稀疏注意力解码
    assert kv_scope is not None # 确保 KV 作用域不为空
    b, s_q, h_q, d_qk = q.shape # 获取查询张量的维度
    assert d_qk == DSV4_D_QK, f"Expected d_qk={DSV4_D_QK} for DSV4, got {d_qk}" # 验证 QK 维度
    total_tokens = b * s_q # 总令牌数

    topk_main = kv_scope.indices_in_kvcache.size(-1) # 主 topk 数量
    topk_extra = ( # 额外 topk 数量
        extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0 # 无额外作用域时为0
    )
    total_topk = topk_main + topk_extra # 总 topk 数量

    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk) # 计算令牌范围

    if len(token_ranges) == 1: # 如果不需要分块
        return _triton_sparse_attn_decode_dsv4_impl( # 直接调用实现函数
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
        )

    outputs = [] # 输出列表
    lses = [] # LSE 列表

    for start_t, end_t in token_ranges: # 遍历每个令牌范围
        chunk_tokens = end_t - start_t # 当前块的令牌数
        q_chunk = q.reshape(total_tokens, h_q, d_qk)[start_t:end_t] # 切片查询
        q_input = q_chunk.reshape(chunk_tokens, 1, h_q, d_qk) # 重塑查询
        chunk_kv_scope = slice_kv_scope_for_tokens(kv_scope, start_t, end_t, s_q) # 切片主 KV 作用域
        chunk_extra_kv_scope = slice_kv_scope_for_tokens( # 切片额外 KV 作用域
            extra_kv_scope, start_t, end_t, s_q
        )

        chunk_out, chunk_lse = _triton_sparse_attn_decode_dsv4_impl( # 处理当前块
            q_input, chunk_kv_scope, chunk_extra_kv_scope, sm_scale, d_v, attn_sink
        )

        outputs.append(chunk_out.reshape(chunk_tokens, h_q, d_v)) # 收集输出
        lses.append(chunk_lse.reshape(chunk_tokens, h_q)) # 收集 LSE

    output = torch.cat(outputs, dim=0).reshape(b, s_q, h_q, d_v) # 拼接并重塑输出
    lse = torch.cat(lses, dim=0).reshape(b, s_q, h_q).transpose(1, 2) # 拼接并重塑 LSE

    return output, lse # 返回输出和 LSE


def _triton_sparse_attn_decode_dsv4_impl( # DSV4 稀疏注意力解码的内部实现
    q: torch.Tensor, # 查询张量
    kv_scope, # KV 作用域
    extra_kv_scope, # 额外 KV 作用域
    sm_scale: float, # softmax 缩放因子
    d_v: int = 512, # 值维度
    attn_sink: Optional[torch.Tensor] = None, # 可选的注意力汇聚值
) -> Tuple[torch.Tensor, torch.Tensor]: # 返回输出和 LSE
    """Internal implementation of sparse attention decode for DSV4. # DSV4 稀疏注意力解码的内部实现

    Assumes KV cache is always FP8 quantized (blocked_k_quantized is not None). # 假设 KV 缓存始终是 FP8 量化的（blocked_k_quantized 不为 None）
    """
    assert kv_scope is not None # 确保 KV 作用域不为空
    b, s_q, h_q, d_qk = q.shape # 获取查询维度
    total_tokens = b * s_q # 总令牌数

    topk_main = kv_scope.indices_in_kvcache.size(-1) # 主 topk 数量
    topk_extra = ( # 额外 topk 数量
        extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0 # 无额外时为0
    )
    total_topk = topk_main + topk_extra # 总 topk 数量

    gathered_kv = torch.empty( # 分配收集后的 KV 缓冲区
        total_tokens, total_topk, d_qk, dtype=torch.bfloat16, device=q.device # BF16 类型
    )
    invalid_mask = torch.empty( # 分配无效掩码缓冲区
        total_tokens, total_topk, dtype=torch.bool, device=q.device # bool 类型
    )

    block_size_main = kv_scope.blocked_k.shape[1] # 主块大小
    indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main) # 重塑主索引

    if extra_kv_scope is not None: # 有额外 KV 作用域
        # Fused gather for both main and extra scope # 融合收集主和额外作用域
        block_size_extra = extra_kv_scope.blocked_k.shape[1] # 额外块大小
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape( # 重塑额外索引
            total_tokens, topk_extra
        )
        fused_gather_dequant_fp8_dsv4( # 调用融合 gather+反量化
            kv_scope.blocked_k_quantized, # 主量化 KV
            indices_main, # 主索引
            block_size_main, # 主块大小
            kv_scope.topk_length, # 主 topk 长度
            extra_kv_scope.blocked_k_quantized, # 额外量化 KV
            indices_extra, # 额外索引
            block_size_extra, # 额外块大小
            extra_kv_scope.topk_length, # 额外 topk 长度
            gathered_kv, # 输出 KV
            invalid_mask, # 输出掩码
            s_q, # 序列长度
        )
    else: # 仅有主 KV 作用域
        # Single gather for main scope only # 仅为主作用域收集
        gather_dequant_fp8_dsv4( # 调用单作用域 gather+反量化
            kv_scope.blocked_k_quantized, # 量化 KV
            indices_main, # 索引
            block_size_main, # 块大小
            gathered_kv, # 输出 KV
            invalid_mask, # 输出掩码
            0, # K 偏移为0
            kv_scope.topk_length, # topk 长度
            s_q, # 序列长度
        )

    q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk) # 转换查询为 BF16 并重塑

    if not q_reshaped.is_contiguous(): # 确保连续
        q_reshaped = q_reshaped.contiguous() # 使其连续

    # Use splitk for large topk to reduce register pressure # 大 topk 使用 splitk 以减少寄存器压力
    if total_topk >= 8192: # 总 topk >= 8192
        # Adaptive split_k selection for optimal performance # 自适应 split_k 选择以优化性能
        # split_k=3 is optimal for topk >= 16384 based on benchmarking # 基准测试表明 topk >= 16384 时 split_k=3 最优
        if total_topk >= 16384: # 总 topk >= 16384
            split_k = 3 # 使用 3 路分割
        else: # 8192 <= 总 topk < 16384
            split_k = 2 # 使用 2 路分割
        output, lse = run_splitk_unified_attention( # 运行 splitk 统一注意力
            q_reshaped, # 查询
            gathered_kv, # 收集的 KV
            invalid_mask, # 无效掩码
            d_v, # 值维度
            sm_scale, # softmax 缩放
            total_tokens, # 总令牌数
            h_q, # 头数
            total_topk, # 总 topk
            d_qk, # QK 维度
            attn_sink=attn_sink, # 注意力汇聚
            split_k=split_k, # 分割数
        )
    elif total_topk <= 65536: # 总 topk <= 65536
        output, lse = run_unified_attention( # 运行统一注意力
            q_reshaped, # 查询
            gathered_kv, # 收集的 KV
            invalid_mask, # 无效掩码
            d_v, # 值维度
            sm_scale, # softmax 缩放
            total_tokens, # 总令牌数
            h_q, # 头数
            total_topk, # 总 topk
            d_qk, # QK 维度
            attn_sink=attn_sink, # 注意力汇聚
        )
    else: # 总 topk > 65536
        output, lse = run_chunked_attention_triton( # 运行分块注意力
            q_reshaped, # 查询
            gathered_kv, # 收集的 KV
            invalid_mask, # 无效掩码
            d_v, # 值维度
            sm_scale, # softmax 缩放
            total_tokens, # 总令牌数
            h_q, # 头数
            total_topk, # 总 topk
            d_qk, # QK 维度
            attn_sink=attn_sink, # 注意力汇聚
            chunk_size=32768, # 块大小
        )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2) # 重塑并返回
