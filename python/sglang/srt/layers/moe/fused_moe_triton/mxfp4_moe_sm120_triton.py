# 文件说明：SM120架构优化的Triton MXFP4 MoE内核模块
# 本模块实现了面向SM120（Blackwell）架构的MXFP4格式MoE计算，
# 使用融合Triton内核替代PyTorch回退实现，支持CUDA图捕获兼容。
# 包含FP4反量化查找表、逐slot GEMV内核、逐专家GEMM内核，
# 以及完整的MXFP4 MoE前向计算流程。

"""SM120-optimized Triton MXFP4 MoE kernel — CUDA graph compatible.
SM120优化的Triton MXFP4 MoE内核——兼容CUDA图捕获。

Replaces the PyTorch fallback (per-expert for-loop + full dequant + matmul)
with fused Triton kernels that:
用融合Triton内核替代PyTorch回退实现（逐专家for循环 + 完整反量化 + 矩阵乘法），这些内核：
1. Fuse FP4 dequant + GEMV (no intermediate BF16 weight materialization)
   融合FP4反量化 + GEMV（无中间BF16权重物化）
2. Process each (token, expert) slot independently — no data-dependent routing
   独立处理每个(令牌, 专家)slot——无数据依赖的路由
3. Respect SM120 shared memory constraint (99 KB/block)
   遵循SM120共享内存约束（99 KB/块）

CUDA graph compatibility:
CUDA图兼容性：
- No .unique(), .item(), .nonzero() — all routing is tensor-level
  无.unique()、.item()、.nonzero()调用——所有路由均为张量级操作
- Fixed grid dimensions (M*topk, N_blocks) per captured batch size
  每个捕获的批次大小使用固定的网格维度(M*topk, N_blocks)
- All control flow is static or within Triton kernels
  所有控制流为静态或在Triton内核内部

SM120 constraints:
SM120约束：
- SMEM: 99 KB/block (vs SM100 228 KB)
  共享内存：99 KB/块（对比SM100的228 KB）
- No TMEM/tcgen05 — uses mma.sync.aligned via Triton
  无TMEM/tcgen05——通过Triton使用mma.sync.aligned
- Max warps: 48/SM
  最大线程束：48/SM
- Registers: ~128/thread practical limit
  寄存器：~128/线程的实际限制
"""

import logging  # 导入日志模块
from typing import Optional  # 导入可选类型注解

import torch  # 导入PyTorch库
import triton  # 导入Triton库
import triton.language as tl  # 导入Triton语言模块

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器


@triton.jit  # Triton JIT编译装饰器
def _dequant_fp4_lut(nibble):  # FP4反量化查找表函数：将4位FP4 E2M1 nibble解码为float32
    """Decode a 4-bit FP4 E2M1 nibble to float32 using arithmetic."""
    """使用算术方法将4位FP4 E2M1 nibble解码为float32。"""
    sign_bit = (nibble >> 3) & 1  # 提取符号位（第3位）
    exp_bits = (nibble >> 1) & 3  # 提取指数位（第1-2位，2位）
    man_bit = nibble & 1  # 提取尾数位（第0位，1位）

    is_subnormal = exp_bits == 0  # 判断是否为次正规数（指数为0）
    mantissa = 1.0 + man_bit.to(tl.float32) * 0.5  # 计算正规数尾数：1 + man*0.5
    exponent = tl.math.exp2((exp_bits - 1).to(tl.float32))  # 计算指数：2^(exp-1)
    val = tl.where(is_subnormal, man_bit.to(tl.float32) * 0.5, mantissa * exponent)  # 次正规数用man*0.5，正规数用尾数*指数
    val = tl.where(sign_bit != 0, -val, val)  # 应用符号位
    return val  # 返回解码后的浮点值


# ── Per-slot GEMV kernel: processes one (token, expert) pair ──
# ── 逐slot GEMV内核：处理一个(令牌, 专家)对 ──


@triton.autotune(  # Triton自动调优装饰器，搜索最优配置
    configs=[  # 候选配置列表
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),  # 配置1：块大小64x64，4个线程束，2个流水线阶段
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),  # 配置2：块大小32x64
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),  # 配置3：块大小64x128
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),  # 配置4：块大小128x64，8个线程束
    ],
    key=["N", "K"],  # 自动调优的键维度（N和K变化时重新调优）
)
@triton.jit  # Triton JIT编译装饰器
def _mxfp4_slot_gemv_kernel(  # 逐slot的MXFP4融合反量化+GEMV内核
    # Pointers  # 指针参数
    A_ptr,  # [M_total, K] bf16 — source rows  # 输入激活矩阵指针
    B_packed_ptr,  # [E, N, K//2] uint8 — packed FP4 expert weights  # 打包的FP4专家权重指针
    B_scale_ptr,  # [E, N, K//32] float32 — weight scales  # 权重缩放因子指针
    C_ptr,  # [num_slots, N] bf16 — output  # 输出矩阵指针
    token_ids_ptr,  # [num_slots] int32 — which A row for each slot  # 每个slot对应的输入行索引
    expert_ids_ptr,  # [num_slots] int32 — which expert's B for each slot  # 每个slot对应的专家权重索引
    # Dimensions  # 维度参数
    N: tl.int32,  # 输出维度N
    K: tl.int32,  # 输入维度K
    # A strides  # A矩阵步幅
    stride_am: tl.int32,  # A矩阵行步幅
    # B strides (within an expert)  # B矩阵步幅（专家内部）
    stride_bn: tl.int32,  # B矩阵N方向步幅
    stride_bk2: tl.int32,  # B矩阵K//2方向步幅
    # B_scale strides (within an expert)  # B缩放步幅（专家内部）
    stride_bsn: tl.int32,  # 缩放因子N方向步幅
    stride_bsk32: tl.int32,  # 缩放因子K//32方向步幅
    # Expert strides (between experts)  # 专家间步幅
    expert_b_stride: tl.int64,  # 专家间B矩阵步幅
    expert_s_stride: tl.int64,  # 专家间缩放步幅
    # C strides  # C矩阵步幅
    stride_cm: tl.int32,  # C矩阵行步幅
    # Block sizes  # 块大小
    BLOCK_N: tl.constexpr,  # N方向块大小（编译时常量）
    BLOCK_K: tl.constexpr,  # K方向块大小（编译时常量）
):
    """Per-slot fused MXFP4 dequant + GEMV.
    逐slot的融合MXFP4反量化 + GEMV。

    Grid: (num_slots, cdiv(N, BLOCK_N))
    网格：(num_slots, cdiv(N, BLOCK_N))
    Each program computes one (token, expert) pair for a BLOCK_N slice of output.
    每个程序计算一个(令牌, 专家)对的BLOCK_N大小输出切片。
    """
    slot_id = tl.program_id(0)  # 获取当前slot ID
    n_block = tl.program_id(1)  # 获取当前N方向块ID

    token_id = tl.load(token_ids_ptr + slot_id).to(tl.int64)  # 加载当前slot对应的令牌ID
    expert_id = tl.load(expert_ids_ptr + slot_id).to(tl.int64)  # 加载当前slot对应的专家ID

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # 计算N方向的偏移量
    n_mask = offs_n < N  # 生成N方向的越界掩码

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)  # 初始化FP32累加器

    # Expert weight base pointers  # 专家权重基指针
    b_base = expert_id * expert_b_stride  # 计算当前专家的B矩阵基偏移
    s_base = expert_id * expert_s_stride  # 计算当前专家的缩放因子基偏移
    a_base = token_id * stride_am  # 计算当前令牌的A矩阵基偏移

    for k_start in range(0, K, BLOCK_K):  # 沿K维度分块循环
        # ── Load packed B: [BLOCK_N, BLOCK_K//2] ──
        # ── 加载打包的B：[BLOCK_N, BLOCK_K//2] ──
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)  # 计算K//2方向的偏移量
        b_mask = n_mask[:, None] & (offs_k2[None, :] < K // 2)  # 生成B矩阵的加载掩码
        b_packed = tl.load(  # 加载打包的FP4权重数据
            B_packed_ptr  # B矩阵基地址
            + b_base  # 加上专家偏移
            + offs_n[:, None] * stride_bn  # 加上N方向偏移
            + offs_k2[None, :] * stride_bk2,  # 加上K//2方向偏移
            mask=b_mask,  # 越界掩码
            other=0,  # 越界填充0
        )

        # ── FP4 dequant ──  # FP4反量化
        b_u8 = b_packed.to(tl.int32)  # 将uint8转为int32以便位操作
        val_lo = _dequant_fp4_lut(b_u8 & 0x0F)  # even K indices  # 解码低4位（偶数K索引）
        val_hi = _dequant_fp4_lut((b_u8 >> 4) & 0x0F)  # odd K indices  # 解码高4位（奇数K索引）

        # ── Load and apply scales: [BLOCK_N, BLOCK_K//2] ──
        # ── 加载并应用缩放因子：[BLOCK_N, BLOCK_K//2] ──
        group_ids = tl.arange(0, BLOCK_K // 2) // 16  # 32 values per group, 2 per byte  # 每32个值一组，每字节2个值
        s_mask = n_mask[:, None] & ((k_start // 32 + group_ids[None, :]) < K // 32)  # 生成缩放因子加载掩码
        scales = tl.load(  # 加载缩放因子
            B_scale_ptr  # 缩放因子基地址
            + s_base  # 加上专家偏移
            + offs_n[:, None] * stride_bsn  # 加上N方向偏移
            + (k_start // 32 + group_ids[None, :]) * stride_bsk32,  # 加上组方向偏移
            mask=s_mask,  # 越界掩码
            other=1.0,  # 越界填充1.0（保持值不变）
        )
        val_lo = val_lo * scales  # 对低4位解码值应用缩放
        val_hi = val_hi * scales  # 对高4位解码值应用缩放

        # ── Load A even/odd: [BLOCK_K//2] each ──
        # ── 加载A的偶数/奇数列：各[BLOCK_K//2] ──
        offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2  # 偶数K索引偏移
        offs_k_odd = offs_k_even + 1  # 奇数K索引偏移

        a_even = tl.load(  # 加载A的偶数列元素
            A_ptr + a_base + offs_k_even,  # A基地址 + 令牌偏移 + 偶数K偏移
            mask=offs_k_even < K,  # K方向越界掩码
            other=0.0,  # 越界填充0.0
        ).to(tl.float32)  # 转为FP32
        a_odd = tl.load(  # 加载A的奇数列元素
            A_ptr + a_base + offs_k_odd,  # A基地址 + 令牌偏移 + 奇数K偏移
            mask=offs_k_odd < K,  # K方向越界掩码
            other=0.0,  # 越界填充0.0
        ).to(tl.float32)  # 转为FP32

        # ── Dot product: acc[n] += sum_k(a_even[k]*B_lo[n,k] + a_odd[k]*B_hi[n,k]) ──
        # ── 点积：acc[n] += sum_k(a_even[k]*B_lo[n,k] + a_odd[k]*B_hi[n,k]) ──
        acc += tl.sum(a_even[None, :] * val_lo, axis=1)  # 累加偶数K部分的点积
        acc += tl.sum(a_odd[None, :] * val_hi, axis=1)  # 累加奇数K部分的点积

    # ── Store output ──  # ── 存储输出 ──
    tl.store(  # 将累加结果写入输出矩阵
        C_ptr + slot_id * stride_cm + offs_n,  # 输出地址 = 基地址 + slot偏移 + N偏移
        acc.to(tl.bfloat16),  # 将FP32结果转为BF16
        mask=n_mask,  # N方向越界掩码
    )


# ── Legacy per-expert GEMM kernel (kept for benchmarking) ──
# ── 遗留的逐专家GEMM内核（保留用于基准测试） ──


@triton.autotune(  # Triton自动调优装饰器
    configs=[  # 候选配置列表
        triton.Config(  # 配置1
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2  # 32x64x64，4线程束，2流水线阶段
        ),
        triton.Config(  # 配置2
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2  # 32x32x64
        ),
        triton.Config(  # 配置3
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2  # 64x32x64
        ),
        triton.Config(  # 配置4
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2  # 64x64x32，8线程束
        ),
        triton.Config(  # 配置5
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2  # 16x64x64
        ),
    ],
    key=["M", "N", "K"],  # 自动调优的键维度
)
@triton.jit  # Triton JIT编译装饰器
def _mxfp4_gemm_kernel(  # 逐专家的MXFP4融合反量化+GEMM内核
    # Pointers  # 指针参数
    A_ptr,  # [M, K] bf16 activation  # 输入激活矩阵指针
    B_packed_ptr,  # [N, K//2] uint8 packed FP4  # 打包的FP4权重指针
    B_scale_ptr,  # [N, K//32] float32 scales  # 缩放因子指针
    C_ptr,  # [M, N] bf16 output  # 输出矩阵指针
    # Dimensions  # 维度参数
    M,  # 令牌数
    N,  # 输出维度
    K,  # 输入维度
    # Strides  # 步幅参数
    stride_am,  # A矩阵M方向步幅
    stride_ak,  # A矩阵K方向步幅
    stride_bn,  # B矩阵N方向步幅
    stride_bk2,  # B矩阵K//2方向步幅
    stride_bsn,  # 缩放因子N方向步幅
    stride_bsk32,  # 缩放因子K//32方向步幅
    stride_cm,  # C矩阵M方向步幅
    stride_cn,  # C矩阵N方向步幅
    # Constexprs  # 编译时常量
    BLOCK_M: tl.constexpr,  # M方向块大小
    BLOCK_N: tl.constexpr,  # N方向块大小
    BLOCK_K: tl.constexpr,  # K方向块大小
):
    """Fused MXFP4 dequant + GEMM: C = A @ dequant(B_packed, B_scale).T"""
    """融合MXFP4反量化 + GEMM：C = A @ dequant(B_packed, B_scale).T"""
    pid_m = tl.program_id(0)  # 获取M方向程序ID
    pid_n = tl.program_id(1)  # 获取N方向程序ID

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 计算M方向偏移量
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 计算N方向偏移量

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # 初始化FP32累加器

    for k_start in range(0, K, BLOCK_K):  # 沿K维度分块循环
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)  # 计算K//2方向偏移量
        b_mask = (offs_n[:, None] < N) & (offs_k2[None, :] < K // 2)  # 生成B矩阵加载掩码
        b_packed = tl.load(  # 加载打包的FP4权重
            B_packed_ptr + offs_n[:, None] * stride_bn + offs_k2[None, :] * stride_bk2,  # 计算加载地址
            mask=b_mask,  # 越界掩码
            other=0,  # 越界填充0
        )

        b_u8 = b_packed.to(tl.int32)  # 将uint8转为int32
        val_lo = _dequant_fp4_lut(b_u8 & 0x0F)  # 解码低4位
        val_hi = _dequant_fp4_lut((b_u8 >> 4) & 0x0F)  # 解码高4位

        group_ids = tl.arange(0, BLOCK_K // 2) // 16  # 计算缩放因子组ID
        scales_per_byte = tl.load(  # 加载缩放因子
            B_scale_ptr  # 缩放因子基地址
            + offs_n[:, None] * stride_bsn  # 加上N方向偏移
            + (k_start // 32 + group_ids[None, :]) * stride_bsk32,  # 加上组方向偏移
            mask=(offs_n[:, None] < N)  # N方向越界掩码
            & ((k_start // 32 + group_ids[None, :]) < K // 32),  # K方向越界掩码
            other=1.0,  # 越界填充1.0
        )
        val_lo = val_lo * scales_per_byte  # 对低4位解码值应用缩放
        val_hi = val_hi * scales_per_byte  # 对高4位解码值应用缩放

        offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2  # 偶数K索引偏移
        offs_k_odd = offs_k_even + 1  # 奇数K索引偏移

        a_even_mask = (offs_m[:, None] < M) & (offs_k_even[None, :] < K)  # A矩阵偶数列加载掩码
        a_even = tl.load(  # 加载A矩阵偶数列
            A_ptr + offs_m[:, None] * stride_am + offs_k_even[None, :] * stride_ak,  # 计算加载地址
            mask=a_even_mask,  # 越界掩码
            other=0.0,  # 越界填充0.0
        ).to(tl.float32)  # 转为FP32

        a_odd_mask = (offs_m[:, None] < M) & (offs_k_odd[None, :] < K)  # A矩阵奇数列加载掩码
        a_odd = tl.load(  # 加载A矩阵奇数列
            A_ptr + offs_m[:, None] * stride_am + offs_k_odd[None, :] * stride_ak,  # 计算加载地址
            mask=a_odd_mask,  # 越界掩码
            other=0.0,  # 越界填充0.0
        ).to(tl.float32)  # 转为FP32

        acc += tl.dot(a_even, tl.trans(val_lo))  # 累加偶数K部分的矩阵乘
        acc += tl.dot(a_odd, tl.trans(val_hi))  # 累加奇数K部分的矩阵乘

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)  # 生成C矩阵存储掩码
    tl.store(  # 将累加结果写入输出矩阵
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,  # 计算存储地址
        acc.to(tl.bfloat16),  # 将FP32结果转为BF16
        mask=c_mask,  # 越界掩码
    )


def mxfp4_gemm_triton(  # Triton融合MXFP4反量化+GEMM接口函数
    A: torch.Tensor,  # 输入激活矩阵 [M, K]
    B_packed: torch.Tensor,  # 打包的FP4权重 [N, K//2]
    B_scale: torch.Tensor,  # 缩放因子 [N, K//32]
    K_full: int,  # 完整的K维度大小
) -> torch.Tensor:  # 返回计算结果 [M, N]
    """Triton fused MXFP4 dequant + GEMM: output = A @ dequant(B).T
    Triton融合MXFP4反量化 + GEMM：output = A @ dequant(B).T

    Kept for standalone benchmarking. The MoE forward uses the slot kernel.
    保留用于独立基准测试。MoE前向使用slot内核。
    """
    M = A.shape[0]  # 获取令牌数
    N = B_packed.shape[0]  # 获取输出维度
    K = K_full  # 获取完整K维度

    if B_scale.dtype == torch.float8_e8m0fnu:  # 如果缩放因子为E8M0格式
        B_scale = B_scale.to(torch.float32)  # 转为FP32
    elif B_scale.dtype != torch.float32:  # 如果缩放因子不是FP32
        B_scale = B_scale.float()  # 转为FP32

    C = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)  # 创建输出张量
    A = A.contiguous()  # 确保A内存连续
    B_packed = B_packed.contiguous()  # 确保B内存连续
    B_scale = B_scale.contiguous()  # 确保缩放因子内存连续

    grid = lambda meta: (  # 定义网格维度
        triton.cdiv(M, meta["BLOCK_M"]),  # M方向网格大小
        triton.cdiv(N, meta["BLOCK_N"]),  # N方向网格大小
    )
    B_u8 = B_packed.view(torch.uint8)  # 将打包权重视图转为uint8

    _mxfp4_gemm_kernel[grid](  # 调用MXFP4 GEMM内核
        A,  # 输入激活矩阵
        B_u8,  # uint8格式的打包权重
        B_scale,  # 缩放因子
        C,  # 输出矩阵
        M,  # 令牌数
        N,  # 输出维度
        K,  # 输入维度
        A.stride(0),  # A的M方向步幅
        A.stride(1),  # A的K方向步幅
        B_u8.stride(0),  # B的N方向步幅
        B_u8.stride(1),  # B的K//2方向步幅
        B_scale.stride(0),  # 缩放因子N方向步幅
        B_scale.stride(1),  # 缩放因子K//32方向步幅
        C.stride(0),  # C的M方向步幅
        C.stride(1),  # C的N方向步幅
    )
    return C  # 返回计算结果


def mxfp4_moe_forward_triton(  # SM120优化的MXFP4 MoE前向计算函数（CUDA图兼容）
    hidden_states: torch.Tensor,  # 输入隐藏状态 [M, K]
    w13_packed: torch.Tensor,  # 打包的w13权重 [E, 2*I, K//2]
    w2_packed: torch.Tensor,  # 打包的w2权重 [E, K, I//2]
    w13_scale: torch.Tensor,  # w13缩放因子 [E, 2*I, K//32]
    w2_scale: torch.Tensor,  # w2缩放因子 [E, K, I//32]
    topk_ids: torch.Tensor,  # Top-k专家索引 [M, topk]
    topk_weights: torch.Tensor,  # Top-k路由权重 [M, topk]
    hidden_size: int,  # 隐藏维度大小
    intermediate_size: int,  # 中间维度大小
    inplace: bool = False,  # 是否原地操作
    routed_scaling_factor: Optional[float] = None,  # 路由缩放因子
    clamp_limit: Optional[float] = None,  # 激活限幅值
) -> torch.Tensor:  # 返回MoE计算结果 [M, K]
    """SM120-optimized MXFP4 MoE forward — CUDA graph compatible.
    SM120优化的MXFP4 MoE前向计算——兼容CUDA图捕获。

    Uses per-slot GEMV kernels instead of per-expert Python loops.
    使用逐slot GEMV内核代替逐专家Python循环。
    Each (token, expert) slot is processed independently with a fixed grid,
    每个(令牌, 专家)slot以固定网格独立处理，
    eliminating .unique()/.item()/.nonzero() that break CUDA graph capture.
    消除会破坏CUDA图捕获的.unique()/.item()/.nonzero()调用。
    """
    import torch.nn.functional as F  # 导入PyTorch函数模块

    M, K = hidden_states.shape  # 获取令牌数M和隐藏维度K
    topk = topk_ids.shape[1]  # 获取top-k值
    I = intermediate_size  # 获取中间维度
    num_slots = M * topk  # 计算总slot数（令牌数 * topk）
    device = hidden_states.device  # 获取计算设备
    dtype = hidden_states.dtype  # 获取数据类型

    # ── Graph-safe routing: flatten topk assignments ──
    # ── 图安全路由：展平topk分配 ──
    # token_ids[slot] = which row of A (original token index)
    # token_ids[slot] = A的哪一行（原始令牌索引）
    # expert_ids[slot] = which expert's weights to use
    # expert_ids[slot] = 使用哪个专家的权重
    # topk_ids may contain -1 for padded/filtered tokens; clamp to 0 for safe
    # topk_ids可能包含-1（填充/过滤的令牌）；钳制到0以保证安全
    # Triton loads, then zero out invalid slots' output after GEMM.
    # Triton加载，然后在GEMM后将无效slot的输出清零。
    flat_expert_ids_raw = topk_ids.reshape(-1).contiguous()  # [M*topk]  # 展平专家ID
    invalid_slot_mask = flat_expert_ids_raw < 0  # [M*topk]  # 标记无效slot（专家ID为-1的）
    flat_expert_ids = flat_expert_ids_raw.clamp(min=0)  # safe for indexing  # 钳制到>=0，确保安全索引
    token_ids = (  # 生成令牌ID映射
        torch.arange(M, device=device, dtype=torch.int32)  # [0, 1, ..., M-1]
        .unsqueeze(1)  # 扩展维度 [M, 1]
        .expand(M, topk)  # 扩展到 [M, topk]
        .reshape(-1)  # 展平为 [M*topk]
        .contiguous()  # 确保内存连续
    )  # [M*topk]  # 每个slot对应的原始令牌索引

    # ── Ensure scales are float32 ──
    # ── 确保缩放因子为float32 ──
    if w13_scale.dtype != torch.float32:  # 如果w13缩放因子不是FP32
        w13_scale = w13_scale.to(torch.float32)  # 转为FP32
    if w2_scale.dtype != torch.float32:  # 如果w2缩放因子不是FP32
        w2_scale = w2_scale.to(torch.float32)  # 转为FP32

    # ── GEMM1: gate_up projection ──
    # ── GEMM1：门控+上投影 ──
    # hidden_states[token] @ w13[expert].T → [num_slots, 2*I]
    # hidden_states[token] @ w13[expert].T → [num_slots, 2*I]
    intermediate = torch.empty(num_slots, 2 * I, dtype=dtype, device=device)  # 创建GEMM1输出缓冲区

    w13_u8 = w13_packed.view(torch.uint8)  # [E, 2*I, K//2]  # 将w13打包权重视图转为uint8
    grid1 = lambda meta: (num_slots, triton.cdiv(2 * I, meta["BLOCK_N"]))  # 定义GEMM1网格维度

    _mxfp4_slot_gemv_kernel[grid1](  # 调用逐slot GEMV内核执行GEMM1
        hidden_states,  # 输入隐藏状态
        w13_u8,  # uint8格式的w13权重
        w13_scale,  # w13缩放因子
        intermediate,  # GEMM1输出
        token_ids,  # 令牌ID映射
        flat_expert_ids,  # 专家ID映射
        2 * I,  # 输出维度（2倍中间维度）
        K,  # 输入维度
        hidden_states.stride(0),  # A矩阵M方向步幅
        w13_u8.stride(1),  # B矩阵N方向步幅
        w13_u8.stride(2),  # B矩阵K//2方向步幅
        w13_scale.stride(1),  # 缩放因子N方向步幅
        w13_scale.stride(2),  # 缩放因子K//32方向步幅
        w13_u8.stride(0),  # 专家间B步幅
        w13_scale.stride(0),  # 专家间缩放步幅
        intermediate.stride(0),  # C矩阵步幅
    )

    # ── SiLU activation (graph-safe vectorized ops) ──
    # ── SiLU激活（图安全的向量化操作） ──
    gate = intermediate[:, :I].float()  # 提取门控部分并转为FP32
    up = intermediate[:, I:].float()  # 提取上投影部分并转为FP32
    if clamp_limit is not None and clamp_limit > 0:  # 如果指定了限幅值
        gate = torch.clamp(gate, max=clamp_limit)  # 对gate进行上限限幅
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)  # 对up进行双向限幅
    activated = (F.silu(gate) * up).to(dtype)  # 计算silu(gate) * up并转回原数据类型

    # ── GEMM2: down projection ──
    # ── GEMM2：下投影 ──
    # activated[slot] @ w2[expert].T → [num_slots, K]
    # activated[slot] @ w2[expert].T → [num_slots, K]
    down = torch.empty(num_slots, K, dtype=dtype, device=device)  # 创建GEMM2输出缓冲区

    # For GEMM2, A is the activated buffer — each slot reads its own row
    # 对于GEMM2，A是激活后的缓冲区——每个slot读取自己的行
    slot_ids = torch.arange(num_slots, device=device, dtype=torch.int32)  # 生成slot ID序列

    w2_u8 = w2_packed.view(torch.uint8)  # [E, K, I//2]  # 将w2打包权重视图转为uint8
    grid2 = lambda meta: (num_slots, triton.cdiv(K, meta["BLOCK_N"]))  # 定义GEMM2网格维度

    _mxfp4_slot_gemv_kernel[grid2](  # 调用逐slot GEMV内核执行GEMM2
        activated,  # 激活后的中间结果
        w2_u8,  # uint8格式的w2权重
        w2_scale,  # w2缩放因子
        down,  # GEMM2输出
        slot_ids,  # slot ID（每个slot读取自己的行）
        flat_expert_ids,  # 专家ID映射
        K,  # 输出维度
        I,  # 输入维度
        activated.stride(0),  # A矩阵M方向步幅
        w2_u8.stride(1),  # B矩阵N方向步幅
        w2_u8.stride(2),  # B矩阵K//2方向步幅
        w2_scale.stride(1),  # 缩放因子N方向步幅
        w2_scale.stride(2),  # 缩放因子K//32方向步幅
        w2_u8.stride(0),  # 专家间B步幅
        w2_scale.stride(0),  # 专家间缩放步幅
        down.stride(0),  # C矩阵步幅
    )

    # ── Zero out invalid slots (padded/filtered tokens with topk_ids == -1) ──
    # ── 将无效slot清零（topk_ids == -1的填充/过滤令牌） ──
    # Use multiplication instead of boolean indexing to stay CUDA-graph-safe
    # 使用乘法而非布尔索引以保持CUDA图安全
    # (no GPU→CPU sync). valid_mask is 1.0 for valid slots, 0.0 for invalid.
    # （无GPU→CPU同步）。valid_mask对有效slot为1.0，对无效slot为0.0。
    valid_mask = (~invalid_slot_mask).unsqueeze(1).to(dtype)  # [M*topk, 1]  # 生成有效slot掩码
    down = down * valid_mask  # 将无效slot的输出清零

    # ── Weighted sum across topk slots (graph-safe) ──
    # ── 跨topk slot的加权和（图安全） ──
    flat_weights = topk_weights.reshape(-1).unsqueeze(1).to(dtype)  # [M*topk, 1]  # 展平路由权重
    output = (down * flat_weights).view(M, topk, K).sum(dim=1)  # 加权求和：乘以权重后reshape并在topk维度求和

    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:  # 如果指定了非1.0的路由缩放因子
        output.mul_(routed_scaling_factor)  # 原地乘以路由缩放因子

    return output  # 返回MoE最终输出
