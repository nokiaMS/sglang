# MurmurHash3 哈希工具模块
# 基于 Triton 实现的 MurmurHash3 32位哈希函数，用于GPU端的高性能哈希计算，
# 支持批量处理种子、位置和列索引的哈希操作。

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言


@triton.jit
def rotl32(x, r: tl.constexpr) -> tl.uint32:  # 32位整数循环左移
    """
    rotate left 32-bit integer x by r bits
    e.g. x = 01110001, r = 2 -> 11000101
    # 将32位整数x循环左移r位
    # 例如 x = 01110001, r = 2 -> 11000101
    """
    x = x.to(tl.uint64)  # 转为64位以避免移位溢出
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF  # 循环左移：左移r位 + 右移(32-r)位，再掩码取低32位


@triton.jit
def fmix32(h: tl.uint32) -> tl.uint32:  # MurmurHash的32位最终混合函数
    """
    final mix of 32-bit hash value for MurmurHash
    # MurmurHash的32位哈希值最终混合
    """
    h ^= h >> 16  # 右移16位后异或
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF  # 乘以混合常数并掩码
    h ^= h >> 13  # 右移13位后异或
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF  # 乘以混合常数并掩码
    h ^= h >> 16  # 右移16位后异或
    return h  # 返回混合后的哈希值


@triton.jit
def murmur3_mix(h: tl.uint32, k: tl.uint32) -> tl.uint32:  # MurmurHash3的单块混合函数
    """
    Mixes a 32-bit key into the hash state.
    # 将32位键混合到哈希状态中。
    """
    c1: tl.uint32 = 0xCC9E2D51  # MurmurHash3混合常数c1
    c2: tl.uint32 = 0x1B873593  # MurmurHash3混合常数c2
    r1: tl.constexpr = 15  # 循环左移位数r1
    r2: tl.constexpr = 13  # 循环左移位数r2
    mm: tl.uint32 = 5  # 乘法常数
    nn: tl.uint32 = 0xE6546B64  # 加法常数

    k = (k * c1) & 0xFFFFFFFF  # 键乘以c1并掩码
    k = rotl32(k, r1)  # 循环左移r1位
    k = (k * c2) & 0xFFFFFFFF  # 键乘以c2并掩码
    h ^= k  # 异或到哈希状态
    h = rotl32(h, r2)  # 哈希状态循环左移r2位
    h = (h * mm + nn) & 0xFFFFFFFF  # 乘以mm加nn并掩码
    return h  # 返回混合后的哈希状态


@triton.jit
def murmur_hash32_kernel(  # MurmurHash 32位 Triton 核函数
    seed_ptr,  # 种子指针
    positions_ptr,  # 位置指针
    col_indices_ptr,  # 列索引指针
    output_ptr,  # 输出指针
    num_rows,  # 行数
    num_cols,  # 列数
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    """
    MurmurHash 32-bit implementation for Triton.
    Reference:
    - https://medium.com/@thealonemusk/murmurhash-the-scrappy-algorithm-that-secretly-powers-half-the-internet-2d3f79b4509b
    - https://en.wikipedia.org/wiki/MurmurHash
    # Triton的MurmurHash 32位实现。
    # 参考：
    # - https://medium.com/@thealonemusk/murmurhash-the-scrappy-algorithm-that-secretly-powers-half-the-internet-2d3f79b4509b
    # - https://en.wikipedia.org/wiki/MurmurHash

    We treat 64-bit seed, 32-bit position, and 32-bit col_index as 4 4-byte blocks, and bit-blend them together.
    # 我们将64位种子、32位位置和32位列索引视为4个4字节块，并将它们进行位混合。
    """
    pid_row = tl.program_id(0)  # 获取行方向程序ID
    pid_col = tl.program_id(1)  # 获取列方向程序ID

    row_idx = pid_row  # 行索引
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 列偏移量
    mask = col_offsets < num_cols  # 列掩码，防止越界

    # Load inputs
    # 加载输入
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64)  # 加载种子并转为64位无符号整数
    pos = tl.load(positions_ptr + row_idx).to(tl.uint32)  # 加载位置并转为32位无符号整数
    col = tl.load(col_indices_ptr + col_offsets, mask=mask, other=0).to(tl.uint32)  # 加载列索引并转为32位无符号整数

    h: tl.uint32 = 0  # hash accumulator  # 哈希累加器

    # Process seed_low
    # 处理种子低32位
    k: tl.uint32 = (seed & 0xFFFFFFFF).to(tl.uint32)  # 提取种子低32位
    h = murmur3_mix(h, k)  # 混合种子低32位

    # Process seed_high
    # 处理种子高32位
    k = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)  # 提取种子高32位
    h = murmur3_mix(h, k)  # 混合种子高32位

    # Process position block starting from seed32
    # 从seed32开始处理位置块
    h = murmur3_mix(h, pos)  # 混合位置

    # Process col block
    # 处理列块
    h = murmur3_mix(h, col)  # 混合列索引

    # Finalize (len=16 for seed + pos + col)
    # 最终化（长度=16，对应seed + pos + col共4个4字节块）
    h ^= 16  # 异或总字节数
    h = fmix32(h)  # 应用最终混合函数

    # Store result as uint32
    # 将结果存储为uint32
    tl.store(output_ptr + row_idx * num_cols + col_offsets, h, mask=mask)  # 存储哈希结果到输出


def murmur_hash32(seed, positions, col_indices):  # 批量计算MurmurHash3 32位哈希
    assert (  # 断言种子和位置形状相同
        seed.shape == positions.shape
    ), "Seed and positions must have the same shape (n,)"  # 种子和位置必须具有相同形状(n,)
    assert (  # 断言输入为一维张量
        len(seed.shape) == 1 and len(col_indices.shape) == 1
    ), f"Inputs must be 1D tensors {seed.shape=} {col_indices.shape=}"  # 输入必须是一维张量
    n = seed.shape[0]  # 行数
    m = col_indices.shape[0]  # 列数
    device = seed.device  # 设备
    hashed = torch.empty((n, m), dtype=torch.uint32, device=device)  # 创建输出张量

    BLOCK_SIZE = 1024  # 块大小
    grid = (n, triton.cdiv(m, BLOCK_SIZE))  # 计算网格大小
    murmur_hash32_kernel[grid](  # 启动Triton核函数
        seed, positions, col_indices, hashed, n, m, BLOCK_SIZE=BLOCK_SIZE
    )
    return hashed  # 返回哈希结果
