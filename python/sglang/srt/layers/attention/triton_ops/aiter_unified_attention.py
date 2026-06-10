# Aiter统一注意力Triton内核文件
# 本文件实现了用于构建分页注意力页表的Triton内核，
# 包括将不规则token级KV索引散列到2D块级页表，
# 以及从req_to_token映射构建验证阶段的页表。
# 支持滑动窗口注意力(SWA)的页表构建。

import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块


@triton.jit  # Triton JIT编译装饰器
def scatter_ragged_to_page_table_kernel(  # 将不规则token级KV索引散列到2D块级页表内核
    kv_flat_ptr,  # 扁平化KV索引指针
    kv_indptr_ptr,  # KV索引偏移指针
    dest_ptr,  # 目标页表指针
    dest_stride,  # 目标页表步长
    sw_page_table_ptr,  # 滑动窗口页表指针
    swa_slot_mapping_ptr,  # 滑动窗口槽位映射指针
    PAGE_SIZE: tl.constexpr,  # 页大小（编译时常量）
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
    HAS_SWA: tl.constexpr,  # 是否有滑动窗口注意力（编译时常量）
):  # 将不规则（ragged）的token级KV索引散列到2D块级页表
    """Scatter ragged token-level kv_indices into a 2D block-level page table."""  # 将不规则token级KV索引散列到2D块级页表。
    """将不规则token级KV索引散列到2D块级页表。"""
    pid = tl.program_id(0)  # 获取序列维度程序ID
    block_id = tl.program_id(1)  # 获取块维度程序ID

    start = tl.load(kv_indptr_ptr + pid).to(tl.int64)  # 加载当前序列的KV起始索引
    kv_len = tl.load(kv_indptr_ptr + pid + 1).to(tl.int64) - start  # 计算当前序列的KV长度
    num_blocks = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE  # 计算当前序列的块数（向上取整）

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算块内偏移
    if block_id * BLOCK_SIZE >= num_blocks:  # 如果当前块超出范围
        return  # 直接返回
    mask = offsets < num_blocks  # 生成有效块掩码
    token_idx = offsets.to(tl.int64) * PAGE_SIZE  # 计算token索引（块索引*页大小）
    vals = tl.load(kv_flat_ptr + start + token_idx, mask=mask, other=0)  # 加载KV索引值
    block_vals = vals // PAGE_SIZE  # 将token级索引转换为块级页号
    tl.store(  # 存储块级页号到目标页表
        dest_ptr + pid.to(tl.int64) * dest_stride + offsets,  # 计算目标地址
        block_vals,  # 块级页号
        mask=mask,  # 掩码
    )

    if HAS_SWA:  # 如果有滑动窗口注意力
        sw_vals = tl.load(swa_slot_mapping_ptr + vals)  # 通过槽位映射获取滑动窗口页号
        block_vals = sw_vals // PAGE_SIZE  # 转换为块级页号
        tl.store(  # 存储滑动窗口块级页号
            sw_page_table_ptr + pid.to(tl.int64) * dest_stride + offsets,  # 计算目标地址
            block_vals,  # 块级页号
            mask=mask,  # 掩码
        )


@triton.jit  # Triton JIT编译装饰器
def scatter_req_to_token_to_page_table_kernel(  # 从req_to_token映射构建验证阶段页表内核
    req_to_token_ptr,  # 请求到token映射指针
    req_pool_indices_ptr,  # 请求池索引指针
    seq_lens_ptr,  # 序列长度指针
    page_table_ptr,  # 目标页表指针
    req_to_token_stride,  # req_to_token步长
    page_table_stride,  # 页表步长
    sw_page_table_ptr,  # 滑动窗口页表指针
    swa_slot_mapping_ptr,  # 滑动窗口槽位映射指针
    DRAFT_NUM: tl.constexpr,  # 草稿token数量（编译时常量）
    PAGE_SIZE: tl.constexpr,  # 页大小（编译时常量）
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
    HAS_SWA: tl.constexpr,  # 是否有滑动窗口注意力（编译时常量）
):  # 从req_to_token映射构建target_verify的2D块级页表
    """Build the 2D block-level page_table for target_verify from req_to_token."""  # 从req_to_token构建target_verify的2D块级页表。
    """从req_to_token映射构建target_verify的2D块级页表。"""
    pid = tl.program_id(0)  # 获取序列维度程序ID
    block_id = tl.program_id(1)  # 获取块维度程序ID

    seq_len = tl.load(seq_lens_ptr + pid).to(tl.int64)  # 加载当前序列长度
    kv_len = seq_len + DRAFT_NUM  # 计算KV长度（序列长度+草稿token数）
    num_blocks = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE  # 计算块数（向上取整）

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算块内偏移
    if block_id * BLOCK_SIZE >= num_blocks:  # 如果当前块超出范围
        return  # 直接返回
    mask = offsets < num_blocks  # 生成有效块掩码

    rp = tl.load(req_pool_indices_ptr + pid).to(tl.int64)  # 加载当前请求的池索引
    token_idx = offsets.to(tl.int64) * PAGE_SIZE  # 计算token索引（块索引*页大小）
    vals = tl.load(  # 加载KV索引值
        req_to_token_ptr + rp * req_to_token_stride + token_idx,  # 通过req_to_token映射获取
        mask=mask,  # 掩码
        other=0,  # 越界填充值
    )
    block_vals = vals // PAGE_SIZE  # 将token级索引转换为块级页号
    tl.store(  # 存储块级页号到页表
        page_table_ptr + pid.to(tl.int64) * page_table_stride + offsets,  # 计算目标地址
        block_vals,  # 块级页号
        mask=mask,  # 掩码
    )

    if HAS_SWA:  # 如果有滑动窗口注意力
        sw_vals = tl.load(swa_slot_mapping_ptr + vals)  # 通过槽位映射获取滑动窗口页号
        block_vals = sw_vals // PAGE_SIZE  # 转换为块级页号
        tl.store(  # 存储滑动窗口块级页号
            sw_page_table_ptr + pid.to(tl.int64) * page_table_stride + offsets,  # 计算目标地址
            block_vals,  # 块级页号
            mask=mask,  # 掩码
        )
