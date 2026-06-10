# 分页对齐的令牌到KV池分配器
# 实现了页对齐的内存分配策略，包含Triton加速的分配内核和朴素CPU分配函数
"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations  # 启用延迟注解评估

"""
Page-aligned memory pool.
"""  # 分页对齐的内存池

from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch张量库
import triton  # 导入Triton GPU编程框架
import triton.language as tl  # 导入Triton语言模块

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator  # 导入基础分配器
from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2  # 导入工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.mem_cache.memory_pool import KVCache  # 导入KV缓存类型


def alloc_extend_naive(  # 朴素扩展分配函数（CPU实现）
    prefix_lens,  # 前缀长度张量
    seq_lens,  # 序列长度张量
    last_loc,  # 每个序列最后一个token的位置
    free_pages,  # 空闲页列表
    out_indices,  # 输出索引张量
    page_size,  # 页大小
    device,  # 设备类型
):
    extend_lens = seq_lens - prefix_lens  # 计算扩展长度
    end_pos = torch.cumsum(extend_lens, 0)  # 计算累积结束位置
    start_pos = end_pos - extend_lens  # 计算每个序列的起始位置
    num_new_pages = (seq_lens + page_size - 1) // page_size - (  # 计算每个序列需要的新页数
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (  # 计算每个序列新增的完整页数
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages  # 计算每个序列是否需要不完整的尾页
    end_new_pages = torch.cumsum(num_new_pages, 0)  # 计算新页累积结束位置
    start_new_pages = end_new_pages - num_new_pages  # 计算每个序列的新页起始位置
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)  # 页内偏移量序列
    for i in range(len(prefix_lens)):  # 遍历每个序列
        num1 = (  # 第一部分：填充旧的未满页中剩余的槽位
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:  # 如果有旧页需要填充
            out_indices[start_pos[i] : start_pos[i] + num1] = (  # 填充旧页的剩余槽位
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)  # 从last_loc+1开始连续填充
            )

        if prefix_lens[i] + num1 == seq_lens[i]:  # 如果旧页填充完毕就结束
            continue

        num2 = (  # 第二部分：填充新增的完整页
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:  # 如果有新增完整页
            pages = (  # 从空闲页列表中取页
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (  # 填充完整页的槽位
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:  # 如果填充完毕就结束
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size  # 第三部分：填充新的未满尾页
        if num3:  # 如果有尾页需要填充
            out_indices[end_pos[i] - num3 : end_pos[i]] = (  # 填充尾页的槽位
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


@triton.jit  # Triton JIT编译的GPU内核：扩展分配
def alloc_extend_kernel(  # 扩展分配GPU内核函数
    pre_lens_ptr,  # 前缀长度指针
    seq_lens_ptr,  # 序列长度指针
    last_loc_ptr,  # 最后位置指针
    free_page_ptr,  # 空闲页指针
    out_indices,  # 输出索引指针
    bs_upper: tl.constexpr,  # 批次大小上界（编译时常量）
    page_size: tl.constexpr,  # 页大小（编译时常量）
):
    pid = tl.program_id(0)  # 获取当前程序ID（序列索引）

    load_offset = tl.arange(0, bs_upper)  # 加载偏移量范围
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)  # 加载所有序列长度
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)  # 加载所有前缀长度
    extend_lens = seq_lens - pre_lens  # 计算所有扩展长度

    seq_len = tl.load(seq_lens_ptr + pid)  # 加载当前序列长度
    pre_len = tl.load(pre_lens_ptr + pid)  # 加载当前前缀长度
    extend_len = seq_len - pre_len  # 计算当前扩展长度

    sum_extend_lens = tl.sum(extend_lens)  # 计算所有扩展长度之和
    output_start_loc = sum_extend_lens - extend_len  # 计算当前序列在输出中的起始位置

    num_pages_after = (seq_lens + page_size - 1) // page_size  # 扩展后每个序列的页数
    num_pages_before = (pre_lens + page_size - 1) // page_size  # 扩展前每个序列的页数
    num_new_pages = num_pages_after - num_pages_before  # 每个序列新增的页数

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (  # 当前序列新增的页数
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)  # 所有序列新增页数之和
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self  # 当前序列在新页列表中的起始位置

    # Part 1: fill the old partial page
    # 第一部分：填充旧的未满页
    last_loc = tl.load(last_loc_ptr + pid)  # 加载当前序列最后位置
    num_part1 = (  # 旧页中剩余槽位数
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)  # 页内偏移量
    tl.store(  # 存储旧页槽位的索引
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,  # 从last_loc+1开始
        mask=offset_one_page < num_part1,  # 只存储有效范围内的值
    )
    if pre_len + num_part1 == seq_len:  # 如果旧页填充完毕则返回
        return

    # Part 2: fill the new full pages using a dynamic blocked loop.
    # 第二部分：使用动态分块循环填充新增的完整页
    # The loop bound is derived from num_part2 (runtime value), so Triton
    # generates a real loop instead of unrolling — no constexpr dependency
    # on extend size and only one kernel compilation.
    # 循环边界来自num_part2（运行时值），Triton生成真正的循环而非展开——
    # 不依赖扩展大小的编译时常量，只需一次内核编译
    num_part2 = (  # 新增完整页中的槽位数
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096  # 每个块处理的元素数
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND  # 计算需要的块数
    for block_id in range(num_blocks):  # 遍历每个块
        offset_in_block = tl.arange(0, BLOCK_EXTEND)  # 块内偏移量
        offset = block_id * BLOCK_EXTEND + offset_in_block  # 全局偏移量
        mask = offset < num_part2  # 有效掩码
        page_start = tl.load(  # 加载页起始索引
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(  # 存储完整页中每个槽位的索引
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,  # 页基址加页内偏移
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:  # 如果填充完毕则返回
        return

    # Part 3: fill the new partial page
    # 第三部分：填充新的未满尾页
    num_part3 = seq_len - seq_len // page_size * page_size  # 尾页中的槽位数
    start_loc = tl.load(  # 加载尾页的页索引
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(  # 存储尾页中每个槽位的索引
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,  # 页基址加页内偏移
        mask=offset_one_page < num_part3,  # 只存储有效范围内的值
    )


@triton.jit  # Triton JIT编译的GPU内核：解码分配
def alloc_decode_kernel(  # 解码分配GPU内核函数
    seq_lens_ptr,  # 序列长度指针
    last_loc_ptr,  # 最后位置指针
    free_page_ptr,  # 空闲页指针
    out_indices,  # 输出索引指针
    bs_upper: tl.constexpr,  # 批次大小上界（编译时常量）
    page_size: tl.constexpr,  # 页大小（编译时常量）
):
    pid = tl.program_id(0)  # 获取当前程序ID（序列索引）

    load_offset = tl.arange(0, bs_upper)  # 加载偏移量范围
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)  # 加载所有序列长度
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)  # 计算前缀长度（解码时前缀=序列-1）

    seq_len = tl.load(seq_lens_ptr + pid)  # 加载当前序列长度
    pre_len = seq_len - 1  # 当前前缀长度

    num_pages_after = (seq_lens + page_size - 1) // page_size  # 解码后每个序列的页数
    num_pages_before = (pre_lens + page_size - 1) // page_size  # 解码前每个序列的页数
    num_new_pages = num_pages_after - num_pages_before  # 每个序列新增的页数

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (  # 当前序列新增的页数
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)  # 所有序列新增页数之和
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self  # 当前序列在新页列表中的起始位置

    if num_page_start_loc_self == 0:  # 如果不需要新页（当前页还有空间）
        last_loc = tl.load(last_loc_ptr + pid)  # 加载最后位置
        tl.store(out_indices + pid, last_loc + 1)  # 直接使用last_loc+1作为新token位置
    else:  # 需要分配新页
        page = tl.load(free_page_ptr + new_page_start_loc)  # 从空闲页列表取一个新页
        tl.store(out_indices + pid, page * page_size)  # 使用新页的第一个槽位


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):  # 分页令牌到KV池分配器
    """
    An allocator managing the indices to kv cache data.
    管理KV缓存数据索引的分配器

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.
    本类与TokenToKVPoolAllocator接口相同，但一个请求的输出总是页对齐的

    TODO: fuse last_loc into the kernel.
    待办：将last_loc融合到内核中
    """

    def __init__(  # 初始化方法
        self,
        size: int,  # 池的总大小（槽位数）
        page_size: int,  # 页大小
        dtype: torch.dtype,  # 数据类型
        device: str,  # 设备类型
        kvcache: KVCache,  # KV缓存对象
        need_sort: bool,  # 是否需要在释放时排序
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)  # 调用父类初始化
        self.num_pages = size // page_size  # 计算总页数
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")  # 读取调试模式环境变量
        self.clear()  # 初始化空闲页列表

    def alloc(self, need_size: int):  # 分配指定大小的空间（页对齐）
        # page-aligned allocation, returning contiguous indices of pages
        # 页对齐分配，返回连续的页索引
        if self.debug_mode:  # 调试模式下检查对齐
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"  # 分配大小必须是页对齐的

        num_pages = need_size // self.page_size  # 计算需要的页数
        if self.need_sort and num_pages > len(self.free_pages):  # 需要排序且空闲页不足
            self.merge_and_sort_free()  # 合并并排序空闲页
        if num_pages > len(self.free_pages):  # 空闲页不足
            return None  # 返回None表示分配失败

        out_pages = self.free_pages[:num_pages]  # 取出需要的页
        self.free_pages = self.free_pages[num_pages:]  # 更新空闲页列表

        out_indices = (  # 将页索引展开为槽位索引
            out_pages[:, None] * self.page_size  # 页基址
            + torch.arange(self.page_size, device=self.device)  # 加页内偏移
        ).reshape(-1)  # 展平为一维

        return out_indices  # 返回槽位索引

    def alloc_extend(  # 扩展阶段分配方法
        self,
        prefix_lens: torch.Tensor,  # 前缀长度张量
        prefix_lens_cpu: torch.Tensor,  # 前缀长度CPU张量
        seq_lens: torch.Tensor,  # 序列长度张量
        seq_lens_cpu: torch.Tensor,  # 序列长度CPU张量
        last_loc: torch.Tensor,  # 每个序列最后一个token的位置
        extend_num_tokens: int,  # 扩展token总数
        num_new_pages: int = None,  # 新页数（可选，为None时自动计算）
    ):
        if self.debug_mode:  # 调试模式下检查对齐一致性
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )  # last_loc+1与prefix_lens的页内偏移应一致

        bs = len(prefix_lens)  # 批次大小
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(  # 需要排序且可能空闲页不足
            self.free_pages
        ):
            self.merge_and_sort_free()  # 合并并排序空闲页

        out_indices = torch.empty(  # 预分配输出索引张量
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_kernel[(bs,)](  # 调用Triton扩展分配内核
            prefix_lens,  # 前缀长度
            seq_lens,  # 序列长度
            last_loc,  # 最后位置
            self.free_pages,  # 空闲页列表
            out_indices,  # 输出索引
            next_power_of_2(bs),  # 批次大小上界（2的幂次）
            self.page_size,  # 页大小
        )

        if self.debug_mode:  # 调试模式下检查索引唯一性
            assert len(torch.unique(out_indices)) == len(out_indices)  # 输出索引应全部唯一

        if num_new_pages is None:  # 如果未提供新页数
            num_new_pages = get_num_new_pages(  # 自动计算新页数
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):  # 空闲页不足
            return None  # 返回None表示分配失败

        self.free_pages = self.free_pages[num_new_pages:]  # 从空闲页列表移除已使用的页
        return out_indices  # 返回槽位索引

    def alloc_decode(  # 解码阶段分配方法
        self,
        seq_lens: torch.Tensor,  # 序列长度张量
        seq_lens_cpu: torch.Tensor,  # 序列长度CPU张量
        last_loc: torch.Tensor,  # 每个序列最后一个token的位置
    ):
        if self.debug_mode:  # 调试模式下检查对齐一致性
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )  # last_loc+2与seq_lens的页内偏移应一致

        bs = len(seq_lens)  # 批次大小
        if self.need_sort and bs > len(self.free_pages):  # 需要排序且可能空闲页不足
            self.merge_and_sort_free()  # 合并并排序空闲页

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)  # 预分配输出索引张量
        alloc_decode_kernel[(bs,)](  # 调用Triton解码分配内核
            seq_lens,  # 序列长度
            last_loc,  # 最后位置
            self.free_pages,  # 空闲页列表
            out_indices,  # 输出索引
            next_power_of_2(bs),  # 批次大小上界（2的幂次）
            self.page_size,  # 页大小
        )

        if self.debug_mode:  # 调试模式下检查索引唯一性
            assert len(torch.unique(out_indices)) == len(out_indices)  # 输出索引应全部唯一

        num_new_pages = get_num_new_pages(  # 计算需要的新页数
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,  # 解码模式
        )
        if num_new_pages > len(self.free_pages):  # 空闲页不足
            return None  # 返回None表示分配失败

        self.free_pages = self.free_pages[num_new_pages:]  # 从空闲页列表移除已使用的页
        return out_indices  # 返回槽位索引

    def free(self, free_index: torch.Tensor):  # 释放指定索引的空间
        if free_index.numel() == 0:  # 如果没有需要释放的索引
            return  # 直接返回

        if self.is_not_in_free_group:  # 不在批量释放模式中
            free_page_indices = torch.unique(free_index // self.page_size)  # 将槽位索引转换为页索引并去重
            if self.need_sort:  # 需要排序模式
                self.release_pages = torch.cat((free_page_indices, self.release_pages))  # 加入待释放页列表
            else:  # 不需要排序模式
                self.free_pages = torch.cat((free_page_indices, self.free_pages))  # 直接加入空闲页列表
        else:  # 在批量释放模式中
            self.free_group.append(free_index)  # 添加到批量释放组

        if self.debug_mode:  # 调试模式下检查空闲页唯一性
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)  # 空闲页应无重复

    def clear(self):  # 清空分配器，重置所有空闲页
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        # 填充槽位0用于写入填充token的虚设输出
        self.free_pages = torch.arange(  # 初始化空闲页为1到num_pages（0号槽位保留）
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True  # 重置批量释放标志
        self.free_group = []  # 清空批量释放组
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)  # 重置待释放页为空

    def get_cpu_copy(self, indices, mamba_indices=None):  # 获取KV缓存的CPU副本
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)  # 委托给KV缓存对象

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):  # 从CPU副本加载KV缓存
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )  # 委托给KV缓存对象
