# 令牌级KV池分配器
# 实现了基于单个token槽位的内存分配策略（page_size=1）
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

from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch张量库

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator  # 导入基础分配器

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.mem_cache.memory_pool import KVCache  # 导入KV缓存类型


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):  # 令牌到KV池分配器（page_size=1）
    """An allocator managing the indices to kv cache data."""  # 管理KV缓存数据索引的分配器

    def __init__(  # 初始化方法
        self,
        size: int,  # 池的总大小（槽位数）
        dtype: torch.dtype,  # 数据类型
        device: str,  # 设备类型
        kvcache: KVCache,  # KV缓存对象
        need_sort: bool,  # 是否需要在释放时排序
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)  # 调用父类初始化，page_size固定为1
        self.clear()  # 初始化空闲页列表

    def clear(self):  # 清空分配器，重置所有空闲页
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        # 填充槽位0用于写入填充token的虚设输出
        self.free_pages = torch.arange(  # 初始化空闲页为1到size（0号槽位保留）
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True  # 重置批量释放标志
        self.free_group = []  # 清空批量释放组
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)  # 重置待释放页为空

    def available_size(self):  # 计算可用空间大小
        # To avoid minor "len(free_pages) * 1" overhead
        # 为避免微小的"len(free_pages) * 1"开销（page_size=1时无需乘法）
        return len(self.free_pages) + len(self.release_pages)  # 直接返回空闲页数加待释放页数

    def alloc(self, need_size: int):  # 分配指定大小的空间
        if self.need_sort and need_size > len(self.free_pages):  # 需要排序且空闲页不足
            self.merge_and_sort_free()  # 合并并排序空闲页

        if need_size > len(self.free_pages):  # 空闲页不足
            return None  # 返回None表示分配失败

        select_index = self.free_pages[:need_size]  # 取出需要的索引
        self.free_pages = self.free_pages[need_size:]  # 更新空闲页列表
        return select_index  # 返回选中的索引

    def free(self, free_index: torch.Tensor):  # 释放指定索引的空间
        if free_index.numel() == 0:  # 如果没有需要释放的索引
            return  # 直接返回

        if self.is_not_in_free_group:  # 不在批量释放模式中
            if self.need_sort:  # 需要排序模式
                self.release_pages = torch.cat((self.release_pages, free_index))  # 加入待释放页列表
            else:  # 不需要排序模式
                self.free_pages = torch.cat((self.free_pages, free_index))  # 直接加入空闲页列表
        else:  # 在批量释放模式中
            self.free_group.append(free_index)  # 添加到批量释放组

    def get_cpu_copy(self, indices, mamba_indices=None):  # 获取KV缓存的CPU副本
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)  # 委托给KV缓存对象

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):  # 从CPU副本加载KV缓存
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )  # 委托给KV缓存对象
