# 令牌到KV池分配器基类
# 定义了内存分配器的抽象接口，包括分配、释放、状态备份恢复等通用操作
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

import abc  # 导入抽象基类模块
from typing import TYPE_CHECKING  # 导入类型检查常量

import torch  # 导入PyTorch张量库

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.mem_cache.memory_pool import KVCache  # 导入KV缓存类型


class BaseTokenToKVPoolAllocator(abc.ABC):  # 令牌到KV池分配器抽象基类
    @abc.abstractmethod
    def __init__(  # 抽象初始化方法
        self,
        size: int,  # 池的总大小（槽位数）
        page_size: int,  # 每页的大小
        dtype: torch.dtype,  # 数据类型
        device: str,  # 设备类型（如"cuda"）
        kvcache: KVCache,  # KV缓存对象
        need_sort: bool,  # 是否需要在释放时排序
    ):
        self.size = size  # 保存池总大小
        self.page_size = page_size  # 保存页大小
        self.dtype = dtype  # 保存数据类型
        self.device = device  # 保存设备类型
        self._kvcache = kvcache  # 保存KV缓存引用
        self.need_sort = need_sort  # 保存是否需要排序标志

        self.free_pages = None  # 空闲页列表
        self.release_pages = None  # 待释放页列表（排序模式下使用）
        self.is_not_in_free_group = True  # 是否不在批量释放组中
        self.free_group = []  # 批量释放组中的待释放索引列表

    @property
    def size_full(self):  # 属性：池的总大小
        return self.size  # 返回总大小

    def debug_print(self) -> str:  # 调试打印方法
        return ""  # 基类返回空字符串

    def available_size(self):  # 计算可用空间大小
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size  # 空闲页加待释放页乘以页大小

    def get_kvcache(self):  # 获取KV缓存对象
        return self._kvcache  # 返回KV缓存引用

    def restore_state(self, state):  # 恢复分配器状态
        self.free_pages, self.release_pages = state  # 从状态元组恢复空闲页和待释放页

    def backup_state(self):  # 备份分配器状态
        return (self.free_pages, self.release_pages)  # 返回空闲页和待释放页的元组

    def free_group_begin(self):  # 开始批量释放组
        self.is_not_in_free_group = False  # 标记进入批量释放模式
        self.free_group = []  # 清空批量释放组

    def free_group_end(self):  # 结束批量释放组
        self.is_not_in_free_group = True  # 标记退出批量释放模式
        if self.free_group:  # 如果批量释放组中有数据
            self.free(torch.cat(self.free_group))  # 将所有索引拼接后一次性释放

    def merge_and_sort_free(self):  # 合并并排序空闲页
        if len(self.release_pages) > 0:  # 如果有待释放页
            self.free_pages = torch.cat((self.free_pages, self.release_pages))  # 合并空闲页和待释放页
            self.free_pages, _ = torch.sort(self.free_pages)  # 对合并后的空闲页排序
            self.release_pages = torch.empty(  # 重置待释放页为空张量
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):  # 获取KV缓存的CPU副本
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        # 待修复：分页分配器实现后复用此方法
        raise NotImplementedError()  # 抛出未实现异常

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):  # 从CPU副本加载KV缓存
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        # 待修复：分页分配器实现后复用此方法
        raise NotImplementedError()  # 抛出未实现异常

    def alloc_extend(self, *args, **kwargs):  # 扩展分配方法
        raise NotImplementedError("alloc_extend is only for paged allocator")  # 仅分页分配器支持

    def alloc_decode(self, *args, **kwargs):  # 解码分配方法
        raise NotImplementedError("alloc_decode is only for paged allocator")  # 仅分页分配器支持

    @abc.abstractmethod
    def clear(self):  # 抽象方法：清空分配器
        raise NotImplementedError()  # 抛出未实现异常

    @abc.abstractmethod
    def alloc(self, need_size: int):  # 抽象方法：分配指定大小的空间
        raise NotImplementedError()  # 抛出未实现异常

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):  # 抽象方法：释放指定索引的空间
        raise NotImplementedError()  # 抛出未实现异常
