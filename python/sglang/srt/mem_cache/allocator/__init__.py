# 令牌到KV槽位分配器模块初始化文件
# 本模块提供不同内存分配策略的实现，每种策略对应一个文件
"""Token-to-KV-slot allocators. One file per allocation strategy."""  # 令牌到KV槽位分配器，每种分配策略一个文件

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator  # 导入基础分配器抽象类
from sglang.srt.mem_cache.allocator.paged import (  # 导入分页分配器相关类和函数
    PagedTokenToKVPoolAllocator,  # 分页令牌到KV池分配器
    alloc_extend_naive,  # 朴素扩展分配函数
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator  # 导入令牌级分配器

__all__ = [  # 模块公开导出列表
    "BaseTokenToKVPoolAllocator",  # 基础分配器抽象类
    "PagedTokenToKVPoolAllocator",  # 分页分配器
    "TokenToKVPoolAllocator",  # 令牌级分配器
    "alloc_extend_naive",  # 朴素扩展分配函数
]
