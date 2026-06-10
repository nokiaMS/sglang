# 分块缓存模块（ChunkCache）：当 RadixCache（基数缓存）被禁用时使用的 KV 缓存实现。
# 本文件实现了不进行前缀匹配的简单分块缓存，主要用于标准分块预填充（chunked-prefill）
# 以及 P/D 分离式推理中未启用解码端基数缓存的场景。

from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


logger = logging.getLogger(__name__)


# ChunkCache：不进行前缀匹配的分块缓存，用于禁用 RadixCache 的场景
class ChunkCache(BasePrefixCache):
    """
    ChunkCache is used when radix cache is disabled.

    That includes standard chunked-prefill setups and the decode side of P/D
    disaggregation when decode radix cache is not enabled.
    """

    # 初始化分块缓存，设置请求到 token 的映射池和 KV 缓存分配器
    def __init__(self, params: CacheInitParams):
        self.req_to_token_pool = params.req_to_token_pool  # 请求到 token 索引的映射池
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator  # token 到 KV 缓存的内存分配器
        self.page_size = params.page_size  # 页大小
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")  # 无分配器时默认使用 CPU

        self.protected_size_ = 0  # 受保护的大小，分块缓存中始终为 0

    # 标识本缓存为分块缓存类型
    def is_chunk_cache(self) -> bool:
        return True

    # NOTE (csy): this is to determine if a cache has prefix matching feature.
    # Chunk cache always return True to indicate no prefix matching.
    # TODO (csy): Using a prefix cache trait to replace this
    # disable 属性：返回 True 表示禁用前缀匹配功能
    @property
    def disable(self):
        return True

    # 重置缓存（分块缓存无需重置操作）
    def reset(self):
        pass

    # 前缀匹配：分块缓存不支持前缀匹配，始终返回空结果
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),  # 空的设备端索引张量
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    # 插入操作：分块缓存不支持前缀缓存，插入为空操作
    def insert(self, params: InsertParams) -> InsertResult:
        # ChunkCache does not support prefix caching, so insert is a no-op
        return InsertResult(prefix_len=0)  # 返回前缀长度为 0

    # 缓存已完成的请求：释放该请求占用的 KV 缓存内存
    def cache_finished_req(self, req: Req, is_insert: bool = True):
        kv_committed_len = req.pop_committed_kv_cache()  # 获取已提交的 KV 缓存长度并从请求中移除
        # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]  # 取出该请求对应的 KV 缓存 token 索引
        self.token_to_kv_pool_allocator.free(kv_indices)  # 释放这些索引对应的 KV 缓存槽位

    # 缓存未完成的请求：将当前已填充的 token 索引保存到请求中，供后续 PrefillAdder 使用
    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]  # 取出该请求已填充的 KV 缓存 token 索引
        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # 将索引拷贝并保存到请求的 prefix_indices 中，供后续 PrefillAdder 使用
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    # 驱逐操作：分块缓存无主动驱逐，返回空结果
    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()

    # 增加锁引用计数：分块缓存不使用引用计数，返回增量为 0
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        return IncLockRefResult(delta=0)

    # 减少锁引用计数：分块缓存不使用引用计数，返回增量为 0
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        return DecLockRefResult(delta=0)

    # 受保护大小：分块缓存中无受保护区域，驱逐与请求生命周期一致
    def protected_size(self):
        # NOTE: no protected size in chunk cache. Chunk cache's eviction is the same with request's lifecycle.
        return 0

    # 格式化打印缓存状态：分块缓存无可打印内容
    def pretty_print(self):
        return ""


# SWAChunkCache：支持滑动窗口注意力的分块缓存
class SWAChunkCache(ChunkCache):
    """ChunkCache with support for sliding window attention."""

    # 初始化滑动窗口分块缓存，验证分配器类型并设置窗口参数
    def __init__(self, params: CacheInitParams):
        # DeepSeek V4 HiSparse wraps SWATokenToKVPoolAllocator and exposes the same API.
        # 验证分配器必须是滑动窗口类型或 DeepSeek V4 HiSparse 类型
        assert isinstance(
            params.token_to_kv_pool_allocator,
            (
                SWATokenToKVPoolAllocator,
                DeepSeekV4HiSparseTokenToKVPoolAllocator,
            ),
        )
        super().__init__(params)

        self.sliding_window_size = params.sliding_window_size  # 滑动窗口大小
        self.chunked_prefill_size = params.chunked_prefill_size  # 分块预填充大小

    # 标识本缓存支持滑动窗口注意力
    def supports_swa(self) -> bool:
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for SWAChunkCache"
        return True

    # 驱逐操作：滑动窗口分块缓存同样无主动驱逐，返回空结果
    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()
