# KV缓存放置事件发射混入类（Mixin）
# 提供BlockStored/BlockRemoved/AllBlocksCleared事件的记录功能，
# 这些事件由KV感知路由器（如dynamo）消费，用于跟踪KV缓存的存储和淘汰。

# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV cache placement event emission mixin.

The mixin produces the ``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared``
events consumed by KV-aware routers (e.g. dynamo).
"""

from typing import Any

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    hash_str_to_int64,
)


class KVCacheEventMixin:
    """KV缓存事件混入类，提供缓存块的存储、移除和清空事件记录功能。"""

    def _record_store_event(self, node: Any, medium=None):
        """记录缓存块存储事件，每个page_size大小的块生成一个BlockStored事件。"""
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if self.enable_kv_cache_events:
            # 默认存储介质为GPU
            if medium is None:
                medium = StorageMedium.GPU

            # Compute hash_value lazily if not already set
            # 延迟计算节点的哈希值（如果尚未设置）
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            # 获取父节点最后一个哈希值作为当前第一个页面的父块哈希
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    # 取父节点最后一个哈希值并转换为int64
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            logical_len = len(node.key)
            is_bigram = node.key.is_bigram
            raw = node.key.token_ids
            # 按page_size分页遍历，每页生成一个BlockStored事件
            for start in range(0, logical_len, self.page_size):
                end = min(start + self.page_size, logical_len)
                if end <= start:
                    continue
                # Preserve historical event payload: bigram pages expose tuples.
                # 二元组模式：每个token表示为(token, next_token)元组
                if is_bigram:
                    page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
                else:
                    page_tokens = list(raw[start:end])

                # 将当前页的哈希值转换为int64
                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=medium,
                    )
                )

                # 当前块的哈希值成为下一块的父块哈希
                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: Any, medium=None):
        """记录缓存块移除事件，每个page_size大小的块生成一个BlockRemoved事件。"""
        # One BlockRemoved per chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        if self.enable_kv_cache_events:
            # 默认存储介质为GPU
            if medium is None:
                medium = StorageMedium.GPU

            # Compute hash_value lazily if not already set (must match what was stored)
            # 延迟计算哈希值，必须与存储时计算的哈希值一致
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            logical_len = len(node.key)
            # 按page_size分页遍历，每页生成一个BlockRemoved事件
            for start in range(0, logical_len, self.page_size):
                end = min(start + self.page_size, logical_len)
                if end <= start:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=medium)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        """记录所有缓存块清空事件。"""
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        # 原子性地取出所有事件并清空队列
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
