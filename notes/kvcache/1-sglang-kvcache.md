# SGLang KV Cache 机制源码导读

本文基于当前仓库源码梳理 SGLang runtime 中 KV cache 的核心机制。主线代码集中在 `python/sglang/srt/mem_cache/`，并由 `model_runner` 初始化、`scheduler`/`schedule_batch` 分配与释放、attention backend 读写真实 K/V 张量。

## 1. 总体模型：三层内存结构

`python/sglang/srt/mem_cache/memory_pool.py` 文件开头直接说明了 SGLang 的两级 memory pool，加上真实 KV 张量后可理解为三层：

1. `ReqToTokenPool`
   - 作用：把“请求 + token 位置”映射到 token-to-KV pool 的物理 slot。
   - 关键字段：`req_to_token`，形状大致为 `(max_running_requests + 1, max_context_len)`。
   - 第 0 行是 padding/dummy 行，正常请求从 slot 1 开始分配。
   - 调度和 attention backend 都会通过它知道某个请求历史 token 对应哪些 KV slot。

2. `TokenToKVPoolAllocator`
   - 作用：管理 KV slot/page 的空闲列表。
   - 普通非分页版本在 `allocator/token.py`，每个 token 一个 slot。
   - 分页版本在 `allocator/paged.py`，以 `page_size` 为单位分配，但仍返回 token 粒度的物理 index。
   - 分配前会调用 tree cache eviction，尽量先驱逐可复用但当前未锁定的 cache。

3. `KVCache`
   - 抽象类在 `memory_pool.py`，真实持有每层 attention 的 K/V buffer。
   - MHA 路径是 `MHATokenToKVPool`，维护 `k_buffer` 和 `v_buffer`。
   - MLA 路径是 `MLATokenToKVPool`，维护合并后的 `kv_buffer`。
   - 写入入口是 `set_kv_buffer(...)`，读取入口是 `get_key_buffer` / `get_value_buffer` / `get_kv_buffer`。

这三层分工很清楚：`ReqToTokenPool` 负责“请求逻辑序列到物理位置”，allocator 负责“哪些物理位置可用”，`KVCache` 负责“物理位置里的 K/V 数据”。

## 2. 初始化路径

主要入口是 `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` 的 `_init_pools`。

初始化顺序：

1. 先创建 `req_to_token_pool`
   - 普通模型使用 `ReqToTokenPool`。
   - Mamba/linear attention 混合模型使用 `HybridReqToTokenPool`。
   - PD disaggregation decode 模式使用专门的 `DecodeReqToTokenPool`。
   - DeepSeek V4 NPU 路径会替换成带额外 per-request state 表的 pool。

2. 再创建真实 `token_to_kv_pool`
   - 普通 MHA：`MHATokenToKVPool`。
   - MLA：`MLATokenToKVPool` 或 DSA 专用 pool。
   - FP4/FP8 KV cache 会选对应量化 pool。
   - SWA 混合模型使用 `SWAKVPool`，内部同时维护 full attention pool 和 sliding-window pool。
   - Mamba/linear 混合模型使用 `HybridLinearKVPool`。

3. 最后创建 allocator
   - `page_size == 1` 时使用 `TokenToKVPoolAllocator`。
   - `page_size > 1` 时使用 `PagedTokenToKVPoolAllocator`。
   - SWA 使用 `SWATokenToKVPoolAllocator`，同时管理 full 和 swa 两套 index。
   - HiSparse、NPU、DSV4 等会进一步替换 allocator。

tree/prefix cache 的选择在 `python/sglang/srt/mem_cache/registry.py`：

- radix cache disabled 且 chunked prefill 时，使用 `ChunkCache` / `SWAChunkCache`。
- 开启 unified radix tree 或 MLX 时，使用 `UnifiedRadixCache`。
- 开启 hierarchical cache 时，普通模型使用 `HiRadixCache`，混合模型走 `UnifiedRadixCache + HiCache`。
- SWA 模型使用 `SWARadixCache`。
- Mamba 模型使用 `MambaRadixCache`。
- LMCache 开启时使用 `LMCRadixCache`。
- 默认使用 `RadixCache`。

## 3. 请求进入调度：prefix 命中

请求对象 `Req` 在 `python/sglang/srt/managers/schedule_batch.py`。每轮调度前会调用 `Req.init_next_round_input(...)`：

1. 拼出当前请求需要填充的完整 token 序列：`origin_input_ids + output_ids`。
2. 用 `RadixKey(token_ids, extra_key, limit)` 调用 `tree_cache.match_prefix(...)`。
3. 把命中结果写回请求字段：
   - `prefix_indices`：命中的 device KV slot。
   - `last_node` / `last_host_node` / `best_match_node`：radix/tree 中命中的节点。
   - `host_hit_length` / `swa_host_hit_length` / `mamba_host_hit_length`：HiCache 或混合缓存命中的 host 部分。
4. 设置 `extend_input_len = input_len - len(prefix_indices)`。

调度策略中也有同样的包装函数：`python/sglang/srt/managers/schedule_policy.py::match_prefix_for_req`。它用于 cache-aware scheduling，例如 longest-prefix-match 和 dfs-weight 调度。

`RadixKey.extra_key` 很重要。它把相同 token 序列但不同 LoRA、salt、cache namespace 的 KV 隔离开，避免错误复用。

## 4. Prefill/extend：只计算未命中 suffix

`ScheduleBatch.prepare_for_extend` 是 prefill/extend 分配 KV 的主入口：

1. 取每个请求未命中的输入：
   - `input_ids = r.get_fill_ids()[len(r.prefix_indices):]`
2. 计算：
   - `prefix_lens = len(prefix_indices)`
   - `extend_lens = extend_input_len`
   - `seq_lens = fill_len`
3. 调 `alloc_for_extend(batch)` 分配 KV slot。

`alloc_for_extend` 在 `python/sglang/srt/mem_cache/common.py`：

1. `alloc_req_slots(...)` 从 `ReqToTokenPool` 分配 request row。
2. 分配 token KV slot：
   - 非分页：`alloc_token_slots(...)`。
   - 分页：`alloc_paged_token_slots_extend(...)`。
3. `write_cache_indices(...)` 写入 `req_to_token_pool.req_to_token`：
   - prefix 部分写入已命中的 `prefix_indices`。
   - extend 部分写入新分配的 `out_cache_loc`。

因此 attention backend 看到的是完整序列的 page table / index table：前缀指向旧 KV，suffix 指向本轮新分配 KV。

## 5. Decode：每步追加新 KV slot

`ScheduleBatch.prepare_for_decode` 处理 decode 阶段：

1. 根据 batch size 和 speculative decoding 配置调用 `alloc_for_decode(batch, token_per_req)`。
2. `alloc_for_decode` 从 allocator 分配每个请求下一个 token 的 slot。
3. 写入 `req_to_token_pool.req_to_token[req_pool_idx, seq_len] = out_cache_loc`。
4. 请求级 bookkeeping 增加：
   - `kv_committed_len += 1`
   - `kv_allocated_len += 1`
   - `seq_lens += 1`

分页 allocator 的 decode 分配会根据每个请求的 `last_loc` 判断是否需要新 page；如果仍在同一 page 内，返回下一 token 位置，否则消耗新 page。

## 6. Attention backend 如何读写 KV

attention backend 的通用模式是：

1. 模型层算出本轮 token 的 `k` 和 `v`。
2. backend 用 `forward_batch.out_cache_loc` 调 `token_to_kv_pool.set_kv_buffer(...)` 写入真实 KV buffer。
3. attention 计算时用 `req_to_token_pool.req_to_token` 构造 kv indices/page table。
4. 再从 `token_to_kv_pool.get_kv_buffer(layer_id)` 取真实 K/V buffer 传给 kernel。

例如 `python/sglang/srt/layers/attention/xpu_backend.py` 中：

- 写入：`self.token_to_kv_pool.set_kv_buffer(layer, KVWriteLoc(cache_loc, swa_out_cache_loc), k, v, ...)`
- 读取：`key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)`
- page table：从 `self.req_to_token_pool.req_to_token[req_pool_indices, ...]` 构造。

不同 backend 会有不同 metadata 组织方式，但核心都依赖这两个表：

- `req_to_token_pool` 给出每个请求历史 token 的 KV slot/page。
- `token_to_kv_pool` 给出这些 slot/page 对应的真实 K/V buffer。

## 7. RadixCache：prefix cache 的基本实现

默认 prefix cache 是 `python/sglang/srt/mem_cache/radix_cache.py::RadixCache`。

核心对象：

- `RadixKey`
  - 包含 token id 序列、`extra_key`、bigram 模式和 limit。
  - `page_aligned(page_size)` 会把长度裁成 page 对齐，分页 KV cache 只缓存完整 page。
  - EAGLE 场景可进入 bigram view。

- `TreeNode`
  - `key`：该边/节点代表的 token 片段。
  - `value`：对应的 device KV indices。
  - `host_value`：HiCache host 备份。
  - `lock_ref`：保护运行中的 prefix 不被驱逐。
  - `hash_value`：按 page 计算的 hash，用于 KV events / HiCache / routing。

- `match_prefix`
  - 从 root 沿 radix tree 找最长公共前缀。
  - 返回 `MatchResult(device_indices, last_device_node, last_host_node, best_match_node, ...)`。

- `insert`
  - 把 token key 和 KV indices 插入 radix tree。
  - 如果已有公共前缀，会 split node。
  - `evictable_size_` / `protected_size_` 随节点状态更新。

- `inc_lock_ref` / `dec_lock_ref`
  - 请求命中某个 prefix 后会 lock 对应路径，避免运行时被 eviction。
  - 请求结束后释放 lock，节点重新变成 evictable。

- `evict`
  - 从 `evictable_leaves` 中按策略选叶子驱逐。
  - 调 `token_to_kv_pool_allocator.free(x.value)` 释放 slot/page。

## 8. 请求结束、中断和释放

释放入口是 `python/sglang/srt/mem_cache/common.py::release_kv_cache(req, tree_cache, is_insert=True)`。

它先调用 `tree_cache.cache_finished_req(...)`：

- 如果 radix cache disabled：直接释放 committed KV indices。
- 如果 enabled 且允许插入：用请求 token 序列和 `req_to_token_pool` 中的 KV indices 插入 radix tree。
- 插入后 radix cache 接管这些 KV slot 的生命周期。
- 请求之前持有的 `last_node` lock 会 `dec_lock_ref`。

然后 `release_kv_cache` 处理没有被 radix 接管的部分：

- speculative decoding 可能预分配多余 KV，`pop_overallocated_kv_cache()` 后释放。
- page mode 会把释放起点向 page 边界对齐。
- 最后 `req_to_token_pool.free(req)` 释放 request row。

对于 chunked prefill 或被暂停的请求，`maybe_cache_unfinished_req(...)` 会调用 `tree_cache.cache_unfinished_req(...)`，把已经计算的部分也插入 prefix cache，后续继续时可以复用。

## 9. Eviction 与 OOM 前置回收

分配 KV 前会先尝试从 tree cache 驱逐：

- `alloc_token_slots(...)` 调 `evict_from_tree_cache(tree_cache, num_tokens)`。
- `alloc_paged_token_slots_extend/decode(...)` 会保守估算需要的新 page 数，再驱逐。
- 普通 allocator 看 `available_size()`。
- SWA allocator 同时看 full 和 swa 两套 pool：`full_available_size()` / `swa_available_size()`。

如果驱逐后仍不够，就抛出 prefill/decode OOM，并打印 available + evictable 信息。

锁机制保证：正在被请求使用的 prefix 属于 protected，不会被 eviction；只有 `lock_ref == 0` 的叶子/路径可被回收。

## 10. 分页 KV cache

分页由 `PagedTokenToKVPoolAllocator` 实现。

特点：

- allocator 内部维护 `free_pages`，不是单个 token slot。
- `alloc(need_size)` 要求 page aligned，返回 page 展开后的 token indices。
- `alloc_extend(...)` 根据 `prefix_lens`、`seq_lens`、`last_loc` 为每个请求补齐当前 page 或分配新 page。
- `alloc_decode(...)` 根据下一 token 是否跨 page 来决定是否消耗新 page。
- `free(...)` 会把 token indices 转成 page index 后释放。

RadixKey 在分页场景下会 `page_aligned(page_size)`，这意味着 prefix cache 只保存 page 对齐的前缀，避免缓存半页造成 page table 语义复杂化。

## 11. SWA、Mamba、HiCache 等扩展

SWA：

- 初始化时使用 `SWAKVPool` + `SWATokenToKVPoolAllocator`。
- full attention 和 sliding-window attention 有不同 pool/index。
- `free_swa_out_of_window_slots(...)` 会释放滑窗外且不在 tree cache 保护范围内的 SWA slot。
- attention backend 写 KV 时可能传 `KVWriteLoc(full_loc, swa_loc)`。

Mamba / hybrid SSM：

- 使用 `HybridReqToTokenPool` 和 `HybridLinearKVPool`。
- Mamba state 不是普通 per-token KV，而是 per-request/per-state slot。
- prefix cache 需要额外管理 mamba state，所以有 `MambaRadixCache`、`HiMambaRadixCache`、unified component。

HiCache / hierarchical cache：

- `HiRadixCache` 或 `UnifiedRadixCache.init_hicache(...)` 把 KV 从 device 扩展到 host/storage 层。
- `MatchResult` 中的 `host_hit_length` / `swa_host_hit_length` / `mamba_host_hit_length` 表示 L2 host 命中，需要 load back 到 device。
- `last_host_node` / `best_match_node` 用于定位 host/storage 的恢复锚点。
- `memory_pool_host.py` 定义多种 host KV pool，如 `MHATokenToKVPoolHost`、`MLATokenToKVPoolHost`。

UnifiedRadixCache：

- 位于 `python/sglang/srt/mem_cache/unified_radix_cache.py`。
- 把 FULL、SWA、MAMBA 等组件放到统一 radix tree/component 框架里。
- 当前 registry 在 MLX、统一 radix env、混合模型 + HiCache 时会优先走它。

## 12. 读源码建议入口

推荐按这个顺序读：

1. `python/sglang/srt/mem_cache/memory_pool.py`
   - 先理解 `ReqToTokenPool`、`KVCache`、`MHATokenToKVPool`、`MLATokenToKVPool`。

2. `python/sglang/srt/mem_cache/allocator/token.py`
   - 理解非分页 slot 分配。

3. `python/sglang/srt/mem_cache/allocator/paged.py`
   - 理解 page mode 的 extend/decode 分配。

4. `python/sglang/srt/mem_cache/common.py`
   - 看 `alloc_for_extend`、`alloc_for_decode`、`release_kv_cache`，这是调度和内存池的连接点。

5. `python/sglang/srt/mem_cache/radix_cache.py`
   - 看 `RadixKey`、`TreeNode`、`match_prefix`、`insert`、`cache_finished_req`、`evict`。

6. `python/sglang/srt/managers/schedule_batch.py`
   - 看 `Req.init_next_round_input`、`ScheduleBatch.prepare_for_extend`、`prepare_for_decode`。

7. `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
   - 看 `_init_pools` 中如何按模型/硬件/参数选择 KV pool 和 allocator。

8. 任一 attention backend
   - 例如 `python/sglang/srt/layers/attention/xpu_backend.py`，看 backend 如何用 `req_to_token` 和 `token_to_kv_pool`。

## 13. 一句话总结

SGLang 的 KV cache 不是单一缓存表，而是“请求到物理 KV slot 的映射 + KV slot allocator + 真实分层 K/V buffer + radix/tree prefix ownership”的组合。运行中请求通过 radix cache 命中已有 prefix，只为 suffix 分配和写入新 KV；请求结束后，已提交 KV slot 被 radix/tree 接管并可被后续请求复用，内存压力大时再按 eviction 策略释放未锁定的缓存节点。
