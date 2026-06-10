# DeepSeek V4 Pro 请求调度挂起问题分析

## 问题描述

### 环境信息

- 模型：DeepSeek-V4-Pro
- 部署：2 节点 H100，tp=16
- 关键启动参数：

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
CUTE_DSL_LOG_LEVEL=40 \
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 172.16.48.80:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port 8080 \
  --moe-dense-tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --page-size 64 \
  --disable-cuda-graph \
  --context-length 1048576 \
  --max-running-requests 64 \
  --log-level debug \
  --log-level-http debug
```

### 现象

使用 `sglang.bench_serving` 进行基准测试：

- `--random-input-len 12000`：正常工作，服务端正常打印 prefill 和 decode 日志
- `--random-input-len 18000`：服务端卡住不动，仅打印到 `Using regular tokenizer for 1 inputs` 后无任何输出

正常输出（12000 tokens）：
```
[2026-06-05 17:17:10] INFO:     172.16.97.53:37764 - "GET /v1/models HTTP/1.1" 200 OK
[2026-06-05 17:17:12] INFO:     172.16.97.53:37768 - "POST /generate HTTP/1.1" 200 OK
[2026-06-05 17:17:12] Using regular tokenizer for 1 inputs
[2026-06-05 17:17:12 TP0] Prefill batch, #new-seq: 1, #new-token: 256, #cached-token: 3584, full token usage: 0.06, swa token usage: 0.60, ...
[2026-06-05 17:17:19 TP0] Decode batch, #running-req: 1, #full token: 3840, full token usage: 0.06, ...
```

异常输出（18000 tokens）：
```
[2026-06-05 17:18:20] INFO:     172.16.97.53:38676 - "GET /v1/models HTTP/1.1" 200 OK
[2026-06-05 17:18:21] INFO:     172.16.97.53:38680 - "POST /generate HTTP/1.1" 200 OK
[2026-06-05 17:18:21] Using regular tokenizer for 1 inputs
（卡住，无后续输出）
```

### 尝试过的修复（均未解决）

1. `--chunked-prefill-size 32768 --swa-full-tokens-ratio 0.2`：服务仍然卡住
   - 测试输出显示 `#Input tokens: 15726`（小于 32768，不应触发分块）
   - 服务端日志仍然卡在 `Using regular tokenizer for 1 inputs`

---

## 分析过程

### 1. 初步分析：定位卡住位置

服务端在 `Using regular tokenizer for 1 inputs` 之后卡住，说明 tokenizer 处理完成，请求已进入调度器，但调度器未能将请求送入 prefill 执行。

### 2. 初步假设：Chunked Prefill 死锁（后证实为不完整）

最初认为问题是 chunked prefill 的 `add_chunked_req` 在 hybrid SWA 场景下的死锁。

**分析依据**：
- `--chunked-prefill-size 16384`，DSV4 hook 将 page_size 设为 256
- 12000 tokens → 12032（< 16384），不触发分块
- 18000 tokens → 18176（> 16384），触发分块
- `add_chunked_req` 在 SWA 预算不足时直接 return req 不加到 can_run_list（line 679-681）

**但此假设不完整**：即使设置 `--chunked-prefill-size 32768`（15726 < 32768，走非分块路径），服务仍然卡住。说明问题不止在 chunked prefill。

### 3. 深入分析：`_swa_budget_for_req` 对所有请求都过度估算

`_swa_budget_for_req`（`schedule_policy.py:543-558`）对**所有请求**（包括非分块请求）都做同样过度保守的预算估算：

```python
def _swa_budget_for_req(self, extend_input_len: int) -> int:
    if self.rem_chunk_tokens is not None:
        alloc = min(extend_input_len, self.rem_chunk_tokens)
    else:
        alloc = extend_input_len  # ← 非分块时，alloc = 全部 extend_input_len
    return max(alloc, self.tree_cache.sliding_window_size) + self.page_size
```

对于 15726 token 的非分块请求（`sliding_window_size=128`, `page_size=256`）：
- `alloc = 15726`（因为 `rem_chunk_tokens is None`）
- `swa_needed = max(15726, 128) + 256 = 15982`

**但实际 SWA 只需 `sliding_window_size=128` 个 token 的 KV 空间！** 因为 prefill 后 `maybe_evict_swa()` 会立即释放超出窗口的 SWA KV。预算高估了约 **124 倍**。

### 4. `add_one_req` 中 SWA 预算检查拒绝请求

`add_one_req`（`schedule_policy.py:857-860`）的 SWA 预算检查：

```python
if self.is_hybrid_swa:
    swa_needed = self._swa_budget_for_req(req.extend_input_len)  # = 15982
    if swa_needed >= self.rem_swa_tokens:                         # SWA pool 不够则拒绝
        return AddReqResult.NO_TOKEN                              # ← 无日志，静默拒绝
```

`rem_swa_tokens`（line 516-521）：
```python
rem_swa_tokens = swa_available_size() + swa_evictable_size() - rem_swa_token_offset
```

如果 SWA pool 总容量不足以通过 `swa_needed = 15982` 的检查，请求就被静默拒绝。

### 5. `_update_prefill_budget` 的预算扣除也是过度估算

即使请求通过了预算检查并被调度，`_update_prefill_budget`（line 582-596）也会按过度估算值扣除 offset：

```python
def _update_prefill_budget(self, prefix_len, extend_input_len, max_new_tokens):
    ...
    if self.is_hybrid_swa:
        self.rem_swa_token_offset += self._swa_budget_for_req(extend_input_len)  # 扣除 15982
```

这意味着第一个请求的 SWA 预算扣除就消耗了 15982 个 SWA token 的 offset，即使实际只用了 128 个。后续请求的 `rem_swa_tokens` 会迅速耗尽。

### 6. 为什么 12000 tokens 能成功但 15726/18000 不能

- 12000 tokens：`swa_needed = max(12000, 128) + 256 = 12256`
- 15726 tokens：`swa_needed = max(15726, 128) + 256 = 15982`
- 18000 tokens：`swa_needed = max(18176, 128) + 256 = 18432`

如果 SWA pool 总容量介于 12256 和 15982 之间（例如 `swa_full_tokens_ratio=0.1` 且 full_pool 不够大时），12000 能通过但更大的请求不能。即使 `swa_full_tokens_ratio=0.2`，如果 full_pool 本身因为 swa_ratio 增大而缩小，SWA pool 可能仍然不够。

### 7. DSV4 HiSparse 分配器的额外瓶颈

`DeepSeekV4HiSparseTokenToKVPoolAllocator.full_available_size()`（`hisparse_memory_pool.py:587-591`）：

```python
def full_available_size(self):
    return min(
        self.logical_attn_allocator.full_available_size(),
        self.hisparse_attn_allocator.available_size() * self.compress_ratio,
    )
```

取逻辑分配器和 HiSparse 分配器的**最小值**。如果 HiSparse 压缩池（c4/c128）空间不足，即使 full pool 有空间，`rem_total_tokens` 也会很小，导致 full pool 预算检查也失败。

### 8. 死锁循环（适用于所有场景，不仅是 chunked）

1. 请求进入 `waiting_queue`
2. `add_one_req` 因 SWA 预算不足返回 `NO_TOKEN`（或因 full pool 预算不足返回 `NO_TOKEN`）
3. `can_run_list` 为空，`_get_new_batch_prefill_raw` 返回 `None`（line 2708-2709，**无日志**）
4. 调度器 `event_loop_normal`/`event_loop_overlap` 调用 `on_idle()` 后再次循环
5. 再次尝试 `add_one_req`，同样失败
6. **无限循环，无任何日志输出**

### 9. `--swa-full-tokens-ratio 0.2` 为什么没解决问题

`swa_full_tokens_ratio=0.2` 虽然增加了 SWA pool，但同时也增加了 `bytes_per_full_token`（因为 swa_ratio 项翻倍），导致 full pool 缩小约 35-40%。full pool 缩小后：

- `swa_tokens = full_pool × 0.2`，full_pool 变小，swa_tokens 可能仍然不够
- `full_available_size()` 因 HiSparse 最小值约束可能更小
- 形成恶性循环：增大 swa_ratio → full_pool 缩小 → SWA pool 也缩小 → 问题依旧

---

## SWA Pool 机制详解

### 什么是 SWA

SWA = **Sliding Window Attention**（滑动窗口注意力），是一种只对序列最近 N 个 token 做 attention 的机制，而非对全部历史 token 做 attention。标准 attention 的计算量和内存与序列长度平方成正比，SWA 将注意力范围限制在最近 `window_size` 个 token 内，将复杂度从 O(L²) 降到 O(L×W)。

### DeepSeek V4 Pro 的注意力架构

对于 DeepSeek V4 Pro，**所有层**都使用滑动窗口（`window_size=128`），部分层额外有 c4/c128 压缩注意力来"记忆"更早的 token。这与 Gemma2 等模型不同——Gemma2 部分层是 full attention，部分层是 SWA；而 DSV4 没有层做全量注意力，全部依赖滑动窗口+压缩。

DSV4 每层的 `compress_ratio` 取值：
- **0**：SWA-only（仅滑动窗口 128 tokens，无压缩注意力）
- **4**：SWA + c4 压缩注意力（4x 压缩 KV + 索引器）
- **128**：SWA + c128 压缩注意力（128x 压缩 KV）

### Full Pool 与 SWA Pool 的关系

```
序列: [t0, t1, t2, ..., t95, t96, t97, t98, t99]   (假设 window_size=128)
                   ↑                              ↑
              可从SWA释放                    SWA pool 保留区

Full pool:  [t0, t1, t2, ..., t99]  ← 存所有 token 的索引映射 (req_to_token)
SWA pool:  [t0, t1, ..., t99]       ← 实际只保留最近128个token的KV数据
                                   ↑ 更早的token的SWA KV会被释放(tombstone)
```

两者共享同一棵 radix tree，但数据可以独立驱逐：
- **Full pool**：存储所有 token 的索引映射，只能驱逐叶节点
- **SWA pool**：只存储最近 128 个 token 的 K/V，可以驱逐内部节点的 SWA 数据（标记为 tombstone），同时保留 full pool 数据

关键不变式：`full_lock_ref >= swa_lock_ref`（full 锁定时 swa 可能锁定也可能不锁定，但 swa 锁定时 full 必须锁定）。

### SWA Pool 的生命周期

1. **Prefill 阶段**：分配 full + SWA 双份内存，所有 token 的 KV 写入 SWA pool
2. **Prefill 完成后**：`maybe_evict_swa()` 释放超出滑动窗口的 SWA KV
3. **Decode 阶段**：每步 forward 后调用 `maybe_evict_swa()`，释放超出滑动窗口的旧 SWA KV
4. **SWA 提前解锁**：decode 超过 `window_size` 步后，prefill 时锁定的 SWA tree lock 被释放（`dec_swa_lock_only`），允许其他请求复用这些 SWA 槽位

### Tombstone 机制

当 SWA KV 被释放但 full KV 仍被锁定时，节点被标记为 `swa_tombstone=True`：
- 节点保留在 radix tree 中（full KV 对 full-attention 层仍有效）
- SWA KV 数据已被释放
- 节点从 `swa_lru_list` 中移除
- 后续前缀匹配将 tombstone 节点视为不连续点

### DSV4 的 6 个内存池

| 池 | 大小 | 存什么 |
|---|---|---|
| **full** | 基准 | 所有 token 的索引映射 |
| **swa** | `full × swa_full_tokens_ratio` | 最近 128 token 的 K/V（所有层） |
| **c4** | `full / 4` | 4x 压缩 K/V（c4 层用） |
| **c128** | `full / 128` | 128x 压缩 K/V（c128 层用） |
| **c4_state** | 基于 swa | c4 压缩器的运行状态（ring buffer） |
| **c128_state** | 基于 swa | c128 压缩器的运行状态 |

### Hybrid SWA 在调度器中的含义

调度器必须同时管理**两个独立的内存预算**：full pool 和 SWA pool。请求只有在两个池都有足够容量时才能被准入。

双预算追踪（`PrefillAdder`）：
- `rem_total_tokens`：`full_available + full_evictable - offset`
- `rem_swa_tokens`：`swa_available + swa_evictable - offset`

每个请求的 SWA 预算：`max(alloc, sliding_window_size) + page_size`（`_swa_budget_for_req`）

### 与挂起问题的关联

`_swa_budget_for_req` 对非分块请求使用 `alloc = extend_input_len`（如 15726），但实际 SWA 只需 `sliding_window_size=128` 个 token。预算高估约 124 倍，导致 SWA 预算检查即使在 SWA pool 有大量实际空闲空间时也失败。

---

## 结论

### 根因

**`_swa_budget_for_req` 对所有请求（包括非分块请求）的 SWA 预算过度估算**，导致调度器在 SWA pool 有充足实际空间时仍然拒绝请求。

具体机制：
1. `_swa_budget_for_req(15726)` 返回 `max(15726, 128) + 256 = 15982`，但实际 SWA 只需 128+256 = 384 个 token
2. `add_one_req` 的 SWA 预算检查 `swa_needed >= rem_swa_tokens` 失败，返回 `NO_TOKEN`
3. `_get_new_batch_prefill_raw` 因 `can_run_list` 为空返回 `None`（**无日志**）
4. 请求永久卡在 `waiting_queue`，无限循环

### 次要问题

1. **Chunked prefill 死锁**：`add_chunked_req` 在 `is_hybrid_swa` 且 `_rem_tokens <= 0` 时直接返回请求对象（line 679-681），不加到 `can_run_list`，导致 chunked 请求永久卡在 `chunked_req` 状态
2. **`_update_prefill_budget` 的过度扣除**：预算扣除也使用过度估算值，即使请求被调度，后续请求的 `rem_swa_token_offset` 也会迅速累积
3. **缺失日志**：`_get_new_batch_prefill_raw` 在 `can_run_list` 为空时返回 `None` 无任何日志，使问题极难排查

### 触发条件

- DeepSeek V4 Pro 模型（hybrid SWA）
- 输入 token 数较大（使 `swa_needed = max(extend_input_len, sliding_window_size)` 超过 `rem_swa_tokens`）
- SWA pool 容量不足以通过过度估算的预算检查

### 为什么之前的修复无效

| 修复 | 为什么无效 |
|---|---|
| `--chunked-prefill-size 32768` | 避免了 chunked prefill 死锁，但非分块请求仍受 `_swa_budget_for_req` 过度估算影响 |
| `--swa-full-tokens-ratio 0.2` | 增大了 SWA pool，但同时也使 `bytes_per_full_token` 增加，full pool 缩小约 35-40%，SWA pool = `full_pool × 0.2` 也随之缩小，可能仍不够 |
| `--disable-radix-cache` | 切换到 SWAChunkCache，`swa_evictable_size()=0`，`rem_swa_tokens` 更小，问题更严重 |

---

## 解决方案

### 方案 1：临时验证——设置 `--swa-full-tokens-ratio 1.0`

```bash
--swa-full-tokens-ratio 1.0
```

设为 1.0 让 SWA pool 等于 full pool，确保 SWA 预算检查不成为瓶颈。如果此参数能解决问题，就确认了 SWA 预算过度估算是根因。

**代价**：`bytes_per_full_token` 大幅增加（swa_ratio 项从 0.1 变为 1.0，增加 9 倍），full pool 容量可能下降 70%+，严重影响并发能力。仅用于验证，不推荐生产使用。

### 方案 2：根治——修复 `_swa_budget_for_req`（推荐）

预算应基于 SWA 的**稳态**需求，而非 prefill 期间的瞬时峰值。因为 prefill 完成后 `maybe_evict_swa` 会立即释放超出窗口的 SWA KV，稳态 SWA 占用仅为 `sliding_window_size`：

```python
def _swa_budget_for_req(self, extend_input_len: int) -> int:
    # SWA pool only needs sliding_window_size tokens in steady state,
    # because maybe_evict_swa frees tokens outside the window after prefill.
    # During prefill, the peak SWA occupancy is extend_input_len, but this
    # is transient and freed before the next scheduling decision.
    return self.tree_cache.sliding_window_size + self.page_size
```

**注意**：这个修改需要确认在 overlap scheduler 下是否安全。在 overlap 模式下，prefill 和 decode 可能同时运行，prefill 的瞬时 SWA 占用可能与 decode 的 SWA 占用叠加。此时可能需要保留一定的余量。

更安全的修复（考虑 overlap scheduler）：

```python
def _swa_budget_for_req(self, extend_input_len: int) -> int:
    if self.rem_chunk_tokens is not None:
        alloc = min(extend_input_len, self.rem_chunk_tokens)
    else:
        # For non-chunked prefill, the SWA pool only needs to hold
        # the sliding window in steady state. The prefill-time peak
        # (extend_input_len tokens) is transient and freed by
        # maybe_evict_swa before the next scheduling iteration.
        alloc = self.tree_cache.sliding_window_size
    return max(alloc, self.tree_cache.sliding_window_size) + self.page_size
```

### 方案 3：修复 `add_chunked_req` 死锁（解决 chunked prefill 场景）

在 `schedule_policy.py:679-681`，当 `is_hybrid_swa` 且 `_rem_tokens <= 0` 时，应该和非 SWA 路径一样回退到 `rem_chunk_tokens`：

```python
if _rem_tokens <= 0:
    if self.is_hybrid_swa:
        _rem_tokens = self.rem_chunk_tokens  # 回退而非死锁
    else:
        _rem_tokens = self.rem_chunk_tokens
```

### 方案 4：添加调度拒绝日志（辅助排查）

在 `add_one_req` 的 SWA 预算检查处（line 857-860）添加日志：

```python
if self.is_hybrid_swa:
    swa_needed = self._swa_budget_for_req(req.extend_input_len)
    if swa_needed >= self.rem_swa_tokens:
        logger.debug(
            f"SWA budget rejected: swa_needed={swa_needed}, "
            f"rem_swa_tokens={self.rem_swa_tokens}, "
            f"extend_input_len={req.extend_input_len}"
        )
        return AddReqResult.NO_TOKEN
```

同样在 `add_chunked_req` 的 early return 处（line 679-681）添加日志。

### 方案对比

| 方案 | 解决范围 | full pool 影响 | 风险 |
|---|---|---|---|
| `--swa-full-tokens-ratio 1.0` | 临时验证 | 下降约 70%+ | 仅用于验证 |
| 修复 `_swa_budget_for_req` | 根治所有场景 | 无影响 | 需验证 overlap scheduler 安全性 |
| 修复 `add_chunked_req` | 仅解决 chunked 死锁 | 无影响 | 可能引入 OOM |
| 添加日志 | 辅助排查 | 无影响 | 无 |

---

## 无效方案分析

### `--disable-radix-cache`：不能解决，反而更糟

`--disable-radix-cache` 将 tree cache 从 `SWARadixCache` 切换为 `SWAChunkCache`（`registry.py:82-89`）：

| | SWARadixCache | SWAChunkCache |
|---|---|---|
| 前缀缓存 | 支持（radix tree 匹配+锁定） | 不支持（`match_prefix` 始终返回空） |
| `swa_evictable_size()` | 有可驱逐的 SWA token | 始终返回 0 |
| `full_evictable_size()` | 有可驱逐的 full token | 始终返回 0 |
| `inc_lock_ref` | 锁定树节点 | 空操作 |
| KV 内存回收 | 通过 LRU 驱逐空闲节点的 KV | 仅在请求完成时释放 |

`rem_swa_tokens = swa_available_size() + swa_evictable_size() - offset`：
- SWAChunkCache 的 `swa_evictable_size() = 0`，`rem_swa_tokens` 更小
- 没有 LRU 驱逐，SWA 槽位直到请求完成才释放
- 问题更严重

额外代价：失去前缀缓存、内存利用率更低。

### `--chunked-prefill-size 32768` 单独使用：不能完全解决

避免了 chunked prefill 死锁，但非分块请求仍受 `_swa_budget_for_req` 过度估算影响。只要 `extend_input_len` 超过 `rem_swa_tokens`，请求就会被拒绝。

### `--swa-full-tokens-ratio 0.2` 单独使用：可能不够

增大 SWA pool，但同时也使 `bytes_per_full_token` 增加，full pool 缩小。SWA pool = `full_pool × 0.2`，如果 full_pool 缩小幅度大，SWA pool 可能仍不够。

---

## 排查步骤建议

1. **查看启动日志中的 pool 大小**：找到 `DSV4 pool sizes: full=XXX, swa=XXX` 行，确认 SWA pool 是否小于输入 token 的 `_swa_budget_for_req` 估算值
2. **临时验证**：设置 `--swa-full-tokens-ratio 1.0`，确认问题消失，锁定根因
3. **添加调试日志**：在 `add_one_req` 的 SWA 预算检查处和 `add_chunked_req` 的 early return 处添加 debug 日志，输出 `swa_needed` 和 `rem_swa_tokens` 的值
4. **修复代码**：根据验证结果，修复 `_swa_budget_for_req` 的预算估算逻辑

---

## 涉及的关键代码文件

| 文件 | 作用 |
|---|---|
| `python/sglang/srt/managers/schedule_policy.py` | PrefillAdder 调度逻辑，**核心 bug 位置**（`_swa_budget_for_req` line 543-558, `add_one_req` SWA 检查 line 857-860, `add_chunked_req` 死锁 line 666-681, `_update_prefill_budget` 过度扣除 line 594-595） |
| `python/sglang/srt/managers/scheduler.py` | 调度器主循环（`_get_new_batch_prefill_raw` line 2548, 无日志返回 None line 2708-2709） |
| `python/sglang/srt/managers/schedule_batch.py` | `maybe_evict_swa()` SWA 驱逐逻辑（line 2646-2740） |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | DeepSeek V4 默认参数设置（page_size=256, swa_ratio=0.1） |
| `python/sglang/srt/model_executor/pool_configurator.py` | DSV4 内存池大小计算（`DSV4PoolConfigurator` line 309-468） |
| `python/sglang/srt/mem_cache/hisparse_memory_pool.py` | HiSparse 分配器（`full_available_size` 取最小值 line 587-591，可能成为额外瓶颈） |
| `python/sglang/srt/mem_cache/swa_radix_cache.py` | SWA radix cache 实现（tombstone 机制、SWA LRU 驱逐） |
| `python/sglang/srt/mem_cache/chunk_cache.py` | SWAChunkCache 实现（disable-radix-cache 时使用） |
| `python/sglang/srt/mem_cache/registry.py` | Cache 工厂路由（根据参数选择 SWARadixCache 或 SWAChunkCache） |
| `python/sglang/srt/mem_cache/swa_memory_pool.py` | SWA 内存池分配器（`swa_available_size`, `full_available_size`） |
| `python/sglang/srt/mem_cache/kv_cache_builder.py` | KV cache 构建入口 |
| `python/sglang/srt/configs/deepseek_v4.py` | DeepSeek V4 模型配置（`window_size=128`） |
| `python/sglang/srt/configs/model_config.py` | `sliding_window_size` 配置读取 |
| `python/sglang/srt/models/deepseek_v4.py` | DSV4 模型层实现（MQALayer、KV 写入 SWA pool） |
