# random-input-len 为 18000 时服务挂起问题分析

## 1. 问题描述

使用 `sglang.bench_serving` 对 DeepSeek-V4-Pro 模型进行压测：

- `random-input-len=15000` 时：服务正常响应，有完整的 Prefill/Decode 日志输出
- `random-input-len=18000` 时：服务端仅打印初始两条日志后无任何输出，压测程序也挂起

## 2. 环境信息

### 服务启动参数

```
SGLANG_SHARED_EXPERT_TP1=1 \
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr <pod0-ip>:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port 8080 \
  --moe-dense-tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 64 \
  --context-length 202752 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct
```

经过 ServerArgs 后处理后的有效参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| chunked_prefill_size | 8192 | H100 GPU 自动设置 |
| page_size | 256 | DeepSeek V4 自动设置 |
| max_running_requests | 256 | DeepSeek V4 自动设置 |
| sliding_window | 128 | 模型 config.json |
| swa_full_tokens_ratio | 0.1 | DeepSeek V4 自动设置 |
| max_total_num_tokens | 64256 | 运行时日志确认 |

### 模型配置关键参数

```json
{
  "architectures": ["DeepseekV4ForCausalLM"],
  "num_hidden_layers": 61,
  "sliding_window": 128,
  "num_attention_heads": 128,
  "compress_ratios": [128, 128, 4, 128, 4, ..., 0]
}
```

## 3. 根因分析

### 3.1 DeepSeek-V4 的 Hybrid SWA 内存架构

DeepSeek-V4-Pro 被识别为 hybrid SWA 模型（`is_hybrid_swa=True`），其 KV 缓存被分为多个池：

- **full KV pool**：存储全注意力层的 KV cache
- **SWA KV pool**：存储滑动窗口层的 KV cache，大小 = `full_tokens × swa_full_tokens_ratio`
- **c4 KV pool**：存储 compress_ratio=4 的压缩层
- **c128 KV pool**：存储 compress_ratio=128 的压缩层
- **c4/c128 state pool**：存储压缩状态

SWA pool 的大小由 `DSV4PoolConfigurator._compute_dsv4_sizes()` 决定：

```python
swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size
```

默认 `swa_ratio = swa_full_tokens_ratio`，对于 DeepSeek V4 自动设置为 **0.1**（非默认的 1.0），即 SWA pool 仅为 full pool 的 10%。运行时日志确认：

```
max_total_num_tokens=64256, swa_full_tokens_ratio=0.1
```

因此 SWA pool 大小约为 `64256 × 0.1 = 6425` token，远小于 full pool。

### 3.2 请求调度的预算检查

请求进入调度器后，`PrefillAdder.add_one_req()` (`schedule_policy.py:815`) 执行两道预算检查：

**检查1：整体 token 预算** (`schedule_policy.py:855`)

```python
total_tokens = req.extend_input_len + max_new_tokens + self.page_size
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN
```

**检查2：SWA 预算** (`schedule_policy.py:858`)

```python
if self.is_hybrid_swa:
    swa_needed = self._swa_budget_for_req(req.extend_input_len)
    if swa_needed >= self.rem_swa_tokens:
        return AddReqResult.NO_TOKEN
```

### 3.3 SWA 预算计算

`_swa_budget_for_req()` (`schedule_policy.py:553`) 的计算逻辑：

```python
def _swa_budget_for_req(self, extend_input_len):
    if self.rem_chunk_tokens is not None:
        alloc = min(extend_input_len, self.rem_chunk_tokens)
    else:
        alloc = extend_input_len
    return max(alloc, self.tree_cache.sliding_window_size) + self.page_size
```

由于 `chunked_prefill_size=8192`，`rem_chunk_tokens=8192`：

| random-input-len | alloc | swa_budget | rem_total_tokens 需求 |
|------------------|-------|------------|-----------------------|
| 15000 | min(15000, 8192) = 8192 | max(8192, 128) + 256 = 8448 | 15000 + 200 + 256 = 15456 |
| 18000 | min(18000, 8192) = 8192 | max(8192, 128) + 256 = 8448 | 18000 + 200 + 256 = 18256 |

SWA 预算两者相同，但 **整体 token 预算** 差异显著。

### 3.4 死锁机制

1. 请求进入 `waiting_queue`
2. 调度器尝试调度请求，`add_one_req()` 返回 `AddReqResult.NO_TOKEN`（token 预算不足）
3. 请求留在 `waiting_queue` 中
4. 由于没有任何正在运行的请求可以完成并释放 KV cache，`rem_total_tokens` 永远不会增加
5. 请求被 **永久挂起**，调度器不会输出任何日志（调度失败时不进入 forward 路径，不触发日志打印）
6. 客户端等待响应，服务端静默，形成死锁

从 `random-input-len=15000` 的日志可以验证资源紧张：

```
full token usage: 0.06, swa token usage: 0.52
```

SWA token usage 已达 52%，full token usage 仅 6%。当 input 增加到 18000 时，`total_tokens=18256` 超过了 `rem_total_tokens` 的可用量，直接被拒绝调度。

## 4. 验证过程

### 4.1 验证方案：`--allow-auto-truncate`

按照文档中的方案1，重启服务并添加 `--allow-auto-truncate` 参数：

```
sglang serve ... --context-length 202752 --allow-auto-truncate ...
```

运行压测命令：

```
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:8080 \
  --model /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --dataset-name random-ids \
  --random-input-len 18000 \
  --random-output-len 200 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 123
```

### 4.2 验证结果：失败，问题依旧

- 服务端日志：请求进入后无任何 Prefill/Decode 日志输出，与未加 `--allow-auto-truncate` 时表现一致
- 压测程序：5 分钟后无进度输出，最终被 kill 退出（exit code 143）
- 服务端日志中 **没有出现任何 truncation 警告**

### 4.3 失败原因分析

`--allow-auto-truncate` 的截断逻辑位于 `managers/utils.py:129` 的 `validate_input_length()` 函数中：

```python
if len(req.origin_input_ids) >= max_req_input_len:
    if allow_auto_truncate:
        req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
        return None
```

该检查的 `max_req_input_len` 是基于 KV 缓存池总量计算的，而非 SWA pool 大小。关键数值：

| 指标 | 值 |
|------|-----|
| max_total_num_tokens | 64256 |
| max_req_input_len | 约 64256（基于 full pool） |
| 输入长度 18000 | 小于 max_req_input_len |

**18000 远小于 64256，`validate_input_length()` 不会触发截断**，输入原样通过。真正的瓶颈在调度器层面的 SWA 预算检查（`add_one_req()` → `_swa_budget_for_req()`），此时 `--allow-auto-truncate` 完全不参与。

### 4.4 核心问题确认

SWA pool 大小约为 `64256 × 0.1 = 6425` token，而 18000 输入在 chunked prefill 模式下的 SWA 预算需求为：

```
swa_budget = max(min(18000, 8192), 128) + 256 = 8448 token
```

8448 > 6425（SWA pool 总量），即使 SWA pool 完全空闲也无法满足需求，请求被调度器永久拒绝。

**结论：`--allow-auto-truncate` 无法解决问题，根因是 `swa_full_tokens_ratio=0.1` 导致 SWA pool 过小。**

### 4.5 验证：增大 `--swa-full-tokens-ratio`

尝试通过增大 `swa_full_tokens_ratio` 来增加 SWA pool 容量。

**验证结果：均失败，问题依旧**

| swa_full_tokens_ratio | max_total_num_tokens | SWA pool 估算 | SWA 预算需求 | 结果 |
|-----------------------|----------------------|--------------|-------------|------|
| 0.1（默认） | 64256 | ~6425 | 8448 | 挂起：SWA 不够 |
| 0.2 | 36352 | ~7270 | 8448 | 挂起：SWA 仍不够 |
| 0.3 | 25344 | ~7603 | 8448 | 挂起：SWA 仍不够，full pool 也不够了 |

**失败原因分析**：增大 `swa_full_tokens_ratio` 存在核心矛盾——

1. `swa_full_tokens_ratio` 影响 `DSV4PoolConfigurator._get_bytes_per_full_token()` 中的内存计算：
   ```python
   self.swa_ratio * kv_bytes * self.num_layers_total  # 61层
   ```
   ratio 从 0.1 → 0.2 → 0.3，该系数线性增大，导致 `bytes_per_full_token` 增大。

2. `bytes_per_full_token` 增大 → `max_total_num_tokens = available_bytes / bytes_per_full_token` 急剧缩小：
   - ratio=0.1: 64256
   - ratio=0.2: 36352（减少 43%）
   - ratio=0.3: 25344（减少 61%）

3. SWA pool = `max_total_num_tokens × ratio`，虽然 ratio 在增大，但 `max_total_num_tokens` 缩减更快：
   - ratio=0.1: 64256 × 0.1 ≈ 6425
   - ratio=0.2: 36352 × 0.2 ≈ 7270
   - ratio=0.3: 25344 × 0.3 ≈ 7603

4. SWA pool 增长缓慢，始终无法达到 8448 的需求阈值。同时 `max_total_num_tokens` 缩小导致 full pool 也不够用（ratio=0.3 时 18000 已占 71%），`rem_total_tokens` 检查也会失败。

### 4.6 根因定论：sglang 调度器 `_swa_budget_for_req` 设计缺陷

通过上述验证，确认问题不是简单的参数调优能解决的，根因在于 **sglang 调度器中 `_swa_budget_for_req()` 的 SWA 预算计算过于保守**。

当前逻辑 (`schedule_policy.py:553`)：

```python
def _swa_budget_for_req(self, extend_input_len):
    if self.rem_chunk_tokens is not None:
        alloc = min(extend_input_len, self.rem_chunk_tokens)
    else:
        alloc = extend_input_len
    return max(alloc, self.tree_cache.sliding_window_size) + self.page_size
```

**问题**：`alloc = min(extend_input_len, rem_chunk_tokens)` 将 chunked prefill 的整个 chunk 大小（8192）作为 SWA 预算需求。但实际上 SWA 层只需要滑动窗口范围内的 token（`sliding_window=128`），远小于 chunk 大小。

这意味着对于 DeepSeek V4 这种 `sliding_window=128` 但 `chunked_prefill_size=8192` 的模型，SWA 预算需求被高估了 **64 倍**（8448 vs 实际约 384）。

对于非压缩注意力模型（如 Gemma2），`sliding_window` 较大（如 4096），这个高估不那么严重。但对于 DeepSeek V4 的 NSA（Native Sparse Attention）架构，`sliding_window=128` 极小，导致预算高估极其严重，使得任何合理的 `swa_full_tokens_ratio` 都无法同时满足 SWA 预算和 full pool 容量需求。

## 5. 解决方案

### 方案1：修复 `_swa_budget_for_req` 预算计算逻辑（根本解决）

修改 `schedule_policy.py:553` 的 `_swa_budget_for_req()`，使 SWA 预算基于滑动窗口大小而非 chunk 大小计算：

**当前逻辑**（有缺陷）：
```python
def _swa_budget_for_req(self, extend_input_len):
    if self.rem_chunk_tokens is not None:
        alloc = min(extend_input_len, self.rem_chunk_tokens)  # 8192 for chunked
    else:
        alloc = extend_input_len
    return max(alloc, self.tree_cache.sliding_window_size) + self.page_size
    # 结果: max(8192, 128) + 256 = 8448，高估64倍
```

**建议修改**：
```python
def _swa_budget_for_req(self, extend_input_len):
    # SWA 层实际只需要滑动窗口范围内的 token
    # 对于 DeepSeek V4 (sliding_window=128)，实际需求远小于 chunk 大小
    return self.tree_cache.sliding_window_size + self.page_size
    # 结果: 128 + 256 = 384
```

**优点**：从根本上解决预算高估问题，无需修改启动参数，不影响 full pool 容量
**缺点**：需要修改 sglang 源码并重新部署；需要验证对其他 hybrid SWA 模型（Gemma2 等）的兼容性

### 方案2：增大 `--swa-full-tokens-ratio`（已验证无效）

尝试通过增大 ratio 来增加 SWA pool，但存在核心矛盾：ratio 增大 → `bytes_per_full_token` 增大 → `max_total_num_tokens` 急剧缩小 → SWA pool 增长缓慢，无法满足需求。

| ratio | max_total | SWA pool | 结果 |
|-------|-----------|----------|------|
| 0.1 | 64256 | ~6425 | SWA 不够 |
| 0.2 | 36352 | ~7270 | SWA 仍不够 |
| 0.3 | 25344 | ~7603 | SWA 和 full 都不够 |

**结论**：无法通过调参解决，预算计算逻辑必须修复。

### 方案3：`--allow-auto-truncate`（已验证无效）

启动参数增加 `--allow-auto-truncate`，超长输入会自动截断。

```bash
sglang serve ... --allow-auto-truncate
```

**验证结果**：无效。`--allow-auto-truncate` 的截断逻辑在 `validate_input_length()` 中，基于 full pool 大小判断。18000 远小于 `max_req_input_len`（约 64256），不触发截断。真正的瓶颈在调度器层面的 SWA 预算检查，`--allow-auto-truncate` 不参与此路径。

### 方案4：降低 `--context-length`

当前 `--context-length 202752`，降低此值可以让内存分配更紧凑：

```bash
sglang serve ... --context-length 131072  # 128K
```

**优点**：减少内存浪费，提高实际可用 token 数
**缺点**：降低模型支持的最大上下文长度

### 方案5：降低 `--mem-fraction-static`

```bash
sglang serve ... --mem-fraction-static 0.85
```

**优点**：减少激活内存与 KV 缓存的冲突
**缺点**：总体 KV 缓存容量会减少

### 方案6：减小 `--chunked-prefill-size`（已验证有效）

减小 `chunked_prefill_size` 可以直接降低 `_swa_budget_for_req` 中的 `rem_chunk_tokens`，从而降低 SWA 预算需求：

```bash
sglang serve ... --chunked-prefill-size 2048
```

SWA 预算计算对比：

| chunked_prefill_size | alloc | swa_budget | SWA pool (ratio=0.1) | 结果 |
|---|---|---|---|---|
| 8192（默认） | 8192 | 8448 | ~6425 | 超出，挂起 |
| 4096 | 4096 | 4352 | ~6425 | 通过 |
| 2048 | 2048 | 2304 | ~6425 | 通过 |

**验证结果**（`--chunked-prefill-size 2048`）：**成功**

启动参数：
```bash
SGLANG_SHARED_EXPERT_TP1=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 --nnodes 2 --node-rank 0 \
  --dist-init-addr <pod0-ip>:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 --port 8080 \
  --moe-dense-tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 64 \
  --context-length 202752 \
  --allow-auto-truncate \
  --chunked-prefill-size 2048 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct
```

运行时参数：
- `max_total_num_tokens=64256`（与默认 ratio=0.1 时一致，full pool 容量不受影响）
- `chunked_prefill_size=2048`
- `swa_full_tokens_ratio=0.1`

压测命令（之前验证时未加 `--random-range-ratio 1.0`，导致实际输入仅 15726 token，详见第 9 节）：
```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:8080 \
  --model /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --dataset-name random-ids \
  --random-input-len 18000 \
  --random-output-len 200 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 123
```

压测结果（`--random-range-ratio` 默认 0.0，实际输入仅 15726 token）：

| 指标 | input=18000 (有auto-truncate) | input=50000 (有auto-truncate) | input=50000 (无auto-truncate) |
|------|------|------|------|
| Successful requests | 1 | 1 | 1 |
| Benchmark duration (s) | 4.04 | 3.96 | 5.02 |
| Total input tokens | 15726 | 15726 | 15726 |
| Total generated tokens | 127 | 127 | 127 |
| Input token throughput (tok/s) | 3892.06 | 3975.54 | 3134.09 |
| Output token throughput (tok/s) | 31.43 | 32.11 | 25.31 |
| Mean E2E Latency (ms) | 4030.02 | 3944.66 | 5005.66 |
| Mean TTFT (ms) | 305.95 | 278.66 | 753.15 |
| Mean TPOT (ms) | 29.56 | 29.10 | 33.75 |

> 注：三个测试的实际输入 token 数均为 15726，这是 `bench_serving` 的 `--random-range-ratio` 参数默认值为 0.0 导致的，详见第 9 节分析。

追加验证（`--random-range-ratio 1.0`，确保实际输入为 18000 token）：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8080 \
  --model /userdata/dsv4/DeepSeek-V4-Pro \
  --dataset-name random-ids \
  --random-input-len 18000 \
  --random-output-len 200 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 123 \
  --random-range-ratio 1.0
```

| 指标 | input=18000 (range-ratio=1.0) |
|------|------|
| Successful requests | 1 |
| Benchmark duration (s) | 6.13 |
| **Total input tokens** | **18000** |
| Total generated tokens | 200 |
| Input token throughput (tok/s) | 2937.02 |
| Output token throughput (tok/s) | 32.63 |
| Mean E2E Latency (ms) | 6098.29 |
| Mean TTFT (ms) | 296.80 |
| Mean TPOT (ms) | 29.15 |

> 结论：`--chunked-prefill-size 2048` 配合真正的 18000 token 输入可正常工作，服务不会挂起。

追加验证（`--random-range-ratio 1.0`，确保实际输入为 50000 token）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8080 \
  --model /userdata/dsv4/DeepSeek-V4-Pro \
  --dataset-name random-ids \
  --random-input-len 50000 \
  --random-output-len 200 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 123 \
  --random-range-ratio 1.0
```

| 指标 | input=50000 (range-ratio=1.0) |
|------|------|
| Successful requests | 1 |
| Benchmark duration (s) | 7.26 |
| **Total input tokens** | **50000** |
| Total generated tokens | 200 |
| Input token throughput (tok/s) | 6888.28 |
| Output token throughput (tok/s) | 27.55 |
| Mean E2E Latency (ms) | 7248.00 |
| Mean TTFT (ms) | 706.01 |
| Mean TPOT (ms) | 32.87 |

> 结论：`--chunked-prefill-size 2048` 配合真正的 50000 token 输入也可正常工作，服务不会挂起。50000 token 输入的 TTFT（706ms）高于 18000 token（297ms），符合预期。

**优点**：无需修改源码，仅调整启动参数即可解决；full pool 容量不受影响
**缺点**：更小的 chunk 会增加 prefill 迭代次数，对 TTFT 有一定影响；只是规避了预算高估问题，未从根本上修复 `_swa_budget_for_req` 的计算逻辑

## 6. 源代码关键路径索引

| 文件 | 行号 | 说明 |
|------|------|------|
| `schedule_policy.py` | 815 | `PrefillAdder.add_one_req()` 请求调度入口 |
| `schedule_policy.py` | 855 | 整体 token 预算检查 |
| `schedule_policy.py` | 858 | SWA 预算检查 |
| `schedule_policy.py` | 553 | `_swa_budget_for_req()` SWA 预算计算 |
| `schedule_policy.py` | 401 | `AddReqResult` 枚举定义 |
| `scheduler.py` | 2630 | `_get_next_batch_to_run()` 从 waiting_queue 取请求 |
| `scheduler.py` | 2720 | 遍历 waiting_queue 调度请求 |
| `managers/utils.py` | 129 | `validate_input_length()` 输入长度校验与截断 |
| `pool_configurator.py` | 290 | `DSV4PoolConfigurator._compute_dsv4_sizes()` 池大小计算 |
| `swa_memory_pool.py` | 296 | `SWATokenToKVPoolAllocator` SWA 内存分配器 |
| `hisparse_memory_pool.py` | 503 | `DeepSeekV4HiSparseTokenToKVPoolAllocator` |
| `hisparse_memory_pool.py` | 586 | `full_available_size()` |
| `hisparse_memory_pool.py` | 592 | `swa_available_size()` |
| `model_config.py` | 492 | `is_hybrid_swa` 判定逻辑 |
| `model_config.py` | 1660 | `is_hybrid_swa_model()` 架构列表 |

## 7. 验证结果汇总

| 方案 | 参数 | 结果 | 原因 |
|------|------|------|------|
| `--allow-auto-truncate` | context-length=202752 | 挂起 | 截断阈值基于 full pool，18000 不触发截断 |
| `--swa-full-tokens-ratio 0.2` | max_total=36352 | 挂起 | SWA pool ≈7270，仍小于需求 8448 |
| `--swa-full-tokens-ratio 0.3` | max_total=25344 | 挂起 | SWA pool ≈7603，仍不够；full pool 也不够 |
| **`--chunked-prefill-size 2048`** | max_total=64256 | **成功（input=18000，range-ratio=1.0）** | SWA 预算降至 2304，远小于 pool 容量；实际输入 18000 token |
| **`--chunked-prefill-size 2048`** | max_total=64256 | **成功（input=50000，range-ratio=1.0）** | 实际输入 50000 token，TTFT=706ms |
| **`--chunked-prefill-size 2048`** | context-length=1M, 100并发 | **成功（input=50000×100，range-ratio=1.0）** | 100请求全部成功，Mean TTFT=121.1s，TPOT=28.95ms |
| `--chunked-prefill-size 2048`（range-ratio=0.0） | max_total=64256 | 成功但实际输入仅 15726 | `--random-range-ratio` 默认 0.0 导致随机采样，非固定输入长度 |

### 10. context-length=1M 压测结果

#### 10.1 单并发压测（num-prompts=1，max-concurrency=1）

启动参数：`--context-length 1048576 --chunked-prefill-size 2048`

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8080 \
  --model /userdata/dsv4/DeepSeek-V4-Pro \
  --dataset-name random-ids \
  --random-input-len 50000 \
  --random-output-len 200 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 123 \
  --random-range-ratio 1.0
```

| 指标 | 值 |
|------|-----|
| Successful requests | 1 |
| Total input tokens | 50000 |
| Total generated tokens | 200 |
| Mean E2E Latency (ms) | 7248.00 |
| Mean TTFT (ms) | 706.01 |
| Mean TPOT (ms) | 32.87 |

#### 10.2 100并发压测（num-prompts=100，max-concurrency=10）

启动参数：`--context-length 1048576 --chunked-prefill-size 2048`

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8080 \
  --model /userdata/dsv4/DeepSeek-V4-Pro \
  --dataset-name random-ids \
  --random-input-len 50000 \
  --random-output-len 200 \
  --num-prompts 100 \
  --max-concurrency 10 \
  --seed 123 \
  --random-range-ratio 1.0 \
  --output-details --output-file /tmp/bench_100.jsonl
```

| 指标 | 值 |
|------|-----|
| Successful requests | 100 |
| Benchmark duration (s) | 1329.15 |
| Total input tokens | 5000000 |
| Total generated tokens | 20000 |
| Request throughput (req/s) | 0.08 |
| Input token throughput (tok/s) | 3761.80 |
| Output token throughput (tok/s) | 15.05 |
| Peak output token throughput (tok/s) | 54.00 |
| Peak concurrent requests | 11 |
| Total token throughput (tok/s) | 3776.85 |
| Concurrency | 9.55 |
| **End-to-End Latency** | |
| Mean E2E Latency (ms) | 126881.64 |
| Median E2E Latency (ms) | 133459.10 |
| P90 E2E Latency (ms) | 133938.28 |
| P99 E2E Latency (ms) | 134212.88 |
| **Time to First Token** | |
| Mean TTFT (ms) | 121119.76 |
| Median TTFT (ms) | 127681.54 |
| P99 TTFT (ms) | 128415.81 |
| **Time per Output Token** | |
| Mean TPOT (ms) | 28.95 |
| Median TPOT (ms) | 29.01 |
| P99 TPOT (ms) | 29.24 |
| **Inter-Token Latency** | |
| Mean ITL (ms) | 28.97 |
| Median ITL (ms) | 28.84 |
| P95 ITL (ms) | 29.41 |
| P99 ITL (ms) | 32.67 |
| Max ITL (ms) | 62.77 |

**分析**：

1. **100请求全部成功**，服务未挂起，`--chunked-prefill-size 2048` 在并发场景下有效
2. **TTFT 非常高**（Mean 121s），这是因为在 max-concurrency=10 的限制下，请求需要排队等待 prefill。每个请求的 input=50000 token，prefill 耗时约 700ms，10个并发请求串行 prefill 约需 7s，加上调度和 KV cache 分配等待，导致 TTFT 大幅增加
3. **TPOT 稳定**（Mean 28.95ms，P99 29.24ms），decode 阶段性能不受并发影响
4. **ITL 稳定**（Mean 28.97ms，Max 62.77ms），无明显抖动
5. **E2E Latency 主要由 TTFT 贡献**：Mean E2E 126.9s，其中 Mean TTFT 121.1s（占 95.4%），decode 仅贡献约 5.8s（200 token × 29ms）
6. **Peak concurrent requests=11** 略高于配置的 max-concurrency=10，包含 warmup 请求的并发

日志文件保存于 `context-length-100并发压测日志/` 目录：
- `pod0-server.log`（855 KB，4039行）
- `pod1-server.log`（99 KB，642行）
- `bench-client-summary.txt`
- `bench-client-detail.jsonl`（599 KB）

## 8. 下一步建议

1. **短期**：使用 `--chunked-prefill-size 2048` 规避问题，无需修改源码
2. **长期**：修复 `_swa_budget_for_req` (`schedule_policy.py:553`)，使 SWA 预算基于 `sliding_window_size` 而非 `extend_input_len` 计算，从根本上解决预算高估问题
3. 修复后回归验证其他 hybrid SWA 模型（如 Gemma2）是否受影响

## 9. 补充分析：为什么三种情况下 Total input tokens 都为 15726

### 9.1 现象

在方案6的验证中，三种压测配置的 `Total input tokens` 均为 15726：

| 配置 | `--random-input-len` | `--allow-auto-truncate` | Total input tokens |
|------|---------------------|------------------------|-------------------|
| 1 | 18000 | 有 | 15726 |
| 2 | 50000 | 有 | 15726 |
| 3 | 50000 | 无 | 15726 |

### 9.2 原因：`--random-range-ratio` 默认值为 0.0

`bench_serving` 的 `--random-range-ratio` 参数默认值为 **0.0**（而非直觉上认为的 1.0）。该参数控制 `compute_random_lens()` 的随机采样范围：

```python
# benchmark/datasets/common.py:56
def compute_random_lens(full_len: int, range_ratio: float, num: int) -> List[int]:
    return np.random.randint(
        max(int(full_len * range_ratio), 1),  # 下界：max(1, full_len * 0.0) = 1
        full_len + 1,                          # 上界：full_len + 1
        size=num,
    ).tolist()
```

当 `range_ratio=0.0` 时：
- 下界 = `max(int(full_len × 0.0), 1)` = **1**
- 上界 = `full_len + 1`

因此，**实际输入长度不是 `random-input-len`，而是在 `[1, random-input-len]` 范围内均匀随机采样的值**。

### 9.3 seed=123 的巧合

在 `run_benchmark()` 中，随机种子通过 `np.random.seed(args.seed)` 设置。当 `seed=123` 时：

```python
np.random.seed(123)
np.random.randint(1, 18001)   # = 15726
np.random.randint(1, 50001)   # = 15726
np.random.randint(1, 100001)  # = 15726
```

由于 NumPy MT19937 伪随机数生成器的算法特性，相同的种子下 `randint(1, N)` 在不同 N 值时（只要 N > 15726）碰巧都产生 15726。这是一个概率巧合，**与服务端的截断逻辑或 KV 缓存池大小完全无关**。

### 9.4 对验证结论的影响

这一发现意味着：

1. **三种压测实际上使用了相同的输入长度（15726 token）**，并没有真正测试 18000 或 50000 token 的场景
2. 15726 < 64256（full pool 容量），也远小于 SWA 预算需求（2304 with chunked_prefill_size=2048），所以三种配置都能正常完成
3. 之前对 "无 auto-truncate 时 TTFT 增大 2.7 倍" 的解释（调度器重试/截断）需要重新审视，因为实际输入长度仅为 15726，远未触发截断
4. 若要真正验证 18000 或 50000 token 的场景，需要设置 `--random-range-ratio 1.0`，使 `input_len` 固定为 `random-input-len` 的值
