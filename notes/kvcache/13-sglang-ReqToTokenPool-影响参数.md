# SGLang `ReqToTokenPool` 影响参数分析

本文基于当前仓库源码，梳理哪些参数会影响 `ReqToTokenPool` 的初始化、容量、设备位置以及变体选择。

核心代码入口：

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/disaggregation/decode.py`
- `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py`

## 1. 直接影响 `ReqToTokenPool` 的构造参数

基础 `ReqToTokenPool` 定义在 `python/sglang/srt/mem_cache/memory_pool.py`：

```python
class ReqToTokenPool:
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        self.size = size
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )
        self.free_slots = list(range(1, self._alloc_size))
```

因此，直接影响 `ReqToTokenPool` 的参数有四个：

| 构造参数 | 影响 |
|---|---|
| `size` | 决定可分配 request row 数量；真实行数是 `size + 1`，第 0 行是 dummy row |
| `max_context_len` | 决定每个 request row 可以记录多少个 token 位置 |
| `device` | 决定 `req_to_token` 张量创建在哪个设备上 |
| `enable_memory_saver` | 决定分配 `req_to_token` 时是否进入 memory saver 区域 |

基础张量大小可以概括为：

```text
req_to_token.shape = (size + 1, max_context_len)
```

## 2. 初始化入口中的实际来源

`ReqToTokenPool` 主要在 `ModelRunner._init_pools` 中创建，入口位于 `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`：

```python
max_num_reqs = self.max_running_requests
extra_max_context_len = get_req_to_token_extra_context_len(self.server_args)

self.req_to_token_pool = ReqToTokenPool(
    size=max_num_reqs,
    max_context_len=self.model_config.context_len + extra_max_context_len,
    device=self.device,
    enable_memory_saver=self.server_args.enable_memory_saver,
)
```

也就是说：

```text
行数 = self.max_running_requests + 1
列数 = self.model_config.context_len + extra_max_context_len
```

其中 `+1` 是第 0 行 dummy row，用于 CUDA graph padding 等场景。

## 3. 影响行数的参数

### `--max-running-requests`

`--max-running-requests` 是影响 `ReqToTokenPool.size` 最直接的用户参数。

但它不是简单原样使用。最终 `self.max_running_requests` 是每个 DP worker 的值，并且会被 KV token 容量限制：

```python
requested_per_worker = max_running_requests // self.dp_size
max_num_reqs = min(requested_per_worker, token_capacity // 2)
```

因此：

- `--max-running-requests` 越大，理论上 request row 越多。
- `dp_size` 越大，每个 worker 分到的 request row 越少。
- KV cache token 容量不足时，实际值会被压低。

### `--max-total-tokens`

`--max-total-tokens` 不直接传给 `ReqToTokenPool`，但会限制 `token_capacity`。当 `token_capacity` 变小时，`self.max_running_requests` 也可能变小，从而减少 `req_to_token` 行数。

相关逻辑在 `_apply_token_constraints`：

```python
token_capacity = min(token_capacity, server_args.max_total_tokens)
```

### `--mem-fraction-static`

`--mem-fraction-static` 影响静态内存池可用显存，进而影响 profiling 得到的 `token_capacity`。它也不直接传给 `ReqToTokenPool`，但会通过 `max_total_num_tokens -> max_running_requests` 间接影响行数。

### 未显式设置 `--max-running-requests` 时的自动推导

如果用户没有显式设置 `--max-running-requests`，代码会根据 token 容量和模型 context length 估算：

```python
estimated = int(token_capacity / self.model_config.context_len * 512)
estimated = max(min(estimated, 4096), 2048)
max_num_reqs = min(estimated, token_capacity // 2)
```

所以在自动模式下：

- `token_capacity` 越大，request row 可能越多。
- `context_len` 越大，同等 token 容量下可并发 request 数越少。

## 4. 影响列数的参数

`ReqToTokenPool` 的列数来自：

```text
max_context_len = model_config.context_len + extra_max_context_len
```

### `--context-length`

`--context-length` 会影响 `model_config.context_len`。它决定每个 request row 的基础列数，也就是每个请求最多能记录多少个逻辑 token 位置。

### speculative decoding 参数

额外列数由 `get_req_to_token_extra_context_len(server_args)` 计算，位于 `python/sglang/srt/mem_cache/common.py`：

```python
extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
if (
    server_args.speculative_algorithm is not None
    and server_args.page_size > 1
    and (server_args.speculative_eagle_topk or 1) > 1
):
    extra = max(extra, get_alloc_reserve_per_decode(server_args))
return extra
```

相关参数包括：

| 参数 | 影响 |
|---|---|
| `--speculative-num-draft-tokens` | 增加每行额外列数 |
| `--speculative-adaptive` | 会改变 `max_speculative_num_draft_tokens` 的最终值 |
| `--speculative-adaptive-config` | adaptive speculative decoding 下用于解析候选 draft steps |
| `--speculative-algorithm` | 开启 speculative 分支 |
| `--speculative-eagle-topk` | 当 `page_size > 1` 且 `topk > 1` 时，可能使用更大的 reserve |
| `--page-size` | 在 speculative + `topk > 1` 场景下参与额外列数计算 |

普通非 speculative 场景下，也会保留固定的 `4` 个额外位置：

```text
extra_max_context_len = 4
```

## 5. 影响设备和分配方式的参数

### `--device`

`ModelRunner` 中：

```python
self.device = server_args.device
```

随后传给 `ReqToTokenPool`：

```python
device=self.device
```

因此 `--device` 决定 `req_to_token` 创建在 `cuda`、`npu`、`cpu` 等设备上。

### `--enable-memory-saver`

`--enable-memory-saver` 传入：

```python
enable_memory_saver=self.server_args.enable_memory_saver
```

它不改变 `req_to_token` 的 shape，但会影响分配时是否进入：

```python
memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE)
```

## 6. 影响 Pool 变体选择的参数和条件

`ReqToTokenPool` 不只有基础类。不同运行模式和模型类型会选择不同变体。

### `--disaggregation-mode decode`

当：

```python
server_args.disaggregation_mode == "decode"
```

会使用 `DecodeReqToTokenPool`，而不是基础 `ReqToTokenPool`。

它的差异是 `_alloc_size` 多了预分配 request 行：

```python
self._alloc_size = size + pre_alloc_size + 1
```

因此 decode disaggregation 下：

```text
req_to_token.shape = (max_running_requests + pre_alloc_size + 1, max_context_len)
```

### `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS`

decode disaggregation 下，`pre_alloc_size` 的来源是：

```python
pre_alloc_size = envs.SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS.get()
pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size
```

因此：

- 当 `max_num_reqs <= 32` 时，预分配行数自动为 `max_num_reqs * 2`。
- 当 `max_num_reqs > 32` 时，使用环境变量 `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS`。

### Mamba / linear attention 模型

如果 `self.mambaish_config` 存在，会使用 `HybridReqToTokenPool`。

它仍然继承基础 `req_to_token` 表，但会额外初始化 Mamba 状态池。相关参数包括：

| 参数 | 影响 |
|---|---|
| `--max-mamba-cache-size` | 影响 Mamba 状态池容量，也可能限制 `max_running_requests` |
| `--mamba-scheduler-strategy` | 决定是否启用 extra buffer 或 lazy extra buffer |
| `--disable-overlap-schedule` | 影响 Mamba ping-pong buffer 数量 |
| speculative decoding 参数 | 影响 Mamba speculative 状态 buffer |

在 Mamba 模型中，`max_running_requests` 还会被 Mamba cache 容量限制：

```python
ratio = self._calculate_mamba_ratio()
max_num_reqs = min(max_num_reqs, self.server_args.max_mamba_cache_size // ratio)
```

### decode disaggregation + Mamba

如果同时满足：

```text
disaggregation_mode == "decode"
self.mambaish_config is not None
```

会使用 `HybridMambaDecodeReqToTokenPool`。它同时具备：

- decode worker 的 `pre_alloc_size` 行扩展。
- Mamba / linear attention 的额外状态池。

### NPU + DeepSeek V4

如果当前平台是 NPU 且模型是 DeepSeek V4：

```python
if _is_npu and is_deepseek_v4(self.model_config.hf_config):
    req_to_token_pool_cls = DSV4NPUReqToTokenPool
```

会使用 `DSV4NPUReqToTokenPool`。

它在基础 `req_to_token` 之外，额外创建五张 per-request 表：

```python
req_to_token_swa
req_to_token_c4
req_to_token_c128
req_to_token_c4_state
req_to_token_c128_state
```

这些表的列数也受 `max_context_len` 影响：

```text
req_to_token_swa:        max_context_len
req_to_token_c4:         max_context_len // 4
req_to_token_c128:       max_context_len // 128
req_to_token_c4_state:   max_context_len
req_to_token_c128_state: max_context_len
```

因此在 NPU + DeepSeek V4 路径下，`context_len`、speculative 额外长度、`max_running_requests` 都会同时放大基础表和这些辅助表。

### draft worker

如果是 speculative decoding 的 draft worker，代码不会重新创建 `req_to_token_pool`：

```python
# Draft worker shares req_to_token_pool with the target worker.
assert self.is_draft_worker
```

也就是说 draft worker 共享 target worker 的 `req_to_token_pool`，避免 target / draft 对同一请求使用不同映射表。

## 7. 参数影响总表

| 参数 / 条件 | 影响对象 | 影响方式 |
|---|---|---|
| `--max-running-requests` | 行数 | 决定 `size`，再加 1 个 dummy row |
| `dp_size` | 行数 | `max_running_requests // dp_size` 后按 worker 分摊 |
| `--max-total-tokens` | 行数 | 限制 token capacity，间接限制 `max_running_requests` |
| `--mem-fraction-static` | 行数 | 影响可分配 KV token 容量，间接影响 `max_running_requests` |
| `--context-length` | 列数 | 改变 `model_config.context_len` |
| `--speculative-num-draft-tokens` | 列数 | 增加 `extra_max_context_len` |
| `--speculative-adaptive` | 列数 | 改变最大 draft token 数 |
| `--speculative-adaptive-config` | 列数 | adaptive 模式下决定候选 draft steps |
| `--speculative-algorithm` | 列数 / 变体 | 开启 speculative 相关额外空间和 draft worker 共享逻辑 |
| `--speculative-eagle-topk` | 列数 | topk > 1 且 page_size > 1 时可能增加 reserve |
| `--page-size` | 列数 | speculative + topk > 1 时参与 reserve 计算 |
| `--device` | 设备 | 决定 `req_to_token` 所在设备 |
| `--enable-memory-saver` | 分配方式 | 不改 shape，只影响分配区域 |
| `--disaggregation-mode decode` | 变体 / 行数 | 使用 `DecodeReqToTokenPool`，增加 `pre_alloc_size` 行 |
| `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS` | 行数 | decode disaggregation 下控制预分配 request 行 |
| Mamba / linear attention 模型 | 变体 / 行数 | 使用 `HybridReqToTokenPool`，Mamba cache 可能限制并发 request |
| `--max-mamba-cache-size` | 行数 / Mamba 状态池 | 限制 Mamba 状态池和最大 request 数 |
| `--mamba-scheduler-strategy` | 变体附加状态 | 控制 Mamba extra buffer |
| `--disable-overlap-schedule` | 变体附加状态 | 影响 Mamba ping-pong buffer 数量 |
| NPU + DeepSeek V4 | 变体 / 额外表 | 使用 `DSV4NPUReqToTokenPool`，额外分配 SWA/C4/C128/state 映射表 |
| speculative draft worker | 共享策略 | draft worker 共享 target worker 的 pool |

## 8. 一句话总结

`ReqToTokenPool` 最核心的容量由两条链路决定：

```text
--max-running-requests / --max-total-tokens / --mem-fraction-static / dp_size
    -> self.max_running_requests
    -> req_to_token 行数

--context-length / 模型 context_len / speculative decoding 参数
    -> max_context_len
    -> req_to_token 列数
```

此外，`--device` 决定张量所在设备，`--enable-memory-saver` 决定分配区域，`--disaggregation-mode decode`、Mamba 模型、NPU + DeepSeek V4 会改变具体使用的 `ReqToTokenPool` 变体。
