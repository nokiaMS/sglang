# SGLang `ReqToTokenPool` 源码导读

本文基于当前仓库源码梳理 `ReqToTokenPool` 的设计与运行路径。核心代码位于：

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/layers/attention/`

## 1. 它解决什么问题

`ReqToTokenPool` 是 SGLang KV cache 体系里的“请求到 token 物理位置映射表”。它不保存真实 K/V 张量，而是保存：

```text
(request slot, logical token position) -> token_to_kv_pool 中的物理 KV slot
```

也就是说，某个请求的第 `i` 个历史 token 对应哪个 KV cache 位置，不由请求对象自己维护长列表，而是统一写入 `req_to_token_pool.req_to_token` 这张二维 GPU tensor。

这张表是调度器、prefix cache、paged KV allocator 和 attention backend 之间的关键连接层：

- 调度器给每个运行中的请求分配一个 `req_pool_idx`。
- KV allocator 给新 token 分配真实 KV slot。
- `ReqToTokenPool` 记录请求逻辑序列到 KV slot 的映射。
- attention backend 通过 `req_pool_indices + seq_lens` 从这张表构造 kv indices 或 page table。

## 2. 类定义和核心字段

`ReqToTokenPool` 定义在 `python/sglang/srt/mem_cache/memory_pool.py`：

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""
```

初始化逻辑很短，但信息密度很高：

```python
self.size = size
self._alloc_size = size + 1
self.max_context_len = max_context_len
self.device = device
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len), dtype=torch.int32, device=device
)
self.free_slots = list(range(1, self._alloc_size))
```

其中这段初始化：

```python
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len), dtype=torch.int32, device=device
)
```

是在指定设备上创建一张全 0 的二维 `int32` tensor，并保存到 `self.req_to_token`。它的第一维 `self._alloc_size` 表示可用 request row 的总数，包含额外的第 0 行 dummy row；第二维 `max_context_len` 表示每个 request 最多能记录多少个 token 位置。因此可以把它理解成：

```text
self.req_to_token[request_row][token_position] = token_slot_index
```

也就是“某个 request 的第 `i` 个 token，对应 `token_to_kv_pool` / KV cache 里的哪个物理 token slot”。初始化为 0 表示还没有写入有效映射，同时也让第 0 行天然可以作为 CUDA graph padding 场景下的安全 dummy row。

关键字段含义：

- `size`：正常可运行请求数，通常来自 `max_running_requests`。
- `_alloc_size = size + 1`：真实 tensor 行数，多出来的第 0 行是 padding/dummy 行。
- `max_context_len`：每个请求最多可映射的 token 位置数，初始化时会用模型 context length 加上一些 speculative decoding 等场景需要的额外长度。
- `req_to_token`：形状为 `(size + 1, max_context_len)` 的 `int32` GPU tensor。
- `free_slots`：可用 request row 列表，从 `1` 开始；第 `0` 行不参与普通分配。

第 0 行的设计很重要。源码注释说明，CUDA graph 的 padded batch 默认会把 padded `req_pool_indices` 置为 0，因此这些 dummy 读写都会落到第 0 行，避免访问非法行。因为 `req_to_token` 初始化为 0，第 0 行天然是安全的 dummy row。

## 3. 基本接口

`ReqToTokenPool` 的公开接口主要有四个：

```python
def write(self, indices, values):
    self.req_to_token[indices] = values

def available_size(self):
    return len(self.free_slots)

def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
    ...

def free(self, req: Req):
    ...
```

### `write`

`write` 只是对 `req_to_token` 做索引赋值。这个接口看起来简单，但使用频率很高：

- extend/prefill 时写入 prefix 命中的 KV indices 和本轮新分配的 KV indices。
- decode 时为每个请求追加本轮生成 token 的 KV slot。
- 某些 backend 或特殊路径会直接读写 `req_to_token` 构造 page table。

### `available_size`

返回当前还剩多少 request row。调度器会用它限制可运行请求数。例如 scheduler 在估算还能加入多少请求时，会和 `req_to_token_pool.available_size()` 取最小值。

### `alloc`

`alloc(reqs)` 给一组请求分配 request row，并把分配结果写回每个 `Req` 的 `req_pool_idx`。

源码里有一个特殊分支：如果请求已经有 `req_pool_idx`，则复用已有 row，不再重新分配。这主要用于 chunked prefill 等跨 chunk 继续处理的场景。相关校验要求复用请求必须是正在分块处理，或者已经有 committed KV：

```python
reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
assert all(
    reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
    for i in reusing
)
```

如果新请求需要的 row 数超过 `free_slots`，`alloc` 返回 `None`，外层 `alloc_req_slots` 会抛出错误，提示调小 `--max-running-requests`。

### `free`

`free(req)` 将 `req.req_pool_idx` 放回 `free_slots`，然后把请求上的 `req_pool_idx` 置为 `None`。

注意：普通 `ReqToTokenPool.free` 不清空 `req_to_token` 对应行的数据。因为 row 已经回收到 `free_slots`，旧映射是否残留不影响语义；新请求后续会覆盖有效 token 范围。真正的 KV slot 释放由 `token_to_kv_pool_allocator` 和 prefix cache 负责。

## 4. 初始化路径

`ReqToTokenPool` 在 `ModelRunner._init_pools` 中创建，入口位于 `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`。

普通 dense/MLA 模型走标准类：

```python
self.req_to_token_pool = ReqToTokenPool(
    size=max_num_reqs,
    max_context_len=self.model_config.context_len + extra_max_context_len,
    device=self.device,
    enable_memory_saver=self.server_args.enable_memory_saver,
)
```

但 `ReqToTokenPool` 也是一组变体的基类/接口约定：

- disaggregation decode 模式使用 `DecodeReqToTokenPool`。
- Mamba/linear attention 混合模型使用 `HybridReqToTokenPool`。
- disaggregation decode + Mamba 使用 `HybridMambaDecodeReqToTokenPool`。
- DeepSeek V4 NPU 路径使用 `DSV4NPUReqToTokenPool`，额外维护 per-request 的 SWA/C4/C128 状态表。
- draft worker 会共享 target worker 的 `req_to_token_pool`，避免 speculative decoding 两套 worker 对同一请求使用不同映射表。

因此，标准 `ReqToTokenPool` 的核心语义是稳定的：维护请求行和 token 位置到 KV slot 的映射；特殊模型只是在这个基础上附加更多 per-request 状态。

## 5. Extend / prefill 阶段如何写入

extend/prefill 的主入口是 `python/sglang/srt/mem_cache/common.py::alloc_for_extend`。

整体流程：

1. `batch.maybe_evict_swa()` 先释放 sliding window attention 已经滑出窗口的 token。
2. 收集每个请求的 `prefix_indices`，这些来自 radix/prefix cache 命中结果。
3. 调用 `alloc_req_slots(batch.req_to_token_pool, batch.reqs, batch.tree_cache)` 分配 request row。
4. 调用 token allocator 为未命中的 suffix 分配 KV slot，得到 `out_cache_loc`。
5. 调用 `write_cache_indices(...)` 写入 `req_to_token_pool.req_to_token`。

`write_cache_indices` 的语义是把完整可见序列写入请求行：

```text
req_to_token[req_idx, 0:prefix_len] = prefix_indices
req_to_token[req_idx, prefix_len:seq_len] = out_cache_loc 对应片段
```

也就是说，一个请求在 prefill 后，`req_to_token[req_pool_idx, :seq_len]` 已经覆盖完整上下文：

- 前半段指向 prefix cache 命中的旧 KV slot。
- 后半段指向本轮新分配并即将写入的 KV slot。

如果 attention backend 支持 Triton，写入会走 `write_req_to_token_pool_triton`，把多请求的 prefix 和 suffix 写入合并成 GPU kernel；否则走 Python 循环逐请求调用 `ReqToTokenPool.write`。

## 6. Decode 阶段如何追加

decode 的主入口是 `python/sglang/srt/mem_cache/common.py::alloc_for_decode`。

非分页 KV cache 下，allocator 直接为 `batch_size * token_per_req` 个 token 分配连续/离散 slot。分页 KV cache 下，需要先读取每个请求上一个 token 的位置：

```python
last_loc = batch.req_to_token_pool.req_to_token[
    batch.req_pool_indices, seq_lens_gpu - 1
]
```

这个 `last_loc` 用于判断当前请求是否仍在上一页内，还是需要分配新 page。

分配完成后，decode 会把新 token 的位置写到当前序列末尾：

```python
locs = seq_lens_gpu.clone()
batch.req_to_token_pool.write(
    (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
)
```

encoder-decoder 模型会用 `encoder_lens + seq_lens_gpu` 作为写入位置。普通 decoder-only 模型则写入 `seq_lens_gpu`，也就是当前新 token 的逻辑位置。

因此 decode 阶段每步都会把请求行向右推进一格：

```text
step t: req_to_token[req_pool_idx, old_seq_len] = new_kv_slot
```

## 7. Attention backend 如何读取

`ReqToTokenPool` 的读取方主要是 attention backend 和 forward batch metadata 构造逻辑。

例如 `python/sglang/srt/layers/attention/triton_ops/kv_indices.py` 中的 Triton kernel 会按如下方式读取：

```text
req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + kv_start + offset
```

也就是用 `req_pool_idx` 选中请求行，再用 token position 选中列，读出真实 KV slot。

不同 backend 读取后的用途略有差异：

- FlashInfer/Triton 路径会构造扁平的 `kv_indices`。
- paged attention 路径会按 page boundary 读取 slot，并转换成 block/page table。
- DeepSeek V4 backend 会在 `make_core_attn_metadata` 中用 `req_to_token[req_pool_indices, :max_seq_len:page_size]` 构造 page table。
- sliding window attention 会读取窗口范围内的 token 映射，只让 kernel 看到窗口内 KV。
- chunked prefix cache 会按 chunk start 和 chunk length 从 `req_to_token` 中取出 chunk 对应 KV indices。

所以 `ReqToTokenPool` 是 attention kernel 的间接寻址表。真实 K/V 数据在 `token_to_kv_pool` 里，kernel 先通过 `req_to_token` 找到要读哪些 slot，再去 KV buffer 中取这些 slot 对应的 K/V。

## 8. 与 prefix cache / allocator 的边界

`ReqToTokenPool` 容易和 `RadixCache`、`TokenToKVPoolAllocator` 混淆。三者边界如下：

| 组件 | 负责内容 | 保存什么 |
|------|----------|----------|
| `ReqToTokenPool` | 当前运行请求的逻辑 token 到物理 KV slot 映射 | `req_to_token` 二维 int32 tensor |
| `TokenToKVPoolAllocator` | KV slot/page 的空闲、分配、释放 | free list、page 状态、allocator 元数据 |
| `KVCache` / token-to-kv pool | 真实 K/V 张量存储 | 每层 K/V buffer |
| `RadixCache` / prefix cache | 已完成或可复用 prefix 的树形索引 | token key 到 KV indices 的树节点 |

prefill 时，prefix cache 命中的 `prefix_indices` 会被写入 `ReqToTokenPool`；未命中的 suffix 由 allocator 新分配 slot，也写入 `ReqToTokenPool`。请求结束时，prefix cache 决定哪些 KV slot 插入树中继续复用，哪些 overallocated 或未保留 slot 释放回 allocator。最后 request row 通过 `ReqToTokenPool.free(req)` 释放。

## 9. 释放路径

请求完成或中止时，通常会走 `python/sglang/srt/mem_cache/common.py::release_kv_cache`。

该函数先调用：

```python
tree_cache.cache_finished_req(req, is_insert=...)
```

这一步会根据 prefix cache 策略处理已完成请求的 KV：可能插入 radix tree，也可能释放不需要保留的 KV。

随后处理 speculative decoding 或 strip thinking cache 产生的 overallocated KV：

```python
indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
    start_p:end_p
]
tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
```

最后调用：

```python
tree_cache.req_to_token_pool.free(req)
```

这一步只释放 request row，不释放整行里所有 KV slot。因为哪些 KV slot 能释放、哪些要进入 prefix cache，是 `tree_cache` 和 allocator 的职责。

## 10. 设计要点

`ReqToTokenPool` 的设计有几个明显取舍：

1. **GPU 常驻映射表**
   `req_to_token` 是 GPU tensor，attention metadata 构造和 Triton kernel 可以直接读取，避免每步从 CPU 传大表。

2. **请求 row 与请求对象解耦**
   `Req` 只持有一个 `req_pool_idx`，历史 token 的物理位置存在全局池中。这样 batch 可以用 `req_pool_indices` 做批量 gather。

3. **第 0 行作为 dummy row**
   CUDA graph / padded batch 场景可以把无效请求指向 0，降低图捕获和 padding 逻辑复杂度。

4. **只管理映射，不管理 KV 生命周期**
   这个类不负责真实 KV 内存，也不负责 prefix cache 复用策略。它只回答“这个请求的这个 token 位置对应哪个 KV slot”。

5. **支持 row 复用**
   chunked prefill 等场景中，请求可能已经有 `req_pool_idx`，后续 chunk 继续复用同一行，保证长请求跨多轮 prefill 时映射表连续。

6. **为特殊模型保留扩展点**
   Mamba、DSV4 NPU、disaggregation decode 都通过子类或替代类扩展 request-level 状态，但 attention/KV 主流程仍围绕 `req_pool_idx` 和 `req_to_token` 展开。

## 11. 一句话总结

`ReqToTokenPool` 是 SGLang runtime 中连接“请求逻辑序列”和“物理 KV cache slot”的 GPU 端二维索引表。它本身不存 KV，也不决定 cache 复用策略；它把 scheduler、allocator、prefix cache 和 attention backend 串起来，使每个请求在 prefill、decode、chunked prefill、paged attention、CUDA graph padding 等场景下都能稳定找到自己的历史 KV。
