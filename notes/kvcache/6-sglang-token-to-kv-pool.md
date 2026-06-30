# SGLang `token_to_kv_pool` 源码导读

本文基于当前仓库源码梳理 SGLang runtime 中 `token_to_kv_pool` 的机制。这里的 `token_to_kv_pool` 不是一个单独的类名，而是 `ModelRunner` 上持有的物理 KV cache pool 实例，具体类型会根据模型结构、attention backend、KV cache dtype、平台和 SWA/Mamba/DSA 等特性选择。

核心代码位置：

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/mem_cache/allocator/token.py`
- `python/sglang/srt/mem_cache/allocator/paged.py`
- `python/sglang/srt/mem_cache/swa_memory_pool.py`
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `python/sglang/srt/layers/attention/`

## 1. 总体定位

`memory_pool.py` 文件开头直接说明了 SGLang 的三层 KV cache 结构：

```text
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
```

因此需要把三个对象区分开：

| 层级 | 典型对象 | 作用 |
|------|----------|------|
| 请求到 token 映射 | `ReqToTokenPool` | 记录 `(req_pool_idx, token_pos) -> kv_slot` |
| KV slot 分配器 | `TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` | 管理哪些 KV slot/page 可用 |
| 真实 KV 存储 | `token_to_kv_pool`，即 `KVCache` 子类 | 持有每层 K/V tensor，并提供读写接口 |

`token_to_kv_pool` 只负责真实 K/V 数据的物理存储和读写。它不决定某个请求应该用哪个 slot，也不维护请求生命周期；这些分别由 allocator、`ReqToTokenPool`、prefix cache 和 scheduler 负责。

## 2. 抽象基类 `KVCache`

所有 token-to-KV pool 都继承自 `memory_pool.py::KVCache`。这个抽象类定义了统一接口：

```python
def get_key_buffer(self, layer_id: int) -> torch.Tensor
def get_value_buffer(self, layer_id: int) -> torch.Tensor
def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]
def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None
```

关键字段：

- `size`：可用于真实 KV 的 token slot 数。
- `page_size`：分页 KV cache 的 page 大小；非分页路径通常为 1。
- `dtype`：逻辑 KV dtype。
- `store_dtype`：实际存储 dtype。FP8 等 dtype 的 `Tensor.index_put` 支持不完整，所以源码里会把 FP8 存成 `torch.uint8`，读取时再 `view(self.dtype)`。
- `layer_num`：当前 runner 管理的有效层数。
- `start_layer` / `end_layer`：pipeline parallel 或分层加载场景下，本 worker 负责的层范围。
- `mem_usage`：根据底层 buffer 大小统计的 KV cache 显存占用。
- `layer_transfer_counter`：用于 layer-wise KV cache loading，同步某层 buffer 可用性。
- `custom_mem_pool`：disaggregation / NVLink 等场景下可选的自定义 CUDA memory pool。

`KVCache` 还提供 `get_cpu_copy` / `load_cpu_copy` 接口约定，用于 KV cache offload 或 disaggregation 场景下在 CPU/GPU 之间搬运缓存。

## 3. 标准 MHA 路径：`MHATokenToKVPool`

普通 MHA/GQA 模型使用 `MHATokenToKVPool`。它为每层分别维护 K 和 V 两组 buffer：

```python
self.k_buffer = [
    torch.zeros((self.size + self.page_size, self.head_num, self.head_dim), ...)
    for _ in range(self.layer_num)
]
self.v_buffer = [
    torch.zeros((self.size + self.page_size, self.head_num, self.v_head_dim), ...)
    for _ in range(self.layer_num)
]
```

这里使用 `size + page_size`，而不是只分配 `size`。注释说明 padded slot 0 用于 padded token 的 dummy 写入；分页场景也需要保留 padding 范围。allocator 的空闲 slot/page 通常从 1 开始，0 作为安全 dummy 位置。

### 3.1 成员变量

`MHATokenToKVPool` 继承自 `KVCache`，所以成员变量可以分成两类：基类提供的通用 KV cache 元数据，以及 MHA pool 自己维护的 K/V buffer 和 kernel 辅助信息。

`KVCache` 基类初始化的通用字段：

| 成员变量 | 含义 |
|---|---|
| `self.size` | 可用于真实 token 的 KV cache slot 数量。实际 buffer 通常会分配 `size + page_size`，多出的部分用于 padding/dummy 写入。 |
| `self.page_size` | paged KV cache 的页大小。非分页路径通常接近 1，paged attention 会按 page/block 组织 slot。 |
| `self.dtype` | attention 逻辑上看到的 KV dtype，例如 fp16、bf16、fp8 等。 |
| `self.store_dtype` | 实际存储 KV cache 的 dtype。通常等于 `dtype`，但 FP8/量化等场景可能用 `torch.uint8` 存储，再通过 `.view(self.dtype)` 还原逻辑视图。 |
| `self.layer_num` | 当前 KV pool 覆盖的层数。pipeline parallel 或分层加载下，它不一定等于模型总层数。 |
| `self.device` | KV cache 张量所在设备，例如 `cuda`、`npu`、`cpu`。 |
| `self.start_layer` | 当前 pool 覆盖的起始全局 layer id。访问 buffer 时通常用 `layer_id - self.start_layer` 转成内部层索引。 |
| `self.end_layer` | 当前 pool 覆盖的结束 layer id。 |
| `self.layer_transfer_counter` | 可选的按层传输同步计数器。layer-wise KV loading 场景下，读某层 KV 前需要等待该层传输完成。 |
| `self.memory_saver_adapter` | memory saver 适配器，用来把 KV cache 分配登记到特定显存区域。 |
| `self.enable_custom_mem_pool` | 是否启用了自定义 CUDA memory pool，常见于 disaggregation / RDMA 相关场景。 |
| `self.custom_mem_pool` | 自定义 memory pool 对象；未启用时通常为空。 |
| `self.mem_usage` | KV cache 分配后的显存占用统计，由 `_finalize_allocation_log` 在初始化末尾设置。 |

`MHATokenToKVPool` 自己维护的字段：

| 成员变量 | 含义 |
|---|---|
| `self.head_num` | KV head 数量。MHA/GQA/MQA 中，这里是 key/value cache 实际保存的 head 数；SWA 子池可用 `swa_head_num` 覆盖。 |
| `self.head_dim` | key head 的维度；SWA 子池可用 `swa_head_dim` 覆盖。 |
| `self.v_head_dim` | value head 的维度。优先级是 `swa_v_head_dim`、`v_head_dim`、`head_dim`。用于支持 K/V 维度不同的模型。 |
| `self.kv_cache_layout` | KV cache 物理布局。默认是 `"nhd"`，即 `[slot, head, dim]`；ROCm AITER 下可为 `"vectorized_5d"`。 |
| `self._kv_vector_x` | 仅 `vectorized_5d` 布局使用。表示最内层向量化宽度，计算方式是 `16 // self.store_dtype.itemsize`。 |
| `self.k_buffer` | 每层 key cache 的实际存储列表。普通 NHD 布局下每层形状近似为 `(size + page_size, head_num, head_dim)`。 |
| `self.v_buffer` | 每层 value cache 的实际存储列表。普通 NHD 布局下每层形状近似为 `(size + page_size, head_num, v_head_dim)`。 |
| `self.k_data_ptrs` | 所有层 `k_buffer` 的底层地址指针 tensor，供 Triton kernel 间接寻址。 |
| `self.v_data_ptrs` | 所有层 `v_buffer` 的底层地址指针 tensor，供 Triton kernel 间接寻址。 |
| `self.data_ptrs` | `k_data_ptrs` 和 `v_data_ptrs` 拼接后的统一指针表。KV 搬迁 kernel 会把所有 K/V buffer 当成一个列表处理。 |
| `self.data_strides` | 每个 K/V buffer 中单个 token 行的字节跨度。KV copy kernel 用它计算搬迁偏移。 |
| `self.device_module` | 当前设备对应的 torch device module，例如 `torch.cuda`。用于创建 stream、获取当前 stream 等。 |
| `self.alt_stream` | 可选辅助 stream。CUDA 或类 CUDA 平台上用于在写 KV 时尝试重叠 K/V 两路写入。 |
| `self._kv_copy_config` | KV cache 搬迁 kernel 的配置。启用 `enable_kv_cache_copy` 时包含 `bytes_per_tile`、`byte_tiles`、`num_warps`、`num_locs_upper`；否则是 `None`。 |
| `self.row_dim` | 单个 key row 的展平元素数，等于 `head_num * head_dim`。写 KV 的 JIT kernel 会用到。 |
| `self.same_kv_dim` | `head_dim == v_head_dim` 的布尔值。用于写入逻辑判断 K/V 是否可以走相同维度的快速路径。 |

核心关系可以概括为：

```text
普通 NHD 布局:
k_buffer[layer].shape = (size + page_size, head_num, head_dim)
v_buffer[layer].shape = (size + page_size, head_num, v_head_dim)

layer_id -> 内部层索引:
internal_layer_idx = layer_id - start_layer

KV cache slot 范围:
[0, size + page_size)
```

其中 `size` 是真实 token 容量，额外的 `page_size` 是 padding/dummy 区；`k_buffer` / `v_buffer` 是真正保存 KV cache 的地方，`data_ptrs` / `data_strides` / `_kv_copy_config` 则是为了让 Triton 或后端 kernel 高效读写和搬迁这些缓存。

### 3.2 buffer layout

默认 layout 是 NHD：

```text
K: [num_slots, num_kv_heads, head_dim]
V: [num_slots, num_kv_heads, v_head_dim]
```

在 HIP + AITER backend 下，可以通过 `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d` 选择 SHUFFLE 5D layout：

```text
K: [num_blocks, H, D_k / X, page_size, X]
V: [num_blocks, H, page_size / X, D_v, X]
```

该 layout 只在 AITER consumer kernel 可用时启用；非 AITER 平台会强制使用 NHD。

### 3.3 读取接口

`get_key_buffer(layer_id)` 和 `get_value_buffer(layer_id)` 会返回当前层的 K/V buffer。如果 `store_dtype != dtype`，会通过 `.view(self.dtype)` 还原逻辑 dtype 视图。

`get_kv_buffer(layer_id)` 只是返回二者：

```python
return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
```

attention backend 通常会这样使用：

```python
k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
```

随后 backend 再配合 `req_to_token_pool.req_to_token` 构造出的 `kv_indices` 或 page table，从这些 buffer 中读取对应 slot 的历史 K/V。

### 3.4 写入接口

`set_kv_buffer(layer, loc_info, cache_k, cache_v, ...)` 是 MHA 路径的核心写入入口。

流程：

1. 通过 `unwrap_write_loc(loc_info)` 取出 full KV pool 的写入位置 `loc`。
2. `maybe_detect_oob(loc, 0, self.size + self.page_size, ...)` 做越界检测。
3. 如果输入 K/V dtype 与 pool dtype 不一致，按需缩放并 cast 到 `self.dtype`。
4. 如果 `store_dtype != dtype`，把 tensor view 成底层存储 dtype。
5. vectorized 5D layout 走 `launch_reshape_and_cache_shuffle_5d`。
6. 默认 NHD layout 走 `_set_kv_buffer_impl(...)`。

`_set_kv_buffer_impl` 会根据平台和 row size 选择不同写入策略：

- CUDA/HIP 且适合时使用 JIT `store_cache`。
- CPU AMX 路径使用 `torch.ops.sgl_kernel.store_cache_cpu`。
- CUDA graph capture 且存在 `alt_stream` 时，K/V 写入可分流到 alternate stream。
- 否则退化为 `k_cache[indices] = k`、`v_cache[indices] = v`。

因此 `set_kv_buffer` 是 backend 无关的统一写入接口，内部再根据平台和 dtype 选择优化写法。

### 3.5 批量复制与 offload

`MHATokenToKVPool` 支持：

- `get_cpu_copy(indices)`：按 layer 和 chunk 把指定 KV slot 拷到 CPU。
- `load_cpu_copy(kv_cache_cpu, indices)`：把 CPU 上的 KV chunk 写回 GPU pool。
- `move_kv_cache(tgt_loc, src_loc)`：在 pool 内搬移所有层的 KV，主要用于 speculative decoding 接受 token 后重排 KV。

当启用 `enable_kv_cache_copy` 时，`move_kv_cache` 会预热并使用 `copy_all_layer_kv_cache_tiled` Triton kernel，一次跨层搬移 K/V。

## 4. MLA 路径：`MLATokenToKVPool`

MLA 模型不再为每层分别保存 K 和 V 两个完整 buffer，而是保存合并后的 latent KV：

```python
self.kv_buffer = [
    torch.zeros((self.size + self.page_size, 1, self.kv_cache_dim), ...)
    for _ in range(self.layer_num)
]
```

默认 `kv_cache_dim = kv_lora_rank + qk_rope_head_dim`。也就是说，一个 slot 里同时保存：

- `k_nope` / latent 部分，长度为 `kv_lora_rank`。
- `k_rope` 部分，长度为 `qk_rope_head_dim`。

`get_key_buffer(layer_id)` 返回整块 combined KV buffer；`get_value_buffer(layer_id)` 返回其中前 `kv_lora_rank` 维：

```python
return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]
```

这和 MHA 路径不同：MLA 的 value 不是单独一份 `v_buffer`，而是 combined latent KV 的前缀视图。

### 4.1 写入 MLA KV

MLA 有两个写入接口：

```python
set_kv_buffer(layer, loc, cache_k, cache_v)
set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
```

普通 `set_kv_buffer` 直接把已经组合好的 `cache_k` 写到 `kv_buffer[loc]`。

更常用的 `set_mla_kv_buffer` 接受拆开的 `cache_k_nope` 和 `cache_k_rope`，然后通过 `set_mla_kv_buffer_triton` 写入同一个 combined buffer。这样可以避免 Python 侧手动 concat 或减少中间 tensor。

DSA + FP8 场景下，源码还会选择更特殊的量化写入：

- HIP FP8 DSA 路径使用 `set_mla_kv_buffer_triton_fp8_quant`。
- `dsa_kv_cache_store_fp8` 路径会用 `quantize_k_cache_separate` 分别量化 nope 和 rope，再写入 combined buffer。

### 4.2 读取 MLA KV

`get_mla_kv_buffer(layer, loc, dst_dtype)` 会从 combined buffer 中按位置取出两段：

```python
cache_k_nope: [num_tokens, 1, kv_lora_rank]
cache_k_rope: [num_tokens, 1, qk_rope_head_dim]
```

底层通过 `get_mla_kv_buffer_triton` 完成拆分读取。

## 5. DSA 路径：`DSATokenToKVPool`

`DSATokenToKVPool` 继承自 `MLATokenToKVPool`，用于 DeepSeek Sparse Attention 相关模型。

它保留 MLA combined KV 的主体设计，同时额外维护 DSA indexer 需要的 K/index buffer 和量化状态。源码中可以看到：

- `quant_block_size = 128`
- `index_k_with_scale_buffer_dtype = torch.uint8`
- `rope_storage_dtype = torch.bfloat16`
- `index_head_dim` 要求为 128
- 非 HIP 路径要求 `page_size == 64`

这说明 DSA pool 的职责不仅是保存普通 latent KV，还要服务 sparse attention 的 indexer 和稀疏检索流程。

模型初始化时，如果 `self.use_mla_backend and is_dsa_model`，会选择：

```python
PoolCls = HiSparseDSATokenToKVPool if self.enable_hisparse else DSATokenToKVPool
```

HiSparse 变体进一步加入 host/device 分层缓存能力。

## 6. 量化 KV pool

源码中有 FP4 变体：

- `MHATokenToKVPoolFP4`
- `MLATokenToKVPoolFP4`

它们的主要特点是：

- K/V 或 combined KV 主体以 `torch.uint8` 保存压缩后的 FP4 数据。
- 额外维护 scale buffer，例如 `k_scale_buffer`、`v_scale_buffer` 或 `kv_scale_buffer`。
- 读取时会通过量化工具反量化或返回逻辑 dtype 视图。
- 写入时会把输入 K/V 做 block FP4 quantize 后存入主体 buffer 和 scale buffer。

FP8 路径则通常通过 `store_dtype = torch.uint8` 实现物理存储，配合 attention backend 或专用 kernel 做量化/反量化。

## 7. SWA 和 Hybrid pool

当模型是 hybrid sliding window attention 时，`ModelRunner._init_pools` 会创建 `SWAKVPool`。

`SWAKVPool` 内部通常维护两套 pool：

- full attention pool：用于 full attention 层。
- SWA pool：用于 sliding window attention 层。

写入时 attention backend 会传入 `KVWriteLoc(loc, swa_loc)`。`loc` 是 full pool 的位置，`swa_loc` 是预先翻译好的 SWA pool 位置。`KVWriteLoc` 的定义位于 `memory_pool.py`：

```python
@dataclass
class KVWriteLoc:
    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None
```

这样 backend 可以统一调用：

```python
token_to_kv_pool.set_kv_buffer(layer, KVWriteLoc(cache_loc, swa_out_cache_loc), k, v)
```

具体 pool 再决定写 full、写 SWA，或两者都写。

Mamba/linear attention 混合模型使用 `HybridLinearKVPool`。它把 attention KV pool 和 Mamba state pool 组合在一起；其中 Mamba state 由 `ReqToTokenPool` 相关的 Mamba allocator 管理，attention KV 仍走 `KVCache` 的读写接口。

## 8. 初始化选择逻辑

`token_to_kv_pool` 的创建集中在 `ModelRunner._init_pools`。

选择顺序大致如下：

1. DeepSeek V4 使用 `DeepSeekV4TokenToKVPool`，NPU 上使用 `DSV4NPUTokenToKVPool`。
2. out-of-tree platform 可以提供平台自定义的 MHA/MLA/DSA pool class。
3. NPU 平台使用 `NPUMHATokenToKVPool` / `NPUMLATokenToKVPool`，hybrid SWA 使用 `SWAKVPool + NPUMHATokenToKVPool`。
4. DSA 模型使用 `DSATokenToKVPool` 或 `HiSparseDSATokenToKVPool`。
5. MLA backend 使用 `MLATokenToKVPool` 或 `MLATokenToKVPoolFP4`。
6. hybrid SWA 使用 `SWAKVPool`。
7. Mamba/linear hybrid 使用 `HybridLinearKVPool`。
8. 普通 MHA 模型使用 `MHATokenToKVPool` 或 `MHATokenToKVPoolFP4`。
9. prefill-only 且禁用 KV cache 的特殊场景使用 `NoOpMHATokenToKVPool`。

创建 pool 后，`ModelRunner` 会继续创建 allocator：

- `TokenToKVPoolAllocator`：`page_size == 1` 的普通 token 粒度分配。
- `PagedTokenToKVPoolAllocator`：`page_size > 1` 的 page 粒度分配。
- `SWATokenToKVPoolAllocator`：hybrid SWA 同时管理 full 和 SWA 两套空间。
- `HiSparseTokenToKVPoolAllocator`：HiSparse host/device 分层场景。
- NPU 或 DeepSeek V4 还有平台专用 allocator。

注意：allocator 持有 `kvcache=self.token_to_kv_pool`，但 allocator 自己不保存真实 K/V tensor。它只分配和释放 index。

## 9. 与 allocator 的配合

普通 `TokenToKVPoolAllocator` 初始化时会把可用 slot 设为：

```python
self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, device=self.device)
```

这里变量名叫 `free_pages`，但在 `page_size == 1` 时它实际就是 free token slot。slot 0 被保留给 dummy/padded token。

分配时：

```python
select_index = self.free_pages[:need_size]
self.free_pages = self.free_pages[need_size:]
return select_index
```

分页 allocator 则以 page 为单位管理空闲列表：

```python
self.num_pages = size // page_size
self.free_pages = torch.arange(1, self.num_pages + 1, ...)
```

`alloc(need_size)` 会返回 page 展开后的 token indices：

```text
page_id * page_size + [0, 1, ..., page_size - 1]
```

`alloc_extend` 和 `alloc_decode` 更复杂：它们会根据 `prefix_lens`、`seq_lens`、`last_loc` 判断当前请求是否还能接在已有 page 后面，以及需要消费多少新 page。返回值仍然是 token 粒度的 `out_cache_loc`，方便写入 `ReqToTokenPool` 和后续 `set_kv_buffer`。

## 10. 运行时写入链路

一次 prefill/extend 或 decode 中，KV 写入大致分为两步。

第一步，调度侧分配 slot，并写入请求映射表：

```text
allocator.alloc_* -> out_cache_loc
ReqToTokenPool.write(req_pool_idx, token_pos, out_cache_loc)
```

第二步，模型 forward / attention backend 计算出本层 K/V 后，把 K/V 写入真实 pool：

```text
token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)
```

也就是说：

- `ReqToTokenPool` 写的是“token 位置到 slot 的索引”。
- `token_to_kv_pool` 写的是“slot 里真实的 K/V 数据”。

两者使用同一个 `out_cache_loc` 作为连接点。

## 11. 运行时读取链路

attention 读取历史 KV 时通常需要两个输入：

1. 从 `ReqToTokenPool` 得到当前 batch 每个请求对应的 KV slot/page table。
2. 从 `token_to_kv_pool` 得到某一层的真实 K/V buffer。

典型代码模式：

```python
k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
```

然后 backend 会把 `k_cache/v_cache` 和 `kv_indices/page_table` 传给 FlashInfer、FlashAttention、Triton、TRTLLM、AITER 等 kernel。

MLA backend 则常见：

```python
k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
```

因为 MLA 的 key buffer 已经是 combined latent KV，kernel 会按 MLA 语义解释它。

## 12. prefix cache 和释放

请求结束时，prefix cache 会决定哪些 KV slot 被插入 radix tree 继续复用，哪些 slot 释放回 allocator。`token_to_kv_pool` 本身不感知“这个 slot 是否还属于某个 prefix node”，它只保存数据。

例如 `release_kv_cache` 会从 `req_to_token_pool.req_to_token` 中取出 overallocated slot：

```python
indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][start_p:end_p]
tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
```

allocator 释放的是 index；真实 `token_to_kv_pool` 中对应位置的数据通常不会被清零。下一次分配到同一 slot 时，新的 `set_kv_buffer` 会覆盖旧值。

Radix cache eviction 也是同理：tree node 中保存的是 KV indices，驱逐时调用 allocator free 这些 indices，而不是操作 KV buffer 内容。

## 13. disaggregation / HiCache 相关接口

`token_to_kv_pool` 还提供一些跨节点或 host/device 缓存需要的低层信息：

- `get_contiguous_buf_infos()`：返回每层 K/V buffer 的 data pointer、总 byte length、单 item byte length。
- `get_cpu_copy(indices)` / `load_cpu_copy(...)`：按 slot 拷贝 KV cache 到 CPU 或从 CPU 写回。
- `data_ptrs` / `data_strides`：为跨层复制、disaggregation 传输或 kernel 批量操作提供指针表。

这些接口说明 `token_to_kv_pool` 不只是 attention kernel 的本地 tensor 容器，也承担了 KV 传输和 offload 的底层数据源角色。

## 14. 特殊 `NoOpMHATokenToKVPool`

`NoOpMHATokenToKVPool` 用在 embedding-mode prefill-only 且满足 FA backend `fa_skip_kv_cache` 前提的场景。它不分配真正的大型 KV cache，只为每层创建很小的 placeholder tensor，保证代码中无条件访问 `k_buffer/v_buffer` 的地方不会崩溃。

该类的 `set_kv_buffer` 会直接抛错，因为一旦真的写 KV，就说明 workload 或 backend 路径不满足 no-op pool 的前提。

这个设计保留了 scheduler 的容量视角，同时避免为不需要 decode 的 workload 分配 GB 级 KV cache。

## 15. 一句话总结

`token_to_kv_pool` 是 SGLang 中真正持有物理 K/V 张量的 KV cache pool。allocator 负责给 token 分配 slot，`ReqToTokenPool` 负责把请求逻辑 token 映射到这些 slot，而 `token_to_kv_pool` 负责在这些 slot 上按层保存和读取真实 K/V 数据。MHA、MLA、DSA、SWA、Mamba hybrid、FP4/FP8、NPU/ROCm/CUDA 等路径都通过 `KVCache` 接口接入，使 attention backend 可以用统一的 `set_kv_buffer` / `get_kv_buffer` 模式访问不同物理布局的 KV cache。
