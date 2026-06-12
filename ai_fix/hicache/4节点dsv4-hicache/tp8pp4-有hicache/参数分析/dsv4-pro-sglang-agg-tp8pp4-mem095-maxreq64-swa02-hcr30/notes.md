# DeepSeek-V4-Pro HiCache 参数分析笔记

## swa_full_tokens_ratio=1.0 对吞吐率及 C60 OOM 的影响分析

### 结论

**swa_full_tokens_ratio=1.0 对 DSV4 是错误方向**：不能缓解 C60 OOM，反而会加剧 OOM 风险，同时大幅降低吞吐率。

### 核心公式

DSV4 的 `bytes_per_full_token` 计算（`pool_configurator.py:396-410`）：

```python
bytes_per_full_token = (
    swa_ratio * kv_bytes * num_layers_total                          # SWA KV
    + c4_frac * kv_bytes * num_layers_ca4                           # C4 KV (不依赖swa_ratio)
    + 1/128 * kv_bytes * num_layers_ca128                           # C128 KV (不依赖swa_ratio)
    + 1/4 * indexer_bytes * num_layers_ca4                          # C4 Indexer (不依赖swa_ratio)
    + swa_ratio * c4_state_ratio * c4_state_bytes * num_layers_ca4  # C4 State
    + swa_ratio * c128_state_ratio * c128_state_bytes * num_layers_ca128  # C128 State
    + swa_ratio * c4_state_ratio * c4_indexer_state_bytes * num_layers_ca4  # C4 Indexer State
)
```

而 `full_token = available_bytes / bytes_per_full_token`，然后所有子池从 full_token 派生：

```python
swa_tokens = full_token * swa_ratio
c4_tokens = full_token / (4 * c4_shrink_factor)
c128_tokens = full_token / 128
```

### swa_ratio 对 bytes_per_full_token 的影响

DSV4 有 61 层，PP=4 每节点约 15 层。`bytes_per_full_token` 中 swa_ratio 乘数项：

| 项 | 系数 | swa_ratio=0.2 | swa_ratio=1.0 | 变化 |
|---|---|---|---|---|
| SWA KV | `swa_ratio × kv_bytes × 15` | 3×kv_bytes | 15×kv_bytes | **5x** |
| C4 State | `swa_ratio × c4_state_ratio × c4_state × 3` | 0.2×base | 1.0×base | **5x** |
| C128 State | `swa_ratio × c128_state_ratio × c128_state × 12` | 0.2×base | 1.0×base | **5x** |
| C4 Indexer State | `swa_ratio × c4_state_ratio × idx_state × 3` | 0.2×base | 1.0×base | **5x** |
| C4 KV | `0.25 × kv_bytes × 3` | 0.75×kv_bytes | 0.75×kv_bytes | 不变 |
| C128 KV | `(1/128) × kv_bytes × 12` | 0.094×kv_bytes | 0.094×kv_bytes | 不变 |
| C4 Indexer | `0.25 × idx_bytes × 3` | 0.75×idx_bytes | 0.75×idx_bytes | 不变 |

**swa_ratio=1.0 使得 bytes_per_full_token 约为 swa_ratio=0.2 的 2.5-3 倍**，导致 full_token 降至约 1/2.5 ~ 1/3，所有子池等比例缩小。

### 对 SWA 容量和 Full token 容量的影响

| swa_ratio | full_token (相对) | swa_tokens | SWA 容量 |
|-----------|-------------------|------------|----------|
| 0.1 | ~1.0x (基准) | 0.1x | 0.1x |
| 0.2 | ~0.85x | 0.17x | 0.17x |
| 1.0 | ~0.35x | 0.35x | 0.35x |

swa_ratio=1.0 时 SWA 容量虽然比 0.2 大 2 倍（0.35 vs 0.17），但 full_token 大幅缩减至 ~35%，意味着 Full KV pool 容量只有 0.2 的 41%，C4/C128 pool 容量也只有 0.2 的 41%，总 token 容量急剧下降。

### 对 C60 OOM 的影响：加剧 OOM

1. **SWA 层的 KV 占用反而增大**：swa_ratio=1.0 时 SWA pool 按 full_token 等量分配，每个 token 在 SWA 层都需要存 KV（而非只存滑动窗口内的），GPU 显存占用大幅增加
2. **Prefill 时激活内存不变**，但可用于分配临时空间的剩余显存更少（因为 SWA pool 变大）
3. **full_token 缩减意味着能同时处理的请求数更少**，C60 的 60 个并发请求可能根本分配不到足够的 KV slot
4. **HiCache host pool 也等比例缩小**（因为 host_pages = device_pages × hicache_ratio），offload 容量更小

### 为什么 MiMoV2 和 Step3p5 用 swa_ratio=1.0？

从 `server_args.py:2272-2303` 可以看到，sglang 对 MiMoV2 和 Step3p5 模型自动设置 `swa_full_tokens_ratio=1.0`。这是因为这些模型的架构不同——它们可能没有 C4/C128 压缩层，或者 SWA 层的 KV 维度极小，使得 1.0 不会造成内存膨胀。DSV4 的 61 层中有大量 C4/C128 压缩层和对应的 State pool，swa_ratio 对这些 pool 的 State 部分影响极大。

### 综合对比

| 维度 | swa=0.1 | swa=0.2 | swa=1.0 |
|------|---------|---------|---------|
| Full token 容量 | 最大 | 中等 | **最小（约35%）** |
| SWA pool 容量 | 最小 | 中等 | 较大（但绝对值仍低） |
| C60 稳定性 | OOM | 稳定(hcr=2.0) | **更易OOM** |
| 吞吐率 | 最高（预热后） | 中等 | **最低** |
| Prefill 内存压力 | 最高 | 中等 | **更高** |

---

## hicache-ratio=3.0 导致 C60 OOM 的根因分析

### 因果链

1. **hicache_ratio=3.0 → Host pool 页数增加 50%**（从 2x 变为 3x device_pages），代码位于 `hybrid_pool_assembler.py:259-260`：
   ```python
   full_host_pages = max(int(device_full_pages * ratio), device_full_pages + 1)
   swa_host_pages = max(int(device_swa_pages * ratio), device_swa_pages + 1)
   ```

2. **DSV4 有 8 个独立的 Host pool**（KV/SWA/C4/C128/State/Indexer 等），每个都按 ratio 倍数分配

3. **所有 Host pool 使用 `cudaHostRegister` 锁定物理页**（`memory_pool_host.py:195`），且通过 `mmap MAP_POPULATE` 预分配（`mmap_allocator.py:124`），全部 Host 内存立即锁定

4. **HiSparse Coordinator 在 GPU 上维护 `req_to_host_pool` 张量**（`hisparse_coordinator.py:125-130`），其第二维 `max_compressed_context_len` 与 host pool 容量成正比。更大的 host pool → 更大的 GPU 跟踪数据

5. **更多 GPU 端跟踪数据**（`hisparse_coordinator.py:147-176`）：
   - `req_device_buffer_tokens`: `(layer_num, max_num_req_slots, padded_buffer_size)` — int32
   - `req_device_buffer_token_locs`: 同上
   - `lru_slots`: `(layer_num, max_num_req_slots, device_buffer_size)` — int16
   - `top_k_device_locs_buffer`, `raw_indices_buffer` 等

6. **C60 高并发 prefill 时**，60个请求同时需要临时 GPU 显存，而 GPU 端 HiCache 跟踪数据已占用更多显存 → Prefill out of memory

### 结论

hicache_ratio 不只影响 Host 内存，还通过 HiSparse Coordinator 的 GPU 端跟踪数据结构间接消耗 GPU 显存。ratio=3.0 比 ratio=2.0 多占用约 50% 的 GPU 端 HiCache 元数据显存，在 C60 高并发 prefill 时触发 OOM。

---

## 正确的优化方向

1. **降低 mem-fraction-static**（从 0.95 降到 0.88-0.90），为 prefill 留出更多临时空间
2. **降低 max-running-requests**（从 64 降到 32-48），减少并发压力
3. **测试 hicache-ratio=2.5**，在 C30 性能和 C60 稳定性间取折中
4. swa_full_tokens_ratio=0.2 是 DSV4 当前最佳平衡点，不宜调大或调小
