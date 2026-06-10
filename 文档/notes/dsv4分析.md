# DeepSeek V4 Pro HiCache 启动报错分析

## 报错信息

启动命令中启用 `--enable-hierarchical-cache` 后，在 `sglang/python/sglang/srt/mem_cache/hiradix_cache.py` 第 102 行报错：

```
ValueError: HiRadixCache only supports MHA, MLA, and NSA(DSA) models.
```

## 根因分析

**报错位置**: `hiradix_cache.py:102` — `raise ValueError("HiRadixCache only supports MHA, MLA, and DSA models")`

**根因**: DeepSeek V4 Pro 使用的是 `DeepSeekV4TokenToKVPool`（继承自 `BaseSWAKVPool`），而 `HiRadixCache.__init__` 只识别三种 KV pool 类型：

| 类型 | 对应模型 |
|------|---------|
| `MHATokenToKVPool` | 标准MHA模型 |
| `MLATokenToKVPool` | MLA模型（如DS-V3） |
| `DSATokenToKVPool` | DSA/NSA模型 |

`DeepSeekV4TokenToKVPool` 不属于以上任何一种，它继承自 `BaseSWAKVPool`，是一个独立的 SWA（Sliding Window Attention）+ 压缩KV pool 实现，因此走到了 `else` 分支直接报错。

**代码执行路径**：

1. 启用 `--enable-hierarchical-cache`
2. 在 `registry.py:124-132`，由于 `enable_hierarchical_cache=True` 且 `is_hybrid_ssm=False`，选择了 `HiRadixCache`
3. `HiRadixCache.__init__` 中的 `isinstance` 检查都不匹配 `DeepSeekV4TokenToKVPool`，触发报错

**关键点**：DS V4 的 HiCache 支持已经实现（commit `d9fa84b25`），但**仅在 `UnifiedRadixCache` 路径中**，不在旧的 `HiRadixCache` 路径中。而 `UnifiedRadixCache` 需要环境变量 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` 才会启用。

## 报错 2：hicache-size 不支持

添加 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` 后，报错：

```
ValueError: DeepSeek V4 HiCache currently does not support --hicache-size; use --hicache-ratio instead.
```

**原因**：`hybrid_pool_assembler.py:257-261` 中，DS V4 HiCache 不支持 `--hicache-size`（绝对大小 GB），只支持 `--hicache-ratio`（相对于 device cache 的比例，默认 `2.0`）。

## 报错 3：SWA host pool 不支持 kernel io_backend

解决报错1、2后，运行时报错：

```
NotImplementedError: swa supports only direct io_backend, got kernel
```

**调用链**：`unified_radix_cache.py:write_backup` → `hybrid_cache_controller.py:start_writing` → `memory_pool_host.py:backup_from_device_all_layer` → `_check_io_backend`

**原因**：DS V4 的 SWA host pool（commit `c2a212bfe` 中实现）在 `_check_io_backend` 中校验，只允许 `direct` io_backend。`kernel` 模式是使用 GPU kernel 做 DMA 搬运，而 SWA pool 的数据布局不支持这种方式。

## 报错 4：dec_lock_ref 时 FULL component invariant 断言失败

解决报错1-3后，服务启动能运行，但 decode 阶段报错：

```
AssertionError
  File "unified_radix_cache.py", line 1406, in writing_check
    self.dec_lock_ref(node, params)
  File "unified_radix_cache.py", line 422, in dec_lock_ref
    component.release_component_lock(node=node, params=params)
  File "full_component.py", line 199, in release_component_lock
    assert cd.value is not None
```

**调用链**：`check_hicache_events` → `writing_check` → `dec_lock_ref` → `release_component_lock` → assert 失败

**根因分析**：

1. `write_backup` 发起异步 D→H 写入时，调用 `inc_lock_ref(node)` 锁定从 node 到 root 的路径，记录 `lock_params`（包含 `skip_lock_node_ids`）
2. 异步写入进行中，SWA 组件的 `_maybe_split_leaf_for_swa_lock`（commit `f59bbef84`）可能在 insert 时将 node 的祖先节点分裂
3. 分裂后，新创建的 parent 节点的 FULL `component_data.value = None`（因为 FULL KV 只保留在子节点中，parent 只保留 SWA 部分）
4. 异步写入完成，`writing_check` 调用 `dec_lock_ref(node, params)` → `release_component_lock`
5. `release_component_lock` 从 node 向 root 遍历，遇到分裂后 value=None 的祖先节点
6. `assert cd.value is not None` 失败 — 因为这个祖先在 `inc_lock_ref` 时不存在（由后续分裂创建），不在 `skip_lock_node_ids` 中

**本质**：`inc_lock_ref` 和 `dec_lock_ref` 之间的树结构发生了变化（SWA split），导致 lock 的"快照"与实际树不一致。这是 `UnifiedRadixCache` 在 SWA + HiCache 场景下的一个已知缺陷。

**可能的解决方向**：

1. **（推荐）使用 write-through 替代 write-back**：将 `--hicache-write-policy write_back` 改为 `write_through`。write-through 模式下 `write_backup` 不调用 `inc_lock_ref`（`lock_params = None`），因此不会触发此断言
2. **等待上游修复**：这是 `UnifiedRadixCache` SWA + HiCache write-back 路径的 bug，需要上游在 `release_component_lock` 中处理 node split 导致的 value=None 情况（类似 `acquire_component_lock` 中 skip tombstone 的逻辑）

## 最终解决方案（含报错4修复）

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 172.16.186.221:20000 \
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
  --cuda-graph-max-bs 64 \
  --enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-io-backend direct
```

相比原命令，做了四处修改：
1. 添加环境变量 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`（解决报错1）
2. 将 `--hicache-size 90` 替换为 `--hicache-ratio 2.0`（解决报错2）
3. 将 `--hicache-io-backend kernel` 替换为 `--hicache-io-backend direct`（解决报错3）
4. 将 `--hicache-write-policy write_back` 替换为 `write_through`（解决报错4）

**write_through vs write_back 的区别**：
- `write_through`：每次写入 device 时同步写入 host，不持有 lock，更稳定但写入延迟略高
- `write_back`：先写 device，异步批量写入 host，持有 lock 防止写入期间被驱逐，吞吐更高但在 SWA 场景下有 bug

## 原理说明

- `registry.py:98`：当 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` 时，走 `UnifiedRadixCache` 路径
- `registry.py:103-106`：`UnifiedRadixCache` 正确处理 `is_hybrid_swa=True` 的情况（DS V4 是 hybrid SWA 模型）
- `registry.py:117-118`：在 `UnifiedRadixCache` 初始化后调用 `init_hicache()`
- `unified_radix_cache.py:371-418`：`init_hicache()` 调用 `attach_hybrid_pool_to_unified_cache()`，该函数包含 DS V4 的 HiCache 支持（commit `d9fa84b25` 中实现）

## 注意事项

- `--page-size 64` 会被 `deepseek_v4_hook.py:16` 覆盖为 256（DS V4 的默认值）
- DS V4 的 KV cache dtype 仅支持 `fp8_e4m3`，`deepseek_v4_hook.py:32-34` 会校验
- DS V4 HiCache 不支持 `--hicache-size`，必须使用 `--hicache-ratio`（默认 2.0，表示 host cache 为 device cache 的 2 倍）
- DS V4 SWA host pool 不支持 `--hicache-io-backend kernel`，必须使用 `direct`
- DS V4 HiCache 的 `write_back` 模式在 SWA split 场景下有 bug（`dec_lock_ref` 时树结构已变导致 assert 失败），建议使用 `write_through`

## 相关代码文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/mem_cache/hiradix_cache.py` | 旧路径 HiCache，不支持 DS V4 |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 新路径 UnifiedRadixCache，支持 DS V4 HiCache |
| `python/sglang/srt/mem_cache/registry.py` | 缓存实例选择逻辑 |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | DS V4 专用 KV pool |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | HiCache 组装逻辑，DS V4 不支持 `--hicache-size` |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | Host KV pool，DS V4 SWA 只支持 `direct` io_backend |
| `python/sglang/srt/mem_cache/unified_cache_components/full_component.py` | FULL component，lock/unlock 逻辑，`release_component_lock` 中的 assert 失败 |
| `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py` | SWA component，`_maybe_split_leaf_for_swa_lock` 会分裂节点导致树结构变化 |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | DS V4 默认参数注入 |
| `python/sglang/srt/environ.py` | `SGLANG_ENABLE_UNIFIED_RADIX_TREE` 环境变量定义 |
