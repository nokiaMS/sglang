# HiCache DSV4 PP 索引错位问题分析

## 问题描述

在 PP=4 配置下启动 DeepSeek-V4-Pro + HiCache 时，PP1/PP2/PP3 节点报错：

```
ValueError: deepseek_v4_c4_state state_pools must not contain None
```

PP0（Node 0）成功分配了 c4_state 和 c128_state host pool，但 PP1~3 全部失败。

## 错误调用链

```
scheduler.__init__()
→ init_cache_with_memory_pool()
→ attach_hybrid_pool_to_unified_cache()         # hybrid_pool_assembler.py:694
→ build_deepseek_v4_hicache_stack()             # hybrid_pool_assembler.py:373
→ DeepSeekV4StateHostPool.__init__()            # memory_pool_host.py:1946
→ raise ValueError("state_pools must not contain None")
```

## 根因分析

### 1. compress_state_pools 的初始化（deepseek_v4_memory_pool.py:555-585）

`compress_state_pools` 是长度为 `total_L`（全部 61 层）的列表，初始化为全 `None`，然后只对当前 PP stage 范围内的层赋值：

```python
self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L

for idx in range(self._stage_start, self._stage_end):  # 绝对索引
    ratio = self.compression_ratios[idx]
    if ratio == 0:
        continue
    self.compress_state_pools[idx] = CompressStatePool(...)
```

- PP0: `_stage_start=0, _stage_end=15` → `compress_state_pools[0:15]` 被赋值
- PP1: `_stage_start=15, _stage_end=30` → `compress_state_pools[15:30]` 被赋值
- PP2: `_stage_start=30, _stage_end=45` → `compress_state_pools[30:45]` 被赋值
- PP3: `_stage_start=45, _stage_end=61` → `compress_state_pools[45:61]` 被赋值

其余索引位置保持 `None`。

### 2. c4_state_global_layers 的构建（hybrid_pool_assembler.py:295-308）

```python
c4_state_global_layers = []
for layer_id, layer_item in enumerate(
    kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
):
    if layer_item.compress_ratio == 4:
        c4_state_global_layers.append(layer_id)  # 局部索引 0,1,2,...
```

`enumerate()` 产生的 `layer_id` 是**从 0 开始的局部偏移量**，而非全局绝对层号。变量名为 `c4_state_global_layers` 但实际存储的是局部索引。

### 3. 用局部索引访问全局数组（hybrid_pool_assembler.py:373-377）

```python
c4_state_host_pool = DeepSeekV4StateHostPool(
    pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),
    state_pools=[
        kvcache.compress_state_pools[layer_id]   # 用局部索引访问全局数组
        for layer_id in c4_state_global_layers
    ],
)
```

### 4. 索引错位的具体表现

| PP Stage | start_layer | end_layer | 局部 layer_id 范围 | compress_state_pools 非 None 范围 | 结果 |
|----------|-------------|-----------|---------------------|-----------------------------------|------|
| PP0 | 0 | 15 | 0~14 | 0~14 | 正常（碰巧对齐） |
| PP1 | 15 | 30 | 0~14 | 15~29 | **索引错位，全为 None** |
| PP2 | 30 | 45 | 0~14 | 30~44 | **索引错位，全为 None** |
| PP3 | 45 | 61 | 0~14 | 45~60 | **索引错位，全为 None** |

### 5. 同样的问题也存在于 c128_state

`c128_state_global_layers` 的构建逻辑与 `c4_state_global_layers` 完全相同，也存在相同的局部/全局索引混淆问题。

### 6. 代码中已有 TODO 提示

`hybrid_pool_assembler.py:289` 行：
```python
# TODO(hzh0425): Support PP for deepseek v4 with hicache
```

说明 HiCache 对 DSV4 的 PP 支持尚未完成。

## 修复方案

### 方案：将局部索引转换为全局索引

在 `build_deepseek_v4_hicache_stack` 函数中，`c4_state_global_layers` 和 `c128_state_global_layers` 收集的 `layer_id` 应加上 `kvcache.start_layer` 偏移量，使其成为全局绝对索引。

**修改位置**：`hybrid_pool_assembler.py:302, 305`

```python
# 修改前：
c4_state_global_layers.append(layer_id)

# 修改后：
c4_state_global_layers.append(layer_id + kvcache.start_layer)

# 修改前：
c128_state_global_layers.append(layer_id)

# 修改后：
c128_state_global_layers.append(layer_id + kvcache.start_layer)
```

**修改位置**：`hybrid_pool_assembler.py:387`（indexer_compress_state_pools 同理）

```python
# 修改前：
state_pools=[
    kvcache.indexer_compress_state_pools[layer_id]
    for layer_id in c4_state_global_layers
],

# 无需修改，因为 c4_state_global_layers 已经被修正为全局索引
```

### 影响范围

此修改仅影响 `c4_state_global_layers` 和 `c128_state_global_layers` 中存储的索引值。需要确认下游使用这些索引的地方是否都期望全局索引：

1. **`c4_state_mapping`** / **`c128_state_mapping`**：用于 `build_pool_entry` 的 `layer_mapping` 参数，其 key 是局部 `layer_id`，value 是 `local_id`。这是 host pool 内部的层映射，不直接依赖全局索引。
2. **`compress_state_pools[layer_id]`** / **`indexer_compress_state_pools[layer_id]`**：这是触发 bug 的地方，需要全局索引。
3. **`c4_kv_pool`** / **`c4_indexer_kv_pool`** 等 PagedHostPool：使用 `device_buffers` 和 `layer_mapping`，不直接使用 `c4_state_global_layers`。

因此，修复方案是安全的，只需将 `c4_state_global_layers` 和 `c128_state_global_layers` 中的值改为全局索引即可。
