# 为什么 `MHATokenToKVPool` 分配 `size + page_size` 个 slots

`MHATokenToKVPool._create_buffers()` 在默认 NHD 布局下会为每层创建 K/V buffer：

```python
self.k_buffer = [
    torch.zeros(
        (self.size + self.page_size, self.head_num, self.head_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
self.v_buffer = [
    torch.zeros(
        (self.size + self.page_size, self.head_num, self.v_head_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
```

这里不是只分配 `size`，而是分配：

```text
真实可用 KV 容量: size
额外保护区域: page_size
实际分配: size + page_size
```

## 1. `size` 是逻辑可用 KV 容量

`size` 表示 allocator 可以管理的真实 token KV slot 数量。也就是说，调度和 allocator 看待容量时，主要认为这个 pool 有 `size` 个可用于真实 token 的位置。

但底层 K/V tensor 需要给一些特殊写入留安全空间，所以实际物理 buffer 会更大一点。

## 2. slot 0 通常作为 dummy / padding 位置

SGLang 的 KV cache 体系里，经常把 `0` 当作安全 dummy index。

典型场景包括：

- batch padding
- CUDA graph padding
- 无效 request / 无效 token 的占位写入
- 一些 backend 中为了避免条件分支而保留的 dummy 访问

如果没有 slot 0，padding token 或 dummy token 一旦尝试写入 K/V cache，就可能访问非法地址。

因此至少需要：

```text
slot 0        -> dummy / padding slot
slot 1..size  -> 正常 KV slot
```

这已经意味着物理 buffer 至少需要 `size + 1` 个 slot。

## 3. paged KV cache 下按 page 预留更安全

SGLang 支持 paged KV cache。此时很多逻辑不是按单个 slot 独立处理，而是按 page/block 处理。

当 `page_size > 1` 时，如果只额外分配 1 个 dummy slot，会带来边界处理复杂度：

```text
只多 1 个 slot:
[dummy][normal slots...]
```

但 paged attention、allocator 展开 page、kernel 计算 page boundary 时，往往天然以 `page_size` 为单位。

因此源码选择多预留一个完整 page：

```text
额外保护区域 = page_size 个 slots
```

这样 dummy / padding 写入、page 展开和边界访问都更容易落在合法范围内。

## 4. 直观例子

假设：

```text
size = 1024
page_size = 16
```

那么实际分配：

```text
size + page_size = 1040 slots
```

可以理解成：

```text
[ padding / dummy page ][ normal KV slots ... ]
  16 slots               1024 slots
```

其中前面的 `page_size` 个 slot 可以承接 padding / dummy 访问，后面的 `size` 个 slot 是 allocator 主要管理的真实 KV 容量。

## 5. 与 allocator 的关系

需要注意，`MHATokenToKVPool` 负责的是底层物理 K/V tensor 分配；allocator 负责的是哪些 slot/page 可用。

普通 token allocator 通常不会把所有物理位置都当作真实容量使用，而是会保留 dummy 位置，从有效 slot 开始分配。

因此：

```text
token_to_kv_pool.size
    是逻辑容量

k_buffer / v_buffer 的第一维
    是物理分配容量 = size + page_size
```

两者不矛盾。前者服务调度和容量控制，后者服务安全访问和 kernel 边界处理。

## 6. 一句话总结

`size` 是真实 token 的逻辑 KV 容量；`page_size` 是额外预留的安全 padding page。`MHATokenToKVPool` 实际分配 `size + page_size` 个 slots，是为了让 dummy token、padding 写入和 paged KV cache 的边界访问都落在合法物理 buffer 内。

## 7. `_create_buffers` 存储分配结果图示

下面用图形化方式展示 `MHATokenToKVPool._create_buffers()` 分配后的存储结构。

### 7.1 默认 NHD 布局

当：

```python
self.kv_cache_layout == "nhd"
```

会为每一层分别创建 `K` 和 `V` buffer：

```text
MHATokenToKVPool
|
+-- k_buffer: List[Tensor], 长度 = layer_num
|   |
|   +-- k_buffer[0]: [size + page_size, head_num, head_dim]
|   +-- k_buffer[1]: [size + page_size, head_num, head_dim]
|   +-- ...
|   +-- k_buffer[layer_num - 1]
|
+-- v_buffer: List[Tensor], 长度 = layer_num
    |
    +-- v_buffer[0]: [size + page_size, head_num, v_head_dim]
    +-- v_buffer[1]: [size + page_size, head_num, v_head_dim]
    +-- ...
    +-- v_buffer[layer_num - 1]
```

单层内部可以理解为：

```text
k_buffer[layer_id]

slot 0              -> dummy / padding slot
slot 1              -> token KV slot
slot 2              -> token KV slot
...
slot size           -> token KV slot
slot size+1 ...     -> padding page 区域

每个 slot 内部:
[head_num, head_dim]
```

对应形状：

```text
K:
+--------------------------------------------+
| size + page_size slots                     |
|                                            |
| slot_i -> [head_num, head_dim]             |
+--------------------------------------------+

V:
+--------------------------------------------+
| size + page_size slots                     |
|                                            |
| slot_i -> [head_num, v_head_dim]           |
+--------------------------------------------+
```

### 7.2 `vectorized_5d` 布局

当：

```python
self.kv_cache_layout == "vectorized_5d"
```

会按 page/block 组织：

```text
total_slots = size + page_size
num_blocks = total_slots // page_size
x = self._kv_vector_x
```

K buffer 形状：

```text
[num_blocks, head_num, head_dim // x, page_size, x]
```

V buffer 形状：

```text
[num_blocks, head_num, page_size // x, v_head_dim, x]
```

图形化表示：

```text
k_buffer[layer]

+----------------------------------------------+
| num_blocks                                   |
|                                              |
| block_i                                      |
|   +-- head_num                               |
|       +-- head_dim // x                      |
|           +-- page_size                      |
|               +-- x                          |
+----------------------------------------------+
```

```text
v_buffer[layer]

+----------------------------------------------+
| num_blocks                                   |
|                                              |
| block_i                                      |
|   +-- head_num                               |
|       +-- page_size // x                     |
|           +-- v_head_dim                     |
|               +-- x                          |
+----------------------------------------------+
```

### 7.3 指针表结构

`_create_buffers()` 最后还会创建指针表：

```text
k_buffer --data_ptr--+
                     +-- k_data_ptrs
                     |
v_buffer --data_ptr--+
                     +-- v_data_ptrs
                     |
                     v
              data_ptrs = [所有 K 指针, 所有 V 指针]
```

整体关系：

```text
MHATokenToKVPool
|
+-- k_buffer
|   +-- layer 0 K tensor
|   +-- layer 1 K tensor
|   +-- ...
|
+-- v_buffer
|   +-- layer 0 V tensor
|   +-- layer 1 V tensor
|   +-- ...
|
+-- k_data_ptrs
|   +-- 每层 K tensor 的地址
|
+-- v_data_ptrs
|   +-- 每层 V tensor 的地址
|
+-- data_ptrs
|   +-- [k_data_ptrs, v_data_ptrs]
|
+-- data_strides
    +-- 每个 K/V buffer 单个 token 行占用的字节数
```

一句话概括：

```text
_create_buffers() 的核心结果是：
为每一层分配一份 K cache 和一份 V cache，
再额外构造 data_ptrs / data_strides，
方便 Triton kernel 或后端 kernel 高效访问这些物理 KV buffer。
```
