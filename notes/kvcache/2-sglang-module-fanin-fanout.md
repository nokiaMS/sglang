# SGLang KV Cache / ModelRunner / ScheduleBatch / Attention Backend Fan-in Fan-out

本文按模块间的依赖、持有、创建和调用方向梳理：

- `A --> B` 表示 `A` 扇出到 `B`。
- 换句话说，`A` import、持有、创建或调用 `B`。

```mermaid
flowchart LR
  Scheduler["scheduler.py<br/>Scheduler"]
  ScheduleBatch["managers/schedule_batch.py<br/>ScheduleBatch / Req"]
  ModelRunner["model_executor/model_runner.py<br/>ModelRunner"]
  KVMixin["model_runner_kv_cache_mixin.py<br/>ModelRunnerKVCacheMixin"]
  ForwardBatch["model_executor/forward_batch_info.py<br/>ForwardBatch"]
  AttnRegistry["layers/attention/attention_registry.py<br/>ATTENTION_BACKENDS"]
  AttnBase["layers/attention/base_attn_backend.py<br/>AttentionBackend"]
  AttnImpl["layers/attention/*_backend.py<br/>具体 Attention Backend"]
  RadixAttention["layers/radix_attention.py<br/>RadixAttention"]
  ForwardContext["model_executor/forward_context.py<br/>ForwardContext / get_attn_backend"]

  KVBuilder["mem_cache/kv_cache_builder.py<br/>KV cache builder"]
  MemCommon["mem_cache/common.py<br/>alloc/release helpers"]
  PrefixCache["mem_cache/base_prefix_cache.py<br/>Radix/Chunk/HiCache"]
  ReqPool["mem_cache/memory_pool.py<br/>ReqToTokenPool"]
  KVAllocator["mem_cache/allocator/*<br/>TokenToKVPoolAllocator"]
  KVPool["mem_cache/memory_pool.py<br/>KVCache / TokenToKVPool"]

  Scheduler -->|"构造/调度 batch"| ScheduleBatch
  Scheduler -->|"初始化/转发经 tp_worker"| ModelRunner
  Scheduler -->|"build tree_cache"| KVBuilder
  Scheduler -->|"release / cache unfinished"| MemCommon
  Scheduler -->|"持有"| PrefixCache
  Scheduler -->|"持有"| ReqPool
  Scheduler -->|"持有"| KVAllocator

  KVBuilder -->|"创建"| PrefixCache
  PrefixCache -->|"记录 prefix -> KV slot"| ReqPool
  PrefixCache -->|"释放/驱逐 KV slot"| KVAllocator

  ScheduleBatch -->|"字段: req_to_token_pool"| ReqPool
  ScheduleBatch -->|"字段: token_to_kv_pool_allocator"| KVAllocator
  ScheduleBatch -->|"字段: tree_cache"| PrefixCache
  ScheduleBatch -->|"alloc_for_extend/decode, release"| MemCommon
  ScheduleBatch -->|"数据流: init_new 输入"| ForwardBatch

  ModelRunner -->|"继承 mixin"| KVMixin
  KVMixin -->|"分配"| ReqPool
  KVMixin -->|"分配"| KVPool
  KVMixin -->|"包装 KVPool"| KVAllocator
  KVAllocator -->|"管理 slot / get_kvcache"| KVPool

  ModelRunner -->|"ForwardBatch.init_new / forward()"| ForwardBatch
  ModelRunner -->|"init_attention_backend()"| AttnRegistry
  AttnRegistry -->|"按 server_args 创建"| AttnImpl
  AttnImpl -.->|"实现接口"| AttnBase
  AttnImpl -->|"构造时读取 runner pools"| ModelRunner
  AttnImpl -->|"读 batch metadata"| ForwardBatch
  AttnImpl -->|"page table / req_to_token"| ReqPool
  AttnImpl -->|"读写 K/V buffer"| KVPool

  ModelRunner -->|"with forward_context(attn_backend)"| ForwardContext
  RadixAttention -->|"get_attn_backend().forward()"| ForwardContext
  ForwardContext -->|"返回当前 backend"| AttnBase
  AttnBase -->|"dispatch decode/extend/mixed"| AttnImpl
```

## 模块扇入扇出摘要

### `ScheduleBatch`

- 扇入：主要来自 `Scheduler`。
- 扇出：`ReqToTokenPool`、`TokenToKVPoolAllocator`、`PrefixCache`、`mem_cache.common`、`ForwardBatch`。
- 作用：承接调度层高层 batch 状态，完成 KV slot 分配、释放、prefix 命中信息维护，并把运行所需字段交给 `ForwardBatch`。

### `ModelRunner`

- 扇入：主要来自 `Scheduler` / `TpModelWorker`。
- 扇出：`ModelRunnerKVCacheMixin`、`ForwardBatch`、attention registry、forward context。
- 作用：初始化模型、KV memory pool、attention backend，并在 forward 时把 `ForwardBatch` 和当前 attention backend 绑定到执行上下文。

### KV Cache

- 扇入：`Scheduler`、`ScheduleBatch`、`ModelRunnerKVCacheMixin`、attention backend。
- 扇出：allocator 管理物理 `KVPool`，prefix cache 管理复用和驱逐。
- 作用：`ReqToTokenPool` 维护请求 token 到物理 KV slot 的映射，`TokenToKVPoolAllocator` 管理 slot/page 分配，`KVPool` 保存真实 K/V tensor。

### Attention Backend

- 扇入：`ModelRunner` 初始化创建，`RadixAttention` 运行时调用。
- 扇出：`ForwardBatch` metadata、`ReqToTokenPool` page table、`KVPool` K/V buffer。
- 作用：按 decode/extend/mixed 模式准备 backend metadata，写入当前 token 的 K/V，并读取历史 K/V 执行 attention kernel。

