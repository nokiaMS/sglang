# DeepSeek-V4-Pro GPU 显存分配机制及原理

## 1. 什么是 MoE 大模型

### 1.1 基本概念

MoE (Mixture of Experts，混合专家模型) 是一种将"稀疏激活"引入神经网络的架构范式。传统 Dense 模型在处理每个 token 时会激活全部参数，而 MoE 模型只激活其中一小部分"专家"，从而以远小于总参数量的计算量实现超大模型的表达能力。

### 1.2 核心结构

一个典型的 MoE 层由三部分组成:

```
输入 token (hidden_state)
       │
       ▼
┌──────────────┐
│   Router     │  ← 门控网络: 为每个 token 选择 Top-K 个专家
│  (Gate)      │     输出: expert_weights[0..K-1] + expert_indices[0..K-1]
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│              Expert Pool (N 个专家)               │
│                                                 │
│  ┌───────┐ ┌───────┐       ┌───────┐           │
│  │Expert0│ │Expert1│  ...  │ExpertN│           │
│  │ (MLP) │ │ (MLP) │       │ (MLP) │           │
│  └───┬───┘ └───┬───┘       └───┬───┘           │
│      │         │               │                │
│  ┌───┴─────────┴───────────────┴───┐           │
│  │   Shared Expert (可选, 始终激活)  │           │
│  └───────────────┬─────────────────┘           │
└──────────────────┼──────────────────────────────┘
                   │
                   ▼
        Weighted Sum (加权求和)
        output = Σ(gate_weight_i × expert_i(input))
              + shared_expert(input)    ← 共享专家始终参与
```

**Routed Experts (路由专家)**: 数量多 (V4 Pro 有 256 个)，每个 token 只激活 Top-K 个 (V4 Pro 为 Top-6)，计算量 = K × 单 expert 计算。

**Shared Expert (共享专家)**: 数量少 (V4 Pro 有 1 个)，对所有 token 始终激活，弥补路由专家的覆盖不足。

**Router/Gate**: 轻量级线性层，输入 hidden_state，输出每个 expert 的选择概率，取 Top-K。

### 1.3 MoE vs Dense 对比

| 维度 | Dense 模型 | MoE 模型 |
|------|-----------|---------|
| 参数总量 | 7B / 70B | 1.6T (V4 Pro) |
| 每 token 激活参数 | = 总参数 | ≈ 总参数 × K/N ≈ 1.6T × 6/256 ≈ 37B |
| 计算量 (FLOPs) | 与参数量成正比 | 仅与激活参数量成正比 |
| 显存需求 | 与参数量成正比 | **总参数量仍需全部加载到显存** |
| 通信模式 | TP all-reduce | EP all-to-all + TP all-reduce |

### 1.4 MoE 的核心矛盾: 参数量大 vs 计算量小

MoE 的根本优势是**用少量计算获得大模型的表达力**，但这带来一个关键矛盾:

```
推理时: 每个 token 只用 6/256 = 2.3% 的专家参数
但部署时: 全部 256 个专家权重都必须加载到 GPU 显存

→ 显存瓶颈远大于计算瓶颈
→ 这正是 MXFP4 量化和 Expert Parallelism 如此重要的原因
```

### 1.5 MoE 相关的并行策略

```
┌────────────────────────────────────────────────────────────────┐
│                    MoE 并行策略全景                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Tensor Parallelism (TP)                                    │
│     每个 expert 的权重沿 intermediate 维度切分到多个 GPU        │
│     每个 GPU 持有全部 N 个 expert, 但每个 expert 只有一片      │
│     通信: all-reduce                                           │
│                                                                │
│  2. Expert Parallelism (EP)                                    │
│     不同 expert 整体分配到不同 GPU                              │
│     每个 GPU 只持有 N/ep_size 个 expert, 每个 expert 完整      │
│     通信: all-to-all (token routing)                           │
│                                                                │
│  3. TP + EP 混合                                               │
│     EP 负责 expert 间分布, TP 负责 expert 内切分                │
│     moe_tp_size = tp_size / ep_size                            │
│     每个 GPU 持 N/ep_size 个 expert, 每个 expert 切 moe_tp_size│
│                                                                │
│  4. DP-Attention                                               │
│     注意力层用数据并行 (各 rank 独立 KV Cache)                  │
│     MoE 层仍用 TP/EP                                           │
│     attn_tp_size = tp_size / dp_size                           │
│                                                                │
│  5. Expert TP-1 (共享专家)                                     │
│     共享专家不做 TP 切分, 每个 GPU 完整复制                     │
│     避免 all-reduce, 但多占显存                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 概述

DeepSeek-V4-Pro 是一个 1.6T 参数的 MoE (Mixture of Experts) 大模型，采用 MXFP4 量化。在 SGLang 推理框架中，其 GPU 显存分配涉及模型权重、KV Cache、激活内存、CUDA Graph 和框架开销等多个方面。由于模型采用了 DSA/NSA (DeepSeek Attention / Native Sparse Attention) 压缩注意力机制和 MHC (Multi-Head Consolidation)，其 KV Cache 分配机制与传统大模型有本质区别。

### 模型关键参数

| 参数 | 值 |
|------|-----|
| 总参数量 | ~1.6T (MoE) |
| hidden_size | 7168 |
| moe_intermediate_size | 3072 (per expert) |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 6 (Top-6 routing) |
| num_hidden_layers | 43 |
| num_attention_heads | 64 |
| qk_nope_head_dim | 448 |
| qk_rope_head_dim | 64 |
| kv_lora_rank | 512 (MQA) |
| o_groups | 16 (MHC) |
| 量化格式 | MXFP4 (expert), FP8 (dense) |
| 注意力机制 | DSA/NSA + MHC |
| window_size | 128 (SWA) |
| index_head_dim | 128 |

---

## 3. GPU 显存总体分配架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 总显存 (e.g. H100 80GB)               │
├─────────────────────────────────────────────────────────────┤
│  mem_fraction_static 部分 (通常 ~82-88%)                     │
│  ┌───────────────────────┐  ┌─────────────────────────────┐ │
│  │   模型静态权重         │  │   KV Cache 动态池            │ │
│  │   (模型加载后固定)     │  │   (运行时按需分配)           │ │
│  │                       │  │                             │ │
│  │  · Dense层权重 (FP8)  │  │  · SWA KV Pool              │ │
│  │  · MoE专家权重 (MXFP4)│  │  · C4 KV Pool               │ │
│  │  · 注意力权重          │  │  · C128 KV Pool             │ │
│  │  · 共享专家权重        │  │  · C4 Indexer Pool          │ │
│  │  · MHC参数            │  │  · Compress State Pools     │ │
│  │  · Embedding/LM Head  │  │                             │ │
│  └───────────────────────┘  └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  (1 - mem_fraction_static) 部分 (通常 ~12-18%)               │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │  激活内存         │  │  CUDA Graph    │  │  框架开销    │ │
│  │  (Prefill中间值)  │  │  捕获缓冲区    │  │  NCCL/PyTorch│ │
│  └──────────────────┘  └────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 核心公式

```
GPU总显存 = 模型权重 + KV Cache + 激活内存 + CUDA Graph + 框架开销

其中:
  (模型权重 + KV Cache) = GPU总显存 × mem_fraction_static
  (激活内存 + CUDA Graph + 框架开销) = GPU总显存 × (1 - mem_fraction_static)
```

---

## 4. mem_fraction_static 机制

### 4.1 定义

`--mem-fraction-static` 控制预留给静态权重和 KV Cache 的显存占 GPU 总显存的比例。剩余部分留给激活、CUDA Graph 和框架开销。

### 4.2 自动计算 (SGLang 默认行为)

当用户未显式指定时，SGLang 通过以下公式自动计算 (`server_args.py:1562-1609`):

```python
reserved_mem = 512                                          # 常量元数据
reserved_mem += max(chunked_prefill_size, 2048) * 1.5      # Prefill 激活
reserved_mem += cuda_graph_max_bs * 2                       # CUDA Graph
reserved_mem += tp_size * pp_size / 8 * 1024               # 并行通信开销

# DP-attention 额外预留
if enable_dp_attention:
    reserved_mem += cuda_graph_max_bs * dp_size * 3
    if cuda_graph_max_bs > 300:
        reserved_mem += cuda_graph_max_bs * dp_size * 1.5

# 大显存GPU (>60GB) 最少预留10GB
if gpu_mem > 60 * 1024:
    reserved_mem = max(reserved_mem, 10 * 1024)

mem_fraction_static = round((gpu_mem - reserved_mem) / gpu_mem, 3)
```

**H100 80GB 示例 (TP=8)**:
```
reserved_mem = 512 + 8192*1.5 + 512*2 + 8*1/8*1024 = ~14,848 MB
mem_fraction_static = (81,920 - 14,848) / 81,920 ≈ 0.819
```

### 4.3 KV Cache 可用空间计算

`_profile_available_bytes()` (`model_runner_kv_cache_mixin.py:62-76`):

```python
available_bytes = (post_model_load_memory - pre_model_load_memory * (1 - mem_fraction_static)) * 1GB
```

其中:
- `pre_model_load_memory`: 模型加载前的可用 GPU 显存 (GB)
- `post_model_load_memory`: 模型加载后的可用 GPU 显存 (GB)
- 模型权重占用 = `pre_model_load_memory - post_model_load_memory`
- KV Cache 预算 = `post_model_load_memory - 预留给激活/Graph的空间`

---

## 5. 模型权重显存分配

### 5.1 权重分类与量化

DeepSeek-V4-Pro 的权重分为两类，采用不同量化策略:

| 权重类型 | 量化格式 | 比特/参数 | 示例 |
|----------|---------|----------|------|
| Dense 层 (注意力、MLP gate) | FP8 (E4M3) | 8 bit | wq_a, wkv, wo_a/b, RMSNorm |
| MoE 专家权重 (routed) | MXFP4 | ~4.25 bit | gate_up, down_proj |
| 共享专家权重 | FP8 | 8 bit | shared_expert MLP |
| MHC 参数 | FP32 | 32 bit | hc_attn_fn, hc_ffn_fn 等 |

### 5.2 MXFP4 量化对显存的影响

MXFP4 将权重存储为 4-bit 打包格式 (uint8，每字节存2个FP4值)，外加 E8M0 block scale (每32个元素1字节 scale):

```
单个 Routed Expert 权重:
  gate_up: 2 × intermediate_size × hidden_size × 0.5 byte (packed FP4)
         + 2 × intermediate_size × (hidden_size/32) × 1 byte (scale)
  down:   hidden_size × intermediate_size × 0.5 byte (packed FP4)
         + hidden_size × (intermediate_size/32) × 1 byte (scale)

对 V4 Pro (hidden=7168, intermediate=3072):
  gate_up 权重: 2 × 3072 × 7168 / 2 = ~22.1 MB (FP4) + ~1.4 MB (scale)
  down 权重:   7168 × 3072 / 2 = ~11.0 MB (FP4) + ~0.7 MB (scale)
  单 expert ≈ ~35 MB
  256 experts ≈ ~8.96 GB (MXFP4)
  对比 FP8: 256 experts ≈ ~17.9 GB
  对比 FP16: 256 experts ≈ ~35.8 GB
```

**MXFP4 相比 FP8 节省约 50%，相比 FP16 节省约 75% 的专家权重显存。**

### 5.3 权重在不同并行策略下的分布

#### Tensor Parallelism (TP)

Dense 层权重按 TP 切分:
- `wq_b` (ColumnParallel): 每个 GPU 持 `1/attn_tp_size`
- `wo_a` (ColumnParallel): 每个 GPU 持 `1/attn_tp_size`
- `wo_b` (RowParallel): 每个 GPU 持 `1/attn_tp_size`
- `wq_a`, `wkv` (Replicated): 每个 GPU 持完整副本

MoE 权重按 `moe_tp_size` 切分:
```
moe_tp_size = tp_size / ep_size / moe_dp_size
intermediate_size_per_partition = moe_intermediate_size / moe_tp_size
每个 GPU 持有: n_routed_experts / ep_size 个专家
每个专家权重: 按 moe_tp_size 切分 intermediate 维度
```

#### Expert Parallelism (EP)

EP 改变专家在各 GPU 间的分布:
```
EP=1:  每个 GPU 持全部 256 个专家 (每个专家按 TP 切分)
EP=16: 每个 GPU 持 256/16 = 16 个专家 (完整权重，不切分)
EP=32: 每个 GPU 持 256/32 = 8 个专家
```

#### 共享专家: SGLANG_SHARED_EXPERT_TP1

当设置 `SGLANG_SHARED_EXPERT_TP1=1` 时:
- 共享专家在每个 GPU 上**完整复制** (TP=1)，不做切分
- **原因**: FP8 量化要求 `intermediate_size / tp_size` 能被 128 整除，TP>=16 时不满足
- **代价**: 每个 GPU 多持有完整共享专家权重 (~0.5-1GB)，但避免了 all-reduce 通信

### 5.4 TP=16 每GPU权重估算

```
Dense 层 (FP8):
  注意力层 × 43:  ~10 GB (TP=16 切分后)

MoE 专家 (MXFP4):
  256 experts / TP=16: 每个 expert ~35MB, 16个 expert = ~0.56 GB
  但每个 expert 的 intermediate 按 TP=16 切分, 所以:
  实际每 GPU: 256 experts × 35MB / 16 = ~0.56 GB... 不对

  正确计算:
  每个 GPU 持有全部 256 个专家 (EP=1)
  每个专家的 intermediate 维度被 TP=16 切分:
    gate_up: 2 × (3072/16) × 7168 / 2 = 2 × 192 × 7168 / 2 = ~1.38 MB
    down:    7168 × (3072/16) / 2 = 7168 × 192 / 2 = ~0.69 MB
    单 expert ≈ ~2.2 MB (切分后)
    256 experts × 2.2 MB ≈ ~0.56 GB... 仍然很小

  实际上 Marlin 后端会要求对齐和 padding, 以及 scale 开销
  加上共享专家 (TP=1): 3072 × 7168 × 2 (gate_up) + 7168 × 3072 (down) ≈ ~66 MB (FP8)

  总 MoE ≈ ~15-20 GB/GPU (含 padding/scale/共享专家)
```

实际运行中 TP=16 每GPU 权重约 **~50 GB** (含框架开销和 CUDA Graph 捕获空间)。

---

## 6. KV Cache 显存分配 (DSV4 核心机制)

### 6.1 三级压缩 KV Cache 架构

DeepSeek-V4-Pro 采用 DSA/NSA 压缩注意力，KV Cache 不是简单的全量存储，而是分为**三个层级**:

```
┌──────────────────────────────────────────────────────────────┐
│                   三级 KV Cache 架构                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐                                    │
│  │ SWA Pool (ratio=0)   │  ← 最近 window_size=128 个 token  │
│  │ 全量 KV，无压缩       │    的完整 KV (所有43层)           │
│  │ page_size = 128      │                                    │
│  │ 584 bytes/token/层   │                                    │
│  └──────────────────────┘                                    │
│           │                                                  │
│           ▼ 压缩                                             │
│  ┌──────────────────────┐                                    │
│  │ C4 Pool (ratio=4)    │  ← 4:1 压缩，每4个token保留1个     │
│  │ NSA 中距离注意力       │    仅 C4 层使用                   │
│  │ page_size = 64       │                                    │
│  │ 584 bytes/token/层   │                                    │
│  └──────────────────────┘                                    │
│           │                                                  │
│           ▼ 压缩                                             │
│  ┌──────────────────────┐                                    │
│  │ C128 Pool (ratio=128)│  ← 128:1 压缩，每128个token保留1个 │
│  │ DSA 远距离注意力       │    仅 C128 层使用                 │
│  │ page_size = 2        │                                    │
│  │ 584 bytes/token/层   │                                    │
│  └──────────────────────┘                                    │
│                                                              │
│  ┌──────────────────────┐                                    │
│  │ C4 Indexer Pool      │  ← C4 层的稀疏索引键               │
│  │ 用于 top-k 选择       │    132 bytes/token/C4层           │
│  │ page_size = 64       │                                    │
│  └──────────────────────┘                                    │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐         │
│  │ C4 Compress State    │  │ C128 Compress State  │         │
│  │ Ring Buffer (8 slots)│  │ Ring Buffer (128 slots)│        │
│  │ 在线压缩状态          │  │ 在线压缩状态          │         │
│  │ 8192 bytes/state-slot│  │ 4096 bytes/state-slot │        │
│  └──────────────────────┘  └──────────────────────┘         │
│                                                              │
│  ┌──────────────────────┐                                    │
│  │ C4 Indexer State     │  ← C4 Indexer 的压缩状态           │
│  │ Ring Buffer (8 slots)│    2048 bytes/state-slot          │
│  └──────────────────────┘                                    │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 层级压缩比 (compress_ratios)

每层根据 `compress_ratios` 配置分配到不同的压缩级别:
- `ratio=0`: 仅 SWA (无压缩注意力)，不使用压缩 KV Pool
- `ratio=4`: C4 压缩 (4:1)，使用 SWA + C4 KV + C4 Indexer + 压缩状态
- `ratio=128`: C128 压缩 (128:1)，使用 SWA + C128 KV + 压缩状态

典型 V4 Pro 配置中，43 层按特定模式分配 ratio=4 和 ratio=128。

### 6.3 KV 单 token 内存布局

每个 token 在 KV Cache 中的布局 (`deepseek_v4_memory_pool.py:93-111`):

```
KV per token per layer:
  ┌──────────────────┬───────────────────┬─────────────┬─────┐
  │ nope_head (FP8)  │ rope_head (BF16)  │ FP8 scales  │ pad │
  │  448 bytes       │  64 × 2 = 128 bytes│ 448/64 = 7  │  1  │
  └──────────────────┴───────────────────┴─────────────┴─────┘
  总计 = 448 + 128 + 7 + 1 = 584 bytes/token/层
```

各字段含义:
- **nope_head (FP8, 448 bytes)**: 不含位置编码的 Key 投影，FP8 量化存储
- **rope_head (BF16, 128 bytes)**: 含旋转位置编码的 Key 投影，BF16 存储 (保留精度)
- **FP8 scales (7 bytes)**: nope 部分的 FP8 block scale (每64个元素1个scale, 448/64=7)
- **padding (1 byte)**: 对齐填充

### 6.4 C4 Indexer 单 token 内存布局

C4 层额外需要 Indexer 键用于稀疏选择:

```
Indexer per token per C4 layer:
  ┌───────────────┬────────────────────┐
  │ index_head    │ FP8 scales         │
  │ 128 bytes     │ 128/128 × 4 = 4   │
  └───────────────┴────────────────────┘
  总计 = 128 + 4 = 132 bytes/token/C4层
```

### 6.5 Compress State 内存

压缩状态使用环形缓冲区 (Ring Buffer)，用于在线压缩的中间累积:

| 状态类型 | Ring Size | 单 slot 大小 | 计算 |
|----------|-----------|-------------|------|
| C4 state | 8 (正常) / 16 (speculative) | 8,192 bytes | 2×2×512×4 (KV+score, overlap, fp32) |
| C128 state | 128 (正常) / 1 (online) | 4,096 bytes | 2×1×512×4 (KV+score, no overlap, fp32) |
| C128 state (online) | 1 | 6,144 bytes | 3×512×4 (max+sum+kv, fp32) |
| C4 indexer state | 8 (正常) | 2,048 bytes | 2×2×128×4 (KV+score, fp32) |

Ring Size 由 `get_compress_state_ring_size()` 决定 (`deepseek_v4_memory_pool.py:30-43`):
```python
def get_compress_state_ring_size(compress_ratio, is_speculative=False):
    if compress_ratio == 128 and ONLINE_C128:
        return 1  # Online C128: 单slot累积模式
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128
```

---

## 7. DSV4PoolConfigurator — 核心内存计算器

### 7.1 工厂方法

`create_memory_pool_configurator()` (`pool_configurator.py:471-480`) 根据 model config 选择配置器:

```python
def create_memory_pool_configurator(mr: ModelRunner):
    if is_deepseek_v4(mr.model_config.hf_config) and mr.is_hybrid_swa:
        return DSV4PoolConfigurator(mr)     # ← V4 Pro 走这条路径
    if mr.is_hybrid_swa:
        return HybridSWAPoolConfigurator(mr)
    return DefaultPoolConfigurator(mr)
```

### 7.2 bytes_per_full_token 计算

这是 DSV4 内存分配的**核心公式** (`pool_configurator.py:372-410`):

```python
def _get_bytes_per_full_token(self) -> float:
    # KV 存储: 584 bytes/token/层
    kv_bytes = qk_nope_head_dim + qk_rope_head_dim * 2 + 8
    #           448           + 64 × 2              + 8 = 584

    # Indexer: 132 bytes/token/C4层
    indexer_bytes = index_head_dim + index_head_dim // 128 * 4
    #               128            + 128 // 128 × 4 = 132

    # 注意力头维度
    attn_head_dim = qk_nope_head_dim + qk_rope_head_dim  # = 512

    # 压缩状态 (fp32)
    c4_state_bytes     = 2 × 2 × attn_head_dim × 4  = 8,192
    c128_state_bytes   = 2 × 1 × attn_head_dim × 4  = 4,096
    c4_indexer_state_bytes = 2 × 2 × 128 × 4        = 2,048

    # 状态比例
    c4_state_ratio   = c4_ring_size / swa_page_size  # 8/128 = 0.0625
    c128_state_ratio = c128_ring_size / swa_page_size # 128/128 = 1.0

    # C4 压缩比例
    c4_frac = 1 / (4 × c4_shrink_factor)  # 默认 = 1/4 = 0.25

    # 总 bytes_per_full_token
    return (
        swa_ratio × kv_bytes × num_layers_total           # ① SWA KV (所有层)
      + c4_frac × kv_bytes × num_layers_ca4               # ② C4 KV
      + 1/128 × kv_bytes × num_layers_ca128               # ③ C128 KV
      + 1/4 × indexer_bytes × num_layers_ca4              # ④ C4 Indexer KV
      + swa_ratio × c4_state_ratio × c4_state_bytes
        × num_layers_ca4                                    # ⑤ C4 压缩状态
      + swa_ratio × c128_state_ratio × c128_state_bytes
        × num_layers_ca128                                  # ⑥ C128 压缩状态
      + swa_ratio × c4_state_ratio × c4_indexer_state_bytes
        × num_layers_ca4                                    # ⑦ C4 Indexer 压缩状态
    )
```

### 7.3 数值示例

假设 V4 Pro 43层中: 22层 C4, 21层 C128 (典型配置), swa_ratio=0.1:

```
kv_bytes = 584
indexer_bytes = 132
attn_head_dim = 512
c4_ring_size = 8, c128_ring_size = 128
swa_page_size = 128 (window_size)

① SWA KV:  0.1 × 584 × 43 = 2,511.2
② C4 KV:   0.25 × 584 × 22 = 3,212.0
③ C128 KV: 1/128 × 584 × 21 = 95.8
④ C4 Indexer: 0.25 × 132 × 22 = 726.0
⑤ C4 State: 0.1 × (8/128) × 8192 × 22 = 1,126.4
⑥ C128 State: 0.1 × (128/128) × 4096 × 21 = 8,601.6
⑦ C4 Indexer State: 0.1 × (8/128) × 2048 × 22 = 281.6

bytes_per_full_token ≈ 16,554.6 bytes
```

### 7.4 Pool 尺寸计算

从 `available_bytes` 计算各池尺寸 (`pool_configurator.py:444-459`):

```python
# Step 1: 计算最大 token 数
full_token = int(available_bytes / bytes_per_full_token)

# Step 2: 对齐到 page_size
full_token = full_token // page_size * page_size   # page_size=256

# Step 3: 计算各子池尺寸
swa_tokens    = int(full_token × swa_ratio) // page_size × page_size
c4_tokens     = full_token // (4 × c4_shrink_factor)
c128_tokens   = full_token // 128
c4_state_size = swa_tokens // swa_page_size × c4_ring_size
c128_state_size = swa_tokens // swa_page_size × c128_ring_size
```

### 7.5 H100 80GB TP=16 完整示例

```
Step 1: 可用 GPU 显存
  pre_model_load_memory  ≈ 79.0 GB  (框架初始化后)
  post_model_load_memory ≈ 29.0 GB  (模型加载后, 权重~50GB)
  mem_fraction_static    ≈ 0.82

  available_bytes = (29.0 - 79.0 × (1 - 0.82)) × 1GB
                  = (29.0 - 14.22) × 1GB
                  ≈ 14.78 GB ≈ 15,874,478,080 bytes

Step 2: 计算最大 token 数
  bytes_per_full_token ≈ 16,555 (假设22 C4 + 21 C128)
  full_token = 15,874,478,080 / 16,555 ≈ 958,814
  对齐到 256: full_token = 958,720

Step 3: 子池尺寸
  swa_tokens    = 958,720 × 0.1 = 95,872 (对齐)
  c4_tokens     = 958,720 / 4 = 239,680
  c128_tokens   = 958,720 / 128 = 7,490
  c4_state_size = 95,872 / 128 × 8 = 5,992
  c128_state_size = 95,872 / 128 × 128 = 95,872

  实际最大 token 容量 ≈ ~959K tokens/GPU
```

---

## 8. 内存分配完整流程

### 8.1 初始化时序

```
ModelRunner.__init__()
  │
  ├─ 1. 记录 pre_model_load_memory (模型加载前可用显存)
  │     get_available_gpu_memory() → ~79 GB
  │
  ├─ 2. 加载模型权重
  │     model = DeepseekV4ForCausalLM(...)
  │     权重分布到各 GPU (TP/EP 切分)
  │
  ├─ 3. init_memory_pool(pre_model_load_memory)
  │     │
  │     ├─ 3a. _profile_available_bytes()
  │     │       get_available_gpu_memory() → post_model_load_memory
  │     │       available_bytes = (post - pre × (1 - frac)) × 1GB
  │     │
  │     ├─ 3b. create_memory_pool_configurator(self)
  │     │       → DSV4PoolConfigurator(mr)
  │     │
  │     ├─ 3c. configurator.calculate_pool_sizes(available_bytes, page_size)
  │     │       → MemoryPoolConfig (各池尺寸)
  │     │
  │     ├─ 3d. _apply_token_constraints() (用户上限、对齐)
  │     │
  │     └─ 3e. _apply_memory_pool_config()
  │             创建 DeepSeekV4TokenToKVPool (物理分配)
  │
  └─ 4. 捕获 CUDA Graph (额外显存)
```

### 8.2 DeepSeekV4TokenToKVPool 物理分配

`DeepSeekV4TokenToKVPool.__init__()` (`deepseek_v4_memory_pool.py:356-500`) 创建以下物理内存池:

| 池名称 | 类型 | 尺寸 | page_size | 层数 | bytes/token |
|--------|------|------|-----------|------|-------------|
| `swa_kv_pool` | DeepSeekV4SingleKVPool | swa_tokens | 128 | 43 (全部) | 584 |
| `c4_kv_pool` | DeepSeekV4SingleKVPool | c4_tokens | 64 | c4_layer_num | 584 |
| `c128_kv_pool` | DeepSeekV4SingleKVPool | c128_tokens | 2 | c128_layer_num | 584 |
| `c4_indexer_kv_pool` | DeepSeekV4IndexerPool | c4_logical_size | 64 | c4_layer_num | 132 |
| `compress_state_pools[]` | CompressStatePool | per-ratio | ring_size | per-layer | 见6.5 |
| `indexer_compress_state_pools[]` | CompressStatePool | c4_state_size | ring_size | c4 layers | 2048/slot |

---

## 9. DSV4 特有参数自动配置

当检测到 DeepSeek V4 模型时，`apply_deepseek_v4_defaults()` (`deepseek_v4_hook.py:10-52`) 自动设置:

| 参数 | 默认值 | 原因 |
|------|--------|------|
| `attention_backend` | `"dsv4"` | V4 使用专用 DSA/NSA 后端 |
| `page_size` | `256` | 压缩注意力要求 page_size 为 128 的倍数 |
| `kv_cache_dtype` | `"fp8_e4m3"` | V4 仅支持 FP8 KV Cache |
| `max_running_requests` | `256` | 限制并发请求数 |
| `swa_full_tokens_ratio` | `0.1` | SWA 仅占 10% (对比默认 0.8)，因压缩KV大幅减少全量存储需求 |
| `state_dtype` | `torch.float32` | 压缩状态使用 fp32 保证精度 |

**`swa_full_tokens_ratio=0.1` 的含义**: 只有 10% 的 token 容量用于 SWA 全量存储，远低于传统模型的 0.8。这是因为 C4/C128 压缩池承担了绝大部分 KV 存储，SWA 只需要覆盖最近 128 个 token 的滑动窗口。

---

## 10. 并行策略对显存的影响

### 10.1 Tensor Parallelism (TP)

TP 将模型权重和 KV Cache 分布到多个 GPU:

```
每 GPU 权重 = 总权重 / tp_size (大部分权重)
每 GPU KV Cache = KV Cache (独立，不共享)

TP 增大 → 每GPU权重减少 → 更多空间给 KV Cache → 更多 token 容量
```

但 TP 增大也受约束:
- **Shape 约束**: `intermediate_size_per_partition = moe_intermediate_size / moe_tp_size` 必须满足后端对齐要求
- **MHC 约束**: `n_local_groups = o_groups / attn_tp_size` 必须 > 0 (TP=32, o_groups=16 时需要 DP-attention)
- **通信开销**: TP 越大，all-reduce 通信量越大

### 10.2 Expert Parallelism (EP)

EP 改变专家分布方式:

```
EP=1:  每个 GPU 持全部 256 experts，每个 expert 按 moe_tp_size 切分
EP>1:  每个 GPU 持 n_routed_experts/ep_size 个 experts

EP 增大 → 每 GPU expert 数减少 → 但 moe_tp_size 也减小 → 每个 expert 切分更少
极端: EP=32, moe_tp_size=1 → 每个 expert 完整存储 → 可能 OOM
```

### 10.3 DP-Attention

DP-Attention 将 TP 组细分为注意力 DP 组:

```
attn_tp_size = tp_size / dp_size

DP-attention 启用后:
  · 注意力权重按 attn_tp_size 切分 (更少切分)
  · 但每个 DP rank 独立持有 KV Cache
  · KV Cache 容量不变 (每个 DP rank 的 KV Cache 独立)
  · 注意力层权重每 GPU 持有更多 (切分更少)
```

### 10.4 moe_dp_size

```
moe_dp_size 增大 → moe_tp_size 减小 → 每个 expert 切分更少 → 每 GPU 权重更多
moe_dp_size=2 (EP=16, TP=32): moe_tp_size=1 → 完整 expert → OOM
moe_dp_size=1 (EP=16, TP=32): moe_tp_size=2 → 每 expert 切一半 → 可行
```

### 10.5 各并行策略组合的每GPU显存分布

以 V4 Pro (1.6T, MXFP4) 为例:

| 配置 | 每 GPU 权重 | KV Cache 可用 | 总可用 | 可行性 |
|------|-----------|-------------|--------|--------|
| TP=8 | ~100GB | OOM | 80GB | ✗ (权重就超了) |
| TP=16, EP=1 | ~50GB | ~15GB | 80GB | ✓ (官方推荐) |
| TP=16, EP=1, DP=2 | ~50GB+ | ~10GB | 80GB | 需 SGLANG_SHARED_EXPERT_TP1 |
| TP=32, EP=1 | ~25GB | ~40GB | 80GB | ✗ (shape约束, 无MoE后端) |
| TP=32, EP=16 | ~15GB | ~50GB | 80GB | ✗ (framework bug) |
| TP=32, EP=32 | ~10GB | ~55GB | 80GB | ✗ (framework bug) |

---

## 11. 显存优化技术

### 11.1 在线 C128 压缩 (Online C128)

通过环境变量 `SGLANG_OPT_USE_ONLINE_COMPRESS=1` 启用:
- C128 ring_size 从 128 降为 1 (单 slot 累积模式)
- 每个压缩索引只维护一个 (max, sum, kv) 状态
- 大幅减少 C128 压缩状态内存 (但每 slot 从 4096→6144 bytes)
- **限制**: 不支持 Speculative Decode (MTP)

### 11.2 HiSparse

通过 `--enable-hisparse` 启用:
- C4 pool 的 `c4_shrink_factor > 1`
- C4 设备端 KV 进一步压缩，部分存储到 CPU (host)
- 减少设备端 C4 内存占用

### 11.3 Speculative Decode 的显存影响

启用 Speculative Decode (EAGLE/MTP) 时:
- `c4_ring_size`: 8 → 16
- `c128_ring_size`: 128 → 256
- `bytes_per_full_token` 膨胀 `(target_layers + draft_layers) / target_layers`
- 压缩状态 ring buffer 翻倍

### 11.4 CUDA Graph 显存

CUDA Graph 捕获需要额外显存:
- 每个 batch size 捕获一份计算图
- 默认捕获 `cuda_graph_max_bs=512` 个 batch size
- DP-attention 时需要更多 (× dp_size × 3 额外预留)
- 可以通过 `--disable-cuda-graph` 禁用 (牺牲性能)

---

## 12. 关键源码索引

| 文件 | 位置 | 功能 |
|------|------|------|
| `pool_configurator.py` | L309-468 | `DSV4PoolConfigurator`: DSV4 内存池配置器 |
| `pool_configurator.py` | L372-410 | `_get_bytes_per_full_token()`: 每 token 内存成本公式 |
| `pool_configurator.py` | L412-422 | `_compute_dsv4_sizes()`: 各子池尺寸计算 |
| `model_runner_kv_cache_mixin.py` | L62-76 | `_profile_available_bytes()`: KV Cache 预算计算 |
| `model_runner_kv_cache_mixin.py` | L926-951 | `_resolve_memory_pool_config()`: 内存池配置解析 |
| `deepseek_v4_memory_pool.py` | L46-120 | `DeepSeekV4SingleKVPool`: 物理KV缓冲区分配 |
| `deepseek_v4_memory_pool.py` | L247-347 | `DeepSeekV4IndexerPool`: Indexer KV 缓冲区 |
| `deepseek_v4_memory_pool.py` | L356-500 | `DeepSeekV4TokenToKVPool`: 统一 KV 池管理 |
| `deepseek_v4_memory_pool.py` | L30-43 | `get_compress_state_ring_size()`: Ring Buffer 尺寸 |
| `deepseek_v4_compress_state.py` | L78+ | `CompressStatePool`: 压缩状态环形缓冲区 |
| `deepseek_v4_hook.py` | L10-52 | DSV4 参数自动配置 (page_size=256, fp8 kv, swa_ratio=0.1) |
| `deepseek_v4.py` | L242-935 | `MQALayer`: DSV4 注意力层 (compressor/indexer) |
| `server_args.py` | L1562-1609 | `mem_fraction_static` 自动计算 |
| `mxfp4.py` | L422-558 | MXFP4 权重创建与内存布局 |
| `deepseek_v2.py` | L526-703 | MoE 层: 专家权重分布, 共享专家 TP1 |
| `deepseek_v4.py` | L47-111 | `DeepSeekV4Config`: 模型架构参数 |

---

## 13. 总结

DeepSeek-V4-Pro 的 GPU 显存分配机制具有以下核心特征:

1. **MXFP4 量化使 1.6T 模型可行**: MoE 专家权重使用 4-bit 量化，相比 FP16 节省 ~75% 显存，使 TP=16 部署成为可能。

2. **三级压缩 KV Cache 是内存效率的关键**: DSA/NSA 机制将 KV Cache 分为 SWA/C4/C128 三级，C128 压缩比达 128:1，使得超长上下文的 KV Cache 不会线性爆炸。`swa_full_tokens_ratio=0.1` 说明全量 KV 只占极小比例。

3. **Compress State 是隐藏的内存开销**: 环形缓冲区 (尤其 C128 ring_size=128) 占用不可忽视的显存。Online C128 模式可将 ring_size 降至 1，是重要的优化手段。

4. **并行策略的选择受多重约束**: Shape 对齐 (MXFP4 要求 intermediate%128=0)、MHC o_groups 约束、EP+DP-attention 的 framework bug，这些约束互相耦合，限制了可行配置空间。

5. **TP=16 EP=1 是当前最可靠的配置**: 避免了 EP 相关的所有 bug，满足所有 shape 约束，每 GPU 约 50GB 权重 + 15GB KV Cache + 15GB 激活/Graph，80GB H100 可容纳。
