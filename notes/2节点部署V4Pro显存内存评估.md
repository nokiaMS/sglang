# DeepSeek-V4-Pro 2节点部署 显存与内存评估

## 部署命令

```
SGLANG_SHARED_EXPERT_TP1=1
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 172.16.48.119:20000 \
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
  --context-length 202752 \
  --max-running-requests 64 \
  --enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-io-backend direct
```

## 模型参数（DeepSeek-V4-Pro HuggingFace config.json）

| 参数 | 值 |
|------|-----|
| hidden_size | 7168 |
| moe_intermediate_size | 3072 |
| n_routed_experts | 384 |
| n_shared_experts | 1 |
| num_hidden_layers | 61 |
| num_attention_heads | 128 |
| head_dim | 512 |
| qk_nope_head_dim | 448 |
| qk_rope_head_dim | 64 |
| kv_lora_rank | 512 |
| q_lora_rank | 1536 |
| o_lora_rank | 1024 |
| o_groups | 16 |
| index_n_heads | 64 |
| index_head_dim | 128 |
| vocab_size | 129280 |
| num_nextn_predict_layers | 1 |
| tie_word_embeddings | False |
| hc_mult | 4 |
| compress_ratios | 1层ratio=0, 29层ratio=4, 31层ratio=128 |
| sliding_window | 256 |

## 框架自动覆盖的参数

| 用户设定值 | 实际生效值 | 来源 |
|-----------|----------|------|
| `page_size=64` | **256** | `deepseek_v4_hook.py:16` 强制覆盖 |
| `swa_full_tokens_ratio` | **0.1** | `deepseek_v4_hook.py:49`（仅当用户未显式指定时覆盖默认值0.8） |
| `kv_cache_dtype` | **fp8_e4m3** | `deepseek_v4_hook.py` 强制 |

## TP=16 分布参数

| 参数 | 值 |
|------|-----|
| attn_tp_size | 16 |
| moe_tp_size | 16 |
| n_local_heads | 128/16 = 8 |
| n_local_groups | 16/16 = 1 |
| intermediate_size_per_partition | 3072/16 = 192 |
| num_local_experts | 384 (全部，EP=1) |

---

# 一、GPU显存评估

## 1. 模型权重

### 1.1 全局权重（非逐层）

| 权重 | 每GPU形状 | dtype | 每GPU大小 |
|------|----------|-------|----------|
| embed_tokens.weight | [8080, 7168] | BF16 | 110.4 MB |
| lm_head.weight | [8080, 7168] | FP8 | 55.2 MB |
| lm_head.weight_scale_inv | [8080, 56] | FP32 | 1.8 MB |
| norm.weight | [7168] | BF16 | 0.01 MB |
| hc_head_fn | [4, 28672] | FP32 | 0.44 MB |
| hc_head_base | [4] | FP32 | ~0 |
| hc_head_scale | [1] | FP32 | ~0 |
| **全局小计** | | | **167.9 MB** |

### 1.2 逐层权重 - 注意力层（所有61层相同）

| 权重 | 每GPU形状 | dtype | 每GPU大小 |
|------|----------|-------|----------|
| wqkv_a.weight | [2048, 7168] | FP8 | 14.0 MB |
| wqkv_a.weight_scale_inv | [2048, 56] | FP32 | 0.44 MB |
| q_norm.weight | [1536] | BF16 | ~0 |
| wq_b.weight | [4096, 1536] | FP8 | 6.0 MB |
| wq_b.weight_scale_inv | [4096, 12] | FP32 | 0.19 MB |
| kv_norm.weight | [512] | BF16 | ~0 |
| wo_a.weight | [1024, 4096] | BF16 | 8.0 MB |
| wo_b.weight | [7168, 1024] | FP8 | 7.0 MB |
| wo_b.weight_scale_inv | [7168, 8] | FP32 | 0.22 MB |
| attn_sink | [128] | FP32 | ~0 |
| **注意力小计** | | | **35.9 MB** |

### 1.3 逐层权重 - Compressor

**ratio=4（29层）**：

| 权重 | 形状 | dtype | 大小 |
|------|------|-------|------|
| ape | [4, 1024] | FP32 | 0.016 MB |
| wkv_gate.weight | [2048, 7168] | BF16 | 28.0 MB |
| norm.weight | [512] | FP32 | ~0 |
| **compressor(ratio=4)小计** | | | **28.0 MB** |

**ratio=128（31层）**：

| 权重 | 形状 | dtype | 大小 |
|------|------|-------|------|
| ape | [128, 512] | FP32 | 0.25 MB |
| wkv_gate.weight | [1024, 7168] | BF16 | 14.0 MB |
| norm.weight | [512] | FP32 | ~0 |
| **compressor(ratio=128)小计** | | | **14.3 MB** |

### 1.4 逐层权重 - Indexer（仅ratio=4的29层）

| 权重 | 形状 | dtype | 大小 |
|------|------|-------|------|
| wq_b.weight | [8192, 1536] | BF16 | 24.0 MB |
| weights_proj.weight | [64, 7168] | BF16 | 0.88 MB |
| compressor.ape | [4, 256] | FP32 | 0.004 MB |
| compressor.wkv_gate.weight | [512, 7168] | BF16 | 7.0 MB |
| compressor.norm.weight | [128] | FP32 | ~0 |
| **indexer小计** | | | **31.9 MB** |

### 1.5 逐层权重 - MHC（所有61层）

| 权重 | 形状 | dtype | 大小 |
|------|------|-------|------|
| hc_attn_fn | [24, 28672] | FP32 | 2.63 MB |
| hc_ffn_fn | [24, 28672] | FP32 | 2.63 MB |
| hc_attn/base/scale | 小维度 | FP32 | ~0 |
| **MHC小计** | | | **5.27 MB** |

### 1.6 逐层权重 - RMSNorm（所有61层）

| 权重 | 大小 |
|------|------|
| input_layernorm + post_attention_layernorm | **0.027 MB** |

### 1.7 逐层权重 - MoE（所有61层）

**Gate**：

| 权重 | 形状 | dtype | 大小 |
|------|------|-------|------|
| gate.weight | [384, 7168] | BF16 | 5.25 MB |
| gate.e_score_correction_bias | [384] | FP32 | ~0 |
| **gate小计** | | | **5.25 MB** |

**Routed Experts（MXFP4 Marlin，384个expert，TP=16，无SM90 padding）**：

| 权重 | 每GPU形状 | dtype | 每GPU大小 |
|------|----------|-------|----------|
| w13_weight | [384, 384, 3584] | uint8 | 504.0 MB |
| w2_weight | [384, 7168, 96] | uint8 | 252.0 MB |
| w13_weight_scale（运行时） | [384, 224, 384] | float8_e8m0fnu | 31.5 MB |
| w2_weight_scale（运行时） | [384, 6, 7168] | float8_e8m0fnu | 15.75 MB |
| w13_weight_scale_inv（泄漏） | [384, 384, 224] | float32 | 126.0 MB |
| w2_weight_scale_inv（泄漏） | [384, 7168, 6] | float32 | 63.0 MB |
| **routed experts小计** | | | **992.3 MB** |

> **显存泄漏**：`mxfp4_marlin_moe.py` 在 `create_weights` 中注册 `w13_weight_scale_inv`（float32），`prepare_moe_mxfp4_layer_for_marlin` 处理后创建新参数 `w13_weight_scale`（float8_e8m0fnu），但**不删除旧参数**。因参数名不同（`_inv` vs 无`_inv`），旧参数作为僵尸属性残留在GPU上，运行时从未被引用。每层浪费 189 MB，61层共 **~11.5 GB**。

**Shared Expert（tp_size=1，完全复制）**：

| 权重 | 每GPU形状 | dtype | 每GPU大小 |
|------|----------|-------|----------|
| gate_up_proj.weight | [6144, 7168] | FP8 | 42.0 MB |
| gate_up_proj.weight_scale_inv | [6144, 56] | FP32 | 1.31 MB |
| down_proj.weight | [7168, 3072] | FP8 | 21.0 MB |
| down_proj.weight_scale_inv | [7168, 24] | FP32 | 0.66 MB |
| **shared expert小计** | | | **65.0 MB** |

**MoE每层总计**：5.25 + 992.3 + 65.0 = **1,062.5 MB**

### 1.8 逐层总汇总

| 组件 | ratio=0（1层） | ratio=4（29层） | ratio=128（31层） |
|------|---------------|----------------|------------------|
| 注意力 | 35.9 | 35.9 | 35.9 |
| Compressor | — | 28.0 | 14.3 |
| Indexer | — | 31.9 | — |
| MHC | 5.27 | 5.27 | 5.27 |
| RMSNorm | 0.027 | 0.027 | 0.027 |
| MoE | 1,062.5 | 1,062.5 | 1,062.5 |
| **每层总计** | **1,103.7** | **1,163.6** | **1,118.0** |

### 1.9 MTP层（1层，compress_ratio=0）

MTP层包含一个完整的 `DeepseekV4DecoderLayer`（含MoE），加上MTP特有的投影层：

| 权重 | 形状 | dtype | 大小 |
|------|------|-------|------|
| enorm.weight | [7168] | BF16 | 0.014 MB |
| hnorm.weight | [7168] | BF16 | 0.014 MB |
| e_proj.weight | [7168, 7168] | FP8 | 49.0 MB |
| e_proj.weight_scale_inv | [7168, 56] | FP32 | 1.56 MB |
| h_proj.weight | [7168, 7168] | FP8 | 49.0 MB |
| h_proj.weight_scale_inv | [7168, 56] | FP32 | 1.56 MB |
| hc_head_fn/base/scale | — | FP32 | 0.44 MB |
| shared_head.norm | [7168] | BF16 | 0.014 MB |
| **MTP特有小计** | | | **101.6 MB** |
| decoder层（同ratio=0） | | | **1,103.7 MB** |
| **MTP层总计** | | | **1,205.3 MB** |

### 1.10 模型权重总计

| 组件 | 数量 | 每GPU大小 |
|------|------|----------|
| 全局权重 | 1 | 167.9 MB |
| ratio=0 层 | 1 | 1,103.7 MB |
| ratio=4 层 | 29 | 33,744.4 MB |
| ratio=128 层 | 31 | 34,658.0 MB |
| MTP层 | 1 | 1,205.3 MB |
| **模型权重总计** | | **70,879.3 MB** |
| **模型权重总计** | | **~69.6 GB** |

> 其中包含泄漏的 scale_inv 参数：61 × 189 MB = **~11.5 GB**（可回收但当前代码未回收）

### 1.11 显存泄漏修复后的模型权重

若删除泄漏的 scale_inv 参数：

| | 含泄漏 | 修复后 |
|---|--------|--------|
| routed experts 每层 | 992.3 MB | 803.3 MB |
| MoE 每层 | 1,062.5 MB | 873.5 MB |
| **模型权重总计** | **~69.6 GB** | **~58.1 GB** |

---

## 2. 运行时GPU显存分配

### 2.1 总可用显存

| 项目 | 值 |
|------|-----|
| GPU总显存 | 80 GB |
| mem-fraction-static | 0.9 |
| 静态分配上限 | 80 × 0.9 = **72 GB** |

### 2.2 模型权重占用

| | 含泄漏 | 修复后 |
|---|--------|--------|
| 模型权重 | ~69.6 GB | ~58.1 GB |

### 2.3 KV Cache

**bytes_per_full_token 计算**（`DSV4PoolConfigurator._get_bytes_per_full_token`）：

```
kv_bytes = qk_nope(448) + qk_rope×2(128) + FP8 scales+pad(8) = 584 bytes/token/层
indexer_bytes = index_head_dim(128) + scale(4) = 132 bytes/token/层
swa_ratio = 0.1（DSV4 hook强制覆盖）
```

| 项 | 公式 | 值 |
|----|------|-----|
| SWA KV | 0.1 × 584 × 61 | 3,562 |
| C4 KV | 0.25 × 584 × 29 | 4,234 |
| C128 KV | (1/128) × 584 × 31 | 141 |
| C4 indexer | 0.25 × 132 × 29 | 957 |
| C4 state | 0.1 × 0.03125 × 8192 × 29 | 742 |
| C128 state | 0.1 × 0.5 × 4096 × 31 | 6,349 |
| C4 indexer state | 0.1 × 0.03125 × 2048 × 29 | 186 |
| **bytes_per_full_token** | | **~16,172 bytes** |

**KV cache 可用空间**：

| | 含泄漏 | 修复后 |
|---|--------|--------|
| 静态分配上限 | 72 GB | 72 GB |
| 模型权重 | ~69.6 GB | ~58.1 GB |
| CUDA上下文+框架 | ~2 GB | ~2 GB |
| CUDA graph缓冲 | ~2-4 GB | ~2-4 GB |
| ReqToToken池 | 65 × 202756 × 4 = 50 MB | 50 MB |
| **KV cache可用** | **< 0 GB** | **~8-10 GB** |

**含泄漏时：显存不足，KV cache几乎无空间。**

**修复后KV cache容量估算**（假设可用 ~9 GB）：

```
full_token = 9 GB / 16,172 bytes ≈ 590,000 tokens
```

各池token数：

| 池 | token数 |
|----|--------|
| SWA tokens | 590,000 × 0.1 = 59,000 |
| C4 tokens | 590,000 / 4 = 147,500 |
| C128 tokens | 590,000 / 128 = 4,609 |

**context-length=202752 与 KV容量**：每个请求最长 202,752 tokens，`max-running-requests=64`，理论峰值需求 = 64 × 202,752 = 12,976,128 tokens。实际可用 ~590,000 tokens，**远不足以支撑64个满长上下文请求**。运行时请求会被调度器限流。

### 2.4 HiCache对GPU显存的影响

**HiCache不占用额外GPU显存。** `hicache-ratio=2.0` 仅控制主机内存（CPU RAM）池大小为GPU池的2倍。所有HiCache host pool分配在CPU侧（mmap）。

`write_through` 策略使GPU中的KV在被淘汰前更早备份到host，不增加GPU显存开销。

### 2.5 CUDA Graph

`cuda-graph-max-bs=64` 捕获 batch size 1~64 的decode图。每张图持有输入/输出缓冲区。主要开销：
- `next_token_logits_buffer`: 64 × vocab_size × 4B ≈ 31.6 MB
- 中间激活：模型相关，估计 ~2-4 GB

### 2.6 显存总结

| 项目 | 含泄漏 | 修复后 |
|------|--------|--------|
| 模型权重 | ~69.6 GB | ~58.1 GB |
| CUDA上下文+框架 | ~2 GB | ~2 GB |
| CUDA graph | ~2-4 GB | ~2-4 GB |
| ReqToToken | ~0.05 GB | ~0.05 GB |
| KV cache可用 | **< 0 GB（不足）** | **~8-10 GB** |
| **总计** | **> 80 GB（OOM）** | **~72 GB** |

### 2.7 结论

**当前配置（含scale_inv泄漏）：80GB H100 显存不足，将导致OOM。**

**若修复泄漏（删除废弃的scale_inv参数）：**
- 模型权重降至 ~58.1 GB，KV cache可用 ~8-10 GB
- KV容量约 59万 full tokens，无法支撑64个满长上下文并发
- 实际运行时 `max-running-requests=64` 会受KV容量限制自动降级
- `context-length=202752` 可用于单请求，但并发数受限于KV池大小

---

# 二、系统内存评估

## 1. 模型权重加载

每个节点需加载模型权重的 1/2（2节点中1个节点的份额）：

| 项目 | 大小/节点 |
|------|----------|
| 模型权重总大小（BF16原始） | ~3.1 TB（估） |
| FP8/MXFP4量化后总大小 | ~1.0 TB（估） |
| 每节点加载量（8/16 GPU） | ~500 GB |

safetensors加载时需暂存CPU内存：**~500 GB**

## 2. HiCache主机内存

`hicache-ratio=2.0` 意味着host池为device池的2倍。以下为修复泄漏后的估算（假设device KV ~9 GB/GPU）：

| Host池 | 每GPU | 每节点（×8） |
|--------|-------|-------------|
| SWA KV host | ~3.8 GB | ~30 GB |
| C4 KV host | ~4.6 GB | ~37 GB |
| C4 Indexer host | ~1.0 GB | ~8 GB |
| C128 KV host | ~0.2 GB | ~2 GB |
| C4 State host | ~0.8 GB | ~6 GB |
| C4 Indexer State host | ~0.2 GB | ~2 GB |
| C128 State host | ~6.8 GB | ~54 GB |
| **HiCache host总计** | **~17.4 GB** | **~139 GB** |

> C128 State host池最大，因为 `state_page_bytes = 128 × 1024 × 4 = 524,288 bytes/page`，31层。

## 3. 框架与运行时开销

| 项目 | 大小/节点 |
|------|----------|
| Python/PyTorch/NCCL | ~20-40 GB |
| NCCL通信缓冲 | ~5-10 GB |
| 临时张量 | ~10-20 GB |
| **框架开销小计** | **~35-70 GB** |

## 4. 系统内存总结

| 项目 | 大小/节点 |
|------|----------|
| 模型权重加载 | ~500 GB |
| HiCache host | ~139 GB |
| 框架开销 | ~35-70 GB |
| **总计** | **~674-709 GB** |
| **每节点内存** | **2 TB** |
| **余量** | **~1.3 TB** |

**系统内存充裕。**

---

# 三、关键风险点

## 1. 显存泄漏（最严重）

`mxfp4_marlin_moe.py` 的 `create_weights` 注册 `w13_weight_scale_inv` / `w2_weight_scale_inv`（float32），`prepare_moe_mxfp4_layer_for_marlin` 创建新参数 `w13_weight_scale` / `w2_weight_scale`（float8_e8m0fnu）但不删除旧参数。参数名不同导致旧参数成为僵尸，61层共浪费 **~11.5 GB** GPU显存。

**修复方法**：在 `marlin_utils_fp4.py` 的 `prepare_moe_mxfp4_layer_for_marlin` 末尾添加：
```python
if hasattr(layer, "w13_weight_scale_inv"):
    del layer.w13_weight_scale_inv
if hasattr(layer, "w2_weight_scale_inv"):
    del layer.w2_weight_scale_inv
```

## 2. page-size被覆盖

`--page-size 64` 被 `deepseek_v4_hook.py` 强制覆盖为 256。DSV4的SWA压缩注意力要求 `page_size == sliding_window == 256`。

## 3. KV容量不足

即使修复泄漏，~9 GB KV cache仅能支撑约59万 full tokens。64个并发请求 × 202K上下文 = 1297万 tokens 的理论峰值远超容量。实际并发数会被调度器自动限流。

## 4. mem-fraction-static=0.9 风险

0.9 是非常激进的分配比例，仅留 10% 显存给非静态分配（CUDA上下文、临时缓冲、CUDA graph等）。若CUDA graph或框架开销超出预估，可能导致运行时OOM。建议降至 0.88。

## 5. MTP层额外开销

MTP层包含完整MoE（1,205 MB），若不使用speculative decoding可考虑禁用以节省 ~1.2 GB。

---

# 四、修复泄漏后的可行配置

修复scale_inv泄漏 + `mem-fraction-static=0.88`：

| 项目 | 大小 |
|------|------|
| 模型权重 | ~58.1 GB |
| CUDA上下文+框架 | ~2 GB |
| CUDA graph | ~3 GB |
| ReqToToken | 0.05 GB |
| **KV cache可用** | **~7.5 GB** |
| **总计** | **~70.7 GB** |
| **80 GB余量** | **~9.3 GB** |

KV容量：7.5 GB / 16,172 B ≈ **48万 full tokens**

实际并发能力（context-length=202752）：
- 48万 / 202,752 ≈ **2个满长上下文请求**
- 若平均上下文较短（如10K tokens），可支撑 ~48个并发

**若要提升KV容量，需要降低模型权重占用**，可考虑：
- 使用更低的TP（如TP=8 单节点）以减少每GPU的expert数量
- 或使用DeepEP+EP来减少每GPU的routed expert权重
