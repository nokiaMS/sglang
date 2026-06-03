# DeepSeek V4 Pro + SGLang + 4×H100 部署完整问题排查文档

> 本文档记录了在 4 台 H100 (每台 8 卡, 共 32 卡) 上使用 SGLang 部署 DeepSeek V4 Pro 单实例的全部尝试过程、遇到的错误、根因分析及最终结论。

---

## 1. 硬件与模型信息

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | 4 台 H100, 每台 8 卡, 80GB/卡, 共 32 卡 |
| IB 网卡 | ib7s400 (非 Mellanox mlx5, 不支持 NVSHMEM/GPU Direct RDMA) |
| 网络 | InfiniBand, 节点间互通 |
| 模型 | DeepSeek V4 Pro (1.6T MoE, MXFP4 量化) |

### 1.2 模型关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 7168 | 隐藏层维度 |
| moe_intermediate_size | 3072 | 每个 expert 的 FFN 中间维度 |
| n_routed_experts | 256 | 路由专家数量 |
| n_shared_experts | 1 | 共享专家数量 |
| n_layers | 43 | Transformer 层数 |
| num_attention_heads | 64 | 注意力头数 |
| head_dim | 512 | 注意力头维度 |
| kv_lora_rank | 512 | MLA 的 KV LoRA 秩 (MQA) |
| o_groups | 16 | MHC (Multi-Head Consolidation) 组数 |
| 量化格式 | MXFP4 (expert), FP8 (dense) | 路由专家用 MXFP4, 密集层用 FP8 |
| 注意力机制 | DSA/NSA + MHC | DeepSeek Attention + Multi-Head Consolidation |

### 1.3 4 节点 32 卡的目标并行度

| 参数 | 目标值 | 说明 |
|------|--------|------|
| TP | 32 | 总 GPU 数 = 张量并行度 |
| EP | ? | 需要引入 EP 来满足 MoE shape 约束 |
| DP | 2 | 数据并行提高吞吐 |
| PP | 1 | V4 不兼容 PP |

---

## 2. 遇到的全部问题

### 问题 1: TP=8, DP=4, DP-attention → OOM

**原始启动命令:**
```bash
python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --tp 8 --dp-size 4 --enable-dp-attention \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend marlin
```

**错误信息:**
```
torch.OutOfMemoryError at mxfp4_marlin_moe.py
```

**根因分析:** DeepSeek V4 Pro 模型总大小约 800GB。TP=8 时每个 GPU 需加载约 100GB 权重, 超过 H100 的 80GB HBM。

**结论:** TP 必须 >= 16, 4 节点必须使用 TP=32。

---

### 问题 2: TP=32 无 EP → MoE shape 约束不满足

**原始启动命令:**
```bash
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --tp 32 --dp-size 2 --enable-dp-attention \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend marlin
```

**错误信息:**
```
RuntimeError: Current DeepSeekV4 MoE layer does not satisfy Marlin constraints.
```
或 (用 flashinfer_mxfp4 后端时):
```
ValueError: Mxfp4FlashinferCutlassMoEMethod requires hidden_size and
intermediate_size_per_partition to be multiples of 128
(got hidden=7168, intermediate=96)
```

**根因分析:** 无 EP 时 `moe_tp_size = 32`, `intermediate_size_per_partition = 3072/32 = 96`:
- Marlin 要求 `96 % 64 == 0` → `96 % 64 = 32 ≠ 0` ✗
- flashinfer_mxfp4 要求 `96 % 128 == 0` → `96 % 128 = 96 ≠ 0` ✗

**结论:** TP=32 无 EP 时没有任何 MoE 后端可用, 必须引入 EP 来增大 `intermediate_size_per_partition`。

**关键公式:**
```
moe_tp_size = tp_size / ep_size / moe_dp_size
intermediate_size_per_partition = moe_intermediate_size / moe_tp_size
```

---

### 问题 3: TP=16, DP=2, DP-attention → 共享专家 FP8 shape 约束

**原始启动命令:**
```bash
python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --tp 16 --dp-size 2 --enable-dp-attention \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend marlin
```

**错误信息:**
```
ValueError: Weight output_partition_size = 192 is not divisible by
weight quantization block_n = 128
```

**根因分析:** DP-attention 下 `attn_tp_size = 16/2 = 8`, 共享专家用 TP=8 分片, `moe_intermediate_size / tp_size = 3072/16 = 192`, 不满足 FP8 block_n=128 的整除约束。

**解决方案:** 设置环境变量 `SGLANG_SHARED_EXPERT_TP1=1`, 让共享专家完全复制在每个 GPU 上 (TP=1), 不参与 TP 分片。此时共享专家使用完整 `intermediate_size=3072`, `3072 % 128 = 0` ✓

**后续所有命令都必须加上 `SGLANG_SHARED_EXPERT_TP1=1`。**

---

### 问题 4: EP + DeepEP → NVSHMEM 崩溃

**原始启动命令:**
```bash
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --tp 32 --ep-size 16 --dp-size 2 --enable-dp-attention \
  --moe-a2a-backend deepep \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend marlin
```

**错误信息:**
```
cuMemCreate failed
illegal memory access
IBGDA init failed
```

**根因分析:** DeepEP 的 all-to-all 通信依赖 NVSHMEM, 而 NVSHMEM 只支持 Mellanox mlx5 系列 IB 网卡。当前硬件使用 ib7s400 网卡, 不兼容。

**结论:** DeepEP 在当前硬件不可用, 只能使用 `--moe-a2a-backend none` (默认值), 即 StandardDispatcher + all-reduce 模式。

---

### 问题 5: PP=2 → CUDA illegal memory access

**错误信息:**
```
CUDA error: an illegal memory access was encountered (NCCL watchdog)
```

**根因分析:** V4 的 NSA/DSA 多流注意力机制和 MHC 与 Pipeline Parallelism 不兼容。层拆分后跨节点通信崩溃。

**结论:** PP 与 V4 架构永久不兼容, 不能使用 `--pipeline-parallel-size`。

---

### 问题 6: TBO → 缺少 compressor 接口

**错误信息:**
```
AttributeError: 'TboAttnBackend' object has no attribute 'forward_core_compressor'
```

**根因分析:** TBO (Two-Batch Overlap) 的 wrapper 未实现 V4 NSA 所需的 compressor 接口。

**结论:** 不能启用 `--enable-two-batch-overlap`。

---

### 问题 7: TP=32, EP=4 (无 DP-attention) → 注意力层 shape 错误

**错误信息:**
```
RuntimeError: shape '[256, 0, -1]' is invalid for input of size 524288
```

**根因分析:** V4 MHC 机制有 `o_groups=16`, TP=32 时 `attn_tp_size=32`, 导致 `n_local_groups = 16/32 = 0`。

**代码位置:** `deepseek_v4.py:908` — `o = o.view(o.shape[0], self.n_local_groups, -1)`

**结论:** TP=32 时必须使用 DP-attention 来降低 `attn_tp_size`, 使 `n_local_groups >= 1`。

---

### 问题 8: EP=16, marlin 后端 → Marlin kernel 实际推理崩溃

**启动命令:**
```bash
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 32 --ep-size 16 --dp-size 2 --enable-dp-attention \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend marlin --disable-cuda-graph
```

**错误信息 (EP=16, moe_tp_size=2):**
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```
发生在 `fused_marlin_moe.py:166` — `intermediate_cache2 = torch.empty(...)`

**错误信息 (EP=8, moe_tp_size=4):**
```
tvm.error.InternalError: Runtime check failed at moe_wna16_marlin.cuh:812:
CUDA error: an illegal memory access was encountered
```

**根因分析:** Marlin MoE kernel 在 EP + `moe_tp_size>1` 组合下有**内核级 bug**, 不论 `moe_tp_size=2` 还是 `moe_tp_size=4` 都触发 illegal memory access。

最初以为只是 CUDA graph 捕获阶段崩溃 (3.8 节中的 `swiglu_limit_func`), 尝试 `--disable-cuda-graph` 绕过, 但实际推理时同样崩溃。`--disable-cuda-graph` 只能跳过 CUDA graph 捕获, 无法修复 kernel 本身的 bug。

**结论:** **Marlin + EP 这条路彻底走不通**, 这是 Marlin MoE kernel 的根本性 bug, 不是配置问题。

---

### 问题 9: EP=16, moe_dp_size=2, attn_cp_size=2 → OOM

**启动命令:**
```bash
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 32 --ep-size 16 --dp-size 2 --enable-dp-attention \
  --moe-data-parallel-size 2 --attn-cp-size 2 \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend marlin
```

**错误信息:**
```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```
日志头标识: `DP1 ATTN_CP1 MOE_DP1 TP30 EP14`

**根因分析:**

`--moe-data-parallel-size 2` 使 `moe_tp_size = 32/16/2 = 1`, MoE 专家权重不做 TP 分片。每 GPU 持有的 MoE 权重从 ~12GB (moe_tp=2 分片) 翻倍到 ~24GB (不分片), 导致 `available_bytes` 为负值 (-15.37 GB)。

**为什么必须同时设 `--attn-cp-size 2`:** 框架约束 (`server_args.py:3242`) 要求 `attn_cp_size != moe_dp_size` 时 `moe_dp_size == 1`。设了 `moe_dp_size=2` 就必须设 `attn_cp_size=2`, 但这又导致 OOM — 是死路。

**关键理解:**
- `moe_dp_size=1` → MoE 权重做 TP 分片 → 内存正常
- `moe_dp_size=2` → MoE 权重不分片 → 内存翻倍 → OOM
- 对大 MoE 模型 (256 experts, 43 layers), 这个翻倍是致命的

**结论:** 4 节点 32 卡部署 V4 Pro 时, `--moe-data-parallel-size` 不可用, 必须保持 `moe_dp_size=1`。

---

### 问题 10: EP=32, flashinfer_cutlass 后端 → 量化方法路由错误

**启动命令:**
```bash
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 32 --ep-size 32 --dp-size 2 --enable-dp-attention \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend flashinfer_cutlass --disable-cuda-graph
```

**错误信息:**
```
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
```
发生在 `fp8.py:1769`

**根因分析:** V4 Pro 的 MXFP4 权重需要走 MXFP4 代码路径, 但 `Fp8Config.get_quant_method()` 的调度逻辑只在 `is_marlin()` 或 `is_flashinfer_mxfp4()` 两个后端时路由到 MXFP4 方法。

`flashinfer_cutlass` 和 `flashinfer_trtllm` 是 **FP8 MoE 后端**, 它们不处理 MXFP4 权重布局, 落到了 `Fp8MoEMethod`, 导致类型错误。

**V4 Pro MXFP4 的后端路由规则** (`fp8.py:252-297`):

| `--moe-runner-backend` | `is_fp4_experts=True` 时路由到 | 说明 |
|---|---|---|
| `marlin` | `Mxfp4MarlinMoEMethod` | ✓ 正确路由 |
| `flashinfer_mxfp4` | SM90: `Mxfp4FlashinferCutlassMoEMethod`; SM100+: `Mxfp4FlashinferTrtllmMoEMethod` | ✓ 正确路由 |
| `flashinfer_cutlass` | `Fp8MoEMethod` | ✗ 错误! 不处理 MXFP4 |
| `flashinfer_trtllm` | `Fp8MoEMethod` | ✗ 错误! 不处理 MXFP4 |

**结论:** V4 Pro MXFP4 模型**只能**用 `--moe-runner-backend marlin` 或 `--moe-runner-backend flashinfer_mxfp4`。

**注意:** `flashinfer_mxfp4` 在 H100 (SM90) 上内部使用 cutlass mixed-input kernel, 名字虽然含 "flashinfer", 但实际 GEMM 实现与 `flashinfer_cutlass` CLI 参数对应的 FP8 后端完全不同。

---

### 问题 11: EP=32, flashinfer_trtllm 后端 → TopKOutput 格式不匹配

**启动命令:**
```bash
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 32 --ep-size 32 --dp-size 2 --enable-dp-attention \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend flashinfer_trtllm
```

**错误信息:**
```
AssertionError at flashinfer_trtllm.py:686:
assert TopKOutputChecker.format_is_bypassed(topk_output)
```

**根因分析:** 与问题 10 同一根因 — `flashinfer_trtllm` 走了 FP8 代码路径, 期望 `BYPASSED` 格式的 TopK 输出, 但 V4 MXFP4 场景下 TopK 输出是 `STANDARD` 格式。

---

### 问题 12: EP=16/32, flashinfer_mxfp4 后端 → dp_scatter 崩溃 (根本性不兼容)

**启动命令:**
```bash
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 32 --ep-size 16 --dp-size 2 --enable-dp-attention \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --moe-runner-backend flashinfer_mxfp4
```

**错误信息:**
```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```
发生在 `deepseek_v4.py:934` → `dp_scatter()` → `memcpy_triton_kernel()`

EP=32 时同样崩溃, 崩溃位置完全一致。

**根因分析 — 两个互锁的 sglang 框架 bug:**

**Bug A: 错误的 DP buffer 分配组**

`deepseek_v4.py:1408,1430` 在分配 DP buffer 时使用了 `get_tp_group()`, 但 EP>1 时应使用 `get_attention_tp_group()`:

```python
# deepseek_v4.py:1408,1430 (BUG)
get_global_dp_buffer(get_tp_group())       # 应该用 get_attention_tp_group()
get_local_dp_buffer(get_tp_group())        # 应该用 get_attention_tp_group()

# communicator.py:1264-1267 (已修复的正确写法)
group = get_tp_group() if tp_size == attn_dp_size else get_attention_tp_group()
```

同类 bug 已在 `communicator.py` 中修复 (commit `86c6c77f2`), 但 `deepseek_v4.py` 遗漏了。

**Bug B: StandardDispatcher 的专家映射错误 (更严重)**

`StandardDispatcher` 在 EP>1 + `moe_a2a_backend=none` 时, 将非本地专家映射为 -1:

```python
# standard.py:209-211
self.local_expert_mapping = torch.full((self.num_experts,), -1, ...)
# EP=16时, 256个expert中只有16个本地expert被映射为0-15, 其余240个都是-1
# MoE kernel尝试访问index=-1的权重 → illegal memory access
```

CUDA 错误是异步的, 所以 actual crash 发生在后续的 `dp_scatter`, 但根源在 MoE kernel 的 expert mapping。

**为什么 DeepEP 能解决:** `--moe-a2a-backend deepep` 走不同的代码路径 (`_use_tp_attn_a2a_scatter`), 正确处理 EP 分发/合并, 避免 `dp_scatter` 和 expert mapping 问题。但 DeepEP 需要 NVSHMEM, 当前硬件不支持。

**结论:** **EP>1 + `moe_a2a_backend=none` + DP-attention 在当前 sglang 版本对 V4 根本性不兼容**, 属于框架 bug, 不是配置问题。

---

## 3. 关键技术概念详解

### 3.1 并行度公式

```
moe_tp_size = tp_size / ep_size / moe_dp_size
attn_tp_size = tp_size / dp_size / attn_cp_size
intermediate_size_per_partition = moe_intermediate_size / moe_tp_size
n_local_groups = o_groups / attn_tp_size
每GPU本地路由专家数 = n_routed_experts / ep_size
```

### 3.2 EP 对 shape 约束的影响 (tp_size=32)

| EP | moe_dp | moe_tp_size | intermediate_per_partition | marlin (%64) | flashinfer_mxfp4 (%128) | 实际结果 |
|----|--------|-------------|---------------------------|-------------|------------------------|---------|
| 1 | 1 | 32 | 96 | 96%64=32 ✗ | 96%128=96 ✗ | 无可用后端 |
| 2 | 1 | 16 | 192 | 192%64=0 ✓ | 192%128=64 ✗ | — |
| 4 | 1 | 8 | 384 | 384%64=0 ✓ | 384%128=0 ✓ | — |
| 8 | 1 | 4 | 768 | 768%64=0 ✓ | 768%128=0 ✓ | Marlin kernel bug |
| 16 | 1 | 2 | 1536 | 1536%64=0 ✓ | 1536%128=0 ✓ | Marlin/flashinfer_mxfp4 都崩 |
| 16 | 2 | 1 | 3072 | 3072%64=0 ✓ | 3072%128=0 ✓ | OOM (MoE 权重不分片) |
| 32 | 1 | 1 | 3072 | 3072%64=0 ✓ | 3072%128=0 ✓ | dp_scatter 崩溃 |

### 3.3 SGLANG_SHARED_EXPERT_TP1 环境变量

**作用:** 让共享专家完全复制在每个 GPU 上 (TP=1), 不参与 TP 分片。

**为什么需要:** V4 Pro 共享专家的 FP8 量化 shape 约束 — `moe_intermediate_size / tp_size` 必须被 128 整除。TP>=16 时 `3072/16=192`, `192%128=64≠0`, 不满足。设为 TP1 后共享专家用完整 intermediate_size=3072, `3072%128=0` ✓

**用法:** 所有启动命令前加 `SGLANG_SHARED_EXPERT_TP1=1`

### 3.4 Expert Parallelism (EP) 两种模式

| 特性 | moe_a2a_backend=none | moe_a2a_backend=deepep |
|------|---------------------|----------------------|
| 通信模式 | StandardDispatcher + all-reduce | all-to-all |
| Token 分发 | 每个 GPU 看到所有 token | 只把 token 发给持有对应 expert 的 GPU |
| 结果合并 | `tensor_model_parallel_all_reduce` | reduce-scatter |
| 通信量 | 大 | 小 |
| 硬件要求 | 无特殊要求 | 需要 NVSHMEM + Mellanox IB |
| V4 + DP-attention 兼容性 | ✗ 有 bug | ✓ 正确路径 |

### 3.5 DP-attention

**作用:** 将 TP 组细分为注意力 DP 组, `attn_tp_size = tp_size / dp_size`

**V4 为什么必须用:** TP=32 时 `attn_tp_size=32`, 但 `o_groups=16`, 导致 `n_local_groups=0` (shape 错误)。DP=2 时 `attn_tp_size=16`, `n_local_groups=1` ✓

### 3.6 moe_dp_size 与内存的关系

| moe_dp_size | moe_tp_size | MoE 权重 | 内存影响 |
|-------------|-------------|---------|---------|
| 1 (默认) | tp/ep | 做 TP 分片 | 正常 |
| 2 | tp/ep/2 | 分片减半 | 权重翻倍, 可能 OOM |
| ep | 1 | 完全不分片 | 每GPU持完整本地专家权重, 大概率 OOM |

### 3.7 日志头标识解读

日志前缀格式: `DP{d} ATTN_CP{c} MOE_DP{m} TP{t} EP{e}`

| 标签 | 条件 | 含义 | 诊断 |
|------|------|------|------|
| DP{n} | `dp_size > 1` | 数据并行 rank | |
| ATTN_CP{n} | `attn_cp_size > 1` | 注意力上下文并行 rank | |
| MOE_DP{n} | `moe_dp_size > 1` | MoE 数据并行 rank | 出现此标签 → 会 OOM |
| TP{n} | `tp_size > 1` | 张量并行 rank | TP30 → moe_dp=2 |
| EP{n} | `ep_size > 1` | 专家并行 rank | EP14 → moe_dp=2 |

### 3.8 KV Cache 内存分配 (DSV4)

DeepSeek V4 使用 `DSV4PoolConfigurator`, 将可用内存分配到 6 个池:

| 池名 | 用途 | 大小比例 |
|------|------|---------|
| full | 全序列 KV cache | 基准 (1x) |
| swa | 滑动窗口 KV cache | `full * swa_full_tokens_ratio` (默认 0.1) |
| c4 | 4x 压缩注意力 | `full / (4 * c4_shrink_factor)` |
| c128 | 128x 压缩注意力 | `full / 128` |
| c4_state | c4 在线状态 | swa 相关 |
| c128_state | c128 在线状态 | swa 相关 |

OOM 发生在 `pool_configurator.py:56`, 当 `available_bytes` 不够分配时。

**不能禁用 KV cache** — 它是自回归推理的基础, 没有它模型无法做增量生成。正确的解决方式是选择合适的并行度配置。

---

## 4. 4 节点单实例部署不可行的完整原因链

```
4节点32卡
  → TP=32
    → 必须EP (无EP时intermediate_per_partition=96, 无MoE后端可用)
    → 必须DP-attention (无DP-attention时n_local_groups=0)
      → EP + DP-attention + moe_a2a_backend=none
        → dp_scatter崩溃 + expert mapping=-1 (framework bug)
      → 需要 moe_a2a_backend=deepep
        → 需要NVSHMEM
          → 需要Mellanox IB
            → ib7s400不支持 ✗
      → Marlin + EP → kernel illegal memory access (kernel bug)
      → moe_dp_size=2 → OOM (MoE权重不分片)
      → flashinfer_cutlass/trtllm → 量化方法路由错误
```

**所有路径汇总:**

| 路径 | 问题 | 性质 |
|------|------|------|
| marlin + EP | kernel illegal memory access | sglang kernel bug |
| marlin + EP + --disable-cuda-graph | 同上, kernel bug 不只影响 CUDA graph | sglang kernel bug |
| flashinfer_mxfp4 + EP | dp_scatter 崩溃 | sglang framework bug |
| flashinfer_cutlass | 错误路由到 FP8 代码, 不处理 MXFP4 | 设计限制 |
| flashinfer_trtllm | 同上 + TopK 格式不匹配 | 设计限制 |
| moe_dp_size=2 | OOM | 设计限制 (模型太大) |
| moe_a2a_backend=deepep | 硬件不支持 NVSHMEM | 硬件限制 |

---

## 5. 需要 sglang 修复的 bug 列表

| # | Bug | 文件位置 | 严重程度 | 说明 |
|---|-----|---------|---------|------|
| 1 | DP buffer 用错 group | `deepseek_v4.py:1408,1430` | HIGH | 应用 `get_attention_tp_group()` 而非 `get_tp_group()`, 同类 bug 已在 `communicator.py` 修复 |
| 2 | StandardDispatcher expert mapping=-1 | `standard.py:209-211` | CRITICAL | EP>1 时非本地专家映射为 -1, MoE kernel 访问非法地址 |
| 3 | Marlin MoE kernel EP 崩溃 | `moe_wna16_marlin.cuh` | HIGH | EP + moe_tp_size>1 时 illegal memory access |
| 4 | should_use_dp_reduce_scatterv 不支持 EP!=DP | `moe/utils.py:418` | MEDIUM | 只处理 EP==DP, EP=16 DP=2 走到 broken path |

---

## 6. 可行的替代方案

### 方案 A: 2 节点 TP=16 (官方推荐, 已验证)

```bash
# 节点0
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --moe-runner-backend marlin \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --host 0.0.0.0 \
  --port 8080

# 节点1
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --moe-runner-backend marlin \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 172.16.186.201:29500
```

**特点:**
- TP=16, EP=1, DP=1, 无需 DP-attention
- moe_tp_size=16, intermediate_per_partition=192, `192%64=0` ✓
- CUDA graph 可正常启用, 无性能损失
- 每GPU持全部 256 个 routed expert, MoE 权重约 ~50GB/GPU, 80GB 可放下
- 使用 2 个节点, 另外 2 个节点空闲

### 方案 B: 4 节点跑 2 个独立实例 (推荐)

用 4 节点跑 2 个独立的 2 节点实例, 外部负载均衡 (nginx/HAProxy) 分发请求:

**实例 1** (节点 0+1):
```bash
# 节点0
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --moe-runner-backend marlin \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 172.16.186.201:29500 \
  --host 0.0.0.0 \
  --port 8080

# 节点1
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --moe-runner-backend marlin \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 172.16.186.201:29500
```

**实例 2** (节点 2+3):
```bash
# 节点2
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --moe-runner-backend marlin \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 172.16.186.202:29500 \
  --host 0.0.0.0 \
  --port 8080

# 节点3
SGLANG_SHARED_EXPERT_TP1=1 python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --moe-runner-backend marlin \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 172.16.186.202:29500
```

**特点:**
- 2 个独立实例, 总吞吐量约 2x 单实例
- 需要 2 个不同的 `--dist-init-addr` (不同 IP 或不同端口)
- CUDA graph 正常启用, 性能无损
- 硬件利用率 100%

### 方案 C: 等待 sglang 修复

需要修复第 5 节列出的 4 个 bug, 可在 sglang GitHub 提 issue。

### 方案 D: 更换 Mellanox IB 网卡

如果更换为 Mellanox ConnectX-7 等 IB 网卡, 可使用 DeepEP (`--moe-a2a-backend deepep`), 正确处理 EP + DP-attention。但 Marlin kernel bug 仍需修复。

---

## 7. 不兼容特性汇总

| 特性 | 原因 | 严重程度 | 是否可能修复 |
|------|------|---------|------------|
| PP (Pipeline Parallelism) | V4 NSA/MHC 不兼容 | 永久 | 否 |
| TBO (Two-Batch Overlap) | V4 NSA compressor 不兼容 | 永久 | 否 |
| DeepEP | ib7s400 不支持 NVSHMEM | 硬件限制 | 换网卡 |
| Marlin + EP | kernel illegal memory access | sglang bug | 是 |
| EP + moe_a2a_backend=none + DP-attention | dp_scatter 崩溃 + expert mapping=-1 | sglang bug | 是 |
| moe_dp_size>1 | MoE 权重不分片 → OOM | 设计限制 | 否 (模型太大) |
| flashinfer_cutlass/trtllm + MXFP4 | 量化方法路由错误 | 设计限制 | 需新增路由逻辑 |
| KV cache 禁用 | 不可行, 自回归推理基础 | 设计限制 | 否 |

---

## 8. 各部署方案对比

| 参数 | 2 节点 (官方) | 4 节点单实例 | 4 节点双实例 (推荐) |
|------|-----------|-----------|----------------|
| 可行性 | ✓ 已验证 | ✗ 不可行 | ✓ |
| TP | 16 | 32 | 16×2 |
| EP | 1 | 16/32 | 1×2 |
| DP | 1 | 2 | 1×2 |
| moe_tp_size | 16 | N/A | 16×2 |
| MoE 后端 | marlin | N/A | marlin×2 |
| CUDA graph | 启用 | N/A | 启用×2 |
| 总吞吐量 | 1x | N/A | ~2x |
| 硬件利用率 | 50% (2/4 节点) | N/A | 100% |
| 单请求延迟 | 正常 | N/A | 正常 |
| 额外组件 | 无 | N/A | 外部负载均衡 |
