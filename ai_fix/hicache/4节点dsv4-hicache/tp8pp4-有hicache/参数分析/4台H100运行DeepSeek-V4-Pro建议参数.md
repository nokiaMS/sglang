# 4台 H100 运行 DeepSeek-V4-Pro 建议参数（基于 sglang 源码分析与实测数据）

## 硬件环境

- 4 节点 × 8×H100 80GB HBM3 (32 GPUs total)
- 节点间互联: NVLink + InfiniBand

## 推荐配置

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python3 -m sglang.launch_server \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --trust-remote-code \
  --tp 8 \
  --pp-size 4 \
  --moe-runner-backend marlin \
  --nnodes 4 \
  --node-rank <RANK> \
  --dist-init-addr <POD0_IP>:20000 \
  --context-length 1048576 \
  --mem-fraction-static 0.90 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --watchdog-timeout 3600 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct \
  --max-running-requests 64 \
  --swa-full-tokens-ratio 0.2 \
  --host 0.0.0.0 \
  --port 8080
```

## 各参数详解与依据

### 1. 并行策略: TP=8, PP=4

| 选项 | 说明 |
|------|------|
| TP=8 | 每节点 8 GPU 做 tensor parallel，节点内 NVLink 高带宽，通信开销低 |
| PP=4 | 4 节点做 pipeline parallel，节点间仅传输少量 inter-layer 数据 |
| DP=1 | 不开启数据并行（4节点全部用于同一模型实例） |

**源码依据**: `deepseek_v4_hook.py:48-68` 要求 `tp_size <= 8`，`dp_size == 1`。DSV4 仅支持 `round-robin-split` CP 模式，当前配置不开启 CP，TP=8 是单节点最大并行度。

**不建议 TP=16 PP=2**: 跨节点 TP 通信量大（MoE expert 通信 + attention 通信），性能远不如 TP=8 + PP=4。

### 2. swa-full-tokens-ratio: 0.2

| 场景 | swa=0.1 | swa=0.2 |
|------|---------|---------|
| C60 稳定性 | R2 即 OOM | 三轮全部成功 |
| C30 R3 Input (tok/s) | 22351 | 10683 |
| C10 R3 Input (tok/s) | 7726 | 6107 |
| 预热依赖 | 严重（R1→R3 翻倍） | 几乎无 |
| 性能可预测性 | 差 | 好 |

**源码依据**: DSV4 默认 swa=0.1（`deepseek_v4_hook.py:38-41`），但实测 4 节点 H800 在 C60 并发下 swa=0.1 必 OOM。swa=0.2 将 full KV pool 缩小（SWA pool 按 0.2 * full_tokens 分配），大幅降低 GPU 内存压力。

**原理**: DSV4 的 `bytes_per_full_token` 公式（`pool_configurator.py`）中，swa_ratio 作为乘数影响 SWA pool + compressed states 的大小。swa=0.2 让更多 token 走 SWA（滑动窗口注意力），减少 full KV cache 占用，为 prefill 腾出内存。代价是 decode 阶段需重算被丢弃的 KV，TPOT 增加 17-29%。

**建议**: 生产环境用 0.2 换稳定性；如果业务并发确定 ≤30 且可接受预热期性能差，可用 0.1。

### 3. mem-fraction-static: 0.90

| 值 | C10 吞吐 | C30 吞吐 | C60 稳定性 |
|----|----------|----------|------------|
| 0.90 | 基准 | 基准 | swa=0.2 三轮成功 |
| 0.95 | +2-3% | +2-3% | swa=0.1 时 R2 OOM |

**源码依据**: `server_args.py:1395-1589` 自动计算公式为 `(gpu_mem - reserved_mem) / gpu_mem`，H100 80GB 自动计算约 0.88-0.90。0.90 是保守安全值。

**原理**: mem-fraction-static 决定 GPU 内存中 KV cache 占比，剩余留给 prefill 激活值。0.95 比 0.90 多 5% 给 KV cache，但 prefill 可用内存减少 4GB/卡，在高并发下更易 OOM。

**建议**: 0.90。多 5% KV cache 空间带来的吞吐提升仅 2-3%，但高并发 OOM 风险显著增加。

### 4. max-running-requests: 64

**源码依据**: DSV4 默认 256（`deepseek_v4_hook.py:25-28`），但 256 对 4 节点 H100 过高。

**原理**: max-running-requests 限制同时运行的请求数。每请求需 prefill 内存（50K token 约 30MB/请求），256 请求理论需 7.5GB prefill 内存，超出 GPU 可用空间。64 是 C60 实测稳定的上限。

**建议**: 64。如果业务并发更低（如 ≤30），可适当降低到 48 或 32，减少调度压力。

### 5. HiCache: enabled, ratio=2.0, write_through, direct

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| enable-hierarchical-cache | True | 开启层级缓存，将冷 KV offload 到 Host 内存 |
| hicache-ratio | 2.0 | Host KV cache = 2.0 × Device KV cache |
| hicache-write-policy | write_through | 新 KV 同时写入 Device 和 Host，保证一致性 |
| hicache-io-backend | direct | 使用 direct IO 传输，避免内核缓冲区开销 |

**源码依据**: `hybrid_pool_assembler.py:230-256` 计算 host pool 大小 = hicache_ratio × device_pages。DSV4 的 HiCache 栈包含 8 种 pool（KV, SWA, C4, C128, C4_INDEXER, C4_STATE, C128_STATE, C4_INDEXER_STATE），ratio=2.0 为每种 pool 在 host 保留 2 倍 device 容量。

**hicache-io-backend**: `server_args.py:3450-3531` 中，FA3 decode + kernel IO 会自动切换到 direct。手动指定 direct 避免自动切换的不确定性。

**注意**: HiCache 有预热效应（swa=0.1 时 C30 R1→R3 吞吐 +87%），swa=0.2 下预热效应消失但性能更稳定。

### 6. disable-cuda-graph: True

**源码依据**: `server_args.py:1301-1368` 列出了 piecewise cuda graph 的多种自动禁用条件。DSV4 使用自定义 `dsv4` attention backend（`deepseek_v4_hook.py:14` 强制设置），不在标准 cuda graph 支持列表中。

**原理**: DSV4 的 page_size=256（强制，`deepseek_v4_hook.py:16`），与标准 cuda graph 的 batch size 对齐方式不兼容。强制启用可能导致 crash。

**建议**: 保持 `--disable-cuda-graph`。如果未来 sglang 版本原生支持 DSV4 cuda graph，可移除此参数。

### 7. disable-overlap-schedule: True

**源码依据**: `server_args.py:3347` 当 `pp_size > 1` 时 overlap schedule 已自动禁用。

**原理**: PP=4 时 pipeline 各 stage 间有 bubble，overlap schedule 无法有效重叠 compute 和 communication。显式禁用避免无效的调度开销。

**建议**: PP>1 时此参数冗余但无害，保持显式指定更清晰。

### 8. context-length: 1048576

**源码依据**: DSV4 模型默认 `max_position_embeddings = 65536`（`configs/deepseek_v4.py`）。超出此值时 `model_config.py:541-570` 会警告，但 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` 时允许覆盖。

**原理**: 1048576 (1M) 支持 1M 上下文长度请求。HiCache 使得长上下文成为可能（冷 KV offload 到 host），但实际单请求最大长度受 KV cache 容量约束。

**建议**: 1048576。如果业务确认不需要超长上下文，可降低到 262144 (256K) 减少 KV cache 预分配。

### 9. moe-runner-backend: marlin

**源码依据**: `server_args.py:245-257` 列出所有可选后端。DSV4 的 FP4 expert 自动检测（`model_config.py:252-268`，`SGLANG_DSV4_FP4_EXPERTS=1` 默认开启），marlin 是 FP4 MoE 推理的高性能后端。

**可选方案对比**:
| 后端 | 适用场景 |
|------|----------|
| marlin | FP4 权重，4节点 TP8PP4，当前最佳 |
| triton | 兼容性好但性能低于 marlin |
| cutlass | FP8/FP4 GEMM，可作为备选 |
| deep_gemm | 需要特定硬件支持 |

**建议**: marlin。如遇兼容性问题可尝试 cutlass。

### 10. kv_cache_dtype: 无需指定

**源码依据**: DSV4 强制 fp8_e4m3（`deepseek_v4_hook.py:29-33`），断言只支持此格式。无需手动指定。

### 11. 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| SGLANG_SHARED_EXPERT_TP1 | 1 | 共享专家不做 TP 切分，减少通信 |
| SGLANG_ENABLE_UNIFIED_RADIX_TREE | 1 | 统一 radix tree 管理 full + SWA token |

**源码依据**: DSV4 的 shared expert 在 TP=8 时只需 1 份副本（`moe_dense_tp_size=1`，由 `deepseek_v4_hook.py:55` 强制设置）。unified radix tree 使 HiCache 可以统一管理 full 和 SWA pool 的 token 映射。

## 不同场景的参数调整

### 场景 A: 高并发生产服务（≥30 并发）

使用上面的推荐配置即可，swa=0.2 保证稳定性。

### 场景 B: 低并发高质量服务（≤10 并发）

可调优参数：
```bash
--swa-full-tokens-ratio 0.1    # 低并发不 OOM，decode 更快
--mem-fraction-static 0.95     # 更多 KV cache 空间，吞吐 +2-3%
--max-running-requests 32      # 降低调度开销
```

代价：冷启动性能差（需预热），突发高并发可能 OOM。

### 场景 C: 最大吞吐离线推理

```bash
--swa-full-tokens-ratio 0.1    # 预热后吞吐最高
--max-running-requests 64      # 保持高并发
--context-length 262144        # 减少预分配，增加 token 容量
```

代价：必须先跑 warmup 请求，不可用于在线服务。

### 场景 D: 超长上下文（>100K token）

```bash
--context-length 1048576       # 支持 1M 上下文
--swa-full-tokens-ratio 0.2    # 必须 0.2，长上下文 KV cache 压力大
--hicache-ratio 4.0            # 增大 host 缓存，offload 更多冷 KV
--max-running-requests 16      # 减少并发，为长上下文 prefill 留内存
```

## 关键源码参考

| 参数 | 源码位置 |
|------|----------|
| swa_full_tokens_ratio | `arg_groups/deepseek_v4_hook.py:38-41` (DSV4 默认 0.1) |
| mem_fraction_static | `server_args.py:1395-1589` (自动计算) |
| max_running_requests | `arg_groups/deepseek_v4_hook.py:25-28` (DSV4 默认 256) |
| DSV4 强制参数 | `arg_groups/deepseek_v4_hook.py:14-36` (attention=dsv4, page_size=256, kv_cache_dtype=fp8_e4m3) |
| KV pool 大小计算 | `pool_configurator.py:238-375` |
| HiCache host pool | `hybrid_pool_assembler.py:230-256` |
| overlap schedule 禁用 | `server_args.py:3347` (pp_size>1 自动禁用) |

## 不建议修改的参数

| 参数 | 默认值 | 原因 |
|------|--------|------|
| kv_cache_dtype | fp8_e4m3 (DSV4 强制) | 源码断言仅支持 fp8_e4m3 |
| page_size | 256 (DSV4 强制) | 源码强制设置，修改会 crash |
| attention_backend | dsv4 (DSV4 强制) | 源码强制设置 |
| moe_dense_tp_size | 1 (DSV4 强制) | 源码强制设置 |
