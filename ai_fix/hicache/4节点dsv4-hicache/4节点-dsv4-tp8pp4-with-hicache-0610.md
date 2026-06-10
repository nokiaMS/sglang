# 4节点 DeepSeek-V4-Pro 性能测试（TP=8 PP=4，有 HiCache，代码修复后）

## 配置信息

- mem-fraction-static: 0.90
- max-running-requests: 64
- swa-full-tokens-ratio: 0.2
- 代码修复: hybrid_pool_assembler.py start_layer 偏移修复

## 脚本

### dist-init-addr地址
172.16.78.7

### 启动命令

**Node 0 (rank=0)** — 进入 Pod 0 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-0 -- bash
```

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
--node-rank 0 \
--dist-init-addr 172.16.78.7:20000 \
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

**Node 1 (rank=1)** — 进入 Pod 1 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-1 -- bash
```

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
--node-rank 1 \
--dist-init-addr 172.16.78.7:20000 \
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

**Node 2 (rank=2)** — 进入 Pod 2 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-2 -- bash
```

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
--node-rank 2 \
--dist-init-addr 172.16.78.7:20000 \
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

**Node 3 (rank=3)** — 进入 Pod 3 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-3 -- bash
```

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
--node-rank 3 \
--dist-init-addr 172.16.78.7:20000 \
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

### 压测命令

在 Node 0 的容器内执行：

```bash
for i in 1 2 3; do
echo "=== Run $i/3 ==="
python3 -m sglang.bench_serving \
--backend sglang \
--base-url http://127.0.0.1:8080 \
--model /userdata/dsv4/DeepSeek-V4-Pro \
--served-model-name deepseek-v4-pro \
--dataset-name random-ids \
--random-input-len 50000 \
--random-output-len 200 \
--num-prompts 100 \
--max-concurrency 10 \
--seed 123
done
```

## 测试结果

### mem-fraction-static=0.90, swa-full-tokens-ratio=0.1

#### c10 (max-concurrency=10)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 376.00 | 349.06 | 291.19 |
| Input throughput (tok/s) | 5800 | 6248 | 7490 |
| Output throughput (tok/s) | 28.69 | 30.90 | 37.04 |
| Mean TTFT (ms) | 2228 | 1881 | 1136 |
| Mean TPOT (ms) | 305 | 285 | 238 |
| Mean E2E Latency (ms) | 35030 | 32400 | 26622 |

#### c30 (max-concurrency=30)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 140.95 | 141.78 | 140.15 |
| Input throughput (tok/s) | 15473 | 15383 | 15562 |
| Output throughput (tok/s) | 76.52 | 76.08 | 76.96 |
| Mean TTFT (ms) | 2774 | 2652 | 2596 |
| Mean TPOT (ms) | 319 | 317 | 316 |
| Mean E2E Latency (ms) | 36318 | 36196 | 35947 |

#### c60 (max-concurrency=60, swa-full-tokens-ratio=0.1)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 127.43 | 崩溃 | 崩溃 |
| Input throughput (tok/s) | 17115 | - | - |
| Output throughput (tok/s) | 84.64 | - | - |
| Mean TTFT (ms) | 21109 | - | - |
| Mean TPOT (ms) | 367 | - | - |
| Mean E2E Latency (ms) | 59606 | - | - |

**c60 说明**: Run 1 正常完成，Run 2 服务端崩溃（TransferEncodingError + ConnectionRefusedError），Run 3 服务未能恢复（60s 超时）。c60 并发下服务稳定性仍有问题，需要进一步调查。

#### c60 (max-concurrency=60, swa-full-tokens-ratio=0.15)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 崩溃 | - | - |
| Input throughput (tok/s) | - | - | - |
| Output throughput (tok/s) | - | - | - |
| Mean TTFT (ms) | - | - | - |
| Mean TPOT (ms) | - | - | - |
| Mean E2E Latency (ms) | - | - | - |

**说明**: ratio=0.15 在 Run 1 阶段即崩溃，OOM 时 SWA 仅剩 3,584 tokens，`component_evictable_size_=0`。0.15 的 SWA 容量仍不足以支撑 c60 并发。

#### c60 (max-concurrency=60, swa-full-tokens-ratio=0.2)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 197.14 | 186.15 | 185.71 |
| Input throughput (tok/s) | 11063 | 11716 | 11744 |
| Output throughput (tok/s) | 54.71 | 57.94 | 58.08 |
| Mean TTFT (ms) | 35845 | 29825 | 28908 |
| Mean TPOT (ms) | 572 | 589 | 614 |
| Mean E2E Latency (ms) | 94709 | 89805 | 89653 |

### mem-fraction-static=0.92, swa-full-tokens-ratio=0.2

max_total_num_tokens=948,992（vs mem=0.90 的 909,312，增加 4.4%）

#### c10 (max-concurrency=10)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 361.52 | 359.60 | 362.61 |
| Input throughput (tok/s) | 6033 | 6065 | 6015 |
| Output throughput (tok/s) | 29.84 | 29.99 | 29.75 |
| Mean TTFT (ms) | 2000 | 1962 | 1940 |
| Mean TPOT (ms) | 295 | 294 | 297 |
| Mean E2E Latency (ms) | 33743 | 33532 | 33840 |

#### c30 (max-concurrency=30)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 206.45 | 205.16 | 202.93 |
| Input throughput (tok/s) | 10564 | 10631 | 10748 |
| Output throughput (tok/s) | 52.25 | 52.57 | 53.15 |
| Mean TTFT (ms) | 4816 | 4969 | 4786 |
| Mean TPOT (ms) | 502 | 494 | 483 |
| Mean E2E Latency (ms) | 55857 | 55400 | 54873 |

#### c60 (max-concurrency=60)

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Duration (s) | 182.35 | 177.30 | 178.60 |
| Input throughput (tok/s) | 11960 | 12301 | 12212 |
| Output throughput (tok/s) | 59.15 | 60.83 | 60.39 |
| Mean TTFT (ms) | 28986 | 26653 | 27438 |
| Mean TPOT (ms) | 602 | 600 | 593 |
| Mean E2E Latency (ms) | 89197 | 85564 | 86816 |

## c60 OOM 崩溃分析

### 崩溃现象

swa-full-tokens-ratio=0.1 时，c60 Run 1 正常完成，Run 2 开始后约 30 秒服务端崩溃，4 个节点全部退出。swa-full-tokens-ratio=0.15 时，Run 1 阶段即崩溃。

### 根本原因：SWA KV Cache 耗尽导致 Prefill OOM

OOM 发生在 Node 0 (PP0)，错误信息：

```
Prefill out of memory. Try to lower your batch size.
Try to allocate 8192 tokens.
Available full tokens: 899840 (full_available_size=11776 + full_evictable_size_=888064)
Available swa: 3840 (available_size=3840 + component_evictable_size_=0)
```

**关键数据**：
- Full KV cache 有余量（可用 899,840 tokens，其中 888,064 可驱逐，可 offload 到 HiCache 磁盘）
- SWA KV cache 完全耗尽（仅剩 3,840 tokens，`component_evictable_size_=0`，无法驱逐）

### 崩溃时间线

| 时间 | 事件 |
|------|------|
| 14:36:20 | Run 2 warmup 完成，100 个请求涌入，swa token usage 飙升至 0.99 |
| 14:36:28 | Node 0 (PP0) 所有 8 个 TP 同时 Prefill OOM |
| 14:36:28 | Node 1 (PP1) 检测到连接断开，收到 sigquit |
| 14:36:29 | Node 2/3 (PP2/PP3) 收到 Connection closed by peer |

### 为什么 Run 1 不崩溃但 Run 2 崩溃

1. Run 1 从空缓存开始，SWA 逐步填满但能通过驱逐旧请求释放空间
2. Run 1 完成后，KV cache 驻留在 SWA 中未被清除
3. Run 2 新请求涌入时，SWA 需要同时容纳 Run 1 残留 + Run 2 新请求
4. SWA 条目无法被驱逐（被 radix tree 锁定），而 Full 条目可以 offload 到 HiCache → SWA 成为瓶颈

### 解决方案：增大 swa-full-tokens-ratio

将 `swa-full-tokens-ratio` 分别调整为 0.15 和 0.2 测试：

| 参数 | ratio=0.1 | ratio=0.15 | ratio=0.2 |
|------|-----------|------------|-----------|
| max_total_num_tokens | 1,607,936 | 1,249,024 | 909,312 |
| SWA 容量 (tokens) | 160,794 | 187,354 | 181,862 |
| Full 容量 (tokens) | 1,447,142 | 1,061,670 | 727,450 |
| c60 SWA 峰值使用率 | 98-99% | OOM | ~92% |
| c60 Full 峰值使用率 | 43% | - | ~60% |
| c60 Run 1 | 通过 | **OOM 崩溃** | 通过 |
| c60 Run 2 | OOM 崩溃 | - | 通过 |
| c60 Run 3 | OOM 崩溃 | - | 通过 |
| c60 吞吐量 (tok/s) | 17115 | - | 11716 |

### 性能影响对比

| 指标 | ratio=0.1 mem=0.90 (c60 Run1) | ratio=0.2 mem=0.90 (c60 Run2) | ratio=0.2 mem=0.92 (c60 Run2) |
|------|-------------------------------|-------------------------------|-------------------------------|
| Duration (s) | 127.43 | 186.15 | 177.30 |
| Input throughput (tok/s) | 17115 | 11716 | 12301 |
| Output throughput (tok/s) | 84.64 | 57.94 | 60.83 |
| Mean TTFT (ms) | 21109 | 29825 | 26653 |
| Mean TPOT (ms) | 367 | 589 | 600 |
| Mean E2E Latency (ms) | 59606 | 89805 | 85564 |
| 稳定性 | Run2崩溃 | 3轮通过 | 3轮通过 |

## 结论

1. **c10、c30 并发下 swa-full-tokens-ratio=0.1 无问题**，3 轮测试全部稳定通过
2. **c60 并发下 swa-full-tokens-ratio=0.1 在 Run 2 崩溃**，0.15 在 Run 1 即崩溃，原因是 SWA KV cache 耗尽且无法驱逐
3. **swa-full-tokens-ratio=0.2 是 c60 稳定运行的最低要求**，3 轮测试全部通过，但性能下降约 28-60%
4. **性能下降根因**：ratio=0.2 使 max_total_num_tokens 从 1,607,936 降至 909,312（-43%），full KV cache 容量减半，吞吐量相应降低
5. **ratio=0.15 不可行**：虽然比 0.1 增加了 SWA 容量，但仍不足以支撑 c60 并发下新旧请求的 SWA 需求，OOM 时 SWA 仅剩 3,584 tokens
6. **mem-fraction-static=0.92 可小幅提升性能**：max_total_num_tokens 从 909,312 提升至 948,992（+4.4%），c60 吞吐量从 11716 提升至 12301（+5%），TTFT 从 29825ms 降至 26653ms（-11%）
7. **去掉 --disable-cuda-graph 会导致崩溃**：c10 即触发 `CUDA error: illegal memory access`，cuda graph 与 PP=4 不兼容
8. **进一步优化方向**：
   - 修复 sglang 调度器，使 SWA 在 OOM 时能驱逐旧请求而非直接崩溃
   - 在 Run 之间通过 API 清除残留 KV cache
   - 优化 HiCache 的 SWA 驱逐策略，允许在内存紧张时释放被 radix tree 锁定的 SWA 条目
