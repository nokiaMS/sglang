# 4节点 DeepSeek-V4-Pro 性能测试报告（TP=8 PP=4，有 HiCache）

## 脚本

### dist-init-addr地址
172.16.107.11

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
--dist-init-addr 172.16.107.11:20000 \
--context-length 1048576 \
--mem-fraction-static 0.85 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 64 \
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
--dist-init-addr 172.16.107.11:20000 \
--context-length 1048576 \
--mem-fraction-static 0.85 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 64 \
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
--dist-init-addr 172.16.107.11:20000 \
--context-length 1048576 \
--mem-fraction-static 0.85 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 64 \
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
--dist-init-addr 172.16.107.11:20000 \
--context-length 1048576 \
--mem-fraction-static 0.85 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 64 \
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
--max-concurrency 60 \
--seed 123
done
```

### 压测结果（c10）

测试配置：input_len=50000, output_len=200, num_prompts=100, max_concurrency=10

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Benchmark duration (s) | 489.46 | 340.34 | 286.34 |
| Request throughput (req/s) | 0.20 | 0.29 | 0.35 |
| Input token throughput (tok/s) | 4455.91 | 6408.22 | 7616.67 |
| Output token throughput (tok/s) | 22.04 | 31.69 | 37.67 |
| Peak output token throughput (tok/s) | 60.00 | 60.00 | 60.00 |
| Total token throughput (tok/s) | 4477.95 | 6439.91 | 7654.34 |
| Concurrency | 9.51 | 9.31 | 9.17 |
| **Mean E2E Latency (ms)** | 46532.61 | 31693.36 | 26255.36 |
| Median E2E Latency (ms) | 45677.39 | 30663.69 | 26460.25 |
| P90 E2E Latency (ms) | 75201.69 | 53098.23 | 42892.47 |
| P99 E2E Latency (ms) | 98762.06 | 62059.74 | 52224.61 |
| **Mean TTFT (ms)** | 3932.67 | 1894.20 | 1337.68 |
| Median TTFT (ms) | 3071.52 | 1728.17 | 528.37 |
| P99 TTFT (ms) | 21245.23 | 7188.61 | 9224.79 |
| **Mean TPOT (ms)** | 412.27 | 277.33 | 235.53 |
| Median TPOT (ms) | 397.79 | 285.38 | 230.61 |
| P99 TPOT (ms) | 876.59 | 367.71 | 347.65 |
| Mean ITL (ms) | 399.03 | 277.33 | 235.53 |
| Median ITL (ms) | 186.25 | 285.38 | 230.61 |
| P95 ITL (ms) | - | - | - |
| P99 ITL (ms) | 4252.81 | - | - |

> HiCache warmup 效果明显：三轮测试性能持续提升。Input throughput 从 4456 提升至 7617 tok/s（+71%），Mean TTFT 从 3.9s 降至 1.3s（-66%），Mean TPOT 从 412ms 降至 236ms（-43%）。

### 压测结果（c30）

测试配置：input_len=50000, output_len=200, num_prompts=100, max_concurrency=30

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Benchmark duration (s) | 143.62 | 136.03 | 138.50 |
| Request throughput (req/s) | 0.70 | 0.74 | 0.72 |
| Input token throughput (tok/s) | 15186.07 | 16032.50 | 15747.31 |
| Output token throughput (tok/s) | 75.10 | 79.29 | 77.88 |
| Peak output token throughput (tok/s) | 174.00 | 176.00 | 176.00 |
| Total token throughput (tok/s) | 15261.17 | 16111.79 | 15825.19 |
| Concurrency | 25.93 | 25.81 | 25.66 |
| **Mean E2E Latency (ms)** | 37241.51 | 35105.98 | 35541.93 |
| Median E2E Latency (ms) | 36973.15 | 34628.37 | 35637.08 |
| P90 E2E Latency (ms) | 60385.06 | 57945.91 | 59088.31 |
| P99 E2E Latency (ms) | 82880.36 | 75636.36 | 75763.90 |
| **Mean TTFT (ms)** | 3101.04 | 2607.82 | 2708.48 |
| Median TTFT (ms) | 1359.61 | 1509.11 | 1547.63 |
| P99 TTFT (ms) | 13952.60 | 9788.24 | 9799.59 |
| **Mean TPOT (ms)** | 331.23 | 309.41 | 313.18 |
| Median TPOT (ms) | 337.40 | 325.41 | 331.62 |
| P99 TPOT (ms) | 615.67 | 458.11 | 457.80 |

> c30 相比 c10：Input throughput 提升至约 15700 tok/s（+106%），Output throughput 提升至约 78 tok/s（+107%），并发能力显著增强。TPOT 从 236ms 增至 313ms（+33%），TTFT 从 1.3s 增至 2.7s，属于正常并发开销。

### 压测结果（c30，代码修复后）

测试配置：input_len=50000, output_len=200, num_prompts=100, max_concurrency=30, mem-fraction-static=0.90, max-running-requests=64

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Benchmark duration (s) | 204.90 | 162.12 | 122.20 |
| Request throughput (req/s) | 0.49 | 0.62 | 0.82 |
| Input token throughput (tok/s) | 10644.17 | 13452.89 | 17847.27 |
| Output token throughput (tok/s) | 52.64 | 66.53 | 88.26 |
| Peak output token throughput (tok/s) | 162.00 | 167.00 | 171.00 |
| Total token throughput (tok/s) | 10696.81 | 13519.43 | 17935.53 |
| Concurrency | 27.07 | 26.22 | 24.88 |
| **Mean E2E Latency (ms)** | 55460.10 | 42499.52 | 30403.32 |
| Median E2E Latency (ms) | 52857.97 | 38267.31 | 28799.68 |
| P90 E2E Latency (ms) | 95276.34 | 74423.45 | 52456.22 |
| P99 E2E Latency (ms) | 120742.92 | 102414.50 | 67264.87 |
| **Mean TTFT (ms)** | 4775.67 | 4280.43 | 2574.79 |
| Median TTFT (ms) | 2236.07 | 1733.97 | 489.61 |
| P99 TTFT (ms) | 25601.36 | 25638.09 | 11549.21 |
| **Mean TPOT (ms)** | 504.91 | 373.88 | 264.12 |
| Median TPOT (ms) | 524.92 | 373.64 | 239.70 |
| P99 TPOT (ms) | 1045.42 | 704.01 | 416.63 |

> 代码修复（hybrid_pool_assembler.py start_layer 偏移修复）后 c30 性能提升：Run 3 Input throughput 从 15747 提升至 17847 tok/s（+13%），TPOT 从 313ms 降至 264ms（-16%）。HiCache warmup 效果更显著：Input throughput 从 10644 提升至 17847 tok/s（+68%），TTFT 从 4.8s 降至 2.6s（-46%），TPOT 从 505ms 降至 264ms（-48%）。

### 压测结果（c60，代码修复后）

测试配置：input_len=50000, output_len=200, num_prompts=100, max_concurrency=60, mem-fraction-static=0.90, max-running-requests=64

| 指标 | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| Benchmark duration (s) | 109.03 | 103.23 | 100.86 |
| Request throughput (req/s) | 0.92 | 0.97 | 0.99 |
| Input token throughput (tok/s) | 20002.48 | 21126.36 | 21624.44 |
| Output token throughput (tok/s) | 98.92 | 104.48 | 106.94 |
| Peak output token throughput (tok/s) | 211.00 | 231.00 | 245.00 |
| Total token throughput (tok/s) | 20101.40 | 21230.84 | 21731.39 |
| Concurrency | 44.43 | 44.09 | 43.93 |
| **Mean E2E Latency (ms)** | 48449.48 | 45520.08 | 44306.93 |
| Median E2E Latency (ms) | 48813.93 | 46158.25 | 45219.33 |
| P90 E2E Latency (ms) | 70522.93 | 66837.36 | 64756.49 |
| P99 E2E Latency (ms) | 84065.05 | 76402.70 | 74659.86 |
| **Mean TTFT (ms)** | 18522.82 | 15495.26 | 14780.73 |
| Median TTFT (ms) | 13079.22 | 12569.00 | 11994.76 |
| P99 TTFT (ms) | 50319.82 | 44888.54 | 42222.58 |
| **Mean TPOT (ms)** | 291.73 | 295.39 | 290.00 |
| Median TPOT (ms) | 294.88 | 288.12 | 285.01 |
| P99 TPOT (ms) | 600.43 | 616.35 | 472.58 |

> 代码修复前 c60 在 Run 2 必现 Prefill OOM 崩溃。修复后三轮稳定运行无崩溃。HiCache warmup 效果：Input throughput 从 20002 提升至 21624 tok/s（+8%），TTFT 从 18.5s 降至 14.8s（-20%），TPOT 稳定在 ~290ms。
>
> c60 vs c30 对比（Run 3）：Input throughput 从 17847 提升至 21624 tok/s（+21%），Output throughput 从 88 提升至 107 tok/s（+21%），但 TTFT 从 2.6s 增至 14.8s（+474%，排队等待增加），TPOT 从 264ms 增至 290ms（+10%，解码性能基本稳定）。
