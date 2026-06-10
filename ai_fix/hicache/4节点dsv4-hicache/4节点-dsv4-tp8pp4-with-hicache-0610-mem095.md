# 4节点 DeepSeek-V4-Pro 性能测试（TP=8 PP=4，有 HiCache，mem=0.95）

## 配置信息

- mem-fraction-static: 0.95
- max-running-requests: 32
- swa-full-tokens-ratio: 0.1（默认）
- disable-cuda-graph: 是
- disable-overlap-schedule: 是
- 代码修复: hybrid_pool_assembler.py start_layer 偏移修复

## 脚本

### dist-init-addr地址
172.16.20.182

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
--dist-init-addr 172.16.20.182:20000 \
--context-length 1048576 \
--mem-fraction-static 0.95 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 32 \
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
--dist-init-addr 172.16.20.182:20000 \
--context-length 1048576 \
--mem-fraction-static 0.95 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 32 \
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
--dist-init-addr 172.16.20.182:20000 \
--context-length 1048576 \
--mem-fraction-static 0.95 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 32 \
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
--dist-init-addr 172.16.20.182:20000 \
--context-length 1048576 \
--mem-fraction-static 0.95 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-io-backend direct \
--max-running-requests 32 \
--host 0.0.0.0 \
--port 8080
```

### 压测命令

**c10** — 在 Node 0 的容器内执行：

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

**c30** — 在 Node 0 的容器内执行：

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
--max-concurrency 30 \
--seed 123
done
```

**c60** — 在 Node 0 的容器内执行：

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

## 测试结果

### 服务启动信息

- max_total_num_tokens: 1,782,784
- max_running_requests: 32
- swa_full_tokens_ratio: 0.1（默认）
- Pod IP: 172.16.20.182 (Pod0), 172.16.107.22 (Pod1), 172.16.78.52 (Pod2), 172.16.123.4 (Pod3)

### c10 详细结果

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 10
Successful requests:                     100
Benchmark duration (s):                  368.56
Total input tokens:                      2180970
Total generated tokens:                  10786
Request throughput (req/s):              0.27
Input token throughput (tok/s):          5917.47
Output token throughput (tok/s):         29.26
Peak output token throughput (tok/s):    59.00
Peak concurrent requests:                12
Total token throughput (tok/s):          5946.73
Concurrency:                             9.33
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   34394.12
Median E2E Latency (ms):                 34984.64
P90 E2E Latency (ms):                    56575.48
P99 E2E Latency (ms):                    69778.62
---------------Time to First Token----------------
Mean TTFT (ms):                          2204.30
Median TTFT (ms):                        1846.25
P99 TTFT (ms):                           9818.39
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          300.57
Median TPOT (ms):                        305.23
P99 TPOT (ms):                           385.01
---------------Inter-Token Latency----------------
Mean ITL (ms):                           301.49
Median ITL (ms):                         190.08
P95 ITL (ms):                            1017.64
P99 ITL (ms):                            2388.93
Max ITL (ms):                            10114.73
==================================================
```

### c30 详细结果

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 30
Successful requests:                     100
Benchmark duration (s):                  184.82
Total input tokens:                      2180970
Total generated tokens:                  10786
Request throughput (req/s):              0.54
Input token throughput (tok/s):          11800.60
Output token throughput (tok/s):         58.36
Peak output token throughput (tok/s):    176.00
Peak concurrent requests:                33
Total token throughput (tok/s):          11858.96
Concurrency:                             26.91
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   49731.11
Median E2E Latency (ms):                 44394.40
P90 E2E Latency (ms):                    84850.44
P99 E2E Latency (ms):                    110380.17
---------------Time to First Token----------------
Mean TTFT (ms):                          4838.16
Median TTFT (ms):                        2509.31
P99 TTFT (ms):                           24754.19
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          447.02
Median TPOT (ms):                        455.15
P99 TPOT (ms):                           1004.15
---------------Inter-Token Latency----------------
Mean ITL (ms):                           448.52
Median ITL (ms):                         188.26
P95 ITL (ms):                            1826.42
P99 ITL (ms):                            2904.28
Max ITL (ms):                            24523.82
==================================================
```

### c60 详细结果

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 60
Successful requests:                     100
Benchmark duration (s):                  132.55
Total input tokens:                      2180970
Total generated tokens:                  10786
Request throughput (req/s):              0.75
Input token throughput (tok/s):          16453.53
Output token throughput (tok/s):         81.37
Peak output token throughput (tok/s):    179.00
Peak concurrent requests:                63
Total token throughput (tok/s):          16534.90
Concurrency:                             44.87
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   59480.03
Median E2E Latency (ms):                 58956.65
P90 E2E Latency (ms):                    85938.99
P99 E2E Latency (ms):                    102705.08
---------------Time to First Token----------------
Mean TTFT (ms):                          26737.98
Median TTFT (ms):                        30881.79
P99 TTFT (ms):                           53458.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          310.85
Median TPOT (ms):                        323.91
P99 TPOT (ms):                           460.51
---------------Inter-Token Latency----------------
Mean ITL (ms):                           311.96
Median ITL (ms):                         190.89
P95 ITL (ms):                            995.95
P99 ITL (ms):                            2423.77
Max ITL (ms):                            9383.57
==================================================
```

### 吞吐量汇总

| 测试 | Input (tok/s) | Output (tok/s) | 状态 |
|------|--------------|----------------|------|
| c10  | 5917.47      | 29.26          | 稳定 |
| c30  | 11800.60     | 58.36          | 稳定 |
| c60  | 16453.53     | 81.37          | 稳定 |

### 延迟汇总

| 指标 | c10 | c30 | c60 |
|------|-----|-----|-----|
| Mean TTFT (ms) | 2204.30 | 4838.16 | 26737.98 |
| Median TTFT (ms) | 1846.25 | 2509.31 | 30881.79 |
| P99 TTFT (ms) | 9818.39 | 24754.19 | 53458.18 |
| Mean TPOT (ms) | 300.57 | 447.02 | 310.85 |
| Median TPOT (ms) | 305.23 | 455.15 | 323.91 |
| P99 ITL (ms) | 2388.93 | 2904.28 | 2423.77 |
| Max ITL (ms) | 10114.73 | 24523.82 | 9383.57 |

### 与其他配置对比

| 配置 | c10 Input | c30 Input | c60 Input | c60稳定? |
|------|-----------|-----------|-----------|----------|
| mem=0.90, swa=0.1, max-run=64 | 5800-7490 | 15473-15562 | 17115 | 否(Run2崩溃) |
| mem=0.90, swa=0.2, max-run=64 | - | - | 11063-11744 | 是 |
| mem=0.92, swa=0.2, max-run=64 | 6015-6065 | 10564-10748 | 11960-12301 | 是 |
| **mem=0.95, swa=0.1, max-run=32** | **5917** | **11801** | **16454** | **是** |

### 结论

1. **mem=0.95 + max-running=32（无swa-ratio参数）是当前最优配置**，c60 吞吐量 16454 tok/s，比 mem=0.92/swa=0.2（11960 tok/s）提升约 **37.5%**
2. 通过限制 max-running-requests=32，有效控制了 SWA KV cache 峰值使用量，避免了 c60 OOM 崩溃，同时无需牺牲 Full KV cache 容量（swa-ratio 保持默认 0.1）
3. c60 吞吐量接近 mem=0.90/swa=0.1 的水平（17115 tok/s），但稳定性大幅提升
4. c30 吞吐量（11801 tok/s）低于 mem=0.90/swa=0.1（15473 tok/s），可能受 max-running-requests=32 限制影响并发能力
5. c60 的 TTFT 明显增高（Mean 26738ms vs c10 的 2204ms），说明 max-running-requests=32 在高并发下请求排队等待时间较长
