# 4节点 DeepSeek-V4-Pro 性能测试报告（TP=8 PP=4，无 HiCache）

## 脚本

### dist-init-addr地址
172.16.78.1

### 启动命令

**Node 0 (rank=0)** — 进入 Pod 0 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-0 -- bash
```

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
python3 -m sglang.launch_server \
--model-path /userdata/dsv4/DeepSeek-V4-Pro \
--served-model-name deepseek-v4-pro \
--trust-remote-code \
--tp 8 \
--pp-size 4 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 0 \
--dist-init-addr 172.16.78.1:20000 \
--context-length 1048576 \
--mem-fraction-static 0.90 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--host 0.0.0.0 \
--port 8080
```

**Node 1 (rank=1)** — 进入 Pod 1 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-1 -- bash
```

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
python3 -m sglang.launch_server \
--model-path /userdata/dsv4/DeepSeek-V4-Pro \
--served-model-name deepseek-v4-pro \
--trust-remote-code \
--tp 8 \
--pp-size 4 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 1 \
--dist-init-addr 172.16.78.1:20000 \
--context-length 1048576 \
--mem-fraction-static 0.90 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--host 0.0.0.0 \
--port 8080
```

**Node 2 (rank=2)** — 进入 Pod 2 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-2 -- bash
```

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
python3 -m sglang.launch_server \
--model-path /userdata/dsv4/DeepSeek-V4-Pro \
--served-model-name deepseek-v4-pro \
--trust-remote-code \
--tp 8 \
--pp-size 4 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 2 \
--dist-init-addr 172.16.78.1:20000 \
--context-length 1048576 \
--mem-fraction-static 0.90 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--host 0.0.0.0 \
--port 8080
```

**Node 3 (rank=3)** — 进入 Pod 3 后执行：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-3 -- bash
```

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
python3 -m sglang.launch_server \
--model-path /userdata/dsv4/DeepSeek-V4-Pro \
--served-model-name deepseek-v4-pro \
--trust-remote-code \
--tp 8 \
--pp-size 4 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 3 \
--dist-init-addr 172.16.78.1:20000 \
--context-length 1048576 \
--mem-fraction-static 0.90 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
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

## 测试环境

| 项目 | 配置 |
|------|------|
| 模型 | DeepSeek-V4-Pro |
| GPU | 4 × 8 NVIDIA H800 80GB HBM3（32 GPU） |
| 框架 | sglang v0.5.12.post1 |
| TP | 8 |
| PP | 4 |
| DP | 1 |
| KV Cache dtype | fp8_e4m3 |
| MoE Runner | marlin |
| HiCache | **未启用** |
| context-length | 1,048,576 (1M) |
| mem-fraction-static | 0.90 |
| disable-cuda-graph | true |
| disable-overlap-schedule | true |

## 显存与 KV Cache

### 汇总对比

| 项目 | tp8pp4 | tp16pp2 (对比) |
|------|--------|---------------|
| 单卡可用显存 | ~77.58 GB | ~77.91 GB |
| 模型权重占用 | 32.02 GB/卡 | 33.92 GB/卡 |
| KV Cache 可用显存 | ~35.57 GB/卡 | ~34.81 GB/卡 |
| bytes_per_full_token | 23,749.84 bytes | 23,749.84 bytes |
| max_total_num_tokens | **1,607,936** | 1,573,888 |
| KV Cache 分配后剩余 | ~35.73 GB/卡 | ~25.75 GB/卡 |
| swa_full_tokens_ratio | 0.1 | 0.1 |
| max_running_requests | 256 | 256 |

> tp8pp4 每卡权重少 1.9GB（TP=8 分片更细），KV Cache 可用空间多 0.76GB，总 token 数多 34,048（+2.2%）。KV Cache 分配后剩余显存多 10GB，内存余量更充裕。

### 各 PP Stage 显存分布

| PP Stage | 节点 | 模型权重 | 权重后可用 | KV Cache 可用 | KV 分配后剩余 |
|----------|------|---------|----------|-------------|-------------|
| PP0 | Node 0 | 32.02 GB | 45.56 GB | 35.57 GB | 35.76 GB |
| PP1 | Node 1 | 31.79 GB | 45.79 GB | 35.57 GB | 35.86 GB |
| PP2 | Node 2 | 31.83 GB | 45.75 GB | 35.57 GB | ~35.73 GB |
| **PP3** | **Node 3** | **34.03 GB** | **43.55 GB** | **35.57 GB** | **33.07 GB** |

> PP3 权重多 2.2GB（34.03 vs 31.79），因为 PP3 包含 embedding 层。这导致 PP3 的 KV Cache 分配后剩余显存仅 33.07GB，比其他 stage 少约 2.7GB。

### 显存分配流程

```
GPU 总显存 80 GB
  │
  ├── CUDA/NCCL 初始化    ~1.0 GB   (mem usage: 0.86~1.25 GB)
  │
  ├── 可用显存             ~77.6 GB  (avail mem at Load weight begin)
  │
  ├── 模型权重             32.0 GB   (PP0) / 31.8 GB (PP1/PP2) / 34.0 GB (PP3)
  │   └── 含 fp8 量化权重 + Marlin MXFP4 experts
  │
  ├── 权重后可用           45.6 GB   (PP0) / 43.5 GB (PP3)
  │
  ├── KV Cache 分配        ~10.0 GB  (权重后可用 - KV 可用 = 45.6 - 35.57 ≈ 10.0 GB)
  │   └── bytes_per_full_token = 23,749.84 bytes
  │   └── full_token = 1,607,936
  │   └── Pool: full=1,607,936, swa=160,768, c4=401,984, c128=12,562
  │
  └── KV 分配后剩余        35.7 GB   (PP0) / 33.1 GB (PP3)
```

### KV Cache 子池分解

| 子池 | Token 数 | 占比 | 用途 |
|------|---------|------|------|
| full | 1,607,936 | 100% | 完整 KV Cache |
| swa | 160,768 | 10% | Sliding Window Attention（= full × swa_full_tokens_ratio） |
| c4 | 401,984 | 25% | C4 chunk（= full × 0.25） |
| c128 | 12,562 | 0.78% | C128 chunk |
| c4_state | 10,048 | 0.62% | C4 状态 |
| c128_state | 160,768 | 10% | C128 状态 |

### 各 PP Stage 显存利用率

| 项目 | PP0 | PP1 | PP2 | PP3 |
|------|-----|-----|-----|-----|
| 总显存 | 80 GB | 80 GB | 80 GB | 80 GB |
| 已使用（权重+KV+运行时） | ~44.3 GB | ~44.1 GB | ~44.3 GB | ~46.9 GB |
| 空闲可利用 | 35.76 GB | 35.86 GB | ~35.73 GB | 33.07 GB |
| 显存利用率 | 55.3% | 55.2% | 55.3% | 58.7% |

> PP3 是显存最紧张的节点（58.7% 利用率），但仍有 33GB 空闲，开启 HiCache 有充足空间。

### 与 tp16pp2 显存对比

| 项目 | tp16pp2 | tp8pp4 | 差异 |
|------|---------|--------|------|
| 单卡权重 (PP0) | 33.92 GB | 32.02 GB | **-1.90 GB** |
| KV 可用显存 | 34.81 GB | 35.57 GB | **+0.76 GB** |
| max_total_num_tokens | 1,573,888 | 1,607,936 | **+34,048 (+2.2%)** |
| KV 分配后剩余 | 25.75 GB | 35.73 GB | **+9.98 GB** |
| 权重加载时间 | ~66s | ~58s | -8s |

> tp8pp4 的 TP=8 使每卡权重分片更细（32GB vs 34GB），加上 PP=4 各 stage 层数相同使得 KV Cache 可用空间更多。KV 分配后剩余显存多出 ~10GB，为开启 HiCache 留出充足空间。

## 压测参数

| 参数 | 值 |
|------|-----|
| 压测工具 | sglang.bench_serving |
| dataset | random-ids |
| input_len | 50,000 tokens |
| output_len | 200 tokens |
| num_prompts | 100 |
| max_concurrency | 10 |
| seed | 123 |
| 运行轮次 | 3 |

## 测试结果

### 3轮结果对比

| 指标 | Run 1 | Run 2 | Run 3 | 平均值 |
|------|-------|-------|-------|--------|
| 成功请求数 | 100 | 100 | 100 | 100 |
| 请求吞吐量 (req/s) | 0.29 | 0.28 | 0.28 | 0.28 |
| 输入 token 吞吐量 (tok/s) | 6,216.29 | 6,204.45 | 6,208.52 | 6,209.75 |
| 输出 token 吞吐量 (tok/s) | 30.74 | 30.68 | 30.70 | 30.71 |
| 峰值输出吞吐量 (tok/s) | 59.00 | 60.00 | 60.00 | 59.67 |
| 总 token 吞吐量 (tok/s) | 6,247.04 | 6,235.13 | 6,239.23 | 6,240.47 |

### 延迟指标

| 指标 | Run 1 | Run 2 | Run 3 | 平均值 |
|------|-------|-------|-------|--------|
| **E2E 延迟 (ms)** | | | | |
| Mean | 32,687 | 32,765 | 32,595 | 32,682 |
| Median | 33,904 | 33,177 | 32,955 | 33,345 |
| P90 | 54,502 | 55,111 | 54,839 | 54,817 |
| P99 | 64,320 | 63,977 | 63,867 | 64,055 |
| **TTFT (ms)** | | | | |
| Mean | 1,980 | 1,889 | 1,950 | 1,940 |
| Median | 1,725 | 1,659 | 1,777 | 1,720 |
| P99 | 6,260 | 6,235 | 6,253 | 6,249 |
| **TPOT (ms)** | | | | |
| Mean | 284.67 | 285.72 | 283.69 | 284.69 |
| Median | 290.50 | 292.53 | 289.08 | 290.70 |
| P99 | 358.76 | 361.06 | 370.45 | 363.42 |
| **ITL (ms)** | | | | |
| Mean | 287.63 | 289.21 | 287.04 | 287.96 |
| Median | 188.30 | 187.32 | 189.15 | 188.26 |
| P95 | 857 | 980 | 866 | 901 |
| P99 | 2,255 | 2,222 | 2,430 | 2,302 |
| Max | 7,353 | 7,379 | 7,357 | 7,363 |

## 与 tp16pp2 基线对比

| 指标 | tp16pp2 基线 | tp8pp4 平均 | 变化幅度 |
|------|-------------|------------|---------|
| 输入吞吐量 (tok/s) | 5,255.72 | 6,209.75 | **+18.1%** |
| 输出吞吐量 (tok/s) | 25.99 | 30.71 | **+18.2%** |
| 峰值输出吞吐量 (tok/s) | 60.00 | 59.67 | -0.6% |
| 请求吞吐量 (req/s) | 0.24 | 0.28 | +16.7% |
| 平均 E2E 延迟 (ms) | 38,987 | 32,682 | **-16.2%** |
| 平均 TTFT (ms) | 2,971 | 1,940 | **-34.7%** |
| 平均 TPOT (ms) | 334.17 | 284.69 | **-14.8%** |
| P99 ITL (ms) | 3,815 | 2,302 | **-39.7%** |
| Max ITL (ms) | 14,956 | 7,363 | **-50.8%** |
| KV max_total_num_tokens | 1,573,888 | 1,607,936 | +2.2% |

## 结论分析

### 1. tp8pp4 全面优于 tp16pp2

在相同硬件（4节点32卡）下，tp8pp4 在所有关键指标上均优于 tp16pp2：
- **吞吐量提升 18%**：输入 5,256 → 6,210 tok/s，输出 26.0 → 30.7 tok/s
- **E2E 延迟降低 16%**：39.0s → 32.7s
- **TTFT 大幅降低 35%**：2.97s → 1.94s，对用户体验影响显著
- **TPOT 降低 15%**：334ms → 285ms，decode 速度明显加快
- **长尾延迟大幅改善**：P99 ITL -40%，Max ITL -51%

### 2. TTFT 改善原因分析

- tp8pp4 中每个 PP stage 只有 1/4 模型层，prefill 时单 stage 计算量更少
- PP=4 使得 prefill 可以流水线化，4 个 stage 可以并行处理不同请求的 prefill
- TP=8 相比 TP=16 减少了 all-reduce 通信开销，单次计算延迟更低
- Mean TTFT 从 2.97s 降至 1.94s，P99 TTFT 从 12.4s 降至 6.2s

### 3. Decode 性能改善

- TPOT 从 334ms 降至 285ms（-15%），对应 decode 速度从 3.0 tok/s/请求 提升至 3.5 tok/s/请求
- 10 并发下总 decode 吞吐 30.7 tok/s vs 26.0 tok/s
- 可能原因：TP=8 的 all-reduce 通信量比 TP=16 少，PP=4 的流水线使 decode 可以重叠执行

### 4. 长尾延迟显著改善

- P99 ITL：3,815ms → 2,302ms（-40%）
- Max ITL：14,956ms → 7,363ms（-51%）
- tp16pp2 中 TP=16 的跨节点通信导致更严重的延迟抖动，tp8pp4 的 TP=8 通信范围限定在单节点内，网络抖动影响更小

### 5. 显存利用

- tp8pp4 每卡权重占用 32.02GB，比 tp16pp2 的 33.92GB 少 1.9GB
- KV Cache 分配后剩余 35.73GB vs 25.75GB，内存余量多 10GB
- max_total_num_tokens 1,607,936 vs 1,573,888，基本持平（+2.2%）
- tp8pp4 在内存效率上也更优，为后续开启 HiCache 留出更多空间

### 6. PP=4 的潜在影响

- PP=4 意味着 4 次跨节点通信（每 stage 间），但 TP=8 限定在单节点内，总体通信模式更优
- PP 增加会引入 pipeline bubble，但在当前 10 并发下影响不大
- 如果并发数更低，PP=4 的 bubble 开销可能更明显

### 7. 推荐配置

在 4 节点 32 卡 H800 环境下，**推荐使用 tp=8 pp=4** 而非 tp=16 pp=2，理由：
1. 吞吐量提升 18%
2. TTFT 降低 35%，用户体感明显
3. 长尾延迟大幅改善，服务更稳定
4. 显存余量更多，适合开启 HiCache 等高级特性

---

## c30 压测（max-concurrency=30）

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
--max-concurrency 30 \
--seed 123
done
```

### 压测参数

| 参数 | 值 |
|------|-----|
| max_concurrency | **30**（c10 基线为 10） |
| 其他参数 | 同 c10 |

### 3轮结果对比

| 指标 | Run 1 | Run 2 | Run 3 | 平均值 |
|------|-------|-------|-------|--------|
| 成功请求数 | 100 | 100 | 100 | 100 |
| 请求吞吐量 (req/s) | 0.50 | 0.50 | 0.51 | 0.50 |
| 输入 token 吞吐量 (tok/s) | 10,960.62 | 10,933.69 | 11,068.68 | 10,987.66 |
| 输出 token 吞吐量 (tok/s) | 54.21 | 54.07 | 54.74 | 54.34 |
| 峰值输出吞吐量 (tok/s) | 175.00 | 175.00 | 175.00 | 175.00 |
| 峰值并发请求数 | 33 | 33 | 33 | 33 |
| 实际并发数 | 27.04 | 26.90 | 26.98 | 26.97 |
| 总 token 吞吐量 (tok/s) | 11,014.82 | 10,987.76 | 11,123.42 | 11,042.00 |

### 延迟指标

| 指标 | Run 1 | Run 2 | Run 3 | 平均值 |
|------|-------|-------|-------|--------|
| **E2E 延迟 (ms)** | | | | |
| Mean | 53,804 | 53,656 | 53,152 | 53,537 |
| Median | 51,354 | 52,095 | 51,585 | 51,678 |
| P90 | 91,404 | 91,916 | 91,645 | 91,655 |
| P99 | 117,198 | 115,918 | 114,733 | 115,950 |
| **TTFT (ms)** | | | | |
| Mean | 4,718 | 4,707 | 4,865 | 4,763 |
| Median | 2,241 | 2,237 | 2,511 | 2,330 |
| P99 | 25,514 | 25,512 | 25,505 | 25,510 |
| **TPOT (ms)** | | | | |
| Mean | 484.53 | 481.06 | 463.22 | 476.27 |
| Median | 512.78 | 505.67 | 496.97 | 505.14 |
| P99 | 880.47 | 882.04 | 674.81 | 812.44 |
| **ITL (ms)** | | | | |
| Mean | 459.82 | 458.45 | 452.25 | 456.84 |
| Median | 190.03 | 192.39 | 190.38 | 190.93 |
| P95 | 1,867 | 1,854 | 1,923 | 1,881 |
| P99 | 3,230 | 3,188 | 3,715 | 3,377 |
| Max | 24,332 | 24,274 | 24,233 | 24,280 |

## c10 vs c30 对比

### 吞吐量对比

| 指标 | c10 平均 | c30 平均 | 变化幅度 |
|------|----------|----------|---------|
| 输入吞吐量 (tok/s) | 6,209.75 | 10,987.66 | **+77.0%** |
| 输出吞吐量 (tok/s) | 30.71 | 54.34 | **+76.9%** |
| 请求吞吐量 (req/s) | 0.28 | 0.50 | **+78.6%** |
| 峰值输出吞吐量 (tok/s) | 59.67 | 175.00 | **+193.5%** |
| 实际并发数 | 9.40 | 26.97 | +186.9% |

### 延迟对比

| 指标 | c10 平均 | c30 平均 | 变化幅度 |
|------|----------|----------|---------|
| Mean E2E 延迟 (ms) | 32,682 | 53,537 | **+63.8%** |
| Median E2E 延迟 (ms) | 33,345 | 51,678 | +54.9% |
| P99 E2E 延迟 (ms) | 64,055 | 115,950 | +81.0% |
| Mean TTFT (ms) | 1,940 | 4,763 | **+145.5%** |
| Median TTFT (ms) | 1,720 | 2,330 | +35.5% |
| P99 TTFT (ms) | 6,249 | 25,510 | **+308.4%** |
| Mean TPOT (ms) | 284.69 | 476.27 | **+67.3%** |
| Median TPOT (ms) | 290.70 | 505.14 | +73.7% |
| P99 TPOT (ms) | 363.42 | 812.44 | +123.6% |
| Mean ITL (ms) | 287.96 | 456.84 | +58.6% |
| P99 ITL (ms) | 2,302 | 3,377 | +46.7% |
| Max ITL (ms) | 7,363 | 24,280 | **+229.7%** |

## c30 结论分析

### 1. 吞吐量线性扩展，效率约 65%

- 并发从 10 提升至 30（3 倍），吞吐量从 30.7 → 54.3 tok/s（+77%），约为 1.8 倍
- **并发效率** = 77% / 3 = ~26%，即每增加 1 个并发请求，吞吐量增长约 26%
- 实际并发 27（低于 30），系统未能完全饱和到 30 并发
- 输入吞吐量从 6,210 → 10,988 tok/s（+77%），说明 prefill 处理能力也有类似提升

### 2. TTFT 显著恶化 — prefill 排队严重

- Mean TTFT 从 1.94s → 4.76s（+146%），**中位数** 从 1.72s → 2.33s（+36%）
- 中位数增长不多，但 **P99 TTFT** 从 6.2s → 25.5s（+308%），说明部分请求排队时间极长
- 原因：30 并发下同时有多个 50K prefill 请求，prefill 阶段互相竞争 GPU 资源，导致严重的排队效应
- Mean TTFT 是中位数的 2 倍，说明分布严重右偏

### 3. TPOT 上升 67% — decode 带宽竞争

- Mean TPOT 从 285ms → 476ms，decode 速度从 3.5 tok/s/请求 → 2.1 tok/s/请求
- 30 并发下 decode 带宽被更多请求分摊，单请求 decode 速度下降 40%
- P99 TPOT 从 363ms → 812ms（+124%），长尾 decode 延迟也恶化

### 4. 长尾延迟急剧恶化

- Max ITL 从 7,363ms → 24,280ms（+230%），出现 24 秒级的 token 间隔
- P99 ITL 从 2,302ms → 3,377ms（+47%），长尾 ITL 也显著增长
- 高并发下 KV cache 管理压力增大，eviction/reload 操作导致更严重的延迟抖动

### 5. KV Cache 容量压力

- max_total_num_tokens = 1,607,936
- 30 并发 × 50K input = 1,500,000 tokens，接近 KV Cache 上限（93%）
- 加上 output tokens（30 × 200 = 6,000），KV 利用率已达 **93.4%**
- 峰值并发 33 时 KV 需求 1,650,000 tokens，**超过 KV Cache 上限**
- 系统需要频繁做 KV eviction，这是长尾延迟急剧恶化的主要原因之一

### 6. 综合评估

| 维度 | c10 | c30 | 评价 |
|------|-----|-----|------|
| 吞吐量 | 30.7 tok/s | 54.3 tok/s | c30 吞吐量更高 |
| 单请求 decode 速度 | 3.5 tok/s | 2.1 tok/s | c10 体感更好 |
| TTFT | 1.94s | 4.76s | c10 明显更优 |
| 长尾延迟 | 稳定 | 严重抖动 | c10 稳定性远优于 c30 |
| KV 利用率 | ~31% | ~93% | c30 接近容量极限 |

**c30 适合追求总吞吐量的批量处理场景**，但延迟和稳定性不适合在线服务。c10 在延迟和稳定性上更优，适合对用户体验有要求的在线推理场景。

### 7. 与 HiCache 的关联

c30 测试暴露了关键瓶颈：**KV Cache 容量不足**。30 并发 × 50K = 1.5M tokens 已接近 1.6M 上限。HiCache 的潜在价值：
- 将部分 KV cache 卸载到 CPU/磁盘，释放 GPU 显存
- 提升有效 KV 容量，减少 eviction 导致的长尾延迟
- 在高并发场景下收益可能更大（c30 比 c10 更能体现 HiCache 价值）

---

## c60 压测（max-concurrency=60）

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

### 压测参数

| 参数 | 值 |
|------|-----|
| max_concurrency | **60**（c10 基线为 10，c30 为 30） |
| 其他参数 | 同 c10 |

### 3轮结果对比

| 指标 | Run 1 | Run 2 | Run 3 | 平均值 |
|------|-------|-------|-------|--------|
| 成功请求数 | 100 | 100 | 100 | 100 |
| 请求吞吐量 (req/s) | 0.49 | 0.50 | 0.51 | 0.50 |
| 输入 token 吞吐量 (tok/s) | 10,783.58 | 10,840.05 | 11,094.43 | 10,906.02 |
| 输出 token 吞吐量 (tok/s) | 53.33 | 53.61 | 54.87 | 53.94 |
| 峰值输出吞吐量 (tok/s) | 223.00 | 219.00 | 224.00 | 222.00 |
| 峰值并发请求数 | 63 | 63 | 63 | 63 |
| 实际并发数 | 50.51 | 50.03 | 49.85 | 50.13 |
| 总 token 吞吐量 (tok/s) | 10,836.91 | 10,893.66 | 11,149.30 | 10,959.96 |

### 延迟指标

| 指标 | Run 1 | Run 2 | Run 3 | 平均值 |
|------|-------|-------|-------|--------|
| **E2E 延迟 (ms)** | | | | |
| Mean | 102,164 | 100,650 | 97,999 | 100,271 |
| Median | 98,875 | 97,422 | 95,875 | 97,391 |
| P90 | 158,225 | 166,411 | 153,177 | 159,271 |
| P99 | 177,438 | 174,222 | 172,688 | 174,783 |
| **TTFT (ms)** | | | | |
| Mean | 37,973 | 35,659 | 36,345 | 36,659 |
| Median | 43,060 | 41,913 | 43,656 | 42,876 |
| P99 | 83,174 | 81,015 | 77,357 | 80,515 |
| **TPOT (ms)** | | | | |
| Mean | 636.90 | 645.82 | 617.23 | 633.32 |
| Median | 676.70 | 679.25 | 648.09 | 668.01 |
| P99 | 1,037.02 | 1,176.28 | 1,202.23 | 1,138.51 |
| **ITL (ms)** | | | | |
| Mean | 601.32 | 608.76 | 577.50 | 595.86 |
| Median | 192.58 | 195.44 | 192.84 | 193.62 |
| P95 | 2,826 | 2,828 | 2,648 | 2,767 |
| P99 | 5,018 | 4,776 | 4,817 | 4,870 |
| Max | 26,208 | 24,148 | 24,364 | 24,907 |

## c10 / c30 / c60 全量对比

### 吞吐量对比

| 指标 | c10 平均 | c30 平均 | c60 平均 | c30 vs c10 | c60 vs c30 | c60 vs c10 |
|------|----------|----------|----------|-----------|-----------|-----------|
| 输入吞吐量 (tok/s) | 6,209.75 | 10,987.66 | 10,906.02 | +77.0% | **-0.7%** | +75.7% |
| 输出吞吐量 (tok/s) | 30.71 | 54.34 | 53.94 | +76.9% | **-0.7%** | +75.6% |
| 请求吞吐量 (req/s) | 0.28 | 0.50 | 0.50 | +78.6% | 0.0% | +78.6% |
| 峰值输出吞吐量 (tok/s) | 59.67 | 175.00 | 222.00 | +193.5% | +26.9% | +272.1% |
| 实际并发数 | 9.40 | 26.97 | 50.13 | +186.9% | +85.9% | +433.3% |

### 延迟对比

| 指标 | c10 平均 | c30 平均 | c60 平均 | c30 vs c10 | c60 vs c30 | c60 vs c10 |
|------|----------|----------|----------|-----------|-----------|-----------|
| Mean E2E (ms) | 32,682 | 53,537 | 100,271 | +63.8% | **+87.2%** | +206.9% |
| Median E2E (ms) | 33,345 | 51,678 | 97,391 | +54.9% | +88.5% | +192.1% |
| P99 E2E (ms) | 64,055 | 115,950 | 174,783 | +81.0% | +50.7% | +172.9% |
| Mean TTFT (ms) | 1,940 | 4,763 | 36,659 | +145.5% | **+669.7%** | +1790.2% |
| Median TTFT (ms) | 1,720 | 2,330 | 42,877 | +35.5% | **+1740.8%** | +2392.9% |
| P99 TTFT (ms) | 6,249 | 25,510 | 80,515 | +308.4% | +215.6% | +1188.4% |
| Mean TPOT (ms) | 284.69 | 476.27 | 633.32 | +67.3% | +33.0% | +122.4% |
| Median TPOT (ms) | 290.70 | 505.14 | 668.01 | +73.7% | +32.2% | +129.9% |
| P99 TPOT (ms) | 363.42 | 812.44 | 1,138.51 | +123.6% | +40.1% | +213.3% |
| P99 ITL (ms) | 2,302 | 3,377 | 4,870 | +46.7% | +44.2% | +111.6% |
| Max ITL (ms) | 7,363 | 24,280 | 24,907 | +229.7% | +2.6% | +238.2% |

## c60 结论分析

### 1. 吞吐量完全饱和 — c30→c60 零增长

- 输出吞吐量：c30 54.34 → c60 53.94 tok/s（**-0.7%**），完全持平
- 输入吞吐量：c30 10,988 → c60 10,906 tok/s（-0.7%），同样持平
- **结论：系统在 c30 时已达 decode 吞吐上限，继续提高并发无法获得任何吞吐收益**
- 系统是 **decode-bound**：decode 阶段的计算量决定了总吞吐上限，与并发数无关

### 2. TTFT 灾难性恶化 — KV Cache 严重不足

- Mean TTFT 从 c30 的 4.76s 暴增至 c60 的 **36.66s**（+670%）
- **Median TTFT 更达 42.88s**，比 Mean 还高，说明大多数请求都在排队等待 KV 空间
- P99 TTFT 达 80.5s，部分请求等待超过 1 分钟才获得首个 token
- 根本原因：
  - KV Cache 总容量 1,607,936 tokens
  - 60 并发 × 50K input = **3,000,000 tokens**，是 KV 容量的 **187%**
  - 远超容量上限，系统必须频繁 evict 和 reload KV cache
  - 大量请求处于"等 KV 空间 → 被 evict → 重新 prefill"的恶性循环
- Median TTFT > Mean TTFT 的异常分布说明：**多数请求的 TTFT 由 KV 排队时间决定**，而非 prefill 计算时间

### 3. 实际并发远低于设定值

- 设定 max_concurrency=60，实际并发仅 ~50（83%）
- 峰值并发 63，但无法维持
- 系统受限于 KV 容量，无法同时调度 60 个请求进入 decode 阶段
- c30 实际并发 ~27，c60 实际并发 ~50，虽然并发数增加了 86%，但吞吐量未增长
- **多出的并发全部变成了排队等待，而非有效计算**

### 4. TPOT 继续上升但幅度可控

- Mean TPOT 从 c30 的 476ms → c60 的 633ms（+33%）
- decode 速度从 2.1 tok/s/请求 → 1.6 tok/s/请求
- TPOT 上升幅度（+33%）远小于 TTFT（+670%），说明 decode 阶段的竞争是渐进的
- 但单请求 decode 速度仅 1.6 tok/s，用户体验很差

### 5. 长尾延迟保持高位

- P99 ITL：c30 3,377ms → c60 4,870ms（+44%）
- Max ITL：c30 24,280ms → c60 24,907ms（+2.6%），基本持平
- 长尾 ITL 在 c30 时已达 ~24s 上限，c60 并未进一步恶化
- 但 Mean ITL 从 457ms → 596ms（+30%），整体 ITL 分布进一步右移

### 6. KV Cache 是硬瓶颈

| 并发级别 | KV 需求 (tokens) | KV 容量 | 利用率 | 评价 |
|---------|-----------------|---------|--------|------|
| c10 | 500,000 | 1,607,936 | 31% | 充裕 |
| c30 | 1,500,000 | 1,607,936 | 93% | 接近上限 |
| c60 | 3,000,000 | 1,607,936 | **187%** | 严重超限 |

- c30→c60 的 KV 需求从 93% 跳至 187%，远超物理容量
- 系统被迫频繁 evict，被 evict 的请求需要重新 prefill，造成 TTFT 暴涨
- **无 HiCache 下，50K 输入的实际最大有效并发约 30**（KV 利用率 93%）

### 7. 三级并发综合评估

| 维度 | c10 | c30 | c60 |
|------|-----|-----|-----|
| 输出吞吐量 (tok/s) | 30.7 | 54.3 | 53.9 |
| 单请求 decode 速度 (tok/s) | 3.5 | 2.1 | 1.6 |
| Mean TTFT | 1.94s | 4.76s | 36.66s |
| Median TTFT | 1.72s | 2.33s | 42.88s |
| Mean E2E | 32.7s | 53.5s | 100.3s |
| KV 利用率 | 31% | 93% | 187% |
| 适用场景 | 在线推理 | 批量处理 | **不推荐** |

**c60 在任何场景下均不推荐**：吞吐量与 c30 持平，延迟却是 c30 的 2 倍、c10 的 10 倍。

### 8. 无 HiCache 下的最优并发

- **c10**：延迟最优，适合在线服务（TTFT < 2s，decode 3.5 tok/s）
- **c30**：吞吐量最优，适合批量处理（TTFT ~5s 可接受，decode 2.1 tok/s）
- **c60**：无任何收益，纯延迟惩罚
- 无 HiCache 时，**decode 吞吐上限约 54 tok/s**，受限于 GPU 计算能力而非并发数

### 9. HiCache 在高并发下的价值

c60 测试进一步验证了 KV Cache 容量是高并发场景的核心瓶颈：
- c30 已达 KV 容量 93%，c60 超限 187%
- HiCache 能将 KV 卸载到 CPU/磁盘，释放 GPU 显存
- 理想情况下 HiCache 可将有效 KV 容量提升 2-3 倍，使 c60 级别的并发不再受 KV 限制
- **但 HiCache 无法突破 decode 计算瓶颈**（~54 tok/s），只能改善 TTFT 和长尾延迟
- HiCache 对 c30 的收益可能最大：KV 从 93% 降到安全水位，消除 eviction 导致的延迟抖动
