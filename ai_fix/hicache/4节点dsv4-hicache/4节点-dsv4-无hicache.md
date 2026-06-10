# 4节点 DeepSeek-V4-Pro 性能测试报告（无 HiCache）

## 测试环境

| 项目 | 配置 |
|------|------|
| 模型 | DeepSeek-V4-Pro |
| GPU | 4 × 8 NVIDIA H800 80GB HBM3（32 GPU） |
| 框架 | sglang v0.5.12.post1 |
| TP | 16 |
| PP | 2 |
| DP | 1 |
| KV Cache dtype | fp8_e4m3 |
| MoE Runner | marlin |
| HiCache | **未启用** |
| context-length | 1,048,576 (1M) |
| mem-fraction-static | 0.90 |
| disable-cuda-graph | true |
| disable-overlap-schedule | true |

## 显存与 KV Cache

| 项目 | 值 |
|------|-----|
| 单卡可用显存 | ~77.91 GB |
| 模型权重占用 | 33.92 GB/卡 |
| KV Cache 可用显存 | ~34.81 GB/卡 |
| bytes_per_full_token | 23,749.84 bytes |
| max_total_num_tokens | 1,573,888 |
| KV Cache 分配后剩余 | ~25.75 GB/卡 |
| swa_full_tokens_ratio | 0.1 |
| max_running_requests | 256 |

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
| 请求吞吐量 (req/s) | 0.24 | 0.24 | 0.24 | 0.24 |
| 输入 token 吞吐量 (tok/s) | 5,163.70 | 5,282.19 | 5,321.28 | 5,255.72 |
| 输出 token 吞吐量 (tok/s) | 25.54 | 26.12 | 26.32 | 25.99 |
| 峰值输出吞吐量 (tok/s) | 60.00 | 60.00 | 60.00 | 60.00 |
| 总 token 吞吐量 (tok/s) | 5,189.24 | 5,308.31 | 5,347.60 | 5,281.72 |
| 实际并发数 | 9.38 | 9.41 | 9.41 | 9.40 |

### 延迟指标

| 指标 | Run 1 | Run 2 | Run 3 | 平均值 |
|------|-------|-------|-------|--------|
| **E2E 延迟 (ms)** | | | | |
| Mean | 39,631 | 38,858 | 38,471 | 38,987 |
| Median | 39,644 | 38,236 | 37,333 | 38,404 |
| P90 | 66,890 | 64,606 | 65,179 | 65,558 |
| P99 | 82,594 | 80,145 | 79,842 | 80,860 |
| **TTFT (ms)** | | | | |
| Mean | 3,011 | 2,968 | 2,933 | 2,971 |
| Median | 2,498 | 2,454 | 2,558 | 2,503 |
| P99 | 13,649 | 12,309 | 11,225 | 12,394 |
| **TPOT (ms)** | | | | |
| Mean | 339.87 | 332.76 | 329.87 | 334.17 |
| Median | 346.67 | 339.87 | 337.62 | 341.39 |
| P99 | 503.34 | 469.98 | 466.46 | 479.93 |
| **ITL (ms)** | | | | |
| Mean | 343.01 | 336.18 | 332.84 | 337.34 |
| Median | 178.62 | 178.06 | 177.90 | 178.19 |
| P95 | 1,312 | 1,216 | 1,197 | 1,242 |
| P99 | 3,855 | 3,793 | 3,798 | 3,815 |
| Max | 16,655 | 14,115 | 14,097 | 14,956 |

## 结论分析

### 1. 吞吐量

- **输入吞吐量** 平均 ~5,256 tok/s，在 50K 输入 + 10 并发下表现稳定
- **输出吞吐量** 平均 ~26 tok/s，对应 decode 阶段的瓶颈
- **峰值输出吞吐量** 60 tok/s，说明在轻负载下单请求 decode 速度可达 60 tok/s
- 实际并发约 9.4，略低于设定的 max_concurrency=10，说明系统基本能满载调度

### 2. 延迟特征

- **TTFT** 平均约 3s，P99 约 12.4s。50K 输入的 prefill 在 TP16 下约 2.5s（中位数），P99 延迟是中位数的 ~5 倍，说明高并发下 prefill 调度存在排队
- **TPOT** 平均约 334ms/token，decode 速度约 3 tok/s/请求，在 10 并发下各请求共享 decode 带宽
- **E2E 延迟** 平均约 39s（含 prefill 3s + decode 200 tokens × ~185ms），其中 decode 占比约 95%

### 3. 稳定性

- 3轮测试结果一致性好，Run 2/3 相对 Run 1 略有提升（热身效应）
- 100 个请求全部成功，无错误
- P99 ITL 约 3.8s，Max ITL 达 14-16s，存在偶发的长尾延迟（可能与 KV cache eviction 或跨节点通信抖动有关）

### 4. 无 HiCache 下的瓶颈

- KV Cache 总容量 1,573,888 tokens，单请求 50K 输入占用约 50,000 tokens
- 理论最大并发 KV 容量约 31 个请求（1,573,888 / 50,000）
- 在 10 并发下 KV 容量充足，不会出现频繁 eviction
- **decode 阶段是主要瓶颈**：平均 334ms/token，对应每请求 3 tok/s，10 并发下总 decode 吞吐 ~26 tok/s
- 输入吞吐量 5,256 tok/s 远高于输出吞吐量 26 tok/s，说明系统是 decode-bound

### 5. 基线参考

本次测试为 **无 HiCache 基线**，可用于后续对比开启 HiCache 后的性能变化。关注点：
- HiCache 能否减少 KV cache 占用，提升可并发请求数
- HiCache 的 prefetch 开销对 TTFT 的影响
- HiCache 对长尾延迟（P99 ITL）的改善效果
