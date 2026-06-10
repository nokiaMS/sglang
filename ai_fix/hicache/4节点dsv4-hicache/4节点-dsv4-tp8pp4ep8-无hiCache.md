# 4节点 DeepSeek-V4-Pro 性能测试报告（TP=8 PP=4 EP=8，无 HiCache）

## 启动配置

### dist-init-addr地址
172.16.123.37

### 启动命令（有 Marlin）

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
--ep-size 8 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 0 \
--dist-init-addr 172.16.123.37:20000 \
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
--ep-size 8 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 1 \
--dist-init-addr 172.16.123.37:20000 \
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
--ep-size 8 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 2 \
--dist-init-addr 172.16.123.37:20000 \
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
--ep-size 8 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 3 \
--dist-init-addr 172.16.123.37:20000 \
--context-length 1048576 \
--mem-fraction-static 0.90 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--host 0.0.0.0 \
--port 8080
```

### 启动命令（无 Marlin）

去掉 `--moe-runner-backend marlin` 参数即可，其余参数相同。

### 压测命令

**c10 压测**（max-concurrency=10）：

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

**c30 压测**（max-concurrency=30）：

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

**c60 压测**（max-concurrency=60）：

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

---

## EP8 启动过程分析（2026-06-09）

### 一、有 Marlin（`--moe-runner-backend marlin`）

#### 启动流程时间线（以 Node 0 为例）

| 时间 | 阶段 | 耗时 | 说明 |
|------|------|------|------|
| 21:52:31 | 模型检测 | - | Auto-detected DSV4 routed-expert layout: is_fp4_experts=True |
| 21:52:45~47 | Init torch distributed begin | - | 8个 TP worker 开始初始化 NCCL |
| 21:53:57 | Init torch distributed ends | ~70s | NCCL 初始化完成，mem usage ~1.0 GB |
| 21:53:58 | Load weight begin | - | avail mem ~77.5 GB |
| 21:54:56~58 | Load weight end | ~59s | mem usage 32.02 GB, avail ~45.5 GB |
| 21:54:58 | Memory pool end | - | avail mem ~35.7 GB |
| 21:54:58~59 | MHC prewarm begin | - | 22种 token 数量预热线型（8192→64） |
| 21:55:03~04 | MHC prewarm finished | ~5s | 所有 EP worker 完成 attn+ffn 预热 |
| 21:55:13~23 | Server warmup (DeepGEMM JIT) | - | DeepGEMM JIT 编译多个 GEMM kernel |
| **21:55:23** | **崩溃** | - | CUDA illegal memory access |

#### 崩溃详情

- **错误类型**: `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
- **崩溃位置**: `fused_marlin_moe.py:216` → `swiglu_limit_func` → `F.silu(gate) * up`
- **调用链**:
  ```
  scheduler.run_event_loop()
  → scheduler.event_loop_pp()
  → scheduler._pp_launch_batch()
  → scheduler.run_batch()
  → model_worker.forward_batch_generation()
  → model_runner.forward()
  → model_runner.forward_extend()
  → DeepseekV4ForCausalLM.forward()
  → DeepseekV4Model.forward() (layer forward)
  → fused_marlin_moe()  ← Marlin MoE runner
  → swiglu_limit_func()
  → F.silu(gate) * up  ← CUDA 非法内存访问
  ```
- **触发场景**: 服务 warmup 阶段，DeepGEMM JIT 编译到第4个 kernel（GEMM_NT_F8F8BF16, N=6144, K=7168）时，TP1 EP1 的 scheduler 在执行 Marlin MoE forward 时触发 CUDA 错误
- **其他节点**: Node 1~3 因 PP 通信对端（Node 0）崩溃而收到 `Connection closed by peer`，随后收到 SIGQUIT 信号退出
- **根因分析**: Marlin MoE runner 在 fp4 权重反量化后执行 SwiGLU 激活函数时，`gate` 张量可能存在非法内存地址。由于 CUDA 错误是异步的，实际的非法访问可能发生在 Marlin 的 fp4 反量化 GEMM kernel 中，而 `F.silu` 只是被 CUDA runtime 报告错误的 API 调用点

---

### 二、无 Marlin（不指定 `--moe-runner-backend`）

#### 启动流程时间线（以 Node 0 为例）

| 时间 | 阶段 | 耗时 | 说明 |
|------|------|------|------|
| 21:35:21 | 模型检测 | - | Auto-detected DSV4 routed-expert layout: is_fp4_experts=True |
| 21:35:35~37 | Init torch distributed begin | - | 8个 TP worker 开始初始化 NCCL |
| 21:37:34 | Init torch distributed ends | ~117s | NCCL 初始化完成（比 Marlin 慢约 47s） |
| 21:37:36 | Load weight begin | - | avail mem ~77.5 GB |
| 21:38:34~35 | Load weight end | ~59s | mem usage 30.54 GB, avail ~47.0 GB |
| 21:38:36 | Memory pool end | - | avail mem ~36.8 GB |
| 21:38:36~37 | MHC prewarm begin | - | 22种 token 数量预热线型（8192→64） |
| 21:38:41 | MHC prewarm finished | ~5s | 所有 EP worker 完成 attn+ffn 预热 |
| 21:38:45 | Weight dequant | - | 权重反量化完成 |
| **21:38:48** | **Uvicorn started** | - | HTTP 服务已启动在 0.0.0.0:8080 |
| 21:38:51~59 | Server warmup (DeepGEMM JIT) | - | DeepGEMM JIT 编译开始 |
| **21:39:01** | **崩溃** | - | Hidden size mismatch |

#### 崩溃详情

- **错误类型**: `AssertionError: Hidden size mismatch`
- **崩溃位置**: `fused_moe.py:828` → `fused_experts_impl` 中的 assert 检查
- **调用链**:
  ```
  scheduler.run_event_loop()
  → scheduler.event_loop_pp()
  → scheduler._pp_launch_batch()
  → scheduler.run_batch()
  → model_worker.forward_batch_generation()
  → model_runner.forward()
  → model_runner.forward_extend()
  → DeepseekV4ForCausalLM.forward()
  → DeepseekV4Model.forward() (layer forward)
  → fused_experts_none_to_triton()  ← Triton MoE runner
  → fused_experts()
  → inplace_fused_experts()
  → fused_experts_impl()  ← assert hidden_states.shape[1] == w1.shape[2] - padded_size
  ```
- **触发场景**: Uvicorn HTTP 服务已成功启动，在 warmup 阶段第一次实际推理时，Triton MoE runner 的 `fused_experts_impl` 发现 hidden_states 的维度与 fp4 量化权重 w1 的维度不匹配
- **其他节点**: 同样因 PP 通信对端崩溃收到 SIGQUIT 信号退出
- **根因分析**: DeepSeek-V4-Pro 使用 MXFP4 (fp4) 量化专家权重，fp4 权重的存储形状与原始 fp8/fp16 权重不同（压缩了4倍）。Triton MoE runner (`fused_moe.py`) 不支持 fp4 权重格式，它期望权重是标准的 fp8/fp16 形状，因此 `w1.shape[2]` 与 `hidden_states.shape[1]` 不匹配

---

### 三、对比总结

| 对比项 | 有 Marlin | 无 Marlin |
|--------|-----------|-----------|
| NCCL 初始化耗时 | ~70s | ~117s |
| 权重加载耗时 | ~59s | ~59s |
| 权重显存占用 | 32.02 GB/TP | 30.54 GB/TP |
| MHC prewarm 耗时 | ~5s | ~5s |
| HTTP 服务启动 | 未启动 | 已启动（21:38:48） |
| 崩溃阶段 | Server warmup（DeepGEMM JIT编译时） | Server warmup（首次推理时） |
| 崩溃时间（距启动） | ~2m52s | ~3m40s |
| 错误类型 | CUDA illegal memory access | AssertionError: Hidden size mismatch |
| 崩溃代码路径 | `fused_marlin_moe.py:216` → `swiglu_limit_func` | `fused_moe.py:828` → `fused_experts_impl` |
| MoE Runner | Marlin (fp4 专用) | Triton (不支持 fp4) |
| 可否修复 | 需排查 Marlin fp4 反量化/EP 分片的内存错误 | 无法修复——Triton runner 不支持 fp4 权重格式 |

### 四、结论

1. **无 Marlin 方案不可行**：Triton MoE runner 根本不支持 MXFP4 (fp4) 量化权重格式，`Hidden size mismatch` 是结构性错误，无法通过调参解决。DeepSeek-V4-Pro 的 fp4 专家权重必须使用 Marlin 后端来执行反量化和计算。

2. **有 Marlin 方案存在 Bug**：Marlin 后端能正确加载 fp4 权重并通过形状检查，但在 warmup 阶段首次执行 MoE forward 时触发 CUDA 非法内存访问。错误发生在 `swiglu_limit_func`（`F.silu(gate) * up`）中，但 CUDA 异步错误意味着实际根因可能在 Marlin 的 fp4 GEMM kernel 中——可能的问题包括：
   - EP=8 分片下专家权重的内存布局/索引错误
   - Marlin fp4 反量化 kernel 在特定 shape 下的越界访问
   - EP 通信与 Marlin 计算之间的内存同步问题

3. **建议排查方向**：
   - 先用 EP=1（无 Expert Parallelism）+ Marlin 测试，排除 EP 分片问题
   - 设置 `CUDA_LAUNCH_BLOCKING=1` 获取精确的 CUDA 错误位置
   - 检查 Marlin MoE runner 在 EP 模式下的专家权重分片逻辑
