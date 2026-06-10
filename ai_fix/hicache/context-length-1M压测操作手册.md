# context-length=1M 压测操作手册

## 前置条件

- 已通过 `tsh ssh --user=guoxu root@hd04-cci-k8s-master-1` 登录跳板机
- 两个 pod 已创建且处于运行状态

## 1. 获取 Pod IP

```bash
kubectl get pod -n elm-test dsv4pro-sg-gx-0 -o wide
```

记下 Pod 0 的 IP 地址，用于更新 `--dist-init-addr` 参数。

## 2. 启动 Pod 0 (rank=0)

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-0 -- bash
```

进入容器后执行：

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr <POD0_IP>:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port 8080 \
  --moe-dense-tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 64 \
  --context-length 1048576 \
  --chunked-prefill-size 2048 \
  --enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-io-backend direct
```

> 将 `<POD0_IP>` 替换为 Pod 0 的实际 IP

## 3. 启动 Pod 1 (rank=1)

另开一个终端：

```bash
kubectl exec -n elm-test -it dsv4pro-sg-gx-1 -- bash
```

进入容器后执行：

```bash
SGLANG_SHARED_EXPERT_TP1=1 \
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr <POD0_IP>:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port 8080 \
  --moe-dense-tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 64 \
  --context-length 1048576 \
  --chunked-prefill-size 2048 \
  --enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-io-backend direct
```

> `--dist-init-addr` 与 Pod 0 保持一致，都是 Pod 0 的 IP

## 4. 等待服务就绪

在 Pod 0 的日志中看到 `Server is ready` 后继续。

## 5. 执行压测

在 Pod 0 的容器内执行：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8080 \
  --model /userdata/dsv4/DeepSeek-V4-Pro \
  --dataset-name random-ids \
  --random-input-len 50000 \
  --random-output-len 200 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 123 \
  --random-range-ratio 1.0
```

## 6. 验证要点

- 服务端不挂起（Pod 0 持续输出日志）
- 压测程序正常完成（100% 进度条）
- `Total input tokens` 等于 `--random-input-len` 的值
- `Successful requests` 为 1
- `errors` 为空

## 与 context-length=202752 的差异

| 参数 | context-length=202752 | context-length=1M |
|------|----------------------|-------------------|
| `--context-length` | 202752 | 1048576 |
| `max_req_input_len` | ~64250 | ~64250（受 KV pool 限制，不变） |
| KV 缓存池大小 | 64256 tokens | 64256 tokens（不变，由显存决定） |

> 注意：`context-length` 只设置模型支持的最大上下文长度，不影响 KV 缓存池的实际大小。KV 缓存池大小由 `mem_fraction_static` 和 `bytes_per_full_token` 决定。当前配置下 KV 池容量约 64256 tokens，因此即使 context-length 设为 1M，实际能处理的输入仍受 KV 池限制。
