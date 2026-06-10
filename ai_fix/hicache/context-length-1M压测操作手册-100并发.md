# context-length=1M 压测操作手册（100并发）

## 1. 获取 Pod IP

```bash
kubectl get pod -n elm-test dsv4pro-sg-gx-0 -o wide
```

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
  --dist-init-addr 172.16.107.51:20000 \
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

## 3. 启动 Pod 1 (rank=1)

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
  --dist-init-addr 172.16.107.51:20000 \
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

## 4. 等待服务就绪

在 Pod 0 的日志中看到 `The server is fired up and ready to roll!` 后继续。

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
  --num-prompts 100 \
  --max-concurrency 10 \
  --seed 123 \
  --random-range-ratio 1.0
```
