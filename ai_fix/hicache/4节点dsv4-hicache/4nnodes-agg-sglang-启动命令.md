# dsv4-pro-sglang-agg-tp16dp1pp2

## 环境

* DeepSeek-V4-Pro
* 4 nodes, 8× NVIDIA H800 80GB HBM3 per node (32 GPUs total)
* sglang v0.5.12.post1

## 脚本
### dist-init-addr地址
172.16.78.1

### 启动脚本模板
SGLANG_SHARED_EXPERT_TP1=1 \
python3 -m sglang.launch_server \
--model-path /userdata/dsv4/DeepSeek-V4-Pro \
--served-model-name deepseek-v4-pro \
--trust-remote-code \
--tp 16 \
--pp-size 2 \
--moe-runner-backend marlin \
--nnodes 4 \
--node-rank 0 \
--dist-init-addr <dist-init-addr地址>:20000 \
--context-length 1048576 \
--mem-fraction-static 0.90 \
--disable-cuda-graph \
--disable-overlap-schedule \
--watchdog-timeout 3600 \
--host 0.0.0.0 \
--port 8080

### 4节点启动命令

#### Node 0 pod名称：dsv4pro-sg-gx-0， 命令空间：elm-test
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
--tp 16 \
--pp-size 2 \
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

#### Node 1 pod名称：dsv4pro-sg-gx-1， 命令空间：elm-test
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
--tp 16 \
--pp-size 2 \
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

#### Node 2 pod名称：dsv4pro-sg-gx-2， 命令空间：elm-test
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
--tp 16 \
--pp-size 2 \
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

#### Node 3 pod名称：dsv4pro-sg-gx-3， 命令空间：elm-test
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
--tp 16 \
--pp-size 2 \
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

> 注意：`--dist-init-addr` 使用的是 Node 0 (rank=0) 的 IP（172.16.78.1），所有节点的该参数保持一致。Pod 重建后 IP 可能变化，需重新获取并更新。

### 性能测试脚本

for i in 1 2 3; do
echo "=== Run $i/3 ==="
python -m sglang.bench\_serving \
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