<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# Checkpoint Engine Integration

The SGLang checkpoint engine integration provides an efficient way to load model weights using a distributed checkpoint loading system. This feature significantly reduces model loading time, especially for large models and multi-node setups, by parallelizing the weight loading process across multiple processes and nodes.

## 概览

The checkpoint engine integration allows SGLang to:
- Load model weights in parallel using multiple processes
- Distribute weight loading across multiple nodes to increase effective disk bandwidth
- Overlap weight loading with other initialization tasks like CUDA graph capture
- 支持 both single-node and multi-node deployments

## 安装

First, install the checkpoint engine package:

```bash
pip install 'checkpoint-engine[p2p]'
```

## Architecture

The system consists of two main components:

1. **SGLang 服务器**: Runs with `--wait-for-initial-weights` flag to wait for weights before becoming ready
2. **Checkpoint Engine Workers**: Separate processes (managed by torchrun) that load and distribute model weights

The checkpoint engine uses a parameter server architecture with support for:
- **Broadcast mode**: Weights are broadcast from loading processes to inference processes
- **P2P mode**: Direct peer-to-peer weight transfer between processes
- **All mode**: Combination of both broadcast and P2P methods

## 用法 示例

### Single Node Setup

**Terminal 1 - Launch SGLang 服务器:**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights
```

**Terminal 2 - Run Checkpoint Engine:**

Using sglang entrypoint:
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### Multi-Node Setup (2 Nodes)

**Node 0:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

**Node 1:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### Multi-Node Setup with Tensor Parallelism (TP=16)

**Node 0:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 0
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

**Node 1:**

Launch SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 1
```

Run checkpoint engine:

Using sglang entrypoint (recommended):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

Using torchrun directly:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

## 配置 选项

### SGLang 服务器 选项

- `--load-format dummy`: Use dummy format for initial loading (allows overlapping with other tasks)
- `--wait-for-initial-weights`: Wait for checkpoint engine to provide weights before becoming ready
- `--host`: Host address for multi-node setups
- `--dist-init-addr`: Distributed initialization address for tensor parallelism

### Checkpoint Engine 选项

- `--update-method`: Weight update method (`broadcast`, `p2p`, or `all`)
- `--checkpoint-path`: Path to model checkpoint directory
- `--inference-parallel-size`: Number of inference parallel processes
- `--endpoint`: SGLang server endpoint (default: `http://localhost:19730`)
- `--checkpoint-name`: Name for the checkpoint (default: `my-checkpoint-iter-0`)
- `--save-metas-file`: File to save checkpoint metadata
- `--load-metas-file`: File to load checkpoint metadata from
- `--uds`: Unix domain socket path for communication
- `--weight-version`: Version identifier for weights

## 性能 Benefits

The checkpoint engine provides significant time savings in two main aspects:

1. **Multi-node Loading**: Each node only loads a portion of weights from disk, effectively increasing disk bandwidth. More participating nodes provide greater acceleration. Preliminary tests show 20-second acceleration when loading DeepSeek-R1 on H20-3e with two nodes.

2. **Single Process Optimization**: Using dummy format allows overlapping disk-to-CPU transfer with CUDA graph capture and other initialization tasks, providing additional time savings.

## 故障排查

- Ensure checkpoint engine package is installed: `pip install 'checkpoint-engine[p2p]'`
- Verify network connectivity between nodes in multi-node setups
- Check that the checkpoint path contains valid model files
- Monitor logs for connection errors between SGLang server and checkpoint engine
- Use `--sleep-time` parameter to add delays if needed for debugging

## 参考资料

- [Checkpoint Engine Repository](https://github.com/MoonshotAI/checkpoint-engine)
