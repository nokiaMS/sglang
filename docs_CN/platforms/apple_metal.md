<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# Apple Silicon with Metal (MLX)

本文档 describes how run SGLang on Apple Silicon using [Metal (MLX)](https://opensource.apple.com/projects/mlx/). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## 安装 SGLang

你可以 install SGLang using one of the methods below.

### 安装 from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install sglang python package
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```

## Launch of the 服务化 Engine

Launch the server with:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --host 0.0.0.0
```

**Key 参数 Explained:**

1. `SGLANG_USE_MLX=1` - Enables the use of MLX as the SGLang runtime backend (if disabled, SGLang will fall back to `torch.mps`, which has less support)
2. `--disable-cuda-graph` - Disables usage of CUDA graph, which is not relevant for Apple Metal.
3. `--disable-overlap-schedule` - Disables overlap scheduling (enabled/not present by default) achieved using MLX's `async_eval()`


## Benchmarking with Requests

`sglang.benchmark_one_batch` calls the synchronous prefill/decode methods directly without going through the scheduler and the overlap code path.

`sglang.benchmark_offline_throughput` can toggle overlap scheduling as it uses the scheduler and the overlap code path by using the flag `--disable-overlap-schedule`.

### Throughput Testing

Basic synchronous one batch throughput:
```bash
SGLANG_USE_MLX=1 python -m sglang.bench_one_batch \
  --model-path <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 10
```

Synchronous offline throughput:
```bash
SGLANG_USE_MLX=1 python -m sglang.bench_offline_throughput \
  --model-path <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --num-prompts 1 \
  --disable-overlap-schedule
```

Asynchronous offline throughput:
```bash
SGLANG_USE_MLX=1 python -m sglang.bench_offline_throughput \
  --model-path <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --num-prompts 1
```
