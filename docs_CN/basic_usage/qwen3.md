# Qwen3-Next 用法

SGLang 从 [这个 PR](https://github.com/sgl-project/sglang/pull/10233) 开始支持 Qwen3-Next-80B-A3B-Instruct 和 Qwen3-Next-80B-A3B-Thinking。

## 使用 SGLang 启动 Qwen3-Next

在 4 张 H100/H200 GPU 上服务 Qwen3-Next 模型：

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

### 配置建议

- `--max-mamba-cache-size`：调大该值可以增加 mamba cache 空间，并提升最大并发请求能力。代价是 KV cache 空间会减少。请根据实际 workload 调整。
- `--mamba-ssm-dtype`：可选 `bfloat16` 或 `float32`。使用 `bfloat16` 可以节省 mamba cache 空间，使用 `float32` 可以获得更准确的结果。默认值为 `float32`。
- `--mamba-full-memory-ratio`：mamba state memory 与 full KV cache memory 的比例。默认值为 0.9。

### Mamba Radix Cache

SGLang 为 Qwen3-Next 模型支持名为 `MambaRadixCache` 的前缀缓存，通过复用计算结果提升推理速度。`MambaRadixCache` 有两个版本：

- `no_buffer`：默认版本，也是其他 hybrid linear 模型的选择。启用时，SGLang 会出于兼容性考虑自动关闭 overlap schedule。
- `extra_buffer`：优化版本，兼容 page size > 1、overlap schedule 和 speculative decoding 等功能，也支持在分支位置存储 mamba state。不过，它要求每个请求额外使用两个 mamba 空间作为 ping-pong buffer。启动服务器时添加 `--mamba-scheduler-strategy extra_buffer` 即可启用。

### EAGLE 投机解码

**说明**：SGLang 已支持在 Qwen3-Next 模型上使用 [EAGLE speculative decoding](/docs_CN/advanced_features/speculative_decoding.md#EAGLE-Decoding)。

**用法**：
添加 `--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 参数即可启用。例如：

``` bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algo NEXTN
```

更多细节见 [这个 PR](https://github.com/sgl-project/sglang/pull/10233)。
