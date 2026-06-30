# Qwen 3.5 用法

Qwen 3.5 是阿里巴巴最新一代大语言模型，具备 hybrid attention 架构、带共享专家的高级 MoE，以及原生多模态能力。

主要架构特性：

- **Hybrid Attention**：将 Gated Delta Networks（线性复杂度 O(n)）与每 4 层一次的 full attention 结合，以获得较强的关联召回能力。
- **带共享专家的 MoE**：64 个 routed experts 中激活 top-8，同时包含一个专用于通用特征的 shared expert。
- **多模态**：使用带 Conv3d 的 DeepStack Vision Transformer，原生支持图像和视频理解。

## 使用 SGLang 启动 Qwen 3.5

### Dense 模型

在 8 张 GPU 上服务 `Qwen/Qwen3.5-397B-A17B`：

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B-A17B \
    --tp 8 \
    --trust-remote-code
```

### AMD GPU（MI300X / MI325X / MI35X）

在 AMD Instinct GPU 上，请使用 `triton` attention backend。full attention 层和 Gated Delta Net（linear attention）层在 ROCm 上都使用基于 Triton 的 kernel：

```bash
SGLANG_USE_AITER=1 python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B-A17B \
    --tp 8 \
    --attention-backend triton \
    --trust-remote-code
```

```{tip}
设置 `SGLANG_USE_AITER=1` 可启用 AMD 为 MoE 和 GEMM 操作优化的 aiter kernel。
```

### 配置建议

- `--attention-backend`：在 AMD GPU 上运行 Qwen 3.5 时使用 `triton`。Hybrid attention 架构（Gated Delta Networks + full attention）在 ROCm 上配合 Triton backend 效果最好。linear attention（GDN）层内部始终通过 `GDNAttnBackend` 使用 Triton kernel。
- `--watchdog-timeout`：该模型较大，加载权重耗时较长，建议增大到 `1200` 或更高。
- `--model-loader-extra-config '{"enable_multithread_load": true}'`：启用并行权重加载，加快启动速度。

### 推理解析与工具调用

Qwen 3.5 可通过 Qwen3 parsers 支持 reasoning 和 tool calling：

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B-A17B \
    --tp 8 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

## 准确率评测

你可以使用 `lm-eval` 评测模型准确率：

```bash
pip install lm-eval[api]

lm_eval --model local-completions \
    --model_args '{"base_url": "http://localhost:8000/v1/completions", "model": "Qwen/Qwen3.5-397B-A17B", "num_concurrent": 256, "max_retries": 10, "max_gen_toks": 2048}' \
    --tasks gsm8k \
    --batch_size auto \
    --num_fewshot 5 \
    --trust_remote_code
```

## 其他资源

- [AMD Day 0 Support for Qwen 3.5 on AMD Instinct GPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html)
- [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
