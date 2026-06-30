from pathlib import Path


FILES = {
    "docs_CN/basic_usage/openai_api.rst": """OpenAI 兼容 API
================

.. toctree::
   :maxdepth: 1

   openai_api_completions.ipynb
   openai_api_vision.ipynb
   openai_api_embeddings.ipynb
""",
    "docs_CN/basic_usage/popular_model_usage.rst": """常见模型用法
==============

.. toctree::
   :maxdepth: 1

   deepseek_v3
   deepseek_v32
   gpt_oss
   qwen3
   qwen3_5
   qwen3_vl
   llama4
   glm45
   glmv
   deepseek_ocr
   minimax_m2
   hy3_preview
""",
    "docs_CN/basic_usage/ollama_api.md": """# Ollama 兼容 API

SGLang 提供 Ollama API 兼容能力，因此你可以把 SGLang 作为推理后端，同时继续使用 Ollama CLI 和 Ollama Python 库。

## 前置条件

```bash
# 安装 Ollama Python 库（用于 Python 客户端）
pip install ollama
```

> **注意**：不需要安装 Ollama server。SGLang 会作为后端运行，你只需要把 `ollama` CLI 或 Python 库作为客户端使用。

## 端点

| Endpoint | Method | 说明 |
|----------|--------|------|
| `/` | GET, HEAD | Ollama CLI 健康检查 |
| `/api/tags` | GET | 列出可用模型 |
| `/api/chat` | POST | Chat completions，支持流式和非流式 |
| `/api/generate` | POST | 文本生成，支持流式和非流式 |
| `/api/show` | POST | 模型信息 |

## 快速开始

### 1. 启动 SGLang Server

```bash
python -m sglang.launch_server \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --port 30001 \\
    --host 0.0.0.0
```

> **注意**：`ollama run` 使用的模型名必须和传给 `--model` 的值完全一致。

### 2. 使用 Ollama CLI

```bash
# 列出可用模型
OLLAMA_HOST=http://localhost:30001 ollama list

# 交互式聊天
OLLAMA_HOST=http://localhost:30001 ollama run "Qwen/Qwen2.5-1.5B-Instruct"
```

如果要连接防火墙后的远程服务器：

```bash
# SSH 隧道
ssh -L 30001:localhost:30001 user@gpu-server -N &

# 然后像上面一样使用 Ollama CLI
OLLAMA_HOST=http://localhost:30001 ollama list
```

### 3. 使用 Ollama Python 库

```python
import ollama

client = ollama.Client(host='http://localhost:30001')

# 非流式
response = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])

# 流式
stream = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## Smart Router

如果希望通过 LLM judge 在本地 Ollama（速度快）和远程 SGLang（能力强）之间做智能路由，请参考 [Smart Router 文档](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/ollama/README.md)。

## 总结

| 组件 | 用途 |
|------|------|
| **Ollama API** | 开发者熟悉的 CLI/API |
| **SGLang Backend** | 高性能推理引擎 |
| **Smart Router** | 智能路由：简单任务走快速本地模型，复杂任务走更强的远程模型 |
""",
    "docs_CN/basic_usage/qwen3.md": """# Qwen3-Next 用法

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
python3 -m sglang.launch_server \\
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
  --tp 4 \\
  --speculative-num-steps 3 \\
  --speculative-eagle-topk 1 \\
  --speculative-num-draft-tokens 4 \\
  --speculative-algo NEXTN
```

更多细节见 [这个 PR](https://github.com/sgl-project/sglang/pull/10233)。
""",
    "docs_CN/basic_usage/qwen3_5.md": """# Qwen 3.5 用法

Qwen 3.5 是阿里巴巴最新一代大语言模型，具备 hybrid attention 架构、带共享专家的高级 MoE，以及原生多模态能力。

主要架构特性：

- **Hybrid Attention**：将 Gated Delta Networks（线性复杂度 O(n)）与每 4 层一次的 full attention 结合，以获得较强的关联召回能力。
- **带共享专家的 MoE**：64 个 routed experts 中激活 top-8，同时包含一个专用于通用特征的 shared expert。
- **多模态**：使用带 Conv3d 的 DeepStack Vision Transformer，原生支持图像和视频理解。

## 使用 SGLang 启动 Qwen 3.5

### Dense 模型

在 8 张 GPU 上服务 `Qwen/Qwen3.5-397B-A17B`：

```bash
python3 -m sglang.launch_server \\
    --model-path Qwen/Qwen3.5-397B-A17B \\
    --tp 8 \\
    --trust-remote-code
```

### AMD GPU（MI300X / MI325X / MI35X）

在 AMD Instinct GPU 上，请使用 `triton` attention backend。full attention 层和 Gated Delta Net（linear attention）层在 ROCm 上都使用基于 Triton 的 kernel：

```bash
SGLANG_USE_AITER=1 python3 -m sglang.launch_server \\
    --model-path Qwen/Qwen3.5-397B-A17B \\
    --tp 8 \\
    --attention-backend triton \\
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
python3 -m sglang.launch_server \\
    --model-path Qwen/Qwen3.5-397B-A17B \\
    --tp 8 \\
    --trust-remote-code \\
    --reasoning-parser qwen3 \\
    --tool-call-parser qwen3_coder
```

## 准确率评测

你可以使用 `lm-eval` 评测模型准确率：

```bash
pip install lm-eval[api]

lm_eval --model local-completions \\
    --model_args '{"base_url": "http://localhost:8000/v1/completions", "model": "Qwen/Qwen3.5-397B-A17B", "num_concurrent": 256, "max_retries": 10, "max_gen_toks": 2048}' \\
    --tasks gsm8k \\
    --batch_size auto \\
    --num_fewshot 5 \\
    --trust_remote_code
```

## 其他资源

- [AMD Day 0 Support for Qwen 3.5 on AMD Instinct GPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html)
- [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
""",
    "docs_CN/basic_usage/minimax_m2.md": """# MiniMax M2.5/M2.1/M2 用法

[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)、[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) 和 [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) 是由 [MiniMax](https://www.minimax.io/) 创建的先进大语言模型。

MiniMax-M2 系列重新定义了 agent 场景下的效率。这些紧凑、快速且成本友好的 MoE 模型（总参数量 230B，激活参数量 10B）面向代码和 agentic 任务的高性能需求，同时保持强大的通用智能。MiniMax-M2 系列仅激活 10B 参数，却能提供接近当今领先模型的端到端工具使用能力，并以更精简的形态降低部署和扩展成本。

## 支持的模型

本指南适用于以下模型。部署时只需要替换模型名即可。下面示例使用 **MiniMax-M2**：

- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## 系统要求

以下是推荐配置，实际要求应根据使用场景调整：

- 4 张 96GB GPU：支持最高 400K tokens 的上下文长度。
- 8 张 144GB GPU：支持最高 3M tokens 的上下文长度。

## 使用 Python 部署

4-GPU 部署命令：

```bash
python -m sglang.launch_server \\
    --model-path MiniMaxAI/MiniMax-M2 \\
    --tp-size 4 \\
    --tool-call-parser minimax-m2 \\
    --reasoning-parser minimax-append-think \\
    --host 0.0.0.0 \\
    --trust-remote-code \\
    --port 8000 \\
    --mem-fraction-static 0.85
```

8-GPU 部署命令：

```bash
python -m sglang.launch_server \\
    --model-path MiniMaxAI/MiniMax-M2 \\
    --tp-size 8 \\
    --ep-size 8 \\
    --tool-call-parser minimax-m2 \\
    --reasoning-parser minimax-append-think \\
    --host 0.0.0.0 \\
    --trust-remote-code \\
    --port 8000 \\
    --mem-fraction-static 0.85
```

### AMD GPU（MI300X/MI325X/MI355X）

8-GPU 部署命令：

```bash
SGLANG_USE_AITER=1 python -m sglang.launch_server \\
    --model-path MiniMaxAI/MiniMax-M2.5 \\
    --tp-size 8 \\
    --ep-size 8 \\
    --attention-backend aiter \\
    --tool-call-parser minimax-m2 \\
    --reasoning-parser minimax-append-think \\
    --host 0.0.0.0 \\
    --trust-remote-code \\
    --port 8000 \\
    --mem-fraction-static 0.85
```

## 测试部署

启动后，可以用下面的命令测试 SGLang 的 OpenAI 兼容 API：

```bash
curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "MiniMaxAI/MiniMax-M2",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```
""",
}


for path, content in FILES.items():
    Path(path).write_text(content, encoding="utf-8")
