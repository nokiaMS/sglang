# Ollama 兼容 API

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
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
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
