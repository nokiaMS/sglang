# MiniMax M2.5/M2.1/M2 用法

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
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

8-GPU 部署命令：

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

### AMD GPU（MI300X/MI325X/MI355X）

8-GPU 部署命令：

```bash
SGLANG_USE_AITER=1 python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp-size 8 \
    --ep-size 8 \
    --attention-backend aiter \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

## 测试部署

启动后，可以用下面的命令测试 SGLang 的 OpenAI 兼容 API：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M2",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```
