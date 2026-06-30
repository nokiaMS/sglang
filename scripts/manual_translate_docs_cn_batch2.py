from pathlib import Path


FILES = {
    "docs_CN/basic_usage/glm45.md": """## 使用 SGLang 启动 GLM-4.5 / GLM-4.6 / GLM-4.7

在 8 张 H100/H200 GPU 上服务 GLM-4.5 / GLM-4.6 FP8 模型：

```bash
python3 -m sglang.launch_server --model zai-org/GLM-4.6-FP8 --tp 8
```

### EAGLE 投机解码

**说明**：SGLang 已支持在 GLM-4.5 / GLM-4.6 模型上使用 [EAGLE speculative decoding](/docs_CN/advanced_features/speculative_decoding.md#EAGLE-Decoding)。

**用法**：
添加 `--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 参数即可启用。例如：

``` bash
python3 -m sglang.launch_server \\
  --model-path zai-org/GLM-4.6-FP8 \\
  --tp-size 8 \\
  --tool-call-parser glm45  \\
  --reasoning-parser glm45  \\
  --speculative-algorithm EAGLE \\
  --speculative-num-steps 3  \\
  --speculative-eagle-topk 1  \\
  --speculative-num-draft-tokens 4 \\
  --mem-fraction-static 0.9 \\
  --served-model-name glm-4.6-fp8 \\
  --enable-custom-logit-processor
```

```{tip}
如果要为 EAGLE 投机解码启用实验性的 overlap scheduler，请设置环境变量 `SGLANG_ENABLE_SPEC_V2=1`。它可以通过让 draft 和 verification 阶段重叠调度来提升性能。
```

### GLM-4.5 / GLM-4.6 的 Thinking Budget

**注意**：对于 GLM-4.7，`--tool-call-parser` 应设置为 `glm47`；对于 GLM-4.5 和 GLM-4.6，应设置为 `glm45`。

在 SGLang 中，可以用 `CustomLogitProcessor` 实现 thinking budget。

启动服务器时需要打开 `--enable-custom-logit-processor`。

示例请求：

```python
import openai
from rich.pretty import pprint
from sglang.srt.sampling.custom_logit_processor import Glm4MoeThinkingBudgetLogitProcessor


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="zai-org/GLM-4.6",
    messages=[
        {
            "role": "user",
            "content": "Question: Is Paris the Capital of France?",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": Glm4MoeThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```
""",
    "docs_CN/basic_usage/deepseek_ocr.md": """# DeepSeek OCR（OCR-1 / OCR-2）

DeepSeek OCR 模型是用于 OCR 和文档理解的多模态（图像 + 文本）模型。

## 启动服务器

```shell
python -m sglang.launch_server \\
  --model-path deepseek-ai/DeepSeek-OCR-2 \\
  --trust-remote-code \\
  --host 0.0.0.0 \\
  --port 30000
```

> 你也可以将 `deepseek-ai/DeepSeek-OCR-2` 替换为 `deepseek-ai/DeepSeek-OCR`。

## Prompt 示例

模型卡中推荐的 prompt：

```
<image>
<|grounding|>Convert the document to markdown.
```

```
<image>
Free OCR.
```

## OpenAI 兼容请求示例

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\\n<|grounding|>Convert the document to markdown."},
                {"type": "image_url", "image_url": {"url": "https://example.com/your_image.jpg"}},
            ],
        }
    ],
    "max_tokens": 512,
}

response = requests.post(url, json=data)
print(response.text)
```
""",
    "docs_CN/basic_usage/llama4.md": """# Llama4 用法

[Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) 是 Meta 最新一代开源大语言模型，具备行业领先的性能。

SGLang 从 [v0.4.5](https://github.com/sgl-project/sglang/releases/tag/v0.4.5) 开始支持 Llama 4 Scout（109B）和 Llama 4 Maverick（400B）。

持续优化进展见 [Roadmap](https://github.com/sgl-project/sglang/issues/5118)。

## 使用 SGLang 启动 Llama 4

在 8 张 H100/H200 GPU 上服务 Llama 4 模型：

```bash
python3 -m sglang.launch_server \\
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \\
  --tp 8 \\
  --context-length 1000000
```

### 配置建议

- **缓解 OOM**：调整 `--context-length` 以避免 GPU out-of-memory。对于 Scout 模型，建议在 8\\*H100 上最高设置到 1M，在 8\\*H200 上最高设置到 2.5M。对于 Maverick 模型，在 8\\*H200 上通常不需要设置 context length。启用 hybrid kv cache 后，Scout 模型的 `--context-length` 在 8\\*H100 上最高可设置到 5M，在 8\\*H200 上最高可设置到 10M。

- **Attention Backend 自动选择**：SGLang 会根据硬件为 Llama 4 自动选择最优 attention backend。通常不需要手动指定 `--attention-backend`：
  - **Blackwell GPU（B200/GB200）**：`trtllm_mha`
  - **Hopper GPU（H100/H200）**：`fa3`
  - **AMD GPU**：`aiter`
  - **Intel XPU**：`intel_xpu`
  - **其他平台**：`triton`（fallback）

  如果要覆盖自动选择，请显式指定 `--attention-backend`，可选后端包括 `fa3`、`aiter`、`triton`、`trtllm_mha` 或 `intel_xpu`。

- **Chat Template**：对 chat completion 任务添加 `--chat-template llama-4`。
- **启用多模态**：添加 `--enable-multimodal`。
- **启用 Hybrid-KVCache**：设置 `--swa-full-tokens-ratio` 来调整 SWA 层（对 Llama4 来说是 local attention 层）KV tokens 与 full 层 KV tokens 的比例。（默认值：0.8，范围：0-1）

### EAGLE 投机解码

**说明**：SGLang 已支持在 Llama 4 Maverick（400B）上使用 [EAGLE speculative decoding](/docs_CN/advanced_features/speculative_decoding.md#EAGLE-Decoding)。

**用法**：
添加 `--speculative-draft-model-path`、`--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 参数即可启用。例如：

```
python3 -m sglang.launch_server \\
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \\
  --speculative-algorithm EAGLE3 \\
  --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 \\
  --speculative-num-steps 3 \\
  --speculative-eagle-topk 1 \\
  --speculative-num-draft-tokens 4 \\
  --trust-remote-code \\
  --tp 8 \\
  --context-length 1000000
```

- **注意**：Llama 4 draft 模型 *nvidia/Llama-4-Maverick-17B-128E-Eagle3* 只能识别 chat mode 中的对话。

## Benchmark 结果

### 使用 `lm_eval` 做准确率测试

SGLang 上 Llama4 Scout 和 Llama4 Maverick 的准确率可以匹配[官方 benchmark 数字](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。

8 张 H100 上 MMLU Pro 数据集的 benchmark 结果：

|                    | Llama-4-Scout-17B-16E-Instruct | Llama-4-Maverick-17B-128E-Instruct  |
|--------------------|--------------------------------|-------------------------------------|
| Official Benchmark | 74.3                           | 80.5                                |
| SGLang             | 75.2                           | 80.7                                |

命令：

```bash
# Llama-4-Scout-17B-16E-Instruct model
python -m sglang.launch_server \\
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \\
  --port 30000 \\
  --tp 8 \\
  --mem-fraction-static 0.8 \\
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Scout-17B-16E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0

# Llama-4-Maverick-17B-128E-Instruct
python -m sglang.launch_server \\
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \\
  --port 30000 \\
  --tp 8 \\
  --mem-fraction-static 0.8 \\
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Maverick-17B-128E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0
```

更多细节见 [这个 PR](https://github.com/sgl-project/sglang/pull/5092)。
""",
}

for path, content in FILES.items():
    Path(path).write_text(content, encoding="utf-8")
