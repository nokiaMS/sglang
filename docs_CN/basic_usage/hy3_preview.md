# Hy3-preview 用法

Hy3-preview 是腾讯混元团队发布的大规模语言模型，参数规模为 295B，其中激活参数为 21B。SGLang 支持部署 Hy3-preview。本指南介绍如何使用原生 BF16 运行 Hy3-preview。

## 安装

### Docker

```bash
docker pull lmsysorg/sglang:hy3-preview
```

### 从源码构建

```bash
# Install SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install "transformers>=5.6.0"
pip3 install -e "python"
```

## 使用 SGLang 启动 Hy3-preview

下面的命令用于在 8 张 GPU 上部署 [Hy3-preview](https://huggingface.co/tencent/Hy3-preview) 模型。在 8x96GB H20 上，SGLang 勉强可以部署 BF16 模型，但只能运行较小 batch size 或较短请求。条件允许时，建议使用 H20-3e 等更大显存的 GPU。

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --served-model-name hy3-preview
```

### EAGLE 推测解码

**说明**：SGLang 支持对 Hy3-preview 模型使用 [EAGLE 推测解码](../advanced_features/speculative_decoding.md#eagle-decoding)。

**用法**：
添加 `--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 来启用该功能。例如：

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --speculative-algorithm EAGLE \
  --served-model-name hy3-preview
```

## OpenAI Client 示例

首先安装 OpenAI Python client：

```bash
uv pip install -U openai
```

可以按如下方式使用 OpenAI client 验证 thinking mode 响应。

```python
from openai import OpenAI

# If running SGLang locally with its default OpenAI-compatible port:
#   http://localhost:30000/v1
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello."},
]

# Thinking mode is disabled by default (no need to pass chat_template_kwargs).
resp = client.chat.completions.create(
    model="hy3-preview",
    messages=messages,
    temperature=1,
    max_tokens=4096,
)
print(resp.choices[0].message.content)

# Thinking mode is enabled only if 'reasoning_effort' and 'interleaved_thinking' are set in 'chat_template_kwargs'.
# 'reasoning_effort' supports: 'high', 'low', 'no_think'.
resp_think = client.chat.completions.create(
    model="hy3-preview",
    messages=messages,
    temperature=1,
    max_tokens=4096,
    extra_body={
      "chat_template_kwargs": {
          "reasoning_effort": "high",
          "interleaved_thinking": True
      },
    },
)
output_msg = resp_think.choices[0].message
# thinking content
print(output_msg.reasoning_content)
# response content
print(output_msg.content)
```

### cURL 用法

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hy3-preview",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello."}
    ],
    "temperature": 1,
    "max_tokens": 4096
  }'
```

## Benchmark 结果

进行 benchmark 时，请在服务器启动命令中添加 `--disable-radix-cache` 来禁用 prefix caching。

下面的示例在 8 张 H20 GPU 上运行 benchmark，每张 GPU 的显存为 96 GB。

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --flush-cache \
    --dataset-name random \
    --random-range-ratio 1.0 \
    --random-input-len 4096 \
    --random-output-len 4096 \
    --num-prompts 5 \
    --max-concurrency 1 \
    --output-file hy3_preview_h20.jsonl \
    --model tencent/Hy3-preview \
    --served-model-name hy3-preview
```

如果运行成功，会看到如下输出。

```shell
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     5
Benchmark duration (s):                  176.41
Total input tokens:                      20480
Total input text tokens:                 20480
Total generated tokens:                  20480
Total generated tokens (retokenized):    20480
Request throughput (req/s):              0.03
Input token throughput (tok/s):          116.09
Output token throughput (tok/s):         116.09
Peak output token throughput (tok/s):    118.00
Peak concurrent requests:                2
Total token throughput (tok/s):          232.19
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   35279.06
Median E2E Latency (ms):                 35275.60
P90 E2E Latency (ms):                    35294.13
P99 E2E Latency (ms):                    35294.41
---------------Time to First Token----------------
Mean TTFT (ms):                          355.93
Median TTFT (ms):                        309.28
P99 TTFT (ms):                           518.36
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.53
Median TPOT (ms):                        8.54
P99 TPOT (ms):                           8.54
---------------Inter-Token Latency----------------
Mean ITL (ms):                           8.53
Median ITL (ms):                         8.54
P95 ITL (ms):                            8.62
P99 ITL (ms):                            8.74
Max ITL (ms):                            31.70
==================================================
```
