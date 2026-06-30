# 采样参数

本文档介绍 SGLang Runtime 的采样参数。这是 runtime 的低层接口。
如果需要能够自动处理 chat template 的高层接口，请考虑使用 [OpenAI Compatible API](openai_api_completions.ipynb)。

## `/generate` 接口

`/generate` 接口接受 JSON 格式的下列参数。详细用法请参考 [native API 文档](native_api.ipynb)。该对象定义在 `io_struct.py::GenerateReqInput`。也可以阅读源代码了解更多参数和说明。

| 参数 | 类型/默认值 | 说明 |
|------|-------------|------|
| text | `Optional[Union[List[str], str]] = None` | 输入 prompt。可以是单个 prompt，也可以是一批 prompt。 |
| input_ids | `Optional[Union[List[List[int]], List[int]]] = None` | 文本对应的 token ID；可以指定 `text` 或 `input_ids`。 |
| input_embeds | `Optional[Union[List[List[List[float]]], List[List[float]]]] = None` | `input_ids` 对应的 embedding；可以指定 `text`、`input_ids` 或 `input_embeds`。 |
| image_data | `Optional[Union[List[List[ImageDataItem]], List[ImageDataItem], ImageDataItem]] = None` | 图像输入。支持三种格式：（1）**原始图像**：PIL Image、文件路径、URL 或 base64 字符串；（2）**Processor 输出**：包含 `format: "processor_output"` 的 Dict，其中包含 HuggingFace processor 输出；（3）**预计算 embedding**：包含 `format: "precomputed_embedding"` 和 `feature` 的 Dict，其中 `feature` 是预先计算好的视觉 embedding。可以是单张图像、图像列表，或图像列表的列表。详情见 [Multimodal Input Formats](#multimodal-input-formats)。 |
| audio_data | `Optional[Union[List[AudioDataItem], AudioDataItem]] = None` | 音频输入。可以是文件名、URL 或 base64 编码字符串。 |
| sampling_params | `Optional[Union[List[Dict], Dict]] = None` | 采样参数，见下文各节说明。 |
| rid | `Optional[Union[List[str], str]] = None` | 请求 ID。 |
| return_logprob | `Optional[Union[List[bool], bool]] = None` | 是否返回 token 的 log probability。 |
| logprob_start_len | `Optional[Union[List[int], int]] = None` | 当 `return_logprob` 启用时，指定从 prompt 中哪个位置开始返回 logprob。默认值为 `-1`，表示只返回输出 token 的 logprob。 |
| top_logprobs_num | `Optional[Union[List[int], int]] = None` | 当 `return_logprob` 启用时，每个位置返回的 top logprob 数量。 |
| token_ids_logprob | `Optional[Union[List[List[int]], List[int]]] = None` | 当 `return_logprob` 启用时，需要返回 logprob 的 token ID。 |
| return_text_in_logprobs | `bool = False` | 是否在返回的 logprob 中将 token detokenize 成文本。 |
| stream | `bool = False` | 是否流式输出。 |
| lora_path | `Optional[Union[List[Optional[str]], Optional[str]]] = None` | LoRA 路径。 |
| custom_logit_processor | `Optional[Union[List[Optional[str]], str]] = None` | 用于高级采样控制的自定义 logit processor。必须是通过 `CustomLogitProcessor` 的 `to_str()` 方法序列化后的实例。用法见下文。 |
| return_hidden_states | `Union[List[bool], bool] = False` | 是否返回 hidden states。 |
| return_routed_experts | `bool = False` | 是否返回 MoE 模型的 routed experts。需要服务端启用 `--enable-return-routed-experts`。返回值是 base64 编码的 int32 expert ID，按逻辑形状 `[num_tokens, num_layers, top_k]` 展平成数组。 |

## 采样参数

该对象定义在 `sampling_params.py::SamplingParams`。也可以阅读源代码了解更多参数和说明。

### 默认值说明

默认情况下，SGLang 会从模型的 `generation_config.json` 初始化若干采样参数（服务端以 `--sampling-defaults model` 启动时，这是默认行为）。如果希望使用 SGLang/OpenAI 的常量默认值，请用 `--sampling-defaults openai` 启动服务端。你始终可以在每个请求中通过 `sampling_params` 覆盖任意参数。

```bash
# Use model-provided defaults from generation_config.json (default behavior)
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults model

# Use SGLang/OpenAI constant defaults instead
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults openai
```

### 核心参数

| 参数 | 类型/默认值 | 说明 |
|------|-------------|------|
| max_new_tokens | `int = 128` | 以 token 数衡量的最大输出长度。 |
| stop | `Optional[Union[str, List[str]]] = None` | 一个或多个 [stop words](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop)。如果采样到其中任意 stop word，生成会停止。 |
| stop_token_ids | `Optional[List[int]] = None` | 以 token ID 形式提供 stop word。如果采样到其中任意 token ID，生成会停止。 |
| stop_regex | `Optional[Union[str, List[str]]] = None` | 当命中列表中的任意正则表达式模式时停止。 |
| temperature | `float (model default; fallback 1.0)` | 采样下一个 token 时使用的 [temperature](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature)。`temperature = 0` 对应 greedy sampling；更高的 temperature 会带来更高多样性。 |
| top_p | `float (model default; fallback 1.0)` | [Top-p](https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p) 会从排序后的最小 token 集合中采样，该集合的累计概率超过 `top_p`。当 `top_p = 1` 时，等价于从全部 token 中不受限制地采样。 |
| top_k | `int (model default; fallback -1)` | [Top-k](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) 会从概率最高的 `k` 个 token 中随机选择。 |
| min_p | `float (model default; fallback 0.0)` | [Min-p](https://github.com/huggingface/transformers/issues/27670) 会从概率大于 `min_p * highest_token_probability` 的 token 中采样。 |

### 惩罚项

| 参数 | 类型/默认值 | 说明 |
|------|-------------|------|
| frequency_penalty | `float = 0.0` | 根据 token 目前在生成结果中出现的频率进行惩罚。取值必须在 `-2` 到 `2` 之间；负值鼓励重复 token，正值鼓励采样新 token。惩罚缩放会随 token 出现次数线性增长。 |
| presence_penalty | `float = 0.0` | 如果 token 已经在生成结果中出现过，则对其进行惩罚。取值必须在 `-2` 到 `2` 之间；负值鼓励重复 token，正值鼓励采样新 token。只要 token 出现过，惩罚缩放就是常量。 |
| repetition_penalty | `float = 1.0` | 缩放此前生成过的 token 的 logits，以抑制（值大于 1）或鼓励（值小于 1）重复。有效范围为 `(0, 2]`；`1.0` 表示概率不变。 |
| min_new_tokens | `int = 0` | 强制模型至少生成 `min_new_tokens` 个 token，直到采样到 stop word 或 EOS token。注意这可能导致非预期行为，例如分布高度偏向这些 token 时。 |

### 约束解码

下列参数请参考专门的 [constrained decoding](../advanced_features/structured_outputs.ipynb) 指南。

| 参数 | 类型/默认值 | 说明 |
|------|-------------|------|
| json_schema | `Optional[str] = None` | 结构化输出使用的 JSON schema。 |
| regex | `Optional[str] = None` | 结构化输出使用的正则表达式。 |
| ebnf | `Optional[str] = None` | 结构化输出使用的 EBNF。 |
| structural_tag | `Optional[str] = None` | 结构化输出使用的 structural tag。 |

### 其他选项

| 参数 | 类型/默认值 | 说明 |
|------|-------------|------|
| n | `int = 1` | 指定每个请求生成的输出序列数量。（不建议在一个请求中生成多个输出，即 `n > 1`；多次重复相同 prompt 通常有更好的控制性和效率。） |
| ignore_eos | `bool = False` | 采样到 EOS token 时不停止生成。 |
| skip_special_tokens | `bool = True` | 解码时移除特殊 token。 |
| spaces_between_special_tokens | `bool = True` | detokenization 时是否在特殊 token 之间添加空格。 |
| no_stop_trim | `bool = False` | 不从生成文本中裁剪 stop word 或 EOS token。 |
| custom_params | `Optional[List[Optional[Dict[str, Any]]]] = None` | 使用 `CustomLogitProcessor` 时使用。用法见下文。 |

## 示例

### 普通请求

启动服务器：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

详细示例见 [send request](./send_request.ipynb)。

### 流式输出

发送请求并流式接收输出：

```python
import requests, json

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"].strip()
        print(output[prev:], end="", flush=True)
        prev = len(output)
print("")
```

详细示例见 [openai compatible api](openai_api_completions.ipynb)。

### 多模态

启动服务器：

```bash
python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov
```

下载图像：

```bash
curl -o example_image.png -L https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\nDescribe this image in a very short sentence.<|im_end|>\n"
                "<|im_start|>assistant\n",
        "image_data": "example_image.png",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

`image_data` 可以是文件名、URL 或 base64 编码字符串。另请参见 `python/sglang/srt/utils.py:load_image`。

流式输出的支持方式与[上文](#流式输出)类似。

详细示例见 [OpenAI API Vision](openai_api_vision.ipynb)。

### 结构化输出（JSON、Regex、EBNF）

可以指定 JSON schema、正则表达式或 [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) 来约束模型输出。模型输出会保证遵循给定约束。每个请求只能指定一个约束参数（`json_schema`、`regex` 或 `ebnf`）。

SGLang 支持两个 grammar 后端：

- [XGrammar](https://github.com/mlc-ai/xgrammar)（默认）：支持 JSON schema、正则表达式和 EBNF 约束。
  - XGrammar 当前使用 [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)。
- [Outlines](https://github.com/dottxt-ai/outlines)：支持 JSON schema 和正则表达式约束。

如果希望初始化 Outlines 后端，可以使用 `--grammar-backend outlines` 参数：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--port 30000 --host 0.0.0.0 --grammar-backend [xgrammar|outlines] # xgrammar or outlines (default: xgrammar)
```

```python
import json
import requests

json_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[\\w]+$"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
})

# JSON (works with both Outlines and XGrammar)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Here is the information of the capital of France in the JSON format.\n",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)
print(response.json())

# Regular expression (Outlines backend only)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Paris is the capital of",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print(response.json())

# EBNF (XGrammar backend only)
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a greeting.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ebnf": 'root ::= "Hello" | "Hi" | "Hey"',
        },
    },
)
print(response.json())
```

详细示例见 [structured outputs](../advanced_features/structured_outputs.ipynb)。

### 自定义 logit processor

启动服务器并开启 `--enable-custom-logit-processor` 参数：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 30000 \
  --enable-custom-logit-processor
```

定义一个自定义 logit processor，使其总是采样指定的 token id。

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class DeterministicLogitProcessor(CustomLogitProcessor):
    """A dummy logit processor that changes the logits to always
    sample the given token id.
    """

    def __call__(self, logits, custom_param_list):
        # Check that the number of logits matches the number of custom parameters
        assert logits.shape[0] == len(custom_param_list)
        key = "token_id"

        for i, param_dict in enumerate(custom_param_list):
            # Mask all other tokens
            logits[i, :] = -float("inf")
            # Assign highest probability to the specified token
            logits[i, param_dict[key]] = 0.0
        return logits
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
            "custom_params": {"token_id": 5},
        },
    },
)
print(response.json())
```

发送 OpenAI chat completion 请求：

```python
import openai
from sglang.utils import print_highlight

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0.0,
    max_tokens=32,
    extra_body={
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "custom_params": {"token_id": 5},
    },
)

print_highlight(f"Response: {response}")
```
