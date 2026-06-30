<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# DeepSeek V3/V3.1/R1 用法

SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended) from Day 0.

本文档 outlines current optimizations for DeepSeek.
For an overview of the implemented features see the completed [Roadmap](https://github.com/sgl-project/sglang/issues/2591).

## Launch DeepSeek V3.1/V3/R1 with SGLang

To run DeepSeek V3.1/V3/R1 models, the recommended settings are as follows:

| Weight Type | 配置 |
|------------|-------------------|
| **Full precision [FP8](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)**<br>*(recommended)* | 8 x H200 |
| | 8 x B200 |
| | 8 x MI300X |
| | 2 x 8 x H100/800/20 |
| | Xeon 6980P CPU |
| **Full precision ([BF16](https://huggingface.co/unsloth/DeepSeek-R1-0528-BF16))** (upcast from original FP8) | 2 x 8 x H200 |
| | 2 x 8 x MI300X |
| | 4 x 8 x H100/800/20 |
| | 4 x 8 x A100/A800 |
| **Quantized weights ([INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8))** | 16 x A100/800 |
| | 32 x L40S |
| | Xeon 6980P CPU |
| | 4 x Atlas 800I A3 |
| **Quantized weights ([W4A8](https://huggingface.co/novita/Deepseek-R1-0528-W4AFP8))** | 8 x H20/100, 4 x H200 |
| **Quantized weights ([AWQ](https://huggingface.co/QuixiAI/DeepSeek-R1-0528-AWQ))** | 8 x H100/800/20 |
| | 8 x A100/A800 |
| **Quantized weights ([MXFP4](https://huggingface.co/amd/DeepSeek-R1-MXFP4-Preview))** | 8, 4 x MI355X/350X |
| **Quantized weights ([NVFP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2))** | 8, 4 x B200 |

<style>
.md-typeset__table {
  width: 100%;
}

.md-typeset__table table {
  border-collapse: collapse;
  margin: 1em 0;
  border: 2px solid var(--md-typeset-table-color);
  table-layout: fixed;
}

.md-typeset__table th {
  border: 1px solid var(--md-typeset-table-color);
  border-bottom: 2px solid var(--md-typeset-table-color);
  background-color: var(--md-default-bg-color--lighter);
  padding: 12px;
}

.md-typeset__table td {
  border: 1px solid var(--md-typeset-table-color);
  padding: 12px;
}

.md-typeset__table tr:nth-child(2n) {
  background-color: var(--md-default-bg-color--lightest);
}
</style>

```{important}
The official DeepSeek V3 is already in FP8 format, so you should not run it with any quantization arguments like `--quantization fp8`.
```

Detailed commands for reference:

- [8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)
- [4 x B200, 8 x B200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-one-b200-node)
- [8 x MI300X](../platforms/amd_gpu.md#running-deepseek-v3)
- [2 x 8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker)
- [4 x 8 x A100](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)
- [8 x A100 (AWQ)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-8-a100a800-with-awq-quantization)
- [16 x A100 (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-16-a100a800-with-int8-quantization)
- [32 x L40S (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-32-l40s-with-int8-quantization)
- [Xeon 6980P CPU](../platforms/cpu_server.md#example-running-deepseek-r1)
- [4 x Atlas 800I A3 (int8)](../platforms/ascend/ascend_npu_deepseek_example.md#running-deepseek-with-pd-disaggregation-on-4-x-atlas-800i-a3)

### Download Weights
If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded. 请参考 [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only) official guide to download the weights.

### Launch with one node of 8 x H200
请参考 [the example](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#installation--launch).

### Running examples on Multi-Node

- [Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP](https://lmsys.org/blog/2025-06-16-gb200-part-1/) ([Part I](https://lmsys.org/blog/2025-06-16-gb200-part-1/), [Part II](https://lmsys.org/blog/2025-09-25-gb200-part-2/)) - Comprehensive guide on GB200 optimizations.

- [Deploying DeepSeek with PD 分离 and Large-Scale 专家并行 on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - Guide on PD disaggregation and large-scale EP.

- [服务化 with two H20*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes).

- [Best Practices for 服务化 DeepSeek-R1 on H20](https://lmsys.org/blog/2025-09-26-sglang-ant-group/) - Comprehensive guide on H20 optimizations, deployment and performance.

- [服务化 with two H200*8 nodes and docker](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker).

- [服务化 with four A100*8 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes).

## Optimizations

### Multi-head Latent Attention (MLA) Throughput Optimizations

**说明**: [MLA](https://arxiv.org/pdf/2405.04434) is an innovative attention mechanism introduced by the DeepSeek team, aimed at improving inference efficiency. SGLang has implemented specific optimizations for this, including:

- **Weight Absorption**: By applying the associative law of matrix multiplication to reorder computation steps, this method balances computation and memory access and improves efficiency in the decoding phase.

- **MLA Attention Backends**: 目前 SGLang 支持 different optimized MLA attention backends, including [FlashAttention3](https://github.com/Dao-AILab/flash-attention), [Flashinfer](https://docs.flashinfer.ai/api/attention.html#flashinfer-mla), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [CutlassMLA](https://github.com/sgl-project/sglang/pull/5390), **TRTLLM MLA** (optimized for Blackwell architecture), and [Triton](https://github.com/triton-lang/triton) backends. The default FA3 provides good performance across wide workloads.

- **FP8 量化**: W8A8 FP8 and KV 缓存 FP8 quantization enables efficient FP8 inference. Additionally, we have implemented Batched Matrix Multiplication (BMM) operator to facilitate FP8 inference in MLA with weight absorption.

- **CUDA Graph & Torch.compile**: Both MLA and Mixture of Experts (MoE) are compatible with CUDA Graph and Torch.compile, which reduces latency and accelerates decoding speed for small batch sizes.

- **Chunked 前缀缓存**: Chunked prefix cache optimization can increase throughput by cutting prefix cache into chunks, processing them with multi-head attention and merging their states. Its improvement can be significant when doing chunked prefill on long sequences. 目前 this optimization is only available for FlashAttention3 backend.

Overall, with these optimizations, we have achieved up to **7x** acceleration in output throughput compared to the previous version.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_3/deepseek_mla.svg" alt="Multi-head Latent Attention for DeepSeek Series 模型">
</p>

**用法**: MLA optimization is enabled by default.

**参考**: Check [Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) and [Slides](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf) for more details.

### 数据并行 Attention

**说明**: This optimization involves data parallelism (DP) for the MLA attention mechanism of DeepSeek Series 模型, which allows for a significant reduction in the KV cache size, enabling larger batch sizes. Each DP worker independently handles different types of batches (prefill, decode, idle), which are then synchronized before and after processing through the Mixture-of-Experts (MoE) layer. If you do not use DP attention, KV cache will be duplicated among all TP ranks.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg" alt="数据并行 Attention for DeepSeek Series 模型">
</p>

With data parallelism attention enabled, we have achieved up to **1.9x** decoding throughput improvement compared to the previous version.

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/deepseek_coder_v2.svg" alt="数据并行 Attention 性能 Comparison">
</p>

**用法**:
- Append `--enable-dp-attention --tp 8 --dp 8` to the server arguments when using 8 H200 GPUs. This optimization improves peak throughput in high batch size scenarios where the server is limited by KV cache capacity.
- DP and TP attention can be flexibly combined. 例如, to deploy DeepSeek-V3/R1 on 2 nodes with 8 H100 GPUs each, you can specify `--enable-dp-attention --tp 16 --dp 2`. This configuration runs attention with 2 DP groups, each containing 8 TP GPUs.

```{caution}
Data parallelism attention is not recommended for low-latency, small-batch use cases. It is optimized for high-throughput scenarios with large batch sizes.
```

**参考**: Check [Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models).

### Multi-Node Tensor Parallelism

**说明**: For users with limited memory on a single node, SGLang 支持 serving DeepSeek Series 模型, including DeepSeek V3, across multiple nodes using tensor parallelism. This approach partitions the model parameters across multiple GPUs or nodes to handle models that are too large for one node's memory.

**用法**: Check [here](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker) for usage examples.

### Block-wise FP8

**说明**: SGLang implements block-wise FP8 quantization with two key optimizations:

- **Activation**: E4M3 format using per-token-per-128-channel sub-vector scales with online casting.

- **Weight**: Per-128x128-block quantization for better numerical stability.

- **DeepGEMM**: The [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernel library optimized for FP8 matrix multiplications.

**用法**: The activation and weight optimization above are turned on by default for DeepSeek V3 models. DeepGEMM is enabled by default on NVIDIA Hopper/Blackwell GPUs and disabled by default on other devices. DeepGEMM can also be manually turned off by setting the environment variable `SGLANG_ENABLE_JIT_DEEPGEMM=0`.

```{tip}
Before serving the DeepSeek model, precompile the DeepGEMM kernels to improve first-run performance. The precompilation process typically takes around 10 minutes to complete.
```

```bash
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

### Multi-token Prediction
**说明**: SGLang implements DeepSeek V3 Multi-Token Prediction (MTP) based on [EAGLE speculative decoding](/docs_CN/advanced_features/speculative_decoding.md#EAGLE-Decoding). With this optimization, the decoding speed can be improved by **1.8x** for batch size 1 and **1.5x** for batch size 32 respectively on H200 TP8 setting.

**用法**:
Add `--speculative-algorithm EAGLE`. Other flags, like `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` are optional. 例如:
```
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --speculative-algorithm EAGLE \
  --trust-remote-code \
  --tp 8
```
- The default configuration for DeepSeek models is `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. The best configuration for `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` can be searched with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) script for given batch size. The minimum configuration is `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`, which can achieve speedup for larger batch sizes.
- Most MLA attention backends fully support MTP usage. See [MLA Backends](../advanced_features/attention_backend.md#mla-backends) for details.

```{note}
To enable DeepSeek MTP for large batch sizes (>48), you need to adjust some parameters (Reference [this discussion](https://github.com/sgl-project/sglang/issues/4543#issuecomment-2737413756)):
- Adjust `--max-running-requests` to a larger number. The default value is `48` for MTP. For larger batch sizes, you should increase this value beyond the default value.
- Set `--cuda-graph-bs`. It's a list of batch sizes for cuda graph capture. The [default captured batch sizes for speculative decoding](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L888-L895) is 48. You can customize this by including more batch sizes.
```

```{tip}
To enable the experimental overlap scheduler for EAGLE speculative decoding, set the environment variable `SGLANG_ENABLE_SPEC_V2=1`. This can improve performance by enabling overlap scheduling between draft and verification stages.
```


### Reasoning Content for DeepSeek R1 & V3.1

See [Reasoning Parser](/docs_CN/advanced_features/separate_reasoning.md) and [Thinking Parameter for DeepSeek V3.1](/docs_CN/basic_usage/openai_api_completions.md#示例:-DeepSeek-V3-模型).


### Function calling for DeepSeek 模型

Add arguments `--tool-call-parser deepseekv3` and `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja`(recommended) to enable this feature. 例如 (running on 1 * H20 node):

```
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3-0324 \
  --tp 8 \
  --port 30000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv3 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja
```

Sample 请求:

```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324", "tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of a city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "How'\''s the weather like in Qingdao today"}]}'
```

Expected 响应

```
{"id":"6501ef8e2d874006bf555bc80cddc7c5","object":"chat.completion","created":1745993638,"model":"deepseek-ai/DeepSeek-V3-0324","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":[{"id":"0","index":null,"type":"function","function":{"name":"query_weather","arguments":"{\"city\": \"Qingdao\"}"}}]},"logprobs":null,"finish_reason":"tool_calls","matched_stop":null}],"usage":{"prompt_tokens":116,"total_tokens":138,"completion_tokens":22,"prompt_tokens_details":null}}

```
Sample Streaming 请求:
```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324","stream":true,"tools": [{"type": "function", "function": {"name": "query_weather", "description": "Get weather of a city, the user should supply a city first", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Beijing"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "How'\''s the weather like in Qingdao today"}]}'
```
Expected Streamed Chunks (simplified for clarity):
```
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"city"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\":\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"Q"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"ing"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"dao"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\"}"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":null}}], "finish_reason": "tool_calls"}
data: [DONE]
```
The client needs to concatenate all arguments fragments to reconstruct the complete tool call:
```
{"city": "Qingdao"}
```

```{important}
1. Use a lower `"temperature"` value for better results.
2. To receive more consistent tool call results, it is recommended to use `--chat-template examples/chat_template/tool_chat_template_deepseekv3.jinja`. It provides an improved unified prompt.
```


### Thinking Budget for DeepSeek R1

In SGLang, we can implement thinking budget with `CustomLogitProcessor`.

Launch a server with `--enable-custom-logit-processor` flag on.

```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --port 30000 --host 0.0.0.0 --mem-fraction-static 0.9 --disable-cuda-graph --reasoning-parser deepseek-r1 --enable-custom-logit-processor
```

Sample 请求:

```python
import openai
from rich.pretty import pprint
from sglang.srt.sampling.custom_logit_processor import DeepSeekR1ThinkingBudgetLogitProcessor


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {
            "role": "user",
            "content": "Question: Is Paris the Capital of France?",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": DeepSeekR1ThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```

## 常见问题

**Q: 模型 loading is taking too long, and I'm encountering an NCCL timeout. What should I do?**

A: If you're experiencing extended model loading times and an NCCL timeout, you can try increasing the timeout duration. Add the argument `--dist-timeout 3600` when launching your model. This will set the timeout to one hour, which often resolves the issue.
