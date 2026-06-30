# GLM-4.6V / GLM-4.5V 用法

## SGLang 启动命令

下面是针对不同硬件和精度模式的推荐启动命令。

### FP8（量化）模式

适用于支持 FP8 checkpoint、追求更高显存效率和更低延迟的部署场景，例如 H100、H200：

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V-FP8 \
  --tp 2 \
  --ep 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --keep-mm-feature-on-device
```

### 非 FP8（BF16 / 全精度）模式

适用于在 A100/H100 上使用 BF16，或不使用 FP8 snapshot 的部署场景：

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --tp 4 \
  --ep 4 \
  --host 0.0.0.0 \
  --port 30000
```

## 硬件相关说明和建议

- 在 H100 上使用 FP8：建议使用 FP8 checkpoint，以获得最佳显存效率。
- 在 A100 / H100 上使用 BF16（非 FP8）：建议使用 `--mm-max-concurrent-calls` 控制图像/视频推理期间的并行吞吐和 GPU 显存占用。
- 在 H200 和 B200 上：模型可以直接运行，支持完整上下文长度，并支持图像和视频的并发处理。

## 发送图像/视频请求

### 图像输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

### 视频输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

## 重要服务器参数和开关

启动支持多模态的模型服务器时，可以使用下面的命令行参数微调性能和行为：

- `--mm-attention-backend`：指定多模态 attention 后端，例如 `fa3`（Flash Attention 3）。
- `--mm-max-concurrent-calls <value>`：指定服务器允许的**最大并发异步多模态数据处理调用数**。可用于控制图像/视频推理期间的并行吞吐和 GPU 显存占用。
- `--mm-per-request-timeout <seconds>`：定义每个多模态请求的**超时时间（秒）**。如果请求超过该时间限制，例如非常大的视频输入，请求会被自动终止。
- `--keep-mm-feature-on-device`：要求服务器在处理后将**多模态特征张量保留在 GPU 上**。这可以避免 device-to-host（D2H）内存拷贝，并提升重复或高频推理负载的性能。
- `--mm-enable-dp-encoder`：将 ViT 放在数据并行中，同时让 LLM 保持张量并行，通常可以降低 TTFT 并提升端到端吞吐。
- `SGLANG_USE_CUDA_IPC_TRANSPORT=1`：为多模态数据传输启用基于共享内存池的 CUDA IPC，可显著改善端到端延迟。

### 使用上述优化的示例：

```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_VLM_CACHE_SIZE_MB=0 \
python -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp-size 8 \
  --enable-cache-report \
  --log-level info \
  --max-running-requests 64 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 8192 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --mm-enable-dp-encoder \
  --enable-metrics
```

### GLM-4.5V / GLM-4.6V 的 Thinking Budget

在 SGLang 中，可以通过 `CustomLogitProcessor` 实现 thinking budget。

启动服务器时添加 `--enable-custom-logit-processor`。随后在请求中使用 `Glm4MoeThinkingBudgetLogitProcessor`，用法类似 [glm45.md](./glm45.md) 中的 `GLM-4.6` 示例。
