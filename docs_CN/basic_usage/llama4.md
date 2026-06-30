# Llama4 用法

[Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) 是 Meta 最新一代开源大语言模型，具备行业领先的性能。

SGLang 从 [v0.4.5](https://github.com/sgl-project/sglang/releases/tag/v0.4.5) 开始支持 Llama 4 Scout（109B）和 Llama 4 Maverick（400B）。

持续优化进展见 [Roadmap](https://github.com/sgl-project/sglang/issues/5118)。

## 使用 SGLang 启动 Llama 4

在 8 张 H100/H200 GPU 上服务 Llama 4 模型：

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 8 \
  --context-length 1000000
```

### 配置建议

- **缓解 OOM**：调整 `--context-length` 以避免 GPU out-of-memory。对于 Scout 模型，建议在 8\*H100 上最高设置到 1M，在 8\*H200 上最高设置到 2.5M。对于 Maverick 模型，在 8\*H200 上通常不需要设置 context length。启用 hybrid kv cache 后，Scout 模型的 `--context-length` 在 8\*H100 上最高可设置到 5M，在 8\*H200 上最高可设置到 10M。

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
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --trust-remote-code \
  --tp 8 \
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
python -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --port 30000 \
  --tp 8 \
  --mem-fraction-static 0.8 \
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Scout-17B-16E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0

# Llama-4-Maverick-17B-128E-Instruct
python -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --port 30000 \
  --tp 8 \
  --mem-fraction-static 0.8 \
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Maverick-17B-128E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0
```

更多细节见 [这个 PR](https://github.com/sgl-project/sglang/pull/5092)。
