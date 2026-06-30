from pathlib import Path


src = Path("docs/advanced_features/server_arguments.md")
dst = Path("docs_CN/advanced_features/server_arguments.md")
text = src.read_text(encoding="utf-8")

start = """# 服务器参数

本页面列出了命令行中用于配置语言模型服务器行为和性能的 server arguments。
这些参数可用于自定义服务器的关键方面，包括模型选择、并行策略、内存管理和优化技术。
你可以通过 `python3 -m sglang.launch_server --help` 查看完整参数列表。

## 常用启动命令

- 如果要使用配置文件，请创建一个包含 server arguments 的 YAML 文件，并通过 `--config` 指定。CLI 参数会覆盖配置文件中的值。

  ```bash
  # Create config.yaml
  cat > config.yaml << EOF
  model-path: meta-llama/Meta-Llama-3-8B-Instruct
  host: 0.0.0.0
  port: 30000
  tensor-parallel-size: 2
  enable-metrics: true
  log-requests: true
  EOF

  # Launch server with config file
  python -m sglang.launch_server --config config.yaml
  ```

- 如果要启用多 GPU tensor parallelism，请添加 `--tp 2`。如果报错 “peer access is not supported between these two devices”，请在服务器启动命令中添加 `--enable-p2p-check`。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
  ```

- 如果要启用多 GPU data parallelism，请添加 `--dp 2`。当显存充足时，data parallelism 更适合提升吞吐；它也可以和 tensor parallelism 一起使用。下面的命令总共使用 4 张 GPU。对于 data parallelism，推荐使用 [SGLang Model Gateway（原 Router）](../advanced_features/sgl_model_gateway.md)。

  ```bash
  python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
  ```

- 如果服务过程中出现 out-of-memory 错误，可以尝试通过减小 `--mem-fraction-static` 来降低 KV cache pool 的内存占用。默认值为 `0.9`。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
  ```

- 如需调优超参数以获得更好性能，请参考 [hyperparameter tuning](hyperparameter_tuning.md)。
- 对于 Docker 和 Kubernetes 运行方式，需要设置共享内存；共享内存用于进程间通信。Docker 请查看 `--shm-size`，Kubernetes manifest 请调整 `/dev/shm` 大小。
- 如果长 prompt 的 prefill 阶段出现 out-of-memory 错误，可以尝试设置更小的 chunked prefill size。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
  ```

- 如果要启用 fp8 权重量化，可以在 fp16 checkpoint 上添加 `--quantization fp8`，或直接加载 fp8 checkpoint 而不指定额外参数。
- 如果要启用 fp8 KV cache 量化，请添加 `--kv-cache-dtype fp8_e4m3` 或 `--kv-cache-dtype fp8_e5m2`。
- 如果要启用确定性推理和 batch invariant operations，请添加 `--enable-deterministic-inference`。更多细节见 [deterministic inference 文档](../advanced_features/deterministic_inference.md)。
- 如果模型的 Hugging Face tokenizer 没有 chat template，可以指定[自定义 chat template](../references/custom_chat_template.md)。如果 tokenizer 有多个命名模板（例如 `default`、`tool_use`），可以用 `--hf-chat-template-name tool_use` 选择其中一个。
- 如果要在多节点上运行 tensor parallelism，请添加 `--nnodes 2`。假设有两个节点、每个节点两张 GPU，并希望运行 TP=4；令第一台节点主机名为 `sgl-dev-0`，`50000` 是可用端口，可以使用下面的命令。如果遇到 deadlock，请尝试添加 `--disable-cuda-graph`。
- （注意：该功能已不再维护，可能出错）如果要启用 `torch.compile` 加速，请添加 `--enable-torch-compile`。它可以加速小模型和小 batch size。默认缓存路径位于 `/tmp/torchinductor_root`，可以通过环境变量 `TORCHINDUCTOR_CACHE_DIR` 自定义。更多信息请参考 [PyTorch 官方文档](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) 和 [为 torch.compile 启用缓存](/docs_CN/references/torch_compile_cache.md)。
"""

marker = "  ```bash\n  # Node 0"
tail = text[text.index(marker) :]
text = start + tail

heading_map = {
    "## Model and tokenizer": "## 模型与 tokenizer",
    "## HTTP server": "## HTTP 服务器",
    "## Quantization and data type": "## 量化与数据类型",
    "## Memory and scheduling": "## 内存与调度",
    "## Runtime options": "## 运行时选项",
    "## Logging": "## 日志",
    "## RequestMetricsExporter configuration": "## RequestMetricsExporter 配置",
    "## API related": "## API 相关",
    "## Data parallelism": "## 数据并行",
    "## Multi-node distributed serving": "## 多节点分布式服务",
    "## Model override args": "## 模型覆盖参数",
    "## LoRA": "## LoRA",
    "## Kernel Backends (Attention, Sampling, Grammar, GEMM)": "## Kernel 后端（Attention、Sampling、Grammar、GEMM）",
    "## Speculative decoding": "## 投机解码",
    "## Ngram speculative decoding": "## Ngram 投机解码",
    "## Multi-layer Eagle speculative decoding": "## 多层 Eagle 投机解码",
    "## MoE": "## MoE",
    "## Mamba Cache": "## Mamba Cache",
    "## Hierarchical cache": "## 分层缓存",
    "## Hierarchical sparse attention": "## 分层稀疏注意力",
    "## LMCache": "## LMCache",
    "## Ktransformers": "## Ktransformers",
    "## Diffusion LLM": "## Diffusion LLM",
    "## Offloading": "## Offloading",
    "## Args for multi-item scoring": "## 多项 scoring 参数",
    "## Optimization/debug options": "## 优化与调试选项",
    "## Dynamic batch tokenizer": "## 动态 batch tokenizer",
    "## Debug tensor dumps": "## 调试 tensor dump",
    "## PD disaggregation": "## PD 分离",
    "## Encode prefill disaggregation": "## Encode prefill 分离",
    "## Custom weight loader": "## 自定义权重加载器",
    "## For PD-Multiplexing": "## PD-Multiplexing 参数",
    "## Configuration file support": "## 配置文件支持",
    "## For Multi-Modal": "## 多模态参数",
    "## For checkpoint decryption": "## checkpoint 解密参数",
    "## Forward hooks": "## Forward hooks",
    "## For MindStudio-probe(msProbe) dump": "## MindStudio-probe(msProbe) dump 参数",
    "## Deprecated arguments": "## 已弃用参数",
}
for a, b in heading_map.items():
    text = text.replace(a, b)

text = text.replace("| Parameter | Description | Defaults | Choices |", "| 参数 | 说明 | 默认值 | 选项 |")
text = text.replace("| Argument | Description | Defaults | Options |", "| 参数 | 说明 | 默认值 | 选项 |")
text = text.replace(
    "Please consult the documentation below and [server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) to learn more about the arguments you may provide when launching a server.",
    "如需了解启动服务器时可提供的更多参数，请参考下方文档以及 [server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)。",
)

phrase_map = {
    "The path of the model weights.": "模型权重路径。",
    "This can be a local folder or a Hugging Face repo ID.": "可以是本地目录或 Hugging Face repo ID。",
    "The path of the tokenizer.": "tokenizer 路径。",
    "Tokenizer mode.": "Tokenizer 模式。",
    "The host of the HTTP server.": "HTTP 服务器监听地址。",
    "The port of the HTTP server.": "HTTP 服务器端口。",
    "If set": "如果设置该参数",
    "If true": "如果为 true",
    "If false": "如果为 false",
    "Whether to": "是否",
    "Enable": "启用",
    "Disable": "禁用",
    "The number of": "数量：",
    "The maximum": "最大",
    "The minimum": "最小",
    "Defaults to": "默认使用",
    "Type: str": "类型：str",
    "Type: int": "类型：int",
    "Type: float": "类型：float",
    "Type: bool": "类型：bool",
    "bool flag (set to enable)": "bool flag（设置即启用）",
    "Path to": "路径：",
    "The size of": "大小：",
    "The format of": "格式：",
}
for a, b in phrase_map.items():
    text = text.replace(a, b)

dst.write_text(text, encoding="utf-8")
