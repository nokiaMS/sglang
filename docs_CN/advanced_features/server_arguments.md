# 服务器参数

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
  ```bash
  # Node 0
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 0

  # Node 1
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 1
  ```

如需了解启动服务器时可提供的更多参数，请参考下方文档以及 [server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)。

## 模型与 tokenizer
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--model-path`<br>`--model` | 模型权重路径。 可以是本地目录或 Hugging Face repo ID。 | `None` | 类型：str |
| `--tokenizer-path` | tokenizer 路径。 | `None` | 类型：str |
| `--tokenizer-mode` | Tokenizer 模式。 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer. | `auto` | `auto`, `slow` |
| `--tokenizer-worker-num` | The worker num of the tokenizer manager. | `1` | 类型：int |
| `--skip-tokenizer-init` | 如果设置该参数, skip init tokenizer and pass input_ids in generate request. | `False` | bool flag（设置即启用） |
| `--load-format` | 格式： the model weights to load. "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. "pt" will load the weights in the pytorch bin format. "safetensors" will load the weights in the safetensors format. "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading. "dummy" will initialize the weights with random values, which is mainly for profiling."gguf" will load the weights in the gguf format. "bitsandbytes" will load the weights using bitsandbytes quantization."layered" loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller. "flash_rl" will load the weights in flash_rl format. "fastsafetensors" and "private" are also supported. "runai_streamer" enables direct model loading from object storage and shared file systems.| `auto` | `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `sharded_state`, `gguf`, `bitsandbytes`, `layered`, `flash_rl`, `remote`, `remote_instance`, `fastsafetensors`, `private`, `runai_streamer` |
| `--model-loader-extra-config` | Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format. | `{}` | 类型：str |
| `--trust-remote-code` | Whether or not to allow for custom models defined on the Hub in their own modeling files. | `False` | bool flag（设置即启用） |
| `--context-length` | The model's maximum context length. 默认使用 None (will use the value from the model's config.json instead). | `None` | 类型：int |
| `--is-embedding` | 是否 use a CausalLM as an embedding model. | `False` | bool flag（设置即启用） |
| `--enable-multimodal` | 启用 the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen | `None` | bool flag（设置即启用） |
| `--revision` | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | `None` | 类型：str |
| `--model-impl` | Which implementation of the model to use. * "auto" will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available. * "sglang" will use the SGLang model implementation. * "transformers" will use the Transformers model implementation. | `auto` | 类型：str |

## HTTP 服务器
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--host` | HTTP 服务器监听地址。 | `127.0.0.1` | 类型：str |
| `--port` | HTTP 服务器端口。 | `30000` | 类型：int |
| `--fastapi-root-path` | App is behind a path based routing proxy. | `""` | 类型：str |
| `--grpc-mode` | 如果设置该参数, use gRPC server instead of HTTP server. | `False` | bool flag（设置即启用） |
| `--skip-server-warmup` | 如果设置该参数, skip warmup. | `False` | bool flag（设置即启用） |
| `--warmups` | Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests | `None` | 类型：str |
| `--nccl-port` | The port for NCCL distributed environment setup. 默认使用 a random port. | `None` | 类型：int |
| `--checkpoint-engine-wait-weights-before-ready` | 如果设置该参数, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests. | `False` | bool flag（设置即启用） |

## 量化与数据类型
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--dtype` | Data type for model weights and activations. * "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. * "half" for FP16. Recommended for AWQ quantization. * "float16" is the same as "half". * "bfloat16" for a balance between precision and range. * "float" is shorthand for FP32 precision. * "float32" for FP32 precision. | `auto` | `auto`, `half`, `float16`, `bfloat16`, `float`, `float32` |
| `--quantization` | The quantization method. | `None` | `awq`, `fp8`, `gptq`, `marlin`, `gptq_marlin`, `awq_marlin`, `bitsandbytes`, `gguf`, `modelopt`, `modelopt_fp8`, `modelopt_fp4`, `petit_nvfp4`, `w8a8_int8`, `w8a8_fp8`, `moe_wna16`, `qoq`, `w4afp8`, `mxfp4`, `mxfp8`, `auto-round`, `compressed-tensors`, `modelslim`, `quark_int4fp8_moe` |
| `--quantization-param-path` | 路径： the JSON file containing the KV cache scaling factors. This should generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling factors default to 1.0, which may cause accuracy issues. | `None` | Type: Optional[str] |
| `--kv-cache-dtype` | Data type for kv cache storage. "auto" will use model data type. "bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" are supported for CUDA 11.8+. "fp4_e2m1" (only mxfp4) is supported for CUDA 12.8+ and PyTorch 2.8.0+ | `auto` | `auto`, `fp8_e5m2`, `fp8_e4m3`, `bf16`, `bfloat16`, `fp4_e2m1` |
| `--enable-fp32-lm-head` | 如果设置该参数, the LM head outputs (logits) are in FP32. | `False` | bool flag（设置即启用） |
| `--modelopt-quant` | The ModelOpt quantization configuration. Supported values: 'fp8', 'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. This requires the NVIDIA Model Optimizer library to be installed: pip install nvidia-modelopt | `None` | 类型：str |
| `--modelopt-checkpoint-restore-path` | 路径： restore a previously saved ModelOpt quantized checkpoint. If provided, the quantization process will be skipped and the model will be loaded from this checkpoint. | `None` | 类型：str |
| `--modelopt-checkpoint-save-path` | 路径： save the ModelOpt quantized checkpoint after quantization. This allows reusing the quantized model in future runs. | `None` | 类型：str |
| `--modelopt-export-path` | 路径： export the quantized model in HuggingFace format after ModelOpt quantization. The exported model can then be used directly with SGLang for inference. If not provided, the model will not be exported. | `None` | 类型：str |
| `--quantize-and-serve` | Quantize the model with ModelOpt and immediately serve it without exporting. This is useful for development and prototyping. For production, it's recommended to use separate quantization and deployment steps. | `False` | bool flag（设置即启用） |
| `--rl-quant-profile` | 路径： the FlashRL quantization profile. Required when using --load-format flash_rl. | `None` | 类型：str |
| `--enable-quant-communications` | 启用 INT8 quantization of TP communications (Supported only for NPU for Qwen3 series). | `False` | bool flag（设置即启用） |

## 内存与调度
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--mem-fraction-static` | The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. | `None` | 类型：float |
| `--max-running-requests` | 最大 number of running requests. | `None` | 类型：int |
| `--max-queued-requests` | 最大 number of queued requests. This option is ignored when using disaggregation-mode. | `None` | 类型：int |
| `--max-total-tokens` | 最大 number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes. | `None` | 类型：int |
| `--chunked-prefill-size` | 最大 number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill. | `None` | 类型：int |
| `--prefill-max-requests` | 最大 number of requests in a prefill batch. If not specified, there is no limit. | `None` | 类型：int |
| `--enable-dynamic-chunking` | 启用 dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks. | `False` | bool flag（设置即启用） |
| `--max-prefill-tokens` | 最大 number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length. | `16384` | 类型：int |
| `--schedule-policy` | The scheduling policy of the requests. | `fcfs` | `lpm`, `random`, `fcfs`, `dfs-weight`, `lof`, `priority`, `routing-key` |
| `--enable-priority-scheduling` | 启用 priority scheduling. Requests with higher priority integer values will be scheduled first by default. | `False` | bool flag（设置即启用） |
| `--abort-on-priority-when-disabled` | 如果设置该参数, abort requests that specify a priority when priority scheduling is disabled. | `False` | bool flag（设置即启用） |
| `--schedule-low-priority-values-first` | If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first. | `False` | bool flag（设置即启用） |
| `--priority-scheduling-preemption-threshold` | Minimum difference in priorities for an incoming request to have to preempt running request(s). | `10` | 类型：int |
| `--schedule-conservativeness` | How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently. | `1.0` | 类型：float |
| `--page-size` | 数量： tokens in a page. | `1` | 类型：int |
| `--swa-full-tokens-ratio` | The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens. | `0.8` | 类型：float |
| `--disable-hybrid-swa-memory` | 禁用 the hybrid SWA memory. | `False` | bool flag（设置即启用） |
| `--radix-eviction-policy` | The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used. | `lru` | `lru`, `lfu` |
| `--enable-prefill-delayer` | 启用 prefill delayer for DP attention to reduce idle time. | `False` | bool flag（设置即启用） |
| `--prefill-delayer-max-delay-passes` | Maximum forward passes to delay prefill. | `30` | 类型：int |
| `--prefill-delayer-token-usage-low-watermark` | Token usage low watermark for prefill delayer. | `None` | 类型：float |
| `--prefill-delayer-queue-min-ratio` | Opt-in to the adaptive queue-based delay trigger (independent of the slot-based one). Defers prefill until the waiting queue reaches `min(running_req * ratio, max_prefill_bs)` so small fragments batch into a larger prefill. Unset keeps the original slot-only behavior. Typical: `0.1`–`0.5`. | `None` | 类型：float |
| `--prefill-delayer-max-delay-ms` | Wall-clock cap (ms) on a single queue-trigger delay; once exceeded, prefill is force-released to bound worst-case TTFT. Only consulted when `--prefill-delayer-queue-min-ratio` is set. Typical: `1000`–`5000`. | `5000` | 类型：float |
| `--prefill-delayer-forward-passes-buckets` | Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added. | `None` | List[float] |
| `--prefill-delayer-wait-seconds-buckets` | Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added. | `None` | List[float] |

## 运行时选项
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--device` | The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). 默认使用 auto-detection if not specified. | `None` | 类型：str |
| `--tensor-parallel-size`<br>`--tp-size` | The tensor parallelism size. | `1` | 类型：int |
| `--pipeline-parallel-size`<br>`--pp-size` | The pipeline parallelism size. | `1` | 类型：int |
| `--attention-context-parallel-size`<br>`--attn-cp-size`| The attention context parallelism size. | `1` | 类型：int|
| `--moe-data-parallel-size`<br>`--moe-dp-size`| The moe data parallelism size. | `1` | 类型：int|
| `--pp-max-micro-batch-size` | 最大 micro batch size in pipeline parallelism. | `None` | 类型：int |
| `--pp-async-batch-depth` | The async batch depth of pipeline parallelism. | `0` | 类型：int |
| `--stream-interval` | The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher | `1` | 类型：int |
| `--incremental-streaming-output` | 是否 output as a sequence of disjoint segments. | `False` | bool flag（设置即启用） |
| `--random-seed` | The random seed. | `None` | 类型：int |
| `--constrained-json-whitespace-pattern` | (outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model to generate consecutive whitespaces, set the pattern to [\n\t ]* | `None` | 类型：str |
| `--constrained-json-disable-any-whitespace` | (xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output. | `False` | bool flag（设置即启用） |
| `--watchdog-timeout` | Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging. | `300` | 类型：float |
| `--soft-watchdog-timeout` | Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging. | `None` | 类型：float |
| `--dist-timeout` | Set timeout for torch.distributed initialization. | `None` | 类型：int |
| `--download-dir` | Model download directory for huggingface. | `None` | 类型：str |
| `--model-checksum` | Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID. | `None` | 类型：str |
| `--base-gpu-id` | The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine. | `0` | 类型：int |
| `--gpu-id-step` | The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,... | `1` | 类型：int |
| `--sleep-on-idle` | Reduce CPU usage when sglang is idle. | `False` | bool flag（设置即启用） |
| `--custom-sigquit-handler` | Register a custom sigquit handler so you can do additional cleanup after the server is shutdown. This is only available for Engine, not for CLI. | `None` | 类型：str |

## 日志
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--log-level` | The logging level of all loggers. | `info` | 类型：str |
| `--log-level-http` | The logging level of HTTP server. If not set, reuse --log-level by default. | `None` | 类型：str |
| `--log-requests` | Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level | `False` | bool flag（设置即启用） |
| `--log-requests-level` | 0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output. | `2` | `0`, `1`, `2`, `3` |
| `--log-requests-format` | Format for request logging: 'text' (human-readable) or 'json' (structured) | `text` | `text`, `json` |
| `--log-requests-target` | Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. | `None` | List[str] |
| `--uvicorn-access-log-exclude-prefixes` | Exclude uvicorn access logs whose request path starts with any of these prefixes. 默认使用 empty (disabled). | `[]` | List[str] |
| `--crash-dump-folder` | Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled. | `None` | 类型：str |
| `--show-time-cost` | Show time cost of custom marks. | `False` | bool flag（设置即启用） |
| `--enable-metrics` | 启用 log prometheus metrics. | `False` | bool flag（设置即启用） |
| `--enable-mfu-metrics` | 启用 estimated MFU-related prometheus metrics. | `False` | bool flag（设置即启用） |
| `--enable-metrics-for-all-schedulers` | 启用 --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0. | `False` | bool flag（设置即启用） |
| `--tokenizer-metrics-custom-labels-header` | Specify the HTTP header for passing custom labels for tokenizer metrics. | `x-custom-labels` | 类型：str |
| `--tokenizer-metrics-allowed-custom-labels` | The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set. | `None` | List[str] |
| `--bucket-time-to-first-token` | The buckets of time to first token, specified as a list of floats. | `None` | List[float] |
| `--bucket-inter-token-latency` | The buckets of inter-token latency, specified as a list of floats. | `None` | List[float] |
| `--bucket-e2e-request-latency` | The buckets of end-to-end request latency, specified as a list of floats. | `None` | List[float] |
| `--collect-tokens-histogram` | Collect prompt/generation tokens histogram. | `False` | bool flag（设置即启用） |
| `--prompt-tokens-buckets` | The buckets rule of prompt tokens. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). | `None` | List[str] |
| `--generation-tokens-buckets` | The buckets rule for generation tokens histogram. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). | `None` | List[str] |
| `--gc-warning-threshold-secs` | The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable. | `0.0` | 类型：float |
| `--decode-log-interval` | The log interval of decode batch. | `40` | 类型：int |
| `--enable-request-time-stats-logging` | 启用 per request time stats logging | `False` | bool flag（设置即启用） |
| `--kv-events-config` | Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used. | `None` | 类型：str |
| `--enable-trace` | 启用 opentelemetry trace | `False` | bool flag（设置即启用） |
| `--otlp-traces-endpoint` | Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port> | `localhost:4317` | 类型：str |

## RequestMetricsExporter 配置
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--export-metrics-to-file` | Export performance metrics for each request to local file (e.g. for forwarding to external systems). | `False` | bool flag（设置即启用） |
| `--export-metrics-to-file-dir` | Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled). | `None` | 类型：str |

## API 相关
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--api-key` | Set API key of the server. It is also used in the OpenAI API compatible server. | `None` | 类型：str |
| `--admin-api-key` | Set **admin API key** for administrative/control endpoints (e.g., weights update, cache flush, `/server_info`). Endpoints marked as admin-only require `Authorization: Bearer <admin_api_key>` when this is set. | `None` | 类型：str |
| `--served-model-name` | Override the model name returned by the v1/models endpoint in OpenAI API server. | `None` | 类型：str |
| `--weight-version` | Version identifier for the model weights. 默认使用 'default' if not specified. | `default` | 类型：str |
| `--chat-template` | The builtin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server. | `None` | 类型：str |
| `--hf-chat-template-name` | When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used. | `None` | 类型：str |
| `--completion-template` | The builtin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently. | `None` | 类型：str |
| `--file-storage-path` | The path of the file storage in backend. | `sglang_storage` | 类型：str |
| `--enable-cache-report` | Return number of cached tokens in usage.prompt_tokens_details for each openai request. | `False` | bool flag（设置即启用） |
| `--reasoning-parser` | Specify the parser for reasoning models. Supported parsers: [deepseek-r1, deepseek-v3, glm45, gpt-oss, kimi, qwen3, qwen3-thinking, step3]. | `None` | `deepseek-r1`, `deepseek-v3`, `glm45`, `gpt-oss`, `kimi`, `qwen3`, `qwen3-thinking`, `step3` |
| `--tool-call-parser` | Specify the parser for handling tool-call interactions. Supported parsers: [deepseekv3, deepseekv31, glm, glm45, glm47, gpt-oss, kimi_k2, llama3, mistral, pythonic, qwen, qwen25, qwen3_coder, step3]. | `None` | `deepseekv3`, `deepseekv31`, `glm`, `glm45`, `glm47`, `gpt-oss`, `kimi_k2`, `llama3`, `mistral`, `pythonic`, `qwen`, `qwen25`, `qwen3_coder`, `step3`, `gigachat3` |
| `--tool-server` | Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used. | `None` | 类型：str |
| `--sampling-defaults` | Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'. | `model` | `openai`, `model` |

## 数据并行
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--data-parallel-size`<br>`--dp-size` | The data parallelism size. | `1` | 类型：int |
| `--load-balance-method` | The load balancing strategy for data parallelism. The `total_tokens` algorithm can only be used when DP attention is applied. This algorithm performs load balancing based on the real-time token load of the DP workers. | `auto` | `auto`, `round_robin`, `follow_bootstrap_room`, `total_requests`, `total_tokens` |

## 多节点分布式服务
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--dist-init-addr`<br>`--nccl-init-addr` | The host address for initializing distributed backend (e.g., `192.168.0.2:25000`). | `None` | 类型：str |
| `--nnodes` | 数量： nodes. | `1` | 类型：int |
| `--node-rank` | The node rank. | `0` | 类型：int |

## 模型覆盖参数
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--json-model-override-args` | A dictionary in JSON string format used to override default model configurations. | `{}` | 类型：str |
| `--preferred-sampling-params` | json-formatted sampling settings that will be returned in /get_model_info | `None` | 类型：str |

## LoRA
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-lora` | 启用 LoRA support for the model. This argument is automatically set to `True` if `--lora-paths` is provided for backward compatibility. | `False` | Bool flag (set to enable) |
| `--enable-lora-overlap-loading` | 启用 asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters. | `False` | Bool flag (set to enable)
| `--max-lora-rank` | 最大 LoRA rank that should be supported. If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of larger LoRA rank after server startup. | `None` | 类型：int |
| `--lora-target-modules` | The union set of all target modules where LoRA should be applied (e.g., `q_proj`, `k_proj`, `gate_proj`). If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. You can also set it to `all` to enable LoRA for all supported modules; note this may introduce minor performance overhead. | `None` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `qkv_proj`, `gate_up_proj`, `all` |
| `--lora-paths` | The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: `<PATH>` \| `<NAME>=<PATH>` \| JSON with schema `{"lora_name": str, "lora_path": str, "pinned": bool}`. | `None` | Type: List[str] / JSON objects |
| `--max-loras-per-batch` | Maximum number of adapters for a running batch, including base-only requests. | `8` | 类型：int |
| `--max-loaded-loras` | If specified, limits the maximum number of LoRA adapters loaded in CPU memory at a time. Must be ≥ `--max-loras-per-batch`. | `None` | 类型：int |
| `--lora-eviction-policy` | LoRA adapter eviction policy when the GPU memory pool is full. | `lru` | `lru`, `fifo` |
| `--lora-backend` | Choose the kernel backend for multi-LoRA serving. | `csgmv` | `triton`, `csgmv`, `ascend`, `torch_native` |
| `--max-lora-chunk-size` | Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when `--lora-backend` is `csgmv`. Larger values may improve performance. | `16` | `16`, `32`, `64`, `128` |
| `--lora-drain-wait-threshold` | When any LoRA adapter request waits longer than this threshold (in seconds), the scheduler will selectively drain one running adapter to make room. This mitigates extreme tail latency under high or skewed workloads by preventing a small set of adapters from monopolizing batch slots. Set to 0 to disable draining (default). | `0.0` | 类型：float |

## Kernel 后端（Attention、Sampling、Grammar、GEMM）
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--attention-backend` | Choose the kernels for attention layers. | `None` | `triton`, `torch_native`, `flex_attention`, `dsa` (canonical; `nsa` is a deprecated alias), `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--prefill-attention-backend` | Choose the kernels for prefill attention layers (have priority over --attention-backend). | `None` | `triton`, `torch_native`, `flex_attention`, `dsa` (canonical; `nsa` is a deprecated alias), `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--decode-attention-backend` | Choose the kernels for decode attention layers (have priority over --attention-backend). | `None` | `triton`, `torch_native`, `flex_attention`, `dsa` (canonical; `nsa` is a deprecated alias), `cutlass_mla`, `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_mla`, `trtllm_mha`, `dual_chunk_flash_attn`, `aiter`, `wave`, `intel_amx`, `ascend` |
| `--sampling-backend` | Choose the kernels for sampling layers. | `None` | `flashinfer`, `pytorch`, `ascend` |
| `--grammar-backend` | Choose the backend for grammar-guided decoding. | `None` | `xgrammar`, `outlines`, `llguidance`, `none` |
| `--mm-attention-backend` | Set multimodal attention backend. | `None` | `sdpa`, `fa3`, `fa4`, `triton_attn`, `ascend_attn`, `aiter_attn` |
| `--dsa-prefill-backend` | Choose the DSA backend for the prefill stage (overrides `--attention-backend` when running DeepSeek DSA-style attention). `--nsa-prefill-backend` is a deprecated alias. | `flashmla_sparse` | `flashmla_sparse`, `flashmla_kv`, `flashmla_auto`, `fa3`, `tilelang`, `aiter`, `trtllm` |
| `--dsa-decode-backend` | Choose the DSA backend for the decode stage when running DeepSeek DSA-style attention. Overrides `--attention-backend` for decoding. `--nsa-decode-backend` is a deprecated alias. | `fa3` | `flashmla_sparse`, `flashmla_kv`, `fa3`, `tilelang`, `aiter`, `trtllm` |
| `--dsa-topk-backend` | Choose the DSA indexer top-k backend. The `torch` backend currently requires `SGLANG_DSA_FUSE_TOPK=false`. | `sgl-kernel` | `sgl-kernel`, `torch`, `flashinfer` |
| `--fp8-gemm-backend` | Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (FlashInfer TRTLLM backend; SM100/SM103 only), 'flashinfer_cutlass' (FlashInfer CUTLASS backend, SM120 only), 'flashinfer_deepgemm' (Hopper SM90 only, uses swapAB optimization for small M dimensions in decoding), 'cutlass' (optimal for Hopper/Blackwell GPUs and high-throughput), 'triton' (fallback, widely compatible), 'aiter' (ROCm only).| `auto` | `auto`, `deep_gemm`, `flashinfer_trtllm`, `flashinfer_cutlass`, `flashinfer_deepgemm`, `cutlass`, `triton`, `aiter` |
| `--fp4-gemm-backend` | Choose the runner backend for NVFP4 GEMM operations. Options: 'flashinfer_cutlass' (default), 'auto' (auto-selects between flashinfer_cudnn/flashinfer_cutlass based on CUDA/cuDNN version), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling). All backends are from FlashInfer; when FlashInfer is unavailable, sgl-kernel CUTLASS is used as an automatic fallback.| `flashinfer_cutlass` | `auto`, `flashinfer_cudnn`, `flashinfer_cutlass`, `flashinfer_trtllm` |
| `--disable-flashinfer-autotune` | Flashinfer autotune is enabled by default. Set this flag to disable the autotune. | `False` | bool flag（设置即启用） |

## 投机解码
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--speculative-algorithm` | Speculative algorithm. | `None` | `EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM` |
| `--speculative-draft-model-path`<br>`--speculative-draft-model` | The path of the draft model weights. 可以是本地目录或 Hugging Face repo ID。 | `None` | 类型：str |
| `--speculative-draft-model-revision` | The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | `None` | 类型：str |
| `--speculative-draft-load-format` | 格式： the draft model weights to load. If not specified, will use the same format as --load-format. Use 'dummy' to initialize draft model weights with random values for profiling. | `None` | Same as --load-format options |
| `--speculative-num-steps` | 数量： steps sampled from draft model in Speculative Decoding. | `None` | 类型：int |
| `--speculative-eagle-topk` | 数量： tokens sampled from the draft model in eagle2 each step. | `None` | 类型：int |
| `--speculative-num-draft-tokens` | 数量： tokens sampled from the draft model in Speculative Decoding. | `None` | 类型：int |
| `--speculative-accept-threshold-single` | Accept a draft token if its probability in the target model is greater than this threshold. | `1.0` | 类型：float |
| `--speculative-accept-threshold-acc` | The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc). | `1.0` | 类型：float |
| `--speculative-token-map` | The path of the draft model's small vocab table. | `None` | 类型：str |
| `--speculative-attention-mode` | Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'. | `prefill` | `prefill`, `decode` |
| `--speculative-draft-attention-backend` | Attention backend for speculative decoding drafting. | `None` | Same as attention backend options |
| `--speculative-moe-runner-backend` | MOE backend for EAGLE speculative decoding, see --moe-runner-backend for options. Same as moe runner backend if unset. | `None` | Same as --moe-runner-backend options |
| `--speculative-moe-a2a-backend` | MOE A2A backend for EAGLE speculative decoding, see --moe-a2a-backend for options. Same as moe a2a backend if unset. | `None` | Same as --moe-a2a-backend options |
| `--speculative-draft-model-quantization` | The quantization method for speculative model. | `None` | Same as --quantization options |

## Ngram 投机解码
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--speculative-ngram-min-bfs-breadth` | 最小 breadth for BFS (Breadth-First Search) in ngram speculative decoding. | `1` | 类型：int |
| `--speculative-ngram-max-bfs-breadth` | 最大 breadth for BFS (Breadth-First Search) in ngram speculative decoding. | `10` | 类型：int |
| `--speculative-ngram-match-type` | Ngram tree-building mode. `BFS` selects recency-based expansion and `PROB` selects frequency-based expansion. This setting is forwarded to the ngram cache implementation. | `BFS` | `BFS`, `PROB` |
| `--speculative-ngram-max-trie-depth` | Maximum suffix length stored and matched by the ngram trie. | `18` | 类型：int |
| `--speculative-ngram-capacity` | The cache capacity for ngram speculative decoding. | `10000000` | 类型：int |

## 多层 Eagle 投机解码
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-multi-layer-eagle` | 启用 multi-layer Eagle speculative decoding. | `False` | bool flag（设置即启用） |

## MoE
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--expert-parallel-size`<br>`--ep-size`<br>`--ep` | The expert parallelism size. | `1` | 类型：int |
| `--moe-a2a-backend` | Select the backend for all-to-all communication for expert parallelism. | `none` | `none`, `deepep`, `mooncake`, `mori`, `nixl`, `ascend_fuseep`|
| `--moe-runner-backend` | Choose the runner backend for MoE. | `auto` | `auto`, `deep_gemm`, `triton`, `triton_kernel`, `flashinfer_trtllm`, `flashinfer_trtllm_routed`, `flashinfer_cutlass`, `flashinfer_mxfp4`, `flashinfer_cutedsl`, `cutlass` |
| `--flashinfer-mxfp4-moe-precision` | Choose the computation precision of flashinfer mxfp4 moe | `default` | `default`, `bf16` |
| `--enable-flashinfer-allreduce-fusion` | 启用 FlashInfer allreduce fusion with Residual RMSNorm. | `False` | bool flag（设置即启用） |
| `--enable-aiter-allreduce-fusion` | 启用 aiter allreduce fusion with Residual RMSNorm. | `False` | bool flag（设置即启用） |
| `--deepep-mode` | Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch. | `auto` | `normal`, `low_latency`, `auto` |
|  `--deepep-dispatcher-output-dtype` | Select DeepEP dispather output dtype, could be `bf16`, `fp8`, `int8` (only Ascend A2/A3 NPU), `nvfp4` or `auto`. Default is `auto`, which follows a priority order (server argument → deprecated env var → input_global_scale check → dispatcher_output_dtype from quant_config → flashinfer/cutlass backend → NPU BF16 default → GPU FP8 default) | `auto` | `bf16`, `fp8`, `int8`, `nvfp4`, `auto` |
| `--ep-num-redundant-experts` | Allocate this number of redundant experts in expert parallel. | `0` | 类型：int |
| `--ep-dispatch-algorithm` | The algorithm to choose ranks for redundant experts in expert parallel. | `None` | 类型：str |
| `--init-expert-location` | Initial location of EP experts. | `trivial` | 类型：str |
| `--enable-eplb` | 启用 EPLB algorithm | `False` | bool flag（设置即启用） |
| `--eplb-algorithm` | Chosen EPLB algorithm | `auto` | 类型：str |
| `--eplb-rebalance-num-iterations` | Number of iterations to automatically trigger a EPLB re-balance. | `1000` | 类型：int |
| `--eplb-rebalance-layers-per-chunk` | Number of layers to rebalance per forward pass. | `None` | 类型：int |
| `--eplb-min-rebalancing-utilization-threshold` | Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0]. | `1.0` | 类型：float |
| `--expert-distribution-recorder-mode` | Mode of expert distribution recorder. | `None` | 类型：str |
| `--expert-distribution-recorder-buffer-size` | Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer. | `None` | 类型：int |
| `--enable-expert-distribution-metrics` | 启用 logging metrics for expert balancedness | `False` | bool flag（设置即启用） |
| `--deepep-config` | Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path. | `None` | 类型：str |
| `--moe-dense-tp-size` | TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports. | `None` | 类型：int |
| `--elastic-ep-backend` | Specify the collective communication backend for elastic EP. Currently supports 'mooncake'. | `none` | `none`, `mooncake` |
| `--enable-elastic-expert-backup` | 启用 elastic EP backend to backup expert weights in DRAM feature. Currently supports 'mooncake'.| `False` | bool flag（设置即启用） |
| `--mooncake-ib-device` | The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled. | `None` | 类型：str |
| `--enable-deepep-waterfill` | 启用 DeepEP Waterfill: dispatch the shared expert as the 9th routed expert to the least-loaded EP rank. Automatically sets `--moe-a2a-backend deepep`, implicitly enables shared-expert fusion, and supports `--deepep-mode auto`, `normal`, or `low_latency`. Use `auto` or `low_latency` for production decode so CUDA graph remains enabled. Supported on DeepSeek-V3/R1 with EP >= 2. By default, Waterfill uses the static local-batch path; set `SGLANG_DISABLE_STATIC_WATERFILL=1` to force dynamic Waterfill with runtime EP all-reduce. | `False` | bool flag（设置即启用） |
| `--elastic-ep-rejoin` | Indicates that this process is a relaunched elastic EP rank that should rejoin an existing process group during rank recovery. | `False` | bool flag（设置即启用） |

## Mamba Cache
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--max-mamba-cache-size` | 最大 size of the mamba cache. | `None` | 类型：int |
| `--mamba-ssm-dtype` | The data type of the SSM states in mamba cache. | `float32` | `float32`, `bfloat16`, `float16` |
| `--mamba-full-memory-ratio` | The ratio of mamba state memory to full kv cache memory. | `0.9` | 类型：float |
| `--mamba-scheduler-strategy` | The strategy to use for mamba scheduler. `auto` currently defaults to `no_buffer`. 1. `no_buffer` does not support overlap scheduler due to not allocating extra mamba state buffers. Branching point caching support is feasible but not implemented. 2. `extra_buffer` supports overlap schedule by allocating extra mamba state buffers to track mamba state for caching (mamba state usage per running req becomes `2x` for non-spec; `1+(1/(2+speculative_num_draft_tokens))x` for spec dec (e.g. 1.16x if speculative_num_draft_tokens==4)). 2a. `extra_buffer` is strictly better for non-KV-cache-bound cases; for KV-cache-bound cases, the tradeoff depends on whether enabling overlap outweighs reduced max running requests. 2b. mamba caching at radix cache branching point is strictly better than non-branch but requires kernel support, currently only extra_buffer supports branching. | `auto` | `auto`, `no_buffer`, `extra_buffer` |
| `--mamba-track-interval` | The interval (in tokens) to track the mamba state during decode. Only used when `--mamba-scheduler-strategy` is `extra_buffer`. Must be divisible by page_size if set, and must be >= speculative_num_draft_tokens when using speculative decoding. | `256` | 类型：int |

## 分层缓存
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-hierarchical-cache` | 启用 hierarchical cache | `False` | bool flag（设置即启用） |
| `--hicache-ratio` | The ratio of the size of host KV cache memory pool to the size of device pool. | `2.0` | 类型：float |
| `--hicache-size` | 大小： host KV cache memory pool in gigabytes, which will override the hicache_ratio if set. | `0` | 类型：int |
| `--hicache-write-policy` | The write policy of hierarchical cache. | `write_through` | `write_back`, `write_through`, `write_through_selective` |
| `--hicache-io-backend` | The IO backend for KV cache transfer between CPU and GPU | `kernel` | `direct`, `kernel`, `kernel_ascend` |
| `--hicache-mem-layout` | The layout of host memory pool for hierarchical cache. | `layer_first` | `layer_first`, `page_first`, `page_first_direct`, `page_first_kv_split`, `page_head` |
| `--hicache-storage-backend` | The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name). | `None` | `file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, `dynamic`, `eic` |
| `--hicache-storage-prefetch-policy` | Control when prefetching from the storage backend should stop. | `best_effort` | `best_effort`, `wait_complete`, `timeout` |
| `--hicache-storage-backend-extra-config` | A dictionary in JSON string format, or a string starting with a `@` followed by a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend. | `None` | 类型：str |

## 分层稀疏注意力
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--hierarchical-sparse-attention-extra-config` | A dictionary in JSON string format for hierarchical sparse attention configuration. Required fields: `algorithm` (str), `backend` (str). All other fields are algorithm-specific and passed to the algorithm constructor. | `None` | 类型：str |

## LMCache
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-lmcache` | Using LMCache as an alternative hierarchical cache solution | `False` | bool flag（设置即启用） |

## Ktransformers
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--kt-weight-path` | [ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder. | `None` | 类型：str |
| `--kt-method` | [ktransformers parameter] Quantization formats for CPU execution. | `AMXINT4` | 类型：str |
| `--kt-cpuinfer` | [ktransformers parameter] 数量： CPUInfer threads. | `None` | 类型：int |
| `--kt-threadpool-count` | [ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA). | `2` | 类型：int |
| `--kt-num-gpu-experts` | [ktransformers parameter] 数量： GPU experts. | `None` | 类型：int |
| `--kt-max-deferred-experts-per-token` | [ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0. | `None` | 类型：int |

## Diffusion LLM

| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--dllm-algorithm` | The diffusion LLM algorithm, such as LowConfidence. | `None` | 类型：str |
| `--dllm-algorithm-config` | The diffusion LLM algorithm configurations. Must be a YAML file. | `None` | 类型：str |

## Offloading
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--cpu-offload-gb` | How many GBs of RAM to reserve for CPU offloading. | `0` | 类型：int |
| `--offload-group-size` | Number of layers per group in offloading. | `-1` | 类型：int |
| `--offload-num-in-group` | Number of layers to be offloaded within a group. | `1` | 类型：int |
| `--offload-prefetch-step` | Steps to prefetch in offloading. | `1` | 类型：int |
| `--offload-mode` | Mode of offloading. | `cpu` | 类型：str |

## 多项 scoring 参数
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--multi-item-scoring-delimiter` | Delimiter token ID for multi-item scoring. Used to combine Query and Items into a single sequence: Query<delimiter>Item1<delimiter>Item2<delimiter>... This enables efficient batch processing of multiple items against a single query. | `None` | 类型：int |

## 优化与调试选项
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--disable-radix-cache` | 禁用 RadixAttention for prefix caching. | `False` | bool flag（设置即启用） |
| `--cuda-graph-max-bs` | Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value. | `None` | 类型：int |
| `--cuda-graph-bs` | Set the list of batch sizes for cuda graph. | `None` | List[int] |
| `--disable-cuda-graph` | 禁用 cuda graph. | `False` | bool flag（设置即启用） |
| `--disable-cuda-graph-padding` | 禁用 cuda graph when padding is needed. Still uses cuda graph when padding is not needed. | `False` | bool flag（设置即启用） |
| `--enable-profile-cuda-graph` | 启用 profiling of cuda graph capture. | `False` | bool flag（设置即启用） |
| `--enable-cudagraph-gc` | 启用 garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process. | `False` | bool flag（设置即启用） |
| `--enable-layerwise-nvtx-marker` | 启用 layerwise NVTX profiling annotations for the model. This adds NVTX markers to every layer for detailed per-layer performance analysis with Nsight Systems. | `False` | bool flag（设置即启用） |
| `--enable-nccl-nvls` | 启用 NCCL NVLS for prefill heavy requests when available. | `False` | bool flag（设置即启用） |
| `--enable-symm-mem` | 启用 NCCL symmetric memory for fast collectives. | `False` | bool flag（设置即启用） |
| `--disable-flashinfer-cutlass-moe-fp4-allgather` | 禁用s quantize before all-gather for flashinfer cutlass moe. | `False` | bool flag（设置即启用） |
| `--enable-tokenizer-batch-encode` | 启用 batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds. | `False` | bool flag（设置即启用） |
| `--disable-tokenizer-batch-decode` | 禁用 batch decoding when decoding multiple completions. | `False` | bool flag（设置即启用） |
| `--disable-outlines-disk-cache` | 禁用 disk cache of outlines to avoid possible crashes related to file system or high concurrency. | `False` | bool flag（设置即启用） |
| `--disable-custom-all-reduce` | 禁用 the custom all-reduce kernel and fall back to NCCL. | `False` | bool flag（设置即启用） |
| `--enable-mscclpp` | 启用 using mscclpp for small messages for all-reduce kernel and fall back to NCCL. | `False` | bool flag（设置即启用） |
| `--enable-torch-symm-mem` | 启用 using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM10 supports world size 6, 8. | `False` | bool flag（设置即启用） |
| `--disable-overlap-schedule` | 禁用 the overlap scheduler, which overlaps the CPU scheduler with GPU model worker. | `False` | bool flag（设置即启用） |
| `--enable-mixed-chunk` | Enabling mixing prefill and decode in a batch when using chunked prefill. | `False` | bool flag（设置即启用） |
| `--enable-dp-attention` | Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported. | `False` | bool flag（设置即启用） |
| `--enable-dp-lm-head` | 启用 vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention. | `False` | bool flag（设置即启用） |
| `--enable-two-batch-overlap` | Enabling two micro batches to overlap. | `False` | bool flag（设置即启用） |
| `--enable-single-batch-overlap` | Let computation and communication overlap within one micro batch. | `False` | bool flag（设置即启用） |
| `--tbo-token-distribution-threshold` | The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap. | `0.48` | 类型：float |
| `--enable-torch-compile` | Optimize the model with torch.compile. Experimental feature. | `False` | bool flag（设置即启用） |
| `--enable-torch-compile-debug-mode` | 启用 debug mode for torch compile. | `False` | bool flag（设置即启用） |
| `--disable-piecewise-cuda-graph` | 禁用 piecewise cuda graph for extend/prefill. PCG is enabled by default. | `False` | bool flag (set to disable) |
| `--enforce-piecewise-cuda-graph` | Enforce piecewise cuda graph, skipping all auto-disable conditions. For testing only. | `False` | bool flag（设置即启用） |
| `--piecewise-cuda-graph-tokens` | Set the list of tokens when using piecewise cuda graph. | `None` | Type: JSON list |
| `--piecewise-cuda-graph-compiler` | Set the compiler for piecewise cuda graph. Choices are: eager, inductor. | `eager` | `eager`, `inductor` |
| `--torch-compile-max-bs` | Set the maximum batch size when using torch compile. | `32` | 类型：int |
| `--piecewise-cuda-graph-max-tokens` | Set the maximum tokens when using piecewise cuda graph. | `4096` | 类型：int |
| `--torchao-config` | Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row | `` | 类型：str |
| `--enable-nan-detection` | 启用 the NaN detection for debugging purposes. | `False` | bool flag（设置即启用） |
| `--enable-p2p-check` | 启用 P2P check for GPU access, otherwise the p2p access is allowed by default. | `False` | bool flag（设置即启用） |
| `--triton-attention-reduce-in-fp32` | Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16. This only affects Triton attention kernels. | `False` | bool flag（设置即启用） |
| `--triton-attention-num-kv-splits` | 数量： KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8. | `8` | 类型：int |
| `--triton-attention-split-tile-size` | 大小： split KV tile in flash decoding Triton kernel. Used for deterministic inference. | `None` | 类型：int |
| `--num-continuous-decode-steps` | Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time. | `1` | 类型：int |
| `--delete-ckpt-after-loading` | Delete the model checkpoint after loading the model. | `False` | bool flag（设置即启用） |
| `--enable-memory-saver` | Allow saving memory using release_memory_occupation and resume_memory_occupation | `False` | bool flag（设置即启用） |
| `--enable-weights-cpu-backup` | Save model weights to CPU memory during release_weights_occupation and resume_weights_occupation | `False` | bool flag（设置即启用） |
| `--enable-draft-weights-cpu-backup` | Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation | `False` | bool flag（设置即启用） |
| `--allow-auto-truncate` | Allow automatically truncating requests that exceed the maximum input length instead of returning an error. | `False` | bool flag（设置即启用） |
| `--enable-custom-logit-processor` | 启用 users to pass custom logit processors to the server (disabled by default for security) | `False` | bool flag（设置即启用） |
| `--flashinfer-mla-disable-ragged` | Not using ragged prefill wrapper when running flashinfer mla | `False` | bool flag（设置即启用） |
| `--disable-shared-experts-fusion` | 禁用 shared experts fusion optimization for deepseek v3/r1. | `False` | bool flag（设置即启用） |
| `--disable-chunked-prefix-cache` | 禁用 chunked prefix cache feature for deepseek, which should save overhead for short sequences. | `False` | bool flag（设置即启用） |
| `--disable-fast-image-processor` | Adopt base image processor instead of fast image processor. | `False` | bool flag（设置即启用） |
| `--keep-mm-feature-on-device` | Keep multimodal feature tensors on device after processing to save D2H copy. | `False` | bool flag（设置即启用） |
| `--enable-return-hidden-states` | 启用 returning hidden states with responses. | `False` | bool flag（设置即启用） |
| `--enable-return-routed-experts` | 启用 returning routed experts of each layer with responses. | `False` | bool flag（设置即启用） |
| `--scheduler-recv-interval` | The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this. | `1` | 类型：int |
| `--numa-node` | Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. | `None` | List[int] |
| `--enable-deterministic-inference` | 启用 deterministic inference mode with batch invariant ops. | `False` | bool flag（设置即启用） |
| `--rl-on-policy-target` | The training system that SGLang needs to match for true on-policy. | `None` | `fsdp` |
| `--enable-attn-tp-input-scattered` | Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent. | `False` | bool flag（设置即启用） |
| `--enable-dsa-prefill-context-parallel` | 启用 context parallelism used in the long sequence prefill phase of DeepSeek v3.2. (`--enable-nsa-prefill-context-parallel` is a deprecated alias.) | `False` | bool flag（设置即启用） |
| `--dsa-prefill-cp-mode` | Token splitting mode for the prefill phase of DeepSeek v3.2 under context parallelism. Optional values: `round-robin-split`(default),`in-seq-split`. `round-robin-split` distributes tokens across ranks based on `token_idx % cp_size`. It supports multi-batch prefill, fused MoE, and FP8 KV cache. (`--nsa-prefill-cp-mode` is a deprecated alias.) | `in-seq-split` | `in-seq-split`, `round-robin-split` |
| `--enable-fused-qk-norm-rope` | 启用 fused qk normalization and rope rotary embedding. | `False` | bool flag（设置即启用） |
| `--enable-precise-embedding-interpolation` | 启用 corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values. | `False` | bool flag（设置即启用） |

## 动态 batch tokenizer
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-dynamic-batch-tokenizer` | 启用 async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently. | `False` | bool flag（设置即启用） |
| `--dynamic-batch-tokenizer-batch-size` | [Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer. | `32` | 类型：int |
| `--dynamic-batch-tokenizer-batch-timeout` | [Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests. | `0.002` | 类型：float |

## 调试 tensor dump
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--debug-tensor-dump-output-folder` | The output folder for dumping tensors. | `None` | 类型：str |
| `--debug-tensor-dump-layers` | The layer ids to dump. Dump all layers if not specified. | `None` | Type: JSON list |
| `--debug-tensor-dump-input-file` | The input filename for dumping tensors | `None` | 类型：str |
| `--debug-tensor-dump-inject` | Inject the outputs from jax as the input of every layer. | `False` | 类型：str |

## PD 分离
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--disaggregation-mode` | Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated | `null` | `null`, `prefill`, `decode` |
| `--disaggregation-transfer-backend` | The backend for disaggregation transfer. Default is mooncake. | `mooncake` | `mooncake`, `nixl`, `ascend`, `fake` |
| `--disaggregation-bootstrap-port` | Bootstrap server port on the prefill server. Default is 8998. | `8998` | 类型：int |
| `--disaggregation-ib-device` | The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when mooncake backend is enabled. | `None` | 类型：str |
| `--disaggregation-decode-enable-offload-kvcache` | 启用 async KV cache offloading on decode server (PD mode). | `False` | bool flag（设置即启用） |
| `--num-reserved-decode-tokens` | Number of decode tokens that will have memory reserved when adding new request to the running batch. | `512` | 类型：int |
| `--disaggregation-decode-polling-interval` | The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this. | `1` | 类型：int |

## Encode prefill 分离
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--encoder-only` | For MLLM with an encoder, launch an encoder-only server | `False` | bool flag（设置即启用） |
| `--language-only` | For VLM, load weights for the language model only. | `False` | bool flag（设置即启用） |
| `--encoder-transfer-backend` | The backend for encoder disaggregation transfer. Default is zmq_to_scheduler. | `zmq_to_scheduler` | `zmq_to_scheduler`, `zmq_to_tokenizer`, `mooncake` |
| `--encoder-urls` | List of encoder server urls. | `[]` | Type: JSON list |

## 自定义权重加载器
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--custom-weight-loader` | The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func | `None` | List[str] |
| `--weight-loader-disable-mmap` | 禁用 mmap while loading weight using safetensors. | `False` | bool flag（设置即启用） |
| `--weight-loader-prefetch-checkpoints` | Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards in a background thread, reducing total network I/O on shared filesystems (NFS/Lustre) from N\*checkpoint to 1\*checkpoint. Recommended for models on network storage. | `False` | bool flag（设置即启用） |
| `--weight-loader-prefetch-num-threads` | Number of threads per rank for checkpoint prefetching. | `4` | 类型：int |
| `--remote-instance-weight-loader-seed-instance-ip` | The ip of the seed instance for loading weights from remote instance. | `None` | 类型：str |
| `--remote-instance-weight-loader-seed-instance-service-port` | The service port of the seed instance for loading weights from remote instance. | `None` | 类型：int |
| `--remote-instance-weight-loader-send-weights-group-ports` | The communication group ports for loading weights from remote instance. | `None` | Type: JSON list |
| `--remote-instance-weight-loader-backend` | The backend for loading weights from remote instance. Can be 'transfer_engine' or 'nccl'. Default is 'nccl'. | `nccl` | `transfer_engine`, `nccl` |
| `--remote-instance-weight-loader-start-seed-via-transfer-engine` | Start seed server via transfer engine backend for remote instance weight loader. | `False` | bool flag（设置即启用） |

## PD-Multiplexing 参数
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-pdmux` | 启用 PD-Multiplexing, PD running on greenctx stream. | `False` | bool flag（设置即启用） |
| `--pdmux-config-path` | The path of the PD-Multiplexing config file. | `None` | 类型：str |
| `--sm-group-num` | Number of sm partition groups. | `8` | 类型：int |

## 配置文件支持
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--config` | Read CLI options from a config file. Must be a YAML file with configuration options. | `None` | 类型：str |

## 多模态参数
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--mm-max-concurrent-calls` | The max concurrent calls for async mm data processing. | `32` | 类型：int |
| `--mm-per-request-timeout` | The timeout for each multi-modal request in seconds. | `10.0` | 类型：int |
| `--enable-broadcast-mm-inputs-process` | 启用 broadcast mm-inputs process in scheduler. | `False` | bool flag（设置即启用） |
| `--mm-process-config` | Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio`. | `{}` | Type: JSON / Dict |
| `--mm-enable-dp-encoder` | Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically. | `False` | bool flag（设置即启用） |
| `--limit-mm-data-per-request` | Limit the number of multimodal inputs per request. e.g. '{"image": 1, "video": 1, "audio": 1}' | `None` | Type: JSON / Dict |
| `--enable-mm-global-cache` | 启用 Mooncake-backed global multimodal embedding cache on encoder servers so repeated images can reuse cached ViT embeddings instead of recomputing them. | `False` | bool flag（设置即启用） |

## checkpoint 解密参数
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--decrypted-config-file` | The path of the decrypted config file. | `None` | 类型：str |
| `--decrypted-draft-config-file` | The path of the decrypted draft config file. | `None` | 类型：str |
| `--enable-prefix-mm-cache` | 启用 prefix multimodal cache. Currently only supports mm-only. | `False` | bool flag（设置即启用） |

## Forward hooks
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--forward-hooks` | JSON-formatted list of forward hook specifications. Each element must include `target_modules` (list of glob patterns matched against `model.named_modules()` names) and `hook_factory` (Python import path to a factory, e.g. `my_package.hooks:make_hook`). An optional `name` field is used for logging, and an optional `config` object is passed as a `dict` to the factory. | `None` | Type: JSON list |

## MindStudio-probe(msProbe) dump 参数
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--msprobe-dump-config` | The path of the JSON configuration file for msProbe. If specified, enables msProbe dump. | `None` | 类型：str |

## 已弃用参数
| 参数 | 说明 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-ep-moe` | NOTE: --enable-ep-moe is deprecated. Please set `--ep-size` to the same value as `--tp-size` instead. | `None` | N/A |
| `--enable-deepep-moe` | NOTE: --enable-deepep-moe is deprecated. Please set `--moe-a2a-backend` to 'deepep' instead. | `None` | N/A |
| `--prefill-round-robin-balance` | Note: Note: --prefill-round-robin-balance is deprecated now. | `None` | N/A |
| `--enable-flashinfer-cutlass-moe` | NOTE: --enable-flashinfer-cutlass-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutlass' instead. | `None` | N/A |
| `--enable-flashinfer-cutedsl-moe` | NOTE: --enable-flashinfer-cutedsl-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutedsl' instead. | `None` | N/A |
| `--enable-flashinfer-trtllm-moe` | NOTE: --enable-flashinfer-trtllm-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_trtllm' instead. | `None` | N/A |
| `--enable-triton-kernel-moe` | NOTE: --enable-triton-kernel-moe is deprecated. Please set `--moe-runner-backend` to 'triton_kernel' instead. | `None` | N/A |
| `--enable-flashinfer-mxfp4-moe` | NOTE: --enable-flashinfer-mxfp4-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_mxfp4' instead. | `None` | N/A |
| `--crash-on-nan` | Crash the server on nan logprobs. | `False` | 类型：str |
| `--hybrid-kvcache-ratio` | Mix ratio in [0,1] between uniform and hybrid kv buffers (0.0 = pure uniform: swa_size / full_size = 1)(1.0 = pure hybrid: swa_size / full_size = local_attention_size / context_length) | `None` | Optional[float] |
| `--load-watch-interval` | The interval of load watching in seconds. | `0.1` | 类型：float |
| `--nsa-prefill` | Deprecated alias for `--dsa-prefill-backend`. Choose the DSA backend for the prefill stage (overrides `--attention-backend` when running DeepSeek DSA-style attention). | `flashmla_sparse` | `flashmla_sparse`, `flashmla_decode`, `fa3`, `tilelang`, `aiter` |
| `--nsa-decode` | Deprecated alias for `--dsa-decode-backend`. Choose the DSA backend for the decode stage when running DeepSeek DSA-style attention. Overrides `--attention-backend` for decoding. | `flashmla_kv` | `flashmla_prefill`, `flashmla_kv`, `fa3`, `tilelang`, `aiter` |
