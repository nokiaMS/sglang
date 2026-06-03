# vLLM 与 SGLang 参数对比文档

> 基于官方文档整理，vLLM 版本基于最新 main 分支，SGLang 版本基于最新 main 分支。
> 最后更新：2026-06-02

---

## 目录

1. [vLLM 参数详解](#1-vllm-参数详解)
2. [SGLang 参数详解](#2-sglang-参数详解)
3. [参数对比分析](#3-参数对比分析)
4. [总结与选型建议](#4-总结与选型建议)

---

## 1. vLLM 参数详解

### 1.1 模型配置（ModelConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace 模型名称或路径 | 指定要加载的模型 |
| `--runner` | `auto` | 模型运行器类型：`auto`/`draft`/`generate`/`pooling` | 推理、draft 模型、池化任务 |
| `--convert` | `auto` | 模型转换适配：`auto`/`classify`/`embed`/`none` | 将生成模型转为分类/嵌入模型 |
| `--tokenizer` | 同 model | 分词器路径 | 分词器与模型分离时指定 |
| `--tokenizer-mode` | `auto` | 分词器模式：`auto`/`hf`/`slow`/`mistral`/`deepseek_v32`/`deepseek_v4` | 特定模型需指定分词器模式 |
| `--trust-remote-code` | `False` | 信任 HuggingFace 远程代码 | 使用自定义模型时必须开启 |
| `--dtype` | `auto` | 数据类型：`auto`/`float16`/`bfloat16`/`float32`/`half` | 控制精度与性能权衡 |
| `--seed` | `0` | 随机种子 | 需要可复现推理结果时设置 |
| `--max-model-len` | 自动推导 | 模型上下文长度（提示+输出） | 限制上下文长度以节省显存 |
| `--quantization` / `-q` | `None` | 量化方法 | 模型量化部署（AWQ/GPTQ/FP8等） |
| `--revision` | 默认版本 | 模型版本（分支/标签/commit id） | 使用特定版本模型 |
| `--served-model-name` | 同 model | API 中使用的模型名称 | 多别名对外服务 |
| `--enforce-eager` | `False` | 强制使用 PyTorch eager 模式 | 调试或 CUDA graph 不兼容时 |
| `--max-logprobs` | `20` | 返回最大 logprobs 数量 | 需要更多 logprobs 信息时调整 |
| `--disable-sliding-window` | `False` | 禁用滑动窗口 | 强制使用完整注意力 |
| `--skip-tokenizer-init` | `False` | 跳过分词器初始化 | 仅使用 token id 输入/输出 |
| `--enable-sleep-mode` | `False` | 启用睡眠模式 | GPU 空闲时释放显存 |
| `--model-impl` | `auto` | 模型实现选择：`auto`/`vllm`/`transformers`/`terratorch` | 选择不同模型实现 |
| `--generation-config` | `auto` | 生成配置路径 | 自定义生成参数 |
| `--override-generation-config` | `{}` | 覆盖生成配置 | 运行时修改生成参数 |

### 1.2 模型加载（LoadConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--load-format` | `auto` | 权重加载格式：`auto`/`safetensors`/`pt`/`gguf`/`bitsandbytes`/`tensorizer` 等 | 指定模型文件格式 |
| `--download-dir` | HF 默认缓存 | 模型下载目录 | 离线环境或自定义缓存位置 |
| `--safetensors-load-strategy` | `lazy` | safetensors 加载策略：`lazy`/`eager`/`prefetch` | NFS 环境用 eager，本地用 lazy |
| `--model-loader-extra-config` | `{}` | 模型加载器额外配置 | 特定加载器需要额外参数 |
| `--ignore-patterns` | `['original/**/*']` | 加载时忽略的文件模式 | 跳过不需要的权重文件 |

### 1.3 并行配置（ParallelConfig）

| 参数 | 默认值 | 说明 | 使用场景 | 备注 |
|------|--------|------|----------|----|
| `--tensor-parallel-size` / `-tp` | `1` | 张量并行度 | 多 GPU 分割模型层内并行 |
| `--pipeline-parallel-size` / `-pp` | `1` | 流水线并行度 | 多 GPU 按层分割 |
| `--data-parallel-size` / `-dp` | `1` | 数据并行度 | 多副本提升吞吐 |
| `--distributed-executor-backend` | 自动 | 分布式后端：`mp`/`ray`/`uni` | 多节点推理选择通信后端 |
| `--enable-expert-parallel` / `-ep` | `False` | MoE 专家并行 | MoE 模型跨 GPU 分配专家 |
| `--decode-context-parallel-size` / `-dcp` | `1` | 解码上下文并行度 | 长序列解码阶段并行 |
| `--prefill-context-parallel-size` / `-pcp` | `1` | 预填充上下文并行度 | 长序列预填充并行 |
| `--all2all-backend` | `allgather_reducescatter` | MoE All2All 后端 | MoE 专家通信选择 |
| `--max-parallel-loading-workers` | 自动 | 并行加载最大 worker 数 | 大模型加载避免 RAM OOM |
| `--nnodes` / `-n` | `1` | 多节点推理节点数 | 多机部署 | done |
| `--node-rank` / `-r` | `0` | 节点排名 | 多节点标识 | done |

### 1.4 KV 缓存配置（CacheConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--gpu-memory-utilization` | `0.92` | GPU 显存利用率（0-1） | 调整显存分配比例 |
| `--kv-cache-dtype` | `auto` | KV 缓存数据类型：`auto`/`fp8`/`fp8_e4m3` 等 | FP8 KV 缓存节省显存 |
| `--block-size` | 默认值 | 缓存块大小（token 数） | 调整 KV 缓存粒度 |
| `--enable-prefix-caching` | 未指定 | 启用前缀缓存 | 共享 system prompt 场景 |
| `--kv-cache-memory-bytes` | 自动 | KV 缓存字节数 | 精确控制 KV 缓存大小 |
| `--num-gpu-blocks-override` | `None` | 覆盖 GPU 块数 | 测试或调试用 |
| `--kv-offloading-size` | `None` | KV 缓存卸载到 CPU 的大小（GiB） | 显存不足时利用 CPU 内存 |
| `--kv-offloading-backend` | `native` | KV 卸载后端：`native`/`lmcache` | 选择 KV 卸载实现 |
| `--prefix-caching-hash-algo` | `sha256` | 前缀缓存哈希算法 | 安全性与性能权衡 |

### 1.5 调度器配置（SchedulerConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--max-num-batched-tokens` | 自动 | 单次迭代最大 token 数 | 控制批处理粒度 |
| `--max-num-seqs` | 自动 | 单次迭代最大序列数 | 控制并发请求数 |
| `--max-num-partial-prefills` | `1` | 最大部分预填充并发数 | chunked prefill 并发控制 |
| `--enable-chunked-prefill` | 自动 | 启用分块预填充 | 长短请求混合时减少延迟 |
| `--scheduling-policy` | `fcfs` | 调度策略：`fcfs`/`priority` | 优先级调度场景 |
| `--stream-interval` | `1` | 流式输出 token 缓冲区大小 | 平衡延迟与吞吐 |
| `--scheduler-reserve-full-isl` | `True` | 调度前检查完整输入长度 | 防止 KV 缓存抖动 |
| `--async-scheduling` | 自动 | 异步调度 | 提高GPU利用率 |

### 1.6 卸载配置（OffloadConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--cpu-offload-gb` | `0` | CPU 卸载空间（GiB） | 显存不足时利用 CPU 内存 |
| `--offload-backend` | `auto` | 卸载后端：`auto`/`prefetch`/`uva` | 选择卸载策略 |
| `--offload-group-size` | `0` | 分组卸载的组大小 | 分层异步预取 |
| `--offload-num-in-group` | `1` | 每组卸载的层数 | 控制卸载粒度 |
| `--offload-prefetch-step` | `1` | 预取步长 | 隐藏延迟但多用显存 |

### 1.7 多模态配置（MultiModalConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--limit-mm-per-prompt` | `{}` | 每提示多模态项限制 | 限制图片/视频数量 |
| `--enable-mm-embeds` | `False` | 启用多模态嵌入 | 传入 tensor 形式多模态数据 |
| `--mm-processor-kwargs` | 未指定 | 多模态处理器参数 | 自定义图像裁剪等 |
| `--mm-processor-cache-gb` | `4` | 多模态处理器缓存大小（GiB） | 大量多模态请求时增大 |
| `--language-model-only` | `False` | 禁用多模态输入 | 纯文本服务 |

### 1.8 LoRA 配置（LoRAConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--enable-lora` | 未指定 | 启用 LoRA | 多租户适配器服务 |
| `--max-loras` | `1` | 单批最大 LoRA 数 | 控制并发适配器数量 |
| `--max-lora-rank` | `16` | 最大 LoRA 秩 | 限制适配器大小 |
| `--fully-sharded-loras` | `False` | 完全分片 LoRA | 高 TP/高序列长度时提升性能 |
| `--lora-target-modules` | 全部支持 | 限制 LoRA 目标模块 | 指定适配模块 |
| `--max-cpu-loras` | 未指定 | CPU 内存中最大 LoRA 数 | 控制热切换缓存 |

### 1.9 可观测性配置（ObservabilityConfig）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--otlp-traces-endpoint` | 未指定 | OpenTelemetry traces 端点 | 分布式追踪 |
| `--collect-detailed-traces` | 未指定 | 详细追踪模块选择 | 性能分析 |
| `--kv-cache-metrics` | `False` | KV 缓存指标 | 分析缓存利用率 |
| `--enable-mfu-metrics` | `False` | MFU 指标 | 模型 FLOPs 利用率分析 |
| `--enable-layerwise-nvtx-tracing` | `False` | 逐层 NVTX 追踪 | GPU 性能剖析 |

### 1.10 编译与内核配置

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--compilation-config` / `-cc` | 大型默认对象 | torch.compile 和 CUDA graph 配置 | 编译优化与调试 |
| `--optimization-level` | `2` | 优化级别：`-O0`到`-O3` | 启动速度 vs 运行性能 |
| `--performance-mode` | `balanced` | 性能模式：`balanced`/`interactivity`/`throughput` | 延迟敏感 vs 吞吐优先 |
| `--cudagraph-capture-sizes` | 自动 | CUDA graph 捕获大小 | 精确控制 CUDA graph |
| `--moe-backend` | `auto` | MoE 内核后端 | 选择 MoE 计算内核 |
| `--linear-backend` | `auto` | 线性层 GEMM 内核后端 | 选择量化线性层内核 |
| `--attention-backend` | 自动 | 注意力后端 | 选择注意力实现 |

### 1.11 推测解码（Speculative Decoding）

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--speculative-config` / `-sc` | 未指定 | 推测解码完整配置 | 使用推测解码加速 |
| `--spec-method` | 未指定 | 推测方法：`eagle`/`medusa`/`ngram`/`mtp` 等 | 选择推测策略 |
| `--spec-model` | 未指定 | Draft 模型路径 | 基于 draft model 的推测 |
| `--spec-tokens` | 未指定 | 推测 token 数 | 控制推测长度 |

### 1.12 其他重要参数

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--disable-log-stats` | `False` | 禁用统计日志 | 减少日志开销 |
| `--shutdown-timeout` | `0` | 关闭超时（秒） | 优雅关闭控制 |
| `--enable-log-requests` | `False` | 记录请求信息 | 调试和审计 |
| `--kv-transfer-config` | 未指定 | KV 传输配置 | 分布式 KV 缓存传输（PD 分离） |
| `--reasoning-parser` | `""` | 推理内容解析器 | 支持 DeepSeek-R1 等推理模型 |
| `--hf-token` | 未指定 | HuggingFace 认证 token | 访问受限模型 |

---

## 2. SGLang 参数详解

### 2.1 模型与分词器

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--model-path` / `--model` | 必填 | 模型路径或 HF repo ID | 指定要加载的模型 |
| `--tokenizer-path` | 同 model-path | 分词器路径 | 分词器与模型分离时指定 |
| `--tokenizer-mode` | `auto` | 分词器模式：`auto`/`slow` | 指定快/慢分词器 |
| `--tokenizer-backend` | `huggingface` | 分词器后端：`huggingface`/`fastokens` | 性能优化用 fastokens |
| `--tokenizer-worker-num` | `1` | 分词器 worker 数量 | 高并发时增加并行分词 |
| `--detokenizer-worker-num` | `1` | 反分词器 worker 数量 | 高并发时增加并行反分词 |
| `--skip-tokenizer-init` | `False` | 跳过分词器初始化 | 仅使用 token id 输入/输出 |
| `--trust-remote-code` | `False` | 信任远程代码 | 使用自定义模型 |
| `--context-length` | `None` | 上下文长度覆盖 | 限制上下文长度以节省显存 |
| `--is-embedding` | `False` | 嵌入模式 | 使用模型做嵌入提取 |
| `--enable-multimodal` | `None` | 启用多模态 | 多模态模型推理 |
| `--revision` | `None` | 模型版本 | 使用特定版本 |
| `--model-impl` | `auto` | 模型实现：`auto`/`sglang`/`transformers` | 选择模型实现 |
| `--model-config-parser` | `auto` | 模型配置解析器 | 自定义模型配置解析 |

### 2.2 HTTP 服务器

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--host` | `127.0.0.1` | 服务器绑定地址 | 网络暴露或本地部署 |
| `--port` | `30000` | 服务器端口 | 端口冲突时修改 |
| `--fastapi-root-path` | `""` | 反向代理路径 | 部署在反向代理后 |
| `--grpc-mode` | `False` | 启用 gRPC 模式 | gRPC 服务 |
| `--skip-server-warmup` | `False` | 跳过预热 | 加快启动（牺牲首次延迟） |
| `--nccl-port` | `None` | NCCL 端口 | 指定分布式通信端口 |
| `--ssl-keyfile` / `--ssl-certfile` | `None` | SSL 证书配置 | HTTPS 服务 |

### 2.3 量化与数据类型

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--dtype` | `auto` | 数据类型：`auto`/`float16`/`bfloat16`/`float32` | 精度与性能权衡 |
| `--quantization` | `None` | 量化方法：`awq`/`fp8`/`gptq`/`marlin` 等 | 模型量化部署 |
| `--kv-cache-dtype` | `auto` | KV 缓存数据类型：`auto`/`fp8_e4m3`/`fp8_e5m2`/`bf16`/`fp4_e2m1` | FP8/FP4 KV 缓存 |
| `--quantization-param-path` | `None` | KV 缓存缩放因子 JSON 路径 | FP8 KV 缓存精度校准 |
| `--enable-fp32-lm-head` | `False` | LM Head 输出用 FP32 | 提升输出精度 |
| `--modelopt-quant` | `None` | ModelOpt 量化配置 | NVIDIA ModelOpt 在线量化 |
| `--modelopt-checkpoint-save-path` | `None` | ModelOpt 量化模型保存路径 | 保存量化后模型供复用 |

### 2.4 内存与调度

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--mem-fraction-static` | `None`（自动） | 静态显存分配比例 | OOM 时减小该值 |
| `--max-running-requests` | `None` | 最大运行请求数 | 控制并发上限 |
| `--max-queued-requests` | `None` | 最大排队请求数 | 防止请求堆积 |
| `--max-total-tokens` | `None` | 总 token 池大小 | 精确控制 token 预算 |
| `--chunked-prefill-size` | `None`（自动） | 分块预填充大小 | 控制预填充粒度 |
| `--max-prefill-tokens` | `16384` | 预填充最大 token 数 | 限制预填充批次 |
| `--prefill-max-requests` | `None` | 预填充最大请求数 | 限制预填充并发 |
| `--schedule-policy` | `fcfs` | 调度策略：`fcfs`/`lpm`/`random`/`priority`/`dfs-weight`/`lof`/`routing-key` | 选择调度策略 |
| `--enable-priority-scheduling` | `False` | 启用优先级调度 | 请求分级处理 |
| `--schedule-conservativeness` | `1.0` | 调度保守程度 | 请求频繁被撤回时增大 |
| `--page-size` | `1` | KV 缓存页大小 | 调整 KV 缓存粒度 |
| `--radix-eviction-policy` | `lru` | Radix 树淘汰策略：`lru`/`lfu`/`slru`/`priority` | 缓存淘汰策略 |
| `--enable-prefill-delayer` | `False` | 启用预填充延迟器 | DP attention 减少空闲 |
| `--enable-dynamic-chunking` | `False` | 动态分块 | PP 场景下均衡执行时间 |

### 2.5 运行时选项

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--device` | 自动 | 设备：`cuda`/`xpu`/`hpu`/`npu`/`cpu` | 非 NVIDIA 设备 |
| `--tensor-parallel-size` / `--tp-size` | `1` | 张量并行度 | 多 GPU 并行 |
| `--pipeline-parallel-size` / `--pp-size` | `1` | 流水线并行度 | 多 GPU 按层分割 |
| `--data-parallel-size` / `--dp-size` | `1` | 数据并行度 | 多副本提升吞吐 |
| `--attention-context-parallel-size` / `--attn-cp-size` | `1` | 注意力上下文并行度 | 长序列并行 |
| `--moe-data-parallel-size` / `--moe-dp-size` | `1` | MoE 数据并行度 | MoE 模型数据并行 |
| `--ep-size` | `1` | 专家并行度 | MoE 专家跨 GPU |
| `--stream-interval` | `1` | 流式输出间隔 | 平衡延迟与吞吐 |
| `--random-seed` | `None` | 随机种子 | 可复现推理 |
| `--watchdog-timeout` | `300` | 看门狗超时（秒） | 防止服务挂起 |
| `--base-gpu-id` | `0` | 起始 GPU ID | 多实例同机部署 |
| `--gpu-id-step` | `1` | GPU ID 步长 | 隔离 GPU 使用 |
| `--sleep-on-idle` | `False` | 空闲时休眠 | 降低空闲功耗 |

### 2.6 日志与监控

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--log-level` | `info` | 日志级别 | 调试时设为 debug |
| `--log-requests` | `False` | 记录请求详情 | 审计与调试 |
| `--log-requests-level` | `2` | 请求日志详细度：0-3 | 控制日志量 |
| `--enable-metrics` | `False` | 启用 Prometheus 指标 | 生产监控 |
| `--enable-mfu-metrics` | `False` | 启用 MFU 指标 | 性能分析 |
| `--enable-trace` | `False` | 启用 OpenTelemetry 追踪 | 分布式追踪 |
| `--otlp-traces-endpoint` | `localhost:4317` | OTLP 追踪端点 | 发送追踪数据 |
| `--crash-dump-folder` | `None` | 崩溃转储目录 | 崩溃诊断 |
| `--export-metrics-to-file` | `False` | 导出指标到文件 | 外部系统集成 |

### 2.7 API 相关

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--api-key` | `None` | API 密钥 | 访问控制 |
| `--admin-api-key` | `None` | 管理员 API 密钥 | 管理端点保护 |
| `--served-model-name` | 同 model-path | API 模型名称 | 自定义对外模型名 |
| `--chat-template` | `None` | 聊天模板 | 自定义对话格式 |
| `--reasoning-parser` | `None` | 推理内容解析器 | 支持 DeepSeek-R1 等推理模型 |
| `--tool-call-parser` | `None` | 工具调用解析器 | Function calling 支持 |
| `--sampling-defaults` | `model` | 采样参数默认来源 | 使用 OpenAI 或模型默认值 |
| `--enable-cache-report` | `False` | 报告缓存命中 | 缓存效果分析 |

### 2.8 LoRA 配置

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--enable-lora` | `False` | 启用 LoRA | 多租户适配器服务 |
| `--lora-paths` | `None` | LoRA 适配器路径列表 | 指定加载的适配器 |
| `--max-lora-rank` | `None`（自动推导） | 最大 LoRA 秩 | 限制适配器大小 |
| `--max-loras-per-batch` | `8` | 每批最大 LoRA 数 | 控制并发适配器数量 |
| `--max-loaded-loras` | `None` | CPU 内存中最大 LoRA 数 | 热切换缓存 |
| `--lora-eviction-policy` | `lru` | LoRA 淘汰策略 | 缓存淘汰 |
| `--lora-backend` | `csgmv` | LoRA 内核后端 | 选择 LoRA 计算内核 |
| `--lora-drain-wait-threshold` | `0.0` | LoRA 排水等待阈值 | 防止适配器垄断 |

### 2.9 内核后端

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--attention-backend` | `None` | 注意力内核后端 | 选择注意力实现 |
| `--prefill-attention-backend` | `None` | 预填充注意力后端 | 独立指定预填充内核 |
| `--decode-attention-backend` | `None` | 解码注意力后端 | 独立指定解码内核 |
| `--sampling-backend` | `None` | 采样后端：`flashinfer`/`pytorch` 等 | 选择采样实现 |
| `--grammar-backend` | `None` | 语法约束后端：`xgrammar`/`outlines`/`llguidance` | 约束解码选择 |
| `--fp8-gemm-runner-backend` | `auto` | FP8 GEMM 内核后端 | 选择 FP8 计算内核 |
| `--fp4-gemm-runner-backend` | `auto` | FP4 GEMM 内核后端 | 选择 FP4 计算内核 |
| `--mamba-backend` | `triton` | Mamba 内核后端 | SSM 模型推理 |
| `--moe-runner-backend` | `auto` | MoE 内核后端 | 选择 MoE 计算内核 |
| `--moe-a2a-backend` | `none` | MoE All2All 后端 | 选择 MoE 通信内核 |

### 2.10 推测解码

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--speculative-algorithm` | `None` | 推测算法：`EAGLE`/`STANDALONE`/`NGRAM` | 选择推测策略 |
| `--speculative-draft-model-path` | `None` | Draft 模型路径 | 基于 draft model 的推测 |
| `--speculative-num-steps` | `None` | 推测步数 | 控制推测长度 |
| `--speculative-eagle-topk` | `None` | EAGLE top-k | EAGLE 算法参数 |
| `--speculative-num-draft-tokens` | `None` | 推测 token 数 | 控制每次推测的 token 数 |
| `--speculative-accept-threshold-single` | `1.0` | 单 token 接受阈值 | 调整推测接受率 |
| `--speculative-adaptive` | `False` | 自适应推测解码 | 动态调整推测策略 |
| `--speculative-ngram-min-bfs-breadth` | `1` | Ngram BFS 最小宽度 | Ngram 推测调参 |
| `--speculative-ngram-max-bfs-breadth` | `10` | Ngram BFS 最大宽度 | Ngram 推测调参 |
| `--enable-multi-layer-eagle` | `False` | 多层 EAGLE | EAGLE 高级模式 |

### 2.11 专家并行

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--ep-size` | `1` | 专家并行度 | MoE 模型跨 GPU 分配专家 |
| `--moe-a2a-backend` | `none` | MoE All2All 后端：`deepep`/`mooncake`/`nixl`/`mori` 等 | 选择 MoE 通信后端 |
| `--moe-runner-backend` | `auto` | MoE 计算内核后端 | 选择 MoE 计算实现 |
| `--deepep-mode` | `auto` | DeepEP 模式：`auto`/`normal`/`low_latency` | DeepEP 调度模式 |
| `--enable-eplb` | `False` | 启用专家并行负载均衡 | MoE 负载均衡 |
| `--ep-num-redundant-experts` | `0` | 冗余专家数 | 提高容错和均衡 |
| `--init-expert-location` | `trivial` | 初始专家放置策略 | 专家初始化布局 |

### 2.12 层级缓存

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--enable-hierarchical-cache` | `False` | 启用层级缓存（GPU+CPU/SSD） | 扩展 KV 缓存容量 |
| `--hicache-ratio` | `2.0` | 层级缓存与 GPU 缓存比率 | 控制扩展比例 |
| `--hicache-write-policy` | `write_through` | 写策略 | 缓存一致性控制 |
| `--hicache-io-backend` | `kernel` | I/O 后端 | 选择 I/O 实现 |
| `--hicache-storage-backend` | `None` | 存储后端 | 选择存储实现 |

### 2.13 PD 分离

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--disaggregation-mode` | `null` | 分离模式：`null`/`prefill`/`decode` | PD 分离部署 |
| `--disaggregation-transfer-backend` | `mooncake` | 传输后端：`mooncake`/`nixl`/`ascend`/`fake`/`mori` | KV 传输实现 |
| `--disaggregation-bootstrap-port` | `8998` | 引导端口 | 分离服务发现 |
| `--disaggregation-ib-device` | `None` | InfiniBand 设备 | 高速 KV 传输 |

### 2.14 卸载配置

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--cpu-offload-gb` | `0` | CPU 卸载空间（GB） | 显存不足时利用 CPU |
| `--offload-group-size` | `-1` | 分组卸载组大小 | 分层卸载策略 |
| `--offload-num-in-group` | `1` | 每组卸载数 | 控制卸载粒度 |
| `--offload-prefetch-step` | `1` | 预取步长 | 隐藏延迟 |
| `--offload-mode` | `cpu` | 卸载模式 | 选择卸载目标 |

### 2.15 Mamba 缓存

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--max-mamba-cache-size` | `None` | Mamba SSM 缓存最大大小 | SSM 模型缓存控制 |
| `--mamba-ssm-dtype` | `None` | Mamba SSM 数据类型 | SSM 缓存精度控制 |
| `--mamba-full-memory-ratio` | `0.9` | Mamba 全内存比例 | SSM 缓存内存分配 |

### 2.16 多节点分布式

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--dist-init-addr` | `None` | 分布式初始化地址 | 多节点部署 |
| `--nnodes` | `1` | 节点数 | 多节点部署 |
| `--node-rank` | `0` | 节点排名 | 多节点标识 |

### 2.17 其他重要参数

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--disable-radix-cache` | `False` | 禁用 Radix 缓存 | 调试或特殊场景 |
| `--cuda-graph-max-bs` | 自动 | CUDA graph 最大批次 | 控制 CUDA graph 捕获范围 |
| `--disable-cuda-graph` | `False` | 禁用 CUDA graph | 调试或不兼容场景 |
| `--enable-torch-compile` | `False` | 启用 torch.compile | 编译优化 |
| `--disable-overlap-schedule` | `False` | 禁用重叠调度 | 调试 |
| `--enable-dp-attention` | `False` | 启用 DP attention | DP 注意力模式 |
| `--enable-two-batch-overlap` | `False` | 启用双批次重叠 | 提高吞吐 |
| `--triton-attention-reduce-in-fp32` | `False` | Triton attention FP32 累加 | 提高精度 |
| `--num-continuous-decode-steps` | `1` | 连续解码步数 | 减少调度开销 |
| `--enable-lmcache` | `False` | 启用 LMCache 集成 | 外部 KV 缓存 |
| `--enable-deterministic-inference` | `False` | 确定性推理 | 可复现推理 |
| `--enable-custom-logit-processor` | `False` | 自定义 logit 处理器 | 自定义采样逻辑 |

---

## 3. 参数对比分析

### 3.1 核心参数对比表

以下为功能对应关系的核心参数对比：

| 功能领域 | vLLM 参数 | SGLang 参数                                                                               | 差异说明                  |  备注 |
|----------|-----------|-----------------------------------------------------------------------------------------|-----------------------|------|
| **模型路径** | `--model` | `--model-path` / `--model`                                                              | 命名不同，功能相同             |
| **分词器路径** | `--tokenizer` | `--tokenizer-path`                                                                      | 命名不同，功能相同             |
| **分词器模式** | `--tokenizer-mode` (`auto`/`hf`/`slow`/`mistral`/`deepseek_v32`/`deepseek_v4`) | `--tokenizer-mode` (`auto`/`slow`)                                                      | vLLM 支持更多模式           |
| **数据类型** | `--dtype` (`auto`/`float16`/`bfloat16`/`float32`/`half`) | `--dtype` (`auto`/`half`/`float16`/`bfloat16`/`float`/`float32`)                        | 功能相同，选项略有不同           |
| **信任远程代码** | `--trust-remote-code` | `--trust-remote-code`                                                                   | 相同（允许加载模型的自定义代码）      | done |
| **上下文长度** | `--max-model-len` | `--context-length`                                                                      | 命名不同，功能相同             | done |
| **量化方法** | `--quantization` | `--quantization`                                                                        | 相同，SGLang 多 `mlx_q4`/`mlx_q8`/`unquant` |
| **KV 缓存类型** | `--kv-cache-dtype` | `--kv-cache-dtype`                                                                      | 相同，vLLM 支持 `nvfp4`/`turboquant` 等更多类型 |
| **张量并行** | `--tensor-parallel-size` / `-tp` | `--tensor-parallel-size` / `--tp-size` / `--tp` |  | done |
| **流水线并行** | `--pipeline-parallel-size` / `-pp` | `--pipeline-parallel-size` / `--pp-size` / `-pp` |  | done |
| **数据并行** | `--data-parallel-size` / `-dp` | `--data-parallel-size` / `--dp-size` / `-pp`  |  | done |
| **GPU 显存** | `--gpu-memory-utilization` (0.92) | `--mem-fraction-static` (自动)                                                            | 命名不同，默认值不同（vLLM 0.92 vs SGLang 自动计算） |
| **分块预填充** | `--enable-chunked-prefill` | `--chunked-prefill-size`                                                                | vLLM 是布尔开关，SGLang 直接指定大小 |
| **前缀缓存** | `--enable-prefix-caching` | `--disable-radix-cache`（默认启用）                                                           | 逻辑相反，SGLang 默认启用 RadixAttention |
| **调度策略** | `--scheduling-policy` (`fcfs`/`priority`) | `--schedule-policy` (`fcfs`/`lpm`/`random`/`priority`/`dfs-weight`/`lof`/`routing-key`) | SGLang 调度策略更丰富        |
| **流式间隔** | `--stream-interval` | `--stream-interval`                                                                     | 相同                    |
| **随机种子** | `--seed` | `--random-seed`                                                                         | 命名不同                  |
| **服务端口** | (OpenAI server 另设) | `--port` (30000)                                                                        | vLLM 在 serve 子命令中设置   |
| **最大序列数** | `--max-num-seqs` | `--max-running-requests`                                                                | 命名不同                  |
| **最大批处理 token** | `--max-num-batched-tokens` | `--max-total-tokens`                                                                    | 语义不同：vLLM 是单次迭代限制，SGLang 是全局 token 池 |
| **LoRA 启用** | `--enable-lora` | `--enable-lora`                                                                         | 相同                    |
| **LoRA 最大秩** | `--max-lora-rank` (16) | `--max-lora-rank` (自动推导)                                                                | 默认值不同                 |
| **卸载到 CPU** | `--cpu-offload-gb` | `--cpu-offload-gb`                                                                      | 相同                    |
| **推测解码** | `--speculative-config` (JSON 配置) | `--speculative-algorithm` + 多个独立参数                                                      | 配置方式不同                |
| **推理解析器** | `--reasoning-parser` | `--reasoning-parser`                                                                    | 相同，SGLang 支持更多预置解析器   |
| **MFU 指标** | `--enable-mfu-metrics` | `--enable-mfu-metrics`                                                                  | 相同                    |
| **OTLP 追踪** | `--otlp-traces-endpoint` | `--otlp-traces-endpoint`                                                                | 相同                    |

- vllm中max-model-len
  - VLLM会为每个请求预留足够容纳max_model_len长度的kv cache。该值越大，每个请求消耗的kv cache内存就越多，即时实际上下文很短，仍然会按照最大长度来预留内存。
  - 较小的 max_model_len 能容纳更多并发，从而提升单位时间处理的请求数 (requests/s)，较大的 max_model_len 允许处理长文本，但代价是并发降低。因此需要在吞吐量和“单个请求能处理多长文本”之间做权衡。
  - 模型一次能处理的最大 token（输入 + 输出） 总数。
- tp
  - 把一个transformer层内的张量切分到多个GPU上同时计算。特点如下：
    - 切分后GPU之间通信频繁，因此需要GPU之间有告诉互联，例如nvlink,nvswitch等。
    - 节省显存。每个GPU只存储模型的一部分权重，因此可以容纳超大模型。
### 3.2 相同点总结

#### 3.2.1 完全相同的参数

以下参数在两个框架中名称、功能、语义基本一致：

- `--trust-remote-code` — 信任远程代码
- `--quantization` — 量化方法
- `--kv-cache-dtype` — KV 缓存数据类型
- `--load-format` — 模型权重加载格式
- `--download-dir` — 模型下载目录
- `--revision` — 模型版本
- `--enable-lora` — 启用 LoRA
- `--cpu-offload-gb` — CPU 卸载大小
- `--stream-interval` — 流式输出间隔
- `--enable-mfu-metrics` — MFU 指标开关
- `--otlp-traces-endpoint` — OTLP 追踪端点
- `--reasoning-parser` — 推理内容解析器
- `--served-model-name` — 对外模型名

#### 3.2.2 功能相同但命名不同的参数

| vLLM | SGLang | 功能 |
|------|--------|------|
| `--model` | `--model-path` | 模型路径 |
| `--tokenizer` | `--tokenizer-path` | 分词器路径 |
| `--max-model-len` | `--context-length` | 上下文长度 |
| `--gpu-memory-utilization` | `--mem-fraction-static` | GPU 显存分配比例 |
| `--seed` | `--random-seed` | 随机种子 |
| `--max-num-seqs` | `--max-running-requests` | 最大并发序列数 |
| `--scheduling-policy` | `--schedule-policy` | 调度策略 |
| `--enable-prefix-caching` | (默认启用，用 `--disable-radix-cache` 关闭) | 前缀/前缀缓存 |
| `--tensor-parallel-size` / `-tp` | `--tp-size` | 张量并行度短参数名 |
| `--pipeline-parallel-size` / `-pp` | `--pp-size` | 流水线并行度短参数名 |
| `--data-parallel-size` / `-dp` | `--dp-size` | 数据并行度短参数名 |
| `--expert-parallel` / `-ep` | `--ep-size` | 专家并行度 |

#### 3.2.3 相同的设计理念

1. **PagedAttention**：两者均使用分页注意力机制管理 KV 缓存
2. **连续批处理**：均支持 iteration-level 调度
3. **分块预填充**：均支持将长 prompt 分块处理，与 decode 混合
4. **张量/流水线/数据并行**：均支持三种并行方式
5. **推测解码**：均支持 EAGLE、Medusa、Ngram 等推测策略
6. **LoRA 热切换**：均支持运行时动态加载/卸载 LoRA
7. **量化支持**：均支持 FP8、GPTQ、AWQ、Marlin 等量化格式
8. **KV 缓存卸载**：均支持将 KV 缓存卸载到 CPU
9. **PD 分离**：均支持 Prefill-Decode 分离部署
10. **OpenAI 兼容 API**：均提供 OpenAI 兼容的 API 服务

### 3.3 差异点分析

#### 3.3.1 vLLM 独有的特性

| 特性 | 说明 |
|------|------|
| `--model-impl` 支持 `terratorh` | 额外支持 TerraTorch 模型实现 |
| `--logprobs-mode` | 支持 raw/processed logprobs 模式选择 |
| `--use-fp64-gumbel` | FP64 Gumbel 噪声采样 |
| `--decode-context-parallel-size` / `-dcp` | 独立的解码上下文并行度参数 |
| `--prefill-context-parallel-size` / `-pcp` | 独立的预填充上下文并行度参数 |
| `--data-parallel-hybrid-lb` / `--data-parallel-external-lb` | 更丰富的 DP 负载均衡模式 |
| `--enable-eplb` + `--eplb-config` | 更细粒度的专家负载均衡配置 |
| `--all2all-backend` 更多选项 | 支持 `pplx`/`nixl_ep`/`mori` 等更多 All2All 后端 |
| `--kv-offloading-size` + `--kv-offloading-backend` | KV 缓存卸载为独立参数（SGLang 整合在层级缓存中）|
| `--prefix-caching-hash-algo` | 可选哈希算法（SHA256/xxHash/CBOR） |
| `--safetensors-load-strategy` | 细粒度 safetensors 加载策略 |
| `--offload-backend` 支持 `uva` | UVA 零拷贝卸载方式 |
| `--numa-bind` + NUMA 绑定 | 完整的 NUMA 拓扑感知配置 |
| `--renderer-num-workers` | 异步渲染器线程池 |
| `--enable-sleep-mode` + `--enable-cumem-allocator` | GPU 睡眠模式与自定义内存分配器 |
| `--kv-sharing-fast-prefill` | KV 共享快速预填充（实验性）|
| `--optimization-level` / `-O0`~`-O3` | 编译优化级别 |
| `--performance-mode` | 性能模式：`balanced`/`interactivity`/`throughput` |
| `--kv-cache-dtype` 更多选项 | 支持 `nvfp4`/`turboquant` 等更多 KV 缓存类型 |
| `--runner` 参数 | 显式选择模型运行器类型 |
| `--convert` 参数 | 模型转换适配器 |

#### 3.3.2 SGLang 独有的特性

| 特性 | 说明 |
|------|------|
| `--tokenizer-backend` | 支持 `fastokens` 分词器后端 |
| `--tokenizer-worker-num` / `--detokenizer-worker-num` | 分词/反分词器 worker 并行度 |
| `--enable-priority-scheduling` + `--schedule-low-priority-values-first` | 更丰富的优先级调度控制 |
| `--priority-scheduling-preemption-threshold` | 优先级抢占阈值 |
| `--lora-drain-wait-threshold` | LoRA 适配器排水阈值 |
| `--lora-backend` 多选项 | 支持 `csgmv`/`triton`/`ascend`/`torch_native` |
| `--grammar-backend` | 独立的语法约束后端选择（`xgrammar`/`outlines`/`llguidance`）|
| `--sampling-backend` | 独立的采样后端选择 |
| `--radix-eviction-policy` | Radix 缓存淘汰策略（`lru`/`lfu`/`slru`/`priority`）|
| `--enable-hierarchical-cache` + 完整层级缓存 | GPU+CPU/SSD 层级 KV 缓存（独立参数体系）|
| `--enable-lmcache` | LMCache 外部缓存集成 |
| `--disaggregation-mode` + 完整 PD 分离参数 | 显式的 PD 分离模式与传输后端 |
| `--tool-call-parser` | 工具调用解析器（独立参数）|
| `--tool-server` | 工具服务器集成 |
| `--sampling-defaults` | 采样默认值来源选择 |
| `--enable-cache-report` | 缓存命中报告 |
| `--admin-api-key` | 管理员 API 密钥 |
| `--grpc-mode` | gRPC 服务模式 |
| `--ssl-*` 系列 | 完整 SSL/TLS 配置 |
| `--watchdog-timeout` + `--soft-watchdog-timeout` | 看门狗超时保护 |
| `--base-gpu-id` + `--gpu-id-step` | GPU ID 分配控制 |
| `--sleep-on-idle` | 空闲时 GPU 休眠 |
| `--schedule-conservativeness` | 调度保守程度参数 |
| `--enable-prefill-delayer` + 完整参数 | 预填充延迟器 |
| `--enable-dynamic-chunking` | 动态分块大小 |
| `--pp-async-batch-depth` | PP 异步批次深度 |
| `--enable-dp-attention` | DP Attention 模式 |
| `--enable-two-batch-overlap` / `--enable-single-batch-overlap` | 批次重叠调度 |
| `--enable-torch-compile` | 独立的 torch.compile 开关 |
| `--num-continuous-decode-steps` | 连续解码步数 |
| `--enable-deterministic-inference` | 确定性推理模式 |
| `--enable-custom-logit-processor` | 自定义 logit 处理器 |
| `--crash-dump-folder` | 崩溃转储目录 |
| `--export-metrics-to-file` | 指标导出到文件 |
| `--enable-mis` | Multi-Item Scoring 优化 |
| `--enable-hisparse` | 层级稀疏注意力 |
| `--model-checksum` | 模型文件校验 |
| `--incremental-streaming-output` | 增量流式输出 |
| `--kt-*` 系列 (KTransformers) | KTransformers CPU/GPU 混合推理 |
| `--dllm-*` 系列 | Diffusion LLM 支持 |
| `--forward-hooks` | 前向钩子配置 |
| `--msprobe-dump-config` | msProbe 调试工具 |
| `--enable-pdmux` | PD-Multiplexing 模式 |

#### 3.3.3 设计哲学差异

| 维度 | vLLM | SGLang |
|------|------|--------|
| **默认 GPU 显存占用** | 0.92（固定值） | 自动计算（基于模型和可用显存） |
| **前缀缓存** | 默认关闭，需手动开启 | 默认启用 RadixAttention，需手动关闭 |
| **调度策略** | 2种（`fcfs`/`priority`） | 7种（`fcfs`/`lpm`/`random`/`priority`/`dfs-weight`/`lof`/`routing-key`） |
| **推测解码配置** | 单个 JSON 配置块 | 多个独立参数，更细粒度 |
| **内核后端选择** | 全局配置或 JSON 配置块 | 独立参数（attention/sampling/grammar/GEMM 各自独立） |
| **PD 分离** | 通过 `--kv-transfer-config` JSON 配置 | 独立参数体系（`--disaggregation-*`） |
| **层级缓存** | 通过 `--kv-offloading-*` 和 `--lmcache` | 完整独立参数体系（`--enable-hierarchical-cache` + `--hicache-*`） |
| **多模态** | 丰富独立参数 | 相对简单，按需扩展 |
| **量化配置** | 通过 `--quantization-config` JSON | 独立参数（`--modelopt-*`）+ `--quantization` |
| **服务安全** | 通过 OpenAI server 参数 | 内置 `--api-key`/`--admin-api-key`/SSL/TLS |
| **调试与诊断** | NVTX 追踪、详细追踪 | 看门狗、崩溃转储、调试张量转储 |
| **参数组织** | 按配置类分组（ModelConfig、CacheConfig 等） | 按功能分类，更扁平化 |

---

## 4. 总结与选型建议

### 4.1 参数规模对比

| 指标 | vLLM | SGLang |
|------|------|--------|
| 核心参数数量 | ~125 | ~200+ |
| 内核后端参数 | 较少（集中在配置块） | 较多（独立参数） |
| 调度参数 | ~10 | ~20+ |
| 推测解码参数 | ~5（配置块式） | ~20+（独立参数式） |
| PD 分离参数 | ~3（配置块式） | ~10+（独立参数式） |
| LoRA 参数 | ~10 | ~14 |

### 4.2 选型建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| **快速上手** | vLLM | 参数更少，默认配置即可工作 |
| **精细调优** | SGLang | 内核后端、调度策略独立可配 |
| **多模态推理** | vLLM | 多模态参数更完善 |
| **MoE 大规模部署** | 看具体需求 | 两者均支持 EP/DP/DeepEP，各有优势 |
| **PD 分离部署** | SGLang | 独立参数体系更直观 |
| **生产监控** | SGLang | 内置看门狗、崩溃转储、指标导出 |
| **安全性要求高** | SGLang | 内置 API key、Admin key、SSL/TLS |
| **灵活调度** | SGLang | 7种调度策略、优先级抢占、预填充延迟 |
| **推测解码** | SGLang | 独立参数更灵活，Ngram 推测更丰富 |
| **CPU/GPU 混合推理** | SGLang | KTransformers 支持 |
| **Diffusion LLM** | SGLang | 独家支持 |
| **NUMA 感知** | vLLM | 完整 NUMA 绑定配置 |
| **编译优化** | vLLM | `-O0`~`-O3` 优化级别 + `--performance-mode` |
| **KV 缓存卸载** | vLLM | 独立参数，支持 `lmcache` 后端 |

### 4.3 关键差异总结

1. **配置风格**：vLLM 偏向"配置块"式（JSON），SGLang 偏向"独立参数"式
2. **默认行为**：SGLang 默认启用前缀缓存（RadixAttention），vLLM 需手动开启
3. **显存管理**：vLLM 固定比例（0.92），SGLang 自动计算
4. **内核控制**：SGLang 提供更细粒度的内核后端选择
5. **安全特性**：SGLang 内置更完善的安全配置
6. **可观测性**：SGLang 内置更多运维友好特性
7. **模型覆盖**：vLLM 在多模态方面参数更完善

---

> **注意**：本文档基于当前最新版本的官方文档整理，两个框架迭代迅速，参数可能随版本更新变化。建议参考官方文档获取最新信息：
> - vLLM：https://docs.vllm.ai/en/latest/configuration/engine_args/
> - SGLang：https://docs.sglang.io/docs/advanced_features/server_arguments
