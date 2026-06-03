# SGLang 总体架构文档

> 本文档基于 SGLang 项目源码自动生成，旨在提供项目总体架构的全景视图。

---

## 1. 项目概述

SGLang (SGLang Runtime) 是一个高性能的大语言模型 (LLM) 推理服务框架，专注于低延迟、高吞吐量的模型部署。其核心特性包括：

- **RadixAttention**：基于 Radix Tree 的 KV Cache 共享与复用机制，显著提升前缀复用场景下的推理效率
- **连续批处理 (Continuous Batching)**：动态调度请求，实现 prefill 与 decode 的交错执行
- **多级并行**：支持 Tensor Parallelism (TP)、Pipeline Parallelism (PP)、Data Parallelism (DP)、Expert Parallelism (EP)
- **推测解码 (Speculative Decoding)**：支持 EAGLE、N-gram、MTP 等多种草稿模型策略
- **多模态支持**：图像、视频、音频等多模态输入处理
- **分离式推理 (Disaggregated Serving)**：Prefill 与 Decode 分离部署
- **LoRA 支持**：多 LoRA 适配器动态加载与推理
- **多硬件后端**：CUDA、ROCm、XPU、NPU、MPS 等

---

## 2. 顶层目录结构

```
sglang/
├── python/sglang/          # 核心 Python 包
│   ├── srt/                # SGLang Runtime (推理引擎核心)
│   ├── lang/               # SGLang 编程语言前端 (DSL)
│   ├── cli/                # 命令行接口
│   ├── benchmark/          # 基准测试工具
│   ├── jit_kernel/         # JIT 编译的 CUDA Kernel
│   ├── eval/               # 评测工具
│   └── test/               # 单元测试
├── sgl-kernel/             # AOT 预编译 CUDA/C++ Kernel 库
├── sgl-model-gateway/      # Rust 编写的模型网关 (负载均衡/路由)
├── rust/                   # Rust 组件 (gRPC 服务等)
├── benchmark/              # 各类基准测试脚本
├── test/                   # 集成测试 / CI 测试
├── examples/               # 示例代码
├── docs/                   # 文档
├── docker/                 # Docker 构建文件
├── scripts/                # 运维与部署脚本
└── 3rdparty/               # 第三方依赖
```

---

## 3. 核心架构：多进程协作模型

SGLang 采用**多进程架构**，各进程通过 **ZMQ** 进行通信。核心进程包括：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client (HTTP/gRPC/Python API)                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                    HTTP Server / Engine (主进程)                     │
│         (FastAPI + Uvicorn / Python Engine API)                     │
│    OpenAI API / Anthropic API / Ollama API 兼容                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ ZMQ
┌──────────────────────────▼──────────────────────────────────────────┐
│                   TokenizerManager (进程)                            │
│    - 文本 Tokenize / 多模态预处理                                     │
│    - 请求分发 (DP 路由)                                              │
│    - 会话管理                                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ ZMQ
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐
│ DP Controller  │ │ DP Controller │ │ DP Controller │  (Data Parallel)
│    (进程)      │ │   (进程)      │ │   (进程)      │
└───────┬────────┘ └──────┬───────┘ └──────┬───────┘
        │                 │                 │
┌───────▼─────────────────▼─────────────────▼───────┐
│              Scheduler (进程)                       │
│    - 请求调度与批处理                                 │
│    - KV Cache 管理 (Radix Tree)                     │
│    - 内存分配与回收                                   │
│    - 约束解码 (Grammar)                              │
│    - 推测解码协调                                     │
└───────────────────────────┬────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────┐
│            ModelRunner / TpWorker (进程)             │
│    - 模型前向推理                                     │
│    - GPU 计算                                        │
│    - CUDA Graph 管理                                 │
│    - 分布式通信 (TP/PP)                              │
└───────────────────────────┬────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────┐
│           DetokenizerManager (进程)                  │
│    - Token ID → 文本解码                              │
│    - 流式输出拼接                                     │
└────────────────────────────────────────────────────┘
```

### 3.1 进程间通信 (IPC)

进程间通过 **ZMQ** 进行消息传递，核心数据结构定义在 `io_struct.py` 中：

- `GenerateReqInput` / `TokenizedGenerateReqInput`：生成请求
- `EmbeddingReqInput`：嵌入请求
- `BatchStrOutput` / `BatchTokenIDOutput`：输出结果
- `UpdateWeightFromDiskReqInput`：权重热更新
- `LoadLoRAAdapterReqInput` / `UnloadLoRAAdapterReqInput`：LoRA 加载/卸载

---

## 4. SGLang Runtime (SRT) 核心模块详解

### 4.1 入口层 (`entrypoints/`)

提供多种 API 接口：

| 子模块 | 说明 |
|--------|------|
| `http_server.py` | FastAPI HTTP 服务器，支持 OpenAI/Anthropic/Ollama 协议 |
| `engine.py` | Python API 入口 (Engine 类)，无需 HTTP 即可直接调用 |
| `openai/` | OpenAI 兼容 API 实现 (Chat/Completions/Embedding/Rerank 等) |
| `anthropic/` | Anthropic Messages API 兼容实现 |
| `ollama/` | Ollama API 兼容实现 |
| `grpc_server.py` | gRPC 服务端 |

**请求处理流程**：
1. HTTP Server 接收请求 → 解析为内部请求对象
2. 通过 TokenizerManager 进行 Tokenize
3. 通过 ZMQ 发送到 Scheduler
4. Scheduler 调度 → ModelRunner 推理
5. 结果返回 DetokenizerManager → 流式/批量返回客户端

### 4.2 调度层 (`managers/`)

调度层是 SGLang 的核心，负责请求的全生命周期管理。

#### 4.2.1 Scheduler (`scheduler.py`)

Scheduler 是推理调度的核心控制器，约 4000 行代码，主要职责：

- **请求接收与排队**：从 TokenizerManager 接收 tokenized 请求
- **批处理调度**：将请求组装为 ScheduleBatch
- **KV Cache 管理**：通过 RadixAttention 进行前缀匹配与缓存分配
- **Prefill/Decode 调度**：决定何时执行 prefill（首次计算）与 decode（逐 token 生成）
- **推测解码协调**：管理 draft model 与 target model 的协作
- **内存管理**：监控与回收 KV Cache 内存

核心数据流：
```
ScheduleBatch (CPU 侧) → ForwardBatch (GPU 侧)
```

#### 4.2.2 TokenizerManager (`tokenizer_manager.py`)

- 文本 Tokenize（使用 HuggingFace Tokenizer）
- 多模态数据预处理（图像/视频/音频编码）
- 请求路由（Data Parallel 路由决策）
- LoRA 适配器管理
- 会话 (Session) 管理

#### 4.2.3 DetokenizerManager (`detokenizer_manager.py`)

- 将 token ID 解码为文本
- 流式输出增量拼接
- 停止词检测
- 请求完成通知

#### 4.2.4 DataParallelController (`data_parallel_controller.py`)

- DP 实例间的请求负载均衡
- 支持 round-robin、cache-aware 等路由策略
- DP 实例健康监控

#### 4.2.5 TpWorker (`tp_worker.py`)

- Tensor Parallel Worker 的抽象
- 管理模型权重加载/更新
- 协调分布式推理中的通信

#### 4.2.6 ScheduleBatch (`schedule_batch.py`)

核心批处理数据结构，约 2700 行，包含：

- `Req`：单个请求的状态（token IDs、采样参数、完成状态等）
- `ScheduleBatch`：一批请求的集合，包含调度所需的所有信息
- 与 KV Cache 的交互接口

#### 4.2.7 SchedulePolicy (`schedule_policy.py`)

请求调度策略实现：
- 请求优先级排序
- Prefill/Decode 比例控制
- KV Cache 压力感知调度

### 4.3 模型执行层 (`model_executor/`)

| 文件 | 说明 |
|------|------|
| `model_runner.py` | ModelRunner 核心，管理模型加载、前向推理、CUDA Graph |
| `forward_batch_info.py` | ForwardBatch 定义，GPU 侧推理所需的张量数据 |
| `forward_context.py` | 前向推理上下文管理 |
| `cuda_graph_runner.py` | CUDA Graph 捕获与重放，减少 kernel launch 开销 |
| `piecewise_cuda_graph_runner.py` | 分段 CUDA Graph（针对长序列等场景） |
| `breakable_cuda_graph_runner.py` | 可中断 CUDA Graph |
| `hook_manager.py` | 前向钩子管理 |

**ModelRunner 核心流程**：
1. 初始化：加载模型权重 → 初始化 KV Cache → 捕获 CUDA Graph
2. 推理：接收 ForwardBatch → 执行模型前向 → 返回 logits/hidden states
3. 采样：logits → 采样策略 → 生成 token

### 4.4 KV Cache 与内存管理 (`mem_cache/`)

SGLang 的 KV Cache 管理是其核心创新之一，基于 **Radix Tree** 实现。

#### 4.4.1 两级内存池

```
┌─────────────────────────────────────────────────────┐
│              ReqToTokenPool                          │
│   请求 → Token 位置映射 (request_id → token_indices) │
├─────────────────────────────────────────────────────┤
│           TokenToKVPoolAllocator                     │
│   Token 位置 → KV Cache 物理内存分配                  │
├─────────────────────────────────────────────────────┤
│              KV Cache (GPU Tensor)                   │
│   实际的 Key/Value 缓存张量                           │
└─────────────────────────────────────────────────────┘
```

#### 4.4.2 RadixAttention 前缀缓存

| 文件 | 说明 |
|------|------|
| `radix_cache.py` | Radix Tree 实现，支持前缀匹配与引用计数 |
| `hiradix_cache.py` | HiRadix Cache（分层 Radix Cache） |
| `chunk_cache.py` | 基于分块的前缀缓存 |
| `swa_radix_cache.py` | Sliding Window Attention Radix Cache |
| `unified_radix_cache.py` | 统一 Radix Cache（整合多种缓存策略） |
| `mamba_radix_cache.py` | Mamba 状态的 Radix Cache |
| `hi_mamba_radix_cache.py` | 分层 Mamba Radix Cache |

#### 4.4.3 其他内存管理组件

| 子模块 | 说明 |
|--------|------|
| `allocator/` | 内存分配器（Base/Paged/Token 三种策略） |
| `memory_pool.py` | 核心内存池实现 (ReqToTokenPool, TokenToKVPool) |
| `memory_pool_host.py` | Host 侧内存池 |
| `hybrid_cache/` | 混合缓存控制器（Attention + Linear Attention） |
| `sparsity/` | 稀疏注意力内存管理 |
| `storage/` | 外部存储后端 (Mooncake/LMCache/HF3FS/NIXL/AIBrix/EIC/SIMM) |
| `multimodal_cache.py` | 多模态数据缓存 |
| `evict_policy.py` | 缓存驱逐策略 |

### 4.5 模型层 (`layers/`)

SGLang 将 Transformer 模型的各层抽象为可组合的模块。

#### 4.5.1 注意力层 (`layers/attention/`)

SGLang 支持多种注意力后端，通过 `AttentionBackend` 抽象接口统一管理：

| 后端 | 说明 |
|------|------|
| `flashinfer_backend.py` | FlashInfer 注意力 (默认) |
| `flashattention_backend.py` | FlashAttention 2 |
| `triton_backend.py` | Triton 实现的注意力 |
| `flashmla_backend.py` | FlashMLA (DeepSeek MLA) |
| `flashinfer_mla_backend.py` | FlashInfer MLA 后端 |
| `trtllm_mha_backend.py` | TensorRT-LLM MHA 后端 |
| `trtllm_mla_backend.py` | TensorRT-LLM MLA 后端 |
| `dsa_backend.py` | DSA (DeepSeek Attention) 后端 |
| `cutlass_mla_backend.py` | CUTLASS MLA 后端 |
| `nsa_backend.py` | Native Sparse Attention |
| `wave_backend.py` | Wave Attention 后端 |
| `hybrid_attn_backend.py` | 混合注意力（Full + Linear） |
| `hybrid_linear_attn_backend.py` | 混合线性注意力 |
| `dual_chunk_flashattention_backend.py` | 双块 FlashAttention |
| `torch_native_backend.py` | PyTorch 原生注意力 |
| `intel_amx_backend.py` | Intel AMX 后端 (CPU) |
| `xpu_backend.py` | Intel XPU 后端 |
| `vision.py` | 视觉注意力 |

#### 4.5.2 MoE 层 (`layers/moe/`)

| 文件 | 说明 |
|------|------|
| `fused_moe_triton/` | Triton 实现的 Fused MoE |
| `cutlass_moe.py` | CUTLASS MoE 实现 |
| `ep_moe/` | Expert Parallelism MoE |
| `mega_moe.py` | Mega MoE（多类型 MoE 融合） |
| `router.py` | MoE 路由器 |
| `topk.py` | Top-K 选择 |
| `token_dispatcher/` | Token 分发器 (EP 场景) |

#### 4.5.3 量化层 (`layers/quantization/`)

| 量化方案 | 说明 |
|----------|------|
| `fp8.py` | FP8 量化 |
| `awq/` | AWQ 量化 |
| `gptq/` | GPTQ 量化 |
| `marlin_utils.py` | Marlin 量化内核 |
| `bitsandbytes.py` | BitsAndBytes 量化 |
| `gguf.py` | GGUF 格式 |
| `compressed_tensors/` | Compressed Tensors 格式 |
| `mxfp4.py` | MX-FP4 量化 |
| `modelopt_quant.py` | NVIDIA ModelOpt 量化 |
| `petit.py` | Petit 量化 |
| `w4afp8.py` | W4A-FP8 混合量化 |
| `w8a8_fp8.py` / `w8a8_int8.py` | W8A8 量化 |

#### 4.5.4 其他重要层

| 文件 | 说明 |
|------|------|
| `linear.py` | 线性层（支持 Tensor Parallel） |
| `radix_attention.py` | RadixAttention 封装层 |
| `sampler.py` | 采样器 |
| `logits_processor.py` | Logits 处理器 |
| `rotary_embedding/` | RoPE 旋转位置编码 |
| `layernorm.py` | LayerNorm / RMSNorm |
| `vocab_parallel_embedding.py` | 词表并行嵌入层 |
| `pooler.py` | 嵌入池化层 |
| `conv.py` | 卷积层 (CV 模型) |
| `multimodal.py` | 多模态投影层 |
| `model_parallel.py` | 模型并行工具 |
| `dp_attention.py` | Data Parallel Attention |

### 4.6 模型库 (`models/`)

SGLang 支持 **190+** 种模型架构，主要包括：

| 模型家族 | 代表模型 |
|----------|----------|
| LLaMA | LLaMA, LLaMA-2, LLaMA-3, LLaMA-4 |
| Qwen | Qwen2, Qwen2-VL, Qwen2.5-VL, Qwen3, Qwen3-MoE, Qwen3-VL |
| DeepSeek | DeepSeek-V2/V3, DeepSeek-V4, DeepSeek-VL2 |
| Gemma | Gemma2, Gemma3, Gemma4 |
| Mistral | Mistral, Mixtral, Mistral-Large-3 |
| GLM | GLM-4, GLM-4-MoE, GLM-OCR |
| Kimi | Kimi-K2.5, Kimi-VL |
| InternVL | InternVL, InternS1, InternS2 |
| MiniCPM | MiniCPM, MiniCPM-V, MiniCPM-O |
| LFM | LFM2, LFM2-MoE, LFM2-VL |
| Nemotron | Nemotron-H |
| 其他 | Falcon, DBRX, Grok, OLMo, Phi, Pixtral, etc. |

模型通过 `registry.py` 自动注册，按 `model_arch` 字符串映射。

### 4.7 推测解码 (`speculative/`)

| 组件 | 说明 |
|------|------|
| `eagle_worker.py` | EAGLE 推测解码 Worker |
| `eagle_worker_v2.py` | EAGLE V2 Worker |
| `multi_layer_eagle_worker.py` | 多层 EAGLE Worker |
| `ngram_worker.py` | N-gram 推测解码 Worker |
| `frozen_kv_mtp_worker.py` | Frozen KV MTP Worker |
| `standalone_worker.py` | 独立草稿模型 Worker |
| `dflash_worker.py` | DFlash 推测解码 Worker |
| `spec_registry.py` | 推测策略注册表 |
| `adaptive_spec_params.py` | 自适应推测参数 |

### 4.8 分离式推理 (`disaggregation/`)

支持 Prefill 与 Decode 分离部署：

| 文件 | 说明 |
|------|------|
| `prefill.py` | Prefill 侧调度逻辑 |
| `decode.py` | Decode 侧调度逻辑 |
| `encode_server.py` | Prefill 编码服务器 |
| `encode_receiver.py` | 接收编码结果 |
| `mooncake/` | Mooncake 传输后端 |
| `nixl/` | NIXL 传输后端 |
| `mori/` | Mori 传输后端 |
| `ascend/` | 华为 Ascend 传输后端 |

### 4.9 分布式 (`distributed/`)

| 文件 | 说明 |
|------|------|
| `parallel_state.py` | 并行状态管理 (TP/PP/DP 组初始化) |
| `communication_op.py` | 分布式通信原语 |
| `device_communicators/` | 设备级通信器 (NCCL, MSCCLPP, Custom AllReduce 等) |

### 4.10 约束解码 (`constrained/`)

| 后端 | 说明 |
|------|------|
| `xgrammar_backend.py` | XGrammar 后端 |
| `llguidance_backend.py` | LLGuidance 后端 |
| `outlines_backend.py` | Outlines 后端 |
| `reasoner_grammar_backend.py` | 推理模式语法后端 |

### 4.11 编译优化 (`compilation/`)

基于 `torch.compile` 的编译优化：

| 文件 | 说明 |
|------|------|
| `compile.py` | 编译入口 |
| `pass_manager.py` | 编译 Pass 管理器 |
| `cuda_piecewise_backend.py` | CUDA 分段编译后端 |
| `inductor_pass.py` | Inductor 自定义 Pass |
| `compilation_config.py` | 编译配置 |

### 4.12 硬件后端 (`hardware_backend/` / `platforms/`)

| 后端 | 说明 |
|------|------|
| `gpu/` | NVIDIA GPU (CUDA) |
| `rocm.py` | AMD GPU (ROCm) |
| `xpu/` | Intel XPU |
| `npu/` | 华为 NPU |
| `musa/` | 摩尔线程 MUSA |
| `mlx/` | Apple MLX |

### 4.13 其他核心模块

| 模块 | 说明 |
|------|------|
| `lora/` | LoRA 适配器管理、动态加载/卸载、内存池 |
| `multimodal/` | 多模态处理器 (VIT CUDA Graph 等) |
| `observability/` | 可观测性 (Metrics/Tracing/计时) |
| `function_call/` | 函数调用 / Tool Use 解析器 (30+ 种格式) |
| `checkpoint_engine/` | 检查点引擎 (权重保存/加载) |
| `model_loader/` | 模型权重加载器 |
| `tokenizer/` | Tokenizer 封装 |
| `session/` | 会话管理 (多轮对话) |
| `eplb/` | Expert Parallelism Load Balancing |
| `elastic_ep/` | 弹性 Expert Parallelism |
| `batch_overlap/` | 批次重叠执行 (Prefill/Decode 重叠) |
| `kv_canary/` | KV Cache 一致性校验 |
| `dllm/` | DLLM (延迟感知 LLM) 混入 |
| `multiplex/` | PDMux 上下文管理 |
| `configs/` | 模型配置 (50+ 种自定义 Config) |
| `plugins/` | 插件系统 |
| `weight_sync/` | 权重同步 |

---

## 5. SGLang 编程语言前端 (`lang/`)

SGLang 提供了一套 DSL (Domain-Specific Language)，用于编写结构化的 LLM 程序：

| 文件 | 说明 |
|------|------|
| `api.py` | 公共 API (gen, select, function, image 等) |
| `ir.py` | 中间表示 (SglExpr, SglFunction, SglGen 等) |
| `interpreter.py` | 解释器执行 SGLang 程序 |
| `tracer.py` | 追踪执行过程 |
| `backend/` | 后端抽象 (Runtime, Engine 等) |
| `choices.py` | 选择采样策略 |
| `chat_template.py` | Chat 模板 |

---

## 6. sgl-kernel：预编译 CUDA Kernel 库

`sgl-kernel` 是 SGLang 的 AOT (Ahead-of-Time) 预编译 CUDA/C++ Kernel 库，通过 CMake 构建。

### 6.1 Kernel 类别

| 目录 | 说明 |
|------|------|
| `csrc/attention/` | 注意力相关 Kernel |
| `csrc/moe/` | MoE 相关 Kernel |
| `csrc/quantization/` | 量化 Kernel |
| `csrc/gemm/` | GEMM Kernel |
| `csrc/allreduce/` | AllReduce 通信 Kernel |
| `csrc/grammar/` | 约束解码 Grammar Kernel |
| `csrc/mamba/` | Mamba 状态更新 Kernel |
| `csrc/memory/` | 内存操作 Kernel |
| `csrc/speculative/` | 推测解码 Kernel |
| `csrc/spatial/` | 空间注意力 Kernel |
| `csrc/kvcacheio/` | KV Cache IO Kernel |
| `csrc/elementwise/` | 逐元素操作 Kernel |
| `csrc/flashmla_extension.cc` | FlashMLA 扩展 |
| `csrc/cpu/` | CPU 优化 Kernel |

### 6.2 Python 接口

`sgl-kernel/python/sgl_kernel/` 提供了 Python 绑定，主要模块包括：`attention`, `moe`, `quantization`, `gemm`, `sampling`, `grammar`, `speculative`, `memory` 等。

---

## 7. sgl-model-gateway：模型网关

用 Rust 编写的高性能模型网关，提供：

- 模型路由与负载均衡
- 多模型实例管理
- gRPC 服务

---

## 8. 请求处理全流程

```
Client Request
      │
      ▼
┌─────────────┐     ┌──────────────────────────────────────────────┐
│ HTTP Server │────►│ 1. 解析请求 (OpenAI/Anthropic/Ollama 格式)    │
│  (FastAPI)  │     │ 2. 验证参数                                   │
└─────────────┘     │ 3. 路由到对应 Serving 模块                     │
      │             └──────────────────────────────────────────────┘
      ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│TokenizerManager  │─►│ 1. Tokenize 文本                              │
│                  │  │ 2. 处理多模态数据 (Image/Video/Audio)          │
│                  │  │ 3. LoRA 适配器选择                            │
│                  │  │ 4. DP 路由决策                                │
└──────────────────┘  └──────────────────────────────────────────────┘
      │ ZMQ
      ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│DP Controller     │─►│ 1. 接收 tokenized 请求                        │
│  (可选)          │  │ 2. 路由到 DP Scheduler 实例                    │
└──────────────────┘  └──────────────────────────────────────────────┘
      │ ZMQ
      ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│  Scheduler       │─►│ 1. 加入等待队列                               │
│                  │  │ 2. Radix Tree 前缀匹配 → KV Cache 复用         │
│                  │  │ 3. 内存分配 (TokenToKVPool)                   │
│                  │  │ 4. 组装 ScheduleBatch                         │
│                  │  │ 5. Prefill/Decode 调度决策                     │
│                  │  │ 6. 构造 ForwardBatch → 发送至 ModelRunner      │
└──────────────────┘  └──────────────────────────────────────────────┘
      │
      ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│  ModelRunner     │─►│ 1. 执行模型前向推理                             │
│  (TpWorker)      │  │ 2. TP/PP 通信                                │
│                  │  │ 3. Logits 采样                                │
│                  │  │  │  ├─ 常规 Decode                             │
│                  │  │  │  ├─ Speculative Decode (EAGLE/Ngram/MTP)   │
│                  │  │  │  └─ Constrained Decode (Grammar)           │
│                  │  │ 4. CUDA Graph 重放 (Decode 阶段)              │
│                  │  │ 5. 返回生成的 token IDs                       │
└──────────────────┘  └──────────────────────────────────────────────┘
      │
      ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│ DetokenizerMgr   │─►│ 1. Token ID → 文本                           │
│                  │  │ 2. 流式增量输出                               │
│                  │  │ 3. 停止词检测                                 │
│                  │  │ 4. 请求完成通知                               │
└──────────────────┘  └──────────────────────────────────────────────┘
      │
      ▼
  Client Response (Streaming / Batch)
```

---

## 9. 并行策略

### 9.1 Tensor Parallelism (TP)

- 模型权重按列/行切分到多个 GPU
- 通过 NCCL/Custom AllReduce 进行通信
- `parallel_state.py` 管理进程组

### 9.2 Pipeline Parallelism (PP)

- 模型层切分到不同 GPU
- 微批次流水线执行
- `scheduler_pp_mixin.py` 管理 PP 调度

### 9.3 Data Parallelism (DP)

- 多个完整模型副本并行
- `data_parallel_controller.py` 负载均衡
- 支持 cache-aware 路由

### 9.4 Expert Parallelism (EP)

- MoE 模型的专家分布到不同 GPU
- `eplb/` 管理专家负载均衡
- `elastic_ep/` 支持弹性专家并行

### 9.5 DP Attention

- Data Parallel + Attention 通信
- `dp_attention.py` 实现 DP Attention 逻辑

---

## 10. 关键优化技术

| 技术 | 说明 | 相关模块 |
|------|------|----------|
| RadixAttention | 基于 Radix Tree 的 KV Cache 前缀共享 | `mem_cache/radix_cache.py` |
| CUDA Graph | 捕获并重放 GPU kernel 序列，减少 launch 开销 | `model_executor/cuda_graph_runner.py` |
| 推测解码 | 草稿模型预生成 → 目标模型验证，加速推理 | `speculative/` |
| 连续批处理 | 动态组装/拆分批次，最大化 GPU 利用率 | `managers/scheduler.py` |
| 批次重叠 | Prefill 与 Decode 重叠执行 | `batch_overlap/` |
| 量化推理 | FP8/INT4/INT8 等低精度推理 | `layers/quantization/` |
| torch.compile | 编译优化，融合 kernel | `compilation/` |
| 分离式推理 | Prefill/Decode 分离部署，独立扩缩容 | `disaggregation/` |
| HiCache | 分层 KV Cache (GPU + Host) | `mem_cache/hicache_storage.py` |
| Tokenizer 并行 | 异步动态批处理 Tokenize | `managers/async_dynamic_batch_tokenizer.py` |

---

## 11. 配置与参数

### 11.1 ServerArgs (`server_args.py`)

核心服务器参数，约 8000 行，涵盖：

- 模型路径与配置
- 并行策略 (tp_size, pp_size, dp_size, ep_size)
- 内存管理 (mem_fraction_static, chunked_prefill_size)
- 调度策略 (schedule_policy, max_running_requests)
- 推测解码配置
- 量化配置
- 分离式推理配置
- 网络配置 (host, port, additional_ports)

### 11.2 环境变量 (`environ.py`)

通过 `SGLANG_*` 环境变量控制运行时行为，如：
- `SGLANG_CUDA_GRAPH_MAX_BS`：CUDA Graph 最大 batch size
- `SGLANG_RADIX_CACHE_TYPE`：Radix Cache 类型选择
- 等等

---

## 12. 测试体系

| 目录 | 说明 |
|------|------|
| `test/srt/` | SRT 引擎集成测试 |
| `test/registered/` | 注册的 CI 测试用例 |
| `test/run_suite.py` | 测试套件运行器 |
| `python/sglang/test/` | Python 单元测试 |
| `sgl-kernel/tests/` | Kernel 级别测试 |

---

## 13. 基准测试 (`benchmark/`)

涵盖多种场景的基准测试：

- **服务吞吐量**：`benchmark_batch/`
- **内核微基准**：`kernels/`
- **调度器**：`scheduler/`
- **长上下文**：`multi_document_qa/`, `long_json_decode/`
- **多模态**：`llava_bench/`
- **LoRA**：`lora/`
- **推测解码**：`bench_adaptive_speculative.py`
- **HiCache**：`hicache/`
- **标准评测**：`gsm8k/`, `mmlu/`, `hellaswag/`, `mtbench/`

---

## 14. 架构总结

SGLang 的架构设计围绕以下核心理念：

1. **多进程解耦**：Tokenizer → Scheduler → ModelRunner → Detokenizer 各自独立进程，通过 ZMQ 通信，避免 GIL 限制
2. **RadixAttention 驱动**：Radix Tree 是 KV Cache 管理的核心，实现高效的前缀共享与自动淘汰
3. **可插拔后端**：注意力、量化、MoE、Grammar 等模块均支持多种后端，通过注册表动态选择
4. **极致优化**：CUDA Graph、torch.compile、Fused Kernel、连续批处理等多层优化
5. **灵活部署**：支持单卡到多卡、单节点到多节点、共享式到分离式的多种部署模式

---

## 附录 A：各文件夹功能详解

### A.1 顶层目录

| 目录 | 功能说明 |
|------|----------|
| `python/sglang/` | 核心 Python 包，包含推理引擎、编程语言前端、CLI、基准测试等 |
| `sgl-kernel/` | AOT 预编译 CUDA/C++ Kernel 库，通过 CMake 构建，提供高性能算子 |
| `sgl-model-gateway/` | Rust 编写的高性能模型网关，负责负载均衡、路由、服务发现、熔断等 |
| `rust/` | Rust 组件，包含 gRPC 服务端 (`sglang-grpc`) |
| `benchmark/` | 各类基准测试脚本，覆盖吞吐量、调度、多模态、LoRA、长上下文等场景 |
| `test/` | 集成测试与 CI 测试，包含 SRT 引擎测试和注册的测试用例 |
| `examples/` | 示例代码，涵盖前端语言用法、运行时配置、监控、检查点、Profiler 等 |
| `docker/` | Docker 构建文件，支持多种硬件平台 (CUDA/ROCm/NPU/XPU/ARM/Xeon) |
| `scripts/` | 运维与部署脚本，含 CI 脚本、发布脚本、调试工具等 |
| `3rdparty/` | 第三方依赖 (如 AMD 相关组件) |
| `proto/` | Protocol Buffers 定义文件 (gRPC 服务接口) |
| `experimental/` | 实验性功能，含 `sgl-router` 路由器 |
| `docs/` | 项目文档 |
| `docs_new/` | 新版文档 |
| `assets/` | 静态资源文件 |

### A.2 `python/sglang/` 核心包

| 目录 | 功能说明 |
|------|----------|
| `srt/` | **SGLang Runtime** — 推理引擎核心，是整个项目最大的模块 (详见 A.3) |
| `lang/` | SGLang 编程语言前端 (DSL)，提供 `gen`、`select`、`function` 等声明式 API |
| `cli/` | 命令行接口，包含 `serve` (启动服务器)、`generate` (生成文本)、`killall` (终止进程) |
| `benchmark/` | 基准测试工具库，包含数据集和工具函数 |
| `jit_kernel/` | JIT 编译的 CUDA Kernel，运行时动态编译轻量级算子 (详见 A.10) |
| `eval/` | 评测工具，含 LLaMA3 评测和 Loogle 长上下文评测 |
| `multimodal_gen/` | 多模态生成框架，含扩散模型运行时、配置、CUDA Kernel 等 |
| `test/` | Python 层面的单元测试 |

### A.3 `python/sglang/srt/` — SGLang Runtime 核心

#### A.3.1 入口与 API 层 (`entrypoints/`)

| 目录/文件 | 功能说明 |
|-----------|----------|
| `http_server.py` | FastAPI HTTP 服务器主入口，定义所有 HTTP 路由端点 |
| `engine.py` | Python API 入口 (`Engine` 类)，无需 HTTP 即可直接调用推理引擎 |
| `EngineBase.py` | Engine 基类，定义通用接口 |
| `engine_score_mixin.py` | Engine Score 功能混入 |
| `engine_info_bootstrap_server.py` | 引擎信息引导服务器 (分布式发现) |
| `http_server_engine.py` | HTTP 服务器与 Engine 的桥接层 |
| `grpc_server.py` | gRPC 服务端实现 |
| `context.py` | 请求上下文管理 |
| `tool.py` | Tool Use / Function Call 工具定义 |
| `ssl_utils.py` | SSL/TLS 工具函数 |
| `warmup.py` | 服务预热逻辑 |
| `v1_loads.py` | V1 版本负载均衡 |
| `harmony_utils.py` | Harmony 协议工具 |
| `openai/` | OpenAI 兼容 API，含 Chat/Completions/Embedding/Rerank/Classify/Score/Tokenize/Transcription/Responses 等 |
| `openai/realtime/` | OpenAI Realtime API (WebSocket 实时对话) |
| `openai/transcription_adapters/` | 语音转录适配器 (Whisper, Qwen3-ASR) |
| `anthropic/` | Anthropic Messages API 兼容实现 |
| `ollama/` | Ollama API 兼容实现，含智能路由 (`smart_router.py`) |

#### A.3.2 调度管理层 (`managers/`)

| 文件 | 功能说明 |
|------|----------|
| `scheduler.py` | **核心调度器**，管理请求排队、批处理调度、KV Cache 分配、Prefill/Decode 决策、推测解码协调 (~4000 行) |
| `tokenizer_manager.py` | **Tokenizer 管理器**，负责文本 Tokenize、多模态预处理、DP 路由、LoRA 管理 (~2900 行) |
| `detokenizer_manager.py` | **Detokenizer 管理器**，将 token ID 解码为文本、流式输出拼接、停止词检测 |
| `tp_worker.py` | Tensor Parallel Worker，管理模型权重加载/更新、分布式推理通信 |
| `schedule_batch.py` | **批处理数据结构**，定义 `Req` 和 `ScheduleBatch`，包含调度所需全部信息 (~2700 行) |
| `schedule_policy.py` | 请求调度策略，含优先级排序、Prefill/Decode 比例控制、内存压力感知 |
| `data_parallel_controller.py` | Data Parallel 控制器，负责 DP 实例间请求负载均衡 |
| `io_struct.py` | **进程间通信数据结构**，定义 TokenizerManager ↔ Scheduler ↔ DetokenizerManager 间传递的所有消息类型 (~2100 行) |
| `communicator.py` | ZMQ 通信原语，实现 Fan-Out 请求分发与结果收集 |
| `cache_controller.py` | KV Cache 控制器，管理缓存生命周期 |
| `async_dynamic_batch_tokenizer.py` | 异步动态批处理 Tokenizer，提高 Tokenize 吞吐 |
| `multimodal_processor.py` | 多模态数据处理器，调度各类多模态前处理 |
| `template_manager.py` | Chat 模板管理器 |
| `template_detection.py` | 自动检测 Chat 模板类型 |
| `mm_utils.py` | 多模态工具函数 (共享内存传输等) |
| `embed_types.py` | 嵌入类型定义 (`PositionalEmbeds`) |
| `overlap_utils.py` | Prefill/Decode 重叠执行工具 |
| `prefill_delayer.py` | Prefill 延迟器 (控制 Prefill 时机) |
| `scheduler_pp_mixin.py` | Pipeline Parallel 调度混入 |
| `scheduler_recv_skipper.py` | 调度器接收跳过逻辑 |
| `scheduler_input_blocker.py` | 调度器输入阻塞器 |
| `tokenizer_control_mixin.py` | Tokenizer 控制逻辑混入 |
| `tokenizer_manager_score_mixin.py` | Tokenizer Score 功能混入 |
| `hisparse_coordinator.py` | HiSparse 协调器 |
| `disagg_service.py` | 分离式推理服务启动 |
| `load_snapshot.py` | 快照加载 |
| `multi_tokenizer_mixin.py` | 多 Tokenizer 路由混入 |
| `configure_logging.py` | 日志配置 |

**`scheduler_components/`** — Scheduler 内部组件：

| 文件 | 功能说明 |
|------|----------|
| `request_receiver.py` | 请求接收器 |
| `output_sender.py` | 输出发送器 |
| `output_streamer.py` | 流式输出器 |
| `batch_result_processor.py` | 批次结果处理器 |
| `logprob_result_processor.py` | Log 概率结果处理器 |
| `dp_attn.py` | Data Parallel Attention 逻辑 |
| `kv_events_publisher.py` | KV Cache 事件发布器 |
| `pool_stats_observer.py` | 内存池状态观察器 |
| `load_inquirer.py` | 负载查询器 |
| `metrics_reporter.py` | 指标上报器 |
| `new_token_ratio_tracker.py` | 新 Token 比率追踪器 |
| `flush_wrapper.py` | 缓存刷新包装器 |
| `idle_sleeper.py` | 空闲休眠器 (减少 CPU 占用) |
| `invariant_checker.py` | 不变量检查器 (调试用) |
| `ipc_channels.py` | IPC 通道管理 |
| `profiler_manager.py` | Profiler 管理器 |
| `weight_updater.py` | 权重更新器 |

#### A.3.3 模型执行层 (`model_executor/`)

| 文件 | 功能说明 |
|------|----------|
| `model_runner.py` | **ModelRunner 核心**，管理模型加载、前向推理、CUDA Graph 捕获与重放 (~3600 行) |
| `forward_batch_info.py` | `ForwardBatch` 定义，GPU 侧推理所需的张量数据 (input_ids, positions, seq_lens 等) |
| `forward_context.py` | 前向推理上下文管理，提供模型层访问全局状态 |
| `cuda_graph_runner.py` | CUDA Graph 捕获与重放，减少 kernel launch 开销 |
| `piecewise_cuda_graph_runner.py` | 分段 CUDA Graph，支持长序列等场景的分段捕获 |
| `breakable_cuda_graph_runner.py` | 可中断 CUDA Graph |
| `cpu_graph_runner.py` | CPU 推理图运行器 |
| `mindspore_runner.py` | MindSpore 框架运行器 |
| `pool_configurator.py` | 内存池配置器，自动计算 KV Cache 可用显存 |
| `hook_manager.py` | 前向钩子管理器 |
| `input_buffers.py` | 输入缓冲区管理 |
| `forward_batch_deepseek_mha_mixin.py` | DeepSeek MHA 前向批次混入 |
| `model_runner_kv_cache_mixin.py` | ModelRunner KV Cache 管理混入 |

**`breakable_cuda_graph/`** — 可中断 CUDA Graph 子模块：

| 文件 | 功能说明 |
|------|----------|
| `breakable_cuda_graph.py` | 可中断 CUDA Graph 实现 |
| `context.py` | CUDA Graph 上下文 |
| `cuda_utils.py` | CUDA 工具函数 |

#### A.3.4 KV Cache 与内存管理 (`mem_cache/`)

| 文件/目录 | 功能说明 |
|-----------|----------|
| `radix_cache.py` | **Radix Tree 前缀缓存**，基于引用计数的自动缓存共享与淘汰 |
| `hiradix_cache.py` | 分层 Radix Cache (HiRadix)，支持 GPU + Host 分层存储 |
| `chunk_cache.py` | 基于分块的前缀缓存 |
| `swa_radix_cache.py` | 滑动窗口注意力 Radix Cache |
| `swa_memory_pool.py` | 滑动窗口注意力内存池 |
| `unified_radix_cache.py` | 统一 Radix Cache，整合 Full/SWA/Mamba/Tree 多种缓存策略 |
| `mamba_radix_cache.py` | Mamba 状态 Radix Cache |
| `hi_mamba_radix_cache.py` | 分层 Mamba Radix Cache |
| `memory_pool.py` | **核心内存池**，实现两级内存管理：`ReqToTokenPool` (请求→Token 位置) 和 `TokenToKVPool` (Token 位置→物理 KV 缓存) |
| `memory_pool_host.py` | Host 侧内存池 |
| `deepseek_v4_memory_pool.py` | DeepSeek-V4 专用内存池 |
| `hisparse_memory_pool.py` | HiSparse 内存池 |
| `deepseek_v4_compress_state.py` | DeepSeek-V4 压缩状态管理 |
| `multimodal_cache.py` | 多模态数据缓存 |
| `mmap_allocator.py` | 内存映射分配器 |
| `kv_cache_builder.py` | KV Cache 构建器 |
| `base_prefix_cache.py` | 前缀缓存基类，定义 `Insert`/`Match`/`Evict`/`IncRef`/`DecRef` 接口 |
| `base_swa_memory_pool.py` | 滑动窗口内存池基类 |
| `cache_init_params.py` | 缓存初始化参数 |
| `common.py` | 公共工具函数 |
| `events.py` | KV Cache 事件混入 |
| `evict_policy.py` | 缓存驱逐策略 |
| `flush_cache.py` | 缓存刷新逻辑 |
| `registry.py` | 缓存注册表 |
| `utils.py` | 工具函数 |
| `radix_cache_cpp.py` | C++ Radix Tree 的 Python 绑定 |

**`allocator/`** — 内存分配器：

| 文件 | 功能说明 |
|------|----------|
| `base.py` | 分配器基类 `BaseTokenToKVPoolAllocator` |
| `paged.py` | 分页内存分配器 |
| `token.py` | Token 级内存分配器 |

**`cpp_radix_tree/`** — C++ 高性能 Radix Tree：

| 文件 | 功能说明 |
|------|----------|
| `tree_v2.h/cpp` | C++ Radix Tree 实现 (V2 版本) |
| `tree_v2_impl.h` | 实现细节 |
| `tree_v2_node.h` | 节点定义 |
| `tree_v2_binding.cpp` | PyBind11 绑定 |
| `tree_v2_debug.cpp` | 调试工具 |
| `common.h` | 公共头文件 |
| `radix_tree.py` | Python 封装 |

**`hybrid_cache/`** — 混合缓存控制器：

| 文件 | 功能说明 |
|------|----------|
| `hybrid_cache_controller.py` | 混合缓存 (Attention + Linear Attention) 控制器 |
| `hybrid_pool_assembler.py` | 混合内存池组装器 |

**`sparsity/`** — 稀疏注意力内存管理：

| 目录/文件 | 功能说明 |
|-----------|----------|
| `factory.py` | 稀疏策略工厂 |
| `algorithms/` | 稀疏算法实现 (基类、DeepSeek DSA、QUEST) |
| `backend/` | 稀疏后端适配器 |
| `core/` | 稀疏协调器核心 |

**`storage/`** — 外部存储后端：

| 目录 | 功能说明 |
|------|----------|
| `aibrix_kvcache/` | AIBrix KV Cache 存储 |
| `eic/` | EIC (External Inference Cache) 存储 |
| `hf3fs/` | HF3FS (HuggingFace 3FS) 分布式文件存储 |
| `lmcache/` | LMCache 外部 KV Cache 存储 |
| `mooncake_store/` | Mooncake 存储后端 (含嵌入缓存) |
| `nixl/` | NIXL 高速传输存储 (含 HiCache NIXL) |
| `simm/` | SIMM 存储 |
| `backend_factory.py` | 存储后端工厂 |
| `serde/` | 序列化/反序列化 |

**`unified_cache_components/`** — 统一缓存组件：

| 文件 | 功能说明 |
|------|----------|
| `full_component.py` | 全注意力缓存组件 |
| `swa_component.py` | 滑动窗口注意力缓存组件 |
| `mamba_component.py` | Mamba 状态缓存组件 |
| `tree_component.py` | Tree 缓存组件 |

#### A.3.5 模型层 (`layers/`)

| 文件 | 功能说明 |
|------|----------|
| `radix_attention.py` | RadixAttention 封装层，将注意力计算与 KV Cache 管理 binding |
| `radix_linear_attention.py` | Radix 线性注意力层 |
| `linear.py` | 线性层 (支持 Column/Row Parallel) |
| `sampler.py` | 采样器，从 logits 中采样 token |
| `logits_processor.py` | Logits 处理器 (温度、top-k、top-p、repetition penalty 等) |
| `layernorm.py` | LayerNorm / RMSNorm |
| `vocab_parallel_embedding.py` | 词表并行嵌入层 |
| `pooler.py` | 嵌入池化层 (用于 Embedding 模型) |
| `sparse_pooler.py` | 稀疏池化层 |
| `conv.py` | 卷积层 (视觉模型) |
| `multimodal.py` | 多模态投影层 |
| `activation.py` | 激活函数 (SiLU, GeLU 等) |
| `elementwise.py` | 逐元素操作 |
| `parameter.py` | 参数管理 |
| `clippable_linear.py` | 可裁剪线性层 |
| `mhc.py` / `mhc_head.py` | Multi-Head Choice 层 |
| `model_parallel.py` | 模型并行工具函数 |
| `communicator.py` | 层级通信器 |
| `communicator_dsa_cp.py` | DSA Context Parallel 通信器 |
| `dp_attention.py` | Data Parallel Attention 逻辑 |
| `flashinfer_comm_fusion.py` | FlashInfer 通信融合 |
| `fused_qk_norm.py` / `fused_qk_norm_rope_store.py` | 融合 QK 归一化 + RoPE |
| `gemma4_fused_ops.py` | Gemma4 融合操作 |
| `deepseek_v4_rope.py` | DeepSeek-V4 RoPE |
| `int4fp8_utils.py` | INT4+FP8 工具 |
| `modelopt_utils.py` / `torchao_utils.py` | ModelOpt/TorchAO 工具 |
| `rocm_linear_utils.py` | ROCm 线性层工具 |
| `n_gram_embedding.py` | N-gram 嵌入 |
| `amx_utils.py` | Intel AMX 工具 |

**`layers/attention/`** — 注意力后端 (41 个子模块)：

| 子模块 | 功能说明 |
|--------|----------|
| `base_attn_backend.py` | 注意力后端抽象基类 `AttentionBackend` |
| `attention_registry.py` | 注意力后端注册表 |
| `flashinfer_backend.py` | **FlashInfer 注意力** (默认后端) |
| `flashattention_backend.py` | FlashAttention 2 后端 |
| `flashmla_backend.py` | FlashMLA 后端 (DeepSeek MLA 架构) |
| `flashinfer_mla_backend.py` | FlashInfer MLA 后端 |
| `flash_mla_sm120.py` / `flash_mla_sm120_triton.py` | SM120 架构 FlashMLA |
| `trtllm_mha_backend.py` | TensorRT-LLM MHA 后端 |
| `trtllm_mla_backend.py` | TensorRT-LLM MLA 后端 |
| `dsa_backend.py` | DSA (DeepSeek Sparse Attention) 后端 |
| `cutlass_mla_backend.py` | CUTLASS MLA 后端 |
| `nsa_backend.py` | Native Sparse Attention 后端 |
| `wave_backend.py` | Wave Attention 后端 |
| `hybrid_attn_backend.py` | 混合注意力 (Full + Linear) 后端 |
| `hybrid_linear_attn_backend.py` | 混合线性注意力后端 |
| `dual_chunk_flashattention_backend.py` | 双块 FlashAttention (长上下文) |
| `tbo_backend.py` | Token-By-Overlay 后端 |
| `tokenspeed_mla_backend.py` | TokenSpeed MLA 后端 |
| `torch_native_backend.py` | PyTorch 原生注意力 |
| `torch_flex_backend.py` | PyTorch Flex Attention |
| `intel_amx_backend.py` | Intel AMX 注意力 (CPU) |
| `xpu_backend.py` | Intel XPU 注意力 |
| `hip_flash_mla.py` | AMD HIP FlashMLA |
| `aiter_backend.py` | AITER 注意力后端 |
| `vision.py` | 视觉注意力 (ViT) |
| `merge_state.py` | 注意力状态合并 |
| `utils.py` / `vision_utils.py` | 工具函数 |
| `triton_ops/` | Triton 实现的注意力 Kernel (decode/extend/prefill/merge_state) |
| `cute_utils/` | NVIDIA CuTe 工具 (TMA Copy 等) |
| `dsa/` | DSA 内部实现 (索引器、量化、Triton/TileLang Kernel、TopK 后端) |
| `dsv4/` | DeepSeek-V4 注意力内部实现 (压缩器、索引器、元数据 Kernel) |
| `nsa/` | NSA 内部实现 (索引器、量化、Triton Decode Kernel) |
| `fla/` | Flash Linear Attention 内部实现 (chunk 级计算、fused recurrent 等) |
| `linear/` | 线性注意力后端 (GDN/KDA/Lightning 等) |
| `mamba/` | Mamba 状态更新 (causal conv1d、selective state update) |
| `wave_ops/` | Wave Attention Kernel (decode/extend/prefill) |

**`layers/moe/`** — MoE (Mixture of Experts) 层：

| 文件/目录 | 功能说明 |
|-----------|----------|
| `router.py` | MoE 路由器，决定每个 token 分配给哪些专家 |
| `topk.py` | Top-K 专家选择 |
| `fused_moe_native.py` | 原生 Fused MoE 实现 |
| `cutlass_moe.py` | CUTLASS MoE 实现 |
| `cutlass_w4a8_moe.py` | CUTLASS W4A8 量化 MoE |
| `cutlass_moe_params.py` | CUTLASS MoE 参数 |
| `flashinfer_cutedsl_moe.py` | FlashInfer + CuteDSL MoE |
| `flashinfer_trtllm_moe.py` | FlashInfer + TRT-LLM MoE |
| `mega_moe.py` | Mega MoE (多类型 MoE 融合) |
| `hash_topk.py` | Hash Top-K 选择 |
| `kt_ep_wrapper.py` | KT EP 包装器 |
| `deepep_waterfill.py` | DeepEP 水位填充算法 |
| `utils.py` | 工具函数 |
| `fused_moe_triton/` | Triton Fused MoE (含 Marlin、MXFP4 量化) |
| `ep_moe/` | Expert Parallelism MoE 实现 |
| `moe_runner/` | MoE Runner (多种后端：Triton/Marlin/AITER/DeepGEMM/FlashInfer) |
| `token_dispatcher/` | Token 分发器 (Standard/DeepEP/FlashInfer/Mooncake/MoriEP/NIXL) |

**`layers/quantization/`** — 量化层 (46 个子模块)：

| 文件/目录 | 功能说明 |
|-----------|----------|
| `fp8.py` / `fp8_kernel.py` / `fp8_utils.py` | FP8 量化实现 |
| `w8a8_fp8.py` / `w8a8_int8.py` | W8A8 量化 |
| `int8_kernel.py` / `int8_utils.py` | INT8 量化 |
| `blockwise_int8.py` | 分块 INT8 量化 |
| `w4afp8.py` | W4A-FP8 混合量化 |
| `mxfp4.py` / `mxfp4_tensor.py` | MX-FP4 量化 |
| `nvfp4_gemm_swiglu_nvfp4_quant.py` | NVFP4 GEMM+SwiGLU 量化 |
| `bitsandbytes.py` | BitsAndBytes 量化 |
| `gguf.py` | GGUF 格式支持 |
| `auto_round.py` | AutoRound 量化 |
| `modelopt_quant.py` | NVIDIA ModelOpt 量化 |
| `petit.py` / `petit_utils.py` | Petit 量化 |
| `qoq.py` | QOQ 量化 |
| `fpgemm_fp8.py` | FPGEMM FP8 量化 |
| `mlx.py` | Apple MLX 量化 |
| `awq/` | AWQ 量化 (含 Triton Kernel 和 Scheme) |
| `gptq/` | GPTQ 量化 (含 Scheme) |
| `compressed_tensors/` | Neural Magic Compressed Tensors 格式 |
| `quark/` | Quark 量化 |
| `modelslim/` | ModelSlim 量化 |
| `marlin_utils*.py` | Marlin 量化工具 (FP4/FP8/通用) |
| `kv_cache.py` | KV Cache 量化 |
| `fp4_kv_cache_quant_method.py` / `fp4_utils.py` / `kvfp4_tensor.py` | FP4 KV Cache 量化 |
| `moe_wna16.py` | MoE WNA16 量化 |
| `configs/` | 预计算的量化配置 (按 GPU 型号/矩阵尺寸索引的 JSON 文件) |
| `base_config.py` / `base_scheme.py` | 量化配置与方案基类 |
| `unquant.py` | 未量化 (FP16/BF16) 回退 |
| `utils.py` | 量化工具函数 |

**`layers/rotary_embedding/`** — RoPE 旋转位置编码：

| 文件 | 功能说明 |
|------|----------|
| `base.py` | RoPE 基类 |
| `factory.py` | RoPE 工厂 (自动选择实现) |
| `mrope.py` / `mrope_rope_index.py` | Multi-dimensional RoPE (用于多模态模型) |
| `yarn.py` | YaRN 扩展 RoPE (长上下文) |
| `rope_variant.py` | RoPE 变体 |
| `triton_kernels.py` | Triton 实现的 RoPE Kernel |
| `utils.py` | 工具函数 |

**`layers/deep_gemm_wrapper/`** — DeepGEMM 包装器：

| 文件 | 功能说明 |
|------|----------|
| `entrypoint.py` | DeepGEMM 入口 |
| `configurer.py` | DeepGEMM 配置 |
| `compile_utils.py` | 编译工具 |

**`layers/utils/`** — 层级工具：

| 文件 | 功能说明 |
|------|----------|
| `common.py` | 公共工具 |
| `cp_utils.py` | Context Parallel 工具 |
| `hash.py` | 哈希函数 |
| `logprob.py` | Log 概率计算 |
| `multi_platform.py` | 多平台适配 |

#### A.3.6 模型库 (`models/`)

190+ 种模型架构实现，按模型家族组织。每个文件对应一种 `model_arch`，通过 `registry.py` 自动注册。

| 文件 | 功能说明 |
|------|----------|
| `registry.py` | 模型注册表，管理 `model_arch → ModelClass` 映射 |
| `utils.py` | 模型工具函数 |
| `transformers.py` | 通用 Transformers 模型兼容层 |
| `torch_native_llama.py` | PyTorch 原生 LLaMA (调试用) |
| 其余 ~190 个文件 | 各模型架构的独立实现 (如 `llama.py`, `qwen3.py`, `deepseek_v4.py` 等) |

#### A.3.7 推测解码 (`speculative/`)

| 文件/目录 | 功能说明 |
|-----------|----------|
| `spec_registry.py` | 推测策略注册表 |
| `spec_info.py` | 推测输入/输出信息基类 |
| `spec_utils.py` | 推测工具函数 |
| `draft_utils.py` | 草稿模型工具 |
| `base_spec_worker.py` | 推测 Worker 基类 |
| `eagle_worker.py` / `eagle_worker_v2.py` | EAGLE 推测解码 (V1/V2) |
| `eagle_utils.py` / `eagle_info.py` / `eagle_info_v2.py` | EAGLE 工具与信息 |
| `eagle_draft_cuda_graph_runner.py` / `eagle_draft_extend_cuda_graph_runner.py` | EAGLE 草稿 CUDA Graph |
| `multi_layer_eagle_worker.py` / `multi_layer_eagle_worker_v2.py` / `multi_layer_eagle_utils.py` | 多层 EAGLE |
| `ngram_worker.py` / `ngram_info.py` | N-gram 推测解码 |
| `frozen_kv_mtp_worker.py` / `frozen_kv_mtp_worker_v2.py` | Frozen KV MTP (Multi-Token Prediction) |
| `frozen_kv_mtp_utils.py` / `frozen_kv_mtp_info.py` | Frozen KV MTP 工具 |
| `frozen_kv_mtp_cuda_graph_runner.py` | Frozen KV MTP CUDA Graph |
| `standalone_worker.py` / `standalone_worker_v2.py` | 独立草稿模型 Worker |
| `dflash_worker.py` / `dflash_info.py` / `dflash_utils.py` | DFlash 推测解码 |
| `adaptive_spec_params.py` / `adaptive_runtime_state.py` | 自适应推测参数与运行时状态 |
| `eagle_disaggregation.py` | EAGLE 分离式推理支持 |
| `external_corpus_manager.py` | 外部语料管理器 (N-gram) |
| `cpp_ngram/` | C++ N-gram 实现 (高性能) |
| `triton_ops/` | 推测解码 Triton Kernel (KV 物化融合) |

#### A.3.8 分离式推理 (`disaggregation/`)

| 文件/目录 | 功能说明 |
|-----------|----------|
| `prefill.py` | Prefill 侧调度逻辑 (含 `SchedulerDisaggregationPrefillMixin`) |
| `decode.py` | Decode 侧调度逻辑 (含 `SchedulerDisaggregationDecodeMixin`) |
| `encode_server.py` | Prefill 编码服务器 |
| `encode_receiver.py` | 接收编码结果 |
| `encode_grpc_server.py` | 编码 gRPC 服务器 |
| `decode_kvcache_offload_manager.py` | Decode 侧 KV Cache 卸载管理 |
| `decode_schedule_batch_mixin.py` | Decode 调度批次混入 |
| `kv_events.py` | KV Cache 事件定义 |
| `utils.py` | 工具函数 (含 `DisaggregationMode`, `TransferBackend` 等) |
| `base/` | 基础传输连接 |
| `common/` | 公共传输组件 (Staging Buffer/Handler) |
| `mooncake/` | Mooncake 传输后端 |
| `nixl/` | NIXL 传输后端 |
| `mori/` | Mori 传输后端 |
| `ascend/` | 华为 Ascend 传输后端 (含专用传输引擎) |
| `fake/` | Fake 传输后端 (测试用) |

#### A.3.9 分布式 (`distributed/`)

| 文件/目录 | 功能说明 |
|-----------|----------|
| `parallel_state.py` | **并行状态管理**，初始化 TP/PP/DP 进程组 (~2400 行) |
| `parallel_state_wrapper.py` | 并行状态包装器 |
| `communication_op.py` | 分布式通信原语 |
| `naive_distributed.py` | 朴素分布式实现 |
| `utils.py` | 工具函数 |
| `device_communicators/` | 设备级通信器 (详见下表) |

**`device_communicators/`** — 设备级通信器：

| 文件 | 功能说明 |
|------|----------|
| `pynccl.py` / `pynccl_wrapper.py` / `pynccl_allocator.py` | NCCL 通信封装 |
| `custom_all_reduce.py` / `custom_all_reduce_v2.py` / `custom_all_reduce_utils.py` / `custom_all_reduce_ops.py` | 自定义 AllReduce (基于 SHM) |
| `quick_all_reduce.py` | 快速 AllReduce |
| `pymscclpp.py` | MSCCL++ 通信 |
| `torch_symm_mem.py` | Torch 对称内存通信 |
| `mooncake_transfer_engine.py` | Mooncake 传输引擎 |
| `shm_broadcast.py` | 共享内存广播 |
| `hpu_communicator.py` | HPU (Habana) 通信器 |
| `npu_communicator.py` | NPU (华为) 通信器 |
| `xpu_communicator.py` | XPU (Intel) 通信器 |
| `all_reduce_utils.py` | AllReduce 工具 |
| `cuda_wrapper.py` | CUDA 通信封装 |

#### A.3.10 约束解码 (`constrained/`)

| 文件/目录 | 功能说明 |
|-----------|----------|
| `base_grammar_backend.py` | 语法后端基类 |
| `grammar_manager.py` | 语法管理器，协调约束解码 |
| `xgrammar_backend.py` | XGrammar 后端 (高性能) |
| `llguidance_backend.py` | LLGuidance 后端 |
| `outlines_backend.py` | Outlines 后端 |
| `outlines_jump_forward.py` | Outlines Jump-Forward 优化 |
| `reasoner_grammar_backend.py` | 推理模式语法后端 (thinking/answer 分离) |
| `utils.py` | 工具函数 |
| `torch_ops/` | Token 过滤的 PyTorch 实现 |
| `triton_ops/` | Bitmask 操作和 Token 过滤的 Triton 实现 |

#### A.3.11 编译优化 (`compilation/`)

| 文件 | 功能说明 |
|------|----------|
| `compile.py` | 编译入口，触发 `torch.compile` |
| `compiler_interface.py` | 编译器接口 |
| `backend.py` | 编译后端 |
| `pass_manager.py` | 编译 Pass 管理器 |
| `inductor_pass.py` | Inductor 自定义 Pass |
| `cuda_piecewise_backend.py` | CUDA 分段编译后端 |
| `npu_piecewise_backend.py` | NPU 分段编译后端 |
| `compilation_config.py` | 编译配置 (含 split op 注册) |
| `compilation_counter.py` | 编译计数器 |
| `piecewise_context_manager.py` | 分段编译上下文管理 |
| `fix_functionalization.py` | 函数化修复 |
| `fx_utils.py` | FX 图工具 |
| `weak_ref_tensor.py` | 弱引用张量 |

#### A.3.12 硬件后端 (`hardware_backend/` / `platforms/`)

| 目录 | 功能说明 |
|------|----------|
| `hardware_backend/gpu/` | NVIDIA GPU 后端 (含量化子模块) |
| `hardware_backend/mlx/` | Apple MLX 后端 (含 KV Cache、ModelRunner、TP Worker、AOT 编译) |
| `hardware_backend/npu/` | 华为 NPU 后端 (含专用注意力、量化、MoE、内存池、图运行器) |
| `hardware_backend/xpu/` | Intel XPU 后端 (含专用 Kernel) |
| `hardware_backend/musa/` | 摩尔线程 MUSA 后端 (含专用注意力、Kernel、Layer) |
| `platforms/cuda.py` | CUDA 平台抽象 |
| `platforms/rocm.py` | ROCm 平台抽象 |
| `platforms/interface.py` | 平台接口定义 |
| `platforms/device_mixin.py` | 设备混入 |

#### A.3.13 LoRA (`lora/`)

| 文件/目录 | 功能说明 |
|-----------|----------|
| `lora.py` | LoRA 层定义 |
| `lora_manager.py` | LoRA 管理器，动态加载/卸载适配器 |
| `lora_config.py` | LoRA 配置 |
| `lora_registry.py` | LoRA 注册表 |
| `lora_drainer.py` | LoRA 卸载排空器 |
| `lora_overlap_loader.py` | LoRA 重叠加载器 (后台加载) |
| `layers.py` | LoRA 注入层定义 |
| `mem_pool.py` | LoRA 内存池 |
| `eviction_policy.py` | LoRA 驱逐策略 |
| `deepseek_mla_correction.py` | DeepSeek MLA LoRA 修正 |
| `lora_moe_runner_marlin.py` / `lora_moe_runners.py` | LoRA MoE Runner |
| `utils.py` | 工具函数 |
| `backend/` | LoRA 后端 (Triton/PyTorch/Ascend/Chunked/LMHead Mixing) |
| `torch_ops/` | LoRA PyTorch 算子 (含 CUDA Graph 支持) |
| `triton_ops/` | LoRA Triton Kernel (SGEMV/Embedding/MoE/量化等) |

#### A.3.14 多模态 (`multimodal/`)

| 文件/目录 | 功能说明 |
|-----------|----------|
| `mm_utils.py` | 多模态工具函数 |
| `customized_mm_processor_utils.py` | 自定义多模态处理器工具 |
| `audio_from_video.py` | 从视频提取音频 |
| `vit_cuda_graph_runner.py` | ViT CUDA Graph 运行器 |
| `internvl_utils.py` / `internvl_vit_cuda_graph_runner.py` | InternVL 工具与 ViT CUDA Graph |
| `processors/` | 41 个多模态处理器 (CLIP/DeepSeek-VL/InternVL/Gemma/Qwen-VL/Phi4MM/Whisper 等) |
| `evs/` | EVS (Efficient Vision System) 多模态处理器 |

#### A.3.15 可观测性 (`observability/`)

| 文件 | 功能说明 |
|------|----------|
| `metrics_collector.py` | **Prometheus 指标收集器** (~1900 行)，收集 GPU/CPU/请求/Cache 等指标 |
| `forward_pass_metrics.py` | 前向推理指标 |
| `func_timer.py` | 函数计时器 |
| `cpu_monitor.py` | CPU 监控线程 |
| `req_time_stats.py` | 请求时间统计 |
| `request_metrics_exporter.py` | 请求指标导出器 |
| `trace.py` | 分布式追踪 |
| `label_transform.py` | 指标标签变换 |
| `startup_func_log_and_timer.py` | 启动函数日志与计时 |

#### A.3.16 函数调用 / Tool Use (`function_call/`)

30+ 种函数调用格式的检测器：

| 文件 | 功能说明 |
|------|----------|
| `function_call_parser.py` | 函数调用解析器入口 |
| `core_types.py` | 核心类型定义 |
| `base_format_detector.py` | 格式检测器基类 |
| `utils.py` | 工具函数 |
| `json_array_parser.py` | JSON 数组解析器 |
| 其余 27 个 `*_detector.py` | 各模型/格式的函数调用检测器 (Hermes/Llama3.2/Qwen2.5/Mistral/Gemma4/DeepSeek/GLM4 等) |

#### A.3.17 其他核心模块

| 目录/文件 | 功能说明 |
|-----------|----------|
| `configs/` | 51 个模型配置文件，覆盖所有支持模型的特殊配置 (含 `model_config.py` 核心配置类) |
| `server_args.py` | 服务器参数定义 (~8000 行)，涵盖所有启动配置 |
| `environ.py` | 环境变量定义 (`SGLANG_*` 前缀) |
| `sampling/sampling_params.py` | 采样参数定义 |
| `sampling/sampling_batch_info.py` | 批次采样信息 |
| `sampling/custom_logit_processor.py` | 自定义 Logit 处理器 |
| `sampling/penaltylib/` | 采样惩罚库 (Frequency/Presence/Repetition/MinNewTokens) |
| `parser/` | 解析器 (Reasoning/Conversation/CodeCompletion/Harmony) |
| `model_loader/` | 模型权重加载器 (支持本地/远程/S3/Azure) |
| `checkpoint_engine/` | 检查点引擎 (权重保存/加载/更新) |
| `tokenizer/` | Tokenizer 封装 (Tiktoken 等) |
| `session/` | 会话管理 (多轮对话、流式会话) |
| `connector/` | 远程连接器 (S3/Azure/Redis/远程实例) |
| `batch_overlap/` | 批次重叠执行策略 (单批次/双批次重叠) |
| `batch_invariant_ops/` | 批次不变操作 |
| `eplb/` | Expert Parallelism Load Balancing (算法: DeepSeek/DeepSeek-Vec/Elasticity-Aware) |
| `elastic_ep/` | 弹性 Expert Parallelism (专家备份/迁移) |
| `kv_canary/` | KV Cache 一致性校验 (金丝雀检测、违规报告、扰动注入、健康检查) |
| `dllm/` | DLLM (延迟感知 LLM) 混入 (算法: JointThreshold/LowConfidence) |
| `multiplex/` | PDMux 上下文管理 (DP Attention 复用) |
| `plugins/` | 插件系统 (钩子注册) |
| `weight_sync/` | 权重同步 (张量桶工具) |
| `state_capturer/` | 状态捕获 (TopK 索引器、路由专家) |
| `debug_utils/` | 调试工具 (CUDA Coredump、张量比较、调度模拟、日志解析) |
| `utils/` | 40+ 个工具模块 (网络、HF 补丁、Tokenizer 补丁、CUDA 工具、NUMA、Watchdog 等) |
| `constants.py` | 全局常量 |
| `ray/` | Ray 集成 (DP Controller/Scheduler Actor/HTTP Server) |
| `grpc/` | gRPC 初始化 |

### A.4 `python/sglang/lang/` — SGLang 编程语言前端

| 文件/目录 | 功能说明 |
|-----------|----------|
| `api.py` | 公共 API：`function`、`gen`、`select`、`image`、`Runtime`、`Engine` 等 |
| `ir.py` | 中间表示：`SglFunction`、`SglGen`、`SglSelect`、`SglImage`、`SglRoleBegin/End` 等 |
| `interpreter.py` | SGLang 程序解释器，执行 IR 节点 |
| `tracer.py` | 追踪执行过程，记录中间结果 |
| `choices.py` | 选择采样策略 (token length normalized 等) |
| `chat_template.py` | Chat 模板处理 |
| `backend/` | 后端实现：OpenAI/Anthropic/VertexAI/LiteLLM/Crusoe/RuntimeEndpoint |

### A.5 `python/sglang/jit_kernel/` — JIT 编译 CUDA Kernel

运行时动态编译的轻量级 CUDA Kernel，覆盖 50+ 种算子：

| 类别 | 文件 | 功能说明 |
|------|------|----------|
| **注意力** | `flash_attention.py`, `flash_attention_v3.py`, `flash_attention_v4.py` | Flash Attention V1/V3/V4 |
| **量化** | `fp8_quantize.py`, `mxfp8.py`, `nvfp4.py`, `per_tensor_quant_fp8.py`, `per_token_group_quant_8bit.py` | 各类量化 Kernel |
| **MoE** | `moe_align.py`, `moe_fused_gate.py`, `moe_lora_align.py`, `moe_wna16_marlin.py`, `grouped_topk.py` | MoE 对齐/门控/TopK |
| **归一化** | `norm.py`, `rmsnorm_hf.py` | RMSNorm/LayerNorm |
| **RoPE** | `rope.py` | 旋转位置编码 |
| **LoRA** | `moe_lora_align.py` | MoE LoRA 对齐 |
| **KV Cache** | `kvcache.py`, `hicache.py`, `hisparse.py`, `set_mla_kv_buffer.py`, `concat_mla.py` | KV Cache 读写/HiCache/HiSparse |
| **融合操作** | `fused_qknorm_rope.py`, `fused_store_index_cache.py`, `fused_metadata_copy.py` | QK Norm+RoPE、存储+索引融合 |
| **MLA** | `mla_kv_pack_quantize_fp8.py` | MLA KV Pack+FP8 量化 |
| **N-gram** | `ngram_corpus.py`, `ngram_embedding.py` | N-gram 语料/嵌入 |
| **Marlin** | `gptq_marlin.py`, `gptq_marlin_repack.py`, `awq_marlin_repack.py`, `awq_dequantize.py` | Marlin/GPTQ/AWQ 反量化 |
| **DeepSeek-V4** | `dsv4/` | DeepSeek-V4 专用 Kernel |
| **扩散模型** | `diffusion/` | 扩散模型相关 Kernel |
| **KV Canary** | `kv_canary/` | KV Cache 一致性校验 Kernel |
| **通信** | `all_reduce.py` | AllReduce 通信 |
| **其他** | `activation.py`, `add_constant.py`, `clamp_position.py`, `fixup_zero_kv.py`, `hadamard.py`, `timestep_embedding.py`, `resolve_future_token_ids.py` | 激活函数、常量添加、位置裁剪等 |

### A.6 `sgl-kernel/` — AOT 预编译 CUDA/C++ Kernel 库

#### A.6.1 C++ 源码 (`csrc/`)

| 目录 | 功能说明 |
|------|----------|
| `attention/` | 注意力 Kernel (CUTLASS MLA, Merge States, Vertical Slash Index) |
| `moe/` | MoE Kernel (CUTLASS MoE, TopK Softmax/Sigmoid, Align, Gate, Sum, FP8 Blockwise) |
| `gemm/` | GEMM Kernel (FP8, INT8, AWQ, GPTQ, Marlin, QServe, DeepSeek Router/Fused GEMM) |
| `quantization/` | 量化 Kernel (GGUF 反量化) |
| `allreduce/` | AllReduce 通信 Kernel (Custom AllReduce, Quick AllReduce, MSCCL++, NCCL) |
| `grammar/` | 约束解码 Grammar Kernel (Token Bitmask) |
| `mamba/` | Mamba Kernel (Causal Conv1D) |
| `memory/` | 内存操作 Kernel (WeakRef Tensor) |
| `speculative/` | 推测解码 Kernel (EAGLE, N-gram, PackBit, Sampling) |
| `spatial/` | 空间注意力 Kernel (GreenCtx Stream) |
| `kvcacheio/` | KV Cache IO Kernel (Transfer) |
| `elementwise/` | 逐元素操作 Kernel (Activation, MLA Concat, Norm+RoPE, TopK, Position Encoding) |
| `expert_specialization/` | 专家特化 Kernel (FP8 Blockwise, SM100 MXFP8 BlockScaled) |
| `cpu/` | CPU 优化 Kernel (GEMM, Attention, MoE, Norm, RoPE, KV Cache, Mamba, SHM 等) |
| `musa/` | MUSA (摩尔线程) Kernel (MoE GEMV, 位置编码, 采样) |
| `cutlass_extensions/` | CUTLASS 扩展 (Epilogue, GEMM) |
| `flash_extension.cc` | FlashMLA 扩展 (Dense/Sparse Decode) |
| `spatial_extension.cc` | 空间注意力扩展 |
| `flashmla_extension.cc` | FlashMLA 扩展 |

#### A.6.2 Python 绑定 (`python/sgl_kernel/`)

| 文件 | 功能说明 |
|------|----------|
| `attention.py` | 注意力 Kernel Python 接口 |
| `flash_attn.py` / `flash_mla.py` / `sparse_flash_attn.py` | Flash Attention/MLA/Sparse 接口 |
| `moe.py` / `cutlass_moe.py` | MoE Kernel Python 接口 |
| `gemm.py` | GEMM Kernel Python 接口 |
| `quantization/` | 量化 Kernel Python 接口 |
| `sampling.py` | 采样 Kernel Python 接口 |
| `grammar.py` | Grammar Kernel Python 接口 |
| `speculative.py` | 推测解码 Kernel Python 接口 |
| `mamba.py` | Mamba Kernel Python 接口 |
| `memory.py` | 内存 Kernel Python 接口 |
| `spatial.py` | 空间注意力 Python 接口 |
| `kvcacheio.py` | KV Cache IO Python 接口 |
| `allreduce.py` | AllReduce Python 接口 |
| `elementwise.py` | 逐元素操作 Python 接口 |
| `expert_specialization.py` | 专家特化 Python 接口 |
| `top_k.py` | Top-K Python 接口 |
| `scalar_type.py` | 标量类型定义 |
| `debug_utils.py` / `test_utils.py` / `testing/` | 调试与测试工具 |
| `load_utils.py` | Kernel 加载工具 |
| `utils.py` | 通用工具 |
| `version.py` | 版本信息 |
| `metal.py` | Apple Metal 接口 |
| `musa.py` | MUSA 接口 |

### A.7 `sgl-model-gateway/` — Rust 模型网关

| 目录 | 功能说明 |
|------|----------|
| `src/main.rs` | 入口 |
| `src/lib.rs` | 库入口 |
| `src/server.rs` | 服务器主逻辑 |
| `src/app_context.rs` | 应用上下文 |
| `src/service_discovery.rs` | 服务发现 |
| `src/middleware.rs` | 中间件 |
| `src/version.rs` | 版本信息 |
| `src/config/` | 配置 (Builder/Types/Validation) |
| `src/core/` | 核心 (Worker 管理、熔断器、重试、令牌桶、Job Queue、指标聚合) |
| `src/policies/` | 路由策略 (RoundRobin/Random/PowerOfTwo/CacheAware/ConsistentHashing/PrefixHash/Bucket/Tree/Manual) |
| `src/routers/` | 路由器 (HTTP/gRPC/OpenAI/Conversations/Mesh/MCP/Tokenize) |
| `src/observability/` | 可观测性 (Metrics/OpenTelemetry Trace/Inflight Tracker/Logging) |
| `src/wasm/` | WASM 插件支持 |
| `bindings/` | Python 绑定 |
| `e2e_test/` | 端到端测试 |
| `benches/` | 性能基准测试 |

### A.8 `benchmark/` — 基准测试

| 目录 | 功能说明 |
|------|----------|
| `benchmark_batch/` | 批处理吞吐量基准测试 |
| `kernels/` | Kernel 级微基准测试 |
| `scheduler/` | 调度器基准测试 |
| `hicache/` | HiCache 基准测试 |
| `lora/` | LoRA 基准测试 |
| `asr/` | 语音识别基准测试 |
| `bench_adaptive_speculative.py` | 自适应推测解码基准测试 |
| `bench_attention_sink/` | 注意力 Sink 基准测试 |
| `bench_in_batch_prefix/` | 批内前缀共享基准测试 |
| `bench_linear_attention/` | 线性注意力基准测试 |
| `bench_pynccl_allocator/` | NCCL 分配器基准测试 |
| `bench_rope/` | RoPE 基准测试 |
| `gsm8k/` / `mmlu/` / `hellaswag/` / `ceval/` / `boolq/` | 标准学术评测 |
| `mtbench/` / `mmmu/` / `llava_bench/` | 多模态/聊天评测 |
| `multi_document_qa/` / `long_json_decode/` / `multi_turn_chat/` | 长上下文/多轮对话评测 |
| `json_decode_regex/` / `json_schema/` / `json_jump_forward/` | JSON 约束解码评测 |
| `react/` / `tree_of_thought_*/` / `generative_agents/` / `dspy/` | Agent/推理评测 |
| `deepseek_v3/` / `hf3fs/` / `io/` / `prefill_only/` / `tip_suggestion/` | 其他专项评测 |
| `blog_v0_2/` | Blog V0.2 评测 |

### A.9 `test/` — 测试体系

| 目录/文件 | 功能说明 |
|-----------|----------|
| `srt/` | SRT 引擎集成测试 |
| `srt/cpu/` | CPU 推理测试 |
| `registered/` | 注册的 CI 测试用例 |
| `run_suite.py` | 测试套件运行器 |
| `lm_eval_configs/` | LM Eval 配置 |
| `manual/` | 手动测试脚本 |
| `pytest.ini` | PyTest 配置 |
| `README.md` | 测试说明文档 |

### A.10 `examples/` — 示例代码

| 目录 | 功能说明 |
|------|----------|
| `runtime/` | 运行时使用示例 (Engine/Server 启动) |
| `frontend_language/` | SGLang 前端语言用法示例 |
| `chat_template/` | Chat 模板示例 |
| `monitoring/` | 监控示例 (Prometheus) |
| `profiler/` | Profiler 使用示例 |
| `checkpoint_engine/` | 检查点引擎示例 |
| `usage/` | 常见用法示例 |
| `sagemaker/` | AWS SageMaker 部署示例 |
| `assets/` | 示例资源文件 |

### A.11 `docker/` — Docker 构建文件

| 文件 | 功能说明 |
|------|----------|
| `Dockerfile` | 默认 CUDA Dockerfile |
| `rocm.Dockerfile` | AMD ROCm Dockerfile |
| `npu.Dockerfile` | 华为 NPU Dockerfile |
| `xpu.Dockerfile` | Intel XPU Dockerfile |
| `xeon.Dockerfile` | Intel Xeon CPU Dockerfile |
| `arm64.Dockerfile` | ARM64 CPU Dockerfile |
| `gateway.Dockerfile` | 模型网关 Dockerfile |
| `sgl-deep-gemm.Dockerfile` | DeepGEMM Dockerfile |
| `sgl-router.Dockerfile` | 路由器 Dockerfile |
| `sagemaker.Dockerfile` | SageMaker Dockerfile |
| `compose.yaml` | Docker Compose 配置 |
| `configs/` | Docker 配置文件 |
| `k8s-sglang-*.yaml` | Kubernetes 部署 YAML |
| `serve` | 服务启动脚本 |

### A.12 `scripts/` — 运维与部署脚本

| 目录/文件 | 功能说明 |
|-----------|----------|
| `ci/` | CI 相关脚本 |
| `ci_monitor/` | CI 监控脚本 |
| `release/` | 发布脚本 |
| `playground/` | 开发调试脚本 |
| `code_sync/` | 代码同步脚本 |
| `build_sgl_deep_gemm.sh` | DeepGEMM 构建脚本 |
| `killall_sglang.sh` | 终止所有 SGLang 进程 |
| `convert_deepseek_nextn.py` | DeepSeek NextN 模型转换 |
| `sort_testcases_alphabetically.py` | 测试用例排序 |
| 其他 `update_*_whl_index.py` | Wheel 包索引更新脚本 |

### A.13 `proto/` — Protocol Buffers 定义

| 目录 | 功能说明 |
|------|----------|
| `sglang/` | SGLang gRPC 服务接口定义 |

### A.14 `rust/sglang-grpc/` — Rust gRPC 服务

| 文件/目录 | 功能说明 |
|-----------|----------|
| `src/` | gRPC 服务实现源码 |
| `Cargo.toml` | Rust 项目配置 |
| `build.rs` | 构建脚本 |
| `rust-toolchain.toml` | Rust 工具链配置 |
