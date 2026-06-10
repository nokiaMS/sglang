# SGLang 整体架构图

> 基于 SGLang 源代码（v0.5.12+）绘制，涵盖进程模型、核心模块、数据流与扩展子系统。

---

## 1. 顶层进程架构

SGLang 采用多进程架构，各进程通过 ZMQ IPC 通信。主进程运行 HTTP Server / Engine / TokenizerManager，子进程运行 Scheduler 和 DetokenizerManager。

```mermaid
flowchart TD
    subgraph MainProcess["主进程 (Main Process)"]
        Client["Client\nOpenAI / Ollama / Anthropic\n离线 Engine API"]
        HTTP["HTTP Server\n(FastAPI)"]
        Engine["Engine\n(API入口)"]
        TM["TokenizerManager\nTokenize / 路由 / 调度"]
        Client --> HTTP --> Engine --> TM
    end

    subgraph SchedulerProcess["Scheduler 子进程"]
        Queue["请求队列\n(waiting / running / decode)"]
        Policy["调度策略\n(PrefillAdder / Policy)"]
        KV["KV Cache 管理\n(RadixTree)"]
        Queue --> Policy --> KV
        subgraph TPWorker["TPWorker"]
            MR["ModelRunner"]
            MW["Model Weights"]
            CG["CUDA Graph Runner"]
            MR --- MW
            MR --- CG
        end
        KV --> TPWorker
    end

    subgraph DetokProcess["DetokenizerManager 子进程"]
        Detok["Token IDs → Detokenize → 增量文本输出"]
    end

    TM -- "ZMQ IPC" --> Queue
    TPWorker -- "ZMQ IPC" --> Detok
    Detok -- "ZMQ IPC" --> TM
```

**关键源码路径：**
- 主进程入口：`python/sglang/srt/entrypoints/engine.py`、`http_server.py`
- TokenizerManager：`python/sglang/srt/managers/tokenizer_manager.py`
- Scheduler：`python/sglang/srt/managers/scheduler.py`
- DetokenizerManager：`python/sglang/srt/managers/detokenizer_manager.py`
- TPWorker：`python/sglang/srt/managers/tp_worker.py`
- ModelRunner：`python/sglang/srt/model_executor/model_runner.py`

---

## 2. 请求完整生命周期

```mermaid
flowchart TD
    Client["Client Request"]

    subgraph TM1["TokenizerManager"]
        T1["1. 接收请求 (GenerateReqInput / EmbeddingReqInput)"]
        T2["2. Tokenize (HuggingFace Tokenizer/Processor)"]
        T3["3. 多模态预处理 (Image/Audio/Video → Tensor)"]
        T4["4. 应用 Chat Template"]
        T5["5. 构造 TokenizedGenerateReqInput"]
        T6["6. 通过 ZMQ 发送至 Scheduler"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6
    end

    subgraph Sched["Scheduler"]
        S1["1. 接收 tokenized 请求，创建 Req 对象"]
        S2["2. 加入 waiting 队列"]
        S3["3. 调度策略选择 Prefill / Decode 请求"]
        S4["4. 构建 ScheduleBatch"]
        S5["5. 分配 KV Cache (RadixTree 前缀匹配)"]
        S6["6. 调用 TPWorker 执行 forward"]
        S7["7. 后处理 (采样、结构化输出、logprobs)"]
        S8["8. 将输出 token ids 发送至 DetokenizerManager"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8
    end

    subgraph Detok["DetokenizerManager"]
        D1["1. 接收 token ids"]
        D2["2. 增量 Detokenize"]
        D3["3. 构造 BatchStrOutput / BatchTokenIDOutput"]
        D4["4. 通过 ZMQ 返回 TokenizerManager"]
        D1 --> D2 --> D3 --> D4
    end

    subgraph TM2["TokenizerManager (回调)"]
        R1["1. 接收 detokenized 输出"]
        R2["2. 组装最终响应"]
        R3["3. 流式或非流式返回给 Client"]
        R1 --> R2 --> R3
    end

    Client --> T1
    T6 -- "ZMQ (ipc)" --> S1
    S8 -- "ZMQ (ipc)" --> D1
    D4 -- "ZMQ (ipc)" --> R1
    R3 --> Client
```

---

## 3. Scheduler 内部架构

Scheduler 是 SGLang 的核心，负责批调度、KV Cache 管理和模型执行协调。它通过 Mixin 模式组合多种功能。

```mermaid
flowchart TD
    subgraph Scheduler["Scheduler"]
        subgraph Mixins["继承 Mixin"]
            M1["SchedulerDisaggDecodeMixin\n(PD分离 Decode端)"]
            M2["SchedulerPPMixin\n(流水线并行)"]
            M3["SchedulerMultiplexMixin\n(多路复用)"]
            M4["SchedulerDisaggPrefillMixin\n(PD分离 Prefill端)"]
            M5["SchedulerDllmMixin\n(扩散模型调度)"]
            M6["SchedulerMlxOverlapMixin\n(Apple MLX)"]
        end

        subgraph Components["Scheduler 组件"]
            C1["RequestReceiver\n(接收/解析请求)"]
            C2["SchedulePolicy\n(FCFS/LPM/DFS/LOCF/HCBS)"]
            C3["IpcChannels\n(ZMQ 通道管理)"]
            C4["BatchResultProcessor\n(结果处理)"]
            C5["MetricsReporter\n(性能指标上报)"]
            C6["ProfilerManager\n(PyTorch Profiler)"]
            C7["WeightUpdaterManager\n(权重热更新)"]
            C8["LoadInquirer\n(负载查询)"]
            C9["IdleSleeper\n(空闲休眠)"]
            C10["KvEventsPublisher\n(KV事件发布)"]
            C11["OutputStreamer\n(输出流推送)"]
            C12["InvariantChecker\n(不变量检查)"]
        end

        subgraph Loop["调度核心循环"]
            L1["1. recv_requests() ← 从 ZMQ 接收新请求"]
            L2["2. update_requests() ← 更新请求状态"]
            L3["3. prepare_batch() ← 构建调度批次"]
            L3a["   3a. prefill_policy() ← 选择 prefill 请求"]
            L3b["   3b. decode_policy() ← 选择 decode 请求"]
            L3c["   3c. alloc_kv_cache() ← RadixTree 分配 KV Cache"]
            L4["4. forward_batch() ← 调用 TPWorker 执行前向"]
            L5["5. process_batch_result() ← 处理采样结果"]
            L6["6. send_output() ← 发送输出到 DetokenizerManager"]
            L1 --> L2 --> L3 --> L3a
            L3a --> L3b --> L3c --> L4
            L4 --> L5 --> L6 --> L1
        end
    end
```

**关键源码路径：**
- Scheduler 主类：`python/sglang/srt/managers/scheduler.py:286`
- 调度策略：`python/sglang/srt/managers/schedule_policy.py`
- 批次构建：`python/sglang/srt/managers/schedule_batch.py`
- 组件目录：`python/sglang/srt/managers/scheduler_components/`

---

## 4. KV Cache 与 RadixAttention 架构

RadixAttention 是 SGLang 的核心创新，通过 Radix Tree 实现前缀共享和自动缓存复用。

```mermaid
flowchart TD
    subgraph KVCache["KV Cache 管理架构"]
        subgraph Unified["Unified Radix Cache"]
            RT["Radix Tree\n(前缀树)\n\nroot\n├── Hello\n│   ├── world\n│   └── SGLang\n└── How are\n    └── you\n\n引用计数管理\nLRU 淘汰策略"]
            subgraph Pools["Token-to-KV Pool"]
                FKV["Full KV Pool\n(全注意力)"]
                SWA["SWA KV Pool\n(滑动窗口)"]
                R2T["ReqToToken Pool\n(请求→KV槽映射)"]
            end
            RT --> Pools
        end

        subgraph HiCache["HiCache (层级缓存)"]
            GPU["GPU KV Cache\n(热数据)"]
            CPU["CPU HiCache\n(温数据)"]
            Disk["Disk HiCache\n(冷数据)"]
            HIRadix["HIRadixCache\n(层级Radix树)"]
            HIStorage["HiCacheStorage\n(存储后端)"]
            RuntimeAttach["Runtime Attach/Detach\n(动态挂载)"]
            GPU <--> CPU <--> Disk
        end

        subgraph Special["特殊 KV Cache 类型"]
            MC["Mamba Radix Cache\n(SSM状态缓存)"]
            HC["Hybrid Cache\n(Full + SWA)"]
            DS["DeepSeek V4 Memory Pool\n(压缩状态)"]
            HS["HiSparse Memory Pool\n(稀疏注意力)"]
            MM["MultiModal Cache\n(多模态特征)"]
        end
    end
```

**关键源码路径：**
- Radix Cache：`python/sglang/srt/mem_cache/radix_cache.py`
- C++ 高性能版：`python/sglang/srt/mem_cache/cpp_radix_tree/`、`radix_cache_cpp.py`
- 内存池：`python/sglang/srt/mem_cache/memory_pool.py`
- HiCache：`python/sglang/srt/mem_cache/hiradix_cache.py`、`hicache_storage.py`
- KV Cache Builder：`python/sglang/srt/mem_cache/kv_cache_builder.py`

---

## 5. 模型执行架构（TPWorker + ModelRunner）

```mermaid
flowchart TD
    subgraph TPWorker["TPWorker (张量并行工作器)"]
        subgraph MR["ModelRunner"]
            subgraph Model["Model (nn.Module)"]
                subgraph Attn["Attention 层"]
                    RA["RadixAttention\n(Prefill+Decode)"]
                    AB["后端选择:\nFlashInfer\nFlashAttention\nTriton\nFlashMLA\nTRT-LLM MHA/MLA\nWave\nDSA\nCutlass MLA"]
                end
                subgraph MoE["MoE 层"]
                    Gate["Gate\n(路由器)"]
                    Expert["Expert\n(并行专家)"]
                    DG["DeepGEMM / Marlin\n(MoE内核)"]
                    Gate --> Expert --> DG
                end
                Quant["量化层\nFP8 / FP4 / INT4 / AWQ /\nGPTQ / Marlin"]
                Linear["Linear\n(TP并行)"]
                RoPE["RoPE\n旋转位置"]
                Sampler["Sampler\n(采样器)"]
                LP["LogitsProcessor\n(约束采样)"]
            end

            subgraph CUDAGraph["CUDA Graph 管理"]
                CGR["CUDAGraphRunner\n(标准图捕获)"]
                BCG["BreakableCUDAGraphRunner\n(可中断图)"]
                PCG["PiecewiseCUDAGraphRunner\n(分段图)"]
            end
        end

        subgraph DataFlow["数据流: ScheduleBatch → ForwardBatch"]
            SB["ScheduleBatch (CPU 侧)\n- Req 对象列表\n- 请求状态/优先级\n- 采样参数\n- KV Cache 分配信息"]
            FB["ForwardBatch (GPU 侧)\n- input_ids (GPU)\n- req_pool_indices\n- seq_lens / positions\n- kv_cache (GPU显存)\n- ForwardMode:\n  EXTEND / DECODE /\n  MIXED / IDLE /\n  TARGET_VERIFY /\n  DRAFT_EXTEND /\n  PREBUILT"]
            SB --> FB
        end
    end
```

**关键源码路径：**
- ModelRunner：`python/sglang/srt/model_executor/model_runner.py`
- ForwardBatch：`python/sglang/srt/model_executor/forward_batch_info.py`
- CUDA Graph Runner：`python/sglang/srt/model_executor/cuda_graph_runner.py`
- Breakable CUDA Graph：`python/sglang/srt/model_executor/breakable_cuda_graph/`
- Piecewise CUDA Graph：`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- Attention 后端：`python/sglang/srt/layers/attention/`
- MoE 层：`python/sglang/srt/layers/moe/`
- 量化层：`python/sglang/srt/layers/quantization/`

---

## 6. 分布式并行架构

```mermaid
flowchart TD
    subgraph Dist["分布式并行架构"]
        subgraph DPC["DataParallelController"]
            LB["负载均衡策略:\nROUND_ROBIN\nFOLLOW_BOOTSTRAP_ROOM\nTOTAL_REQUESTS\nTOTAL_TOKENS"]
            subgraph DPW["DP Workers"]
                DW0["DP Worker 0\nTokenizerManager → Scheduler → TPWorker"]
                DW1["DP Worker 1\nTokenizerManager → Scheduler → TPWorker"]
                DWN["DP Worker N\nTokenizerManager → Scheduler → TPWorker"]
            end
            LB --> DPW
        end

        subgraph Parallel["单 Worker 内并行策略"]
            TP["张量并行 TP\n同一请求切分到多GPU"]
            PP["流水线并行 PP\n长上下文分段到多GPU"]
            EP["专家并行 EP\nMoE专家分布到多GPU"]
            DP2["数据并行 DP\n(DP Attention)\n请求级并行，多副本推理"]
        end

        subgraph Comm["通信后端"]
            NCCL["NCCL"]
            PyNCCL["PyNCCL"]
            CAR["Custom All-Reduce"]
            SM["Symmetric Memory"]
            MSCCLPP["MSCCL++"]
        end
    end
```

**关键源码路径：**
- DataParallelController：`python/sglang/srt/managers/data_parallel_controller.py`
- 分布式通信：`python/sglang/srt/distributed/`
- 并行状态：`python/sglang/srt/distributed/parallel_state.py`
- 设备通信器：`python/sglang/srt/distributed/device_communicators/`
- DP Attention：`python/sglang/srt/layers/dp_attention.py`
- Elastic EP：`python/sglang/srt/elastic_ep/`
- EP 负载均衡：`python/sglang/srt/eplb/`

---

## 7. PD 分离部署架构（Prefill-Decode Disaggregation）

```mermaid
flowchart LR
    subgraph Prefill["Prefill 实例"]
        PTM["TokenizerManager"]
        PS["Scheduler\n(PrefillMixin)\n- 专注于 Prefill\n- 大批量长序列处理\n- KV Cache 写入"]
        PTW["TPWorker\n(大显存 / 大算力)"]
        PTM --> PS --> PTW
    end

    subgraph KVTransfer["KV Cache 传输层"]
        MC["Mooncake\n(RDMA)"]
        NIXL["NIXL"]
        MORI["MORI\n(AMD)"]
    end

    subgraph Decode["Decode 实例"]
        DTM["TokenizerManager"]
        DS["Scheduler\n(DecodeMixin)\n- 专注于 Decode\n- 低延迟单 token 生成\n- KV Cache 读取"]
        DTW["TPWorker\n(高带宽 / 低延迟)"]
        DTM --> DS --> DTW
    end

    PS -- "KV Cache 传输" --> KVTransfer --> DS
```

**关键源码路径：**
- PD Prefill 端：`python/sglang/srt/disaggregation/prefill.py`
- PD Decode 端：`python/sglang/srt/disaggregation/decode.py`
- 传输后端：`python/sglang/srt/disaggregation/mooncake/`、`nixl/`、`mori/`
- 通用基础：`python/sglang/srt/disaggregation/base/`、`common/`
- KV 事件：`python/sglang/srt/disaggregation/kv_events.py`

---

## 8. 推测解码架构

```mermaid
flowchart TD
    subgraph Spec["推测解码架构 (Speculative Decoding)"]
        subgraph Registry["SpecRegistry (注册表)"]
            E["Eagle / Eagle3\n(草稿模型投机)"]
            MTP["FrozenKV-MTP\n(多token预测)"]
            NG["N-gram\n(统计投机)"]
            DF["DFlash\n(扩散投机)"]
            MLE["MultiLayer-Eagle\n(多层Eagle)"]
            SA["Standalone\n(独立草稿模型)"]
        end

        subgraph Flow["推测解码执行流程"]
            Draft["Draft Worker\n(草稿模型)\n生成 k 个候选 token"]
            Target["Target Model\n(目标模型)\n一次前向验证所有候选 token"]
            Verify["Verify (验证)\n对比草稿token与目标token\n接受匹配 / 拒绝不匹配\n接受 n 个正确 token\n+ 1 个修正 token"]
            Draft --> Target --> Verify
        end

        FM["ForwardMode:\nDRAFT_EXTEND ← 草稿模型前向\nTARGET_VERIFY ← 目标模型验证\nDRAFT_EXTEND_V2 ← Eagle V2 草稿模型前向"]
    end
```

**关键源码路径：**
- 注册表：`python/sglang/srt/speculative/spec_registry.py`
- Eagle Worker：`python/sglang/srt/speculative/eagle_worker.py`、`eagle_worker_v2.py`
- MTP Worker：`python/sglang/srt/speculative/frozen_kv_mtp_worker.py`、`frozen_kv_mtp_worker_v2.py`
- N-gram Worker：`python/sglang/srt/speculative/ngram_worker.py`
- DFlash Worker：`python/sglang/srt/speculative/dflash_worker.py`
- 自适应推测：`python/sglang/srt/speculative/adaptive_runtime_state.py`

---

## 9. 结构化输出架构

```mermaid
flowchart TD
    subgraph Struct["结构化输出架构 (Structured Outputs)"]
        subgraph GM["GrammarManager (语法管理器)"]
            subgraph Backends["语法后端 (可插拔)"]
                LLG["llguidance\n(默认后端)"]
                XG["xgrammar"]
                OL["Outlines"]
                RGB["ReasonerGrammarBackend\n(推理模型约束)"]
                CFSM["Compressed FSM\n(压缩有限状态机 - 3x 加速)"]
            end
        end

        Fmts["支持的约束格式: JSON / Regex / EBNF / JSON Schema"]

        subgraph Workflow["工作流程"]
            W1["请求"] --> W2["GrammarManager"] --> W3["编译为 FSM"] --> W4["每 token 应用约束"] --> W5["合法 token 输出"]
        end
    end
```

**关键源码路径：**
- GrammarManager：`python/sglang/srt/constrained/grammar_manager.py`
- 后端基类：`python/sglang/srt/constrained/base_grammar_backend.py`
- llguidance：`python/sglang/srt/constrained/llguidance_backend.py`
- xgrammar：`python/sglang/srt/constrained/xgrammar_backend.py`
- Outlines：`python/sglang/srt/constrained/outlines_backend.py`
- Triton 算子：`python/sglang/srt/constrained/triton_ops/`

---

## 10. API 层与入口架构

```mermaid
flowchart TD
    subgraph API["API 入口架构"]
        subgraph EngineAPI["Engine (离线引擎 API)"]
            EG["Engine.generate() ← 生成请求"]
            EAG["Engine.async_generate() ← 异步生成请求"]
            EE["Engine.encode() ← Embedding 请求"]
            EUW["Engine.update_weights() ← 权重热更新"]
            EP["Engine.start_profile() ← 启动 Profiler"]
        end

        subgraph HTTP["HTTP Server (FastAPI)"]
            subgraph APIs["协议 API"]
                OAI["OpenAI API\n/v1/completions\n/v1/chat/completions\n/v1/embeddings\n/v1/images/generations"]
                OLL["Ollama API\n/api/chat\n/api/generate\n/api/show"]
                ANT["Anthropic API\n/v1/messages"]
            end
            subgraph Mgmt["管理 API"]
                HA["Health API\n/health\n/health_generate"]
                LA["LoRA API\n/load_lora\n/unload_lora"]
                WA["Weight Update API\n/update_weights\n/get_model_info"]
                SA["Session API\n/open_session\n/close_session"]
                CA["Config API\n/configure_logging"]
            end
        end

        subgraph GRPC["gRPC Server (可选)"]
            PB["Protocol Buffers:\nproto/sglang/runtime/v1/"]
            RG["Rust gRPC:\nrust/sglang-grpc/"]
        end
    end
```

**关键源码路径：**
- Engine：`python/sglang/srt/entrypoints/engine.py`
- HTTP Server：`python/sglang/srt/entrypoints/http_server.py`
- OpenAI API：`python/sglang/srt/entrypoints/openai/`
- Ollama API：`python/sglang/srt/entrypoints/ollama/`
- Anthropic API：`python/sglang/srt/entrypoints/anthropic/`
- gRPC Server：`python/sglang/srt/entrypoints/grpc_server.py`
- EngineBase：`python/sglang/srt/entrypoints/EngineBase.py`

---

## 11. LoRA 服务架构

```mermaid
flowchart TD
    subgraph LoRA["LoRA 服务架构"]
        Reg["LoRA Registry (注册表)\nLoRARef: name, path, device\n← 管理所有已注册的 LoRA 适配器"]

        subgraph LM["LoRA Manager (管理器)"]
            LL["LoRALayer\n(层适配器)"]
            LMP["LoRAMemPool\n(内存池)"]
            LD["LoRADrainer\n(卸载管理)"]
            LC["LoRAConfig\n(配置)"]
            EP2["EvictionPolicy\n(淘汰策略)"]
            LOL["LoRAOverlapLoader\n(重叠加载)"]
            MLR["MoE-LoRA Runner"]
            MLR2["Marlin-LoRA Runner"]
        end

        Reg --> LM

        subgraph Workflow["工作流程"]
            W1["请求携带 lora_path"] --> W2["Registry 查找"] --> W3["Manager 加载权重"] --> W4["注入对应层的 LoRA 权重"] --> W5["前向计算时混合 Base + LoRA"]
        end
    end
```

**关键源码路径：**
- LoRA Manager：`python/sglang/srt/lora/lora_manager.py`
- LoRA Registry：`python/sglang/srt/lora/lora_registry.py`
- LoRA Layer：`python/sglang/srt/lora/layers.py`
- LoRA Config：`python/sglang/srt/lora/lora_config.py`
- MoE-LoRA：`python/sglang/srt/lora/lora_moe_runners.py`

---

## 12. 多模态处理架构

```mermaid
flowchart TD
    subgraph MM["多模态处理架构"]
        subgraph TMM["TokenizerManager (多模态入口)"]
            subgraph MMP["MultimodalProcessor"]
                IP["Image Processor"]
                VP["Video Processor"]
                AP["Audio Processor"]
                PR["Processor Registry"]
            end
            Note["输入: Image / Video / Audio → 预处理 → Tensor Features\n每个多模态项有独立 hash，支持 RadixAttention 缓存复用"]
        end

        subgraph ModelSide["Model 侧 (视觉/音频编码器)"]
            CLIP["CLIP ViT"]
            SigLIP["SigLIP ViT"]
            MiniCPM["MiniCPM-V ViT"]
            QwenVL["Qwen2-VL ViT"]
            Whisper["Whisper (Audio)"]
            Gemma4V["Gemma4 Vision"]
            InternVL["InternVL ViT"]
        end
    end
```

**关键源码路径：**
- 多模态处理器：`python/sglang/srt/managers/multimodal_processor.py`
- 多模态工具：`python/sglang/srt/multimodal/`
- 多模态层：`python/sglang/srt/layers/multimodal.py`
- 多模态缓存：`python/sglang/srt/mem_cache/multimodal_cache.py`
- 模型实现：`python/sglang/srt/models/llava.py`、`qwen2_vl.py`、`gemma3_mm.py` 等

---

## 13. sgl-kernel 内核架构

```mermaid
flowchart TD
    subgraph Kernel["sgl-kernel 内核架构"]
        subgraph AOT["AOT 内核 (sgl-kernel)"]
            Att["attention\n(注意力)"]
            MoE["moe\n(专家混合)"]
            QK["quantization\n(量化)"]
            Samp["sampling\n(采样)"]
            Norm["norm\n(归一化)"]
            Act["activation\n(激活函数)"]
            Rot["rotary\n(位置编码)"]
            Other["csrc/\n(其他)"]
            Build["构建: CMake + pyproject.toml + setup.py\n平台: NVIDIA / AMD ROCm / MUSA / Metal"]
        end

        subgraph JIT["JIT 内核 (jit_kernel)"]
            JITDesc["轻量级 JIT CUDA 内核，运行时编译加载\n适用于快速迭代和实验性内核"]
        end
    end
```

**关键源码路径：**
- sgl-kernel 目录：`sgl-kernel/csrc/`、`sgl-kernel/python/`
- JIT 内核：`python/sglang/jit_kernel/`
- 测试：`tests/kernels/`
- 基准：`sgl-kernel/benchmark/`

---

## 14. 可观测性与监控架构

```mermaid
flowchart TD
    subgraph Observability["可观测性与监控架构"]
        subgraph Metrics["Metrics (指标)"]
            SMC["SchedulerMetricsCollector\n- 请求延迟\n- Prefill/Decode 吞吐量\n- KV Cache 命中率\n- Batch 大小\n- GPU 利用率"]
            TMC["TokenizerMetricsCollector\n- API 延迟\n- Tokenize 耗时\n- 请求计数"]
        end

        subgraph Tracing["Tracing (追踪)"]
            OT["OpenTelemetry → OTLP Exporter → Jaeger / Tempo / ..."]
            OTDesc["请求级别链路追踪:\nClient → Tokenizer → Scheduler → ModelRunner"]
        end

        subgraph Exporter["Request Metrics Exporter"]
            RME["将每个请求的详细时间指标导出到外部系统"]
        end
    end
```

**关键源码路径：**
- 指标收集：`python/sglang/srt/observability/metrics_collector.py`
- 追踪：`python/sglang/srt/observability/trace.py`
- 请求时间统计：`python/sglang/srt/observability/req_time_stats.py`
- CPU 监控：`python/sglang/srt/observability/cpu_monitor.py`
- 指标导出：`python/sglang/srt/observability/request_metrics_exporter.py`

---

## 15. 全景架构总览

```mermaid
flowchart TD
    subgraph ClientLayer["客户端层 (Client Layer)"]
        C1["OpenAI SDK"]
        C2["Anthropic SDK"]
        C3["Ollama CLI"]
        C4["HTTP Client"]
        C5["Python Engine API"]
    end

    subgraph APILayer["API 入口层 (API Layer)"]
        A1["OpenAI API"]
        A2["Ollama API"]
        A3["Anthropic API"]
        A4["Engine (离线)"]
        A5["gRPC Server"]
        A6["Health API"]
        A7["LoRA API"]
    end

    subgraph PreprocessLayer["预处理层 (Preprocess Layer)"]
        TMFull["TokenizerManager\nTokenize | Chat Template | 多模态预处理 | LoRA 路由 | 请求分派"]
    end

    subgraph SchedLayer["调度层 (Scheduling Layer)"]
        Sch["Scheduler\n(批调度核心)"]
        DPC["DPController\n(数据并行控制)"]
        SP["调度策略\nFCFS/LPM/DFS/LOCF/HCBS"]
        RA["RadixAttention\n(前缀缓存)"]
        HC["HiCache\n(层级缓存)"]
        Gram["GrammarManager\n(结构化输出)"]
    end

    subgraph ExecLayer["执行层 (Execution Layer)"]
        MR2["ModelRunner + TPWorker"]
        Attn2["Attention Backends"]
        MoE2["MoE Layers"]
        Lin2["Linear (TP)"]
        Samp2["Sampler"]
        CG2["CUDA Graph Runners"]
        RoPE2["RoPE"]
        Quant2["Quantize Layers"]
        LoRA2["LoRA Layers"]
        LP2["LogitsProcessor"]
    end

    subgraph KernelLayer["内核层 (Kernel Layer)"]
        AOTK["sgl-kernel (AOT)\nCUDA/C++ 预编译内核"]
        JITK["jit_kernel (JIT)\n运行时编译轻量内核"]
        FIK["FlashInfer\n第三方注意力库"]
    end

    subgraph HWLayer["硬件层 (Hardware Layer)"]
        NV["NVIDIA GPU"]
        AMD["AMD GPU"]
        Intel["Intel Xeon CPU"]
        TPU["Google TPU"]
        NPU["Ascend NPU"]
        MT["Moore Threads"]
    end

    subgraph CrossCut["横切关注点 (Cross-cutting Concerns)"]
        Obs["Observability\n(指标/追踪)"]
        Dist["分布式通信\n(NCCL/RDMA)"]
        Wgt["权重管理\n(加载/更新)"]
        Plat["平台适配\n(Platform抽象)"]
    end

    ClientLayer --> APILayer --> PreprocessLayer
    PreprocessLayer -- "ZMQ" --> SchedLayer
    SchedLayer --> ExecLayer
    ExecLayer --> KernelLayer --> HWLayer
```

---

## 附录：进程间通信（IPC）数据结构

```mermaid
sequenceDiagram
    participant TM as TokenizerManager
    participant S as Scheduler
    participant DM as DetokenizerManager
    participant DPC as DPController
    participant E as Engine

    Note over TM,S: ZMQ IPC 通信
    TM->>S: TokenizedGenerateReqInput<br/>(tokenized 输入 + 采样参数 + 多模态数据)
    TM->>S: TokenizedEmbeddingReqInput<br/>(embedding 请求)
    TM->>S: AbortReq (中止请求)
    TM->>S: UpdateWeightFromDiskReqInput (权重更新)
    TM->>S: LoadLoRAAdapterReqInput (加载 LoRA)

    Note over S,DM: ZMQ IPC 通信
    S->>DM: BatchTokenIDOutput (token ids 输出)
    S->>DM: BatchEmbeddingOutput (embedding 输出)

    Note over DM,TM: ZMQ IPC 通信
    DM->>TM: BatchStrOutput (解码后的文本输出)
    DM->>TM: BatchTokenIDOutput (原始 token ids)

    Note over DPC,S: ZMQ IPC 通信
    DPC->>S: TokenizedGenerateReqInput (路由后的请求)

    Note over E,S: ZMQ RPC 通信
    E->>S: RpcReqInput / RpcReqOutput (远程过程调用)
    E->>S: ProfileReq (性能分析)
```

**IPC 数据结构源码：** `python/sglang/srt/managers/io_struct.py`
