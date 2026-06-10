# SGLang 模块扇入扇出分析

## 1. 顶层模块依赖关系

```mermaid
graph TD
    CLI[cli/<br/>命令行入口] --> SRT[srt/<br/>核心运行时]
    LANG[lang/<br/>前端语言API] --> SRT
    BENCH[bench_*<br/>基准测试] --> SRT
    JIT[jit_kernel/<br/>JIT内核] --> SRT
    EVAL[eval/<br/>评估工具] --> SRT
    SGL_KERNEL[sgl-kernel/<br/>AOT CUDA内核] --> SRT

    SRT --> SERVER_ARGS[server_args<br/>服务参数]
    SRT --> CONFIGS[configs/<br/>模型/设备配置]
    SRT --> DISTRIBUTED[distributed/<br/>分布式通信]
```

## 2. SRT 核心三进程架构与数据流

```mermaid
graph LR
    subgraph 主进程
        EP[entrypoints/<br/>HTTP/gRPC/Python API] --> TM[TokenizerManager<br/>分词 & 多模态预处理]
    end

    subgraph 调度子进程
        TM -->|ZMQ PUSH<br/>TokenizedGenerateReqInput| SCH[Scheduler<br/>调度 & 批管理]
        SCH --> TpW[TpModelWorker<br/>批转换桥梁]
        TpW --> MR[ModelRunner<br/>模型前向推理]
        MR --> MODEL[Models<br/>nn.Module模型]
        MODEL --> LAYERS[Layers<br/>注意力/MoE/采样]
    end

    subgraph 反分词子进程
        SCH -->|ZMQ PUSH<br/>BatchTokenIDOutput| DM[DetokenizerManager<br/>反分词 & 停止检测]
        DM -->|ZMQ PUSH<br/>BatchStrOutput| TM
    end

    LAYERS --> MEM[mem_cache/<br/>KV缓存 & 前缀树]
    SCH --> MEM
```

## 3. 核心模块扇入扇出图

```mermaid
graph TD
    %% ===== 入口层 =====
    EP[entrypoints/<br/>扇出: 6]

    %% ===== 管理层 =====
    TM[TokenizerManager<br/>扇入: 25 / 扇出: 8]
    SCH[Scheduler<br/>扇入: 40+ / 扇出: 12]
    DM[DetokenizerManager<br/>扇入: 5 / 扇出: 3]
    TpW[TpModelWorker<br/>扇入: 10 / 扇出: 6]
    DP[DataParallelController<br/>扇入: 8 / 扇出: 4]

    %% ===== 执行层 =====
    MR[ModelRunner<br/>扇入: 50+ / 扇出: 15]
    FB[ForwardBatch<br/>扇入: 5 / 扇出: 8]
    CGR[CudaGraphRunner<br/>扇入: 8 / 扇出: 5]

    %% ===== 模型层 =====
    MODELS[Models<br/>扇入: 3 / 扇出: 1]
    LAYERS[Layers<br/>扇入: 15 / 扇出: 5]
    ATTN[Attention Backends<br/>扇入: 3 / 扇出: 1]
    MOE[MoE<br/>扇入: 4 / 扇出: 2]
    QUANT[Quantization<br/>扇入: 4 / 扇出: 1]
    SAMPLER[Sampler<br/>扇入: 6 / 扇出: 4]

    %% ===== 基础设施层 =====
    MEM[mem_cache/<br/>扇入: 12 / 扇出: 8]
    IO[io_struct<br/>扇出: 10]
    SA[server_args<br/>扇出: 30+]
    CFG[configs/<br/>扇出: 15]
    DIST[distributed/<br/>扇出: 15]
    SPEC[speculative/<br/>扇入: 6 / 扇出: 4]
    CONS[constrained/<br/>扇入: 2 / 扇出: 1]
    LORA[LoRA<br/>扇入: 4 / 扇出: 3]
    DISAGG[disaggregation/<br/>扇入: 3 / 扇出: 2]

    %% ===== 入口 -> 管理层 =====
    EP --> TM
    EP --> SCH
    EP --> DM
    EP --> DP

    %% ===== 管理层内部 =====
    TM -->|ZMQ| SCH
    SCH -->|ZMQ| DM
    DM -->|ZMQ| TM
    SCH --> TpW
    DP --> TM

    %% ===== 管理层 -> 执行层 =====
    TpW --> MR
    TpW --> FB
    MR --> CGR

    %% ===== 执行层 -> 模型层 =====
    MR --> MODELS
    MR --> LAYERS
    MR --> SAMPLER
    MODELS --> LAYERS
    LAYERS --> ATTN
    LAYERS --> MOE
    LAYERS --> QUANT

    %% ===== 各层 -> 基础设施 =====
    EP -.-> IO
    TM -.-> IO
    SCH -.-> IO
    DM -.-> IO
    TM -.-> SA
    SCH -.-> SA
    MR -.-> SA
    SCH -.-> CFG
    MR -.-> CFG
    TM -.-> CFG
    MR -.-> DIST
    SCH -.-> DIST
    LAYERS -.-> DIST
    SCH --> SPEC
    SCH --> CONS
    SCH --> LORA
    SCH --> DISAGG
    MR --> MEM
    SCH --> MEM
    LAYERS --> MEM

    %% ===== 样式 =====
    classDef entry fill:#e74c3c,color:#fff,stroke:#c0392b
    classDef manager fill:#3498db,color:#fff,stroke:#2980b9
    classDef executor fill:#2ecc71,color:#fff,stroke:#27ae60
    classDef model fill:#9b59b6,color:#fff,stroke:#8e44ad
    classDef infra fill:#f39c12,color:#fff,stroke:#e67e22

    class EP entry
    class TM,SCH,DM,TpW,DP manager
    class MR,FB,CGR executor
    class MODELS,LAYERS,ATTN,MOE,QUANT,SAMPLER model
    class MEM,IO,SA,CFG,DIST,SPEC,CONS,LORA,DISAGG infra
```

## 4. 数据结构流转图

```mermaid
flowchart LR
    A["GenerateReqInput<br/>(io_struct)"] -->|TokenizerManager| B["TokenizedGenerateReqInput<br/>(io_struct)"]
    B -->|ZMQ → Scheduler| C["Req<br/>(schedule_batch)"]
    C -->|调度批处理| D["ScheduleBatch<br/>(schedule_batch)"]
    D -->|ForwardBatch.init_new| E["ForwardBatch<br/>(forward_batch_info)"]
    E -->|ModelRunner.forward| F["LogitsProcessorOutput<br/>(layers/logits_processor)"]
    F -->|Sampler采样| G["next_token_ids<br/>(Tensor)"]
    G -->|Scheduler处理| H["BatchTokenIDOutput<br/>(io_struct)"]
    H -->|ZMQ → Detokenizer| I["BatchStrOutput<br/>(io_struct)"]
    I -->|ZMQ → TokenizerManager| J["客户端响应<br/>(dict/stream)"]
```

## 5. 高扇出模块（被多方依赖）

| 模块 | 扇出数 | 依赖者 |
|------|--------|--------|
| `server_args` | 30+ | 几乎所有模块 |
| `managers/io_struct` | 10 | TokenizerManager, Scheduler, DetokenizerManager, Engine, 各entrypoint |
| `configs/model_config` | 15 | ModelRunner, Scheduler, TokenizerManager, TpModelWorker, model_loader, kv_cache_builder |
| `distributed/` | 15 | ModelRunner, Scheduler, TpModelWorker, layers, model_loader |
| `model_executor/forward_batch_info` | 8 | ModelRunner, TpModelWorker, Scheduler, schedule_batch, layers, speculative |

## 6. 高扇入模块（依赖多方）

| 模块 | 扇入数 | 主要依赖来源 |
|------|--------|-------------|
| `model_executor/model_runner` | 50+ | model_loader, layers/*, mem_cache, configs, distributed, sampling, speculative, lora |
| `managers/scheduler` | 40+ | schedule_batch, schedule_policy, io_struct, tp_worker, speculative, constrained, lora, mem_cache, disaggregation, observability, session |
| `managers/tokenizer_manager` | 25 | io_struct, mm_utils, multimodal_processor, sampling_params, spec_info, lora_registry, configs |
| `entrypoints/engine` | 20 | TokenizerManager, Scheduler, DetokenizerManager, DataParallelController, io_struct, server_args |

## 7. Scheduler 子组件扇入扇出

```mermaid
graph TD
    SCH[Scheduler 主类<br/>扇入: 40+]

    SCH --> CMP[scheduler_components/]
    CMP --> IPC[IPCChannels<br/>ZMQ通道管理]
    CMP --> RR[RequestReceiver<br/>请求接收]
    CMP --> OS[OutputStreamer<br/>输出流]
    CMP --> BRP[BatchResultProcessor<br/>批结果处理]
    CMP --> LRP[LogprobResultProcessor<br/>logprob处理]
    CMP --> MR2[MetricsReporter<br/>指标上报]
    CMP --> PS[PoolStatsObserver<br/>池状态观察]
    CMP --> LI[LoadInquirer<br/>负载查询]
    CMP --> WU[WeightUpdater<br/>权重更新]
    CMP --> PM[ProfilerManager<br/>性能分析]
    CMP --> IC[InvariantChecker<br/>不变量检查]
    CMP --> IS[IdleSleeper<br/>空闲休眠]
    CMP --> FW[FlushWrapper<br/>刷新包装]
    CMP --> DPA[DPAttnAdapter<br/>DP注意力适配]
    CMP --> KVE[KvEventsPublisher<br/>KV事件发布]
    CMP --> OSE[OutputSender<br/>输出发送]

    SCH --> SP[SchedulePolicy<br/>调度策略]
    SCH --> SB[ScheduleBatch<br/>批数据结构]
    SCH --> GRAM[GrammarManager<br/>语法约束]
    SCH --> LDR[LoRADrainer<br/>LoRA调度]
    SCH --> SPEC2[SpecWorker<br/>投机解码]
    SCH --> SC[SessionController<br/>会话管理]
    SCH --> DISAGG2[DisaggMixin<br/>分离式推理]
    SCH --> MEM2[KVCache<br/>缓存管理]
