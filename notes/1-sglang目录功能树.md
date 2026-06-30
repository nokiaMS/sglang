# SGLang 目录功能树

本文梳理 `E:\codex_home\code\sglang` 下的目录职责。范围以**顶层目录**为主，并对代码阅读中最常用的关键子目录做展开；benchmark case、测试 case 等数量较多的子目录只按类别概括。

```text
E:\codex_home\code\sglang
├─ 3rdparty/
│  └─ amd/
│     第三方依赖或厂商相关补充代码。目前可见 AMD 相关内容，用于硬件/后端适配。
│
├─ assets/
│     项目图片、logo、文档素材等静态资源。README 和文档页面会引用这里的资源。
│
├─ benchmark/
│  ├─ asr/
│  ├─ benchmark_batch/
│  ├─ bench_in_batch_prefix/
│  ├─ bench_rope/
│  ├─ deepseek_v3/
│  ├─ gpt_oss/
│  ├─ hicache/
│  ├─ json_schema/
│  ├─ kernels/
│  ├─ lora/
│  ├─ scheduler/
│  └─ ...
│     性能和效果评测集合。覆盖吞吐/延迟、batch、prefix cache、HiCache、kernel、LoRA、
│     DeepSeek/GPT-OSS 等模型或功能场景，也包含 MMLU、GSM8K、MT-Bench 等任务评测。
│
├─ docker/
│  └─ configs/
│     Docker 镜像构建、运行和容器配置相关文件，用于部署或复现实验环境。
│
├─ docs/
│  ├─ advanced_features/
│  ├─ basic_usage/
│  ├─ developer_guide/
│  ├─ diffusion/
│  ├─ get_started/
│  ├─ performance_dashboard/
│  ├─ platforms/
│  ├─ references/
│  ├─ release_lookup/
│  ├─ supported_models/
│  └─ _static/
│     英文官方文档源码。覆盖安装、基础用法、高级特性、开发者指南、平台支持、
│     supported models、reference API、performance dashboard 等。
│
├─ docs_CN/
│  ├─ advanced_features/
│  ├─ basic_usage/
│  ├─ developer_guide/
│  ├─ diffusion/
│  ├─ get_started/
│  ├─ performance_dashboard/
│  ├─ platforms/
│  ├─ references/
│  ├─ release_lookup/
│  ├─ supported_models/
│  └─ _static/
│     中文官方文档源码，与 `docs/` 结构基本对应。
│
├─ docs_new/
│  ├─ cards/
│  ├─ cookbook/
│  ├─ docs/
│  ├─ fonts/
│  ├─ images/
│  ├─ logo/
│  ├─ scripts/
│  └─ src/
│     新版文档站点或新文档系统的源码与素材。包含页面源码、cookbook、图片、字体和构建脚本。
│
├─ examples/
│  ├─ assets/
│  ├─ chat_template/
│  ├─ checkpoint_engine/
│  ├─ frontend_language/
│  ├─ monitoring/
│  ├─ profiler/
│  ├─ runtime/
│  ├─ sagemaker/
│  └─ usage/
│     示例代码和使用样例。覆盖 runtime 调用、chat template、监控、profiler、checkpoint、
│     SageMaker 部署和前端语言接口等。
│
├─ experimental/
│  └─ sgl-router/
│     实验性功能区。`sgl-router` 是早期/实验 router 相关实现或验证代码，
│     与正式 `sgl-model-gateway/` 的定位不同。
│
├─ notes/
│  ├─ final/
│  ├─ kvcache/
│  ├─ kv_router/
│  ├─ radix_cache/
│  ├─ sgl-model-gateway/
│  ├─ startup/
│  ├─ user_req/
│  └─ notes.md
│     本地分析笔记、源码阅读记录和专题文档。不是上游主功能代码，但适合沉淀调研结论。
│
├─ proto/
│  └─ sglang/
│     Protocol Buffers 定义目录。主要用于 gRPC 协议、跨语言接口和生成代码。
│
├─ python/
│  ├─ sglang/
│  │  ├─ benchmark/
│  │  ├─ cli/
│  │  ├─ eval/
│  │  ├─ jit_kernel/
│  │  ├─ lang/
│  │  ├─ multimodal_gen/
│  │  ├─ srt/
│  │  └─ test/
│  └─ tools/
│     SGLang Python 包主体。既包含用户侧 frontend/lang API，也包含核心 serving runtime。
│
│     关键子目录：
│     ├─ python/sglang/lang/
│     │     SGLang frontend language 层，提供面向用户的结构化生成/编程接口。
│     ├─ python/sglang/cli/
│     │     Python CLI 入口和命令行工具。
│     ├─ python/sglang/srt/
│     │     SGLang Runtime 核心代码，是推理服务主实现。
│     └─ python/tools/
│           辅助工具脚本。
│
│     srt 关键模块：
│     ├─ arg_groups/              启动参数分组和 CLI 参数组织。
│     ├─ batch_overlap/           batch overlap 相关调度/执行优化。
│     ├─ compilation/             编译、capture、graph 或 kernel 编译相关逻辑。
│     ├─ connector/               外部系统或组件连接器。
│     ├─ constrained/             constrained decoding / structured output 支持。
│     ├─ disaggregation/          prefill/decode 分离和跨节点协作。
│     ├─ distributed/             分布式执行、通信和并行相关基础设施。
│     ├─ entrypoints/             server/API 启动入口。
│     ├─ eplb/                    Expert Parallel Load Balancing。
│     ├─ function_call/           tool/function call 解析与格式处理。
│     ├─ grpc/                    SGLang gRPC 协议和服务端/客户端支持。
│     ├─ hardware_backend/        CUDA/ROCm/NPU/TPU/MLX 等硬件后端适配。
│     ├─ layers/                  模型层、attention、quantization 等底层实现。
│     ├─ lora/                    LoRA 和 multi-LoRA serving。
│     ├─ managers/                scheduler、tokenizer、detokenizer 等 manager。
│     ├─ mem_cache/               KV cache、radix cache、HiCache、memory pool 核心。
│     ├─ models/                  模型适配层，支持 Llama/Qwen/DeepSeek/GLM 等。
│     ├─ model_executor/          模型执行器、forward 执行路径。
│     ├─ model_loader/            权重加载、格式适配和加载优化。
│     ├─ multimodal/              多模态输入、图像/视频等支持。
│     ├─ observability/           metrics、日志和观测能力。
│     ├─ parser/                  输出解析、reasoning parser 等。
│     ├─ platforms/               平台抽象与平台能力检测。
│     ├─ sampling/                采样参数、logits processor、sampling 实现。
│     ├─ speculative/             speculative decoding。
│     ├─ tokenizer/               tokenizer 加载和运行时 tokenization。
│     └─ utils/                   通用工具函数。
│
├─ rust/
│  └─ sglang-grpc/
│     Rust 侧 gRPC 相关 crate 或辅助实现。与 `proto/` 和 gateway 的 gRPC 路径配合使用。
│
├─ scripts/
│  ├─ ci/
│  ├─ ci_monitor/
│  ├─ code_sync/
│  ├─ playground/
│  └─ release/
│     项目脚本集合。包括 CI、CI 监控、代码同步、实验 playground 和 release 管理脚本。
│
├─ sgl-kernel/
│  ├─ benchmark/
│  ├─ cmake/
│  ├─ csrc/
│  ├─ include/
│  ├─ python/
│  └─ tests/
│     SGLang 自研 kernel 子项目。包含 C++/CUDA/底层 kernel 源码、头文件、Python binding、
│     benchmark 和测试。服务于 attention、sampling、quantization 等高性能路径。
│
├─ sgl-model-gateway/
│  ├─ benches/
│  ├─ bindings/
│  ├─ e2e_test/
│  ├─ examples/
│  ├─ scripts/
│  ├─ src/
│  └─ tests/
│     SGLang Model Gateway / Router。负责模型网关、worker 注册、负载均衡、HTTP/gRPC/OpenAI
│     兼容路由、PD 路由、限流、重试、观测、MCP、history backend 等。
│
│     src 关键模块：
│     ├─ config/                  Router/Gateway 配置类型和 builder。
│     ├─ core/                    worker registry、job queue、workflow、worker abstraction。
│     ├─ observability/           Gateway metrics、logging、trace。
│     ├─ policies/                路由策略，包括 cache_aware、prefix_hash、power_of_two 等。
│     ├─ routers/                 HTTP、PD、gRPC、OpenAI 等数据面 router。
│     └─ wasm/                    WASM 扩展相关路由/执行能力。
│
└─ test/
   ├─ lm_eval_configs/
   ├─ manual/
   ├─ registered/
   └─ srt/
      测试目录。包含手工测试、注册测试、SRT runtime 测试、lm-eval 配置等。
```

## 阅读建议

如果目标是理解 SGLang runtime 主流程，优先看：

```text
python/sglang/srt/entrypoints/
python/sglang/srt/managers/
python/sglang/srt/mem_cache/
python/sglang/srt/model_executor/
python/sglang/srt/models/
```

如果目标是理解 KV cache / radix cache / HiCache，优先看：

```text
python/sglang/srt/mem_cache/
python/sglang/srt/managers/scheduler*
benchmark/hicache/
notes/kvcache/
notes/radix_cache/
```

如果目标是理解 KV router / cache-aware 路由，优先看：

```text
sgl-model-gateway/src/policies/cache_aware.rs
sgl-model-gateway/src/policies/tree.rs
sgl-model-gateway/src/policies/prefix_hash.rs
sgl-model-gateway/src/policies/registry.rs
sgl-model-gateway/src/routers/
notes/kv_router/
```

如果目标是理解高性能 kernel，优先看：

```text
sgl-kernel/csrc/
sgl-kernel/include/
sgl-kernel/python/
python/sglang/srt/layers/
python/sglang/srt/model_executor/
```

如果目标是部署和使用，优先看：

```text
docs/get_started/
docs/basic_usage/
docs/advanced_features/
examples/
docker/
sgl-model-gateway/README.md
```
