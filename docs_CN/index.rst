SGLang 中文文档
================

.. raw:: html

  <a class="github-button" href="https://github.com/sgl-project/sglang" data-size="large" data-show-count="true" aria-label="Star sgl-project/sglang on GitHub">Star</a>
  <a class="github-button" href="https://github.com/sgl-project/sglang/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork sgl-project/sglang on GitHub">Fork</a>
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <br></br>

SGLang 是一个面向大语言模型和多模态模型的高性能服务框架。
它面向从单卡 GPU 到大规模分布式集群的多种部署形态，目标是在推理服务中提供低延迟和高吞吐。
核心能力包括：

- **高速运行时**：通过 RadixAttention 前缀缓存、零开销 CPU 调度器、Prefill-Decode 分离、投机解码、连续批处理、Paged Attention、张量/流水线/专家/数据并行、结构化输出、分块 prefill、量化（FP4/FP8/INT4/AWQ/GPTQ）以及 multi-LoRA batching，提供高效的推理服务。
- **广泛的模型支持**：支持多种语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等）、嵌入模型（e5-mistral、gte、mcdse）、奖励模型（Skywork）和扩散模型（WAN、Qwen-Image），并提供便捷的新模型扩展能力。兼容大多数 Hugging Face 模型和 OpenAI API。
- **丰富的硬件支持**：可运行在 NVIDIA GPU（GB200/B300/H100/A100/Spark/5090）、AMD GPU（MI355/MI300）、Intel Xeon CPU、Google TPU、昇腾 NPU 等硬件平台上。
- **活跃的社区**：SGLang 是开源项目，拥有活跃社区和广泛的行业采用，目前支撑全球超过 400,000 张 GPU 的工作负载。
- **RL 与后训练基础设施**：SGLang 是经过验证的 rollout 后端，被用于训练许多前沿模型；它原生支持 RL 集成，并被 AReaL、Miles、slime、Tunix、verl 等后训练框架采用。

.. toctree::
   :maxdepth: 1
   :caption: 快速开始

   get_started/install.md

.. toctree::
   :maxdepth: 1
   :caption: 基础用法

   basic_usage/send_request.ipynb
   basic_usage/openai_api.rst
   basic_usage/ollama_api.md
   basic_usage/offline_engine_api.ipynb
   basic_usage/native_api.ipynb
   basic_usage/sampling_params.md
   basic_usage/popular_model_usage.rst

.. toctree::
   :maxdepth: 1
   :caption: 高级功能

   advanced_features/server_arguments.md
   advanced_features/object_storage.md
   advanced_features/hyperparameter_tuning.md
   advanced_features/attention_backend.md
   advanced_features/speculative_decoding.ipynb
   advanced_features/adaptive_speculative_decoding.md
   advanced_features/structured_outputs.ipynb
   advanced_features/structured_outputs_for_reasoning_models.ipynb
   advanced_features/tool_parser.ipynb
   advanced_features/separate_reasoning.ipynb
   advanced_features/quantization.md
   advanced_features/quantized_kv_cache.md
   advanced_features/expert_parallelism.md
   advanced_features/dp_dpa_smg_guide.md
   advanced_features/lora.ipynb
   advanced_features/pd_disaggregation.md
   advanced_features/epd_disaggregation.md
   advanced_features/pipeline_parallelism.md
   advanced_features/hicache.rst
   advanced_features/pd_multiplexing.md
   advanced_features/vlm_query.ipynb
   advanced_features/dp_for_multi_modal_encoder.md
   advanced_features/cuda_graph_for_multi_modal_encoder.md
   advanced_features/piecewise_cuda_graph.md
   advanced_features/breakable_cuda_graph.md
   advanced_features/sgl_model_gateway.md
   advanced_features/deterministic_inference.md
   advanced_features/observability.md
   advanced_features/checkpoint_engine.md
   advanced_features/sglang_for_rl.md

.. toctree::
   :maxdepth: 2
   :caption: 支持的模型

   supported_models/text_generation/index
   supported_models/retrieval_ranking/index
   supported_models/specialized/index
   supported_models/extending/index

.. toctree::
   :maxdepth: 2
   :caption: SGLang Diffusion

   diffusion/index
   diffusion/installation
   diffusion/compatibility_matrix
   diffusion/api/cli
   diffusion/api/openai_api
   diffusion/performance/index
   diffusion/performance/ring_sp_performance
   diffusion/performance/attention_backends
   diffusion/performance/cache/index
   diffusion/quantization
   diffusion/contributing

.. toctree::
   :maxdepth: 1
   :caption: 硬件平台

   platforms/amd_gpu.md
   platforms/cpu_server.md
   platforms/tpu.md
   platforms/nvidia_jetson.md
   platforms/ascend/ascend_npu_support.rst
   platforms/xpu.md

.. toctree::
   :maxdepth: 1
   :caption: 开发者指南

   developer_guide/contribution_guide.md
   developer_guide/development_guide_using_docker.md
   developer_guide/development_jit_kernel_guide.md
   developer_guide/benchmark_and_profiling.md
   developer_guide/bench_serving.md
   developer_guide/evaluating_new_models.md

.. toctree::
   :maxdepth: 1
   :caption: 参考资料

   references/faq.md
   references/environment_variables.md
   references/production_metrics.md
   references/production_request_trace.md
   references/multi_node_deployment/multi_node_index.rst
   references/custom_chat_template.md
   references/frontend/frontend_index.rst
   references/post_training_integration.md
   references/release_lookup
   references/learn_more.md

.. toctree::
   :maxdepth: 1
   :caption: 安全致谢

   security/acknowledgements.md
