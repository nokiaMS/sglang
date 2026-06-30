<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# SGLang 扩散模型

SGLang 扩散模型 is a high-performance inference framework for image and video generation. It provides native SGLang pipelines, diffusers backend support, an OpenAI-compatible server, and an optimized kernel stack built on both precompiled `sgl-kernel` operators and JIT kernels for key inference paths.

## Key 功能

- Broad model support across Wan, Hunyuan, Cosmos3, Qwen-Image, FLUX, Z-Image, GLM-Image, and more
- Fast inference with `sgl-kernel`, JIT kernels, scheduler improvements, and caching acceleration
- Multiple interfaces: `sglang generate`, `sglang serve`, and an OpenAI-compatible API
- Multi-platform support for NVIDIA, AMD, Intel XPU, Ascend, Apple Silicon, and Moore Threads

## 快速开始

```bash
uv pip install "sglang[diffusion]" --prerelease=allow
```

```bash
sglang generate --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output
```

```bash
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

## Start Here

- [安装](installation.md): install SGLang 扩散模型 and platform dependencies
- [兼容性矩阵](compatibility_matrix.md): check model, optimization, and component override support
- [CLI](api/cli.md): run one-off generation jobs or launch a persistent server
- [兼容 OpenAI 的 API](api/openai_api.md): send image and video requests to the HTTP server
- [Attention Backends](performance/attention_backends.md): choose the best backend for your model and hardware
- [Caching Acceleration](performance/cache/index.md): use 缓存-DiT or TeaCache to reduce denoising cost
- [量化](quantization.md): load quantized transformer checkpoints
- [Contributing](contributing.md): contribution workflow, adding new models, and CI perf baselines

## Additional Documentation

- [后处理](api/post_processing.md): frame interpolation and upscaling
- [性能 概览](performance/index.md): overview of attention, caching, and profiling
- [环境变量](environment_variables.md): platform, caching, storage, and debugging configuration
- [支持新模型](support_new_models.md): implementation guide for new diffusion pipelines
- [CI 性能](ci_perf.md): performance baseline generation

## 参考资料

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [缓存-DiT](https://github.com/vipshop/cache-dit)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [xDiT](https://github.com/xdit-project/xDiT)
- [Diffusers](https://github.com/huggingface/diffusers)
