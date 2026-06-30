<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# Caching Acceleration

SGLang provides two complementary caching strategies for 扩散模型 Transformer (DiT) models. Both reduce denoising cost by skipping redundant computation, but they operate at different levels.

## 概览

SGLang 支持 two complementary caching approaches:

| Strategy | Scope | Mechanism | Best For |
|----------|-------|-----------|----------|
| **缓存-DiT** | Block-level | Skip individual transformer blocks dynamically | Advanced, higher speedup |
| **TeaCache** | Timestep-level | Skip entire denoising steps based on L1 similarity | Simple, built-in |

## 缓存-DiT

[缓存-DiT](https://github.com/vipshop/cache-dit) provides block-level caching with
advanced strategies like DBCache and TaylorSeer. It can achieve up to **1.69x speedup**.

See [cache_dit.md](cache_dit.md) for detailed configuration.

### 快速开始

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

### Key 功能

- **DBCache**: Dynamic block-level caching based on residual differences
- **TaylorSeer**: Taylor expansion-based calibration for optimized caching
- **SCM**: Step-level computation masking for additional speedup

## TeaCache

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

See [teacache.md](teacache.md) for detailed documentation.

### Quick 概览

- Tracks L1 distance between modulated inputs across timesteps
- When accumulated distance is below threshold, reuses cached residual
- Supports CFG with separate positive/negative caches

### 支持的模型

- Wan (wan2.1, wan2.2)
- Hunyuan (HunyuanVideo)
- Z-Image

For Flux and Qwen models, TeaCache is automatically disabled when CFG is enabled.

```{toctree}
:maxdepth: 1

cache_dit
teacache
```

## 参考资料

- [缓存-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
