<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# 性能

This section covers the main performance levers for SGLang 扩散模型: attention backends, caching acceleration, and profiling.

## 概览

| Optimization | Type | 说明 |
|--------------|------|-------------|
| **缓存-DiT** | Caching | Block-level caching with DBCache, TaylorSeer, and SCM |
| **TeaCache** | Caching | Timestep-level caching based on temporal similarity |
| **Attention Backends** | Kernel | Optimized attention implementations (FlashAttention, SageAttention, etc.) |
| **性能分析** | Diagnostics | PyTorch Profiler and Nsight Systems guidance |

## Start Here

- Use [Attention Backends](attention_backends.md) to choose the best backend for your model and hardware.
- Use [部署 Cookbook](deployment_cookbook.md) to choose CPU offload, FSDP, CFG parallelism, SP, and TP.
- Use [Caching Acceleration](cache/index.md) to reduce denoising cost with 缓存-DiT or TeaCache.
- Use [性能分析](profiling.md) when you need to diagnose a bottleneck rather than guess.

## Caching at a Glance

- [缓存-DiT](cache/cache_dit.md) is block-level caching for diffusers pipelines and higher speedup-oriented tuning.
- [TeaCache](cache/teacache.md) is timestep-level caching built into SGLang model families.

```{toctree}
:maxdepth: 1

attention_backends
deployment_cookbook
cache/index
profiling
```

## Current Baseline Snapshot

For Ring SP benchmark details, see:

- [Ring SP 性能](ring_sp_performance.md)

## 参考资料

- [缓存-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
