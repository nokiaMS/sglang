<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# TeaCache

> **注意**: This is one of two caching strategies available in SGLang.
> For an overview of all caching options, see [caching](../index.md).

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

## 概览

TeaCache works by:
1. Tracking the L1 distance between modulated inputs across consecutive timesteps
2. Accumulating the rescaled L1 distance over steps
3. When accumulated distance is below a threshold, reusing the cached residual
4. Supporting CFG (Classifier-Free Guidance) with separate positive/negative caches

## How It Works

### L1 Distance Tracking

At each denoising step, TeaCache computes the relative L1 distance between the current and previous modulated inputs:

```
rel_l1 = |current - previous|.mean() / |previous|.mean()
```

This distance is then rescaled using polynomial coefficients and accumulated:

```
accumulated += poly(coefficients)(rel_l1)
```

### 缓存 Decision

- If `accumulated >= threshold`: Force computation, reset accumulator
- If `accumulated < threshold`: Skip computation, use cached residual

### CFG 支持

For models that support CFG cache separation (Wan, Hunyuan, Z-Image), TeaCache maintains separate caches for positive and negative branches:
- `previous_modulated_input` / `previous_residual` for positive branch
- `previous_modulated_input_negative` / `previous_residual_negative` for negative branch

For models that don't support CFG separation (Flux, Qwen), TeaCache is automatically disabled when CFG is enabled.

## 配置

TeaCache is configured via `TeaCacheParams` in the sampling parameters:

```python
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

params = TeaCacheParams(
    teacache_thresh=0.1,           # Threshold for accumulated L1 distance
    coefficients=[1.0, 0.0, 0.0],  # Polynomial coefficients for L1 rescaling
)
```

### 参数

| Parameter | Type | 说明 |
|-----------|------|-------------|
| `teacache_thresh` | float | Threshold for accumulated L1 distance. Lower = more caching, faster but potentially lower quality |
| `coefficients` | list[float] | Polynomial coefficients for L1 rescaling. 模型-specific tuning |

### 模型-Specific Configurations

Different models may have different optimal configurations. The coefficients are typically tuned per-model to balance speed and quality.

## 支持的模型

TeaCache is built into the following model families:

| 模型 Family | CFG 缓存 Separation | Notes |
|--------------|---------------------|-------|
| Wan (wan2.1, wan2.2) | Yes | Full support |
| Hunyuan (HunyuanVideo) | Yes | To be supported |
| Z-Image | Yes | To be supported |
| Flux | No | To be supported |
| Qwen | No | To be supported |


## 参考资料

- [TeaCache: Accelerating 扩散模型 模型 with Temporal Similarity](https://arxiv.org/abs/2411.14324)
