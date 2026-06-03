# DeepSeek V4 Pro vs GLM-5.1 对比

## 1. 基础架构

| 维度 | DeepSeek V4 Pro | GLM-5.1 |
|---|---|---|
| **开发商** | DeepSeek-AI | 智谱AI (Zhipu) |
| **发布日期** | 2026-04-24 | 2026-04-07 |
| **架构类型** | MoE + Hybrid Attention (CSA+HCA) | MoE + GlmMoeDSA (MLA + DSA) |
| **总参数** | **1.6T** | **754B** |
| **激活参数** | **49B** | **40B** |
| **激活比** | 3.1% | 5.3% |
| **层数** | 61 (全部MoE) | 78 (3 Dense + 75 MoE) |
| **Hidden Size** | 7168 | 6144 |
| **MoE Expert中间层** | 3072 | 2048 |
| **Dense FFN中间层** | — | 12288 |
| **总路由专家数** | 384 | 256 |
| **共享专家数** | 1 | 1 |
| **激活专家/每token** | 6 routed + 1 shared | 8 routed + 1 shared |
| **路由缩放因子** | 2.5 | 2.5 |
| **量化格式** | FP4(MoE专家) + FP8(其他) | FP8 (官方) / W4A8 (后处理) |
| **词表大小** | 129,280 | 154,880 |
| **许可证** | MIT | MIT |

## 2. 注意力机制（核心差异）

| | DeepSeek V4 Pro | GLM-5.1 |
|---|---|---|
| **注意力类型** | CSA + HCA 混合注意力 | MLA + DSA 稀疏注意力 |
| **Q头数** | 128 | 64 |
| **KV头数** | 1 (MQA) | 64 (MLA，通过latent展开) |
| **Head Dim** | 512 (448 nope + 64 rope) | 256 (192 nope + 64 rope) |
| **Q LoRA Rank** | 1536 | 2048 |
| **KV LoRA Rank** | 512 | 512 |
| **O LoRA Rank** | 1024 | — |
| **O Groups** | 16 | — |
| **位置编码** | RoPE (θ=10000, YaRN×16) | RoPE (NeoX/Llama风格, θ=1M) |
| **滑动窗口** | 128 | — |
| **HCA压缩** | 多层4×/128×压缩 | — |
| **DSA Indexer** | — | 32 heads, head_dim=128, topk=2048 |
| **上下文长度** | **1M** (1,048,576) | **200K** (202,752) |
| **最大输出** | 384K | 128K |

V4 Pro 用 CSA+HCA 替代 MLA，通过多层混合压缩（4×/128×）实现1M上下文推理 FLOPs 仅为 V3.2 的27%。GLM-5.1 用 MLA 压缩 KV 存储（~10×压缩），DSA 压缩注意力计算（仅关注 top-2048 keys），二者互补实现 200K 高效推理。

## 3. Benchmark 对比

| Benchmark | DeepSeek V4 Pro | GLM-5.1 | 胜出 |
|---|---|---|---|
| **GPQA Diamond** (综合) | **90.10** | 86.20 | V4 Pro (+3.9) |
| **HLE** (综合) | 48.20 | **52.30** | GLM-5.1 (+4.1) |
| **SWE-Bench Pro** (编程) | 55.40 | **58.40** | GLM-5.1 (+3.0) |
| **BrowseComp** (信息收集) | **83.40** | 79.30 | V4 Pro (+4.1) |
| **Terminal Bench 2.0** (工具使用) | **67.90** | 63.50 | V4 Pro (+4.4) |
| **IMO-AnswerBench** (数学推理) | **89.80** | 83.80 | V4 Pro (+6.0) |
| **AIME 2026** | — | 95.3 | — |

**总胜出**: V4 Pro 赢 4/6 项，GLM-5.1 赢 2/6 项（HLE + SWE-Bench Pro）。综合得分 V4 Pro 领先约 1.88 分。

## 4. 关键技术创新对比

| | DeepSeek V4 Pro | GLM-5.1 |
|---|---|---|
| **核心创新** | CSA+HCA 混合注意力、FP4+FP8 混合量化、Muon优化器 | MLA+DSA、Gated DeltaNet 线性注意力、长时Agent能力 |
| **后训练** | 两阶段：SFT+GRPO → On-policy Distillation | Slime异步强化学习 |
| **推理模式** | Non-think / Think High / Think Max | 思维链开关 |
| **高速推理** | — | 高速版 400 tokens/s |
| **MHC** | hc_mult=4, 多流形隐藏状态 | — |
| **Expert路由** | noaux_tc (Sigmoid + bias校正) | Sigmoid + e_score_correction_bias |

## 5. 2×H100 (16×80GB) 显存占用详细对比

### 5.1 硬件环境

- 2台 H100 服务器，每台8卡，80GB/GPU
- 总GPU数：16
- 总显存：1,280 GB
- 并行策略：TP=16（无EP）

### 5.2 模型权重显存

#### DeepSeek V4 Pro (MXFP4 量化)

| 组件 | 参数量 | 量化格式 | 字节/参数 | 总大小 |
|---|---|---|---|---|
| MoE路由专家 (384×61层) | 1,547.4B | MXFP4 | 0.5 | 773.7 GB |
| MoE共享专家 (1×61层) | 4.0B | MXFP4 | 0.5 | 2.0 GB |
| 注意力+嵌入+其他 | ~48.6B | FP8 | 1 | 48.6 GB |
| **合计** | **~1,600B** | — | — | **~824.3 GB** |

每GPU权重：824.3 / 16 = **~51.5 GB**

> 计算方法：每个MoE专家含 gate_proj(7168×3072) + up_proj(7168×3072) + down_proj(3072×7168) = 66,060,288 参数。
> 384专家 × 61层 × 66,060,288 = 1,547,395,786,112 ≈ 1.547T 参数。

#### GLM-5.1 (FP8 量化)

| 组件 | 参数量 | 量化格式 | 字节/参数 | 总大小 |
|---|---|---|---|---|
| 嵌入层 (embed_tokens) | 951.4M | FP8 | 1 | 0.95 GB |
| 3层Dense FFN (intermediate=12288) | 1.20B | FP8 | 1 | 1.20 GB |
| 75层MoE (256路由+1共享专家) | 740.3B | FP8 | 1 | 740.3 GB |
| 78层注意力 (MLA+DSA Indexer) | ~13.6B | FP8 | 1 | 13.6 GB |
| lm_head (untied) | 951.4M | FP8 | 1 | 0.95 GB |
| FP32保留模块 (indexer.weights_proj等) | ~0.3M | FP32 | 4 | 0.001 GB |
| 量化缩放因子+元数据 | — | — | — | ~3 GB |
| **合计** | **~754B** | — | — | **~760 GB** |

每GPU权重：760 / 16 = **~47.5 GB**

> MoE每专家参数：gate_proj(6144×2048) + up_proj(6144×2048) + down_proj(2048×6144) = 37,748,736。
> 256专家 × 75层 × 37,748,736 = 723,775,488,000 ≈ 723.8B 参数。
> 共享专家：75 × 37,748,736 ≈ 2.8B 参数。

### 5.3 KV Cache 显存

#### DeepSeek V4 Pro KV Cache

V4 Pro 使用 MQA + MLA 风格压缩 KV：
- kv_lora_rank = 512，qk_rope_head_dim = 64
- 每token每层 KV 存储 = (512 + 64) × dtype_size = 576 × dtype_size

| KV格式 | 每token每层 | 61层总量 | 说明 |
|---|---|---|---|
| BF16 | 1,152 bytes = 1.125 KB | 68.6 KB/token | MLA压缩latent |
| FP8 | 576 bytes = 0.5625 KB | 34.3 KB/token | MLA压缩latent |

> **注意**: V4 Pro 的 CSA/HCA 机制在滑动窗口(128 tokens)外使用4×/128×压缩，实际KV cache占用可能低于上述线性估算。上述为最坏情况（全量MLA latent存储）。

#### GLM-5.1 KV Cache

GLM-5.1 使用 MLA + DSA：
- kv_lora_rank = 512，qk_rope_head_dim = 64
- 每token每层 KV 存储 = (512 + 64) × dtype_size = 576 × dtype_size
- DSA Indexer 额外缓存：每层128 dims × 2 bytes ≈ 256 bytes（可忽略）

| KV格式 | 每token每层 | 78层总量 | 说明 |
|---|---|---|---|
| BF16 (MLA压缩) | 1,152 bytes = 1.125 KB | 87.75 KB/token | MLA压缩latent |
| FP8 (MLA压缩) | 576 bytes = 0.5625 KB | 43.9 KB/token | MLA压缩latent |
| BF16 (展开KV) | ~65.5 KB | ~5,109 KB/token | 64头×(256+256)×2 bytes，无MLA压缩 |

> MLA压缩比约 **10.3×**：89 KB/token vs 5,109 KB/token (BF16)。使用 flash-mla 内核可直接消费 latent，无需展开。

### 5.4 每GPU显存汇总

假设 batch=1，75% 剩余显存用于 KV Cache，25% 用于激活/CUDA工作区/框架开销。

| 项目 | DeepSeek V4 Pro | GLM-5.1 |
|---|---|---|
| **GPU总显存** | 80 GB | 80 GB |
| **模型权重** | **51.5 GB** | **47.5 GB** |
| 剩余可用 | 28.5 GB | 32.5 GB |
| KV Cache 可用 (~75%) | **~21.4 GB** | **~24.4 GB** |
| 激活/工作区 (~25%) | ~7.1 GB | ~8.1 GB |

### 5.5 可支持的上下文长度

| KV格式 | DeepSeek V4 Pro | GLM-5.1 |
|---|---|---|
| **BF16 MLA** | 21.4GB / 68.6KB ≈ **312K tokens** | 24.4GB / 87.75KB ≈ **278K tokens** |
| **FP8 MLA** | 21.4GB / 34.3KB ≈ **624K tokens** | 24.4GB / 43.9KB ≈ **556K tokens** |
| **模型最大上下文** | 1,048,576 (1M) | 202,752 (200K) |
| **能否跑满最大上下文** | BF16: 否 (312K < 1M)；FP8: 否 (624K < 1M) | BF16: 是 (278K > 200K)；FP8: 是 (556K > 200K) |

> V4 Pro 在2节点H100上**无法跑满1M上下文**，但可支持312K-624K tokens。如需1M上下文需增加节点或使用更强的KV压缩。
> GLM-5.1 在2节点H100上**可轻松跑满200K上下文**，且有约38-176%的KV余量。

### 5.6 显存占用可视化

```
DeepSeek V4 Pro (每GPU 80GB)
┌─────────────────────────────────────────────────┐
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 51.5GB 权重 │
│░░░░░░░░░░░░░░░░░░░░░ 21.4GB KV Cache          │
│                        7.1GB 激活/工作区         │
└─────────────────────────────────────────────────┘
 权重 64.4% │ KV 26.7% │ 其他 8.9%

GLM-5.1 (每GPU 80GB)
┌─────────────────────────────────────────────────┐
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 47.5GB 权重      │
│░░░░░░░░░░░░░░░░░░░░░░░░ 24.4GB KV Cache       │
│                         8.1GB 激活/工作区        │
└─────────────────────────────────────────────────┘
 权重 59.4% │ KV 30.5% │ 其他 10.1%
```

### 5.7 不同并行策略的显存对比

| 配置 | DS V4 Pro 权重/GPU | GLM-5.1 权重/GPU | DS V4 Pro 可行? | GLM-5.1 可行? |
|---|---|---|---|---|
| TP=8 (1节点H200 141GB) | 103 GB | 95 GB | 否 (>141GB加KV) | 勉强 (需FP8 KV) |
| TP=8 (1节点H100 80GB) | 103 GB | 95 GB | 否 (OOM) | 否 (OOM) |
| **TP=16 (2节点H100)** | **51.5 GB** | **47.5 GB** | **是** | **是** |
| TP=16 (2节点H200 141GB) | 51.5 GB | 47.5 GB | 是 (充裕) | 是 (充裕) |
| EP=8 + TP=16 (2节点H100) | 变化* | 变化* | 受限 (需DeepEP) | 受限 (需测试) |

> *EP模式下MoE权重按EP分片（每GPU仅存部分专家），但注意力权重仍TP分片。DS V4 Pro的EP部署需要Mellanox IB网络（详见`dsv4+sglang+4xh100.md`）。

## 6. 部署差异（实战角度）

| | DeepSeek V4 Pro | GLM-5.1 |
|---|---|---|
| **2×H100 推荐配置** | TP=16, MXFP4 | TP=16, FP8 |
| **单节点8×H100** | 不可行 (权重需103GB > 80GB) | 不可行 (权重需95GB > 80GB) |
| **单节点8×H200** | 不可行 (权重+KV超141GB) | 官方推荐 (vLLM TP=8) |
| **4节点部署** | 需Mellanox IB (DeepEP) | 模型更小，并行策略更灵活 |
| **MXFP4/FP4量化** | 官方提供 | 需后处理 (W4A8) |
| **长上下文成本** | 1M FLOPs 极低，但2节点跑不满1M | 200K上限，2节点充裕 |
| **框架支持** | sglang, vLLM | vLLM (官方), llama.cpp |
| **KV Cache效率** | MQA + MLA → 极小 (68.6KB/token) | MLA → 极小 (87.75KB/token) |

## 7. 各自优势总结

**DeepSeek V4 Pro 优势**:
- 1.6T参数 + 仅49B激活 → 模型容量大但推理成本可控
- 1M超长上下文（CSA+HCA效率极高）
- KV cache 极小 (MQA + MLA, 68.6KB/token)
- 数学推理、Agent信息收集、工具使用更强
- 官方MXFP4量化，权重显存仅51.5GB/GPU

**GLM-5.1 优势**:
- 754B参数更小 → 权重仅47.5GB/GPU，多4GB用于KV
- 2×H100可跑满200K上下文（KV余量38-176%）
- SWE-Bench Pro 编程排第一 (58.4%)
- DSA稀疏注意力 → 注意力计算与序列长度解耦
- 长时自主Agent能力（8小时持续执行）
- 高速版 400 tokens/s 推理速度
- 部署灵活，不需要特殊网络硬件

## 8. 选型建议

| 场景 | 推荐 | 原因 |
|---|---|---|
| 超长上下文 (100K+) | V4 Pro | CSA/HCA专为长上下文设计，312K+ tokens |
| 数学/科学推理 | V4 Pro | GPQA 90.1, IMO 89.8 |
| 软件工程/编程 | GLM-5.1 | SWE-Bench Pro #1 (58.4%) |
| 低成本部署 | GLM-5.1 | 权重更小，KV余量更大 |
| Agent自动化 | GLM-5.1 | 8h持续自主执行 + 400 tok/s |
| 信息检索/浏览 | V4 Pro | BrowseComp 83.4 |
| 工具调用 | V4 Pro | Terminal Bench 67.9 |
| 极速推理 | GLM-5.1 | 高速版 400 tokens/s |
| 2×H100跑满最大上下文 | GLM-5.1 | V4 Pro跑不满1M，GLM-5.1可跑满200K |

**总结**: 在2×H100硬件上，GLM-5.1部署更轻松（权重更小、KV余量更大、可跑满200K上下文）；V4 Pro模型容量更大、推理能力更强，但1M上下文在2节点上无法跑满。两者都是国产第一梯队开源 MoE，各有所长。
