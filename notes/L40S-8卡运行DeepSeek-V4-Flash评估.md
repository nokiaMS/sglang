# L40S 8 卡运行 DeepSeek-V4-Flash 可行性评估

## 1. 评估结论

基于当前 SGLang 仓库 commit `fd8679510`，**8×L40S 不能开箱即用地运行 DeepSeek-V4-Flash**。

主要结论如下：

- **显存容量基本可行**：官方 FP4+FP8 混合精度 checkpoint 约 160GB，8×L40S 聚合显存为 384GB。
- **当前内核不兼容**：L40S 是 Ada SM89，而 SGLang 的 DeepSeek V4 FlashMLA Attention 当前要求 SM90a 及以上。
- **原生 FP4 Expert 路径不兼容**：当前 DeepSeek V4 MXFP4 Marlin 实现明确要求 SM90 或 SM120。
- **全 FP8 不能直接解决问题**：虽然可以规避 MXFP4 Expert，但仍受 Attention 内核限制，并且约 294GB 的 FP8 权重只留下较小的运行时显存余量。
- **多卡性能风险较高**：L40S 没有 NVLink，TP8 只能依赖 PCIe 拓扑完成频繁的跨卡通信。

因此，当前应将该配置判定为：

> 硬件显存满足 DeepSeek-V4-Flash 的权重容量要求，但当前 SGLang 缺少 SM89 的 DeepSeek V4 Attention 和原生 MXFP4 MoE Kernel，不能直接运行；需要专项内核适配，且不适合作为近期生产部署方案。

---

## 2. 硬件与模型基础信息

### 2.1 L40S

NVIDIA L40S 的主要规格为：

- Ada Lovelace 架构。
- Compute Capability 8.9，即 SM89。
- 单卡 48GB GDDR6 ECC。
- 单卡显存带宽 864GB/s。
- 支持 FP8 Tensor Core。
- PCIe Gen4 x16，标称双向 64GB/s。
- 不支持 NVLink。

8 卡总物理显存为：

```text
8 × 48GB = 384GB
```

参考资料：

- [NVIDIA L40S 产品规格](https://www.nvidia.com/en-au/data-center/l40s/)
- [NVIDIA CUDA GPU Compute Capability 列表](https://developer.nvidia.com/cuda/gpus)

需要注意，384GB 是物理显存总量，不是一个自动共享的统一地址空间。模型必须通过 TP、EP 或其他并行方式将权重和运行状态分布到不同 GPU。

### 2.2 DeepSeek-V4-Flash

DeepSeek 官方资料显示：

- 总参数量约 284B。
- 每 Token 激活参数约 13B。
- 最大上下文长度为 1M Token。
- 官方 Instruct checkpoint 使用 FP4+FP8 混合精度。
- MoE Expert 参数使用 FP4，大部分其他参数使用 FP8。

官方 Hugging Face 仓库总体积约 160GB：

- [DeepSeek-V4-Flash 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [DeepSeek-V4-Flash 文件列表](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main)

SGLang 还提供了一个全 FP8 重打包版本，仓库总体积约 294GB：

- [sgl-project/DeepSeek-V4-Flash-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8/tree/main)

---

## 3. 显存容量评估

### 3.1 官方 FP4+FP8 混合精度 checkpoint

按仓库总体积 160GB 粗略平均到 8 张卡：

```text
160GB ÷ 8 ≈ 20GB/卡
48GB - 20GB ≈ 28GB/卡
```

从纯权重容量看，8×L40S 有较充分的空间装载官方 checkpoint。剩余空间还需容纳：

- KV Cache。
- C4/C128 压缩 Attention 状态池。
- Indexer Cache。
- CUDA Graph Pool。
- Attention、MoE 和通信 Workspace。
- 临时激活和 JIT Kernel Workspace。
- NCCL Buffer 和框架运行时开销。

实际分片可能不完全均匀，部分参数或 Buffer 也可能按 Rank 复制。因此 20GB/卡只能用于初步容量判断，不能作为最终峰值显存结论。

### 3.2 全 FP8 checkpoint

按 294GB 粗略平均到 8 张卡：

```text
294GB ÷ 8 ≈ 36.75GB/卡
48GB - 36.75GB ≈ 11.25GB/卡
```

该路径理论上仍能装下权重，但只剩约 11GB/卡的名义空间。考虑模型加载峰值、CUDA Context、KV Cache、压缩状态、Workspace 和 CUDA Graph 后，OOM 风险较高。

如果未来完成 SM89 Attention 适配并尝试全 FP8 路径，应从以下限制开始：

- TP8。
- Batch Size 1。
- 短上下文。
- `--max-running-requests 1`。
- 关闭 MTP。
- 关闭 CUDA Graph。
- 使用较小的 Chunked Prefill。
- 逐步增加 KV Cache 和并发，而不是直接按生产参数启动。

### 3.3 BF16

284B 参数若按 BF16 粗略估算，仅权重约需：

```text
284B × 2 Bytes ≈ 568GB
```

超过 8×L40S 的 384GB 聚合显存，因此 BF16 全权重路径不可行。

---

## 4. 关键阻塞一：DeepSeek V4 Attention 不支持 SM89

当前 DeepSeek V4 CUDA Attention Backend 的核心路径依赖 FlashMLA，包括：

- SWA Attention。
- C4/C128 压缩稀疏 Attention。
- Sparse Prefill。
- Decode Attention。
- FP8 KV Cache 读取。

仓库构建脚本明确说明：

```cmake
# The FlashMLA kernels only work on hopper and require CUDA 12.4 or later.
```

并以以下架构参数编译 Hopper Kernel：

```cmake
-gencode=arch=compute_90a,code=sm_90a
```

源码证据：

- `python/sglang/kernels/aot/cmake/flashmla.cmake:29`
- `python/sglang/kernels/aot/cmake/flashmla.cmake:34`
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`

L40S 是 SM89，无法执行 SM90a 专用 Kernel，预计会在扩展加载、调度或第一次 Attention Forward 时出现设备架构不支持或 `no kernel image is available` 类错误。

SGLang 的普通 FA3 虽然可以支持 L40S，但不能直接替代 DeepSeek V4 的混合压缩 Attention，因为两者的缓存布局、压缩层、Indexer 结果和 Attention 语义不同。因此简单设置普通 `fa3` Attention Backend 不能解决问题。

---

## 5. 关键阻塞二：原生 MXFP4 MoE Expert 不支持 SM89

DeepSeek-V4-Flash 官方 Instruct checkpoint 的 Expert 参数使用 FP4。SGLang 在 Hopper 上通常通过 W4A16 Marlin 路径运行这些 Expert。

当前实现有明确架构检查：

```python
if not is_sm90_supported() and not is_sm120_supported():
    raise RuntimeError("MXFP4 Marlin requires SM90 or SM120.")
```

源码位置：

- `python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py:118`
- `python/sglang/srt/layers/quantization/mxfp4.py:522`

因此 L40S 使用以下配置会在权重后处理阶段失败：

```bash
--moe-runner-backend marlin
```

FlashInfer MXFP4 路径当前也主要面向 SM90、SM100 和 SM120，不覆盖 SM89。

---

## 6. 全 FP8 或加载时反量化能否绕过问题

可以考虑两个方向：

1. 使用 `sgl-project/DeepSeek-V4-Flash-FP8`。
2. 加载官方混合 checkpoint 时，将 FP4 Expert 转为 FP8。

它们能够避免直接执行原生 MXFP4 Expert Kernel，并且 L40S 本身具有 FP8 Tensor Core。但是这两个方向都不能直接让模型运行：

- DeepSeek V4 Attention 的 SM90a FlashMLA 限制仍然存在。
- FP8 权重显存约 294GB，TP8 后每卡运行时空间较紧。
- 加载时 FP4→FP8 转换还可能产生临时峰值显存。
- 需要验证 FP8 Block Quant、Router、MoE 和共享 Expert 的 SM89 Kernel 路径。

因此，全 FP8 只能作为完成 SM89 Attention 适配后的候选验证路线。

---

## 7. 多卡互联与性能风险

L40S 不支持 NVLink。8 卡服务器通常通过 PCIe Switch 和 CPU Root Complex 互联，具体性能取决于主机拓扑。

DeepSeek-V4-Flash 使用 TP8 时会频繁出现：

- Attention TP all-reduce/all-gather。
- MoE 分片后的数据聚合。
- reduce-scatter。
- MTP Draft/Verify 的额外同步。
- Prefill 大 Token Batch 下的跨卡数据移动。

PCIe Gen4 x16 的标称双向带宽为 64GB/s，明显低于 Hopper NVLink/NVSwitch 系统的卡间带宽。即使完成 SM89 Kernel 移植，TP8 Decode 延迟和吞吐也可能受到显著影响。

实际评估前必须执行：

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
```

并确认：

- GPU 之间是否支持 CUDA P2P。
- 是否跨 NUMA Node。
- 是否被 ACS/IOMMU 配置阻断 P2P。
- NCCL 实际使用 P2P、SHM 还是 Socket 路径。

---

## 8. SGLang 官方验证矩阵

SGLang DeepSeek V4 Cookbook 当前列出的 DeepSeek-V4-Flash 单机硬件包括：

- B200/B300/GB200/GB300。
- H200 TP4。
- H100 TP8。
- RTX PRO 6000 TP2。

其中没有 L40S：

- [SGLang DeepSeek V4 Cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)

这与当前仓库中的 SM90+ Kernel 限制相符，因此不能把 L40S 理解成“仅未做性能认证”；当前存在明确的软件兼容性缺口。

---

## 9. 可行性分级

### 9.1 当前直接部署

结论：**不可行。**

确定阻塞包括：

- SM89 无法执行当前 DeepSeek V4 FlashMLA Attention Kernel。
- 官方 checkpoint 的 MXFP4 Marlin Expert 路径拒绝 SM89。

### 9.2 完成内核改造后进行功能验证

结论：**理论可行，但属于专项适配项目。**

至少需要：

1. 为 SM89 实现 DeepSeek V4 SWA、CSA/HCA Prefill 和 Decode Attention fallback。
2. 支持对应的 FP8 KV Cache、Indexer 结果和额外压缩 Cache 布局。
3. 在以下 MoE 方案中选择一种：
   - 将 MXFP4 W4A16/Marlin 路径移植到 SM89；
   - 使用全 FP8 checkpoint；
   - 加载时将 FP4 Expert 转换为 FP8。
4. 验证 MHC、Compressor、C4 Indexer、RoPE 和 fused cache-store Kernel 在 SM89 上的正确性。
5. 验证 TP8 NCCL 通信、P2P 能力和实际拓扑。
6. 完成算子级、单层、短请求和完整模型正确性测试。

### 9.3 生产部署

结论：**不建议。**

即使适配完成，仍存在：

- 缺少官方正确性和性能验证。
- PCIe-only TP8 通信瓶颈。
- 全 FP8 路径显存余量较小。
- 1M Context 和高并发容量无法直接保证。
- 后续 SGLang Kernel 更新需要持续维护 SM89 分支。

---

## 10. 如果必须使用 L40S 的建议路线

### 路线 A：全 FP8，优先减少适配范围

目标 checkpoint：

```text
sgl-project/DeepSeek-V4-Flash-FP8
```

优点：

- 避免原生 MXFP4 Expert Kernel。
- L40S 有 FP8 Tensor Core。
- 主要集中解决 DeepSeek V4 Attention 的 SM89 fallback。

缺点：

- 权重约 294GB，运行时显存非常紧。
- 仍需验证 FP8 MoE 和 Block Quant 路径。
- 上下文和并发能力受限。

### 路线 B：保留官方 FP4+FP8 checkpoint

优点：

- 权重约 160GB，显存余量明显更好。
- 更适合保留 KV Cache 和运行时 Workspace。

缺点：

- 除 Attention 外，还必须实现或移植 SM89 MXFP4 Expert Kernel。
- 内核适配范围和正确性验证工作量更大。

### 建议选择

- 如果目标是尽快完成“能生成正确文本”的 PoC，优先考虑路线 A。
- 如果目标是后续高并发或长上下文，路线 B 的显存结构更合理，但开发成本更高。
- 如果目标是近期生产上线，应改用 SGLang 已验证的 H100、H200 或 Blackwell 平台，而不是启动 SM89 移植项目。

---

## 11. 适配后的最小 Bring-up 顺序

如果后续完成 SM89 Kernel 开发，建议按以下顺序验证：

1. 单个 DeepSeek V4 Attention Kernel 正确性测试。
2. Compressor 和 Indexer 单元测试。
3. 单层 Decoder Forward。
4. TP8 模型加载，不分配大 KV Cache。
5. BS=1、短输入、输出 1 Token。
6. 关闭 CUDA Graph、MTP、DP Attention 和复杂 MoE 通信后端。
7. 完成短 Decode 正确性测试。
8. 增加 Prefill 长度和输出长度。
9. 开启 CUDA Graph。
10. 增加并发和 KV Cache。
11. 最后评估 MTP、DP/EP 和生产吞吐。

建议首轮限制：

```text
TP=8
Batch Size=1
Input Length<=512
Output Length<=16
Max Running Requests=1
MTP=Off
CUDA Graph=Off
MoE A2A Backend=None
```

---

## 12. 最终建议

| 目标 | 建议 |
| --- | --- |
| 直接启动官方模型 | 不执行，当前确定会遇到架构 Kernel 阻塞 |
| 验证权重能否装下 | 可以，官方混合 checkpoint 容量上满足 |
| 做 SM89 技术 PoC | 可以立项，但应定义为内核适配项目 |
| 快速 PoC 路线 | 全 FP8 checkpoint + SM89 Attention fallback，接受短上下文和低并发 |
| 高并发/长上下文 | 不推荐 L40S，优先选择 H200/Blackwell |
| 近期生产部署 | No-Go |

本评估中的“不能运行”指当前 SGLang 实现不能在 L40S 上开箱即用地完成正确推理，不代表 L40S 的理论算力或显存永远无法承载该模型。若未来 SGLang 或上游 FlashMLA/MoE Kernel 增加 SM89 fallback，应重新执行源码兼容性、显存峰值、正确性和性能评估。
