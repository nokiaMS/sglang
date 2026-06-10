# 4节点单实例部署 DeepSeek-V4-Pro 可行性分析

## 结论：不可行

在不安装 NVSHMEM/DeepEP 的条件下（ib7s400 网卡不支持），4节点32卡单实例部署 V4 Pro **不可行**。

## 环境

- 4节点，每节点8卡（H100 80GB），共32卡
- 网卡：ib7s400（不支持 NVSHMEM）

## 死路汇总

| 路径 | 问题 |
|------|------|
| TP=32, EP=1 | `intermediate_per_partition=96`，无 MoE 后端可用 |
| TP=32, EP>1, 无 DP-attention | MHC `n_local_groups=0`，shape 错误 |
| TP=32, EP>1, DP-attention, 无 DeepEP | `dp_scatter` 崩溃 + expert mapping=-1 |
| TP=32, EP>1, DP-attention, DeepEP | 需 NVSHMEM → 需 Mellanox IB → ib7s400 不支持 |
| Marlin + EP, moe_tp_size>1 | kernel illegal memory access |
| moe_dp_size>1 | MoE 权重不分片 → OOM |

## 详细分析

4节点32卡意味着 TP=32（4节点×8卡=32）。TP=32 时 MoE 层无法工作，核心问题是死路链条：

### 路径1：TP=32, EP=1

`intermediate_size_per_partition = 3072/32 = 96`，不满足任何 MoE 后端的整除约束：

- marlin 要求 %64==0
- flashinfer_mxfp4 要求 %128==0

**无 MoE 后端可用。**

### 路径2：TP=32, EP>1（降低 moe_tp_size）

EP>=4 时 shape 约束可满足，但触发连锁问题：

1. **必须启用 DP-attention**：V4 Pro 的 MHC 机制 `o_groups=8`，TP=32 时 `n_local_groups=8/32=0`，shape 崩溃
2. **EP + DP-attention + 无 DeepEP**：`_use_tp_moe_gather` 代码路径存在两个 bug（错误的 DP buffer 分配组 + expert mapping=-1 导致 illegal memory access）
3. **必须用 DeepEP**：DeepEP 依赖 NVSHMEM，NVSHMEM 需要 Mellanox IB 网卡，而服务器使用 ib7s400 网卡**不支持 NVSHMEM**
4. **Marlin + EP**：即使装了 DeepEP，Marlin MoE kernel 在 EP + `moe_tp_size>1` 下存在 illegal memory access 的内核级 bug
5. **moe_dp_size>1**：MoE 权重不分片导致每 GPU 权重翻倍，80GB 显存不足（OOM）

### 根本卡点

服务器 ib7s400 网卡不支持 NVSHMEM → 无法安装 DeepEP → 4节点单实例的唯二可行路径（EP+DeepEP）被完全阻断。
