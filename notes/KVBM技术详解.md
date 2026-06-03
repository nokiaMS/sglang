# KVBM (KV Block Manager) 技术详解

## 目录

- [1. KVBM 简介](#1-kvbm-简介)
- [2. KVBM 如何充当统一内存层及写穿缓存](#2-kvbm-如何充当统一内存层及写穿缓存)
  - [统一内存层](#统一内存层)
  - [写穿缓存（Write-Through Cache）](#写穿缓存write-through-cache)
- [3. 为什么说 KVBM 是可扩展的](#3-为什么说-kvbm-是可扩展的)
  - [3.1 横向扩展（Scale-Out）](#31-横向扩展scale-out)
  - [3.2 纵向扩展（Scale-Up）](#32-纵向扩展scale-up)
  - [3.3 框架扩展](#33-框架扩展)
  - [3.4 负载扩展](#34-负载扩展)
- [4. KVBM 的写穿缓存如何保证低延迟](#4-kvbm-的写穿缓存如何保证低延迟)
  - [4.1 读写不对称性——读远多于写](#41-读写不对称性读远多于写)
  - [4.2 计算与写入流水线化](#42-计算与写入流水线化)
  - [4.3 RDMA 零拷贝传输](#43-rdma-零拷贝传输)
  - [4.4 本地优先调度](#44-本地优先调度)
  - [4.5 批量写穿（Batched Write-Through）](#45-批量写穿batched-write-through)
- [5. KVBM 与 vLLM 原有 KV 管理的区别](#5-kvbm-与-vllm-原有-kv-管理的区别)
- [6. KVBM 远程 KV 块迁移的实现](#6-kvbm-远程-kv-块迁移的实现)
  - [阶段一：元数据交换（Metadata Exchange）](#阶段一元数据交换metadata-exchange)
  - [阶段二：分布式搜索（Search Phase）](#阶段二分布式搜索search-phase)
  - [阶段三：分阶段数据准备（Staging）](#阶段三分阶段数据准备staging)
  - [阶段四：RDMA 拉取（核心迁移路径）](#阶段四rdma-拉取核心迁移路径)
- [7. KVBM Event Plane 机制](#7-kvbm-event-plane-机制)
  - [7.1 事件类型（三层协议）](#71-事件类型三层协议)
  - [7.2 事件发射策略（Policy）](#72-事件发射策略policy)
  - [7.3 EventsManager——事件生成核心](#73-eventsmanager事件生成核心)
  - [7.4 EventBatcher——批量处理与排序](#74-eventbatcher批量处理与排序)
  - [7.5 KvbmCacheEventsPublisher——发布到消息系统](#75-kvbmcacheeventspublisher发布到消息系统)
  - [7.6 Consolidator——多源事件去重合并](#76-consolidator多源事件去重合并)
  - [7.7 设计理念：观察者模式解耦](#77-设计理念观察者模式解耦)
- [8. NIXL 在 KVBM 中的作用](#8-nixl-在-kvbm-中的作用)
  - [8.1 NIXL 在 KVBM 中的六层集成](#81-nixl-在-kvbm-中的六层集成)
  - [8.2 配置层——声明启用哪些后端](#82-配置层声明启用哪些后端)
  - [8.3 Agent 层——NIXL 运行时实例](#83-agent-层nixl-运行时实例)
  - [8.4 存储注册层——让 NIXL 知道内存在哪](#84-存储注册层让-nixl-知道内存在哪)
  - [8.5 Layout 层——将块布局映射到 NIXL 描述符](#85-layout-层将块布局映射到-nixl-描述符)
  - [8.6 Transfer 层——策略选择与执行](#86-transfer-层策略选择与执行)
  - [8.7 Manager 层——本地与远端布局统一管理](#87-manager-层本地与远端布局统一管理)
  - [8.8 NIXL 的价值总结](#88-nixl-的价值总结)
- [9. KVBM Block State Machine](#9-kvbm-block-state-machine)
  - [9.1 核心设计：类型状态模式（Typestate Pattern）](#91-核心设计类型状态模式typestate-pattern)
  - [9.2 RAII 保证：永不泄漏](#92-raii-保证永不泄漏)
  - [9.3 Primary 与 Duplicate](#93-primary-与-duplicate)
  - [9.4 WeakBlock 与复活机制](#94-weakblock-与复活机制)
  - [9.5 LifecyclePinRef——类型擦除的 keepalive](#95-lifecyclepinref类型擦除的-keepalive)
  - [9.6 BlockDuplicationPolicy](#96-blockduplicationpolicy)
  - [9.7 set_evict_on_reset——逐块行为覆盖](#97-set_evict_on_reset逐块行为覆盖)
  - [9.8 完整生命周期示例](#98-完整生命周期示例)
- [10. KVBM Framework Connector 架构](#10-kvbm-framework-connector-架构)
  - [10.1 Leader 侧（调度侧）](#101-leader-侧调度侧)
  - [10.2 Worker 侧（执行侧）](#102-worker-侧执行侧)
  - [10.3 Leader-Worker 通信协议](#103-leader-worker-通信协议)
  - [10.4 vLLM vs TRT-LLM Connector 差异](#104-vllm-vs-trt-llm-connector-差异)
  - [10.5 Python-Rust 分层](#105-python-rust-分层)
  - [10.6 分布式层](#106-分布式层)
  - [10.7 端到端数据流](#107-端到端数据流)
- [11. KVBM 存储层级 G1-G4 与 Offload/Onboard 流水线](#11-kvbm-存储层级-g1-g4-与-offloadonboard-流水线)
  - [11.1 Offload 流水线架构](#111-offload-流水线架构)
  - [11.2 Onboard 流水线（升级）](#112-onboard-流水线升级)
  - [11.3 源块类型（SourceBlock）](#113-源块类型sourceblock)
  - [11.4 存储后端实现](#114-存储后端实现)
  - [11.5 ParallelismMode](#115-parallelismmode)
  - [11.6 完整数据流](#116-完整数据流)
- [12. KVBM 聚合与分离式部署模式](#12-kvbm-聚合与分离式部署模式)
  - [12.1 聚合式部署（Aggregated）](#121-聚合式部署aggregated)
  - [12.2 分离式部署（Disaggregated）](#122-分离式部署disaggregated)
  - [12.3 多 Prefill + 多 Decode 拓扑（2P+2D）](#123-多-prefill--多-decode-拓扑2p2d)
  - [12.4 KV 事件流与路由](#124-kv-事件流与路由)
  - [12.5 DynamoConnector——聚合模式的核心](#125-dynamoconnector聚合模式的核心)
  - [12.6 模式对比](#126-模式对比)
  - [12.7 关键配置变量](#127-关键配置变量)
- [13. 总结](#13-总结)
- [14. 术语表](#14-术语表)

---

## 1. KVBM 简介

原文：

> The Dynamo KV Block Manager (KVBM) is a scalable runtime component designed to handle memory allocation, management, and remote sharing of Key-Value (KV) blocks for inference tasks across heterogeneous and distributed environments. It acts as a unified memory layer and write-through cache for frameworks like vLLM and TensorRT-LLM.

翻译：

Dynamo KV Block Manager（KVBM）是一个可扩展的运行时组件，旨在处理异构和分布式环境中推理任务的键值（KV）块的内存分配、管理和远程共享。它充当统一内存层和写穿缓存（write-through cache），服务于 vLLM 和 TensorRT-LLM 等框架。

---

## 2. KVBM 如何充当统一内存层及写穿缓存

### 统一内存层

KVBM 作为"统一内存层"，意味着它为不同的推理框架（如 vLLM、TensorRT-LLM）提供了一致的 KV 块管理接口，屏蔽了底层硬件和内存类型的差异：

- **跨硬件统一**：GPU 显存、CPU 内存、远程节点内存等异构存储资源，对上层框架呈现为统一的 KV 块地址空间
- **跨框架统一**：不同框架不需要各自实现 KV 块的管理逻辑，都通过 KVBM 进行分配、寻址和访问
- **跨节点统一**：分布式环境中多个节点的 KV 块可以通过 KVBM 进行远程共享和迁移，对请求调度而言如同本地访问

简单类比：就像操作系统为应用程序提供统一的虚拟内存抽象，KVBM 为推理引擎提供统一的 KV 块内存抽象。

### 写穿缓存（Write-Through Cache）

写穿缓存是一种缓存策略，核心规则是：

> **数据写入缓存时，同时写入后端存储，两者保持强一致性。**

在 KVBM 的语境下：

1. **写入路径**：当推理框架生成新的 KV 块并写入 KVBM（缓存层）时，KVBM 会同步将数据持久化到后端存储（如分布式共享内存或远程节点）
2. **读取路径**：优先从缓存（本地 GPU/CPU 内存）读取，命中则直接返回；未命中则从后端存储拉取并缓存
3. **一致性保证**：缓存与后端存储始终一致，不存在缓存中的数据比后端更新的情况

**与写回缓存（Write-Back）的对比**：

| 特性 | 写穿（Write-Through） | 写回（Write-Back） |
|------|----------------------|-------------------|
| 写入延迟 | 较高（等后端写入完成） | 较低（只写缓存，异步刷回） |
| 一致性 | 强一致 | 最终一致，有窗口期不一致 |
| 崩溃安全性 | 高（数据已持久化） | 低（可能丢失未刷回数据） |

**为什么 KVBM 选择写穿**：在分布式推理场景中，KV 块可能被多个节点共享和复用（例如 prefix caching），强一致性至关重要——一个节点生成的 KV 块必须立即可被其他节点可靠访问，否则会导致推理结果错误或重复计算。

---

## 3. 为什么说 KVBM 是可扩展的

KVBM 的"可扩展"体现在多个维度：

### 3.1 横向扩展（Scale-Out）

- **分布式架构**：KVBM 不局限于单节点，可以将 KV 块分布存储在多个节点的内存中，随着节点增加，总可用内存容量线性增长
- **远程共享**：KV 块可以通过网络被远程节点访问，新节点加入集群后即可共享已有的 KV 块资源

### 3.2 纵向扩展（Scale-Up）

- **异构内存支持**：同一节点内，KVBM 可以管理 GPU 显存、CPU 内存、甚至 NVMe 等不同层级的存储，构成层次化的存储池
- **热数据驻留高速介质，冷数据下沉低速介质**，在不增加 GPU 显存的情况下容纳更多 KV 块

### 3.3 框架扩展

- **框架无关设计**：作为统一内存层，KVBM 不绑定特定框架。新增推理框架只需对接 KVBM 的接口，无需重新实现整套 KV 块管理逻辑
- 当前支持 vLLM 和 TensorRT-LLM，未来可扩展到其他框架

### 3.4 负载扩展

- **动态分配**：KV 块按需分配和回收，无需预分配固定大小的内存池，适应不同请求量和序列长度
- **Prefix Caching 复用**：相同前缀的 KV 块可跨请求共享，请求量增长时内存消耗不一定同比例增长

### 核心原因

传统方案中，KV 块管理是各框架各自为政的——每个框架管理自己的 GPU 内存池，无法跨节点、跨框架共享。当模型变大、上下文变长、推理集群规模增长时，这种封闭式的管理会成为瓶颈。

KVBM 通过**解耦 KV 块管理与推理框架**、**统一本地和远程内存访问**，使得系统可以通过增加资源来应对增长，而不是被单点内存容量或单框架能力所限制——这就是"可扩展"的本质含义。

---

## 4. KVBM 的写穿缓存如何保证低延迟

写穿缓存天然在写入路径上有额外开销，KVBM 主要通过以下机制来保证低延迟：

### 4.1 读写不对称性——读远多于写

LLM 推理中，每次 decode 步骤的核心操作是：

- **读**：加载整个 KV 缓存做注意力计算（延迟敏感）
- **写**：追加 1 个 token 的 KV 块（量小，非关键路径）

写穿策略下，**读始终命中本地缓存**，零网络开销。写入虽然需要同步写后端，但写的数据量远小于读，对端到端延迟影响有限。

### 4.2 计算与写入流水线化

KV 块的生成和持久化可以重叠执行：

```
Token 1: [Prefill/Decode计算] → [KV写入本地缓存] → [异步写穿透至后端]
Token 2:                              ↑ 计算无需等待写穿完成
```

- 当前 token 的 KV 块写入本地缓存后，即可立即被下一次注意力计算使用
- 写穿后端的操作可以与下一个 token 的计算并行，**不阻塞推理关键路径**

### 4.3 RDMA 零拷贝传输

远程写穿使用 RDMA（Remote Direct Memory Access）：

- **绕过 CPU**：数据从 GPU 内存直接通过网卡传输到远程节点内存，无需 CPU 参与
- **绕过操作系统内核**：无内核态切换和内存拷贝开销
- **延迟极低**：RDMA 的网络延迟通常在微秒级，远低于传统 TCP

### 4.4 本地优先调度

KVBM 的调度策略会尽量让请求命中本地已有的 KV 块：

- **Prefix Cache 感知调度**：将新请求调度到已有相同前缀 KV 块的节点，避免远程读取
- **亲和性调度**：同一会话的请求优先路由到同一节点，最大化本地缓存命中率

命中率越高，写穿带来的开销在实际体验中就越不明显。

### 4.5 批量写穿（Batched Write-Through）

多个 KV 块的写穿操作不是逐个执行，而是批量提交：

- 一次网络传输写入多个 KV 块，摊薄单次写穿的网络固定开销
- 在 prefill 阶段尤其有效，因为会一次性生成大量 KV 块

### 总结

| 机制 | 解决的问题 |
|------|-----------|
| 读命中本地缓存 | 读取零延迟 |
| 计算与写穿流水线化 | 写入不阻塞推理 |
| RDMA 零拷贝 | 降低远程写穿延迟 |
| 本地优先调度 | 减少远程访问频率 |
| 批量写穿 | 摊薄写穿固定开销 |

核心思路是：**写穿保证一致性，但通过让写穿操作避开推理关键路径、用高速网络降低传输开销、用智能调度减少远程依赖，来将延迟影响降到最低。**

---

## 5. KVBM 与 vLLM 原有 KV 管理的区别

### vLLM 原有方案：PagedAttention + BlockManager

vLLM 的核心创新是 **PagedAttention**，将 KV 缓存按固定大小的 Block（页）管理，类比操作系统的虚拟内存分页：

- **BlockManager** 管理物理块的分配与释放
- **BlockSpaceManager** 维护逻辑块到物理块的映射
- 所有 KV 块管理逻辑**内置在 vLLM 内部**

### 核心区别

| 维度 | vLLM 原有 KV 管理 | KVBM |
|------|-------------------|------|
| **架构位置** | 内嵌于 vLLM 引擎内部 | 独立运行时组件，通过接口与框架对接 |
| **内存范围** | 仅管理本节点 GPU 显存 | 统一管理 GPU 显存、CPU 内存、远程节点内存 |
| **跨节点共享** | 不支持，每个节点各自管理 | 支持，KV 块可跨节点远程访问和复用 |
| **跨框架共享** | 不支持，KV 块对其他框架不可见 | 支持，vLLM 生成的 KV 块可被 TensorRT-LLM 等复用 |
| **缓存策略** | 本地缓存，无写穿 | 写穿缓存，保证多节点间强一致 |
| **Prefix Caching** | 支持，但仅限单节点内复用 | 支持，可跨节点、跨框架复用 |
| **块迁移** | 不支持 | 支持，KV 块可在节点间动态迁移 |
| **解耦程度** | KV 管理与调度紧密耦合 | KV 管理与调度解耦，独立演进 |

### 架构对比

**vLLM 原有架构**：
```
┌─────────────────────────┐
│        vLLM Engine      │
│  ┌───────┐ ┌──────────┐ │
│  │Scheduler│ │BlockMgr  │ │  ← KV管理内嵌，仅管理本地GPU
│  └───┬───┘ └────┬─────┘ │
│      └────┬─────┘       │
│        GPU Memory       │
└─────────────────────────┘
     各节点独立，互不知晓
```

**KVBM 架构**：
```
┌──────────┐  ┌───────────┐
│  vLLM    │  │TRT-LLM    │   ← 多框架共享
└────┬─────┘  └─────┬─────┘
     └──────┬───────┘
     ┌──────▼──────┐
     │    KVBM     │         ← 独立统一层
     │ (write-    │
     │  through   │
     │  cache)    │
     └──┬──┬──┬───┘
    GPU  CPU  Remote          ← 异构存储统一管理
   Mem  Mem  Nodes
```

### 实际影响

1. **Disaggregated Serving（分离式推理）**：vLLM 原有方案中 prefill 节点和 decode 节点的 KV 块无法直接传递，需要通过应用层序列化传输。KVBM 原生支持 KV 块跨节点迁移，prefill 和 decode 可以真正解耦部署。

2. **多框架混布**：同一个集群中 vLLM 做 prefill、TensorRT-LLM 做 decode，原有方案下不可能，KVBM 下 KV 块是框架无关的，可以无缝衔接。

3. **内存利用率**：vLLM 原有方案中，每个节点独占自己的 GPU 显存池，空闲内存无法被其他节点利用。KVBM 统一调度，空闲节点的内存可以承接其他节点的 KV 块溢出。

4. **故障恢复**：vLLM 原有方案中，节点故障意味着该节点所有 KV 缓存丢失。KVBM 的写穿缓存保证 KV 块在后端有副本，节点故障后可从后端恢复。

### 一句话总结

vLLM 原有 KV 管理是**单节点、单框架、紧耦合**的本地内存管理器；KVBM 是**多节点、多框架、松耦合**的分布式统一内存层——从"各自管理自己的显存"到"集群共享统一的 KV 存储池"。

---

## 6. KVBM 远程 KV 块迁移的实现

整个过程基于 **Leader/Worker 架构 + NIXL RDMA + Session 协议**，分为四个阶段：

### 阶段一：元数据交换（Metadata Exchange）

两个节点在迁移前必须互相了解对方的内存布局：

```
Node A Leader                    Node B Leader
    │                                │
    │─── request_metadata() ────────>│  (Velo RPC)
    │<── SerializedLayout ──────────│
    │                                │
    │  connect_remote(instance_id,   │
    │                 metadata)      │
    │  → 导入到本地 TransferManager  │
    │  → 建立 (InstanceId, LogicalType) │
    │    → LayoutHandle 映射表       │
```

关键点：
- `SerializedLayout` 包含 `LayoutConfig`（num_layers, page_size, inner_dim, dtype）、基地址 + 步长、设备 ID + 内存类型
- 即使两个节点的 **TP 配置不同**（如 TP=4 vs TP=8），元数据交换能桥接语义差异，使 NIXL 能正确执行 gather-scatter

### 阶段二：分布式搜索（Search Phase）

InitiatorSession 发起跨节点搜索，采用 **先到先得（first-responder-wins）** 策略：

```
Initiator                        Responder A    Responder B    G4 (S3)
   │                                 │              │            │
   │─ 本地 G2 搜索 ─> 命中部分        │              │            │
   │─ 本地 G3 搜索 ─> 命中部分        │              │            │
   │                                 │              │            │
   │─── CreateSession(hashes) ──────>│              │            │
   │─── CreateSession(hashes) ─────────────────────>│            │
   │─── G4 has_blocks() ───────────────────────────────────────>│
   │                                 │              │            │
   │<─── G2Results ─────────────────│              │            │
   │<─── G3Results ────────────────│              │            │
   │<─── SearchComplete ───────────│              │            │
   │<─── G2Results ──────────────────────────────│            │
   │<─── G4Results ──────────────────────────────────────────│
   │                                 │              │            │
   │  对每个 hash：第一个响应者获胜    │              │            │
   │─── HoldBlocks(hold, drop) ────>│              │            │
```

搜索覆盖四个层级：本地 G2 → 本地 G3 → 远程 G2/G3 → G4 对象存储。

### 阶段三：分阶段数据准备（Staging）

根据 `StagingMode` 有三种策略：

| 模式 | 操作 | 适用场景 |
|------|------|---------|
| **Hold** | 仅搜索和锁定块，不做数据移动 | 确认缓存是否存在 |
| **Prepare** | G3→G2 提升（本地 + 远程），保持 session | 预热缓存 |
| **Full** | G3→G2 + RDMA 远程 G2→本地 G2，完成后关闭 | 一步到位迁移 |

**G3→G2 提升（本地和远程）**：
- 本地：`stage_g3_to_g2()` 通过 TransferManager 执行 NIXL Read (Disk→Host)
- 远程：发送 `StageBlocks` 消息，对端执行 G3→G2 后回复 `BlocksReady`

### 阶段四：RDMA 拉取（核心迁移路径）

`pull_remote_blocks()` 是远程迁移的核心：

```
Initiator Node                           Remote Node
    │                                        │
    │ 1. 检查是否已导入远端元数据              │
    │    若否 → request_metadata()           │
    │           → connect_remote()            │
    │                                        │
    │ 2. 分配本地 G2 目标块                    │
    │    g2_manager.allocate_blocks(N)        │
    │                                        │
    │ 3. RDMA 传输                            │
    │    parallel_worker                      │
    │      .execute_remote_onboard_for_instance(│
    │        remote_instance,                 │
    │        LogicalLayoutHandle::G2,  // src │
    │        block_ids,                       │
    │        LogicalLayoutHandle::G2,  // dst │
    │        dst_ids,                         │
    │        TransferOptions::default()       │
    │      )                                  │
    │ ──────── NIXL RDMA Read ──────────────> │
    │ <─────── DMA 直传到本地 G2 ─────────── │
    │                                        │
    │ 4. 注册拉取到的块                        │
    │    dst.stage(seq_hash)                  │
    │    g2_manager.register_block(complete)  │
    │                                        │
    │ 5. 合并 & 按位置排序                     │
    │    consolidate_blocks()                 │
    └────────────────────────────────────────┘
```

底层传输路径：

- **RDMA 路径**：`RemoteDescriptor::Layout { handle, block_ids }` → `TransferManager.execute_transfer()` → NIXL RDMA 直接从远端内存读到本地
- **对象存储路径**：`RemoteDescriptor::Object { keys }` → 异步 `get_blocks_with_layout()` 从 S3/MinIO 下载到本地 G2

### 跨 TP 配置的迁移

当源和目标 TP 配置不同时（如 Prefill TP=8 → Decode TP=1），元数据中的布局信息使 NIXL 能正确执行 gather/scatter：

```
源节点 (TP=8):                    目标节点 (TP=1):
┌──┬──┬──┬──┬──┬──┬──┬──┐        ┌──────────────┐
│R0│R1│R2│R3│R4│R5│R6│R7│  ──→   │   完整块      │
└──┴──┴──┴──┴──┴──┴──┴──┘        └──────────────┘
  每个rank持有 1/8 数据             NIXL gather后完整
```

### 生命周期管理

迁移完成后：
- 发送 `CloseSession` 给所有远端，释放远端持有的块引用
- 本地拉取的块通过 RAII `PublishHandle` 管理，块被驱逐时自动发布 Remove 事件到 Event Plane，NIXL 层同步取消远端注册

---

## 7. KVBM Event Plane 机制

Event Plane 是 KVBM 的**发布/订阅（Pub/Sub）协调层**，用于在分布式环境中广播 KV 块的生命周期事件，使各组件能够被动感知块状态变化，而无需直接耦合。

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Event Plane 数据流                        │
│                                                                 │
│  块生命周期事件源          事件处理管道              事件消费者    │
│                                                                 │
│  ┌──────────┐           ┌──────────────┐         ┌──────────┐  │
│  │ 块注册    │──Create──>│              │         │ KV Router │  │
│  │ (Registry)│           │ EventsManager│         │ (调度路由)│  │
│  └──────────┘           │   ↓ 策略过滤  │         └──────────┘  │
│                          │   ↓ 广播发送  │         ┌──────────┐  │
│  ┌──────────┐           │              │──批量──>│Consolidator│ │
│  │ 块驱逐/  │──Remove──>│   ↓ EventBatcher        │(去重合并) │  │
│  │ 引用释放  │           │   ↓ 批量+排序 │         └─────┬────┘  │
│  │ (RAII Drop)│          │              │               │       │
│  └──────────┘           │   ↓ Publisher │               ▼       │
│                          │              │         ┌──────────┐  │
│                          └──────────────┘         │ ZMQ/NATS │  │
│                                                     │ 发布     │  │
│                                                     └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.1 事件类型（三层协议）

事件按粒度从细到粗分为三层（`kvbm-logical/src/events/protocol.rs`）：

| 层级 | 类型 | 用途 |
|------|------|------|
| `KvCacheEvent` | `Create(SequenceHash)` / `Remove(SequenceHash)` | 内部流上的单个事件 |
| `KvCacheEvents` | `Create(Vec<SeqHash>)` / `Remove(Vec<SeqHash>)` / `Shutdown` | 批量事件，同类型聚合 |
| `KvbmCacheEvents` | `{ events: KvCacheEvents, instance_id: InstanceId }` | 线路传输格式，附带实例上下文 |

关键设计：**批量事件按类型分组**——一个批次要么全是 Create，要么全是 Remove，不会混合。

### 7.2 事件发射策略（Policy）

并非每个块的注册/移除都发射事件。`EventEmissionPolicy` trait 控制过滤（`kvbm-logical/src/events/policy.rs`）：

**PowerOfTwoPolicy**（生产默认）：
- 仅在位置为 2 的幂次方时发射事件：16, 32, 64, 128, ..., 65536
- 创建了对序列空间的**稀疏采样**
- 目的：让 KV Router 在定位跨集群块时能**高效缩小搜索空间**，而无需跟踪每个块

```
位置:  0 1 ... 15 [16] 17 ... 31 [32] 33 ... 63 [64] ...
                      ↑                ↑               ↑
                   发射事件          发射事件        发射事件
```

**AllEventsPolicy**（测试用）：每个块都发射事件，用于验证逻辑正确性。

### 7.3 EventsManager——事件生成核心

`EventsManager` 挂钩到 `BlockRegistry`，在块注册和移除时自动发射事件（`kvbm-logical/src/events/manager.rs`）：

**注册时（Create）**：
```
on_block_registered(handle)
  1. policy.should_emit(seq_hash)?   → 策略过滤
  2. event_tx.send(Create(seq_hash)) → 广播到所有订阅者
  3. handle.attach_unique(EventReleaseHandle) → 附加 RAII 句柄
```

**移除时（Remove）**：
```
EventReleaseHandle::drop()  → 块引用全部释放时自动触发
  event_tx.send(Remove(seq_hash)) → 广播到所有订阅者
```

RAII 机制确保：块的最后一个引用被丢弃时，Remove 事件**必然自动发出**，不存在遗忘或手动取消注册的 bug。

**多订阅者**：基于 `tokio::broadcast` 通道，支持多个消费者同时订阅，每个消费者都能收到完整事件流。

### 7.4 EventBatcher——批量处理与排序

`EventBatcher` 将单个事件流转换为批量线路格式（`kvbm-logical/src/events/batcher.rs`）：

**刷新触发条件**（任一满足即刷新）：
1. **事件类型切换**：Create → Remove 或反之，立即刷新当前批次
2. **达到最大批次大小**：默认 1024 个事件
3. **窗口超时**：默认 10ms

**排序优化**（针对基数树操作）：
- Create 事件：**按位置升序排列**（低→高），便于基数树高效插入
- Remove 事件：**按位置降序排列**（高→低），便于基数树高效删除

```
输入: Create(pos=15), Create(pos=3), Create(pos=64)
                         ↓ 批量+排序
输出: KvCacheEvents::Create([pos=3, pos=15, pos=64])  // 升序
```

### 7.5 KvbmCacheEventsPublisher——发布到消息系统

`KvbmCacheEventsPublisher` 消费批量事件流，通过 `Publisher` trait 发布到外部消息系统（`kvbm-logical/src/events/publisher.rs`）：

- 序列化使用 **MessagePack**（rmp_serde），比 JSON 更紧凑
- 作为后台 tokio 任务运行
- 流结束时自动发送 `Shutdown` 事件并 flush
- 可通过 `abort()` 或 Drop 停止

`Publisher` trait（`kvbm-logical/src/pubsub.rs`）抽象了消息传输：
- `publish(subject, payload)` — 发布消息
- `flush()` — 确保消息已投递
- 具体实现：NATS（生产）、ZMQ（替代）、Stub（测试）

### 7.6 Consolidator——多源事件去重合并

`kvbm-consolidator` 是 Event Plane 的核心消费者，负责将多源事件合并为去重的、kv-router 兼容的输出流（`kvbm-consolidator/src/lib.rs`、`kvbm-consolidator/src/tracker.rs`）：

**三个事件源**：

| 源 | 传输方式 | 内容 |
|----|---------|------|
| vLLM Worker (G1) | ZMQ SUB | 框架级块事件，携带 token_ids、block_size |
| TRT-LLM Worker (G1) | ZMQ SUB | 同上 |
| KVBM (G2/G3) | EventsManager 广播 | 内部块事件，仅携带 SequenceHash |

**去重规则**：

```
Store 事件：同一 SequenceHash 的第一个源发射时发布，后续源注册为静默
Remove 事件：同一 SequenceHash 的最后一个源移除时才发布
```

示例：vLLM、TRT-LLM、KVBM 三个源都持有同一块：
1. vLLM Store → 发布（第一个源）
2. TRT-LLM Store → 静默（第二个源，去重）
3. KVBM Store → 静默（第三个源，去重）
4. vLLM Remove → 静默（还有两个源持有）
5. TRT-LLM Remove → 静默（还有一个源持有）
6. KVBM Remove → 发布 Remove（最后一个源移除）

**Hash 统一**：外部源（vLLM/TRT-LLM）使用自己的字符串 hash，Consolidator 通过 `dynamo_kv_hashing` 将其转换为与 KVBM 内部相同的 `PositionalLineageHash`，确保跨源去重能正确匹配。

### 7.7 设计理念：观察者模式解耦

Event Plane 的核心价值是**将 KV 块的消费者与生产者完全解耦**：

- **KVBM** 只管管理块的生命周期，不知道谁在消费事件
- **KV Router** 订阅事件流，构建索引，做调度决策
- **存储后端** 订阅事件流，实现智能分层（热块提升到 SSD，冷块下沉到对象存储）
- **监控/可观测性** 订阅事件流，生成指标

任何新组件只需订阅 Event Plane 即可获得全局块状态视图，无需修改 KVBM 代码。

---

## 8. NIXL 在 KVBM 中的作用

NIXL（NVIDIA Infrastructure Exchange Layer）是 KVBM 的**底层传输引擎**，负责所有跨存储层级和跨节点的数据搬移。KVBM 的上层逻辑只管"哪些块需要搬到哪里"，NIXL 负责"怎么搬"。

### NIXL 是什么

NIXL 是 NVIDIA 提供的高性能数据传输抽象层，统一了多种传输后端：

| 后端 | 用途 |
|------|------|
| **UCX** | RDMA 远程传输（RoCE、InfiniBand） |
| **POSIX** | 本地文件 I/O（Disk ↔ Host） |
| **GDS** | GPUDirect Storage（Disk ↔ GPU 直传，绕过 CPU） |

### 8.1 NIXL 在 KVBM 中的六层集成

```
┌──────────────────────────────────────────────┐
│  KVBM 上层逻辑（块管理、调度、会话）          │
├──────────────────────────────────────────────┤
│  6. Manager 层    LocalLayout / RemoteLayout  │
│  5. Transfer 层   策略选择 + 执行器 + 通知    │
│  4. Layout 层     PhysicalLayout + NixlMetadata│
│  3. Storage 层    NixlStorage + 内存注册       │
│  2. Agent 层      NixlAgent（后端管理）        │
│  1. Config 层     后端配置（TOML/环境变量）    │
├──────────────────────────────────────────────┤
│  nixl_sys（NIXL C 库的 Rust FFI）            │
└──────────────────────────────────────────────┘
```

### 8.2 配置层——声明启用哪些后端

```toml
[nixl.backends.UCX]          # RDMA 传输
[nixl.backends.GDS]          # GPU 直连存储
threads = "4"
[nixl.backends.POSIX]        # 本地文件 I/O
```

也支持环境变量：`DYN_KVBM_NIXL_BACKEND_UCX=true`

### 8.3 Agent 层——NIXL 运行时实例

每个 KVBM Worker 持有一个 `NixlAgent`，它是 NIXL 的运行时入口：

- 创建时根据配置初始化后端（UCX、GDS、POSIX 等）
- 追踪可用后端列表，供策略选择时查询
- 所有内存注册和传输操作都通过 Agent 进行

### 8.4 存储注册层——让 NIXL 知道内存在哪

KVBM 管理的每一块内存（GPU、CPU Pinned、Disk）都必须向 NIXL Agent 注册：

```
GPU VRAM  → agent.register_memory(ptr, size, Vram, device_id)
CPU Pinned → agent.register_memory(ptr, size, Dram, 0)
Disk FD   → agent.register_memory(fd_offset, size, File, fd)
```

注册后返回 `NixlRegistrationHandle`，这是一个 RAII 句柄——**Drop 时先取消注册再释放内存**，确保不会出现 NIXL 访问已释放内存的悬空引用。

`NixlStorage` 是远程内存的非拥有描述符（addr, size, mem_type, device_id），用于构建 RDMA 传输描述符，指向的地址**不能本地访问**。

### 8.5 Layout 层——将块布局映射到 NIXL 描述符

`PhysicalLayout` 将逻辑块结构（多少层、页大小、内维）与 NIXL 注册信息绑定：

```
PhysicalLayout = Layout（抽象布局） + StorageKind（位置）+ NixlMetadata
                                                    ├─ agent_name（哪个 Agent 拥有）
                                                    ├─ mem_type（Vram/Dram/File）
                                                    └─ device_id
```

**序列化/反序列化**是跨节点迁移的关键：
- `to_descriptor()` → 将布局序列化为可传输的 `LayoutDescriptor`
- `from_descriptor()` → 远端反序列化，创建 `RemoteMemoryDescriptor`，地址指向远端内存

这就是第 6 章中元数据交换的底层实现。

### 8.6 Transfer 层——策略选择与执行

这是 NIXL 最核心的集成层，包含三个子系统：

#### 8.6.1 策略选择

根据源/目标的存储类型和位置，自动选择最优传输路径（`kvbm-physical/src/transfer/strategy.rs`）：

**本地传输**：

| 源 → 目标 | 默认策略 | 启用 GDS/RDMA 后 |
|-----------|---------|-----------------|
| Host → Host | Memcpy | 同左 |
| Pinned → Device | CudaAsyncH2D | 同左 |
| Device → Pinned | CudaAsyncD2H | 同左 |
| Host → Disk | NixlWrite | 同左 |
| Disk → Host | NixlReadFlipped | 同左 |
| Device → Disk | **两跳**：D2H → H2Disk | **直传**：GDS |
| Disk → Device | **两跳**：Disk2H → H2D | **直传**：GDS |

**远程传输**：

| 源 → 目标 | 默认策略 | 启用 GPU RDMA 后 |
|-----------|---------|-----------------|
| Host → Remote | NixlWrite（直传） | 同左 |
| Device → Remote | **两跳**：D2H → NixlWrite | **直传**：GPU RDMA |
| Disk → Remote | 不支持（需先提升到 Host） | — |

**两跳传输（TwoHop）**示意：
```
Device → Remote（无 GPU RDMA）:
  [GPU VRAM] ──CUDA D2H──> [CPU Pinned] ──NIXL Write/RDMA──> [Remote Node]
              第一跳                    第二跳
              bounce buffer
```

局部性检测通过比较 `src.nixl_metadata().agent_name()` 与 `ctx.nixl_agent().name()` 实现。

#### 8.6.2 传输执行器

类型状态（typestate）构建器，编译期保证所有必要参数已设置（`kvbm-physical/src/transfer/executor/nixl.rs`）：

```
NixlTransferBuilder
  .src(PhysicalLayout)         // 源布局
  .dst(PhysicalLayout)         // 目标布局
  .src_blocks(&[BlockId])      // 源块 ID
  .dst_blocks(&[BlockId])      // 目标块 ID
  .strategy(TransferStrategy)  // 传输策略
  .layer_range(0..8)           // 可选：仅传输指定层
  .write_notif("callback")     // 可选：远端通知
  .execute(&TransferContext)   // 执行
```

执行过程：
1. 验证布局兼容性（层数、维度）
2. 根据 NIXL 操作类型（Read/Write）验证局部性约束：Write 要求源必须是本地，Read 要求目标必须是本地
3. 构建 `XferDescList`（源和目标的描述符列表）——支持整块优化或逐层传输
4. 处理"翻转"策略：某些后端（如 POSIX）要求 Host 内存始终作为"local"描述符列表
5. 创建并提交传输请求：`agent.create_xfer_req()` + `post_xfer_req()`
6. 注册异步完成通知

#### 8.6.3 完成通知（两种机制）

**事件驱动**（`kvbm-physical/src/transfer/notifications/nixl_events.rs`，推荐）：
- 注册 `RegisterNixlNotification`（UUID + XferRequest + EventHandle）
- 后台任务轮询 `agent.get_notifications()`
- 处理早到通知（通知在注册前到达的竞态条件）
- 超过 60 秒未完成则告警（30 秒节流）

**轮询检查**（`kvbm-physical/src/transfer/notifications/nixl_status.rs`，备用）：
- 注册 `NixlStatusChecker`（持有 Agent + XferRequest）
- 后台任务周期性调用 `agent.get_xfer_status()`
- 状态为 success 时触发完成事件

### 8.7 Manager 层——本地与远端布局统一管理

`LocalLayout` / `RemoteLayout` 对 `PhysicalLayout` 做了轻量包装：

- `LocalLayout`：直接持有 NIXL 注册的本地内存，可本地访问
- `RemoteLayout`：从导入的元数据重建，内存区域指向远端地址，**仅用于构建 NIXL RDMA 描述符**

两者对 `TransferManager` 呈现统一接口，使本地传输和远程传输使用相同的 `execute_transfer()` 代码路径。

### 8.8 NIXL 的价值总结

| 能力 | 没有 NIXL | 有 NIXL |
|------|----------|---------|
| GPU ↔ CPU | CUDA D2H/H2D | 同左 + NIXL 统一接口 |
| CPU ↔ Disk | POSIX I/O | NIXL POSIX 后端，零拷贝 |
| GPU ↔ Disk | 必须经 CPU 中转 | GDS 直传（可选） |
| 节点间传输 | 应用层序列化 + TCP | RDMA 零拷贝 |
| GPU ↔ 远程 GPU | 不可能 | GPU RDMA（可选） |
| 跨 TP 配置传输 | 手动对齐 | NIXL gather/scatter 自动处理 |

**一句话概括**：NIXL 是 KVBM 的传输脊梁——它将 GPU/CPU/Disk/远程内存的异构传输统一为一致的 Read/Write 抽象，让 KVBM 上层无需关心数据在不同介质和节点间如何搬运。

---

## 9. KVBM Block State Machine

KVBM 的块状态机通过 **Rust 类型系统** 在编译期强制执行状态转换，确保非法转换不可能发生。

### 状态与守卫类型映射

```
                 stage / complete          register_block
  Reset ──────────────────────► Staged ─────────────────► Registered
    ▲                              │                          │
    │          reset               │                          │
    ├──────────────────────────────┘                          │
    │                          drop                           │
    └──────────────────────────────────────────────────────────┘
```

| 状态 | 守卫类型 | 含义 | 可执行操作 |
|------|---------|------|-----------|
| **Reset** | `MutableBlock<T>` | 块空闲或正在写入 | `stage()` → Staged，`complete()` → Staged |
| **Staged** | `CompleteBlock<T>` | 块已填满并分配了 SequenceHash，但尚未注册 | `register_block()` → Registered，`reset()` → Reset |
| **Registered** | `ImmutableBlock<T>` | 块已注册，可被查找和共享 | `downgrade()` → WeakBlock，`pin()` → LifecyclePinRef |

此外还有两个辅助类型：
- **`WeakBlock<T>`**：Registered 状态的非拥有弱引用，不阻止驱逐
- **`LifecyclePinRef`**：类型擦除的生命周期引脚，跨不同 `T` 参数使用

### 9.1 核心设计：类型状态模式（Typestate Pattern）

状态转换通过**消费 self** 实现——转换方法接收 `self` 所有权，返回下一个状态的守卫。旧状态的守卫不再存在，编译器保证不可能在错误的状态调用方法：

```rust
// MutableBlock::stage — 消费 self，返回 CompleteBlock
pub fn stage(mut self, seq_hash: SequenceHash, block_size: usize)
    -> Result<CompleteBlock<T>, BlockError<MutableBlock<T>>>

// CompleteBlock::reset — 消费 self，返回 MutableBlock
pub fn reset(mut self) -> MutableBlock<T>

// CompleteBlock 只能通过 BlockManager::register_block 注册为 ImmutableBlock
// （不由 CompleteBlock 自身方法完成，避免循环依赖）
```

### 9.2 RAII 保证：永不泄漏

每个守卫类型都实现了 `Drop`，通过 `armed` 标志控制行为：

```rust
// MutableBlock::Drop — armed 时自动归还到 reset 池
impl<T: BlockMetadata> Drop for MutableBlock<T> {
    fn drop(&mut self) {
        if self.armed {
            self.store.release_mutable(self.block_id);
        }
    }
}

// CompleteBlock::Drop — armed 时自动释放回 Reset
impl<T: BlockMetadata> Drop for CompleteBlock<T> {
    fn drop(&mut self) {
        if self.armed {
            self.store.release_staged(self.block_id);
        }
    }
}
```

- 正常转换时 `armed = false`（`disarm()`），Drop 为空操作，槽位由新守卫接管
- 异常/提前丢弃时 `armed = true`，Drop 自动回收槽位
- `BlockError` 变体携带原始块返回给调用者，**失败路径也不会泄漏块**

### 9.3 Primary 与 Duplicate

进入 Registered 状态后，块分为两种角色：

**Primary（主块）**：
- 同一 SequenceHash 的规范持有者
- 最后一个 `Arc` clone 被 Drop 时，槽位转为 **Inactive**（可被驱逐）
- 通过 `BlockRegistry` 可被查找和复活

**Duplicate（副本块）**：
- 共享同一 SequenceHash 的额外物理块
- 持有对 Primary 的强引用 `_primary_keepalive`，**保证 Primary 在 Duplicate 存活期间不会被驱逐**
- 最后一个 clone 被 Drop 时，槽位直接 **Reset**（归还空闲列表）

```
SequenceHash = H
┌─────────────────┐       ┌─────────────────┐
│  Primary Block   │◄──────│ Duplicate Block  │
│  is_primary=true │  强引用│ is_primary=false │
│  G1 (GPU)       │       │ G2 (CPU Pinned) │
└─────────────────┘       └─────────────────┘
  Drop → Inactive            Drop → Reset
```

Duplicate 的存在意义：同一个逻辑块可能同时存在于多个存储层级（GPU + CPU），每个物理副本都是 Duplicate，但只有一个是 Primary。

### 9.4 WeakBlock 与复活机制

`WeakBlock` 是 `ImmutableBlock` 的弱引用，通过 `downgrade()` 创建：

```rust
pub fn upgrade(&self) -> Option<ImmutableBlock<T>> {
    // 快速路径：Arc 还活着，直接升级
    if let Some(strong) = self.inner.upgrade() {
        return Some(ImmutableBlock::from_inner(strong));
    }
    // 慢速路径：Primary 已进入 Inactive 池，通过 Registry 复活
    let inner = upgrade_or_resurrect(&self.handle, &self.store, false)?;
    Some(ImmutableBlock::from_inner(inner))
}
```

**两条升级路径**：
1. **快速路径**：`Weak::upgrade()`——Arc 引用计数 > 0，Primary 仍然活跃
2. **慢速路径**：`upgrade_or_resurrect()`——Primary 已进入 Inactive 池，通过 `BlockRegistry` 按序列哈希查找并复活

### 9.5 LifecyclePinRef——类型擦除的 keepalive

`LifecyclePin` trait 擦除了 `T` 参数，使不同类型的已注册块可以存入同一集合：

```rust
pub trait LifecyclePin: Send + Sync {
    fn block_id(&self) -> BlockId;
    fn sequence_hash(&self) -> SequenceHash;
    fn manager_id(&self) -> ManagerId;
    fn registration_handle(&self) -> BlockRegistrationHandle;
}
```

用途场景：KV offload connector 需要暂存异构的进行中传输列表，用 `(manager_id, block_id)` 作为查找键。只要 `LifecyclePinRef` 存活，槽位就不会从 Active 转为 Inactive。

### 9.6 BlockDuplicationPolicy

注册时控制是否允许同一 SequenceHash 有多个物理块：

- **Allow**：允许多副本（GPU + CPU 各一份）
- **Reject**：每个哈希只保留一个，注册重复时返回已有的 Primary

### 9.7 set_evict_on_reset——逐块行为覆盖

`ImmutableBlock::set_evict_on_reset(value)` 控制最后一个引用 Drop 时的行为：

- `true`：直接 Reset（归还空闲列表）
- `false`：进入 Inactive 池（可被驱逐或复活）

此覆盖设置**跨复活持久化**——块被复活后仍保持该设置，仅在真正被驱逐回 Mutable 时才重置为全局默认值。

### 9.8 完整生命周期示例

```
1. allocate_blocks(1) → MutableBlock（Reset）
2. mb.complete(&token_block) → CompleteBlock（Staged，已分配 SequenceHash）
3. manager.register_blocks(vec![cb]) → ImmutableBlock（Registered，Primary）
4. block.downgrade() → WeakBlock（弱引用，不阻止驱逐）
5. block.pin() → LifecyclePinRef（类型擦除 keepalive）
6. block.set_evict_on_reset(false) → Drop 后进入 Inactive 而非 Reset
7. drop(block) → Primary 转入 Inactive 池
8. weak.upgrade() → 慢速路径复活 → 新的 ImmutableBlock
9. drop(all_references) → Inactive → 驱逐 → MutableBlock（Reset，可重新分配）
```

---

## 10. KVBM Framework Connector 架构

Connector 是 KVBM 与推理框架之间的桥梁，采用 **Leader-Worker 分裂模式**，精确映射推理框架自身的 Scheduler-Worker 架构。

### 整体架构

```
┌───────────────────────┐   序列化 ConnectorMetadata   ┌───────────────────────┐
│   LEADER（调度侧）     │  ──────────────────────────> │   WORKER（执行侧）     │
│                       │                              │                       │
│  KvConnectorLeader    │                              │  KvConnectorWorker    │
│   - create_slot       │                              │   - register_kv_caches│
│   - get_num_new_      │                              │   - bind_connector_   │
│     matched_tokens    │                              │     metadata          │
│   - update_state_     │                              │   - save_kv_layer     │
│     after_alloc       │                              │   - start_load_kv     │
│   - build_connector_  │                              │   - get_finished      │
│     metadata          │                              │                       │
│   - request_finished  │                              │                       │
│                       │                              │                       │
│  ConnectorSlotManager │                              │  WorkerSchedulerClient│
│  VllmConnectorSlot    │                              │  TransferScheduler    │
│  LocalTransferEngine  │                              │  KvbmWorker (NIXL)   │
└───────────────────────┘                              └───────────────────────┘
          │                                                       │
          └──────────────────┬──────────────────────────────────┘
                             ▼
                ┌─────────────────────────┐
                │  分布式层               │
                │  KvbmLeader (ZMQ/Velo)  │
                │  KvbmWorker (NIXL)      │
                └─────────────────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │  KVBM-Engine 核心       │
                │  InstanceLeader         │
                │  BlockManager<G2/G3>    │
                │  InitiatorSession       │
                │  ResponderSession       │
                └─────────────────────────┘
```

### 10.1 Leader 侧（调度侧）

Leader 在推理框架的 Scheduler 进程中运行，负责**决策**而非执行。

#### Leader Trait

```rust
pub trait Leader: Send + Sync + std::fmt::Debug {
    fn get_num_new_matched_tokens(...) -> Result<(usize, bool)>;
    fn update_state_after_alloc(...) -> Result<()>;
    fn build_connector_metadata(...) -> Result<Vec<u8>>;
    fn request_finished(...) -> Result<bool>;
    fn create_slot(...) -> Result<()>;
}
```

#### 核心流程

**1. 创建 Slot（create_slot）**

每个推理请求进入调度器时，Leader 创建一个 `VllmConnectorSlot`，关联 request_id、lora_name 和 salt_hash。

**2. 查找缓存命中（get_num_new_matched_tokens）**

```
请求进入 → acquire_local_matches()
              ├─ 搜索 G2 (Host) → 命中部分块
              ├─ 搜索 G3 (Disk) → 命中部分块
              └─ 返回 (matched_count, needs_async)
```

- 返回值：`(匹配的 token 数, 是否需要异步 onboard)`
- 如果命中了 Host/Disk 上的块，需要异步将其搬到 GPU

**3. 更新分配状态（update_state_after_alloc）**

Scheduler 为请求分配了新的 GPU 块后，通知 Leader：
- 追加新的 mutable device blocks 到 slot
- 触发 onboarding（将 Host/Disk 命中的块搬到 GPU）

**4. 构建元数据（build_connector_metadata）**

将调度决策序列化为 `ConnectorMetadata`，传递给 Worker：

```rust
pub struct ConnectorMetadata {
    pub iteration: u64,
    pub new_slots: Vec<NewSlotInfo>,
    pub operations: Vec<WorkerTransferRequest>,
}
```

`WorkerTransferRequest` 包含：
- `request_id`：请求标识
- `transfer_type`：Load（onboard）或 Store（offload）
- `request_type`：Immediate（立即执行）或 Scheduled（延迟到 forward pass 后）

**5. 请求完成（request_finished）**

请求生成完毕后，标记 slot 为 Finished。返回值控制是否保持块活跃直到 Worker 确认。

#### SlotState 生命周期

```
Initialized → OnboardStaged → Onboarding → Prefilling → Decoding → Finishing → Finished
                  │                                            ▲
                  └──── SkippedPrefill / SkippedDecode ───────┘
```

- **Initialized**：刚创建，等待缓存查找
- **OnboardStaged**：缓存命中，等待触发 onboard
- **Onboarding**：块正在从 Host/Disk 搬到 Device
- **Prefilling**：正在 prefill
- **Decoding**：正在 decode
- **Finishing**：请求结束，等待 Worker 确认
- **Finished**：可释放块

#### LocalTransferEngine

Leader 侧的后台传输引擎，处理实际的块搬运：

- **Onboard**：Host/Disk → Device（G2/G3 → G1）
- **Offload**：Device → Host/Disk（G1 → G2/G3），支持优先级过滤（`offload_min_priority`）

### 10.2 Worker 侧（执行侧）

Worker 在推理框架的 Model Runner 进程中运行，负责**执行**传输操作。

#### Worker Trait

```rust
pub trait Worker: Send + Sync {
    fn register_kv_caches(...) -> Result<()>;
    fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> Result<()>;
    fn clear_connector_metadata(&mut self);
    fn save_kv_layer(&mut self, layer_name: String) -> Result<()>;
    fn get_finished(&mut self, finished_requests: HashSet<String>) -> (HashSet<String>, HashSet<String>);
}
```

#### 核心流程

**1. 注册 KV 缓存（register_kv_caches）**

将 vLLM 分配的 KV Cache 张量注册到 NIXL，使远程节点可以通过 RDMA 访问。关键方法 `with_external_device_regions()` 使 KVBM 能够直接操作框架管理的 GPU 内存。

**2. 绑定元数据（bind_connector_metadata）**

反序列化 Leader 传来的 `ConnectorMetadata`：
- 创建 Worker 侧的 slot
- 入队 Immediate 类型的 onboard 操作
- 延迟 Scheduled 类型的 offload 操作（到 forward pass 后执行）

**3. 保存 KV 层（save_kv_layer）**

在每层 forward pass 完成后调用：
- 追踪层完成进度
- 当所有层完成后，触发 offload 操作
- 记录 CUDA event 用于异步完成通知

**4. 获取完成状态（get_finished）**

返回两组完成的请求 ID：
- `finished_sending`：offload 完成的请求
- `finished_receiving`：onboard 完成的请求

### 10.3 Leader-Worker 通信协议

Leader 和 Worker 之间通过**序列化的 ConnectorMetadata** 通信，这是唯一的接口边界：

```
Leader (Scheduler 进程)                    Worker (Model Runner 进程)
    │                                           │
    │  build_connector_metadata()               │
    │  ──── ConnectorMetadata (bytes) ────────> │  bind_connector_metadata()
    │                                           │
    │                                           │  save_kv_layer() × N layers
    │                                           │  （每层完成后触发 offload）
    │                                           │
    │  <──── finished_requests ─────────────── │  get_finished()
    │                                           │
```

这种设计确保 Leader 和 Worker 可以运行在不同进程甚至不同节点上。

### 10.4 vLLM vs TRT-LLM Connector 差异

| 维度 | vLLM Connector | TRT-LLM Connector |
|------|---------------|-------------------|
| **请求 ID 类型** | `String` | `u64` |
| **KV 张量注册** | 逐层注册 | 单个连续张量 + NCCL replicated 模式 |
| **Onboard 触发** | bind_connector_metadata 时入队 | start_load_kv() 显式调用 |
| **Offload 触发** | save_kv_layer 后自动 | execute_offload_operations() 或 submit_offload_on_event() |
| **Offload 模式** | 同步（每层检查完成） | 异步（CUDA event 轮询线程） |
| **MLA 支持** | 无 | NCCL replicated 模式（DYN_KVBM_NCCL_MLA_MODE） |
| **request_finished 返回** | 始终 `true`（保持块直到 Worker 确认） | `false`（块立即释放） |
| **update_state_after_alloc** | 接收 `num_external_tokens` | 接收 `context_current_position` |
| **Consolidator 端点** | vllm 专用端点 | trtllm 专用端点 |

### 10.5 Python-Rust 分层

每个框架的 Connector 都有三层：

```
┌─────────────────────────────────────────┐
│  Python 层（框架接口适配）               │
│  - 实现 vLLM/TRT-LLM 的 Connector 协议  │
│  - 处理框架特定的数据类型和回调           │
│  - 薄包装，逻辑在 Rust 层               │
├─────────────────────────────────────────┤
│  Rust 层（PyO3 绑定）                   │
│  - KvConnectorLeader / KvConnectorWorker│
│  - ConnectorSlotManager / SlotState     │
│  - LocalTransferEngine                  │
│  - ConnectorMetadata 序列化/反序列化     │
├─────────────────────────────────────────┤
│  KVBM-Engine 核心（纯 Rust）            │
│  - InstanceLeader / BlockAccessor       │
│  - BlockManager<G2/G3>                  │
│  - Session 协议（Initiator/Responder）  │
│  - NIXL Agent / TransferManager         │
└─────────────────────────────────────────┘
```

Python 层极薄——只做框架协议适配和数据类型转换，核心逻辑全部在 Rust 中实现，确保性能。

### 10.6 分布式层

`KvbmLeader` 和 `KvbmWorker` 通过 ZMQ/Velo 协调：
- Worker 向 Leader 注册并同步 KV Cache 张量信息
- Leader 路由跨实例的块传输请求
- 支持非对称 TP 配置（如 Prefill TP=4 → Decode TP=1）

### 10.7 端到端数据流

```
1. 请求到达 vLLM Scheduler
2. Leader.create_slot(request)
3. Leader.get_num_new_matched_tokens()
   → 搜索 G2/G3 缓存，找到 N 个命中块
4. Scheduler 分配 GPU 块
5. Leader.update_state_after_alloc()
   → 触发 LocalTransferEngine: G2/G3 → G1
6. Leader.build_connector_metadata()
   → 序列化 onboard/offload 操作
7. Worker.bind_connector_metadata()
   → 入队 Immediate onboard 操作
8. Worker 执行 forward pass
   → 每层: save_kv_layer() → 追踪完成 → 触发 offload
9. Worker.get_finished()
   → 返回已完成的 onboard/offload 请求
10. Leader.request_finished()
   → 释放 slot
```

---

## 11. KVBM 存储层级 G1-G4 与 Offload/Onboard 流水线

### G1-G4 四级存储层次

```
┌─────────────────────────────────────────────────────────────────────┐
│  G1  GPU HBM（显存）        容量最小，速度最快，推理时活跃 KV 缓存  │
│      StorageKind::Device    由推理框架拥有，KVBM 通过 External 引用 │
├─────────────────────────────────────────────────────────────────────┤
│  G2  Host Pinned DRAM       CPU 锁页内存，GPU DMA 高效传输的暂存区  │
│      StorageKind::Pinned    RDMA 源/目标，配置: DYN_KVBM_CPU_CACHE_GB│
├─────────────────────────────────────────────────────────────────────┤
│  G3  NVMe/SSD               本地磁盘，支持 GDS 直传或 POSIX 回退    │
│      StorageKind::Disk      持久化热块存储，配置: DYN_KVBM_DISK_CACHE_GB│
├─────────────────────────────────────────────────────────────────────┤
│  G4  S3/MinIO/NIXL-OBJ     远程对象存储，最大容量，KVBM 视为不透明  │
│      StorageKind::Object    通过 ObjectBlockOps trait 访问          │
└─────────────────────────────────────────────────────────────────────┘

容量递增 ──────────────────────────────────────────────> 
延迟递增 ──────────────────────────────────────────────>
持久性递增 ──────────────────────────────────────────────>
```

**容量规则**：每一层的容量必须 >= 上一层。否则 KVBM 会无效地反复 offload/onboard 同一批块，产生抖动。

### 11.1 Offload 流水线架构

Offload（降级）将块从高速层搬到低速层。`OffloadEngine` 管理三条流水线：

```rust
pub struct OffloadEngine {
    g1_to_g2: Option<Pipeline<G1, G2>>,       // GPU → Host
    g2_to_g3: Option<Pipeline<G2, G3>>,       // Host → Disk
    g2_to_g4: Option<ObjectPipeline<G2>>,     // Host → Object Storage
}
```

#### 五阶段流水线

每条 offload 流水线包含五个串行阶段：

```
┌──────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌────────────────┐
│ 1. Policy    │──>│ 2. Batch     │──>│ 3. Precondition │──>│ 4. Block     │──>│ 5. Transfer    │
│    Evaluator │   │    Collector  │   │    Awaiter      │   │    Upgrader  │   │    Executor    │
└──────────────┘   └──────────────┘   └─────────────────┘   └──────────────┘   └────────────────┘
   过滤不需要      积攒到批量大小      等待前置条件         Weak→Strong       执行实际数据传输
   offload的块     再往下传递          （如forward完成）     升级引用
```

**阶段 1：Policy Evaluator（策略过滤）**

| 流水线 | 策略 | 行为 |
|--------|------|------|
| G1→G2 | `PresenceFilter` | 跳过已在 G2 的块 + 正在传输中的块 |
| G2→G3 | `PresenceAndLFUFilter` | 跳过已在 G3 的块 + LFU 计数 >= 8 的热块 |
| G2→G4 | `ObjectPresenceFilter` / `ObjectLockPresenceFilter` | 异步检查对象是否已存在 / 分布式锁检查 |

`PendingTracker` 追踪传输中的块，防止并发场景下重复传输。

**阶段 2：Batch Collector（批量收集）**

```
max_batch_size: 1024    // 达到上限立即刷新
flush_interval: 10ms    // 定时刷新
min_batch_size: 8       // 至少积攒 8 个块才刷新
```

**阶段 3：Precondition Awaiter（前置条件等待）**

等待 forward pass 完成后再执行 offload——确保不会搬运正在被 GPU 计算使用的块。

**阶段 4：Block Upgrader（引用升级）**

将 `WeakBlock` 升级为 `ImmutableBlock`（强引用）。已被驱逐的块记录在 `evicted` 列表中，跳过传输。

**阶段 5：Transfer Executor（传输执行）**

- **BlockTransferExecutor**（G2/G3 目标）：
  1. 在目标层分配块：`dst_manager.allocate_blocks()`
  2. 执行数据传输：`leader.execute_local_transfer()`
  3. 等待传输完成通知
  4. 注册到目标层：`dst_manager.register_block()`
  5. 如果启用 auto_chain，将注册结果送入下游流水线

- **ObjectTransferExecutor**（G4 目标）：
  1. 调用 `ObjectBlockOps::put_blocks()`
  2. 如果配置了 lock_manager：创建 `.meta` 文件，释放 `.lock` 文件
  3. 无需目标 BlockManager 注册

#### Auto-Chaining（自动链式降级）

当 G1→G2 流水线启用 `auto_chain` 时，完成 G2 注册的块**自动送入**下游 G2→G3 和 G2→G4 流水线：

```
G1 ──offload──> G2 ──auto_chain──> G3 (Disk)
                 │
                 └──auto_chain──> G4 (S3)
```

链式降级使用 `WeakBlock`（弱引用）——如果块在被下游处理前已被驱逐，则跳过（优雅降级，不阻塞）。

#### 分布式 G4 Offload

在分布式部署中，Leader 可能没有物理布局信息：
- 启用 `with_enable_remote_g4(true)` 代替本地 G2→G4 流水线
- Leader 通过 RPC 调用 Worker 的 `ObjectBlockOps::put_blocks()`
- Worker 从本地 G2 上传到对象存储

### 11.2 Onboard 流水线（升级）

Onboard（升级）将块从低速层搬回高速层，发生在请求需要复用缓存时：

```
                    ONBOARD PATH
                    ============

G4 (S3)       ── get_blocks() ──────────>  G2 (Host) ──> G1 (GPU)
G3 (NVMe)     ── find_matches() ──────>  G2 (Host) ──> G1 (GPU)  
G2 (Host)     ── find_matches() ──────>  G1 (GPU)
```

Onboard 路径在第 6 章已有详细描述，核心是 Session 协议驱动的分布式搜索和 RDMA 拉取。

### 11.3 源块类型（SourceBlock）

Offload 流水线中块的引用方式：

```rust
pub enum SourceBlock<T> {
    External(ExternalBlock<T>),   // G1 块：vLLM 拥有 GPU 内存，KVBM 只持有 block_id + seq_hash
    Strong(ImmutableBlock<T>),    // 强引用：直接持有的块
    Weak(WeakBlock<T>),           // 弱引用：auto_chain 输出，可能已被驱逐
}
```

G1 块始终是 `External`——GPU 内存由推理框架管理，KVBM 无需持有强引用。

### 11.4 存储后端实现

| 层级 | 后端 | 特性 |
|------|------|------|
| G1 | `DeviceStorage` | CUDA VRAM，支持包装 torch 张量（`DeviceStorageType::Torch`） |
| G2 | `PinnedStorage` | CUDA 锁页内存，NUMA 感知分配，GPU DMA 零拷贝 |
| G3 | `DiskStorage` | O_DIRECT 文件 I/O，GDS 可选；支持 `mkostemp` 和 `fcntl` 两种 O_DIRECT 策略 |
| G4 | `ObjectStorage` | NIXL OBJ 后端，桶名 + u64 key 标识，无直接内存访问 |

G3 的 O_DIRECT 策略可通过环境变量选择：
- `DYN_KVBM_DISK_ALLOCATOR_TYPE=open-direct`：用于 IBM Storage Scale
- 默认：`fcntl` 方式（适用于 ext4、XFS、Lustre）

### 11.5 ParallelismMode

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `TensorParallel` | 所有 worker 都有 G1+G2+G3 | 标准张量并行 |
| `ReplicatedData` | 只有 rank-0 有 G2+G3，onboard 后通过 NCCL 广播给其他 rank | MLA 模型（如 DeepSeek），减少重复存储 |

### 11.6 完整数据流

```
OFFLOAD（降级）:
  G1 (GPU) ──[PresenceFilter]──> G2 (Host) ──[Presence+LFU]──> G3 (Disk)
                                      │
                                      └──[ObjectPresence]──> G4 (S3)

ONBOARD（升级）:
  G4 (S3)    ──get_blocks()────> G2 ──H2D──> G1
  G3 (Disk)  ──NIXL Read──────> G2 ──H2D──> G1
  G3 (Disk)  ──GDS─────────────> G1（绕过 G2，需要 GDS 支持）
  G2 (Host)  ──CUDA H2D───────> G1

AUTO-CHAIN:
  G1 ──offload──> G2 ──auto_chain──> G3/Disk
                     └──auto_chain──> G4/S3

DISAGGREGATED:
  Prefill Node G1 ──offload──> G2 ──RDMA──> Decode Node G2 ──onboard──> G1
```

---

## 12. KVBM 聚合与分离式部署模式

KVBM 支持两种部署模式：**聚合式（Aggregated）** 和 **分离式（Disaggregated）**，分别适用于单卡/少卡场景和多卡/多节点的大规模推理场景。

### 12.1 聚合式部署（Aggregated）

单 GPU 同时承担 prefill 和 decode，KVBM 作为 KV 缓存的分级存储管理器：

```
┌─────────────────────────────────────────────┐
│  单 Worker（Prefill + Decode）               │
│                                             │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │  G1 (GPU)    │    │  DynamoConnector │   │
│  │  活跃 KV 缓存│◄──►│  kv_role=kv_both │   │
│  └──────┬───────┘    └────────┬─────────┘   │
│         │ offload/onboard     │              │
│         ▼                     │              │
│  ┌──────────────┐             │              │
│  │  G2 (Host)   │◄────────────┘              │
│  │  冷 KV 缓存  │                            │
│  └──────────────┘                            │
│       DYN_KVBM_CPU_CACHE_GB=20              │
└─────────────────────────────────────────────┘
```

**配置要点：**

```bash
DYN_KVBM_CPU_CACHE_GB=20 \
python -m dynamo.vllm --model $MODEL \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_connector_module_path": "kvbm.vllm_integration.connector",
    "kv_role": "kv_both"
  }'
```

- 使用 `DynamoConnector` 作为唯一的 KV connector
- `kv_role=kv_both`：同一进程同时负责 KV 的产出和消费
- KVBM 将冷块 offload 到 G2（Host Pinned DRAM），请求复用时 onboard 回 G1

### 12.2 分离式部署（Disaggregated）

Prefill 和 Decode 分别运行在不同 GPU 上，通过 **PdConnector** 组合 KVBM offloading 和 NIXL RDMA 传输：

```
┌─────────────────────────────┐          RDMA          ┌──────────────────────────┐
│  Prefill Worker (GPU 1)     │  ◄──────────────────►  │  Decode Worker (GPU 0)   │
│                             │       NIXL RDMA        │                          │
│  ┌───────────────────────┐  │                        │  ┌────────────────────┐  │
│  │  PdConnector          │  │                        │  │  NixlConnector     │  │
│  │  ┌─────────────────┐  │  │                        │  │  kv_role=kv_both   │  │
│  │  │ DynamoConnector │  │  │                        │  │  从 Prefill 拉取 KV│  │
│  │  │ (KVBM offload)  │  │  │                        │  └────────────────────┘  │
│  │  ├─────────────────┤  │  │                        │                          │
│  │  │ NixlConnector   │──┼──┼──────────────────────►│  G1 (GPU)               │
│  │  │ (RDMA 传输)     │  │  │                        │  解码用 KV 缓存         │
│  │  └─────────────────┘  │  │                        └──────────────────────────┘
│  └───────────────────────┘  │
│                             │
│  G1 (GPU) ──offload──> G2   │
│  DYN_KVBM_CPU_CACHE_GB=20  │
│  --disaggregation-mode      │
│    prefill                  │
└─────────────────────────────┘
```

#### PdConnector 架构

`PdConnector` 继承 `MultiConnector`，严格组合两个子 connector：

```python
class PdConnector(MultiConnector):
    """
    第一个 connector: DynamoConnector（KVBM），负责 KV 块的 offload/onboard
    第二个 connector: NixlConnector，负责跨节点的 RDMA KV 传输
    """
```

| 方法 | 委派目标 | 说明 |
|------|---------|------|
| `get_num_new_matched_tokens` | `connectors[0]`（DynamoConnector） | 缓存命中查找走 KVBM |
| `update_state_after_alloc` | `connectors[0]` + `connectors[1]` | KVBM 分配 onboard 块；NIXL 获空块（Prefill 侧不需要分配） |
| `build_connector_meta` | 两个 connector 各自构建 | 合并为 `PdConnectorMetadata` |
| `get_handshake_metadata` | `connectors[1]`（NixlConnector） | 握手元数据来自 NIXL，Decode Worker 由此连接 |
| `set_xfer_handshake_metadata` | 两个 connector 都传播 | 确保 NIXL 握手监听器启动 |

**Prefill Worker 配置：**

```bash
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
DYN_KVBM_CPU_CACHE_GB=20 \
CUDA_VISIBLE_DEVICES=1 \
python -m dynamo.vllm --model $MODEL \
  --disaggregation-mode prefill \
  --kv-transfer-config '{
    "kv_connector": "PdConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "connectors": [
        {"kv_connector": "DynamoConnector",
         "kv_connector_module_path": "kvbm.vllm_integration.connector",
         "kv_role": "kv_both"},
        {"kv_connector": "NixlConnector",
         "kv_role": "kv_both"}
      ]
    },
    "kv_connector_module_path": "kvbm.vllm_integration.connector"
  }' \
  --kv-events-config '{
    "publisher": "zmq",
    "topic": "kv-events",
    "endpoint": "tcp://*:20081",
    "enable_kv_cache_events": true
  }'
```

**Decode Worker 配置：**

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m dynamo.vllm --model $MODEL \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_both"
  }'
```

注意：Decode Worker **不启用 KVBM**，只使用 `NixlConnector` 从 Prefill Worker 拉取 KV。

### 12.3 多 Prefill + 多 Decode 拓扑（2P+2D）

4 GPU 部署示例，2 个 Prefill Worker + 2 个 Decode Worker，配合 KV Router：

```
┌──────────────┐
│  Frontend    │  --router-mode kv
│  KV Router   │  根据 KV 命中率路由请求
└──────┬───────┘
       │
   ┌───┴───────────────────────┐
   │                           │
   ▼                           ▼
┌─────────────┐          ┌─────────────┐
│ Prefill #0  │   RDMA   │ Prefill #1  │
│ GPU 2       │◄────────►│ GPU 3       │
│ PdConnector │          │ PdConnector │
│ NIXL:20098  │          │ NIXL:20099  │
│ ZMQ:20082   │          │ ZMQ:20083   │
│ ZMQ_PUB:56001│         │ ZMQ_PUB:56003│
│ ZMQ_ACK:56002│         │ ZMQ_ACK:56004│
└──────┬──────┘          └──────┬──────┘
       │ RDMA                   │ RDMA
   ┌───┴──────┐           ┌────┴──────┐
   ▼          ▼           ▼           ▼
┌───────┐ ┌───────┐  ┌───────┐  ┌───────┐
│Decode │ │Decode │  │Decode │  │Decode │
│ GPU 0 │ │ GPU 1 │  │ GPU 0 │  │ GPU 1 │
│Nixl   │ │Nixl   │  │Nixl   │  │Nixl   │
└───────┘ └───────┘  └───────┘  └───────┘
```

**关键差异（相比 1P+1D）：**

1. **KV Router**：Frontend 使用 `--router-mode kv`，基于 Consolidator 提供的 KV 命中信息路由请求到最佳 Prefill Worker
2. **端口隔离**：每个 Prefill Worker 使用独立的端口号，避免 ZMQ 和 NIXL 冲突
3. **KVBM Leader 端口**：第二个 Prefill Worker 需要配置独立的 `DYN_KVBM_LEADER_ZMQ_PUB_PORT` 和 `DYN_KVBM_LEADER_ZMQ_ACK_PORT`
4. **Decode Worker** 标记 `--disaggregation-mode decode`

### 12.4 KV 事件流与路由

分离式部署中，KV 命中信息通过事件流传递给路由器：

```
Prefill Worker                  Consolidator                KV Router
     │                              │                         │
     │  KV 事件（ZMQ 发布）         │                         │
     │  KvCacheEvent::Create/Remove │                         │
     ├─────────────────────────────►│                         │
     │                              │  去重 + 聚合             │
     │                              │  (DedupTracker)         │
     │                              ├────────────────────────►│
     │                              │                         │
     │                              │  路由查询：              │
     │                              │  哪个 Prefill 有最多命中？│
     │                              │◄────────────────────────┤
     │                              │                         │
     │  请求路由到最优 Prefill       │                         │
     │◄─────────────────────────────┼─────────────────────────┤
```

配置 KV 事件的 Prefill Worker 会启用 `--kv-events-config`，将块生命周期事件发布到 ZMQ，Consolidator 消费后提供给 KV Router 做智能路由。

### 12.5 DynamoConnector——聚合模式的核心

`DynamoConnector` 是 `KVConnectorBase_V1` 协议的实现，内部按角色分裂：

```python
class DynamoConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = KvConnectorLeader(vllm_config, engine_id)
            self._worker = None
        elif role == KVConnectorRole.WORKER:
            self._worker = KvConnectorWorker(vllm_config, engine_id)
            self._scheduler = None
```

| 侧 | 实现类 | 职责 |
|----|--------|------|
| Scheduler | `KvConnectorLeader` | 缓存命中查找、状态分配、元数据构建、请求生命周期管理 |
| Worker | `KvConnectorWorker` | KV Cache 张量注册、onboard/offload 执行、完成状态追踪 |

元数据通过 `DynamoConnectorMetadata` 传递原始字节，Leader 端序列化，Worker 端反序列化后执行。

### 12.6 模式对比

| 维度 | 聚合式 | 分离式 |
|------|--------|--------|
| **GPU 数量** | 1 | 2+ (1P+1D, 2P+2D, ...) |
| **Prefill/Decode** | 同一 GPU | 不同 GPU |
| **KV Connector** | `DynamoConnector` | Prefill: `PdConnector[Dynamo+Nixl]`，Decode: `NixlConnector` |
| **KVBM 位置** | 唯一 Worker | 仅 Prefill Worker |
| **跨节点传输** | 无 | NIXL RDMA |
| **KV 路由** | 不需要 | KV Router + Consolidator |
| **延迟优势** | 简单部署 | Prefill/Decode 独立扩展，互不干扰 |
| **吞吐优势** | 受单卡限制 | 可水平扩展 Prefill/Decode 数量 |
| **典型场景** | 开发/测试/小规模 | 生产/大规模推理服务 |

### 12.7 关键配置变量

| 变量 | 作用 | 示例 |
|------|------|------|
| `DYN_KVBM_CPU_CACHE_GB` | G2 (Host) 缓存大小 | `20` |
| `DYN_KVBM_DISK_CACHE_GB` | G3 (Disk) 缓存大小 | `100` |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | NIXL RDMA 通信端口 | `20097` |
| `DYN_KVBM_LEADER_ZMQ_PUB_PORT` | KVBM Leader ZMQ 发布端口（多 Prefill 时需隔离） | `56003` |
| `DYN_KVBM_LEADER_ZMQ_ACK_PORT` | KVBM Leader ZMQ 确认端口（多 Prefill 时需隔离） | `56004` |
| `--disaggregation-mode` | Worker 角色：`prefill` / `decode` | `prefill` |
| `--router-mode` | Frontend 路由模式：`kv` 启用 KV 感知路由 | `kv` |
| `--kv-events-config` | KV 事件发布配置（ZMQ endpoint + topic） | `{"publisher":"zmq","endpoint":"tcp://*:20081"}` |

---

## 13. 总结

### 核心问题与 KVBM 的解法

KVBM 解决的核心问题是：**大语言模型推理中 KV 缓存的内存管理**。随着上下文长度增长，KV 缓存占用成为推理系统的首要瓶颈。KVBM 从四个维度系统性解决了这一问题：

| 维度 | 问题 | KVBM 解法 |
|------|------|-----------|
| **容量** | GPU 显存放不下所有 KV 缓存 | G1-G4 四级存储层次，冷块自动降级，热块自动升级 |
| **一致性** | 多副本间数据如何保持同步 | 写穿缓存策略，写入同时持久化到后端存储 |
| **共享** | 跨节点/跨实例如何复用 KV 缓存 | Session 协议 + NIXL RDMA 传输，远程块如同本地访问 |
| **扩展** | 如何随负载水平扩展 | 事件驱动架构 + 分离式部署，Prefill/Decode 独立伸缩 |

### 架构分层回顾

KVBM 的设计遵循清晰的分层原则，每一层只关注自己的职责：

```
┌─────────────────────────────────────────────────────────────────┐
│  部署层        聚合式 / 分离式 / KV Router 路由                 │
├─────────────────────────────────────────────────────────────────┤
│  接入层        Framework Connector（Leader-Worker 分裂）         │
│                DynamoConnector / PdConnector / TRT-LLM          │
├─────────────────────────────────────────────────────────────────┤
│  逻辑层        Block State Machine                              │
│                Typestate 保证编译期安全，RAII 保证运行期不泄漏    │
├─────────────────────────────────────────────────────────────────┤
│  协调层        Event Plane                                      │
│                发布-订阅解耦，Consolidator 去重，驱动 KV Router  │
├─────────────────────────────────────────────────────────────────┤
│  物理层        G1-G4 存储管理 + Offload/Onboard 流水线           │
│                五阶段流水线：策略→批量→等待→升级→传输             │
├─────────────────────────────────────────────────────────────────┤
│  传输层        NIXL 抽象                                        │
│                UCX(RDMA) / POSIX / GDS 后端自动选择              │
│                TransferStrategy: Memcpy/CudaAsync/Nixl/TwoHop   │
├─────────────────────────────────────────────────────────────────┤
│  分布式层      Session 协议（Initiator/Responder）               │
│                元数据交换 → 分布式搜索 → Staging → RDMA 拉取    │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计原则

1. **类型安全优先**：Rust Typestate Pattern 将块生命周期状态编码到类型系统中，非法状态转换在编译期被拒绝，而非运行时崩溃

2. **零拷贝传输**：NIXL + RDMA 实现跨节点 KV 传输无需 CPU 参与，GPU 显存直接到远端 GPU 显存

3. **写穿一致性**：所有 KV 写入同步持久化到后端存储，确保节点间缓存一致性，代价是写延迟可接受（读远多于写）

4. **异步流水线**：五阶段 Offload 流水线将策略决策、批量收集、前置等待、引用升级、数据传输解耦并行，最大化吞吐

5. **事件驱动协调**：Event Plane 发布-订阅模式将块生命周期事件与消费者完全解耦，Consolidator 去重后驱动 KV Router 做智能路由

6. **渐进式部署**：从单卡聚合式起步，无缝切换到分离式部署，同一套 KVBM 核心代码覆盖两种模式

### 数据流全景

```
                    ┌──────── 请求进入 ────────┐
                    │                         │
                    ▼                         │
            ┌──────────────┐                  │
            │ Leader 查找   │                  │
            │ G2/G3 缓存命中│                  │
            └──────┬───────┘                  │
                   │                          │
         ┌─────────┴─────────┐                │
         ▼                   ▼                │
    命中 → Onboard       未命中 → Prefill     │
    G2/G3 → G1          GPU 计算 KV           │
         │                   │                │
         └─────────┬─────────┘                │
                   ▼                          │
            ┌──────────────┐                  │
            │ Decode 生成   │                  │
            │ 逐 token 输出 │                  │
            └──────┬───────┘                  │
                   │                          │
                   ▼                          │
            ┌──────────────┐                  │
            │ Offload 冷块  │                  │
            │ G1 → G2 → G3  │                 │
            │  (auto_chain) │                  │
            └──────┬───────┘                  │
                   │                          │
                   ▼                          │
            ┌──────────────┐                  │
            │ Event 发布    │                  │
            │ Create/Remove │──── Consolidator │
            └──────────────┘           │      │
                                       ▼      │
                                  KV Router    │
                                  智能路由 ────┘
```

### 一句话总结

**KVBM 是 NVIDIA Dynamo 推理框架的 KV 缓存基础设施，通过四级存储层次、写穿缓存、类型安全的块状态机、事件驱动协调和 NIXL 零拷贝传输，实现了 KV 缓存的跨层级管理、跨节点共享和跨框架统一，支撑从单卡到多节点的大规模 LLM 推理服务。**

---

## 14. 术语表

### A

| 术语 | 英文 | 释义 |
|------|------|------|
| Auto-Chain | Auto-Chain | 自动链式降级，G1→G2 offload 完成后自动将块送入下游 G2→G3/G4 流水线 |

### B

| 术语 | 英文 | 释义 |
|------|------|------|
| Block | KV Block | KV 缓存的基本管理单元，包含固定数量的 token 对应的 Key/Value 张量数据 |
| BlockDuplicationPolicy | BlockDuplicationPolicy | 块重复策略，控制同一 SequenceHash 是否允许多个物理块（Allow/Reject） |

### C

| 术语 | 英文 | 释义 |
|------|------|------|
| CompleteBlock | CompleteBlock | 处于 Staged 状态的块，已完成数据写入并分配了 SequenceHash，等待注册 |
| Consolidator | Consolidator | 多源事件去重合并器，消费多个 KVBM 实例发布的 KV 事件，去重后提供给 KV Router |
| Connector | Framework Connector | KVBM 与推理框架之间的桥梁组件，采用 Leader-Worker 分裂模式 |
| ConnectorMetadata | ConnectorMetadata | Leader 与 Worker 之间的序列化通信协议，携带 onboard/offload 操作指令 |

### D

| 术语 | 英文 | 释义 |
|------|------|------|
| DeviceStorage | DeviceStorage | G1 层存储后端，管理 CUDA VRAM 上的 KV 块，支持包装 torch 张量 |
| DiskStorage | DiskStorage | G3 层存储后端，使用 O_DIRECT 文件 I/O，支持 GDS 直传 |
| DynamoConnector | DynamoConnector | KVBM 的核心框架连接器，实现 vLLM KVConnectorBase_V1 协议，内部按角色分裂为 Leader/Worker |
| Drop | Drop (Rust) | Rust 的 RAII 析构机制，变量离开作用域时自动调用，用于释放块资源 |

### E

| 术语 | 英文 | 释义 |
|------|------|------|
| Event Batcher | EventBatcher | 事件批量处理器，积攒事件到阈值后刷新，支持按 block_id 排序减少后端压力 |
| Event Plane | Event Plane | 事件平面，KVBM 的发布-订阅协调层，广播 KV 块生命周期事件 |
| EventsManager | EventsManager | 事件生成核心，为 BlockManager 提供事件发布接口，内部包含 Policy、Batcher、Publisher |
| Eviction | Eviction | 驱逐，将 Inactive 池中长期未访问的块回收为 MutableBlock（Reset），归还空闲列表 |
| ExternalBlock | ExternalBlock | 外部块，G1 层的块引用方式，GPU 内存由推理框架拥有，KVBM 只持有 block_id + seq_hash |

### G

| 术语 | 英文 | 释义 |
|------|------|------|
| G1 | GPU HBM | 第一级存储，GPU 显存，速度最快容量最小，存放推理活跃 KV 缓存 |
| G2 | Host Pinned DRAM | 第二级存储，CPU 锁页内存，GPU DMA 高效传输的暂存区 |
| G3 | NVMe/SSD | 第三级存储，本地磁盘，支持 GDS 直传或 POSIX 回退 |
| G4 | S3/MinIO/Object | 第四级存储，远程对象存储，容量最大延迟最高 |
| GDS | GPUDirect Storage | NVIDIA 技术，允许 GPU 直接访问存储设备，绕过 CPU 内存拷贝 |

### I

| 术语 | 英文 | 释义 |
|------|------|------|
| ImmutableBlock | ImmutableBlock | 处于 Registered 状态的块，数据不可变，持有强引用计数，是 KVBM 最重要的块类型 |
| InitiatorSession | InitiatorSession | 发起方会话，请求远程块的节点创建，负责搜索远端布局并拉取数据 |
| Inactive Pool | InactiveBlockPool | 不活跃块池，存储 Primary Drop 后的块，支持被驱逐或通过 WeakBlock 复活 |

### K

| 术语 | 英文 | 释义 |
|------|------|------|
| KV Cache | Key-Value Cache | Transformer 推理中缓存的历史 Key/Value 张量，避免重复计算 |
| KV Router | KV Router | KV 感知路由器，根据 Consolidator 提供的 KV 命中信息将请求路由到最优 Worker |

### L

| 术语 | 英文 | 释义 |
|------|------|------|
| Leader | KvConnectorLeader | Connector 的调度侧实现，运行在 Scheduler 进程，负责缓存查找和决策 |
| LifecyclePinRef | LifecyclePinRef | 类型擦除的 keepalive 引用，只要存活就阻止块从 Active 转为 Inactive |
| LocalTransferEngine | LocalTransferEngine | Leader 侧的后台传输引擎，处理本节点内的 G1↔G2/G3 块搬运 |

### M

| 术语 | 英文 | 释义 |
|------|------|------|
| MutableBlock | MutableBlock | 处于 Reset 状态的块，可写入数据，尚未关联 SequenceHash |
| MultiConnector | MultiConnector | vLLM 的多连接器基类，PdConnector 继承此类组合多个子连接器 |

### N

| 术语 | 英文 | 释义 |
|------|------|------|
| NCCL | NVIDIA Collective Communications Library | NVIDIA 集合通信库，用于 GPU 间数据同步，ReplicatedData 模式下 broadcast KV 块 |
| NIXL | NVIDIA Infrastructure Exchange Layer | NVIDIA 基础设施交换层，高性能数据传输抽象，支持 UCX(RDMA)/POSIX/GDS 后端 |
| NixlConnector | NixlConnector | vLLM 的 NIXL 连接器，处理跨节点 RDMA KV 传输 |
| NixlStorage | NixlStorage | 实现了 NIXL 注册接口的存储后端，支持通过 NIXL Agent 进行远程访问 |

### O

| 术语 | 英文 | 释义 |
|------|------|------|
| ObjectStorage | ObjectStorage | G4 层存储后端，通过 NIXL OBJ 后端访问远程对象存储（S3/MinIO） |
| Offload | Offload | 降级，将 KV 块从高速层搬到低速层（如 G1→G2），释放高速层空间 |
| Onboard | Onboard | 升级，将 KV 块从低速层搬回高速层（如 G2→G1），用于缓存命中后复用 |

### P

| 术语 | 英文 | 释义 |
|------|------|------|
| PdConnector | PdConnector | Prefill-Decode 连接器，组合 DynamoConnector(KVBM) + NixlConnector(RDMA)，用于分离式部署 |
| PinnedStorage | PinnedStorage | G2 层存储后端，CUDA 锁页内存，NUMA 感知分配，GPU DMA 零拷贝 |
| PowerOfTwoPolicy | PowerOfTwoPolicy | 2 的幂次发射策略，仅在块数为 2 的幂次时发射事件，减少事件量 |
| Primary | Primary | 主块角色，持有数据的规范副本，Drop 时转入 Inactive 池而非直接释放 |
| PagedAttention | PagedAttention | vLLM 的注意力机制实现，将 KV 缓存按块管理，借鉴操作系统分页思想 |

### R

| 术语 | 英文 | 释义 |
|------|------|------|
| RAII | Resource Acquisition Is Initialization | C++/Rust 资源管理范式，构造时获取资源，析构时自动释放，KVBM 用其保证块不泄漏 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问，网卡直接读写远端内存，无需 CPU 参与，延迟极低 |
| ReplicatedData | ReplicatedData | 并行模式，仅 rank-0 持有 G2/G3，onboard 后通过 NCCL 广播给其他 rank，适用于 MLA 模型 |
| ResponderSession | ResponderSession | 响应方会话，拥有数据的节点创建，为 Initiator 提供 NIXL 描述符并等待拉取 |
| Resurrection | WeakBlock Resurrection | 弱引用块复活，通过 Arc::upgrade（快速路径）或 BlockRegistry 查找 Inactive 池（慢速路径）恢复强引用 |

### S

| 术语 | 英文 | 释义 |
|------|------|------|
| SequenceHash | SequenceHash | 序列哈希，KV 块的唯一标识，基于 token 序列内容计算，用于跨节点缓存匹配 |
| Session Protocol | Session Protocol | 分布式块搜索与传输协议，InitiatorSession（请求方）+ ResponderSession（响应方） |
| SlotState | SlotState | Connector 槽位状态，描述请求在 Connector 中的生命周期阶段 |
| SourceBlock | SourceBlock | Offload 流水线中的源块引用，区分 External（G1）、Strong、Weak 三种持有方式 |
| Staged | Staged | 块状态之一，数据已写入完成并分配了 SequenceHash，等待注册到 BlockManager |
| Typestate Pattern | Typestate Pattern | 类型状态模式，将状态编码到 Rust 类型系统中，非法状态转换在编译期被拒绝 |

### T

| 术语 | 英文 | 释义 |
|------|------|------|
| TensorParallel | TensorParallel | 张量并行模式，所有 worker rank 都拥有 G1+G2+G3 存储层 |
| TransferStrategy | TransferStrategy | 传输策略，根据源/目标存储类型和位置自动选择：Memcpy/CudaAsync/NixlRead/NixlWrite/TwoHop |
| TwoHop | TwoHop Transfer | 两跳传输，数据经中间节点中转到达目标（如 G3→G2→G1） |

### W

| 术语 | 英文 | 释义 |
|------|------|------|
| WeakBlock | WeakBlock | 弱引用块，不阻止 Primary 被驱逐，用于 auto_chain 和观察者场景 |
| Worker | KvConnectorWorker | Connector 的执行侧实现，运行在 Model Runner 进程，负责 KV 张量注册和传输执行 |
| Write-Through Cache | Write-Through Cache | 写穿缓存，写入缓存时同步持久化到后端存储，确保多节点间强一致性 |

### Z

| 术语 | 英文 | 释义 |
|------|------|------|
| ZMQ | ZeroMQ | 高性能异步消息库，KVBM 用于 Event 发布和 Leader-Worker 分布式协调 |
