# SGLang KVCache 命中率提升方案

> 基于 `E:\codex_home\code\sglang` 当前源码分析。核心源码入口：
> `python/sglang/srt/mem_cache/radix_cache.py`、
> `python/sglang/srt/managers/schedule_policy.py`、
> `python/sglang/srt/managers/schedule_batch.py`、
> `python/sglang/srt/mem_cache/registry.py`、
> `python/sglang/srt/mem_cache/unified_radix_cache.py`、
> `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`。

## 1. 目标与判断标准

这里的“缓存命中率”主要指 runtime prefill prefix cache 命中率，而不是 router 层的近似 affinity 命中。当前统计链路在 `PrefillAdder` 中累加：

```text
hit_tokens = len(req.prefix_indices)
input_tokens = ceil_page(req.extend_input_len)
cache_hit_rate = effective_hit_tokens / (effective_input_tokens + effective_hit_tokens)
```

对应源码：

- `schedule_policy.py::match_prefix_for_req()`：写入 `req.prefix_indices`、`req.host_hit_length`、`req.num_matched_prefix_tokens`。
- `schedule_policy.py::PrefillAdder._update_prefill_budget()`：写入 `log_hit_tokens` 和 `log_input_tokens`。
- `metrics_reporter.py`：扣除 retracted 后计算 `sglang:cache_hit_rate`。
- `schedule_batch.py` / `output_streamer.py`：输出请求级 `cached_tokens` 和 `cached_tokens_details`。

优化目标不应只看 `sglang:cache_hit_rate`，还要同时看 TTFT、prefill throughput、evict 次数、retract 次数、device/host/storage 命中来源。Host/storage 命中提高但 TTFT 变差时，不应算有效优化。

## 2. 当前命中率的关键瓶颈

### 2.1 命名空间过细导致无法共享

`RadixKey` 的匹配条件不是只有 token 序列，还包括 `extra_key`。源码中 `RadixKey._check_compatible()` 要求 `extra_key` 完全相同，不同 LoRA、`cache_salt`、请求级 `extra_key` 会形成互相隔离的 cache namespace。

影响：

- 每个用户、请求或 session 都随机生成 `cache_salt` 会让相同系统提示词无法共享。
- 多 LoRA 服务天然不能跨 adapter 共享 KV。
- RAG 文档顺序、时间戳、request id 放在 prompt 前部，会破坏最长公共前缀。

### 2.2 `page_size` 向下取整损失短前缀

`RadixCache.match_prefix()` 和 `insert()` 都会调用 `RadixKey.page_aligned(self.page_size)`。如果 `page_size=64`，公共前缀 100 token 只按 64 token 命中，短于 64 token 的公共前缀直接不计入 device hit。

影响：

- 短系统提示词、短 few-shot 模板的统计命中率低。
- chunked prefill 的切分点如果不按 page 对齐，会产生尾部重复 prefill。

### 2.3 调度没有把相似请求聚在一起

`SchedulePolicy` 支持 `lpm`、`dfs-weight`、`routing-key`，但默认 `fcfs` 更重公平，不能主动聚合同前缀请求。源码里 `lpm` 在 waiting queue 超过 128 时会自动退化到 `fcfs`，避免 prefix match 计算开销；这会让大队列场景下相似请求重新被打散。

`in-batch prefix caching` 只在已有缓存命中较短时，用临时 radix tree 识别 waiting queue 内部相同前缀，并把后续相似请求临时降优先级。它依赖两个环境变量：

- `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`
- `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`

### 2.4 KV pool 容量不足导致高价值前缀被驱逐

`RadixCache.evict()` 只从 `evictable_leaves` 驱逐，正在运行或被 session/load-back 锁住的节点会进入 `protected_size`。容量不足时，即使有热前缀，也可能因为可驱逐空间小、活跃请求多而被迫删除历史叶子。

直接相关参数：

- `--max-total-tokens`
- `--mem-fraction-static`
- `--max-running-requests`
- speculative decoding 预留 token
- streaming session 长期锁定的 KV

### 2.5 驱逐策略只看节点级局部信号

当前 `RadixCache` 的节点有 `last_access_time`、`hit_count`、`priority`，策略由 `get_eviction_strategy()` 选择，支持 `lru/lfu/fifo/mru/filo/priority/slru`。但驱逐发生在叶子节点上，现有策略缺少“前缀未来复用收益”的综合评分，例如 subtree 热度、共享长度、reload 成本、tenant 重要性。

## 3. P0：无需改代码的落地方案

### 3.1 启用正确 cache 实现

先确认启动日志：

```text
Tree cache initialized: source=... impl=...
```

原则：

- 不要设置 `--disable-radix-cache`，除非只做无缓存对照实验。
- 确认 `SGLANG_RADIX_FORCE_MISS=0`。
- 普通 MHA/MLA 服务优先使用默认 `RadixCache`。
- GPU KV 容量不足但长前缀复用明显时，开启 `--enable-hierarchical-cache`，让 GPU 被驱逐后的前缀仍可在 host/storage 命中。

推荐起点：

```bash
--schedule-policy lpm \
--chunked-prefill-size 8192 \
--radix-eviction-policy lru \
--mem-fraction-static 0.90
```

若热点前缀稳定、长尾请求多，再试：

```bash
--radix-eviction-policy lfu
```

或：

```bash
--radix-eviction-policy slru
```

### 3.2 规范请求侧前缀

把高复用内容放在 prompt 最前面，把高变化内容后移：

```text
推荐：system prompt -> 固定工具描述 -> 稳定 RAG 模板 -> 用户问题 -> 动态时间/请求 ID
避免：动态时间/trace id/随机 few-shot -> system prompt -> 用户问题
```

治理项：

- 不要给每个请求生成随机 `cache_salt`。
- `extra_key` 只承载确实需要隔离的维度，例如 tenant、cache schema version、LoRA adapter。
- RAG 文档排序要稳定，例如按 doc id 或固定 score tie-breaker。
- chat template 版本变化要显式进入低基数 `extra_key`，避免新旧模板混用污染 cache。
- 多 LoRA 时按 `lora_id` 路由到固定实例组。

### 3.3 调整 page 与 chunk 对齐

原则：

- `chunked_prefill_size` 尽量是 `page_size` 的整数倍。
- 如果业务主要公共前缀短于当前 `page_size`，优先评估降低 `page_size`，前提是 attention backend 支持。
- 如果业务主要是超长公共前缀，保持后端推荐 page size，重点调容量和路由。

### 3.4 控制活跃请求对 protected KV 的挤压

当 `protected_size` 高、`evictable_size` 低、retract 增多时，命中率下降通常不是匹配问题，而是容量被活跃请求占满。

调参顺序：

1. 提高 `--max-total-tokens` 或 `--mem-fraction-static`。
2. 降低过高的 `--max-running-requests`。
3. 降低 speculative 预留压力，或确认 `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` 没有过小/过大导致 admission 失真。
4. streaming session 场景限制 session 数或 session KV 上限。

## 4. P1：建议优先实现的代码级优化

### 4.1 增加“收益感知”的驱逐策略

位置：

- `python/sglang/srt/mem_cache/evict_policy.py`
- `python/sglang/srt/mem_cache/utils.py`
- `python/sglang/srt/mem_cache/radix_cache.py`

当前节点已有信号：

```text
last_access_time, creation_time, hit_count, priority, len(node.key)
```

建议新增策略 `reuse-score`：

```text
score = age_weight * normalized_idle_time
      - hit_weight * log1p(subtree_hit_count)
      - len_weight * log1p(subtree_token_len)
      - priority_weight * priority
      + reload_cost_weight * is_host_backed_penalty
```

驱逐时优先删除 score 高的叶子。这里的核心不是简单 LFU，而是保留“长、热、重载成本高、优先级高”的前缀。

实现步骤：

1. 在 `TreeNode` 增加可选 `subtree_hit_count`、`subtree_token_len`，或在驱逐建堆前沿 parent 聚合。
2. 在 `_inc_hit_count()` 时向祖先累计热度，避免只统计命中终点。
3. 在 `get_eviction_strategy()` 注册 `reuse-score`。
4. 增加单测覆盖：热长前缀不应被冷短前缀挤掉；priority 高的节点在相同热度下后驱逐。

风险：

- 每次 hit 向祖先更新会增加 CPU 开销。可先只在 insert/match 终点更新，驱逐时懒聚合。
- 策略过度保护长前缀可能损害短 prompt 延迟，需要用 workload A/B。

### 4.2 放宽 LPM 大队列退化阈值，改成预算式 prefix match

位置：`schedule_policy.py::SchedulePolicy._determine_active_policy()`。

当前逻辑：

```python
if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
    return CacheAgnosticPolicy.FCFS
```

问题：高并发时恰恰更需要 cache-aware 聚合，但直接全量 LPM 又可能 CPU 开销过大。

建议改造：

- 新增参数或环境变量：`SGLANG_LPM_MAX_MATCH_REQUESTS`，默认 128。
- 当 waiting queue 很长时，只对前 N 个或按 routing_key 分桶后的每桶 top K 做 prefix match。
- 对未参与 prefix match 的请求保持 FCFS。

示意：

```text
waiting_queue <= N：全量 LPM
waiting_queue > N：按 received time 取前 N + 每个 routing_key 补充 K 个，执行 LPM
```

收益：

- 避免大队列完全退化为 FCFS。
- 对热点前缀请求集中到 batch 更友好。

风险：

- 调度公平性略降，需要保留老化机制，例如等待超过阈值强制参与。

### 4.3 强化 in-batch prefix caching：从“降优先级”改为“leader/follower 分组”

位置：`schedule_policy.py::_compute_prefix_matches()` 和 `PrefillAdder.add_one_req()`。

当前做法：发现 waiting queue 内部长前缀相同后，把后续请求临时降优先级，等待 leader 先跑并写入 cache。

建议：

- 为相同 in-batch prefix 建立 group id。
- 每组本轮最多放入一个 leader。
- follower 的优先级不是简单放到最后，而是在 leader 完成 `cache_unfinished_req` 或 `cache_finished_req` 后，下轮优先调度。

实现路径：

1. 在 `_compute_prefix_matches()` 中记录 `req.in_batch_prefix_group_key` 和 `req.in_batch_prefix_leader`。
2. scheduler 在一轮 prefill 后，如果 leader 已写入 cache，把同组 follower 移到 waiting queue 前部。
3. 指标增加：in-batch leader 数、follower 延迟轮数、follower 二次命中 token。

收益：

- 对并发到达的一批相同系统提示词/RAG 模板请求，减少重复 prefill。
- 比单纯降优先级更可控。

风险：

- 如果 leader 被 retract/abort，follower 会额外等待。需要超时或 fallback。

### 4.4 请求级 cache namespace 规范化

位置：

- `schedule_batch.py::Req.__init__()`
- OpenAI/native request 解析处
- `RadixKey(extra_key=...)`

建议引入结构化 `cache_namespace`，替代随意拼接字符串的 `extra_key`：

```text
cache_namespace = hash({
  model_id,
  tokenizer_revision,
  chat_template_version,
  lora_id,
  tenant_cache_group,
  cache_schema_version
})
```

并明确禁止将 request id、timestamp、trace id 进入 namespace。

收益：

- 降低误隔离。
- 便于 metrics 按 namespace 统计命中率。

风险：

- 需要兼容已有 `extra_key` 行为。建议先以 opt-in 参数启用。

### 4.5 对 page 尾部做“软命中”统计与可选复用

位置：`radix_cache.py::RadixKey.match()`、`match_prefix_for_req()`、metrics。

当前 page 对齐导致尾部 partial page 不计入 `prefix_indices`。建议先做观测，再决定是否复用：

- 新增 `raw_matched_len` 和 `page_aligned_matched_len` 指标。
- 暴露 `page_tail_lost_tokens = raw - aligned`。
- 当尾部损失长期很高时，指导降低 `page_size` 或调整 prompt/chunk 对齐。

可选更进一步：

- 对非 paged backend 或 `page_size=1` 后端保持现状。
- 对支持 token 粒度 KV index 的 backend，允许 partial tail 进入 `prefix_indices`，但需要严格验证 attention backend 是否允许非整页 prefix。

风险：

- 很多 paged attention 后端依赖整页布局，直接复用 partial page 可能破坏 kernel 假设。建议先只做统计。

## 5. P2：多级缓存与多实例路由方案

### 5.1 HiCache 使用条件

位置：

- `mem_cache/registry.py::default_radix_cache_factory()`
- `mem_cache/unified_radix_cache.py`
- `mem_cache/hiradix_cache.py`

适合开启 HiCache 的场景：

- 长前缀复用明显。
- GPU KV pool 放不下全部热点。
- TTFT 可以接受 host 到 device 的 load-back。

推荐：

```bash
--enable-hierarchical-cache \
--hicache-size <host_cache_size> \
--hicache-write-policy write_through
```

如果 host 写带宽成为瓶颈，再评估：

```bash
--hicache-write-policy write_through_selective
```

或：

```bash
--hicache-write-policy write_back
```

注意：

- `UnifiedRadixCache` 中 `load_back_threshold` 默认很小，短 host hit 不一定值得加载。
- Host/storage 命中要看 TTFT，不是越高越好。
- storage backend 只适合跨重启、跨实例或超长前缀复用明显的场景。

### 5.2 Router 层做 prefix locality

多实例部署时，runtime 内部命中率的前提是相似请求落到同一 worker。router 层可以使用：

- `cache_aware`：基于请求文本的近似 radix tree。
- `prefix_hash`：基于前 N 个 token 的一致性 hash。
- `routing-key`：上游明确提供稳定 key。

建议：

- 固定 system prompt / tenant / LoRA 场景：优先 `prefix_hash` 或显式 routing key，稳定且开销低。
- RAG 长文本相似但不是完全相同：使用 `cache_aware`。
- PD disaggregation：prefill pool 更应该 cache-aware；decode pool 更关注负载和连接局部性。

router 命中只是“更可能有 KV”，最终以 runtime 的 `cached_tokens` 和 `sglang:cache_hit_rate` 为准。

## 6. 观测闭环

### 6.1 必看指标

Prometheus/runtime：

- `sglang:cache_hit_rate`
- `sglang:cached_tokens_total{source="device|host|storage"}`
- prefill input throughput
- TTFT P50/P90/P99
- retract 请求数和 reprocessed tokens
- KV pool available/evictable/protected
- HiCache load-back tokens、load-back duration

请求级：

- `meta_info.cached_tokens`
- `meta_info.cached_tokens_details.device`
- `meta_info.cached_tokens_details.host`
- `meta_info.cached_tokens_details.storage`

建议新增指标：

- `radix_page_tail_lost_tokens_total`
- `radix_match_raw_tokens_total`
- `radix_match_aligned_tokens_total`
- `radix_eviction_tokens_total{policy,reason}`
- `radix_eviction_reuse_score`
- `in_batch_prefix_groups_total`
- `in_batch_prefix_followers_hit_tokens_total`
- `cache_namespace_cardinality`

### 6.2 A/B 验证矩阵

至少跑四组：

| 组别 | 配置 | 目的 |
|---|---|---|
| baseline | 当前线上参数 | 真实基线 |
| force miss | `SGLANG_RADIX_FORCE_MISS=1` | 估算 cache 收益上限 |
| LPM | `--schedule-policy lpm` | 验证调度聚合收益 |
| LPM + LFU/SLRU | 加 `--radix-eviction-policy lfu/slru` | 验证热点保留收益 |
| HiCache | 加 `--enable-hierarchical-cache` | 验证 GPU 容量不足时的 host 命中收益 |

每组至少记录：

```text
cache_hit_rate
cached_tokens/device/host/storage
TTFT P50/P90/P99
prefill tokens/s
decode tokens/s
evict tokens/s
retract count
GPU memory headroom
```

## 7. 推荐实施顺序

### 第一阶段：参数与请求治理

1. 确认 `Tree cache initialized` 是预期实现。
2. 移除随机 `cache_salt` / 高基数 `extra_key`。
3. 统一 prompt 前缀结构，把动态字段后移。
4. 开启 `--schedule-policy lpm`。
5. 在 `lru/lfu/slru` 中压测选择最优 eviction policy。
6. 调高 KV pool 容量，直到 evict 不再频繁或 TTFT 收益边际变小。

### 第二阶段：调度改造

1. 实现预算式 LPM，避免 waiting queue > 128 时完全退化。
2. 实现 in-batch leader/follower 分组。
3. 增加 page tail lost 指标。

### 第三阶段：驱逐策略改造

1. 实现 `reuse-score` 驱逐策略。
2. 增加 subtree 热度和长度统计。
3. 做单测和 replay 压测，确认热长前缀保留率提高。

### 第四阶段：多级缓存与路由

1. GPU 容量瓶颈明确后开启 HiCache。
2. 多实例部署时引入 `prefix_hash` / `cache_aware` router。
3. 按 tenant/model/LoRA/prefix hash 做路由和指标分组。

## 8. 风险与边界

- 命中率提高不必然降低延迟：host/storage load-back 可能比重算更慢。
- `page_size` 不是纯命中率参数，attention backend 可能依赖固定页大小。
- LPM 会消耗 CPU 调度时间，大队列下必须有预算和老化机制。
- LoRA、不同 tokenizer/template、不同 positional override 不能强行共享 KV。
- streaming session 提升同 session 命中，但会长期占用 protected KV，影响全局共享。
- speculative decoding 会增加 KV 预留，对 prefix cache 容量有间接挤压。

## 9. 最小可执行建议

如果只做一轮改动，建议先执行：

```bash
--schedule-policy lpm \
--chunked-prefill-size 8192 \
--radix-eviction-policy lfu \
--mem-fraction-static 0.90
```

同时确保：

- 不使用 `--disable-radix-cache`。
- `SGLANG_RADIX_FORCE_MISS=0`。
- 请求不携带随机 `cache_salt`。
- `extra_key` 低基数且稳定。
- 上游按 model/tenant/LoRA/prefix 做稳定路由。

如果此时命中率仍低，优先排查 namespace 和 prompt 前缀一致性；如果命中先高后低，优先排查 KV pool 容量、evict 策略和 protected KV 占用。
