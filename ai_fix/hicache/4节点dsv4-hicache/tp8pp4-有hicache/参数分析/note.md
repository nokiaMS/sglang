- 结论：
  - dsv4代码中强制要求 kv_cache_dtype 为 FP8_E4M3，因此在dsv4的性能测试中不需要对此参数的各个值做对比了。
  - DSV4 PP=4 + HiCache 配置下必须使用 --disable-cuda-graph。
    - 根因：CUDA Graph 与 DSV4 PP=4 HiCache 模式不兼容。CUDA Graph 固化了 decode 阶段的 GPU 计算图，但 HiCache 的异步 Host↔Device DMA
      搬运操作需要在运行时动态调度，破坏了 CUDA Graph 的静态性。当 decode 使用 CUDA Graph 后约3-4秒，gloo 通信断开导致服务崩溃。此问题与并发数无关。
    - chunked_prefill_size调整为4096，默认是8192，C10,C30,C60正常运行。
  - DeepSeek V3/R1 8 卡	--tp 8 --ep 8 --moe-a2a-backend deepep，标准配置  ---- DSV4+H800 不支持此配置。
- TODO:
  - 验证cp
  - 投机解码
  **- 23.4.2 微批调度**

## TP/PP/DP 参数组合建议

- 当前 4 节点 H800 环境的最佳组合建议为：`TP=8, PP=4, DP=1`。
- 硬件规模为 4 台 H800，每台 8 张 GPU，共 32 张 GPU。`TP=8, PP=4` 正好覆盖 32 张 GPU，可以把每个节点的 8 张卡都用满。
- `DP=1` 可以避开当前 SGLang 在 `PP + multi-node DP attention` 场景下暴露的问题。此前测试 `TP=4, PP=4, DP=2` 时，multi-node DP 必须开启 `--enable-dp-attention`，但随后触发 `_DpGatheredBufferWrapper._dp_max_padding` 未初始化异常，导致服务崩溃。
- 对 DSV4-Pro 这类大模型，`PP=4` 能把模型层切到 4 个 pipeline stage，显存压力比 `PP=1` 更合理，也与当前已验证的 `dsv4-pro-sglang-agg-tp8pp4-mem095-maxreq64-swa02-cps4096` 基线配置一致。
- 不建议当前使用 `TP=4, PP=4, DP=2`：虽然表面上 `4 * 4 * 2 = 32`，但实测会进入 SGLang 的 multi-node DP attention 问题路径。
- 不建议当前使用 `TP=8, PP=1, DP=8`：该组合在 SGLang 的资源划分下很可能只使用约 8 张 GPU，无法吃满 32 卡；同时 `PP=1` 会显著增加单卡权重显存压力，在 H800 80G 上存在较高 OOM 风险。
- 结论：当前最稳、最符合硬件利用率和显存约束的组合是 `TP=8, PP=4, DP=1`。

## CP 并行参数建议

- 在当前 `TP=8, PP=4, DP=1` 配置下，`CP=4` 和 `CP=8` 从 SGLang 参数约束上都可以尝试。
- SGLang 源码约束为：
  - `tp_size % attn_cp_size == 0`
  - `tp_size % (dp_size * attn_cp_size) == 0`
- 因此在 `TP=8, DP=1` 下：
  - `CP=4` 满足 `8 % 4 == 0`，可以启动尝试。
  - `CP=8` 满足 `8 % 8 == 0`，也可以启动尝试。
- 启动参数写法：
  - `--attention-context-parallel-size 4`
  - `--attention-context-parallel-size 8`
- 但不建议直接使用 `CP=8`。CP 越大，attention context 被切得越细，额外通信和调度开销越高。
- 已完成的 `CP=2` 测试显示，相比无 CP 的 `TP=8, PP=4, DP=1` 基线，C10/C30/C60 吞吐均略有下降，说明当前 50000 input len 的压测场景下 CP 没有体现收益。
- 建议验证顺序：先测 `CP=4`；如果 `CP=4` 仍无收益，则不建议继续测 `CP=8`。
- 当前默认推荐仍是：`TP=8, PP=4, DP=1, CP=1`。
