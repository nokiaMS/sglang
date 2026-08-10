# DeepSeek V4 算子列表整理方案

## 1. 文档目的

本文用于指导后续系统化查询、验证和整理 DeepSeek V4 在 SGLang 中使用的算子，最终形成可追溯、可复现、可用于芯片适配与性能优化的算子清单。

本文是执行方案，不是最终算子列表。后续执行应同时回答两个不同问题：

1. **指定部署配置实际执行了哪些算子？**
   - 对应一个确定的模型、权重精度、硬件、SGLang commit、启动参数、环境变量和输入负载。
   - 结论主要来自动态 Profiling。
2. **当前 SGLang 代码支持 DeepSeek V4 走哪些潜在算子路径？**
   - 包含不同硬件、精度、Attention/MoE 后端和优化开关下的条件分支。
   - 结论主要来自静态源码分析，再由分场景动态测试验证。

两类清单必须分别保存，不能把“源码中可能调用”写成“运行时一定调用”。

---

## 2. 目标与非目标

### 2.1 目标

- 建立统一的“算子”定义和分层口径。
- 找出 DeepSeek V4 推理主链路中的模型级、框架级、GPU/NPU Kernel 级和通信级算子。
- 区分 Prefill、Decode、MTP Draft/Verify 等阶段的算子差异。
- 识别不同模型版本、量化精度、硬件和后端导致的条件算子。
- 建立逻辑模块、源码函数、框架算子、设备 Kernel 之间的映射。
- 给每个最终条目附上源码或 Trace 证据。
- 统计调用次数、输入形状、耗时和耗时占比，为后续算子开发、性能优化和芯片适配提供依据。

### 2.2 非目标

- 本阶段不实现或优化任何算子。
- 本阶段不以单次 Trace 代表所有 DeepSeek V4 配置。
- 默认不统计模型下载、权重加载、权重转换和编译安装过程；如后续需要分析启动性能，应另建“加载期算子清单”。
- 不把纯 Python 调度函数直接等同于设备算子。

---

## 3. 算子口径

最终清单采用五层口径。每个条目必须注明所属层级。

| 层级 | 定义 | 示例 | 主要证据 |
| --- | --- | --- | --- |
| L1 模型逻辑算子 | 从模型结构理解的计算模块 | RMSNorm、RoPE、Attention、MoE Router、TopK、SwiGLU | 模型源码、配置 |
| L2 SGLang 实现算子 | SGLang 中可定位的 Python/C++/Triton/TileLang 调用单元 | `fused_q_norm_rope`、`mhc_fused_post_pre` | 源码符号、调用栈 |
| L3 框架算子 | PyTorch Dispatcher/Profiler 中的算子 | `aten::linear`、`aten::einsum`、自定义 `torch.ops.*` | Torch Profiler |
| L4 设备 Kernel | GPU/NPU 实际执行的 Kernel | FlashMLA、DeepGEMM、Triton、CUTLASS、NCCL Kernel | Nsight Systems、Torch Profiler |
| L5 通信与数据移动 | 跨卡通信、缓存读写和必要的数据重排 | all-reduce、all-gather、reduce-scatter、MoE dispatch/combine | Trace、通信后端源码 |

注意：一个 L1 算子可能融合成一个 L4 Kernel，也可能展开为多个 Kernel；多个 L1 算子也可能被融合成一个 Kernel。因此最终表中应保存多对多映射，而不是强制一一对应。

---

## 4. 调研范围冻结

正式采集前必须先生成一次 `run_manifest`，冻结以下信息：

- SGLang Git commit、分支及工作区状态。
- 模型 ID、本地模型目录、模型 revision/commit。
- `config.json`、量化配置和 `architectures`。
- DeepSeek V4 版本：Flash、Pro 或其他变体。
- 权重精度：BF16、FP8、MXFP4/NVFP4 等。
- 硬件型号、GPU/NPU 数量、计算能力。
- 驱动、CUDA/ROCm/CANN、PyTorch、Triton、SGLang Kernel 版本。
- TP、DP、EP、PP、CP 配置。
- Attention、MoE、通信和 Speculative Decode 后端。
- 全部启动参数和影响执行路径的环境变量。
- 输入长度、输出长度、Batch Size、并发数和数据集。

建议保留以下基础信息：

```bash
git rev-parse HEAD
git status --short
python -c "from sglang.version import __version__; print(__version__)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

实际模型配置必须从目标 checkpoint 读取，不能只使用 SGLang 配置类中的默认值：

```bash
python - <<'PY'
from transformers import AutoConfig

model_path = "deepseek-ai/DeepSeek-V4-Pro"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
for key in (
    "architectures",
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "compress_ratios",
    "window_size",
    "index_topk",
    "n_routed_experts",
    "num_experts_per_tok",
    "quantization_config",
):
    print(f"{key}: {getattr(config, key, None)}")
PY
```

---

## 5. 当前源码入口与候选算子族

后续静态分析至少覆盖以下入口：

| 范围 | 文件 | 重点 |
| --- | --- | --- |
| 模型主干 | `python/sglang/srt/models/deepseek_v4.py` | Decoder Layer、MHC、Q/KV/O 投影、RoPE、Attention 调用、输出头 |
| 模型配置 | `python/sglang/srt/configs/deepseek_v4.py` | 结构参数、压缩比例、Indexer、MoE、量化配置 |
| CUDA Attention | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | SWA、压缩 Attention、FlashMLA、Prefill/Decode 分支 |
| ROCm Attention | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` | HIP/AITER/ROCm 专用路径 |
| Indexer | `python/sglang/srt/layers/attention/dsv4/indexer.py` | Index Query、Hadamard、量化、MQA Logits、TopK |
| Compressor | `python/sglang/srt/layers/attention/dsv4/compressor.py`、`compressor_v2.py` | 压缩、Norm、RoPE、缓存写入 |
| MoE | `python/sglang/srt/models/deepseek_v2.py` | Router、Grouped TopK、Experts、Shared Expert、SwiGLU |
| MoE Kernel 后端 | `python/sglang/srt/layers/moe/` | Triton、DeepGEMM、FlashInfer、CUTLASS、AITER 等实现 |
| DSV4 Kernel 包装 | `python/sglang/kernels/ops/attention/dsv4/` | Triton/JIT/自定义融合算子入口 |
| MHC Kernel | `python/sglang/kernels/ops/layernorm/mhc.py` | Split/Sinkhorn、MHC pre/post 融合 |
| 内存池 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | SWA、压缩 KV、Indexer Cache 布局及写入 |
| NPU 路径 | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` | Ascend 专用 Attention 与算子 |
| NextN/MTP | `python/sglang/srt/models/deepseek_v4_nextn.py`、`deepseek_v4_dspark.py` | Draft/Verify 和辅助预测路径 |

第一轮预计会出现的逻辑算子族如下；这些只作为静态扫描的起始分类，不直接作为最终实测结论：

- Embedding、LM Head、Logits Processing。
- MHC mixing、RMS 统计、Sigmoid、Sinkhorn、MHC pre/post 及跨层融合。
- Q/KV/O Linear、Router GEMM、Expert GEMM、FP8/FP4 GEMM。
- RMSNorm、RMSNorm + 量化融合。
- RoPE、逆 RoPE、Norm + RoPE + KV Cache 写入融合。
- SWA、压缩稀疏 Attention、FlashMLA、Sparse Prefill。
- Compressor 投影、压缩、量化和缓存写入。
- C4 Indexer Query、Hadamard、FP8/FP4 量化、MQA Logits、TopK 与索引变换。
- MoE Router、Grouped TopK、Dispatch、Expert 计算、Combine、Shared Expert。
- SiLU/SwiGLU、clamped SwiGLU 及其量化融合。
- KV Cache 量化、反量化、Pack、Paged Cache 读写。
- all-gather、all-reduce、reduce-scatter、MoE all-to-all/dispatch/combine。

---

## 6. 总体执行流程

```text
冻结配置与范围
    -> 静态扫描得到“候选算子全集”
    -> 建立分支条件和源码调用链
    -> 设计最小覆盖场景矩阵
    -> Torch Profiler 获取框架算子和调用栈
    -> Nsight Systems 获取真实设备 Kernel 和通信
    -> 归一化、去重、建立多层映射
    -> 对静态未命中和动态无来源项进行补查
    -> 形成最终清单、差异清单和热点清单
```

执行时按以下阶段推进。

### 阶段 A：配置和路径识别

1. 保存目标 checkpoint 的完整配置。
2. 记录 SGLang 对该模型应用的默认 override。
3. 记录实际 Attention Backend、MoE Runner Backend、通信 Backend 和量化 Backend。
4. 从服务启动日志中保存最终生效配置，而不是只保存命令行原始参数。
5. 标记 `compress_ratios` 中出现的所有比例；当前实现重点关注 0、4、128 路径。
6. 标记是否启用 MTP/EAGLE、DP Attention、CP、CUDA Graph、多流重叠等。

阶段产物：`manifest.json`、`model_config.json`、`server_args.json`、`environment.txt`。

### 阶段 B：静态源码扫描

静态扫描目标是建立“候选算子全集”和“条件分支表”。

建议先执行：

```bash
rg -n "torch\.ops|sglang\.kernels\.ops|from sgl_kernel|deep_gemm|flash_mla|FusedMoE|RMSNorm|Linear" \
  python/sglang/srt/models/deepseek_v4.py \
  python/sglang/srt/models/deepseek_v4_nextn.py \
  python/sglang/srt/layers/attention/deepseek_v4_backend.py \
  python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py \
  python/sglang/srt/layers/attention/dsv4 \
  python/sglang/srt/models/deepseek_v2.py \
  python/sglang/srt/layers/moe
```

随后人工沿 `forward()` 调用链整理：

1. `DeepseekV4ForCausalLM.forward`
2. `DeepseekV4Model.forward`
3. `DeepseekV4DecoderLayer.forward`
4. `MQALayer.forward`
5. `DeepseekV4AttnBackend.forward`
6. `C4Indexer.forward`
7. `Compressor`/`CompressorBackendMixin`
8. `DeepseekV2MoE.forward` 及具体 MoE Runner

每个源码调用项至少记录：

- Python 符号和文件行号。
- 所属模型阶段和子模块。
- 进入该分支的条件。
- 预期底层实现：ATen、Triton、DeepGEMM、FlashMLA、CUTLASS、TileLang、AITER、NCCL 等。
- 是否存在融合以及被融合的逻辑算子。
- 是否只在特定硬件、精度、Batch/Token Shape 或 CUDA Graph 路径出现。

阶段产物：`static_candidates.csv`、`branch_conditions.md`、源码调用链图。

### 阶段 C：离线分阶段 Profiling

优先使用 `sglang.benchmark.one_batch`，原因是输入形状固定、可重复，并且可以分别采集 Prefill 和 Decode。

示例以 Linux/NVIDIA 环境为准，实际并行和后端参数应替换为目标部署配置：

```bash
export SGLANG_TORCH_PROFILER_DIR=$PWD/profile_artifacts/torch

python -m sglang.benchmark.one_batch \
  --model-path deepseek-ai/DeepSeek-V4-Pro \
  --batch-size 1 \
  --input-len 1024 \
  --output-len 16 \
  --profile \
  --profile-stage prefill \
  --profile-record-shapes \
  --profile-prefix dsv4_prefill \
  --profile-activities CPU GPU \
  <实际并行与后端参数>

python -m sglang.benchmark.one_batch \
  --model-path deepseek-ai/DeepSeek-V4-Pro \
  --batch-size 1 \
  --input-len 1024 \
  --output-len 16 \
  --profile \
  --profile-stage decode \
  --profile-record-shapes \
  --profile-prefix dsv4_decode \
  --profile-activities CPU GPU \
  <实际并行与后端参数>
```

Torch Profiler 重点提取：

- `cpu_op`/自定义 Operator 名称。
- CUDA Kernel 名称及其上层调用关系。
- 输入 Shape、调用次数和总耗时。
- Python/C++ 调用栈。
- Prefill 与 Decode 的差异。

正式采集前至少完成 2～3 次 Warmup，避免把首次 JIT 编译、Autotune 和缓存初始化误计入稳定执行算子集。需要单独研究首次执行时，可额外保存 `cold_start` Trace，但不得和稳态 Trace 混合统计。

### 阶段 D：在线真实配置 Profiling

离线结果需要用线上启动方式复核，特别是以下路径：

- CUDA Graph。
- Continuous Batching。
- DP Attention、TP/EP/PP/CP。
- DeepEP、MegaMoE 等 MoE 通信后端。
- MTP/EAGLE Draft 与 Verify。
- Chunked Prefill、Disaggregation、多流重叠。

推荐流程：

1. 使用生产等价参数启动服务。
2. 预热到 JIT、Autotune 和 CUDA Graph 捕获完成。
3. 调用 `/start_profile`，指定 `start_step` 和 `num_steps`。
4. 使用固定随机种子和固定请求集施加载荷。
5. 等待 Trace 落盘，并保存同批次的服务日志和请求描述。

Torch Profiler 请求示例：

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{
        "activities": ["CPU", "GPU"],
        "start_step": 10,
        "num_steps": 5,
        "record_shapes": true,
        "with_stack": true,
        "output_dir": "profile_artifacts/online_torch"
      }' &

# 此处运行固定的请求或 benchmark workload，并等待 profile 请求结束。
```

阶段产物：在线 Torch Trace、请求清单、服务日志、实际吞吐与时延结果。

### 阶段 E：Nsight Systems 设备 Kernel 采集

Torch Profiler 用于关联框架调用，Nsight Systems 用于得到更可靠的 CUDA Kernel、Stream、CUDA Graph 和 NCCL 时间线。两者应配套使用。

离线示例：

```bash
nsys profile \
  -c cudaProfilerApi \
  --capture-range-end stop \
  --force-overwrite=true \
  -o profile_artifacts/nsys/dsv4_decode \
  python -m sglang.benchmark.one_batch \
    --model-path deepseek-ai/DeepSeek-V4-Pro \
    --batch-size 1 \
    --input-len 1024 \
    --output-len 16 \
    --profile \
    --profile-stage decode \
    --profile-activities CUDA_PROFILER \
    <实际并行与后端参数>
```

在线服务可使用仓库测试中的同类方式：以 `nsys profile -c cudaProfilerApi` 包裹 `sglang.launch_server`，再通过 `/start_profile` 发送 `activities=["CUDA_PROFILER"]`。

采集后至少生成：

- CUDA Kernel 汇总：名称、调用次数、总时长、平均时长、占比。
- CUDA API 汇总。
- NCCL/通信 Kernel 汇总。
- Kernel 时间线，保留 Stream 和 CUDA Graph 信息。

可以复用仓库工具：

```bash
python examples/profiler/nsys_profile_tools/gputrc2graph.py \
  --in_file profile_artifacts/nsys/dsv4_decode.nsys-rep,sglang,ds,0 \
  --out_dir profile_artifacts/nsys/summary \
  --title "DeepSeek V4 Decode Kernel Summary"
```

如果工具无法识别新的 DeepSeek V4 Kernel 名称，只扩展其分类映射，不修改原始 Trace 和原始 Kernel 名称。

### 阶段 F：结果归一化与映射

同一类 Kernel 可能包含模板参数、Shape、数据类型和自动生成后缀。必须同时保留两个字段：

- `raw_kernel_name`：Trace 原始名称，用于审计。
- `normalized_kernel_name`：去除地址、编译 hash 和非语义后缀后的名称，用于聚合。

归一化时不能删除会改变实现语义的信息，例如：

- FP8/FP4/BF16。
- Prefill/Decode。
- Head Dim、Tile、Split-K 等关键模板参数。
- FlashMLA、DeepGEMM、CUTLASS、Triton、AITER 等实现来源。
- NCCL 操作类型。

映射顺序建议为：

1. 用 Torch Profiler 调用栈建立 L3 → L2 映射。
2. 用 Torch Profiler correlation 和时间区间建立 L3 → L4 映射。
3. 用源码调用链建立 L2 → L1 映射。
4. 用 Nsight 时间线补齐 Torch Profiler 未完整记录的 L4/L5 项。
5. 对无法映射的高耗时 Kernel 单独建 `unmapped_hot_kernels.csv`，不得直接归入“其他”后结束分析。

### 阶段 G：交叉验证与收敛

完成归并后进行三类核对：

1. **静态有、动态无**
   - 判断是否是其他硬件、精度、阶段、Shape 或开关才会触发。
   - 如果属于目标范围，补充最小触发用例。
2. **动态有、静态无**
   - 通过调用栈、扩展库和 JIT 生成目录反查来源。
   - 重点检查 cuBLAS、NCCL、内存操作和融合 Kernel。
3. **逻辑算子与 Kernel 数量不匹配**
   - 明确是融合、拆分、重计算、并行分片还是 Speculative Decode 多步调用造成。

每补充一个场景，都应更新覆盖矩阵和 manifest，避免无边界地重复采集。

---

## 7. 覆盖矩阵设计

“全部算子”不能靠一个用例覆盖，应采用“基线场景 + 差异场景”，避免做无意义的全笛卡尔积。

### 7.1 必选维度

| 维度 | 最低覆盖要求 |
| --- | --- |
| 模型 | 实际使用的 V4-Flash/V4-Pro；若两者都要支持则分别采集 |
| 精度 | 每种实际要部署的权重精度分别采集，如 FP8、FP4 |
| 硬件 | 每个目标硬件架构分别采集，不跨架构推断 |
| 阶段 | Prefill、Decode 必须分开 |
| Batch | Decode 至少覆盖低延迟小 Batch 和吞吐型大 Batch |
| Context | 至少覆盖短、中、长三类上下文，触发不同 Prefill/Indexer 分支 |
| 压缩层 | 确认配置中 0/4/128 等 `compress_ratio` 路径均在 Trace 中出现 |
| 并行 | 覆盖目标 TP/DP/EP/CP 配置，通信算子不能用单卡结果代替 |
| MoE 后端 | 覆盖生产计划使用的 Runner 和 A2A 后端 |
| 推测解码 | 生产启用 MTP/EAGLE 时，单独采集 Draft 和 Verify |
| CUDA Graph | 分别验证 Eager 与生产 CUDA Graph 路径，至少保留生产路径结果 |

### 7.2 推荐首轮场景

| 场景 ID | 目的 | 建议负载 |
| --- | --- | --- |
| S01 | Prefill 基线 | BS=1，Input=1024，仅采 Prefill |
| S02 | 长 Prefill/稀疏路径 | BS=1，Input=8192 或生产长上下文 |
| S03 | 低延迟 Decode | BS=1，稳定 Decode 5～20 step |
| S04 | 中 Batch Decode | BS=16 或生产常见 Batch |
| S05 | 高吞吐 Decode | BS=64/128，使用生产 DP/EP/MoE 后端 |
| S06 | MTP Draft/Verify | 使用生产 speculative 参数，分阶段标注 |
| S07 | 多卡通信 | 使用生产 TP/DP/EP，重点采集 NCCL/DeepEP/MegaMoE |

模型和精度变化不与所有 Shape 重复组合。先以一个主配置覆盖 S01～S07，再对其他模型/精度运行差异场景；若静态分析显示其后端完全不同，再扩展完整场景。

---

## 8. 最终数据结构

### 8.1 主清单 `dsv4_operator_catalog.csv`

建议字段：

| 字段 | 含义 |
| --- | --- |
| `operator_id` | 稳定 ID，例如 `DSV4-ATTN-001` |
| `logical_category` | Attention、MoE、MHC、Norm、Quant、Comm 等 |
| `logical_operator` | L1 逻辑算子名称 |
| `sglang_symbol` | L2 Python/C++/Triton 符号 |
| `source_location` | 文件和行号/函数 |
| `framework_operator` | L3 ATen 或自定义算子名 |
| `raw_kernel_name` | L4/L5 Trace 原始名称 |
| `normalized_kernel_name` | 归一化名称 |
| `implementation` | FlashMLA、DeepGEMM、Triton、CUTLASS、TileLang、AITER、NCCL 等 |
| `stage` | Prefill、Decode、Draft、Verify |
| `model_variant` | Flash、Pro 等 |
| `dtype` | BF16、FP8、FP4 等 |
| `hardware` | GPU/NPU 型号与架构 |
| `shape_signature` | 关键输入 Shape |
| `parallel_config` | TP/DP/EP/PP/CP |
| `condition` | 触发分支的配置、环境变量或 Shape 条件 |
| `call_count` | Trace 中调用次数 |
| `total_time_us` | 总设备耗时 |
| `time_percent` | 当前 Trace 耗时占比 |
| `evidence_type` | Source、Torch Trace、Nsight Trace |
| `evidence_path` | 证据文件路径及 Trace/场景 ID |
| `verification_status` | candidate、observed、mapped、verified |
| `notes` | 融合关系、限制和异常说明 |

### 8.2 配套产物

```text
notes/dsv4_operator_inventory/
├── manifests/
│   └── <scenario_id>/
│       ├── manifest.json
│       ├── model_config.json
│       ├── server_args.json
│       └── environment.txt
├── static/
│   ├── static_candidates.csv
│   ├── branch_conditions.md
│   └── call_graph.md
├── traces/
│   ├── torch/<scenario_id>/
│   └── nsys/<scenario_id>/
├── normalized/
│   ├── framework_ops.csv
│   ├── device_kernels.csv
│   ├── communication_ops.csv
│   └── unmapped_hot_kernels.csv
├── dsv4_operator_catalog.csv
├── dsv4_operator_summary.md
└── coverage_matrix.csv
```

原始 Trace 通常体积较大，是否提交 Git 应遵循仓库策略；即使不提交，也必须在 manifest 中记录存储位置和校验值。

---

## 9. 验收标准

只有同时满足以下条件，才能认为某个目标部署配置的算子清单整理完成：

1. 配置可复现：Git commit、模型 revision、硬件、软件版本、启动参数、环境变量和负载完整。
2. Prefill 和 Decode 已分别采集，生产启用 MTP 时 Draft/Verify 也已覆盖。
3. 目标 `compress_ratio`、Attention Backend、MoE Backend 和通信路径均有证据。
4. 清单中的每个“已执行”算子至少有一份动态 Trace 证据。
5. 清单中的每个高耗时设备 Kernel 都能映射到逻辑模块或被列入待分析项。
6. 设备耗时前 95% 的 Kernel 必须完成分类；其余未映射项应保留原始名称和证据。
7. Torch Profiler 和 Nsight Systems 的主要 Kernel 类别、调用阶段和耗时排序不存在无法解释的冲突。
8. 静态候选项已被标记为 `observed` 或注明未触发原因。
9. 至少重复采集两次，稳定路径的核心 Kernel 集合应一致；JIT/Autotune 差异单独说明。
10. 最终文档明确区分“该配置实测集合”和“当前代码潜在集合”。

---

## 10. 风险与处理原则

### 10.1 融合导致名称不一致

同一个融合 Kernel 可能覆盖 Norm、RoPE、Quant 和 Cache Store。处理时在 `logical_operator` 中记录多个逻辑标签，不要只按 Kernel 名称猜测单一功能。

### 10.2 CUDA Graph 隐藏上层调用关系

CUDA Graph Replay 时上层框架事件可能不足。先用 Eager/Torch Profiler 建立映射，再用生产 CUDA Graph/Nsight 验证实际 Kernel 集合和耗时。

### 10.3 首次 JIT 和 Autotune 污染

冷启动和稳态分开采集。正式清单以预热后的稳态 Trace 为准，JIT 编译相关活动进入启动期附表。

### 10.4 多 Rank Trace 重复计数

每个 Trace 必须携带 TP/DP/EP/PP Rank。既要提供单 Rank 视角，也要提供全系统聚合视角；不能简单把所有 Rank 调用次数相加后解释为单请求调用次数。

### 10.5 Kernel 名称被截断或模板化

保留原始 `.nsys-rep` 和 Torch Trace。归一化仅用于聚合，禁止覆盖原始名称。

### 10.6 单一输入无法触发全部分支

以静态分支表指导补充最小触发用例。不要依赖随机流量“碰到”低频路径。

### 10.7 `torch.fx`/模型结构清单不完整

DeepSeek V4 的动态控制流、分布式通信、自定义算子、JIT Kernel 和 CUDA Graph 使单纯 `torch.fx` 或 `named_modules()` 无法代表真实 Kernel。它们只能作为结构辅助信息，不能作为最终证据。

---

## 11. 建议执行顺序与里程碑

### M1：范围冻结与静态候选清单

- 确认目标模型、精度、硬件和生产启动配置。
- 生成 manifest。
- 完成主调用链和分支条件表。
- 输出第一版 `static_candidates.csv`。

### M2：单卡/最小并行基线

- 完成 Prefill 和 Decode 的 Torch Profiler、Nsight 采集。
- 建立逻辑算子到设备 Kernel 的初版映射。
- 验证 0/4/128 压缩层路径。

### M3：生产并行与 MoE 通信

- 使用生产 TP/DP/EP、MoE Runner 和 A2A 后端采集。
- 补齐通信、Dispatch/Combine、跨 Rank 数据移动。
- 输出多 Rank 聚合视角。

### M4：MTP、长上下文和差异配置

- 补齐 Draft/Verify、长 Prefill、大 Batch 和其他精度/硬件差异。
- 收敛静态未命中与动态未映射项。

### M5：最终交付

- 发布 `dsv4_operator_catalog.csv`。
- 发布按模块、阶段、精度、硬件和耗时分类的汇总文档。
- 输出后续算子实现/适配优先级：先覆盖必需算子，再按设备耗时占比优化热点算子。

---

## 12. 后续执行检查表

- [ ] 明确要回答“单一配置实际集合”还是“全代码潜在集合”，或两者都要。
- [ ] 确认模型 ID、revision、权重精度和本地路径。
- [ ] 确认目标硬件和生产并行配置。
- [ ] 保存 SGLang commit、依赖版本、启动参数和环境变量。
- [ ] 提取实际 `config.json` 和 `compress_ratios`。
- [ ] 完成主调用链静态扫描。
- [ ] 完成硬件/精度/后端条件分支表。
- [ ] 设计并冻结场景 ID 和负载矩阵。
- [ ] 完成 Warmup，并记录冷启动与稳态边界。
- [ ] 分别采集 Prefill、Decode Torch Trace。
- [ ] 分别采集 Prefill、Decode Nsight Trace。
- [ ] 生产启用 MTP 时采集 Draft/Verify。
- [ ] 生产使用多卡时采集并标注各 Rank。
- [ ] 提取并归一化 Framework Op、Device Kernel 和通信操作。
- [ ] 建立 L1～L5 多层映射。
- [ ] 处理静态未命中和动态未映射项。
- [ ] 完成覆盖率与重复性验证。
- [ ] 输出最终算子清单、证据索引和热点优先级。

---

## 13. 最终决策原则

- 如果目标是**芯片能否运行 DeepSeek V4**，优先交付按目标配置验证过的“必需设备算子与通信算子集合”。
- 如果目标是**适配 SGLang 的全部 DeepSeek V4 能力**，必须在目标硬件范围内覆盖不同精度、Attention/MoE 后端、MTP 和并行模式的潜在集合。
- 如果目标是**性能优化**，在完整性清单之外，按 Prefill/Decode 分别输出总耗时、调用次数和关键 Shape，优先处理累计设备耗时前 80%～95% 的 Kernel。
- 如果目标是**算子正确性测试**，从最终清单中进一步提取输入/输出 Shape、DType、容差、随机数据分布、参考实现和融合前后等价关系，形成独立的算子测试计划。

---

## 14. 各阶段对真实 GPU 环境的依赖

各阶段应区分三种环境要求：

1. **完全不需要 GPU**：可以在普通开发机上完成。
2. **分析过程不需要 GPU，但依赖已有 Trace**：Trace 可由其他 GPU 机器采集后交付分析。
3. **必须使用真实 GPU**：需要执行目标模型或采集设备 Kernel，无法仅靠源码可靠替代。

### 14.1 阶段依赖总表

| 阶段 | 是否需要真实 GPU | 说明 |
| --- | --- | --- |
| 阶段 A：配置和路径识别 | 部分需要 | 读取模型配置、整理参数和分析 `compress_ratios` 不需要 GPU；确认实际生效 Backend、CUDA Graph、量化 Kernel 路径及生产启动日志通常需要目标 GPU |
| 阶段 B：静态源码扫描 | 不需要 | 可以完成候选算子全集、主调用链、条件分支和硬件/精度适用范围整理 |
| 阶段 C：离线分阶段 Profiling | 需要 | 必须执行模型才能获得真实 Prefill/Decode 框架算子、设备 Kernel、Shape、调用次数和耗时 |
| 阶段 D：在线真实配置 Profiling | 需要 | 必须在真实服务环境验证 Continuous Batching、CUDA Graph、DP/EP、MTP 等运行路径 |
| 阶段 E：Nsight Systems 设备 Kernel 采集 | 需要 | NVIDIA 路径必须在真实 GPU 上采集 CUDA Kernel、NCCL、Stream 和 CUDA Graph；其他硬件需使用对应设备和 Profiler |
| 阶段 F：结果归一化与映射 | 不需要 GPU，但依赖 Trace | 获得 Torch/Nsight Trace 后，可以在无 GPU 环境中完成解析、去重、聚合和 L1～L5 映射 |
| 阶段 G：交叉验证与收敛 | 部分需要 | 静态与动态结果的对照分析不需要 GPU；发现覆盖缺口并补采场景时需要 GPU |

### 14.2 完全不需要 GPU 的工作

#### 14.2.1 范围和统计口径设计

可以直接完成：

- 确定统计“指定部署实际集合”“当前代码潜在集合”或两者都统计。
- 确定 L1～L5 算子分层。
- 确定清单字段、证据要求、验收标准和版本管理方法。
- 设计场景 ID、目录结构、结果模板和交付格式。

#### 14.2.2 模型配置静态分析

只要能够读取目标模型的 `config.json`，就可以分析：

- 模型架构、层数和 Hidden Size。
- Attention Head、KV Head、Head Dim。
- `compress_ratios` 和 SWA Window。
- Indexer TopK 和相关维度。
- MoE Expert 数量、每 Token 激活 Expert 数量和 Shared Expert 配置。
- 量化配置及 Flash/Pro 等模型变体差异。

这一步不需要加载完整权重。模型配置尚未缓存在本地时可能需要网络访问，但仍不需要 GPU。

#### 14.2.3 阶段 B 的静态源码分析

无 GPU 环境可以完整执行静态扫描并产出：

- DeepSeek V4 主调用链。
- Attention、Indexer、Compressor、MHC、MoE 的候选算子。
- CUDA、ROCm、NPU 的条件分支。
- FP8、FP4、BF16 的潜在执行路径。
- Prefill、Decode、MTP 的代码路径。
- FlashMLA、DeepGEMM、Triton、TileLang、AITER、CUTLASS 等潜在实现。
- 环境变量、启动参数和 Shape 对算子选择的影响。
- `static_candidates.csv`、`branch_conditions.md` 和 `call_graph.md`。

静态结果只能标记为 `candidate`，不能据此标记为 `observed` 或“实际执行”。

#### 14.2.4 覆盖矩阵和采集脚本设计

不需要 GPU 即可提前完成：

- Prefill、Decode、Draft、Verify 场景矩阵。
- Batch Size、上下文长度和输出长度组合。
- FP8、FP4、TP、DP、EP、CP 场景设计。
- Torch Profiler、Nsight Systems 和在线 API 采集命令模板。
- Warmup、稳定采集窗口、Trace 命名和 Manifest 规范。
- Trace 解析、Kernel 归一化和结果比对脚本的开发。

### 14.3 不需要本机 GPU、但依赖已有 Trace 的工作

如果 GPU 执行机器已经提供以下数据，阶段 F 和阶段 G 的大部分分析可以在普通开发机上完成：

- Torch Profiler `.trace.json.gz`。
- Nsight Systems `.nsys-rep` 或其导出 CSV。
- `manifest.json`、`model_config.json`、`server_args.json`。
- 服务启动日志、Profiler 日志和固定请求描述。
- 各 Rank、各场景的硬件和并行配置信息。

可离线完成的分析包括：

- 提取 Framework Operator、设备 Kernel 和通信操作。
- Kernel 名称归一化与去重。
- 调用次数、总耗时、平均耗时和占比聚合。
- Prefill、Decode、Draft、Verify 差异比较。
- 单 Rank 与全系统视角的多 Rank 汇总。
- 建立 L1～L5 多层映射。
- 生成静态未命中项和动态未映射项清单。
- 计算场景覆盖率并生成最终 CSV、Markdown 和图表。

解析 `.nsys-rep` 的机器通常需要安装与采集文件兼容的 Nsight Systems 版本，但不要求安装 GPU。若无法安装兼容版本，应要求 GPU 执行机器同时导出原始 CUDA GPU Trace CSV、Kernel Summary 和 CUDA API Summary。

### 14.4 必须使用真实 GPU 的工作

以下结论不能由源码分析或 CPU/Dummy Weight 运行可靠替代：

- 指定配置实际执行的 Framework Operator 和设备 Kernel 集合。
- Prefill、Decode、Draft、Verify 的真实差异。
- Kernel 的输入 Shape、调用次数、耗时及占比。
- FP8、FP4、FlashMLA、DeepGEMM、CUTLASS 等实际派发结果。
- CUDA Graph Capture/Replay 下的 Kernel 集合和执行顺序。
- Triton、TileLang、DeepGEMM 等 JIT Kernel 的最终生成形态。
- 不同 GPU 架构触发的专用 Kernel，例如 SM90、SM100、SM120 分支。
- TP/DP/EP/CP、NCCL、DeepEP、MegaMoE 的通信 Kernel 和重叠效果。
- 多流执行、Continuous Batching、Chunked Prefill 和 Disaggregation 路径。
- 性能数据及其稳定性。

这些工作至少需要与目标软件栈兼容的 GPU。涉及架构专用实现时，必须使用目标 GPU 架构，不能用其他 GPU 的 Trace 推断；涉及生产多卡通信时，必须使用与目标并行方式相匹配的多 GPU 环境。

### 14.5 按里程碑划分 GPU 依赖

| 里程碑 | GPU 依赖 | 可在无 GPU 环境完成的部分 |
| --- | --- | --- |
| M1：范围冻结与静态候选清单 | 基本不需要 | Manifest 模板、模型配置、调用链、候选清单、分支表；实际生效配置需后续在目标环境复核 |
| M2：单卡/最小并行基线 | 需要 | 可提前准备采集命令、负载和解析程序 |
| M3：生产并行与 MoE 通信 | 需要多 GPU | 可提前设计 Rank 标注、通信聚合和验收规则 |
| M4：MTP、长上下文和差异配置 | 需要 GPU | 可提前确定补充场景及其触发条件 |
| M5：最终交付 | 已有 Trace 后不需要 | 归一化、映射、覆盖率计算、汇总和文档交付 |

### 14.6 推荐的无 GPU 先行顺序

在 GPU 资源到位前，建议依次完成：

1. 冻结统计口径、目标模型和目标部署范围。
2. 读取实际模型配置，建立模型特征表。
3. 建立 DeepSeek V4 主调用链和源码索引。
4. 输出候选算子全集及其分支条件。
5. 设计 GPU 场景矩阵、Manifest 和采集命令。
6. 开发 Trace 解析、Kernel 归一化、多 Rank 聚合和覆盖率检查工具。
7. 准备 GPU 执行人员可直接运行的操作手册。
8. 定义 GPU 数据回传包的完整性检查，确保 Trace 到位后可以直接进入阶段 F。

### 14.7 GPU 环境交接条件

无 GPU 阶段完成后，提交 GPU 机器执行的任务包至少应包含：

- 明确的 SGLang commit 和模型 revision。
- 每个场景的唯一 ID、目的和预期触发路径。
- 可复制执行的启动、Warmup、负载和 Profiling 命令。
- 所需 GPU 型号、数量、显存和软件栈。
- 每个场景的完成判据和失败日志收集方式。
- Trace、日志、配置和结果文件的目录规范。
- 回传文件清单、文件大小检查和 SHA256 校验要求。

按照以上拆分，在没有 GPU 的情况下可以先完成候选算子、条件分支、覆盖矩阵、采集工具和分析工具；但最终的“实际执行算子集合、真实 Kernel 名称、调用次数、Shape 和耗时”必须由目标 GPU 环境实测确认。
