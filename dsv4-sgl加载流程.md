# DeepSeek V4 Pro SGLang 加载流程分析

基于源代码完整分析 DeepSeek V4 Pro 模型在 SGLang 中的加载流程，从命令行入口到模型完全就绪。

## 1. 总体流程概览

```
python -m sglang.launch_server
  │
  ▼
launch_server.py → prepare_server_args() → ServerArgs
  │                                         ├─ apply_deepseek_v4_defaults()  [DSV4专属默认值]
  │                                         └─ validate_deepseek_v4_cp()    [CP配置校验]
  ▼
run_server() → http_server.py → Engine()
  │
  ▼
Engine._launch_scheduler_processes()  (每个TP rank一个子进程)
  │
  ▼
run_scheduler_process() → Scheduler.__init__()
  │
  ▼
TpModelWorker.__init__()
  ├─ _init_model_config() → ModelConfig.from_server_args()
  │     ├─ 加载HF config.json → DeepSeekV4Config
  │     ├─ is_deepseek_v4() 检测
  │     ├─ try_detect_fp4_experts() 自动检测FP4/FP8权重
  │     └─ 强制 topk_group == n_group
  │
  └─ _init_model_runner() → ModelRunner()
        │
        ▼
      ModelRunner.initialize()
        ├─ init_distributed_environment()
        ├─ initialize_model_parallel()     [TP/EP/PP/DP/CP进程组]
        ├─ load_model()                   [模型加载核心]
        │     ├─ get_model_loader()        → DefaultModelLoader
        │     ├─ _get_quantization_config() → Fp8Config/W4AFp8Config
        │     ├─ _initialize_model()       → DeepseekV4ForCausalLM(config, quant_config)
        │     ├─ model.load_weights()      [权重加载+重映射+融合]
        │     └─ quant_method.process_weights_after_loading()
        ├─ configure_kv_cache_dtype()      [FP8 KV cache]
        └─ init_memory_pool()              [KV Cache池初始化]
              ├─ DSV4PoolConfigurator       [C4/C128池大小计算]
              ├─ DeepSeekV4TokenToKVPool    [多层KV池]
              ├─ CompressStatePool          [压缩状态池]
              └─ create_dsv4_backend()      → DeepseekV4AttnBackend
```

## 2. 第一阶段：命令行解析与DSV4默认值

### 2.1 入口点

**文件**: `python/sglang/launch_server.py`

```python
# 旧入口: python -m sglang.launch_server
# 新入口: sglang serve (python/sglang/cli/serve.py)
```

两者最终都调用 `prepare_server_args()` 和 `run_server()`。

### 2.2 ServerArgs 解析

**文件**: `python/sglang/srt/server_args.py`

`prepare_server_args()` (line 7799) 解析CLI参数，创建 `ServerArgs` 对象。关键步骤：

1. **下载HF config.json** → 获取 `model_arch = hf_config.architectures[0]`
2. **DSV4 默认值注入** (line 1798-1805):
   ```python
   if model_arch in ["DeepseekV4ForCausalLM"]:
       from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
       apply_deepseek_v4_defaults(self, model_arch)
   ```
3. **DSV4 CP 校验** (line 2062-2067):
   ```python
   if model_arch in ["DeepseekV4ForCausalLM"]:
       from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v4_cp
       validate_deepseek_v4_cp(self)
   ```
4. **SM120 MoE后端选择** (line 2069-2081): 在SM120上自动选择 `marlin` MoE后端

### 2.3 DSV4 专属默认值

**文件**: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`

`apply_deepseek_v4_defaults()` 强制设置：

| 参数 | 强制值 | 说明 |
|---|---|---|
| `attention_backend` | `"dsv4"` | DSV4专用注意力后端 |
| `page_size` | `256` | DSV4要求的页大小 |
| `max_running_requests` | `256` | 默认最大并发请求数 |
| `kv_cache_dtype` | `"fp8_e4m3"` | 仅支持FP8 KV cache |
| `swa_full_tokens_ratio` | `0.1` | SWA全注意力token比例 |

`validate_deepseek_v4_cp()` 校验：
- 要求 `dsa_prefill_cp_mode == "round-robin-split"`
- 启用 `enable_dp_attention = True`
- 设置 `moe_dense_tp_size = 1`
- 限制 `dp_size == 1`, `tp_size <= 8`

## 3. 第二阶段：进程启动与并行初始化

### 3.1 Engine 启动子进程

**文件**: `python/sglang/srt/entrypoints/engine.py`

`Engine` 类根据 `tp_size` 和 `dp_size` 计算子进程数量，调用 `_launch_scheduler_processes()` 为每个 TP rank 启动一个子进程。

### 3.2 Scheduler 进程

**文件**: `python/sglang/srt/managers/scheduler.py`

每个子进程执行 `run_scheduler_process()` (line 3926)，创建 `Scheduler` 实例，其中包含 `TpModelWorker`。

### 3.3 TpModelWorker 初始化

**文件**: `python/sglang/srt/managers/tp_worker.py`

```python
class TpModelWorker(BaseTpWorker):
    def __init__(self, server_args, gpu_id, tp_rank, ...):
        self._init_model_config()    # 步骤3.4
        self._init_model_runner()    # 步骤3.5
```

### 3.4 ModelConfig 创建

**文件**: `python/sglang/srt/configs/model_config.py`

`ModelConfig.from_server_args()` 执行：

1. **加载HF config.json** → 解析为 `DeepSeekV4Config` 数据类
2. **DSV4检测** (line 117-121):
   ```python
   def is_deepseek_v4(config) -> bool:
       return _hf_arch(config) in ("DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMNextN")
   ```
3. **FP4/FP8自动检测** (line 252-263):
   ```python
   if is_deepseek_v4(self.hf_config):
       self.is_fp4_experts = envs.SGLANG_DSV4_FP4_EXPERTS.get()  # 默认True
       if not envs.SGLANG_DSV4_FP4_EXPERTS.is_set():
           detected = try_detect_fp4_experts(self.model_path)
           # 探测safetensors中专家权重的dtype: U8/I8/F4 → mxfp4, F8_E4M3 → fp8
   ```
4. **强制topk_group** (line 272-274): DSV4使用全专家top-k，强制 `topk_group == n_group`

### 3.5 DeepSeekV4Config 数据类

**文件**: `python/sglang/srt/configs/deepseek_v4.py`

```python
@dataclass(kw_only=True)
class DeepSeekV4Config(PretrainedConfig):
    # 核心维度
    hidden_size: int = 4096           # V4 Pro: 7168
    num_hidden_layers: int = 43       # V4 Pro: 61
    num_attention_heads: int = 64     # V4 Pro: 128
    num_key_value_heads: int = 1      # MQA
    kv_lora_rank: int = 512
    q_lora_rank: int = 1024           # V4 Pro: 1536
    qk_nope_head_dim: int = 448       # V4 Pro: 448
    qk_rope_head_dim: int = 64
    v_head_dim: int = 512

    # MoE配置
    n_routed_experts: int = 256       # V4 Pro: 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048 # V4 Pro: 3072
    routed_scaling_factor: float = 1.5  # V4 Pro: 2.5

    # 注意力机制
    o_lora_rank: int = 1024           # V4 Pro: 1024
    o_groups: int = 8                 # V4 Pro: 16
    window_size: int = 128            # 滑动窗口
    topk_method: str = "noaux_tc"

    # 压缩注意力
    compress_ratios: List[int] = []   # 各层压缩率 [0,4,128,...]
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512             # V4 Pro: 1024

    # MHC (Multi-Head Conditioning)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
```

### 3.6 FP4专家权重自动检测

**文件**: `python/sglang/srt/configs/deepseek_v4.py` + `python/sglang/srt/model_loader/weight_utils.py`

`try_detect_fp4_experts()` 流程：
1. 定位本地模型缓存目录
2. 调用 `probe_routed_expert_weight_dtype(local_path)` 读取safetensors header
3. 检查专家权重的dtype：
   - `"U8"`, `"I8"`, `"F4"` → MXFP4打包格式 (返回 `True`)
   - `"F8_E4M3"` → 转换后的FP8格式 (返回 `False`)

检测结果 `is_fp4_experts` 传递给 `Fp8Config`，影响MoE量化和内核选择。

## 4. 第三阶段：模型加载

### 4.1 ModelRunner.initialize()

**文件**: `python/sglang/srt/model_executor/model_runner.py`

`initialize()` (line 613) 是模型加载的主入口，执行顺序：

```python
def initialize(self, pre_model_load_memory):
    # 1. 分布式环境初始化
    init_distributed_environment(...)
    initialize_model_parallel(...)

    # 2. 加载模型
    self.load_model()

    # 3. 应用torchao量化
    apply_torchao_config_to_model(self.model, ...)

    # 4. 配置KV cache dtype
    self.configure_kv_cache_dtype()

    # 5. 初始化内存池
    self.init_memory_pool(pre_model_load_memory)

    # 6. 初始化注意力后端
    self.init_attention_backend()

    # 7. CUDA graph warmup
    self.init_device_graphs()
```

### 4.2 分布式环境初始化

**文件**: `python/sglang/srt/distributed/parallel_state.py`

`initialize_model_parallel()` 创建以下进程组：
- **TP组**: 张量并行 (`tp_size`)
- **EP组**: 专家并行 (`ep_size`)
- **PP组**: 流水线并行 (`pp_size`)
- **DP组**: 数据并行 (`dp_size`, DP-attention时启用)
- **CP组**: 上下文并行 (`attn_cp_size`, DSA prefill CP时启用)
- **Attention TP组**: 注意力专用TP子组
- **MoE DP组**: MoE数据并行子组

关键关系：
```
moe_tp_size = tp_size / ep_size / moe_dp_size
attn_tp_size = tp_size / dp_size / attn_cp_size
n_local_groups = o_groups / attn_tp_size
intermediate_size_per_partition = moe_intermediate_size / moe_tp_size
```

### 4.3 ModelRunner.load_model()

**文件**: `python/sglang/srt/model_executor/model_runner.py` (line 1255)

```python
def load_model(self):
    # 1. 创建LoadConfig
    self.load_config = LoadConfig(load_format=..., download_dir=..., ...)

    # 2. 获取模型加载器
    self.loader = get_model_loader(load_config=self.load_config, model_config=self.model_config)

    # 3. 加载模型
    self.model = self.loader.load_model(
        model_config=self.model_config,
        device_config=DeviceConfig(self.device, self.gpu_id),
    )

    # 4. 加载KV cache缩放因子
    if self.server_args.kv_cache_dtype == "fp8_e4m3":
        if self.server_args.quantization_param_path is not None:
            self.model.load_kv_cache_scales(...)
```

### 4.4 DefaultModelLoader.load_model()

**文件**: `python/sglang/srt/model_loader/loader.py` (line 702)

```python
def load_model(self, *, model_config, device_config):
    target_device = torch.device(device_config.device)

    # 1. 获取量化配置
    quant_config = _get_quantization_config(model_config, self.load_config)
    # DSV4: Fp8Config (is_fp4_experts=True → 内部路由到W4AFp8Config/Mxfp4Config)

    # 2. 在目标设备上实例化模型
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = _initialize_model(model_config, self.load_config, quant_config)

    # 3. 加载权重
    self.load_weights_and_postprocess(
        model, self._get_all_weights(model_config, model), target_device
    )

    return model.eval()
```

### 4.5 量化配置解析

**文件**: `python/sglang/srt/model_loader/loader.py` (line 194) + `python/sglang/srt/layers/quantization/`

`_get_quantization_config()` 流程：
1. 调用 `get_quant_config()` 读取 `quantize_config.json` 或从HF config推断
2. DSV4 checkpoint 的 `quantize_config.json` 声明 `quant_method: "fp8"`
3. 创建 `Fp8Config` 实例，设置 `is_fp4_experts` 标志
4. `Fp8Config.get_quant_method()` 根据权重类型路由：
   - MoE专家权重 (MXFP4) → `W4AFp8Method` / `Mxfp4MarlinMoEMethod` / `FlashinferMxfp4MoEMethod`
   - 其他权重 (FP8) → `Fp8LinearMethod` / `Fp8MoEMethod`

### 4.6 模型实例化

**文件**: `python/sglang/srt/model_loader/loader.py` (line 273) + `python/sglang/srt/models/deepseek_v4.py`

`_initialize_model()` → `DeepseekV4ForCausalLM(config, quant_config)`

```
DeepseekV4ForCausalLM
  ├─ config: DeepSeekV4Config
  ├─ model: DeepseekV4Model
  │     ├─ embed_tokens: VocabParallelEmbedding
  │     ├─ layers[0..60]: DeepseekV4DecoderLayer × 61
  │     │     ├─ self_attn: MQALayer
  │     │     │     ├─ wqkv_a: ColumnParallelLinear (MLA q+kv融合)
  │     │     │     ├─ wq_b: ColumnParallelLinear
  │     │     │     ├─ wkv_b: ColumnParallelLinear
  │     │     │     ├─ wo_a: RowParallelLinear (O LoRA降维)
  │     │     │     ├─ wo_b: RowParallelLinear (O LoRA升维)
  │     │     │     ├─ Compressor (C4/C128压缩)
  │     │     │     └─ C4Indexer (C4索引)
  │     │     ├─ mlp: DeepseekV2MoE
  │     │     │     ├─ gate: MoEGate (路由器)
  │     │     │     ├─ experts[0..383]: FusedMoE专家
  │     │     │     └─ shared_expert: 共享专家
  │     │     ├─ input_layernorm: RMSNorm
  │     │     ├─ post_attention_layernorm: RMSNorm
  │     │     └─ hc_attn_fn, hc_ffn_fn, ... (MHC参数)
  │     ├─ norm: RMSNorm
  │     └─ hc_head_fn, hc_head_scale, hc_head_base (MHC头部)
  ├─ lm_head: ParallelLMHead
  └─ logits_processor: LogitsProcessor
```

**关键实例化细节**:

1. **MQALayer** (line 954): DSV4的注意力层，使用MQA (1个KV头)，包含：
   - `wqkv_a`: 融合的q_a + kv_a投影 (可选 `SGLANG_OPT_FUSE_WQA_WKV`)
   - `Compressor`: 根据 `compress_ratios[layer_id]` 创建C4(4×)或C128(128×)压缩器
   - `C4Indexer`: 仅在 `compress_ratio == 4` 的层创建

2. **DeepseekV2MoE** (line 970): 复用DSV2的MoE实现，`is_deepseek_v4=True` 标志启用DSV4特定行为

3. **MHC参数** (line 985-996): 每层6个 `nn.Parameter`:
   - `hc_attn_fn`, `hc_ffn_fn`: (mix_hc, hc_dim) 混合函数
   - `hc_attn_base`, `hc_ffn_base`: (mix_hc,) 偏置
   - `hc_attn_scale`, `hc_ffn_scale`: (3,) 缩放

4. **共享专家融合** (line 1695-1721): DSV4默认**禁用**共享专家融合 (`disable_shared_experts_fusion=True`)，除非启用了 `enable_deepep_waterfill`

5. **attn_tp_context 初始化** (line 1670): `get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)`

## 5. 第四阶段：权重加载

### 5.1 权重加载主流程

**文件**: `python/sglang/srt/models/deepseek_v4.py` (line 1877)

`DeepseekV4ForCausalLM.load_weights()` 是权重加载的核心方法。

### 5.2 权重名称重映射

`remap_weight_name_to_dpsk_hf_format()` (line 1817) 将HF格式名称映射到SGLang内部名称：

| HF checkpoint 名称 | SGLang 内部名称 |
|---|---|
| `model.embed_tokens.weight` | `embed.weight` |
| `lm_head.weight` | `head.weight` |
| `.attn.` | `.self_attn.` |
| `.ffn.` | `.mlp.` |
| `.attn_norm.` | `.input_layernorm.` |
| `.ffn_norm.` | `.post_attention_layernorm.` |
| `.w1.` | `.gate_proj.` |
| `.w2.` | `.down_proj.` |
| `.w3.` | `.up_proj.` |
| `.self_attn.scale` | `.self_attn.weight_scale_inv` |
| `.gate.bias` | `.gate.e_score_correction_bias` |

### 5.3 权重融合优化

1. **gate_up_proj 融合** (line 1902-1905): `gate_proj` + `up_proj` → `gate_up_proj`
2. **Compressor wkv+wgate 融合** (line 2084-2115): `.compressor.wkv.weight` + `.compressor.wgate.weight` → `.compressor.wkv_gate.weight`
3. **wqkv_a 融合** (line 2116-2146, 需 `SGLANG_OPT_FUSE_WQA_WKV=1`): `wq_a` + `wkv` → `wqkv_a`
4. **FP8 wo_a 反量化** (line 1893-1900, 需 `SGLANG_OPT_FP8_WO_A_GEMM=0`): 将FP8 `wo_a` 反量化为BF16

### 5.4 专家权重加载

MoE专家权重通过 `FusedMoE.make_expert_params_mapping()` 生成的映射表加载：

```python
expert_params_mapping = FusedMoE.make_expert_params_mapping(
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=config.n_routed_experts + num_fused_shared_experts,
)
```

每个专家权重通过 `weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)` 加载，支持TP/EP分片。

**MXFP4专家额外映射** (line 1914-1917):
```python
if quant_config.get_name() == "w4afp8":
    expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
        num_experts=config.n_routed_experts
    )
```

### 5.5 权重加载后处理

`post_load_weights()` (line 1798):
1. **FP8 wo_a scale 布局转换** (需 `SGLANG_OPT_FP8_WO_A_GEMM=1`): 调用 `deep_gemm.transform_sf_into_required_layout()` 重新排列scale因子
2. **Compressor APE热修复** (line 1807-1813): 对C4/C128压缩器应用APE (Approximate Positional Encoding) 修正
3. **MHC norm权重缓存** (line 1814): 缓存BF16 norm权重供融合MHC内核使用

### 5.6 量化后处理

**文件**: `python/sglang/srt/model_loader/loader.py` (line 733)

`load_weights_and_postprocess()` 在 `model.load_weights()` 之后遍历所有模块：

```python
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None:
        quant_method.process_weights_after_loading(module)
```

对于DSV4，`process_weights_after_loading()` 执行：
- **MXFP4专家**: 权重重打包为Marlin/flashinfer格式
- **FP8线性层**: 权重scale校准

## 6. 第五阶段：KV Cache 与内存池初始化

### 6.1 KV Cache dtype 配置

**文件**: `python/sglang/srt/model_executor/model_runner.py`

`configure_kv_cache_dtype()` (line 756):
- DSV4强制 `kv_cache_dtype = "fp8_e4m3"` (在 `apply_deepseek_v4_defaults` 中已设置)
- 可选加载KV cache缩放因子 (`quantization_param_path`)

### 6.2 内存池初始化

**文件**: `python/sglang/srt/model_executor/model_runner.py` + `model_runner_kv_cache_mixin.py`

`init_memory_pool()` 流程：

1. **池大小计算**: 调用 `_resolve_memory_pool_config()`
2. **DSV4池配置器**: 当 `is_deepseek_v4() and is_hybrid_swa` 时使用 `DSV4PoolConfigurator`

### 6.3 DSV4PoolConfigurator

**文件**: `python/sglang/srt/model_executor/pool_configurator.py` (line 309)

DSV4需要多个KV Cache池：

| 池类型 | 用途 | 大小计算 |
|---|---|---|
| `full` | 全注意力 (最近128 tokens的滑动窗口) | 基于SWA tokens |
| `swa` | 滑动窗口注意力 | 基于滑动窗口大小 |
| `c4` | C4压缩注意力 (4×压缩) | 基于compress_ratio==4的层数 |
| `c128` | C128压缩注意力 (128×压缩) | 基于compress_ratio==128的层数 |
| `c4_state` | C4压缩状态 (max/sum/score) | 配合C4池 |
| `c128_state` | C128压缩状态 | 配合C128池 |

`_DSV4PoolSizes` 数据类计算各池大小，考虑因素：
- 可用GPU显存 (80GB - 权重 - 框架开销)
- `swa_full_tokens_ratio` (默认0.1): SWA池中全注意力token占比
- 各层的 `compress_ratio` 决定池类型
- page_size = 256

### 6.4 DeepSeekV4TokenToKVPool

**文件**: `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`

DSV4的KV池管理器，包含：

```
DeepSeekV4TokenToKVPool
  ├─ full_pool: DeepSeekV4SingleKVPool    [全注意力KV, fp8_e4m3]
  │     每层存储: kv_lora_rank(512) + qk_rope_head_dim(64) = 576 fp8元素/token
  ├─ swa_pool: DeepSeekV4SingleKVPool     [滑动窗口KV]
  │     同上格式
  ├─ c4_pool: DeepSeekV4SingleKVPool      [C4压缩KV]
  │     存储压缩后的KV
  └─ c128_pool: DeepSeekV4SingleKVPool    [C128压缩KV]
        存储高度压缩的KV
```

`CompressStatePool` 管理C4/C128的压缩状态 (max/sum/KV scores)。

### 6.5 注意力后端创建

**文件**: `python/sglang/srt/layers/attention/attention_registry.py`

`create_dsv4_backend()` (line 127):
- CUDA: 返回 `DeepseekV4AttnBackend`
- HIP (ROCm): 返回 `DeepseekV4HipRadixBackend`

**文件**: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`

`DeepseekV4AttnBackend` 是DSV4的核心注意力后端，管理：
- **全注意力**: 滑动窗口内128 tokens的标准注意力
- **SWA**: 滑动窗口注意力
- **C4压缩**: 4×压缩注意力 (通过 `Compressor` 和 `C4Indexer`)
- **C128压缩**: 128×压缩注意力 (通过 `Compressor`)

### 6.6 HiSparse 后端 (可选)

**文件**: `python/sglang/srt/arg_groups/hisparse_hook.py`

当启用 `--enable-hisparse` 时，DSV4使用HiSparse后端，在host和device之间offload KV cache：
- 选择 `flashmla_kv` 或 `flashmla_sparse` 后端 (基于KV cache dtype)
- 使用 `DeepSeekV4HiSparseTokenToKVPoolAllocator`

## 7. 第六阶段：CUDA Graph 与预热

### 7.1 注意力后端初始化

**文件**: `python/sglang/srt/model_executor/model_runner.py` (line 806)

```python
self.init_attention_backend()
```

创建 `DeepseekV4AttnBackend` 实例，初始化：
- Compressor 的频率缓存 (`freqs_cis_c4`, `freqs_cis_c128`)
- C4Indexer 的元数据
- FlashMLA 相关状态

### 7.2 内核预热

```python
self.kernel_warmup()
```

JIT编译和预热所有DSV4专用内核：
- `fused_norm_rope`, `fused_rope_inplace` (RoPE融合)
- C4/C128 压缩内核
- MHC pre/post 内核
- TopK 路由内核
- MoE 内核

### 7.3 CUDA Graph 捕获

```python
self.init_device_graphs()
```

为不同batch size捕获CUDA graph，DSV4特殊处理：
- MHC prewarm: 对不同token数量预热MHC pre/post内核
- 分层CUDA graph: 支持按层捕获 (`disable_piecewise_cuda_graph`)

## 8. 模型注册表机制

**文件**: `python/sglang/srt/models/registry.py`

SGLang使用自动注册机制：

```python
# python/sglang/srt/models/deepseek_v4.py (line 2228)
EntryClass = [DeepseekV4ForCausalLM]

# python/sglang/srt/models/deepseek_v4_nextn.py (line 283)
EntryClass = [DeepseekV4ForCausalLMNextN]
```

`ModelRegistry` 在初始化时扫描 `sglang.srt.models` 包下所有模块，找到 `EntryClass` 属性，以类名为key注册。

`get_model_architecture()` 调用 `ModelRegistry.resolve_model_cls(["DeepseekV4ForCausalLM"])` 返回对应的Python类。

## 9. 关键环境变量

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `SGLANG_DSV4_FP4_EXPERTS` | `True` | DSV4专家是否为MXFP4格式 |
| `SGLANG_OPT_FP8_WO_A_GEMM` | `True` (SM90+), `False` (SM120) | 是否使用FP8 wo_a GEMM (DeepGEMM) |
| `SGLANG_OPT_FUSE_WQA_WKV` | `False` | 是否融合wq_a+wkv投影 |
| `SGLANG_OPT_USE_TOPK_V2` | `True` (SM90), `False` (SM120) | 是否使用topk_v2内核 |
| `SGLANG_OPT_USE_TILELANG_MHC_PRE` | `True` | 是否使用TileLang MHC pre内核 |
| `SGLANG_SHARED_EXPERT_TP1` | `0` | 强制共享专家TP=1 (避免FP8 block_n约束) |
| `SGLANG_ENABLE_SPEC_V2` | `False` | EAGLE投机解码v2 |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` | `False` (SM90), `True` (SM120) | 使用torch实现FP8 paged MQA |

## 10. 完整调用链 (关键行号)

| 步骤 | 文件 | 行号 | 函数/方法 |
|---|---|---|---|
| 1. CLI入口 | `launch_server.py` | — | `main()` |
| 2. 参数解析 | `server_args.py` | 7799 | `prepare_server_args()` |
| 3. DSV4默认值 | `deepseek_v4_hook.py` | 10 | `apply_deepseek_v4_defaults()` |
| 4. DSV4 CP校验 | `deepseek_v4_hook.py` | 55 | `validate_deepseek_v4_cp()` |
| 5. Engine启动 | `engine.py` | — | `Engine._launch_scheduler_processes()` |
| 6. Scheduler | `scheduler.py` | 3926 | `run_scheduler_process()` |
| 7. TpModelWorker | `tp_worker.py` | 218 | `TpModelWorker.__init__()` |
| 8. ModelConfig | `tp_worker.py` | 326 | `_init_model_config()` |
| 9. FP4检测 | `deepseek_v4.py` (config) | 13 | `try_detect_fp4_experts()` |
| 10. ModelRunner | `tp_worker.py` | 344 | `_init_model_runner()` |
| 11. 分布式初始化 | `parallel_state.py` | — | `initialize_model_parallel()` |
| 12. 模型加载入口 | `model_runner.py` | 1255 | `load_model()` |
| 13. 加载器选择 | `loader.py` | — | `get_model_loader()` |
| 14. 量化配置 | `loader.py` | 194 | `_get_quantization_config()` |
| 15. 模型实例化 | `loader.py` | 273 | `_initialize_model()` |
| 16. DSV4模型类 | `deepseek_v4.py` | 1639 | `DeepseekV4ForCausalLM.__init__()` |
| 17. DSV4骨干 | `deepseek_v4.py` | 1449 | `DeepseekV4Model.__init__()` |
| 18. Decoder层 | `deepseek_v4.py` | 938 | `DeepseekV4DecoderLayer.__init__()` |
| 19. 权重加载 | `deepseek_v4.py` | 1877 | `load_weights()` |
| 20. 权重重映射 | `deepseek_v4.py` | 1817 | `remap_weight_name_to_dpsk_hf_format()` |
| 21. 权重后处理 | `deepseek_v4.py` | 1798 | `post_load_weights()` |
| 22. 量化后处理 | `loader.py` | 733 | `quant_method.process_weights_after_loading()` |
| 23. KV配置 | `model_runner.py` | 756 | `configure_kv_cache_dtype()` |
| 24. 内存池 | `model_runner.py` | 759 | `init_memory_pool()` |
| 25. 池大小计算 | `pool_configurator.py` | 309 | `DSV4PoolConfigurator` |
| 26. KV池创建 | `deepseek_v4_memory_pool.py` | — | `DeepSeekV4TokenToKVPool` |
| 27. 注意力后端 | `attention_registry.py` | 127 | `create_dsv4_backend()` |
| 28. CUDA Graph | `model_runner.py` | 809 | `init_device_graphs()` |
