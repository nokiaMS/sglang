# 关于 MHATokenToKVPoolFP4

## 结论

`MHATokenToKVPoolFP4` 的 FP4 量化存储及反量化过程会有精度损失。

原因是它不是把 K/V cache 按原始 `float16` 或 `bfloat16` 直接保存，而是在写入缓存时先把 K/V 量化成 FP4 压缩格式，并额外保存 scale buffer。读取时再根据 FP4 数据和 scale buffer 反量化回计算用的 dtype。

FP4 只有 4 bit 表示一个数值，能表达的数值范围和精度都明显低于 FP16/BF16。因此量化阶段会发生舍入、截断或饱和，反量化只能恢复近似值，不能完全恢复原始 K/V。

## 代码路径

在 `python/sglang/srt/mem_cache/memory_pool.py` 中，`MHATokenToKVPoolFP4` 写入 KV cache 时调用：

```python
BlockFP4KVQuantizeUtil.batched_quantize(cache_k)
BlockFP4KVQuantizeUtil.batched_quantize(cache_v)
```

量化后的数据分别保存到：

```python
self.k_buffer
self.v_buffer
self.k_scale_buffer
self.v_scale_buffer
```

读取内部 K/V buffer 时，如果物理存储 dtype 和计算 dtype 不一致，会调用：

```python
BlockFP4KVQuantizeUtil.batched_dequantize(...)
```

也就是通过保存的 FP4 数据和 scale buffer 反量化得到近似的 K/V 张量。

## 精度损失来源

1. FP4 bit 数很低，无法表示 FP16/BF16 中的大量中间值。
2. 量化时需要把连续值映射到有限的 FP4 表示，会产生舍入误差。
3. 超出可表示范围的值可能被裁剪或饱和。
4. 反量化只是根据 FP4 编码和 scale 重建近似值，不能恢复量化前的完整信息。

## 收益与代价

收益是显存占用显著下降，K/V cache 可以保存更多 token 或降低运行时内存压力。

代价是 K/V cache 中的 key/value 数值带有量化误差，attention 结果可能产生轻微偏差，最终模型输出质量也可能受到影响。影响大小取决于模型、任务、量化实现、scale 粒度以及 K/V 分布。

从当前实现看，它使用的是块级 FP4 量化，每个 block 配套 scale。相比全局 scale，这种方式能更好地保留局部动态范围，通常可以降低量化误差，但本质上仍然是有损压缩。

## 如何启用 MHATokenToKVPoolFP4

启动 SGLang server 时设置：

```bash
python -m sglang.launch_server \
  --model-path <你的模型路径或HF模型名> \
  --kv-cache-dtype fp4_e2m1
```

如果模型走普通 MHA/GQA/MQA KV cache 路径，`--kv-cache-dtype fp4_e2m1` 会先被转换为 `torch.float4_e2m1fn_x2`，随后在创建 token-to-KV pool 时实例化 `MHATokenToKVPoolFP4`。

相关代码路径：

```python
# python/sglang/srt/model_executor/model_runner.py
elif self.server_args.kv_cache_dtype == "fp4_e2m1":
    if hasattr(torch, "float4_e2m1fn_x2"):
        self.kv_cache_dtype = torch.float4_e2m1fn_x2
```

```python
# python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
if is_float4_e2m1fn_x2(self.kv_cache_dtype):
    self.token_to_kv_pool = MHATokenToKVPoolFP4(...)
```

也就是说，启用 `MHATokenToKVPoolFP4` 的关键参数是：

```bash
--kv-cache-dtype fp4_e2m1
```

## 注意事项

1. 需要当前 PyTorch 支持 `torch.float4_e2m1fn_x2`。如果不支持，代码会 fallback 到模型默认 dtype，不会真正启用 FP4 KV cache。
2. 参数说明中要求 `fp4_e2m1` 依赖 CUDA 12.8+ 和 PyTorch 2.8.0+。
3. `--kv-cache-dtype fp4_e2m1` 是 KV cache 量化参数，不是模型权重量化参数。
4. `--quantization modelopt_fp4` 用于模型权重量化，不能替代 `--kv-cache-dtype fp4_e2m1`。
5. 如果模型走 MLA 路径，会实例化 `MLATokenToKVPoolFP4`，不是 `MHATokenToKVPoolFP4`。要使用 `MHATokenToKVPoolFP4`，模型需要走普通 MHA/GQA/MQA attention KV cache 路径。
6. FP4 KV cache 当前属于实验特性，可能带来精度下降，建议结合具体模型和任务做验证。

最小示例：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp4_e2m1
```
