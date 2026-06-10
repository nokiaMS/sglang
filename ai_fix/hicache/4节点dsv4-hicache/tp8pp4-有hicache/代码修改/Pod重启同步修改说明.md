# Pod 重启后同步代码修改

Pod 重启后容器内修改会丢失，需要重新同步。使用 `kubectl cp` 将本地修改后的文件拷贝到 Pod 中。

## 前提条件

在跳板机上已保存修改后的文件 `hybrid_pool_assembler.py`。

## 操作步骤

### 1. 将修改文件拷贝到跳板机

从本地上传到跳板机：

```bash
scp hybrid_pool_assembler.py guoxu@hd04-cci-k8s-master-1:/tmp/
```

或者通过 tsh：

```bash
tsh scp hybrid_pool_assembler.py --user=guoxu root@hd04-cci-k8s-master-1:/tmp/
```

### 2. 使用 kubectl cp 同步到所有 Pod

在跳板机上执行：

```bash
for i in 0 1 2 3; do
  kubectl cp /tmp/hybrid_pool_assembler.py elm-test/dsv4pro-sg-gx-$i:/sgl-workspace/sglang/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
done
```

### 3. 验证修改是否生效

```bash
for i in 0 1 2 3; do
  echo "=== Pod $i ==="
  kubectl exec -n elm-test dsv4pro-sg-gx-$i -- grep 'start_layer)' /sgl-workspace/sglang/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py | head -2
done
```

预期输出：

```
=== Pod 0 ===
            c4_state_global_layers.append(layer_id + kvcache.start_layer)
            c128_state_global_layers.append(layer_id + kvcache.start_layer)
=== Pod 1 ===
            c4_state_global_layers.append(layer_id + kvcache.start_layer)
            c128_state_global_layers.append(layer_id + kvcache.start_layer)
=== Pod 2 ===
            c4_state_global_layers.append(layer_id + kvcache.start_layer)
            c128_state_global_layers.append(layer_id + kvcache.start_layer)
=== Pod 3 ===
            c4_state_global_layers.append(layer_id + kvcache.start_layer)
            c128_state_global_layers.append(layer_id + kvcache.start_layer)
```

### 4. 启动 sglang 服务

修改同步完成后再执行启动命令。
