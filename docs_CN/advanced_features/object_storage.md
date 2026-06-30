<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# Loading 模型 from 对象存储

SGLang 支持 direct loading of models from object storage (S3 and Google Cloud Storage) without requiring a full local download. This feature uses the `runai_streamer` load format to stream model weights directly from cloud storage, significantly reducing startup time and local storage requirements.

## 概览

When loading models from object storage, SGLang uses a two-phase approach:

1. **Metadata Download** (once, before process launch): 配置 files and tokenizer files are downloaded to a local cache
2. **Weight Streaming** (lazy, during model loading): 模型 weights are streamed directly from object storage as needed

## 已支持 Storage Backends

1. **Amazon S3**: `s3://bucket-name/path/to/model/`
2. **Google Cloud Storage**: `gs://bucket-name/path/to/model/`
3. **Azure Blob**: `az://some-azure-container/path/`
4. **S3 compatible**: `s3://bucket-name/path/to/model/`

## 快速开始

### 基础用法

Simply provide an object storage URI as the model path:

```bash
# S3
python -m sglang.launch_server \
  --model-path s3://my-bucket/models/llama-3-8b/ \
  --load-format runai_streamer

# Google Cloud Storage
python -m sglang.launch_server \
  --model-path gs://my-bucket/models/llama-3-8b/ \
  --load-format runai_streamer
```

**注意**: The `--load-format runai_streamer` is automatically detected when using object storage URIs, so you can omit it:

```bash
python -m sglang.launch_server \
  --model-path s3://my-bucket/models/llama-3-8b/
```

### With Tensor Parallelism

```bash
python -m sglang.launch_server \
  --model-path gs://my-bucket/models/llama-70b/ \
  --tp 4 \
  --model-loader-extra-config '{"distributed": true}'
```

## 配置

### Load Format

The `runai_streamer` load format is specifically designed for object storage, ssd and shared file systems

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --load-format runai_streamer
```

### Extended 配置 参数

Use `--model-loader-extra-config` to pass additional configuration as a JSON string:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --model-loader-extra-config '{
    "distributed": true,
    "concurrency": 8,
    "memory_limit": 2147483648
  }'
```

#### Available 参数

| Parameter | Type | 说明 | 默认值 |
|-----------|------|-------------|---------|
| `distributed` | bool | Enable distributed streaming for multi-GPU setups. Automatically set to `true` for object storage paths and cuda alike devices. | Auto-detected |
| `concurrency` | int | Number of concurrent download streams. Higher values can improve throughput for large models. | 4 |
| `memory_limit` | int | 内存 limit (in bytes) for the streaming buffer. | System-dependent |


## 性能 Considerations

### Distributed Streaming

For multi-GPU setups, enable distributed streaming to parallelize weight loading between the processes:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --tp 8 \
  --model-loader-extra-config '{"distributed": true}'
```

## Limitations

- **已支持 Formats**: 目前 only supports `.safetensors` weight format (recommended format)
- **已支持 Device**: Distributed streaming is supported on cuda alike devices. Otherwise fallback to non distributed streaming

## See Also

- [Runai model streamer documentation](https://github.com/run-ai/runai-model-streamer)
