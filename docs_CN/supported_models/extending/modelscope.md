<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# Use 模型 From ModelScope

要使用 a model from [ModelScope](https://www.modelscope.cn), set the environment variable `SGLANG_USE_MODELSCOPE`.

```bash
export SGLANG_USE_MODELSCOPE=true
```

We take [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) as an example.

Launch the 服务器:
```bash
python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
```

Or start it by docker:

```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    --env "SGLANG_USE_MODELSCOPE=true" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

注意 that modelscope uses a different cache directory than huggingface. You may need to set it manually to avoid running out of disk space.
