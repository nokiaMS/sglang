# 安装 SGLang

你可以使用下面任一方式安装 SGLang。
本页面主要适用于常见的 NVIDIA GPU 平台。
对于其他平台或较新的平台，请参考对应页面：[AMD GPU](../platforms/amd_gpu.md)、[Intel Xeon CPU](../platforms/cpu_server.md)、[TPU](../platforms/tpu.md)、[NVIDIA DGX Spark](https://lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/)、[NVIDIA Jetson](../platforms/nvidia_jetson.md)、[昇腾 NPU](../platforms/ascend/ascend_npu.md) 和 [Intel XPU](../platforms/xpu.md)。

## 方法 1：使用 pip 或 uv

推荐使用 uv，以获得更快的安装速度：

```bash
pip install --upgrade pip
pip install uv
uv pip install sglang
```

### CUDA 13

推荐使用 Docker（见方法 3 中关于 B300/GB300/CUDA 13 的说明）。如果你无法使用 Docker，请按以下步骤安装：

1. 先安装支持 CUDA 13 的 PyTorch：

```bash
# 将 X.Y.Z 替换为你的 SGLang 安装所需的版本
uv pip install torch==X.Y.Z torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

2. 安装 sglang：

```bash
uv pip install sglang
```

3. 从 [sgl-project whl releases](https://github.com/sgl-project/whl/blob/gh-pages/cu130/sglang-kernel/index.html) 安装 CUDA 13 对应的 `sglang-kernel` wheel。将 `X.Y.Z` 替换为当前 SGLang 安装所需的 `sglang-kernel` 版本（可以通过 `uv pip show sglang-kernel` 查看）。示例：

```bash
# x86_64
uv pip install "https://github.com/sgl-project/whl/releases/download/vX.Y.Z/sglang_kernel-X.Y.Z+cu130-cp310-abi3-manylinux2014_x86_64.whl"

# aarch64
uv pip install "https://github.com/sgl-project/whl/releases/download/vX.Y.Z/sglang_kernel-X.Y.Z+cu130-cp310-abi3-manylinux2014_aarch64.whl"
```

4. 如果你在 B300/GB300 上遇到 `ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'`，可以这样修复：

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

### 常见问题快速修复

- 如果遇到 `OSError: CUDA_HOME environment variable is not set`，请用以下任一方式把它设置为 CUDA 安装根目录：
  1. 使用 `export CUDA_HOME=/usr/local/cuda-<your-cuda-version>` 设置 `CUDA_HOME` 环境变量。
  2. 按照 [FlashInfer 安装文档](https://docs.flashinfer.ai/installation.html) 先安装 FlashInfer，然后再按上面的方式安装 SGLang。

## 方法 2：从源码安装

```bash
# 使用最新发布分支
git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
cd sglang

# 安装 Python 包
pip install --upgrade pip
pip install -e "python"
```

**常见问题快速修复**

- 如果你要开发 SGLang，可以使用开发版 Docker 镜像。请参考[设置 Docker 容器](../developer_guide/development_guide_using_docker.md#setup-docker-container)。Docker 镜像为 `lmsysorg/sglang:dev`。

## 方法 3：使用 Docker

Docker 镜像发布在 Docker Hub 的 [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags)，由 [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker) 构建。
将下面的 `<secret>` 替换为你的 Hugging Face Hub [token](https://huggingface.co/docs/hub/en/security-tokens)。

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

生产部署建议使用 `runtime` 变体。它移除了构建工具和开发依赖，体积明显更小（约减少 40%）：

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest-runtime \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

你也可以在[这里](https://hub.docker.com/r/lmsysorg/sglang/tags?name=nightly)找到 nightly Docker 镜像。

注意：

- 在 B300/GB300（SM103）或 CUDA 13 环境中，推荐使用 nightly 镜像 `lmsysorg/sglang:dev-cu13`，或稳定镜像 `lmsysorg/sglang:latest-cu130-runtime`。请不要在该 Docker 镜像内以 editable 方式重新安装项目，因为这会覆盖 cu13 Docker 镜像指定的库版本。

## 方法 4：使用 Kubernetes

请参考 [OME](https://github.com/sgl-project/ome)。OME 是一个 Kubernetes operator，用于企业级大语言模型（LLM）管理和服务化。

<details>
<summary>更多</summary>

1. 选项 1：单节点服务（通常适用于模型大小可以放入单节点 GPU 的情况）

   执行 `kubectl apply -f docker/k8s-sglang-service.yaml`，以 llama-31-8b 为例创建 k8s deployment 和 service。

2. 选项 2：多节点服务（通常适用于大模型需要多个 GPU 节点的情况，例如 `DeepSeek-R1`）

   按需修改 LLM 模型路径和参数，然后执行 `kubectl apply -f docker/k8s-sglang-distributed-sts.yaml`，创建双节点 k8s statefulset 和 serving service。

</details>

## 方法 5：使用 docker compose

<details>
<summary>更多</summary>

> 如果你计划将它作为服务运行，推荐使用这种方式。
> 更好的做法是使用 [k8s-sglang-service.yaml](https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-service.yaml)。

1. 将 [compose.yml](https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml) 复制到本机。
2. 在终端中执行 `docker compose up -d`。
</details>

## 方法 6：使用 SkyPilot 在 Kubernetes 或云平台上运行

<details>
<summary>更多</summary>

如果要部署到 Kubernetes 或 12 个以上云平台，可以使用 [SkyPilot](https://github.com/skypilot-org/skypilot)。

1. 安装 SkyPilot，并配置 Kubernetes 集群或云访问权限。参见 [SkyPilot 文档](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)。
2. 用单条命令部署到你的基础设施，并获取 HTTP API endpoint：

<details>
<summary>SkyPilot YAML：<code>sglang.yaml</code></summary>

```yaml
# sglang.yaml
envs:
  HF_TOKEN: null

resources:
  image_id: docker:lmsysorg/sglang:latest
  accelerators: A100
  ports: 30000

run: |
  conda deactivate
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

</details>

```bash
# 部署到任意云或 Kubernetes 集群。使用 --cloud <cloud> 选择特定云厂商。
HF_TOKEN=<secret> sky launch -c sglang --env HF_TOKEN sglang.yaml

# 获取 HTTP API endpoint
sky status --endpoint 30000 sglang
```

3. 如果要通过自动扩缩容和故障恢复进一步扩展部署，请查看 [SkyServe + SGLang guide](https://github.com/skypilot-org/skypilot/tree/master/llm/sglang#serving-llama-2-with-sglang-for-more-traffic-using-skyserve)。

</details>

## 方法 7：在 AWS SageMaker 上运行

<details>
<summary>更多</summary>

如需在 AWS SageMaker 上部署 SGLang，请参考 [AWS SageMaker Inference](https://aws.amazon.com/sagemaker/ai/deploy)。

Amazon Web Services 为 SGLang 容器提供支持，并包含常规安全补丁。可用的 SGLang 容器见 [AWS SGLang DLCs](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sglang-containers)。

如果要使用自己的容器托管模型，请按以下步骤操作：

1. 使用 [sagemaker.Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/sagemaker.Dockerfile) 以及 [serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) 脚本构建 Docker 容器。
2. 将容器推送到 AWS ECR。

<details>
<summary>Dockerfile 构建脚本：<code>build-and-push.sh</code></summary>

```bash
#!/bin/bash
AWS_ACCOUNT="<YOUR_AWS_ACCOUNT>"
AWS_REGION="<YOUR_AWS_REGION>"
REPOSITORY_NAME="<YOUR_REPOSITORY_NAME>"
IMAGE_TAG="<YOUR_IMAGE_TAG>"

ECR_REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Starting build and push process..."

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Build the image
echo "Building Docker image..."
docker build -t ${IMAGE_URI} -f sagemaker.Dockerfile .

echo "Pushing ${IMAGE_URI}"
docker push ${IMAGE_URI}

echo "Build and push completed successfully!"
```

</details>

3. 在 AWS SageMaker 上部署模型服务，请参考 [deploy_and_serve_endpoint.py](https://github.com/sgl-project/sglang/blob/main/examples/sagemaker/deploy_and_serve_endpoint.py)。更多信息见 [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk)。
   1. 默认情况下，SageMaker 上的模型服务器会用以下命令运行：`python3 -m sglang.launch_server --model-path opt/ml/model --host 0.0.0.0 --port 8080`。这适合用 SageMaker 托管你自己的模型。
   2. 如需修改模型服务参数，[serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) 脚本允许通过 `SM_SGLANG_` 前缀的环境变量指定 `python3 -m sglang.launch_server --help` CLI 中的所有可用选项。
   3. serve 脚本会自动把所有带 `SM_SGLANG_` 前缀的环境变量从 `SM_SGLANG_INPUT_ARGUMENT` 转换为 `--input-argument`，再交给 `python3 -m sglang.launch_server` CLI 解析。
   4. 例如，要运行带 reasoning parser 的 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，只需增加环境变量 `SM_SGLANG_MODEL_PATH=Qwen/Qwen3-0.6B` 和 `SM_SGLANG_REASONING_PARSER=qwen3`。

</details>

## 通用注意事项

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 是默认 attention kernel 后端。它只支持 sm75 及以上架构。如果你在 sm75+ 设备（例如 T4、A10、A100、L4、L40S、H100）上遇到 FlashInfer 相关问题，请通过添加 `--attention-backend triton --sampling-backend pytorch` 切换到其他 kernel，并在 GitHub 上提交 issue。
- 如果要在本地重新安装 flashinfer，请运行：`pip3 install --upgrade flashinfer-python --force-reinstall --no-deps`，然后用 `rm -rf ~/.cache/flashinfer` 删除缓存。
