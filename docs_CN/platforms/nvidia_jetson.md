<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# NVIDIA Jetson Orin

## 前置条件

Before starting, ensure the following:

- [**NVIDIA Jetson AGX Orin Devkit**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) is set up with **JetPack 6.1** or later.
- **CUDA Toolkit** and **cuDNN** are installed.
- Verify that the Jetson AGX Orin is in **high-performance mode**:
```bash
sudo nvpmodel -m 0
```
* * * * *
## Installing and running SGLang with Jetson Containers
Clone the jetson-containers github repository:
```
git clone https://github.com/dusty-nv/jetson-containers.git
```
Run the installation script:
```
bash jetson-containers/install.sh
```
Build the container image:
```
jetson-containers build sglang
```
Run the container:
```
jetson-containers run $(autotag sglang)
```
Or you can also manually run a container with this command:
```
docker run --runtime nvidia -it --rm --network=host IMAGE_NAME
```
* * * * *

Running 推理
-----------------------------------------

Launch the server:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192
```
The quantization and limited context length (`--dtype half --context-length 8192`) are due to the limited computational resources in [Nvidia jetson kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/). A detailed explanation can be found in [服务器参数](../advanced_features/server_arguments.md).

After launching the engine, refer to [Chat completions](/docs_CN/basic_usage/openai_api_completions.md#用法) to test the usability.
* * * * *
Running quantization with TorchAO
-------------------------------------
TorchAO is suggested to NVIDIA Jetson Orin.
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```
This enables TorchAO's int4 weight-only quantization with a 128-group size. The usage of `--torchao-config int4wo-128` is also for memory efficiency.


* * * * *
Structured output with XGrammar
-------------------------------
请参考 [SGLang doc structured output](../advanced_features/structured_outputs.ipynb).
* * * * *

Thanks to the support from [Nurgaliyev Shakhizat](https://github.com/shahizat), [Dustin Franklin](https://github.com/dusty-nv) and [Johnny Núñez Cano](https://github.com/johnnynunez).

参考资料
----------
-   [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
