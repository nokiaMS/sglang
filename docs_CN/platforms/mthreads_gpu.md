<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# Moore Threads GPUs

本文档 describes how run SGLang on Moore Threads GPUs. If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## 安装 SGLang

你可以 install SGLang using one of the methods below.

### 安装 from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# Install sglang python package
cd ..
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```
