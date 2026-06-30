<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# PyPI Package 发布流程

## Update the version in code
Update the package version in `python/pyproject.toml` and `python/sglang/__init__.py`.

## Upload the PyPI package

```
pip install build twine
```

```
cd python
bash upload_pypi.sh
```

## Make a release in GitHub
Make a new release https://github.com/sgl-project/sglang/releases/new.
