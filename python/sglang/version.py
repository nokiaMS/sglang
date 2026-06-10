# SGLang 版本号管理模块，依次尝试从构建版本、包元数据和 setuptools_scm 获取版本号
try:
    # 优先从构建时生成的 _version 模块导入版本号
    from sglang._version import __version__, __version_tuple__
except ImportError:
    try:
        # 其次从包元数据中获取版本号
        import importlib.metadata

        __version__ = importlib.metadata.version("sglang")
        __version_tuple__ = tuple(__version__.split("."))
    except Exception:
        try:
            # 再次从 setuptools_scm 动态获取版本号（开发环境）
            import pathlib

            from setuptools_scm import get_version

            # point to the directory containing pyproject.toml.
            # 指向包含 pyproject.toml 的项目根目录
            project_root = pathlib.Path(__file__).parent.parent.parent
            __version__ = get_version(
                root=str(project_root), fallback_version="0.0.0.dev0"
            )
            __version_tuple__ = tuple(__version__.split("."))
        except Exception:
            # 所有方式均失败时的回退版本号
            # Fallback for development without build
            __version__ = "0.0.0.dev0"
            __version_tuple__ = (0, 0, 0, "dev0")
