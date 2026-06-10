# 文件名: conftest.py - Pytest配置和共享夹具
"""Shared fixtures for source_patcher tests.

The sample module is defined as an inline string and written to a temp file
at test time, avoiding CI complaints about fixture files without test registration.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

SAMPLE_MODULE_NAME = "_source_patcher_test_fixtures.sample_module"

SAMPLE_MODULE_SOURCE = '''\
GLOBAL_VAR = "global_value"


class HelperClass:
    """Utility class referenced by SampleClass to test cross-class calls."""

    @staticmethod
    # 执行formatvalue
    def format_value(value: str) -> str:
        return f"[{value}]"


class SampleClass:
    # 执行greet
    def greet(self, name: str) -> str:
        greeting = f"hello {name}"
        return greeting

    # 执行compute
    def compute(self, x: int) -> int:
        result = x * 2 + 1
        return result

    # 执行usesglobal
    def uses_global(self) -> str:
        return f"value={GLOBAL_VAR}"

    # 执行useshelper
    def uses_helper(self, value: str) -> str:
        return HelperClass.format_value(value)


# 执行standalonefunction
def standalone_function(a: int, b: int) -> int:
    return a + b
'''


@pytest.fixture(scope="session")
# 执行samplemodule
def sample_module() -> ModuleType:
    """Load the sample module from a temp file and register it in sys.modules."""
    if SAMPLE_MODULE_NAME in sys.modules:
        return sys.modules[SAMPLE_MODULE_NAME]

    tmpdir = tempfile.mkdtemp(prefix="source_patcher_fixtures_")
    module_path = Path(tmpdir) / "sample_module.py"
    module_path.write_text(SAMPLE_MODULE_SOURCE)

    spec = importlib.util.spec_from_file_location(SAMPLE_MODULE_NAME, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[SAMPLE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
