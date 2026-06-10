# SGLang 包的初始化文件，定义了公共 API 并处理平台兼容性补丁
# SGLang public APIs

# Install stubs early for platforms where certain dependencies are unavailable
# (e.g. macOS/MPS has no triton, and torch.mps lacks Stream / set_device /
# get_device_properties).  This must run before any downstream imports.
# 提前为某些依赖不可用的平台安装桩模块（例如 macOS/MPS 没有 triton）
import platform as _platform
import sys as _sys

# 检测是否为 macOS ARM64 平台
if _sys.platform == "darwin" and _platform.machine() == "arm64":
    try:
        import torch as _torch

        # 检查 MPS 后端是否可用
        if _torch.backends.mps.is_available():
            # 安装 triton 桩模块以兼容 MPS 平台
            from sglang._triton_stub import install as _install_triton_stub

            _install_triton_stub()
            del _install_triton_stub

            # 安装 MPS 桩模块以补全缺失的 API
            from sglang._mps_stub import install as _install_mps_stub

            _install_mps_stub()
            del _install_mps_stub
        del _torch
    except ImportError:
        pass
del _platform
del _sys

# 应用 HuggingFace transformers 补丁
from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches

_apply_hf_patches()
del _apply_hf_patches

# 前端语言 API 导入
# Frontend Language APIs
from sglang.global_config import global_config
from sglang.lang.api import (
    Engine,
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    separate_reasoning,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)

# 延迟导入某些库以加速 import sglang
# Lazy import some libraries
from sglang.utils import LazyImport
from sglang.version import __version__

# 后端实现的延迟导入
Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
Crusoe = LazyImport("sglang.lang.backend.crusoe", "Crusoe")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# 运行时引擎 API 的延迟导入
# Runtime Engine APIs
ServerArgs = LazyImport("sglang.srt.server_args", "ServerArgs")
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")

# 公共导出列表
__all__ = [
    "Engine",
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "RuntimeEndpoint",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "ServerArgs",
    "Anthropic",
    "Crusoe",
    "LiteLLM",
    "OpenAI",
    "VertexAI",
    "global_config",
    "__version__",
]
