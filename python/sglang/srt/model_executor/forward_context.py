# 本文件实现了每次前向调用（forward call）的控制上下文机制。
# 核心类 ForwardContext 是一个不可变的数据类，保存模型层在深层调用时
# 所需的控制配置。通过模块级全局变量 _current 管理当前活跃的上下文，
# 并提供 get_forward_context() 等访问接口。forward_context() 上下文
# 管理器支持临时覆盖上下文，用于 PDmux、MTP draft loop、TBO 等场景。

"""Per-forward-call control context.

Owns ``ForwardContext`` — a frozen dataclass holding control configs the model
layer reads at depth via ``get_forward_context()``. The only mandatory field
today is ``attn_backend``; pool refs are derived from ``attn_backend.*``
(every backend caches them at ``__init__``), so a published ``ForwardContext``
is enough to resolve the active pools without a separate global.

``ModelRunner._forward_raw`` publishes a fresh ``ForwardContext`` for the
duration of each forward; callers that need a per-call override (PDmux
per-stream backend, frozen-KV MTP draft loop, TBO per-child dispatch) use
``dataclasses.replace`` and wrap the override scope with ``forward_context()``.

Distinct from ``sglang.srt.compilation.piecewise_context_manager.ForwardContext``,
which collects compilation-time refs for the piecewise CUDA graph backend.

Concurrency: ``_current`` is a plain module-level global, not thread-local.
This matches the ``global_server_args`` precedent and is safe because each
forward runs synchronously on a single Python thread per worker process. If
worker threads ever share a process, migrate to ``contextvars.ContextVar``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs. Read via ``get_forward_context()``;
    extend by adding fields here. Frozen so accidental mutation raises at
    write time — use ``dataclasses.replace`` for per-call overrides."""

    # 注意力后端实例，是 ForwardContext 中唯一的必填字段，
    # 其他池引用（token_to_kv_pool、req_to_token_pool）均可从中派生
    attn_backend: AttentionBackend


# 模块级全局变量，保存当前活跃的前向上下文；
# 非线程局部变量，因为每个 worker 进程中的前向调用在单线程上同步执行
_current: Optional[ForwardContext] = None


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set the active context; return the previous one for explicit
    save/restore. Prefer the ``forward_context()`` context manager."""
    # 设置当前活跃上下文，同时返回之前的上下文以便后续恢复
    global _current
    prev, _current = _current, ctx
    return prev


def has_forward_context() -> bool:
    """检查当前是否存在活跃的前向上下文"""
    return _current is not None


def get_forward_context() -> ForwardContext:
    """获取当前活跃的前向上下文；若不存在则抛出断言错误"""
    assert _current is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return _current


def get_attn_backend() -> AttentionBackend:
    """从当前前向上下文中获取注意力后端"""
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> KVCache:
    """从当前注意力后端中获取 token 到 KV 缓存的映射池"""
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    """从当前注意力后端中获取请求到 token 位置的映射池"""
    return get_attn_backend().req_to_token_pool


@contextmanager
def forward_context(ctx: ForwardContext):
    """前向上下文的上下文管理器，在 with 块内临时切换上下文，
    退出时自动恢复为之前的上下文。用于 PDmux、MTP draft loop、
    TBO 等需要逐次调用覆盖上下文的场景。"""
    # 保存旧上下文并设置新上下文
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        # 无论是否异常，都恢复之前的上下文
        set_forward_context(prev)
