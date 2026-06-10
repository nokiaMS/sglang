# 异步不变量探测模块
# 提供无需CPU同步的GPU端异步断言检查（NaN、Inf、越界、页对齐）
# 所有探测都受SGLANG_ENABLE_ASYNC_ASSERT环境变量控制（生产环境默认关闭）
# 违规将在下一次CUDA同步点以断言形式抛出，而非静默NaN级联或非法地址崩溃

"""Async invariant probes — fire torch._assert_async without CPU sync.

All probes are gated on SGLANG_ENABLE_ASYNC_ASSERT (default off in prod).
When the gate is on, a violation surfaces as an assertion at the next CUDA
sync point instead of as a silent NaN cascade or illegal-address crash.
"""

import torch  # PyTorch深度学习框架

from sglang.srt.environ import envs  # SGLang环境变量配置


def maybe_detect_nan(tensor: torch.Tensor, msg: str = ""):  # 异步NaN检测，无GPU-CPU同步，错误在下一次同步点抛出
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    # 异步NaN检查 — 无GPU-CPU同步，错误在下一次同步点抛出。
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():  # 如果未启用异步断言
        return  # 直接返回
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")  # 异步断言：张量中无NaN


def maybe_detect_inf(tensor: torch.Tensor, msg: str = ""):  # 异步Inf检测，fp16溢出先出现Inf再变NaN
    """Async Inf check — fp16 overflow surfaces as Inf before NaN."""
    # 异步Inf检查 — fp16溢出先以Inf形式出现再变NaN。
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():  # 如果未启用异步断言
        return  # 直接返回
    torch._assert_async(~torch.any(torch.isinf(tensor)), f"Inf detected! {msg}")  # 异步断言：张量中无Inf


def maybe_detect_oob(indices: torch.Tensor, low: int, high: int, msg: str):  # 异步越界检测，无GPU-CPU同步
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point."""
    # 异步越界检查 — 无GPU-CPU同步，错误在下一次同步点抛出。
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():  # 如果未启用异步断言
        return  # 直接返回
    if indices.numel() == 0:  # 如果索引张量为空
        return  # 直接返回
    torch._assert_async(  # 异步断言
        (indices.min() >= low) & (indices.max() < high),  # 检查所有索引在[low, high)范围内
        f"OOB indices not in [{low}, {high}): {msg}",  # 越界错误消息
    )


def maybe_detect_page_aligned(indices: torch.Tensor, page_size: int, msg: str):  # 异步页对齐检测，检查slot ID是否按页大小对齐
    """Async page-alignment check on slot ids."""
    # 对slot ID的异步页对齐检查。
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():  # 如果未启用异步断言
        return  # 直接返回
    if indices.numel() == 0 or page_size <= 1:  # 如果索引为空或页大小为1（无需对齐）
        return  # 直接返回
    torch._assert_async(  # 异步断言
        (indices % page_size == 0).all(),  # 检查所有索引都能被页大小整除
        f"page-misaligned indices (page_size={page_size}): {msg}",  # 页未对齐错误消息
    )
