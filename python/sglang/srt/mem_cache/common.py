# 内存缓存通用工具模块
# 本文件提供了 KV 缓存（键值缓存）内存管理所需的基础工具函数和 Triton 内核，
# 包括：KV 索引到页面索引的转换、请求到令牌池的写入、令牌槽位的分配与释放、
# 以及针对预填充（extend）和解码（decode）阶段的 KV 缓存分配逻辑。

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_hip, support_triton
from sglang.srt.utils.common import ceil_align

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
# 带前缀缓存的 Mamba 请求需要 2+1 个槽位：2 个用于乒乓缓存，1 个用于运行中的 Mamba 状态
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# 不带前缀缓存的 Mamba 请求每个请求需要 1 个状态槽位
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    """将 KV 索引数组转换为页面索引数组。通过按 page_size 步长采样并整除得到页号。"""
    # The page is guaranteed to be full except the last page.
    # 除了最后一页，其余页面保证是满的
    if page_size == 1:
        return kv_indices

    # 每隔 page_size 取一个索引，再整除 page_size 得到页号
    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    """根据 KV 索引数量和页面大小，计算所需的页面数量（向上取整）。"""
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    """将长度向下对齐到页面大小的整数倍。"""
    return (length // page_size) * page_size


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    """如果请求未被标记为跳过，则将其作为未完成请求缓存到基数树缓存中。"""
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    """Triton 内核：将前缀和扩展的 KV 缓存索引写入请求到令牌的映射池。"""
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    # 加载当前请求的池索引、前缀长度、序列长度和前缀张量指针
    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    # 写入前缀部分：将前缀的 KV 缓存索引复制到 req_to_token 池
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    # 计算当前请求在 out_cache_loc 中的起始偏移量（对所有先前请求的 extend_len 求和）
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    # 写入扩展部分：将新分配的 KV 缓存位置写入 req_to_token 池中前缀之后的位置
    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    """将缓存索引写入请求到令牌池。优先使用 Triton 内核加速，回退到逐请求 CPU 写入。"""
    if support_triton(get_global_server_args().attention_backend):
        # 收集所有前缀张量的数据指针，构造设备端指针张量
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        # 使用 Triton 内核批量写入
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        # 回退路径：逐请求在 CPU 上写入前缀和扩展部分的缓存索引
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            # 写入前缀部分
            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            # 写入扩展部分
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """获取每个请求前缀部分的最后一个 KV 缓存位置索引。
    根据硬件平台和注意力后端选择合适的实现。"""
    attn_backend = get_global_server_args().attention_backend
    uses_triton_dispatch = attn_backend not in ("ascend", "torch_native")

    if _is_hip and uses_triton_dispatch:
        # HIP-only: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend). The bug is in the Triton HIP codegen, not
        # in any particular attention backend, so route every HIP path
        # that would otherwise use get_last_loc_triton through the
        # int32-safe variant. Non-HIP hardware keeps the original
        # dispatcher below.
        # HIP 平台专用：旧版 Triton 内核在 HIP 上存在 int32->int64 混合宽度存储的
        # 编译错误，可能导致 last_loc 值越界。因此 HIP 路径全部使用安全变体。
        return get_last_loc_triton_safe(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    if uses_triton_dispatch:
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """使用 PyTorch 操作获取每个请求前缀的最后一个 KV 缓存位置。
    若前缀长度为 0 则返回 -1。"""
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def _get_last_loc_safe_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result_i32,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
    PREFIX_DTYPE_IS_I64: tl.constexpr,
):
    """安全的 get_last_loc Triton 内核：内核内使用 int32 结果缓冲区，
    避免在 HIP 平台上出现 int32->int64 混合宽度存储的编译错误。"""
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    # 根据 prefix_lens 的数据类型选择不同的计算路径，避免类型溢出
    if PREFIX_DTYPE_IS_I64:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    else:
        prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
        req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)
        token_index = req_pool_indices.to(tl.int64) * req_to_token_stride + (
            prefix_lens.to(tl.int64) - 1
        )

    # 仅当前缀长度 > 0 时才加载对应的 token 值
    token_mask = mask & (prefix_lens > 0)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)
    # Result stays int32 (req_to_token dtype); caller promotes after return.
    # 结果保持 int32（req_to_token 的数据类型），调用方在返回后提升精度
    tl.store(result_i32 + offset, tokens, mask=mask)


def get_last_loc_triton_safe(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """Fused `last_loc` Triton kernel whose in-kernel result buffer is int32
    (the dtype of req_to_token). The consumer-dtype promotion happens in
    torch after the kernel returns, so Triton never issues a mixed-width
    store — avoiding the HIP int32->int64 store bug hit by the legacy kernel.
    """
    """安全版 get_last_loc Triton 实现：内核内结果缓冲区使用 int32，
    在内核返回后由 PyTorch 进行精度提升，避免 HIP 上的混合宽度存储错误。"""
    num_tokens = prefix_lens_tensor.shape[0]
    BLOCK_SIZE = 256
    # 创建 int32 结果缓冲区，与 req_to_token 的数据类型一致
    result_i32 = torch.empty(
        num_tokens, dtype=torch.int32, device=prefix_lens_tensor.device
    )
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)
    _get_last_loc_safe_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result_i32,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        PREFIX_DTYPE_IS_I64=(prefix_lens_tensor.dtype == torch.int64),
    )
    # 内核返回后将结果转换为 prefix_lens_tensor 的数据类型
    return result_i32.to(prefix_lens_tensor.dtype)


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """标准版 get_last_loc Triton 内核：获取每个请求前缀的最后一个 KV 缓存位置。"""
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    # 计算前缀最后一个 token 在 req_to_token 中的索引
    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    """使用标准 Triton 内核获取每个请求前缀的最后一个 KV 缓存位置。"""
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    """从 KV 缓存池中分配指定数量的令牌槽位。
    先尝试驱逐树缓存中的条目腾出空间，分配失败时抛出内存不足异常。
    若 backup_state 为 True，则在分配前备份分配器状态并一起返回。"""
    allocator = tree_cache.token_to_kv_pool_allocator
    # 先尝试通过驱逐树缓存中的条目腾出足够空间
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        # 备份分配器状态，用于后续可能的回滚
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        # 分配失败，构造详细的内存不足错误信息
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    # 若需要备份状态，则返回 (缓存位置, 状态) 元组；否则仅返回缓存位置
    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    """从树缓存中驱逐条目以腾出至少 num_tokens 个 KV 缓存槽位。
    根据分配器类型（混合分配器或标准分配器）选择不同的驱逐策略。"""
    if tree_cache is None:
        return

    # 分块缓存不需要驱逐
    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        # 混合分配器：需要同时检查全量可用空间和 SWA 可用空间
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            # 计算需要驱逐的 token 数量（全量和 SWA 分别计算）
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        # 标准分配器：仅检查总可用空间
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    """为预填充（extend）阶段分配分页 KV 缓存槽位。
    估算所需 token 数时会多估每个请求一个页面的空间，以确保分配成功。"""
    # Over estimate the number of tokens: assume each request needs a new page.
    # 高估所需 token 数：假设每个请求需要一个新页面，避免分配不足
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    # 使用分页分配器的扩展分配接口
    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    """从请求到令牌池中分配请求槽位。对于混合池，还需处理 Mamba 状态的分配和驱逐。"""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        # 混合池：需要额外检查 Mamba 状态的可用空间
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        # 根据是否支持 Mamba 前缀缓存确定每个请求所需的 Mamba 状态数
        factor = (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE
            if tree_cache.supports_mamba()
            else MAMBA_STATE_PER_REQ_NO_CACHE
        )
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            # Mamba 状态不足时，尝试从树缓存中驱逐以释放空间
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    # 从池中分配请求槽位
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.
    为预填充（extend）批次分配 KV 缓存，并将缓存索引写入请求到令牌池。

    Returns:
        out_cache_loc: allocated cache locations（已分配的缓存位置）
        req_pool_indices_device: request pool indices as a device tensor（设备端请求池索引张量）
        req_pool_indices_cpu: request pool indices as a CPU tensor (host mirror)（CPU 端请求池索引张量，主机镜像）
    """
    # free out-of-window swa tokens
    # 释放超出滑动窗口的 SWA token
    batch.maybe_evict_swa()

    # 获取每个请求的前缀索引张量
    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    # 创建分配所需的张量（CPU 和设备端）
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    # 分配请求槽位
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    # 分配 KV 缓存（失败时抛出异常）
    if batch.tree_cache.page_size == 1:
        # 非分页模式：直接分配 token 槽位
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        # 分页模式：构建 last_loc（每个请求前缀的最后一个 token 位置）
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
        )

    # Write to req_to_token_pool
    # 将缓存索引写入请求到令牌池
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    """为解码（decode）批次分配分页 KV 缓存槽位。"""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    # 高估所需 token 数：假设每个请求需要一个新页面
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    # 使用分页分配器的解码分配接口
    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.
    为解码（decode）批次分配 KV 缓存，并将缓存索引写入请求到令牌池。

    Returns:
        out_cache_loc: allocated cache locations（已分配的缓存位置）
    """

    # 释放超出滑动窗口的 SWA token
    batch.maybe_evict_swa()

    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        # 非分页模式：直接分配 token 槽位
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        # 分页模式：获取每个请求最后一个 token 的位置，并分配分页槽位
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        # 计算解码后的新序列长度
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    # 将新分配的缓存位置写入请求到令牌池中对应的序列位置
    if batch.model_config.is_encoder_decoder:
        # 编码器-解码器模型：位置偏移需要加上编码器长度
        locs = batch.encoder_lens + seq_lens_gpu
    else:
        # 仅解码器模型：直接使用当前序列长度作为写入位置
        locs = seq_lens_gpu.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    """释放请求的 KV 缓存资源，包括将完成的请求插入树缓存、
    释放过度分配的 KV 缓存、释放 Mamba 状态和请求槽位。"""
    # MambaRadixCache may alloc mamba state before alloc KV cache
    # MambaRadixCache 可能在分配 KV 缓存之前就分配了 Mamba 状态
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        # 清理早期分配的 Mamba 状态
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    # 将完成的请求缓存到树缓存中（如果未被标记为跳过）
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # and bookkeeping flag sync internally, then sets req_pool_idx = None.
    # StreamingSession 内部处理推测解码的尾部修剪和簿记标志同步，然后将 req_pool_idx 置为 None
    if req.req_pool_idx is None:
        return

    # 获取过度分配的 KV 缓存范围 [start_p, end_p)
    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    # strip_thinking_cache 有意将输出 token 报告为过度分配，以便走下面的释放路径
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    # 分页模式下，将起始位置向上对齐到页面边界
    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    # 释放过度分配的 KV 缓存 token
    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    # 如果前缀缓存不管理 Mamba 状态，则必须在此处释放
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # 释放请求到令牌池中的请求槽位
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    """返回树缓存中可用和可驱逐空间的描述字符串，用于内存不足时的错误信息。"""
    return tree_cache.available_and_evictable_str()
