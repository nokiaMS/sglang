"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.triton_ops.cache_move import (
    copy_all_layer_kv_cache_tiled,
    set_kv_buffer_prefix_valid_tiled,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()
# `SGLANG_AITER_KV_CACHE_LAYOUT` is only meaningful on the ROCm AITER backend
# (HIP + --enable-aiter / SGLANG_USE_AITER=1). On any other platform / backend
# the SHUFFLE 5D pool layout has no consumer kernels, so the env var is
# silently ignored and the legacy NHD layout is used.
_use_aiter = bool(envs.SGLANG_USE_AITER.get()) and _is_hip


def conv_window_dedup_enabled(
    is_npu: bool, is_cpu: bool, speculative_eagle_topk: Optional[int]
) -> bool:
    """Whether the deduplicated sliding-window conv-intermediate layout is safe.

    It is only correct for a *linear* draft chain (``speculative_eagle_topk <= 1``,
    i.e. NEXTN / MTP): consecutive draft tokens then form a true sliding window, so
    the overlapping physical columns hold identical values. Under EAGLE *tree*
    verify (``topk > 1``) the conv kernel walks per-token tree ancestors, so aliased
    columns can need different values from different parent chains -> fall back to
    the dense layout. NPU/CPU also keep the dense layout (their kernels assume
    contiguous per-step windows). See ``MambaPool.__init__``.
    """
    return (
        not is_npu
        and not is_cpu
        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)
    )


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    size_limit: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
            size_limit=size_limit,
        )

    if _is_cpu and _cpu_has_amx_support:
        return torch.ops.sgl_kernel.store_cache_cpu(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            row_dim,
        )

    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


def _set_kv_buffer_prefix_valid_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_dim: int,
    store_dtype: torch.dtype,
) -> None:
    if k.numel() == 0 or loc_2d.numel() == 0 or commit_lens.numel() == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()

    row_bytes = row_dim * store_dtype.itemsize
    if row_bytes <= 0:
        return

    if row_bytes >= 8192:
        bytes_per_tile = 512
        num_warps = 8
    elif row_bytes >= 4096:
        bytes_per_tile = 256
        num_warps = 4
    else:
        bytes_per_tile = 128
        num_warps = 4

    grid = (
        int(loc_2d.shape[0]),
        int(loc_2d.shape[1]),
        triton.cdiv(row_bytes, bytes_per_tile),
    )

    set_kv_buffer_prefix_valid_tiled[grid](
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        int(k.stride(0) * k.element_size()),
        int(v.stride(0) * v.element_size()),
        int(k_cache.stride(0) * k_cache.element_size()),
        int(v_cache.stride(0) * v_cache.element_size()),
        int(loc_2d.shape[1]),
        ROW_BYTES=row_bytes,
        BYTES_PER_TILE=bytes_per_tile,
        num_warps=num_warps,
        num_stages=2,
    )


class ReqToTokenPool:
    """请求到 token 物理 KV 位置的映射池。

    该类维护一个二维矩阵 ``req_to_token``，行表示 request slot
    （即 ``req_pool_idx``），列表示请求内部的 token 位置，单元格保存
    对应 token 在第三层 KV cache 中的物理 slot/page index。它只保存
    映射关系，不保存真实 K/V tensor。

    A memory pool that maps a request to its token locations.
    """

    enable_mamba_extra_buffer_lazy: bool = False

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        """初始化 request 到 token 位置的映射矩阵和空闲 request 行。

        Args:
            size: 可同时分配的 request slot 数量。
            max_context_len: 每个 request 最多可记录的 token 位置数量。
            device: ``req_to_token`` 张量所在设备。
            enable_memory_saver: 是否在分配映射矩阵时启用显存节省区域。
        """
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),    # 表示最多能够管理多少request/slot；每个请求最多能保留多少token。
                dtype=torch.int32,                      # 每个元素的类型。
                device=device                           # 指定张量所在设备。
            )
        self.free_slots = list(range(1, self._alloc_size))

    def write(self, indices, values):
        """把 token 的物理 KV 位置写入指定 request 行/列。"""
        self.req_to_token[indices] = values

    def available_size(self):
        """返回当前还可分配的 request slot 数量。"""
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        """为一批请求分配或复用 request slot。

        已经带有 ``req_pool_idx`` 的请求会复用原行，常见于 chunked
        prefill 跨 chunk 继续执行或已有 committed KV 的请求；新请求会
        从 ``free_slots`` 中取空闲行。如果空闲行不足则返回 ``None``。

        Returns:
            每个请求对应的 ``req_pool_idx`` 列表；分配失败时返回 ``None``。
        """
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].inflight_middle_chunks > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        """释放单个请求占用的 request slot，并清空请求上的 ``req_pool_idx``。"""
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        """重置所有 request slot 为空闲状态。"""
        self.free_slots = list(range(1, self._alloc_size))


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)

        self.size = size
        self.device = device

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    _init_npu_conv_state,
                )

                conv_state = _init_npu_conv_state(
                    conv_state[0], conv_state_shape, speculative_num_draft_tokens
                )

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                if _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    temporal_state_shape = (
                        *temporal_state_shape[:-2],
                        temporal_state_shape[-1],
                        temporal_state_shape[-2],
                    )
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token
                # during target verify.
                #
                # On CUDA (Triton conv kernel + Triton scatter) we use a
                # *deduplicated sliding-window* layout: consecutive draft tokens'
                # (K-1)-wide windows overlap by (K-2), so instead of D separate
                # [dim, K-1] windows we store one shared [dim, D+K-2] buffer per
                # (layer, slot) and expose an overlapping `as_strided` view of
                # logical shape [num_layers, size+1, draft_tokens, dim, K-1] where
                # step `t`'s window is the slice shared[..., :, t:t+K-1]. This
                # halves the conv-intermediate footprint (D*(K-1) -> D+K-2 columns)
                # with no numerical change: both the conv kernel write (idempotent
                # overlapping stores) and `fused_conv_window_scatter_with_mask`
                # consume the view through its strides.
                #
                # Dedup the sliding-window conv-intermediate only when it is safe:
                # CUDA + a linear draft chain (topk <= 1). NPU/CPU and EAGLE tree
                # verify (topk > 1) keep the dense layout -- see
                # `conv_window_dedup_enabled` for the full rationale. The
                # `fused_conv_window_scatter_with_mask` scatter is layout-agnostic,
                # so the dense fallback reads correctly through the same code path.
                dedup_conv_window = conv_window_dedup_enabled(
                    _is_npu, _is_cpu, speculative_eagle_topk
                )
                self._intermediate_conv_window_phys = []
                if dedup_conv_window:
                    intermediate_conv_window_cache = []
                    for conv_shape in conv_state_shape:
                        conv_dim, win = conv_shape  # win == conv_kernel - 1 == K-1
                        shared_win = (
                            speculative_num_draft_tokens + win - 1
                        )  # D + (K-1) - 1
                        phys = torch.zeros(
                            size=(
                                num_mamba_layers,
                                spec_state_size + 1,
                                conv_dim,
                                shared_win,
                            ),
                            dtype=conv_dtype,
                            device="cuda",
                        )
                        # view[l, s, step, d, w] = phys[l, s, d, step + w]
                        view = phys.as_strided(
                            (
                                phys.shape[0],
                                phys.shape[1],
                                speculative_num_draft_tokens,
                                conv_dim,
                                win,
                            ),
                            (
                                phys.stride(0),
                                phys.stride(1),
                                phys.stride(3),  # step -> shared-win axis (stride 1)
                                phys.stride(2),  # dim
                                phys.stride(3),  # win -> shared-win axis (stride 1)
                            ),
                        )
                        self._intermediate_conv_window_phys.append(phys)
                        intermediate_conv_window_cache.append(view)
                else:
                    # Original dense layout (NPU/CPU, or EAGLE tree verify): one
                    # [dim, K-1] window per draft token.
                    # Shape: [num_layers, size+1, draft_tokens, dim, K-1]
                    intermediate_conv_window_cache = [
                        torch.zeros(
                            size=(
                                num_mamba_layers,
                                spec_state_size + 1,
                                speculative_num_draft_tokens,
                                conv_shape[0],
                                conv_shape[1],
                            ),
                            dtype=conv_dtype,
                            device="cuda",
                        )
                        for conv_shape in conv_state_shape
                    ]
                    self._intermediate_conv_window_phys = intermediate_conv_window_cache
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    # Report the deduplicated PHYSICAL conv-window buffers (the view
                    # over-reports its logical, un-deduplicated size).
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(self._intermediate_conv_window_phys) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            mem_usage_bytes = self.mamba_cache.mem_usage_bytes()
            if isinstance(self.mamba_cache, self.SpeculativeState):
                # `intermediate_conv_window` is an as_strided view whose logical
                # shape over-reports its real footprint; charge the physical buffers
                # instead. No-op for the dense layout, where the view and the
                # physical tensors coincide.
                mem_usage_bytes -= get_tensor_size_bytes(
                    self.mamba_cache.intermediate_conv_window
                )
                mem_usage_bytes += get_tensor_size_bytes(
                    self._intermediate_conv_window_phys
                )
            self.mem_usage = mem_usage_bytes / GB
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def clear_slots(self, indices: torch.Tensor):
        """Zero out mamba state at the given pool indices. Must run on forward stream."""
        need_size = len(indices)
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, indices] = z

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_indices] = self.mamba_cache.conv[i][
                :, src_indices
            ]
        self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
            :, src_indices
        ]

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )
        current_platform.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        conv_cpu, temporal_cpu = mamba_cache_cpu
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        current_platform.synchronize()

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = enable_mamba_extra_buffer_lazy
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
    ):
        self.mamba_pool = MambaPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
        )
        self.mamba_allocator = MambaSlotAllocator(
            size=mamba_size,
            device=device,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        # Optional int8 checkpoint pool: the radix caches states here (int8) instead
        # of holding them in the active bf16 pool -> ~2x cached-prefix capacity at
        # fixed memory. Strategy-agnostic (no_buffer / extra_buffer / spec).
        from sglang.srt.mem_cache.mamba_checkpoint_pool import (
            maybe_init_int8_mamba_checkpoint_pool,
        )

        self.mamba_ckpt_pool = maybe_init_int8_mamba_checkpoint_pool(
            mamba_size=mamba_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
        )

        self.device = device
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            if req.mamba_pool_idx is not None:  # for radix cache / continuing chunked
                pass
            else:
                mid = self.mamba_allocator.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_allocator.available_size()=}, {len(reqs)=}"
                req.mamba_pool_idx = mid[0]
                req.mamba_needs_clear = True
            mamba_indices.append(req.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    self._alloc_ping_pong_buffer(req)
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_state_buf_infos(self):
        return self.mamba_pool.get_contiguous_buf_infos()

    def get_state_dim_per_tensor(self):
        return self.mamba_pool.get_state_dim_per_tensor()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def get_mamba_ping_pong_keep_idx(self, req: Req) -> int:
        """Return the ping-pong index holding the most recent tracked state.

        In lazy mode the valid state stays at next_track_idx (no eager swap).
        In normal mode it is at the "other" index (swapped after each track).
        """
        if self.enable_mamba_extra_buffer_lazy:
            return req.mamba_next_track_idx
        return self.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)

    def _alloc_ping_pong_buffer(self, req: Req):
        """Allocate the ping-pong track buffer for a new request.

        Lazy mode allocates 1 slot with the second set to -1 (allocated
        on demand at track boundaries). Normal mode allocates all slots upfront.
        """
        n = (
            1
            if self.enable_mamba_extra_buffer_lazy
            else self.mamba_ping_pong_track_buffer_size
        )
        slots = self.mamba_allocator.alloc(n)
        assert slots is not None, (
            "Not enough space for mamba ping pong idx, "
            "try to increase --mamba-full-memory-ratio."
        )
        buf = torch.full(
            (self.mamba_ping_pong_track_buffer_size,),
            -1,
            dtype=slots.dtype,
            device=slots.device,
        )
        buf[:n] = slots
        req.mamba_ping_pong_track_buffer = buf
        req.mamba_next_track_idx = 0

    def set_mamba_ping_pong_slot(self, req: Req, idx: int, value):
        """Update a ping-pong slot value and sync the device-side mapping.

        The req holds the authoritative buffer; this keeps the
        req_index_to_mamba_ping_pong_track_buffer_mapping in sync so that
        set_mamba_track_indices_from_reqs reads correct slot indices.
        """
        req.mamba_ping_pong_track_buffer[idx] = value
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx] = (
            req.mamba_ping_pong_track_buffer
        )

    def donate_mamba_ping_pong_slot(
        self, req: Req, new_slot: torch.Tensor
    ) -> torch.Tensor:
        """Donate the tracked-state ping-pong slot to the radix cache.

        Returns the old slot index (shape [1]) for cache insertion and
        replaces it with new_slot so the request can continue tracking.
        In lazy mode the valid state is at next_track_idx; in normal mode
        it is at the "other" index.
        """
        donate_idx = self.get_mamba_ping_pong_keep_idx(req)
        mamba_value_donated = (
            req.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
        )
        assert mamba_value_donated.item() != -1, (
            f"Donated mamba slot is -1: donate_idx={donate_idx}, "
            f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
            f"next_track_idx={req.mamba_next_track_idx}, "
            f"rid={req.rid}"
        )
        self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])
        return mamba_value_donated

    def free_mamba_cache(
        self, req: Req, mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_allocator.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            if self.enable_mamba_extra_buffer_lazy:
                mamba_ping_pong_track_buffer_to_free = (
                    mamba_ping_pong_track_buffer_to_free[
                        mamba_ping_pong_track_buffer_to_free != -1
                    ]
                )
            self.mamba_allocator.free(mamba_ping_pong_track_buffer_to_free)
            # Match the req.mamba_pool_idx=None clear above so the next
            # alloc() doesn't see a stale ping-pong reference on the req
            # and skip allocation (which would silently reuse a freed
            # tensor on the req side while the new pool slot leaks).
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_allocator.clear()
        # The int8 checkpoint pool holds radix-cached states in its own slots; a
        # flush/reset drops the radix tree, so its slots must be released too,
        # otherwise the (now unreferenced) slots leak and break the int8-pool
        # invariant (int8_available + radix_cached != int8_total).
        if self.mamba_ckpt_pool is not None:
            self.mamba_ckpt_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


@dataclass
class KVWriteLoc:
    """Write target(s) for ``KVCache.set_kv_buffer``.

    ``loc`` is the full-pool write location; ``swa_loc`` is the pre-translated
    full->SWA location for hybrid SWA pools (``None`` otherwise). Bundling them
    lets a backend issue one ``set_kv_buffer`` call regardless of pool type.
    """

    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None


def unwrap_write_loc(loc_info):
    """Return ``(loc, swa_loc)`` from a ``KVWriteLoc`` or a bare loc tensor."""
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc
    return loc_info, None


class KVCache(abc.ABC):
    """KV 缓存池抽象基类。

    该类统一描述物理 KV cache 的基本能力：按层访问 key/value 缓冲区、
    写入新生成 token 的 KV、在需要时进行 CPU/GPU 拷贝，以及维护缓存
    分配后的显存统计。具体的缓存布局和量化格式由子类实现。
    """

    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        """初始化 KV 缓存池的通用元数据。

        Args:
            size: 可用于实际 token 的缓存槽数量。
            page_size: 分页 KV cache 的页大小，非分页布局通常为 1。
            dtype: 模型计算或反量化后使用的 KV 数据类型。
            layer_num: 当前缓存池覆盖的层数。
            device: 缓存张量所在设备。
            enable_memory_saver: 是否启用显存节省适配器。
            start_layer: 当前缓存池覆盖的起始 layer id。
            end_layer: 当前缓存池覆盖的结束 layer id。
        """
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """完成 KV cache 分配后的统一日志记录和显存统计。

        支持子类返回 ``(K, V)`` 两个独立大小，也支持返回合并后的单个
        KV 大小；最终会把 GB 级显存用量写入 ``self.mem_usage``。
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """获取指定层的 key cache 物理缓冲区。

        子类需要根据自身的层偏移、布局和量化格式返回可供 attention
        kernel 读取的 key 张量。
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """获取指定层的 value cache 物理缓冲区。

        对于只存储合并 KV 或 latent KV 的实现，子类可以返回与自身布局
        兼容的 value 视图或占位结果。
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """同时获取指定层的 key/value cache 缓冲区。

        返回值通常是 ``(key_buffer, value_buffer)``，用于需要一次拿到
        K/V 两个物理缓存的 attention 或迁移逻辑。
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """将一批 token 的 KV 写入指定层的缓存位置。

        Args:
            layer: 当前写入对应的 attention 层对象。
            loc: 目标 token 槽位，部分子类也可接受 ``KVWriteLoc``。
            cache_k: 待写入的 key cache。
            cache_v: 待写入的 value cache。
        """
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        """注册按层传输计数器，用于访问缓存前等待对应层传输完成。"""
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        """把指定缓存槽位拷贝到 CPU，供 CPU offloading 或回收恢复使用。"""
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        """把先前保存到 CPU 的 KV cache 数据加载回指定缓存槽位。"""
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        """返回自定义显存池；未启用时返回 ``None``。"""
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):
    """多头注意力（MHA/GQA/MQA）使用的 token 到 KV 物理缓存池。

    该缓存池按层维护独立的 K/V 张量，默认布局为
    ``[slot, head_num, head_dim]``。在 ROCm AITER 后端可选用
    ``vectorized_5d`` 物理布局以匹配后端 kernel。该类负责缓存张量
    分配、按 token 槽位写入 KV、按层读取 K/V、CPU offloading 拷贝，
    以及接收 token 后的 KV cache 位置搬迁。
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        """初始化 MHA KV 缓存池并分配底层 K/V 缓冲区。

        Args:
            size: 可用 token 缓存槽数量。
            page_size: 分页 KV cache 的页大小。
            dtype: attention 计算看到的 KV 数据类型。
            head_num: key/value head 数量。
            head_dim: key head 维度。
            layer_num: 当前缓存池覆盖的层数。
            device: 缓存张量所在设备。
            enable_memory_saver: 是否在分配缓存时启用显存节省区域。
            v_head_dim: value head 维度，未指定时与 ``head_dim`` 相同。
            swa_head_num: SWA 子池覆盖时使用的 head 数量。
            swa_head_dim: SWA 子池覆盖时使用的 key head 维度。
            swa_v_head_dim: SWA 子池覆盖时使用的 value head 维度。
            start_layer: 当前池覆盖的起始 layer id。
            end_layer: 当前池覆盖的结束 layer id。
            enable_alt_stream: 是否允许写 K/V 时使用辅助 stream。
            enable_kv_cache_copy: 是否初始化 Triton KV 搬迁 kernel 配置。
        """
        # 先初始化 KVCache 基类中的通用容量、分页、dtype、设备和层范围元数据。
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # SWA 子池可覆盖 head 数；未覆盖时使用普通 MHA/GQA 的 head 数。
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        # SWA 子池可覆盖 key head 维度；未覆盖时使用传入的 key head 维度。
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        # value head 维度优先使用 SWA 覆盖值，其次使用显式 v_head_dim，最后回退到 key head_dim。
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        # 默认使用传统 NHD 物理布局，即每层 K/V 都是 [slot, head, dim]。
        self.kv_cache_layout = "nhd"
        # 只有 ROCm AITER 后端支持通过环境变量切换到 vectorized_5d 物理布局。
        if _use_aiter:
            # 读取 AITER KV cache 布局配置，并统一转成小写便于校验。
            layout = envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower()
            # 只允许传统 NHD 或 AITER 专用的 5D shuffle 布局。
            if layout not in ("nhd", "vectorized_5d"):
                raise ValueError(
                    f"Unsupported SGLANG_AITER_KV_CACHE_LAYOUT={layout!r}; "
                    "expected 'nhd' or 'vectorized_5d'."
                )
            # 保存最终选中的物理布局，后续 buffer 分配和写入逻辑都会读取它。
            self.kv_cache_layout = layout
            # vectorized_5d 需要额外计算内部向量化宽度并验证维度可整除。
            if layout == "vectorized_5d":
                # X 是 5D shuffle 布局的最内层向量宽度，由实际存储 dtype 的字节数决定。
                self._kv_vector_x = 16 // self.store_dtype.itemsize
                # 总 slot 数必须正好按 page_size 切成完整 page。
                assert (self.size + self.page_size) % self.page_size == 0
                # page_size 必须能被 X 整除，否则 V 的 page // X 维度无法成立。
                assert self.page_size % self._kv_vector_x == 0, (
                    f"page_size={self.page_size} must be divisible by "
                    f"X={self._kv_vector_x} for vectorized_5d layout"
                )
                # key head 维度必须能被 X 整除，才能拆成 D_k // X 和 X 两维。
                assert self.head_dim % self._kv_vector_x == 0
                # value head 维度同样必须能被 X 整除，才能匹配 V 的 5D 布局。
                assert self.v_head_dim % self._kv_vector_x == 0

        # 按最终布局分配每层 K/V 缓冲区，并构造指针表和 stride 表。
        self._create_buffers()

        # 获取当前设备模块，例如 torch.cuda 或对应平台的 device module。
        self.device_module = torch.get_device_module(self.device)

        # CUDA 或类 CUDA 平台支持辅助 stream，用于在小 batch 下重叠 K/V 写入。
        _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
        # 如果平台和配置都允许，则创建辅助 stream；否则保持 None。
        self.alt_stream = (
            self.device_module.Stream()
            if _use_alt_stream and enable_alt_stream
            else None
        )

        # 需要 KV cache 搬迁时，初始化并预热 Triton copy kernel。
        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            # 不启用搬迁能力时，用 None 标记后续 move_kv_cache 不应走 Triton 路径。
            self._kv_copy_config = None

        # 记录 KV cache 分配后的显存占用日志。
        self._finalize_allocation_log(size)

        # store_cache JIT kernel 需要一行 K cache 的展平元素数。
        self.row_dim = self.head_num * self.head_dim
        # 标记 K/V 最后一维是否相同，便于写入 kernel 选择快速路径。
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        """初始化跨层 KV cache 搬迁 kernel 的分块参数并预热 Triton kernel。"""
        # 零层池没有任何 K/V buffer，例如全 SWA 模型里的 full attention 子池。
        if self.layer_num == 0:
            # 没有 buffer 时不需要也不能初始化 copy kernel 配置。
            self._kv_copy_config = None
            return

        # 根据单个 token 的 KV 行字节数，启发式选择 Triton copy tile 大小。
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        # data_strides[0] 表示单个 K/V token 行占用的字节数。
        stride_bytes = int(self.data_strides[0].item())
        # 大 stride 使用更大的 tile，减少 tile 数量。
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        # 中等 stride 使用中等 tile。
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        # 小 stride 使用较小 tile，避免过度并行带来的浪费。
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # 限制单次 kernel 专门化的 loc 上界，避免生成过大的 Triton specialization。
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        # 保存 copy kernel 的静态配置，后续 move_kv_cache 会复用。
        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        # 构造预热用的 dummy 槽位，避免首次真实搬迁时触发编译开销。
        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        # grid 第一维覆盖所有 K/V buffer 指针，第二维覆盖单行字节 tile。
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        # 预热 Triton tiled copy kernel。
        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        """按当前物理布局为每一层创建 key/value 缓冲区和指针表。"""
        # 所有 KV cache 张量都登记到 memory saver 的 KV cache 区域。
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # 如果启用了自定义 CUDA mempool，则在该 mempool 中完成底层张量分配。
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # AITER vectorized_5d 布局按 page/block 组织物理存储。
                if self.kv_cache_layout == "vectorized_5d":
                    # 总 slot 包含真实 token slot 和一个 page_size 大小的 padding 区。
                    total_slots = self.size + self.page_size
                    # 5D 布局的第一维是 page/block 数。
                    num_blocks = total_slots // self.page_size
                    # x 是 16 字节向量内能容纳的元素数。
                    x = self._kv_vector_x
                    # K 的 5D 形状为 (num_blocks, H, D_k // X, page, X)。
                    self.k_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.head_dim // x,
                                self.page_size,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    # V 的 5D 形状为 (num_blocks, H, page // X, D_v, X)。
                    self.v_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.page_size // x,
                                self.v_head_dim,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    # 传统 NHD 布局下，每层 K 的形状是 [slot, head_num, head_dim]。
                    # 额外 page_size 个 padding slot 用于承接 padded/dummy token 写入。
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    # 传统 NHD 布局下，每层 V 的 value head 维度可以不同于 key head 维度。
                    self.v_buffer = [
                        torch.zeros(
                            (
                                self.size + self.page_size,
                                self.head_num,
                                self.v_head_dim,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        # 保存每层 K buffer 的原始 data_ptr，供 Triton kernel 间接寻址。
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        # 保存每层 V buffer 的原始 data_ptr，供 Triton kernel 间接寻址。
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        # 拼接 K/V 指针表，copy kernel 会把它当成统一 buffer 列表。
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        # 记录每个 buffer 中单个 token/page 行对应的字节跨度。
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        """释放当前池持有的 key/value 缓冲区引用。"""
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        """统计所有层 key/value 缓冲区分别占用的字节数。"""
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        """返回用于 disagg 场景的连续 KV 缓冲区指针、总字节数和单页字节数。"""
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices, mamba_indices=None):
        """按槽位把所有层的 K/V 缓冲区分块拷贝到 CPU。"""
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        """把 ``get_cpu_copy`` 保存的 K/V 分块加载回对应缓存槽位。"""
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        current_platform.synchronize()

    def _get_key_buffer(self, layer_id: int):
        """获取指定层 key 缓冲区的内部视图，不等待层级传输计数器。"""
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        """等待可选的层级传输完成后，返回指定层 key 缓冲区。"""
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        """获取指定层 value 缓冲区的内部视图，不等待层级传输计数器。"""
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        """等待可选的层级传输完成后，返回指定层 value 缓冲区。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        """同时返回指定层的 key 和 value 缓冲区。"""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """把一批 token 的 key/value 写入指定层的物理 KV cache。

        写入前会做越界检测、必要的数据类型转换和量化存储视图转换；
        对 ``vectorized_5d`` 布局会调用专用 reshape/cache kernel，
        其他布局走通用的 KV 写入实现。
        """
        loc, _ = unwrap_write_loc(loc_info)
        # Catch stale slot ids here instead of as illegal-addr / silent KV
        # corruption in the store_kvcache write (gated on SGLANG_ENABLE_ASYNC_ASSERT).
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.kv_cache_layout == "vectorized_5d":
            # Late-import to keep the NHD path import-clean.
            from sglang.srt.layers.attention.utils import (
                launch_reshape_and_cache_shuffle_5d,
            )

            # The writer kernel uses key.stride(0) directly as the source
            # token stride; head/dim are assumed contiguous within each
            # token (stride(1)=head_size, stride(2)=1). Both hold for K/V
            # produced by QKV split + RoPE in upstream attention even when
            # the outer per-token stride is non-canonical, so we skip the
            # protective .contiguous() copies that would otherwise fire
            # large per-layer elementwise kernels.
            launch_reshape_and_cache_shuffle_5d(
                cache_k,
                cache_v,
                self.k_buffer[layer_id - self.start_layer],
                self.v_buffer[layer_id - self.start_layer],
                loc,
            )
            return

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            # size + page_size = real slots + the reserved padding slot (padded /
            # dummy tokens write there); valid index range is [0, size + page_size).
            size_limit=self.size + self.page_size,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """只提交每个序列前缀中有效 token 的 KV 到缓存。

        ``loc_2d`` 给出二维目标槽位，``commit_lens`` 给出每行实际有效
        长度。CUDA/HIP 上使用按有效前缀写入的 Triton kernel，其他平台
        先展开有效位置后复用 ``set_kv_buffer``。
        """
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
            raise ValueError(
                "commit_lens must match loc_2d batch size: "
                f"{tuple(commit_lens.shape)=} {tuple(loc_2d.shape)=}."
            )

        num_rows = int(loc_2d.numel())
        if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
            raise ValueError(
                "dense KV rows must match loc_2d size: "
                f"{tuple(cache_k.shape)=} {tuple(cache_v.shape)=} {tuple(loc_2d.shape)=}."
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)
        else:
            cache_k = cache_k.contiguous()
            cache_v = cache_v.contiguous()

        if loc_2d.device != self.k_buffer[0].device:
            loc_2d = loc_2d.to(device=self.k_buffer[0].device, non_blocking=True)
        if commit_lens.device != self.k_buffer[0].device:
            commit_lens = commit_lens.to(
                device=self.k_buffer[0].device, non_blocking=True
            )
        if loc_2d.dtype != torch.int64:
            loc_2d = loc_2d.to(torch.int64)
        if commit_lens.dtype != torch.int32:
            commit_lens = commit_lens.to(torch.int32)

        if not (_is_cuda or _is_hip):
            row_offsets = torch.arange(loc_2d.shape[1], device=loc_2d.device)
            valid_mask = row_offsets[None, :] < commit_lens.to(torch.int64)[:, None]
            valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                return
            self.set_kv_buffer(
                layer,
                loc_2d.reshape(-1).index_select(0, valid_idx),
                cache_k.index_select(0, valid_idx),
                cache_v.index_select(0, valid_idx),
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
            return

        _set_kv_buffer_prefix_valid_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc_2d,
            commit_lens,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """把所有层的 K/V cache 从源槽位搬迁到目标槽位。

        该方法用于接受 speculative token 或压缩缓存位置等场景。小批量
        和大批量搬迁都会使用预先初始化的 tiled copy kernel；如环境变量
        要求原生路径，则回退到逐层张量索引复制。
        """
        # Zero-layer pool (e.g. all-SWA model's full sub-pool) has no buffers.
        if self.layer_num == 0:
            return

        # Catch stale indices here instead of as illegal-addr or silent KV corruption.
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class NoOpMHATokenToKVPool(MHATokenToKVPool):
    """跳过真实 K/V 物理缓存分配的 no-op MHA KV 缓存池。

    该类用于只做 prefill 且 attention 后端不会读写 paged KV cache 的场景。
    它保留调度器看到的逻辑容量，但每层只分配很小的占位 K/V 张量，
    用来满足少量仍会引用 ``k_buffer`` / ``v_buffer`` 的公共代码路径。
    真实的 KV 写入、读取和搬迁都不应发生在该池上；如果误调用写入
    路径，会主动抛错暴露配置问题。

    KV cache pool that skips physical K/V buffer allocation.

    Used in embedding-mode prefill-only workloads with the FA
    fa_skip_kv_cache path, where no layer reads or writes KV cache because
    attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths
    such as scoring/MIS may benefit from the same idea later, but some still
    stage K/V through paged cache today.

    This class keeps the scheduler's view of pool capacity (self.size is
    honored for admission) but allocates only (page_size, head_num, head_dim)
    placeholder tensors per layer to satisfy any code paths that dereference
    the buffers.

    Callers MUST ensure no real set_kv_buffer/get_*_buffer calls happen against
    this pool; those paths raise loudly so misuse is visible.
    """

    def _create_buffers(self):
        """创建每层最小占位 K/V 缓冲区和对应的指针/stride 表。

        占位张量仅用于让后续公共逻辑中的引用、view 和指针表构造保持
        可用，不承载真实 token 的 KV cache 数据。
        """
        # Allocate minimal placeholder buffers. They exist purely so that code
        # paths holding `k_buffer` / `v_buffer` references (pointer tables,
        # layer-transfer counters, stride arithmetic) keep working without
        # None-guards scattered across the codebase. Shape is
        # [page_size, head_num, head_dim] per layer so that the unconditional
        # `key_cache.view(-1, page_size, head_num, head_dim)` in the FA backend
        # at the top of forward_extend succeeds regardless of --page-size.
        # Total footprint is still on the order of KB vs GBs for a real pool.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """记录 no-op 池的逻辑容量和占位张量大小，并把显存统计置零。"""
        self.mem_usage = 0.0
        placeholder_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * max(self.head_dim, self.v_head_dim)
            * self.store_dtype.itemsize
        )
        logger.info(
            f"KV Cache skipped (no-op pool). Logical #tokens: {num_tokens}, "
            f"physical K/V size: ~{placeholder_bytes / 1024:.1f} KB placeholder"
        )

    def get_kv_size_bytes(self):
        """返回零大小，避免调度和日志把占位张量当成真实 KV cache 计费。"""
        # Report zero so downstream memory accounting matches reality.
        return (0, 0)

    def set_kv_buffer(self, *args, **kwargs):
        """禁止真实 KV 写入；no-op 池只允许在跳过 KV cache 的路径中使用。"""
        raise RuntimeError(
            "NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "
            "valid in prefill-only modes (e.g. --is-embedding, scoring) with "
            "the FA backend's fa_skip_kv_cache path active; the attention "
            "backend must never write to it. Check that the workload truly "
            "performs no decode and that the FA backend's fa_skip_kv_cache "
            "preconditions are met."
        )

    def get_key_buffer(self, layer_id: int):
        """返回指定层的 key 占位缓冲区，供跳过 KV cache 前的公共路径引用。"""
        # Return the placeholder. The FA backend reads this before taking the
        # fa_skip_kv_cache branch (which does not use it); the placeholder shape
        # is (page_size, head_num, head_dim) so downstream .view() calls succeed.
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        """返回指定层的 value 占位缓冲区。"""
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        """同时返回指定层的 key/value 占位缓冲区。"""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """跳过 KV 搬迁；该池没有真实 KV cache 需要移动。"""
        # no-op; embedding mode has no KV cache to move
        return


class MHATokenToKVPoolFP4(MHATokenToKVPool):
    """使用块级 FP4 量化存储的 MHA KV 缓存池。

    该类继承 MHA KV cache 的大部分调度和访问接口，但把 K/V 物理缓存
    存成 FP4 压缩字节，并为每层分别维护 K/V 的 scale buffer。写入时
    会对输入 K/V 做块级 FP4 量化，读取内部 buffer 时会按 scale 反量化
    成父类接口期望的 dtype。
    """

    def _create_buffers(self):
        """创建 FP4 压缩 K/V 缓冲区以及对应的 K/V scale 缓冲区。"""
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        """释放 FP4 K/V 缓冲区及其 scale 缓冲区引用。"""
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        """获取指定层 key 缓冲区；FP4 存储时先按 scale 反量化。"""
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        """获取指定层 value 缓冲区；FP4 存储时先按 scale 反量化。"""
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_v_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """把一批 token 的 K/V 量化为 FP4 后写入物理缓存。

        写入前会先做槽位越界检测和可选的输入 scale 还原，然后使用
        ``BlockFP4KVQuantizeUtil`` 生成 FP4 数据及 scale buffer。CUDA
        graph capture 且存在辅助 stream 时，会尝试重叠 K/V 两路写入。
        """
        loc, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-FP4)")
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HybridLinearKVPool(KVCache):
    """混合 full attention 与 linear/Mamba attention 的 KV/状态缓存池。

    该类本身不直接保存所有物理缓存，而是把 full attention 层的 KV cache
    委托给 ``full_kv_pool``，把 linear/Mamba 层的状态缓存委托给
    ``mamba_pool``。由于 full attention 层在混合模型中通常不是连续层，
    本类还负责把模型全局 layer id 映射到 ``full_kv_pool`` 内部的连续
    layer id。
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        start_layer: Optional[int] = None,
    ):
        """初始化混合 KV/状态缓存池。

        Args:
            size: 可用 token 缓存槽数量。
            dtype: full attention KV cache 的逻辑 dtype。
            page_size: 分页 KV cache 的页大小。
            head_num: MHA/GQA full attention 的 KV head 数量。
            head_dim: MHA/GQA full attention 的 head 维度。
            full_attention_layer_ids: 使用 full attention 的全局 layer id 列表。
            enable_kvcache_transpose: 是否启用转置 KV cache；当前不支持。
            device: 缓存张量所在设备。
            mamba_pool: linear/Mamba 层使用的状态缓存池。
            enable_memory_saver: 是否启用显存节省区域。
            enable_kv_cache_copy: 是否为 full KV pool 初始化 KV 搬迁能力。
            use_mla: full attention 子池是否使用 MLA KV 布局。
            kv_lora_rank: MLA latent KV 维度，仅 ``use_mla`` 时使用。
            qk_rope_head_dim: MLA RoPE key 维度，仅 ``use_mla`` 时使用。
            start_layer: 当前 worker/pipeline stage 覆盖的起始 layer id。
        """
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        if not use_mla:
            TokenToKVPoolClass = MHATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
                enable_kv_cache_copy=enable_kv_cache_copy,
            )
        else:
            TokenToKVPoolClass = MLATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mla_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        """返回 full attention 子池的 KV cache 字节数。"""
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        """返回 full attention KV cache 的连续缓冲区信息。"""
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        """返回 linear/Mamba 状态缓存的连续缓冲区信息。"""
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """返回每个 Mamba 状态张量可切分维度的大小。"""
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        """返回 full attention 子池使用的自定义显存池。"""
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        """把全局 full attention layer id 转换为子池内部连续 layer id。"""
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        """注册层级传输计数器，并由混合池统一执行等待逻辑。"""
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        """等待指定全局 layer id 的层级传输完成。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int):
        """获取指定 full attention 层的 key/合并 KV 缓冲区。"""
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        """获取指定 full attention 层的 value 缓冲区或 value 视图。"""
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        """同时获取指定 full attention 层的 K/V 缓冲区。"""
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):
        """临时把 ``layer.layer_id`` 改为 full KV 子池内部 layer id。"""
        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        """把 full attention 层的一批 K/V 写入对应的 full KV 子池。

        MHA/GQA 路径直接传入转换后的内部 layer id；MLA 路径需要临时
        patch ``layer.layer_id``，因为底层 MLA pool 会从 layer 对象读取 id。
        """
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """委托 full attention 子池搬迁 KV cache 槽位。"""
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        """把 full KV cache 和可选 Mamba 状态一起拷贝到 CPU。"""
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices)
        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(mamba_indices)
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        """把 CPU 上保存的 full KV cache 和可选 Mamba 状态加载回设备。"""
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(mamba_cpu, mamba_indices)

    def get_v_head_dim(self):
        """返回 full attention 子池 value head 的最后一维大小。"""
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        """把 MLA full attention 层的 ``k_nope`` 和 ``k_rope`` 写入子池。"""
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        """从 MLA full attention 子池读取指定槽位的 ``k_nope`` 和 ``k_rope``。"""
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    """MLA attention 使用的 token 到 KV 物理缓存池。

    MLA 不像 MHA 那样分别保存完整的 key/value head，而是把 latent KV
    部分和 RoPE key 部分合并存入单个 ``kv_buffer``。该类负责按层分配
    合并 KV 缓冲区、写入/读取 MLA 所需的 ``k_nope`` 与 ``k_rope``、
    支持 DSA/FP8 特殊存储分支，以及提供 KV 搬迁和 CPU offloading 能力。
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_dsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        """初始化 MLA KV 缓存池并创建每层合并 KV 缓冲区。

        Args:
            size: 可用 token 缓存槽数量。
            page_size: 分页 KV cache 的页大小。
            dtype: KV cache 的逻辑 dtype。
            kv_lora_rank: MLA latent KV 部分的维度。
            qk_rope_head_dim: RoPE key 部分的维度。
            layer_num: 当前缓存池覆盖的层数。
            device: 缓存张量所在设备。
            enable_memory_saver: 是否启用显存节省区域。
            start_layer: 当前池覆盖的起始 layer id。
            end_layer: 当前池覆盖的结束 layer id。
            use_dsa: 是否用于 DSA 模型的 KV cache。
            override_kv_cache_dim: DSA 等特殊路径指定的实际 KV 缓存维度。
        """
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_dsa = use_dsa
        self.dsa_kv_cache_store_fp8 = (
            use_dsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with dsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_dsa:
            # DSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        """为每一层创建合并 MLA KV 缓冲区。"""
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        """释放合并 MLA KV 缓冲区引用。"""
        del self.kv_buffer

    def get_kv_size_bytes(self):
        """统计所有层合并 KV 缓冲区占用的总字节数。"""
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        """返回 disagg 场景所需的合并 KV 缓冲区指针、总字节数和单页字节数。"""
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        """返回指定层的合并 MLA KV 缓冲区，供 attention 后端作为 key cache 使用。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        """返回指定层 latent KV 部分的视图，供需要 value cache 的路径使用。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        """同时返回指定层的合并 KV 缓冲区和 latent KV 视图。"""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """把已经拼好的 MLA KV 张量写入指定层缓存。

        该通用路径主要写入 ``cache_k``，因为 MLA 的 value 信息已经包含
        在合并 KV/latent KV 表示中；DSA FP8 特殊存储路径不允许走这里。
        """
        loc, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        """把 MLA 的 latent key 和 RoPE key 写入合并 KV cache。

        普通路径会把 ``cache_k_nope`` 和 ``cache_k_rope`` 通过 Triton
        kernel 写入同一个 ``kv_buffer``；DSA/FP8 路径会按各自布局先做
        量化或转换再写入。
        """
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id

        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:
            # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
            # Fuse BF16/FP16 -> FP8 cast with paged KV write.
            set_mla_kv_buffer_triton_fp8_quant(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.dsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        """从合并 KV cache 中读取指定槽位的 ``k_nope`` 和 ``k_rope``。"""
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """按层把合并 MLA KV cache 从源槽位搬迁到目标槽位。"""
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for kv_cache in self.kv_buffer:
            kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]

    def get_cpu_copy(self, indices, mamba_indices=None):
        """按槽位把所有层的合并 MLA KV 缓冲区分块拷贝到 CPU。"""
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        """把 ``get_cpu_copy`` 保存的 MLA KV 分块加载回对应缓存槽位。"""
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):
    """使用块级 FP4 量化存储的 MLA KV 缓存池。

    该类在 ``MLATokenToKVPool`` 的合并 MLA KV 布局基础上，将物理
    ``kv_buffer`` 存成 FP4 压缩字节，并额外维护 ``kv_scale_buffer``。
    写入时会对 latent KV/RoPE KV 做块级 FP4 量化，读取 key buffer 时
    会根据 scale 反量化回逻辑 dtype。
    """

    def _create_buffers(self):
        """创建 FP4 压缩的合并 MLA KV 缓冲区和对应 scale 缓冲区。"""
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        """释放 FP4 合并 KV 缓冲区及其 scale 缓冲区引用。"""
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        """返回指定层 key buffer；FP4 存储时先按 scale 反量化。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """把已经合并好的 MLA KV 张量量化为 FP4 后写入缓存。

        该路径用于调用方已经提供完整 ``cache_k`` 的情况，会同时写入
        FP4 数据和对应的 scale buffer；DSA FP8 特殊路径不允许走这里。
        """
        # loc_info may be a KVWriteLoc; MLA pools have no SWA target.
        loc, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA-FP4)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_fp4, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(
                cache_k
            )

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        """把 MLA 的 ``k_nope`` 与 ``k_rope`` 分别量化为 FP4 后写入缓存。

        普通 FP4 路径会分别生成 latent 部分和 RoPE 部分的 FP4 数据及
        scale，再通过 Triton kernel 写入合并 ``kv_buffer`` 和
        ``kv_scale_buffer``。DSA FP8 路径则沿用专门的 FP8 存储格式。
        """
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA-FP4)"
        )
        layer_id = layer.layer_id

        if self.dsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    BlockFP4KVQuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class DSATokenToKVPool(MLATokenToKVPool):
    """DSA attention 使用的 token 到 KV 物理缓存池。

    该类在 ``MLATokenToKVPool`` 的合并 latent KV cache 基础上，额外维护
    DSA indexer 使用的 ``index_k_with_scale_buffer``。这个 buffer 以页为
    单位保存 index K 的 FP8 数据及其 scale，必须在 KV 搬迁、CPU offload
    和恢复时与主 KV cache 保持同步，否则 DSA attention 会读到错位的
    index/scale 数据。
    """

    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
    ):
        """初始化 DSA KV 缓存池并分配 index K + scale 页缓存。

        Args:
            size: 可用 token 缓存槽数量。
            page_size: 分页 KV cache 的页大小。
            kv_lora_rank: MLA latent KV 部分的维度。
            dtype: 主 KV cache 的逻辑 dtype。
            qk_rope_head_dim: RoPE key 部分的维度。
            layer_num: 当前缓存池覆盖的层数。
            device: 缓存张量所在设备。
            index_head_dim: DSA index K 的 head 维度，当前要求为 128。
            enable_memory_saver: 是否启用显存节省区域。
            kv_cache_dim: 主 KV cache 的实际存储维度。
            start_layer: 当前池覆盖的起始 layer id。
            end_layer: 当前池覆盖的结束 layer id。
            index_buf_size: index K 页缓存的逻辑 token 容量，默认等于 ``size``。
        """
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        if index_buf_size is None:
            index_buf_size = size
        # num head == 1 and head dim == 128 for index_k in DSA
        assert index_head_dim == 128

        if _is_hip:
            if aiter_can_use_preshuffle_paged_mqa():
                assert (
                    self.page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {self.page_size}"
            else:
                assert (
                    self.page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {self.page_size}"
        else:
            assert self.page_size == 64
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (index_buf_size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
        self._finalize_allocation_log(size)

    def _clear_buffers(self):
        """释放主 KV 缓冲区和 DSA index K + scale 缓冲区引用。"""
        del self.kv_buffer
        del self.index_k_with_scale_buffer

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """同步搬迁主 latent KV cache 和 DSA index K + scale cache。"""
        super().move_kv_cache(tgt_loc, src_loc)

        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for index_k in self.index_k_with_scale_buffer:
            index_k[tgt_loc_flat] = index_k[src_loc_flat]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """返回指定层原始的 DSA index K + scale 页缓存。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        """从页缓存中连续读取指定层的 index K FP8 数据。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        """从页缓存中连续读取指定层的 index K scale 数据。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """一次性读取指定层的 index K FP8 数据和 scale 数据。

        该融合 Triton 路径比分别调用 ``get_index_k_continuous`` 和
        ``get_index_k_scale_continuous`` 更高效。

        Returns:
            ``(k_fp8, k_scale)``，其中 ``k_fp8`` 形状约为
            ``(seq_len_sum, index_head_dim)``，``k_scale`` 保存对应 scale。
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        """把 index K FP8 数据和 scale 写入指定页缓存位置。"""
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        """把主 KV cache 和 DSA index K + scale cache 一起拷贝到 CPU。"""
        # DSA keeps a page-indexed index_k_with_scale_buffer alongside kv_buffer.
        # Retract frees the slots/pages and they get reused by other reqs'
        # set_index_k_scale_buffer, so we must offload it here too -- otherwise
        # resume restores kv_buffer but leaves foreign index/scale in place and
        # DSA attention reads garbage at those token positions.
        kv_cache_cpu = super().get_cpu_copy(indices, mamba_indices=mamba_indices)

        page_indices = indices[:: self.page_size] // self.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            index_k_cpu.append([])
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.index_k_with_scale_buffer[layer_id][
                    chunk_page_indices
                ].to("cpu", non_blocking=True)
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()

        return {"kv": kv_cache_cpu, "index_k": index_k_cpu}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        """把 CPU 上保存的主 KV cache 和 DSA index K + scale cache 加载回设备。"""
        super().load_cpu_copy(
            kv_cache_cpu_dict["kv"], indices, mamba_indices=mamba_indices
        )

        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(
                    self.index_k_with_scale_buffer[0].device, non_blocking=True
                )
                self.index_k_with_scale_buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()

    def get_state_buf_infos(self):
        """返回 DSA index K + scale 状态缓存的指针、总字节数和单页字节数。"""
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        """统计主 KV cache 和 DSA index K + scale cache 的总字节数。"""
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]
