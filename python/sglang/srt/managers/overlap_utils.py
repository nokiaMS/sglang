# 本文件实现了跨迭代重叠调度的工具类和函数。
# 核心是 FutureMap 类，它作为始终启用的、按请求池索引的中继器，
# 在前向计算写入值（通过 publish/stash）和下一迭代读取值
# （通过 resolve_forward_inputs / resolve_seq_lens_cpu）之间传递数据，
# 实现前向计算与采样之间的流水线重叠，从而提升吞吐量。
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    server_args: "ServerArgs",
    attn_backends: Sequence["AttentionBackend"],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    OR over per-backend needs_cpu_seq_lens; force True under TBO / piecewise CG
    (they read the CPU mirror outside the backend layer).
    """
    # 判断 FutureMap 是否需要发布 CPU 端的序列长度信息
    if server_args.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        # 启用双批次重叠时，必须在 CPU 端维护序列长度
        return True
    if not server_args.disable_piecewise_cuda_graph:
        # FIXME: support PCG without seq lens cpu value
        # 启用分段 CUDA Graph 时，也必须在 CPU 端维护序列长度
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    # 对每个后端取逻辑或：任意后端需要 CPU 序列长度则返回 True；
    # 未声明该标志的后端默认返回 True 以保持兼容旧路径
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
# Token 缓冲区消费跟踪：初始化为 -1，gather 时断言非负，读完后写回 -1。
# 用于捕获"未暂存就直接 gather"的 bug。CI 环境下启用，生产环境关闭。
_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch."""
    # 融合操作：断言所有值非负，并将缓冲区对应位置写回 -1（标记已消费）
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


@torch.compile(dynamic=True, disable=_is_npu)
def _gather_spec_extras(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: Optional[torch.Tensor],
):
    """Compiled gather of spec extras. `hidden_states_buf` is None when the
    build does not capture hidden states."""
    # 编译优化的投机采样额外数据 gather 操作，从缓冲区中按索引收集投机采样所需数据
    topk_p = topk_p_buf[indices]
    topk_index = topk_index_buf[indices]
    bonus_tokens = output_tokens_buf[indices]
    hidden_states = (
        hidden_states_buf[indices] if hidden_states_buf is not None else None
    )
    return topk_p, topk_index, bonus_tokens, hidden_states


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Two sources:

    - Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu).
    - Decode/spec_v2: gather from FutureMap (last iter's sampled token).
    """
    # 在前向计算入口处物化 input_ids，有两个来源：
    # 1. Prefill：从 pinned CPU 暂存区拷贝到 GPU
    # 2. Decode/spec_v2：从 FutureMap 中 gather 上一轮采样的 token
    if batch.prefill_input_ids_cpu is not None:
        # 将 prefill 的 input_ids 从 CPU 拷贝到 GPU
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            # 混合模式：同时有 prefill 和 decode 请求，从 FutureMap gather decode 的 token
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                # 调试断言：检查 gather 的 token 值非负，然后将缓冲区对应位置置为 -1
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            # 拼接 prefill 和 decode 的 input_ids
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            # 纯 prefill 模式
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        # 纯 decode 模式（非投机采样）：从 FutureMap 中 gather 上一轮的采样 token
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )

    # spec_v1 (non-overlap spec) doesn't relay extras; only spec_v2 does.
    # spec_v1（非重叠投机采样）不中继额外数据；只有 spec_v2 需要
    if batch.is_spec_v2:
        future_map._resolve_spec_extras(batch)


class FutureMap:
    """Always-on pool-indexed relay for cross-iter values. Forward writes via
    publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.
    """
    # FutureMap：始终启用的、按请求池索引的跨迭代值中继器。
    # 前向计算通过 publish/stash 写入值，下一迭代通过
    # resolve_forward_inputs / resolve_seq_lens_cpu 读取值。

    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
    ):
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        # 缓冲区按 req_pool_idx 索引；slot 0 镜像 KV 填充行，
        # 这样 CUDA Graph 填充批次（req_pool_idx == 0）不会产生副作用
        self.device = device
        self.spec_algo = spec_algo
        # Computed by decide_needs_cpu_seq_lens(); see that helper for the
        # full decision (per-backend flag + TBO / piecewise CG overrides).
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        # 调试模式下用 -1 初始化以检测未暂存就 gather 的 bug，否则用 empty 节省开销
        self.output_tokens_buf = (
            torch.full((self.req_pool_size,), -1, dtype=torch.int64, device=self.device)
            if _DEBUG_ASSERT
            else torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        )
        # 新序列长度缓冲区，用于在迭代间传递序列长度
        self.new_seq_lens_buf = torch.empty(
            (self.req_pool_size,), dtype=torch.int64, device=self.device
        )
        # Pinned host copy of new_seq_lens_buf + private stream for fwd-prepare
        # D2H pulls (gated only on publish, off the schedule stream). CUDA-only:
        # recovers occupancy lost to the WAR barrier (also CUDA-only); other
        # platforms have no barrier and use the plain .cpu() bootstrap path.
        # 固定（pinned）主机端副本 + 独立流用于前向准备的 D2H 拷贝。
        # 仅在 publish 时触发，不在调度流上执行。仅 CUDA 端使用：
        # 恢复因 WAR 屏障损失的占用率；其他平台无屏障，使用普通 .cpu() 引导路径。
        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            # 独立的 CUDA 流，用于异步 D2H 拷贝，避免阻塞调度流
            self.fwd_prepare_d2h_stream = torch.get_device_module(self.device).Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None
        if self.spec_algo.is_some():
            # 投机采样前向缓冲区是否已初始化的标志
            self._forward_buf_initialized = False

        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it
        # 延迟初始化的 CUDA 事件，仅 spec_v2 需要它来同步 publish 操作

    def _lazy_init_forward_buf(self, draft_input: EagleDraftInput):
        """懒初始化投机采样前向缓冲区，根据首个 draft_input 的形状分配缓冲区。"""
        self._forward_buf_initialized = True

        # 根据首个 draft_input 的张量形状分配缓冲区
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        self.topk_p_buf = torch.empty(
            (self.req_pool_size, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.req_pool_size, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        if spec_need_hidden_states():
            # 仅当构建需要隐藏状态时才分配 hidden_states 缓冲区
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        """从 FutureMap 缓冲区中收集投机采样所需的额外数据（topk_p、topk_index、bonus_tokens、hidden_states）。"""
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            # 仅 prefill 时 draft_input 为 None，当前不兼容混合模式
            return
        indices = draft_input.future_indices
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        hidden_states_buf = (
            self.hidden_states_buf if spec_need_hidden_states() else None
        )
        # 使用编译优化的 gather 函数一次性收集所有投机采样额外数据
        (
            draft_input.topk_p,
            draft_input.topk_index,
            draft_input.bonus_tokens,
            hidden_states,
        ) = _gather_spec_extras(
            indices,
            self.topk_p_buf,
            self.topk_index_buf,
            self.output_tokens_buf,
            hidden_states_buf,
        )
        if hidden_states is not None:
            draft_input.hidden_states = hidden_states
        if _DEBUG_ASSERT:
            # 调试断言：检查 bonus_tokens 非负并置无效标记
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        """将序列长度从 FutureMap 缓冲区解析到批次对象中，包括 GPU gather 和可选的 D2H 拷贝。"""
        # seq_lens_cpu may be needed on the host for kernel-launch prep (some backends).
        # Run this D2H on a standalone stream to avoid chain-blocking forward_n ->
        # prepare_{n+1}: a sync on the schedule stream would inherit its WAR barrier and
        # stall the host until forward_n ends.
        # 某些后端需要在主机端获取 seq_lens_cpu 用于 kernel 启动准备。
        # 在独立流上执行 D2H 拷贝，避免链式阻塞 forward_n -> prepare_{n+1}：
        # 在调度流上同步会继承 WAR 屏障并阻塞主机直到 forward_n 结束。
        fi = batch.spec_info.future_indices if batch.spec_info is not None else None
        if fi is None:
            return
        if self.publish_ready is not None:
            # 等待 publish 操作完成，确保缓冲区数据已写入
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                # AMD MI355 上的临时变通方案：Event.wait() 会导致 TPOT 退化
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        # 从缓冲区中 gather 新序列长度
        batch.seq_lens = self.new_seq_lens_buf[fi]

        if not self.needs_cpu_seq_lens:
            # GPU gather above is kept (SB.seq_lens must advance each verify);
            # skip the .cpu() D2H. Downstream takes the GPU-only path.
            # 保留上面的 GPU gather（验证时 SB.seq_lens 必须推进）；
            # 跳过 .cpu() 的 D2H 拷贝，下游走纯 GPU 路径
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            # 引导阶段或非 CUDA 平台：直接 .cpu() 同步拷贝
            batch.seq_lens_cpu = batch.seq_lens.cpu()  # bootstrap / non-CUDA
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            return

        # Mechanism: don't sync the schedule stream; gate a private stream on the
        # publish event and copy into the static pinned buffer.
        # 机制：不同步调度流；将私有流门控在 publish 事件上，
        # 然后拷贝到静态 pinned 缓冲区
        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            # 在独立流上异步执行 D2H 拷贝到 pinned 缓冲区
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        # 同步等待 D2H 拷贝完成
        self.fwd_prepare_d2h_stream.synchronize()

        # FIXME: fi == batch.req_pool_indices; unify future_indices and req_pool_indices.
        # 从 pinned 缓冲区中按 req_pool_indices 索引取出序列长度
        batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

    def publish(self, future_indices: torch.Tensor, new_seq_lens: torch.Tensor) -> None:
        """将新序列长度写入缓冲区，并记录 CUDA 事件以同步后续读取操作。"""
        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
            # 数据并行空闲，无数据需发布
        # 将新序列长度写入对应索引位置
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        # 仅 spec_v2 需要 CUDA 事件，用于门控私有流上的 seq_lens D2H 拷贝
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            # 记录当前 CUDA 流上的事件，标记 publish 完成
            self.publish_ready.record()

    def stash(
        self,
        future_indices: torch.Tensor,
        payload: Union[torch.Tensor, EagleDraftInput],
    ) -> None:
        """将上一轮采样的输出（token 或投机采样额外数据）暂存到缓冲区，供下一迭代读取。"""
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            # 数据并行空闲：payload 为空桩，懒初始化时访问形状会 IndexError
            return
        # Dispatch by payload type, not spec_algo: spec_v1 (non-overlap spec)
        # also passes a token Tensor here.
        # FIXME(lsyin): unify this relay path with a dataclass instead of the
        # Tensor / EagleDraftInput type switch.
        # 根据载荷类型分发，而非 spec_algo：spec_v1（非重叠投机）也传入 token 张量
        if isinstance(payload, torch.Tensor):
            # 普通张量载荷：直接暂存采样的 token
            self.output_tokens_buf[indices] = payload.to(torch.int64)
            return

        # 投机采样载荷：暂存所有投机采样额外数据
        draft_input: EagleDraftInput = payload
        if not self._forward_buf_initialized:
            # 首次使用投机采样时懒初始化缓冲区
            self._lazy_init_forward_buf(draft_input)
        # 暂存 bonus_tokens、topk_p、topk_index 到对应缓冲区
        self.output_tokens_buf[indices] = draft_input.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )
        self.topk_p_buf[indices] = draft_input.topk_p.to(self.topk_p_buf.dtype)
        self.topk_index_buf[indices] = draft_input.topk_index.to(
            self.topk_index_buf.dtype
        )
        if spec_need_hidden_states():
            # 暂存 hidden_states（仅当构建需要时）
            self.hidden_states_buf[indices] = draft_input.hidden_states.to(
                self.hidden_states_buf.dtype
            )
