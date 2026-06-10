# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# 分段 CUDA Graph 运行器：使用 CUDA Graph 和 torch.compile 来加速模型的 prefill（扩展）阶段的前向传播。
# 与传统的整图 CUDA Graph 不同，分段式（piecewise）方法将模型拆分为多个段，
# 每段单独编译和捕获 CUDA Graph，从而支持动态形状和更灵活的编译优化。
# 主要功能包括：
#   - 对不同 token 数量的 prefill 请求进行 CUDA Graph 捕获与回放
#   - 支持 torch.compile (inductor) 和 eager 两种编译模式
#   - 支持多模态、Mamba 跟踪、LoRA、DP 注意力等特性
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import bisect
import gc
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import tqdm

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    enable_piecewise_cuda_graph_compile,
    set_forward_context,
    set_pcg_capture_stream,
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.utils import (
    get_available_gpu_memory,
    is_musa,
    is_npu,
    log_info_on_rank0,
    require_gathered_buffer,
)

# Suppress Dynamo warning about tracing through lru_cache-wrapped functions (e.g., is_arch_support_pdl).
warnings.filterwarnings("ignore", message=".*lru_cache.*", module="torch._dynamo")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_is_musa = is_musa()


# Prefill 阶段的输入缓冲区数据结构，保存 CUDA Graph 捕获所需的静态输入张量
@dataclass
class PrefillInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    out_cache_loc: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    mamba_track_mask: Optional[torch.Tensor]
    mamba_track_seqlens: Optional[torch.Tensor]
    positions: torch.Tensor
    input_embeds: Optional[torch.Tensor]
    mrope_positions: Optional[torch.Tensor]


# 上下文管理器：在 CUDA Graph 捕获期间优化垃圾回收行为
# 先执行一次 GC 清理，然后根据配置决定是否冻结 GC，避免捕获期间对象被回收导致问题
@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()  # 先执行一次垃圾回收，清理无用对象
    should_freeze = not enable_cudagraph_gc  # 如果捕获期间禁用 GC，则需要冻结
    if should_freeze:
        gc.freeze()  # 冻结所有当前对象，使其不会被后续 GC 回收
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()  # 解冻对象，恢复正常的 GC 行为


# 递归地将模型中的 MultiPlatformOp 子模块切换到 torch.compile 模式或恢复原模式
def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()  # 退出 torch.compile 模式
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)  # 进入 torch.compile 模式
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)  # 递归处理子模块


# 上下文管理器：在编译期间临时将模型切换到 torch.compile 兼容模式，结束后恢复
@contextmanager
def patch_model(model: torch.nn.Module, compiler: str):
    try:
        if compiler != "eager":
            _to_torch(model, reverse=False, num_tokens=16)  # 切换到编译模式
        yield model
    finally:
        _to_torch(model, reverse=True, num_tokens=16)  # 恢复原模式


# Reuse this memory pool across all cuda graph runners.
# 全局 CUDA Graph 内存池，所有 CUDA Graph 运行器共享，避免重复分配显存
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


# 设置 torch.compile 的配置，包括缓存大小限制和平台特定补丁
def set_torch_compile_config():
    import torch._dynamo.config

    # Resolve torch._dynamo.exc.FailOnRecompileLimitHit
    torch._dynamo.config.accumulated_cache_size_limit = 1024  # 增大编译缓存大小限制
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024  # 增大缓存大小限制，避免重编译失败

    if _is_musa:
        from sglang.srt.hardware_backend.musa.utils.patch_torch import (
            patch_fx_custom_device,
        )

        patch_fx_custom_device()  # 对摩尔线程(MUSA)平台应用自定义设备补丁


# 分段 CUDA Graph 运行器：将模型前向传播拆分为多个段，
# 分别使用 CUDA Graph 和 torch.compile 进行捕获和加速
class PiecewiseCudaGraphRunner:
    """A PiecewiseCudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    # 检查是否启用 Mamba 跟踪功能，需要同时满足：
    # 1. 启用了 Mamba 额外缓冲区
    # 2. 未禁用 radix cache
    # 3. 没有使用投机解码算法
    def is_mamba_track_enabled(self):
        return (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and not self.model_runner.server_args.disable_radix_cache
            and self.model_runner.spec_algorithm.is_none()
        )

    # 初始化分段 CUDA Graph 运行器，包括解析参数、分配输入缓冲区、编译模型和捕获 CUDA Graph
    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}  # 保存捕获的 CUDA Graph
        self.output_buffers = {}  # 保存各 graph 的输出缓冲区
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        self.attn_tp_size = get_attention_tp_size()  # 注意力张量并行大小
        self.attn_tp_rank = get_attention_tp_rank()  # 注意力张量并行秩

        set_torch_compile_config()  # 设置 torch.compile 配置

        assert (
            self.model_runner.server_args.piecewise_cuda_graph_tokens is not None
        ), "piecewise_cuda_graph_tokens is not set"
        assert self.model_runner.server_args.piecewise_cuda_graph_compiler in [
            "eager",
            "inductor",
        ], "By now, only eager and inductor are supported for piecewise cuda graph compiler."
        # 创建编译配置，包含分段 token 数、编译器和调试模式等参数
        self.compile_config = CompilationConfig(
            self.model_runner.server_args.piecewise_cuda_graph_tokens,
            self.model_runner.server_args.piecewise_cuda_graph_compiler,
            self.model_runner.server_args.enable_torch_compile_debug_mode,
        )
        # 如果 MoE 使用 DeepEP 或 Mooncake all-to-all 后端，需要添加分割算子
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.compile_config.add_split_op(
                "sglang.moe_forward_piecewise_cuda_graph_impl"
            )

        self.quant_config = getattr(self.model_runner.model, "quant_config", None)

        # Batch sizes to capture
        # 获取需要捕获 CUDA Graph 的 token 数量列表
        self.capture_num_tokens = self.compile_config.get_capture_sizes()
        # When the layer communicator scatters/gathers across the attention TP
        # group (e.g. with --moe-dense-tp-size 1), the model's reduce_scatter
        # requires the token count to be divisible by attn_tp_size * attn_cp_size.
        # Drop captures that would violate this (mirrors the filter used by
        # the regular CUDA graph runner in get_batch_sizes_to_capture).
        # 过滤掉不满足整除要求的 token 数量，避免 reduce_scatter 操作出错
        if require_gathered_buffer(self.model_runner.server_args):
            mul_base = self.attn_tp_size
            attn_cp_size = get_attention_cp_size()
            if mul_base % attn_cp_size != 0:
                mul_base *= attn_cp_size  # 调整整除基数
            filtered = [n for n in self.capture_num_tokens if n % mul_base == 0]
            assert (
                len(filtered) > 0
            ), f"No piecewise CUDA graph capture sizes are multiples of {mul_base}"
            self.capture_num_tokens = filtered
        log_info_on_rank0(
            logger, f"Capture cuda graph num tokens {self.capture_num_tokens}"
        )
        self.capture_forward_mode = ForwardMode.EXTEND  # 捕获时使用 EXTEND 模式
        self.capture_hidden_mode = CaptureHiddenMode.NULL  # 默认不捕获隐藏状态

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        # 如果启用了隐藏状态返回，则设置捕获隐藏模式为 FULL，避免启动时重复捕获
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        self.max_num_tokens = (
            max(self.capture_num_tokens) if self.capture_num_tokens else 8192
        )
        self.max_bs = model_runner.req_to_token_pool.size  # 最大批大小等于请求池大小

        self.is_multimodal = model_runner.is_multimodal
        self.mamba_track_enabled = self.is_mamba_track_enabled()
        # Classification/reward forwards branch on return_pooled_hidden_states; piecewise
        # CUDA graph capture must use the same flag value as replay for those models.
        # 分类/奖励模型需要返回池化隐藏状态，捕获和回放必须使用相同的标志值
        self.capture_return_pooled_hidden_states = not model_runner.is_generation

        # Graph inputs
        # 预分配 CUDA Graph 所需的输入缓冲区，确保捕获时内存地址固定
        with torch.device(self.device):
            input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
            out_cache_loc = torch.zeros(
                (self.max_num_tokens,), dtype=self._cache_loc_dtype()
            )
            mamba_track_indices = (
                torch.zeros((self.max_bs,), dtype=torch.int64)
                if self.mamba_track_enabled
                else None
            )
            mamba_track_mask = (
                torch.zeros((self.max_bs,), dtype=torch.bool)
                if self.mamba_track_enabled
                else None
            )
            mamba_track_seqlens = (
                torch.zeros((self.max_bs,), dtype=torch.int32)
                if self.mamba_track_enabled
                else None
            )
            positions = torch.zeros((self.max_num_tokens,), dtype=torch.int64)

            self.tbo_plugin = TboCudaGraphRunnerPlugin()  # 双批重叠插件

            if (
                self.is_multimodal
            ):  # Only create input_embeds and mrope_positions for multimodal model to save memory
                # 1. In multimodal, we only compile and capture the language model part.
                # 2. The embedder is outside of the graph, but cuda graph requires the input embeds to have a fixed memory address.
                # 3. Input embeds is a pre-allocated buffer. In model.forward, we copy the embed output to this buffer.
                # 对于多模态模型，需要预分配 input_embeds 和 mrope_positions 缓冲区
                # 因为 CUDA Graph 要求输入张量的内存地址固定，而 embedder 在 graph 之外
                input_embeds = torch.zeros(
                    (self.max_num_tokens, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )
                mrope_positions = torch.zeros(
                    (3, self.max_num_tokens), dtype=torch.int64
                )  # 多模态旋转位置编码，3个维度
            else:
                input_embeds = None
                mrope_positions = None

        # 创建输入缓冲区对象并共享底层存储
        self.buffers = PrefillInputBuffers(
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            positions=positions,
            input_embeds=input_embeds,
            mrope_positions=mrope_positions,
        )
        self.buffers.share_buffers()

        self.attention_layers = self.model_runner.attention_layers  # 注意力层列表
        self.moe_layers = self.model_runner.moe_layers  # MoE 层列表
        self.moe_fusions = self.model_runner.moe_fusions  # MoE 融合层列表
        self.dsa_indexers = getattr(self.model_runner, "dsa_indexers", None)  # DSA 索引器

        # 初始化全局 CUDA Graph 内存池（所有 graph 共享）
        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())

        # 启用分段 CUDA Graph 上下文，编译模型并捕获 CUDA Graph
        with enable_piecewise_cuda_graph():
            # 获取语言模型部分（多模态模型的视觉编码器不在 graph 内）
            language_model = getattr(
                self.model_runner.model, "language_model", self.model_runner.model
            )
            # 获取包含层列表的模型部分
            layer_model = (
                language_model.model
                if hasattr(language_model, "model")
                and hasattr(language_model.model, "layers")
                else language_model
            )
            with patch_model(
                layer_model, self.compile_config.compiler
            ) as patched_model:

                # Dummy warmup for jit kernel
                # 对第一个 token 数量进行预热编译，确保 JIT 内核已初始化
                self.warmup_compile(num_tokens=self.capture_num_tokens[0])

                # 安装 torch.compile 编译后的模型
                install_torch_compiled(
                    patched_model,
                    fullgraph=True,
                    dynamic_arg_dims=None,
                    compile_config=self.compile_config,
                    graph_pool=get_global_graph_memory_pool(),
                )

                # 对所有捕获 token 数量进行编译预热（从大到小，复用内存池）
                with enable_piecewise_cuda_graph_compile():
                    compile_range = (
                        tqdm.tqdm(list(reversed(self.capture_num_tokens)))
                        if get_tensor_model_parallel_rank() == 0
                        else reversed(self.capture_num_tokens)
                    )
                    for _, num_tokens in enumerate(compile_range):
                        if get_tensor_model_parallel_rank() == 0:
                            compile_range.set_description(
                                f"Compiling num tokens ({num_tokens=})"
                            )
                        self.warmup_compile(num_tokens=num_tokens)

                # 更新全局内存池句柄（编译后可能已改变）
                set_global_graph_memory_pool(self.device_module.graph_pool_handle())
                set_graph_pool_id(get_global_graph_memory_pool())

                self.device_module.synchronize()  # 同步设备，确保所有操作完成
                self.model_runner.tp_group.barrier()  # 张量并行组同步
                # Capture
                self.capture()  # 执行 CUDA Graph 捕获

        self.raw_num_tokens = 0  # 记录实际 token 数量（用于回放时截取输出）

    # 预热编译：在 CUDA Graph 捕获之前执行一次简单的前向传播，
    # 确保 torch.compile 已编译好所有需要的内核
    def warmup_compile(self, num_tokens: int):
        """Warmup the model with a simple forward pass before CUDA graph capture."""
        buffers = self.buffers
        input_ids = buffers.input_ids[:num_tokens]
        input_embeds = buffers.input_embeds[:num_tokens] if self.is_multimodal else None
        positions = buffers.positions[:num_tokens]
        mrope_positions = (
            buffers.mrope_positions[:, :num_tokens] if self.is_multimodal else None
        )
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        mamba_track_indices = (
            buffers.mamba_track_indices[:1]
            if buffers.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            buffers.mamba_track_mask[:1]
            if buffers.mamba_track_mask is not None
            else None
        )
        mamba_track_seqlens = (
            buffers.mamba_track_seqlens[:1]
            if buffers.mamba_track_seqlens is not None
            else None
        )
        # 构造虚拟的 ForwardBatch 用于预热
        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=input_ids,
                input_embeds=input_embeds,
                req_pool_indices=torch.arange(1, device=self.device),
                seq_lens=torch.tensor([num_tokens], device=self.device),
                next_token_logits_buffer=None,
                orig_seq_lens=torch.tensor([num_tokens], device=self.device),
                seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                out_cache_loc=out_cache_loc,
                seq_lens_sum=num_tokens,
                mamba_track_indices=mamba_track_indices,
                mamba_track_mask=mamba_track_mask,
                mamba_track_seqlens=mamba_track_seqlens,
                encoder_lens=None,
                return_logprob=False,
                extend_num_tokens=num_tokens,
                extend_seq_lens=torch.tensor([num_tokens], device=self.device),
                extend_prefix_lens=torch.tensor([0], device=self.device),
                extend_start_loc=torch.tensor([0], device=self.device),
                extend_prefix_lens_cpu=torch.tensor([0], device="cpu"),
                extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                positions=positions,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=None,
                mrope_positions=mrope_positions,
                spec_algorithm=None,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_token_non_padded=None,
                num_token_non_padded_cpu=num_tokens,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
                return_pooled_hidden_states=self.capture_return_pooled_hidden_states,
            )

        # Attention backend
        # 初始化注意力后端的元数据
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(None, num_tokens, forward_batch.dp_padding_mode.is_max_len())
        set_is_extend_in_batch(False)
        # 在前向上下文中执行一次模型前向传播来完成预热
        with forward_context(
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ):
            with set_forward_context(
                forward_batch,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
                dsa_indexers=self.dsa_indexers,
            ):
                _ = self.model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                )

    # 获取 cache_loc 张量的数据类型（NPU 平台使用 int32，其他平台使用 int64）
    def _cache_loc_dtype(self):
        return torch.int64 if not is_npu() else torch.int32

    # 判断当前 forward_batch 是否可以使用分段 CUDA Graph 来执行
    # 需要满足多个条件：无 input_embeds、非 target_verify、隐藏模式匹配、无嵌入替换、token 数在范围内等
    def can_run(self, forward_batch: ForwardBatch):
        # Disable piecewise cuda graph for input embeddings
        # TODO(yuwei): fix it
        # 不支持 input_embeds 输入（多模态嵌入由外部处理）
        if forward_batch.input_embeds is not None:
            return False
        # PCG graphs are captured with ForwardMode.EXTEND and spec_info=None.
        # TARGET_VERIFY has different spec_info and capture_hidden_mode,
        # so it must not use PCG-captured graphs.
        # TARGET_VERIFY 模式的 spec_info 和捕获时不同，不能使用 PCG 图
        if forward_batch.forward_mode.is_target_verify():
            return False
        # PCG graphs are captured with the runner's capture_hidden_mode.
        # If the batch needs a different mode (e.g. FULL for speculative
        # decoding), PCG replay would return wrong/missing hidden_states.
        # 隐藏模式不匹配时不能使用（否则会返回错误或缺失的隐藏状态）
        if forward_batch.capture_hidden_mode != self.capture_hidden_mode:
            return False
        # Disable for token embedding overrides (dynamic per-request)
        # 动态嵌入替换不支持，因为每个请求的替换内容不同
        if forward_batch.replace_embeds is not None:
            return False
        num_tokens = len(forward_batch.input_ids)
        # 如果需要返回 logprob，检查是否有部分 token 需要计算 logprob
        # 目前只支持所有 token 都需要 logprob 的情况
        if forward_batch.return_logprob:
            for start_len, seq_len in zip(
                forward_batch.extend_logprob_start_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
            ):
                if start_len is not None and start_len < seq_len:
                    return False
        # token 数量必须在已捕获的最大 token 数之内
        if num_tokens <= self.max_num_tokens:
            return True
        return False

    # 捕获 CUDA Graph：对所有预定义的 token 数量分别捕获 CUDA Graph
    # 从大到小捕获，使较小的 graph 可以复用较大 graph 分配的内存池
    def capture(self) -> None:
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with (
            freeze_gc(self.model_runner.server_args.enable_cudagraph_gc),
            graph_capture() as graph_capture_context,
        ):
            stream = graph_capture_context.stream  # 获取专用的 graph 捕获流
            with set_pcg_capture_stream(stream):
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                # Reverse the order to enable better memory sharing across cuda graphs.
                # 从大到小遍历，使大 graph 先分配内存，小 graph 可以复用
                capture_range = (
                    tqdm.tqdm(list(reversed(self.capture_num_tokens)))
                    if get_tensor_model_parallel_rank() == 0
                    else reversed(self.capture_num_tokens)
                )
                for i, num_tokens in enumerate(capture_range):
                    if get_tensor_model_parallel_rank() == 0:
                        avail_mem = get_available_gpu_memory(
                            self.model_runner.device,
                            self.model_runner.gpu_id,
                            empty_cache=False,
                        )
                        capture_range.set_description(
                            f"Capturing num tokens ({num_tokens=} {avail_mem=:.2f} GB)"
                        )

                    self.capture_one_batch_size(num_tokens)  # 捕获单个 token 数量的 CUDA Graph

    # 捕获单个批大小的 CUDA Graph：构造虚拟输入，运行两次前向传播
    # 第一次为预热，第二次为实际的 CUDA Graph 捕获
    def capture_one_batch_size(self, num_tokens: int):
        buffers = self.buffers
        bs = 1  # 捕获时固定 batch_size=1

        # Graph inputs
        # 从预分配的缓冲区中截取对应 token 数量的切片
        input_ids = buffers.input_ids[:num_tokens]
        input_embeds = buffers.input_embeds[:num_tokens] if self.is_multimodal else None

        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        mamba_track_indices = (
            buffers.mamba_track_indices[:bs]
            if buffers.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            buffers.mamba_track_mask[:bs]
            if buffers.mamba_track_mask is not None
            else None
        )
        mamba_track_seqlens = (
            buffers.mamba_track_seqlens[:bs]
            if buffers.mamba_track_seqlens is not None
            else None
        )
        positions = buffers.positions[:num_tokens]
        mrope_positions = (
            buffers.mrope_positions[:, :num_tokens] if self.is_multimodal else None
        )

        global_dp_buffer_len = None

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            # 启用 LoRA 时使用空 LoRA id 捕获 graph，内核会立即返回
            lora_ids = [None] * bs
        else:
            lora_ids = None

        # 构造捕获用的 ForwardBatch
        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=bs,
                input_ids=input_ids,
                input_embeds=input_embeds,
                req_pool_indices=torch.arange(bs, device=self.device),
                seq_lens=torch.tensor([num_tokens], device=self.device),
                next_token_logits_buffer=None,
                orig_seq_lens=torch.tensor([num_tokens], device=self.device),
                seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                out_cache_loc=out_cache_loc,
                seq_lens_sum=num_tokens,
                mamba_track_indices=mamba_track_indices,
                mamba_track_mask=mamba_track_mask,
                mamba_track_seqlens=mamba_track_seqlens,
                encoder_lens=None,
                return_logprob=False,
                extend_num_tokens=num_tokens,
                extend_seq_lens=torch.tensor([num_tokens], device=self.device),
                extend_prefix_lens=torch.tensor([0], device=self.device),
                extend_start_loc=torch.tensor([0], device=self.device),
                extend_prefix_lens_cpu=torch.tensor([0], device="cpu"),
                extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                positions=positions,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=None,
                mrope_positions=mrope_positions,
                spec_algorithm=None,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_token_non_padded=None,
                num_token_non_padded_cpu=num_tokens,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
                return_pooled_hidden_states=self.capture_return_pooled_hidden_states,
            )
        # Setup hooks below read get_attn_backend() and must run inside the
        # same ForwardContext as the warmup/capture forward.
        # 在与前向传播相同的 ForwardContext 中设置钩子
        with forward_context(
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ):
            self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

            if lora_ids is not None:
                self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

            self.model_runner.attn_backend.init_forward_metadata(forward_batch)

            # Run and capture
            # 定义单次运行函数，用于 CUDA Graph 捕获
            def run_once():
                # Invalidate SWA loc cache — same fix as in cuda_graph_runner.run_once.
                # 滑动窗口注意力(SWA)需要使缓存的 loc 失效
                if self.model_runner.is_hybrid_swa:
                    self.model_runner.token_to_kv_pool.invalidate_loc_cache()

                # Clean intermediate result cache for DP attention
                # 清理 DP 注意力的中间结果缓存
                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )
                set_dp_buffer_len(
                    global_dp_buffer_len,
                    num_tokens,
                    forward_batch.dp_padding_mode.is_max_len(),
                )
                # FIXME: the implementation is hacky. `is_extend_in_batch`` is for determining the deepep mode.
                # It is True in this context but we need to set it to use low latency deepep mode.
                # 临时设置 is_extend_in_batch=False 以使用低延迟 DeepEP 模式
                set_is_extend_in_batch(False)

                kwargs = {}
                with set_forward_context(
                    forward_batch,
                    self.attention_layers,
                    self.quant_config,
                    self.moe_layers,
                    self.moe_fusions,
                    dsa_indexers=self.dsa_indexers,
                ):
                    self.model_runner.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
                return

            # run twice for warmup at the first time and cuda graph capture at the second time
            # detail lies in sglang/python/sglang/srt/compilation/cuda_piecewise_backend.py
            # 运行两次：第一次预热，第二次触发 CUDA Graph 捕获
            for _ in range(2):
                self.device_module.synchronize()
                self.model_runner.tp_group.barrier()
                run_once()

        return

    # 回放准备：将 forward_batch 的数据拷贝到静态缓冲区，
    # 构造与捕获时形状匹配的 static_forward_batch 供 CUDA Graph 回放使用
    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        buffers = self.buffers
        num_tokens = len(forward_batch.input_ids)
        # 使用二分查找找到大于等于当前 token 数的最小捕获 token 数
        index = bisect.bisect_left(self.capture_num_tokens, num_tokens)
        static_num_tokens = self.capture_num_tokens[index]
        self.raw_num_tokens = num_tokens  # 保存实际 token 数，用于截取输出
        # 如果静态 token 数大于实际 token 数，需要将多余的位置清零
        if static_num_tokens != num_tokens:
            buffers.out_cache_loc.zero_()
            buffers.input_ids[num_tokens:static_num_tokens].zero_()
            buffers.positions[num_tokens:static_num_tokens].zero_()
            if self.is_multimodal:
                buffers.input_embeds[num_tokens:static_num_tokens].zero_()
            if forward_batch.mrope_positions is not None:
                buffers.mrope_positions[:, num_tokens:static_num_tokens].zero_()

        bs = forward_batch.batch_size

        # 将 forward_batch 的数据拷贝到静态缓冲区中
        buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        buffers.positions[:num_tokens].copy_(forward_batch.positions)
        buffers.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)

        # 拷贝 Mamba 跟踪相关的缓冲区
        if (
            buffers.mamba_track_indices is not None
            and forward_batch.mamba_track_indices is not None
        ):
            buffers.mamba_track_indices[:bs].copy_(forward_batch.mamba_track_indices)
        if (
            buffers.mamba_track_mask is not None
            and forward_batch.mamba_track_mask is not None
        ):
            buffers.mamba_track_mask[:bs].copy_(forward_batch.mamba_track_mask)
        if (
            buffers.mamba_track_seqlens is not None
            and forward_batch.mamba_track_seqlens is not None
        ):
            buffers.mamba_track_seqlens[:bs].copy_(forward_batch.mamba_track_seqlens)

        # 截取静态大小的输入张量（与捕获时形状一致）
        input_ids = buffers.input_ids[:static_num_tokens]
        positions = buffers.positions[:static_num_tokens]
        out_cache_loc = buffers.out_cache_loc[:static_num_tokens]

        mamba_track_indices = (
            buffers.mamba_track_indices[:bs]
            if buffers.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            buffers.mamba_track_mask[:bs]
            if buffers.mamba_track_mask is not None
            else None
        )
        mamba_track_seqlens = (
            buffers.mamba_track_seqlens[:bs]
            if buffers.mamba_track_seqlens is not None
            else None
        )
        if forward_batch.mrope_positions is not None:
            buffers.mrope_positions[:, :num_tokens].copy_(forward_batch.mrope_positions)

        input_ids = buffers.input_ids[:static_num_tokens]
        input_embeds = (
            buffers.input_embeds[:static_num_tokens] if self.is_multimodal else None
        )

        mrope_positions = (
            buffers.mrope_positions[:, :static_num_tokens]
            if forward_batch.mrope_positions is not None
            else None
        )

        next_token_logits_buffer = None

        # Normalize MIXED→EXTEND so dynamo's guard (captured with EXTEND=1) doesn't fail on MIXED=3.
        # 将 MIXED 模式归一化为 EXTEND 模式，因为 CUDA Graph 捕获时使用 EXTEND 模式
        pcg_forward_mode = (
            ForwardMode.EXTEND
            if forward_batch.forward_mode == ForwardMode.MIXED
            else forward_batch.forward_mode
        )
        pcg_global_forward_mode = (
            ForwardMode.EXTEND
            if forward_batch.global_forward_mode == ForwardMode.MIXED
            else forward_batch.global_forward_mode
        )

        # 构造与捕获时形状匹配的静态 ForwardBatch
        static_forward_batch = ForwardBatch(
            forward_mode=pcg_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=forward_batch.orig_seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=forward_batch.seq_lens_sum,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            encoder_lens=forward_batch.encoder_lens,
            return_logprob=False,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_start_loc=forward_batch.extend_start_loc,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_num_tokens=forward_batch.extend_num_tokens,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            positions=positions,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=forward_batch.dp_padding_mode,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=forward_batch.spec_algorithm,
            spec_info=forward_batch.spec_info,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            num_token_non_padded=forward_batch.num_token_non_padded,
            num_token_non_padded_cpu=forward_batch.num_token_non_padded_cpu,
            global_forward_mode=pcg_global_forward_mode,
            lora_ids=forward_batch.lora_ids,
            sampling_info=forward_batch.sampling_info,
            mm_inputs=forward_batch.mm_inputs,
            temperature=forward_batch.temperature,
            top_p=forward_batch.top_p,
            dimensions=forward_batch.dimensions,
            return_pooled_hidden_states=(
                self.capture_return_pooled_hidden_states
                or forward_batch.return_pooled_hidden_states
            ),
        )

        return static_forward_batch

    # 回放 CUDA Graph：将 forward_batch 准备为静态形状后执行模型前向传播，
    # 并根据输出类型截取实际 token 数量的结果
    def replay(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        with enable_piecewise_cuda_graph():
            static_forward_batch = self.replay_prepare(forward_batch, **kwargs)
            # Replay
            # 执行模型前向传播（使用静态形状的输入）
            with set_forward_context(
                static_forward_batch,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
                dsa_indexers=self.dsa_indexers,
            ):
                # Due to the dispatch kernel for MLA model, we init the metadata with original forward_batch
                # MLA 模型的分发内核需要使用原始 forward_batch 初始化元数据
                self.model_runner.attn_backend.init_forward_metadata(forward_batch)
                output = self.model_runner.model.forward(
                    static_forward_batch.input_ids,
                    static_forward_batch.positions,
                    static_forward_batch,
                    **kwargs,
                )
                if isinstance(output, LogitsProcessorOutput):
                    # Preserve mm_input_embeds when speculative decoding is
                    # enabled. The speculative draft's prefill path
                    # (eagle_worker_v2._draft_extend_for_prefill) reads
                    # mm_input_embeds off this LogitsProcessorOutput to reuse
                    # the target's encoder embeddings instead of re-embedding
                    # multimodal placeholder token ids.
                    # 投机解码时保留多模态输入嵌入，避免重复计算
                    mm_input_embeds = None
                    if (
                        self.model_runner.spec_algorithm.is_speculative()
                        and output.mm_input_embeds is not None
                    ):
                        mm_input_embeds = output.mm_input_embeds[: self.raw_num_tokens]
                    # 截取实际 token 数量的输出，去除 padding 部分
                    return LogitsProcessorOutput(
                        next_token_logits=output.next_token_logits[
                            : self.raw_num_tokens
                        ],
                        hidden_states=(
                            output.hidden_states[: self.raw_num_tokens]
                            if output.hidden_states is not None
                            else None
                        ),
                        mm_input_embeds=mm_input_embeds,
                    )
                elif isinstance(output, EmbeddingPoolerOutput):
                    return output
                else:
                    assert isinstance(output, PPProxyTensors)
                    # TODO(Yuwei): support PP Support
                    # 流水线并行代理张量暂不支持
                    raise NotImplementedError(
                        "PPProxyTensors is not supported in PiecewiseCudaGraphRunner yet."
                    )

    # 获取投机解码的 spec_info，目前仅支持 EAGLE 和 standalone 投机算法
    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                # 构造 EAGLE 验证输入，用于投机解码的验证阶段
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )

        return spec_info
