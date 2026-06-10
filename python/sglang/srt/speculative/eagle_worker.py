# EAGLE推测解码工作器模块
# 本文件实现了EAGLE（Efficient Accuracy-Guided Likelihood Enhancement）推测解码的工作器。
# EAGLE通过使用轻量级草稿模型快速生成候选token树，再由目标模型验证，从而加速推理。
# 主要包含：草稿模型的初始化、多步草稿前向传播、目标模型验证、
# 自适应运行时状态管理、CUDA图捕获与重放等功能。

import contextlib
import logging
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.observability.trace import get_global_tracing_enabled
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    apply_eagle_prefill_input_rotation,
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    get_last_loc_large_page_size_large_top_k,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    empty_context,
    get_available_gpu_memory,
    is_cuda,
    is_musa,
    is_npu,
    log_info_on_rank0,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

# 检测当前是否为NPU或MUSA设备
_is_npu = is_npu()
_is_musa = is_musa()

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


# EAGLE推测解码工作器类，继承自TpModelWorker
# 负责管理草稿模型和目标模型的协作推理，包括草稿生成、验证和自适应运行时控制
class EAGLEWorker(TpModelWorker):

    # 初始化EAGLE工作器
    # 参数包括服务器配置、GPU ID、张量并行秩、数据并行秩、MoE专家并行秩等
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        # 解析参数并保存关键配置
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        # 根据字符串解析推测解码算法类型
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Adaptive speculative
        # 自适应推测解码控制器，根据运行时状态动态调整推测步数
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self, config_path=server_args.speculative_adaptive_config
            )

        # Override the context length of the draft model to be the same as the target model.
        # 将草稿模型的上下文长度覆盖为目标模型的上下文长度
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        # 暂时禁用CUDA图捕获，后续会单独处理
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        # 与目标工作器共享内存池分配器，草稿和目标工作器各自拥有KV缓存池
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        # 加载热词token ID映射表，用于EAGLE3模型的词表裁剪
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            # 从文件加载token映射表，并设置模型的hot_vocab_size参数
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        # 初始化草稿模型工作器，在适当的上下文中执行
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism  # 推测工作器不支持流水线并行
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # 获取目标模型的嵌入层和语言模型头
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            # 大多数EAGLE3模型不共享lm_head，但部分模型（如nvidia/gpt-oss-120b-Eagle3）共享
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                # 同时设置嵌入层和语言模型头
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                # 仅设置嵌入层
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            # 从EAGLE3模型中获取热词token ID
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                # 根据热词ID裁剪语言模型头，只保留热词对应的行
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            # 与草稿模型共享嵌入层和语言模型头
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        # 初始化注意力后端和CUDA图
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        # 根据是否启用DP注意力选择草稿模型的张量并行上下文
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        # EAGLE3模型使用辅助隐藏状态
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            # 从EAGLE配置中读取是否使用辅助隐藏状态
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()
            self.init_cuda_graphs()
            # 如果启用了自适应控制器，注册运行时状态并初始化
            if self.adaptive_controller is not None:
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self.draft_attn_backend,
                        cuda_graph_runner=self.cuda_graph_runner,
                        target_attn_backend=self.target_worker.model_runner.attn_backend,
                        target_graph_runner=self.target_worker.model_runner.graph_runner,
                        draft_extend_attn_backend=self.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states()

        # Some dummy tensors
        # 占位张量，用于CUDA内核中传递参数
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    # 初始化注意力后端，包括解码注意力和草稿扩展注意力
    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        # 创建多步注意力后端和CUDA图运行器
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        # 初始化解码注意力后端
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        # 初始化草稿扩展注意力后端（遵循speculative_attention_mode设置）
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        # 将解码注意力后端绑定到草稿模型运行器
        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    # 初始化CUDA图，包括草稿解码图和草稿扩展图
    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # 不同设备对应的草稿CUDA图运行器
        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        # 捕获草稿解码的CUDA图（仅在多步推测时）
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            # 根据目标设备类型创建对应的CUDA图运行器
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

        # Capture extend
        # 捕获草稿扩展的CUDA图（NPU不支持）
        if self.draft_extend_attn_backend and not _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

    # 应用预构建的运行时状态到当前工作器
    # 用于自适应推测解码中切换不同的步数配置
    def apply_runtime_state(self, state: SpecRuntimeState):
        """Apply a pre-built runtime state to this worker."""
        # 如果步数相同则无需切换
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # 更新推测步数和草稿token数
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        # Draft stage
        # 草稿阶段：更新注意力后端和CUDA图运行器
        self.draft_attn_backend = state.draft_attn_backend
        self.draft_model_runner.draft_attn_backend = state.draft_attn_backend
        self.cuda_graph_runner = state.cuda_graph_runner
        # Verify stage
        # 验证阶段：更新目标模型的注意力后端和CUDA图运行器
        self.target_worker.model_runner.attn_backend = state.target_attn_backend
        self.target_worker.model_runner.graph_runner = state.target_graph_runner
        # Extend stage
        # 扩展阶段：更新草稿扩展注意力后端和CUDA图运行器
        self.draft_extend_attn_backend = state.draft_extend_attn_backend
        self.cuda_graph_runner_for_draft_extend = (
            state.cuda_graph_runner_for_draft_extend
        )
        # Sync server_args
        # 同步更新服务器参数中的推测步数配置
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    # 构建指定步数配置的自适应运行时状态
    # 包括重新初始化注意力后端、捕获CUDA图等
    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        # 临时覆盖工作器状态用于图捕获
        with self._override_worker_state(
            speculative_num_steps, speculative_num_draft_tokens
        ):
            # Reuse existing init methods for draft attention backend and cuda graphs
            # 复用已有的初始化方法创建草稿注意力后端和CUDA图
            self.init_attention_backend()
            self.init_cuda_graphs()

            # Capture target attention backend and CUDA graph
            # 捕获目标模型的注意力后端和CUDA图
            target_model_runner = self.target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            # 为目标模型构建新的CUDA图运行器
            target_graph_runner = None
            if not self.server_args.disable_cuda_graph:
                TargetGraphRunnerCls = NPUGraphRunner if _is_npu else CudaGraphRunner
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            # 构建运行时状态对象
            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                # Draft stage
                # 草稿阶段
                draft_attn_backend=self.draft_attn_backend,
                cuda_graph_runner=self.cuda_graph_runner,
                # Verify stage
                # 验证阶段
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                # Extend stage
                # 扩展阶段
                draft_extend_attn_backend=self.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    @contextmanager
    # 临时覆盖工作器状态，用于CUDA图捕获
    # 在上下文管理器退出时自动恢复原始状态
    def _override_worker_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        sa = self.server_args
        # 备份当前状态
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.draft_attn_backend,
            self.draft_extend_attn_backend,
            self.draft_model_runner.draft_attn_backend,
            self.cuda_graph_runner,
            self.cuda_graph_runner_for_draft_extend,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
        )
        # 临时设置新的步数参数
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        try:
            yield
        finally:
            # 恢复原始状态
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                self.draft_attn_backend,
                self.draft_extend_attn_backend,
                self.draft_model_runner.draft_attn_backend,
                self.cuda_graph_runner,
                self.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
            ) = backup

    @property
    # 草稿模型运行器属性，返回内部的模型运行器
    def draft_model_runner(self):
        return self.model_runner

    # 推测解码的主前向传播入口
    # 根据批次类型（扩展/解码）分别处理，返回生成结果
    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        # 扩展模式（prefill）路径
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            # 在草稿模型的张量并行上下文中运行草稿扩展
            with (
                self.draft_tp_context(self.draft_model_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.forward_draft_extend(
                    batch,
                    logits_output.hidden_states,
                    next_token_ids,
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:
            # 解码模式路径：草稿 -> 验证 -> 草稿扩展
            set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)

            # 草稿阶段：生成候选token树
            with (
                self.draft_tp_context(self.draft_model_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                verify_input = self.draft(batch)

            set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)
            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

            # Install verify_input as `batch.spec_info` for the verify forward.
            # 将验证输入安装为batch.spec_info，用于验证前向传播
            batch.spec_info = verify_input
            # 验证阶段：用目标模型验证候选token
            verify_output = self.verify(batch)

            # 记录追踪信息（如果启用）
            if get_global_tracing_enabled():
                for idx, req in enumerate(batch.reqs):
                    num_correct_drafts = verify_output.num_correct_drafts_per_req_cpu[
                        idx
                    ]
                    req.time_stats.set_spec_verify_end_time(
                        num_correct_drafts=num_correct_drafts
                    )

            set_time_batch(
                batch.reqs, "set_spec_draft_extend_start_time", trace_only=True
            )

            # 草稿扩展阶段：根据验证结果准备下一轮草稿输入
            with (
                self.draft_tp_context(self.draft_model_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                draft_extend_input = verify_output.draft_extend_input
                if (
                    self.server_args.enable_dp_attention
                    or draft_extend_input.input_ids.shape[0] > 0
                ):
                    # decode is not finished; install draft_extend_input for
                    # the extend forward, then install the next-iter
                    # EagleDraftInput it returns.
                    # 解码未完成：安装草稿扩展输入，执行扩展前向，然后安装下一轮的草稿输入
                    batch.spec_info = draft_extend_input
                    next_draft_input = self.forward_draft_extend_after_decode(batch)
                    batch.spec_info = next_draft_input
                else:
                    # All reqs finished and dp_attention isn't forcing extend.
                    # Install an idle EagleDraftInput so next iter's scheduler
                    # ops (merge_batch / filter_batch) see well-typed empty
                    # tensors instead of None.
                    # 所有请求已完成且DP注意力未强制扩展：安装空闲的草稿输入
                    self._draft_preprocess_idle(batch)

            set_time_batch(
                batch.reqs, "set_spec_draft_extend_end_time", trace_only=True
            )

            # 自适应控制器回调：通知验证完成
            if self.adaptive_controller is not None:
                self.adaptive_controller.on_verify_complete(
                    verify_output.num_correct_drafts_per_req_cpu
                )

            return GenerationBatchResult(
                logits_output=verify_output.logits_output,
                next_token_ids=verify_output.accept_tokens,
                num_correct_drafts=sum(verify_output.num_correct_drafts_per_req_cpu),
                num_correct_drafts_per_req_cpu=verify_output.num_correct_drafts_per_req_cpu,
                can_run_cuda_graph=verify_output.can_run_cuda_graph,
            )

    # 运行目标模型的扩展（prefill）前向传播
    # 返回逻辑输出、下一个token ID、序列长度和CUDA图可运行标志
    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor], bool]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            seq_lens_cpu: CPU copy of sequence lengths for the draft prefill path.
            can_run_cuda_graph: Whether the target prefill ran with cuda graph.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        # 使用目标模型前向传播并获取隐藏状态，用于填充草稿模型的KV缓存
        # 根据算法类型决定是否捕获完整隐藏状态
        capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.capture_hidden_mode = capture_mode
        batch_result = self.target_worker.forward_batch_generation(batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            batch.seq_lens_cpu,
            batch_result.can_run_cuda_graph,
        )

    # 草稿解码的预处理：分配KV缓存位置、设置惩罚、构建注意力掩码等
    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        # 可能驱逐滑动窗口注意力中的旧token
        batch.maybe_evict_swa()
        # 递增每个请求的解码批次索引
        for req in batch.reqs:
            req.decode_batch_idx += 1

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        # 累积采样惩罚（如重复惩罚等）
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            # 这是推测解码的惩罚累积（宽松版本）
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.bonus_tokens.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        # 分配KV缓存位置，布局为[topk0的各步, topk1的各步, ...]
        if self.page_size == 1:
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            # 分配token槽位并备份KV池状态
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            # 大页面大小的处理逻辑
            if self.topk == 1:
                # topk=1的简化路径
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens
                # topk>1时需要复制最后一个不完整页面的KV缓存

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                # 计算每个topk分支需要的新页面数
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                # 计算扩展后的序列长度
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                # 计算需要扩展的总token数
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            # 分配分页token槽位
            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        # 大页面且topk>1时，需要复制最后一个不完整页面的KV缓存
        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            # 准备源和目标缓存位置用于KV缓存复制
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            # 不需要复制缓存时跳过
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        # 调用CUDA内核分配草稿缓存位置
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        # 执行KV缓存复制（大页面+多topk时）
        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            # 移除填充的槽位，只保留实际需要的缓存位置
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        # 设置批次的输出缓存位置和序列长度信息
        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        # 每个序列的位置为原始序列长度，按topk重复
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        # 恢复KV池分配器的状态
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    # 空闲状态的草稿预处理：创建空闲的草稿输入
    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        # 独立算法不捕获隐藏状态，否则捕获最后一个token的隐藏状态
        capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=EagleDraftInput.hidden_size_for(self),
            dtype=EagleDraftInput.dtype_for(self),
            topk=self.topk,
            capture_hidden_mode=capture_mode,
        )

    # 草稿阶段：生成候选token树
    # 返回EagleVerifyInput供验证阶段使用
    def draft(self, batch: ScheduleBatch):
        # Parse args
        # 空闲模式或解码模式的预处理
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        # 设置草稿阶段的隐藏状态捕获模式
        draft_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        spec_info.capture_hidden_mode = draft_capture_mode
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        # 构建前向传播批次
        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        assert forward_batch.capture_hidden_mode == draft_capture_mode
        # 检查是否可以使用CUDA图加速
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_cuda_graph:
            # 使用CUDA图重放获取草稿结果
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for idle mode or 1-step draft
                # 空闲模式或单步草稿跳过注意力后端初始化
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            # Run forward steps
            # 运行多步草稿前向传播
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        # 空闲模式返回空闲验证输入
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # 构建树形注意力掩码和检索索引
        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        # 目标模型需要捕获完整隐藏状态用于验证
        target_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=target_capture_mode,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    # 多步草稿前向传播
    # 在每步中选择topk token，构建草稿token树
    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # 检测初始topk概率中的NaN
        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # 如果存在热词映射，将topk索引映射到实际词表索引
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        # 将缓存位置重塑为[步数, 序列数*topk]的布局
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        # 存储每步的得分、token和父节点信息
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        # 多步前向传播循环
        scores = None
        for i in range(self.speculative_num_steps):
            # 选择topk token作为当前步的输入
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            # 记录当前步的树结构信息
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            # 最后一步不需要运行前向传播，因为已经从prefill获得了1个token
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            # 设置当前步的输入
            forward_batch.input_ids = input_ids
            # Some draft model RoPE kernels need cache_loc to be contiguous.
            # 某些草稿模型的RoPE内核需要cache_loc连续
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ) or self.model_config.hf_config.architectures[0] == (
                "Qwen3MoeForCausalLMMTP"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            spec_info.hidden_states = hidden_states

            # Run forward under a per-step ForwardContext so the model layer
            # reads attn_backends[i] for the i-th draft step. ``_forward_raw``
            # is no-op for the attn_backend half when a context is already
            # active, so this outer wrap is what reaches RadixAttention.
            # 在每步的前向上下文中运行，使模型层使用对应步的注意力后端
            with forward_context(
                ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[i])
            ):
                logits_output = self.draft_model_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
            # 检测异常值
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            maybe_detect_inf(logits_output.next_token_logits, f"draft_forward step {i}")
            # 计算概率分布并选择topk
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            # 映射热词索引
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            # 更新隐藏状态用于下一步
            hidden_states = logits_output.hidden_states
            maybe_detect_nan(hidden_states, f"draft_forward step {i}: hidden_states")
            maybe_detect_inf(hidden_states, f"draft_forward step {i}: hidden_states")
            # 递增位置编码
            forward_batch.positions.add_(1)

        # 整理所有步的草稿结果，构建最终的父节点列表、得分索引和草稿token
        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    # 清理缓存池（草稿和目标工作器共享分配器，无需额外操作）
    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        pass

    # 验证阶段：用目标模型验证草稿token，确定接受的token
    def verify(self, batch: ScheduleBatch):
        spec_info: EagleVerifyInput = batch.spec_info
        # 保存验证前的序列长度（用于Mamba状态更新）
        seq_lens_pre_verify = batch.seq_lens.clone()
        # 准备验证所需的数据结构
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        # 设置前向模式为验证模式
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        # 提前将验证所需的token数据转移到CPU（用于语法约束的bitmask生成）
        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()

        # Forward
        # 运行目标模型的前向传播进行验证
        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu
        batch_result = self.target_worker.forward_batch_generation(
            batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        # 生成语法约束的词表掩码（与目标模型前向传播重叠执行）
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                # 清除之前的词表掩码，避免使用错误的掩码
                batch.sampling_info.vocab_mask = None

        # 检测目标模型logits中的异常值
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")

        # 将隐藏状态传递给spec_info，然后执行验证逻辑
        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        # 根据验证结果筛选接受的token对应的logits和隐藏状态
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accept_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                res.accept_indices
            ]

        # 如果目标模型使用了Mamba/混合架构，需要更新Mamba状态
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        # 计算并添加logprobs（如果请求了返回logprob）
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        # 恢复前向模式为解码模式，为下一轮草稿做准备
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )

        res.can_run_cuda_graph = can_run_cuda_graph
        return res

    # Mamba/混合架构验证后的状态更新
    # 更新Mamba的SSM状态以反映验证接受的token
    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        # Under DP attention, some ranks can be IDLE during target verify and never
        # initialize mamba forward metadata for this step.
        # DP注意力下，某些rank可能在验证时处于空闲状态，无需更新Mamba
        if batch.forward_mode.is_idle():
            return

        # 计算每个请求接受的草稿token数
        num_correct_drafts = torch.tensor(
            res.num_correct_drafts_per_req_cpu,
            device=logits_output.next_token_logits.device,
            dtype=torch.int64,
        )
        # 计算累积接受token数的前缀和
        cumulative_num_accept_tokens = torch.cumsum(num_correct_drafts + 1, dim=0)
        # prepend 0 to the cumulative_num_accept_tokens
        # 在累积和前添加0，用于计算每个请求的起始索引
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_num_accept_tokens.dtype,
                    device=cumulative_num_accept_tokens.device,
                ),
                cumulative_num_accept_tokens[:-1],
            ]
        )
        # 计算每个请求在展平索引中的偏移量
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accept_indices.shape[0] > 0 skips DP attn idle batch
        # topk>1时需要计算每个请求最后一个正确步的索引
        if spec_info.topk > 1 and res.accept_indices.shape[0] > 0:
            # accept_indices=[0,2,3,4,5,7,9,10,11], num_accept_tokens=[4, 3, 2], cumulative_num_accept_tokens=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accept_indices[cumulative_num_accept_tokens[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accept_indices[cumulative_num_accept_tokens - 1] = [4, 9, 11] (last token ID of each req)
            # last_correct_step_indices = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # equivalent: last_correct_step_indices = last_token_indices_per_req - first_token_indices_per_req;
            # `accepted_indices_offset` equals `first_token_indices_per_req` because the first accepted slot of each req is its "current token" at logical position i * draft_token_num.
            # 计算每个请求中最后一个正确Mamba缓存对应的步偏移
            last_correct_step_indices = (
                res.accept_indices[cumulative_num_accept_tokens - 1]
                - accepted_indices_offset
            )
        else:
            # topk=1时，正确步数等于接受的草稿token数
            last_correct_step_indices = num_correct_drafts

        # 处理Mamba跟踪点的更新
        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            # 如果验证后序列长度跨越了Mamba跟踪间隔，需要在跨越点更新Mamba状态
            mamba_track_interval = self.server_args.mamba_track_interval
            # 找出跨越了跟踪间隔的请求
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            # 计算跟踪点位置
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            # 计算需要跟踪的步偏移
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            # 计算每个需要跟踪的请求对应的Mamba步索引
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accept_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        # 调用注意力后端更新Mamba状态
        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    # 草稿模型的扩展（prefill）前向传播
    # 在目标模型扩展后调用，用目标模型的隐藏状态初始化草稿模型
    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # 创建草稿输入，使用目标模型的隐藏状态和下一个token
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        # 应用EAGLE预填充旋转位置编码
        apply_eagle_prefill_input_rotation(batch, next_token_ids)
        # 设置隐藏状态捕获模式
        capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        batch.spec_info.capture_hidden_mode = capture_mode
        batch.seq_lens_cpu_cache = seq_lens_cpu
        # 构建前向传播批次并运行草稿模型
        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        forward_batch.return_logprob = False
        # 如果有多模态输入嵌入，传递给前向批次
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        logits_output = self.draft_model_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        # 捕获logits用于后续解码步骤
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    # 解码后的草稿扩展前向传播
    # 在验证后根据接受的token更新草稿模型状态，准备下一轮草稿输入
    def forward_draft_extend_after_decode(
        self, batch: ScheduleBatch
    ) -> EagleDraftInput:
        draft_extend_input: EagleDraftExtendInput = batch.spec_info

        # Backup fields that will be modified in-place
        # 备份将被原地修改的批次字段
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        return_logprob_backup = batch.return_logprob

        input_is_idle = batch.forward_mode.is_idle()

        draft_extend_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        # 如果输入为空（空闲或所有请求完成），创建空闲输入
        if draft_extend_input.input_ids.shape[0] == 0:
            # Single source for hidden_size via hidden_size_for(self) (incl.
            # EAGLE-3 aux widening). Two stub origins from verify(): fully-idle
            # batch (DP attn rank w/o reqs) and active batch with all reqs
            # finished. prepare_for_idle() is idempotent on already-idle.
            batch = batch.copy()
            batch.prepare_for_idle()
            draft_extend_input = EagleDraftExtendInput.create_idle_input(
                device=self.device,
                hidden_size=EagleDraftExtendInput.hidden_size_for(self),
                dtype=EagleDraftExtendInput.dtype_for(self),
                capture_hidden_mode=draft_extend_capture_mode,
            )
            batch.spec_info = draft_extend_input

        # Phase 1: prepare extend (kernel writes draft_extend_input.{positions, bonus_tokens})
        # 阶段1：准备扩展（内核写入位置和bonus_tokens）
        draft_extend_input.num_tokens_per_req = self.speculative_num_steps + 1
        draft_extend_input.num_tokens_for_logprob_per_req = 1
        draft_extend_input.prepare_extend_after_decode(
            batch,
            speculative_num_steps=self.speculative_num_steps,
        )
        # 设置前向模式为草稿扩展模式
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        # Verify-time construction of EagleDraftExtendInput uses the dataclass
        # default (LAST); override here so ForwardBatch.init_new picks up the
        # correct mode (NULL for STANDALONE).
        # 覆盖隐藏状态捕获模式
        draft_extend_input.capture_hidden_mode = draft_extend_capture_mode
        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        assert forward_batch.capture_hidden_mode == draft_extend_capture_mode
        # 计算序列长度总和
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        # Phase 2: run draft-extend forward
        # 阶段2：运行草稿扩展前向传播
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            # 使用CUDA图重放
            logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            # cuda-graph replay populates logits_output.{topk_p, topk_index, hidden_states}.
            # CUDA图重放填充topk概率、索引和隐藏状态
            topk_p = logits_output.topk_p
            topk_index = logits_output.topk_index
            hidden_states = logits_output.hidden_states
        else:
            # 非CUDA图路径：正常前向传播
            forward_batch.can_run_dp_cuda_graph = False
            attn_backend = None
            if not forward_batch.forward_mode.is_idle():
                # 选择草稿扩展或默认注意力后端
                attn_backend = (
                    self.draft_extend_attn_backend
                    or self.draft_model_runner.attn_backend
                )
                attn_backend.init_forward_metadata(forward_batch)
            # Publish the chosen backend via ForwardContext so model code
            # picks it up for this forward (no runner-attr mutation).
            # 通过ForwardContext发布选择的注意力后端
            if attn_backend is not None:
                ctx_mgr = forward_context(ForwardContext(attn_backend=attn_backend))
            else:
                ctx_mgr = contextlib.nullcontext()
            with ctx_mgr:
                logits_output = self.draft_model_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
            # Non-cuda-graph path: compute topk_p / topk_index inline.
            # 非CUDA图路径：在线计算topk概率和索引
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            hidden_states = logits_output.hidden_states

        # 检测logits中的异常值
        maybe_detect_nan(
            logits_output.next_token_logits,
            f"draft_extend_after_decode (cuda_graph={can_cuda_graph})",
        )

        # Phase 3: assemble next-iter EagleDraftInput from extend output
        # 阶段3：从扩展输出组装下一轮的草稿输入
        next_decode_capture_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        next_draft_input = EagleDraftInput(
            bonus_tokens=draft_extend_input.bonus_tokens,
            hidden_states=hidden_states,
            topk_p=topk_p,
            topk_index=topk_index,
            capture_hidden_mode=next_decode_capture_mode,
        )

        # Restore batch fields. `seq_lens` etc. were modified by
        # `prepare_extend_after_decode`. Caller installs `next_draft_input` as
        # `batch.spec_info`.
        # 恢复批次字段（prepare_extend_after_decode修改了它们）
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.return_logprob = return_logprob_backup
        return next_draft_input

    # 从logits输出中捕获topk概率和索引，保存到草稿输入中供后续解码使用
    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    # 从张量更新模型权重，同时更新草稿模型和目标模型
    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        # 反序列化接收到的命名张量
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        # 先更新草稿模型的权重
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        # 再更新目标模型的权重
        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message


# 获取大页面大小且topk=1时的最后位置信息
# 使用torch.compile优化，在NPU和MUSA上禁用
@torch.compile(dynamic=True, disable=(_is_npu or _is_musa))
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    # 前缀长度等于当前序列长度
    prefix_lens = seq_lens
    # 扩展后的序列长度
    seq_lens = prefix_lens + speculative_num_steps
    # 获取每个请求最后一个token的KV缓存位置
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc
