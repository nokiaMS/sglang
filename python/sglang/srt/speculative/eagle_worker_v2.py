# EAGLE Worker V2 模块
# 实现EAGLE投机解码的V2 Worker，包含草稿Worker（EagleDraftWorker）和
# 主Worker（EAGLEWorkerV2），支持自适应投机解码、CUDA图捕获、验证和KV缓存管理。
import contextlib  # 导入上下文管理工具
import logging  # 导入日志模块
import time  # 导入时间模块
from typing import List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,  # NPU草稿扩展图运行器
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,  # NPU草稿图运行器
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner  # NPU图运行器
from sglang.srt.kv_canary.runner.canary_manager import context_tuple  # 金丝雀上下文元组
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend  # Tokenspeed MLA后端
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  # Triton注意力后端
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,  # TRTLLM MLA后端
)
from sglang.srt.layers.dp_attention import get_attention_tp_group  # DP注意力工具
from sglang.srt.layers.moe.utils import (  # MoE工具
    speculative_moe_a2a_backend_context,  # 投机MoE A2A后端上下文
    speculative_moe_backend_context,  # 投机MoE后端上下文
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs  # 投机V2 logprob计算
from sglang.srt.managers.io_struct import (  # IO结构
    UpdateWeightFromDiskReqInput,  # 磁盘权重更新请求
    UpdateWeightsFromIPCReqInput,  # IPC权重更新请求
    UpdateWeightsFromTensorReqInput,  # 张量权重更新请求
)
from sglang.srt.managers.schedule_batch import ScheduleBatch  # 调度批次
from sglang.srt.managers.scheduler import GenerationBatchResult  # 生成批次结果
from sglang.srt.managers.tp_worker import TpModelWorker  # TP模型Worker
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner  # CUDA图运行器
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch  # 前向批次信息
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context  # 前向上下文
from sglang.srt.server_args import ServerArgs  # 服务器参数
from sglang.srt.speculative.adaptive_runtime_state import (  # 自适应运行时状态
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker  # 基类
from sglang.srt.speculative.draft_utils import DraftBackendFactory  # 草稿后端工厂
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,  # EAGLE草稿CUDA图运行器
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,  # EAGLE草稿扩展CUDA图运行器
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput  # EAGLE输入类
from sglang.srt.speculative.eagle_info_v2 import (  # EAGLE V2信息
    assign_extend_cache_locs,  # 分配扩展缓存位置
    fill_accepted_out_cache_loc,  # 填充接受输出缓存位置
    fill_bonus_tokens,  # 填充奖励token
)
from sglang.srt.speculative.eagle_utils import (  # EAGLE工具
    TreeMaskMode,  # 树掩码模式
    _eagle_prefill_tail_tokens,  # EAGLE预填充尾部token
    build_tree_kernel_efficient,  # 高效构建树内核
    organize_draft_results,  # 组织草稿结果
    per_step_draft_out_cache_loc,  # 每步草稿输出缓存位置
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 投机算法枚举
from sglang.srt.speculative.spec_utils import (  # 投机工具
    draft_tp_context,  # 草稿TP上下文
    generate_token_bitmask,  # 生成token位掩码
    load_token_map,  # 加载token映射
    record_stream_each,  # 记录流（逐个）
    record_stream_for_v2_verify,  # V2验证记录流
    select_top_k_tokens,  # 选择top-k token
)
from sglang.srt.utils.async_probe import (  # 异步探测
    maybe_detect_inf,  # 检测无穷
    maybe_detect_nan,  # 检测NaN
    maybe_detect_oob,  # 检测越界
)
from sglang.srt.utils.common import (  # 通用工具
    MultiprocessingSerializer,  # 多进程序列化器
    empty_context,  # 空上下文
    fast_topk,  # 快速topk
    get_available_gpu_memory,  # 获取可用GPU内存
    is_cuda,  # CUDA检测
    is_hip,  # HIP检测
    is_musa,  # MUSA检测
    is_npu,  # NPU检测
    log_info_on_rank0,  # rank0日志
    next_power_of_2,  # 下一个2的幂
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # torch reduction猴子补丁

_is_npu = is_npu()  # 是否为NPU
_is_cuda = is_cuda()  # 是否为CUDA
_is_musa = is_musa()  # 是否为MUSA
_is_hip = is_hip()  # 是否为HIP

logger = logging.getLogger(__name__)  # 获取日志记录器


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    # 获取规划流及其上下文管理器，用于重叠规划与计算
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():  # 启用重叠规划流
        plan_stream = torch.get_device_module(device).Stream()  # 创建新流
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)  # 创建流上下文
        return plan_stream, plan_stream_ctx  # 返回流和上下文
    else:  # 未启用
        return None, contextlib.nullcontext()  # 返回None和空上下文


class EagleDraftWorker(BaseDraftWorker):
    # EAGLE草稿Worker，负责草稿模型的多步自回归生成
    def __init__(
        self,
        server_args: ServerArgs,  # 服务器参数
        gpu_id: int,  # GPU ID
        tp_rank: int,  # 张量并行排名
        dp_rank: int,  # 数据并行排名
        moe_ep_rank: int,  # MoE专家并行排名
        attn_cp_rank: int,  # 注意力上下文并行排名
        moe_dp_rank: int,  # MoE数据并行排名
        nccl_port: int,  # NCCL端口
        target_worker: TpModelWorker,  # 目标Worker
    ):
        # copy args
        # 复制参数
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        # 便于访问的参数
        self.device = server_args.device  # 设备
        self.topk = server_args.speculative_eagle_topk  # top-k值
        self.speculative_num_steps = server_args.speculative_num_steps  # 投机步数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens  # 投机草稿token数
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )  # 投机算法

        # Pre-allocated constants for the topk=1 chain fast path in draft_forward.
        # 为draft_forward中topk=1链式快速路径预分配的常量。
        self._topk1_parents_prealloc = None  # topk=1父节点预分配
        self._topk1_score_indices_prealloc = None  # topk=1分数索引预分配
        self._rebuild_topk1_chain_buffers()  # 重建topk=1链式缓冲区

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        # 不在TpModelWorker初始化中捕获CUDA图，稍后通过init_cuda_graphs()捕获
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # 与目标Worker共享分配器。
        # Draft and target worker own their own KV cache pools.
        # 草稿和目标Worker各自拥有KV缓存池。
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Init draft worker
        # 初始化草稿Worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():  # DP注意力+EAGLE3
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # Init draft worker
            # 初始化草稿Worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism 投机worker不支持流水线并行
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

        # Alias for better readability
        # 别名以提高可读性
        self.draft_runner = self.draft_worker.model_runner  # 草稿运行器
        self.eagle_use_aux_hidden_state = False  # 是否使用辅助隐藏状态
        if self.speculative_algorithm.is_eagle3():  # EAGLE3算法
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        self.init_token_map()  # 初始化token映射
        self.init_lm_head()  # 初始化语言模型头

        # Init attention backend and cuda graphs
        # 初始化注意力后端和CUDA图
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()  # 初始化注意力后端
            if server_args.enable_breakable_cuda_graph:  # 可中断CUDA图
                self.draft_runner.init_piecewise_cuda_graphs(
                    force_for_draft_worker=True
                )
            self.init_cuda_graphs()  # 初始化CUDA图

        if (c := self.draft_runner.canary_manager) is not None:  # 金丝雀管理器
            c.mark_init_finished()  # 标记初始化完成

        self.tree_mask_mode = TreeMaskMode.FULL_MASK  # 树掩码模式

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)  # 规划流

    def _rebuild_topk1_chain_buffers(self) -> None:
        # 重建topk=1链式缓冲区（步数或草稿token数变化时调用）
        # For topk=1 the draft tree degenerates to a chain, so parent_list and
        # top_scores_index are runtime-invariant. Must be rebuilt after any
        # change to speculative_num_steps / speculative_num_draft_tokens.
        # topk=1时草稿树退化为链，parent_list和top_scores_index是运行时不变的。
        # 必须在speculative_num_steps/speculative_num_draft_tokens变化后重建。
        if self.topk != 1:
            return
        # _override_worker_state can set both directly, bypassing the hook that
        # pins this relation; the fast path is only valid when it holds.
        # _override_worker_state可以直接设置两者，绕过固定此关系的钩子；
        # 快速路径仅在关系成立时有效。
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        num_steps = self.speculative_num_steps
        sa = self.server_args
        max_bs = max(
            sa.cuda_graph_max_bs or 0,
            sa.max_running_requests or 0,
            1,
        )
        # A single-step chain has no parent entries (slow path drops the last
        # step). repeat (not expand): the kernel reads these as contiguous.
        # 单步链没有父条目（慢路径会丢弃最后一步）。
        # repeat（不是expand）：内核将这些作为连续读取。
        parent_width = num_steps if num_steps > 1 else 0
        self._topk1_parents_prealloc = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._topk1_score_indices_prealloc = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)

    def init_token_map(self):
        # 初始化热token映射
        # Load hot token ids
        # 加载热token ID
        if self.speculative_algorithm.is_eagle3():  # EAGLE3
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:  # 指定了token映射
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:  # 无映射
            self.hot_token_id = None

    def init_lm_head(self):
        # 初始化语言模型头（共享目标模型的嵌入和LM头）
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():  # EAGLE3
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            # 大多数EAGLE3模型不共享lm_head
            # 但某些模型（如nvidia/gpt-oss-120b-Eagle3）共享
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)  # 共享嵌入和头
            else:
                self.draft_runner.model.set_embed(embed)  # 仅共享嵌入

            # grab hot token ids
            # 获取热token ID
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:  # EAGLE（非EAGLE3）
            if self.hot_token_id is not None:
                head = head.clone()  # 克隆头
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]  # 只保留热token对应的权重

            # Share the embedding and lm_head
            # 共享嵌入和语言模型头
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # 初始化注意力后端
        # Create multi-step attn backends and cuda graph runners
        # 创建多步注意力后端和CUDA图运行器

        self.draft_extend_attn_backend = None  # 草稿扩展注意力后端

        draft_backend_factory = DraftBackendFactory(  # 草稿后端工厂
            self.server_args,
            self.draft_runner,
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

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        # 初始化CUDA图
        """Capture cuda graphs."""
        self.cuda_graph_runner = None  # 草稿CUDA图运行器
        self.cuda_graph_runner_for_draft_extend = None  # 草稿扩展CUDA图运行器

        if self.server_args.disable_cuda_graph:  # 禁用CUDA图
            return

        if self.server_args.model_impl == "mindspore":  # MindSpore不支持
            return

        Device2DraftCudaGraphRunner = {  # 设备到草稿图运行器映射
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        # 捕获草稿CUDA图
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)  # 创建草稿图运行器
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {  # 设备到扩展图运行器映射
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:  # HIP设备
            # Keep import local so non-HIP environments do not require aiter.
            # 保持局部导入，非HIP环境不需要aiter。
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and (  # CUDA/MUSA支持的后端
            isinstance(self.draft_extend_attn_backend, TritonAttnBackend)
            or isinstance(self.draft_extend_attn_backend, TRTLLMMLABackend)
            or isinstance(self.draft_extend_attn_backend, TokenspeedMLABackend)
        )
        # Capture extend
        # 捕获扩展CUDA图
        # TODO: support draft extend cuda graph for more attention backends
        # TODO: 支持更多注意力后端的草稿扩展CUDA图
        if self.draft_extend_attn_backend and (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)  # 创建扩展图运行器
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

    def draft(self, batch: ScheduleBatch):
        # 执行草稿阶段：生成草稿token并构建树掩码
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        n_inner = self.speculative_num_steps - 1  # 内部步数
        canary_outside_ctx = (  # 金丝雀外部上下文
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            # 运行草稿
            if can_cuda_graph:  # 可以使用CUDA图
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
            else:  # 不能使用CUDA图
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    # 跳过1步草稿的注意力后端初始化，
                    # draft_forward在此情况下仅做采样。
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

        if batch.forward_mode.is_idle():  # 空闲模式
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # 构建树掩码
        # Directly write to cuda graph buffers for verify attn
        # 直接写入验证注意力的CUDA图缓冲区
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        # build_tree_kernel uses seq_lens_sum only to size the (non-preallocated)
        # tree mask; over-size is safe. Skip per-iter .sum().item() D2H via UB.
        # build_tree_kernel仅使用seq_lens_sum来确定（非预分配）树掩码大小；
        # 过大是安全的。通过UB跳过每次迭代的.sum().item() D2H。
        seq_lens_sum = batch.seq_lens_sum
        if seq_lens_sum is None:
            if tree_mask_buf is None:
                max_context_len = (
                    self.target_worker.model_runner.attn_backend.max_context_len
                )
                seq_lens_sum = batch.seq_lens.shape[0] * max_context_len
            else:
                # tree_mask_buf preallocated -> kernel ignores seq_lens_sum.
                # tree_mask_buf已预分配 -> 内核忽略seq_lens_sum。
                seq_lens_sum = 0

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(  # 高效构建树内核
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(  # 返回验证输入
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
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # 执行草稿模型的多步前向传播
        # Parse args
        # 解析参数
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")  # 检测NaN

        if self.hot_token_id is not None:  # 有热token映射
            topk_index = self.hot_token_id[topk_index]  # 映射到实际token ID

        out_cache_loc = per_step_draft_out_cache_loc(  # 每步草稿输出缓存位置
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        # Return values
        # 返回值
        score_list: List[torch.Tensor] = []  # 分数列表
        token_list: List[torch.Tensor] = []  # token列表
        parents_list: List[torch.Tensor] = []  # 父节点列表

        # Forward multiple steps
        # 多步前向
        scores = None
        for i in range(self.speculative_num_steps):  # 遍历每一步
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )  # 选择top-k token
            score_list.append(tree_info[0])  # 添加分数
            token_list.append(tree_info[1])  # 添加token
            parents_list.append(tree_info[2])  # 添加父节点

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            # 不需要运行最后一次前向。从草稿预填充获得1个token，这里获得(spec_steps-1)个token
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            # 设置输入
            forward_batch.input_ids = input_ids
            # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
            # argument must be contiguous.
            # Qwen3-MoE MTP使用融合的RoPE + KV存储路径，cache_loc参数必须连续。
            if (
                self.draft_runner.model_config.hf_config.architectures[0]
                == "Qwen3MoeForCausalLMMTP"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            spec_info.hidden_states = hidden_states

            # Run forward under a per-step ForwardContext so the model layer
            # reads attn_backends[i] for the i-th draft step, plus a canary
            # index context so canary tracks which draft step is active.
            # 在每步ForwardContext下运行前向，使模型层为第i步读取attn_backends[i]，
            # 加上金丝雀索引上下文以跟踪活动的草稿步骤。
            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with forward_context(
                ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[i])
            ), canary_index_ctx:
                logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output  # 运行草稿前向
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")  # 检测NaN
            maybe_detect_inf(logits_output.next_token_logits, f"draft_forward step {i}")  # 检测无穷
            if self.topk == 1 and not _is_hip:  # topk=1快速路径（仅CUDA）
                # topk=1 → degenerate single-path tree; `topk_p` is unused
                # downstream, so skip softmax and just argmax over logits.
                # topk=1 → 退化的单路径树；topk_p下游不使用，因此跳过softmax直接argmax。
                # Gated to CUDA: on ROCm the argmax tie-break diverges from
                # the softmax+max path on FP8 logits and corrupts MTP draft
                # selection (DSV3.2 MTP GSM8K, see #26358).
                # 仅CUDA：在ROCm上argmax决胜与softmax+max路径在FP8 logits上分歧，
                # 会损坏MTP草稿选择。
                topk_index = torch.argmax(
                    logits_output.next_token_logits, dim=-1, keepdim=True
                )
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)
            else:  # topk>1或HIP
                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(  # 检测越界
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:  # 映射热token
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states  # 更新隐藏状态
            forward_batch.positions.add_(1)  # 递增位置

        # Organize the results
        # 组织结果
        if (
            self.topk == 1
            and token_list[0].shape[0] <= self._topk1_parents_prealloc.shape[0]
        ):
            # Chain topology: draft_tokens = concat of per-step tokens; the
            # full-length topk/sort/gather over score_list collapses to an
            # identity. parent_list and top_scores_index are runtime-invariant
            # constants pre-allocated on the worker. Oversized batches (rare,
            # would silently truncate the slice) fall through to the slow path.
            # 链式拓扑：draft_tokens = 每步token的拼接；对score_list的全长度
            # topk/sort/gather退化为恒等。parent_list和top_scores_index是运行时
            # 不变的常量，预分配在worker上。过大的批次（罕见，会静默截断切片）
            # 会落入慢路径。
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)  # 拼接所有步骤的token
            top_scores_index = self._topk1_score_indices_prealloc[:bs]  # 使用预分配索引
            parent_list = self._topk1_parents_prealloc[:bs]  # 使用预分配父节点
            return parent_list, top_scores_index, draft_tokens

        return organize_draft_results(  # 慢路径：组织草稿结果
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

    def draft_extend(self):
        # 草稿扩展（V2由_draft_extend_for_decode实现）
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,  # 调度批次
        target_hidden_states: torch.Tensor,  # 目标隐藏状态
        next_token_ids: torch.Tensor,  # 下一个token ID
        mm_input_embeds: Optional[torch.Tensor] = None,  # 多模态输入嵌入
    ):
        # 预填充阶段的草稿扩展：填充草稿KV缓存
        """
        Run draft model extend to correctly fill the KV cache.
        # 运行草稿模型扩展以正确填充KV缓存。

        Args:
        # 参数：
            batch: The batch to run.
            # 要运行的批次。
            target_hidden_states: Hidden states from the target model forward
            # 目标模型前向的隐藏状态
            next_token_ids: Next token ids generated from the target forward.
            # 目标前向生成的下一个token ID。
        """
        # Construct input_ids
        # 构建输入ID
        if not batch.forward_mode.is_idle():  # 非空闲模式
            # Chunked-prefill-aware tail tokens (see PR #26329).
            # 感知分块预填充的尾部token（见PR #26329）。
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):  # 遍历每个请求
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], tail_tokens[i].reshape(1))
                )  # 替换最后一个token
                pt += extend_len

        # Construct spec_info
        # 构建spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            bonus_tokens=next_token_ids,
            # draft mode is same with decode mode, only 1 token per req
            # 草稿模式与解码模式相同，每个请求只有1个token
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        batch.spec_info = next_draft_input

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # 运行前向（LAST模式：每个请求只有最终的隐藏状态，
        # 以馈送期望[bs, hidden_dim]的下一个草稿步骤）。
        # STANDALONE skips hidden states end-to-end.
        # STANDALONE端到端跳过隐藏状态。
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        batch.capture_hidden_mode = capture_hidden_mode
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)  # 初始化前向批次
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:  # 有多模态输入
            forward_batch.mm_input_embeds = mm_input_embeds

        canary_ctx = (  # 金丝雀上下文
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output  # 运行前向
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")  # 检测NaN
        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")  # 检测无穷

        # Update spec_info for the next draft step
        # 为下一步草稿更新spec_info
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )  # 计算top-k
        next_draft_input.hidden_states = logits_output.hidden_states  # 更新隐藏状态
        return next_draft_input  # 返回下一步草稿输入

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # 解码阶段的草稿扩展：将接受的token的KV缓存填充到草稿模型
        # Batch 2: Draft extend
        # 批次2：草稿扩展
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=self.speculative_num_steps + 1,
        )
        select_index = (  # 选择索引
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )

        # Prepare for draft extend in a separate stream
        # 在单独的流中准备草稿扩展
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )

        if self.plan_stream:  # 等待规划流完成
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        if forward_batch.spec_info.num_correct_drafts is None:  # 设置正确草稿数
            # `batch_result.accept_lens` already includes the bonus token, so use it
            # directly for `num_accept_tokens` and subtract 1 for `num_correct_drafts`.
            # batch_result.accept_lens已包含奖励token，因此直接用于num_accept_tokens，
            # 减1得到num_correct_drafts。
            forward_batch.spec_info.num_correct_drafts = batch_result.accept_lens - 1
            forward_batch.spec_info.num_accept_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        # 在主计算流中运行草稿扩展批次
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )

        canary_ctx = (  # 金丝雀上下文
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_cuda_graph:  # 使用CUDA图
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                    forward_batch
                )
            else:  # 不使用CUDA图
                draft_logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output

        maybe_detect_nan(  # 检测NaN
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )
        maybe_detect_inf(  # 检测无穷
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        # Reorganize the spec info for the next batch
        # 为下一批重组spec_info
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]  # 选择对应位置的logits
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]  # 选择对应位置的隐藏状态
        if self.topk == 1 and not _is_hip:  # topk=1快速路径（仅CUDA）
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            # 仅CUDA：见#26358 — ROCm的argmax决胜在FP8 logits上损坏MTP草稿选择。
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
        else:  # topk>1或HIP
            probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        # 构建返回值
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    # EAGLE V2 Worker，协调草稿和目标模型的投机解码流程
    def __init__(
        self,
        server_args: ServerArgs,  # 服务器参数
        gpu_id: int,  # GPU ID
        tp_rank: int,  # 张量并行排名
        dp_rank: Optional[int],  # 数据并行排名
        moe_ep_rank: int,  # MoE专家并行排名
        attn_cp_rank: int,  # 注意力上下文并行排名
        moe_dp_rank: int,  # MoE数据并行排名
        nccl_port: int,  # NCCL端口
        target_worker: TpModelWorker,  # 目标Worker
    ):
        # Parse arguments
        # 解析参数
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        # 覆盖章稿模型的上下文长度与目标模型相同。
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(  # 创建草稿Worker
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Adaptive speculative
        # 自适应投机
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:  # 启用自适应
            self.adaptive_controller = AdaptiveController(
                self, config_path=server_args.speculative_adaptive_config
            )

        # Some dummy tensors
        # 一些虚拟张量
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)  # 规划流

        # Build adaptive runtime states (must be after draft worker is fully initialized)
        # 构建自适应运行时状态（必须在草稿Worker完全初始化之后）
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(  # 注册初始状态
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states()  # 初始化所有状态

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # 获取spec_v2前向涉及的所有注意力后端
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        # spec_v2前向触及的每个注意力后端；由decide_needs_cpu_seq_lens消费
        # 以门控seq_lens_cpu D2H。
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend,
        )

    @property
    def target_worker(self):
        # 获取目标Worker
        return self._target_worker

    @property
    def draft_worker(self):
        # 获取草稿Worker
        return self._draft_worker

    def clear_cache_pool(self):
        # 清除缓存池（与目标Worker共享，由调度器清除）
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        # EAGLE V2前向批次生成：协调预填充、草稿、验证和扩展四个阶段
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:  # 预填充/扩展模式
            # Target prefill
            # 目标预填充
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(batch)

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # spec_v2约定：batch.seq_lens = 本次迭代token之前的长度。
            # Extend processed L prompt tokens; next verify iter expects same L.
            # 扩展处理了L个提示token；下一次验证迭代期望相同的L。
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft_extend so the fence is at target-end.
            # 在draft_extend之前发布，以便围栏在target-end。
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Draft prefill
            # 草稿预填充
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:  # 解码模式
            if batch.spec_info is None:  # 无spec_info（首次解码）
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=EagleDraftInput.hidden_size_for(self.draft_worker),
                    dtype=EagleDraftInput.dtype_for(self.draft_worker),
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                )
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                verify_input: EagleVerifyInput = self.draft_worker.draft(batch)  # 草稿阶段
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch)  # 验证阶段
            # Publish before draft_extend so the fence is at verify-end.
            # 在draft_extend之前发布，以便围栏在verify-end。
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.draft_worker._draft_extend_for_decode(batch, batch_output)  # 扩展阶段

            return batch_output

    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        # 验证完成回调，更新自适应控制器
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req)

    # -- Adaptive speculative decoding protocol --
    # -- 自适应投机解码协议 --

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState:
        # 为给定的步数配置构建运行时状态
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps, speculative_num_draft_tokens
        ):
            self._draft_worker.init_attention_backend()  # 初始化注意力后端
            self._draft_worker.init_cuda_graphs()  # 初始化CUDA图

            # Build target attention backend and CUDA graph runner
            # 构建目标注意力后端和CUDA图运行器
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not self.server_args.disable_cuda_graph:  # 启用CUDA图
                TargetGraphRunnerCls = NPUGraphRunner if _is_npu else CudaGraphRunner
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            state = SpecRuntimeState(  # 创建运行时状态
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        # 应用预构建的运行时状态到Worker
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:  # 步数相同无需切换
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # Top-level
        # 顶层
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        # 草稿侧
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()  # 重建topk=1缓冲区

        # Target side
        # 目标侧
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.graph_runner = state.target_graph_runner

        # Sync server_args
        # 同步服务器参数
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        # 临时覆盖Worker状态用于图捕获
        """Temporarily override server_args and worker attributes for graph capture."""
        sa = self.server_args
        dw = self._draft_worker
        backup = (  # 备份所有需要覆盖的属性
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
        )

        self.speculative_num_steps = speculative_num_steps  # 覆盖
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw._rebuild_topk1_chain_buffers()  # 重建缓冲区

        try:
            yield  # 执行图捕获
        finally:
            (  # 恢复所有属性
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
            ) = backup
            dw._rebuild_topk1_chain_buffers()  # 恢复缓冲区

    def verify(self, batch: ScheduleBatch):
        # 验证阶段：运行目标模型验证草稿token
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)  # 记录流

        verify_input.num_tokens_per_req = self.speculative_num_steps + 1  # 每请求token数
        bs = len(batch.seq_lens)  # 批次大小

        # Batch 1: Target verify
        # 批次1：目标验证
        # Prepare for target verify in a separate stream
        # 在单独的流中准备目标验证
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Cover post-prepare rebinds: draft_token, plan_stream-allocated out_cache_loc.
        # 覆盖准备后重新绑定：draft_token、plan_stream分配的out_cache_loc。
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        # Correct some buffers due to the overlap plan
        # 由于重叠规划，修正一些缓冲区
        if self.plan_stream:  # 有规划流
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )  # 等待规划流
            if (
                _is_npu
                and self._target_worker.model_runner.model_is_mrope
                and batch.spec_info is not None
                and getattr(batch.spec_info, "positions", None) is not None
                and not batch.forward_mode.is_idle()
            ):
                # mrope_position depends on draft output in default stream and is computed in plan stream,
                # causing errors. Compute it here for correct values.
                # mrope_position依赖于默认流中的草稿输出，但在规划流中计算，导致错误。
                # 在这里计算以获得正确的值。
                verify_forward_batch.compute_spec_mrope_positions(
                    self._target_worker.model_runner, batch
                )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            # 某些值如custom_mask和position依赖于草稿的输出，因此上一次规划步骤使用了错误的值。
            # 这里需要重新运行相关计算以更新为正确的值。
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Prepare grammar data on CPU if needed
        # 如果需要，在CPU上准备语法数据
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute).
        # 在主计算流中运行目标验证批次（GPU计算）。
        # Only skip metadata init when cuda-graph already ran replay_prepare;
        # the non-cuda-graph path needs forward_extend's init (post-pad).
        # 仅在CUDA图已运行replay_prepare时跳过元数据初始化；
        # 非CUDA图路径需要forward_extend的初始化（后填充）。
        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=can_run_cuda_graph,
        )
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        # 为约束解码生成词表掩码
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # 为结构化输出生成logit掩码。
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                # 注意：否则此词表掩码将是上一次扩展阶段的，会被错误应用
                batch.sampling_info.vocab_mask = None

        # Sample
        # 采样
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")  # 检测NaN
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")  # 检测无穷
        (
            predict,
            accept_lens,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)  # 采样验证
        new_seq_lens = batch.seq_lens + accept_lens  # 新序列长度

        # Update mamba state for hybrid GDN models after verification
        # 验证后更新混合GDN模型的Mamba状态
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_lens, accept_index, bs
            )

        if not batch.forward_mode.is_idle():  # 非空闲模式
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            fill_bonus_tokens[(bs,)](  # 填充奖励token
                accept_tokens,
                accept_lens,
                bonus_tokens,
                self.speculative_num_draft_tokens,
            )
        else:  # 空闲模式
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():  # 计算logprob
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)  # 下一步草稿输入

        # verify_forward_batch transitively holds verify-time GPU tensors
        # (draft_token / out_cache_loc / ...) that must outlive the imminent
        # batch.input_ids rebind in prepare_for_extend_to_fill_draft_kvcache.
        # Scheduler pins it in batch_record_buf for the 2-iter window.
        # verify_forward_batch间接持有验证时GPU张量（draft_token/out_cache_loc/...），
        # 这些必须比即将在prepare_for_extend_to_fill_draft_kvcache中的
        # batch.input_ids重绑定存活更久。调度器在batch_record_buf中固定它2个迭代窗口。
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,  # 调度批次
        verify_input: EagleVerifyInput,  # 验证输入
        accept_lens: torch.Tensor,  # 接受长度
        accept_index: torch.Tensor,  # 接受索引
        bs: int,  # 批次大小
    ):
        # 验证后更新混合GDN模型的Mamba状态
        """Update mamba state for hybrid GDN models after verification."""
        # `accept_lens` already includes the bonus token (drafts + 1 per req).
        # accept_lens已包含奖励token（每个请求drafts + 1）。
        if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
            if verify_input.topk != 1:  # 目前仅支持topk=1
                raise ValueError("Spec v2 currently only supports topk = 1.")

            accepted_indices_offset = torch.arange(  # 接受索引偏移
                0,
                bs * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=accept_lens.dtype,
                device=accept_lens.device,
            )
            last_correct_step_indices = accept_lens - 1  # 最后正确步骤索引

            if batch.mamba_track_indices is not None:  # 有Mamba跟踪索引
                # If after verify, the request's seq_lens has crossed a mamba track interval,
                # we need to update the mamba state for the request at the crossing point.
                # 如果验证后请求的seq_lens跨越了Mamba跟踪间隔，
                # 需要在跨越点更新请求的Mamba状态。
                seq_lens_pre_verify = batch.seq_lens
                seq_lens_post_verify = batch.seq_lens + accept_lens
                mamba_track_interval = self.server_args.mamba_track_interval
                to_track_mask = (
                    seq_lens_pre_verify // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify // mamba_track_interval * mamba_track_interval
                )
                to_track_ith = torch.clamp(
                    tracking_point - seq_lens_pre_verify - 1, min=0
                ).to(torch.int64)
                req_idx = torch.arange(
                    bs,
                    dtype=torch.int64,
                    device=accept_lens.device,
                )
                candidate_track_steps = (
                    accept_index[req_idx, to_track_ith] - accepted_indices_offset
                )
                mamba_steps_to_track = torch.where(
                    to_track_mask,
                    candidate_track_steps,
                    torch.full_like(candidate_track_steps, -1),
                )
            else:  # 无跟踪索引
                mamba_steps_to_track = None

            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=self.target_worker.model_runner.model,
            )  # 更新Mamba状态

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ScheduleBatch,  # 调度批次
        accept_index: torch.Tensor,  # 接受索引
        num_correct_drafts: torch.Tensor,  # 正确草稿数
    ):
        # 将接受的token移动到目标KV缓存
        """
        Move accepted tokens (drafts + bonus) to the target KV cache.
        # 将接受的token（草稿+奖励）移动到目标KV缓存。

        Args:
        # 参数：
            batch: The batch to run.
            # 要运行的批次。
            accept_index: The index of the accepted tokens (incl. bonus).
            # 接受token的索引（包括奖励）。
            num_correct_drafts: Per-req count of correct drafts (excludes bonus);
                seq_lens is advanced by ``num_correct_drafts + 1`` to cover the bonus slot.
            # 每个请求的正确草稿计数（不包括奖励）；
            # seq_lens前进num_correct_drafts + 1以覆盖奖励槽位。
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        # fill_accepted_out_cache_loc reads out_cache_loc[accept_index]; -1 sentinel ok.
        # fill_accepted_out_cache_loc读取out_cache_loc[accept_index]；-1哨兵值可以。
        maybe_detect_oob(
            accept_index,
            -1,
            batch.out_cache_loc.size(0),
            "eagle v2 move_accepted_tokens accept_index",
        )

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](  # 分配扩展缓存位置
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](  # 填充接受输出缓存位置
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )  # 移动KV缓存

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        # 从磁盘更新模型权重
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        # 通过IPC更新模型权重
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        # 通过张量更新模型权重
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
