# EAGLE草稿扩展CUDA图运行器模块
# 为EAGLE投机解码的草稿扩展阶段提供CUDA图捕获和重放功能，
# 在验证后将接受的token的KV缓存填充到草稿模型中。
from __future__ import annotations  # 启用延迟注解求值

import bisect  # 导入二分查找模块
import contextlib  # 导入上下文管理工具
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Callable, Optional  # 导入类型注解

import torch  # 导入PyTorch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len  # 导入DP注意力工具
from sglang.srt.model_executor.cuda_graph_runner import (  # 导入CUDA图运行器相关
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    LogitsProcessorOutput,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context  # 导入前向上下文
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers  # 导入前向输入缓冲区
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput  # 导入EAGLE草稿扩展输入
from sglang.srt.speculative.spec_utils import fast_topk  # 导入快速topk工具
from sglang.srt.utils import (  # 导入工具函数
    is_hip,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

_is_hip = is_hip()  # 是否为HIP（AMD ROCm）设备

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


@dataclass  # 数据类装饰器
class EagleDraftExtendInputBuffers(ForwardInputBuffers):
    # EAGLE草稿扩展CUDA图的输入缓冲区
    input_ids: torch.Tensor  # 输入ID
    req_pool_indices: torch.Tensor  # 请求池索引
    out_cache_loc: torch.Tensor  # 输出缓存位置
    positions: torch.Tensor  # 位置编码
    mrope_positions: torch.Tensor  # 多维RoPE位置
    hidden_states: Optional[torch.Tensor]  # 隐藏状态
    seq_lens: torch.Tensor  # 序列长度
    seq_lens_cpu: torch.Tensor  # CPU上的序列长度
    extend_seq_lens: torch.Tensor  # 扩展序列长度
    num_correct_drafts: torch.Tensor  # 正确草稿数
    num_accept_tokens: torch.Tensor  # 接受token数
    next_token_logits_buffer: torch.Tensor  # 下一个token的logits缓冲区
    global_num_tokens_gpu: Optional[torch.Tensor]  # GPU上的全局token数
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]  # GPU上用于logprob的全局token数


class EAGLEDraftExtendCudaGraphRunner:
    # EAGLE草稿扩展CUDA图运行器，捕获和重放草稿扩展阶段的CUDA图
    def __init__(
        self,
        eagle_worker: EAGLEWorker,  # EAGLE Worker
        *,
        draft_extend_attn_backend=None,  # 草稿扩展注意力后端
        speculative_num_steps: Optional[int] = None,  # 投机步数
    ):
        # Parse args
        # 解析参数
        self.eagle_worker = eagle_worker  # EAGLE Worker引用
        if not hasattr(eagle_worker, "model_runner"):  # V2版本
            # V2: EagleDraftWorker
            self.model_runner = model_runner = eagle_worker.draft_runner  # 使用草稿运行器
            self.forward_mode = ForwardMode.DRAFT_EXTEND_V2  # V2扩展模式
        else:  # V1版本
            self.model_runner = model_runner = eagle_worker.model_runner  # 使用模型运行器
            self.forward_mode = ForwardMode.DRAFT_EXTEND  # V1扩展模式

        self.graphs = {}  # 批次大小到CUDA图的映射
        self.output_buffers = {}  # 批次大小到输出缓冲区的映射
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile  # 是否启用torch编译
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding  # 是否禁用填充
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)  # 是否需要收集缓冲区
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)  # 是否需要MLP TP收集
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)  # 是否需要MLP同步
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)  # 是否需要注意力TP收集
        self.tp_size = self.model_runner.tp_size  # 张量并行大小
        self.dp_size = self.model_runner.dp_size  # 数据并行大小
        self.speculative_num_steps = (  # 投机步数
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.topk = model_runner.server_args.speculative_eagle_topk  # top-k值
        self.draft_extend_attn_backend = (  # 草稿扩展注意力后端
            draft_extend_attn_backend or eagle_worker.draft_extend_attn_backend
        )
        self.enable_profile_cuda_graph = (  # 是否启用CUDA图分析
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False  # 不启用PDMUX
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()  # DeepEP适配器

        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)  # 要捕获的批次大小
        self.padded_static_len = -1  # 填充的静态长度

        # Attention backend
        # 注意力后端
        self.num_tokens_per_bs = self.speculative_num_steps + 1  # 每个批次的token数
        self.max_bs = max(self.capture_bs)  # 最大批次大小
        self.max_num_token = self.max_bs * self.num_tokens_per_bs  # 最大token数

        self.draft_extend_attn_backend.init_cuda_graph_state(  # 初始化CUDA图状态
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = (  # 序列长度填充值
            self.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        seq_lens_cpu = torch.full(  # CPU序列长度
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs  # 扩展序列长度列表

        if self.enable_torch_compile:  # 启用torch编译
            set_torch_compile_config()

        # Graph inputs
        # 图输入
        with torch.device(model_runner.device):  # 在设备上创建张量
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 输入ID
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)  # 请求池索引
            out_cache_loc = torch.ones(  # 输出缓存位置（初始化为1以避免0索引问题）
                (self.max_num_token,), dtype=self._cache_loc_dtype()
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 位置
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)  # 多维RoPE位置

            _hidden_size = EagleDraftExtendInput.hidden_size_for(self.eagle_worker)  # 隐藏层大小
            hidden_states = (  # 隐藏状态
                torch.zeros(
                    (self.max_num_token, _hidden_size),
                    dtype=EagleDraftExtendInput.dtype_for(self.eagle_worker),
                )
                if _hidden_size is not None
                else None
            )
            self.seq_len_fill_value = (  # 从模型注意力后端获取填充值
                self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
            )
            seq_lens = torch.full(  # 序列长度
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            extend_seq_lens = torch.full(  # 扩展序列长度
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_correct_drafts = torch.full(  # 正确草稿数
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_accept_tokens = torch.full(  # 接受token数
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )

            if self.require_gathered_buffer:  # 需要收集缓冲区
                if self.require_mlp_tp_gather:  # MLP TP收集
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:  # 注意力TP收集
                    assert self.require_attn_tp_gather
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:  # 不需要
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

            if hasattr(  # llama_eagle模型
                self.model_runner.model_config.hf_config, "draft_vocab_size"
            ):
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size  # 使用草稿词表大小
            elif hasattr(  # llama_eagle3模型
                self.model_runner.model_config.hf_config, "hot_vocab_size"
            ):
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size  # 使用热词表大小
            else:  # 默认
                vocab_size = self.model_runner.model_config.vocab_size  # 使用完整词表大小

            next_token_logits_buffer = torch.zeros(  # logits缓冲区
                (
                    (
                        self.max_bs * self.num_tokens_per_bs
                        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2
                        else self.max_bs
                    ),
                    vocab_size,
                ),
                dtype=torch.float,
            )

        self.buffers = EagleDraftExtendInputBuffers(  # 创建输入缓冲区
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
            next_token_logits_buffer=next_token_logits_buffer,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        self.buffers.share_buffers()  # 共享缓冲区

        # Capture
        # 捕获CUDA图
        try:
            with model_capture_mode():  # 模型捕获模式
                self.capture()
        except RuntimeError as e:  # 捕获失败
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        # 检查CUDA图是否可以运行给定批次
        if self.require_mlp_tp_gather:  # MLP TP收集模式
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:  # 普通模式
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (  # 批次大小是否支持
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:  # 需要MLP同步
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def _create_graph(self):
        # 创建CUDA图对象
        return torch.cuda.CUDAGraph()

    def _cache_loc_dtype(self):
        # 返回缓存位置的数据类型
        return torch.int64

    def _capture_init(self, run_once_fn):
        # 捕获前初始化：预热运行
        for _ in range(2):  # 预热2次
            torch.cuda.synchronize()  # 同步CUDA
            self.model_runner.tp_group.barrier()  # TP组屏障
            run_once_fn()  # 运行一次

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # 捕获CUDA图
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()  # 运行一次
        return out  # 返回输出

    def _replay(self, forward_batch: ForwardBatch):
        # 重放CUDA图
        ctx = (  # 计时上下文
            self.model_runner.device_timer.wrap(
                metadata={"category": "eagle_draft_extend"}
            )
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:  # 在计时上下文中重放
            self.graphs[self.bs].replay()  # 重放图

    def capture(self):
        # 捕获所有批次大小的CUDA图
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(self, bs: int, forward: Callable, stream_idx: int = 0):
        # 捕获指定批次大小的CUDA图
        buffers = self.buffers  # 输入缓冲区
        graph = self._create_graph()  # 创建图
        stream = self.stream  # CUDA流
        num_tokens = bs * self.num_tokens_per_bs  # token数

        # Graph inputs
        # 图输入
        input_ids = buffers.input_ids[:num_tokens]  # 输入ID
        req_pool_indices = buffers.req_pool_indices[:bs]  # 请求池索引
        seq_lens = buffers.seq_lens[:bs]  # 序列长度
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]  # CPU序列长度
        extend_seq_lens = buffers.extend_seq_lens[:bs]  # 扩展序列长度
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]  # CPU扩展序列长度
        out_cache_loc = buffers.out_cache_loc[:num_tokens]  # 输出缓存位置
        positions = buffers.positions[:num_tokens]  # 位置
        mrope_positions = buffers.mrope_positions[:, :num_tokens]  # 多维RoPE位置
        hidden_states = (  # 隐藏状态
            buffers.hidden_states[:num_tokens]
            if buffers.hidden_states is not None
            else None
        )
        num_correct_drafts = buffers.num_correct_drafts[:bs]  # 正确草稿数
        num_accept_tokens = buffers.num_accept_tokens[:bs]  # 接受token数
        next_token_logits_buffer = buffers.next_token_logits_buffer[  # logits缓冲区
            : bs if self.forward_mode == ForwardMode.DRAFT_EXTEND else num_tokens
        ]

        # V1 (DRAFT_EXTEND): pruned_states = bs (last token per seq)
        # V1 (DRAFT_EXTEND): pruned_states = bs（每个序列最后一个token）
        # V2 (DRAFT_EXTEND_V2): pruned_states = num_tokens (all tokens)
        # V2 (DRAFT_EXTEND_V2): pruned_states = num_tokens（所有token）
        num_tokens_for_logprob = (
            num_tokens if self.forward_mode.is_draft_extend_v2() else bs
        )

        if self.require_mlp_tp_gather:  # MLP TP收集
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens_for_logprob] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:  # 注意力TP收集
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens_for_logprob],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:  # 不需要收集
            global_dp_buffer_len = None

        spec_info = EagleDraftExtendInput(  # 创建草稿扩展输入
            hidden_states=hidden_states,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
        )

        # Forward batch
        # 前向批次
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            padded_static_len=self.padded_static_len,
        )

        def run_once():
            # 运行一次草稿扩展前向
            if self.model_runner.is_hybrid_swa:  # 混合滑动窗口注意力
                self.model_runner.token_to_kv_pool.invalidate_loc_cache()

            # Clean intermediate result cache for DP attention
            # 清除DP注意力的中间结果缓存
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(  # 设置DP缓冲区长度
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)  # 标记非扩展

            # Backup two fields, which will be modified in-place in `draft_forward`.
            # 备份两个字段，将在draft_forward中被原地修改。
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.model_runner.model.forward(  # 运行模型前向
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            # ROCm's argmax tie-breaks differently from CUDA's softmax+max
            # path on FP8 logits, which corrupts MTP draft selection on AMD.
            # Keep the fastpath CUDA-only.
            # ROCm的argmax在FP8 logits上的决胜规则与CUDA的softmax+max路径不同，
            # 会损坏AMD上的MTP草稿选择。保持快速路径仅CUDA可用。
            if self.topk == 1 and not _is_hip:  # topk=1且非HIP
                ret.topk_index = torch.argmax(  # 直接argmax
                    ret.next_token_logits, dim=-1, keepdim=True
                )
                ret.topk_p = torch.ones_like(ret.topk_index, dtype=torch.float32)  # 概率设为1
            else:  # topk>1或HIP
                probs = torch.softmax(ret.next_token_logits, dim=-1)  # softmax
                ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk

            forward_batch.out_cache_loc = output_cache_loc_backup  # 恢复
            forward_batch.spec_info.hidden_states = hidden_states_backup  # 恢复
            return ret  # 返回结果

        with forward_context(  # 前向上下文
            ForwardContext(attn_backend=self.draft_extend_attn_backend)
        ):
            self.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
                bs=bs,
                num_tokens=num_tokens,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                encoder_lens=None,
                forward_mode=self.forward_mode,
                spec_info=spec_info,
            )  # 初始化前向元数据
            self.deepep_adapter.capture(is_extend_in_batch=True)  # DeepEP捕获

            canary_ctx = (  # 金丝雀上下文
                c.with_active_single_forward_manager(0)
                if (c := self.model_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with canary_ctx:  # 在金丝雀上下文中
                self._capture_init(run_once)  # 预热

                out = self._capture_graph(  # 捕获图
                    graph, get_global_graph_memory_pool(), stream, run_once
                )

        set_global_graph_memory_pool(graph.pool())  # 设置全局图内存池
        return graph, out  # 返回图和输出

    def replay(self, forward_batch: ForwardBatch):
        # 重放CUDA图以执行草稿扩展前向
        assert forward_batch.out_cache_loc is not None  # 断言输出缓存位置存在
        self.deepep_adapter.replay()  # DeepEP重放
        buffers = self.buffers  # 输入缓冲区

        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        # 批次中可能有已完成的示例不被计入num_seqs，因此batch_size和num_seqs可能不同
        raw_bs = forward_batch.batch_size  # 实际批次大小
        num_tokens = forward_batch.input_ids.shape[0]  # 实际token数
        if self.require_mlp_tp_gather:  # MLP TP收集
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:  # 普通模式
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]  # 找到最近的捕获批次大小
        if bs * self.num_tokens_per_bs != num_tokens:  # 需要填充
            buffers.seq_lens.fill_(self.seq_len_fill_value)  # 填充序列长度
            buffers.out_cache_loc.zero_()  # 清零输出缓存位置
            buffers.positions.zero_()  # 清零位置
            # Pair with seq_lens fill: padded rows must point at reserved
            # req_pool slot 0 (req_to_token[0, :] is all zeros from init).
            # 与序列长度填充配对：填充行必须指向保留的req_pool槽0
            buffers.req_pool_indices.zero_()  # 清零请求池索引
            buffers.num_correct_drafts.fill_(self.num_tokens_per_bs)  # 填充正确草稿数
            buffers.num_accept_tokens.fill_(self.num_tokens_per_bs)  # 填充接受token数
            buffers.extend_seq_lens.fill_(self.num_tokens_per_bs)  # 填充扩展序列长度

        # Common inputs
        # 通用输入
        buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)  # 复制输入ID
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)  # 复制序列长度
        if forward_batch.extend_seq_lens is not None:  # 有扩展序列长度
            buffers.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)
        else:  # 无扩展序列长度
            buffers.extend_seq_lens[:raw_bs].fill_(self.num_tokens_per_bs)  # 填充默认值
        buffers.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)  # 复制输出缓存位置
        buffers.positions[:num_tokens].copy_(forward_batch.positions)  # 复制位置
        if (  # 复制隐藏状态（维度匹配时）
            buffers.hidden_states is not None
            and forward_batch.spec_info.hidden_states is not None
            and forward_batch.spec_info.hidden_states.shape[1]
            == buffers.hidden_states.shape[1]
        ):
            buffers.hidden_states[:num_tokens].copy_(
                forward_batch.spec_info.hidden_states
            )
        if forward_batch.spec_info.num_correct_drafts is not None:  # 复制正确草稿数和接受token数
            buffers.num_correct_drafts[:raw_bs].copy_(
                forward_batch.spec_info.num_correct_drafts
            )
            buffers.num_accept_tokens[:raw_bs].copy_(
                forward_batch.spec_info.num_accept_tokens
            )
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)  # 复制请求池索引

        # TODO(ch-wan): support num_token_non_padded
        # TODO(ch-wan): 支持num_token_non_padded
        if self.require_gathered_buffer:  # 需要收集缓冲区
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            # V1: pruned_states = bs; V2: pruned_states = num_tokens
            # V1: pruned_states = bs; V2: pruned_states = num_tokens
            if self.forward_mode.is_draft_extend_v2():
                buffers.global_num_tokens_for_logprob_gpu.fill_(
                    bs * self.num_tokens_per_bs
                )
            else:
                buffers.global_num_tokens_for_logprob_gpu.fill_(bs)

        if forward_batch.seq_lens_cpu is not None:  # 处理CPU序列长度
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        if forward_batch.extend_seq_lens_cpu is not None:  # 有CPU扩展序列长度
            self.extend_seq_lens_cpu[:raw_bs] = forward_batch.extend_seq_lens_cpu
        else:  # 无CPU扩展序列长度
            self.extend_seq_lens_cpu[:raw_bs] = [self.num_tokens_per_bs] * raw_bs
        if bs > raw_bs:  # 需要填充扩展序列长度
            self.extend_seq_lens_cpu[raw_bs:bs] = [self.num_tokens_per_bs] * (
                bs - raw_bs
            )
        forward_batch.spec_info.extend_seq_lens_cpu = list(
            self.extend_seq_lens_cpu[:bs]
        )
        forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]

        if bs != raw_bs:  # 需要填充时更新spec_info
            forward_batch.spec_info.positions = buffers.positions[:num_tokens]
            forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[:bs]
            forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[:bs]

        seq_lens_sum = forward_batch.seq_lens_sum  # 序列长度总和
        if seq_lens_sum is not None:
            seq_lens_sum = seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value  # 调整填充
        self.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu,
        )  # 初始化重放元数据

        # Replay
        # 重放
        self.raw_bs = raw_bs  # 保存原始批次大小
        self.bs = bs  # 保存填充后批次大小
        self._replay(forward_batch)  # 执行重放
        out = self.output_buffers[bs]  # 获取输出

        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:  # V2模式
            # DRAFT_EXTEND_V2: all tokens calculations whether accepted or not.
            # DRAFT_EXTEND_V2: 所有token无论是否接受都参与计算。
            unpadding_bs = num_tokens
        elif bs != raw_bs:  # V1模式需要截断
            forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[
                :raw_bs
            ]
            forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[
                :raw_bs
            ]
            unpadding_bs = raw_bs
        else:  # 无需截断
            unpadding_bs = None

        if unpadding_bs is not None:  # 需要去填充
            out_copy = out  # 保存原始输出
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:unpadding_bs],  # 截断logits
                hidden_states=out.hidden_states[:unpadding_bs],  # 截断隐藏状态
            )
            out.topk_p = out_copy.topk_p[:unpadding_bs]  # 截断top-k概率
            out.topk_index = out_copy.topk_index[:unpadding_bs]  # 截断top-k索引
        return out  # 返回输出
