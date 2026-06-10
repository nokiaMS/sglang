# Frozen-KV MTP CUDA图运行器模块
# 为Frozen-KV MTP循环草稿步骤提供CUDA图捕获和重放功能，
# 使用冻结的目标KV池运行草稿模型。
from __future__ import annotations  # 启用延迟注解求值

import bisect  # 导入二分查找模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Callable, Optional  # 导入类型注解

import torch  # 导入PyTorch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len  # 导入DP注意力工具
from sglang.srt.model_executor.cuda_graph_runner import (  # 导入CUDA图运行器相关
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
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
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput  # 导入Frozen-KV MTP草稿输入
from sglang.srt.utils import (  # 导入工具函数
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker


@dataclass  # 数据类装饰器
class FrozenKVMTPInputBuffers(ForwardInputBuffers):
    # Frozen-KV MTP CUDA图的输入缓冲区
    req_pool_indices: torch.Tensor  # 请求池索引
    positions: torch.Tensor  # 位置编码
    mrope_positions: torch.Tensor  # 多维RoPE位置
    seq_lens: torch.Tensor  # 序列长度
    seq_lens_cpu: torch.Tensor  # CPU上的序列长度
    topk_p: torch.Tensor  # top-k概率
    topk_index: torch.Tensor  # top-k索引
    hidden_states: torch.Tensor  # 隐藏状态
    # Consumed by the captured seed iter; see `FrozenKVMTPWorker.draft_forward`.
    # 被捕获的种子迭代消费；见FrozenKVMTPWorker.draft_forward。
    bonus_tokens: torch.Tensor  # 奖励token
    global_num_tokens_gpu: Optional[torch.Tensor]  # GPU上的全局token数
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]  # GPU上用于logprob的全局token数


class FrozenKVMTPCudaGraphRunner:
    # Frozen-KV MTP循环草稿步骤的CUDA图运行器
    """CUDA graph runner for the Frozen-KV MTP recurrent draft-loop step."""

    def __init__(self, frozen_kv_mtp_worker: FrozenKVMTPWorker):
        # 初始化Frozen-KV MTP CUDA图运行器
        self.frozen_kv_mtp_worker = frozen_kv_mtp_worker  # Frozen-KV MTP Worker引用
        self.model_runner = model_runner = frozen_kv_mtp_worker.draft_model_runner  # 草稿模型运行器
        self.graphs = {}  # 批次大小到CUDA图的映射
        self.output_buffers = {}  # 批次大小到输出缓冲区的映射
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile  # 是否启用torch编译
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding  # 是否禁用填充
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)  # 是否需要收集缓冲区
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)  # MLP TP收集
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)  # MLP同步
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)  # 注意力TP收集
        self.tp_size = self.model_runner.tp_size  # 张量并行大小
        self.dp_size = self.model_runner.dp_size  # 数据并行大小
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps  # 投机步数
        self.topk = model_runner.server_args.speculative_eagle_topk  # top-k值
        self.draft_attn_backend = frozen_kv_mtp_worker.draft_attn_backend  # 草稿注意力后端
        self.enable_profile_cuda_graph = (  # CUDA图分析
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False  # 不启用PDMUX
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()  # DeepEP适配器

        self.num_tokens_per_bs = self.topk  # 每个批次的token数
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(  # 要捕获的批次大小
            model_runner, self.num_tokens_per_bs
        )
        self.max_bs = max(self.capture_bs)  # 最大批次大小
        self.max_num_token = self.max_bs * self.num_tokens_per_bs  # 最大token数

        self.draft_attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)  # 初始化CUDA图状态
        self.seq_len_fill_value = (  # 序列长度填充值
            self.draft_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        seq_lens_cpu = torch.full(  # CPU序列长度
            (self.max_num_token,), self.seq_len_fill_value, dtype=torch.int32
        )

        if self.enable_torch_compile:  # 启用torch编译
            set_torch_compile_config()

        with torch.device(model_runner.device):  # 在设备上创建张量
            req_pool_indices = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 请求池索引
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 位置
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)  # 多维RoPE位置
            seq_lens = torch.full(  # 序列长度
                (self.max_num_token,), self.seq_len_fill_value, dtype=torch.int32
            )
            topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)  # top-k概率
            topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)  # top-k索引
            hidden_states = torch.zeros(  # 隐藏状态
                (self.max_bs, frozen_kv_mtp_worker._recurrent_hidden_size),
                dtype=self.model_runner.dtype,
            )
            bonus_tokens = torch.zeros((self.max_bs,), dtype=torch.int64)  # 奖励token

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

        self.buffers = FrozenKVMTPInputBuffers(  # 创建输入缓冲区
            req_pool_indices=req_pool_indices,
            positions=positions,
            mrope_positions=mrope_positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            bonus_tokens=bonus_tokens,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        self.buffers.share_buffers()  # 共享缓冲区

        try:
            with model_capture_mode():  # 模型捕获模式
                self.capture()  # 捕获CUDA图
        except RuntimeError as e:  # 捕获失败
            raise Exception(
                f"Capture frozen-KV MTP cuda graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        # 检查CUDA图是否可以运行给定批次
        if self.require_mlp_tp_gather:  # MLP TP收集模式
            cuda_graph_bs = max(forward_batch.global_num_tokens_cpu) // (
                self.topk * self.topk
            )
        else:  # 普通模式
            cuda_graph_bs = (
                forward_batch.batch_size // self.topk
                if self.topk > 1
                else forward_batch.batch_size
            )

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

    def _replay(self):
        # 重放CUDA图
        self.graphs[self.bs].replay()  # 重放

    def capture(self):
        # 捕获所有批次大小的CUDA图
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        # 捕获指定批次大小的CUDA图
        del forward, stream_idx  # 未使用
        buffers = self.buffers  # 输入缓冲区
        graph = self._create_graph()  # 创建图
        stream = self.stream  # CUDA流
        request_bs = num_seqs  # 请求批次大小
        expanded_bs = request_bs * self.num_tokens_per_bs  # 展开后的批次大小

        req_pool_indices = buffers.req_pool_indices[:expanded_bs]  # 请求池索引
        positions = buffers.positions[:expanded_bs]  # 位置
        mrope_positions = buffers.mrope_positions[:, :expanded_bs]  # 多维RoPE位置
        seq_lens = buffers.seq_lens[:expanded_bs]  # 序列长度
        seq_lens_cpu = buffers.seq_lens_cpu[:expanded_bs]  # CPU序列长度
        topk_p = buffers.topk_p[:request_bs]  # top-k概率
        topk_index = buffers.topk_index[:request_bs]  # top-k索引
        hidden_states = buffers.hidden_states[:request_bs]  # 隐藏状态
        bonus_tokens = buffers.bonus_tokens[:request_bs]  # 奖励token

        if self.require_mlp_tp_gather:  # MLP TP收集
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [expanded_bs] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [expanded_bs] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = expanded_bs * self.dp_size
        elif self.require_attn_tp_gather:  # 注意力TP收集
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [expanded_bs],
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [expanded_bs],
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = expanded_bs
        else:  # 不需要收集
            global_num_tokens = None
            global_num_tokens_for_logprob = None
            global_dp_buffer_len = None

        spec_info = FrozenKVMTPDraftInput(  # 创建草稿输入
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            bonus_tokens=bonus_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = self.topk  # 每请求token数
        spec_info.num_tokens_for_logprob_per_req = self.topk  # logprob每请求token数
        spec_info.positions = positions  # 位置

        forward_batch = ForwardBatch(  # 创建前向批次
            forward_mode=ForwardMode.DECODE,
            batch_size=expanded_bs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=None,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        def run_once():
            # 运行一次Frozen-KV MTP草稿前向
            if self.model_runner.is_hybrid_swa:  # 混合滑动窗口注意力
                self.model_runner.token_to_kv_pool.invalidate_loc_cache()

            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(  # 设置DP缓冲区长度
                global_dp_buffer_len,
                expanded_bs,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)  # 标记非扩展

            hidden_states_backup = forward_batch.spec_info.hidden_states  # 备份隐藏状态
            ret = self.frozen_kv_mtp_worker.draft_forward(
                forward_batch, skip_attn_backend_init=True
            )  # 运行草稿前向
            forward_batch.spec_info.hidden_states = hidden_states_backup  # 恢复隐藏状态
            return ret  # 返回结果

        # Swap the draft backend's token_to_kv_pool to the frozen target pool
        # for the capture; the single backend-attr swap is seen by both
        # ``get_token_to_kv_pool()`` (via ``get_attn_backend()``) and the
        # backend's own reads.
        # 将草稿后端的token_to_kv_pool交换为冻结的目标池用于捕获；
        # 单个后端属性交换被get_token_to_kv_pool()（通过get_attn_backend()）
        # 和后端自身的读取共同看到。
        target_pool = self.frozen_kv_mtp_worker.kv_context.target_token_to_kv_pool  # 目标KV池
        saved_backend_pool = self.draft_attn_backend.token_to_kv_pool  # 保存原池
        self.draft_attn_backend.token_to_kv_pool = target_pool  # 交换到目标池
        try:
            with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):  # 前向上下文
                self.frozen_kv_mtp_worker._init_frozen_kv_metadata_capture_cuda_graph(
                    forward_batch
                )  # 初始化元数据
                self.deepep_adapter.capture(is_extend_in_batch=False)  # DeepEP捕获
                self._capture_init(run_once)  # 预热
                out = self._capture_graph(  # 捕获图
                    graph, get_global_graph_memory_pool(), stream, run_once
                )
        finally:
            self.draft_attn_backend.token_to_kv_pool = saved_backend_pool  # 恢复原池
        set_global_graph_memory_pool(graph.pool())  # 设置全局图内存池
        return graph, out  # 返回图和输出

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        # 后处理：将输出截断到实际批次大小
        parent_list, top_scores_index, draft_tokens = (t[:raw_bs] for t in out)  # 截断
        return parent_list, top_scores_index, draft_tokens  # 返回截断后的结果

    def replay(self, forward_batch: ForwardBatch):
        # 重放CUDA图以执行Frozen-KV MTP草稿前向
        self.deepep_adapter.replay()  # DeepEP重放
        buffers = self.buffers  # 输入缓冲区

        raw_expanded_bs = forward_batch.batch_size  # 实际展开批次大小
        raw_bs = (  # 实际请求批次大小
            raw_expanded_bs // self.num_tokens_per_bs
            if self.topk > 1
            else raw_expanded_bs
        )
        raw_num_token = raw_expanded_bs  # 实际token数

        if self.require_mlp_tp_gather:  # MLP TP收集
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = max_num_tokens // (
                self.num_tokens_per_bs * self.num_tokens_per_bs
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:  # 普通模式
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]  # 找到最近的捕获批次大小
        expanded_bs = bs * self.num_tokens_per_bs  # 展开后的批次大小
        if bs != raw_bs:  # 需要填充
            buffers.seq_lens.fill_(self.seq_len_fill_value)  # 填充序列长度
            buffers.positions.zero_()  # 清零位置
            # Pair with seq_lens fill: padded rows must point at reserved
            # req_pool slot 0 (req_to_token[0, :] is all zeros from init).
            # 与序列长度填充配对：填充行必须指向保留的req_pool槽0。
            buffers.req_pool_indices.zero_()  # 清零请求池索引

        num_tokens = expanded_bs  # token数
        buffers.seq_lens[:raw_expanded_bs].copy_(forward_batch.seq_lens)  # 复制序列长度
        buffers.positions[:raw_num_token].copy_(forward_batch.positions)  # 复制位置
        if forward_batch.mrope_positions is not None:  # 复制多维RoPE位置
            buffers.mrope_positions[:, :raw_num_token].copy_(
                forward_batch.mrope_positions
            )
        # `topk_p`/`topk_index` are produced by the captured seed iter.
        # topk_p/topk_index由捕获的种子迭代产生。
        buffers.bonus_tokens[:raw_bs].copy_(forward_batch.spec_info.bonus_tokens)  # 复制奖励token
        buffers.hidden_states[:raw_bs].copy_(forward_batch.spec_info.hidden_states)  # 复制隐藏状态
        buffers.req_pool_indices[:raw_expanded_bs].copy_(forward_batch.req_pool_indices)  # 复制请求池索引

        if self.require_gathered_buffer:  # 需要收集缓冲区
            buffers.global_num_tokens_gpu.fill_(expanded_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(expanded_bs)

        if bs != raw_bs:  # 需要填充时更新前向批次
            forward_batch.batch_size = expanded_bs
            forward_batch.seq_lens = buffers.seq_lens[:expanded_bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:expanded_bs]
            forward_batch.positions = buffers.positions[:num_tokens]
            if forward_batch.mrope_positions is not None:
                forward_batch.mrope_positions = buffers.mrope_positions[:, :num_tokens]

        if forward_batch.seq_lens_cpu is not None:  # 处理CPU序列长度
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_expanded_bs].copy_(forward_batch.seq_lens_cpu)
            forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:expanded_bs]

        self.frozen_kv_mtp_worker._init_frozen_kv_metadata_replay_cuda_graph(
            forward_batch,
            expanded_bs,
            forward_batch.seq_lens_sum
            + (expanded_bs - raw_expanded_bs) * self.seq_len_fill_value,
        )  # 初始化重放元数据

        self.raw_bs = raw_bs  # 保存原始批次大小
        self.bs = bs  # 保存填充后批次大小
        # NVTX span: the graph bypasses `model_runner.forward`'s record_function.
        # NVTX跨度：图绕过了model_runner.forward的record_function。
        span_name = f"step[DRAFT_LOOP raw_bs={raw_bs} bs={bs} topk={self.topk}]"
        if torch.autograd._profiler_enabled():  # 启用了profiler
            with torch.profiler.record_function(span_name):
                self._replay()  # 在profiler范围内重放
        else:  # 未启用
            self._replay()  # 直接重放
        out = self.output_buffers[bs]  # 获取输出

        if bs != raw_bs:  # 需要截断
            out = self._postprocess_output_to_raw_bs(out, raw_bs)  # 后处理截断
            forward_batch.batch_size = raw_expanded_bs  # 恢复原始批次大小
            forward_batch.positions = buffers.positions[:raw_num_token]  # 恢复位置
            forward_batch.seq_lens = buffers.seq_lens[:raw_expanded_bs]  # 恢复序列长度
            forward_batch.req_pool_indices = buffers.req_pool_indices[:raw_expanded_bs]  # 恢复请求池索引
            if forward_batch.mrope_positions is not None:
                forward_batch.mrope_positions = buffers.mrope_positions[
                    :, :raw_num_token
                ]
            if forward_batch.seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:raw_expanded_bs]

        return out  # 返回输出
