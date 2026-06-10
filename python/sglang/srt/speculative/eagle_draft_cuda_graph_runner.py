# EAGLE草稿CUDA图运行器模块
# 为EAGLE投机解码的草稿阶段提供CUDA图捕获和重放功能，
# 优化多步自回归草稿生成的性能。
from __future__ import annotations  # 启用延迟注解求值

import bisect  # 导入二分查找模块
import contextlib  # 导入上下文管理工具
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Callable, Optional  # 导入类型注解

import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量
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
from sglang.srt.speculative.eagle_info import EagleDraftInput  # 导入EAGLE草稿输入
from sglang.srt.utils import (  # 导入工具函数
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.async_probe import maybe_detect_nan, maybe_detect_oob  # 导入异步探测工具

if TYPE_CHECKING:  # 类型检查时才导入
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


@dataclass  # 数据类装饰器
class EagleDraftInputBuffers(ForwardInputBuffers):
    # EAGLE草稿CUDA图的输入缓冲区
    input_ids: torch.Tensor  # 输入ID
    req_pool_indices: torch.Tensor  # 请求池索引
    out_cache_loc: torch.Tensor  # 输出缓存位置
    positions: torch.Tensor  # 位置编码
    mrope_positions: torch.Tensor  # 多维RoPE位置
    rids_int: Optional[torch.Tensor]  # 请求ID（整数）
    bootstrap_room_ids_int: Optional[torch.Tensor]  # 引导房间ID（整数）
    seq_lens: torch.Tensor  # 序列长度
    seq_lens_cpu: torch.Tensor  # CPU上的序列长度
    extend_seq_lens: torch.Tensor  # 扩展序列长度
    topk_p: torch.Tensor  # top-k概率
    topk_index: torch.Tensor  # top-k索引
    hidden_states: Optional[torch.Tensor]  # 隐藏状态
    global_num_tokens_gpu: Optional[torch.Tensor]  # GPU上的全局token数
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]  # GPU上用于logprob的全局token数


class EAGLEDraftCudaGraphRunner:
    # EAGLE草稿CUDA图运行器，捕获和重放草稿阶段的CUDA图
    def __init__(
        self,
        eagle_worker: EAGLEWorker,  # EAGLE Worker
        *,
        draft_attn_backend=None,  # 草稿注意力后端
        speculative_num_steps: Optional[int] = None,  # 投机步数
    ):
        # Parse args
        # 解析参数
        self.eagle_worker = eagle_worker  # EAGLE Worker引用
        if not hasattr(eagle_worker, "model_runner"):  # V2版本的EagleDraftWorker
            # V2: EagleDraftWorker
            self.model_runner = model_runner = eagle_worker.draft_runner  # 使用草稿运行器
        else:  # V1版本
            self.model_runner = model_runner = eagle_worker.model_runner  # 使用模型运行器
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
        self.draft_attn_backend = draft_attn_backend or model_runner.draft_attn_backend  # 草稿注意力后端
        self.enable_profile_cuda_graph = (  # 是否启用CUDA图分析
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False  # 不启用PDMUX
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()  # DeepEP适配器

        # Batch sizes to capture
        # 要捕获的批次大小
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)

        # Attention backend
        # 注意力后端
        self.num_tokens_per_bs = self.topk  # 每个批次的token数
        self.max_bs = max(self.capture_bs)  # 最大批次大小
        self.max_num_token = self.max_bs * self.num_tokens_per_bs  # 最大token数

        self.draft_attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)  # 初始化CUDA图状态
        self.seq_len_fill_value = self.draft_attn_backend.attn_backends[  # 序列长度填充值
            0
        ].get_cuda_graph_seq_len_fill_value()
        seq_lens_cpu = torch.full(  # CPU序列长度
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_seq_lens_cpu = [self.seq_len_fill_value] * self.max_bs  # 扩展序列长度列表

        if self.enable_torch_compile:  # 启用torch编译
            set_torch_compile_config()

        # Graph inputs
        # 图输入
        with torch.device(model_runner.device):  # 在设备上创建张量
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 输入ID
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)  # 请求池索引
            out_cache_loc = torch.zeros(  # 输出缓存位置
                (self.max_num_token * self.speculative_num_steps,),
                dtype=self._cache_loc_dtype(),
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 位置
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)  # 多维RoPE位置
            rids_int = (  # 请求ID（KV金丝雀启用时）
                torch.zeros((self.max_bs,), dtype=torch.int64)
                if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get()
                else None
            )
            bootstrap_room_ids_int = (  # 引导房间ID（KV金丝雀启用时）
                torch.full((self.max_bs,), -1, dtype=torch.int64)
                if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get()
                else None
            )
            seq_lens = torch.full(  # 序列长度
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)  # 扩展序列长度
            topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)  # top-k概率
            topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)  # top-k索引
            _hidden_size = EagleDraftInput.hidden_size_for(self.eagle_worker)  # 隐藏层大小
            hidden_states = (  # 隐藏状态
                torch.zeros(
                    (self.max_bs, _hidden_size),
                    dtype=EagleDraftInput.dtype_for(self.eagle_worker),
                )
                if _hidden_size is not None
                else None
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
            else:  # 不需要收集缓冲区
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

        self.buffers = EagleDraftInputBuffers(  # 创建输入缓冲区
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
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

    def _cache_loc_dtype(self):
        # 返回缓存位置的数据类型
        return torch.int64

    def can_run(self, forward_batch: ForwardBatch):
        # 检查CUDA图是否可以运行给定批次
        if self.require_mlp_tp_gather:  # MLP TP收集模式
            cuda_graph_bs = (  # 计算CUDA图批次大小
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:  # 普通模式
            cuda_graph_bs = forward_batch.batch_size

        is_bs_supported = (  # 批次大小是否支持
            cuda_graph_bs in self.graphs
            if self.disable_padding  # 禁用填充时需精确匹配
            else cuda_graph_bs <= self.max_bs  # 否则只需不超过最大值
        )

        if self.require_mlp_sync:  # 需要MLP同步
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported  # 返回是否支持

    def _create_graph(self):
        # 创建CUDA图对象
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        # 捕获前初始化：预热运行
        for _ in range(2):  # 预热2次
            torch.cuda.synchronize()  # 同步CUDA
            self.model_runner.tp_group.barrier()  # TP组屏障
            run_once_fn()  # 运行一次
            hook = getattr(  # 获取后处理钩子
                self.model_runner.draft_attn_backend,
                "on_after_cuda_graph_warmup",
                None,
            )
            if hook is not None:  # 有钩子
                hook()  # 调用钩子

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # 捕获CUDA图
        with torch.cuda.graph(graph, pool=pool, stream=stream):  # 开始图捕获
            out = run_once_fn()  # 运行一次
        return out  # 返回输出

    def _replay(self, forward_batch: ForwardBatch):
        # 重放CUDA图
        ctx = (  # 计时上下文
            self.model_runner.device_timer.wrap(metadata={"category": "eagle_draft"})
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:  # 在计时上下文中重放
            self.graphs[self.bs].replay()  # 重放图

    def capture(self):
        # 捕获所有批次大小的CUDA图
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        # 捕获指定批次大小的CUDA图
        buffers = self.buffers  # 输入缓冲区
        graph = self._create_graph()  # 创建图
        stream = self.stream  # CUDA流
        num_tokens = num_seqs * self.num_tokens_per_bs  # token数

        # Graph inputs
        # 图输入
        req_pool_indices = buffers.req_pool_indices[:num_seqs]  # 请求池索引
        seq_lens = buffers.seq_lens[:num_seqs]  # 序列长度
        seq_lens_cpu = buffers.seq_lens_cpu[:num_seqs]  # CPU序列长度
        extend_seq_lens = buffers.extend_seq_lens[:num_seqs]  # 扩展序列长度
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:num_seqs]  # CPU扩展序列长度
        out_cache_loc = buffers.out_cache_loc[: num_tokens * self.speculative_num_steps]  # 输出缓存位置
        positions = buffers.positions[:num_tokens]  # 位置
        mrope_positions = buffers.mrope_positions[:, :num_tokens]  # 多维RoPE位置
        rids_int = buffers.rids_int[:num_seqs] if buffers.rids_int is not None else None  # 请求ID
        bootstrap_room_ids_int = (  # 引导房间ID
            buffers.bootstrap_room_ids_int[:num_seqs]
            if buffers.bootstrap_room_ids_int is not None
            else None
        )
        hidden_states = (  # 隐藏状态
            buffers.hidden_states[:num_seqs]
            if buffers.hidden_states is not None
            else None
        )
        topk_p = buffers.topk_p[:num_seqs]  # top-k概率
        topk_index = buffers.topk_index[:num_seqs]  # top-k索引

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
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu  # 全局token数
            global_dp_buffer_len = num_tokens * self.dp_size  # 全局DP缓冲区长度
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu  # logprob全局token数
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
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_dp_buffer_len = num_tokens
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
        else:  # 不需要收集
            global_num_tokens = None
            global_dp_buffer_len = None
            global_num_tokens_for_logprob = None

        capture_mode = (  # 捕获模式
            CaptureHiddenMode.NULL
            if self.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        spec_info = EagleDraftInput(  # 创建草稿输入
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            capture_hidden_mode=capture_mode,
        )

        # Forward batch
        # 前向批次
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=num_seqs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            out_cache_loc=out_cache_loc,
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
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
            capture_hidden_mode=(
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            ),
        )

        def run_once():
            # 运行一次草稿前向
            if self.model_runner.is_hybrid_swa:  # 混合滑动窗口注意力
                self.model_runner.token_to_kv_pool.invalidate_loc_cache()  # 使位置缓存失效

            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None  # 清除DP本地位置
            set_dp_buffer_len(  # 设置DP缓冲区长度
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)  # 标记非扩展

            output_cache_loc_backup = forward_batch.out_cache_loc  # 备份输出缓存位置
            hidden_states_backup = forward_batch.spec_info.hidden_states  # 备份隐藏状态

            ret = self.eagle_worker.draft_forward(forward_batch)  # 运行草稿前向

            forward_batch.out_cache_loc = output_cache_loc_backup  # 恢复输出缓存位置
            forward_batch.spec_info.hidden_states = hidden_states_backup  # 恢复隐藏状态
            forward_batch.positions.sub_(self.eagle_worker.speculative_num_steps - 1)  # 恢复位置
            return ret  # 返回结果

        with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):  # 前向上下文
            self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
                forward_batch
            )  # 初始化前向元数据
            self.deepep_adapter.capture(is_extend_in_batch=False)  # DeepEP捕获
            self._capture_init(run_once)  # 预热
            out = self._capture_graph(  # 捕获图
                graph, get_global_graph_memory_pool(), stream, run_once
            )

        set_global_graph_memory_pool(graph.pool())  # 设置全局图内存池
        return graph, out  # 返回图和输出

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        # 后处理：将输出截断到实际批次大小
        # Keep the variables name for readability
        # 保持变量名以提高可读性
        parent_list, top_scores_index, draft_tokens = (t[:raw_bs] for t in out)  # 截断
        return parent_list, top_scores_index, draft_tokens  # 返回截断后的结果

    def replay(self, forward_batch: ForwardBatch):
        # 重放CUDA图以执行草稿前向
        assert forward_batch.out_cache_loc is not None  # 断言输出缓存位置存在
        self.deepep_adapter.replay()  # DeepEP重放
        buffers = self.buffers  # 输入缓冲区

        raw_bs = forward_batch.batch_size  # 实际批次大小
        raw_num_token = raw_bs * self.num_tokens_per_bs  # 实际token数

        # Pad
        # 填充到捕获的批次大小
        if self.require_mlp_tp_gather:  # MLP TP收集
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:  # 普通模式
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]  # 找到最近的捕获批次大小
        if bs != raw_bs:  # 需要填充
            buffers.seq_lens.fill_(self.seq_len_fill_value)  # 填充序列长度
            buffers.out_cache_loc.zero_()  # 清零输出缓存位置
            buffers.positions.zero_()  # 清零位置
            if buffers.rids_int is not None:  # 清零请求ID
                buffers.rids_int.zero_()
            if buffers.bootstrap_room_ids_int is not None:  # 填充引导房间ID
                buffers.bootstrap_room_ids_int.fill_(-1)
            buffers.topk_p.zero_()  # 清零top-k概率
            buffers.topk_index.zero_()  # 清零top-k索引
            if buffers.hidden_states is not None:  # 清零隐藏状态
                buffers.hidden_states.zero_()
            buffers.req_pool_indices.zero_()  # 清零请求池索引

        num_tokens = bs * self.num_tokens_per_bs  # token数

        # Common inputs
        # 通用输入
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)  # 复制序列长度
        buffers.out_cache_loc[: raw_num_token * self.speculative_num_steps].copy_(
            forward_batch.out_cache_loc
        )  # 复制输出缓存位置
        buffers.positions[:raw_num_token].copy_(forward_batch.positions)  # 复制位置
        if buffers.rids_int is not None and forward_batch.rids_int is not None:  # 复制请求ID
            buffers.rids_int[:raw_bs].copy_(forward_batch.rids_int)
        if (  # 复制引导房间ID
            buffers.bootstrap_room_ids_int is not None
            and forward_batch.bootstrap_room_ids_int is not None
        ):
            buffers.bootstrap_room_ids_int[:raw_bs].copy_(
                forward_batch.bootstrap_room_ids_int
            )
        maybe_detect_nan(  # 检测NaN
            forward_batch.spec_info.topk_p,
            "EagleDraftCudaGraphRunner.replay: topk_p",
        )
        maybe_detect_oob(  # 检测越界
            forward_batch.spec_info.topk_index,
            0,
            self.model_runner.model_config.vocab_size,
            "EagleDraftCudaGraphRunner.replay: topk_index vs vocab_size="
            f"{self.model_runner.model_config.vocab_size}",
        )
        buffers.topk_p[:raw_bs].copy_(forward_batch.spec_info.topk_p)  # 复制top-k概率
        buffers.topk_index[:raw_bs].copy_(forward_batch.spec_info.topk_index)  # 复制top-k索引
        if (  # 复制隐藏状态
            buffers.hidden_states is not None
            and forward_batch.spec_info.hidden_states is not None
        ):
            buffers.hidden_states[:raw_bs].copy_(forward_batch.spec_info.hidden_states)
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)  # 复制请求池索引

        # TODO(ch-wan): support num_token_non_padded
        # TODO(ch-wan): 支持num_token_non_padded
        if self.require_gathered_buffer:  # 需要收集缓冲区
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)  # 填充全局token数
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)  # 填充logprob全局token数

        # Attention backend
        # 注意力后端
        if bs != raw_bs:  # 需要填充时更新前向批次
            forward_batch.batch_size = bs  # 更新批次大小
            forward_batch.seq_lens = buffers.seq_lens[:bs]  # 使用缓冲区序列长度
            forward_batch.req_pool_indices = buffers.req_pool_indices[:bs]  # 使用缓冲区请求池索引
            forward_batch.positions = buffers.positions[:num_tokens]  # 使用缓冲区位置
            if buffers.rids_int is not None and forward_batch.rids_int is not None:
                forward_batch.rids_int = buffers.rids_int[:bs]
            if (
                buffers.bootstrap_room_ids_int is not None
                and forward_batch.bootstrap_room_ids_int is not None
            ):
                forward_batch.bootstrap_room_ids_int = buffers.bootstrap_room_ids_int[
                    :bs
                ]

        if forward_batch.seq_lens_cpu is not None:  # 处理CPU序列长度
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)  # 填充
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)  # 复制
            forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:bs]  # 使用缓冲区

        self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
            forward_batch, bs
        )  # 初始化重放元数据
        self.raw_bs = raw_bs  # 保存原始批次大小
        self.bs = bs  # 保存填充后批次大小
        # TODO: The forward_batch.seq_len_sum might need to be updated to reflect the padding in the cuda graph
        # TODO: forward_batch.seq_len_sum可能需要更新以反映CUDA图中的填充

        # Replay
        # 重放
        self._replay(forward_batch)  # 执行重放
        out = self.output_buffers[bs]  # 获取输出

        if bs != raw_bs:  # 需要截断
            out = self._postprocess_output_to_raw_bs(out, raw_bs)  # 后处理截断
            forward_batch.batch_size = raw_bs  # 恢复原始批次大小
            forward_batch.positions = buffers.positions[:raw_num_token]  # 恢复原始位置
            forward_batch.seq_lens = buffers.seq_lens[:raw_bs]  # 恢复原始序列长度
            forward_batch.req_pool_indices = buffers.req_pool_indices[:raw_bs]  # 恢复原始请求池索引
            if buffers.rids_int is not None and forward_batch.rids_int is not None:
                forward_batch.rids_int = buffers.rids_int[:raw_bs]
            if (
                buffers.bootstrap_room_ids_int is not None
                and forward_batch.bootstrap_room_ids_int is not None
            ):
                forward_batch.bootstrap_room_ids_int = buffers.bootstrap_room_ids_int[
                    :raw_bs
                ]
            if forward_batch.seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:raw_bs]

        return out  # 返回输出
