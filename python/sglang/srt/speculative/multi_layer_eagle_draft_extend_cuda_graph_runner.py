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
# 多层EAGLE草稿扩展CUDA图运行器模块。
# 实现了多层EAGLE推测解码中草稿扩展阶段的CUDA图捕获和重放，
# 支持每步独立捕获和链式多步捕获两种模式。

from __future__ import annotations  # 启用延迟类型注解评估

import bisect  # 导入二分查找模块
import logging  # 导入日志模块
import time  # 导入时间模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Callable, List, Optional  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len  # 导入DP注意力相关
from sglang.srt.model_executor.cuda_graph_runner import (  # 导入CUDA图运行器相关
    CUDA_GRAPH_CAPTURE_FAILED_MSG,  # CUDA图捕获失败消息
    CudaGraphRunner,  # CUDA图运行器基类
    DeepEPCudaGraphRunnerAdapter,  # DeepEP CUDA图运行器适配器
    LogitsProcessorOutput,  # Logits处理器输出
    get_batch_sizes_to_capture,  # 获取需要捕获的批次大小
    get_global_graph_memory_pool,  # 获取全局图内存池
    model_capture_mode,  # 模型捕获模式
    set_global_graph_memory_pool,  # 设置全局图内存池
    set_is_extend_in_batch,  # 设置是否批次内扩展
    set_torch_compile_config,  # 设置torch编译配置
)
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    CaptureHiddenMode,  # 隐藏状态捕获模式
    ForwardBatch,  # 前向批次
    ForwardMode,  # 前向模式
)
from sglang.srt.model_executor.forward_context import (  # 导入前向上下文
    ForwardContext,  # 前向上下文类
    forward_context,  # 前向上下文管理器
    get_req_to_token_pool,  # 获取请求到token映射池
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers  # 导入前向输入缓冲区基类
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput  # 导入EAGLE草稿扩展输入
from sglang.srt.speculative.multi_layer_eagle_utils import assign_new_state_triton  # 导入Triton状态分配函数
from sglang.srt.speculative.spec_utils import fast_topk  # 导入快速topk函数
from sglang.srt.utils import (  # 导入工具函数
    get_available_gpu_memory,  # 获取可用GPU内存
    require_attn_tp_gather,  # 是否需要注意力TP聚合
    require_gathered_buffer,  # 是否需要聚合缓冲区
    require_mlp_sync,  # 是否需要MLP同步
    require_mlp_tp_gather,  # 是否需要MLP TP聚合
)

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (  # 导入多层EAGLE草稿工作器
        MultiLayerEagleDraftWorker,
    )


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclass
class MultiLayerEagleDraftExtendInputBuffers(ForwardInputBuffers):
    """多层EAGLE草稿扩展输入缓冲区数据类，存储CUDA图所需的输入缓冲区。"""
    # Sliced from shared parent buffers
    input_ids: torch.Tensor  # 输入token ID
    out_cache_loc: torch.Tensor  # 输出缓存位置
    positions: torch.Tensor  # 位置编码
    # Shared from parent
    seq_lens: torch.Tensor  # 序列长度
    seq_lens_cpu: torch.Tensor  # CPU端序列长度
    req_pool_indices: torch.Tensor  # 请求池索引
    num_correct_drafts: torch.Tensor  # 正确草稿数
    num_accept_tokens: torch.Tensor  # 接受token数
    # Per-step buffers
    extend_seq_lens: torch.Tensor  # 扩展序列长度
    extend_start_loc: torch.Tensor  # 扩展起始位置
    mrope_positions: torch.Tensor  # 多模态旋转位置编码
    hidden_states: torch.Tensor  # 隐藏状态
    next_token_logits_buffer: torch.Tensor  # 下一个token的logits缓冲区
    global_num_tokens_gpu: Optional[torch.Tensor]  # GPU端全局token数
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]  # GPU端全局logprob token数


class MultiLayerEagleDraftExtendCudaGraphRunner:
    """多层EAGLE草稿扩展CUDA图运行器，管理单步的CUDA图捕获和重放。"""

    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker, step: int):
        """初始化单步草稿扩展CUDA图运行器。"""
        # Parse args
        self.step = step  # 当前步骤编号
        self.eagle_worker = eagle_worker  # EAGLE工作器引用
        self.model_runner = model_runner = eagle_worker.mtp_model_runner(self.step)  # 获取当前步骤的模型运行器
        self.forward_mode = ForwardMode.DRAFT_EXTEND_V2  # 设置前向模式为草稿扩展V2

        self.graphs = {}  # 存储不同批次大小的CUDA图
        self.output_buffers = {}  # 存储不同批次大小的输出缓冲区
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile  # 是否启用torch编译
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding  # 是否禁用CUDA图填充
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)  # 是否需要聚合缓冲区
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)  # 是否需要MLP TP聚合
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)  # 是否需要MLP同步
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)  # 是否需要注意力TP聚合
        self.tp_size = self.model_runner.tp_size  # 张量并行大小
        self.dp_size = model_runner.server_args.dp_size  # 数据并行大小
        self.enable_pdmux = model_runner.server_args.enable_pdmux  # 是否启用PD多路复用
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps  # 推测步数
        self.speculative_num_draft_tokens = (  # 推测草稿token数
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.topk = model_runner.server_args.speculative_eagle_topk  # topk参数
        self.enable_profile_cuda_graph = (  # 是否启用CUDA图性能分析
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)  # 获取需要捕获的批次大小
        self.padded_static_len = -1  # 填充静态长度
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()  # DeepEP适配器

        # For Attention Backend
        self.num_tokens_per_bs = self.speculative_num_steps + 1 + step  # 每批次token数
        self.max_bs = max(self.capture_bs)  # 最大批次大小
        self.max_num_token = self.max_bs * self.num_tokens_per_bs  # 最大token数

        self.eagle_worker.draft_extend_attn_backend_list[  # 初始化注意力后端的CUDA图状态
            self.step
        ].init_cuda_graph_state(self.max_bs, self.max_num_token)
        self.seq_len_fill_value = self.eagle_worker.draft_extend_attn_backend_list[  # 获取序列长度填充值
            self.step
        ].get_cuda_graph_seq_len_fill_value()

    def init_buffers_and_capture(
        self,
        cuda_graph_buffers,  # CUDA图缓冲区
        offset,  # 偏移量
        next_cuda_graph_runner,  # 下一个CUDA图运行器
    ):
        """初始化缓冲区并捕获CUDA图。"""
        self.next_cuda_graph_runner = next_cuda_graph_runner  # 保存下一个运行器引用
        seq_lens_cpu = cuda_graph_buffers["seq_lens_cpu"]  # 获取CPU序列长度
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs  # 初始化扩展序列长度CPU列表

        if self.enable_torch_compile:  # 如果启用torch编译
            set_torch_compile_config()  # 设置torch编译配置

        # Graph inputs
        with torch.device(self.model_runner.device):  # 在指定设备上分配
            # sliced buffers
            # slice according to max_num_token
            input_ids = cuda_graph_buffers["input_ids"][  # 从共享缓冲区切片输入ID
                offset : offset + self.max_num_token
            ]
            out_cache_loc = cuda_graph_buffers["out_cache_loc"][  # 从共享缓冲区切片输出缓存位置
                offset : offset + self.max_num_token
            ]
            positions = cuda_graph_buffers["positions"][  # 从共享缓冲区切片位置
                offset : offset + self.max_num_token
            ]

            # shared states
            seq_lens = cuda_graph_buffers["seq_lens"]  # 获取共享序列长度
            req_pool_indices = cuda_graph_buffers["req_pool_indices"]  # 获取共享请求池索引
            num_correct_drafts = cuda_graph_buffers["num_correct_drafts"]  # 获取共享正确草稿数
            num_accept_tokens = cuda_graph_buffers["num_accept_tokens"]  # 获取共享接受token数

            extend_seq_lens = torch.full(  # 创建扩展序列长度张量
                (self.max_bs,),  # 最大批次大小
                self.num_tokens_per_bs,  # 每批次token数
                dtype=torch.int32,  # int32类型
            )
            extend_start_loc = torch.arange(  # 创建扩展起始位置张量
                0,  # 起始值
                self.max_bs * self.num_tokens_per_bs,  # 结束值
                step=self.num_tokens_per_bs,  # 步长
                dtype=torch.int32,  # int32类型
            )

            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)  # 创建多模态旋转位置编码

            hidden_states = torch.zeros(  # 创建隐藏状态张量
                (
                    self.max_num_token,  # 最大token数
                    EagleDraftExtendInput.hidden_size_for(self.eagle_worker),  # 隐藏维度
                ),
                dtype=EagleDraftExtendInput.dtype_for(self.eagle_worker),  # 数据类型
            )

            if self.require_gathered_buffer:  # 如果需要聚合缓冲区
                if self.require_mlp_tp_gather:  # 如果需要MLP TP聚合
                    global_num_tokens_gpu = torch.zeros(  # 全局token数GPU
                        (self.dp_size,), dtype=torch.int32  # DP大小的int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(  # 全局logprob token数GPU
                        (self.dp_size,), dtype=torch.int32  # DP大小的int32
                    )
                else:
                    assert self.require_attn_tp_gather  # 断言需要注意力TP聚合
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)  # 全局token数GPU
                    global_num_tokens_for_logprob_gpu = torch.zeros(  # 全局logprob token数GPU
                        (1,), dtype=torch.int32  # 大小为1
                    )
            else:
                global_num_tokens_gpu = None  # 不需要
                global_num_tokens_for_logprob_gpu = None  # 不需要

            if hasattr(  # 检查是否有draft_vocab_size属性（llama_eagle）
                self.model_runner.model_config.hf_config, "draft_vocab_size"
            ):  # llama_eagle
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size  # 使用draft词汇大小
            elif hasattr(  # 检查是否有hot_vocab_size属性（llama_eagle3）
                self.model_runner.model_config.hf_config, "hot_vocab_size"
            ):  # llama_eagle3
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size  # 使用hot词汇大小
            else:
                vocab_size = self.model_runner.model_config.vocab_size  # 使用默认词汇大小

            next_token_logits_buffer = torch.zeros(  # 创建下一个token的logits缓冲区
                (
                    (
                        self.max_bs * self.num_tokens_per_bs  # 草稿扩展V2模式
                        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2
                        else self.max_bs  # 其他模式
                    ),
                    vocab_size,  # 词汇大小
                ),
                dtype=torch.float,  # float类型
            )

        self.buffers = MultiLayerEagleDraftExtendInputBuffers(  # 创建输入缓冲区数据类
            input_ids=input_ids,  # 输入ID
            out_cache_loc=out_cache_loc,  # 输出缓存位置
            positions=positions,  # 位置
            seq_lens=seq_lens,  # 序列长度
            seq_lens_cpu=seq_lens_cpu,  # CPU序列长度
            req_pool_indices=req_pool_indices,  # 请求池索引
            num_correct_drafts=num_correct_drafts,  # 正确草稿数
            num_accept_tokens=num_accept_tokens,  # 接受token数
            extend_seq_lens=extend_seq_lens,  # 扩展序列长度
            extend_start_loc=extend_start_loc,  # 扩展起始位置
            mrope_positions=mrope_positions,  # 多模态旋转位置编码
            hidden_states=hidden_states,  # 隐藏状态
            next_token_logits_buffer=next_token_logits_buffer,  # 下一个token logits缓冲区
            global_num_tokens_gpu=global_num_tokens_gpu,  # 全局token数GPU
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,  # 全局logprob token数GPU
        )

        # Capture
        try:
            with model_capture_mode():  # 在模型捕获模式下
                self.capture()  # 捕获CUDA图
        except RuntimeError as e:  # 捕获运行时错误
            raise Exception(  # 抛出异常
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        """检查当前CUDA图是否可以运行给定的前向批次。"""
        if self.require_mlp_tp_gather:  # 如果需要MLP TP聚合
            cuda_graph_bs = (  # 计算CUDA图批次大小
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs  # EAGLE算法
                if self.model_runner.spec_algorithm.is_eagle()
                else max(forward_batch.global_num_tokens_cpu)  # 其他算法
            )
        else:
            cuda_graph_bs = forward_batch.seq_lens.numel()  # 使用序列长度数量

        is_bs_supported = (  # 检查批次大小是否支持
            cuda_graph_bs in self.graphs  # 禁用填充时必须精确匹配
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs  # 启用填充时只需不超过最大值
        )

        if self.require_mlp_sync:  # 如果需要MLP同步
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph  # 还需检查DP CUDA图

        return is_bs_supported  # 返回是否支持

    def _create_graph(self):
        """创建新的CUDA图对象。"""
        return torch.cuda.CUDAGraph()  # 返回新的CUDA图

    def _capture_init(self, run_once_fn):
        """CUDA图捕获前的预热初始化，运行两次以确保稳定性。"""
        for _ in range(2):  # 运行两次预热
            torch.cuda.synchronize()  # 同步CUDA
            self.model_runner.tp_group.barrier()  # TP组屏障同步
            run_once_fn()  # 运行一次

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        """在指定流上捕获CUDA图。"""
        with torch.cuda.graph(graph, pool=pool, stream=stream):  # 开始图捕获
            out = run_once_fn()  # 运行一次获取输出
        return out  # 返回输出

    def _replay(self, forward_batch: ForwardBatch):
        """重放指定批次大小的CUDA图。"""
        self.graphs[self.bs].replay()  # 重放当前批次大小的图

    def capture(self):
        """执行CUDA图捕获，委托给CudaGraphRunner基类。"""
        CudaGraphRunner.capture(self)  # 调用基类捕获方法

    def get_forward_batch(self, bs: int) -> ForwardBatch:
        """根据批次大小构造前向批次对象，用于CUDA图捕获和重放。"""
        buffers = self.buffers  # 获取缓冲区
        num_tokens = bs * self.num_tokens_per_bs  # 计算token数

        # Graph inputs
        input_ids = buffers.input_ids[:num_tokens]  # 截取输入ID
        req_pool_indices = buffers.req_pool_indices[:bs]  # 截取请求池索引
        seq_lens = buffers.seq_lens[:bs]  # 截取序列长度
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]  # 截取CPU序列长度
        extend_seq_lens = buffers.extend_seq_lens[:bs]  # 截取扩展序列长度
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]  # 截取CPU扩展序列长度
        extend_start_loc = buffers.extend_start_loc[:bs]  # 截取扩展起始位置
        num_correct_drafts = buffers.num_correct_drafts[:bs]  # 截取正确草稿数
        num_accept_tokens = buffers.num_accept_tokens[:bs]  # 截取接受token数
        out_cache_loc = buffers.out_cache_loc[:num_tokens]  # 截取输出缓存位置
        positions = buffers.positions[:num_tokens]  # 截取位置
        mrope_positions = buffers.mrope_positions[:, :num_tokens]  # 截取多模态旋转位置
        hidden_states = buffers.hidden_states[:num_tokens]  # 截取隐藏状态
        next_token_logits_buffer = buffers.next_token_logits_buffer[  # 截取logits缓冲区
            : bs if self.forward_mode == ForwardMode.DRAFT_EXTEND else num_tokens  # 根据模式选择大小
        ]

        if self.require_mlp_tp_gather:  # 如果需要MLP TP聚合
            buffers.global_num_tokens_gpu.copy_(  # 复制全局token数到GPU
                torch.tensor(
                    [num_tokens] * self.dp_size,  # 每个DP大小的token数
                    dtype=torch.int32,  # int32类型
                    device=buffers.input_ids.device,  # 设备
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(  # 复制全局logprob token数到GPU
                torch.tensor(
                    [num_tokens] * self.dp_size,  # 每个DP大小的token数
                    dtype=torch.int32,  # int32类型
                    device=buffers.input_ids.device,  # 设备
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size  # 计算全局DP缓冲区长度
        elif self.require_attn_tp_gather:  # 如果需要注意力TP聚合
            buffers.global_num_tokens_gpu.copy_(  # 复制全局token数到GPU
                torch.tensor(
                    [num_tokens],  # token数
                    dtype=torch.int32,  # int32类型
                    device=buffers.input_ids.device,  # 设备
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(  # 复制全局logprob token数到GPU
                torch.tensor(
                    [bs],  # 批次大小
                    dtype=torch.int32,  # int32类型
                    device=buffers.input_ids.device,  # 设备
                )
            )
            global_dp_buffer_len = num_tokens  # 全局DP缓冲区长度等于token数
        else:
            global_dp_buffer_len = None  # 不需要

        spec_info = EagleDraftExtendInput(  # 创建草稿扩展输入
            hidden_states=hidden_states,  # 隐藏状态
            num_correct_drafts=num_correct_drafts,  # 正确草稿数
            num_accept_tokens=num_accept_tokens,  # 接受token数
        )
        spec_info.positions = None  # 位置设为None

        capture_mode = (  # 确定捕获模式
            CaptureHiddenMode.NULL  # 独立模式不捕获隐藏状态
            if self.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL  # 其他模式捕获全部隐藏状态
        )

        # Forward batch
        forward_batch = ForwardBatch(  # 创建前向批次
            forward_mode=self.forward_mode,  # 前向模式
            batch_size=bs,  # 批次大小
            input_ids=input_ids,  # 输入ID
            req_pool_indices=req_pool_indices,  # 请求池索引
            seq_lens=seq_lens,  # 序列长度
            seq_lens_cpu=seq_lens_cpu,  # CPU序列长度
            next_token_logits_buffer=next_token_logits_buffer,  # logits缓冲区
            out_cache_loc=out_cache_loc,  # 输出缓存位置
            seq_lens_sum=seq_lens.sum().item(),  # 序列长度总和
            return_logprob=False,  # 不返回logprob
            positions=positions,  # 位置
            mrope_positions=mrope_positions,  # 多模态旋转位置
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,  # 全局token数GPU
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,  # 全局logprob token数GPU
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),  # DP填充模式
            global_dp_buffer_len=global_dp_buffer_len,  # 全局DP缓冲区长度
            spec_algorithm=self.model_runner.spec_algorithm,  # 推测算法
            spec_info=spec_info,  # 推测信息
            capture_hidden_mode=capture_mode,  # 捕获隐藏状态模式
            extend_seq_lens=extend_seq_lens,  # 扩展序列长度
            extend_seq_lens_cpu=extend_seq_lens_cpu,  # CPU扩展序列长度
            padded_static_len=self.padded_static_len,  # 填充静态长度
            # added args
            extend_start_loc=extend_start_loc,  # 扩展起始位置
            extend_num_tokens=self.num_tokens_per_bs * bs,  # 扩展token数
            num_token_non_padded_cpu=self.num_tokens_per_bs * bs,  # 非填充token数CPU
            return_hidden_states_before_norm=True,  # 返回归一化前的隐藏状态
        )
        return forward_batch  # 返回前向批次

    def capture_one_batch_size(self, bs: int, forward: Callable, stream_idx: int = 0):
        """捕获单个批次大小的CUDA图。"""
        buffers = self.buffers  # 获取缓冲区
        graph = self._create_graph()  # 创建CUDA图
        stream = self.stream  # 获取流

        num_tokens = bs * self.num_tokens_per_bs  # 计算token数
        forward_batch = self.get_forward_batch(bs)  # 获取前向批次
        attn_backend = self.eagle_worker.draft_extend_attn_backend_list[self.step]  # 获取注意力后端

        def run_once():
            """CUDA图内的一次运行函数。"""
            if self.model_runner.is_hybrid_swa:  # 如果是混合滑动窗口注意力
                self.model_runner.token_to_kv_pool.invalidate_loc_cache()  # 使位置缓存失效

            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None  # 清除DP注意力中间缓存
            set_dp_buffer_len(  # 设置DP缓冲区长度
                forward_batch.global_dp_buffer_len,  # 全局DP缓冲区长度
                num_tokens,  # token数
                forward_batch.dp_padding_mode.is_max_len(),  # 是否使用最大长度模式
            )
            set_is_extend_in_batch(False)  # 设置非批次内扩展

            # Backup two fields, which will be modified in-place in `draft_forward`.
            output_cache_loc_backup = forward_batch.out_cache_loc  # 备份输出缓存位置
            hidden_states_backup = forward_batch.spec_info.hidden_states  # 备份隐藏状态

            ret = self.model_runner.model.forward(  # 运行模型前向
                forward_batch.input_ids,  # 输入ID
                forward_batch.positions,  # 位置
                forward_batch,  # 前向批次
            )

            # Chain-style MTP: overwrite buffers.hidden_states with the draft model's
            # output (hidden_states_before_norm) so that assign_new_state_triton
            # propagates each MTP layer's own output to the next MTP layer,
            # rather than always feeding the target model's hidden states.
            if (  # 链式MTP：用草稿模型输出覆盖隐藏状态
                self.eagle_worker.chain_mtp_hidden_states
                and ret.hidden_states is not None
            ):
                buffers.hidden_states[:num_tokens].copy_(ret.hidden_states[:num_tokens])  # 复制隐藏状态

            # num_correct_drafts is drafts-only; the last accepted draft sits at index
            # `num_correct_drafts` within the (current_token + drafts) slot range.
            select_index = (  # 计算选择索引
                torch.arange(bs, device=self.model_runner.device)  # 0到bs-1的范围
                * (self.speculative_num_draft_tokens + self.step)  # 乘以草稿token数加步数
                + buffers.num_correct_drafts[:bs]  # 加上正确草稿数
                + self.step  # 加上步数偏移
            )

            probs = torch.softmax(ret.next_token_logits[select_index], dim=-1)  # 计算softmax概率
            ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk选择

            if self.next_cuda_graph_runner is not None:  # 如果有下一个运行器
                next_buffers = self.next_cuda_graph_runner.buffers  # 获取下一个运行器的缓冲区
                # rejected drafts = proposed drafts - accepted drafts.
                # speculative_num_draft_tokens includes the current-token slot, so -1.
                padding_lens = (  # 计算填充长度（被拒绝的草稿数）
                    self.speculative_num_draft_tokens - 1
                ) - buffers.num_correct_drafts[:bs]
                assign_new_state_triton(  # 使用Triton核分配新状态
                    ret.topk_index,  # topk索引
                    buffers.input_ids,  # 输入ID
                    buffers.positions,  # 位置
                    buffers.hidden_states,  # 隐藏状态
                    buffers.out_cache_loc,  # 输出缓存位置
                    buffers.extend_seq_lens,  # 扩展序列长度
                    buffers.extend_start_loc,  # 扩展起始位置
                    next_buffers.input_ids,  # 下一步输入ID
                    next_buffers.positions,  # 下一步位置
                    next_buffers.hidden_states,  # 下一步隐藏状态
                    next_buffers.out_cache_loc,  # 下一步输出缓存位置
                    next_buffers.extend_seq_lens,  # 下一步扩展序列长度
                    next_buffers.extend_start_loc,  # 下一步扩展起始位置
                    next_buffers.seq_lens,  # 下一步序列长度
                    padding_lens,  # 填充长度
                    forward_batch.batch_size,  # 批次大小
                    self.step,  # 步数
                    forward_batch.req_pool_indices,  # 请求池索引
                    get_req_to_token_pool().req_to_token,  # 请求到token映射
                    self.eagle_worker.req_to_hidden_states_pool,  # 请求到隐藏状态池
                )
            forward_batch.out_cache_loc = output_cache_loc_backup  # 恢复输出缓存位置
            forward_batch.spec_info.hidden_states = hidden_states_backup  # 恢复隐藏状态
            return ret  # 返回结果

        with forward_context(ForwardContext(attn_backend=attn_backend)):  # 使用注意力后端上下文
            attn_backend.init_forward_metadata_capture_cuda_graph(  # 初始化CUDA图捕获的前向元数据
                bs=bs,  # 批次大小
                num_tokens=num_tokens,  # token数
                req_pool_indices=forward_batch.req_pool_indices,  # 请求池索引
                seq_lens=forward_batch.seq_lens,  # 序列长度
                encoder_lens=None,  # 编码器长度
                forward_mode=self.forward_mode,  # 前向模式
                spec_info=forward_batch.spec_info,  # 推测信息
            )
            self.deepep_adapter.capture(is_extend_in_batch=True)  # DeepEP适配器捕获
            self._capture_init(run_once)  # 初始化捕获
            out = self._capture_graph(  # 捕获CUDA图
                graph, get_global_graph_memory_pool(), stream, run_once  # 图、内存池、流、运行函数
            )

        set_global_graph_memory_pool(graph.pool())  # 设置全局图内存池
        return graph, out  # 返回图和输出

    def init_replay_state(
        self, forward_batch: ForwardBatch, bs: int, raw_bs: int, num_tokens: int  # 前向批次, # 填充后批次大小, # 原始批次大小, # token数
    ):
        """初始化重放状态，将前向批次数据复制到缓冲区中。"""
        buffers = self.buffers  # 获取缓冲区
        # Common inputs
        buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)  # 复制输入ID
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)  # 复制序列长度
        if forward_batch.extend_seq_lens is not None:  # 如果扩展序列长度存在
            buffers.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)  # 复制扩展序列长度
            buffers.extend_start_loc[:raw_bs].copy_(forward_batch.extend_start_loc)  # 复制扩展起始位置
        buffers.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)  # 复制输出缓存位置
        buffers.positions[:num_tokens].copy_(forward_batch.positions)  # 复制位置
        if (  # 如果隐藏状态维度匹配
            forward_batch.spec_info.hidden_states.shape[1]
            == buffers.hidden_states.shape[1]
        ):
            buffers.hidden_states[:num_tokens].copy_(  # 复制隐藏状态
                forward_batch.spec_info.hidden_states
            )
        if forward_batch.spec_info.num_correct_drafts is not None:  # 如果正确草稿数存在
            buffers.num_correct_drafts[:raw_bs].copy_(  # 复制正确草稿数
                forward_batch.spec_info.num_correct_drafts
            )
            buffers.num_accept_tokens[:raw_bs].copy_(  # 复制接受token数
                forward_batch.spec_info.num_accept_tokens
            )
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)  # 复制请求池索引

        if forward_batch.seq_lens_cpu is not None:  # 如果CPU序列长度存在
            if bs != raw_bs:  # 如果填充后与原始不同
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)  # 填充默认值
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)  # 复制CPU序列长度

        if forward_batch.extend_seq_lens_cpu is not None:  # 如果CPU扩展序列长度存在
            self.extend_seq_lens_cpu[:raw_bs] = forward_batch.extend_seq_lens_cpu  # 复制CPU扩展序列长度

    def replay(self, forward_batch: ForwardBatch, init_state: bool = True):
        """重放CUDA图，执行草稿扩展前向计算。"""
        assert forward_batch.out_cache_loc is not None  # 断言输出缓存位置存在
        self.deepep_adapter.replay()  # DeepEP适配器重放
        buffers = self.buffers  # 获取缓冲区

        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = forward_batch.batch_size  # 原始批次大小
        num_tokens = raw_bs * self.num_tokens_per_bs  # token数
        # num_tokens = forward_batch.input_ids.shape[0]
        if self.require_mlp_tp_gather:  # 如果需要MLP TP聚合
            max_batch_size = max(forward_batch.original_global_num_tokens_cpu)  # 获取最大全局token数
            index = bisect.bisect_left(self.capture_bs, max_batch_size)  # 二分查找索引
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)  # 二分查找索引

        bs = self.capture_bs[index]  # 获取填充后的批次大小

        if init_state:  # 如果需要初始化状态
            self.init_replay_state(forward_batch, bs, raw_bs, num_tokens)  # 初始化重放状态

        if self.require_gathered_buffer:  # 如果需要聚合缓冲区
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)  # 填充全局token数
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)  # 填充全局logprob token数

        forward_batch.spec_info.hidden_states = buffers.hidden_states[:num_tokens]  # 设置隐藏状态
        forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[:bs]  # 设置正确草稿数
        forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[:bs]  # 设置接受token数
        forward_batch.spec_info.num_tokens_per_req = self.num_tokens_per_bs  # 设置每请求token数
        forward_batch.spec_info.num_tokens_for_logprob_per_req = 1  # 设置每请求logprob token数
        forward_batch.spec_info.positions = buffers.positions[:num_tokens]  # 设置位置
        forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]  # 设置扩展序列长度

        self.eagle_worker.draft_extend_attn_backend_list[  # 初始化重放的注意力元数据
            self.step
        ].init_forward_metadata_replay_cuda_graph(
            bs=bs,  # 填充后批次大小
            req_pool_indices=buffers.req_pool_indices,  # 请求池索引
            seq_lens=buffers.seq_lens,  # 序列长度
            seq_lens_sum=forward_batch.seq_lens_sum  # 序列长度总和
            + (bs - raw_bs) * self.seq_len_fill_value,  # 加上填充部分
            encoder_lens=None,  # 编码器长度
            forward_mode=self.forward_mode,  # 前向模式
            spec_info=forward_batch.spec_info,  # 推测信息
            seq_lens_cpu=buffers.seq_lens_cpu,  # CPU序列长度
        )

        # Replay
        self.raw_bs = raw_bs  # 保存原始批次大小
        self.bs = bs  # 保存填充后批次大小
        self._replay(forward_batch)  # 执行重放
        out = self.output_buffers[bs]  # 获取输出缓冲区

        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:  # 如果是草稿扩展V2模式
            # DRAFT_EXTEND_V2: all tokens calculations whether accepted or not.
            unpadding_bs = num_tokens  # 不填充批次大小为token数
        elif bs != raw_bs:  # 如果填充后与原始不同
            forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[  # 截取正确草稿数
                :raw_bs
            ]
            forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[  # 截取接受token数
                :raw_bs
            ]
            unpadding_bs = raw_bs  # 不填充批次大小为原始大小
        else:
            unpadding_bs = None  # 不需要去填充

        if unpadding_bs is not None:  # 如果需要去填充
            out_copy = out  # 复制输出
            out = LogitsProcessorOutput(  # 创建新的logits输出
                next_token_logits=out.next_token_logits[:unpadding_bs],  # 截取logits
                hidden_states=out.hidden_states[:unpadding_bs],  # 截取隐藏状态
            )
            out.topk_p = out_copy.topk_p[:raw_bs]  # 截取topk概率
            out.topk_index = out_copy.topk_index[:raw_bs]  # 截取topk索引
        return out  # 返回输出


class MultiLayerEagleMultiStepDraftExtendCudaGraphRunner:
    """多层EAGLE多步草稿扩展CUDA图运行器，管理所有步骤的CUDA图。"""

    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker):
        """初始化多步草稿扩展CUDA图运行器，创建每步运行器并捕获图。"""
        self.eagle_worker = eagle_worker  # EAGLE工作器引用
        self.device = eagle_worker.device  # 设备
        self.gpu_id = eagle_worker.gpu_id  # GPU ID
        self.speculative_num_steps = eagle_worker.speculative_num_steps  # 推测步数
        self.draft_extend_attn_backend_list = (  # 草稿扩展注意力后端列表
            eagle_worker.draft_extend_attn_backend_list
        )

        self.runners = []  # 每步运行器列表
        self.cuda_graph_buffers = {}  # CUDA图缓冲区
        self.seq_len_fill_value = 1  # 序列长度填充值
        self.max_bs = 1  # 最大批次大小
        self.offsets = [0]  # 缓冲区偏移量列表

        self._init_and_capture()  # 初始化并捕获

    def _init_and_capture(self):
        """初始化每步运行器并捕获CUDA图。"""
        if self.eagle_worker.server_args.disable_cuda_graph:  # 如果禁用CUDA图
            self.runners = [None] * self.speculative_num_steps  # 所有步骤设为None
            return

        self.runners: List[Optional[MultiLayerEagleDraftExtendCudaGraphRunner]] = []  # 运行器列表
        buffer_len_list: List[int] = []  # 缓冲区长度列表

        # 1. Capture loop
        for step in range(self.speculative_num_steps):  # 遍历每步
            if self.draft_extend_attn_backend_list[step]:  # 如果注意力后端存在
                runner = MultiLayerEagleDraftExtendCudaGraphRunner(  # 创建运行器
                    self.eagle_worker, step  # 工作器和步数
                )
                self.runners.append(runner)  # 添加到列表

                self.seq_len_fill_value = runner.seq_len_fill_value  # 更新填充值
                self.max_bs = runner.max_bs  # 更新最大批次大小
                buffer_len_list.append(runner.max_num_token)  # 添加缓冲区长度
                self.offsets.append(self.offsets[-1] + runner.max_num_token)  # 计算偏移量
            else:
                self.runners.append(None)  # 添加None

        # 2. Allocate buffers
        self.cuda_graph_buffers["seq_lens_cpu"] = torch.full(  # 分配CPU序列长度缓冲区
            (self.max_bs,),  # 最大批次大小
            self.seq_len_fill_value,  # 填充值
            dtype=torch.int32,  # int32类型
        )

        with torch.device(self.device):  # 在指定设备上分配
            # Sliced buffers
            self.cuda_graph_buffers["input_ids"] = torch.zeros(  # 分配输入ID缓冲区
                (self.offsets[-1],), dtype=torch.int64  # 总长度，int64类型
            )
            self.cuda_graph_buffers["out_cache_loc"] = torch.ones(  # 分配输出缓存位置缓冲区
                (self.offsets[-1],), dtype=torch.int64  # 总长度，int64类型
            )
            self.cuda_graph_buffers["positions"] = torch.zeros(  # 分配位置缓冲区
                (self.offsets[-1],), dtype=torch.int64  # 总长度，int64类型
            )

            # Shared states
            self.cuda_graph_buffers["seq_lens"] = torch.full(  # 分配序列长度缓冲区
                (self.max_bs,),  # 最大批次大小
                self.seq_len_fill_value,  # 填充值
                dtype=torch.int32,  # int32类型
            )
            self.cuda_graph_buffers["req_pool_indices"] = torch.zeros(  # 分配请求池索引缓冲区
                (self.max_bs,), dtype=torch.int64  # 最大批次大小，int64类型
            )
            self.cuda_graph_buffers["num_correct_drafts"] = torch.full(  # 分配正确草稿数缓冲区
                (self.max_bs,), 1, dtype=torch.int32  # 初始值为1
            )
            self.cuda_graph_buffers["num_accept_tokens"] = torch.full(  # 分配接受token数缓冲区
                (self.max_bs,), 1, dtype=torch.int32  # 初始值为1
            )

        for step in range(self.speculative_num_steps - 1, -1, -1):  # 从后向前遍历步骤
            if self.runners[step] is not None:  # 如果运行器存在
                tic = time.perf_counter()  # 记录开始时间
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)  # 获取可用内存
                logger.info(  # 记录信息
                    f"Capture draft extend cuda graph begin (step {step}). This can take up to several minutes. avail mem={before_mem:.2f} GB"
                )

                self.runners[step].init_buffers_and_capture(  # 初始化缓冲区并捕获
                    self.cuda_graph_buffers,  # CUDA图缓冲区
                    self.offsets[step],  # 偏移量
                    (  # 下一个运行器
                        self.runners[step + 1]  # 下一步运行器
                        if step + 1 < self.speculative_num_steps  # 如果还有下一步
                        else None  # 否则为None
                    ),
                )

                after_mem = get_available_gpu_memory(self.device, self.gpu_id)  # 获取捕获后可用内存
                logger.info(  # 记录信息
                    f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
                )

    def reset_buffers(self, forward_batch, batch_result):
        """重置CUDA图缓冲区，根据验证结果设置正确草稿数和接受token数。"""
        self.cuda_graph_buffers["input_ids"].zero_()  # 清零输入ID
        self.cuda_graph_buffers["seq_lens"].fill_(self.seq_len_fill_value)  # 填充序列长度
        self.cuda_graph_buffers["out_cache_loc"].zero_()  # 清零输出缓存位置
        self.cuda_graph_buffers["positions"].zero_()  # 清零位置
        # `batch_result.accept_lens` is drafts + bonus.
        bs = forward_batch.batch_size  # 获取批次大小
        self.cuda_graph_buffers["num_correct_drafts"][:bs].copy_(  # 设置正确草稿数
            batch_result.accept_lens - 1  # 接受长度减1（去掉bonus）
        )
        self.cuda_graph_buffers["num_accept_tokens"][:bs].copy_(  # 设置接受token数
            batch_result.accept_lens  # 接受长度
        )

    def get_runner(self, step):
        """获取指定步骤的CUDA图运行器。"""
        return self.runners[step]  # 返回运行器

    def get_last_runner(self):
        """获取最后一步的CUDA图运行器。"""
        return self.runners[-1] if self.runners else None  # 返回最后一个运行器或None

    def can_run(self, forward_batch):
        """检查第一步运行器是否可以运行给定批次。"""
        return self.runners[0].can_run(forward_batch)  # 委托给第一步运行器检查
