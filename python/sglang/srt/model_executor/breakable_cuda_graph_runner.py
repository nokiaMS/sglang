# 可中断CUDA图（BCG）运行器模块
# 将模型前向传播捕获为一系列在注意力层处分割的torch.cuda.CUDAGraph段，
# 功能上与基于torch.compile的PCG运行器并行，但不依赖torch.compile或FX图分割，
# 图中断通过eager_on_graph装饰的可调用对象（密集模型的radix attention，混合模型的mamba）即时插入。

# Copyright 2023-2026 SGLang Team
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
"""Breakable CUDA graph (BCG) runner.  # 可中断CUDA图（BCG）运行器。

Captures the model forward as a sequence of ``torch.cuda.CUDAGraph`` segments  # 将模型前向传播捕获为一系列torch.cuda.CUDAGraph段
split at attention layers. Functionally parallel to the torch.compile-based  # 在注意力层处分割。功能上与基于torch.compile的
PCG runner but does not depend on torch.compile or FX graph splitting — graph  # PCG运行器并行，但不依赖torch.compile或FX图分割——
breaks are inserted eagerly via :func:`eager_on_graph` decorated callables  # 图中断通过:eager_on_graph装饰的可调用对象即时插入
(radix attention for dense models, mamba for hybrid models).  # （密集模型的radix attention，混合模型的mamba）。
"""

# 可中断 CUDA 图（BCG）运行器。
# 将模型前向传播捕获为一系列在注意力层处分割的 ``torch.cuda.CUDAGraph`` 段。
# 功能上与基于 torch.compile 的 PCG 运行器并行，但不依赖 torch.compile 或 FX 图分割——
# 图中断通过 :func:`eager_on_graph` 装饰的可调用对象（密集模型的 radix attention，
# 混合模型的 mamba）即时插入。

from __future__ import annotations  # 启用延迟类型注解评估

import bisect  # 导入二分查找模块，用于查找合适的捕获token数
import inspect  # 导入检查模块，用于获取函数签名
import logging  # 导入日志模块
from typing import TYPE_CHECKING, Union  # 导入类型注解

import torch  # 导入PyTorch
import tqdm  # 导入进度条模块

from sglang.srt.compilation.piecewise_context_manager import set_forward_context  # 导入前向上下文设置
from sglang.srt.distributed import get_tensor_model_parallel_rank  # 导入获取张量并行排名函数
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # 导入NCCL图池ID设置
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture  # 导入图捕获上下文
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力相关设置
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出类
from sglang.srt.layers.pooler import EmbeddingPoolerOutput  # 导入嵌入池化输出类
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (  # 导入可中断CUDA图核心类
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.breakable_cuda_graph.context import (  # 导入BCG上下文管理
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.cuda_graph_runner import (  # 导入全局图内存池管理
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息和捕获隐藏模式
    CaptureHiddenMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context  # 导入前向上下文
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (  # 导入分段CUDA图运行器和GC冻结工具
    PiecewiseCudaGraphRunner,
    freeze_gc,
)
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0  # 导入GPU内存查询和日志工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批次信息类
    from sglang.srt.model_executor.model_runner import ModelRunner  # 模型运行器类


class BreakableCudaGraphRunner:  # 可中断CUDA图运行器，在注意力层处分割捕获模型前向传播
    """Breakable CUDA graph runner.  # 可中断CUDA图运行器。

    Captures the model forward as a series of ``torch.cuda.CUDAGraph`` segments  # 将模型前向传播捕获为一系列torch.cuda.CUDAGraph段
    with graph breaks at attention layers. Simpler than the torch.compile-based  # 在注意力层处有图断点。比基于torch.compile的
    PCG runner: no FX tracing, no compiled-kernel fusion — just segment-level  # PCG运行器更简单：无FX追踪，无编译内核融合——仅段级别的
    graph capture of the eager kernel stream.  # 急切内核流的图捕获。
    """

    def __init__(self, model_runner: ModelRunner):  # 初始化可中断CUDA图运行器
        self.model_runner = model_runner  # 保存模型运行器引用
        self.device = model_runner.device  # 获取计算设备
        self.device_module = torch.get_device_module(self.device)  # 获取设备模块（cuda/npu等）
        self.graphs = {}  # 存储每种token数对应的可中断CUDA图
        self.output_buffers = {}  # 存储每种token数对应的输出缓冲区

        self.quant_config = getattr(model_runner.model, "quant_config", None)  # 获取量化配置
        self.is_multimodal = model_runner.is_multimodal  # 是否为多模态模型
        # Read by the shared replay_prepare (bound from PiecewiseCudaGraphRunner).  # 由共享的replay_prepare读取（从PiecewiseCudaGraphRunner绑定）。
        self.capture_return_pooled_hidden_states = not model_runner.is_generation  # 非生成模型需要返回池化隐藏状态

        # Capture sizes  # 捕获尺寸配置
        capture_tokens = model_runner.server_args.piecewise_cuda_graph_tokens  # 获取分段CUDA图的token数列表
        assert capture_tokens is not None  # 断言token数列表不为空
        self.capture_num_tokens = sorted(capture_tokens)  # 排序后的token数列表
        self.max_num_tokens = (  # 最大token数
            max(self.capture_num_tokens) if self.capture_num_tokens else 8192  # 取列表最大值或默认8192
        )
        self.max_bs = model_runner.req_to_token_pool.size  # 最大批大小等于请求到token池的大小

        self.capture_hidden_mode = CaptureHiddenMode.NULL  # 默认不捕获隐藏状态
        if model_runner.server_args.enable_return_hidden_states:  # 如果启用了返回隐藏状态
            self.capture_hidden_mode = CaptureHiddenMode.FULL  # 设置为完全捕获模式
        if (  # 如果使用EAGLE推测算法
            model_runner.spec_algorithm is not None  # 推测算法不为空
            and model_runner.spec_algorithm.is_eagle()  # 且为EAGLE算法
        ):
            if model_runner.is_draft_worker:  # 如果是草稿工作者
                self.capture_hidden_mode = CaptureHiddenMode.LAST  # 仅捕获最后一层隐藏状态
            else:  # 否则
                self.capture_hidden_mode = CaptureHiddenMode.FULL  # 捕获全部隐藏状态

        log_info_on_rank0(  # 在rank0上记录日志
            logger,
            f"[BCG] Capture num tokens: {self.capture_num_tokens}",  # 记录捕获的token数列表
        )

        self._init_buffers(model_runner)  # 初始化输入缓冲区

        self.attention_layers = model_runner.attention_layers  # 获取注意力层列表
        self.moe_layers = model_runner.moe_layers  # 获取MoE层列表
        self.moe_fusions = model_runner.moe_fusions  # 获取MoE融合配置

        # Resolve the inner transformer-stack module (the same boundary PCG draws  # 解析内部transformer栈模块（与PCG通过patch_model绘制的边界相同）。
        # via patch_model). At replay we monkey-patch this module's forward with  # 在重放时，我们用重放捕获的CUDAGraph并返回捕获的
        # a closure that replays the captured CUDAGraph and returns the captured  # hidden_states的闭包来猴子补丁此模块的forward；
        # hidden_states; the outer model.forward then runs logits_processor /  # 外部model.forward然后使用活跃的（多请求）forward_batch
        # pooler eagerly with the live (multi-req) forward_batch.  # 急切地运行logits_processor/pooler。
        language_model = getattr(  # 获取语言模型
            model_runner.model, "language_model", model_runner.model  # 尝试获取language_model属性，否则使用模型本身
        )
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):  # 如果语言模型有model.layers
            self.layer_model = language_model.model  # 设置层模型为内部model
        else:  # 否则
            # If we can't find the inner layer_model, disable BCG.  # 如果找不到内部layer_model，禁用BCG。
            self.layer_model = None  # 设为None表示禁用
            logger.warning(  # 记录警告
                "[BCG] Could not resolve inner layer_model on %s. BCG is "  # 无法解析内部layer_model
                "disabled for this model; prefill will fall back to eager.",  # BCG对该模型禁用；预填充将回退到急切模式
                type(language_model).__name__,  # 语言模型类型名
            )
            return  # 提前返回，不继续初始化
        self.use_input_embeds = self.is_multimodal  # 多模态模型使用输入嵌入
        if self.use_input_embeds:  # 如果使用输入嵌入
            sig = inspect.signature(self.layer_model.forward)  # 获取层模型forward函数签名
            params = list(sig.parameters)  # 获取参数列表
            if "input_embeds" not in params:  # 如果参数中没有input_embeds
                raise ValueError(  # 抛出值错误
                    f"layer_model.forward must accept 'input_embeds' for "  # layer_model.forward必须接受'input_embeds'
                    f"multimodal BCG, got params: {params}"  # 用于多模态BCG
                )
            self._input_embeds_arg_idx = params.index("input_embeds")  # 记录input_embeds参数的位置索引

        # Memory pool  # 内存池
        if get_global_graph_memory_pool() is None:  # 如果全局图内存池未设置
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())  # 使用设备图池句柄设置全局池
        set_graph_pool_id(get_global_graph_memory_pool())  # 设置NCCL图池ID

        # Warmup then capture  # 预热然后捕获
        self._warmup()  # 执行预热
        self.device_module.synchronize()  # 同步设备
        self.model_runner.tp_group.barrier()  # 张量并行组同步屏障
        self._capture_all()  # 捕获所有token数的CUDA图

        self.raw_num_tokens = 0  # 记录实际token数，用于截取输出

    def _init_buffers(self, model_runner):  # 初始化捕获用的输入缓冲区
        """Initialize input buffers."""  # 初始化输入缓冲区
        from sglang.srt.model_executor.piecewise_cuda_graph_runner import (  # 导入预填充输入缓冲区类
            PrefillInputBuffers,
        )
        from sglang.srt.utils import is_npu  # 导入NPU检测函数

        with torch.device(self.device):  # 在指定设备上创建张量
            input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.int64)  # 输入ID缓冲区
            out_cache_loc = torch.zeros(  # 输出缓存位置缓冲区
                (self.max_num_tokens,),
                dtype=torch.int64 if not is_npu() else torch.int32,  # NPU使用int32，其他使用int64
            )
            positions = torch.zeros((self.max_num_tokens,), dtype=torch.int64)  # 位置ID缓冲区
            if self.is_multimodal:  # 如果是多模态模型
                input_embeds = torch.zeros(  # 输入嵌入缓冲区
                    (self.max_num_tokens, model_runner.model_config.hidden_size),  # 形状为(最大token数, 隐藏维度)
                    dtype=model_runner.dtype,  # 使用模型数据类型
                )
                mrope_positions = torch.zeros(  # 多模态旋转位置编码缓冲区
                    (3, self.max_num_tokens), dtype=torch.int64  # 3行对应3个维度的位置编码
                )
            else:  # 非多模态模型
                input_embeds = None  # 不需要输入嵌入
                mrope_positions = None  # 不需要多模态旋转位置编码

        if model_runner.is_draft_worker:  # 如果是草稿工作者
            from sglang.srt.speculative.eagle_utils import get_draft_hidden_dim  # 导入获取草稿隐藏维度函数

            hidden_dim = get_draft_hidden_dim(model_runner)  # 获取草稿隐藏维度
            self.static_draft_hidden_states = torch.zeros(  # 草稿隐藏状态缓冲区
                (self.max_num_tokens, hidden_dim),  # 形状为(最大token数, 草稿隐藏维度)
                dtype=model_runner.dtype,  # 使用模型数据类型
                device=self.device,  # 在指定设备上
            )

        self.buffers = PrefillInputBuffers(  # 创建预填充输入缓冲区对象
            input_ids=input_ids,  # 输入ID
            out_cache_loc=out_cache_loc,  # 输出缓存位置
            mamba_track_indices=None,  # mamba跟踪索引（暂未使用）
            mamba_track_mask=None,  # mamba跟踪掩码（暂未使用）
            mamba_track_seqlens=None,  # mamba跟踪序列长度（暂未使用）
            positions=positions,  # 位置ID
            input_embeds=input_embeds,  # 输入嵌入
            mrope_positions=mrope_positions,  # 多模态旋转位置编码
        )
        self.buffers.share_buffers()  # 共享缓冲区到各层

    @torch.no_grad()  # 禁用梯度计算
    def _run_forward(self, forward_batch, num_tokens):  # 运行层栈前向传播（含正确的上下文设置）
        """Run layer-stack forward with proper context.  # 在正确的上下文中运行层栈前向传播。

        Captures only the inner transformer stack (layer_model). The outer  # 仅捕获内部transformer栈（layer_model）。外部的
        model.forward's tail (logits_processor / pooler) is intentionally  # model.forward的尾部（logits_processor/pooler）被故意
        excluded — it has bs-shaped kernels that would bake batch_size=1  # 排除——它有bs形状的内核，会将batch_size=1
        into the captured graph.  # 烘焙到捕获的图中。

        ``@torch.no_grad`` mirrors the decorator on the outer ``*ForCausalLM.forward``  # @torch.no_grad镜像了外部*ForCausalLM.forward上的装饰器
        (e.g. qwen3.py:507). Calling ``layer_model.forward`` directly skips that  # （例如qwen3.py:507）。直接调用layer_model.forward跳过了
        decorator, so we apply it here — without it some MoE @torch.compile  # 该装饰器，所以我们在这里应用——没有它，一些MoE @torch.compile
        kernels (``torch.sum(out=...)``) fail dynamo with "out= doesn't support  # 内核（torch.sum(out=...)）会在dynamo中因"out=不支持
        autograd", and mamba state ops can spuriously track gradients.  # autograd"而失败，且mamba状态操作可能错误地追踪梯度。
        """
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None  # 重置数据并行局部位置和token数
        set_dp_buffer_len(None, num_tokens, forward_batch.dp_padding_mode.is_max_len())  # 设置数据并行缓冲区长度
        set_is_extend_in_batch(False)  # 设置不在批次中执行extend操作

        with set_forward_context(  # 设置前向上下文
            forward_batch,  # 前向批次
            self.attention_layers,  # 注意力层
            self.quant_config,  # 量化配置
            self.moe_layers,  # MoE层
            self.moe_fusions,  # MoE融合
        ):
            output = self.layer_model.forward(  # 执行层模型前向传播
                forward_batch.input_ids,  # 输入ID
                forward_batch.positions,  # 位置ID
                forward_batch,  # 前向批次
                input_embeds=forward_batch.input_embeds,  # 输入嵌入（多模态时使用）
            )
        return output  # 返回层模型输出

    def _build_capture_forward_batch(self, num_tokens):  # 构建bs=1的占位ForwardBatch用于捕获
        """Build a bs=1 placeholder ForwardBatch for capture.  # 构建bs=1的占位ForwardBatch用于捕获。

        bs=1 here is only a placeholder for attention/mamba breaks' metadata  # 这里的bs=1仅是注意力/mamba断点的元数据
        shapes; replay supplies live multi-req metadata via replay_prepare.  # 形状的占位符；重放通过replay_prepare提供活跃的多请求元数据。
        Captured kernels run only on the token-major layer stack and are  # 捕获的内核仅在token主导的层栈上运行，
        bs-invariant.  # 且与批大小无关。
        """
        from sglang.srt.layers.dp_attention import DpPaddingMode  # 导入数据并行填充模式
        from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息类
            ForwardBatch,
            ForwardMode,
        )

        spec_info = None  # 推测信息初始化为None
        if self.model_runner.is_draft_worker:  # 如果是草稿工作者
            from sglang.srt.speculative.eagle_info import EagleDraftInput  # 导入EAGLE草稿输入

            spec_info = EagleDraftInput(  # 创建EAGLE草稿输入
                hidden_states=self.static_draft_hidden_states[:num_tokens],  # 使用草稿隐藏状态的前num_tokens个
            )

        buffers = self.buffers  # 获取缓冲区引用
        bs = 1  # 批大小设为1（占位用）
        with torch.device(self.device):  # 在指定设备上创建张量
            seq_lens = torch.full((bs,), num_tokens, dtype=torch.int64)  # 序列长度，全填num_tokens
            extend_seq_lens = torch.full((bs,), num_tokens, dtype=torch.int64)  # 扩展序列长度
            extend_prefix_lens = torch.zeros((bs,), dtype=torch.int64)  # 扩展前缀长度，全零
            extend_start_loc = torch.zeros((bs,), dtype=torch.int64)  # 扩展起始位置，全零
            req_pool_indices = torch.arange(bs, dtype=torch.int64)  # 请求池索引，0到bs-1
            orig_seq_lens = torch.full((bs,), num_tokens, dtype=torch.int64)  # 原始序列长度

        return ForwardBatch(  # 返回构建的ForwardBatch对象
            forward_mode=ForwardMode.EXTEND,  # 前向模式为扩展
            batch_size=bs,  # 批大小
            input_ids=buffers.input_ids[:num_tokens],  # 输入ID（截取到num_tokens）
            input_embeds=(  # 输入嵌入
                buffers.input_embeds[:num_tokens] if self.is_multimodal else None  # 多模态时截取，否则None
            ),
            req_pool_indices=req_pool_indices,  # 请求池索引
            seq_lens=seq_lens,  # 序列长度
            next_token_logits_buffer=None,  # 下一token的logits缓冲区（不使用）
            orig_seq_lens=orig_seq_lens,  # 原始序列长度
            seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),  # CPU上的序列长度
            out_cache_loc=buffers.out_cache_loc[:num_tokens],  # 输出缓存位置
            seq_lens_sum=num_tokens,  # 序列长度总和
            mamba_track_indices=None,  # mamba跟踪索引（不使用）
            mamba_track_mask=None,  # mamba跟踪掩码（不使用）
            mamba_track_seqlens=None,  # mamba跟踪序列长度（不使用）
            encoder_lens=None,  # 编码器长度（不使用）
            return_logprob=False,  # 不返回log概率
            extend_num_tokens=num_tokens,  # 扩展token数
            extend_seq_lens=extend_seq_lens,  # 扩展序列长度
            extend_prefix_lens=extend_prefix_lens,  # 扩展前缀长度
            extend_start_loc=extend_start_loc,  # 扩展起始位置
            extend_prefix_lens_cpu=torch.tensor([0], device="cpu"),  # CPU上的扩展前缀长度
            extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),  # CPU上的扩展序列长度
            extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),  # CPU上的扩展logprob起始长度
            positions=buffers.positions[:num_tokens],  # 位置ID
            global_num_tokens_gpu=None,  # 全局token数GPU张量（不使用）
            global_num_tokens_for_logprob_gpu=None,  # 全局logprob token数GPU张量（不使用）
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),  # 数据并行填充模式
            global_dp_buffer_len=None,  # 全局数据并行缓冲区长度（不使用）
            mrope_positions=(  # 多模态旋转位置编码
                buffers.mrope_positions[:, :num_tokens] if self.is_multimodal else None  # 多模态时截取，否则None
            ),
            spec_algorithm=None,  # 推测算法（不使用）
            spec_info=spec_info,  # 推测信息
            capture_hidden_mode=self.capture_hidden_mode,  # 隐藏状态捕获模式
            num_token_non_padded=None,  # 非填充token数（不使用）
            global_forward_mode=ForwardMode.EXTEND,  # 全局前向模式为扩展
            lora_ids=None,  # LoRA ID（不使用）
        )

    def _warmup(self):  # 预热模型，执行一次前向传播
        """Warmup the model with a forward pass."""  # 通过一次前向传播预热模型
        num_tokens = self.capture_num_tokens[0]  # 使用最小的token数进行预热
        forward_batch = self._build_capture_forward_batch(num_tokens)  # 构建预热用的前向批次
        with forward_context(  # 设置前向上下文
            ForwardContext(attn_backend=self.model_runner.attn_backend)  # 使用注意力后端
        ):
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)  # 初始化前向元数据
            self._run_forward(forward_batch, num_tokens)  # 执行预热前向传播

    def _capture_all(self):  # 捕获所有token数的可中断CUDA图
        """Capture breakable CUDA graphs for all token sizes."""  # 为所有token大小捕获可中断CUDA图
        with (  # 使用多个上下文管理器
            freeze_gc(self.model_runner.server_args.enable_cudagraph_gc),  # 可选地冻结垃圾回收
            graph_capture() as graph_capture_context,  # 图捕获上下文
            enable_breakable_cuda_graph(),  # 启用BCG上下文
        ):
            stream = graph_capture_context.stream  # 获取捕获流
            pool = get_global_graph_memory_pool()  # 获取全局图内存池

            capture_range = (  # 创建捕获范围迭代器
                tqdm.tqdm(list(reversed(self.capture_num_tokens)))  # rank0显示进度条，从大到小捕获
                if get_tensor_model_parallel_rank() == 0  # 仅主rank显示进度条
                else reversed(self.capture_num_tokens)  # 其他rank使用普通迭代器
            )
            for num_tokens in capture_range:  # 遍历每种token数
                if get_tensor_model_parallel_rank() == 0:  # 仅主rank更新进度信息
                    avail_mem = get_available_gpu_memory(  # 获取可用GPU内存
                        self.model_runner.device,  # 设备
                        self.model_runner.gpu_id,  # GPU ID
                        empty_cache=False,  # 不清空缓存
                    )
                    capture_range.set_description(  # 设置进度条描述
                        f"[BCG] Capturing ({num_tokens=} {avail_mem=:.2f} GB)"  # 显示当前token数和可用内存
                    )

                graph, output = self._capture_one(num_tokens, pool, stream)  # 捕获一个token数的CUDA图
                self.graphs[num_tokens] = graph  # 存储捕获的图
                self.output_buffers[num_tokens] = output  # 存储输出缓冲区

    def can_run(self, forward_batch: "ForwardBatch"):  # 判断当前BCG运行器能否处理给定的前向批次
        """Return True if this BCG runner can handle the given forward batch."""  # 如果此BCG运行器能处理给定的前向批次则返回True
        if self.layer_model is None:  # 如果层模型未解析（BCG被禁用）
            return False  # 不能运行
        if forward_batch.forward_mode.is_target_verify():  # 如果是目标验证模式
            return False  # 不能运行
        if forward_batch.capture_hidden_mode != self.capture_hidden_mode:  # 如果隐藏状态捕获模式不匹配
            return False  # 不能运行
        if forward_batch.input_embeds is not None:  # 如果有输入嵌入（多模态运行时）
            return False  # 不能运行
        if forward_batch.replace_embeds is not None:  # 如果有替换嵌入
            return False  # 不能运行
        num_tokens = len(forward_batch.input_ids)  # 获取输入token数
        if forward_batch.return_logprob:  # 如果需要返回log概率
            for start_len, seq_len in zip(  # 遍历每个序列的logprob起始长度和序列长度
                forward_batch.extend_logprob_start_lens_cpu,  # logprob起始长度
                forward_batch.extend_seq_lens_cpu,  # 扩展序列长度
            ):
                if start_len is not None and start_len < seq_len:  # 如果起始长度小于序列长度
                    return False  # 需要部分logprob，不能运行
        return num_tokens <= self.max_num_tokens  # token数不超过最大值时可运行

    def _capture_one(self, num_tokens, pool, stream):  # 为单个token数捕获可中断CUDA图
        """Capture a breakable CUDA graph for one token size."""  # 为一个token大小捕获可中断CUDA图
        forward_batch = self._build_capture_forward_batch(num_tokens)  # 构建捕获用的前向批次
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)  # 初始化前向元数据

        def run_once():  # 执行一次前向传播（用于预热和捕获）
            # Invalidate SWA loc cache — same fix as in cuda_graph_runner.run_once.  # 使SWA位置缓存失效——与cuda_graph_runner.run_once中的修复相同。
            if self.model_runner.is_hybrid_swa:  # 如果使用混合滑动窗口注意力
                self.model_runner.token_to_kv_pool.invalidate_loc_cache()  # 使位置缓存失效
            return self._run_forward(forward_batch, num_tokens)  # 执行前向传播

        with forward_context(  # 设置前向上下文
            ForwardContext(attn_backend=self.model_runner.attn_backend)  # 使用注意力后端
        ):
            for _ in range(2):  # 预热2次
                self.device_module.synchronize()  # 同步设备
                self.model_runner.tp_group.barrier()  # 张量并行组同步屏障
                run_once()  # 执行一次前向传播

            graph = BreakableCUDAGraph()  # 创建可中断CUDA图容器
            with BreakableCUDAGraphCapture(cuda_graph=graph, pool=pool, stream=stream):  # 开始捕获上下文
                output = run_once()  # 在捕获上下文中执行一次前向传播

        return graph, output  # 返回捕获的图和输出

    def replay_prepare(self, forward_batch, **kwargs):  # 准备重放，设置静态前向批次数据
        """Prepare static forward batch data for replay."""  # 为重放准备静态前向批次数据
        # TODO: fix PiecewiseCudaGraphRunner to support draft workers as well.  # TODO: 修复PiecewiseCudaGraphRunner以支持草稿工作者。
        static_forward_batch = PiecewiseCudaGraphRunner.replay_prepare(  # 调用父类的replay_prepare
            self, forward_batch, **kwargs  # 传入自身和前向批次
        )
        if self.model_runner.is_draft_worker and forward_batch.spec_info is not None:  # 如果是草稿工作者且有推测信息
            num_tokens = len(forward_batch.input_ids)  # 获取token数
            self.static_draft_hidden_states[:num_tokens].copy_(  # 复制草稿隐藏状态到静态缓冲区
                forward_batch.spec_info.hidden_states  # 从推测信息中获取隐藏状态
            )
        return static_forward_batch  # 返回准备好的静态前向批次

    def replay(  # 重放捕获的CUDA图并获取模型输出
        self,
        forward_batch: ForwardBatch,  # 前向批次
        **kwargs,  # 额外关键字参数
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:  # 返回类型为logits输出、PP代理张量或嵌入池化输出
        """Replay the captured breakable CUDA graph for the given forward batch."""  # 为给定的前向批次重放捕获的可中断CUDA图
        num_tokens = len(forward_batch.input_ids)  # 获取实际token数
        index = bisect.bisect_left(self.capture_num_tokens, num_tokens)  # 二分查找找到>=num_tokens的最小捕获token数索引
        static_num_tokens = self.capture_num_tokens[index]  # 获取对应的捕获token数

        captured_graph = self.graphs[static_num_tokens]  # 获取对应token数的捕获图
        captured_hidden = self.output_buffers[static_num_tokens]  # 获取对应token数的输出缓冲区

        # Closure replaces layer_model.forward for the duration of the outer  # 闭包在外部model.forward调用期间替换layer_model.forward。
        # model.forward call. Replays the captured CUDAGraph and hands the  # 重放捕获的CUDAGraph并将捕获的hidden_states
        # outer forward the captured hidden_states; logits_processor / pooler  # 传递给外部forward；logits_processor/pooler
        # then runs eagerly on top with the live multi-req forward_batch.  # 然后使用活跃的多请求forward_batch在其上急切运行。
        def replay_layer_forward(*args, **layer_kwargs):  # 替换层模型forward的闭包，重放CUDA图并返回捕获的隐藏状态
            ie = layer_kwargs.get("input_embeds") or (  # 从关键字参数获取input_embeds
                args[self._input_embeds_arg_idx]  # 或从位置参数获取
                if self.use_input_embeds and len(args) > self._input_embeds_arg_idx  # 如果使用输入嵌入且参数足够
                else None  # 否则为None
            )
            if self.use_input_embeds:  # 如果使用输入嵌入
                if ie is None:  # 但没有提供
                    raise ValueError("BCG replay expects input_embeds but got None")  # 抛出值错误
                self.buffers.input_embeds[:static_num_tokens].copy_(  # 将输入嵌入复制到静态缓冲区
                    ie[:static_num_tokens]  # 截取到捕获token数
                )
            else:  # 不使用输入嵌入
                if ie is not None:  # 但却收到了
                    raise ValueError(  # 抛出值错误
                        "BCG replay got unexpected input_embeds on non-multimodal model"  # BCG重放在非多模态模型上收到了意外的input_embeds
                    )
            captured_graph.replay()  # 重放捕获的CUDA图
            return captured_hidden  # 返回捕获的隐藏状态

        with enable_breakable_cuda_graph():  # 在BCG上下文中执行重放
            static_forward_batch = self.replay_prepare(forward_batch, **kwargs)  # 准备重放数据

            original_layer_forward = self.layer_model.forward  # 保存原始的层模型forward
            self.layer_model.forward = replay_layer_forward  # 猴子补丁替换为重放闭包
            try:  # 尝试执行
                self.model_runner.attn_backend.init_forward_metadata(forward_batch)  # 初始化前向元数据
                with set_forward_context(  # 设置前向上下文
                    static_forward_batch,  # 使用静态前向批次
                    self.attention_layers,  # 注意力层
                    self.quant_config,  # 量化配置
                    self.moe_layers,  # MoE层
                    self.moe_fusions,  # MoE融合
                ):
                    output = self.model_runner.model.forward(  # 执行完整模型前向传播
                        static_forward_batch.input_ids,  # 静态输入ID
                        static_forward_batch.positions,  # 静态位置ID
                        static_forward_batch,  # 静态前向批次
                        **kwargs,  # 额外关键字参数
                    )
            finally:  # 无论如何恢复原始forward
                self.layer_model.forward = original_layer_forward  # 恢复原始层模型forward
        if isinstance(output, LogitsProcessorOutput):  # 如果输出是logits处理器输出
            return LogitsProcessorOutput(  # 截取到实际token数并返回
                next_token_logits=output.next_token_logits[: self.raw_num_tokens],  # 截取logits到实际token数
                hidden_states=(  # 截取隐藏状态到实际token数
                    output.hidden_states[: self.raw_num_tokens]  # 如果有隐藏状态
                    if output.hidden_states is not None  # 且不为None
                    else None  # 否则返回None
                ),
            )
        elif isinstance(output, EmbeddingPoolerOutput):  # 如果输出是嵌入池化输出
            return output  # 直接返回
        else:  # 其他类型
            assert isinstance(output, PPProxyTensors)  # 断言为PP代理张量
            raise NotImplementedError(  # 抛出未实现错误
                "PPProxyTensors is not supported in BreakableCudaGraphRunner."  # BreakableCudaGraphRunner不支持PPProxyTensors
            )
