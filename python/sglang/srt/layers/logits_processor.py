# Logits处理器模块，实现语言模型输出的logits计算、logprobs处理、分块计算和扩散LLM等功能
# SPDX-License-Identifier: Apache-2.0
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
"""Logits processing.
Logits处理模块。"""

import dataclasses  # 导入数据类装饰器
import logging  # 导入日志模块
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch核心库
import triton  # 导入Triton编译器
import triton.language as tl  # 导入Triton语言
from torch import nn  # 导入PyTorch神经网络模块
from triton.language.extra import libdevice  # 导入Triton libdevice扩展

from sglang.srt.distributed import (  # 导入分布式通信相关函数
    get_tensor_model_parallel_world_size,  # 获取张量模型并行世界大小
    tensor_model_parallel_all_gather,  # 张量模型并行全收集
)
from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力相关工具
    DpPaddingMode,  # DP填充模式枚举
    attn_tp_all_gather,  # 注意力TP全收集
    attn_tp_all_gather_into_tensor,  # 注意力TP全收集到张量
    dp_gather_replicate,  # DP收集复制
    dp_scatter,  # DP散射
    get_attention_dp_rank,  # 获取注意力DP rank
    get_attention_dp_size,  # 获取注意力DP大小
    get_attention_tp_size,  # 获取注意力TP大小
    get_dp_device,  # 获取DP设备
    get_dp_dtype,  # 获取DP数据类型
    get_dp_hidden_size,  # 获取DP隐藏层大小
)
from sglang.srt.layers.utils.logprob import (  # 导入logprob计算工具
    InputLogprobsResult,  # 输入logprobs结果类
    get_token_ids_logprobs_chunk,  # 分块获取token ID的logprobs
    get_token_ids_logprobs_prefill,  # 预填充获取token ID的logprobs
    get_top_logprobs_chunk,  # 分块获取top-k logprobs
    get_top_logprobs_prefill,  # 预填充获取top-k logprobs
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    CaptureHiddenMode,  # 隐藏状态捕获模式
    ForwardBatch,  # 前向批次类
    ForwardMode,  # 前向模式枚举
)
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数
from sglang.srt.utils.common import (  # 导入通用工具函数
    is_cpu,  # 判断是否为CPU
    is_npu,  # 判断是否为NPU
    is_pin_memory_available,  # 判断是否支持锁页内存
    use_intel_amx_backend,  # 判断是否使用Intel AMX后端
)

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

_is_npu = is_npu()  # 检测是否为NPU平台
_is_cpu = is_cpu()  # 检测是否为CPU平台


@dataclasses.dataclass  # 数据类装饰器
class LogitsProcessorOutput:  # Logits处理器输出数据类
    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    ## 第一部分：此部分将在LogitsProcessor中赋值
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    # 下一个token的logits。形状：[#seq, vocab_size]
    # Can be None for certain prefill-only requests (e.g., multi-item scoring) that don't need next token generation
    # 对于某些不需要下一个token生成的纯预填充请求（如多item评分），可以为None
    next_token_logits: Optional[torch.Tensor]  # 下一个token的logits
    # Used by speculative decoding (EAGLE)
    # 用于投机解码（EAGLE）
    # The last hidden layers
    # 最后的隐藏层状态
    hidden_states: Optional[torch.Tensor] = None  # 隐藏层状态，默认为None

    ## Part 2: This part will be assigned in python/sglang/srt/layers/sampler.py::Sampler
    ## 第二部分：此部分将在Sampler中赋值
    # he log probs of output tokens, if SGLANG_RETURN_ORIGINAL_LOGPROB = True, will get the log probs before applying temperature. If False, will get the log probs before applying temperature.
    # 输出token的log概率，若SGLANG_RETURN_ORIGINAL_LOGPROB=True，获取应用温度前的log概率。若False，同样获取应用温度前的log概率。
    next_token_logprobs: Optional[torch.Tensor] = None  # 下一个token的log概率
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    # 输出位置top-k token的log概率和ID。形状：[#seq, k]
    next_token_top_logprobs_val: Optional[List] = None  # top-k logprobs值
    next_token_top_logprobs_idx: Optional[List] = None  # top-k logprobs索引
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    # 输出位置请求token ID的log概率和ID。形状：[#seq, n]（n为请求的token ID数）
    # Can contain either lists or GPU tensors (for delayed copy optimization in prefill-only requests)
    # 可以包含列表或GPU张量（用于纯预填充请求中的延迟复制优化）
    next_token_token_ids_logprobs_val: Optional[
        List[Union[List[float], torch.Tensor]]
    ] = None  # 请求token ID的logprobs值
    next_token_token_ids_logprobs_idx: Optional[List] = None  # 请求token ID的logprobs索引

    ## Part 3: Prefill-only. This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    ## 第三部分：仅预填充。此部分将在LogitsProcessor中赋值
    # The logprobs of input tokens.        shape: [#token]
    # 输入token的log概率。形状：[#token]
    input_token_logprobs: Optional[torch.Tensor] = None  # 输入token的log概率
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    # 输入位置top-k token的log概率和ID。形状：[#seq, #token, k]
    input_top_logprobs_val: Optional[List] = None  # 输入top-k logprobs值
    input_top_logprobs_idx: Optional[List] = None  # 输入top-k logprobs索引
    # The logprobs and ids of the requested token ids in input positions. shape: [#seq, n] (n is the number of requested token ids)
    # 输入位置请求token ID的log概率和ID。形状：[#seq, n]（n为请求的token ID数）
    # Can contain either lists or GPU tensors (for delayed GPU-to-CPU transfer optimization)
    # 可以包含列表或GPU张量（用于延迟GPU到CPU传输优化）
    input_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None  # 输入请求token ID的logprobs值
    )
    input_token_ids_logprobs_idx: Optional[List] = None  # 输入请求token ID的logprobs索引

    ## Part 4: Diffusion LLM only.
    ## 第四部分：仅扩散LLM使用。
    full_logits: Optional[torch.Tensor] = None  # 完整logits（扩散LLM使用）

    ## Part 5: Customized Info
    ## 第五部分：自定义信息
    customized_info: Optional[Dict[str, List[Any]]] = None  # 自定义信息字典

    mm_input_embeds: Optional[torch.Tensor] = None  # 多模态输入嵌入


@dataclasses.dataclass  # 数据类装饰器
class LogitsMetadata:  # Logits元数据类
    forward_mode: ForwardMode  # 前向模式
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL  # 隐藏状态捕获模式，默认NULL
    next_token_logits_buffer: Optional[torch.Tensor] = None  # 下一个token logits缓冲区

    extend_return_logprob: bool = False  # 扩展模式是否返回logprob
    extend_return_top_logprob: bool = False  # 扩展模式是否返回top logprob
    extend_token_ids_logprob: bool = False  # 扩展模式是否返回token ID logprob
    extend_seq_lens: Optional[torch.Tensor] = None  # 扩展序列长度
    extend_seq_lens_cpu: Optional[List[int]] = None  # 扩展序列长度（CPU端）
    extend_logprob_start_lens_cpu: Optional[List[int]] = None  # 扩展logprob起始长度（CPU端）
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None  # 扩展logprob裁剪后长度（CPU端）
    top_logprobs_nums: Optional[List[int]] = None  # top logprobs数量列表
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None  # 扩展输入logprob的token ID（GPU端）
    token_ids_logprobs: Optional[List[List[int]]] = None  # token ID logprobs列表

    # logits and logprobs post processing
    # logits和logprobs后处理参数
    temperature: torch.Tensor = None  # 温度参数
    top_p: torch.Tensor = None  # top_p参数

    # DP attention metadata. Not needed when DP attention is not used.
    # DP注意力元数据。不使用DP注意力时不需要。
    # Number of tokens in the request.
    # 请求中的token数量。
    global_num_tokens_gpu: Optional[torch.Tensor] = None  # 全局token数量（GPU端）
    # The start position of local hidden states.
    # 本地隐藏状态的起始位置。
    dp_local_start_pos: Optional[torch.Tensor] = None  # DP本地起始位置
    dp_local_num_tokens: Optional[torch.Tensor] = None  # DP本地token数量
    global_dp_buffer_len: Optional[int] = None  # 全局DP缓冲区长度
    # Number of tokens to sample per DP rank
    # 每个DP rank要采样的token数量
    global_num_tokens_for_logprob_cpu: Optional[torch.Tensor] = None  # 用于logprob的全局token数（CPU端）
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None  # 用于logprob的全局token数（GPU端）
    # The gather mode for DP attention
    # DP注意力的收集模式
    dp_padding_mode: Optional[DpPaddingMode] = None  # DP填充模式
    # for padding
    # 用于填充
    padded_static_len: int = -1  # 填充静态长度，默认-1

    # Whether this batch is prefill-only (no token generation needed)
    # 此批次是否为纯预填充（不需要生成token）
    is_prefill_only: bool = False  # 纯预填充标志，默认False

    mm_input_embeds: Optional[torch.Tensor] = None  # 多模态输入嵌入

    @classmethod  # 类方法装饰器
    def from_forward_batch(cls, forward_batch: ForwardBatch):  # 从前向批次创建LogitsMetadata
        if (  # 如果是扩展模式且需要返回logprob且不是目标验证
            forward_batch.forward_mode.is_extend()
            and forward_batch.return_logprob
            and not forward_batch.forward_mode.is_target_verify()
        ):
            extend_return_top_logprob = any(  # 检查是否任何请求需要top logprob
                x > 0 for x in forward_batch.top_logprobs_nums
            )
            extend_token_ids_logprob = any(  # 检查是否任何请求需要token ID logprob
                x is not None for x in forward_batch.token_ids_logprobs
            )
            extend_return_logprob = False  # 初始化为False
            extend_logprob_pruned_lens_cpu = []  # 初始化裁剪后长度列表
            for extend_len, start_len in zip(  # 遍历扩展长度和起始长度
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            ):
                if extend_len - start_len > 0:  # 如果有需要计算logprob的token
                    extend_return_logprob = True  # 设置为True
                extend_logprob_pruned_lens_cpu.append(extend_len - start_len)  # 添加裁剪后长度
        else:  # 非扩展模式或不需要logprob
            extend_return_logprob = extend_return_top_logprob = (
                extend_token_ids_logprob
            ) = extend_logprob_pruned_lens_cpu = False  # 全部设为False

        return cls(  # 创建并返回LogitsMetadata实例
            forward_mode=forward_batch.forward_mode,  # 前向模式
            capture_hidden_mode=forward_batch.capture_hidden_mode,  # 隐藏状态捕获模式
            next_token_logits_buffer=forward_batch.next_token_logits_buffer,  # logits缓冲区
            extend_return_logprob=extend_return_logprob,  # 是否返回扩展logprob
            extend_return_top_logprob=extend_return_top_logprob,  # 是否返回top logprob
            extend_token_ids_logprob=extend_token_ids_logprob,  # 是否返回token ID logprob
            extend_seq_lens=forward_batch.extend_seq_lens,  # 扩展序列长度
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,  # 扩展序列长度（CPU）
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,  # logprob起始长度
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,  # 裁剪后长度
            top_logprobs_nums=forward_batch.top_logprobs_nums,  # top logprobs数量
            token_ids_logprobs=forward_batch.token_ids_logprobs,  # token ID logprobs
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,  # 输入logprob token ID
            padded_static_len=forward_batch.padded_static_len,  # 填充静态长度
            is_prefill_only=forward_batch.is_prefill_only,  # 纯预填充标志
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,  # 全局token数
            dp_local_start_pos=forward_batch.dp_local_start_pos,  # DP本地起始位置
            dp_local_num_tokens=forward_batch.dp_local_num_tokens,  # DP本地token数
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,  # 全局DP缓冲区长度
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,  # logprob全局token数（CPU）
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,  # logprob全局token数（GPU）
            dp_padding_mode=DpPaddingMode.SUM_LEN,  # DP填充模式
            mm_input_embeds=forward_batch.mm_input_embeds,  # 多模态输入嵌入
        )

    def compute_dp_attention_metadata(self):  # 计算DP注意力元数据

        cumtokens = torch.cumsum(self.global_num_tokens_for_logprob_gpu, dim=0)  # 计算累积token数
        dp_rank = get_attention_dp_rank()  # 获取当前DP rank
        if dp_rank == 0:  # 如果是第0个rank
            dp_local_start_pos = torch.zeros_like(  # 起始位置为0
                self.global_num_tokens_for_logprob_gpu[0]
            )
        else:  # 非第0个rank
            dp_local_start_pos = cumtokens[dp_rank - 1]  # 起始位置为前一个rank的累积值

        self.dp_local_start_pos = dp_local_start_pos  # 设置本地起始位置
        self.dp_local_num_tokens = self.global_num_tokens_for_logprob_gpu[dp_rank]  # 设置本地token数

        hidden_size = get_dp_hidden_size()  # 获取DP隐藏层大小
        dtype = get_dp_dtype()  # 获取DP数据类型
        device = get_dp_device()  # 获取DP设备

        if self.global_num_tokens_for_logprob_cpu is not None:  # 如果CPU端有token数信息
            # create a smaller buffer to reduce peak memory usage
            # 创建较小的缓冲区以减少峰值内存使用
            self.global_dp_buffer_len = sum(self.global_num_tokens_for_logprob_cpu)  # 计算总缓冲区长度
        else:  # 没有CPU端信息
            self.global_dp_buffer_len = self.global_dp_buffer_len  # 保持原值

        self.gathered_buffer = torch.empty(  # 分配收集缓冲区
            (
                self.global_dp_buffer_len,  # 缓冲区长度
                hidden_size,  # 隐藏层大小
            ),
            dtype=dtype,  # 数据类型
            device=device,  # 设备
        )


class LogitsProcessor(nn.Module):  # Logits处理器类，继承自nn.Module
    def __init__(  # 初始化方法
        self,
        config,  # 模型配置
        skip_all_gather: bool = False,  # 是否跳过全收集，默认False
        logit_scale: Optional[float] = None,  # logit缩放因子，默认None
        return_full_logits: bool = False,  # 是否返回完整logits，默认False
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存模型配置
        self.vocab_size = config.vocab_size  # 获取词表大小
        self.logit_scale = logit_scale  # 保存logit缩放因子
        self.use_attn_tp_group = get_global_server_args().enable_dp_lm_head  # 是否使用注意力TP组
        self.use_fp32_lm_head = get_global_server_args().enable_fp32_lm_head  # 是否使用fp32 lm_head
        if self.use_attn_tp_group:  # 如果使用注意力TP组
            self.attn_tp_size = get_attention_tp_size()  # 获取注意力TP大小
            self.do_tensor_parallel_all_gather = (  # 是否执行TP全收集
                not skip_all_gather and self.attn_tp_size > 1  # 不跳过且TP大小>1
            )
            self.do_tensor_parallel_all_gather_dp_attn = False  # DP注意力全收集标志，默认False
        else:  # 不使用注意力TP组
            self.do_tensor_parallel_all_gather = (  # 是否执行TP全收集
                not skip_all_gather and get_tensor_model_parallel_world_size() > 1  # 不跳过且TP世界大小>1
            )
            self.do_tensor_parallel_all_gather_dp_attn = (  # DP注意力全收集标志
                self.do_tensor_parallel_all_gather and get_attention_dp_size() != 1  # TP全收集且DP大小不为1
            )
        self.final_logit_softcapping = getattr(  # 获取最终logit软裁剪参数
            self.config, "final_logit_softcapping", None  # 从配置中获取，默认None
        )
        if (  # 如果软裁剪值存在且为负
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None  # 设为None（禁用）

        self.return_full_logits = return_full_logits  # 保存是否返回完整logits标志
        self.enable_mis = get_global_server_args().enable_mis  # 是否启用多item评分

        # enable chunked logprobs processing
        # 启用分块logprobs处理
        self.enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK.get()  # 获取分块开关
        # chunk size for logprobs processing
        # logprobs处理的块大小
        self.logprobs_chunk_size = envs.SGLANG_LOGITS_PROCESSER_CHUNK_SIZE.get()  # 获取块大小

    def forward(  # 前向传播方法
        self,
        input_ids,  # 输入token ID
        hidden_states,  # 隐藏状态
        lm_head: VocabParallelEmbedding,  # 语言模型头
        logits_metadata: Union[LogitsMetadata, ForwardBatch],  # logits元数据或前向批次
        aux_hidden_states: Optional[torch.Tensor] = None,  # 辅助隐藏状态
        hidden_states_before_norm: Optional[torch.Tensor] = None,  # 归一化前的隐藏状态
    ) -> LogitsProcessorOutput:  # 返回Logits处理器输出
        # Extract MIS indices before ForwardBatch → LogitsMetadata conversion
        # 在ForwardBatch转换为LogitsMetadata之前提取MIS索引
        multi_item_delimiter_indices = None  # 初始化多item分隔符索引为None
        if isinstance(logits_metadata, ForwardBatch):  # 如果是ForwardBatch类型
            multi_item_delimiter_indices = logits_metadata.multi_item_delimiter_indices  # 提取MIS分隔符索引
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)  # 转换为LogitsMetadata

        # Multi-item scoring only for prefill-only requests with pre-computed indices.
        # 多item评分仅用于带有预计算索引的纯预填充请求。
        if multi_item_delimiter_indices is not None and logits_metadata.is_prefill_only:  # 如果有MIS索引且为纯预填充
            return self.compute_logprobs_for_multi_item_scoring(  # 返回多item评分的logprobs结果
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                multi_item_delimiter_indices,
            )

        # Diffusion LLM only.
        # 仅扩散LLM使用。
        if logits_metadata.forward_mode.is_dllm_extend():  # 如果是扩散LLM扩展模式
            return self._get_dllm_logits(hidden_states, lm_head, logits_metadata)  # 返回扩散LLM的logits

        # Get the last hidden states and last logits for the next token prediction
        # 获取用于下一个token预测的最后隐藏状态和logits
        (
            pruned_states,  # 裁剪后的隐藏状态
            pruned_states_before_norm,  # 归一化前的裁剪状态
            aux_pruned_states,  # 辅助裁剪状态
            sample_indices,  # 采样索引
            input_logprob_indices,  # 输入logprob索引
            token_to_seq_idx,  # token到序列的映射索引
        ) = self._get_pruned_states(  # 调用裁剪状态获取方法
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            logits_metadata,
        )

        hidden_states_to_store = self._get_hidden_states_to_store(  # 获取需要存储的隐藏状态
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            logits_metadata,
        )
        del hidden_states  # 释放原始隐藏状态以节省内存

        if not logits_metadata.extend_return_logprob:  # 如果不需要返回扩展logprob
            # Compute logits for both input and sampled tokens.
            # 为输入和采样token计算logits。
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)  # 计算logits
            sampled_logits = (  # 提取采样token的logits
                logits[sample_indices] if sample_indices is not None else logits
            )

            # Decode mode or extend mode without return_logprob.
            # 解码模式或不返回logprob的扩展模式。
            return LogitsProcessorOutput(  # 返回处理结果
                next_token_logits=sampled_logits,  # 采样token的logits
                hidden_states=hidden_states_to_store,  # 存储的隐藏状态
                # FIXME: These fields are not logits-related but are passed through here as a
                # workaround since ForwardBatch is local to forward_batch_generation().
                # They should be moved to GenerationBatchResult to keep this class clean.
                # FIXME: 这些字段与logits无关，但作为变通方法在此传递，
                # 因为ForwardBatch仅限于forward_batch_generation()。
                # 应移至GenerationBatchResult以保持此类的整洁。
                mm_input_embeds=logits_metadata.mm_input_embeds,  # 多模态输入嵌入
            )

        # Start to process input logprobs
        # 开始处理输入logprobs
        # Determine whether to use chunked or non-chunked logits processing.
        # 确定是否使用分块logits处理。
        # Skip chunking if:
        # 1. Chunking is disabled
        # 2. Total count is below chunk size threshold
        # 3. DP attention all-gather is enabled (can use "enable_dp_lm_head" to enable chunking)
        # 跳过分块的条件：
        # 1. 分块被禁用
        # 2. 总数低于块大小阈值
        # 3. DP注意力全收集已启用（可使用"enable_dp_lm_head"启用分块）
        should_skip_chunking = (  # 判断是否跳过分块
            not self.enable_logprobs_chunk  # 分块未启用
            or pruned_states.shape[0] <= self.logprobs_chunk_size  # token数不超过块大小
            or self.do_tensor_parallel_all_gather_dp_attn  # DP注意力全收集已启用
        )

        if should_skip_chunking:  # 如果跳过分块
            # Compute logits for both input and sampled tokens.
            # 为输入和采样token计算logits。
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)  # 计算logits
            sampled_logits = (  # 提取采样token的logits
                logits[sample_indices] if sample_indices is not None else logits
            )
            input_logits = logits[input_logprob_indices]  # 提取输入logprob的logits
            del logits  # 释放logits以节省内存

            logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)  # 处理输入logprobs
        else:  # 使用分块处理
            logprobs_result, sampled_logits = self.process_input_logprobs_by_chunk(  # 分块处理输入logprobs
                pruned_states,
                sample_indices,
                input_logprob_indices,
                token_to_seq_idx,
                lm_head,
                logits_metadata,
            )

        return LogitsProcessorOutput(  # 返回处理结果
            next_token_logits=sampled_logits,  # 采样token的logits
            hidden_states=hidden_states_to_store,  # 存储的隐藏状态
            input_token_logprobs=logprobs_result.input_token_logprobs,  # 输入token logprobs
            input_top_logprobs_val=logprobs_result.input_top_logprobs_val,  # 输入top logprobs值
            input_top_logprobs_idx=logprobs_result.input_top_logprobs_idx,  # 输入top logprobs索引
            input_token_ids_logprobs_val=logprobs_result.input_token_ids_logprobs_val,  # 输入token ID logprobs值
            input_token_ids_logprobs_idx=logprobs_result.input_token_ids_logprobs_idx,  # 输入token ID logprobs索引
            mm_input_embeds=logits_metadata.mm_input_embeds,  # 多模态输入嵌入
        )

    def _get_pruned_states(  # 获取裁剪后的隐藏状态
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        hidden_states_before_norm: Optional[torch.Tensor],  # 归一化前的隐藏状态
        aux_hidden_states: Optional[torch.Tensor],  # 辅助隐藏状态
        logits_metadata: LogitsMetadata,  # logits元数据
    ):
        pruned_states_before_norm: Optional[torch.Tensor] = None  # 初始化归一化前裁剪状态
        aux_pruned_states = None  # 初始化辅助裁剪状态
        token_to_seq_idx = []  # 初始化token到序列索引映射

        if (  # 如果是解码/空闲模式、目标验证模式或草稿扩展v2模式
            logits_metadata.forward_mode.is_decode_or_idle()
            or logits_metadata.forward_mode.is_target_verify()
            or logits_metadata.forward_mode.is_draft_extend_v2()
        ):
            pruned_states = hidden_states  # 解码模式下不需要裁剪
            pruned_states_before_norm = hidden_states_before_norm  # 保留归一化前状态
            if aux_hidden_states is not None:  # 如果有辅助隐藏状态
                aux_pruned_states = [hidden for hidden in aux_hidden_states]  # 复制辅助状态列表
            sample_indices = None  # 采样索引为None
            input_logprob_indices = None  # 输入logprob索引为None

        elif (  # 如果是扩展模式且不需要返回logprob
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):
            # Prefill without input logprobs.
            # 无输入logprobs的预填充。
            if logits_metadata.padded_static_len < 0:  # 如果没有静态填充
                last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1  # 计算每个序列最后token的索引
            else:  # 有静态填充
                # If padding_static length is 5 and extended_seq_lens is [2, 3],
                # then our batch looks like [t00, t01, p, p, p, t10, t11, t12, p, p]
                # and this retrieves t01 and t12, which are the valid last tokens
                # 如果padding_static长度为5，extended_seq_lens为[2, 3]，
                # 则批次为[t00, t01, p, p, p, t10, t11, t12, p, p]，
                # 此处获取t01和t12，即有效的最后token
                idx = torch.arange(  # 创建序列索引
                    len(logits_metadata.extend_seq_lens),
                    device=logits_metadata.extend_seq_lens.device,
                )
                last_index = (  # 计算带填充的最后token索引
                    idx * logits_metadata.padded_static_len
                    + logits_metadata.extend_seq_lens
                    - 1
                )
            pruned_states = hidden_states[last_index]  # 提取最后token的隐藏状态
            if hidden_states_before_norm is not None:  # 如果有归一化前状态
                pruned_states_before_norm = hidden_states_before_norm[last_index]  # 提取对应的归一化前状态
            if aux_hidden_states is not None:  # 如果有辅助隐藏状态
                aux_pruned_states = [hidden[last_index] for hidden in aux_hidden_states]  # 提取辅助状态
            sample_indices = None  # 采样索引为None
            input_logprob_indices = None  # 输入logprob索引为None
        else:  # 扩展模式且需要返回logprob
            # Prefill with input logprobs.
            # 带输入logprobs的预填充。
            # Find 4 different indices.
            # 查找4种不同的索引。
            # 1. pruned_states: hidden states that we want logprobs from.
            # 1. pruned_states：需要计算logprobs的隐藏状态。
            # 2. sample_indices: Indices that have sampled tokens.
            # 2. sample_indices：有采样token的索引。
            # 3. input_logprob_indices: Indices that have input logprob tokens.
            # 3. input_logprob_indices：有输入logprob token的索引。
            # 4. token_to_seq_idx: map each token to its sequence index
            # 4. token_to_seq_idx：将每个token映射到其序列索引
            #
            # Example
            # 示例
            # -------
            # Suppose a batch (flattened by sequence):
            # 假设一个批次（按序列展平）：
            # [t00, t01, t02, t03, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t25]
            # extend_seq_lens_cpu           = [4, 5, 6]
            # extend_logprob_start_lens_cpu = [0, 5, 3]
            #
            # Then, the indices are:
            # 那么，索引为：
            # pruned_states         -> [t00, t01, t02, t03, t14, t23, t24, t25]
            # sample_indices        -> [3, 4, 7]
            # input_logprob_indices -> [0, 1, 2, 3, 5, 6, 7]
            # token_to_seq_idx      -> [0, 0, 0, 0, 1, 2, 2, 2]
            #
            # If chunk is enabled and chunk_size = 3, the chunks will be computed in a chunked manner:
            # 如果启用分块且chunk_size = 3，则分块计算：
            # [t00, t01, t02], [t03, t14, t23], [t24, t25]

            sample_index_pt = -1  # 采样索引指针，初始为-1
            sample_indices = []  # 采样索引列表
            input_logprob_indices_pt = 0  # 输入logprob索引指针
            input_logprob_indices = []  # 输入logprob索引列表
            pt, pruned_states_list, pruned_states_before_norm_list = 0, [], []  # 指针和状态列表
            aux_pruned_states_lists = (  # 辅助裁剪状态列表
                [[] for _ in aux_hidden_states]  # 为每个辅助状态创建列表
                if aux_hidden_states is not None  # 如果有辅助状态
                else None  # 否则为None
            )

            for idx, (extend_logprob_start_len, extend_len) in enumerate(  # 遍历每个序列
                zip(
                    logits_metadata.extend_logprob_start_lens_cpu,
                    logits_metadata.extend_seq_lens_cpu,
                )
            ):
                # It can happen in chunked prefill. We still need to sample 1 token,
                # But we don't want to include it in input logprob.
                # 分块预填充中可能发生。我们仍需采样1个token，
                # 但不想将其包含在输入logprob中。
                if extend_len == extend_logprob_start_len:  # 如果扩展长度等于logprob起始长度
                    start_len = extend_logprob_start_len - 1  # 起始长度回退1
                else:  # 正常情况
                    start_len = extend_logprob_start_len  # 使用logprob起始长度

                # We always need at least 1 token to sample because that's required
                # by a caller.
                # 我们始终需要至少1个token进行采样，因为调用者要求如此。
                assert extend_len > start_len  # 断言扩展长度大于起始长度
                pruned_states_list.append(  # 添加裁剪后的隐藏状态片段
                    hidden_states[pt + start_len : pt + extend_len]
                )
                if hidden_states_before_norm is not None:  # 如果有归一化前状态
                    pruned_states_before_norm_list.append(  # 添加归一化前状态片段
                        hidden_states_before_norm[pt + start_len : pt + extend_len]
                    )
                if aux_pruned_states_lists is not None:  # 如果有辅助状态列表
                    for j, hidden in enumerate(aux_hidden_states):  # 遍历辅助状态
                        aux_pruned_states_lists[j].append(  # 添加辅助状态片段
                            hidden[pt + start_len : pt + extend_len]
                        )
                # Map each token to its sequence index, for chunked computation
                # of input logprobs
                # 将每个token映射到其序列索引，用于分块计算输入logprobs
                token_to_seq_idx.extend([idx] * (extend_len - start_len))  # 添加序列索引映射
                pt += extend_len  # 更新指针
                sample_index_pt += extend_len - start_len  # 更新采样索引指针
                sample_indices.append(sample_index_pt)  # 添加采样索引
                input_logprob_indices.extend(  # 添加输入logprob索引
                    [
                        input_logprob_indices_pt + i  # 基于当前指针计算索引
                        for i in range(extend_len - extend_logprob_start_len)  # 遍历logprob范围
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len  # 更新logprob索引指针

            # Set the last token of the last sequence
            # 设置最后一个序列的最后一个token
            token_to_seq_idx.append(len(logits_metadata.extend_seq_lens_cpu) - 1)  # 添加最后一个序列索引
            pruned_states = torch.cat(pruned_states_list)  # 拼接所有裁剪后的状态
            if hidden_states_before_norm is not None:  # 如果有归一化前状态
                pruned_states_before_norm = torch.cat(pruned_states_before_norm_list)  # 拼接归一化前状态
            if aux_pruned_states_lists is not None:  # 如果有辅助状态
                aux_pruned_states = [torch.cat(lst) for lst in aux_pruned_states_lists]  # 拼接辅助状态

            # Build the index tensors via pinned host memory + non-blocking H2D
            # so the small copy doesn't drain the stream.
            # 通过锁页主机内存+非阻塞H2D构建索引张量，
            # 这样小的复制不会占用流。
            sample_indices = torch.tensor(  # 创建采样索引张量
                sample_indices,  # 索引列表
                dtype=torch.int64,  # 64位整数类型
                pin_memory=is_pin_memory_available(),  # 如果可用则使用锁页内存
            ).to(pruned_states.device, non_blocking=True)  # 非阻塞传输到设备
            input_logprob_indices = torch.tensor(  # 创建输入logprob索引张量
                input_logprob_indices,  # 索引列表
                dtype=torch.int64,  # 64位整数类型
                pin_memory=is_pin_memory_available(),  # 如果可用则使用锁页内存
            ).to(pruned_states.device, non_blocking=True)  # 非阻塞传输到设备

        return (  # 返回所有裁剪结果
            pruned_states,  # 裁剪后的隐藏状态
            pruned_states_before_norm,  # 归一化前的裁剪状态
            aux_pruned_states,  # 辅助裁剪状态
            sample_indices,  # 采样索引
            input_logprob_indices,  # 输入logprob索引
            token_to_seq_idx,  # token到序列的映射
        )

    def _get_hidden_states_to_store(  # 获取需要存储的隐藏状态
        self,
        hidden_states: torch.Tensor,  # 原始隐藏状态
        hidden_states_before_norm: Optional[torch.Tensor],  # 归一化前的隐藏状态
        aux_hidden_states: Optional[List[torch.Tensor]],  # 辅助隐藏状态列表
        pruned_states: torch.Tensor,  # 裁剪后的隐藏状态
        pruned_states_before_norm: Optional[torch.Tensor],  # 归一化前的裁剪状态
        aux_pruned_states: Optional[List[torch.Tensor]],  # 辅助裁剪状态列表
        sample_indices: Optional[torch.Tensor],  # 采样索引
        logits_metadata: LogitsMetadata,  # logits元数据
    ) -> Optional[torch.Tensor]:  # 返回可选的隐藏状态张量
        hidden_states_to_store: Optional[torch.Tensor] = None  # 初始化待存储隐藏状态
        hidden_states_to_store_before_norm: Optional[torch.Tensor] = None  # 初始化归一化前待存储状态
        if logits_metadata.capture_hidden_mode.need_capture():  # 如果需要捕获隐藏状态
            if logits_metadata.capture_hidden_mode.is_full():  # 如果是全量捕获模式
                if aux_hidden_states is not None:  # 如果有辅助隐藏状态
                    aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)  # 拼接辅助状态
                    hidden_states_to_store = aux_hidden_states  # 使用辅助状态作为存储
                else:  # 没有辅助状态
                    hidden_states_to_store = hidden_states  # 使用原始隐藏状态
                hidden_states_to_store_before_norm = hidden_states_before_norm  # 保留归一化前状态
            elif logits_metadata.capture_hidden_mode.is_last():  # 如果是最后token捕获模式
                # Get the last token hidden states. If sample_indices is None,
                # pruned states only contain the last tokens already.
                # 获取最后token的隐藏状态。如果sample_indices为None，
                # 裁剪后的状态已经只包含最后token。
                if aux_hidden_states is not None:  # 如果有辅助隐藏状态
                    aux_pruned_states = torch.cat(aux_pruned_states, dim=-1)  # 拼接辅助裁剪状态
                    hidden_states_to_store = (  # 提取最后token的辅助状态
                        aux_pruned_states[sample_indices]
                        if sample_indices is not None
                        else aux_pruned_states
                    )
                else:  # 没有辅助状态
                    hidden_states_to_store = (  # 提取最后token的裁剪状态
                        pruned_states[sample_indices]
                        if sample_indices is not None
                        else pruned_states
                    )
                    if hidden_states_before_norm is not None:  # 如果有归一化前状态
                        hidden_states_to_store_before_norm = (  # 提取最后token的归一化前状态
                            pruned_states_before_norm[sample_indices]
                            if sample_indices is not None
                            else pruned_states_before_norm
                        )
            else:  # 其他模式
                assert False, "Should never reach"  # 不应到达此处

        if hidden_states_to_store_before_norm is not None:  # 如果有归一化前状态
            # NOTE: when hidden_states_before_norm is provided, we always
            # prefer to return it.
            # 注意：当提供了归一化前的隐藏状态时，我们始终优先返回它。
            hidden_states_to_store = hidden_states_to_store_before_norm  # 使用归一化前状态

        return hidden_states_to_store  # 返回待存储的隐藏状态

    def process_input_logprobs(self, input_logits, logits_metadata: LogitsMetadata):  # 处理输入logprobs（非分块）
        input_logprobs = torch.nn.functional.log_softmax(input_logits, dim=-1)  # 对输入logits进行log_softmax

        # Get the logprob of top-k tokens
        # 获取top-k token的logprob
        if logits_metadata.extend_return_top_logprob:  # 如果需要返回top logprob
            (
                input_top_logprobs_val,  # top logprobs值
                input_top_logprobs_idx,  # top logprobs索引
            ) = get_top_logprobs_prefill(input_logprobs, logits_metadata)  # 计算top logprobs
        else:  # 不需要top logprob
            input_top_logprobs_val = input_top_logprobs_idx = None  # 设为None

        # Get the logprob of given token id
        # 获取给定token ID的logprob
        if logits_metadata.extend_token_ids_logprob:  # 如果需要token ID logprob
            (
                input_token_ids_logprobs_val,  # token ID logprobs值
                input_token_ids_logprobs_idx,  # token ID logprobs索引
            ) = get_token_ids_logprobs_prefill(input_logprobs, logits_metadata)  # 计算token ID logprobs
        else:  # 不需要token ID logprob
            input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None  # 设为None

        input_token_logprobs = input_logprobs[  # 提取每个位置目标token的logprob
            torch.arange(input_logprobs.shape[0], device=input_logprobs.device),  # 行索引
            logits_metadata.extend_input_logprob_token_ids_gpu,  # 列索引（目标token ID）
        ]

        return InputLogprobsResult(  # 返回输入logprobs结果
            input_token_logprobs=input_token_logprobs,  # 输入token logprobs
            input_top_logprobs_val=input_top_logprobs_val,  # top logprobs值
            input_top_logprobs_idx=input_top_logprobs_idx,  # top logprobs索引
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,  # token ID logprobs值
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,  # token ID logprobs索引
        )

    def process_input_logprobs_by_chunk(  # 分块处理输入logprobs
        self,
        pruned_states: torch.Tensor,  # 裁剪后的隐藏状态
        sample_indices: torch.Tensor,  # 采样索引
        input_logprob_indices: torch.Tensor,  # 输入logprob索引
        token_to_seq_idx: list[int],  # token到序列索引映射
        lm_head: VocabParallelEmbedding,  # 语言模型头
        logits_metadata: LogitsMetadata,  # logits元数据
    ) -> Tuple[InputLogprobsResult, torch.Tensor]:  # 返回logprobs结果和采样logits
        """
        compute logprobs for the output token from the hidden states.
        To avoid using too much memory, we split pruned_states into chunks of
        rows to compute input_logprobs separately, then concatenate the results.
        从隐藏状态计算输出token的logprobs。
        为避免使用过多内存，将pruned_states按行分块分别计算输入logprobs，然后拼接结果。

        Returns:
            InputLogprobsResult: logprobs result
            InputLogprobsResult: logprobs结果
            torch.Tensor: sampled logits
            torch.Tensor: 采样logits
        """

        # The peak memory usage is proportional to the chunk size.
        # 峰值内存使用与块大小成正比。
        chunk_size = self.logprobs_chunk_size  # 获取块大小
        total_size = pruned_states.shape[0]  # 获取总token数
        num_chunks = (total_size + chunk_size - 1) // chunk_size  # 计算块数

        input_token_logprobs = []  # 初始化输入token logprobs列表
        if logits_metadata.extend_return_top_logprob:  # 如果需要top logprob
            input_top_logprobs_val = []  # top logprobs值列表
            input_top_logprobs_idx = []  # top logprobs索引列表
        else:  # 不需要top logprob
            input_top_logprobs_val = None  # 设为None
            input_top_logprobs_idx = None  # 设为None
        if logits_metadata.extend_token_ids_logprob:  # 如果需要token ID logprob
            input_token_ids_logprobs_val = []  # token ID logprobs值列表
            input_token_ids_logprobs_idx = []  # token ID logprobs索引列表
        else:  # 不需要token ID logprob
            input_token_ids_logprobs_val = None  # 设为None
            input_token_ids_logprobs_idx = None  # 设为None

        # If a single sequence is split into multiple chunks, we need to keep track
        # of the pruned length of the sequences in the previous chunks.
        # 如果单个序列被分成多个块，需要跟踪前面块中序列的裁剪长度。
        split_len_topk = 0  # topk已分割长度
        split_len_token_ids = 0  # token_ids已分割长度

        for i in range(num_chunks):  # 遍历每个块
            start_idx = i * chunk_size  # 计算块起始索引
            end_idx = min((i + 1) * chunk_size, total_size)  # 计算块结束索引

            # Notify lm_head LoRA about the current chunk so it can swap
            # to the precomputed per-chunk batch_info.  This is a no-op
            # for non-LoRA lm_head modules.
            # 通知lm_head LoRA当前块，以便切换到预计算的每块batch_info。
            # 对非LoRA的lm_head模块无操作。
            if hasattr(lm_head, "set_lm_head_pass"):  # 如果lm_head有set_lm_head_pass方法
                lm_head.set_lm_head_pass(i)  # 设置当前块索引

            # Get indices for this chunk
            # 获取当前块的索引
            chunk_mask = (input_logprob_indices >= start_idx) & (  # 创建块掩码
                input_logprob_indices < end_idx
            )
            global_indices = input_logprob_indices[chunk_mask]  # 获取块内的全局索引
            chunk_indices = global_indices - start_idx  # 转换为块内局部索引
            # Get the positions in the original array where chunk_mask is True
            # This is needed to correctly index into extend_input_logprob_token_ids_gpu
            # 获取原始数组中chunk_mask为True的位置
            # 这对于正确索引extend_input_logprob_token_ids_gpu是必需的
            mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]  # 获取掩码为True的位置

            # Get the logits for this chunk
            # 获取当前块的logits
            chunk_states = pruned_states[start_idx:end_idx]  # 提取当前块的隐藏状态
            chunk_logits = self._get_logits(chunk_states, lm_head, logits_metadata)  # 计算块内logits

            # Initialize sampled_logits on first chunk
            # 在第一个块上初始化sampled_logits
            if i == 0:  # 如果是第一个块
                sampled_logits = torch.empty(  # 分配采样logits张量
                    (sample_indices.shape[0], chunk_logits.shape[1]),  # 形状：[采样数, 词表大小]
                    dtype=chunk_logits.dtype,  # 数据类型
                    device=chunk_logits.device,  # 设备
                )

            # Handle sampled logits for the chunk if needed
            # This must be done before the continue statement to ensure all sampled_logits are filled
            # 如果需要，处理当前块的采样logits
            # 必须在continue语句之前完成，以确保所有sampled_logits都被填充
            chunk_sample_mask = (sample_indices >= start_idx) & (  # 创建采样掩码
                sample_indices < end_idx
            )
            if chunk_sample_mask.any():  # 如果当前块有采样token
                chunk_sample_indices = sample_indices[chunk_sample_mask] - start_idx  # 转换为块内索引
                sampled_logits[chunk_sample_mask] = chunk_logits[chunk_sample_indices]  # 填充采样logits

            # If there are no input logprobs in this chunk, skip the rest
            # 如果当前块没有输入logprobs，跳过剩余处理
            if chunk_indices.numel() == 0:  # 如果没有输入logprob索引
                continue  # 跳到下一个块

            # Compute the logprobs of the chunk
            # 计算当前块的logprobs
            chunk_input_logprobs = chunk_logits[chunk_indices]  # 提取输入logprob位置的logits
            chunk_input_logprobs = torch.nn.functional.log_softmax(  # 对块内logits进行log_softmax
                chunk_input_logprobs, dim=-1
            )

            # For each chunk, we need to get the slice of the token_to_seq_idx
            # 对于每个块，需要获取token_to_seq_idx的切片
            chunk_slice = slice(  # 创建切片
                token_to_seq_idx[start_idx], token_to_seq_idx[end_idx] + 1
            )

            # Get the logprob of top-k tokens
            # 获取top-k token的logprob
            if logits_metadata.extend_return_top_logprob:  # 如果需要top logprob
                top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]  # 当前块的top-k数量
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[  # 当前块的裁剪长度
                    chunk_slice
                ]
                split_len_topk = get_top_logprobs_chunk(  # 分块计算top logprobs
                    chunk_input_logprobs,
                    logits_metadata,
                    top_k_nums,
                    pruned_lens,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    split_len_topk,
                )

            # Get the logprob of given token id
            # 获取给定token ID的logprob
            if logits_metadata.extend_token_ids_logprob:  # 如果需要token ID logprob
                token_ids_logprobs = logits_metadata.token_ids_logprobs[chunk_slice]  # 当前块的token ID列表
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[  # 当前块的裁剪长度
                    chunk_slice
                ]
                split_len_token_ids = get_token_ids_logprobs_chunk(  # 分块计算token ID logprobs
                    chunk_input_logprobs,
                    token_ids_logprobs,
                    pruned_lens,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    split_len_token_ids,
                )

            # Get the logprob of the requested token ids
            # 获取请求token ID的logprob
            chunk_input_token_logprobs = chunk_input_logprobs[  # 提取目标token的logprob
                torch.arange(
                    chunk_input_logprobs.shape[0], device=chunk_input_logprobs.device
                ),
                logits_metadata.extend_input_logprob_token_ids_gpu[mask_indices],  # 使用掩码索引
            ]
            input_token_logprobs.append(chunk_input_token_logprobs)  # 添加到结果列表

        # Restore the full-pruned lm_head batch_info after chunk iteration.
        # 分块迭代后恢复完整裁剪的lm_head batch_info。
        if hasattr(lm_head, "reset_lm_head_pass"):  # 如果有reset方法
            assert hasattr(  # 断言同时有set和reset方法
                lm_head, "set_lm_head_pass"
            ), "lm_head must have set_lm_head_pass method and reset_lm_head_pass method at the same time"
            lm_head.reset_lm_head_pass()  # 重置lm_head块索引

        # Concatenate the results
        # 拼接结果
        input_token_logprobs = torch.cat(input_token_logprobs, dim=0)  # 拼接所有块的输入token logprobs

        return (  # 返回结果
            InputLogprobsResult(  # 创建输入logprobs结果
                input_token_logprobs=input_token_logprobs,  # 输入token logprobs
                input_top_logprobs_val=input_top_logprobs_val,  # top logprobs值
                input_top_logprobs_idx=input_top_logprobs_idx,  # top logprobs索引
                input_token_ids_logprobs_val=input_token_ids_logprobs_val,  # token ID logprobs值
                input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,  # token ID logprobs索引
            ),
            sampled_logits,  # 采样logits
        )

    def _get_logits(  # 获取logits的内部方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        lm_head: VocabParallelEmbedding,  # 语言模型头
        logits_metadata: LogitsMetadata,  # logits元数据
        embedding_bias: Optional[torch.Tensor] = None,  # 嵌入偏置
    ) -> torch.Tensor:  # 返回logits张量
        """Get logits from hidden_states.
        从隐藏状态获取logits。

        If sampled_logits_only is True, it means hidden_states only contain the
        last position (e.g., extend without input logprobs). The caller should
        guarantee the given hidden_states follow this constraint.
        如果sampled_logits_only为True，表示hidden_states只包含最后位置
        （例如不返回输入logprobs的扩展模式）。调用者应确保给定的hidden_states遵循此约束。
        """
        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(  # 收集DP注意力隐藏状态
            hidden_states, logits_metadata
        )

        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)  # 计算lm_head的logits

        if self.logit_scale is not None:  # 如果有logit缩放因子
            logits.mul_(self.logit_scale)  # 原地乘以缩放因子

        if self.do_tensor_parallel_all_gather:  # 如果需要TP全收集
            if self.use_attn_tp_group:  # 如果使用注意力TP组
                logits = self._gather_attn_tp_logits(logits)  # 收集注意力TP logits
            else:  # 不使用注意力TP组
                logits = tensor_model_parallel_all_gather(logits)  # 常规TP全收集

        logits = self._scatter_dp_attn_logits(  # 散射DP注意力logits
            logits, local_hidden_states, logits_metadata
        )

        logits = self._copy_logits_to_buffer(logits, logits_metadata)  # 复制logits到缓冲区

        if self.final_logit_softcapping:  # 如果有logit软裁剪
            if not _is_npu:  # 如果不是NPU
                fused_softcap(logits, self.final_logit_softcapping)  # 使用融合软裁剪内核
            else:  # NPU平台
                logits = self.final_logit_softcapping * torch.tanh(  # 手动计算软裁剪
                    logits / self.final_logit_softcapping
                )

        return logits  # 返回logits

    def _compute_lm_head(  # 计算lm_head的内部方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        lm_head: VocabParallelEmbedding,  # 语言模型头
        embedding_bias: Optional[torch.Tensor] = None,  # 嵌入偏置
    ) -> torch.Tensor:  # 返回logits张量
        if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):  # 如果是LoRA模块
            # This is a LoRA-wrapped module, use its forward method
            # 这是LoRA包装的模块，使用其forward方法
            logits = lm_head(hidden_states)  # 通过LoRA前向传播
        elif hasattr(lm_head, "weight"):  # 如果是普通线性层
            # Normal linear layer
            # 普通线性层
            if self.use_fp32_lm_head:  # 如果使用fp32 lm_head
                logits = torch.matmul(  # 使用fp32精度计算
                    hidden_states.to(torch.float32), lm_head.weight.to(torch.float32).T
                )
            elif use_intel_amx_backend(lm_head):  # 如果使用Intel AMX后端
                logits = torch.ops.sgl_kernel.weight_packed_linear(  # 使用AMX打包线性计算
                    hidden_states.to(lm_head.weight.dtype),
                    lm_head.weight,
                    None,  # bias
                    True,  # is_vnni
                )
            elif get_global_server_args().rl_on_policy_target is not None:  # 如果有RL策略目标
                # Due to tie-weight, we may not be able to change lm_head's weight dtype
                # 由于权重绑定，可能无法更改lm_head的权重数据类型
                logits = torch.matmul(  # 使用bfloat16精度计算
                    hidden_states.bfloat16(), lm_head.weight.T.bfloat16()
                )
            else:  # 默认路径
                logits = torch.matmul(  # 使用权重精度计算
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:  # GGUF模型
            # GGUF models
            # GGUF模型
            # TODO: use weight_packed_linear for GGUF models
            # TODO: 为GGUF模型使用weight_packed_linear
            if self.use_fp32_lm_head:  # 如果使用fp32 lm_head
                with torch.cuda.amp.autocast(enabled=False):  # 禁用自动混合精度
                    logits = lm_head.quant_method.apply(  # 使用量化方法计算
                        lm_head, hidden_states.to(torch.float32), embedding_bias
                    )
            else:  # 不使用fp32
                logits = lm_head.quant_method.apply(  # 使用量化方法计算
                    lm_head, hidden_states, embedding_bias
                )
        return logits  # 返回logits

    def _gather_dp_attn_hidden_states(  # 收集DP注意力隐藏状态
        self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata  # 隐藏状态和元数据
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回全局和本地隐藏状态
        if self.do_tensor_parallel_all_gather_dp_attn:  # 如果需要DP注意力全收集
            logits_metadata.compute_dp_attention_metadata()  # 计算DP注意力元数据
            local_hidden_states = hidden_states  # 保存本地隐藏状态
            hidden_states = logits_metadata.gathered_buffer  # 使用收集缓冲区
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)  # 执行DP收集复制
            return hidden_states, local_hidden_states  # 返回全局和本地状态
        return hidden_states, hidden_states  # 不需要收集时，两个返回值相同

    def _gather_attn_tp_logits(self, logits: torch.Tensor) -> torch.Tensor:  # 收集注意力TP logits
        if self.vocab_size % self.attn_tp_size == 0:  # 如果词表大小可被TP大小整除
            global_logits = torch.empty(  # 分配全局logits张量
                (
                    self.attn_tp_size,  # TP大小维度
                    logits.shape[0],  # token数维度
                    self.vocab_size // self.attn_tp_size,  # 每个TP分片的词表大小
                ),
                device=logits.device,  # 设备
                dtype=logits.dtype,  # 数据类型
            )
            attn_tp_all_gather_into_tensor(global_logits, logits)  # 执行TP全收集到张量
            global_logits = global_logits.permute(1, 0, 2).reshape(  # 重排并重塑形状
                logits.shape[0], self.vocab_size
            )
        else:  # 词表大小不可被TP大小整除
            global_logits = torch.empty(  # 分配全局logits张量
                (self.vocab_size, logits.shape[0]),  # 形状：[词表大小, token数]
                device=logits.device,  # 设备
                dtype=logits.dtype,  # 数据类型
            )
            global_logits = global_logits.T  # 转置
            attn_tp_all_gather(  # 执行TP全收集
                list(global_logits.tensor_split(self.attn_tp_size, dim=-1)),  # 按TP大小分割
                logits,
            )
        return global_logits  # 返回全局logits

    def _scatter_dp_attn_logits(  # 散射DP注意力logits
        self,
        logits: torch.Tensor,  # 全局logits
        local_hidden_states: torch.Tensor,  # 本地隐藏状态
        logits_metadata: LogitsMetadata,  # logits元数据
    ) -> torch.Tensor:  # 返回本地logits
        if self.do_tensor_parallel_all_gather_dp_attn:  # 如果需要DP注意力散射
            global_logits = logits  # 保存全局logits
            logits = torch.empty(  # 分配本地logits
                (local_hidden_states.shape[0], global_logits.shape[1]),  # 形状
                device=global_logits.device,  # 设备
                dtype=global_logits.dtype,  # 数据类型
            )
            dp_scatter(logits, global_logits, logits_metadata)  # 执行DP散射
        return logits  # 返回logits

    def _copy_logits_to_buffer(  # 复制logits到缓冲区
        self, logits: torch.Tensor, logits_metadata: LogitsMetadata  # logits和元数据
    ) -> torch.Tensor:  # 返回logits
        if logits_metadata.next_token_logits_buffer is not None:  # 如果有logits缓冲区
            logits_buffer = logits_metadata.next_token_logits_buffer  # 获取缓冲区
            assert logits_buffer.dtype == torch.float  # 断言缓冲区为float类型
            logits_buffer.copy_(logits[:, : self.vocab_size])  # 复制有效词表范围的logits
            logits = logits_buffer  # 使用缓冲区作为结果
        else:  # 没有缓冲区
            logits = logits[:, : self.vocab_size].float()  # 截取有效词表范围并转为float
        return logits  # 返回logits

    def _get_dllm_logits(  # 获取扩散LLM的logits
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        lm_head: VocabParallelEmbedding,  # 语言模型头
        logits_metadata: LogitsMetadata,  # logits元数据
    ) -> LogitsProcessorOutput:  # 返回Logits处理器输出
        assert self.return_full_logits  # 断言需要返回完整logits
        full_logits = self._get_logits(hidden_states, lm_head, logits_metadata)  # 计算完整logits
        return LogitsProcessorOutput(  # 返回处理结果
            full_logits=full_logits,  # 完整logits
            next_token_logits=None,  # 下一个token logits为None
        )

    def compute_logprobs_for_multi_item_scoring(  # 计算多item评分的logprobs
        self,
        input_ids,  # 输入token ID
        hidden_states,  # 隐藏状态
        lm_head: VocabParallelEmbedding,  # 语言模型头
        logits_metadata: Union[LogitsMetadata, ForwardBatch],  # logits元数据
        multi_item_delimiter_indices: List[torch.Tensor],  # 多item分隔符索引列表
    ):
        """
        Compute logprobs for multi-item scoring using pre-computed delimiter indices.
        使用预计算的分隔符索引计算多item评分的logprobs。

        Sequence format: Query<delimiter>Item1<delimiter>Item2<delimiter>...
        序列格式：Query<delimiter>Item1<delimiter>Item2<delimiter>...
        Scoring positions: Extracts logprobs at positions before each <delimiter>
        评分位置：提取每个<delimiter>之前位置的logprobs

        Args:
            input_ids: Input token IDs. Shape: [total_sequence_length].
            input_ids: 输入token ID。形状：[total_sequence_length]。
            hidden_states: Hidden states from the model. Shape: [sequence_length, hidden_dim].
            hidden_states: 模型的隐藏状态。形状：[sequence_length, hidden_dim]。
            lm_head: Language model head for computing logits.
            lm_head: 用于计算logits的语言模型头。
            logits_metadata: Metadata containing batch info and logprob specs.
            logits_metadata: 包含批次信息和logprob规格的元数据。
            multi_item_delimiter_indices: Pre-computed delimiter positions per request (CPU tensors).
            multi_item_delimiter_indices: 每个请求的预计算分隔符位置（CPU张量）。
        """
        # Compute positions just before each delimiter.
        # 计算每个分隔符之前的位置。
        # Build offset-adjusted indices on CPU, then do a single CPU→GPU transfer.
        # 在CPU上构建偏移调整索引，然后进行一次CPU→GPU传输。
        device = input_ids.device  # 获取设备
        all_tensors = []  # 初始化张量列表
        if logits_metadata.extend_seq_lens_cpu is not None:  # 如果有序列长度信息
            offset = 0  # 初始化偏移
            for req_seq_len, indices_tensor in zip(  # 遍历每个请求
                logits_metadata.extend_seq_lens_cpu, multi_item_delimiter_indices
            ):
                if len(indices_tensor) > 0:  # 如果有分隔符索引
                    # Note: if the first delimiter is at position 0 (empty query),
                    # indices - 1 wraps to -1. This is harmless — the first
                    # delimiter entry is always discarded by
                    # _process_multi_item_scoring_results.
                    # 注意：如果第一个分隔符在位置0（空查询），indices - 1会变为-1。
                    # 这是无害的——第一个分隔符条目总是被_process_multi_item_scoring_results丢弃。
                    all_tensors.append(indices_tensor + (offset - 1))  # 添加偏移调整后的索引
                offset += req_seq_len  # 更新偏移
        else:  # 没有序列长度信息
            all_tensors.append(multi_item_delimiter_indices[0] - 1)  # 添加分隔符前一个位置
        multi_item_indices = torch.cat(all_tensors).to(device, non_blocking=True)  # 拼接并传输到GPU

        # Extract hidden states at delimiter positions for multi-item scoring
        # 提取分隔符位置的隐藏状态用于多item评分
        sliced_hidden = hidden_states[multi_item_indices]  # 切片获取分隔符位置的隐藏状态

        sliced_logits = self._get_logits(sliced_hidden, lm_head, logits_metadata)  # 计算切片logits
        sliced_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1)  # 计算log softmax

        # Initialize return values
        # 初始化返回值
        input_token_ids_logprobs_val = []  # token ID logprobs值列表
        input_token_ids_logprobs_idx = []  # token ID logprobs索引列表
        input_top_logprobs_val = None  # top logprobs值
        input_top_logprobs_idx = None  # top logprobs索引

        # Recalculate extend_logprob_pruned_lens_cpu to match delimiter counts per request
        # 重新计算extend_logprob_pruned_lens_cpu以匹配每个请求的分隔符数量
        if (  # 如果需要token ID logprob或top logprob
            logits_metadata.token_ids_logprobs
            or logits_metadata.extend_return_top_logprob
        ):
            logits_metadata.extend_logprob_pruned_lens_cpu = [  # 更新裁剪长度
                len(t) for t in multi_item_delimiter_indices  # 使用分隔符数量
            ]

        # Get the logprobs of specified token ids
        # 获取指定token ID的logprobs
        if logits_metadata.extend_token_ids_logprob:  # 如果需要token ID logprob
            (
                input_token_ids_logprobs_val,  # token ID logprobs值
                input_token_ids_logprobs_idx,  # token ID logprobs索引
            ) = get_token_ids_logprobs_prefill(  # 计算token ID logprobs
                sliced_logprobs, logits_metadata, no_copy_to_cpu=True  # 不复制到CPU
            )

        # Get the logprob of top-k tokens
        # 获取top-k token的logprob
        if logits_metadata.extend_return_top_logprob:  # 如果需要top logprob
            (
                input_top_logprobs_val,  # top logprobs值
                input_top_logprobs_idx,  # top logprobs索引
            ) = get_top_logprobs_prefill(sliced_logprobs, logits_metadata)  # 计算top logprobs

        # MIS scores come from input_token_ids_logprobs_val (label-token logprobs),
        # not from per-position input_token_logprobs. However, the shared logprob
        # pipeline (add_input_logprob_return_values) asserts input_token_logprobs is
        # non-None, converts it to a tuple, slices it, and validates its length —
        # all before score_request() ever sees the result. We can't set it to None
        # without changing those shared asserts, so we fill with zeros to satisfy
        # the pipeline. score_request() ignores this field entirely.
        # MIS分数来自input_token_ids_logprobs_val（标签token的logprobs），
        # 而非逐位置的input_token_logprobs。然而，共享的logprob管道
        # （add_input_logprob_return_values）断言input_token_logprobs不为None，
        # 将其转换为元组、切片并验证长度——这些都在score_request()看到结果之前完成。
        # 我们不能设为None而不修改那些共享断言，因此用零填充以满足管道。
        # score_request()完全忽略此字段。
        input_token_logprobs = torch.zeros(multi_item_indices.shape[0], device=device)  # 用零填充

        return LogitsProcessorOutput(  # 返回处理结果
            next_token_logits=None,  # 无下一个token logits
            input_token_logprobs=input_token_logprobs,  # 输入token logprobs（零填充）
            input_top_logprobs_val=input_top_logprobs_val,  # top logprobs值
            input_top_logprobs_idx=input_top_logprobs_idx,  # top logprobs索引
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,  # token ID logprobs值
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,  # token ID logprobs索引
            # FIXME: These fields are not logits-related but are passed through here as a
            # workaround since ForwardBatch is local to forward_batch_generation().
            # They should be moved to GenerationBatchResult to keep this class clean.
            # FIXME: 这些字段与logits无关，但作为变通方法在此传递，
            # 因为ForwardBatch仅限于forward_batch_generation()。
            # 应移至GenerationBatchResult以保持此类的整洁。
            mm_input_embeds=logits_metadata.mm_input_embeds,  # 多模态输入嵌入
        )


@triton.jit  # Triton JIT编译装饰器
def fused_softcap_kernel(  # 融合软裁剪内核
    full_logits_ptr,  # 完整logits指针
    softcapping_value,  # 软裁剪值
    ncols,  # 列数
    row_stride,  # 行步长
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    row = tl.program_id(1).to(tl.int64)  # 获取行索引
    pid = tl.program_id(0).to(tl.int64)  # 获取块索引
    block_start = pid * BLOCK_SIZE  # 计算块起始位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 计算偏移量
    mask = offsets < ncols  # 创建边界掩码

    # Load values
    # 加载值
    row_ptr = full_logits_ptr + row * row_stride  # 计算行指针
    x = tl.load(row_ptr + offsets, mask=mask)  # 加载数据

    # Perform operations in place
    # 原地执行操作
    x = x / softcapping_value  # 除以软裁剪值
    x = libdevice.tanh(x)  # 计算tanh
    x = x * softcapping_value  # 乘以软裁剪值

    # Store result
    # 存储结果
    tl.store(row_ptr + offsets, x, mask=mask)  # 存储计算结果


def fused_softcap(full_logits, final_logit_softcapping):  # 融合软裁剪函数
    if full_logits.is_contiguous():  # 如果张量连续
        nrows, ncols = 1, full_logits.numel()  # 1行，总元素数为列数
        row_stride = ncols  # 行步长等于列数
    else:  # 张量不连续
        assert full_logits.ndim == 2, "non-contiguous softcap requires 2D tensor"  # 断言为2维
        assert (  # 断言列维度连续
            full_logits.stride(1) == 1
        ), "non-contiguous softcap requires contiguous columns"
        nrows, ncols = full_logits.shape  # 获取行数和列数
        row_stride = full_logits.stride(0)  # 获取行步长

    BLOCK_SIZE = 1024  # 块大小
    grid = ((ncols + BLOCK_SIZE - 1) // BLOCK_SIZE, nrows)  # 计算网格大小

    fused_softcap_kernel[grid](  # 启动软裁剪内核
        full_logits_ptr=full_logits,  # logits指针
        softcapping_value=final_logit_softcapping,  # 软裁剪值
        ncols=ncols,  # 列数
        row_stride=row_stride,  # 行步长
        BLOCK_SIZE=BLOCK_SIZE,  # 块大小
    )
    return full_logits  # 返回处理后的logits
