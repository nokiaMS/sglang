# Mamba2元数据模块 - 定义Mamba2前向传播所需的元数据结构，
# 包括前向元数据基类、Mamba2元数据类及其分块索引计算方法
# SPDX-License-Identifier: Apache-2.0 # SPDX许可证标识符
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project # SPDX版权声明
# Copyright 2025 SGLang Team # 版权声明
# Licensed under the Apache License, Version 2.0 (the "License"); # 在Apache许可证2.0版下授权
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件。
# You may obtain a copy of the License at # 可在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意，否则按"原样"分发
# distributed under the License is distributed on an "AS IS" BASIS, # 软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不提供任何明示或暗示的保证或条件
# See the License for the specific language governing permissions and # 参见许可证了解管理权限和
# limitations under the License. # 限制的特定语言
# ==============================================================================
# Adapted from https://github.com/vllm-project/vllm/blob/2c58742dff8613a3bd7496f2008ce927e18d38d1/vllm/model_executor/layers/mamba/mamba2_metadata.py # 改编自vLLM项目的mamba2_metadata.py


import math  # 导入数学模块 # 导入标准库math用于数学计算
from dataclasses import dataclass  # 导入数据类装饰器 # 导入dataclass用于定义数据类
from typing import Optional  # 导入可选类型 # 导入Optional类型注解

import torch  # 导入PyTorch # 导入PyTorch深度学习框架

from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息 # 导入前向传播批次信息类


@dataclass(kw_only=True)  # 数据类装饰器，强制使用关键字参数 # 定义前向元数据基类
class ForwardMetadata:  # 前向元数据基类 - 存储所有Mamba层共享的前向传播元数据
    query_start_loc: torch.Tensor  # 查询起始位置张量 # 变长序列的累积起始索引
    mamba_cache_indices: torch.Tensor  # Mamba缓存索引张量 # 每个请求对应的缓存行索引
    mamba_cache_indices_gdn: Optional[torch.Tensor] = None  # Mamba缓存索引（门控），可选 # 门控模块的缓存索引
    # For topk > 1 eagle # 用于topk > 1的eagle推测解码
    retrieve_next_token: Optional[torch.Tensor] = None  # 检索下一个token指针，可选 # Eagle树中每个token的第一个子token
    retrieve_next_sibling: Optional[torch.Tensor] = None  # 检索下一个兄弟token指针，可选 # Eagle树中每个token的第一个兄弟token
    retrieve_parent_token: Optional[torch.Tensor] = None  # 检索父token指针，可选 # Eagle树中每个token的父token
    # For prefill radix cache # 用于预填充基数缓存
    track_conv_indices: Optional[torch.Tensor] = None  # 跟踪卷积索引，可选 # 需要跟踪的卷积状态索引
    track_ssm_h_src: Optional[torch.Tensor] = None  # 跟踪SSM隐藏状态源索引，可选 # SSM隐藏状态散射的源索引
    track_ssm_h_dst: Optional[torch.Tensor] = None  # 跟踪SSM隐藏状态目标索引，可选 # SSM隐藏状态散射的目标索引
    track_ssm_final_src: Optional[torch.Tensor] = None  # 跟踪SSM最终状态源索引，可选 # SSM最终状态散射的源索引
    track_ssm_final_dst: Optional[torch.Tensor] = None  # 跟踪SSM最终状态目标索引，可选 # SSM最终状态散射的目标索引

    is_target_verify: bool = False  # 是否为目标验证模式 # 推测解码的目标验证标志
    draft_token_num: int = 1  # 草稿token数量 # 每个请求的草稿token数

    has_mamba_track_mask: bool = False  # 是否有Mamba跟踪掩码 # 基数缓存跟踪标志
    mamba_track_mask_indices: Optional[torch.Tensor] = None  # Mamba跟踪掩码索引，可选 # 跟踪掩码对应的索引
    conv_states_mask_indices: Optional[torch.Tensor] = None  # 卷积状态掩码索引，可选 # 卷积状态跟踪的掩码索引


@dataclass(kw_only=True)  # 数据类装饰器，强制使用关键字参数 # 定义Mamba2元数据类
class Mamba2Metadata(ForwardMetadata):  # Mamba2元数据类 - 继承前向元数据，添加Mamba2特有字段
    """stable metadata across all mamba2 layers in the forward pass""" # 前向传播中所有mamba2层共享的稳定元数据

    num_prefills: int  # 预填充请求数量
    num_prefill_tokens: int  # 预填充token总数
    num_decodes: int  # 解码请求数量

    @dataclass(kw_only=True, frozen=True)  # 冻结的数据类装饰器 # 定义混合元数据类
    class MixedMetadata:  # 混合元数据类 - 存储预填充请求的特有元数据
        has_initial_states: torch.Tensor  # 是否有初始状态的张量 # 每个请求是否有先前的SSM状态
        prep_initial_states: bool  # 是否需要准备初始状态 # 是否有任何请求需要准备初始状态

        chunk_size: int  # 分块大小 # Mamba的物理块大小
        seq_idx: torch.Tensor  # 序列索引张量 # 每个token所属的序列编号
        chunk_indices: torch.Tensor  # 块索引张量 # 每个逻辑块对应的物理块索引
        chunk_offsets: torch.Tensor  # 块偏移张量 # 每个逻辑块在物理块内的起始偏移

        extend_seq_lens_cpu: list[int]  # CPU上的扩展序列长度列表 # 每个预填充请求的新增长度

    mixed_metadata: MixedMetadata | None = None  # 混合元数据，可选 # 预填充请求的额外元数据
    """`mixed_metadata` is used for extend/mixed requests""" # `mixed_metadata`用于扩展/混合请求

    @staticmethod  # 静态方法 # 将查询起始位置转换为块索引和偏移
    def _query_start_loc_to_chunk_indices_offsets(
        query_start_loc: torch.Tensor, chunk_size: int, total_seqlens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回块索引和偏移的元组
        """ # 将查询起始位置转换为块索引和偏移的文档字符串
        Args: # 参数：
            query_start_loc (torch.Tensor): 1D tensor of cumulative sequence # query_start_loc (torch.Tensor): 累积序列的一维张量
                lengths, shape (num_seqs + 1,). # 长度，形状(num_seqs + 1,)。
                The first element should be 0. Each entry represents the starting # 第一个元素应为0。每个条目表示
                index of a sequence in the flattened token array. # 扁平化token数组中序列的起始索引。
            chunk_size (int): The size of each physical mamba chunk # chunk_size (int)：每个物理mamba块的大小
                (number of tokens per chunk). # （每块的token数）。
            total_seqlens (int): The total number of tokens in the batch. # total_seqlens (int)：批次中的总token数。

        Returns: # 返回：
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing: # Tuple[torch.Tensor, torch.Tensor]：包含以下内容的元组：
                - chunk_indices (torch.Tensor): 1D tensor of indices # - chunk_indices (torch.Tensor)：索引的一维张量
                    indicating the physical chunk for each logical chunk. # 指示每个逻辑块的物理块。
                - chunk_offsets (torch.Tensor): 1D tensor of offsets # - chunk_offsets (torch.Tensor)：偏移的一维张量
                    indicating the starting index of each logical chunk within # 指示每个逻辑块在
                    its physical chunk. # 其物理块内的起始索引。

        This function computes the chunk indices and offsets for the given # 此函数计算给定
        query_start_loc and chunk_size. Both are tensors of integers with length N, # query_start_loc和chunk_size的块索引和偏移。两者都是长度为N的整数张量，
        where N is the number of logical (pseudo) chunks. # 其中N是逻辑（伪）块的数量。
        A logical chunk is a sequence of tokens that are all part of the same # 逻辑块是一系列token，它们都属于同一
        sequence and are all in the same physical mamba chunk. # 序列且都在同一物理mamba块中。
        In other words, a logical chunk changes every time we cross a sequence # 换句话说，每当跨越序列
        boundary or a physical mamba chunk boundary. # 边界或物理mamba块边界时，逻辑块会改变。
        Logical chunks are needed to handle batched requests with initial states # 逻辑块用于处理具有初始状态的批量请求
        (see _state_passing_fwd and _chunk_scan_fwd). # （参见_state_passing_fwd和_chunk_scan_fwd）。
        The chunk_indices tensor contains the index of the physical chunk for each # chunk_indices张量包含每个
        logical chunk. # 逻辑块的物理块索引。
        The chunk_offsets tensor contains the offset (AKA starting index) of the # chunk_offsets张量包含
        logical chunk in the physical chunk. # 逻辑块在物理块中的偏移（即起始索引）。

        Example: # 示例：
        query_start_loc = [0, 5, 10] # 查询起始位置 = [0, 5, 10]
        chunk_size = 8 # 块大小 = 8
        total_seqlens = 10 # 总序列长度 = 10
        -> chunk_indices = [0, 0, 1] # -> 块索引 = [0, 0, 1]
        -> chunk_offsets = [0, 5, 0] # -> 块偏移 = [0, 5, 0]

        In this example, we have 2 sequences, each with 5 tokens. The physical # 在此示例中，我们有2个序列，每个5个token。物理
        chunk size is 8 tokens. # 块大小为8个token。
        We have three logical chunks: # 我们有三个逻辑块：
        - the first logical chunk starts at token 0 in the first physical chunk # - 第一个逻辑块从第一个物理块的token 0开始
            and contains all 5 tokens from the first sequence # 包含第一个序列的所有5个token
        - the second logical chunk starts at token 5 in the first physical chunk # - 第二个逻辑块从第一个物理块的token 5开始
            and contains first 3 tokens from the second sequence # 包含第二个序列的前3个token
        - the third logical chunk starts at token 0 in the second physical chunk # - 第三个逻辑块从第二个物理块的token 0开始
            and contains the remaining 2 tokens from the second sequence # 包含第二个序列的剩余2个token
        """

        cu_seqlens = query_start_loc[1:]  # remove prepended 0 # 移除前导的0 # 获取每个序列的结束位置

        # outputs will have length expansion of chunks that do not divide # 输出将扩展不整除
        # chunk_size # chunk_size的块的长度
        N = (
            math.ceil(total_seqlens / chunk_size)
            + (cu_seqlens[:-1] % chunk_size > 0).sum()
        )  # 计算逻辑块数量N # 总物理块数+序列边界处跨块的额外块数
        chunk_indices = torch.arange(N, dtype=torch.int, device=query_start_loc.device)  # 初始化块索引 # 创建递增索引数组
        chunk_offsets = torch.zeros(
            (N,), dtype=torch.int, device=query_start_loc.device
        )  # 初始化块偏移为全零 # 创建全零偏移数组

        p = 0  # num of insertions # 插入计数器 # 跨块边界导致的额外块数
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):  # 遍历相邻序列边界 # 遍历每对序列起止位置

            # if does not divide chunk_size, then there is one chunk insertion # 如果不整除chunk_size，则需要插入一个块
            p += s % chunk_size > 0  # 累加插入计数 # 如果序列起始不与块边界对齐则增加计数

            # get the dimensions # 获取维度
            # - the + 1 for _e is to shift the boundary by one chunk # - _e的+1是为了将边界移动一个块
            # - this shifting is not needed if chunk_size divides e # - 如果chunk_size整除e则不需要此移动
            _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size > 0)  # 计算逻辑块的起止索引 # 考虑插入偏移后计算

            # adjust indices and offsets # 调整索引和偏移
            chunk_indices[_s:_e] -= p  # 调整块索引以补偿插入 # 减去插入偏移得到正确的物理块索引
            chunk_offsets[_s] = s % chunk_size  # 设置块的起始偏移 # 第一个逻辑块在物理块内的偏移

        return chunk_indices, chunk_offsets  # 返回块索引和偏移 # 返回计算结果

    @staticmethod  # 静态方法 # 准备解码阶段的Mamba2元数据
    def prepare_decode(
        forward_metadata: ForwardMetadata,  # 前向元数据
        seq_lens: torch.Tensor,  # 序列长度张量
        *,
        is_target_verify: bool,  # 是否为目标验证模式
        draft_token_num: int,  # 草稿token数量
    ) -> "Mamba2Metadata":  # 返回Mamba2元数据
        """This path is run during CUDA graph capture, i.e. decode only, so `num_prefills` is 0""" # 此路径在CUDA图捕获期间运行，即仅解码，因此`num_prefills`为0
        return Mamba2Metadata(  # 创建并返回Mamba2元数据实例 # 构建解码模式的元数据
            query_start_loc=forward_metadata.query_start_loc,  # 查询起始位置
            mamba_cache_indices=forward_metadata.mamba_cache_indices,  # Mamba缓存索引
            retrieve_next_token=forward_metadata.retrieve_next_token,  # 检索下一个token
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,  # 检索下一个兄弟token
            retrieve_parent_token=forward_metadata.retrieve_parent_token,  # 检索父token
            track_conv_indices=forward_metadata.track_conv_indices,  # 跟踪卷积索引
            track_ssm_h_src=forward_metadata.track_ssm_h_src,  # 跟踪SSM隐藏状态源索引
            track_ssm_h_dst=forward_metadata.track_ssm_h_dst,  # 跟踪SSM隐藏状态目标索引
            track_ssm_final_src=forward_metadata.track_ssm_final_src,  # 跟踪SSM最终状态源索引
            track_ssm_final_dst=forward_metadata.track_ssm_final_dst,  # 跟踪SSM最终状态目标索引
            has_mamba_track_mask=forward_metadata.has_mamba_track_mask,  # 是否有Mamba跟踪掩码
            num_decodes=len(seq_lens),  # 解码请求数量 # 序列数即为解码请求数
            num_prefills=0,  # 解码模式下预填充数为0 # CUDA图捕获仅用于解码
            num_prefill_tokens=0,  # 解码模式下预填充token数为0
            is_target_verify=is_target_verify,  # 是否为目标验证模式
            draft_token_num=draft_token_num,  # 草稿token数量
        )

    @classmethod  # 类方法 # 准备混合（预填充+解码）模式的Mamba2元数据
    def prepare_mixed(
        cls,
        forward_metadata: ForwardMetadata,  # 前向元数据
        chunk_size: int,  # 分块大小
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> "Mamba2Metadata":  # 返回Mamba2元数据
        """This path cannot run with CUDA graph, as it contains extend requests.""" # 此路径不能与CUDA图一起运行，因为它包含扩展请求。
        if forward_batch.extend_num_tokens is None:  # 如果没有扩展token # 纯解码模式
            draft_token_num = (
                forward_batch.spec_info.draft_token_num
                if forward_batch.spec_info is not None
                else 1
            )  # 获取草稿token数量 # 从推测信息中读取，无则为1
            return cls.prepare_decode(
                forward_metadata,
                forward_batch.seq_lens,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                draft_token_num=draft_token_num,
            )  # 退化为纯解码模式 # 调用prepare_decode处理
        num_prefills = len(forward_batch.extend_seq_lens)  # 预填充请求数量 # 扩展序列长度列表的长度
        num_prefill_tokens = forward_batch.extend_num_tokens  # 预填充token总数 # 扩展token总数
        num_decodes = len(forward_batch.seq_lens) - num_prefills  # 解码请求数量 # 总序列数减去预填充数
        context_lens_tensor = forward_batch.extend_prefix_lens  # 上下文长度张量 # 每个请求的前缀长度
        assert context_lens_tensor is not None  # 断言上下文长度不为None
        # precompute flag to avoid device syncs later # 预计算标志以避免后续设备同步
        has_initial_states = context_lens_tensor > 0  # 计算是否有初始状态 # 前缀长度>0表示有历史状态
        prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()  # 是否需要准备初始状态 # 检查是否有任何预填充请求需要初始状态

        query_start_loc = forward_metadata.query_start_loc[: num_prefills + 1]  # 截取预填充部分的查询起始位置
        seq_idx = torch.repeat_interleave(
            torch.arange(
                num_prefills, dtype=torch.int32, device=query_start_loc.device
            ),
            query_start_loc.diff(),
            output_size=num_prefill_tokens,
        )  # 创建序列索引 # 每个token对应的序列编号
        seq_idx.unsqueeze_(0)  # 在第0维增加维度 # 添加batch维度

        # We compute metadata for chunked prefill once at the top level model # 我们在顶层模型一次性计算分块预填充的元数据
        # forward and reuse them in mamba layers. If not needed, they will be # 前向并在mamba层中复用。如果不需要，它们将
        # ignored inside mamba kernels. # 在mamba核函数中被忽略。
        chunk_offsets, chunk_indices = None, None  # 初始化块偏移和索引为None
        if prep_initial_states:  # 如果需要准备初始状态 # 仅在有初始状态时计算块元数据
            chunk_indices, chunk_offsets = (
                cls._query_start_loc_to_chunk_indices_offsets(
                    query_start_loc, chunk_size, num_prefill_tokens
                )
            )  # 计算块索引和偏移 # 调用静态方法计算

        draft_token_num = (
            getattr(forward_batch.spec_info, "draft_token_num", 1)
            if forward_batch.spec_info is not None
            else 1
        )  # 获取草稿token数量 # 从推测信息读取，默认为1
        return Mamba2Metadata(  # 创建并返回Mamba2元数据实例 # 构建混合模式的元数据
            query_start_loc=query_start_loc,  # 查询起始位置
            mamba_cache_indices=forward_metadata.mamba_cache_indices,  # Mamba缓存索引
            retrieve_next_token=forward_metadata.retrieve_next_token,  # 检索下一个token
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,  # 检索下一个兄弟token
            retrieve_parent_token=forward_metadata.retrieve_parent_token,  # 检索父token
            track_conv_indices=forward_metadata.track_conv_indices,  # 跟踪卷积索引
            track_ssm_h_src=forward_metadata.track_ssm_h_src,  # 跟踪SSM隐藏状态源索引
            track_ssm_h_dst=forward_metadata.track_ssm_h_dst,  # 跟踪SSM隐藏状态目标索引
            track_ssm_final_src=forward_metadata.track_ssm_final_src,  # 跟踪SSM最终状态源索引
            track_ssm_final_dst=forward_metadata.track_ssm_final_dst,  # 跟踪SSM最终状态目标索引
            has_mamba_track_mask=forward_metadata.has_mamba_track_mask,  # 是否有Mamba跟踪掩码
            num_prefills=num_prefills,  # 预填充请求数量
            num_prefill_tokens=num_prefill_tokens,  # 预填充token总数
            num_decodes=num_decodes,  # 解码请求数量
            is_target_verify=forward_batch.forward_mode.is_target_verify(),  # 是否为目标验证模式
            draft_token_num=draft_token_num,  # 草稿token数量
            mixed_metadata=cls.MixedMetadata(  # 创建混合元数据 # 预填充特有的元数据
                has_initial_states=has_initial_states,  # 是否有初始状态
                prep_initial_states=prep_initial_states,  # 是否需要准备初始状态
                chunk_size=chunk_size,  # 分块大小
                seq_idx=seq_idx,  # 序列索引
                chunk_indices=chunk_indices,  # 块索引
                chunk_offsets=chunk_offsets,  # 块偏移
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,  # CPU上的扩展序列长度
            ),
        )
