# XPU FlashAttention后端模块
# 实现Intel XPU平台上的FlashAttention后端，支持MHA和MLA注意力架构，
# 包含预填充扩展、解码、推测解码和滑动窗口注意力等功能

from __future__ import annotations  # 启用延迟注解评估

from typing import TYPE_CHECKING, Optional  # 类型提示

import torch  # PyTorch核心库

from sglang.srt.configs.model_config import AttentionArch  # 注意力架构枚举
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 注意力后端基类
from sglang.srt.layers.attention.flashattention_backend import (  # FlashAttention后端工具
    FlashAttentionMetadata,  # FlashAttention元数据
    make_local_attention_virtual_batches,  # 创建局部注意力虚拟批次
    merge_state_v2_wrapper,  # 状态合并v2封装
    prepare_swa_spec_page_table_triton,  # 准备SWA推测页表Triton内核
)
from sglang.srt.managers.schedule_batch import get_global_server_args  # 获取全局服务器参数
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 前向批次信息

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.layers.radix_attention import RadixAttention  # 基数注意力
    from sglang.srt.model_executor.model_runner import ModelRunner  # 模型运行器

from sgl_kernel import flash_mla_decode, flash_mla_get_workspace_size, merge_state_v2  # sgl_kernel MLA解码和合并
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # Flash Attention函数


class XPUAttentionBackend(AttentionBackend):  # XPU FlashAttention后端，当前基于FlashAttentionBackend，将来会重构
    """XPU FlashAttention backend, currently based on FlashAttentionBackend, will be refactored later.
    XPU FlashAttention后端，当前基于FlashAttentionBackend，将来会重构。

    TODO:
    - Prefill and Decode disaggregation, currently only chunked prefill is supported
    - 预填充和解码分离，当前仅支持分块预填充
    - Speculative Decoding support
    - 推测解码支持
    - XPU Graph support, see https://github.com/pytorch/pytorch/issues/162143
    - XPU图支持，参见https://github.com/pytorch/pytorch/issues/162143
    - MLA Prefill support
    - MLA预填充支持
    """

    def __init__(  # 初始化方法
        self,
        model_runner: ModelRunner,  # 模型运行器
        skip_prefill: bool = False,  # 是否跳过预填充
        speculative_step_id=0,  # 推测步ID
        topk=0,  # topk值
        speculative_num_steps=0,  # 推测步数
    ):
        super().__init__()  # 调用父类初始化

        assert not (  # 断言不同时支持滑动窗口和交叉注意力
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"  # 不支持滑动窗口和交叉注意力同时使用

        self.forward_metadata: FlashAttentionMetadata = None  # 前向元数据
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        # 处理推测解码topk>1、扩展草稿解码和验证的额外元数据
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None  # 推测解码扩展元数据
        self.max_context_len = model_runner.model_config.context_len  # 最大上下文长度
        self.num_attention_heads = (  # 注意力头数
            model_runner.model_config.hf_text_config.num_attention_heads  # 从配置获取
        )
        self.tp_size = model_runner.tp_size  # 张量并行大小
        assert self.num_attention_heads % self.tp_size == 0  # 头数必须可被TP大小整除
        self.num_local_heads = self.num_attention_heads // self.tp_size  # 本地头数
        self.device = model_runner.device  # 设备
        self.decode_cuda_graph_metadata = {}  # 解码CUDA图元数据
        self.target_verify_metadata = {}  # 目标验证元数据
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        # 池引用 - 在构造时捕获，以便在ForwardBatch字段删除后仍能存活。
        self.req_to_token_pool = model_runner.req_to_token_pool  # 请求到token映射池
        self.token_to_kv_pool = model_runner.token_to_kv_pool  # token到KV映射池
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到token映射表
        self.kv_cache_dtype = model_runner.kv_cache_dtype  # KV缓存数据类型
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype  # KV缓存数据类型字符串
        self.page_size = model_runner.page_size  # 页大小
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA  # 是否使用MLA
        self.skip_prefill = skip_prefill  # 是否跳过预填充
        self.is_hybrid_swa = model_runner.is_hybrid_swa  # 是否混合滑动窗口注意力
        if self.is_hybrid_swa:  # 如果使用混合SWA
            self.full_to_swa_index_mapping = (  # 全量到SWA索引映射
                model_runner.token_to_kv_pool.full_to_swa_index_mapping  # 从KV池获取
            )
        self.topk = model_runner.server_args.speculative_eagle_topk or 0  # 推测Eagle topk
        self.speculative_num_steps = speculative_num_steps  # 推测步数
        self.speculative_num_draft_tokens = (  # 推测草稿token数
            model_runner.server_args.speculative_num_draft_tokens  # 从服务器参数获取
        )
        self.speculative_step_id = speculative_step_id  # 推测步ID

        # Local attention settings  # 局部注意力设置
        self.attention_chunk_size = (  # 注意力分块大小
            model_runner.attention_chunk_size  # 从运行器获取
            if hasattr(model_runner, "attention_chunk_size")  # 如果存在
            else None  # 否则为None
        )

        # For each layer, the sliding_window_size can be different. This is only used for preparing SWA metadata.
        # We use `layer.sliding_window_size` to decide whether to use SWA for each layer.
        # 每层的sliding_window_size可以不同。这仅用于准备SWA元数据。
        # 我们使用`layer.sliding_window_size`来决定每层是否使用SWA。
        self.sliding_window_size = model_runner.sliding_window_size  # 滑动窗口大小
        self.has_swa = (  # 是否有滑动窗口注意力
            self.sliding_window_size is not None and self.sliding_window_size > -1  # 窗口大小非负
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):  # 初始化前向元数据，使所有层可复用
        """Initialize forward metadata hence all layers in the forward pass can reuse it.
        初始化前向元数据，使前向传播中的所有层可复用。"""
        metadata = FlashAttentionMetadata()  # 创建FlashAttention元数据
        seqlens_in_batch = forward_batch.seq_lens  # 批次中的序列长度
        batch_size = forward_batch.batch_size  # 批量大小
        device = seqlens_in_batch.device  # 设备

        if forward_batch.forward_mode.is_decode_or_idle():  # 解码或空闲模式
            # Draft Decode  # 草稿解码
            if forward_batch.spec_info is not None:  # 如果有推测信息
                assert (  # 断言失败
                    False
                ), "XPUAttentionBackend doesn't support speculative decoding yet, please use --attention-backend triton instead."  # XPU后端尚不支持推测解码，请使用triton后端
                if self.topk <= 1:  # topk<=1
                    metadata.cache_seqlens_int32 = (  # 缓存序列长度int32
                        seqlens_in_batch + (self.speculative_step_id + 1)  # 加上推测步
                    ).to(torch.int32)  # 转int32
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (  # 最大KV序列长度
                        self.speculative_step_id + 1  # 加上推测步
                    )
                    metadata.cu_seqlens_q = torch.arange(  # 查询累积序列长度
                        0, batch_size + 1, dtype=torch.int32, device=device  # 0到batch_size
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度
                        torch.cumsum(  # 累积和
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32  # 沿第0维
                        ),
                        (1, 0),  # 左填充1
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引、最大长度
                    ]
                else:  # topk>1
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)  # 缓存序列长度
                    metadata.max_seq_len_q = self.topk  # 查询最大序列长度为topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()  # KV最大序列长度
                    metadata.cu_seqlens_q = torch.arange(  # 查询累积序列长度
                        0,  # 起始
                        batch_size * self.topk + 1,  # 结束
                        step=self.topk,  # 步长
                        dtype=torch.int32,  # int32
                        device=device,  # 设备
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度
                        torch.cumsum(  # 累积和
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32  # 沿第0维
                        ),
                        (1, 0),  # 左填充1
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引、最大长度
                    ]

                    metadata_expand = FlashAttentionMetadata()  # 扩展元数据
                    decode_length = self.speculative_step_id + 1  # 解码长度
                    metadata_expand.cache_seqlens_int32 = torch.full(  # 扩展缓存序列长度
                        (seqlens_in_batch.numel() * self.topk,),  # 形状
                        decode_length,  # 填充值
                        device=device,  # 设备
                        dtype=torch.int32,  # int32
                    )
                    metadata_expand.max_seq_len_q = 1  # 每次解码1个token
                    metadata_expand.cu_seqlens_q = torch.arange(  # 扩展查询累积序列长度
                        0,  # 起始
                        metadata_expand.cache_seqlens_int32.numel() + 1,  # 结束
                        dtype=torch.int32,  # int32
                        device=device,  # 设备
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(  # 扩展KV累积序列长度
                        0,  # 起始
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,  # 结束
                        step=decode_length,  # 步长
                        dtype=torch.int32,  # int32
                        device=device,  # 设备
                    )
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    # 形状：[bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = forward_batch.out_cache_loc.view(  # 缓存位置
                        -1, self.speculative_num_steps  # 重塑形状
                    )
                    metadata_expand.page_table = (  # 扩展页表
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)  # 取前decode_length步
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand  # 保存扩展元数据
            else:  # 无推测信息
                # Normal Decode  # 普通解码
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)  # 缓存序列长度
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()  # 最大KV序列长度
                metadata.cu_seqlens_q = torch.arange(  # 查询累积序列长度
                    0, batch_size + 1, dtype=torch.int32, device=device  # 0到batch_size
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)  # 累积和并左填充
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引、最大长度
                ]
            # TODO: we need to test this part for llama 4 eagle case
            # TODO: 需要针对llama 4 eagle情况测试此部分
            self._init_local_attn_metadata(forward_batch, metadata, device)  # 初始化局部注意力元数据
        elif forward_batch.forward_mode.is_target_verify():  # 目标验证模式
            if self.topk <= 1:  # topk<=1
                metadata.cache_seqlens_int32 = (  # 缓存序列长度
                    forward_batch.seq_lens + self.speculative_num_draft_tokens  # 加上草稿token数
                ).to(torch.int32)  # 转int32
                metadata.max_seq_len_q = self.speculative_num_draft_tokens  # 查询最大长度
                metadata.max_seq_len_k = (  # KV最大长度
                    forward_batch.seq_lens_cpu.max().item()  # CPU最大值
                    + self.speculative_num_draft_tokens  # 加上草稿数
                )
                metadata.cu_seqlens_q = torch.arange(  # 查询累积序列长度
                    0,  # 起始
                    batch_size * self.speculative_num_draft_tokens + 1,  # 结束
                    self.speculative_num_draft_tokens,  # 步长
                    dtype=torch.int32,  # int32
                    device=device,  # 设备
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度
                    torch.cumsum(  # 累积和
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32  # 沿第0维
                    ),
                    (1, 0),  # 左填充1
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引、最大长度
                ]

                self._init_local_attn_metadata(forward_batch, metadata, device)  # 初始化局部注意力元数据
            else:  # topk>1
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)  # 缓存序列长度
                metadata.max_seq_len_q = self.speculative_num_draft_tokens  # 查询最大长度
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()  # KV最大长度
                metadata.cu_seqlens_q = torch.arange(  # 查询累积序列长度
                    0,  # 起始
                    batch_size * self.speculative_num_draft_tokens + 1,  # 结束
                    step=self.speculative_num_draft_tokens,  # 步长
                    dtype=torch.int32,  # int32
                    device=device,  # 设备
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度
                    torch.cumsum(  # 累积和
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32  # 沿第0维
                    ),
                    (1, 0),  # 左填充1
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引、最大长度
                ]

                metadata_expand = FlashAttentionMetadata()  # 扩展元数据

                metadata_expand.max_seq_len_q = 1  # 查询最大长度为1
                metadata_expand.cu_seqlens_q = torch.arange(  # 扩展查询累积序列长度
                    0,  # 起始
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens  # 结束
                    + 1,  # 加1
                    dtype=torch.int32,  # int32
                    device=device,  # 设备
                )

                # create expand page table  # 创建扩展页表
                offsets = torch.arange(  # 偏移量
                    self.speculative_num_draft_tokens, device=device  # 草稿token数
                ).unsqueeze(  # 增加维度
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)  # 形状：(1, speculative_num_draft_tokens)
                cols = offsets.expand(  # 列索引
                    forward_batch.seq_lens.numel(), -1  # 扩展到批量大小
                ) + forward_batch.seq_lens.unsqueeze(1)  # 加上序列长度
                cum_len = torch.nn.functional.pad(  # 累积长度
                    torch.cumsum(  # 累积和
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens  # 序列长度+草稿数
                        ).repeat_interleave(self.speculative_num_draft_tokens),  # 重复草稿数次
                        dim=0,  # 沿第0维
                    ),
                    (1, 0),  # 左填充1
                )[:-1]  # 去掉最后一个
                mask_extraction_indices = (  # 掩码提取索引
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)  # 重复列索引
                    + cum_len[:, None]  # 加上累积长度
                ).view(1, -1)  # 重塑为1D
                mask = forward_batch.spec_info.custom_mask[  # 从自定义掩码中提取
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens  # 重塑为2D
                )  # (bsz * draft_num, draft_num)  # 形状

                # shift table indices to avoid padding
                # 移位表索引以避免填充
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                # 注意cache_seqlens_int32为[1, 2, 2]，因此每行多余的页索引将被忽略
                col_indices = offsets.expand(  # 列索引
                    mask.shape[0], self.speculative_num_draft_tokens  # 扩展到掩码形状
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                # 构建键：如果条目有效(mask==True)，保留其原始索引；
                # 如果无效，加上speculative_num_draft_tokens使其排序在所有有效条目之后。
                keys = torch.where(  # 构建排序键
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens  # 有效用原始索引，无效用大值
                )
                _, sort_order = torch.sort(keys, dim=1)  # 排序获取顺序
                non_masked_page_table = (  # 未掩码页表
                    self.req_to_token_pool.req_to_token[  # 从映射表获取
                        forward_batch.req_pool_indices, :  # 请求索引
                    ]
                    .gather(1, cols)  # 按列索引收集
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)  # 重复草稿数次
                )  # (bsz, draft_num)  # 形状
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)  # 按排序顺序收集
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)  # 掩码和即为有效长度
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(  # 扩展KV累积序列长度
                    torch.cumsum(  # 累积和
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32  # 沿第0维
                    ),
                    (1, 0),  # 左填充1
                )
                self.forward_metadata_spec_decode_expand = metadata_expand  # 保存扩展元数据

                if self.has_swa:  # 如果有滑动窗口注意力
                    self._init_sliding_window_attn_spec_metadata(  # 初始化SWA推测元数据
                        metadata, metadata_expand  # 传入元数据和扩展元数据
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():  # 扩展或草稿扩展模式
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)  # 缓存序列长度
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()  # 最大KV序列长度
            metadata.cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)  # 累积和并左填充
            )
            metadata.page_table = self.req_to_token_pool.req_to_token[  # 页表
                forward_batch.req_pool_indices, : metadata.max_seq_len_k  # 请求索引、最大长度
            ]

            if (  # 如果有前缀长度或草稿扩展
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                extend_seq_lens = forward_batch.extend_seq_lens  # 扩展序列长度
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)  # 查询最大长度
                metadata.cu_seqlens_q = torch.nn.functional.pad(  # 查询累积序列长度
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)  # 累积和并左填充
                )
            else:  # 无前缀
                metadata.max_seq_len_q = metadata.max_seq_len_k  # 查询长度等于KV长度
                metadata.cu_seqlens_q = metadata.cu_seqlens_k  # 累积长度相同

            # Setup local attention if enabled  # 如果启用则设置局部注意力
            if forward_batch.forward_mode == ForwardMode.EXTEND:  # 扩展模式
                self._init_local_attn_metadata(forward_batch, metadata, device)  # 初始化局部注意力元数据

        # Encoder metadata for cross attention  # 交叉注意力的编码器元数据
        if forward_batch.encoder_lens is not None:  # 如果有编码器长度
            assert (  # 断言编码器数量为1
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"  # 当前仅支持编码器大小为1

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)  # 编码器长度int32
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(  # 编码器KV累积序列长度
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),  # 累积和
                (1, 0),  # 左填充1
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()  # 编码器最大长度
            metadata.encoder_page_table = self.req_to_token_pool.req_to_token[  # 编码器页表
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k  # 请求索引、最大长度
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            # 当前仅支持forward_batch.encoder_lens.numel() == 1
            metadata.page_table = self.req_to_token_pool.req_to_token[  # 主页表（编码器之后）
                forward_batch.req_pool_indices,  # 请求索引
                metadata.encoder_max_seq_len_k : (  # 从编码器长度开始
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k  # 到编码器+KV长度
                ),
            ]

        if self.use_mla:  # 如果使用MLA
            workspace_size = flash_mla_get_workspace_size(  # 获取MLA工作空间大小
                max_seq_len=self.max_context_len,  # 最大序列长度
                num_batches=batch_size,  # 批量大小
                num_heads=self.num_local_heads,  # 本地头数
                page_size=self.page_size,  # 页大小
                num_kv_splits=-1,  # 自动决定KV分片数
            )
            if (  # 如果工作空间不存在或大小不足
                not hasattr(self, "workspace")
                or self.workspace.numel() < workspace_size  # 当前空间不够
            ):
                self.workspace = torch.empty(  # 创建工作空间
                    workspace_size, device=self.device, dtype=torch.uint8  # uint8类型
                )

        # Convert the page table to a strided format which is needed by FA3 API
        # 将页表转换为FA3 API所需的步进格式
        if self.page_size > 1:  # 如果页大小>1
            self.strided_indices = torch.arange(  # 步进索引
                0, metadata.page_table.shape[1], self.page_size, device=self.device  # 按页大小步进
            )
            metadata.page_table = (  # 转换为页粒度
                metadata.page_table[:, self.strided_indices] // self.page_size  # 整除页大小
            )

        self.forward_metadata = metadata  # 保存前向元数据

    def forward_extend(  # 扩展注意力前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
        # For multi-head latent attention  # 多头潜在注意力用
        q_rope: Optional[torch.Tensor] = None,  # 旋转位置编码查询
        k_rope: Optional[torch.Tensor] = None,  # 旋转位置编码键
        sinks: Optional[torch.Tensor] = None,  # sink参数
    ):
        if k is not None:  # 如果K不为None
            assert v is not None  # V也不应为None
            if save_kv_cache:  # 如果保存KV缓存
                cache_loc = (  # 缓存位置
                    forward_batch.out_cache_loc  # 输出缓存位置
                    if not layer.is_cross_attention  # 非交叉注意力
                    else forward_batch.encoder_out_cache_loc  # 交叉注意力用编码器缓存位置
                )
                if not self.use_mla:  # 非MLA
                    self.token_to_kv_pool.set_kv_buffer(  # 设置KV缓存
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale  # 层、位置、K、V、缩放
                    )
                else:  # MLA
                    self.token_to_kv_pool.set_mla_kv_buffer(  # 设置MLA KV缓存
                        layer,  # 层
                        cache_loc,  # 缓存位置
                        k,  # 压缩KV
                        k_rope,  # 旋转位置编码键
                    )

        # Use precomputed metadata across all layers  # 在所有层间使用预计算的元数据
        metadata = self.forward_metadata  # 获取元数据

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # 计算窗口大小（如果层属性不变，可移到元数据中）
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # 我们不做layer.sliding_window_size - 1，因为在model.get_attention_sliding_window_size()中已经-1了
        # here is two side inclusive  # 这里是两侧包含
        is_hybrid_swa = (  # 是否混合SWA
            layer.sliding_window_size is not None and layer.sliding_window_size > -1  # 窗口大小非负
        )
        window_size = (layer.sliding_window_size, 0) if is_hybrid_swa else (-1, -1)  # 窗口大小

        # currently no FP8 KV cache supported  # 当前不支持FP8 KV缓存
        k_descale, v_descale = None, None  # KV反缩放为None
        # # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # # has corresponding quantization method so that layer.k_scale is not None,
        # # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        # # 仅在以下条件同时满足时使用KV缩放：1) 显式启用fp8 kv，2) RadixAttention
        # # 有相应的量化方法使layer.k_scale不为None，
        # # 3) layer.head_dim <= 256，因为FA3内核在此情况下要求fp16和bf16数据类型。
        # if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
        #     if layer.k_scale is not None:
        #         descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        #         k_descale = layer.k_scale.expand(descale_shape)
        #         v_descale = layer.v_scale.expand(descale_shape)
        #     q = q.to(self.kv_cache_dtype)
        #     q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
        #     k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = not layer.is_cross_attention  # 非交叉注意力为因果

        # Check if we should use local attention  # 检查是否应使用局部注意力
        use_local_attn = (  # 使用局部注意力条件
            self.attention_chunk_size is not None  # 有分块大小
            and metadata.local_attn_metadata is not None  # 有局部注意力元数据
            and (hasattr(layer, "use_irope") and layer.use_irope)  # 层启用irope
        )

        # We do cascade attention for Target Verify with topk > 1
        # We don't use cascade attention for Sliding Window Attention:
        # - Different window sizes should be passed in for each q in the first stage of cascade attention, but FA3 interface doesn't support pass in a list of window sizes.
        # - The overhead of duplicated computation of the common prefix part is small for sliding window layers (seq_len <= window_size), so we can just expand it.
        # 我们对topk>1的目标验证使用级联注意力
        # 我们不对滑动窗口注意力使用级联注意力：
        # - 级联注意力的第一阶段应该为每个q传入不同的窗口大小，但FA3接口不支持传入窗口大小列表。
        # - 滑动窗口层的公共前缀部分重复计算开销很小（seq_len <= window_size），所以可以直接展开。
        use_cascade_attn = (  # 使用级联注意力条件
            forward_batch.forward_mode.is_target_verify()  # 目标验证模式
            and self.topk > 1  # topk>1
            and not is_hybrid_swa  # 非混合SWA
        )

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        # 为了fa3接口版本兼容性，我们将新字段放入条件关键字参数
        kwargs = {}  # 额外参数
        if sinks is not None:  # 如果有sink
            kwargs["sinks"] = sinks  # 添加sink

        # Get the appropriate page table based on whether we're using local attention
        # 根据是否使用局部注意力获取相应的页表
        if use_local_attn:  # 使用局部注意力
            local_metadata = metadata.local_attn_metadata  # 局部元数据
            page_table = local_metadata.local_block_table  # 局部块表
            cu_seqlens_q = local_metadata.local_query_start_loc  # 局部查询起始位置
            cache_seqlens = local_metadata.local_seqused_k  # 局部KV序列长度
            max_seqlen_q = local_metadata.local_max_query_len  # 局部最大查询长度
        elif is_hybrid_swa and metadata.swa_spec_metadata is not None:  # 混合SWA推测
            swa_spec_metadata = metadata.swa_spec_metadata  # SWA推测元数据
            page_table = swa_spec_metadata.page_table  # SWA页表
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q  # SWA查询累积长度
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32  # SWA缓存序列长度
            max_seqlen_q = swa_spec_metadata.max_seq_len_q  # SWA最大查询长度
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k  # SWA KV累积长度
        else:  # 标准模式
            page_table = metadata.page_table  # 页表
            cu_seqlens_q = metadata.cu_seqlens_q  # 查询累积长度
            cache_seqlens = metadata.cache_seqlens_int32  # 缓存序列长度
            max_seqlen_q = metadata.max_seq_len_q  # 最大查询长度
            cu_seqlens_k = metadata.cu_seqlens_k  # KV累积长度

        # Use Flash Attention for prefill  # 使用Flash Attention进行预填充
        if not self.use_mla:  # 非MLA
            # Do multi-head attention  # 多头注意力
            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)  # 获取KV缓存
            key_cache = key_cache.view(  # 重塑键缓存
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim  # [页数, 页大小, KV头数, 头维度]
            )
            value_cache = value_cache.view(  # 重塑值缓存
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim  # [页数, 页大小, V头数, 头维度]
            )
            if layer.is_cross_attention:  # 交叉注意力
                page_table = metadata.encoder_page_table  # 使用编码器页表
                cache_seqlens = metadata.encoder_lens_int32  # 使用编码器长度
                cu_seqlens_k = metadata.encoder_cu_seqlens_k  # 使用编码器累积长度
                window_size = (-1, -1)  # 无窗口限制

            result = flash_attn_with_kvcache(  # 调用带KV缓存的Flash Attention
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),  # 重塑Q
                k_cache=key_cache,  # 键缓存
                v_cache=value_cache,  # 值缓存
                page_table=page_table,  # 页表
                cache_seqlens=cache_seqlens,  # 缓存序列长度
                cu_seqlens_q=cu_seqlens_q,  # 查询累积长度
                cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,  # KV新累积长度
                max_seqlen_q=max_seqlen_q,  # 最大查询长度
                softmax_scale=layer.scaling,  # softmax缩放
                causal=False if use_cascade_attn else causal,  # 因果性
                window_size=window_size,  # 窗口大小
                softcap=layer.logit_cap,  # softmax上限
                k_descale=k_descale,  # K反缩放
                v_descale=v_descale,  # V反缩放
                return_softmax_lse=use_cascade_attn,  # 级联注意力需要LSE
                **kwargs,  # 额外参数
            )

            if use_cascade_attn:  # 如果使用级联注意力
                o, softmax_lse, *rest = result  # 解包结果
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(  # 第二阶段：扩展注意力
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),  # 重塑Q
                    k_cache=key_cache,  # 键缓存
                    v_cache=value_cache,  # 值缓存
                    page_table=self.forward_metadata_spec_decode_expand.page_table,  # 扩展页表
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,  # 扩展缓存长度
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,  # 扩展查询长度
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,  # 扩展KV长度
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,  # 扩展最大查询长度
                    softmax_scale=layer.scaling,  # softmax缩放
                    causal=False,  # 非因果
                    window_size=window_size,  # 窗口大小
                    softcap=layer.logit_cap,  # softmax上限
                    k_descale=k_descale,  # K反缩放
                    v_descale=v_descale,  # V反缩放
                    return_softmax_lse=True,  # 需要LSE
                    **kwargs,  # 额外参数
                )
                o, _ = merge_state_v2_wrapper(  # 合并两阶段结果
                    o,  # 第一阶段输出
                    softmax_lse.T.contiguous(),  # 第一阶段LSE
                    o_expand,  # 扩展输出
                    softmax_lse_expand.T.contiguous(),  # 扩展LSE
                )
            else:  # 非级联
                o = result  # 直接使用结果
        else:  # MLA
            if (  # 如果有前缀缓存且非验证非草稿扩展
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                # Do multi-head attention with chunked prefix cache
                # 使用分块前缀缓存做多头注意力
                if forward_batch.attn_attend_prefix_cache:  # 如果需要关注前缀缓存
                    assert not get_global_server_args().disable_chunked_prefix_cache  # 不应禁用分块前缀缓存
                    # MHA for chunked prefix kv cache when running model with MLA
                    # 使用MLA运行模型时分块前缀KV缓存的多头注意力
                    assert forward_batch.prefix_chunk_idx is not None  # 需要前缀块索引
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None  # 需要前缀块累积长度
                    assert forward_batch.prefix_chunk_max_seq_lens is not None  # 需要前缀块最大长度

                    chunk_idx = forward_batch.prefix_chunk_idx  # 当前块索引
                    assert chunk_idx >= 0  # 索引非负

                    assert forward_batch.mha_return_lse  # 需要返回LSE
                    output = flash_attn_varlen_func(  # 变长Flash Attention
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),  # Q
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),  # K
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),  # V
                        cu_seqlens_q=metadata.cu_seqlens_q,  # 查询累积长度
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],  # 前缀块KV累积长度
                        max_seqlen_q=metadata.max_seq_len_q,  # 最大查询长度
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],  # 前缀块最大KV长度
                        softmax_scale=layer.scaling,  # softmax缩放
                        causal=False,  # 非因果（关注全部前缀）
                        return_softmax_lse=True,  # 返回LSE
                    )
                else:  # 不关注前缀缓存
                    # MHA for extend part of sequence without attending prefix kv cache
                    # 不关注前缀KV缓存的序列扩展部分的MHA
                    output = flash_attn_varlen_func(  # 变长Flash Attention
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),  # Q
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),  # K
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),  # V
                        cu_seqlens_q=metadata.cu_seqlens_q,  # 查询累积长度
                        cu_seqlens_k=metadata.cu_seqlens_q,  # KV累积长度与查询相同
                        max_seqlen_q=metadata.max_seq_len_q,  # 最大查询长度
                        max_seqlen_k=metadata.max_seq_len_q,  # 最大KV长度与查询相同
                        softmax_scale=layer.scaling,  # softmax缩放
                        causal=True,  # 因果注意力
                        return_softmax_lse=forward_batch.mha_return_lse,  # 是否返回LSE
                    )
                if forward_batch.mha_return_lse:  # 如果需要返回LSE
                    output, lse, *rest = output  # 解包输出和LSE
                    lse = torch.transpose(lse, 0, 1).contiguous()  # 转置LSE
                    return output, lse  # 返回输出和LSE
                return output  # 返回输出
            else:  # 无前缀缓存
                # Do absorbed multi-latent attention  # 吸收式多头潜在注意力
                kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(  # 获取KV缓存
                    q.dtype  # 转为查询精度
                )
                k_rope = kv_cache[:, :, layer.v_head_dim :]  # 旋转位置编码部分
                c_kv = kv_cache[:, :, : layer.v_head_dim]  # 压缩KV部分
                k_rope_cache = k_rope.view(  # 重塑旋转位置编码缓存
                    -1,  # 页数
                    self.page_size,  # 页大小
                    layer.tp_k_head_num,  # KV头数
                    layer.head_dim - layer.v_head_dim,  # 旋转维度
                )
                c_kv_cache = c_kv.view(  # 重塑压缩KV缓存
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim  # [页数, 页大小, V头数, V头维度]
                )
                if q_rope is not None:  # 如果提供了旋转位置编码查询
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)  # 非旋转部分查询
                    q_rope = q_rope.view(  # 重塑旋转部分查询
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim  # [token, 头数, 旋转维度]
                    )
                else:  # 未分离提供
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)  # 合并查询
                    q_nope = q_all[:, :, : layer.v_head_dim]  # 非旋转部分
                    q_rope = q_all[:, :, layer.v_head_dim :]  # 旋转部分

                result = flash_attn_with_kvcache(  # 带KV缓存的MLA Flash Attention
                    q=q_rope,  # 旋转部分查询
                    k_cache=k_rope_cache,  # 旋转部分键缓存
                    v_cache=c_kv_cache,  # 压缩KV缓存
                    qv=q_nope,  # 非旋转部分查询（MLA特有）
                    page_table=page_table,  # 页表
                    cache_seqlens=cache_seqlens,  # 缓存序列长度
                    cu_seqlens_q=cu_seqlens_q,  # 查询累积长度
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,  # KV新累积长度
                    max_seqlen_q=max_seqlen_q,  # 最大查询长度
                    softmax_scale=layer.scaling,  # softmax缩放
                    causal=False if use_cascade_attn else causal,  # 因果性
                    softcap=layer.logit_cap,  # softmax上限
                    k_descale=k_descale,  # K反缩放
                    v_descale=v_descale,  # V反缩放
                    return_softmax_lse=use_cascade_attn,  # 级联需要LSE
                )
                if use_cascade_attn:  # 如果使用级联注意力
                    o, softmax_lse, *rest = result  # 解包
                    o_expand, softmax_lse_expand, *rest_expand = (  # 第二阶段
                        flash_attn_with_kvcache(  # 扩展MLA Flash Attention
                            q=q_rope,  # 旋转部分查询
                            k_cache=k_rope_cache,  # 旋转部分键缓存
                            v_cache=c_kv_cache,  # 压缩KV缓存
                            qv=q_nope,  # 非旋转部分查询
                            page_table=self.forward_metadata_spec_decode_expand.page_table,  # 扩展页表
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,  # 扩展缓存长度
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,  # 扩展查询长度
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,  # 扩展KV长度
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,  # 扩展最大查询长度
                            softmax_scale=layer.scaling,  # softmax缩放
                            causal=False,  # 非因果
                            window_size=window_size,  # 窗口大小
                            softcap=layer.logit_cap,  # softmax上限
                            k_descale=k_descale,  # K反缩放
                            v_descale=v_descale,  # V反缩放
                            return_softmax_lse=True,  # 需要LSE
                        )
                    )
                    o, _ = merge_state_v2_wrapper(  # 合并结果
                        o,  # 第一阶段
                        softmax_lse.T.contiguous(),  # 第一阶段LSE
                        o_expand,  # 扩展
                        softmax_lse_expand.T.contiguous(),  # 扩展LSE
                    )
                else:  # 非级联
                    o = result  # 直接使用结果

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)  # 返回重塑后的输出

    def forward_decode(  # 解码注意力前向传播
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
        # For multi-head latent attention  # 多头潜在注意力用
        q_rope: Optional[torch.Tensor] = None,  # 旋转位置编码查询
        k_rope: Optional[torch.Tensor] = None,  # 旋转位置编码键
        sinks: Optional[torch.Tensor] = None,  # sink参数
    ) -> torch.Tensor:  # 返回输出张量
        if k is not None:  # 如果K不为None
            assert v is not None  # V也不应为None
            if save_kv_cache:  # 如果保存KV缓存
                cache_loc = (  # 缓存位置
                    forward_batch.out_cache_loc  # 输出缓存位置
                    if not layer.is_cross_attention  # 非交叉注意力
                    else forward_batch.encoder_out_cache_loc  # 交叉注意力用编码器缓存位置
                )
                if not self.use_mla:  # 非MLA
                    self.token_to_kv_pool.set_kv_buffer(  # 设置KV缓存
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale  # 层、位置、K、V、缩放
                    )
                else:  # MLA
                    k_rope_val = (  # 旋转位置编码键值
                        k_rope if k_rope is not None else k[:, :, layer.v_head_dim :]  # 使用提供的或从K中提取
                    )
                    self.token_to_kv_pool.set_mla_kv_buffer(  # 设置MLA KV缓存
                        layer,  # 层
                        cache_loc,  # 缓存位置
                        k,  # 压缩KV
                        k_rope_val,  # 旋转位置编码键
                    )

        # Use precomputed metadata across all layers  # 在所有层间使用预计算的元数据
        metadata = self.forward_metadata  # 获取元数据
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)  # 获取局部注意力元数据
        use_local_attn = (  # 使用局部注意力条件
            self.attention_chunk_size is not None  # 有分块大小
            and local_attn_metadata is not None  # 有局部元数据
            and (hasattr(layer, "use_irope") and layer.use_irope)  # 层启用irope
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don't need cascade attention, spec_info will be none in this case
        # 当启用推测解码时，forward_decode会以两种模式调用：
        # 1. DRAFT_DECODE：top_k>1时启用级联注意力
        # 2. IDLE：不需要级联注意力，spec_info为None
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1  # 级联注意力条件

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # 计算窗口大小（如果层属性不变，可移到元数据中）
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # 我们不做layer.sliding_window_size - 1，因为在model.get_attention_sliding_window_size()中已经-1了
        # here is two side inclusive  # 这里是两侧包含
        window_size = (  # 窗口大小
            (layer.sliding_window_size, 0)  # SWA窗口
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1  # 有SWA
            else (-1, -1)  # 无窗口限制
        )
        causal = not layer.is_cross_attention  # 非交叉注意力为因果

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        # 为了fa3接口版本兼容性，我们将新字段放入条件关键字参数
        kwargs = {}  # 额外参数
        if sinks is not None:  # 如果有sink
            kwargs["sinks"] = sinks  # 添加sink

        k_descale, v_descale = None, None  # KV反缩放初始化
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        # 仅在以下条件同时满足时使用KV缩放：1) 显式启用fp8 kv，2) RadixAttention
        # 有相应的量化方法使layer.k_scale不为None，
        # 3) layer.head_dim <= 256，因为FA3内核在此情况下要求fp16和bf16数据类型。
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:  # 需要FP8 KV缩放
            if layer.k_scale is not None:  # 有缩放因子
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)  # 缩放形状
                k_descale = layer.k_scale.expand(descale_shape)  # K反缩放
                v_descale = layer.v_scale.expand(descale_shape)  # V反缩放
            q = q.to(self.kv_cache_dtype)  # 转Q为KV缓存精度
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None  # 转q_rope
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None  # 转k_rope
        if not self.use_mla:  # 非MLA
            # Do multi-head attention  # 多头注意力

            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)  # 获取KV缓存
            key_cache = key_cache.view(  # 重塑键缓存
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim  # [页数, 页大小, KV头数, 头维度]
            )
            value_cache = value_cache.view(  # 重塑值缓存
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim  # [页数, 页大小, V头数, 头维度]
            )

            if layer.is_cross_attention:  # 交叉注意力
                # Always use non-chunked logic for cross-attention  # 交叉注意力始终使用非分块逻辑
                o = flash_attn_with_kvcache(  # 带KV缓存的Flash Attention
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),  # Q
                    k_cache=key_cache,  # 键缓存
                    v_cache=value_cache,  # 值缓存
                    page_table=metadata.encoder_page_table,  # 编码器页表
                    cache_seqlens=metadata.encoder_lens_int32,  # 编码器长度
                    cu_seqlens_q=metadata.cu_seqlens_q,  # 查询累积长度
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,  # 编码器KV累积长度
                    max_seqlen_q=1,  # 解码时查询长度为1
                    softmax_scale=layer.scaling,  # softmax缩放
                    causal=False,  # 非因果
                    window_size=(-1, -1),  # 无窗口限制
                    softcap=layer.logit_cap,  # softmax上限
                    k_descale=k_descale,  # K反缩放
                    v_descale=v_descale,  # V反缩放
                    **kwargs,  # 额外参数
                )
            elif use_local_attn:  # 局部注意力
                # Use chunked (local) attention batching for self-attention
                # 自注意力使用分块（局部）注意力批处理
                o = flash_attn_with_kvcache(  # 带KV缓存的Flash Attention
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),  # Q
                    k_cache=key_cache,  # 键缓存
                    v_cache=value_cache,  # 值缓存
                    page_table=local_attn_metadata.local_block_table,  # 局部块表
                    cache_seqlens=local_attn_metadata.local_seqused_k,  # 局部KV长度
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,  # 局部查询起始位置
                    cu_seqlens_k_new=None,  # 无新KV累积长度
                    max_seqlen_q=local_attn_metadata.local_max_query_len,  # 局部最大查询长度
                    softmax_scale=layer.scaling,  # softmax缩放
                    causal=True,  # 因果注意力
                    window_size=(-1, -1),  # 无窗口限制
                    softcap=layer.logit_cap,  # softmax上限
                    k_descale=k_descale,  # K反缩放
                    v_descale=v_descale,  # V反缩放
                    **kwargs,  # 额外参数
                )
            else:  # 标准解码
                page_table = metadata.page_table  # 页表
                cache_seqlens = metadata.cache_seqlens_int32  # 缓存序列长度
                cu_seqlens_k = metadata.cu_seqlens_k  # KV累积长度
                max_seqlen_q = metadata.max_seq_len_q  # 最大查询长度
                q_reshaped = q.contiguous().view(  # 重塑Q
                    -1, layer.tp_q_head_num, layer.head_dim  # [token, 头数, 头维度]
                )

                # Default: single-token self-attention  # 默认：单token自注意力
                result = flash_attn_with_kvcache(  # 带KV缓存的Flash Attention
                    q=q_reshaped,  # Q
                    k_cache=key_cache,  # 键缓存
                    v_cache=value_cache,  # 值缓存
                    page_table=page_table,  # 页表
                    cache_seqlens=cache_seqlens,  # 缓存序列长度
                    cu_seqlens_q=metadata.cu_seqlens_q,  # 查询累积长度
                    cu_seqlens_k_new=cu_seqlens_k,  # KV累积长度
                    max_seqlen_q=max_seqlen_q,  # 最大查询长度
                    softmax_scale=layer.scaling,  # softmax缩放
                    causal=False if use_cascade_attn else causal,  # 因果性
                    window_size=window_size,  # 窗口大小
                    softcap=layer.logit_cap,  # softmax上限
                    k_descale=k_descale,  # K反缩放
                    v_descale=v_descale,  # V反缩放
                    return_softmax_lse=use_cascade_attn,  # 级联需要LSE
                    **kwargs,  # 额外参数
                )
                if use_cascade_attn:  # 级联注意力
                    o, softmax_lse, *rest = result  # 解包
                    o_expand, softmax_lse_expand, *rest_expand = (  # 扩展阶段
                        flash_attn_with_kvcache(  # 扩展Flash Attention
                            q=q_reshaped,  # Q
                            k_cache=key_cache,  # 键缓存
                            v_cache=value_cache,  # 值缓存
                            page_table=self.forward_metadata_spec_decode_expand.page_table,  # 扩展页表
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,  # 扩展缓存长度
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,  # 扩展查询长度
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,  # 扩展KV长度
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,  # 扩展最大查询长度
                            softmax_scale=layer.scaling,  # softmax缩放
                            causal=False,  # 非因果
                            window_size=window_size,  # 窗口大小
                            softcap=layer.logit_cap,  # softmax上限
                            k_descale=k_descale,  # K反缩放
                            v_descale=v_descale,  # V反缩放
                            return_softmax_lse=True,  # 需要LSE
                            **kwargs,  # 额外参数
                        )
                    )
                    o, _ = merge_state_v2(  # 合并结果
                        o,  # 第一阶段
                        softmax_lse.T.contiguous(),  # 第一阶段LSE
                        o_expand,  # 扩展
                        softmax_lse_expand.T.contiguous(),  # 扩展LSE
                    )
                else:  # 非级联
                    o = result  # 直接使用结果
        else:  # MLA
            # Do absorbed multi-latent attention  # 吸收式多头潜在注意力
            kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)  # 获取KV缓存
            assert not use_cascade_attn, "Cascade attention is not supported with MLA"  # MLA不支持级联注意力

            if q_rope is not None:  # 如果提供了旋转位置编码查询
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)  # 非旋转部分
                q_rope = q_rope.view(  # 旋转部分
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim  # [token, 头数, 旋转维度]
                )
            else:  # 未分离提供
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)  # 合并查询
                q_nope = q_all[:, :, : layer.v_head_dim]  # 非旋转部分
                q_rope = q_all[:, :, layer.v_head_dim :]  # 旋转部分

            o = flash_mla_decode(  # MLA解码Flash Attention
                q_nope,  # 非旋转部分查询
                q_rope,  # 旋转部分查询
                kv_cache.view(-1, self.page_size, layer.head_dim),  # 重塑KV缓存
                metadata.cache_seqlens_int32,  # 缓存序列长度
                metadata.page_table,  # 页表
                self.workspace,  # 工作空间
                layer.scaling,  # softmax缩放
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)  # 返回重塑后的输出

    def get_cuda_graph_seq_len_fill_value(self):  # 获取CUDA图中序列长度的填充值
        """Get the fill value for sequence length in CUDA graph.
        获取CUDA图中序列长度的填充值。"""
        return 1  # 返回1

    def _init_local_attn_metadata(  # 初始化局部注意力元数据
        self, forwardbatch: ForwardBatch, metadata: FlashAttentionMetadata, device  # 前向批次、元数据、设备
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled.
        如果启用了分块注意力，初始化local_attn_metadata的集中工具。"""
        if self.attention_chunk_size is None:  # 如果没有分块大小
            metadata.local_attn_metadata = None  # 无局部注意力
            return  # 直接返回

        cu_seqlens_q = metadata.cu_seqlens_q  # 查询累积长度
        cache_seqlens_int32 = metadata.cache_seqlens_int32  # 缓存序列长度
        if self.is_hybrid_swa:  # 如果是混合SWA
            page_table = self.full_to_swa_index_mapping[metadata.page_table].to(  # 映射到SWA索引
                torch.int32  # 转int32
            )
        else:  # 非混合SWA
            page_table = metadata.page_table  # 使用原始页表
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:  # 缺少必要数据
            metadata.local_attn_metadata = None  # 无局部注意力
            return  # 直接返回

        # make_local_attention_virtual_batches expects a page-granularity block table:
        # column p is the logical page number, and the value stored at that column is the
        # physical page index. The raw req_to_token table is token-granularity (column i =
        # the KV slot for token i), so when page_size > 1 we must stride and divide first
        # so that block_starts = k_seqstarts_absolute // page_size correctly indexes the table.
        # make_local_attention_virtual_batches期望页粒度的块表：
        # 列p是逻辑页号，该列存储的值是物理页索引。原始req_to_token表是token粒度的
        # （列i = token i的KV槽），因此当page_size>1时，我们必须先步进和除法，
        # 使得block_starts = k_seqstarts_absolute // page_size正确索引表。
        if self.page_size > 1:  # 如果页大小>1
            strided_indices = torch.arange(  # 步进索引
                0, page_table.shape[1], self.page_size, device=page_table.device  # 按页大小步进
            )
            page_table = page_table[:, strided_indices] // self.page_size  # 转为页粒度

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()  # 查询累积长度转numpy
        seq_lens_np = cache_seqlens_int32.cpu().numpy()  # 序列长度转numpy
        (  # 创建局部注意力虚拟批次
            seqlens_q_local_np,  # 局部查询长度
            cu_seqlens_q_local_np,  # 局部查询累积长度
            seqlens_k_local_np,  # 局部KV长度
            block_table_local,  # 局部块表
        ) = make_local_attention_virtual_batches(  # 创建虚拟批次
            self.attention_chunk_size,  # 分块大小
            cu_seqlens_q_np,  # 查询累积长度
            seq_lens_np,  # 序列长度
            page_table,  # 页表
            self.page_size,  # 页大小
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(  # 创建局部注意力元数据
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),  # 局部查询起始位置
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),  # 局部KV长度
            local_block_table=block_table_local.to(device),  # 局部块表
            local_max_query_len=int(seqlens_q_local_np.max()),  # 局部最大查询长度
            local_max_seq_len=int(seqlens_k_local_np.max()),  # 局部最大KV长度
        )
        metadata.local_attn_metadata = local_metadata  # 保存局部注意力元数据

    def _init_sliding_window_attn_spec_metadata(  # 初始化滑动窗口推测注意力元数据
        self,
        metadata: FlashAttentionMetadata,  # 主元数据
        metadata_expand: FlashAttentionMetadata,  # 扩展元数据
        metadata_swa: Optional[FlashAttentionMetadata] = None,  # SWA元数据（可选）
    ):
        # TODO: support page_size > 1 for swa spec  # TODO: 支持page_size>1的SWA推测
        assert (  # 断言页大小为1
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"  # FlashAttention后端不支持page_size>1和topk>1的滑动窗口推测解码

        cache_seqlens_int32 = (  # 合并缓存序列长度
            metadata.cache_seqlens_int32.repeat_interleave(  # 重复原始长度
                self.speculative_num_draft_tokens  # 按草稿数重复
            )
            + metadata_expand.cache_seqlens_int32  # 加上扩展长度
        )
        cu_seqlens_k = torch.nn.functional.pad(  # KV累积序列长度
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)  # 累积和并左填充
        )
        bs = cache_seqlens_int32.shape[0]  # 批量大小
        page_table = (  # 页表
            metadata.page_table.new_zeros(  # 创建零页表
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])  # 形状
            )
            if metadata_swa is None  # 无已有SWA元数据
            else metadata_swa.page_table  # 使用已有的
        )

        prepare_swa_spec_page_table_triton(  # Triton内核准备SWA推测页表
            page_table,  # 输出页表
            metadata.page_table,  # 主页表
            metadata_expand.page_table,  # 扩展页表
            metadata.cache_seqlens_int32,  # 主缓存长度
            metadata_expand.cache_seqlens_int32,  # 扩展缓存长度
            self.speculative_num_draft_tokens,  # 草稿token数
        )

        if metadata_swa is None:  # 无已有SWA元数据
            metadata_swa = FlashAttentionMetadata()  # 创建新的
            metadata_swa.max_seq_len_q = 1  # 查询最大长度为1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q  # 使用扩展的查询累积长度
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32  # 缓存序列长度
            metadata_swa.cu_seqlens_k = cu_seqlens_k  # KV累积长度
            metadata_swa.page_table = page_table  # 页表
        else:  # 有已有SWA元数据
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)  # 复制缓存长度
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)  # 复制KV累积长度

        metadata.swa_spec_metadata = metadata_swa  # 保存SWA推测元数据
