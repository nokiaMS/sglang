# FlashAttention后端实现 - SGLang中基于FlashAttention的注意力计算后端
# 本文件实现了FlashAttention v3/v4后端，支持解码、扩展（预填充）、投机解码、
# CUDA图、滑动窗口注意力、局部注意力、多潜在注意力(MLA)和上下文并行等功能。

from __future__ import annotations  # 启用延迟类型注解评估

from dataclasses import dataclass  # 数据类装饰器
from typing import TYPE_CHECKING, Optional  # 类型检查和可选类型

import numpy as np  # NumPy数值计算库
import torch  # PyTorch深度学习框架
import triton  # Triton GPU编程框架
import triton.language as tl  # Triton GPU编程语言

from sglang.srt.configs.model_config import AttentionArch  # 注意力架构枚举
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 注意力后端基类
from sglang.srt.layers.radix_attention import AttentionType  # 注意力类型枚举
from sglang.srt.layers.utils.cp_utils import (  # 上下文并行工具函数
    cp_allgather_and_save_kv_cache,  # CP全聚合并保存KV缓存
    cp_attn_forward_extend,  # CP注意力前向扩展
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool  # 滑动窗口注意力KV池
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 前向批次信息和前向模式
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.speculative.spec_info import SpecInput  # 投机解码输入信息
from sglang.srt.utils import get_compiler_backend  # 获取编译器后端

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.radix_attention import RadixAttention  # Radix注意力层
    from sglang.srt.model_executor.model_runner import ModelRunner  # 模型运行器

from sgl_kernel import merge_state_v2  # 合并状态的v2版本内核

from sglang.jit_kernel.flash_attention import (  # JIT编译的FlashAttention函数
    flash_attn_varlen_func,  # 变长序列FlashAttention函数
    flash_attn_with_kvcache,  # 带KV缓存的FlashAttention函数
)


# FlashAttention元数据类，在模型前向传播中初始化一次，各层可复用
@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.
    在模型前向传播中初始化一次的元数据，每层的前向传播可以复用该元数据。

    For each init metadata function, we will try set up them in below order
    对于每个初始化元数据函数，我们将按以下顺序设置
    """

    # 前向批次的序列长度
    cache_seqlens_int32: torch.Tensor = None
    # 查询的最大序列长度
    max_seq_len_q: int = 1
    # 键的最大序列长度
    max_seq_len_k: int = 0
    # 查询的累计序列长度
    cu_seqlens_q: torch.Tensor = None
    # 键的累计序列长度
    cu_seqlens_k: torch.Tensor = None
    # 窗口大小（通常用于Gemma模型）
    window_size: tuple = (-1, -1)
    # 页表，KV缓存表/块的索引
    page_table: torch.Tensor = None
    # 滑动窗口注意力的页表
    swa_page_table: torch.Tensor = None
    # 预计算的FA3调度器元数据（避免每层调用prepare_varlen_num_blocks）
    scheduler_metadata: torch.Tensor = None

    # 编码器元数据
    # 编码器键的累计序列长度
    encoder_cu_seqlens_k: torch.Tensor = None
    # 编码器键的最大序列长度
    encoder_max_seq_len_k: int = 0
    # 前向批次的序列长度
    encoder_lens_int32: torch.Tensor = None
    # 编码器的页表
    encoder_page_table: torch.Tensor = None

    # 局部注意力元数据类
    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
        local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
        local_block_table: torch.Tensor = None  # block table for local attention
        local_max_query_len: int = 0  # max query length for local attention
        local_max_seq_len: int = 0  # max sequence length for local attention

    local_attn_metadata: Optional[LocalAttentionMetadata] = None

    # 用于滑动窗口注意力topk>1投机解码
    swa_spec_metadata: Optional[FlashAttentionMetadata] = None


# FlashAttention后端实现类，支持解码、扩展、投机解码、CUDA图等
class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation.

    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    # 初始化FlashAttention后端
    def __init__(
        self,
        model_runner: ModelRunner,  # 模型运行器
        skip_prefill: bool = False,  # 是否跳过预填充
        speculative_step_id=0,  # 投机步ID
        topk=0,  # topk值
        speculative_num_steps=0,  # 投机步数
        fa_impl_ver=3,  # FlashAttention实现版本
    ):
        super().__init__()  # 调用父类初始化

        assert not (  # 断言不同时支持滑动窗口和交叉注意力
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"  # 滑动窗口和交叉注意力不能同时使用

        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder  # 是否为编码器-解码器模型
        self.forward_metadata: FlashAttentionMetadata = None  # 前向传播元数据
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify  # 处理投机解码topk>1、扩展草稿解码和验证的额外元数据
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None  # 投机解码扩展元数据
        self.max_context_len = model_runner.model_config.context_len  # 最大上下文长度
        self.device = model_runner.device  # 计算设备
        self.decode_cuda_graph_metadata = {}  # 解码CUDA图元数据字典
        self.target_verify_metadata = {}  # 目标验证元数据字典
        # 池引用 — captured at construction so they survive deletion of the  # 池引用 - 在构造时捕获，以便在
        # corresponding ForwardBatch fields.  # 对应的ForwardBatch字段被删除后仍然存活。
        self.req_to_token_pool = model_runner.req_to_token_pool  # 请求到令牌的池
        self.token_to_kv_pool = model_runner.token_to_kv_pool  # 令牌到KV的池
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到令牌的映射表
        self.kv_cache_dtype = model_runner.kv_cache_dtype  # KV缓存数据类型
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype  # KV缓存数据类型字符串
        self.page_size = model_runner.page_size  # 页大小
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA  # 是否使用MLA注意力
        self.skip_prefill = skip_prefill  # 是否跳过预填充
        self.attn_cp_size = model_runner.attn_cp_size  # 注意力上下文并行大小

        self.use_sliding_window_kv_pool = (  # 是否使用滑动窗口KV池
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)  # 检查token_to_kv_pool是否为SWAKVPool实例
            and model_runner.token_to_kv_pool.swa_layer_nums > 0  # 且SWA层数大于0
        )

        self.topk = model_runner.server_args.speculative_eagle_topk or 0  # EAGLE投机解码的topk值
        self.speculative_num_steps = speculative_num_steps  # 投机步数
        self.speculative_num_draft_tokens = (  # 投机草稿令牌数
            model_runner.server_args.speculative_num_draft_tokens  # 从服务器参数获取
        )
        self.speculative_step_id = speculative_step_id  # 投机步ID

        # 局部注意力设置
        self.has_local_attention = model_runner.model_config.is_local_attention_model
        if self.has_local_attention:  # 如果有局部注意力
            assert (  # 断言注意力块大小不为None
                model_runner.attention_chunk_size is not None
            ), "Attention chunk size is required for local attention"  # 局部注意力需要注意力块大小
            self.attention_chunk_size = model_runner.attention_chunk_size  # 注意力块大小

        # 每层的滑动窗口大小可能不同。这仅用于准备SWA元数据。
        # 使用`layer.sliding_window_size`决定每层是否使用SWA。
        self.sliding_window_size = model_runner.sliding_window_size  # 滑动窗口大小
        self.has_swa = (  # 是否有滑动窗口注意力
            self.sliding_window_size is not None and self.sliding_window_size > -1  # 滑动窗口大小非None且>-1
        )

        # 选择FA版本
        self.fa_impl_ver = fa_impl_ver  # FA实现版本
        if self.fa_impl_ver == 3:  # 如果版本为3
            from sgl_kernel.flash_attn import (  # 从sgl_kernel导入FA3函数
                flash_attn_varlen_func,  # 变长FlashAttention函数
                flash_attn_with_kvcache,  # 带KV缓存的FlashAttention函数
                get_scheduler_metadata,  # 获取调度器元数据
            )

            self._get_scheduler_metadata = get_scheduler_metadata  # 保存调度器元数据获取函数
        elif self.fa_impl_ver == 4:  # 如果版本为4
            from sglang.jit_kernel.flash_attention_v4 import (  # 从JIT内核导入FA4函数
                flash_attn_varlen_func,  # 变长FlashAttention函数
                flash_attn_with_kvcache,  # 带KV缓存的FlashAttention函数
            )

            self._get_scheduler_metadata = None  # FA4不需要调度器元数据
        else:  # 其他版本
            raise ValueError(f"Invalid version: {self.fa_impl_ver=}")  # 抛出版本无效错误

        self.flash_attn_varlen_func = flash_attn_varlen_func  # 变长FlashAttention函数
        self.flash_attn_with_kvcache = flash_attn_with_kvcache  # 带KV缓存的FlashAttention函数

        # 存储头信息用于预计算FA3调度器元数据
        self.head_dim = model_runner.model_config.head_dim  # 头维度
        self.num_attention_heads = (  # 注意力头数
            model_runner.model_config.hf_text_config.num_attention_heads  # 总注意力头数
            // model_runner.tp_size  # 除以张量并行大小
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(  # KV头数
            model_runner.tp_size  # 张量并行大小
        )
        _softcapping = getattr(  # 获取注意力logit软上限
            model_runner.model_config.hf_text_config, "attn_logit_softcapping", None  # 从配置获取软上限参数
        )
        self.has_softcap = _softcapping is not None and _softcapping > 0.0  # 是否有软上限

        # 如果num_splits==0，使用启发式自动确定分割数。
        # 启用确定性推理时将分割数设为1。
        # 详见该博客了解更多细节。
        # 此外，FA4不支持CUDA图下num_splits=0，启用CUDA图时设为1。
        self.num_splits = (  # 分割数
            1  # 设为1
            if model_runner.server_args.enable_deterministic_inference  # 如果启用确定性推理
            or (  # 或者
                self.fa_impl_ver == 4  # FA4版本
                and not model_runner.server_args.disable_cuda_graph  # 且未禁用CUDA图
            )
            else 0  # 否则设为0（自动确定）
        )

        # 在嵌入模式下，无分块预填充且禁用radix缓存时，
        # 跳过KV缓存写入，使用原始K/V调用flash_attn_varlen_func
        # 而非flash_attn_with_kvcache，完全绕过分页KV缓存。
        # 限制为非MLA后端：读取跳过的elif位于
        # forward_extend中的`if not self.use_mla:`分支，而写入跳过
        # 保护同时包裹set_kv_buffer和set_mla_kv_buffer。没有这个
        # 门控，MLA+is_embedding会跳过写入但仍读取过期
        # 通过absorbed-MLA路径中get_key_buffer读取的缓存。
        server_args = model_runner.server_args  # 服务器参数
        self.fa_skip_kv_cache = (  # 是否跳过KV缓存
            server_args.is_embedding  # 是否为嵌入模式
            and server_args.chunked_prefill_size == -1  # 且无分块预填充
            and server_args.disable_radix_cache  # 且禁用radix缓存
            and not self.use_mla  # 且不使用MLA
        )

        # 在DP注意力下跳过FA3 scheduler_metadata预计算
        # 预计算的缓冲区可能与
        # C++ mha_fwd内核从实时cache_seqlens推导的num_splits不一致
        # 导致split-KV合并内核中出现越界读取
        # （flash_fwd_combine_launch_template.h:52）。保持scheduler_metadata
        # 未设置则使用现有的逐层元数据路径。
        self._disable_scheduler_metadata_precompute = bool(  # 是否禁用调度器元数据预计算
            getattr(server_args, "enable_dp_attention", False)  # 获取是否启用DP注意力
        )

    # 计算FA3调度器元数据，用于解码阶段避免逐层调用prepare_varlen_num_blocks
    def _compute_scheduler_metadata(
        self, batch_size, max_seq_len_k, cache_seqlens, cu_seqlens_q
    ):
        """Compute FA3 scheduler metadata for decode.
        为解码计算FA3调度器元数据。

        Returns the scheduler_metadata tensor, or None if not applicable.
        返回scheduler_metadata张量，如果不适用则返回None。
        """
        if self._get_scheduler_metadata is None or self.use_mla:
            return None  # 返回None
        if self._disable_scheduler_metadata_precompute:
            return None  # 返回None
        # Always use window_size=(-1, -1) because scheduler_metadata is only
        # consumed by non-SWA layers (SWA layers skip it in forward_decode).
        return self._get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_seq_len_k,
            num_heads=self.num_attention_heads,
            num_heads_k=self.num_kv_heads,
            headdim=self.head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self.kv_cache_dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=self.page_size,
            causal=True,
            has_softcap=self.has_softcap,
            num_splits=self.num_splits,
        )

    # 初始化前向传播元数据，使所有层可复用
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it.
    初始化前向元数据，使前向传播中的所有层可以复用。"""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # 草稿解码
            if forward_batch.spec_info is not None:
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    # 形状：[批大小,步数,topk]->[批大小*topk,步数]
                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand  # 保存投机解码扩展元数据
            else:
                # 普通解码
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
                # 预计算FA3调度器元数据以避免每层
                # 调用prepare_varlen_num_blocks内核
                metadata.scheduler_metadata = self._compute_scheduler_metadata(
                    batch_size,
                    metadata.max_seq_len_k,
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_q,
                )
            # TODO：需要为llama 4 eagle情况测试此部分
            self._maybe_init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._maybe_init_local_attn_metadata(forward_batch, metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # 创建扩展页表
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (批大小*草稿数, 草稿数)

                # 移动表索引以避免填充
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # 注意cache_seqlens_int32为[1,2,2]，每行中多余的页索引将被忽略
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # 构建键：如果条目有效（mask==True），保持原始索引；
                # 否则加上投机草稿令牌数使其排序在所有有效条目之后。
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (批大小, 草稿数)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand  # 保存投机解码扩展元数据

                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )

            # MLA/MHA CP：prepare_mlp_sync_batch将扩展令牌填充至
            # lcm(注意力TP大小,注意力CP大小)，因此cache_seqlens_cp可能超过
            # seq_lens_cpu.max()。通过填充增量扩展page_table以保持
            # FA3的因果读取在范围内；扩展的列索引KV槽0
            # （req_to_token零初始化）且填充查询的输出
            # 在下游被丢弃。
            if (
                self.attn_cp_size > 1
                and forward_batch.global_num_tokens_cpu is not None
                and forward_batch.extend_num_tokens is not None
                and forward_batch.extend_seq_lens_cpu is not None
            ):
                padded_extend = int(forward_batch.extend_num_tokens)
                real_extend = int(sum(forward_batch.extend_seq_lens_cpu))
                pad_delta = padded_extend - real_extend
                if pad_delta > 0:
                    metadata.max_seq_len_k += pad_delta

            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if any(
                forward_batch.extend_prefix_lens_cpu
            ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # 如果启用则设置局部注意力
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._maybe_init_local_attn_metadata(forward_batch, metadata, device)

        # 编码器元数据 for cross attention. Supports per-request varlen
        # 编码器长度（如MossVL每个请求有不同图像大小）。
        if forward_batch.encoder_lens is not None:
            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()

            # 交叉注意力页表：每请求行。cache_seqlens
            # （encoder_lens_int32）限制每请求读取，因此
            # 超过encoder_lens[i]的垃圾数据不会被消费。
            metadata.encoder_page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # 自注意力（文本）页表：文本从每请求偏移量开始
            # encoder_lens[i]，而非单一最大值。使用花式索引收集。
            text_max = metadata.max_seq_len_k
            arange_text = torch.arange(
                text_max, device=forward_batch.req_pool_indices.device
            )
            text_col = forward_batch.encoder_lens.long().unsqueeze(
                1
            ) + arange_text.unsqueeze(
                0
            )  # (bs, max_seq_len_k)
            text_row = forward_batch.req_pool_indices.unsqueeze(1).expand(-1, text_max)
            metadata.page_table = self.req_to_token_pool.req_to_token[
                text_row, text_col
            ]

        if self.use_sliding_window_kv_pool:
            metadata.swa_page_table = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                )
            )

        # 将页表转换为FA3 API所需的跨步格式
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )

            if self.use_sliding_window_kv_pool:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

            if (
                self.topk > 1
                and forward_batch.forward_mode.is_decode_or_idle()
                and forward_batch.spec_info is not None
            ):
                # 修改cache_seqlens_int32和page_table(B, speculative_num_steps).
                last_page_lens = forward_batch.seq_lens % self.page_size
                # 第一次注意力处理前缀-最后一页长度的部分.
                metadata.cache_seqlens_int32 -= last_page_lens  # Both (B, )

                # 第二次注意力处理最后一页长度+解码部分.
                expanded_last_page_lens = last_page_lens.repeat_interleave(self.topk)
                self.forward_metadata_spec_decode_expand.cache_seqlens_int32 += (
                    expanded_last_page_lens
                )
                # NOTE: the max decode length is speculative_num_steps - 1 (one token always generated by draft extend)
                # and we leave one extra for last_page_len, which -> speculative_num_steps for the page table
                expand_page_table = torch.zeros(
                    forward_batch.batch_size * self.topk,
                    self.speculative_num_steps,
                    dtype=torch.int32,
                    device=self.device,
                )
                # 形状：[批大小,步数,topk]->[批大小*topk,步数]
                cache_loc = forward_batch.out_cache_loc.view(
                    -1, self.speculative_num_steps
                )
                draft_decode_set_expand_metadata(
                    cache_seqlens_int32=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    page_table=expand_page_table,
                    last_page_lens=last_page_lens,
                    decode_length=decode_length,
                    cache_loc=cache_loc,
                    topk=self.topk,
                    page_size=self.page_size,
                )
                self.forward_metadata_spec_decode_expand.page_table = expand_page_table

        self.forward_metadata = metadata  # 保存前向传播元数据

    # 前向扩展（预填充）方法，处理新令牌的注意力计算
    def forward_extend(
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
        # 用于多头潜在注意力
        q_rope: Optional[torch.Tensor] = None,  # 旋转位置编码查询
        k_rope: Optional[torch.Tensor] = None,  # 旋转位置编码键
        sinks: Optional[torch.Tensor] = None,  # 注意力汇聚点
    ):
        is_cp_mode = (  # 是否为上下文并行模式
            forward_batch.forward_mode.is_context_parallel_extend()  # 是否是上下文并行扩展
            and forward_batch.attn_cp_metadata is not None  # 且有CP元数据
            and self.attn_cp_size > 1  # 且CP大小>1
        )

        if k is not None:
            assert v is not None

            if save_kv_cache and not self.fa_skip_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if self.use_mla:
                    # MLA：在CP下，k和k_rope以完整序列到达
                    # （rebuild_cp_kv_cache在上游运行于
                    # forward_absorb_prepare中）；否则为rank本地。
                    # out_cache_loc不会zigzag分割，因此写入
                    # 在两种情况下都会落在每个rank的正确槽位。
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                elif is_cp_mode:
                    # 稠密MHA CP：k,v仍是rank本地的；后端
                    # 执行全聚合并写入每个rank的池。
                    cp_allgather_and_save_kv_cache(
                        forward_batch, layer, k, v, self.attn_cp_size
                    )
                else:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

        # 使用跨所有层预计算的元数据
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # 不做-1因为在get_attention_sliding_window_size()中已经-1
        # 这里是两侧包含
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
        k_descale, v_descale = None, None  # KV反缩放因子初始化为None
        # 仅在以下条件满足时使用KV缩放：1)显式启用fp8 KV，2)RadixAttention
        # 有对应量化方法使layer.k_scale非None，
        # 3)head_dim<=256因为fa3内核需fp16/bf16，
        # 4)fa_impl_ver!=4因为fa4不支持fp8查询和键。
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = True  # 默认启用因果注意力
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        # 检查是否应使用局部注意力
        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # 对topk>1的目标验证使用级联注意力
        # 对滑动窗口注意力不使用级联注意力：
        # - Different window sizes should be passed in for each q in the first stage of cascade attention, but FA3 interface doesn't support pass in a list of window sizes.
        # - The overhead of duplicated computation of the common prefix part is small for sliding window layers (seq_len <= window_size), so we can just expand it.
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and not is_swa_layer
        )

        kwargs = {}  # 额外参数字典
        if sinks is not None:
            kwargs["sinks"] = sinks

        _fa_out = (
            forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if getattr(forward_batch, "_attn_output", None) is not None
            else None
        )

        # 根据是否使用局部注意力获取合适的页表
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_swa_layer and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        else:
            page_table = metadata.page_table
            if is_swa_layer and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        metadata.page_table
                    )
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        # 使用FlashAttention进行预填充
        if not self.use_mla:
            # 执行多头注意力
            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            if (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            ):

                # FA上下文并行注意力闭包函数
                def _fa_cp_attn(
                    q_chunk, cu_seqlens_q_cp, cache_seqlens_cp, max_seqlen_q_cp
                ):
                    return flash_attn_with_kvcache(
                        q=q_chunk,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens_cp,
                        cu_seqlens_q=cu_seqlens_q_cp,
                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                        max_seqlen_q=max_seqlen_q_cp,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        window_size=window_size,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=use_cascade_attn,
                        num_splits=self.num_splits,
                        ver=self.fa_impl_ver,
                        **kwargs,
                    )

                result = cp_attn_forward_extend(
                    forward_batch,
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    self.device,
                    _fa_cp_attn,
                )
            elif self.fa_skip_kv_cache:
                # 嵌入模式：跳过KV缓存读取，使用原始K/V张量
                # directly via flash_attn_varlen_func. The KV cache write is
                # 也被跳过（上面已防护）。这消除了store_kvcache
                # 和每层的prepare_varlen_num_blocks开销。
                assert k is not None, "fa_skip_kv_cache requires k to be provided"
                assert k_descale is None and v_descale is None, (
                    "fa_skip_kv_cache uses raw K/V tensors, "
                    "FP8 KV cache descaling is not supported in this mode"
                )
                result = flash_attn_varlen_func(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v=v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    **kwargs,
                )
            else:
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    # 这里的metadata_expand.page_table未除以page_size。
                    # 这是因为我们放松了对注意力令牌的精细控制，
                    # 而是必须完整地关注某个块。
                    k_cache=key_cache.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
                    v_cache=value_cache.view(
                        -1, 1, layer.tp_v_head_num, layer.head_dim
                    ),
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                # 执行多头注意力 with chunked prefix cache
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_global_server_args().disable_chunked_prefix_cache
                    # 运行MLA模型时分块前缀KV缓存的MHA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                        out=_fa_out,
                        ver=self.fa_impl_ver,
                        **kwargs,
                    )
                else:
                    # 不关注前缀KV缓存的序列扩展部分的MHA
                    cu_seqlens_k = (
                        metadata.cu_seqlens_q
                        if not forward_batch.mha_one_shot
                        else metadata.cu_seqlens_k
                    )
                    max_seqlen_k = (
                        metadata.max_seq_len_q
                        if not forward_batch.mha_one_shot
                        else metadata.max_seq_len_k
                    )
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                        out=_fa_out,
                        ver=self.fa_impl_ver,
                        **kwargs,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                assert self.fa_impl_ver == 3, "Only FA3 support here"
                # 执行吸收式多潜在注意力
                kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                    q.dtype
                )
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                if is_cp_mode:
                    # MLA CP：q是rank本地的zigzag分割；运行
                    # 吸收式MLA内核两次（前/后半部分）针对
                    # 完整的潜在KV池（rebuild_cp_kv_cache
                    # 在上游填充）通过cp_attn_forward_extend。
                    # 沿dim=-1拼接q_nope+q_rope以便包装器的
                    # chunk(2, dim=0)保持对齐；在闭包内拆分回来
                    # 。
                    assert (
                        not use_cascade_attn
                    ), "Cascade attention under MLA CP is not supported in v1."
                    q_fused = torch.cat([q_nope, q_rope], dim=-1)

                    # MLA上下文并行注意力闭包函数
                    def _mla_cp_attn(
                        q_chunk,
                        cu_seqlens_q_cp,
                        cache_seqlens_cp,
                        max_seqlen_q_cp,
                    ):
                        q_nope_chunk = q_chunk[..., : layer.v_head_dim]
                        q_rope_chunk = q_chunk[..., layer.v_head_dim :]
                        return flash_attn_with_kvcache(
                            q=q_rope_chunk,
                            qv=q_nope_chunk,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            page_table=page_table,
                            cache_seqlens=cache_seqlens_cp,
                            cu_seqlens_q=cu_seqlens_q_cp,
                            cu_seqlens_k_new=(
                                cu_seqlens_k if not use_local_attn else None
                            ),
                            max_seqlen_q=max_seqlen_q_cp,
                            softmax_scale=layer.scaling,
                            causal=causal,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            num_splits=self.num_splits,
                            ver=self.fa_impl_ver,
                        )

                    o = cp_attn_forward_extend(
                        forward_batch, q_fused, self.device, _mla_cp_attn
                    )
                else:
                    result = flash_attn_with_kvcache(
                        q=q_rope,
                        k_cache=k_rope_cache,
                        v_cache=c_kv_cache,
                        qv=q_nope,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                        max_seqlen_q=max_seqlen_q,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=use_cascade_attn,
                        num_splits=self.num_splits,
                        ver=self.fa_impl_ver,
                    )
                    if use_cascade_attn:
                        o, softmax_lse, *rest = result
                        o_expand, softmax_lse_expand, *rest_expand = (
                            flash_attn_with_kvcache(
                                q=q_rope,
                                k_cache=k_rope_cache,
                                v_cache=c_kv_cache,
                                qv=q_nope,
                                page_table=self.forward_metadata_spec_decode_expand.page_table,
                                cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                                cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                                cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                                max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                                softmax_scale=layer.scaling,
                                causal=False,
                                window_size=window_size,
                                softcap=layer.logit_cap,
                                k_descale=k_descale,
                                v_descale=v_descale,
                                return_softmax_lse=True,
                                num_splits=self.num_splits,
                                ver=self.fa_impl_ver,
                            )
                        )
                        o, _ = merge_state_v2_wrapper(
                            o,
                            softmax_lse.T.contiguous(),
                            o_expand,
                            softmax_lse_expand.T.contiguous(),
                        )
                    else:
                        o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # 前向解码方法，处理单令牌解码的注意力计算
    def forward_decode(
        self,
        q: torch.Tensor,  # 查询张量
        k: torch.Tensor,  # 键张量
        v: torch.Tensor,  # 值张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 前向批次
        save_kv_cache=True,  # 是否保存KV缓存
        # 用于多头潜在注意力
        q_rope: Optional[torch.Tensor] = None,  # 旋转位置编码查询
        k_rope: Optional[torch.Tensor] = None,  # 旋转位置编码键
        sinks: Optional[torch.Tensor] = None,  # 注意力汇聚点
    ) -> torch.Tensor:  # 返回输出张量
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # 使用跨所有层预计算的元数据
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # 启用投机解码时，forward_decode会在两种模式下被调用：
        # 1. DRAFT_DECODE：top_k>1时启用级联注意力
        # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # 不做-1因为在get_attention_sliding_window_size()中已经-1
        # 这里是两侧包含
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        causal = True  # 默认启用因果注意力
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        kwargs = {}  # 额外参数字典
        if sinks is not None:
            kwargs["sinks"] = sinks

        _fa_out = (
            forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if getattr(forward_batch, "_attn_output", None) is not None
            else None
        )

        k_descale, v_descale = None, None  # KV反缩放因子初始化为None
        # 仅在以下条件满足时使用KV缩放：1)显式启用fp8 KV，2)RadixAttention
        # 有对应量化方法使layer.k_scale非None，
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # 执行多头注意力

            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if layer.is_cross_attention:
                # 交叉注意力始终使用非分块逻辑
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
            elif use_local_attn:
                # 使用分块（局部）注意力批处理进行自注意力
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
            else:
                page_table = metadata.page_table
                if is_swa_layer and self.use_sliding_window_kv_pool:
                    if metadata.swa_page_table is not None:
                        page_table = metadata.swa_page_table
                    else:
                        page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                metadata.page_table
                            )
                        )
                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # 默认：单令牌自注意力
                # 可用时使用预计算的scheduler_metadata。
                # scheduler_metadata仅对非SWA、非级联解码有效。
                sched_meta = None
                if (
                    metadata.scheduler_metadata is not None
                    and not is_swa_layer
                    and not use_cascade_attn
                ):
                    sched_meta = metadata.scheduler_metadata
                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    ver=self.fa_impl_ver,
                    scheduler_metadata=sched_meta,
                    **kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            num_splits=self.num_splits,
                            ver=self.fa_impl_ver,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # 执行吸收式多潜在注意力
            kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,  # 合并状态需要softmax_lse
                num_splits=self.num_splits,
                ver=self.fa_impl_ver,
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # 初始化CUDA图状态，预分配固定大小张量
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.
        为注意力后端初始化CUDA图状态。

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs
            max_bs (int): CUDA图支持的最大批大小

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        创建固定大小的张量，在CUDA图重放期间复用以避免内存分配。
        """
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

        # 此数据用于topk==1时的普通解码和草稿解码
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }
        # 为CUDA图预分配scheduler_metadata缓冲区
        # 大小：1(信号量)+round_up(max_bs,4)*4(因果解码向量)
        if self._get_scheduler_metadata is not None and not self.use_mla:
            b_rounded = ((max_bs + 3) // 4) * 4
            self._sched_meta_buf = torch.zeros(
                1 + b_rounded * 4, dtype=torch.int32, device=self.device
            )
        else:
            self._sched_meta_buf = None

        # 仅在启用局部注意力时分配缓冲区
        # 这防止未使用局部注意力时出现OOM错误
        if self.has_local_attention:
            # 估算局部注意力元数据的最大尺寸
            max_seq_len = self.max_context_len
            page_size = self.page_size or 1
            attn_chunk_size = self.attention_chunk_size
            max_virtual_batches = max_bs * (
                (max_seq_len + attn_chunk_size - 1) // attn_chunk_size
            )
            max_pages_per_block = (attn_chunk_size + page_size - 1) // page_size

            self.decode_cuda_graph_local_attn_metadata = {
                "local_query_start_loc": torch.zeros(
                    max_virtual_batches + 1, dtype=torch.int32, device=self.device
                ),
                "local_seqused_k": torch.zeros(
                    max_virtual_batches, dtype=torch.int32, device=self.device
                ),
                "local_block_table": torch.zeros(
                    max_virtual_batches,
                    max_pages_per_block,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if self.use_sliding_window_kv_pool:
            self.decode_cuda_graph_metadata["swa_page_table"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )

        # topk>1时草稿解码前半部分元数据使用
        if self.topk > 1:
            self.draft_decode_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    step=self.topk,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            # topk>1时草稿解码后半部分元数据使用
            decode_length = self.speculative_step_id + 1
            self.draft_decode_metadata_topk_expand = {
                "cache_seqlens": torch.full(
                    (max_bs * self.topk,),
                    decode_length,
                    device=self.device,
                    dtype=torch.int32,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.arange(
                    0,
                    max_bs * self.topk * decode_length + 1,
                    step=decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.topk,
                    decode_length + 1,  # 额外一页用于最后一个不完整页
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            # "page_table_draft_decode"仅在启用投机解码时设置以节省内存
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )

            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            if self.use_sliding_window_kv_pool:
                self.target_verify_metadata["swa_page_table"] = torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.draft_extend_metadata["swa_page_table"] = torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                )

        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            self.target_verify_metadata_topk_expand = {
                "cache_seqlens": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            if self.has_swa:
                self.target_verify_metadata_topk_swa = {
                    "cache_seqlens": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "cu_seqlens_k": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "cu_seqlens_q": torch.arange(
                        0,
                        max_bs * self.speculative_num_draft_tokens + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "page_table": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        self.max_context_len,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                }

        # 仅为编码器-解码器模型分配编码器元数据
        if self.is_encoder_decoder:
            self.encoder_metadata = {
                "encoder_page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "encoder_lens_int32": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "encoder_cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
            }
        else:
            # 对于仅解码器模型，跳过encoder_metadata分配
            self.encoder_metadata = {}

    # 绑定元数据缓冲区，创建预分配缓冲区切片引用
    def _bind_metadata_buffers(
        self,
        bs: int,
        num_tokens: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        device: torch.device,
    ) -> tuple:
        """Create FlashAttentionMetadata with pre-allocated buffer slice refs.
        创建带有预分配缓冲区切片引用的FlashAttentionMetadata。

        Assigns all buffer slice references but does NOT fill data values.
        分配所有缓冲区切片引用但不填充数据值。
        Stores the new metadata object(s) in the appropriate lookup dicts.
        将新的元数据对象存储在相应的查找字典中。
        Returns (metadata, metadata_expand).
        返回(metadata, metadata_expand)。
        """
        metadata = FlashAttentionMetadata()
        metadata_expand = FlashAttentionMetadata()

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                if self.topk <= 1:
                    # 草稿解码 topk=1
                    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                        "cache_seqlens"
                    ][:bs]
                    metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.decode_cuda_graph_metadata[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.decode_cuda_graph_metadata[
                        "page_table_draft_decode"
                    ][:bs, :]
                    if self.use_sliding_window_kv_pool:
                        metadata.swa_page_table = self.decode_cuda_graph_metadata[
                            "swa_page_table"
                        ][:bs, :]
                    self.decode_cuda_graph_metadata[bs] = metadata
                else:
                    # 草稿解码 topk>1: two metadata objects
                    metadata.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_normal["cache_seqlens"][:bs]
                    )
                    metadata.max_seq_len_q = self.topk
                    metadata.cu_seqlens_q = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.draft_decode_metadata_topk_normal[
                        "page_table"
                    ][:bs, :]

                    metadata_expand.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_expand["cache_seqlens"][
                            : bs * self.topk
                        ]
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_q"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.cu_seqlens_k = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_k"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.page_table = self.draft_decode_metadata_topk_expand[
                        "page_table"
                    ][: bs * self.topk]
                    self.draft_decode_metadata_topk_normal[bs] = metadata
                    self.draft_decode_metadata_topk_expand[bs] = metadata_expand
            else:
                # 普通解码
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = self.decode_cuda_graph_metadata["cu_seqlens_k"][
                    : bs + 1
                ]
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                if self.use_sliding_window_kv_pool:
                    metadata.swa_page_table = self.decode_cuda_graph_metadata[
                        "swa_page_table"
                    ][:bs, :]
                self.decode_cuda_graph_metadata[bs] = metadata

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.cu_seqlens_q = self.target_verify_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                    : (bs + 1)
                ]
                metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]
                if self.use_sliding_window_kv_pool:
                    metadata.swa_page_table = self.target_verify_metadata[
                        "swa_page_table"
                    ][:bs, :]
                self.target_verify_metadata[bs] = metadata
            else:
                # 目标验证topk>1：两个（或带SWA三个）元数据对象
                metadata.cache_seqlens_int32 = self.target_verify_metadata_topk_normal[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.cu_seqlens_q = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_q"
                ][: bs + 1]
                metadata.cu_seqlens_k = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_k"
                ][: bs + 1]
                metadata.page_table = self.target_verify_metadata_topk_normal[
                    "page_table"
                ][:bs, :]

                metadata_expand.cache_seqlens_int32 = (
                    self.target_verify_metadata_topk_expand["cache_seqlens"][
                        : bs * self.speculative_num_draft_tokens
                    ]
                )
                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_q"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.cu_seqlens_k = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_k"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.page_table = self.target_verify_metadata_topk_expand[
                    "page_table"
                ][: bs * self.speculative_num_draft_tokens]

                self.target_verify_metadata_topk_normal[bs] = metadata
                self.target_verify_metadata_topk_expand[bs] = metadata_expand

                if self.has_swa:
                    metadata_swa = FlashAttentionMetadata()
                    metadata_swa.cache_seqlens_int32 = (
                        self.target_verify_metadata_topk_swa["cache_seqlens"][
                            : bs * self.speculative_num_draft_tokens
                        ]
                    )
                    metadata_swa.max_seq_len_q = 1
                    metadata_swa.cu_seqlens_q = self.target_verify_metadata_topk_swa[
                        "cu_seqlens_q"
                    ][: bs * self.speculative_num_draft_tokens + 1]
                    metadata_swa.cu_seqlens_k = self.target_verify_metadata_topk_swa[
                        "cu_seqlens_k"
                    ][: bs * self.speculative_num_draft_tokens + 1]
                    metadata_swa.page_table = self.target_verify_metadata_topk_swa[
                        "page_table"
                    ][: bs * self.speculative_num_draft_tokens]
                    self.target_verify_metadata_topk_swa[bs] = metadata_swa
                    metadata.swa_spec_metadata = metadata_swa

        elif forward_mode.is_draft_extend(include_v2=True):
            num_tokens_per_bs = num_tokens // bs
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.cu_seqlens_q = self.draft_extend_metadata["cu_seqlens_q"][: bs + 1]
            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]
            if self.use_sliding_window_kv_pool:
                metadata.swa_page_table = self.draft_extend_metadata["swa_page_table"][
                    :bs, :
                ]
            self.draft_extend_metadata[bs] = metadata

        if encoder_lens is not None:
            encoder_bs = encoder_lens.numel()
            metadata.encoder_lens_int32 = self.encoder_metadata["encoder_lens_int32"][
                :encoder_bs
            ]
            metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                "encoder_cu_seqlens_k"
            ][: (encoder_bs + 1)]
            metadata.encoder_page_table = self.encoder_metadata["encoder_page_table"][
                :bs, :
            ]

        return metadata, metadata_expand

    # 初始化前向元数据用于捕获CUDA图
    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize forward metadata for capturing CUDA graph.
        初始化前向元数据用于捕获CUDA图。"""
        seq_lens_cpu = seq_lens.cpu()
        self._bind_metadata_buffers(
            bs, num_tokens, encoder_lens, forward_mode, spec_info, seq_lens.device
        )

        if forward_mode.is_decode_or_idle() and spec_info is not None and self.topk > 1:
            # topk>1草稿解码：重放需要out_cache_loc但捕获时没有；
            # 直接设置forward_metadata，让实际CUDA图重放填充数据。
            self.forward_metadata = self.draft_decode_metadata_topk_normal[bs]
            self.forward_metadata_spec_decode_expand = (
                self.draft_decode_metadata_topk_expand[bs]
            )
            return

        if forward_mode.is_target_verify() and self.topk > 1:
            # topk>1目标验证：重放需要spec_info.positions和.custom_mask
            # 这些在捕获时未填充。
            self.forward_metadata = self.target_verify_metadata_topk_normal[bs]
            self.forward_metadata_spec_decode_expand = (
                self.target_verify_metadata_topk_expand[bs]
            )
            return

        self.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=None,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        if forward_mode.is_decode_or_idle() and spec_info is None:
            # 局部注意力和调度器元数据需要捕获时切片大小设置。
            # 两者都依赖上面重放已填充的数据。
            metadata = self.decode_cuda_graph_metadata[bs]
            self._maybe_update_local_attn_metadata_for_capture(metadata, bs)
            if self._sched_meta_buf is not None:
                sched = self._compute_scheduler_metadata(
                    bs,
                    max(metadata.max_seq_len_k, 1),
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_q,
                )
                if sched is not None:
                    n = sched.shape[0]
                    self._sched_meta_buf[:n] = sched
                    self._sched_meta_buf[n:] = 0
                    metadata.scheduler_metadata = self._sched_meta_buf[:n]

        if forward_mode.is_draft_extend(include_v2=True):
            # CUDA图将max_seq_len_q烘焙为常量。replay()将其设为
            # max(num_accept_tokens_cpu)，这在捕获时为None/空，
            # 回退到1。恢复正确的上限使内核
            # 在此图的所有重放中看到num_tokens_per_bs（非1）。
            self.forward_metadata.max_seq_len_q = num_tokens // bs

    # 初始化前向元数据用于重放CUDA图
    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph.
        初始化前向元数据用于重放CUDA图。"""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None
        metadata_expand = None

        if forward_mode.is_decode_or_idle():

            if spec_info is not None:
                # 草稿解码
                if self.topk <= 1:
                    # topk=1时使用普通解码元数据
                    metadata = self.decode_cuda_graph_metadata[bs]
                    max_len = seq_lens_cpu.max().item()
                    metadata.max_seq_len_k = max_len + self.speculative_step_id + 1
                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size

                    normal_decode_set_metadata(
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_k,
                        metadata.page_table,
                        self.req_to_token,
                        req_pool_indices,
                        self.decode_cuda_graph_metadata["strided_indices"],
                        max_seq_pages,
                        seq_lens,
                        self.speculative_step_id + 1,
                        self.page_size,
                        metadata.swa_page_table,
                        (
                            self.token_to_kv_pool
                            if self.use_sliding_window_kv_pool
                            else None
                        ),
                    )

                else:
                    # topk>1时需要两个特定的草稿解码元数据，然后合并状态
                    # 1. 前缀令牌的元数据前半部分
                    metadata = self.draft_decode_metadata_topk_normal[bs]
                    if self.page_size > 1:
                        # 如果page_size>1，第一次注意力处理seq_lens-last_page_lens。
                        last_page_lens = seq_lens % self.page_size
                        seq_lens = seq_lens - last_page_lens
                    metadata.cache_seqlens_int32.copy_(seq_lens)
                    # metadata.max_seq_len_q=self.topk，已在捕获时设置
                    # metadata.cu_seqlens_q已在捕获时设置
                    # metadata.cu_seqlens_k不需要

                    metadata.max_seq_len_k = seq_lens_cpu.max().item()
                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size
                    strided_indices = self.decode_cuda_graph_metadata["strided_indices"]
                    strided_indices = strided_indices[:max_seq_pages]
                    page_table = (
                        self.req_to_token[
                            req_pool_indices[:, None],  # shape [bs, 1]
                            strided_indices[None, :],  # shape [1, max_seq_pages]
                        ]
                        // self.page_size
                    )
                    metadata.page_table[:, :max_seq_pages].copy_(page_table)
                    # 2. 草稿令牌的元数据后半部分（每批令牌数=topk）
                    metadata_expand = self.draft_decode_metadata_topk_expand[bs]
                    decode_length = self.speculative_step_id + 1
                    # 形状：[批大小,步数,topk]->[批大小*topk,步数]
                    cache_loc = out_cache_loc.view(-1, self.speculative_num_steps)
                    if self.page_size > 1:
                        draft_decode_set_expand_metadata(
                            cache_seqlens_int32=metadata_expand.cache_seqlens_int32,
                            page_table=metadata_expand.page_table,
                            last_page_lens=last_page_lens,
                            decode_length=decode_length,
                            cache_loc=cache_loc,
                            topk=self.topk,
                            page_size=self.page_size,
                        )
                    else:
                        num_seqs = cache_loc.shape[0]
                        metadata_expand.page_table[:num_seqs, :decode_length].copy_(
                            cache_loc[:, :decode_length]
                        )
                # TODO：支持llama4 eagle时处理草稿解码的局部注意力元数据
            else:
                # 普通解码
                metadata = self.decode_cuda_graph_metadata[bs]
                max_len = seq_lens_cpu.max().item()
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
                metadata.max_seq_len_k = max_len

                normal_decode_set_metadata(
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table,
                    self.req_to_token,
                    req_pool_indices,
                    self.decode_cuda_graph_metadata["strided_indices"],
                    max_seq_pages,
                    seq_lens,
                    0,
                    self.page_size,
                    metadata.swa_page_table,
                    self.token_to_kv_pool if self.use_sliding_window_kv_pool else None,
                )

                self._maybe_update_local_attn_metadata_for_replay(
                    metadata,
                    bs,
                )

                # 重新计算scheduler_metadata到预分配缓冲区
                if (
                    self._sched_meta_buf is not None
                    and metadata.scheduler_metadata is not None
                ):
                    sched = self._compute_scheduler_metadata(
                        bs,
                        metadata.max_seq_len_k,
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_q,
                    )
                    if sched is not None:
                        n = sched.shape[0]
                        self._sched_meta_buf[:n] = sched
                        self._sched_meta_buf[n:] = 0

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata = self.target_verify_metadata[bs]
                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + self.speculative_num_draft_tokens)
                )

                metadata.max_seq_len_k = (
                    seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                if (
                    self.use_sliding_window_kv_pool
                    and metadata.swa_page_table is not None
                ):
                    swa_page_indices = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            page_indices
                        )
                    )
                    metadata.swa_page_table[:, :max_seq_pages].copy_(
                        swa_page_indices // self.page_size
                    )
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
            else:
                # topk>1时需要两个特定的目标验证元数据，然后合并状态
                # 1. 前缀令牌的元数据前半部分
                metadata = self.target_verify_metadata_topk_normal[bs]
                metadata.cache_seqlens_int32.copy_(seq_lens)
                # metadata.max_seq_len_q=投机草稿令牌数，已在捕获时设置
                metadata.max_seq_len_k = seq_lens_cpu.max().item()
                # metadata.cu_seqlens_q已在捕获时设置
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)

                # 2. 草稿令牌的元数据后半部分（每批令牌数=topk）
                metadata_expand = self.target_verify_metadata_topk_expand[bs]

                # metadata_expand.max_seq_len_q=1，已在捕获时设置
                # metadata_expand.cu_seqlens_q已在捕获时设置
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)

                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                # 避免提取越界的填充序列索引
                mask_extraction_indices[
                    :,
                    spec_info.positions.numel() * self.speculative_num_draft_tokens :,
                ].fill_(0)
                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )  # (批大小*草稿数, 草稿数)

                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask,
                    col_indices,
                    col_indices + self.speculative_num_draft_tokens,
                )
                _, sort_order = torch.sort(keys, dim=1)

                non_masked_page_table = (
                    self.req_to_token[req_pool_indices, :]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (批大小, 草稿数)

                metadata_expand.page_table.copy_(
                    non_masked_page_table.gather(1, sort_order)
                )
                metadata_expand.cache_seqlens_int32.copy_(mask.sum(dim=1))
                metadata_expand.cu_seqlens_k[1:].copy_(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32,
                        dim=0,
                        dtype=torch.int32,
                    )
                )
                if self.has_swa:
                    metadata_swa = self.target_verify_metadata_topk_swa[bs]
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand, metadata_swa
                    )

        elif forward_mode.is_draft_extend():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            extend_lens = spec_info.num_accept_tokens[:bs]
            if spec_info.num_accept_tokens_cpu:
                metadata.max_seq_len_q = max(spec_info.num_accept_tokens_cpu)
            else:
                metadata.max_seq_len_q = 1

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(extend_lens, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            if self.use_sliding_window_kv_pool and metadata.swa_page_table is not None:
                swa_page_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    page_indices
                )
                metadata.swa_page_table[:, :max_seq_pages].copy_(
                    swa_page_indices // self.page_size
                )
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)

        elif forward_mode.is_draft_extend_v2():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )

            extend_seq_lens_tensor = getattr(spec_info, "extend_seq_lens_tensor", None)
            extend_seq_lens_cpu = getattr(spec_info, "extend_seq_lens_cpu", None)
            if extend_seq_lens_tensor is not None:
                extend_seq_lens = extend_seq_lens_tensor.to(torch.int32)
            elif extend_seq_lens_cpu is not None:
                extend_seq_lens = torch.as_tensor(
                    extend_seq_lens_cpu,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                default_extend = getattr(
                    spec_info, "num_tokens_per_req", self.speculative_num_steps + 1
                )
                extend_seq_lens = torch.full(
                    (bs,), default_extend, dtype=torch.int32, device=device
                )
                extend_seq_lens_cpu = [default_extend] * bs

            if extend_seq_lens_cpu:
                metadata.max_seq_len_q = int(max(extend_seq_lens_cpu))
            else:
                metadata.max_seq_len_q = getattr(
                    spec_info, "num_tokens_per_req", self.speculative_num_steps + 1
                )

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            if self.use_sliding_window_kv_pool and metadata.swa_page_table is not None:
                swa_page_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    page_indices
                )
                metadata.swa_page_table[:, :max_seq_pages].copy_(
                    swa_page_indices // self.page_size
                )
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)

        if encoder_lens is not None:
            # 每请求变长编码器支持（如MossVL不同图像）。
            metadata.encoder_max_seq_len_k = int(encoder_lens.max().item())
            metadata.encoder_lens_int32[:bs].copy_(encoder_lens[:bs].to(torch.int32))
            metadata.encoder_cu_seqlens_k[1 : bs + 1].copy_(
                torch.cumsum(metadata.encoder_lens_int32[:bs], dim=0, dtype=torch.int32)
            )

            metadata.encoder_page_table[:bs, : metadata.encoder_max_seq_len_k].copy_(
                self.req_to_token[req_pool_indices, : metadata.encoder_max_seq_len_k]
            )

            # 自注意力（文本）页表：每请求偏移量=encoder_lens[i]。
            text_max = metadata.max_seq_len_k
            arange_text = torch.arange(text_max, device=req_pool_indices.device)
            text_col = encoder_lens[:bs].long().unsqueeze(1) + arange_text.unsqueeze(0)
            text_row = req_pool_indices.unsqueeze(1).expand(-1, text_max)
            metadata.page_table[:bs, :text_max].copy_(
                self.req_to_token[text_row, text_col]
            )

        self.forward_metadata = metadata  # 保存前向传播元数据
        self.forward_metadata_spec_decode_expand = metadata_expand  # 保存投机解码扩展元数据

    # 获取CUDA图序列长度填充值
    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph.
        获取CUDA图序列长度填充值。"""
        return 1  # 返回填充值1

    # 可能初始化局部注意力元数据
    def _maybe_init_local_attn_metadata(
        self, forwardbatch: ForwardBatch, metadata: FlashAttentionMetadata, device
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled.
        如果启用了分块注意力，则集中初始化local_attn_metadata。"""
        if not self.has_local_attention:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        if self.use_sliding_window_kv_pool:
            page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                metadata.page_table
            )
        else:
            page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    # 在CUDA图捕获阶段更新局部注意力元数据
    def _maybe_update_local_attn_metadata_for_capture(
        self, metadata: FlashAttentionMetadata, bs: int
    ):
        """Update local attention metadata during CUDA graph capture phase.
        在CUDA图捕获阶段更新局部注意力元数据。

        This method calculates the exact buffer sizes needed for local attention metadata
        during the CUDA graph capture phase, optimizing memory usage by creating views of
        pre-allocated buffers with exactly the sizes needed.
        此方法在CUDA图捕获阶段计算局部注意力元数据所需的精确缓冲区大小，
        通过创建预分配缓冲区精确大小的视图来优化内存使用。
        """
        if not self.has_local_attention:
            return

        seq_lens_capture = metadata.cache_seqlens_int32
        max_seq_len = int(seq_lens_capture.max().item())
        page_table_capture = metadata.page_table

        cu_seqlens_q_np = metadata.cu_seqlens_q.cpu().numpy()
        seqlens_np = seq_lens_capture.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local_np,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            page_table_capture,
            self.page_size,
        )

        # 从计算中获取精确维度
        q_len = len(cu_seqlens_q_local_np)
        k_len = len(seqlens_k_local_np)
        b0 = block_table_local_np.shape[0] if block_table_local_np.shape[0] > 0 else bs
        b1 = block_table_local_np.shape[1] if block_table_local_np.shape[1] > 0 else 1

        # 创建预分配缓冲区的精确大小视图
        # 这是关键优化 - 仅使用实际需要的内存
        local_query_start_loc = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ][:q_len]

        local_seqused_k = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"][
            :k_len
        ]

        local_block_table = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ][:b0, :b1]

        metadata.local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seqused_k=local_seqused_k,
            local_block_table=local_block_table,
            local_max_query_len=1,
            local_max_seq_len=max_seq_len,
        )

    # 在CUDA图重放前就地更新预分配的局部注意力元数据
    def _maybe_update_local_attn_metadata_for_replay(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
    ):
        """Update preallocated local attention metadata in-place before CUDA graph replay.
        在CUDA图重放前就地更新预分配的局部注意力元数据。"""
        if not self.has_local_attention:
            return

        # 访问预分配缓冲区
        local_q_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ]
        local_k_buf = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"]
        local_block_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ]
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"]

        # 创建仅处理最后一个令牌的局部注意力修改版本
        # 这模仿普通解码模式
        cu_seqlens_q = torch.arange(
            bs + 1, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        seqlens = metadata.cache_seqlens_int32[:bs]
        # 切片page_table以匹配批大小和实际序列长度
        # 这有三个重要目的：
        # 1. 确保仅处理实际批大小(bs)而非最大批大小
        # 2. 限制序列长度以防止处理填充令牌或垃圾值
        # 3. 防止块表中的零导致重放时产生垃圾输出
        #
        # 没有此切片，预分配的page_table可能包含零或无效索引
        # 超出实际序列长度，导致不正确的注意力计算
        max_seq_len = int(seqlens.max().item())
        if self.use_sliding_window_kv_pool:
            sliced_page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                metadata.page_table[:bs, :max_seq_len]
            )
        else:
            sliced_page_table = metadata.page_table[:bs, :max_seq_len]

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seqlens_np = seqlens.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            sliced_page_table,
            self.page_size,
        )

        # 转回张量
        device = local_q_buf.device
        cu_seqlens_q_local = torch.from_numpy(cu_seqlens_q_local_np).to(device)
        seqlens_k_local = torch.from_numpy(seqlens_k_local_np).to(device)
        block_table_local = block_table_local.to(device)
        # 获取大小
        q_len = cu_seqlens_q_local.shape[0]
        k_len = seqlens_k_local.shape[0]
        b0, b1 = block_table_local.shape

        # 就地更新预分配张量并将未使用空间清零
        local_q_buf[:q_len].copy_(cu_seqlens_q_local)
        local_q_buf[q_len:].fill_(0)
        local_k_buf[:k_len].copy_(seqlens_k_local)
        local_k_buf[k_len:].fill_(0)
        local_block_buf[:b0, :b1].copy_(block_table_local)
        local_block_buf[b0:, :].fill_(0)
        local_block_buf[:b0, b1:].fill_(0)

        if metadata.local_attn_metadata is not None:
            lam = metadata.local_attn_metadata
            lam.local_max_query_len = int(seqlens_q_local_np.max())
            lam.local_max_seq_len = int(seqlens_k_local_np.max())

    # 初始化滑动窗口注意力投机解码元数据
    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: Optional[FlashAttentionMetadata] = None,
    ):
        # TODO：支持SWA投机解码的page_size>1
        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        cache_seqlens_int32 = (
            metadata.cache_seqlens_int32.repeat_interleave(
                self.speculative_num_draft_tokens
            )
            + metadata_expand.cache_seqlens_int32
        )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        bs = cache_seqlens_int32.shape[0]
        page_table = (
            metadata.page_table.new_zeros(
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])
            )
            if metadata_swa is None
            else metadata_swa.page_table
        )

        page_table_a = metadata.page_table
        page_table_b = metadata_expand.page_table
        if self.use_sliding_window_kv_pool:
            page_table_a = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                page_table_a
            )
            page_table_b = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                page_table_b
            )

        prepare_swa_spec_page_table_triton(
            page_table,
            page_table_a,
            page_table_b,
            metadata.cache_seqlens_int32,
            metadata_expand.cache_seqlens_int32,
            self.speculative_num_draft_tokens,
        )

        if metadata_swa is None:
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.cu_seqlens_k = cu_seqlens_k
            metadata_swa.page_table = page_table
        else:
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)

        metadata.swa_spec_metadata = metadata_swa


@triton.jit
# 准备SWA投机页表的Triton GPU内核
def _prepare_swa_spec_page_table_kernel(
    dst_ptr,
    src_a_ptr,
    src_b_ptr,
    seq_len_a_ptr,
    seq_len_b_ptr,
    dst_stride_m,
    dst_stride_n,
    a_stride_m,
    a_stride_n,
    b_stride_m,
    b_stride_n,
    LEN_A: tl.constexpr,
    LEN_B: tl.constexpr,
    REPEAT_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx_a = pid_m // REPEAT_STEP
    idx_b = pid_m
    seq_len_a = tl.load(seq_len_a_ptr + idx_a)
    seq_len_b = tl.load(seq_len_b_ptr + idx_b)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_len = seq_len_a + seq_len_b

    if pid_n * BLOCK_N >= total_len:
        return

    mask = offs_n < total_len
    dst = dst_ptr + pid_m * dst_stride_m + offs_n * dst_stride_n

    if (pid_n + 1) * BLOCK_N < seq_len_a:
        a_ptr = src_a_ptr + idx_a * a_stride_m + offs_n * a_stride_n
        a_mask = mask & (offs_n < LEN_A)
        val = tl.load(a_ptr, mask=a_mask, other=0)
        tl.store(dst, val, mask=mask)
    elif pid_n * BLOCK_N >= seq_len_a:
        offs_b = offs_n - seq_len_a
        b_ptr = src_b_ptr + idx_b * b_stride_m + offs_b * b_stride_n
        b_mask = mask & (offs_b < LEN_B)
        val = tl.load(b_ptr, mask=b_mask, other=0)
        tl.store(dst, val, mask=mask)
    else:
        # 混合部分
        a_offs = offs_n
        a_mask = (a_offs < seq_len_a) & (a_offs < LEN_A)
        a_ptr = src_a_ptr + idx_a * a_stride_m + a_offs * a_stride_n
        a_val = tl.load(a_ptr, mask=a_mask, other=0)

        b_offs = offs_n - seq_len_a
        b_mask = (b_offs >= 0) & (b_offs < seq_len_b) & (b_offs < LEN_B)
        b_ptr = src_b_ptr + idx_b * b_stride_m + b_offs * b_stride_n
        b_val = tl.load(b_ptr, mask=b_mask, other=0)

        result = tl.where(offs_n < seq_len_a, a_val, b_val)
        tl.store(dst, result, mask=mask)


# 使用Triton准备SWA投机页表，拼接page_table和扩展page_table
def prepare_swa_spec_page_table_triton(
    page_table_dst: torch.Tensor,
    page_table_a: torch.Tensor,
    page_table_b: torch.Tensor,  # expand page table
    seq_len_a: torch.Tensor,
    seq_len_b: torch.Tensor,  # expand seq lens
    speculative_num_draft_tokens: int,
):
    # concat page_table and expand page_table by kv seq length
    bs = seq_len_a.numel()
    bs_expand = seq_len_b.numel()
    assert bs_expand == bs * speculative_num_draft_tokens

    LEN_A = page_table_a.shape[1]
    LEN_B = page_table_b.shape[1]
    LEN_OUT = LEN_A + LEN_B
    REPEAT_STEP = speculative_num_draft_tokens
    BLOCK_N = 256

    grid = (bs_expand, triton.cdiv(LEN_OUT, BLOCK_N))
    _prepare_swa_spec_page_table_kernel[grid](
        page_table_dst,
        page_table_a,
        page_table_b,
        seq_len_a,
        seq_len_b,
        page_table_dst.stride(0),
        page_table_dst.stride(1),
        page_table_a.stride(0),
        page_table_a.stride(1),
        page_table_b.stride(0),
        page_table_b.stride(1),
        LEN_A=LEN_A,
        LEN_B=LEN_B,
        REPEAT_STEP=REPEAT_STEP,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


# FlashAttention多步后端，用于投机解码的草稿工作器
class FlashAttentionMultiStepBackend:

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )
            )

    # 初始化前向传播元数据，使所有层可复用
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    # 初始化CUDA图状态，预分配固定大小张量
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    # 初始化前向元数据用于捕获CUDA图
    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    # 初始化前向元数据用于重放CUDA图
    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):
            # TODO：增量更新后续步骤的元数据，
            # 使其不需要从头重新计算所有内容。
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )


@triton.jit
# 融合元数据通用Triton内核，替换4-5个顺序CUDA内核
def _fused_metadata_kernel_general(
    # 输入张量
    seq_lens,
    seq_lens_stride_0,
    req_to_token,
    req_to_token_stride_0,
    req_to_token_stride_1,
    req_pool_indices,
    req_pool_indices_stride_0,
    # 输出缓冲区
    cache_seqlens_int32,
    cache_seqlens_int32_stride_0,
    cu_seqlens_k,
    cu_seqlens_k_stride_0,
    page_table,
    page_table_stride_0,
    page_table_stride_1,
    swa_page_table,
    swa_page_table_stride_0,
    swa_page_table_stride_1,
    full_to_swa_mapping,
    full_to_swa_mapping_stride_0,
    # 标量参数
    B,
    max_seq_pages,
    page_size: tl.constexpr,
    seq_len_delta: tl.constexpr,
    use_swa: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)  # 批次索引
    pid_c = tl.program_id(1)  # 列块索引

    # 1. Prefix sum (only one block does it)
    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + seq_len_delta).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)

    # 2. Gather for this batch and column chunk
    if max_seq_pages == 0:
        return

    i = pid_b
    # 加载此批次的行索引（块中所有线程有相同的i）
    row_idx = tl.load(req_pool_indices + i * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_start = pid_c * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < max_seq_pages

    # 计算源张量中的列索引（令牌偏移量）
    if page_size == 1:
        col_idx = col_offsets
    else:
        col_idx = col_offsets << SHIFT  # 对2的幂比乘法更快

    # 从req_to_token加载页索引
    rt_offsets = row_offset + col_idx * req_to_token_stride_1
    page_index = tl.load(
        req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg"
    )

    # 计算page_table
    if page_size == 1:
        page_table_val = page_index
    else:
        page_table_val = page_index >> SHIFT

    # 存储到page_table
    pt_offsets = i * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_table_val, mask=mask, cache_modifier=".cg")

    if use_swa:
        swa_slot = tl.load(
            full_to_swa_mapping + page_index * full_to_swa_mapping_stride_0,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        )
        if page_size == 1:
            swa_val = swa_slot
        else:
            swa_val = swa_slot >> SHIFT
        swa_offsets = (
            i * swa_page_table_stride_0 + col_offsets * swa_page_table_stride_1
        )
        tl.store(swa_page_table + swa_offsets, swa_val, mask=mask, cache_modifier=".cg")


@triton.jit
# 页大小为1且无SWA的融合元数据专用Triton内核
def _fused_metadata_kernel_ps1_no_swa(
    # 输入张量
    seq_lens,
    seq_lens_stride_0,
    req_to_token,
    req_to_token_stride_0,
    req_to_token_stride_1,
    req_pool_indices,
    req_pool_indices_stride_0,
    # 输出缓冲区
    cache_seqlens_int32,
    cache_seqlens_int32_stride_0,
    cu_seqlens_k,
    cu_seqlens_k_stride_0,
    page_table,
    page_table_stride_0,
    page_table_stride_1,
    # 标量参数
    B,
    max_seq_pages,
    seq_len_delta: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)  # 批次索引
    pid_c = tl.program_id(1)  # 列块索引

    # 1. Prefix sum (only one block does it)
    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + seq_len_delta).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)

    # 2. Gather for this batch and column chunk
    if max_seq_pages == 0:
        return

    i = pid_b
    # 加载此批次的行索引（块中所有线程有相同的i）
    row_idx = tl.load(req_pool_indices + i * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_start = pid_c * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < max_seq_pages

    # page_size=1：col_idx=col_offsets
    rt_offsets = row_offset + col_offsets * req_to_token_stride_1
    page_index = tl.load(
        req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg"
    )

    # page_table=page_index//1=page_index
    pt_offsets = i * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_index, mask=mask, cache_modifier=".cg")


# Fused Triton kernel implementation
# 融合Triton实现的普通解码元数据设置
def normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
    swa_page_table: Optional[torch.Tensor] = None,
    token_to_kv_pool: Optional[SWAKVPool] = None,
):
    """
    Fused Triton implementation that replaces 4-5 sequential CUDA kernels with 1-2 kernels
    融合Triton实现，将4-5个顺序CUDA内核替换为1-2个内核:
      1. cache_seqlens = seq_lens + seq_len_delta (int64→int32 cast)
      2. cu_seqlens_k = cumsum(cache_seqlens) (prefix-sum)
      3. page_indices = req_to_token[pool_idx, stride_idx] (2-D gather)
      4. page_table = page_indices // page_size (floor-divide)
      5. (optional) swa_page_table for sliding window attention

    Achieves ~5.2x speedup on H200 hardware for typical decode workloads.
    """
    assert (
        page_size > 0 and (page_size & (page_size - 1)) == 0
    ), f"page_size must be a power of two, got {page_size}"

    batch_size = cache_seqlens_int32.shape[0]
    device = seq_lens.device

    # 确保连续内存布局以高效Triton访问
    seq_lens = seq_lens.contiguous()
    req_to_token = req_to_token.contiguous()
    req_pool_indices = req_pool_indices.contiguous()

    # 准备张量步幅
    seq_lens_stride_0 = seq_lens.stride(0)
    req_to_token_stride_0 = req_to_token.stride(0)
    req_to_token_stride_1 = req_to_token.stride(1)
    req_pool_indices_stride_0 = req_pool_indices.stride(0)
    cache_seqlens_int32_stride_0 = cache_seqlens_int32.stride(0)
    cu_seqlens_k_stride_0 = cu_seqlens_k.stride(0)
    page_table_stride_0 = page_table.stride(0)
    page_table_stride_1 = page_table.stride(1)

    # 检查是否应使用page_size=1无SWA的专用快速路径
    use_swa = swa_page_table is not None and token_to_kv_pool is not None

    if page_size == 1 and not use_swa:
        # 常见情况（page_size=1，无SWA）的专用内核
        BLOCK_COLS = 256
        if max_seq_pages == 0:
            grid = (1, 1)
        else:
            num_blocks_j = triton.cdiv(max_seq_pages, BLOCK_COLS)
            grid = (batch_size, num_blocks_j)

        _fused_metadata_kernel_ps1_no_swa[grid](
            seq_lens,
            seq_lens_stride_0,
            req_to_token,
            req_to_token_stride_0,
            req_to_token_stride_1,
            req_pool_indices,
            req_pool_indices_stride_0,
            cache_seqlens_int32,
            cache_seqlens_int32_stride_0,
            cu_seqlens_k,
            cu_seqlens_k_stride_0,
            page_table,
            page_table_stride_0,
            page_table_stride_1,
            batch_size,
            max_seq_pages,
            seq_len_delta,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=8,
            num_stages=3,
        )
    else:
        # page_size>1或SWA情况的通用内核
        # SWA参数
        if use_swa:
            assert isinstance(token_to_kv_pool, SWAKVPool)
            swa_page_table = swa_page_table.contiguous()
            swa_page_table_stride_0 = swa_page_table.stride(0)
            swa_page_table_stride_1 = swa_page_table.stride(1)
            # Extract the full_to_swa_index_mapping from token_to_kv_pool
            full_to_swa_mapping = (
                token_to_kv_pool.full_to_swa_index_mapping.contiguous()
            )
            full_to_swa_mapping_stride_0 = full_to_swa_mapping.stride(0)
        else:
            # 虚拟张量（未使用）
            swa_page_table = torch.empty(0, dtype=torch.int32, device=device)
            swa_page_table_stride_0 = 0
            swa_page_table_stride_1 = 0
            full_to_swa_mapping = torch.empty(0, dtype=torch.int32, device=device)
            full_to_swa_mapping_stride_0 = 0

        # 内核配置
        BLOCK_COLS = 128
        shift = (page_size).bit_length() - 1 if page_size > 1 else 0

        if max_seq_pages == 0:
            grid = (1, 1)
        else:
            num_blocks_j = triton.cdiv(max_seq_pages, BLOCK_COLS)
            grid = (batch_size, num_blocks_j)

        _fused_metadata_kernel_general[grid](
            seq_lens,
            seq_lens_stride_0,
            req_to_token,
            req_to_token_stride_0,
            req_to_token_stride_1,
            req_pool_indices,
            req_pool_indices_stride_0,
            cache_seqlens_int32,
            cache_seqlens_int32_stride_0,
            cu_seqlens_k,
            cu_seqlens_k_stride_0,
            page_table,
            page_table_stride_0,
            page_table_stride_1,
            swa_page_table,
            swa_page_table_stride_0,
            swa_page_table_stride_1,
            full_to_swa_mapping,
            full_to_swa_mapping_stride_0,
            batch_size,
            max_seq_pages,
            page_size,
            seq_len_delta,
            use_swa,
            shift,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=4,
            num_stages=3,
        )


@torch.compile(dynamic=True, backend=get_compiler_backend())
# 设置草稿解码扩展元数据
def draft_decode_set_expand_metadata(
    cache_seqlens_int32: torch.Tensor,  # Modifies
    page_table: torch.Tensor,  # Modifies
    last_page_lens: torch.Tensor,
    decode_length: int,
    cache_loc: torch.Tensor,
    topk: int,
    page_size: int,
):
    expanded_last_page_lens = last_page_lens.repeat_interleave(topk)
    cache_seqlens_int32.copy_(decode_length + expanded_last_page_lens)
    cache_loc = (cache_loc // page_size).to(torch.int32)
    if cache_loc.dim() == 1:
        cache_loc = cache_loc.unsqueeze(0)
    # 向量化torch.unique_consecutive：跟踪值变化点然后scatter
    mask = torch.ones_like(cache_loc, dtype=torch.bool)
    mask[:, 1:] = cache_loc[:, 1:] != cache_loc[:, :-1]
    positions = mask.cumsum(dim=1) - 1
    num_seqs = cache_loc.shape[0]
    page_table[:num_seqs, :].scatter_(1, positions, cache_loc)


# Copied from:
# https://github.com/houseroad/vllm/blob/4e45bfcaf928bdb9bd952b4ac922a3c205589ae8/vllm/v1/attention/backends/flash_attn.py
#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
# 将序列分割为局部注意力虚拟批次
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
    local attention blocks, where each block is passed to the attention kernel
    as an independent local ("virtual") batch item.
    将序列分割为局部注意力块，每个块作为独立的局部（"虚拟"）批次项传入注意力内核。

    Args:
        attn_chunk_size: Size of local attention chunks / 局部注意力块大小
        query_start_loc_np: Cumulative sum of query lengths (numpy array) / 查询长度的累计和
        seq_lens_np: Sequence lengths (numpy array) / 序列长度
        block_table: Block table for KV cache / KV缓存的块表
        page_size: Size of each page in the KV cache / KV缓存中每页的大小

    Returns:
        seqlens_q_local: Query sequence lengths for local attention / 局部注意力的查询序列长度
        cu_seqlens_q_local: Cumulative sum of query sequence lengths / 查询序列长度的累计和
        seqlens_k_local: Key sequence lengths for local attention / 局部注意力的键序列长度
        block_table_local: Block table for local attention / 局部注意力的块表
    """
    # 根据实际序列长度调整attention_chunk_size
    # 以避免索引越界错误
    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    # 确保effective_chunk_size可被page_size整除
    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    if effective_chunk_size < page_size:
        effective_chunk_size = page_size
    attn_chunk_size = effective_chunk_size

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # 处理从局部注意力块中间开始的情况，
    #  假设q_seqlens>0（对所有元素），对每个批次索引计算
    #  不在第一个局部注意力块中的令牌数，以及
    #  然后对剩余部分使用cdiv。
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # 知道局部块数后可计算请求跨度
    #  对每个批次索引，确定"虚拟"请求数
    #  ，
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # 首先获取批量arange。（如[2,4,2]->[0,1,0,1,2,3,0,1]）
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange步骤1。[2,4,2]->[2,6,8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange步骤2。[2,6,8]->[0,0,2,2,2,2,6,6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange步骤3。[0,1,0,1,2,3,0,1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # 同时计算反向arange（如[1,0,3,2,1,0,1,0]）
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # 然后计算seqlens_q_local，处理
    #  首尾块可能不完整的情况
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # 设置第一个块，因为可能是部分块
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # 设置剩余块
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # 从q_seqlens转换为cu_seqlens_q
    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)

    # 计算seqlens_k_local，
    #  基本上每个批次的最后一个块之外都是完整的局部注意力块
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // page_size

    assert attn_chunk_size % page_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by page_size {page_size}"
    )
    pages_per_local_batch = attn_chunk_size // page_size

    # 为局部注意力块创建block_table
    # For out example if we have a block-table like (assuming page_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)
    # 确保block_indices不超过block_table维度
    # 这是防止索引越界错误的关键安全检查
    # 当处理大序列（>8192令牌）或block_table
    # 维度小于完整注意力块大小所需时。
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # 注意：该PR导致使用numpy数组索引torch张量时
    # 出现性能退化。作为变通方法，先将numpy数组转为
    # torch张量，可恢复性能。
    # tensor first, which recovers perf.
    batch_indices_torch = torch.from_numpy(batch_indices)
    block_indices_torch = torch.from_numpy(block_indices)
    block_table_local = block_table[batch_indices_torch, block_indices_torch].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


# 向上取整除法
def cdiv(a: int, b: int) -> int:
    """Ceiling division. 向上取整除法。"""
    return -(a // -b)


# TODO(hebiao064): remove this once we have a better way to handle the merge_state_v2 torch.compile issue
@torch._dynamo.disable()
# merge_state_v2的包装函数，禁用torch.compile以避免问题
def merge_state_v2_wrapper(o, s_a, o_exp, s_b):
    return merge_state_v2(o, s_a, o_exp, s_b)
