# 文件说明：__init__.py - 基于Triton的稀疏注意力解码内核包初始化模块，提供DeepSeek V4的Triton实现
"""
Triton-based sparse attention decode kernels for DeepSeek V4. # 基于Triton的DeepSeek V4稀疏注意力解码内核

This package provides an alternative to the tilelang implementation, # 本包提供了tilelang实现的替代方案
controlled by the environment variable SGLANG_HACK_FLASHMLA_BACKEND=triton. # 通过环境变量 SGLANG_HACK_FLASHMLA_BACKEND=triton 控制
"""

from typing import Optional, Tuple  # 导入类型提示 # import type hints

import torch  # 导入PyTorch # import PyTorch

from sglang.srt.layers.attention.nsa.triton_decode.triton_mla_kernels_decode_optimized import (  # 从优化版Triton MLA解码内核导入 # import from optimized Triton MLA decode kernel
    triton_sparse_attn_decode,  # 稀疏注意力解码函数 # sparse attention decode function
)


class _KVScopeAdapter:  # KV范围适配器类，为Triton内核提供kv_scope接口 # KV scope adapter class, provides kv_scope interface for Triton kernels
    """Lightweight adapter providing the kv_scope interface expected by # 轻量级适配器，提供以下接口
    ``triton_sparse_attn_decode``. # 供 triton_sparse_attn_decode 使用的 kv_scope 接口

    The Triton kernels access four fields: # Triton内核访问以下四个字段：
      * ``blocked_k_quantized`` – the raw FP8 KV cache tensor. # 原始FP8 KV缓存张量
      * ``blocked_k``          – only ``blocked_k.shape[1]`` (block size) # 仅读取 blocked_k.shape[1]（块大小）
                                   is read, so we reuse the same tensor. # 因此复用同一个张量
      * ``indices_in_kvcache`` – sparse top-k page indices. # 稀疏top-k页索引
      * ``topk_length``        – valid length per batch element. # 每个批次元素的有效长度
    """

    __slots__ = [  # 定义槽位以节省内存 # define slots to save memory
        "blocked_k",  # 块状K张量 # blocked K tensor
        "blocked_k_quantized",  # 量化后的块状K张量 # quantized blocked K tensor
        "indices_in_kvcache",  # KV缓存中的索引 # indices in KV cache
        "topk_length",  # topk有效长度 # valid topk length
    ]

    def __init__(  # 初始化方法 # initialization method
        self,
        k_cache: torch.Tensor,  # K缓存张量 # K cache tensor
        indices: torch.Tensor,  # 索引张量 # indices tensor
        topk_length: Optional[torch.Tensor],  # topk长度（可选） # topk length (optional)
    ):
        self.blocked_k_quantized = k_cache  # 量化K缓存指向k_cache # quantized K cache points to k_cache
        self.blocked_k = k_cache  # 块状K也指向k_cache（复用同一张量） # blocked K also points to k_cache (reuse same tensor)
        self.indices_in_kvcache = indices  # 设置KV缓存索引 # set KV cache indices
        self.topk_length = topk_length  # 设置topk有效长度 # set topk valid length


def triton_fp8_attention_fwd(  # Triton FP8注意力前向传播函数 # Triton FP8 attention forward function
    q: torch.Tensor,  # 查询张量 # query tensor
    k_cache: torch.Tensor,  # K缓存张量 # K cache tensor
    head_dim_v: int,  # V的头部维度 # head dimension of V
    softmax_scale: float,  # softmax缩放因子 # softmax scale factor
    indices: torch.Tensor,  # 稀疏索引张量 # sparse indices tensor
    attn_sink: Optional[torch.Tensor] = None,  # 注意力汇（可选） # attention sink (optional)
    extra_k_cache: Optional[torch.Tensor] = None,  # 额外K缓存（可选） # extra K cache (optional)
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,  # 额外KV缓存索引（可选） # extra KV cache indices (optional)
    topk_length: Optional[torch.Tensor] = None,  # topk有效长度（可选） # topk valid length (optional)
    extra_topk_length: Optional[torch.Tensor] = None,  # 额外topk长度（可选） # extra topk length (optional)
    **_unused,  # 未使用的关键字参数（静默忽略） # unused keyword args (silently ignored)
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回输出和lse的元组 # return tuple of output and lse
    """Sparse MLA decode via Triton kernels. # 通过Triton内核进行稀疏MLA解码

    Accepts the same ``**kwargs`` dict that the caller builds for # 接受与调用方为以下函数构建的相同 **kwargs 字典
    ``flash_mla_with_kvcache`` / ``dpsk_v4_fp8_attention_fwd``, but only # flash_mla_with_kvcache / dpsk_v4_fp8_attention_fwd
    uses the subset of arguments relevant to the Triton implementation. # 但仅使用与Triton实现相关的参数子集
    Unused keys (``block_table``, ``cache_seqlens``, # 未使用的键（block_table, cache_seqlens,
    ``tile_scheduler_metadata``, ``num_splits``, ``causal``, # tile_scheduler_metadata, num_splits, causal,
    ``is_fp8_kvcache``) are silently ignored via ``**_unused``. # is_fp8_kvcache）通过 **_unused 静默忽略

    Returns: # 返回值：
        ``(output, lse)`` where *output* has shape # (output, lse)，其中 output 的形状为
        ``[batch, seq_len, num_heads, head_dim_v]`` and *lse* has shape # [batch, seq_len, num_heads, head_dim_v]，lse 的形状为
        ``[batch, seq_len, num_heads]``. # [batch, seq_len, num_heads]
    """
    kv_scope = _KVScopeAdapter(k_cache, indices, topk_length)  # 创建KV范围适配器 # create KV scope adapter

    extra_kv_scope = None  # 初始化额外KV范围为None # initialize extra KV scope to None
    if extra_k_cache is not None:  # 如果存在额外K缓存 # if extra K cache exists
        extra_kv_scope = _KVScopeAdapter(  # 为额外K缓存创建适配器 # create adapter for extra K cache
            extra_k_cache,  # 额外K缓存张量 # extra K cache tensor
            extra_indices_in_kvcache,  # 额外KV缓存索引 # extra KV cache indices
            extra_topk_length,  # 额外topk长度 # extra topk length
        )

    output, lse = triton_sparse_attn_decode(  # 调用Triton稀疏注意力解码内核 # call Triton sparse attention decode kernel
        q=q,  # 查询张量 # query tensor
        kv_scope=kv_scope,  # KV范围 # KV scope
        extra_kv_scope=extra_kv_scope,  # 额外KV范围 # extra KV scope
        sm_scale=softmax_scale,  # softmax缩放 # softmax scale
        d_v=head_dim_v,  # V的头部维度 # head dimension of V
        attn_sink=attn_sink,  # 注意力汇 # attention sink
    )

    # Triton kernel returns lse as (b, h_q, s_q); transpose to # Triton内核返回的lse形状为(b, h_q, s_q)；转置为
    # (b, s_q, h_q) to match the tilelang / flash_mla convention. # (b, s_q, h_q) 以匹配 tilelang / flash_mla 的约定
    lse = lse.transpose(1, 2)  # 转置lse的第1和第2维 # transpose lse dims 1 and 2

    return output, lse  # 返回输出和lse # return output and lse
