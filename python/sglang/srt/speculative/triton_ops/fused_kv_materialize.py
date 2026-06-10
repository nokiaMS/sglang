# 融合KV物化Triton核模块。
# 实现了融合的RMSNorm + RoPE物化操作，用于DFlash KV缓存。
# 将KV投影（cuBLAS）+ RMSNorm + RoPE（Triton）组合在一起，
# 然后通过池管理的KV写入操作完成物化。
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
"""Fused Triton kernel for DFlash KV materialization.

Combines: KV projection (cuBLAS) + RMSNorm + RoPE (Triton), then pool-managed KV writes.
"""

from typing import Callable, List  # 导入类型提示

import torch  # 导入PyTorch
import triton  # 导入Triton
import triton.language as tl  # 导入Triton语言


@triton.jit
def _fused_norm_rope_kernel(
    kv_ptr,  # [total_ctx, kv_size * 2]  # KV投影输出指针
    k_norm_weight_ptr,  # [head_dim]  # K归一化权重指针
    cos_sin_cache_ptr,  # [max_pos, rotary_dim]  # cos/sin缓存指针
    positions_ptr,  # [total_ctx]  # 位置指针
    k_out_ptr,  # [total_ctx, num_kv_heads, head_dim]  # K输出指针
    v_out_ptr,  # [total_ctx, num_kv_heads, head_dim]  # V输出指针
    kv_stride_ctx,  # KV上下文步长
    cos_sin_stride_pos,  # cos/sin位置步长
    k_out_stride_ctx,  # K输出上下文步长
    k_out_stride_head,  # K输出头步长
    v_out_stride_ctx,  # V输出上下文步长
    v_out_stride_head,  # V输出头步长
    total_ctx,  # 总上下文数
    num_kv_heads: tl.constexpr,  # KV头数常量
    head_dim: tl.constexpr,  # 头维度常量
    kv_size: tl.constexpr,  # KV大小常量
    rotary_dim: tl.constexpr,  # 旋转维度常量
    half_rotary_dim: tl.constexpr,  # 半旋转维度常量
    eps: tl.constexpr,  # epsilon常量
    BLOCK_HD: tl.constexpr,  # 头维度块大小常量
):
    """融合RMSNorm(K) + RoPE(K)物化核函数。网格：(total_ctx, num_kv_heads)。"""
    """Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads)."""
    ctx_id = tl.program_id(0)  # 获取上下文ID
    head_id = tl.program_id(1)  # 获取头ID
    if ctx_id >= total_ctx:  # 如果超出范围
        return  # 返回

    # Load metadata
    position = tl.load(positions_ptr + ctx_id)  # 加载位置

    # Compute base pointers
    kv_base = kv_ptr + ctx_id * kv_stride_ctx  # KV基指针
    k_base = kv_base + head_id * head_dim  # K基指针
    v_base = kv_base + kv_size + head_id * head_dim  # V基指针
    k_write = k_out_ptr + ctx_id * k_out_stride_ctx + head_id * k_out_stride_head  # K写入指针
    v_write = v_out_ptr + ctx_id * v_out_stride_ctx + head_id * v_out_stride_head  # V写入指针

    # Load K and V
    offs = tl.arange(0, BLOCK_HD)  # 头维度偏移量
    mask_hd = offs < head_dim  # 头维度掩码
    mask_half = offs < half_rotary_dim  # 半旋转维度掩码

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)  # 加载原始K
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)  # 加载原始V

    # RMSNorm on K
    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)  # 计算RMS逆
    norm_w = tl.load(k_norm_weight_ptr + offs, mask=mask_hd, other=1.0).to(tl.float32)  # 加载归一化权重
    k_normed = k_raw * inv_rms * norm_w  # 应用RMSNorm

    # RoPE (neox style): k_first, k_second -> rotated
    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos  # cos/sin基指针
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)  # 加载cos值
    sin_v = tl.load(  # 加载sin值
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    # Extract first/second halves of K for rotation
    k_first = tl.where(mask_half, k_normed, 0.0)  # 提取K的前半部分
    k_second_raw = tl.load(  # 加载K的后半部分原始值
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(  # 加载后半部分归一化权重
        k_norm_weight_ptr + half_rotary_dim + offs, mask=mask_half, other=1.0
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second  # 应用RMSNorm到后半部分

    # Apply rotation
    k_rot_first = k_first * cos_v - k_second * sin_v  # 旋转前半部分
    k_rot_second = k_second * cos_v + k_first * sin_v  # 旋转后半部分

    # Store V (no transform)
    tl.store(v_write + offs, v_raw, mask=mask_hd)  # 存储V（无变换）

    # Store K: rotated halves + pass-through
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)  # 存储旋转后的前半部分
    tl.store(  # 存储旋转后的后半部分
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)  # 不旋转部分的掩码
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)  # 存储不旋转部分


def _fused_norm_rope(
    kv: torch.Tensor,  # [total_ctx, kv_size*2]  # KV投影输出
    k_norm_weight: torch.Tensor,  # [head_dim]  # K归一化权重
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]  # cos/sin缓存
    positions: torch.Tensor,  # [total_ctx]  # 位置
    num_kv_heads: int,  # KV头数
    head_dim: int,  # 头维度
    rotary_dim: int,  # 旋转维度
    eps: float = 1e-6,  # epsilon值
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合RMSNorm + RoPE物化函数，处理单层的KV投影输出。"""
    """Fused RMSNorm + RoPE materialization for a single layer."""
    total_ctx = kv.shape[0]  # 获取总上下文数
    if total_ctx == 0:  # 如果为空
        empty = torch.empty(  # 创建空张量
            (0, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        return empty, empty  # 返回空K和V

    kv_size = num_kv_heads * head_dim  # 计算KV大小
    if kv.shape[1] != kv_size * 2:  # 检查形状
        raise ValueError(  # 形状不匹配抛出异常
            "Invalid fused KV projection shape: "
            f"got {tuple(kv.shape)}, expected second dim {kv_size * 2}."
        )
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:  # 检查旋转维度
        raise ValueError(  # 旋转维度无效抛出异常
            "Invalid fused KV rotary/head dim pair: "
            f"rotary_dim={rotary_dim}, head_dim={head_dim}."
        )

    half_rotary_dim = rotary_dim // 2  # 计算半旋转维度
    BLOCK_HD = triton.next_power_of_2(head_dim)  # 计算块大小（2的幂次）

    # Ensure int64 for indexing
    if positions.device != kv.device:  # 如果位置不在同一设备
        positions = positions.to(device=kv.device, dtype=torch.int64)  # 转移到KV设备
    elif positions.dtype != torch.int64:  # 如果位置不是int64
        positions = positions.to(torch.int64)  # 转换为int64

    k_out = torch.empty(  # 创建K输出张量
        (total_ctx, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
    )
    v_out = torch.empty_like(k_out)  # 创建V输出张量

    _fused_norm_rope_kernel[(total_ctx, num_kv_heads)](  # 启动核函数
        kv,  # KV投影输出
        k_norm_weight,  # K归一化权重
        cos_sin_cache,  # cos/sin缓存
        positions,  # 位置
        k_out,  # K输出
        v_out,  # V输出
        kv.stride(0),  # KV上下文步长
        cos_sin_cache.stride(0),  # cos/sin位置步长
        k_out.stride(0),  # K输出上下文步长
        k_out.stride(1),  # K输出头步长
        v_out.stride(0),  # V输出上下文步长
        v_out.stride(1),  # V输出头步长
        total_ctx,  # 总上下文数
        num_kv_heads,  # KV头数
        head_dim,  # 头维度
        kv_size,  # KV大小
        rotary_dim,  # 旋转维度
        half_rotary_dim,  # 半旋转维度
        eps,  # epsilon
        BLOCK_HD,  # 块大小
    )
    return k_out, v_out  # 返回K和V输出


class FusedKVMaterializeHelper:
    """融合KV物化辅助类，使用批量投影和Triton核。
    使用torch.einsum进行跨层的批量KV投影，
    然后使用Triton核进行融合的RMSNorm + RoPE物化。
    """
    """Fused KV materialization helper using batched projection.

    Uses torch.einsum for batched KV projection across all layers,
    then a Triton kernel for fused RMSNorm + RoPE materialization per layer.
    """

    def __init__(
        self,
        layers: List,  # 层列表
        rotary_emb,  # 旋转嵌入
        num_kv_heads: int,  # KV头数
        head_dim: int,  # 头维度
        device: torch.device,  # 设备
    ):
        """初始化融合KV物化辅助类，提取并堆叠各层权重。"""
        self.num_kv_heads = num_kv_heads  # KV头数
        self.head_dim = head_dim  # 头维度
        self.rotary_emb = rotary_emb  # 旋转嵌入
        self.n_layers = len(layers)  # 层数
        self.device = device  # 设备

        self.rotary_dim = int(getattr(rotary_emb, "rotary_dim", head_dim))  # 旋转维度
        self.is_neox_style = bool(getattr(rotary_emb, "is_neox_style", True))  # 是否为neox风格

        if not self.is_neox_style:  # 如果不是neox风格
            raise NotImplementedError("Only neox-style RoPE is supported.")  # 不支持
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:  # 检查旋转维度
            raise ValueError(  # 旋转维度无效
                "Invalid fused KV rotary/head dim pair: "
                f"rotary_dim={self.rotary_dim}, head_dim={self.head_dim}."
            )

        # Pre-extract and stack weights for batched projection.
        kv_weights = []  # KV权重列表
        self.k_norm_weights = []  # K归一化权重列表
        self.eps_values = []  # epsilon值列表

        for layer_id, layer in enumerate(layers):  # 遍历每层
            attn = layer.self_attn  # 获取自注意力层
            if int(attn.num_kv_heads) != self.num_kv_heads:  # 检查KV头数一致性
                raise ValueError(
                    "num_kv_heads mismatch across layers for fused KV path: "
                    f"expected {self.num_kv_heads}, got {int(attn.num_kv_heads)} at layer {layer_id}."
                )
            if int(attn.head_dim) != self.head_dim:  # 检查头维度一致性
                raise ValueError(
                    "head_dim mismatch across layers for fused KV path: "
                    f"expected {self.head_dim}, got {int(attn.head_dim)} at layer {layer_id}."
                )
            layer_rotary_dim = int(  # 获取层旋转维度
                getattr(attn.rotary_emb, "rotary_dim", self.head_dim)
            )
            layer_is_neox = bool(getattr(attn.rotary_emb, "is_neox_style", True))  # 获取层neox风格
            if (  # 检查RoPE配置一致性
                layer_rotary_dim != self.rotary_dim
                or layer_is_neox != self.is_neox_style
            ):
                raise ValueError(
                    "RoPE config mismatch across layers for fused KV path: "
                    f"expected (rotary_dim={self.rotary_dim}, neox={self.is_neox_style}), "
                    f"got (rotary_dim={layer_rotary_dim}, neox={layer_is_neox}) at layer {layer_id}."
                )

            # Extract KV portion of QKV weight
            qkv_w = attn.qkv_proj.weight  # 获取QKV投影权重
            kv_weight = qkv_w[attn.q_size : attn.q_size + 2 * attn.kv_size]  # 提取KV部分
            kv_weights.append(kv_weight)  # 添加到列表
            self.k_norm_weights.append(attn.k_norm.weight)  # 添加K归一化权重
            self.eps_values.append(attn.k_norm.variance_epsilon)  # 添加epsilon值

        # Stack for batched einsum: [n_layers, kv_size*2, hidden_size]
        self.batched_kv_weight = torch.stack(kv_weights)  # 堆叠KV权重

    def materialize(
        self,
        ctx_hidden: torch.Tensor,  # 上下文隐藏状态
        positions: torch.Tensor,  # 位置
        write_layer_kv: Callable[[int, torch.Tensor, torch.Tensor], None],  # 写入层KV的回调
    ) -> None:
        """物化所有层的KV缓存，使用批量投影和融合归一化/旋转。"""
        """Materialize KV cache for all layers using batched projection."""
        total_ctx = ctx_hidden.shape[0]  # 获取总上下文数
        if total_ctx == 0:  # 如果为空
            return  # 返回

        if positions.ndim != 1:  # 如果位置不是1维
            positions = positions.reshape(-1)  # 重塑为1维
        if positions.numel() != total_ctx:  # 检查位置数量
            raise ValueError(
                "positions must match ctx_hidden token count for fused KV materialization: "
                f"positions={positions.numel()}, total_ctx={total_ctx}."
            )

        max_position = int(positions.max().item())  # 获取最大位置
        ensure_cos_sin_cache_length = getattr(  # 获取确保缓存长度的方法
            self.rotary_emb, "_ensure_cos_sin_cache_length", None
        )
        if callable(ensure_cos_sin_cache_length):  # 如果方法可调用
            ensure_cos_sin_cache_length(max_position)  # 确保缓存长度足够

        cos_sin_cache = self.rotary_emb.cos_sin_cache  # 获取cos/sin缓存
        if max_position >= int(cos_sin_cache.shape[0]):  # 检查缓存是否足够
            raise RuntimeError(
                "RoPE cos/sin cache is too short for fused KV materialization: "
                f"max_position={max_position}, cache_len={int(cos_sin_cache.shape[0])}."
            )
        if cos_sin_cache.device != ctx_hidden.device:  # 检查设备一致性
            cos_sin_cache = cos_sin_cache.to(ctx_hidden.device)  # 转移到正确设备

        # Batched KV projection: [n_layers, total_ctx, kv_size*2]
        kv_all = torch.einsum("th,loh->lto", ctx_hidden, self.batched_kv_weight)  # 批量KV投影

        # Per-layer fused norm/RoPE/materialize, then delegate writes to the KV pool.
        for layer_id in range(self.n_layers):  # 遍历每层
            cache_k, cache_v = _fused_norm_rope(  # 融合归一化和旋转
                kv_all[layer_id],  # 当前层KV投影
                self.k_norm_weights[layer_id],  # 当前层K归一化权重
                cos_sin_cache,  # cos/sin缓存
                positions,  # 位置
                self.num_kv_heads,  # KV头数
                self.head_dim,  # 头维度
                self.rotary_dim,  # 旋转维度
                self.eps_values[layer_id],  # 当前层epsilon
            )
            write_layer_kv(layer_id, cache_k, cache_v)  # 写入层KV缓存
