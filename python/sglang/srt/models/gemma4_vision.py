# Gemma4视觉编码器实现
# 本文件实现了Gemma4视觉模型的完整视觉编码器，包括：
# 2D多维旋转位置编码(RoPE)、视觉注意力、视觉MLP、
# 编码器层、视觉Transformer、补丁嵌入器、池化器和
# 顶层视觉编码器，支持张量并行和量化。

# Copyright 2025 SGLang Team
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
from __future__ import annotations  # 启用延迟类型注解求值

from typing import Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式神经网络接口
from einops import rearrange  # 导入张量重排工具
from transformers import Gemma4VisionConfig  # 导入Gemma4视觉配置

from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL  # 导入QKV后端实现
from sglang.srt.layers.clippable_linear import (  # 导入可裁剪线性层
    ClippableGateUpParallelLinear,  # 可裁剪的Gate-Up并行线性层
    ClippableQKVParallelLinear,  # 可裁剪的QKV并行线性层
    ClippableRowParallelLinear,  # 可裁剪的行并行线性层
)
from sglang.srt.layers.dp_attention import get_attention_tp_size  # 导入获取注意力张量并行大小的函数
from sglang.srt.layers.layernorm import Gemma4RMSNorm  # 导入Gemma4 RMS归一化层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.utils import add_prefix, get_device_capability, is_cuda, is_hip  # 导入工具函数

# ---------------------------------------------------------------------------
# 2-D Multidimensional RoPE (matches HF Gemma4RotaryEmbedding for vision)  # 2维多维旋转位置编码（匹配HF Gemma4RotaryEmbedding视觉版本）
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:  # 旋转半边函数，用于RoPE计算
    x1 = x[..., : x.shape[-1] // 2]  # 取前半部分
    x2 = x[..., x.shape[-1] // 2 :]  # 取后半部分
    return torch.cat((-x2, x1), dim=-1)  # 拼接旋转后的结果：(-x2, x1)


def _apply_rotary(  # 应用旋转位置编码函数
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor  # 输入张量、cos值、sin值
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)  # 旋转位置编码公式：x*cos + rotate_half(x)*sin


class Gemma4VisionRotaryEmbedding(nn.Module):
    """Compute 2-D multidimensional RoPE cos/sin for patch positions."""  # 计算补丁位置的2维多维RoPE cos/sin值

    def __init__(self, config: Gemma4VisionConfig):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.head_dim = config.head_dim  # 每个注意力头的维度
        self.rope_theta: float = config.rope_parameters["rope_theta"]  # RoPE的theta参数（基数）

    @torch.no_grad()  # 禁用梯度计算
    def forward(  # 前向传播方法
        self, x: torch.Tensor, patch_positions: torch.Tensor  # 输入张量（仅用于设备/数据类型）、补丁位置
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, hidden] – only used for device/dtype.  # x: [批次, 序列, 隐藏维度] – 仅用于获取设备和数据类型
            patch_positions: [batch, num_patches, 2] – (x, y) coordinates.  # patch_positions: [批次, 补丁数, 2] – (x, y)坐标
        Returns:
            (cos, sin) each of shape [batch, num_patches, head_dim].  # 返回：(cos, sin)，形状均为[批次, 补丁数, 头维度]
        """
        ndim = patch_positions.shape[-1]  # 2  # 维度数为2（x和y坐标）
        head_dim_per_dim = self.head_dim // ndim  # 每个维度分配的头维度

        all_embs = []  # 存储所有维度的嵌入
        for d in range(ndim):  # 遍历每个维度（x和y）
            dim_inv_freq = 1.0 / (  # 计算逆频率
                self.rope_theta  # theta基数
                ** (  # 的幂次
                    torch.arange(  # 生成等差数列
                        0, head_dim_per_dim, 2, device=x.device, dtype=torch.float  # 从0到head_dim_per_dim，步长2
                    )
                    / head_dim_per_dim  # 除以头维度
                )
            )
            dim_inv_freq_expanded = dim_inv_freq[None, :, None].expand(  # 扩展逆频率维度
                patch_positions.shape[0], -1, 1  # 扩展到[批次, 频率数, 1]
            )
            dim_positions = patch_positions[:, :, d].float()  # 获取第d维的位置并转为float
            dim_positions_expanded = dim_positions[:, None, :]  # 扩展位置维度为[批次, 1, 补丁数]

            dim_freqs = (dim_inv_freq_expanded @ dim_positions_expanded).transpose(1, 2)  # 计算频率并转置
            dim_emb = torch.cat((dim_freqs, dim_freqs), dim=-1)  # 拼接频率（cos和sin共用）
            all_embs.append(dim_emb)  # 添加到列表

        emb = torch.cat(all_embs, dim=-1)  # 拼接所有维度的嵌入
        cos = emb.cos().to(dtype=x.dtype)  # 计算cos值并转为输入数据类型
        sin = emb.sin().to(dtype=x.dtype)  # 计算sin值并转为输入数据类型
        return cos, sin  # 返回cos和sin


def _apply_multidimensional_rope(  # 应用多维旋转位置编码函数
    x: torch.Tensor,  # 输入张量
    cos: torch.Tensor,  # cos值
    sin: torch.Tensor,  # sin值
) -> torch.Tensor:
    """Apply 2-D RoPE to x of shape [batch*seq, heads, head_dim].  # 将2维RoPE应用到形状为[批次*序列, 头数, 头维度]的x上

    cos/sin have shape [batch, seq, head_dim]. We split along head_dim into  # cos/sin形状为[批次, 序列, 头维度]。沿head_dim分割为
    ndim=2 parts and apply standard rotary to each independently.  # ndim=2部分，各自独立应用标准旋转编码
    """
    ndim = 2  # 维度数为2
    chunk_size = x.shape[-1] // ndim  # 每个块的维度大小
    x_parts = x.split(chunk_size, dim=-1)  # 将x按维度分割为2部分
    cos_parts = cos.split(chunk_size, dim=-1)  # 将cos按维度分割为2部分
    sin_parts = sin.split(chunk_size, dim=-1)  # 将sin按维度分割为2部分
    y_parts = [  # 对每个维度分别应用旋转位置编码
        _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k]) for k in range(ndim)  # 第k维的旋转编码
    ]
    return torch.cat(y_parts, dim=-1)  # 拼接所有维度的结果


# ---------------------------------------------------------------------------
# Vision Attention (TP-sharded, fused QKV)  # 视觉注意力（张量并行分片，融合QKV）
# ---------------------------------------------------------------------------


class Gemma4VisionAttention(nn.Module):
    """Multi-head attention for the Gemma 4 vision encoder.  # Gemma 4视觉编码器的多头注意力

    QKV uses a fused ``ClippableQKVParallelLinear`` for efficient matmul with  # QKV使用融合的ClippableQKVParallelLinear进行高效矩阵乘法
    per-projection clip bounds.  Output projection uses ``ClippableLinear``.  # 带每投影裁剪边界。输出投影使用ClippableLinear。
    """

    def __init__(  # 初始化方法
        self,
        config: Gemma4VisionConfig,  # Gemma4视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.head_dim = config.head_dim  # 每个头的维度

        tp_size = get_attention_tp_size()  # 获取注意力张量并行大小
        self.num_heads_per_partition = config.num_attention_heads // tp_size  # 每个分区的头数
        self.num_kv_heads_per_partition = config.num_key_value_heads // tp_size  # 每个分区的KV头数

        self.qkv = ClippableQKVParallelLinear(  # 可裁剪的QKV并行线性层
            hidden_size=config.hidden_size,  # 隐藏层大小
            head_size=config.head_dim,  # 头大小
            total_num_heads=config.num_attention_heads,  # 总头数
            total_num_kv_heads=config.num_key_value_heads,  # 总KV头数
            bias=config.attention_bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=prefix,  # 前缀
        )
        self.o_proj = ClippableRowParallelLinear(  # 可裁剪的行并行输出投影层
            input_size=config.num_attention_heads * config.head_dim,  # 输入大小
            output_size=config.hidden_size,  # 输出大小
            bias=config.attention_bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("o_proj", prefix),  # 添加前缀
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q归一化层
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K归一化层
        self.v_norm = Gemma4RMSNorm(  # V归一化层
            self.head_dim, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False  # 无缩放
        )

        backend = self._select_backend()  # 选择注意力后端
        self.qkv_backend = QKV_BACKEND_IMPL[backend](  # 创建QKV后端实例
            head_dim=config.head_dim,  # 头维度
            num_heads=self.num_heads_per_partition,  # 每分区头数
            num_kv_heads=self.num_kv_heads_per_partition,  # 每分区KV头数
            dropout=0.0,  # dropout率为0
            flatten_batch=True,  # 展平批次
            softmax_in_single_precision=False,  # 不使用单精度softmax
            softmax_scale=1.0,  # softmax缩放因子
        )

    @staticmethod  # 静态方法
    def _select_backend() -> str:  # 选择注意力后端
        """Mirror VisionAttention._determine_attention_backend for consistency."""  # 为保持一致性，镜像VisionAttention的后端选择逻辑
        from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数获取函数

        override = get_global_server_args().mm_attention_backend  # 获取用户指定的多模态注意力后端
        if override is not None:  # 如果用户指定了后端
            return override  # 返回用户指定的后端
        if is_cuda():  # 如果是CUDA平台
            major, _ = get_device_capability()  # 获取设备计算能力
            if major == 9:  # 如果是计算能力9.x（Hopper架构）
                from sglang.srt.utils import is_blackwell_supported  # 导入Blackwell支持检测

                if is_blackwell_supported():  # 如果支持Blackwell架构
                    return "triton_attn"  # 使用triton注意力
                return "fa3"  # 使用FlashAttention3
            return "triton_attn"  # 其他CUDA架构使用triton注意力
        if is_hip():  # 如果是HIP（AMD）平台
            # ROCm: use triton_attn to avoid SDPA flatten_batch issues  # ROCm：使用triton_attn以避免SDPA的flatten_batch问题
            # with multi-image/video inputs  # 多图像/视频输入时的问题
            return "triton_attn"  # 使用triton注意力
        return "sdpa"  # 其他平台使用SDPA

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cos: torch.Tensor,  # 旋转位置编码cos值
        sin: torch.Tensor,  # 旋转位置编码sin值
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape  # 获取批次大小和序列长度

        q, k, v = self.qkv(hidden_states)  # 通过QKV投影层

        q = q.reshape(bsz * seq_len, self.num_heads_per_partition, self.head_dim)  # 重塑Q的形状
        k = k.reshape(bsz * seq_len, self.num_kv_heads_per_partition, self.head_dim)  # 重塑K的形状
        v = v.reshape(bsz * seq_len, self.num_kv_heads_per_partition, self.head_dim)  # 重塑V的形状

        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)  # Q归一化并恢复形状
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)  # K归一化并恢复形状
        v = self.v_norm(v.reshape(-1, self.head_dim)).reshape(v.shape)  # V归一化并恢复形状

        cos_flat = cos.reshape(bsz * seq_len, 1, self.head_dim)  # 将cos展平
        sin_flat = sin.reshape(bsz * seq_len, 1, self.head_dim)  # 将sin展平
        q = _apply_multidimensional_rope(q, cos_flat, sin_flat)  # 对Q应用多维旋转位置编码
        k = _apply_multidimensional_rope(k, cos_flat, sin_flat)  # 对K应用多维旋转位置编码

        if attention_mask is not None:  # 如果提供了注意力掩码
            attn_mask_4d = (  # 构建4D注意力掩码
                attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)  # 外积生成2D掩码
            ).unsqueeze(1)  # 添加头维度
        else:  # 否则
            attn_mask_4d = None  # 不使用掩码

        output = self.qkv_backend.forward(  # 通过QKV后端计算注意力
            q=q,  # 查询
            k=k,  # 键
            v=v,  # 值
            cu_seqlens=None,  # 未使用累积序列长度
            bsz=bsz,  # 批次大小
            seq_len=seq_len,  # 序列长度
            attention_mask=attn_mask_4d,  # 4D注意力掩码
            softmax_scale=1.0,  # softmax缩放因子
        )

        output = rearrange(output, "(b s) h d -> b s (h d)", b=bsz)  # 重排输出形状
        output = self.o_proj(output)  # 通过输出投影层
        return output  # 返回输出


# ---------------------------------------------------------------------------
# Vision MLP (GatedGELU, TP-sharded)  # 视觉MLP（门控GELU，张量并行分片）
# ---------------------------------------------------------------------------


class Gemma4VisionMLP(nn.Module):
    """Gemma4视觉MLP，使用门控GELU激活函数。"""

    def __init__(  # 初始化方法
        self,
        config: Gemma4VisionConfig,  # Gemma4视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        if config.hidden_activation != "gelu_pytorch_tanh":  # 如果隐藏激活函数不是gelu_pytorch_tanh
            raise ValueError(  # 抛出值错误
                f"Gemma4VisionMLP expects hidden_activation='gelu_pytorch_tanh', "  # Gemma4VisionMLP期望hidden_activation='gelu_pytorch_tanh'
                f"got {config.hidden_activation!r}"  # 但得到的是
            )
        self.gate_up = ClippableGateUpParallelLinear(  # 可裁剪的Gate-Up并行线性层
            input_size=config.hidden_size,  # 输入大小
            intermediate_size=config.intermediate_size,  # 中间层大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=prefix,  # 前缀
        )
        self.down_proj = ClippableRowParallelLinear(  # 可裁剪的行并行下投影层
            input_size=config.intermediate_size,  # 输入大小
            output_size=config.hidden_size,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 添加前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        gate, up = self.gate_up(x)  # 通过gate_up层得到门控和上投影
        x = F.gelu(gate, approximate="tanh") * up  # GELU激活乘以上投影（门控机制）
        x = self.down_proj(x)  # 通过下投影层
        return x  # 返回输出


# ---------------------------------------------------------------------------
# Encoder Layer  # 编码器层
# ---------------------------------------------------------------------------


class Gemma4VisionEncoderLayer(nn.Module):
    """Gemma4视觉编码器层，包含自注意力、MLP和多个归一化层。"""

    def __init__(  # 初始化方法
        self,
        config: Gemma4VisionConfig,  # Gemma4视觉配置
        layer_idx: int,  # 层索引
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.self_attn = Gemma4VisionAttention(  # 自注意力层
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
        )
        self.mlp = Gemma4VisionMLP(  # MLP层
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 添加前缀
        )
        eps = config.rms_norm_eps  # RMS归一化epsilon
        hs = config.hidden_size  # 隐藏层大小
        self.input_layernorm = Gemma4RMSNorm(hs, eps=eps)  # 输入层归一化
        self.post_attention_layernorm = Gemma4RMSNorm(hs, eps=eps)  # 注意力后归一化
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hs, eps=eps)  # 前馈前归一化
        self.post_feedforward_layernorm = Gemma4RMSNorm(hs, eps=eps)  # 前馈后归一化

        self.register_buffer("layer_scalar", torch.ones(()))  # 注册层缩放因子，初始值为1

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cos: torch.Tensor,  # 旋转位置编码cos值
        sin: torch.Tensor,  # 旋转位置编码sin值
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选
    ) -> torch.Tensor:
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 输入层归一化
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)  # 自注意力
        hidden_states = self.post_attention_layernorm(hidden_states)  # 注意力后归一化
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 更新残差
        hidden_states = self.pre_feedforward_layernorm(hidden_states)  # 前馈前归一化
        hidden_states = self.mlp(hidden_states)  # MLP
        hidden_states = self.post_feedforward_layernorm(hidden_states)  # 前馈后归一化
        hidden_states = residual + hidden_states  # 残差连接

        hidden_states = hidden_states * self.layer_scalar  # 乘以层缩放因子
        return hidden_states  # 返回输出


# ---------------------------------------------------------------------------
# Vision Transformer (stack of encoder layers + RoPE)  # 视觉Transformer（编码器层堆叠 + RoPE）
# ---------------------------------------------------------------------------


class Gemma4VisionTransformer(nn.Module):
    """Gemma4视觉Transformer，由多个编码器层堆叠而成。"""

    def __init__(  # 初始化方法
        self,
        config: Gemma4VisionConfig,  # Gemma4视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)  # 旋转位置编码
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                Gemma4VisionEncoderLayer(  # 编码器层
                    config,  # 配置
                    layer_idx=i,  # 层索引
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 添加前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历所有隐藏层
            ]
        )

    def forward(  # 前向传播方法
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入
        attention_mask: torch.Tensor,  # 注意力掩码
        patch_positions: torch.Tensor,  # 补丁位置
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: [batch, seq, hidden_size]  # 输入嵌入：[批次, 序列, 隐藏大小]
            attention_mask: [batch, seq] — True = valid token  # 注意力掩码：[批次, 序列] — True = 有效token
            patch_positions: [batch, seq, 2]  # 补丁位置：[批次, 序列, 2]
        Returns:
            last_hidden_state: [batch, seq, hidden_size]  # 返回：最后隐藏状态：[批次, 序列, 隐藏大小]
        """
        cos, sin = self.rotary_emb(inputs_embeds, patch_positions)  # 计算旋转位置编码
        hidden_states = inputs_embeds  # 初始化隐藏状态
        for layer in self.layers:  # 遍历每一层
            hidden_states = layer(hidden_states, cos, sin, attention_mask)  # 通过编码器层
        return hidden_states  # 返回最后的隐藏状态


# ---------------------------------------------------------------------------
# Patch Embedder  # 补丁嵌入器
# ---------------------------------------------------------------------------


class Gemma4VisionPatchEmbedder(nn.Module):
    """Gemma4视觉补丁嵌入器，将像素值投影到模型空间并添加位置编码。"""

    def __init__(self, config: Gemma4VisionConfig):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.patch_size = config.patch_size  # 补丁大小
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.position_embedding_size = config.position_embedding_size  # 位置嵌入大小

        self.input_proj = nn.Linear(  # 输入投影层
            3 * self.patch_size**2, self.hidden_size, bias=False  # 输入维度为3*补丁大小^2（RGB像素），无偏置
        )
        self.position_embedding_table = nn.Parameter(  # 位置嵌入表参数
            torch.ones(2, self.position_embedding_size, self.hidden_size)  # 形状为[2, 位置嵌入大小, 隐藏大小]
        )

    def _position_embeddings(  # 计算位置嵌入
        self, patch_positions: torch.Tensor, padding_positions: torch.Tensor  # 补丁位置、填充位置
    ) -> torch.Tensor:
        clamped_positions = patch_positions.clamp(min=0)  # 将位置限制为非负值
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)  # 转为one-hot编码
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)  # 置换维度并转为位置嵌入表的数据类型
        position_embeddings = one_hot @ self.position_embedding_table  # 矩阵乘法得到位置嵌入
        position_embeddings = position_embeddings.sum(dim=1)  # 沿维度1求和，合并x和y的位置嵌入
        position_embeddings = torch.where(  # 处理填充位置
            padding_positions.unsqueeze(-1), 0.0, position_embeddings  # 填充位置的位置嵌入设为0
        )
        return position_embeddings  # 返回位置嵌入

    def _patch_projection(self, pixel_values: torch.Tensor) -> torch.Tensor:  # 补丁投影方法
        """Project pre-patchified pixels into model space.  # 将预补丁化的像素投影到模型空间

        Args:
            pixel_values: [batch, num_patches, patch_pixels] — already patchified  # pixel_values: [批次, 补丁数, 补丁像素数] — 已补丁化
                          by the image processor, values in [0, 1].  # 由图像处理器处理，值在[0, 1]范围内
        """
        patches = 2 * (pixel_values - 0.5)  # 将[0,1]范围的值归一化到[-1,1]
        return self.input_proj(patches.to(self.input_proj.weight.dtype))  # 投影到模型空间

    def forward(  # 前向传播方法
        self,
        pixel_values: torch.Tensor,  # 像素值
        pixel_position_ids: torch.Tensor,  # 像素位置ID
        padding_positions: torch.Tensor,  # 填充位置
    ) -> torch.Tensor:
        """Compute patch embeddings with positional information.  # 计算带位置信息的补丁嵌入

        Args:
            pixel_values: [batch, num_patches, patch_pixels] — pre-patchified.  # pixel_values: [批次, 补丁数, 补丁像素数] — 预补丁化
            pixel_position_ids: [batch, num_patches, 2] — (x, y) positions,  # pixel_position_ids: [批次, 补丁数, 2] — (x, y)位置
                                -1 for padding patches.  # -1表示填充补丁
            padding_positions: [batch, num_patches] — True for padding patches.  # padding_positions: [批次, 补丁数] — True表示填充补丁
        """
        hidden_states = self._patch_projection(pixel_values)  # 补丁投影
        position_embeddings = self._position_embeddings(  # 计算位置嵌入
            pixel_position_ids, padding_positions  # 传入位置ID和填充位置
        )
        return hidden_states + position_embeddings  # 返回补丁嵌入加位置嵌入


# ---------------------------------------------------------------------------
# Pooler  # 池化器
# ---------------------------------------------------------------------------


class Gemma4VisionPooler(nn.Module):
    """Gemma4视觉池化器，通过位置平均池化将补丁特征降维。"""

    def __init__(self, config: Gemma4VisionConfig):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.root_hidden_size = self.hidden_size**0.5  # 隐藏大小的平方根，用于缩放

    def _avg_pool_by_positions(  # 按位置平均池化方法
        self, x: torch.Tensor, patch_positions: torch.Tensor, length: int  # 输入、补丁位置、目标长度
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = x.shape[1]  # 输入序列长度
        k = int((input_seq_len // length) ** 0.5)  # 池化核大小
        k_squared = k**2  # 池化核面积的平方
        if k_squared * length != input_seq_len:  # 如果无法整除
            raise ValueError(  # 抛出值错误
                f"Cannot pool {x.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."  # 无法池化到目标长度
            )
        clamped_positions = patch_positions.clamp(min=0)  # 将位置限制为非负值
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1  # 最大x坐标+1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")  # 计算核索引
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]  # 合并x和y方向的核索引

        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared  # 计算池化权重（one-hot除以核面积）
        output = weights.transpose(1, 2).to(x.dtype) @ x  # 加权平均池化
        mask = torch.logical_not((weights == 0).all(dim=1))  # 计算有效token掩码
        return output, mask  # 返回池化输出和掩码

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        patch_positions: torch.Tensor,  # 补丁位置
        padding_positions: torch.Tensor,  # 填充位置
        output_length: Optional[int] = None,  # 输出长度，可选
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (pooled_hidden_states, mask) where mask is True for valid tokens.  # 返回：(池化后的隐藏状态, 掩码)，掩码True表示有效token
        """
        if output_length is None:  # 如果未指定输出长度
            raise ValueError("output_length is required for Gemma4VisionPooler")  # 抛出值错误
        if output_length > hidden_states.shape[1]:  # 如果输出长度大于输入长度
            raise ValueError(  # 抛出值错误
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"  # 不能输出比补丁数更多的软token
                f" ({hidden_states.shape[1]}). Change the value of `num_soft_tokens` when processing."  # （请求的软token数），请在处理时更改num_soft_tokens的值
            )
        length = output_length  # 获取输出长度
        if isinstance(length, (list, tuple)):  # 如果是列表或元组
            length = length[0]  # 取第一个元素
        if hidden_states.shape[1] == length:  # 如果输入长度等于输出长度
            mask = padding_positions  # 直接使用填充位置作为掩码
        else:  # 否则需要池化
            hidden_states, mask = self._avg_pool_by_positions(  # 按位置平均池化
                hidden_states, patch_positions, length  # 传入隐藏状态、位置和目标长度
            )
        hidden_states = hidden_states * self.root_hidden_size  # 乘以隐藏大小平方根进行缩放
        return hidden_states, mask  # 返回池化后的隐藏状态和掩码


# ---------------------------------------------------------------------------
# Top-level Vision Encoder (patch_embedder → transformer → pooler)  # 顶层视觉编码器（补丁嵌入器 → Transformer → 池化器）
# ---------------------------------------------------------------------------


class Gemma4VisionEncoder(nn.Module):
    """Drop-in replacement for HF ``Gemma4VisionEncoder`` with TP support."""  # 替代HF Gemma4VisionEncoder的实现，支持张量并行

    def __init__(  # 初始化方法
        self,
        config: Gemma4VisionConfig,  # Gemma4视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.patch_size = config.patch_size  # 补丁大小
        self.pooling_kernel_size = config.pooling_kernel_size  # 池化核大小

        self.patch_embedder = Gemma4VisionPatchEmbedder(config)  # 补丁嵌入器
        self.encoder = Gemma4VisionTransformer(  # 视觉Transformer编码器
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("encoder", prefix),  # 添加前缀
        )
        self.pooler = Gemma4VisionPooler(config)  # 池化器

        # Post-pooling standardization (normalizes vision tokens before projection)  # 池化后标准化（在投影前归一化视觉token）
        self.standardize = getattr(config, "standardize", False)  # 是否启用标准化
        if self.standardize:  # 如果启用标准化
            self.register_buffer("std_bias", torch.zeros(config.hidden_size))  # 标准化偏置
            self.register_buffer("std_scale", torch.ones(config.hidden_size))  # 标准化缩放

    @property  # 属性装饰器
    def device(self) -> torch.device:  # 设备属性
        return self.patch_embedder.input_proj.weight.device  # 返回补丁嵌入器投影层的设备

    def forward(  # 前向传播方法
        self,
        pixel_values: torch.Tensor,  # 像素值
        pixel_position_ids: torch.Tensor,  # 像素位置ID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-patchified pixel_values into soft tokens.  # 将预补丁化的像素值编码为软token

        Args:
            pixel_values: [batch, num_patches, patch_pixels] — pre-patchified  # pixel_values: [批次, 补丁数, 补丁像素数] — 预补丁化
                          by the image processor.  # 由图像处理器处理
            pixel_position_ids: [batch, num_patches, 2] — (x, y) positions,  # pixel_position_ids: [批次, 补丁数, 2] — (x, y)位置
                                -1 for padding patches.  # -1表示填充补丁

        Returns:
            (hidden_states, pooler_mask) — hidden_states [batch, output_len, hidden],  # 返回：(隐藏状态, 池化掩码) — 隐藏状态 [批次, 输出长度, 隐藏维度]
            pooler_mask [batch, output_len] True = valid.  # 池化掩码 [批次, 输出长度] True = 有效
        """
        k2 = self.pooling_kernel_size * self.pooling_kernel_size  # 池化核面积
        output_length = pixel_values.shape[-2] // k2  # 计算输出长度（补丁数除以池化核面积）

        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # 计算填充位置（所有坐标为-1的补丁）

        inputs_embeds = self.patch_embedder(  # 通过补丁嵌入器
            pixel_values, pixel_position_ids, padding_positions  # 传入像素值、位置ID和填充位置
        )

        last_hidden = self.encoder(  # 通过Transformer编码器
            inputs_embeds=inputs_embeds,  # 输入嵌入
            attention_mask=~padding_positions,  # 注意力掩码（非填充位置为True）
            patch_positions=pixel_position_ids,  # 补丁位置
        )

        pooled, pooler_mask = self.pooler(  # 通过池化器
            last_hidden,  # 最后的隐藏状态
            pixel_position_ids,  # 补丁位置
            padding_positions,  # 填充位置
            output_length=output_length,  # 输出长度
        )

        if self.standardize:  # 如果启用标准化
            pooled = (pooled - self.std_bias) * self.std_scale  # 应用标准化：(x - bias) * scale

        return pooled, pooler_mask  # 返回池化后的隐藏状态和掩码
