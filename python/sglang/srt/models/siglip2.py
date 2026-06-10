# Siglip2 视觉编码器推理实现文件
# 本文件实现了Siglip2视觉模型，支持NaFlex可变分辨率图像处理
# 与Siglip v1不同，Siglip2通过打包序列和cu_seqlens处理不同尺寸的图像
# 包含视觉嵌入、注意力、MLP、编码器和权重加载等核心组件

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 Liquid AI. All rights reserved.
#
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
#
# Adapted from vLLM's implementation of Siglip2VisionModel
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/lfm2_siglip2.py
#
# Siglip2 is a vision encoder that supports variable-resolution images via NaFlex.
# Unlike Siglip v1 which uses fixed-size images, Siglip2 handles images of different
# sizes by packing them into sequences and using cu_seqlens for attention.

from collections.abc import Iterable  # 导入可迭代类型
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
from transformers import Siglip2VisionConfig  # 导入Siglip2视觉配置

from sglang.srt.layers.activation import get_act_fn  # 导入激活函数
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力
from sglang.srt.layers.linear import (  # 导入线性层
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.utils import add_prefix  # 导入前缀工具


class Siglip2VisionEmbeddings(nn.Module):
    """Siglip2 vision embeddings with NaFlex variable-resolution support."""
    """Siglip2视觉嵌入，支持NaFlex可变分辨率"""

    def __init__(self, config: Siglip2VisionConfig):
        """初始化Siglip2视觉嵌入，配置补丁嵌入和位置编码"""
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        # Siglip2 uses Linear instead of Conv2d for patch embedding
        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )  # 补丁线性嵌入
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)  # 位置嵌入

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,  # 打包的像素值
        spatial_shapes: torch.LongTensor,  # 空间形状
    ) -> torch.Tensor:
        """Embed patchified pixel values in packed (unpadded) form.

        Args:
            pixel_values_packed: (1, total_tokens, patch_dim) or
                (total_tokens, patch_dim), packed in tile order.
            spatial_shapes: (num_tiles, 2) on CPU (height, width) per tile.

        Returns:
            (1, total_tokens, embed_dim) packed embeddings.
        """
        """将打包的像素值嵌入为补丁嵌入，支持可变分辨率"""
        assert spatial_shapes.device.type == "cpu", (
            "Expected `spatial_shapes` on CPU to avoid device-to-host sync in "
            "variable-length packing."
        )

        if pixel_values_packed.dim() == 3:
            assert pixel_values_packed.shape[0] == 1
            pixel_values_flat = pixel_values_packed[0]
        else:
            pixel_values_flat = pixel_values_packed

        lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(dtype=torch.int64)  # 每个图像的token数
        lengths_list = lengths.tolist()
        total_tokens = int(sum(lengths_list))
        if total_tokens != pixel_values_flat.shape[0]:
            raise ValueError(
                "Packed pixel_values token count does not match spatial_shapes: "
                f"{pixel_values_flat.shape[0]} vs {total_tokens}."
            )

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values_flat.to(dtype=target_dtype))  # 补丁嵌入

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )  # 重塑位置嵌入
        packed_pos_embeds = self.resize_positional_embeddings_packed(
            positional_embeddings,
            spatial_shapes,
            lengths_list=lengths_list,
        )  # 调整位置嵌入大小

        embeddings = patch_embeds + packed_pos_embeds  # 补丁嵌入+位置嵌入
        return embeddings.unsqueeze(0)

    @staticmethod
    def resize_positional_embeddings_packed(
        positional_embeddings: torch.Tensor,  # 位置嵌入
        spatial_shapes: torch.LongTensor,  # 空间形状
        lengths_list: list[int],  # 长度列表
    ) -> torch.Tensor:
        """Resize positional embeddings per image and return a packed tensor.

        Args:
            positional_embeddings: (height, width, embed_dim) base grid.
            spatial_shapes: (batch_size, 2) on CPU, (height, width) per image.
            lengths_list: flattened token length per image (height * width).

        Returns:
            (total_tokens, embed_dim) packed positional embeddings.
        """
        """逐图像调整位置嵌入大小并返回打包张量"""
        assert spatial_shapes.device.type == "cpu"

        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        total_tokens = int(sum(lengths_list))
        packed_pos_embeds = torch.empty(
            (total_tokens, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width)
        pos_4d = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for
        # bfloat16/float16 on CPU.
        if pos_4d.device.type == "cpu":
            pos_4d = pos_4d.to(torch.float32)

        offset = 0
        for i, length in enumerate(lengths_list):  # 逐图像插值位置编码
            if length <= 0:
                continue
            height, width = spatial_shapes[i].tolist()
            resized = F.interpolate(
                pos_4d,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape(embed_dim, height * width).transpose(0, 1)
            resized = resized.to(source_dtype)
            packed_pos_embeds[offset : offset + length] = resized
            offset += length

        return packed_pos_embeds


class Siglip2Attention(nn.Module):
    """Multi-headed attention for Siglip2 using optimized VisionAttention backend."""
    """Siglip2多头注意力，使用优化的VisionAttention后端"""

    def __init__(
        self,
        config: Siglip2VisionConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化Siglip2注意力"""
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Use SGLang's optimized VisionAttention with automatic backend selection
        self.attn = VisionAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            projection_size=self.embed_dim,
            use_qkv_parallel=True,
            dropout=config.attention_dropout,
            flatten_batch=True,  # For variable-length sequence support  支持可变长度序列
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cu_seqlens: torch.Tensor,  # 累计序列长度
        max_seqlen: int | torch.Tensor,  # 最大序列长度
    ) -> torch.Tensor:
        """Forward pass with variable-length attention.

        Args:
            hidden_states: (1, total_tokens, embed_dim) packed hidden states
            cu_seqlens: Cumulative sequence lengths for variable-length attention
            max_seqlen: Maximum sequence length (unused, VisionAttention computes internally)

        Returns:
            (1, total_tokens, embed_dim) attention output
        """
        """可变长度注意力前向传播"""
        return self.attn(hidden_states, cu_seqlens=cu_seqlens)


class Siglip2MLP(nn.Module):
    """MLP for Siglip2 encoder layers."""
    """Siglip2编码器层的MLP"""

    def __init__(
        self,
        config: Siglip2VisionConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化MLP"""
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)  # 激活函数

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP前向传播：fc1+激活+fc2"""
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    """Single encoder layer for Siglip2."""
    """Siglip2单个编码器层"""

    def __init__(
        self,
        config: Siglip2VisionConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        """初始化编码器层"""
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cu_seqlens: torch.Tensor,  # 累计序列长度
        max_seqlen: int | torch.Tensor,  # 最大序列长度
    ) -> torch.Tensor:
        """Forward pass for encoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, embed_dim).
            cu_seqlens: Cumulative sequence lengths tensor.
            max_seqlen: Maximum sequence length.
        """
        """编码器层前向传播：Pre-Norm注意力+Pre-Norm MLP"""
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)  # 注意力前归一化
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)  # MLP前归一化
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # 残差连接
        return hidden_states


class Siglip2Encoder(nn.Module):
    """Transformer encoder for Siglip2."""
    """Siglip2 Transformer编码器"""

    def __init__(
        self,
        config: Siglip2VisionConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        num_hidden_layers_override: Optional[int] = None,  # 层数覆盖
        prefix: str = "",  # 前缀
    ):
        """初始化编码器，创建编码器层列表"""
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                )
                for idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入
        cu_seqlens: torch.Tensor,  # 累计序列长度
        max_seqlen: int | torch.Tensor,  # 最大序列长度
        return_all_hidden_states: bool = False,  # 是否返回所有隐藏状态
    ) -> torch.Tensor | list[torch.Tensor]:
        """编码器前向传播"""
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


def resolve_visual_encoder_outputs(
    encoder_outputs: torch.Tensor | list[torch.Tensor],  # 编码器输出
    post_layer_norm: Optional[nn.LayerNorm],  # 后层归一化
    select_layers: Optional[list[int]] = None,  # 选择层
    max_possible_layers: Optional[int] = None,  # 最大可能层数
) -> torch.Tensor:
    """Resolve outputs from visual encoder based on select_layers."""
    """根据选择层解析视觉编码器输出"""
    if select_layers is None:  # 不选择层时
        if isinstance(encoder_outputs, list):
            encoder_outputs = encoder_outputs[-1]
        if post_layer_norm is not None:
            encoder_outputs = post_layer_norm(encoder_outputs)
        return encoder_outputs

    if max_possible_layers is None:
        raise ValueError(
            "`max_possible_layers` must be provided alongside `select_layers`"
        )

    if not isinstance(encoder_outputs, list):
        raise ValueError(
            "Expected encoder_outputs to be a list when select_layers is provided"
        )

    # Get the hidden states corresponding to the layer indices
    num_loaded_layers = len(encoder_outputs) - 1
    offset = max_possible_layers - num_loaded_layers
    hs_pool = [
        (
            encoder_outputs[layer_idx]
            if layer_idx >= 0
            else encoder_outputs[layer_idx + offset]
        )
        for layer_idx in select_layers
    ]

    uses_last_layer = select_layers[-1] in (max_possible_layers - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(hs_pool[-1])

    return torch.cat(hs_pool, dim=-1)


class Siglip2VisionTransformer(nn.Module):
    """Siglip2 Vision Transformer with NaFlex variable-resolution support."""
    """Siglip2视觉Transformer，支持NaFlex可变分辨率"""

    def __init__(
        self,
        config: Siglip2VisionConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        num_hidden_layers_override: Optional[int] = None,  # 层数覆盖
        require_post_norm: Optional[bool] = None,  # 是否需要后归一化
        prefix: str = "",  # 前缀
    ):
        """初始化Siglip2视觉Transformer"""
        super().__init__()
        embed_dim = config.hidden_size
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=add_prefix("encoder", prefix),
        )
        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    @property
    def dtype(self) -> torch.dtype:
        """获取模型数据类型"""
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return self.embeddings.patch_embedding.weight.device

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,  # 打包像素值
        spatial_shapes: torch.LongTensor,  # 空间形状
        cu_seqlens: torch.Tensor,  # 累计序列长度
        max_seqlen: torch.Tensor,  # 最大序列长度
        select_layers: Optional[list[int]] = None,  # 选择层
    ) -> torch.Tensor:
        """Forward pass through the vision transformer.

        Args:
            pixel_values_packed: Packed pixel values
            spatial_shapes: (batch_size, 2) tensor with (height, width) per image
            cu_seqlens: Cumulative sequence lengths
            max_seqlen: Maximum sequence length
            select_layers: Optional layer indices to select hidden states from

        Returns:
            Vision features tensor
        """
        """视觉Transformer前向传播"""
        hidden_states = self.embeddings(pixel_values_packed, spatial_shapes)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            return_all_hidden_states=select_layers is not None,
        )

        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs,
            self.post_layernorm,
            select_layers=select_layers,
            max_possible_layers=self.config.num_hidden_layers,
        )

        return encoder_outputs


class Siglip2Model(nn.Module):
    """Siglip2 Vision Model for use in vision-language models."""
    """Siglip2视觉模型，用于视觉语言模型"""

    def __init__(
        self,
        config: Siglip2VisionConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        num_hidden_layers_override: Optional[int] = None,  # 层数覆盖
        require_post_norm: Optional[bool] = None,  # 是否需要后归一化
        prefix: str = "",  # 前缀
    ):
        """初始化Siglip2模型"""
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=add_prefix("vision_model", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        """获取数据类型"""
        return self.vision_model.dtype

    @property
    def device(self) -> torch.device:
        """获取设备"""
        return self.vision_model.device

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,  # 打包像素值
        spatial_shapes: torch.LongTensor,  # 空间形状
        cu_seqlens: torch.Tensor,  # 累计序列长度
        max_seqlen: torch.Tensor,  # 最大序列长度
        select_layers: Optional[list[int]] = None,  # 选择层
    ) -> torch.Tensor:
        """Forward pass through the vision model."""
        """视觉模型前向传播"""
        return self.vision_model(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            select_layers=select_layers,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载模型权重，处理堆叠参数和重命名映射"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # VisionAttention uses attn.qkv_proj for fused Q/K/V
            ("attn.qkv_proj", "q_proj", "q"),
            ("attn.qkv_proj", "k_proj", "k"),
            ("attn.qkv_proj", "v_proj", "v"),
        ]
        # VisionAttention uses attn.proj instead of out_proj
        params_rename_mapping = {
            "out_proj": "attn.proj",
        }
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is optional in Siglip2Model
            if (
                name.startswith("vision_model.post_layernorm")
                and self.vision_model.post_layernorm is None
            ):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):  # 跳过超出的层
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 堆叠参数
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:  # 非堆叠参数
                # Apply rename mappings (e.g., out_proj -> attn.proj)
                for old_name, new_name in params_rename_mapping.items():
                    if old_name in name:
                        name = name.replace(old_name, new_name)
                        break

                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
