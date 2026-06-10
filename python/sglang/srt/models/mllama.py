# Mllama多模态模型推理实现 - 基于Llama架构的视觉语言模型，支持图像理解和交叉注意力
# 本文件实现了Mllama模型的完整推理流程，包含视觉编码器、文本模型、交叉注意力和权重加载

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://github.com/vllm-project/vllm/blob/7193774b1ff8603ad5bf4598e5efba0d9a39b436/vllm/model_executor/models/mllama.py
"""PyTorch Mllama model."""  # PyTorch Mllama模型

from __future__ import annotations  # 启用延迟注解评估

import math  # 导入数学模块
from array import array  # 导入数组类型
from typing import Iterable, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入神经网络函数模块
import torch.utils.checkpoint  # 导入梯度检查点
import transformers.models.mllama.configuration_mllama as config_mllama  # 导入Mllama配置
from torch import nn  # 导入神经网络模块
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast  # 导入模型输出类
from transformers.models.mllama.modeling_mllama import (  # 导入Mllama建模工具
    _prepare_aspect_ratio_attention_mask,  # 准备宽高比注意力掩码
)

import sglang.srt.distributed.parallel_state as ps  # 导入并行状态
from sglang.srt.distributed import get_tensor_model_parallel_world_size  # 获取张量并行世界大小
from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取器
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化
from sglang.srt.layers.linear import (  # 导入线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.radix_attention import RadixAttention  # 导入基数注意力
from sglang.srt.layers.vocab_parallel_embedding import (  # 导入词表并行嵌入
    DEFAULT_VOCAB_PADDING_SIZE,  # 默认词表填充大小
    ParallelLMHead,  # 并行语言模型头
    VocabParallelEmbedding,  # 词表并行嵌入
)
from sglang.srt.managers.schedule_batch import MultimodalInputs  # 导入多模态输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaMLP  # 导入Llama解码器层和MLP
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """
    # 列并行Conv2D分块层，将图像分块并做列并行线性投影
    # 输入：(bsz, in_channels, width, height)，输出：(bsz, num_tokens, out_channels)

    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        kernel_size: Union[int, Tuple[int, int]],  # 卷积核大小
        stride: Union[int, Tuple[int, int]],  # 卷积步长
        bias: bool = False,  # 是否使用偏置
    ) -> None:
        super().__init__()  # 调用父类初始化
        if isinstance(kernel_size, int):  # 如果核大小是整数
            kernel_size = (kernel_size, kernel_size)  # 转换为元组
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)  # 展开操作，提取图像块
        self._linear = ColumnParallelLinear(  # 列并行线性投影
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：展开图像块 -> 线性投影"""
        x = self._unfold(x)  # 展开图像为块序列
        x = x.permute(0, 2, 1)  # 调整维度顺序
        x, _ = self._linear(x)  # 线性投影
        return x  # 返回投影后的特征


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    """预计算的宽高比嵌入，用于编码图像分块的空间布局信息"""

    def __init__(self, config: config_mllama.MllamaVisionConfig, is_gated: bool = True):
        super().__init__()  # 调用父类初始化
        self.max_num_tiles = config.max_num_tiles  # 最大分块数
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.max_aspect_ratio_id = config.max_aspect_ratio_id  # 最大宽高比ID
        self.is_gated = is_gated  # 是否使用门控

        self.embedding = nn.Embedding(  # 宽高比嵌入层
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size
        )
        if is_gated:  # 如果使用门控
            self.gate = nn.Parameter(torch.zeros(1))  # 门控参数，初始化为0

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        """前向传播：获取宽高比嵌入，可选门控调制，加到隐藏状态上"""
        embeddings = self.embedding(aspect_ratio_ids)  # 获取宽高比嵌入
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)  # 调整形状

        if self.is_gated:  # 使用门控时
            embeddings = embeddings * self.gate.tanh()  # 用tanh门控调制嵌入

        hidden_state = hidden_state + embeddings  # 将嵌入加到隐藏状态上
        return hidden_state  # 返回更新后的隐藏状态


class MllamaPrecomputedPositionEmbedding(nn.Module):
    """预计算的位置嵌入，包含patch位置嵌入和分块位置嵌入，使用门控融合"""

    def __init__(self, config: config_mllama.MllamaVisionConfig):
        super().__init__()  # 调用父类初始化
        self.max_num_tiles = config.max_num_tiles  # 最大分块数
        self.max_aspect_ratio_id = config.max_aspect_ratio_id  # 最大宽高比ID
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1  # patch数量加1个CLS token
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.scale = config.hidden_size**-0.5  # 缩放因子

        self.gate = nn.Parameter(torch.zeros(1))  # 门控参数

        # position embedding  # 位置嵌入
        position_embedding = torch.randn(self.num_patches, self.hidden_size)  # 随机初始化位置嵌入
        self.embedding = nn.Parameter(self.scale * position_embedding)  # 缩放后作为可学习参数

        # tile position embedding  # 分块位置嵌入
        self.tile_embedding = nn.Embedding(  # 分块位置嵌入层
            self.max_aspect_ratio_id + 1,
            self.max_num_tiles * self.num_patches * self.hidden_size,
        )

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        """前向传播：门控融合patch位置嵌入和分块位置嵌入"""
        # position embeddings  # patch位置嵌入
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding  # 门控patch位置嵌入
        hidden_state = hidden_state + gated_position_embedding.view(  # 加到隐藏状态上
            1, 1, self.num_patches, self.hidden_size
        )

        # precomputed tile position embeddings  # 预计算的分块位置嵌入
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)  # 获取分块位置嵌入
        batch_size = hidden_state.shape[0]  # 获取批次大小
        tile_position_embedding = tile_position_embedding.reshape(  # 调整形状
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding  # 门控分块位置嵌入
        hidden_state = hidden_state + gated_tile_position_embedding  # 加到隐藏状态上

        return hidden_state  # 返回更新后的隐藏状态


class MllamaVisionMLP(nn.Module):
    """Mllama视觉MLP层，包含两个线性层和激活函数"""

    def __init__(
        self,
        config,  # 视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.activation_fn = get_act_fn(config.hidden_act)  # 获取激活函数
        self.fc1 = ColumnParallelLinear(  # 第一个全连接层（列并行）
            config.hidden_size,
            config.intermediate_size,
            bias=True,  # 使用偏置
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(  # 第二个全连接层（行并行）
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP前向传播：fc1 -> 激活 -> fc2"""
        hidden_states, _ = self.fc1(hidden_states)  # 第一个全连接
        hidden_states = self.activation_fn(hidden_states)  # 激活函数
        hidden_states, _ = self.fc2(hidden_states)  # 第二个全连接

        return hidden_states  # 返回MLP输出


class MllamaVisionEncoderLayer(nn.Module):
    """Mllama视觉编码器层，包含自注意力和MLP，可选tanh门控"""

    def __init__(
        self,
        config: config_mllama.MllamaVisionConfig,  # 视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        is_gated: bool = False,  # 是否使用门控
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化

        self.hidden_size = config.hidden_size  # 隐藏大小
        self.num_attention_heads = config.attention_heads  # 注意力头数
        self.is_gated = is_gated  # 是否门控
        self.intermediate_size = config.intermediate_size  # 中间层大小

        self.self_attn = VisionAttention(  # 视觉自注意力
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size,
            use_qkv_parallel=True,  # 使用QKV并行
            quant_config=quant_config,
            flatten_batch=False,  # 不展平批次
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = MllamaVisionMLP(  # 视觉MLP
            config, quant_config, prefix=add_prefix("mlp", prefix)
        )

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)  # 输入层归一化
        self.post_attention_layernorm = nn.LayerNorm(  # 注意力后层归一化
            self.hidden_size, eps=config.norm_eps
        )

        # there used to be an if else here, no code path  # 此处之前有if/else分支，现已无此代码路径
        if is_gated:  # 如果使用门控
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)  # 注意力门控参数
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)  # FFN门控参数

    def forward(
        self,
        hidden_state: torch.Tensor,  # 隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
    ):
        """编码器层前向传播：自注意力 -> 残差 -> MLP -> 残差"""
        # Self Attention  # 自注意力
        residual = hidden_state  # 保存残差
        hidden_state = self.input_layernorm(hidden_state)  # 层归一化
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)  # 自注意力
        gate_attn = 1 if not self.is_gated else self.gate_attn.tanh()  # 计算注意力门控值
        hidden_state = residual + gate_attn * hidden_state  # 残差连接加门控

        # Feed forward  # 前馈网络
        residual = hidden_state  # 保存残差
        hidden_state = self.post_attention_layernorm(hidden_state)  # 层归一化
        hidden_state = self.mlp(hidden_state)  # MLP
        gate_ffn = 1 if not self.is_gated else self.gate_ffn.tanh()  # 计算FFN门控值
        hidden_state = residual + gate_ffn * hidden_state  # 残差连接加门控

        return hidden_state  # 返回编码器层输出


class MllamaVisionEncoder(nn.Module):
    """Mllama视觉编码器，堆叠多个视觉编码器层"""

    def __init__(
        self,
        config: config_mllama.MllamaVisionConfig,  # 视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        num_layers=32,  # 编码器层数
        is_gated=False,  # 是否使用门控
        output_hidden_states=None,  # 需要输出隐藏状态的层索引
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                MllamaVisionEncoderLayer(
                    config,
                    quant_config,
                    is_gated,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(num_layers)  # 按指定层数创建
            ]
        )
        self.output_hidden_states = output_hidden_states or []  # 需要输出隐藏状态的层

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
    ) -> Union[Tuple, BaseModelOutput]:
        """编码器前向传播：依次通过各编码器层，收集中间层隐藏状态"""
        encoder_states = ()  # 初始化编码器状态元组

        for i, encoder_layer in enumerate(self.layers):  # 遍历所有层
            if i in self.output_hidden_states:  # 如果当前层需要输出隐藏状态
                encoder_states = encoder_states + (hidden_states,)  # 收集隐藏状态
            hidden_states = encoder_layer(  # 通过编码器层
                hidden_states,
                attention_mask,
            )

        if len(self.layers) - 1 in self.output_hidden_states:  # 最后一层也需要收集
            encoder_states = encoder_states + (hidden_states,)

        return hidden_states, encoder_states  # 返回最终隐藏状态和中间状态


class MllamaVisionModel(nn.Module):
    """Mllama视觉模型，包含分块嵌入、位置编码、局部编码器和全局编码器"""

    def __init__(
        self,
        config: config_mllama.MllamaVisionConfig,  # 视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.image_size = config.image_size  # 图像大小
        self.patch_size = config.patch_size  # 分块大小
        self.max_num_tiles = config.max_num_tiles  # 最大分块数
        self.hidden_size = config.hidden_size  # 隐藏大小
        self.in_channels = config.num_channels  # 输入通道数
        self.intermediate_layers_indices = config.intermediate_layers_indices  # 中间层索引

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1  # patch数量
        self.scale = config.hidden_size**-0.5  # 缩放因子

        self.patch_embedding = ColumnParallelConv2dPatch(  # 分块嵌入层
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))  # CLS token嵌入
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)  # 门控位置嵌入

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(  # 编码器前的分块嵌入
            config, is_gated=True
        )
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(  # 编码器后的分块嵌入
            config, is_gated=True
        )

        # layer norms  # 层归一化
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)  # 编码器前归一化
        self.layernorm_post = nn.LayerNorm(self.hidden_size)  # 编码器后归一化

        # encoders  # 编码器
        self.transformer = MllamaVisionEncoder(  # 局部编码器
            config,
            quant_config,
            config.num_hidden_layers,
            is_gated=False,  # 局部编码器不使用门控
            output_hidden_states=config.intermediate_layers_indices,  # 输出中间层
            prefix=add_prefix("transformer", prefix),
        )
        self.global_transformer = MllamaVisionEncoder(  # 全局编码器
            config,
            quant_config,
            config.num_global_layers,
            is_gated=True,  # 全局编码器使用门控
            prefix=add_prefix("global_transformer", prefix),
        )

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """将CLS token嵌入拼接到序列开头"""
        batch_size, _, hidden_size = hidden_state.shape  # 获取形状
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)  # 扩展CLS token
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)  # 拼接CLS token
        return hidden_state  # 返回添加CLS后的隐藏状态

    def forward(
        self,
        pixel_values: torch.Tensor,  # 像素值
        aspect_ratio_ids: torch.Tensor,  # 宽高比ID
        aspect_ratio_mask: torch.Tensor,  # 宽高比掩码
    ) -> torch.Tensor:
        """视觉模型前向传播：分块嵌入 -> 位置编码 -> 局部编码器 -> 全局编码器 -> 拼接中间层"""
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = (
            pixel_values.shape  # 解包像素值形状
        )

        pixel_values = pixel_values.reshape(  # 展平批次和分块维度
            batch_size * num_concurrent_media * num_tiles, num_channels, height, width
        )
        aspect_ratio_ids = aspect_ratio_ids.reshape(  # 展平宽高比ID
            batch_size * num_concurrent_media, -1
        )

        # patch embedding  # 分块嵌入
        patch_embeds = self.patch_embedding(  # 分块嵌入
            pixel_values.to(self.layernorm_pre.weight.dtype)
        )
        hidden_state = patch_embeds  # 保存嵌入结果
        hidden_state = ps.get_tp_group().all_gather(hidden_state)  # 全收集张量并行结果

        # tile embeddings  # 分块嵌入
        _, num_patches, dim = hidden_state.shape  # 获取形状
        hidden_state = hidden_state.reshape(  # 重塑为分块形式
            batch_size * num_concurrent_media, num_tiles, -1, dim
        )
        hidden_state = self.pre_tile_positional_embedding(  # 应用编码器前的分块嵌入
            hidden_state, aspect_ratio_ids
        )

        # apply cls token  # 添加CLS token
        hidden_state = hidden_state.reshape(  # 展平
            batch_size * num_concurrent_media * num_tiles, num_patches, dim
        )
        hidden_state = self.apply_class_embedding(hidden_state)  # 拼接CLS token
        num_patches += 1  # patch数加1（CLS token）

        # apply position embeddings  # 应用位置嵌入
        hidden_state = hidden_state.reshape(  # 重塑形状
            batch_size * num_concurrent_media, num_tiles, num_patches, dim
        )
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)  # 门控位置嵌入

        # apply encoder  # 应用编码器
        hidden_state = self.layernorm_pre(hidden_state)  # 编码器前归一化

        # Compute the number of tokens to pad  # 计算需要填充的token数
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8  # 对齐到8的倍数
        # Compute padding tuple for pad function  # 计算填充参数
        padding = (
            0,
            0,
            0,
            num_padding_patches,
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)  # 填充参数
        # Pad the tensor  # 填充张量
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)  # 零填充
        slice_index = -num_padding_patches if num_padding_patches > 0 else None  # 切片索引

        attention_mask = aspect_ratio_mask.reshape(  # 重塑宽高比掩码
            batch_size * num_concurrent_media, -1
        )
        attention_mask = _prepare_aspect_ratio_attention_mask(  # 准备宽高比注意力掩码
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.layernorm_pre.weight.dtype,
        )

        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)  # 重塑形状
        output = self.transformer(  # 局部编码器
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state, intermediate_hidden_states = output[0], output[1]  # 分离输出和中间状态
        intermediate_hidden_states = torch.stack(intermediate_hidden_states, dim=-1)  # 堆叠中间状态

        # apply global encoder  # 应用全局编码器
        hidden_state = self.layernorm_post(hidden_state)  # 编码器后归一化
        hidden_state = hidden_state.reshape(  # 重塑为分块形式
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = self.post_tile_positional_embedding(  # 应用编码器后的分块嵌入
            hidden_state, aspect_ratio_ids
        )
        hidden_state = hidden_state.reshape(  # 展平分块和patch维度
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches),
            dim,
        )
        hidden_state = self.global_transformer(  # 全局编码器
            hidden_state, attention_mask=attention_mask
        )[0]
        hidden_state = hidden_state.reshape(  # 重塑回分块形式
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = hidden_state[:, :, :slice_index]  # 去掉填充部分

        # adding intermediate layer outputs  # 拼接中间层输出
        hidden_state = hidden_state.reshape(  # 重塑为原始批次形状
            batch_size, num_concurrent_media, num_tiles, num_patches, dim
        )
        intermediate_hidden_states = intermediate_hidden_states.reshape(  # 重塑中间状态
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            -1,
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]  # 去掉填充
        intermediate_hidden_states = intermediate_hidden_states.reshape(  # 重塑回原始形状
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)  # 拼接最终和中间层输出
        return hidden_state  # 返回视觉模型输出


class MllamaTextCrossAttention(nn.Module):
    """Mllama文本交叉注意力层，文本查询关注视觉键值"""

    def __init__(
        self,
        config: Optional[config_mllama.MllamaTextConfig] = None,  # 文本配置
        layer_id: Optional[int] = None,  # 层ID
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.model_parallel_size = get_tensor_model_parallel_world_size()  # 张量并行大小
        self.num_heads = self.config.num_attention_heads  # 注意力头数
        self.num_local_heads = self.num_heads // self.model_parallel_size  # 当前并行头数
        self.num_key_value_heads = self.config.num_key_value_heads  # KV头数
        self.num_local_key_value_heads = (  # 当前并行KV头数
            self.num_key_value_heads // self.model_parallel_size
        )
        self.dropout = config.dropout  # Dropout率
        self.hidden_size = config.hidden_size  # 隐藏大小
        self.head_dim = config.hidden_size // self.num_heads  # 每个头的维度
        self.layer_id = layer_id  # 层ID
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # KV分组数
        self.q_local_size = self.num_local_heads * self.head_dim  # 本地Q大小
        self.kv_local_size = self.num_local_key_value_heads * self.head_dim  # 本地KV大小

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(  # 输出投影
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,  # 输入已是并行的
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # Q归一化
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # K归一化
        self.scaling = self.head_dim**-0.5  # 缩放因子

        self.attn = RadixAttention(  # 基数注意力
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            self.num_local_key_value_heads,
            layer_id=layer_id,
            is_cross_attention=True,  # 交叉注意力模式
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # 文本隐藏状态
        attention_mask: Optional[torch.Tensor],  # 注意力掩码
        cross_attention_states: Optional[torch.Tensor],  # 交叉注意力状态（视觉特征）
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        """交叉注意力前向传播：文本Q关注视觉KV"""
        qkv_dec, _ = self.qkv_proj(hidden_states)  # 对解码器隐藏状态做QKV投影
        q, _, _ = qkv_dec.split(  # 只取Q部分
            [self.q_local_size, self.kv_local_size, self.kv_local_size], dim=-1
        )
        if cross_attention_states is None:  # 无视觉输入时
            k = None  # K为None
            v = None  # V为None
        else:  # 有视觉输入时
            qkv_enc, _ = self.qkv_proj(cross_attention_states)  # 对视觉状态做QKV投影
            _, k, v = qkv_enc.split(  # 取KV部分
                [self.q_local_size, self.kv_local_size, self.kv_local_size], dim=-1
            )
            k = k.view(-1, self.num_local_key_value_heads, self.head_dim)  # 重塑K形状
            v = v.view(-1, self.num_local_key_value_heads, self.head_dim)  # 重塑V形状
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(  # K归一化
                -1, self.num_local_key_value_heads, self.head_dim
            )
        q = q.view(-1, self.num_local_heads, self.head_dim)  # 重塑Q形状
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(  # Q归一化
            -1, self.num_local_heads, self.head_dim
        )

        output = self.attn(q, k, v, forward_batch)  # 计算交叉注意力
        output = output.view(-1, self.num_local_heads * self.head_dim)  # 重塑输出形状
        out, _ = self.o_proj(output)  # 输出投影
        return out  # 返回交叉注意力输出


class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention
    and feedforward."""
    # 交叉注意力解码器层，包含tanh门控的注意力和前馈网络

    def __init__(
        self,
        config: config_mllama.MllamaTextConfig,  # 文本配置
        layer_id: int,  # 层ID
        quant_config: Optional[QuantizationConfig],  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.layer_id = layer_id  # 保存层ID
        self.cross_attn = MllamaTextCrossAttention(  # 交叉注意力层
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("cross_attn", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 输入层归一化
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))  # 注意力门控参数

        self.mlp = LlamaMLP(  # MLP层
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.post_attention_layernorm = RMSNorm(  # 注意力后层归一化
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))  # MLP门控参数

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cross_attention_states: torch.Tensor,  # 交叉注意力状态
        cross_attention_mask: torch.Tensor,  # 交叉注意力掩码
        full_text_row_masked_out_mask: torch.Tensor,  # 文本行掩码
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> torch.Tensor:
        """交叉注意力解码器层前向传播：交叉注意力 -> 门控残差 -> MLP -> 门控残差"""
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 层归一化

        hidden_states = self.cross_attn(  # 交叉注意力
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            forward_batch=forward_batch,
        )
        hidden_states = full_text_row_masked_out_mask * hidden_states  # 应用文本行掩码
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states  # 门控残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.post_attention_layernorm(hidden_states)  # 层归一化
        hidden_states = self.mlp(hidden_states)  # MLP
        hidden_states = full_text_row_masked_out_mask * hidden_states  # 应用文本行掩码
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states  # 门控残差连接
        return hidden_states  # 返回解码器层输出


class MllamaTextModel(nn.Module):
    """Mllama文本模型，包含词嵌入和混合的自注意力/交叉注意力解码器层"""

    config_class = config_mllama.MllamaTextConfig  # 配置类
    base_model_prefix = "model"  # 基础模型前缀

    def __init__(
        self,
        config: config_mllama.MllamaTextConfig,  # 文本配置
        quant_config: Optional[QuantizationConfig],  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.padding_id = config.pad_token_id  # 填充token ID
        self.vocab_size = config.vocab_size  # 词表大小
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入层，词表+8用于特殊token
            config.vocab_size + 8,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.cross_attention_layers = config.cross_attention_layers  # 交叉注意力层索引

        layers = []  # 层列表
        for layer_id in range(config.num_hidden_layers):  # 遍历所有层
            if layer_id in self.cross_attention_layers:  # 交叉注意力层
                layers.append(
                    MllamaCrossAttentionDecoderLayer(
                        config,
                        layer_id,
                        quant_config=quant_config,
                        prefix=add_prefix(f"layers.{layer_id}", prefix),
                    )
                )
            else:  # 自注意力层
                # TODO: force LlamaDecoderLayer to config.attention_bias=False  # 待办：强制LlamaDecoderLayer使用attention_bias=False
                layers.append(
                    LlamaDecoderLayer(
                        config,
                        quant_config=quant_config,
                        layer_id=layer_id,
                        prefix=add_prefix(f"layers.{layer_id}", prefix),
                    )
                )

        self.layers = nn.ModuleList(layers)  # 创建模块列表
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化层

    def forward(
        self,
        input_ids: torch.LongTensor,  # 输入token ID
        positions: Optional[torch.LongTensor],  # 位置索引
        cross_attention_states: Optional[torch.LongTensor],  # 交叉注意力状态
        cross_attention_mask: Optional[torch.LongTensor],  # 交叉注意力掩码
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]],  # 文本行掩码
        forward_batch: ForwardBatch,  # 前向批次信息
        skip_cross_attention: bool,  # 是否跳过交叉注意力
    ) -> torch.Tensor:
        """文本模型前向传播：词嵌入 -> 解码器层 -> 归一化"""
        inputs_embeds = self.embed_tokens(input_ids)  # 词嵌入
        hidden_states = inputs_embeds  # 初始化隐藏状态

        for _, decoder_layer in enumerate(self.layers):  # 遍历所有解码器层
            if isinstance(decoder_layer, MllamaCrossAttentionDecoderLayer):  # 交叉注意力层
                if not skip_cross_attention:  # 不跳过交叉注意力
                    hidden_states = decoder_layer(
                        hidden_states=hidden_states,
                        cross_attention_states=cross_attention_states,
                        cross_attention_mask=cross_attention_mask,
                        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                        forward_batch=forward_batch,
                    )
            elif isinstance(decoder_layer, LlamaDecoderLayer):  # 自注意力层
                hidden_states, residual = decoder_layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=None,
                )
                hidden_states = hidden_states + residual  # 残差连接
            else:
                raise ValueError(f"Unknown decoder layer type {type(decoder_layer)}")  # 未知层类型
        hidden_states = self.norm(hidden_states)  # 最终归一化
        return hidden_states  # 返回文本模型输出


class MllamaForCausalLM(nn.Module):
    """Mllama因果语言模型，包装文本模型和语言模型头"""

    config_class = config_mllama.MllamaTextConfig  # 配置类
    base_model_prefix = "language_model"  # 基础模型前缀
    _no_split_modules = [  # 不可拆分模块
        "MllamaCrossAttentionDecoderLayer",
        "MllamaSelfAttentionDecoderLayer",
    ]

    def __init__(
        self,
        config: config_mllama.MllamaTextConfig,  # 文本配置
        quant_config: Optional[QuantizationConfig],  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.vocab_size = config.vocab_size  # 词表大小
        self.model = MllamaTextModel(  # 文本模型
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(  # 语言模型头
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,  # 输入token ID
        positions: Optional[torch.LongTensor],  # 位置索引
        cross_attention_states: Optional[torch.LongTensor],  # 交叉注意力状态
        cross_attention_mask: Optional[torch.LongTensor],  # 交叉注意力掩码
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]],  # 文本行掩码
        forward_batch: ForwardBatch,  # 前向批次信息
        skip_cross_attention: bool,  # 是否跳过交叉注意力
    ) -> torch.Tensor:
        """因果语言模型前向传播"""
        hidden_states = self.model(  # 文本模型前向传播
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
        )
        return hidden_states  # 返回隐藏状态


class MllamaForConditionalGeneration(nn.Module):
    """Mllama条件生成模型，整合视觉模型、语言模型和多模态投影器"""

    # BitandBytes specific attributes  # BitandBytes特定属性
    default_bitsandbytes_target_modules = [  # 默认BitandBytes目标模块
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)  # 张量并行中沿列维度切分的权重
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {  # BitandBytes堆叠参数映射
        # shard_name, weight_name, index  # 分片名, 权重名, 索引
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: config_mllama.MllamaConfig,  # Mllama配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.quant_config = quant_config  # 保存量化配置
        self.vocab_size = config.text_config.vocab_size  # 词表大小
        self.hidden_size = config.text_config.hidden_size  # 隐藏大小
        self.max_num_tiles = config.vision_config.max_num_tiles  # 最大分块数
        self.vision_output_dim = config.vision_config.vision_output_dim  # 视觉输出维度
        self.pad_token_id = (  # 填充token ID
            config.pad_token_id if config.pad_token_id is not None else -1
        )
        self.image_size = config.vision_config.image_size  # 图像大小

        self.vision_model = MllamaVisionModel(  # 视觉模型
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
        )
        self.language_model = MllamaForCausalLM(  # 语言模型
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.multi_modal_projector = ReplicatedLinear(  # 多模态投影器
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="multi_modal_projector",
        )
        self.logits_processor = LogitsProcessor(config.text_config)  # logits处理器

    def pad_input_ids(
        self, input_ids: array[int], mm_inputs: MultimodalInputs
    ) -> array[int]:
        """为输入ID添加多模态填充token前缀"""
        pixel_values = torch.cat([item.feature for item in mm_inputs.mm_items], dim=0)  # 拼接像素值
        pad_values = array("q", (item.pad_value for item in mm_inputs.mm_items))  # 填充值

        num_concurrent_media, num_tiles = pixel_values.shape[1:3]  # 获取媒体和分块数量
        num_patches = self.vision_model.num_patches  # patch数量
        image_len = num_concurrent_media * num_tiles * num_patches  # 图像token总长度
        mm_inputs.num_image_tokens = image_len  # 设置图像token数

        pad_ids = pad_values * ((image_len + len(pad_values)) // len(pad_values))  # 扩展填充ID

        return pad_ids[:image_len] + input_ids  # 返回填充后的输入ID

    def _batch_image_inputs(self, forward_batch: ForwardBatch):
        """批处理图像输入，将多个请求的图像整理为统一批次"""
        if forward_batch.forward_mode.is_decode() or all(forward_batch.encoder_cached):  # 解码模式或全部已缓存
            return None, None, None, None  # 返回None

        # pixel_values: shape (bs, num_image, num_tiles, 3, image_res, image_res)  # 像素值形状
        max_num_images = max_num_tiles = bs = 0  # 初始化统计值
        for i, mm_input in enumerate(forward_batch.mm_inputs):  # 遍历多模态输入

            if not forward_batch.encoder_cached[i] and mm_input is not None:  # 未缓存且有输入
                pixel_values = torch.cat(  # 拼接像素值
                    [item.feature for item in mm_input.mm_items], dim=0
                )
                max_num_images = max(max_num_images, pixel_values.shape[1])  # 更新最大图像数

                max_num_tiles = max(max_num_tiles, pixel_values.shape[2])  # 更新最大分块数
                bs += 1  # 批次大小加1

        if max_num_images * max_num_tiles * bs == 0:  # 无图像输入
            return None, None, None, None  # 返回None

        with forward_batch.out_cache_loc.device:  # 在缓存设备上
            batched_images = torch.zeros(  # 初始化批处理图像张量
                bs,
                max_num_images,
                max_num_tiles,
                3,
                self.image_size,
                self.image_size,
                dtype=torch.float32,
            )
            batched_ar_ids = torch.ones(bs, max_num_images, dtype=torch.int64)  # 宽高比ID
            batched_ar_mask = torch.zeros(  # 宽高比掩码
                bs, max_num_images, max_num_tiles, dtype=torch.int64
            )
            i = 0  # 批次索引
            encoder_lens_need = []  # 需要计算的编码器长度

            for k, mm_input in enumerate(forward_batch.mm_inputs):  # 遍历多模态输入
                if forward_batch.encoder_cached[k] or mm_input is None:  # 已缓存或无输入
                    continue  # 跳过

                encoder_lens_need.append(forward_batch.encoder_lens[k])  # 记录编码器长度
                pixel_values = torch.cat(  # 拼接像素值
                    [item.feature for item in mm_input.mm_items], dim=0
                )
                for j in range(pixel_values.shape[1]):  # 遍历图像
                    img = pixel_values[0, j]  # 获取图像
                    num_tiles = img.shape[0]  # 分块数
                    batched_images[i, j, :num_tiles] = img  # 填充到批次中
                    batched_ar_ids[i, j] = mm_input.mm_items[0].model_specific_data[  # 宽高比ID
                        "aspect_ratio_ids"
                    ][0, j]

                    batched_ar_mask[i, j, :num_tiles] = mm_input.mm_items[  # 宽高比掩码
                        0
                    ].model_specific_data["aspect_ratio_mask"][0, j]
                i += 1  # 批次索引加1

        return batched_images, batched_ar_ids, batched_ar_mask, encoder_lens_need  # 返回批处理结果

    def flat_encoder_result(
        self, cross_attention_states: torch.Tensor, encoder_lens_need: List[int]
    ):
        """将编码器结果展平为连续的一维序列"""
        # NOTE: not all encoders need computation, some are cached  # 注意：不是所有编码器都需要计算，有些已缓存
        head_dim = cross_attention_states.shape[-1]  # 头维度
        total_encoder_len = sum(encoder_lens_need)  # 总编码器长度
        cross_attention_states_flat = torch.zeros(  # 初始化展平张量
            total_encoder_len,
            head_dim,
            device=cross_attention_states.device,
            dtype=cross_attention_states.dtype,
        )

        i = start_pos = 0  # 初始化索引
        for encoder_len in encoder_lens_need:  # 遍历编码器长度
            if encoder_len == 0:  # 长度为0跳过
                continue
            end_pos = start_pos + encoder_len  # 计算结束位置
            cross_attention_states_flat[start_pos:end_pos] = cross_attention_states[i][  # 复制对应片段
                :encoder_len
            ]
            i += 1  # 请求索引加1
            start_pos += encoder_len  # 更新起始位置

        return cross_attention_states_flat  # 返回展平结果

    def get_full_text_row_masked_out_mask(self, forward_batch: ForwardBatch):
        """获取文本行掩码，标记哪些token可以看到视觉token"""
        if forward_batch.forward_mode.is_decode():  # 解码模式
            full_text_row_masked_out_mask = forward_batch.encoder_lens != 0  # 有编码器长度的为True
        else:  # 预填充模式
            full_text_row_masked_out_mask = torch.ones(  # 初始化为全True
                forward_batch.extend_seq_lens.sum(), dtype=torch.bool
            )
            start_pos = 0  # 起始位置

            for seq_len, encoder_len in zip(  # 遍历序列
                forward_batch.seq_lens.tolist(), forward_batch.encoder_lens_cpu
            ):
                if encoder_len == 0:  # 无视觉输入
                    full_text_row_masked_out_mask[start_pos : start_pos + seq_len] = (  # 标记为False
                        False
                    )
                start_pos += encoder_len  # 更新起始位置

            full_text_row_masked_out_mask = full_text_row_masked_out_mask.to(  # 移到目标设备
                forward_batch.seq_lens.device
            )

        return full_text_row_masked_out_mask.reshape(-1, 1)  # 重塑为列向量

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入token ID
        positions: torch.Tensor,  # 位置索引
        forward_batch: ForwardBatch,  # 前向批次信息
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """条件生成模型前向传播：视觉编码 -> 投影 -> 语言模型"""
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode  # 延迟导入

        batched_images, batched_ar_ids, batched_ar_mask, encoder_lens_need = (
            self._batch_image_inputs(forward_batch)  # 批处理图像输入
        )

        # TODO: support multi-image by this mask  # 待办：通过此掩码支持多图像
        cross_attention_mask = None  # 交叉注意力掩码
        cross_attention_states = None  # 交叉注意力状态

        if get_is_capture_mode():  # CUDA图捕获模式
            # NOTE: when doing cuda graph capture, we do not want to skip cross attention  # 注意：CUDA图捕获时不跳过交叉注意力
            # Make is a constant value to avoid cuda graph capture issue  # 设为常量避免CUDA图捕获问题
            skip_cross_attention = False
        else:  # 正常模式
            # NOTE: we do not need image_inputs when prefill  # 注意：预填充时不需要图像输入
            assert len(forward_batch.encoder_lens) == len(forward_batch.seq_lens)  # 断言长度一致
            assert len(forward_batch.encoder_lens_cpu) == len(forward_batch.seq_lens)  # 断言长度一致
            skip_cross_attention = forward_batch.encoder_lens.max() == 0  # 无视觉输入时跳过交叉注意力

        if not skip_cross_attention:  # 不跳过交叉注意力
            full_text_row_masked_out_mask = self.get_full_text_row_masked_out_mask(  # 获取文本行掩码
                forward_batch
            )
        else:  # 跳过交叉注意力
            full_text_row_masked_out_mask = None  # 掩码为None

        if batched_images is not None:  # 有图像输入
            # NOTE: llama's reference implementation runs vision model on CPU  # 注意：Llama参考实现在CPU上运行视觉模型
            cross_attention_states = self.vision_model(  # 视觉模型编码
                batched_images, batched_ar_ids, batched_ar_mask
            )
            cross_attention_states, _ = self.multi_modal_projector(  # 多模态投影
                cross_attention_states
            )

            bs, _, _, _, image_token_dim = cross_attention_states.shape  # 获取形状
            cross_attention_states = cross_attention_states.view(  # 展平分块维度
                bs, -1, image_token_dim
            )

            cross_attention_states = self.flat_encoder_result(  # 展平编码器结果
                cross_attention_states, encoder_lens_need
            )

        hidden_states = self.language_model(  # 语言模型前向传播
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
        )
        return self.logits_processor(  # 处理logits
            input_ids, hidden_states, self.language_model.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理分块嵌入和堆叠参数映射"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())  # 参数字典
        updated_params = set()  # 已更新参数集合
        for name, loaded_weight in weights:  # 遍历权重
            if "patch_embedding.weight" in name:  # 分块嵌入权重特殊处理
                name = name.replace(  # 替换名称
                    "patch_embedding.weight", "patch_embedding._linear.weight"
                )
                loaded_weight = loaded_weight.view(loaded_weight.shape[0], -1)  # 重塑权重形状
            for param_name, weight_name, shard_id in stacked_params_mapping:  # 检查堆叠参数
                if weight_name not in name:  # 不匹配则跳过
                    continue
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                param = params_dict[name]  # 获取参数
                updated_params.add(name)  # 记录已更新
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break
            else:  # 非堆叠参数
                if "vision_model" in name:  # 视觉模型权重
                    # adapt to VisionAttention  # 适配VisionAttention
                    name = name.replace("self_attn.o_proj", "self_attn.proj")  # 替换输出投影名称
                param = params_dict.pop(name)  # 获取并移除参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = MllamaForConditionalGeneration  # 入口类，用于模型注册
