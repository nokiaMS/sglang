# 文件说明：DeepSeek OCR多模态因果语言模型实现
# 本文件实现了DeepSeek OCR模型，包含SAM视觉编码器、CLIP视觉模型、
# Qwen2解码器即编码器（Decoder-as-Encoder）、MLP投影器等组件，
# 支持OCR1和OCR2两种视觉编码方案，用于文档图像理解和OCR任务。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The SwissAI Initiative
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

# Adapted from
# 适配自
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only Apertus model compatible with HuggingFace weights."""  # 仅推理的Apertus模型，兼容HuggingFace权重

import copy  # 导入深拷贝模块
import logging  # 导入日志模块
import math  # 导入数学模块
from functools import partial  # 导入偏函数
from typing import Iterable, List, Optional, Set, Tuple, Type, TypeAlias, Union  # 导入类型注解

import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入PyTorch神经网络功能模块
import transformers  # 导入Transformers库
from torch import Tensor, nn  # 导入张量和神经网络模块
from transformers.models.vitdet.modeling_vitdet import get_rel_pos  # 导入ViTDet相对位置编码

from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config  # 导入DeepSeek OCR视觉语言配置
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式（多模态令牌）
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs  # 导入多模态数据项和输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.deepseek import DeepseekForCausalLM  # 导入DeepSeek因果语言模型
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM  # 导入DeepSeek V2/V3模型
from sglang.srt.models.transformers import maybe_prefix  # 导入前缀工具
from sglang.srt.utils import cpu_has_amx_support, is_cpu  # 导入CPU AMX支持和CPU判断

_is_cpu_amx_available = cpu_has_amx_support()  # CPU AMX指令集是否可用
_is_cpu = is_cpu()  # 是否在CPU上运行

NestedTensors: TypeAlias = Union[  # 嵌套张量类型别名
    list["NestedTensors"],  # 嵌套列表
    list["torch.Tensor"],  # 张量列表
    "torch.Tensor",  # 张量
    tuple["torch.Tensor", ...],  # 张量元组
]

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]  # 多模态嵌入类型别名

logger = logging.getLogger(__name__)  # 获取日志记录器


# 递归展平并拼接嵌套张量
def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """
    """递归地在除最后维度外的所有维度上展平并拼接嵌套张量。"""

    if isinstance(embeddings, torch.Tensor):  # 如果是张量
        # Flatten all but the last dimension.
        # 展平除最后维度外的所有维度。
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))  # 递归展平并拼接


# 构建嵌套张量中嵌入数量的调试表达式
def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """
    """构建嵌套张量中嵌入数量的调试表示。"""

    if isinstance(embeddings, torch.Tensor):  # 如果是张量
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])  # 用x连接维度

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)  # 递归构建


# 将多模态嵌入合并到输入嵌入中
def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,  # 输入嵌入
    multimodal_embeddings: NestedTensors,  # 多模态嵌入
    is_multimodal: torch.Tensor,  # 多模态标记张量
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    """通过覆写`inputs_embeds`中对应`input_ids`中占位符令牌的位置，
    将`multimodal_embeddings`合并到`inputs_embeds`中。

    注意：
        此操作原地更新`inputs_embeds`。
    """
    if len(multimodal_embeddings) == 0:  # 如果没有多模态嵌入
        return inputs_embeds  # 直接返回

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)  # 展平多模态嵌入
    input_dtype = inputs_embeds.dtype  # 保存输入数据类型

    try:
        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        # 注意：这可以避免D2H同步（#22105），但如果
        # is_multimodal.sum() < len(mm_embeds_flat)则无法抛出错误
        inputs_embeds.masked_scatter_(  # 使用掩码散射操作
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:  # 捕获运行时错误
        num_actual_tokens = len(mm_embeds_flat)  # 实际令牌数
        num_expected_tokens = is_multimodal.sum().item()  # 期望令牌数

        if num_actual_tokens != num_expected_tokens:  # 如果数量不匹配
            expr = _embedding_count_expression(multimodal_embeddings)  # 构建调试表达式

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e  # 掩码散射操作错误

    return inputs_embeds  # 返回合并后的嵌入


# 判断元素是否在列表中
def isin_list(
    elements: torch.Tensor,  # 输入元素张量
    test_elements_list: list[int],  # 测试元素列表
) -> torch.Tensor:
    use_pin = torch.cuda.is_available() and not getattr(torch.version, "hip", None)  # 是否使用固定内存
    test_elements = torch.tensor(test_elements_list, pin_memory=use_pin).to(  # 创建测试元素张量
        device=elements.device, non_blocking=use_pin
    )

    return torch.isin(elements, test_elements)  # 返回成员判断结果


# 合并多模态嵌入到输入嵌入中（支持多占位符令牌）
def merge_multimodal_embeddings(
    input_ids: torch.Tensor,  # 输入令牌ID
    inputs_embeds: torch.Tensor,  # 输入嵌入
    multimodal_embeddings: NestedTensors,  # 多模态嵌入
    placeholder_token_id: int | list[int],  # 占位符令牌ID
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    `placeholder_token_id` can be a list of token ids (e.g, token ids
    of img_start, img_break, and img_end tokens) when needed: This means
    the order of these tokens in the `input_ids` MUST MATCH the order of
    their embeddings in `multimodal_embeddings` since we need to
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.

    Then the image embeddings (that correspond to I's) from vision encoder
    must be padded with embeddings of S, B, and E in the same order of
    input_ids for a correct embedding merge.

    Note:
        This updates `inputs_embeds` in place.
    """
    """通过覆写`inputs_embeds`中对应`input_ids`中占位符令牌的位置，
    将`multimodal_embeddings`合并到`inputs_embeds`中。

    `placeholder_token_id`可以是令牌ID列表（例如img_start、img_break
    和img_end令牌的ID）：这意味着这些令牌在`input_ids`中的顺序
    必须与它们在`multimodal_embeddings`中的嵌入顺序匹配，
    因为我们需要切片合并而不是单独散射。

    例如，如果input_ids是"TTTTTSIIIBIIIBIIIETTT"，其中
    - T是文本令牌
    - S是图像起始令牌
    - I是图像嵌入令牌
    - B是图像断行令牌
    - E是图像结束令牌。

    那么来自视觉编码器的图像嵌入（对应I）必须按照input_ids的
    相同顺序用S、B和E的嵌入进行填充，才能正确合并嵌入。

    注意：
        此操作原地更新`inputs_embeds`。
    """
    if isinstance(placeholder_token_id, list):  # 如果占位符是列表
        is_multimodal = isin_list(input_ids, placeholder_token_id)  # 判断哪些位置是多模态
    else:  # 单个占位符
        is_multimodal = input_ids == placeholder_token_id  # 直接比较

    return _merge_multimodal_embeddings(  # 调用内部合并函数
        inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )


# MLP投影器模块
class MlpProjector(nn.Module):

    def __init__(
        self,
        projector_type,  # 投影器类型
        input_dim,  # 输入维度
        n_embed,  # 嵌入维度
        depth=1,  # 深度
        mlp_ratio=1,  # MLP比率
        downsample_ratio=4,  # 下采样比率
    ):
        self.projector_type = projector_type  # 投影器类型
        self.input_dim = input_dim  # 输入维度
        self.n_embed = n_embed  # 嵌入维度
        self.depth = depth  # 深度
        self.token_pooling = False  # 令牌池化标志
        self.conv_fusion_high_low_features = False  # 卷积融合高低特征标志

        super().__init__()  # 调用父类初始化

        if projector_type == "identity":  # 恒等投影
            modules = nn.Identity()

        elif projector_type == "linear":  # 线性投影
            modules = nn.Linear(input_dim, n_embed)

        elif projector_type == "mlp_gelu":  # MLP+GELU投影
            mlp_depth = depth
            modules = [nn.Linear(input_dim, n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "normlayer_downsample_mlp_gelu":  # 带归一化的下采样MLP
            mlp_depth = depth
            mlp_ratio = mlp_ratio
            modules = [
                nn.LayerNorm(input_dim * downsample_ratio * downsample_ratio),  # 层归一化
                nn.Linear(  # 线性层
                    input_dim * downsample_ratio * downsample_ratio,
                    n_embed * mlp_ratio,
                ),
            ]
            for _ in range(1, mlp_depth - 1):  # 隐藏层
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed * mlp_ratio, n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(n_embed * mlp_ratio, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "downsample_mlp_gelu":  # 下采样MLP
            mlp_depth = depth
            mlp_ratio = mlp_ratio
            modules = [
                nn.Linear(  # 线性层
                    input_dim * downsample_ratio * downsample_ratio,
                    n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):  # 隐藏层
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed * mlp_ratio, n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(n_embed * mlp_ratio, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "low_high_hybrid_split_mlp_gelu":  # 低高分辨率混合分割MLP
            mlp_depth = depth
            self.high_up_proj = nn.Linear(input_dim, n_embed // 2)  # 高分辨率上投影
            self.low_up_proj = nn.Linear(input_dim, n_embed // 2)  # 低分辨率上投影

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "hybrid_split_feature_mlp_gelu":  # 混合分割特征MLP
            mlp_depth = depth
            channel_div = 0.5  # 通道分割比例
            self.high_up_proj = nn.Linear(input_dim[0], int(n_embed * channel_div))  # 高分辨率上投影
            self.low_up_proj = nn.Linear(  # 低分辨率上投影
                input_dim[1], n_embed - int(n_embed * channel_div)
            )

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "low_high_split_mlp_gelu":  # 低高分割MLP
            mlp_depth = depth
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed // 2, n_embed // 2))
            modules = nn.Sequential(*modules)
            self.high_layers = nn.Sequential(*modules)  # 高分辨率层
            self.low_layers = copy.deepcopy(modules)  # 低分辨率层（深拷贝）

        else:  # 未知投影类型
            raise ValueError(f"Unknown projector type: {projector_type}")

        self.layers = modules  # 投影层

    # 前向传播：MLP投影器
    def forward(self, x):
        if self.token_pooling:  # 如果启用令牌池化
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # 展开为2x2块
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # Concatenate on channel dimension
            # 在通道维度上拼接
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )

            # Pass through linear layer
            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)  # 令牌池化层

        if self.conv_fusion_high_low_features:  # 如果启用卷积融合高低特征
            x = self.fusion_layer(x[:, 0]) + x[:, 1]

        if self.projector_type == "low_high_hybrid_split_mlp_gelu":  # 低高混合分割
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.projector_type == "hybrid_split_feature_mlp_gelu":  # 混合分割特征
            high_x = x[..., : self.input_dim[0]]  # 提取高分辨率部分
            low_x = x[..., self.input_dim[0] :]  # 提取低分辨率部分
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.projector_type == "low_high_split_mlp_gelu":  # 低高分割
            high_x, low_x = x[0], x[1]
            high_x = self.high_layers(high_x)
            low_x = self.low_layers(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
            return x

        if (  # 下采样类型
            self.projector_type == "downsample_mlp_gelu"
            or self.projector_type == "normlayer_downsample_mlp_gelu"
        ):
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            """计算填充"""
            if h % self.downsample_ratio:  # 如果高度不能被下采样比率整除
                pad = self.downsample_ratio - h % self.downsample_ratio
            else:  # 无需填充
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:  # 如果需要填充
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            """4合1拼接"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(  # 展开操作
                x,
                kernel_size=self.downsample_ratio,
                stride=self.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)  # 返回投影结果


# 2D层归一化模块
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # 缩放权重
        self.bias = nn.Parameter(torch.zeros(num_channels))  # 偏置
        self.eps = eps  # epsilon值

    # 前向传播：2D层归一化
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)  # 计算均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 缩放和偏移
        return x


# MLP块模块
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,  # 嵌入维度
        mlp_dim: int,  # MLP维度
        act: Type[nn.Module] = nn.GELU,  # 激活函数
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)  # 第一个线性层
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)  # 第二个线性层
        self.act = act()  # 激活函数

    # 前向传播：MLP块
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))  # 两层MLP


# 添加分解相对位置编码
def add_decomposed_rel_pos(
    q: torch.Tensor,  # 查询张量
    rel_pos_h: torch.Tensor,  # 高度轴相对位置嵌入
    rel_pos_w: torch.Tensor,  # 宽度轴相对位置嵌入
    q_size: Tuple[int, int],  # 查询空间尺寸
    k_size: Tuple[int, int],  # 键空间尺寸
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    """从:paper:`mvitv2`计算分解的相对位置嵌入。
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    参数：
        q (张量): 注意力层中的查询q，形状为(B, q_h * q_w, C)。
        rel_pos_h (张量): 高度轴的相对位置嵌入(Lh, C)。
        rel_pos_w (张量): 宽度轴的相对位置嵌入(Lw, C)。
        q_size (元组): 查询q的空间序列尺寸(q_h, q_w)。
        k_size (元组): 键k的空间序列尺寸(k_h, k_w)。
    返回：
        attn (张量): 添加了相对位置嵌入的注意力图。
    """
    q_h, q_w = q_size  # 查询高度和宽度
    k_h, k_w = k_size  # 键高度和宽度
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # 获取高度轴相对位置
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # 获取宽度轴相对位置

    B, _, dim = q.shape  # 批次、长度、维度
    r_q = q.reshape(B, q_h, q_w, dim)  # 重塑查询
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)  # 高度轴相对位置贡献
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)  # 宽度轴相对位置贡献
    rel_h = rel_h.unsqueeze(-1)  # 增加维度
    rel_w = rel_w.unsqueeze(-2)  # 增加维度
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)  # 重塑
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)  # 重塑

    return rel_h, rel_w  # 返回高度和宽度相对位置


# 带相对位置编码的多头注意力模块
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""
    """带相对位置嵌入的多头注意力块。"""

    def __init__(
        self,
        dim: int,  # 输入维度
        num_heads: int = 8,  # 头数
        qkv_bias: bool = True,  # QKV偏置
        use_rel_pos: bool = False,  # 是否使用相对位置编码
        rel_pos_zero_init: bool = True,  # 相对位置零初始化
        input_size: Optional[Tuple[int, int]] = None,  # 输入分辨率
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.  # 输入通道数。
            num_heads (int): Number of attention heads.  # 注意力头数。
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.  # 如果为True，为Q/K/V添加可学习偏置。
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.  # 如果为True，零初始化相对位置参数。
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.  # 计算相对位置参数大小的输入分辨率。
        """
        super().__init__()
        self.num_heads = num_heads  # 头数
        head_dim = dim // num_heads  # 每头维度
        self.scale = head_dim**-0.5  # 缩放因子

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # QKV线性层
        self.proj = nn.Linear(dim, dim)  # 输出投影

        self.use_rel_pos = use_rel_pos  # 是否使用相对位置
        if self.use_rel_pos:  # 如果使用相对位置
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            # 初始化相对位置嵌入
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))  # 高度轴相对位置
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))  # 宽度轴相对位置

    # 前向传播：注意力模块
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape  # 批次、高度、宽度、通道
        # qkv with shape (3, B, nHead, H * W, C)
        # QKV形状为(3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        # Q, K, V形状为(B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:  # 如果使用相对位置
            rel_h, rel_w = add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        q = q.view(B, self.num_heads, H * W, -1)  # 重塑Q
        k = k.view(B, self.num_heads, H * W, -1)  # 重塑K
        v = v.view(B, self.num_heads, H * W, -1)  # 重塑V

        if self.use_rel_pos:  # 如果使用相对位置
            rel_h = rel_h.view(
                B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
            )
            rel_w = rel_w.view(
                B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
            )
            attn_bias = (rel_h + rel_w).view(  # 合并相对位置偏差
                B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
            )
            x = torch.nn.functional.scaled_dot_product_attention(  # 带偏差的缩放点积注意力
                q, k, v, attn_mask=attn_bias
            )
            # x = _attention_rel_h_rel_w(q, k, v, rel_h, rel_w)
        else:  # 不使用相对位置
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = (  # 重塑输出
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        x = self.proj(x)  # 输出投影

        return x  # 返回注意力输出


# 窗口分区函数
def window_partition(
    x: torch.Tensor, window_size: int  # 输入张量和窗口大小
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    """分区为非重叠窗口，如果需要则填充。
    参数：
        x (张量): 输入令牌，形状[B, H, W, C]。
        window_size (int): 窗口大小。
    返回：
        windows: 分区后的窗口，形状[B * num_windows, window_size, window_size, C]。
        (Hp, Wp): 分区前的填充高度和宽度
    """
    B, H, W, C = x.shape  # 批次、高度、宽度、通道

    pad_h = (window_size - H % window_size) % window_size  # 高度填充量
    pad_w = (window_size - W % window_size) % window_size  # 宽度填充量
    if pad_h > 0 or pad_w > 0:  # 如果需要填充
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w  # 填充后的高度和宽度

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)  # 重塑
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  # 排列并展平
    )
    return windows, (Hp, Wp)  # 返回窗口和填充尺寸


# 窗口逆分区函数
def window_unpartition(
    windows: torch.Tensor,  # 窗口张量
    window_size: int,  # 窗口大小
    pad_hw: Tuple[int, int],  # 填充后高宽
    hw: Tuple[int, int],  # 原始高宽
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    """窗口逆分区为原始序列并移除填充。
    参数：
        windows (张量): 输入令牌，形状[B * num_windows, window_size, window_size, C]。
        window_size (int): 窗口大小。
        pad_hw (元组): 填充后的高度和宽度(Hp, Wp)。
        hw (元组): 填充前的原始高度和宽度(H, W)。
    返回：
        x: 逆分区后的序列，形状[B, H, W, C]。
    """
    Hp, Wp = pad_hw  # 填充后高宽
    H, W = hw  # 原始高宽
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)  # 计算批次大小
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)  # 重塑

    if Hp > H or Wp > W:  # 如果有填充则裁剪
        x = x[:, :H, :W, :].contiguous()
    return x  # 返回逆分区结果


# Transformer块，支持窗口注意力和残差传播
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""
    """支持窗口注意力和残差传播块的Transformer块"""

    def __init__(
        self,
        dim: int,  # 嵌入维度
        num_heads: int,  # 头数
        mlp_ratio: float = 4.0,  # MLP比率
        qkv_bias: bool = True,  # QKV偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,  # 归一化层
        act_layer: Type[nn.Module] = nn.GELU,  # 激活层
        use_rel_pos: bool = False,  # 是否使用相对位置
        rel_pos_zero_init: bool = True,  # 相对位置零初始化
        window_size: int = 0,  # 窗口大小
        input_size: Optional[Tuple[int, int]] = None,  # 输入分辨率
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.  # 输入通道数。
            num_heads (int): Number of attention heads in each ViT block.  # 每个ViT块中的注意力头数。
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.  # MLP隐藏维度与嵌入维度的比率。
            qkv_bias (bool): If True, add a learnable bias to query, key, value.  # 如果为True，为Q/K/V添加可学习偏置。
            norm_layer (nn.Module): Normalization layer.  # 归一化层。
            act_layer (nn.Module): Activation layer.  # 激活层。
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.  # 如果为True，为注意力图添加相对位置嵌入。
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.  # 如果为True，零初始化相对位置参数。
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.  # 窗口注意力的窗口大小。如果等于0，则使用全局注意力。
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.  # 计算相对位置参数大小的输入分辨率。
        """
        super().__init__()
        self.norm1 = norm_layer(dim)  # 第一个归一化
        self.attn = Attention(  # 注意力模块
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)  # 第二个归一化
        self.mlp = MLPBlock(  # MLP块
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size  # 窗口大小

    # 前向传播：Transformer块
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # 保存残差
        x = self.norm1(x)  # 归一化
        # Window partition
        # 窗口分区
        if self.window_size > 0:  # 如果使用窗口注意力
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)  # 注意力计算
        # Reverse window partition
        # 窗口逆分区
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x  # 注意力残差连接
        x = x + self.mlp(self.norm2(x))  # MLP残差连接

        return x  # 返回输出


# 补丁嵌入模块
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    """
    图像到补丁嵌入。
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),  # 卷积核大小
        stride: Tuple[int, int] = (16, 16),  # 步长
        padding: Tuple[int, int] = (0, 0),  # 填充
        in_chans: int = 3,  # 输入通道数
        embed_dim: int = 768,  # 嵌入维度
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.  # 投影层的卷积核大小。
            stride (Tuple): stride of the projection layer.  # 投影层的步长。
            padding (Tuple): padding size of the projection layer.  # 投影层的填充大小。
            in_chans (int): Number of input image channels.  # 输入图像通道数。
            embed_dim (int): Patch embedding dimension.  # 补丁嵌入维度。
        """
        super().__init__()

        self.proj = nn.Conv2d(  # 投影卷积层
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    # 前向传播：补丁嵌入
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # 卷积投影
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)  # 转置为HWC格式
        return x


# 获取SAM风格的绝对位置编码
def get_abs_pos_sam(abs_pos, tgt_size):  # 绝对位置编码和目标大小
    dtype = abs_pos.dtype  # 保存数据类型

    src_size = abs_pos.size(1)  # 源大小

    if src_size != tgt_size:  # 如果需要插值
        old_pos_embed = abs_pos.permute(0, 3, 1, 2)  # 转置
        old_pos_embed = old_pos_embed.to(torch.float32)  # 转为float32
        new_pos_embed = F.interpolate(  # 双三次插值
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)  # 转回
        return new_pos_embed
    else:  # 无需插值
        return abs_pos


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
# 此类及其支持函数轻度适配自ViTDet主干：https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
# 图像编码器ViT模块
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,  # 输入图像大小
        patch_size: int = 16,  # 补丁大小
        in_chans: int = 3,  # 输入通道数
        embed_dim: int = 768,  # 嵌入维度
        depth: int = 12,  # ViT深度
        num_heads: int = 12,  # 注意力头数
        mlp_ratio: float = 4.0,  # MLP比率
        out_chans: int = 256,  # 输出通道数
        qkv_bias: bool = True,  # QKV偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,  # 归一化层
        act_layer: Type[nn.Module] = nn.GELU,  # 激活层
        use_abs_pos: bool = True,  # 是否使用绝对位置编码
        use_rel_pos: bool = False,  # 是否使用相对位置编码
        rel_pos_zero_init: bool = True,  # 相对位置零初始化
        window_size: int = 0,  # 窗口大小
        global_attn_indexes: Tuple[int, ...] = (),  # 全局注意力层索引
        net_3_out_channels: int = 1024,  # net_3输出通道数
    ) -> None:
        """
        Args:
            img_size (int): Input image size.  # 输入图像大小。
            patch_size (int): Patch size.  # 补丁大小。
            in_chans (int): Number of input image channels.  # 输入图像通道数。
            embed_dim (int): Patch embedding dimension.  # 补丁嵌入维度。
            depth (int): Depth of ViT.  # ViT深度。
            num_heads (int): Number of attention heads in each ViT block.  # 每个ViT块的注意力头数。
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.  # MLP隐藏维度与嵌入维度的比率。
            qkv_bias (bool): If True, add a learnable bias to query, key, value.  # 如果为True，为Q/K/V添加可学习偏置。
            norm_layer (nn.Module): Normalization layer.  # 归一化层。
            act_layer (nn.Module): Activation layer.  # 激活层。
            use_abs_pos (bool): If True, use absolute positional embeddings.  # 如果为True，使用绝对位置嵌入。
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.  # 如果为True，为注意力图添加相对位置嵌入。
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.  # 如果为True，零初始化相对位置参数。
            window_size (int): Window size for window attention blocks.  # 窗口注意力的窗口大小。
            global_attn_indexes (list): Indexes for blocks using global attention.  # 使用全局注意力的块索引。
        """
        super().__init__()
        self.img_size = img_size  # 图像大小

        self.patch_embed = PatchEmbed(  # 补丁嵌入层
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None  # 位置嵌入
        if use_abs_pos:  # 如果使用绝对位置编码
            # Initialize absolute positional embedding with pretrain image size.
            # 用预训练图像大小初始化绝对位置嵌入。
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()  # Transformer块列表
        for i in range(depth):  # 创建每个Transformer块
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,  # 非全局注意力层使用窗口
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(  # 颈部网络
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)  # 第二级卷积
        self.net_3 = nn.Conv2d(  # 第三级卷积
            512, net_3_out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )

    # 前向传播：图像编码器
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # 补丁嵌入
        if self.pos_embed is not None:  # 添加位置编码
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))

        for blk in self.blocks:  # 遍历Transformer块
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))  # 颈部处理（转置为NCHW）
        x2 = self.net_2(x)  # 第二级特征
        x3 = self.net_3(x2.clone())  # 第三级特征

        return x3  # 返回第三级特征


# 构建SAM图像编码器
def _build_sam(
    encoder_embed_dim,  # 编码器嵌入维度
    encoder_depth,  # 编码器深度
    encoder_num_heads,  # 编码器头数
    encoder_global_attn_indexes,  # 全局注意力索引
    checkpoint=None,  # 检查点路径
    net_3_out_channels: int = 1024,  # net_3输出通道数
):
    prompt_embed_dim = 256  # 提示嵌入维度
    image_size = 1024  # 图像大小
    vit_patch_size = 16  # ViT补丁大小
    image_encoder = ImageEncoderViT(  # 创建图像编码器
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        net_3_out_channels=net_3_out_channels,
    )
    image_encoder.eval()  # 设为评估模式
    if checkpoint is not None:  # 如果有检查点
        state_dict = torch.load(checkpoint)  # 加载状态字典
        image_encoder.load_state_dict(  # 加载权重
            {k[30:]: v for k, v in state_dict.items() if "vision_tower_high" in k},  # 仅加载vision_tower_high部分
            strict=True,
        )
    return image_encoder  # 返回图像编码器


# 构建SAM ViT-B模型
def build_sam_vit_b(checkpoint=None, net_3_out_channels: int = 1024):
    return _build_sam(  # 构建SAM ViT-B
        encoder_embed_dim=768,  # 嵌入维度768
        encoder_depth=12,  # 深度12
        encoder_num_heads=12,  # 头数12
        encoder_global_attn_indexes=[2, 5, 8, 11],  # 全局注意力层索引
        checkpoint=checkpoint,
        net_3_out_channels=net_3_out_channels,
    )


# 获取绝对位置编码（CLIP风格）
def get_abs_pos(abs_pos, tgt_size):  # 绝对位置编码和目标大小
    # abs_pos: L, C
    # abs_pos: L, C
    # tgt_size: M
    # tgt_size: M
    # return: M, C
    # 返回: M, C
    dim = abs_pos.size(-1)  # 嵌入维度
    abs_pos_new = abs_pos.squeeze(0)  # 去掉批次维度
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]  # 分离CLS令牌和位置嵌入

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))  # 源网格大小
    tgt_size = int(math.sqrt(tgt_size))  # 目标网格大小
    dtype = abs_pos.dtype  # 保存数据类型

    if src_size != tgt_size:  # 如果需要插值
        old_pos_embed = (
            old_pos_embed.view(1, src_size, src_size, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        old_pos_embed = old_pos_embed.to(torch.float32)  # 转为float32
        new_pos_embed = F.interpolate(  # 双三次插值
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)  # 转回
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)  # 展平
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)  # 拼接CLS令牌
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)  # 添加批次维度
        return vision_pos_embed
    else:  # 无需插值
        return abs_pos


# CLIP视觉嵌入模块
class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size  # 嵌入维度
        self.image_size = image_size  # 图像大小
        self.patch_size = patch_size  # 补丁大小

        self.class_embedding = torch.nn.Parameter(torch.randn(self.embed_dim))  # 类嵌入

        self.patch_embedding = torch.nn.Conv2d(  # 补丁嵌入卷积
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 补丁数
        self.num_positions = self.num_patches + 1  # 位置数（补丁+CLS）
        self.position_embedding = torch.nn.Embedding(self.num_positions, self.embed_dim)  # 位置嵌入
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1))  # 位置ID
        )

    # 前向传播：CLIP视觉嵌入
    def forward(self, pixel_values, patch_embeds):  # 像素值和预计算的补丁嵌入
        batch_size = pixel_values.shape[0]  # 批次大小

        if patch_embeds is not None:  # 如果提供了预计算的补丁嵌入
            patch_embeds = patch_embeds
        else:  # 从像素值计算补丁嵌入
            patch_embeds = self.patch_embedding(pixel_values)

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 展平并转置

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # 扩展类嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接CLS和补丁

        embeddings = embeddings + get_abs_pos(  # 添加位置编码
            self.position_embedding(self.position_ids), embeddings.size(1)
        )
        return embeddings  # 返回嵌入


# 非张量并行注意力模块
class NoTPAttention(torch.nn.Module):
    def __init__(self, cfg):  # 配置字典
        super().__init__()
        self.num_heads = cfg["num_attention_heads"]  # 头数
        self.n_local_heads = cfg["num_attention_heads"]  # 本地头数（无并行）
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]  # 每头维度
        self.max_seq_len = cfg["seq_length"]  # 最大序列长度
        self.use_flash_attention = cfg["use_flash_attn"]  # 是否使用Flash注意力

        self.qkv_proj = torch.nn.Linear(  # QKV投影
            cfg["hidden_size"], cfg["hidden_size"] * 3, bias=True
        )
        self.out_proj = torch.nn.Linear(  # 输出投影
            cfg["hidden_size"], cfg["hidden_size"], bias=True
        )

        # self.core_attention = CoreAttention(cfg, AttnType.self_attn)

        self.attn_drop = cfg["attention_dropout"]  # 注意力Dropout率

    # 前向传播：非TP注意力
    def forward(
        self,
        x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape  # 批次大小、序列长度、特征维度
        xqkv = self.qkv_proj(x)  # QKV投影
        xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)  # 重塑

        if self.use_flash_attention:  # 使用Flash注意力

            xq, xk, xv = torch.split(xqkv, 1, dim=2)  # 分割Q/K/V
            xq = xq.squeeze(2)
            xk = xk.squeeze(2)
            xv = xv.squeeze(2)
            # xq, xk, xv = xqkv[:, :, 0, ...], xqkv[:, :, 1, ...], xqkv[:, :, 2, ...]

            # （B, num_head, S, head_size)
            xq = xq.permute(0, 2, 1, 3)  # 重排Q
            xk = xk.permute(0, 2, 1, 3)  # 重排K
            xv = xv.permute(0, 2, 1, 3)  # 重排V
            output = torch.nn.functional.scaled_dot_product_attention(  # 缩放点积注意力
                xq, xk, xv, attn_mask=None
            )
            output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)  # 重排输出
        else:  # 不使用Flash注意力
            xq, xk, xv = torch.split(xqkv, 1, dim=2)  # 分割Q/K/V
            xq = xq.squeeze(2)
            xk = xk.squeeze(2)
            xv = xv.squeeze(2)

            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xv = xv.permute(0, 2, 1, 3)
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None
            )
            output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        output = self.out_proj(output)  # 输出投影
        return output


@torch.jit.script
# 快速GELU激活函数
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)  # 近似GELU


# 非TP前馈网络模块
class NoTPFeedForward(nn.Module):
    def __init__(
        self,
        cfg,  # 配置
        dim: int,  # 输入维度
        hidden_dim: int,  # 隐藏维度
    ):
        super().__init__()

        self.fc1 = torch.nn.Linear(dim, hidden_dim, bias=True)  # 第一个全连接层
        self.fc2 = torch.nn.Linear(hidden_dim, dim, bias=True)  # 第二个全连接层

    # 前向传播：前馈网络
    def forward(self, x):
        output = self.fc2(quick_gelu(self.fc1(x)))  # FC1 -> GELU -> FC2
        return output


# FP32层归一化模块
class LayerNormfp32(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""  # 子类化torch的LayerNorm以处理fp16

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype  # 保存原始类型
        ret = super().forward(x.type(torch.float32))  # 在float32中计算
        return ret.type(orig_type)  # 转回原始类型


# 非TP Transformer块
class NoTPTransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id: int, multiple_of=256):
        super().__init__()

        self.n_heads = cfg["num_attention_heads"]  # 头数
        self.dim = cfg["hidden_size"]  # 隐藏维度
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]  # 每头维度
        self.self_attn = NoTPAttention(cfg)  # 自注意力
        self.mlp = NoTPFeedForward(  # MLP
            cfg, dim=cfg["hidden_size"], hidden_dim=cfg["ffn_hidden_size"]
        )
        self.layer_id = layer_id  # 层ID
        self.layer_norm1 = torch.nn.LayerNorm(  # 第一个层归一化
            cfg["hidden_size"], eps=cfg["layernorm_epsilon"]
        )
        self.layer_norm2 = torch.nn.LayerNorm(  # 第二个层归一化
            cfg["hidden_size"], eps=cfg["layernorm_epsilon"]
        )

    # 前向传播：Transformer块
    def forward(self, x: torch.Tensor):
        residual = self.self_attn.forward(self.layer_norm1(x))  # 自注意力
        h = x + residual  # 注意力残差
        out = h + self.mlp.forward(self.layer_norm2(h))  # MLP残差
        return out


# 非TP Transformer模型
class NoTPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg  # 配置
        self.num_layers = cfg["num_layers"]  # 层数

        self.layers = torch.nn.ModuleList()  # 层列表
        for layer_id in range(self.num_layers):  # 创建每层
            self.layers.append(
                NoTPTransformerBlock(
                    cfg,
                    layer_id + 1,
                )
            )

    # 前向传播：Transformer
    def forward(
        self,
        hidden_states,  # 隐藏状态
    ):

        for layer in self.layers:  # 遍历每层
            hidden_states = layer(hidden_states)

        return hidden_states  # 返回隐藏状态


# ViT模型
class VitModel(nn.Module):
    def __init__(self, cfg, freeze_embed=False, freeze_pre_norm=False) -> None:
        super().__init__()

        self.embeddings = CLIPVisionEmbeddings(  # CLIP视觉嵌入
            hidden_size=cfg["hidden_size"],
            image_size=cfg["image_size"],
            patch_size=cfg["patch_size"],
        )

        if freeze_embed:  # 如果冻结嵌入
            for _, param in self.embeddings.named_parameters():
                param.requires_grad = False

        self.transformer = NoTPTransformer(cfg=cfg)  # 非TP Transformer

        if cfg.get("fp32norm", False):  # 如果使用FP32归一化
            logger.info("Load fp32 layernorm for ViT.")
            self.pre_layrnorm = LayerNormfp32(  # FP32层归一化
                cfg["hidden_size"],
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )
        else:  # 标准层归一化
            self.pre_layrnorm = torch.nn.LayerNorm(
                cfg["hidden_size"],
                eps=cfg.get("pre_layernorm_epsilon", 1e-5),
            )

        if freeze_pre_norm:  # 如果冻结预归一化
            for _, param in self.pre_layrnorm.named_parameters():
                param.requires_grad = False

        for p in self.parameters():  # 设置micro_dp标志
            p.micro_dp = True

    @property
    def dtype(self):  # 数据类型
        return next(self.parameters()).dtype

    # 设置输入张量
    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.transformer.set_input_tensor(input_tensor[0])

    def __str__(self) -> str:
        return "open_clip"

    # 前向传播：ViT模型
    def forward(self, x, patch_embeds):  # 输入和补丁嵌入
        x = self.embeddings(x, patch_embeds)  # 嵌入
        hidden_states = self.pre_layrnorm(x)  # 预归一化

        output = self.transformer(hidden_states)  # Transformer

        return output


# ViT模型配置
vit_model_cfg = dict(
    num_layers=24,  # 层数
    hidden_size=1024,  # 隐藏维度
    num_heads=16,  # 头数
    num_attention_heads=16,  # 注意力头数
    ffn_hidden_size=4096,  # FFN隐藏维度
    seq_length=256,  # 序列长度
    max_position_embeddings=256,  # 最大位置嵌入
    use_flash_attn=False,  # 是否使用Flash注意力
    understand_projector_stride=2,  # 理解投影器步长
    hidden_dropout=0.0,  # 隐藏Dropout
    attention_dropout=0.0,  # 注意力Dropout
    no_persist_layer_norm=False,  # 非持久层归一化
    layernorm_epsilon=1e-5,  # 层归一化epsilon
    pre_layernorm_epsilon=1e-5,  # 预层归一化epsilon
    image_size=224,  # 图像大小
    patch_size=14,  # 补丁大小
    recompute_list=[],  # 重计算列表
)


# 构建CLIP-L模型
def build_clip_l():
    return VitModel(  # 创建ViT模型
        cfg=vit_model_cfg,
        freeze_embed=False,  # 不冻结嵌入
        freeze_pre_norm=False,  # 不冻结预归一化
    )


# 自定义Qwen2解码器，带混合因果掩码用于OCR2视觉编码器
class CustomQwen2Decoder(nn.Module):
    """Qwen2 decoder with mixed causal masking for OCR2 vision encoder."""
    """带混合因果掩码的Qwen2解码器，用于OCR2视觉编码器。"""

    def __init__(
        self,
        decoder_layer: int = 24,  # 解码器层数
        max_position_embeddings: int = 131072,  # 最大位置嵌入
        hidden_dimension: int = 896,  # 隐藏维度
        num_attention_heads: int = 14,  # 注意力头数
        num_key_value_heads: int = 2,  # KV头数
        intermediate_size: int = 4864,  # 中间层大小
        vocab_size: int = 151936,  # 词表大小
        attn_implementation: str = "sdpa",  # 注意力实现
        rms_norm_eps: float = 1e-6,  # RMS归一化epsilon
        rope_theta: float = 1000000.0,  # RoPE theta
        attention_dropout: float = 0.0,  # 注意力Dropout
        hidden_act: str = "silu",  # 隐藏激活
        initializer_range: float = 0.02,  # 初始化范围
    ):
        super().__init__()
        if attn_implementation == "flash_attention_2":  # 不支持Flash注意力2
            raise ValueError(
                "CustomQwen2Decoder does not support flash_attention_2; "
                "use sdpa or eager."
            )

        Qwen2Model = getattr(transformers.models.qwen2.modeling_qwen2, "Qwen2Model")  # 获取Qwen2Model类
        Qwen2Config = getattr(transformers, "Qwen2Config")  # 获取Qwen2Config类

        config = Qwen2Config(  # 创建Qwen2配置
            hidden_size=hidden_dimension,
            num_hidden_layers=decoder_layer,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            _attn_implementation=attn_implementation,
        )

        self.model = self._create_custom_model(Qwen2Model, config)  # 创建自定义模型
        del self.model.embed_tokens  # 删除嵌入层

    # 创建自定义Qwen2模型（带混合因果掩码）
    def _create_custom_model(self, Qwen2Model, config):
        class CustomQwen2ModelInner(Qwen2Model):
            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                token_type_ids=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                cache_position=None,
            ):
                self._current_token_type_ids = token_type_ids  # 保存令牌类型ID
                causal_mask_mapping = {  # 因果掩码映射
                    "full_attention": self._update_causal_mask(
                        attention_mask,
                        inputs_embeds,
                        cache_position,
                        past_key_values,
                        output_attentions,
                    )
                }
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=causal_mask_mapping,  # 使用混合因果掩码
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            # 更新因果掩码（支持混合掩码）
            def _update_causal_mask(
                self,
                attention_mask,
                input_tensor,
                cache_position,
                past_key_values,
                output_attentions,
            ):
                dtype, device = input_tensor.dtype, input_tensor.device
                min_dtype = torch.finfo(dtype).min
                batch_size, sequence_length = (
                    input_tensor.shape[0],
                    input_tensor.shape[1],
                )

                token_type_ids = getattr(self, "_current_token_type_ids", None)
                if token_type_ids is None:  # 如果没有令牌类型ID，使用标准因果掩码
                    return super()._update_causal_mask(
                        attention_mask,
                        input_tensor,
                        cache_position,
                        past_key_values,
                        output_attentions,
                    )

                causal_mask = self._create_custom_4d_mask(  # 创建自定义4D掩码
                    sequence_length=sequence_length,
                    dtype=dtype,
                    device=device,
                    batch_size=batch_size,
                    token_type_ids=token_type_ids,
                )

                if attention_mask is not None and attention_mask.dim() == 2:  # 处理padding掩码
                    padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
                    padding_mask = (1.0 - padding_mask) * min_dtype
                    causal_mask = causal_mask + padding_mask

                return causal_mask

            # 创建自定义4D混合因果掩码
            def _create_custom_4d_mask(
                self,
                sequence_length,
                dtype,
                device,
                batch_size,
                token_type_ids,
            ):
                min_dtype = torch.finfo(dtype).min
                masks = []
                for b in range(batch_size):  # 遍历每个样本
                    mask = torch.full(
                        (sequence_length, sequence_length),
                        fill_value=min_dtype,  # 初始化为最小值（遮蔽）
                        dtype=dtype,
                        device=device,
                    )

                    type_ids = token_type_ids[b]
                    image_positions = (type_ids == 0).nonzero(as_tuple=True)[0]  # 图像令牌位置
                    text_positions = (type_ids == 1).nonzero(as_tuple=True)[0]  # 文本令牌位置

                    if len(image_positions) > 0:  # 图像令牌之间全连接
                        mask[image_positions[:, None], image_positions] = 0.0

                    for i, text_pos in enumerate(text_positions):  # 文本令牌因果掩码
                        if len(image_positions) > 0:  # 文本可以看到图像
                            mask[text_pos, image_positions] = 0.0
                        mask[text_pos, text_positions[: i + 1]] = 0.0  # 文本因果

                    masks.append(mask)

                mask = torch.stack(masks, dim=0).unsqueeze(1)  # 堆叠并添加头维度
                return mask

        return CustomQwen2ModelInner(config)  # 返回自定义模型实例

    # 前向传播：自定义Qwen2解码器
    def forward(self, inputs_embeds, token_type_ids, attention_mask=None, **kwargs):
        return self.model(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


# Qwen2解码器即编码器模块，用于OCR2视觉令牌
class Qwen2Decoder2Encoder(nn.Module):
    """Decoder-as-encoder for OCR2 vision tokens."""
    """用于OCR2视觉令牌的解码器即编码器。"""

    def __init__(
        self,
        decoder_layer: int,  # 解码器层数
        hidden_dimension: int,  # 隐藏维度
        num_attention_heads: int,  # 注意力头数
        num_key_value_heads: int,  # KV头数
        intermediate_size: int,  # 中间层大小
        max_query: int,  # 最大查询数
    ):
        super().__init__()
        self.model = CustomQwen2Decoder(  # 自定义Qwen2解码器
            decoder_layer=decoder_layer,
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            attn_implementation="sdpa",
        )

        self.query_768 = nn.Embedding(144, hidden_dimension)  # 768分辨率查询嵌入
        self.query_1024 = nn.Embedding(256, hidden_dimension)  # 1024分辨率查询嵌入

    # 前向传播：解码器即编码器
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)  # 展平并转置
        bs, n_query, _ = x.shape  # 批次大小和查询数

        if n_query == 144:  # 768分辨率
            param_img = self.query_768.weight
        elif n_query == 256:  # 1024分辨率
            param_img = self.query_1024.weight
        else:  # 其他分辨率需要插值
            base = (
                self.query_1024.weight
                if n_query > self.query_768.num_embeddings
                else self.query_768.weight
            )
            param_img = (  # 线性插值
                F.interpolate(
                    base.T.unsqueeze(0),
                    size=n_query,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )

        batch_query_imgs = param_img.unsqueeze(0).expand(bs, -1, -1)  # 扩展查询
        x_combined = torch.cat([x, batch_query_imgs], dim=1)  # 拼接特征和查询
        token_type_ids = torch.cat(  # 创建令牌类型ID（0=图像，1=查询）
            [
                torch.zeros(bs, n_query, dtype=torch.long, device=x.device),
                torch.ones(bs, n_query, dtype=torch.long, device=x.device),
            ],
            dim=1,
        )
        y = self.model(x_combined, token_type_ids)[0]  # 通过解码器
        y = y[:, n_query:, :]  # 取查询位置的输出
        return y  # 返回编码后的特征


# 构建Qwen2解码器即编码器
def build_qwen2_decoder_as_encoder(
    decoder_layer: int = 24,  # 解码器层数
    hidden_dimension: int = 896,  # 隐藏维度
    num_attention_heads: int = 14,  # 注意力头数
    num_key_value_heads: int = 2,  # KV头数
    intermediate_size: int = 4864,  # 中间层大小
    max_query: int = 400,  # 最大查询数
    checkpoint=None,  # 检查点路径
):
    decoder_as_encoder = Qwen2Decoder2Encoder(  # 创建解码器即编码器
        decoder_layer=decoder_layer,
        hidden_dimension=hidden_dimension,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_query=max_query,
    )
    if checkpoint is not None:  # 如果有检查点
        state_dict = torch.load(checkpoint)
        decoder_as_encoder.load_state_dict(state_dict, strict=True)
    return decoder_as_encoder  # 返回解码器即编码器


# DeepSeek OCR因果语言模型
class DeepseekOCRForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        config: DeepseekVLV2Config,  # DeepSeek OCR视觉语言配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()

        self.config = config  # 保存配置

        self.vision_config = config.vision_config  # 视觉配置
        self.projector_config = config.projector_config  # 投影器配置
        self.text_config = config.text_config  # 文本配置
        self.is_ocr2 = (  # 判断是否为OCR2
            str(getattr(self.vision_config, "model_name", "")).lower()
            == "deepencoderv2"
            or getattr(self.projector_config, "input_dim", None) == 896
        )
        n_embed = getattr(self.projector_config, "n_embed", 1280)  # 嵌入维度

        self.tile_tag = config.tile_tag  # 切片标签
        self.global_view_pos = config.global_view_pos  # 全局视图位置

        # special token for image token sequence format
        # 图像令牌序列格式的特殊令牌
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":  # 2D切片标签
            # <|view_separator|>, <|\n|>
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)  # 视图分隔符
            if not self.is_ocr2:  # OCR1有换行符
                self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:  # 不支持的切片标签
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        if not self.is_ocr2:  # OCR1模式
            if self.text_config.topk_method == "noaux_tc":  # V3模型
                self.model = DeepseekV3ForCausalLM(
                    config=config.text_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "language"),
                )
            elif not self.text_config.use_mla:  # 非MLA DeepSeek
                self.model = DeepseekForCausalLM(
                    config=config.text_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "language"),
                )
            else:  # MLA DeepSeek V2
                self.model = DeepseekV2ForCausalLM(
                    config=config.text_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "language"),
                )
        else:  # OCR2模式
            # OCR2 language_config uses non-MLA attention (qk_* dims are 0).
            # OCR2的language_config使用非MLA注意力（qk_*维度为0）。
            # Use the non-MLA Deepseek model to avoid MLA-specific assumptions.
            # 使用非MLA Deepseek模型以避免MLA特有的假设。
            self.model = DeepseekForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )

        if not self.is_ocr2:  # OCR1视觉模型
            self.sam_model = build_sam_vit_b()  # SAM视觉编码器
            self.vision_model = build_clip_l()  # CLIP视觉模型
        else:  # OCR2视觉模型
            projector_input_dim = getattr(self.projector_config, "input_dim", 896)
            self.sam_model = build_sam_vit_b(net_3_out_channels=projector_input_dim)  # SAM编码器
            self.qwen2_model = build_qwen2_decoder_as_encoder(  # Qwen2解码器即编码器
                hidden_dimension=projector_input_dim
            )

        self.projector = MlpProjector(  # MLP投影器
            projector_type=self.projector_config.projector_type,
            input_dim=self.projector_config.input_dim,
            n_embed=n_embed,
            depth=self.projector_config.depth,
            mlp_ratio=self.projector_config.mlp_ratio,
            downsample_ratio=self.projector_config.downsample_ratio,
        )

    @staticmethod
    # 收集多模态数据项中的标志
    def _collect_mm_flag(
        items: List[MultimodalDataItem], flag_name: str  # 数据项列表和标志名
    ) -> Optional[List[bool]]:
        values = []
        for item in items:
            value = getattr(item, flag_name, None)  # 获取标志值
            if value is None:  # 如果任何项没有该标志
                return None
            values.append(bool(value))
        return values  # 返回标志值列表

    # 编码OCR2特征
    def _encode_ocr2_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.sam_model(images)  # SAM视觉特征
        features = self.qwen2_model(features)  # Qwen2解码器即编码器
        features = self.projector(features)  # MLP投影
        return features.view(-1, features.shape[-1])  # 展平

    # 编码OCR1特征
    def _encode_ocr1_features(self, images: torch.Tensor) -> torch.Tensor:
        features_1 = self.sam_model(images)  # SAM视觉特征
        features_2 = self.vision_model(images, features_1)  # CLIP视觉特征（使用SAM特征作为补丁嵌入）
        features = torch.cat(  # 拼接CLIP和SAM特征
            (
                features_2[:, 1:],  # CLIP特征（去掉CLS）
                features_1.flatten(2).permute(0, 2, 1),  # SAM特征展平
            ),
            dim=-1,
        )
        return self.projector(features)  # 通过投影器

    # 格式化OCR1全局特征
    def _format_ocr1_global_features(self, features: torch.Tensor) -> torch.Tensor:
        _, hw, n_dim = features.shape
        h = w = int(hw**0.5)
        features = features.view(h, w, n_dim)
        features = torch.cat(
            [features, self.image_newline[None, None, :].expand(h, 1, n_dim)],  # 每行末尾添加换行符
            dim=1,
        )
        return features.view(-1, n_dim)  # 展平

    # 格式化OCR1局部特征
    def _format_ocr1_local_features(
        self, features: torch.Tensor, crop_shape: torch.Tensor
    ) -> torch.Tensor:
        _, hw2, n_dim2 = features.shape
        h2 = w2 = int(hw2**0.5)
        width_crop_num, height_crop_num = int(crop_shape[0]), int(crop_shape[1])
        features = (  # 重排为网格
            features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
            .permute(0, 2, 1, 3, 4)
            .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
        )
        features = torch.cat(
            [
                features,
                self.image_newline[None, None, :].expand(
                    height_crop_num * h2, 1, n_dim2
                ),  # 每行末尾添加换行符
            ],
            dim=1,
        )
        return features.view(-1, n_dim2)  # 展平

    # 解析和验证图像输入
    def _parse_and_validate_image_input(self, **kwargs: object):

        pixel_values = kwargs.pop("pixel_values", None)  # 像素值
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)  # 空间裁剪信息
        images_crop = kwargs.pop("images_crop", None)  # 裁剪图像
        has_images = kwargs.pop("has_images", None)  # 是否有图像

        if pixel_values is None:  # 没有像素值
            return None
        if has_images is not None:  # 有has_images标志
            if not has_images:
                return None
        elif torch.sum(pixel_values).item() == 0:  # 像素值全为零
            return None

        if pixel_values is not None:  # 验证输入类型
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image sizes. "
                    f"Got type: {type(images_spatial_crop)}"
                )

            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image crop. " f"Got type: {type(images_crop)}"
                )

            return [pixel_values, images_crop, images_spatial_crop]  # 返回验证后的输入

        raise AssertionError("This line should be unreachable.")  # 不应到达此处

    # 将像素值转换为嵌入
    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,  # 像素值
        images_crop: torch.Tensor,  # 裁剪图像
        images_spatial_crop: torch.Tensor,  # 空间裁剪信息
        has_local_crops: Optional[List[bool]] = None,  # 是否有局部裁剪
    ) -> NestedTensors:

        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # Pixel_values（全局视图）: [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # images_crop（局部视图）: [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1
        # 分割pixel和image_crop，所有batch_size = 1

        images_in_this_batch = []  # 批次中的图像列表

        if not self.is_ocr2:  # OCR1模式
            with torch.no_grad():  # 禁用梯度
                for jdx in range(images_spatial_crop.size(0)):  # 遍历每张图像
                    patches = images_crop[jdx][0].to(torch.bfloat16)  # 局部裁剪块
                    image_ori = pixel_values[jdx]  # 原始图像
                    crop_shape = images_spatial_crop[jdx][0]  # 裁剪形状
                    use_local_crops = (  # 判断是否使用局部裁剪
                        has_local_crops[jdx]
                        if has_local_crops is not None
                        else torch.sum(patches).item() != 0
                    )

                    global_features = self._encode_ocr1_features(image_ori)  # 编码全局特征
                    global_features = self._format_ocr1_global_features(global_features)  # 格式化全局特征

                    if use_local_crops:  # 如果使用局部裁剪
                        local_features = self._encode_ocr1_features(patches)  # 编码局部特征
                        local_features = self._format_ocr1_local_features(
                            local_features, crop_shape
                        )
                        global_local_features = torch.cat(  # 拼接局部和全局特征
                            [
                                local_features,
                                global_features,
                                self.view_seperator[None, :],
                            ],
                            dim=0,
                        )
                    else:  # 仅全局特征
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :]], dim=0
                        )

                    images_in_this_batch.append(global_local_features)

            return images_in_this_batch  # 返回OCR1图像嵌入

        with torch.no_grad():  # OCR2模式，禁用梯度
            for jdx in range(images_spatial_crop.size(0)):
                patches = images_crop[jdx][0].to(torch.bfloat16)
                image_ori = pixel_values[jdx]
                use_local_crops = (
                    has_local_crops[jdx]
                    if has_local_crops is not None
                    else torch.sum(patches).item() != 0
                )

                global_features = self._encode_ocr2_features(image_ori)  # 编码OCR2全局特征
                if use_local_crops:
                    local_features = self._encode_ocr2_features(patches)  # 编码OCR2局部特征
                    global_local_features = torch.cat(
                        [local_features, global_features, self.view_seperator[None, :]],
                        dim=0,
                    )
                else:
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch  # 返回OCR2图像嵌入

    # 处理图像输入
    def _process_image_input(self, mm_items: List[MultimodalDataItem]) -> torch.Tensor:
        target_dtype = (  # 目标数据类型
            next(self.sam_model.parameters()).dtype
            if self.is_ocr2
            else self.vision_model.dtype
        )
        has_local_crops = self._collect_mm_flag(mm_items, "has_local_crops")  # 收集局部裁剪标志
        pixel_values = torch.stack([item.feature for item in mm_items], dim=0).type(
            target_dtype
        )

        images_crop = (  # 裁剪图像
            torch.stack([item.images_crop for item in mm_items], dim=0)
            .type(target_dtype)
            .to(device=pixel_values.device)
        )
        images_spatial_crop = (  # 空间裁剪信息
            torch.cat([item.images_spatial_crop for item in mm_items], dim=0)
            .type(torch.long)
            .to(device=pixel_values.device)
        )

        assert images_crop.dim() == 6  # 断言裁剪图像维度
        assert images_spatial_crop.dim() == 3  # 断言空间裁剪维度

        vision_feature_lists = self._pixel_values_to_embedding(  # 像素值转嵌入
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            has_local_crops=has_local_crops,
        )
        vision_features = torch.cat(vision_feature_lists, dim=0).type(target_dtype)  # 拼接视觉特征

        return vision_features  # 返回视觉特征

    # 获取语言模型
    def get_language_model(self) -> torch.nn.Module:
        return self.model

    # 获取多模态嵌入
    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)  # 解析图像输入
        if image_input is None:  # 无图像输入
            return None
        vision_embeddings = self._process_image_input(image_input)  # 处理图像输入
        return vision_embeddings  # 返回视觉嵌入

    # 获取输入嵌入
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,  # 输入令牌ID
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,  # 多模态嵌入
    ) -> torch.Tensor:

        inputs_embeds = self.model.get_input_embeddings(input_ids)  # 获取文本嵌入

        if multimodal_embeddings is not None:  # 如果有多模态嵌入
            inputs_embeds = merge_multimodal_embeddings(  # 合并多模态嵌入
                input_ids, inputs_embeds, multimodal_embeddings, self.image_token_id
            )

        return inputs_embeds  # 返回合并后的嵌入

    # 填充输入ID
    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 返回填充后的令牌

    # 获取图像特征
    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        vision_embeddings = self._process_image_input(items)  # 处理图像输入
        return vision_embeddings  # 返回视觉嵌入

    # 前向传播
    def forward(
        self,
        input_ids: torch.Tensor,  # 输入令牌ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次信息
        **kwargs: object,
    ):
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )

        return hidden_states  # 返回隐藏状态

    # 加载权重
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())  # 参数字典
        loaded_params: Set[str] = set()  # 已加载参数集合
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转频率
                continue
            is_qwen2_weight = "qwen2_model." in name  # 是否为Qwen2权重
            if name == "lm_head.weight":  # 语言模型头权重
                name = "model.lm_head.weight"
            elif name.startswith("model."):  # model前缀的权重
                if (
                    "image_newline" in name
                    or ".projector" in name
                    or "vision_model" in name
                    or "qwen2_model" in name
                    or "sam_model" in name
                    or "view_seperator" in name
                ):
                    name = name[len("model.") :]  # 去掉model.前缀
                elif not (
                    ".projector" in name
                    or "vision_model" in name
                    or "qwen2_model" in name
                    or "sam_model" in name
                    or "image_newline" in name
                ):
                    name = name.replace("model.", "model.model.")  # 替换前缀

            if is_qwen2_weight:  # Qwen2权重的特殊处理
                target_name = name
                if target_name not in params_dict:
                    if ".model.model." in target_name:
                        alt_name = target_name.replace(".model.model.", ".model.")
                    else:
                        alt_name = target_name.replace(".model.", ".model.model.", 1)
                    if alt_name in params_dict:
                        target_name = alt_name
                if target_name.endswith(".bias") and target_name not in params_dict:  # 跳过不存在的偏置
                    continue
                if target_name in params_dict:
                    param = params_dict[target_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(target_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 处理堆叠参数
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                # 跳过GPTQ模型的额外偏置加载。
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                # 跳过未分配给此工作者的专家。
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:  # 非堆叠参数
                # Skip loading extra bias for GPTQ models.
                # 跳过GPTQ模型的额外偏置加载。
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                # 跳过未分配给此工作者的专家。
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params  # 未加载的参数
        if unloaded_params:  # 如果有未加载的参数则报错
            raise RuntimeError(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )
        self.post_load_weights()  # 权重加载后处理

    # 权重加载后处理
    def post_load_weights(self):
        if _is_cpu and _is_cpu_amx_available:  # CPU且有AMX支持
            from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading

            layer_ids = int(self.config.num_hidden_layers)  # 隐藏层数
            first_k_dense_replace_id = (  # 前k个密集层替换ID
                self.config.first_k_dense_replace
                if hasattr(self.config, "first_k_dense_replace")
                else -1
            )
            moe_layer_freq_id = (  # MoE层频率
                self.config.moe_layer_freq
                if hasattr(self.config, "moe_layer_freq")
                else 1
            )
            for layer_id in range(0, layer_ids):  # 遍历每层
                if (
                    layer_id >= first_k_dense_replace_id
                    and layer_id % moe_layer_freq_id == 0
                ):  # MoE层
                    if (
                        hasattr(self.model, "model")
                        and hasattr(self.model.model, "layers")
                        and hasattr(self.model.model.layers[layer_id], "mlp")
                    ):
                        self_moe = self.model.model.layers[layer_id].mlp
                        if hasattr(self_moe, "w1") and hasattr(self_moe, "w2"):
                            _amx_process_weight_after_loading(self_moe, ["w1", "w2"])  # AMX处理权重


EntryClass = [DeepseekOCRForCausalLM]  # 入口类列表
