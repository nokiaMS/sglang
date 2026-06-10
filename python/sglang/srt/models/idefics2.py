# IDEFICS2 视觉语言模型实现
# 本文件实现了 IDEFICS2 多模态模型的视觉编码器部分，包括视觉 MLP、编码器层、
# 编码器、视觉嵌入和视觉变换器等组件，支持可变分辨率图像输入。

# Copyright 2023 The SGLang team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import Optional  # 导入可选类型注解

import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取工具
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.conv import Conv2dLayer  # 导入二维卷积层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear  # 导入列并行和行并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.utils import add_prefix, is_npu  # 导入前缀添加工具和 NPU 检测


class Idefics2VisionMLP(nn.Module):  # IDEFICS2 视觉 MLP 类，实现视觉特征的两层全连接变换

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存模型配置
        self.activation_fn = get_act_fn(config.hidden_act)  # 根据配置获取激活函数
        self.fc1 = ColumnParallelLinear(  # 第一个全连接层（列并行），用于升维
            config.hidden_size,  # 输入维度
            config.intermediate_size,  # 中间维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc1", prefix),  # 参数前缀
        )
        self.fc2 = RowParallelLinear(  # 第二个全连接层（行并行），用于降维
            config.intermediate_size,  # 输入维度
            config.hidden_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc2", prefix),  # 参数前缀
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        hidden_states, _ = self.fc1(hidden_states)  # 通过第一个全连接层升维
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states, _ = self.fc2(hidden_states)  # 通过第二个全连接层降维
        return hidden_states  # 返回变换后的隐藏状态


class Idefics2EncoderLayer(nn.Module):  # IDEFICS2 编码器层类，实现单个 Transformer 编码器层

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.num_heads = config.num_attention_heads  # 注意力头数
        self.self_attn = VisionAttention(  # 自注意力层（视觉专用）
            embed_dim=config.hidden_size,  # 嵌入维度
            num_heads=self.num_heads,  # 注意力头数
            projection_size=config.intermediate_size,  # 投影维度
            use_qkv_parallel=True,  # 使用 QKV 并行
            quant_config=quant_config,  # 量化配置
            dropout=config.attention_dropout,  # Dropout 概率
            softmax_in_single_precision=True,  # 在单精度下计算 softmax
            flatten_batch=False,  # 不展平批次
            prefix=add_prefix("self_attn", prefix),  # 参数前缀
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层归一化
        self.mlp = Idefics2VisionMLP(  # 视觉 MLP 模块
            config,  # 模型配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 参数前缀
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层归一化

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        cu_seqlens: torch.Tensor,  # 累积序列长度，用于变长注意力
    ) -> torch.Tensor:  # 返回输出隐藏状态
        """
        Args:  # 参数说明
            hidden_states (`torch.FloatTensor`):  # 隐藏状态
                Input to the layer of shape `(batch, seq_len, embed_dim)`.  # 层输入，形状为 (batch, seq_len, embed_dim)

        """
        residual = hidden_states  # 保存残差连接
        hidden_states = self.layer_norm1(hidden_states)  # 通过第一个层归一化
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens)  # 通过自注意力层

        hidden_states = residual + hidden_states  # 残差连接
        residual = hidden_states  # 更新残差
        hidden_states = self.layer_norm2(hidden_states)  # 通过第二个层归一化
        hidden_states = self.mlp(hidden_states)  # 通过 MLP 层
        hidden_states = residual + hidden_states  # 残差连接
        return hidden_states  # 返回输出隐藏状态


class Idefics2Encoder(nn.Module):  # IDEFICS2 编码器类，由多个编码器层堆叠而成
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention
    layers. Each layer is a
    [`Idefics2EncoderLayer`].
    # 由 config.num_hidden_layers 个自注意力层组成的 Transformer 编码器。每层是一个 Idefics2EncoderLayer。

    Args:  # 参数说明
        config: Idefics2Config  # IDEFICS2 配置
    """

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存模型配置
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                Idefics2EncoderLayer(  # 每个编码器层
                    config,  # 模型配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 参数前缀
                )
                for i in range(config.num_hidden_layers)  # 根据配置的层数创建
            ]
        )

    def forward(  # 前向传播方法
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入
        cu_seqlens: torch.Tensor,  # 累积序列长度
    ) -> torch.Tensor:  # 返回编码器输出
        r"""
        Args:  # 参数说明
            inputs_embeds (torch.Tensor):  # 输入嵌入
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation.
                This is useful if you want more control over how to convert
                `input_ids` indices into associated vectorsthan the model's
                internal embedding lookup matrix.
                # 可选地，你可以选择直接传入嵌入表示而非 input_ids。
                # 如果你想比模型内部嵌入查找矩阵更精细地控制 input_ids 索引到向量的转换，这很有用。
        """
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        # cu_seqlens 必须在 CPU 上，因为 npu_flash_attention_unpad 算子的限制
        if is_npu():  # 如果运行在 NPU 上
            cu_seqlens = cu_seqlens.to("cpu")  # 将 cu_seqlens 移到 CPU
        hidden_states = inputs_embeds  # 初始化隐藏状态
        for encoder_layer in self.layers:  # 遍历每个编码器层
            layer_outputs = encoder_layer(  # 通过编码器层
                hidden_states,  # 隐藏状态
                cu_seqlens=cu_seqlens,  # 累积序列长度
            )
            hidden_states = layer_outputs  # 更新隐藏状态
        return hidden_states  # 返回最终的隐藏状态


class Idefics2VisionEmbeddings(nn.Module):  # IDEFICS2 视觉嵌入类，实现可变分辨率的视觉位置嵌入
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings
    ` to enable images of variable
    resolution.
    # 这是 SiglipVisionEmbeddings 的修改版本，支持可变分辨率图像。

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision
    Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
    which allows treating images in their native aspect ratio and without the
    need to resize them to the same fixed size. In particular, we start from the
    original pre-trained SigLIP model(which uses images of fixed-size square
    images) and adapt it by training on images of variable resolutions.
    # 修改方案来自 [Patch n' Pack: NaViT]，允许以原始宽高比处理图像，无需缩放到固定尺寸。
    # 具体而言，从预训练的 SigLIP 模型（使用固定尺寸正方形图像）出发，通过可变分辨率图像训练进行适配。
    """

    def __init__(self, config: PretrainedConfig):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.image_size = config.image_size  # 图像尺寸
        self.patch_size = config.patch_size  # 补丁尺寸
        self.patch_embedding = Conv2dLayer(  # 补丁嵌入卷积层
            in_channels=config.num_channels,  # 输入通道数
            out_channels=self.embed_dim,  # 输出通道数（嵌入维度）
            kernel_size=self.patch_size,  # 卷积核大小等于补丁尺寸
            stride=self.patch_size,  # 步长等于补丁尺寸
            padding="valid",  # 不使用填充
        )
        self.num_patches_per_side = self.image_size // self.patch_size  # 每边的补丁数
        self.num_patches = self.num_patches_per_side**2  # 总补丁数
        self.num_positions = self.num_patches  # 位置数等于补丁数
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 位置嵌入层

    def get_position_ids(  # 获取位置 ID，用于可变分辨率的位置编码
        self,
        pixel_values: torch.FloatTensor,  # 像素值
        patch_attention_mask: torch.BoolTensor,  # 补丁注意力掩码
        tgt_sizes: Optional[torch.IntTensor] = None,  # 目标尺寸，可选
    ):  # 返回位置 ID 张量
        batch_size, _, max_im_h, max_im_w = pixel_values.shape  # 获取批次大小和最大图像高宽

        max_nb_patches_h, max_nb_patches_w = (  # 计算最大补丁数
            max_im_h // self.patch_size,  # 高度方向最大补丁数
            max_im_w // self.patch_size,  # 宽度方向最大补丁数
        )
        boundaries = torch.arange(  # 计算边界值，用于分桶操作
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side  # 在 [0, 1] 之间均匀划分
        )
        position_ids = torch.full(  # 初始化位置 ID
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0  # 填充为 0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):  # 遍历每个样本的注意力掩码

            if tgt_sizes is not None:  # 如果提供了目标尺寸
                nb_patches_h = tgt_sizes[batch_idx][0]  # 高度方向的补丁数
                nb_patches_w = tgt_sizes[batch_idx][1]  # 宽度方向的补丁数
            else:  # 否则
                nb_patches_h = p_attn_mask[:, 0].sum()  # 从注意力掩码推断高度方向的补丁数
                nb_patches_w = p_attn_mask[0].sum()  # 从注意力掩码推断宽度方向的补丁数
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)  # 高度方向的分数坐标
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)  # 宽度方向的分数坐标
            bucket_coords_h = torch.bucketize(  # 将高度坐标分桶
                fractional_coords_h, boundaries, right=True  # 右边界包含
            )
            bucket_coords_w = torch.bucketize(  # 将宽度坐标分桶
                fractional_coords_w, boundaries, right=True  # 右边界包含
            )
            pos_ids = (  # 计算 2D 位置 ID
                bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w  # 行优先索引
            ).flatten()  # 展平
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids  # 将位置 ID 填入对应位置
        position_ids = position_ids.to(self.position_embedding.weight.device)  # 将位置 ID 移到与权重相同的设备
        return position_ids  # 返回位置 ID

    def forward(  # 前向传播方法
        self,
        pixel_values: torch.FloatTensor,  # 像素值
        patch_attention_mask: torch.BoolTensor,  # 补丁注意力掩码
        tgt_sizes: Optional[torch.IntTensor] = None,  # 目标尺寸，可选
    ) -> torch.Tensor:  # 返回视觉嵌入
        target_dtype = self.patch_embedding.weight.dtype  # 获取目标数据类型
        pixel_values = pixel_values.to(  # 将像素值转换到目标设备和数据类型
            device=self.patch_embedding.weight.device, dtype=target_dtype  # 与补丁嵌入层一致
        )
        patch_embeds = self.patch_embedding(pixel_values)  # 通过补丁嵌入卷积层
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # 展平并转置，变为 (batch, seq_len, embed_dim)
        position_ids = self.get_position_ids(  # 获取位置 ID
            pixel_values, patch_attention_mask, tgt_sizes  # 传入像素值、掩码和目标尺寸
        )

        embeddings = embeddings + self.position_embedding(position_ids)  # 加上位置嵌入
        return embeddings  # 返回带位置信息的嵌入


class Idefics2VisionTransformer(nn.Module):  # IDEFICS2 视觉变换器类，完整的视觉编码器

    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        require_post_norm: bool = True,  # 是否需要后归一化
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        embed_dim = config.hidden_size  # 嵌入维度
        self.config = config  # 保存模型配置
        self.embeddings = Idefics2VisionEmbeddings(config)  # 视觉嵌入模块
        self.encoder = Idefics2Encoder(  # 视觉编码器
            config=config,  # 模型配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("encoder", prefix),  # 参数前缀
        )
        self.post_layernorm = (  # 后归一化层
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 如果需要则使用层归一化
            if require_post_norm  # 根据配置决定
            else nn.Identity()  # 否则使用恒等变换
        )

    def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入层
        return self.embeddings  # 返回视觉嵌入模块

    def compute_cu_seqlens(  # 计算累积序列长度，用于变长注意力
        self,
        tgt_sizes: Optional[torch.Tensor] = None,  # 目标尺寸，可选
        input_embeds: Optional[torch.Tensor] = None,  # 输入嵌入，可选
    ) -> torch.Tensor:  # 返回累积序列长度张量
        # shape: (batch_size,)  # 形状：(batch_size,)
        if tgt_sizes is not None:  # 如果提供了目标尺寸
            seqlen = tgt_sizes[:, 0] * tgt_sizes[:, 1]  # 计算每个样本的序列长度（高 * 宽）
        elif input_embeds is not None:  # 如果提供了输入嵌入
            seqlen = torch.full(  # 用嵌入的序列长度填充
                size=(input_embeds.shape[0],),  # 批次大小
                fill_value=input_embeds.shape[1],  # 填充值为序列长度
                dtype=torch.int32,  # 数据类型为 int32
                device=input_embeds.device,  # 设备与输入嵌入一致
            )
        else:  # 否则
            raise ValueError(  # 抛出异常
                "Either `tgt_sizes` or `input_embeds` must be provided to compute cu_seqlens."  # 必须提供 tgt_sizes 或 input_embeds
            )

        cu_seqlens = torch.cat(  # 拼接累积序列长度
            [
                torch.tensor([0], device=seqlen.device, dtype=torch.int32),  # 起始位置为 0
                torch.cumsum(seqlen, dim=0, dtype=torch.int32),  # 累积求和
            ],
            dim=0,  # 在第 0 维拼接
        ).to(seqlen.device)  # 移到与 seqlen 相同的设备
        return cu_seqlens  # 返回累积序列长度

    def forward(  # 前向传播方法
        self,
        pixel_values,  # 像素值
        patch_attention_mask: Optional[torch.BoolTensor] = None,  # 补丁注意力掩码，可选
        tgt_sizes: Optional[torch.IntTensor] = None,  # 目标尺寸，可选
    ) -> torch.Tensor:  # 返回最后的隐藏状态
        hidden_states = self.embeddings(  # 通过视觉嵌入模块
            pixel_values=pixel_values,  # 像素值
            patch_attention_mask=patch_attention_mask,  # 补丁注意力掩码
            tgt_sizes=tgt_sizes,  # 目标尺寸
        )
        cu_seqlens = self.compute_cu_seqlens(tgt_sizes, hidden_states)  # 计算累积序列长度
        encoder_outputs = self.encoder(  # 通过编码器
            hidden_states,  # 隐藏状态
            cu_seqlens=cu_seqlens,  # 累积序列长度
        )
        last_hidden_state = self.post_layernorm(encoder_outputs)  # 通过后归一化层
        return last_hidden_state  # 返回最后的隐藏状态
