# Radio视觉模型推理实现文件
# 本文件实现了Radio视觉编码器模型，基于InternVision编码器架构
# 支持可变分辨率图像处理(NaFlex)、视频时序压缩和动态尺寸图像打包
# 适配自vLLM的Radio模型实现

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/radio.py

import logging  # 导入日志模块
import math  # 导入数学模块
from collections.abc import Iterable  # 导入可迭代类型
from itertools import repeat  # 导入重复迭代器
from typing import TypeAlias  # 导入类型别名

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
from einops import rearrange  # 导入张量重排库
from transformers import PretrainedConfig  # 导入预训练配置
from transformers.modeling_outputs import BaseModelOutput  # 导入基础模型输出

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.model_loader.weight_utils import (  # 导入权重加载工具
    default_weight_loader,  # 默认权重加载器
    replace_prefix,  # 替换前缀
    replace_substrings,  # 替换子串
)
from sglang.srt.models.internvl import InternVisionEncoder  # 导入InternVision编码器

logger = logging.getLogger(__name__)  # 创建日志记录器

input_dim_t: TypeAlias = int | tuple[int, int]  # 输入维度类型别名
norm_t: TypeAlias = tuple[float, float, float] | torch.Tensor  # 归一化类型别名


def _ntuple(n):
    """创建将输入转换为n元组的函数"""
    def parse(x):
        """将输入解析为n元组"""
        if isinstance(x, Iterable) and not isinstance(x, str):  # 如果已是可迭代对象
            return tuple(x)  # 转为元组
        return tuple(repeat(x, n))  # 重复n次

    return parse  # 返回解析函数


to_1tuple = _ntuple(1)  # 转换为1元组
to_2tuple = _ntuple(2)  # 转换为2元组
to_3tuple = _ntuple(3)  # 转换为3元组
to_4tuple = _ntuple(4)  # 转换为4元组
to_ntuple = _ntuple  # 转换为n元组的工厂函数


class ClsToken(nn.Module):
    """CLS令牌模块，为输入序列添加类别令牌和寄存器令牌"""
    def __init__(
        self,
        ndim: int,  # 嵌入维度
        num_tokens: int = 1,  # 令牌数量
        enabled: bool = True,  # 是否启用
        register_multiple: int | None = None,  # 寄存器倍数
        num_registers: int | None = None,  # 寄存器数量
    ):
        """初始化CLS令牌，创建可学习的令牌参数"""
        super().__init__()

        self.ndim = ndim  # 保存维度
        self.enabled = enabled  # 保存启用状态
        self.num_registers = 0  # 初始化寄存器数量
        self.num_tokens = num_tokens  # 保存令牌数量
        if enabled:  # 如果启用
            if num_registers:  # 如果指定寄存器数量
                self.num_registers = num_registers
            elif register_multiple:  # 如果指定寄存器倍数
                self.num_registers = register_multiple - (
                    num_tokens % register_multiple
                )

            scale = ndim**-0.5  # 缩放因子
            self.token = nn.Parameter(
                torch.randn(num_tokens + self.num_registers, ndim) * scale  # 创建令牌参数
            )

        else:
            self.token = None  # 不创建令牌

        self.num_patches = self.num_tokens + self.num_registers  # 总补丁数

    def forward(self, x: torch.Tensor):
        """将CLS令牌和寄存器令牌拼接到输入序列前面"""
        if self.token is None:  # 如果无令牌
            return x

        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)  # 扩展令牌维度
        x = torch.cat(
            [
                token,  # 令牌
                x,  # 原始输入
            ],
            dim=1,  # 沿序列维度拼接
        )

        return x  # 返回拼接结果


class ViTPatchGenerator(nn.Module):
    """ViT补丁生成器，将图像转换为补丁嵌入序列"""
    def __init__(
        self,
        patch_size: int,  # 补丁大小
        embed_dim: int,  # 嵌入维度
        input_dims: input_dim_t,  # 输入维度
        abs_pos: bool = True,  # 是否使用绝对位置编码
        normalize_patches: bool = False,  # 是否归一化补丁
        cls_token: bool = False,  # 是否使用CLS令牌
        max_input_dims: input_dim_t | None = None,  # 最大输入维度
        pos_dropout: float = 0.0,  # 位置编码dropout
        return_pos_enc: bool = False,  # 是否返回位置编码
        num_cls_tokens: int = 1,  # CLS令牌数量
        register_multiple: int | None = None,  # 寄存器倍数
        num_registers: int | None = None,  # 寄存器数量
        patch_bias: bool = False,  # 补丁偏置
        video_temporal_patch_size: int = 1,  # 视频时序补丁大小
        separate_video_embedder: bool = True,  # 是否使用独立视频嵌入器
        device=None,  # 设备
        dtype=None,  # 数据类型
    ):
        """初始化ViT补丁生成器，配置补丁嵌入、位置编码和CLS令牌"""
        super().__init__()
        if isinstance(input_dims, int):  # 如果输入维度是整数
            input_dims = (input_dims, input_dims)  # 转为元组

        if max_input_dims is None:  # 如果未指定最大维度
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):  # 如果最大维度是整数
            max_input_dims = (max_input_dims, max_input_dims)

        max_input_dims = tuple(
            int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims
        )  # 向上取整到补丁大小的倍数

        self.cpe_mode = max_input_dims != input_dims  # 是否使用CPE模式
        self.pos_dropout = pos_dropout  # 位置dropout
        self.return_pos_enc = return_pos_enc  # 是否返回位置编码

        factory = dict(device=device, dtype=dtype)  # 设备和数据类型参数

        self.patch_size = patch_size  # 补丁大小
        self.abs_pos = abs_pos  # 绝对位置编码
        self.embed_dim = embed_dim  # 嵌入维度

        self.num_rows = max_input_dims[0] // patch_size  # 行数
        self.num_cols = max_input_dims[1] // patch_size  # 列数
        self.input_dims = tuple(d // patch_size for d in input_dims)  # 输入维度
        self.num_patches = self.num_rows * self.num_cols  # 补丁数
        self.max_input_dims = max_input_dims  # 最大输入维度

        self.im_to_patches = Im2Patches(patch_size)  # 图像转补丁
        self.embedder = ViTPatchLinear(
            patch_size, embed_dim, bias=patch_bias, **factory
        )  # 补丁线性嵌入

        if abs_pos:  # 如果使用绝对位置编码
            scale = embed_dim**-0.5  # 缩放因子
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim, **factory) * scale
            )  # 位置编码参数

        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
            num_registers=num_registers,
        )  # CLS令牌

        self.patch_normalizer = (
            nn.LayerNorm(embed_dim) if normalize_patches else nn.Identity()
        )  # 补丁归一化

        self.video_temporal_patch_size = video_temporal_patch_size  # 视频时序补丁大小
        self.video_embedder = None  # 视频嵌入器
        self._video_embedder_loaded = False  # 视频嵌入器是否已加载
        if video_temporal_patch_size > 1 and separate_video_embedder:  # 如果需要视频嵌入器
            self.video_embedder = nn.Linear(
                3 * video_temporal_patch_size * patch_size * patch_size,
                embed_dim,
                bias=False,
                **factory,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """图像前向传播：补丁嵌入、位置编码、CLS令牌、归一化"""
        patches = self.embed_patches(x)  # 补丁嵌入
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])  # 应用位置编码
        patches = self.cls_token(patches)  # 添加CLS令牌
        patches = self.patch_normalizer(patches)  # 补丁归一化
        if self.return_pos_enc:  # 如果需要返回位置编码
            return patches, pos_enc
        return patches

    def forward_video(self, x: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
        """Embed video frames with temporal compression via tubelet grouping."""
        """视频前向传播：通过管状分组实现时序压缩"""
        assert (
            self.video_embedder is not None
        ), "video_embedder is required for temporal compression"
        T = temporal_patch_size  # 时序补丁大小
        num_frames = x.shape[0]  # 帧数

        if num_frames % T != 0:  # 如果帧数不能被T整除
            pad = T - (num_frames % T)  # 计算填充数
            x = torch.cat(
                [x, x[-1:].expand(pad, -1, -1, -1)],  # 重复最后一帧填充
                dim=0,
            )

        padded_frames = x.shape[0]  # 填充后帧数
        num_tubelets = padded_frames // T  # 管状数

        patches = self.im_to_patches(x)  # 图像转补丁
        num_spatial = patches.shape[1]  # 空间补丁数
        feat_dim = patches.shape[2]  # 特征维度

        patches = patches.reshape(num_tubelets, T, num_spatial, feat_dim)  # 重塑
        patches = patches.permute(0, 2, 1, 3).reshape(
            num_tubelets, num_spatial, T * feat_dim
        )  # 排列并重塑

        patches = self.video_embedder(patches)  # 视频嵌入

        patches, _ = self.apply_pos_enc(patches, input_size=x.shape[2:])  # 应用位置编码
        patches = self.cls_token(patches)  # 添加CLS令牌
        patches = self.patch_normalizer(patches)  # 补丁归一化
        return patches

    @property
    def apply_cls_token(self):
        """是否应用CLS令牌"""
        return self.cls_token.enabled

    @property
    def num_cls_tokens(self):
        """CLS令牌数量"""
        return self.cls_token.num_tokens

    @property
    def num_cls_patches(self):
        """CLS补丁数量"""
        return self.cls_token.num_patches

    @property
    def num_registers(self):
        """寄存器数量"""
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        """跳过的令牌数（CLS+寄存器）"""
        return self.num_cls_tokens + self.num_registers

    def _load_embed(self, src_embed: torch.Tensor, targ_embed: nn.Parameter):
        """加载位置嵌入，支持不同尺寸的双三次插值"""
        if src_embed.shape != targ_embed.shape:  # 形状不匹配时插值
            src_size = int(math.sqrt(src_embed.shape[1]))

            assert (
                src_size**2 == src_embed.shape[1]
            ), "Unable to interpolate non-square embedding"

            src_embed = rearrange(
                src_embed, "b (h w) c -> b c h w", h=src_size, w=src_size
            )
            src_embed = F.interpolate(
                src_embed,
                size=(self.num_rows, self.num_cols),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_embed = rearrange(src_embed, "b c h w -> b (h w) c")
        targ_embed.data.copy_(src_embed)  # 复制嵌入

    def _load_projection(
        self, src_proj_weight: torch.Tensor, targ_proj_weight: torch.Tensor
    ):
        """加载投影权重，支持不同补丁尺寸的双三次插值"""
        if src_proj_weight.shape != targ_proj_weight.shape:  # 形状不匹配时插值
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))

            assert (src_patch_size**2) * 3 == src_proj_weight.shape[
                1
            ], "Unable to interpolate non-square patch size"

            src_proj_weight = rearrange(
                src_proj_weight,
                "b (c h w) -> b c h w",
                c=3,
                h=src_patch_size,
                w=src_patch_size,
            )
            src_proj_weight = F.interpolate(
                src_proj_weight,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_proj_weight = rearrange(src_proj_weight, "b c h w -> b (c h w)")
        targ_proj_weight.data.copy_(src_proj_weight)  # 复制投影权重

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """将图像转换为补丁嵌入"""
        patches = self.im_to_patches(x)  # 图像转补丁
        patches = self.embedder(patches)  # 补丁嵌入
        return patches

    def apply_pos_enc(
        self,
        patches: torch.Tensor,  # 补丁
        patch_idxs: torch.Tensor | None = None,  # 补丁索引
        input_size: tuple[int, int] | None = None,  # 输入尺寸
    ) -> torch.Tensor:
        """应用位置编码到补丁"""
        if not self.abs_pos:  # 不使用绝对位置编码
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)  # 获取位置编码

        if self.training and self.pos_dropout > 0:  # 训练时应用dropout
            keeps = (
                torch.rand(
                    patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device
                )
                > self.pos_dropout
            )
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc  # 返回添加位置编码后的补丁

    def get_pos_enc(
        self,
        batch_size: int,  # 批次大小
        patch_idxs: torch.Tensor | None = None,  # 补丁索引
        input_size: tuple[int, int] | None = None,  # 输入尺寸
    ) -> torch.Tensor:
        """获取位置编码"""
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)  # 计算输入维度

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)  # 获取位置嵌入

        if patch_idxs is None:  # 无索引时直接返回
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])  # 扩展索引

        pos_embed = torch.gather(
            pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs
        )  # 根据索引收集
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: tuple[int, int]):
        """获取位置嵌入，支持插值和窗口选择"""
        if (self.num_rows, self.num_cols) == input_dims:  # 尺寸匹配
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(
            0, 3, 1, 2
        )  # 重塑位置嵌入

        def window_select(pos_embed):
            """窗口选择，裁剪到输入尺寸"""
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.cpe_mode:  # CPE模式
            max_dim = max(input_dims)  # 最大维度
            pos_embed = F.interpolate(
                pos_embed.float(),
                size=(max_dim, max_dim),
                align_corners=False,
                mode="bilinear",
            ).to(pos_embed.dtype)  # 双线性插值

            pos_embed = window_select(pos_embed)  # 窗口选择
        else:
            pos_embed = window_select(pos_embed)  # 窗口选择

        if pos_embed.shape[-2:] != input_dims:  # 仍然不匹配时插值
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)  # 展平

        return pos_embed


class Im2Patches(nn.Module):
    """图像转补丁模块"""
    def __init__(self, patch_size: int):
        """初始化，设置补丁大小"""
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将图像转换为补丁序列"""
        if self.patch_size == 1:  # 补丁大小为1时直接展平
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size  # 行方向补丁数
        px = x.shape[-1] // self.patch_size  # 列方向补丁数
        patches = rearrange(
            x,
            "b c (py yy) (px xx) -> b (py px) (c yy xx)",
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )  # 重排为补丁序列
        return patches


class ViTPatchLinear(nn.Linear):
    """ViT补丁线性投影层"""
    def __init__(self, patch_size: int, embed_dim: int, bias: bool = False, **factory):
        """初始化补丁线性投影，输入维度为3*patch_size^2"""
        super().__init__(3 * (patch_size**2), embed_dim, bias=bias, **factory)
        self.patch_size = patch_size


class RadioInternVisionModel(nn.Module):
    """Radio InternVision视觉模型"""
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig = None,  # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
    ) -> None:
        """初始化Radio InternVision模型，配置补丁生成器和编码器"""
        super().__init__()

        self.config = config  # 保存配置
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            to_2tuple(config.patch_size), config.image_size
        )
        max_img_size = int(
            round(config.max_img_size / config.patch_size) * config.patch_size
        )
        video_temporal_patch_size = getattr(config, "video_temporal_patch_size", 1)
        separate_video_embedder = getattr(config, "separate_video_embedder", True)

        self.patch_generator = ViTPatchGenerator(
            config.patch_size,
            config.hidden_size,
            input_dims=self.img_size,
            max_input_dims=max_img_size,
            cls_token=True,
            register_multiple=config.reg_tokens,
            video_temporal_patch_size=video_temporal_patch_size,
            separate_video_embedder=separate_video_embedder,
        )

        self.encoder = InternVisionEncoder(config=config, quant_config=quant_config)

    def _init_img_size(self, patch_size, img_size: int | tuple[int, int]):
        """初始化图像尺寸，计算网格大小和补丁数"""
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def get_input_embeddings(self):
        """获取输入嵌入"""
        return self.embeddings

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        """视觉模型前向传播"""
        assert self.patch_generator is not None
        hidden_states = self.patch_generator(x)  # 生成补丁
        encoder_outputs = self.encoder.forward(inputs_embeds=hidden_states)  # 编码器前向
        assert isinstance(encoder_outputs, BaseModelOutput)
        return encoder_outputs.last_hidden_state  # 返回最后隐藏状态


class RadioModel(nn.Module):
    """Radio视觉模型，支持标准图像、动态尺寸和视频时序压缩"""
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: QuantizationConfig | None = None,  # 量化配置
    ) -> None:
        """初始化Radio模型"""
        super().__init__()

        self.config = config  # 保存配置
        self.model = RadioInternVisionModel(
            config=config,
            quant_config=quant_config,
        )

    def forward(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor] | None = None,  # 像素值
        num_frames: int | None = None,  # 帧数
    ) -> torch.FloatTensor:
        """Radio模型前向传播，根据输入类型选择处理模式"""
        if (
            num_frames is not None
            and getattr(self.config, "video_temporal_patch_size", 1) > 1
        ):
            return self._forward_video_temporal(pixel_values, num_frames)  # 视频时序模式
        if isinstance(pixel_values, list):
            return self._forward_dynamic(pixel_values)  # 动态尺寸模式
        y = self.model(pixel_values)  # 标准模式
        return self._extract_final(y)

    def _forward_dynamic(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int]]:
        """Process variable-size images with ragged packing via cu_seqlens."""
        """使用不规则打包处理可变尺寸图像"""
        patch_gen = self.model.patch_generator
        all_patches = []
        seqlens = [0]

        for img in images:  # 遍历每张图像
            patches = patch_gen(img)  # 生成补丁
            seq_len = patches.shape[1]  # 序列长度
            all_patches.append(patches.squeeze(0))  # 添加到列表
            seqlens.append(seqlens[-1] + seq_len)  # 累计长度

        hidden = torch.cat(all_patches, dim=0).unsqueeze(0)  # 拼接所有补丁
        cu_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=hidden.device)  # 累计序列长度

        out = self.model.encoder.forward(inputs_embeds=hidden, cu_seqlens=cu_seqlens)  # 编码器前向
        features = out.last_hidden_state  # 获取特征

        num_skip = patch_gen.num_skip  # 跳过的令牌数
        per_image_features = []
        num_patches_list = []
        for i in range(len(images)):  # 遍历每张图像
            start = seqlens[i] + num_skip  # 起始位置（跳过CLS+寄存器）
            end = seqlens[i + 1]  # 结束位置
            per_image_features.append(features[0, start:end])  # 提取特征
            num_patches_list.append(end - start)  # 记录补丁数

        return (
            torch.cat(per_image_features, dim=0).unsqueeze(0),
            num_patches_list,
        )

    def _forward_video_temporal(
        self, pixel_values: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Process video frames with temporal compression (tubelet grouping)."""
        """使用时序压缩（管状分组）处理视频帧"""
        T = self.config.video_temporal_patch_size
        patch_gen = self.model.patch_generator

        patches = patch_gen.forward_video(pixel_values, T)  # 生成视频补丁
        num_tubelets = patches.shape[0]  # 管状数
        seq_per_tubelet = patches.shape[1]  # 每个管状的序列长度

        cu_seqlens = torch.arange(
            0,
            (num_tubelets + 1) * seq_per_tubelet,
            seq_per_tubelet,
            dtype=torch.int32,
            device=patches.device,
        )  # 累计序列长度
        packed = patches.reshape(1, -1, patches.shape[-1])  # 打包

        out = self.model.encoder.forward(inputs_embeds=packed, cu_seqlens=cu_seqlens)  # 编码
        features = out.last_hidden_state.reshape(num_tubelets, seq_per_tubelet, -1)  # 重塑

        num_skip = patch_gen.num_skip  # 跳过令牌数
        return features[:, num_skip:]  # 返回去除CLS+寄存器的特征

    def load_weights(self, weights) -> set[str]:
        """加载模型权重，处理名称重映射"""
        remap_substrings = {
            "attn": "attn.attn",
            "qkv": "qkv_proj",
            "blocks": "encoder.layers",
        }
        remap_prefixes = {
            "radio_model.": "",
        }

        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters())

        if isinstance(weights, dict):  # 字典格式权重
            weights_list = list(weights.items())
        else:
            weights_list = list(weights)

        for name, weight in weights_list:
            if not name.startswith("radio_model."):
                # Skip non-radio weights
                continue
            name = replace_substrings(name, remap_substrings)  # 替换子串
            name = replace_prefix(name, remap_prefixes)  # 替换前缀
            if name and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(name)
                if "video_embedder" in name:  # 标记视频嵌入器已加载
                    self.model.patch_generator._video_embedder_loaded = True

        return loaded_params

    def _extract_final(self, y: torch.Tensor):
        """提取最终特征，去除CLS和寄存器令牌"""
        # Remove CLS + REGISTERS tokens
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            all_feat = y[:, patch_gen.num_skip :]

        return all_feat
