# MiniCPM-V视觉语言模型实现（多版本支持）
# 该模块实现了MiniCPM-V系列视觉语言模型，支持2.6、4.0、4.5、4.6四个版本
# 核心组件：BaseResampler、Resampler2_5、Resampler4_5、MiniCPMBaseModel、
# MiniCPMV2_6、MiniCPMV4_0、MiniCPMV4_5、MiniCPMV4_6、MiniCPMV
# 版本差异：2.6/4.0使用Qwen2/Llama+Resampler2_5，4.5使用Qwen3+Resampler4_5，
# 4.6使用Qwen3.5+MiniCPMV_VisionTransformer+MiniCPMV_Merger（纯MLP连接器）
# 位置编码：2D sincos位置嵌入，4.5版本额外支持时间位置嵌入

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
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
"""Inference-only MiniCPM-V model compatible with HuggingFace weights."""

import types  # 导入类型模块
from functools import partial  # 导入偏函数
from itertools import chain  # 导入链式迭代器
from typing import (  # 导入类型注解
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import torch.types  # 导入PyTorch类型
from PIL import Image  # 导入PIL图像
from torch import nn  # 导入PyTorch神经网络模块
from torch.nn.init import trunc_normal_  # 导入截断正态初始化
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入logits处理器
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternTokenPairs,  # 多模态token对填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputFormat,  # 多模态输入格式
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.utils import set_default_torch_dtype  # 导入默认数据类型设置
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.idefics2 import Idefics2VisionTransformer  # 导入Idefics2视觉Transformer
from sglang.srt.models.llama import LlamaConfig, LlamaForCausalLM  # 导入Llama模型
from sglang.srt.models.minicpmv_vit import (  # 导入MiniCPM-V ViT组件
    MiniCPMV_Merger,  # MiniCPM-V合并器
    MiniCPMV_VisionTransformer,  # MiniCPM-V视觉Transformer
)
from sglang.srt.models.qwen2 import Qwen2Config, Qwen2ForCausalLM  # 导入Qwen2模型
from sglang.srt.models.qwen3 import Qwen3Config, Qwen3ForCausalLM  # 导入Qwen3模型
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM  # 导入Qwen3.5模型
from sglang.srt.utils import add_prefix, flatten_nested_list, get_device  # 导入工具函数

RawImageType = Union[Image.Image, torch.Tensor]  # 原始图像类型


# sin/cos positional embedding helpers are adapted from:  # sin/cos位置嵌入辅助函数适配自
# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray, version: Tuple[int, int] = (2, 0)
) -> torch.Tensor:
    """从1D网格生成sincos位置嵌入"""
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,) / (H, W)
    out: (M, D) / (H, W, D)
    """
    assert embed_dim % 2 == 0  # 嵌入维度必须为偶数
    omega = np.arange(embed_dim // 2, dtype=np.float32)  # 频率索引
    omega /= embed_dim / 2.0  # 归一化
    omega = 1.0 / 10000**omega  # (D/2,)  # 频率

    if version == (2, 0):  # 版本2.0
        pos = pos.reshape(-1)  # (M,)  # 展平
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product  # 外积
        emb_sin = np.sin(out)  # (M, D/2)  # sin分量
        emb_cos = np.cos(out)  # (M, D/2)  # cos分量
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)  # 拼接
    else:  # 版本2.5+
        out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product  # 外积
        emb_sin = np.sin(out)  # (H, W, D/2)  # sin分量
        emb_cos = np.cos(out)  # (H, W, D/2)  # cos分量
        emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)  # 拼接
    return emb


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: np.ndarray, version: Tuple[int, int] = (2, 0)
) -> torch.Tensor:
    """从2D网格生成sincos位置嵌入"""
    assert embed_dim % 2 == 0  # 嵌入维度必须为偶数

    # use half of dimensions to encode grid_h  # 用一半维度编码高度
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], version
    )  # (H*W, D/2) or (H, W, D/2)  # 高度方向嵌入
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], version
    )  # (H*W, D/2) or (H, W, D/2)  # 宽度方向嵌入

    if version == (2, 0):  # 版本2.0
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    else:  # 版本2.5+
        emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    cls_token: bool = False,
    version: Tuple[int, int] = (2, 0),
) -> torch.Tensor:
    """生成2D sincos位置嵌入"""
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
                [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):  # 正方形网格
        grid_h_size, grid_w_size = grid_size, grid_size
    else:  # 矩形网格
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)  # 高度网格
    grid_w = np.arange(grid_w_size, dtype=np.float32)  # 宽度网格
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first  # w先
    grid = np.stack(grid, axis=0)  # 堆叠
    assert isinstance(grid, np.ndarray) and grid.shape == (2, grid_h_size, grid_w_size)

    if version == (2, 0):  # 版本2.0
        grid = grid.reshape([2, 1, grid_h_size, grid_w_size])  # 重塑
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
        if cls_token:  # 有CLS token
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    else:  # 版本2.5+
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
    return pos_embed


class MiniCPMVImagePixelInputs(TypedDict):
    """MiniCPM-V图像像素输入"""
    type: Literal["pixel_values"]
    data: List[torch.Tensor]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that the image size may vary, so we pass it as a list
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """

    tgt_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class MiniCPMVImageEmbeddingInputs(TypedDict):
    """MiniCPM-V图像嵌入输入"""
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """


MiniCPMVImageInputs = Union[MiniCPMVImagePixelInputs, MiniCPMVImageEmbeddingInputs]  # 图像输入联合类型

DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)  # 默认层归一化


class BaseResampler(nn.Module):
    """基础重采样器：2D感知器重采样网络"""
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb.
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        do_post_projection: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.num_queries = num_queries  # 查询数量
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 头数

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))  # 可学习查询
        trunc_normal_(self.query, std=0.02)  # 截断正态初始化
        if kv_dim is not None and kv_dim != embed_dim:  # 需要KV投影
            self.kv_proj = ReplicatedLinear(
                kv_dim,
                embed_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_proj", prefix),
            )
        else:  # 不需要KV投影
            # Maintain the same return value with ReplicatedLinear.forward  # 保持与ReplicatedLinear.forward相同的返回值
            self.kv_proj = lambda *args, **kwargs: (  # type: ignore # noqa
                nn.Identity()(*args, **kwargs),
                None,
            )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)  # 多头交叉注意力
        self.ln_q = norm_layer(embed_dim)  # 查询层归一化
        self.ln_kv = norm_layer(embed_dim)  # KV层归一化
        self.do_post_projection = do_post_projection  # 是否做后投影
        self.ln_post = norm_layer(embed_dim) if do_post_projection else None  # 后归一化
        self.proj = (  # 后投影矩阵
            nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))
            if do_post_projection
            else None
        )

    def _init_weights(self, m: nn.Module) -> None:
        """初始化权重"""
        if isinstance(m, nn.Linear):  # 线性层
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):  # 层归一化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _repeat(self, query, N: int):
        """重复查询N次"""
        return query.unsqueeze(1).repeat(1, N, 1)


class Resampler2_5(BaseResampler):
    """2.5版本重采样器：带2D sincos位置嵌入的感知器重采样"""

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        max_size: Tuple[int, int] = (70, 70),
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_queries,
            embed_dim,
            num_heads,
            kv_dim,
            norm_layer,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.max_size = max_size  # 最大网格尺寸
        self._set_2d_pos_cache(self.max_size)  # 设置2D位置缓存

        self.apply(self._init_weights)  # 初始化权重

    def _set_2d_pos_cache(
        self, max_size: Tuple[int, int], device: torch.types.Device = "cpu"
    ) -> None:
        """设置2D位置嵌入缓存"""
        pos_embed_arr = get_2d_sincos_pos_embed(
            self.embed_dim, max_size, version=(2, 5)
        )
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)  # 注册缓冲区

    def _adjust_pos_cache(
        self, tgt_sizes: torch.Tensor, device: torch.types.Device
    ) -> None:
        """调整位置嵌入缓存（按需扩展）"""
        max_h = tgt_sizes[:, 0].max().item()  # 最大高度
        max_w = tgt_sizes[:, 1].max().item()  # 最大宽度
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:  # 需要扩展
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)  # 重新设置缓存

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        """2.5重采样器前向传播"""
        assert x.shape[0] == tgt_sizes.shape[0]  # 批次大小必须匹配
        bs = x.shape[0]

        device = x.device  # 设备
        dtype = x.dtype  # 数据类型

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]  # 每个图像的补丁数

        self._adjust_pos_cache(tgt_sizes, device=device)  # 调整位置缓存

        max_patch_len = patch_len.max().item()  # 最大补丁数
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros(  # 键填充掩码
            (bs, max_patch_len), dtype=torch.bool, device=device
        )

        pos_embed = []  # 位置嵌入列表
        for i in range(bs):  # 遍历每个图像
            tgt_h, tgt_w = tgt_sizes[i].tolist()
            pos_embed.append(
                self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
            )  # patches * D
            key_padding_mask[i, patch_len[i] :] = True  # 标记填充位置
        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0
        ).permute(
            1, 0, 2
        )  # BLD => L * B * D  # 填充并转置
        x, _ = self.kv_proj(x)  # B * L * D  # KV投影
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D  # 层归一化并转置

        q = self.ln_q(self.query)  # Q * D  # 查询归一化

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D  # 重复查询
            x + pos_embed,  # L * B * D +  L * B * D  # K+位置嵌入
            x,  # V
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D  # 转置回来

        x = self.ln_post(x)  # 后归一化
        x = x @ self.proj  # 后投影
        return x


class Resampler4_5(BaseResampler):
    """4.5版本重采样器：带2D和时间位置嵌入的感知器重采样"""

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        max_size: tuple[int, int] = (70, 70),
        max_temporal_size=36000,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_queries,
            embed_dim,
            num_heads,
            kv_dim,
            norm_layer,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.max_size = max_size  # 最大空间尺寸
        self.max_temporal_size = max_temporal_size  # 最大时间尺寸

        self._set_2d_pos_cache(self.max_size)  # 设置2D位置缓存
        self._set_temporal_pos_cache(self.max_temporal_size)  # 设置时间位置缓存
        self.apply(self._init_weights)  # 初始化权重

    def get_1d_sincos_pos_embed_from_temporal_size(
        self, embed_dim: int, pos: np.ndarray
    ):
        """从时间尺寸生成1D sincos位置嵌入"""
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0  # 嵌入维度必须为偶数
        omega = np.arange(embed_dim // 2, dtype=np.float32)  # 频率索引
        omega /= embed_dim / 2.0  # 归一化
        omega = 1.0 / 10000**omega  # (D/2,)  # 频率

        pos = pos.reshape(-1)  # (M,)  # 展平
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product  # 外积

        emb_sin = np.sin(out)  # (M, D/2)  # sin分量
        emb_cos = np.cos(out)  # (M, D/2)  # cos分量

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)  # 拼接
        return emb

    def _set_2d_pos_cache(
        self, max_size: tuple[int, int], device: torch.types.Device = "cpu"
    ) -> None:
        """设置2D位置嵌入缓存"""
        pos_embed_arr = get_2d_sincos_pos_embed(
            self.embed_dim, max_size, version=(2, 5)
        )
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(
        self, tgt_sizes: torch.Tensor, device: torch.types.Device
    ) -> None:
        """调整2D位置嵌入缓存"""
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def _set_temporal_pos_cache(
        self, max_temporal_size: int, device: torch.types.Device = "cpu"
    ) -> None:
        """设置时间位置嵌入缓存"""
        temporal_size = np.arange(max_temporal_size, dtype=np.float32)
        pos_embed = (
            torch.from_numpy(
                self.get_1d_sincos_pos_embed_from_temporal_size(
                    self.embed_dim, temporal_size
                )
            )
            .float()
            .to(device)
        )
        self.register_buffer("temporal_pos_embed", pos_embed, persistent=False)  # 注册缓冲区

    def _adjust_temporal_pos_cache(
        self, max_temporal_size: int, device: torch.types.Device = "cpu"
    ):
        """调整时间位置嵌入缓存"""
        if max_temporal_size > self.max_temporal_size:
            self.max_temporal_size = max_temporal_size
            self._set_temporal_pos_cache(self.max_temporal_size, device)

    def forward(
        self, x: torch.Tensor, tgt_sizes: torch.Tensor, temporal_ids=None
    ) -> torch.Tensor:
        """4.5重采样器前向传播（支持时间位置嵌入）"""
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        temporal_pos_emb = False  # 是否使用时间位置嵌入
        temporal_ids_flatten = None
        if temporal_ids is not None:  # 有时间ID
            # example: [[-1], [-1], [2, 6, 9]]
            temporal_ids_flatten = list(chain.from_iterable(temporal_ids))  # 展平
            max_temporal_size = max(temporal_ids_flatten)
            if max_temporal_size > -1:  # 有效的时间ID
                temporal_pos_emb = True
            if max_temporal_size > self.max_temporal_size:
                self._adjust_temporal_pos_cache(max_temporal_size, device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros(
            (bs, max_patch_len), dtype=torch.bool, device=device
        )

        x, _ = self.kv_proj(x)  # B * L * D  # KV投影
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D  # 层归一化并转置
        q = self.ln_q(self.query)  # Q * D  # 查询归一化

        pos_embed_2d = []  # 2D位置嵌入
        pos_embed_temporal = []  # 时间位置嵌入
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            if temporal_pos_emb:  # 使用时间位置嵌入
                if temporal_ids_flatten[i] == -1:  # 无时间ID
                    pos_embed_temporal.append(
                        torch.zeros(self.embed_dim, dtype=dtype, device=device)
                    )
                else:  # 有时间ID
                    pos_embed_temporal.append(
                        self.temporal_pos_embed[temporal_ids_flatten[i]].to(dtype)
                    )  # D

            pos_embed_2d.append(
                self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
            )  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed_2d = torch.nn.utils.rnn.pad_sequence(
            pos_embed_2d, batch_first=True, padding_value=0.0
        ).permute(
            1, 0, 2
        )  # BLD => L * B * D

        k = x  # K
        v = x + pos_embed_2d  # V + 2D位置嵌入

        if pos_embed_temporal:  # 有时间位置嵌入
            k += torch.stack(pos_embed_temporal, dim=0)  # K += 时间嵌入
            bs = len(temporal_ids)
            merge_k = []
            merge_v = []
            merge_key_padding_mask = []

            start = 0
            for tp in temporal_ids:  # 按时间组重排
                end = start + len(tp)
                # # L * (end-start) * D -> (end-start) * L * D -> 1 * L*(end-start) * D
                merge_k.append(
                    k[:, start:end, :].permute(1, 0, 2).reshape(-1, self.embed_dim)
                )
                merge_v.append(
                    v[:, start:end, :].permute(1, 0, 2).reshape(-1, self.embed_dim)
                )
                merge_key_padding_mask.append(
                    key_padding_mask[start:end, :].reshape(-1, 1)
                )

                start = end

            k = torch.nn.utils.rnn.pad_sequence(
                merge_k, batch_first=True, padding_value=0.0
            ).permute(
                1, 0, 2
            )  # L*(end-start)
            v = torch.nn.utils.rnn.pad_sequence(
                merge_v, batch_first=True, padding_value=0.0
            ).permute(
                1, 0, 2
            )  # L*(end-start)
            key_padding_mask = torch.nn.utils.rnn.pad_sequence(
                merge_key_padding_mask, batch_first=True, padding_value=True
            ).squeeze(-1)

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            k,  # L * B * D +  L * B * D
            v,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)  # 后归一化
        x = x @ self.proj  # 后投影
        return x


def get_version_by_config(config: PretrainedConfig) -> Tuple[int, ...]:
    """根据配置获取版本号"""
    # 4.6 ships its own ``model_type`` instead of a numeric ``version``.  # 4.6使用自己的model_type而非数字version
    if getattr(config, "model_type", None) == "minicpmv4_6":
        return 4, 6

    version_float = getattr(config, "version", None)

    # The old configs do not include version number  # 旧配置不包含版本号
    # TODO: Remove this after the HF repos are updated  # 待办：HF仓库更新后移除
    if version_float is None:
        if config.hidden_size == 2304 and config.query_num == 64:  # 2.0版本
            return 2, 0
        return 2, 5  # 默认2.5

    version_str = str(version_float)
    return tuple(int(x) for x in version_str.split("."))


class MiniCPMBaseModel(nn.Module):
    """MiniCPM-V基础模型抽象类"""
    """
    The abstract class of MiniCPMV can only be inherited, but cannot be
    instantiated.
    """

    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # All MiniCPM-V models disable `tie_word_embeddings` but  # 所有MiniCPM-V模型禁用tie_word_embeddings
        # `PretrainedConfig.tie_word_embeddings` defaults to True; we cannot  # 但PretrainedConfig默认为True
        # check `tie_word_embeddings` until SGLang integrate MiniCPM-V model  # 在SGLang集成MiniCPM-V模型前无法检查
        # and config class  # 和配置类
        self.config = config

        self.version = get_version_by_config(self.config)  # 版本号
        self.llm = self.init_llm(
            config=config, quant_config=quant_config, prefix=add_prefix("llm", prefix)
        )  # LLM
        self.vpm = self.init_vision_module(
            config, quant_config, add_prefix("vpm", prefix)
        )  # 视觉模块
        self.vision_dim = (  # 视觉维度
            self.vpm.embed_dim
            if self.version == (2, 0)
            else self.vpm.embeddings.embed_dim
        )
        self.embed_dim = self.config.hidden_size  # 嵌入维度

        self.resampler = self.init_resampler(
            self.embed_dim,
            self.vision_dim,
            quant_config=quant_config,
            prefix=add_prefix("resampler", prefix),
        )  # 重采样器

        self.logits_processor = LogitsProcessor(config)  # logits处理器

    def _get_image_bounds(
        self,
        input_ids: torch.Tensor,
        pad_values: List[int],
        im_start_id: int,
        im_end_id: int,
        slice_start_id: Optional[int] = None,
        slice_end_id: Optional[int] = None,
    ) -> torch.Tensor:
        """获取图像边界（起始和结束token ID）"""
        """
        Returns a tensor indicating the bounds (start and end token ids) of the images
        """
        # All the images in the batch should share the same special image  # 批次中所有图像应共享相同的特殊图像
        # bound token ids.  # 边界token ID
        start_cond = input_ids == im_start_id  # 起始条件
        end_cond = input_ids == im_end_id  # 结束条件
        if slice_start_id is not None:  # 有切片起始ID
            start_cond |= input_ids == slice_start_id
            end_cond |= input_ids == slice_end_id

        (image_start_tokens,) = torch.where(start_cond)  # 起始token位置
        image_start_tokens += 1  # 偏移1
        (image_end_tokens,) = torch.where(end_cond)  # 结束token位置

        # the im_start_id sometimes can be cached as prefix, but it is needed for the embedding of the images  # im_start_id有时作为前缀缓存，但图像嵌入需要它
        if len(image_start_tokens) != len(image_end_tokens):  # 数量不匹配
            if (
                len(image_start_tokens) + 1 == len(image_end_tokens)
                and input_ids[0] in pad_values
                and len(image_start_tokens) != 0
                and len(image_end_tokens) != 0
                and image_end_tokens[0] < image_start_tokens[0]
            ):  # 前缀缓存情况
                image_start_tokens = torch.cat(
                    [
                        torch.tensor([0], device=image_start_tokens.device),
                        image_start_tokens,
                    ]
                )
        valid_image_nums = min(len(image_start_tokens), len(image_end_tokens))  # 有效图像数

        if valid_image_nums == 0:  # 无图像
            return torch.zeros((0, 2), device=input_ids.device)

        # Filter out pairs where start_token >= end_token  # 过滤起始>=结束的对
        valid_pairs = []
        for i in range(valid_image_nums):
            start_token = image_start_tokens[i]
            end_token = image_end_tokens[i]
            if start_token < end_token:
                valid_pairs.append((start_token, end_token))

        if not valid_pairs:  # 无有效对
            return torch.zeros((0, 2), device=input_ids.device)

        # Convert valid pairs to tensor  # 转换为张量
        valid_pairs_tensor = torch.tensor(valid_pairs, device=input_ids.device)
        return valid_pairs_tensor

    def _parse_and_validate_inputs(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> Optional[MiniCPMVImageInputs]:
        """解析和验证多模态输入"""
        pixel_values = kwargs.pop("pixel_values", [])  # 像素值
        tgt_sizes = kwargs.pop("tgt_sizes", [])  # 目标尺寸
        im_start_id = kwargs.pop("im_start_id", None)  # 图像起始ID
        im_end_id = kwargs.pop("im_end_id", None)  # 图像结束ID
        slice_start_id = kwargs.pop("slice_start_id", None)  # 切片起始ID
        slice_end_id = kwargs.pop("slice_end_id", None)  # 切片结束ID
        image_embeds = kwargs.pop("image_embeds", None)  # 图像嵌入
        pad_values = kwargs.pop("pad_values", None)  # 填充值

        if image_embeds is not None:  # 使用预计算嵌入
            image_bounds = self._get_image_bounds(
                input_ids=input_ids,
                pad_values=pad_values,
                im_start_id=im_start_id,
                im_end_id=im_end_id,
                slice_start_id=slice_start_id,
                slice_end_id=slice_end_id,
            )
            if not isinstance(image_embeds, (torch.Tensor, list)):  # 类型检查
                raise ValueError(
                    f"Incorrect type of image embeds. "
                    f"Got type: {type(image_embeds)}"
                )

            if isinstance(image_embeds, list):  # 列表转张量
                image_embeds = torch.cat(image_embeds)

            return MiniCPMVImageEmbeddingInputs(
                image_bounds=image_bounds,
                data=image_embeds,
                type="image_embeds",
            )

        image_bounds = self._get_image_bounds(  # 获取图像边界
            input_ids=input_ids,
            pad_values=pad_values,
            im_start_id=im_start_id,
            im_end_id=im_end_id,
            slice_start_id=slice_start_id,
            slice_end_id=slice_end_id,
        )
        return MiniCPMVImagePixelInputs(
            image_bounds=image_bounds.to(device=input_ids.device),
            data=pixel_values,
            tgt_sizes=tgt_sizes,
            type="pixel_values",
        )

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        image_inputs: Optional[MiniCPMVImageInputs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取多模态嵌入（文本+视觉）"""
        vlm_embedding: torch.Tensor = self.llm.get_input_embeddings(input_ids)  # 文本嵌入

        if image_inputs is None:  # No image  # 无图像
            vision_hidden_states = torch.tensor([], device=input_ids.device)
        else:
            if image_inputs["type"] == "image_embeds":  # 预计算嵌入
                vision_hidden_states = (
                    image_inputs["data"]
                    .type(vlm_embedding.dtype)
                    .to(vlm_embedding.device)
                )
            else:  # 从像素值计算
                vision_hidden_states = self.get_vision_hidden_states(image_inputs)
            # See NOTE in _parse_and_validate_inputs
            image_bounds = image_inputs["image_bounds"]  # 图像边界
            if len(image_bounds) > 0:  # 有图像
                image_indices = torch.stack(
                    [
                        torch.arange(start, end, dtype=torch.long)
                        for start, end in image_bounds.tolist()
                    ]
                ).to(vlm_embedding.device)  # 图像token索引

                vlm_embedding.scatter_(  # 将视觉嵌入散布到文本嵌入中
                    0,
                    image_indices.view(-1, 1).repeat(1, vlm_embedding.shape[-1]),
                    vision_hidden_states.view(-1, vision_hidden_states.shape[-1]),
                )

        return vlm_embedding, vision_hidden_states

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.llm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,

        **kwargs: Any,
    ) -> torch.Tensor:
        """基础模型前向传播"""
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.llm,
            positions=positions,
        )
        return hidden_states

    def init_llm(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化LLM（子类实现）"""
        raise NotImplementedError

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        """初始化视觉模块（子类实现）"""
        raise NotImplementedError

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化重采样器（子类实现）"""
        raise NotImplementedError

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取视觉嵌入（子类实现）"""
        raise NotImplementedError

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取图像特征（子类实现）"""
        raise NotImplementedError


class MiniCPMV2_6(MiniCPMBaseModel):
    """MiniCPM-V 2.6版本：Qwen2 + Idefics2VisionTransformer + Resampler2_5"""
    packed_modules_mapping = {  # 打包模块映射
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes  # LoRA特定属性
    supported_lora_modules = [
        # vision encoder  # 视觉编码器
        "fc1",
        "fc2",
        "out_proj",
        # language model  # 语言模型
        "qkv_proj",  # same name with vision encoder  # 与视觉编码器同名
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler  # 重采样器
        "kv_proj",
    ]

    # BitandBytes specific attributes  # BitandBytes特定属性
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        assert self.version == (2, 6)

    def init_llm(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化Qwen2 LLM"""
        return Qwen2ForCausalLM(config=config, quant_config=quant_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        """初始化Idefics2视觉Transformer"""
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:  # 丢弃最后一层
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化2.5重采样器"""
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.  # 2.6的重采样器与2.5一致
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device=get_device(), dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取视觉嵌入"""
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取图像特征"""
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:  # 预计算嵌入
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        # list of tensors  # 张量列表
        pixel_values = flatten_nested_list([item.feature for item in items])  # 像素值
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )  # 目标尺寸
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()  # 最大补丁数
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )  # 补丁注意力掩码

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]  # 掩码形状
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)  # 设置掩码

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )  # 视觉编码
        return self.resampler(vision_embedding, tgt_sizes)  # 重采样

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        """填充输入ID"""
        # Get all special token IDs  # 获取所有特殊token ID
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [(im_start_id, im_end_id), (slice_start_id, slice_end_id)]
        # Only increment data_idx on im_start (not slice_start) so all slices  # 只在im_start上递增data_idx，这样一张图像的所有切片
        # within one image share the same pad_value for per-image caching.  # 共享相同的pad_value用于每图像缓存
        pattern = MultiModalityDataPaddingPatternTokenPairs(
            media_token_pairs, data_start_token_ids=[im_start_id]
        )

        return pattern.pad_input_tokens(input_ids, image_inputs)


class MiniCPMV4_0(MiniCPMBaseModel):
    """MiniCPM-V 4.0版本：Llama + Idefics2VisionTransformer + Resampler2_5"""
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        # vision encoder
        "fc1",
        "fc2",
        "out_proj",
        # language model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler
        "kv_proj",
    ]

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        assert self.version == (4, 0)

    def init_llm(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化Llama LLM"""
        return LlamaForCausalLM(config=config, quant_config=quant_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        """初始化Idefics2视觉Transformer"""
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化2.5重采样器"""
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device=get_device(), dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取视觉嵌入"""
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取图像特征"""
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        # list of tensors
        pixel_values = flatten_nested_list([item.feature for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return self.resampler(vision_embedding, tgt_sizes)

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        """填充输入ID"""
        # Get all special token IDs
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [(im_start_id, im_end_id), (slice_start_id, slice_end_id)]
        # Only increment data_idx on im_start (not slice_start) so all slices
        # within one image share the same pad_value for per-image caching.
        pattern = MultiModalityDataPaddingPatternTokenPairs(
            media_token_pairs, data_start_token_ids=[im_start_id]
        )

        return pattern.pad_input_tokens(input_ids, image_inputs)


class MiniCPMV4_5(MiniCPMBaseModel):
    """MiniCPM-V 4.5版本：Qwen3 + Idefics2VisionTransformer + Resampler4_5"""
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        # vision encoder
        "fc1",
        "fc2",
        "out_proj",
        # language model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler
        "kv_proj",
    ]

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        assert self.version == (4, 5)

    def init_llm(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化Qwen3 LLM"""
        llm = Qwen3ForCausalLM(config=config, quant_config=quant_config, prefix=prefix)
        llm.get_input_embeddings = types.MethodType(
            lambda self: self.model.get_input_embeddings(), llm
        )  # 修补get_input_embeddings方法
        return llm

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        """初始化Idefics2视觉Transformer"""
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化4.5重采样器（带时间位置嵌入）"""
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler4_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device=get_device(), dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取视觉嵌入"""
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取图像特征"""
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        # list of tensors
        pixel_values = flatten_nested_list([item.feature for item in items])
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return self.resampler(vision_embedding, tgt_sizes)

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        """填充输入ID"""
        # Get all special token IDs
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [(im_start_id, im_end_id), (slice_start_id, slice_end_id)]
        # Only increment data_idx on im_start (not slice_start) so all slices
        # within one image share the same pad_value for per-image caching.
        pattern = MultiModalityDataPaddingPatternTokenPairs(
            media_token_pairs, data_start_token_ids=[im_start_id]
        )

        return pattern.pad_input_tokens(input_ids, image_inputs)

    def eval(self):
        """设置评估模式"""
        super().eval()
        return self


class MiniCPMV4_6(MiniCPMBaseModel):
    """MiniCPM-V 4.6版本：Qwen3.5 + MiniCPMV_VisionTransformer + MiniCPMV_Merger"""
    """MiniCPM-V 4.6.

    Differences vs 4.5:
      * mid-ViT compression (``MiniCPMV_VisionTransformer`` fires a 2x2 window
        attention + 2x2 fold at ``config.insert_layer_id``);
      * post-encoder connector is a pure MLP chain (``MiniCPMV_Merger``),
        not a Perceiver resampler;
      * LLM backbone is Qwen3.5;
      * ``config.downsample_mode`` toggles ``"16x"`` (mid-ViT + post merger)
        vs ``"4x"`` (skip mid-ViT, keep 4x more visual tokens).
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    supported_lora_modules = [
        # vision encoder + mid-ViT merger
        "fc1",
        "fc2",
        "out_proj",
        "linear_1",
        "linear_2",
        # language model
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]

    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        assert self.version == (4, 6)
        # ``Qwen3_5ForCausalLM`` returns plain hidden states (body only, no LM  # Qwen3_5ForCausalLM返回纯隐藏状态（仅主体，无LM头）
        # head, no LogitsProcessor). Add them here so the downstream sampler  # 无LogitsProcessor）。在此添加，使下游采样器
        # sees a ``LogitsProcessorOutput``. With ``tie_word_embeddings=True``  # 能看到LogitsProcessorOutput。使用tie_word_embeddings=True
        # (4.6 default) the head shares weights with the embedding.  # （4.6默认）头与嵌入共享权重
        text_config = config.text_config
        if getattr(text_config, "tie_word_embeddings", False):  # 权重绑定
            self.lm_head = self.llm.embed_tokens
        else:  # 独立LM头
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

    def init_llm(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化Qwen3.5 LLM"""
        # 4.6 nests the LLM config under ``text_config``.  # 4.6将LLM配置嵌套在text_config下
        return Qwen3_5ForCausalLM(
            config=config.text_config, quant_config=quant_config, prefix=prefix
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        """4.6前向传播：在基础例程上添加LM头和LogitsProcessor"""
        # Apply our lm_head + LogitsProcessor on top of the base routine; the  # 在基础例程之上应用LM头和LogitsProcessor
        # 4.6 LLM body (``Qwen3_5ForCausalLM``) returns plain hidden states,  # 4.6 LLM主体返回纯隐藏状态
        # unlike the ``Qwen3ForCausalLM`` 4.5 used.  # 与4.5使用的Qwen3ForCausalLM不同
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            **kwargs,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        """初始化MiniCPMV视觉Transformer（带中间层压缩）"""
        model = MiniCPMV_VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )
        if getattr(self.config, "drop_vision_last_layer", False):
            # The mid-ViT merger sits on the transformer (not encoder.layers),  # 中间ViT合并器位于transformer上（不在encoder.layers上）
            # so popping the last encoder layer leaves it untouched — same  # 因此弹出最后一个编码器层不会影响它——与4.5行为相同
            # behaviour as 4.5.
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化4.6合并器（纯MLP，替代Perceiver重采样器）"""
        # 4.6 replaces Resampler4_5 with a pure MLP. Method name kept so  # 4.6用纯MLP替代Resampler4_5。方法名保留以使
        # ``MiniCPMBaseModel.__init__`` doesn't need to branch.  # MiniCPMBaseModel.__init__不需要分支
        with set_default_torch_dtype(torch.float16):
            merger = MiniCPMV_Merger(
                config=self.config,
                quant_config=quant_config,
                prefix=prefix,
            )
        return merger.to(device=get_device(), dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取视觉嵌入（4.6返回隐藏状态和更新后的目标尺寸）"""
        hidden, _ = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            target_sizes=tgt_sizes,
        )
        return hidden

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取图像特征"""
        if items and items[0].format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            result = torch.cat([item.feature for item in items])
            return result.reshape(-1, result.shape[-1])

        pixel_values = flatten_nested_list([item.feature for item in items])  # 像素值
        tgt_sizes = torch.stack(
            flatten_nested_list([item.tgt_size for item in items]), dim=0
        )  # 目标尺寸
        assert len(pixel_values) == tgt_sizes.shape[0]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)
        mask_shapes = tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]
        patch_attn_mask[:, 0, :] = torch.arange(
            patch_attn_mask.size(2), device=patch_attn_mask.device
        ).unsqueeze(0) < mask_shapes.unsqueeze(1)

        use_vit_merger = getattr(self.config, "downsample_mode", "16x") != "4x"  # 是否使用ViT合并器

        vision_embedding, tgt_sizes_out = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            target_sizes=tgt_sizes,
            use_vit_merger=use_vit_merger,
        )  # 视觉编码+可选中间层压缩
        return self.resampler(vision_embedding, tgt_sizes_out)  # 合并器

    # Video frames take the same vision path as image patches; the mm  # 视频帧与图像补丁走相同的视觉路径
    # processor emits one ``MultimodalDataItem`` per patch regardless of  # mm处理器无论来源都为每个补丁发出一个MultimodalDataItem
    # source. sglang's dispatcher routes by ``get_{modality}_feature``.  # sglang的调度器按get_{modality}_feature路由
    get_video_feature = get_image_feature  # 视频特征与图像特征共用

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        """填充输入ID"""
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id
        slice_start_id: int = image_inputs.slice_start_id
        slice_end_id: int = image_inputs.slice_end_id

        media_token_pairs = [(im_start_id, im_end_id), (slice_start_id, slice_end_id)]
        pattern = MultiModalityDataPaddingPatternTokenPairs(
            media_token_pairs, data_start_token_ids=[im_start_id]
        )
        return pattern.pad_input_tokens(input_ids, image_inputs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载4.6权重：重映射前缀并委托给Qwen3.5"""
        """Remap 4.6 prefixes (``model.{vision_tower,merger,language_model}``)
        to sglang's (``vpm`` / ``resampler`` / ``llm``) and delegate the LLM
        portion to ``Qwen3_5ForCausalLM.load_weights`` — the Qwen3.5 hybrid
        backbone has its own stacked-param logic (``in_proj_a/b -> in_proj_ba``,
        ``in_proj_qkv/z -> in_proj_qkvz``) the legacy loader doesn't know.
        Vision-side still needs QKV stacking + ``out_proj -> proj`` rename.
        """

        llm_weights: List[Tuple[str, torch.Tensor]] = []  # LLM权重
        vision_weights: List[Tuple[str, torch.Tensor]] = []  # 视觉权重
        for name, w in weights:
            if name.startswith("model.language_model."):  # LLM权重
                llm_weights.append((name[len("model.language_model.") :], w))
                continue
            if name.startswith("model.vision_tower."):  # 视觉塔权重
                name = "vpm." + name[len("model.vision_tower.") :]
            elif name.startswith("model.merger."):  # 合并器权重
                name = "resampler." + name[len("model.merger.") :]
            vision_weights.append((name, w))

        self.llm.load_weights(iter(llm_weights))  # 加载LLM权重

        stacked_params_mapping = [  # 视觉侧堆叠参数映射
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in vision_weights:
            name = name.replace("self_attn.out_proj", "self_attn.proj")  # 重命名

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 匹配堆叠参数
                if weight_name not in name:
                    continue
                target = name.replace(weight_name, param_name)
                if target not in params_dict:
                    continue
                param = params_dict[target]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:  # 直接加载
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


_SUPPORT_VERSION = {  # 支持的版本映射
    (2, 6): MiniCPMV2_6,
    (4, 0): MiniCPMV4_0,
    (4, 5): MiniCPMV4_5,
    (4, 6): MiniCPMV4_6,
}


class MiniCPMV:
    """MiniCPM-V版本分发器：根据配置选择对应版本"""
    """
    Different versions of MiniCPMV use different visual encoders and LLMs,
    which is not conducive to the current integration logic of LoRA and
    bitsandbytes in SGLang. Therefore, it is necessary to separate them.
    """

    # Ensure that the LoRA support check passes when the class is not  # 确保类未初始化时LoRA支持检查通过
    # initialized, but set all these attributes to empty.  # 但将所有这些属性设为空
    packed_modules_mapping = {}
    supported_lora_modules = []
    embedding_modules = {}
    embedding_padding_modules = []

    minicpmv: nn.Module

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # 4.6 carries ``model_type == "minicpmv4_6"`` instead of a numeric  # 4.6使用model_type代替数字版本号
        # ``config.version``; older versionless configs keep the legacy  # 较旧的无版本配置保留旧的
        # ``(2, 6)`` default.  # (2, 6)默认值
        if getattr(config, "model_type", None) == "minicpmv4_6":
            version = (4, 6)
        elif not hasattr(config, "version"):
            version = (2, 6)
        else:
            version = str(config.version).split(".")
            version = tuple([int(x) for x in version])
        # Dispatch class based on version  # 根据版本分发类
        instance_class = _SUPPORT_VERSION.get(version)
        if instance_class is None:
            supported_versions = ", ".join(
                [f"{v[0]}.{v[1]}" for v in sorted(_SUPPORT_VERSION.keys())]
            )
            raise ValueError(
                f"Currently, MiniCPMV only supports versions "
                f"{supported_versions}. Got version: {version}"
            )

        try:
            minicpmv = instance_class(
                config=config, quant_config=quant_config, prefix=prefix
            )
            self.minicpmv = minicpmv
        except Exception as e:
            print(f"Failed to instantiate MiniCPMV: {e}")
            raise e
        self.config = config

    def __getattr__(self, name):
        """属性访问委托给内部minicpmv实例"""
        if name == "minicpmv":
            return None
        return getattr(self.minicpmv, name)

    def __call__(self, *args, **kwargs):
        """调用委托给内部minicpmv实例"""
        return self.minicpmv(*args, **kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载权重：委托给版本特定的加载器"""
        # Defer to the version-specific subclass loader if it overrides the  # 委托给版本特定的子类加载器
        # base (4.6 does — it needs prefix remap + Qwen3.5 LLM delegation).  # 4.6需要前缀重映射+Qwen3.5 LLM委托
        sub_loader = getattr(type(self.minicpmv), "load_weights", None)
        base_loader = getattr(MiniCPMBaseModel, "load_weights", None)
        if sub_loader is not None and sub_loader is not base_loader:
            return self.minicpmv.load_weights(weights)

        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.minicpmv.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq~" in name or "projector" in name:  # 跳过
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存
                # Models trained using ColossalAI may include these tensors in  # ColossalAI训练的模型可能包含
                # the checkpoint. Skip them.  # 跳过
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:  # 跳过不存在的视觉塔权重
                continue

            # adapt to VisionAttention  # 适配视觉注意力
            name = name.replace(r"self_attn.out_proj", r"self_attn.proj")

            if "sampler" in name:  # 重采样器权重
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 匹配堆叠参数
                # replace the name and load with customized loader  # 替换名称并自定义加载
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:  # 直接加载
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# Real subclass (not an `=` alias) so the model registry — which keys by  # 真子类（非=别名），使模型注册表——按键
# ``__name__`` — resolves the canonical 4.6 architecture name through  # __name__——通过MiniCPMV的版本分发工厂
# ``MiniCPMV``'s version-dispatch factory.  # 解析规范的4.6架构名
class MiniCPMV4_6ForConditionalGeneration(MiniCPMV):
    """MiniCPM-V 4.6条件生成模型（兼容旧名称）"""
    pass


EntryClass = [MiniCPMV, MiniCPMV4_6ForConditionalGeneration]  # 入口类列表
