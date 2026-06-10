# Kimi VL MoonViT 视觉编码器实现
# 该文件实现了 MoonViT 视觉编码器，包含2D旋转位置编码（RoPE）、
# 多头注意力（Flash Attention 2 / SDPA）、可学习2D插值位置编码、
# Patch嵌入、2D patch合并、MLP2、MoonViT编码器层和完整的视觉模型。
# 该文件仅供 kimi_vl.py 使用。
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
# This file is meant to be used in kimi_vl.py only  # 此文件仅供kimi_vl.py使用
# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math  # 导入数学库
from copy import deepcopy  # 导入深拷贝
from functools import cached_property  # 导入缓存属性装饰器
from typing import List, Optional, Sequence, Tuple, Union  # 导入类型注解

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from transformers.activations import ACT2FN  # 导入激活函数映射
from transformers.modeling_utils import PreTrainedModel  # 导入预训练模型基类

from sglang.kernel_api_logging import debug_kernel_api  # 导入内核API调试日志

try:  # 尝试导入Flash Attention
    from flash_attn.flash_attn_interface import flash_attn_varlen_func  # 导入变长Flash Attention
except ImportError:  # 导入失败
    flash_attn_varlen_func = None  # 设为None

from sglang.srt.configs import MoonViTConfig  # 导入MoonViT配置
from sglang.srt.layers.conv import Conv2dLayer  # 导入2D卷积层
from sglang.srt.layers.linear import ReplicatedLinear  # 导入复制线性层
from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig  # 导入ModelSlim量化
from sglang.srt.utils import add_prefix, get_device  # 导入工具函数


@debug_kernel_api  # 内核API调试装饰器
def multihead_attention(
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    v: torch.Tensor,  # 值张量
    q_cu_seqlens: Optional[torch.Tensor] = None,  # Q的累计序列长度
    k_cu_seqlens: Optional[torch.Tensor] = None,  # K的累计序列长度
):
    """使用Flash Attention 2的多头注意力函数。"""
    """Multi-head attention using flash attention 2.  # 使用Flash Attention 2的多头注意力
    This function is used to handle the case where the query, key, and value are packed.  # 用于处理打包的QKV
    Args:  # 参数
        q, k, v: tensor of shape (tot_seqlens, num_heads, head_dim).  # 张量形状
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.  # Q的累计序列长度
            The first element should be 0 and the last element should be q.shape[0].  # 第一个为0，最后一个为q长度
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.  # K的累计序列长度
            The first element should be 0 and the last element should be k.shape[0].  # 第一个为0，最后一个为k长度

    Returns:  # 返回
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,  # 输出形状
            where dim = num_heads * head_dim  # dim = 头数 * 头维度
    """
    if flash_attn_varlen_func is None:  # 如果Flash Attention不可用
        raise ImportError(  # 抛出导入错误
            "flash_attn is not installed, this function needs flash_attn_varlen_func from flash_attn"
        )
    # Unified format legal check  # 统一格式合法性检查
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"  # QKV必须3维
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"  # Q序列长度检查
    assert (  # KV序列长度检查
        k_cu_seqlens[-1] == k.shape[0] == v.shape[0]
    ), "k_cu_seqlens must sum to k.shape[0]"
    assert q.dtype in [  # 检查数据类型
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()  # Q最大序列长度
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()  # K最大序列长度
    attn_out = flash_attn_varlen_func(  # 调用Flash Attention变长函数
        q,  # 查询
        k,  # 键
        v,  # 值
        q_cu_seqlens,  # Q累计序列长度
        k_cu_seqlens,  # K累计序列长度
        max_seqlen_q,  # Q最大序列长度
        max_seqlen_k,  # K最大序列长度
        causal=False,  # 非因果注意力
    )
    attn_out = attn_out.flatten(start_dim=-2)  # 展平最后两个维度

    return attn_out  # 返回注意力输出


def sdpa_attention(
    q: torch.Tensor,  # 查询张量
    k: torch.Tensor,  # 键张量
    v: torch.Tensor,  # 值张量
    q_cu_seqlens: Optional[torch.Tensor] = None,  # Q累计序列长度
    k_cu_seqlens: Optional[torch.Tensor] = None,  # K累计序列长度
) -> torch.Tensor:
    """使用PyTorch缩放点积注意力的多头注意力函数。"""
    """Multi-head attention using torch scaled dot product attention.  # 使用PyTorch SDPA的多头注意力
    This function is used to handle the case where the query, key, and value are packed.  # 用于处理打包的QKV
    Args:  # 参数
        q, k, v: tensor of shape (tot_seqlens, num_heads, head_dim).  # 张量形状
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.  # Q累计序列长度
            The first element should be 0 and the last element should be q.shape[0].  # 第一个为0，最后一个为q长度
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.  # K累计序列长度
            The first element should be 0 and the last element should be k.shape[0].  # 第一个为0，最后一个为k长度

    Returns:  # 返回
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,  # 输出形状
            where dim = num_heads * head_dim  # dim = 头数 * 头维度
    """
    # Unified format legal check  # 统一格式合法性检查
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"  # QKV必须3维
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"  # Q序列长度检查
    seq_length = q.shape[0]  # 序列总长度
    attention_mask = torch.zeros(  # 创建注意力掩码
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool  # 形状和设备
    )
    for i in range(1, len(q_cu_seqlens)):  # 遍历每个样本
        attention_mask[  # 设置注意力掩码
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],  # 行范围
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],  # 列范围
        ] = True  # 允许自注意力
    q = q.transpose(0, 1)  # 转置Q：(L, H, D) -> (H, L, D)
    k = k.transpose(0, 1)  # 转置K
    v = v.transpose(0, 1)  # 转置V
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)  # SDPA计算
    attn_output = attn_output.transpose(0, 1)  # 转回：(H, L, D) -> (L, H, D)
    attn_output = attn_output.reshape(seq_length, -1)  # 重塑：(L, H*D)
    return attn_output  # 返回注意力输出


VL_VISION_ATTENTION_FUNCTIONS = {  # 视觉注意力函数映射
    "flash_attention_2": multihead_attention,  # Flash Attention 2
    "sdpa": sdpa_attention,  # SDPA
}


def _apply_rope_input_validation(x, freqs_cis):
    """验证RoPE输入的形状和类型。"""
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)  # 维度检查
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)  # 前导维度检查
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)  # 最后一维检查
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype  # 复数类型检查


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """对查询和键张量应用2D旋转位置编码（RoPE）。"""
    """
    Args: (The leading dimensions of all inputs should be the same)  # 所有输入的前导维度应相同
        xq: query, tensor of shape (..., num_heads, head_dim)  # 查询张量
        xk: key, tensor of shape (..., num_heads, head_dim)  # 键张量
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.  # 预计算的复数旋转频率
    Returns:  # 返回
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)  # 应用RoPE后的Q和K
    """
    _apply_rope_input_validation(xq, freqs_cis)  # 验证Q输入
    _apply_rope_input_validation(xk, freqs_cis)  # 验证K输入

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2 扩展维度以匹配多头
    # ..., num_heads, head_dim/2  # 将Q转换为复数视图
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))  # 将Q转为复数
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))  # 将K转为复数
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim 复数乘法后转回实数
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim 复数乘法后转回实数
    return xq_out.type_as(xq), xk_out.type_as(xk)  # 转回原始数据类型并返回


class Learnable2DInterpPosEmb(nn.Module):
    """可学习的2D插值位置编码，支持多分辨率。"""

    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"  # 高度、宽度、维度、插值模式
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.height = height  # 保存高度
        self.width = width  # 保存宽度
        self.interpolation_mode = interpolation_mode  # 保存插值模式
        self.weight = nn.Parameter(torch.empty(height, width, dim))  # 可学习的2D位置编码
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        """重置参数：使用正态分布初始化位置编码。"""
        nn.init.normal_(self.weight)  # 正态分布初始化

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        """前向传播：根据网格尺寸计算位置编码并加到输入上。"""
        pos_embs = []  # 位置编码列表
        for shape in grid_hws.tolist():  # 遍历每个样本的网格尺寸
            if shape == self.weight.shape[:-1]:  # 如果尺寸匹配
                pos_embs.append(self.weight.flatten(end_dim=1))  # 直接展平
            else:  # 否则需要插值
                pos_embs.append(
                    F.interpolate(  # 双三次插值
                        self.weight.permute((2, 0, 1)).unsqueeze(0),  # 调整维度
                        size=shape,  # 目标形状
                        mode=self.interpolation_mode,  # 插值模式
                    )
                    .squeeze(0)  # 去除batch维度
                    .permute((1, 2, 0))  # 恢复维度
                    .flatten(end_dim=1)  # 展平
                )
        out = x + torch.cat(pos_embs)  # 将位置编码加到输入上
        return out  # 返回结果


class MoonVisionPatchEmbed(nn.Module):
    """MoonVision Patch嵌入模块，将图像分割为patch并添加位置编码。"""

    def __init__(
        self,
        out_dim: int,  # 输出维度
        in_dim: int = 3,  # 输入通道数
        patch_size: Union[int, Tuple[int, int]] = (14, 14),  # patch大小
        pos_emb_height: int = 14,  # 位置编码高度
        pos_emb_width: int = 14,  # 位置编码宽度
    ):
        super().__init__()  # 调用父类初始化
        assert isinstance(  # 验证patch_size类型
            patch_size, (int, Sequence)
        ), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):  # 如果是整数
            patch_size = (patch_size, patch_size)  # 转为元组
        assert (  # 验证patch_size长度
            len(patch_size) == 2
        ), f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size  # 保存patch大小

        self.proj = Conv2dLayer(  # 2D卷积投影
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size  # 卷积参数
        )

        self.pos_emb = Learnable2DInterpPosEmb(  # 可学习2D位置编码
            height=pos_emb_height, width=pos_emb_width, dim=out_dim  # 高度、宽度、维度
        )

    def forward(self, x: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        """前向传播：将输入投影为patch并添加位置编码。"""
        """
        Args:  # 参数
            x (L, Channels): input tensor  # 输入张量
            grid_hw (N, 2): grid height and width  # 网格高度和宽度

        Returns:  # 返回
            (L, Cout) tensor  # 输出张量
        """
        x = self.proj(x).view(x.size(0), -1)  # 通过卷积投影并展平
        # apply positional embedding  # 应用位置编码
        x = self.pos_emb(x, grid_hw)  # 添加位置编码
        return x  # 返回带位置编码的patch


class Rope2DPosEmb(nn.Module):
    """2D旋转位置编码，支持多分辨率。"""
    """2D rotary position embedding with multi-resolution support.  # 2D旋转位置编码，多分辨率支持

    This class is intended to be used in the following way:  # 使用方式
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.  # 训练前创建实例
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.  # 前向传播前获取频率
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.  # 传递给注意力层
        The rope is shared across all attention layers and all heads.  # RoPE在所有层和头之间共享

    Refs:  # 参考
    - RoFormer: https://arxiv.org/abs/2104.09864  # RoFormer论文
    - VisionLLaMA: https://arxiv.org/abs/2403.00522  # VisionLLaMA论文
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py  # 实现代码

    Args:  # 参数
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)  # 注意力维度
        max_height (int): the maximum height of the 2D grid  # 最大高度
        max_width (int): the maximum width of the 2D grid  # 最大宽度
        theta_base (float): the base of the theta  # theta基数
        device (str): the device to store the precomputed cis  # 存储设备
    """

    def __init__(
        self, dim: int, max_height: int, max_width: int, theta_base=10000, device=None  # 维度、最大高宽、theta基数、设备
    ):
        super().__init__()  # 调用父类初始化
        self.dim = dim  # 保存维度
        assert self.dim % 4 == 0, "dim must be divisible by 4"  # 维度必须被4整除
        self.max_height = max_height  # 保存最大高度
        self.max_width = max_width  # 保存最大宽度
        self.theta_base = theta_base  # 保存theta基数
        self.device = device if device is not None else get_device()  # 保存设备

    def extra_repr(self):
        """返回模块的额外表示字符串。"""
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"  # 格式化输出

    @cached_property  # 缓存属性
    def precomputed_freqs_cis(self) -> torch.Tensor:
        """预计算2D网格中每个位置的复数旋转频率。"""
        """Calculate the cis(freqs) for each position in the 2D grid.  # 计算2D网格中的复数频率

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:  # 返回复数张量
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))  # 高度轴编码
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))  # 宽度轴编码
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,  # cis数学记号
        """
        N = self.max_height * self.max_width  # 总位置数
        flat_pos = torch.arange(0, N).float().to(self.device)  # 展平位置索引
        x_pos = flat_pos % self.max_width  # x坐标
        y_pos = flat_pos // self.max_width  # y坐标
        dim_range = (  # 维度范围
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(self.device)
        )  # C/4 每4个维度取一个频率
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))  # 计算频率
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4 x方向频率
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4 y方向频率
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4 x方向复数旋转
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4 y方向复数旋转
        # N, C/4, 2  # 拼接x和y
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2  # 重排为2D网格
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis  # 返回预计算结果

    def get_freqs_cis_by_seqlens(self, grid_hws: torch.Tensor) -> torch.Tensor:
        """根据网格高度宽度获取对应的复数旋转频率。"""
        """
        Args:  # 参数
            grid_hws (torch.Tensor): containing list of (height, width) or (t, height, width) tuples.  # 网格高宽
        Returns:  # 返回
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)  # 复数频率
        """
        shapes = grid_hws.tolist()  # 获取网格形状
        assert all(  # 验证形状在合法范围
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,  # 当前形状
            self.max_height,  # 最大高度
            self.max_width,  # 最大宽度
        )
        freqs_cis = torch.cat(  # 拼接所有样本的频率
            [
                self.precomputed_freqs_cis[:h, :w].reshape(-1, self.dim // 2)  # 截取对应区域
                for h, w in shapes  # 遍历每个形状
            ],
            dim=0,  # 在序列维度拼接
        )
        return freqs_cis  # 返回频率

    def get_freqs_cis_by_idx(
        self, pos_idx: torch.Tensor, pos_idx_mask: torch.Tensor
    ) -> torch.Tensor:
        """根据位置索引获取对应的复数旋转频率（支持掩码）。"""
        """
        Args:  # 参数
            pos_idx: tensor of shape (..., 2), It contains the (h, w) position indices of each 2D token.  # 位置索引
            pos_idx_mask: a mask of shape (...), the leading dimensions should be the same as pos_idx.  # 位置掩码
                Rope will only be applied to the tokens with True mask. `freqs_cis` for the tokens with False mask with be ones.  # 仅对True掩码的token应用RoPE
        Return:  # 返回
            freqs_cis: tensor of shape (..., dim//2)  # 复数频率
        """
        assert (  # 验证形状
            pos_idx.shape[:-1] == pos_idx_mask.shape
            and pos_idx.shape[-1] == 2
            and pos_idx.ndim == pos_idx_mask.ndim + 1
        ), (pos_idx.shape, pos_idx_mask.shape)
        assert pos_idx_mask.dtype == torch.bool, pos_idx_mask.dtype  # 验证掩码类型

        shp = pos_idx_mask.shape + (self.dim // 2,)  # ..., head_dim/2 输出形状
        freqs_cis = torch.ones(  # 初始化为1（不应用RoPE的位置）
            shp, dtype=torch.complex64, device=self.device
        )  # ..., head_dim/2
        freqs_cis[pos_idx_mask] = self.precomputed_freqs_cis[  # 对掩码位置应用RoPE
            pos_idx[..., 0][pos_idx_mask], pos_idx[..., 1][pos_idx_mask]  # 使用位置索引
        ]
        return freqs_cis  # 返回频率


class MLP2(nn.Module):
    """两层MLP模块，支持量化配置。"""
    """
    Args:  # 参数
        dims: [in_dim, hidden_dim, out_dim]  # 维度列表
        bias: whether to use bias in linear layer.  # 是否使用偏置
    """

    def __init__(
        self,
        dims: list[int],  # 维度列表
        activation,  # 激活函数
        bias: bool = True,  # 是否使用偏置
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        assert len(dims) == 3  # 必须是3个维度

        self.quant_config = quant_config  # 保存量化配置
        if isinstance(self.quant_config, ModelSlimConfig):  # 如果使用ModelSlim量化
            self.fc0 = ReplicatedLinear(  # 第一层（量化版）
                dims[0],  # 输入维度
                dims[1],  # 隐藏维度
                bias=bias,  # 偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("fc0", prefix),  # 参数前缀
            )
            self.fc1 = ReplicatedLinear(  # 第二层（量化版）
                dims[1],  # 隐藏维度
                dims[2],  # 输出维度
                bias=bias,  # 偏置
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix("fc1", prefix),  # 参数前缀
            )
        else:  # 非量化版
            self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)  # 第一层
            self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)  # 第二层
            for m in [self.fc0, self.fc1]:  # 初始化权重
                nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))  # 截断正态初始化
                if m.bias is not None:  # 如果有偏置
                    nn.init.zeros_(m.bias)  # 零初始化偏置
        self.activation = activation  # 保存激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：两层MLP计算。"""
        if isinstance(self.quant_config, ModelSlimConfig):  # 量化版
            x = x.flatten(0, 1)  # 展平
            x, _ = self.fc0(x)  # 第一层
            x = self.activation(x)  # 激活
            x, _ = self.fc1(x)  # 第二层
        else:  # 非量化版
            x = self.fc0(x)  # 第一层
            x = self.activation(x)  # 激活
            x = self.fc1(x)  # 第二层
        return x  # 返回结果


class MoonVitEncoderLayer(nn.Module):
    """MoonViT编码器层，包含自注意力和MLP。"""

    def __init__(
        self,
        num_heads: int,  # 注意力头数
        hidden_dim: int,  # 隐藏维度
        mlp_dim: int,  # MLP中间维度
        *,  # 以下为关键字参数
        attn_implementation: str = "flash_attention_2",  # use fa2 in sglang by default  # 默认使用FA2
        activation=F.gelu,  # 激活函数
        attn_bias: bool = False,  # 注意力偏置
    ):
        super().__init__()  # 调用父类初始化
        self.num_heads = num_heads  # 保存头数
        self.hidden_dim = hidden_dim  # 保存隐藏维度
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads  # 每头维度
        self.attn_implementation = attn_implementation  # 保存注意力实现方式

        self.norm0 = nn.LayerNorm(hidden_dim)  # 注意力前层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)  # MLP前层归一化
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)  # MLP模块
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)  # QKV投影
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)  # 输出投影

    def attention_qkvpacked(
        self,
        x: torch.Tensor,  # 输入张量
        cu_seqlens: torch.Tensor,  # 累计序列长度
        rope_freqs_cis: Optional[torch.Tensor] = None,  # RoPE复数频率
    ):
        """打包QKV的注意力计算。"""
        """
        Args:  # 参数
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)  # 输入张量
            cu_seqlens (torch.Tensor):  # 累计序列长度
        """
        xqkv = self.wqkv(x)  # QKV投影

        qkv_shape = xqkv.size()[:-1] + (  # QKV形状
            3,  # QKV三部分
            self.num_heads,  # 头数
            self.hidden_size_per_attention_head,  # 每头维度
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)  # QKV张量
        xqkv = xqkv.view(*qkv_shape)  # 重排形状
        xq, xk, xv = torch.unbind(xqkv, dim=-3)  # 拆分QKV

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)  # 应用2D RoPE

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]  # 获取注意力函数
        attn_out = attn_func(  # 执行注意力计算
            xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens  # 传入参数
        )

        attn_out = self.wo(attn_out)  # 输出投影
        return attn_out  # 返回注意力输出

    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cu_seqlens: torch.Tensor,  # 累计序列长度
        rope_freqs_cis: Union[torch.Tensor, None] = None,  # RoPE复数频率
    ) -> torch.Tensor:
        """前向传播：执行注意力+MLP的编码器层计算。"""
        """
        Args:  # 参数
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set  # 隐藏状态

        Returns:  # 返回
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input  # 输出
        """
        residual = hidden_states  # 保存残差
        hidden_states = self.norm0(hidden_states)  # 注意力前归一化
        attn_out = self.attention_qkvpacked(  # 执行注意力
            hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis  # 传入参数
        )
        hidden_states = residual + attn_out  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.mlp(self.norm1(hidden_states))  # MLP前归一化后通过MLP
        hidden_states = residual + hidden_states  # 残差连接
        return hidden_states  # 返回隐藏状态


class MoonVitEncoder(nn.Module):
    """MoonViT编码器，包含多个编码器层。"""

    def __init__(
        self,
        hidden_dim: int,  # 隐藏维度
        num_layers: int,  # 层数
        block_cfg: dict,  # 块配置
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.rope_2d = Rope2DPosEmb(  # 创建2D RoPE
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512  # 维度和最大分辨率
        )
        self.blocks = nn.ModuleList(  # 创建编码器层列表
            [MoonVitEncoderLayer(**block_cfg) for _ in range(num_layers)]  # 每层使用相同配置
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)  # 最终层归一化

    def forward(
        self, hidden_states: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        """前向传播：通过所有编码器层处理隐藏状态。"""
        rope_freqs_cis = self.rope_2d.get_freqs_cis_by_seqlens(grid_hws=grid_hw)  # 获取2D RoPE频率

        lengths = torch.cat(  # 计算每个样本的序列长度
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_hw.dtype),  # 起始0
                grid_hw[:, 0] * grid_hw[:, 1],  # h*w
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)  # 累计序列长度

        for _, block in enumerate(self.blocks):  # 遍历所有编码器层
            hidden_states = block(  # 通过编码器层
                hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis  # 传入参数
            )

        hidden_states = self.final_layernorm(hidden_states)  # 最终层归一化

        return hidden_states  # 返回编码后的隐藏状态


def patch_merger(
    x: torch.Tensor,  # 输入张量
    grid_hw: torch.Tensor,  # 网格高度宽度
    merge_kernel_size: list[int, int] = (2, 2),  # 合并核大小
) -> List[torch.Tensor]:
    """Patch合并器，沿空间维度将patch按合并核大小合并。"""
    d_model = x.size(-1)  # 获取模型维度

    outputs = []  # 输出列表
    pre_sum = 0  # 前序累计和
    for x_shape in grid_hw.tolist():  # 遍历每个样本的网格尺寸
        height, width = x_shape[0], x_shape[1]  # 获取高度和宽度
        # Get the current sequence  # 获取当前序列
        seq = x[pre_sum : pre_sum + height * width]  # 截取当前样本
        # Reshape along self.merge_kernel_size and concat to the last dimension  # 按合并核大小重排
        kernel_height, kernel_width = merge_kernel_size  # 获取合并核大小
        new_height, new_width = height // kernel_height, width // kernel_width  # 合并后的高宽
        reshaped_seq = seq.view(  # 重排
            new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = reshaped_seq.permute(0, 2, 1, 3, 4).contiguous()  # 重排维度
        padded_seq = reshaped_seq.view(  # 展平合并后的patch
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)  # 添加到输出列表
        pre_sum += height * width  # 更新累计和

    return outputs  # 返回合并后的patch列表


class MoonVitVLProjector(nn.Module):
    """MoonViT VL投影器，将视觉特征投影到语言模型空间。"""

    def __init__(
        self,
        in_channels: int,  # 输入通道数
        merge_kernel_size: list[int, int],  # 合并核大小
        hidden_act: str = "gelu",  # 隐藏层激活
        ln_eps: float = 1e-5,  # 层归一化epsilon
        out_dim: int = 4096,  # 输出维度
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = in_channels * merge_kernel_size[0] * merge_kernel_size[1]  # 隐藏大小

        self.pre_norm = nn.nn.LayerNorm(in_channels, eps=ln_eps)  # 预归一化（注意：nn.nn是原始代码）
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # 第一层线性
        self.act = ACT2FN[hidden_act]  # 激活函数
        self.linear_2 = nn.Linear(self.hidden_size, out_dim, bias=True)  # 第二层线性

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播：将视觉特征投影到语言模型空间。"""
        hidden_states = self.pre_norm(hidden_states).view(-1, self.hidden_size)  # 预归一化并重塑
        hidden_states = self.linear_1(hidden_states)  # 第一层线性
        hidden_states = self.act(hidden_states)  # 激活
        hidden_states = self.linear_2(hidden_states)  # 第二层线性
        return hidden_states  # 返回投影结果


class MoonVitPretrainedModel(PreTrainedModel):
    """MoonViT预训练模型，包含Patch嵌入和Transformer编码器。"""
    config_class = MoonViTConfig  # 配置类
    model_type = "moonvit"  # 模型类型
    _no_split_modules = ["PackingTransformer"]  # 不可分割模块
    _supports_flash_attn_2 = True  # 支持Flash Attention 2
    _supports_sdpa = True  # 支持SDPA

    def __init__(self, config: MoonViTConfig, *inputs, **kwargs):  # 初始化
        from transformers.activations import GELUTanh  # 导入GELU Tanh激活

        super().__init__(config, *inputs, **kwargs)  # 调用父类初始化
        config = deepcopy(config)  # 深拷贝配置
        self.merge_kernel_size = config.merge_kernel_size  # 合并核大小
        self.patch_size = config.patch_size  # patch大小
        self.patch_embed = MoonVisionPatchEmbed(  # 创建Patch嵌入
            out_dim=config.hidden_size,  # 输出维度
            patch_size=config.patch_size,  # patch大小
            pos_emb_height=config.init_pos_emb_height,  # 位置编码高度
            pos_emb_width=config.init_pos_emb_width,  # 位置编码宽度
        )

        self.encoder = MoonVitEncoder(  # 创建编码器
            hidden_dim=config.hidden_size,  # 隐藏维度
            num_layers=config.num_hidden_layers,  # 层数
            block_cfg={  # 块配置
                "num_heads": config.num_attention_heads,  # 头数
                "hidden_dim": config.hidden_size,  # 隐藏维度
                "mlp_dim": config.intermediate_size,  # MLP中间维度
                "activation": GELUTanh(),  # 激活函数
                "attn_bias": True,  # 使用注意力偏置
                "attn_implementation": config._attn_implementation,  # 注意力实现
            },
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        """前向传播：处理图像像素值，返回视觉token。"""
        """
        Args:  # 参数
            pixel_values (torch.Tensor): The input pixel values.  # 输入像素值
            grid_hw (torch.Tensor): The grid height and width.  # 网格高度和宽度

        Returns:  # 返回
            torch.Tensor: The output tokens.  # 输出token
        """
        hidden_states = self.patch_embed(pixel_values, grid_hw)  # Patch嵌入
        hidden_states = self.encoder(hidden_states, grid_hw)  # 编码器
        hidden_states = patch_merger(  # Patch合并
            hidden_states, grid_hw, merge_kernel_size=self.merge_kernel_size  # 传入参数
        )
        return hidden_states  # 返回合并后的视觉token
