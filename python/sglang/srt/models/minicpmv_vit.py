# MiniCPM-V 4.6视觉Transformer实现
# 该模块实现了MiniCPM-V 4.6的视觉编码器，包含ViT窗口注意力合并器和纯MLP合并器
# 相比4.5版本，4.6版本在ViT中间层插入2x2窗口注意力+折叠进行压缩，并在编码器后使用MLP链替换感知器重采样器
# 核心组件：MiniCPMV_ViTWindowAttentionMerger、MiniCPMV_DownsampleMLP、MiniCPMV_Merger、MiniCPMV_VisionTransformer

# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Vision Transformer for MiniCPM-V 4.6.

Compared to 4.5 (Idefics2VisionTransformer end-to-end + Perceiver-style
Resampler4_5), 4.6 compresses visual tokens *twice*:

    patchify -> [layer 0 .. insert_layer_id]     full-res tokens
             -> ViTWindowAttentionMerger         2x2 window attn + 2x2 fold
             -> [layer insert_layer_id+1 .. N-1] compressed tokens
             -> post_layernorm
             -> Merger (merger_times x DownsampleMLP, project to LLM dim)

With defaults (insert_layer_id=6, merger_times=1) the combined compression
is 16x. ``downsample_mode="4x"`` skips the mid-ViT merger.

Class structure mirrors the HF ref one-to-one to make weight loading and
upstream tracking easy.
"""

from typing import List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch函数式接口
from torch import nn  # 导入PyTorch神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取器
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear  # 导入并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.models.idefics2 import (  # 从idefics2导入视觉编码器组件
    Idefics2Encoder,  # Idefics2编码器
    Idefics2EncoderLayer,  # Idefics2编码器层
    Idefics2VisionEmbeddings,  # Idefics2视觉嵌入
)
from sglang.srt.utils import add_prefix, is_npu  # 导入工具函数


class MiniCPMV_ViTWindowAttentionMerger(nn.Module):
    """ViT中间层的2x2窗口注意力+2x2折叠合并器"""
    """Mid-ViT 2x2 window attention + 2x2 fold.

    Stage 1: reorder tokens so each 2x2 spatial window becomes 4 contiguous
    tokens; run packed self-attention with one window per cu_seqlens segment;
    un-reorder; add residual. (No length reduction yet.)

    Stage 2: fold each 2x2 window into a single token by concatenating the
    four hidden vectors along channel; pass through ``hidden*4 ->
    intermediate*4 -> hidden`` MLP; add the mean of the four window vectors
    as residual. ``target_sizes`` halves on each axis; ``cu_seqlens`` /
    ``max_seqlens`` are rebuilt for the compressed grid.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.window_kernel_size = (2, 2)  # 窗口核大小
        self.embed_dim = config.hidden_size  # 嵌入维度

        # The "FFN" here is the linear_1/linear_2 pair applied after the 2x2
        # fold below (it operates on hidden*4 -> intermediate*4 -> hidden).
        # ``flatten_batch=True``: input is one packed sequence
        # ``(1, sum_windows * window_area, D)`` with cu_seqlens demarcating
        # per-window segments. The outer encoder layers use ``False`` because
        # there each batch row is one image padded to max_patches.
        self.self_attn = VisionAttention(  # 窗口内自注意力
            embed_dim=config.hidden_size,  # 嵌入维度
            num_heads=config.num_attention_heads,  # 头数
            projection_size=config.hidden_size,  # 投影尺寸
            use_qkv_parallel=True,  # 使用QKV并行
            quant_config=quant_config,  # 量化配置
            dropout=config.attention_dropout,  # dropout率
            softmax_in_single_precision=True,  # 单精度softmax
            flatten_batch=True,  # 展平批次
            prefix=add_prefix("self_attn", prefix),  # 前缀
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 层归一化1

        window_area = self.window_kernel_size[0] * self.window_kernel_size[1]  # 窗口面积=4
        hidden_4x = self.embed_dim * window_area  # 4倍隐藏维度
        inter_4x = config.intermediate_size * window_area  # 4倍中间维度

        self.pre_norm = nn.LayerNorm(hidden_4x, eps=config.layer_norm_eps)  # 折叠后归一化
        self.linear_1 = ColumnParallelLinear(  # 第一个线性层
            hidden_4x,  # 输入：4倍隐藏维度
            inter_4x,  # 输出：4倍中间维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear_1", prefix),  # 前缀
        )
        self.act = get_act_fn("gelu_pytorch_tanh")  # GELU激活函数
        self.linear_2 = RowParallelLinear(  # 第二个线性层
            inter_4x,  # 输入：4倍中间维度
            self.embed_dim,  # 输出：嵌入维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear_2", prefix),  # 前缀
        )

    def get_window_index(
        self, target_sizes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """获取窗口索引、累计序列长度和最大序列长度"""
        """Return ``(permutation, per-window cu_seqlens, max_seqlens=4)``.

        Kept on CPU because mixing device-bound offsets with CPU arange trips
        strict dtype checks in PyTorch 2.10+.
        """
        window_h, window_w = self.window_kernel_size  # 窗口大小
        max_seqlens = window_h * window_w  # 4

        window_index_list: List[torch.Tensor] = []  # 窗口索引列表
        cu_seqlens: List[int] = [0]  # 累计序列长度
        token_offset = 0  # token偏移

        for height, width in target_sizes:  # 遍历每个目标尺寸
            height, width = int(height), int(width)  # 转换为整数
            if height % window_h != 0 or width % window_w != 0:  # 检查可整除性
                raise ValueError(
                    f"height={height}, width={width} must be divisible by "
                    f"window size ({window_h}, {window_w})"
                )
            index = torch.arange(height * width).reshape(height, width)  # 生成空间索引
            num_windows_h = height // window_h  # 高度方向窗口数
            num_windows_w = width // window_w  # 宽度方向窗口数
            num_windows = num_windows_h * num_windows_w  # 总窗口数

            index = index.reshape(num_windows_h, window_h, num_windows_w, window_w)  # 重塑为窗口形状
            index = index.permute(0, 2, 1, 3).reshape(num_windows, window_h * window_w)  # 重排为窗口索引

            window_index_list.append(index.reshape(-1) + token_offset)  # 添加偏移后的索引

            cu_this = (  # 计算当前窗口的累计长度
                torch.arange(1, num_windows + 1) * (window_h * window_w)
                + cu_seqlens[-1]
            )
            cu_seqlens.extend(cu_this.tolist())  # 添加到累计序列长度列表

            token_offset += height * width  # 更新token偏移

        window_index = torch.cat(window_index_list)  # 拼接所有窗口索引
        cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32)  # 转换为张量
        return window_index, cu_seqlens_t, max_seqlens  # 返回索引、累计长度和最大长度

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sizes: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """合并器前向传播：窗口注意力+2x2折叠"""
        device = hidden_states.device  # 获取设备

        # Stage 1: 2x2 window self-attention + residual.
        residual = hidden_states  # 保存残差
        hidden_states = self.layer_norm1(hidden_states)  # 层归一化

        window_index, window_cu_seqlens, _ = self.get_window_index(target_sizes)  # 获取窗口索引
        window_index = window_index.to(device)  # 转换设备
        window_cu_seqlens = window_cu_seqlens.to(device)  # 转换设备
        if is_npu():  # 如果是NPU设备
            window_cu_seqlens = window_cu_seqlens.to("cpu")  # NPU需要放在CPU上

        hidden_states = hidden_states[:, window_index, :]  # 按窗口索引重排
        hidden_states = self.self_attn(hidden_states, cu_seqlens=window_cu_seqlens)  # 窗口注意力
        hidden_states = hidden_states[:, torch.argsort(window_index), :]  # 恢复原始顺序
        hidden_states = residual + hidden_states  # 残差连接

        # Stage 2: 2x2 spatial fold + MLP + mean residual.
        if (target_sizes % 2 != 0).any():  # 检查目标尺寸可被2整除
            raise ValueError(
                f"All target_sizes must be divisible by 2, got {target_sizes}"
            )
        new_target_sizes = target_sizes // 2  # 新的目标尺寸（减半）

        window_h, window_w = self.window_kernel_size  # 窗口大小
        batch_size = target_sizes.shape[0]  # 批次大小
        all_pixel_values = []  # 处理后的像素值列表
        for batch_idx in range(batch_size):  # 遍历每个批次
            height, width = target_sizes[batch_idx]  # 当前高度和宽度
            patch = hidden_states[  # 获取当前图像的补丁
                0, cu_seqlens[batch_idx] : cu_seqlens[batch_idx + 1], :
            ].squeeze(0)

            embed_dim = patch.shape[-1]  # 嵌入维度
            merged_h, merged_w = height // window_h, width // window_w  # 合并后的高度和宽度
            patch_5d = patch.view(  # 重塑为5D
                merged_h, window_h, merged_w, window_w, embed_dim
            ).permute(0, 2, 1, 3, 4)  # 重排维度
            hidden_state = patch_5d.reshape(  # 折叠：拼接4个窗口向量
                merged_h * merged_w, window_h * window_w * embed_dim
            )
            res = patch_5d.reshape(  # 计算窗口均值作为残差
                merged_h * merged_w, window_h, window_w, embed_dim
            ).mean(dim=1)

            hidden_state = self.pre_norm(hidden_state)  # 折叠后归一化
            hidden_state, _ = self.linear_1(hidden_state)  # 第一个线性层
            hidden_state = self.act(hidden_state)  # 激活函数
            hidden_state, _ = self.linear_2(hidden_state)  # 第二个线性层

            all_pixel_values.append(hidden_state + res)  # 添加MLP输出加残差

        new_hidden_states = torch.concat(all_pixel_values, dim=0).unsqueeze(0)  # 拼接所有批次
        new_cu_seqlens = F.pad(  # 计算新的累计序列长度
            torch.cumsum(
                new_target_sizes[:, 0] * new_target_sizes[:, 1],  # 新的序列长度
                dim=0,
                dtype=torch.int32,
            ).to(device),
            (1, 0),  # 前端补0
        )
        if max_seqlens % 4 != 0:  # 检查最大序列长度可被4整除
            raise ValueError(f"max_seqlens ({max_seqlens}) must be divisible by 4")
        new_max_seqlens = max_seqlens // 4  # 新的最大序列长度

        return new_hidden_states, new_target_sizes, new_cu_seqlens, new_max_seqlens  # 返回合并结果


class MiniCPMV_DownsampleMLP(nn.Module):
    """2x2空间合并+MLP的下采样模块"""
    """One round of 2x2 spatial merge + MLP, used inside ``MiniCPMV_Merger``.

    Input channel dim is ``hidden_size * 4`` (already folded by the caller).
    Output is ``hidden_size`` for an intermediate round or ``llm_embed_dim``
    for the final round.
    """

    def __init__(
        self,
        hidden_size: int,
        llm_embed_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        merged_hidden_size = hidden_size * 4  # 合并后的隐藏维度

        self.pre_norm = nn.LayerNorm(merged_hidden_size, eps=1e-6)  # 归一化
        self.linear_1 = ColumnParallelLinear(  # 第一个线性层
            merged_hidden_size,  # 输入维度
            merged_hidden_size,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear_1", prefix),  # 前缀
        )
        self.act = nn.GELU()  # GELU激活函数
        self.linear_2 = RowParallelLinear(  # 第二个线性层
            merged_hidden_size,  # 输入维度
            llm_embed_dim,  # 输出维度（LLM嵌入维度）
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("linear_2", prefix),  # 前缀
        )
        self.in_features = merged_hidden_size  # 输入特征数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """下采样MLP前向传播"""
        hidden_states = self.pre_norm(hidden_states).view(-1, self.in_features)  # 归一化并重塑
        hidden_states, _ = self.linear_1(hidden_states)  # 第一个线性层
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states, _ = self.linear_2(hidden_states)  # 第二个线性层
        return hidden_states  # 返回输出


class MiniCPMV_Merger(nn.Module):
    """迭代式2x2折叠+MLP链，连接ViT和LLM"""
    """Iterative 2x2 fold + MLP chain between ViT and LLM.

    With ``merger_times == 1`` (the 4.6 release default) it's a single
    DownsampleMLP projecting straight into ``text_config.hidden_size``. Each
    additional round halves the grid and keeps the channel width at
    ``vision_config.hidden_size`` until the last round.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.merge_kernel_size = tuple(config.merge_kernel_size)  # 合并核大小
        self.merger_times = config.merger_times  # 合并次数
        hidden_size = config.vision_config.hidden_size  # 视觉隐藏维度
        llm_embed_dim = config.text_config.hidden_size  # LLM嵌入维度

        self.mlp = nn.ModuleList(  # MLP列表，每次合并一个MLP
            [
                MiniCPMV_DownsampleMLP(
                    hidden_size,  # 输入隐藏维度
                    llm_embed_dim if i == self.merger_times - 1 else hidden_size,  # 最后一轮投影到LLM维度
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"mlp.{i}", prefix),  # 前缀
                )
                for i in range(self.merger_times)  # 遍历合并次数
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """合并器前向传播：迭代2x2折叠+MLP"""
        merge_h, merge_w = self.merge_kernel_size  # 合并核大小

        start = 0  # 起始索引
        processed = []  # 处理结果列表
        for batch_idx in range(len(target_sizes)):  # 遍历每个批次
            height, width = target_sizes[batch_idx]  # 当前高度和宽度
            num_patches = int(height * width)  # 补丁数

            embed_dim = hidden_states.shape[-1]  # 嵌入维度
            merged_h, merged_w = int(height) // merge_h, int(width) // merge_w  # 合并后的尺寸
            hidden_state = (  # 折叠2x2窗口
                hidden_states[0, start : start + num_patches, :]  # 获取当前图像的补丁
                .view(merged_h, merge_h, merged_w, merge_w, embed_dim)  # 重塑
                .permute(0, 2, 1, 3, 4)  # 重排维度
                .reshape(merged_h * merged_w, merge_h * merge_w * embed_dim)  # 折叠
            )
            hidden_state = self.mlp[0](hidden_state)  # 通过第一个MLP

            height, width = int(height), int(width)  # 转换为整数
            for i in range(1, self.merger_times):  # 遍历后续合并轮次
                if height % merge_h != 0 or width % merge_w != 0:  # 检查可整除性
                    raise ValueError(
                        f"Patch grid ({height}, {width}) must be divisible by "
                        f"merge kernel size {self.merge_kernel_size} at round {i}"
                    )
                height //= merge_h  # 高度减半
                width //= merge_w  # 宽度减半

                inner_dim = hidden_state.shape[-1]  # 内部维度
                merged_h, merged_w = height // merge_h, width // merge_w  # 合并后的尺寸
                hidden_state = (  # 折叠2x2窗口
                    hidden_state.view(merged_h, merge_h, merged_w, merge_w, inner_dim)  # 重塑
                    .permute(0, 2, 1, 3, 4)  # 重排维度
                    .reshape(merged_h * merged_w, merge_h * merge_w * inner_dim)  # 折叠
                )
                hidden_state = self.mlp[i](hidden_state)  # 通过MLP

            start += num_patches  # 更新起始索引
            processed.append(hidden_state)  # 添加处理结果

        return torch.cat(processed, dim=0)  # 拼接所有结果


class MiniCPMV_VisionEncoderLayer(Idefics2EncoderLayer):
    """SigLip风格的预归一化编码器层，用于打包的NaViT输入"""
    """SigLip-style pre-norm encoder layer for packed NaViT input.

    Inherits Idefics2's forward and submodule layout (so HF weights map
    verbatim), then rebuilds ``self_attn`` with ``flatten_batch=True`` for
    per-image block-diagonal attention on a single packed sequence
    (Idefics2 uses padded ``(B, max_patches, D)``) and the SigLip-correct
    ``projection_size = hidden_size`` (Idefics2 sets it to ``intermediate_size``).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)  # 调用父类初始化
        self.self_attn = VisionAttention(  # 重建自注意力，使用打包模式
            embed_dim=config.hidden_size,  # 嵌入维度
            num_heads=config.num_attention_heads,  # 头数
            projection_size=config.hidden_size,  # 投影尺寸
            use_qkv_parallel=True,  # 使用QKV并行
            quant_config=quant_config,  # 量化配置
            dropout=config.attention_dropout,  # dropout率
            softmax_in_single_precision=True,  # 单精度softmax
            flatten_batch=True,  # 展平批次
            prefix=add_prefix("self_attn", prefix),  # 前缀
        )


class MiniCPMV_VisionEncoder(Idefics2Encoder):
    """MiniCPMV视觉编码器层堆叠"""
    """Stack of ``MiniCPMV_VisionEncoderLayer``.

    ``vit_merger`` lives one level up on ``MiniCPMV_VisionTransformer`` so the
    HF checkpoint key ``vision_tower.vit_merger.*`` lands at the matching
    sglang path.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)  # 调用父类初始化
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                MiniCPMV_VisionEncoderLayer(  # 使用自定义编码器层
                    config,
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{i}", prefix),  # 前缀
                )
                for i in range(config.num_hidden_layers)  # 遍历层数
            ]
        )


class MiniCPMV_VisionTransformer(nn.Module):
    """MiniCPM-V 4.6视觉Transformer"""
    """Vision Transformer for MiniCPM-V 4.6.

    Reuses sglang's SigLIP-style ``Idefics2VisionEmbeddings`` + encoder layers,
    inserts ``MiniCPMV_ViTWindowAttentionMerger`` after layer ``insert_layer_id``,
    and applies post-encoder LayerNorm. ``forward`` returns
    ``(hidden_states, target_sizes)``; in ``"16x"`` mode ``target_sizes``
    reflects the post-merger grid, which downstream code must use.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        require_post_norm: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        embed_dim = config.hidden_size  # 嵌入维度
        self.config = config  # 保存配置

        if not hasattr(config, "insert_layer_id"):  # 检查insert_layer_id属性
            raise ValueError(
                "MiniCPMV_VisionTransformer requires `config.insert_layer_id`"
            )

        self.insert_layer_id = config.insert_layer_id  # 插入合并器的层ID
        self.embeddings = Idefics2VisionEmbeddings(config)  # 视觉嵌入层
        self.encoder = MiniCPMV_VisionEncoder(  # 视觉编码器
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.post_layernorm = (  # 编码器后归一化
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            if require_post_norm  # 是否需要后归一化
            else nn.Identity()  # 否则使用恒等映射
        )
        self.vit_merger = MiniCPMV_ViTWindowAttentionMerger(  # ViT窗口注意力合并器
            config,
            quant_config=quant_config,
            prefix=add_prefix("vit_merger", prefix),
        )

    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层"""
        return self.embeddings  # 返回视觉嵌入层

    @staticmethod
    def compute_cu_seqlens(target_sizes: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """根据目标尺寸计算累计序列长度和最大序列长度"""
        seqlen = (target_sizes[:, 0] * target_sizes[:, 1]).to(torch.int32)  # 计算每个序列的长度
        cu_seqlens = torch.cat(  # 计算累计序列长度
            [
                torch.tensor([0], device=seqlen.device, dtype=torch.int32),  # 起始为0
                torch.cumsum(seqlen, dim=0, dtype=torch.int32),  # 累计求和
            ],
            dim=0,
        )
        max_seqlens = int(seqlen.max().item())  # 最大序列长度
        return cu_seqlens, max_seqlens  # 返回累计长度和最大长度

    @staticmethod
    def _pad_to_pack(padded: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        """将填充的张量转换为打包格式"""
        """``(B, max_patches, D) -> (1, sum_patches, D)``.

        ``Idefics2VisionEmbeddings`` emits padded shape with valid tokens at
        ``[0, h_b * w_b)`` of each batch row. Strip the padding so the rest
        of the ViT runs in flat NaViT form.
        """
        seqlens = (target_sizes[:, 0] * target_sizes[:, 1]).to(torch.long)  # 计算每个序列长度
        if padded.shape[0] == 1:  # 如果批次大小为1
            return padded[:, : int(seqlens[0].item()), :]  # 直接截取
        parts = [padded[b, : int(seqlens[b].item()), :] for b in range(padded.shape[0])]  # 逐个截取
        return torch.cat(parts, dim=0).unsqueeze(0)  # 拼接并增加维度

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        target_sizes: Optional[torch.IntTensor] = None,
        use_vit_merger: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """视觉Transformer前向传播"""
        if target_sizes is None:  # 如果没有目标尺寸
            raise ValueError("MiniCPMV_VisionTransformer requires `target_sizes`.")  # 抛出异常

        hidden_states = self.embeddings(  # 补丁嵌入
            pixel_values=pixel_values,  # 像素值
            patch_attention_mask=patch_attention_mask,  # 补丁注意力掩码
            tgt_sizes=target_sizes,  # 目标尺寸
        )
        hidden_states = self._pad_to_pack(hidden_states, target_sizes)  # 转换为打包格式
        cu_seqlens, max_seqlens = self.compute_cu_seqlens(target_sizes)  # 计算累计序列长度
        if is_npu():  # 如果是NPU设备
            cu_seqlens = cu_seqlens.to("cpu")  # 放在CPU上

        if use_vit_merger:  # 如果使用ViT合并器
            # Encoder loop lives here (not inside ``MiniCPMV_VisionEncoder``)
            # so we can fire ``vit_merger`` after layer ``insert_layer_id``
            # without coupling the encoder module to it.
            for layer_index, layer in enumerate(self.encoder.layers):  # 遍历编码器层
                hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens)  # 通过编码器层
                if layer_index == self.insert_layer_id:  # 在插入层后执行合并
                    (
                        hidden_states,
                        target_sizes,
                        cu_seqlens,
                        max_seqlens,
                    ) = self.vit_merger(  # ViT合并器
                        hidden_states, target_sizes, cu_seqlens, max_seqlens
                    )
                    if is_npu():  # NPU设备
                        cu_seqlens = cu_seqlens.to("cpu")  # 放在CPU上
        else:  # 不使用ViT合并器
            hidden_states = self.encoder(hidden_states, cu_seqlens=cu_seqlens)  # 直接通过编码器

        hidden_states = self.post_layernorm(hidden_states)  # 编码器后归一化
        return hidden_states, target_sizes  # 返回隐藏状态和目标尺寸
