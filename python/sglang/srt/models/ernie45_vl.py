# 文件说明：Ernie4.5 VL视觉语言多模态模型完整实现，兼容HuggingFace权重
# 包含视觉编码器（ViT）、可变分辨率重采样器、视觉旋转嵌入及多模态因果语言模型

# Copyright 2023-2025 SGLang Team
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
# limitations under the License. # 详见License限制条件
"""Inference-only Ernie45-VL model compatible with HuggingFace weights.""" # 仅推理的Ernie45-VL模型，兼容HuggingFace权重

import logging # 导入日志模块
from functools import lru_cache, partial # 导入LRU缓存和偏函数工具
from typing import Iterable, List, Optional, Tuple, Type # 导入类型提示工具

import numpy as np # 导入NumPy库
import torch # 导入PyTorch库
import torch.nn as nn # 导入神经网络模块
import torch.nn.functional as F # 导入PyTorch函数式API
from einops import rearrange # 导入张量重排工具
from transformers import PretrainedConfig # 导入预训练配置基类

from sglang.srt.layers.activation import QuickGELU # 导入QuickGELU激活函数
from sglang.srt.layers.attention.vision import VisionAttention # 导入视觉注意力层
from sglang.srt.layers.layernorm import RMSNorm # 导入RMS归一化层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear # 导入列并行和行并行线性层
from sglang.srt.layers.logits_processor import LogitsProcessor # 导入logits处理器
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE # 导入融合MoE层
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.rotary_embedding import get_rope # 导入RoPE获取函数
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead # 导入并行语言模型头
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens, # 导入多模态数据填充模式
    general_mm_embed_routine, # 导入通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs # 导入多模态数据项和输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批信息
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.models.ernie45_moe_vl import Ernie4_5_VLMoeModel # 导入Ernie4.5 VL MoE模型主体
from sglang.srt.utils import add_prefix # 导入添加前缀工具函数
from sglang.srt.utils.hf_transformers_utils import get_processor # 导入处理器获取函数

logger = logging.getLogger(__name__) # 获取当前模块日志记录器


# === Vision Encoder === # # === 视觉编码器 === #


class Ernie4_5_VisionMLP(nn.Module): # Ernie4.5视觉MLP模块

    def __init__( # Ernie4.5视觉MLP初始化
        self,
        in_features: int, # 输入特征维度
        hidden_features: int = None, # 隐藏层特征维度
        act_layer: Type[nn.Module] = QuickGELU, # 激活函数层类型
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ):
        super().__init__() # 调用父类初始化
        self.fc1 = ColumnParallelLinear( # 第一个全连接层（升维）
            in_features, # 输入维度
            hidden_features, # 隐藏维度
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("fc1", prefix), # 参数前缀
        )
        self.act = act_layer() # 创建激活函数实例
        self.fc2 = RowParallelLinear( # 第二个全连接层（降维）
            hidden_features, # 隐藏维度
            in_features, # 输出维度
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("fc2", prefix), # 参数前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Ernie4.5视觉MLP前向传播
        x_parallel, _ = self.fc1(x) # 升维
        x_parallel = self.act(x_parallel) # 激活
        x, _ = self.fc2(x_parallel) # 降维
        return x # 返回输出


class Ernie4_5_VisionBlock(nn.Module): # Ernie4.5视觉Transformer块，包含注意力和MLP

    def __init__( # Ernie4.5视觉块初始化
        self,
        dim: int, # 特征维度
        num_heads: int, # 注意力头数
        mlp_ratio: float, # MLP扩展比例
        act_layer: Type[nn.Module] = QuickGELU, # 激活函数层类型
        norm_layer: Type[nn.Module] = None, # 归一化层类型
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        if norm_layer is None: # 未指定归一化层
            norm_layer = partial(nn.LayerNorm, eps=1e-6) # 默认使用LayerNorm
        self.norm1 = norm_layer(dim) # 注意力前归一化
        self.norm2 = norm_layer(dim) # MLP前归一化
        mlp_hidden_dim = int(dim * mlp_ratio) # 计算MLP隐藏维度

        self.attn = VisionAttention( # 视觉注意力层
            embed_dim=dim, # 嵌入维度
            num_heads=num_heads, # 注意力头数
            projection_size=dim, # 投影大小
            use_qkv_parallel=True, # 使用QKV并行
            flatten_batch=True, # 展平批次
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("attn", prefix), # 参数前缀
        )
        self.mlp = Ernie4_5_VisionMLP( # 视觉MLP
            dim, # 输入维度
            mlp_hidden_dim, # 隐藏维度
            act_layer=act_layer, # 激活函数
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("mlp", prefix), # 参数前缀
        )

    def forward( # Ernie4.5视觉块前向传播
        self,
        x: torch.Tensor, # 输入张量
        cu_seqlens: torch.Tensor, # 累积序列长度
        rotary_pos_emb_cos: torch.Tensor, # 旋转位置编码余弦
        rotary_pos_emb_sin: torch.Tensor, # 旋转位置编码正弦
    ) -> torch.Tensor: # 返回输出
        hidden_states = self.norm1(x) # 注意力前归一化
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...") # 重排为批次优先
        attn = self.attn( # 计算注意力
            hidden_states,
            cu_seqlens=cu_seqlens, # 累积序列长度
            rotary_pos_emb_cos=rotary_pos_emb_cos, # 旋转位置编码余弦
            rotary_pos_emb_sin=rotary_pos_emb_sin, # 旋转位置编码正弦
        )
        attn = rearrange(attn, "b s ... -> s b ...") # 重排为序列优先
        x = x + attn # 残差连接
        x = x + self.mlp(self.norm2(x)) # MLP残差连接
        return x # 返回输出


class Ernie4_5_VisionPatchEmbed(nn.Module): # Ernie4.5视觉Patch嵌入层，将图像patch转换为嵌入向量

    def __init__( # Ernie4.5视觉Patch嵌入初始化
        self,
        patch_size: int = 14, # Patch大小
        in_chans: int = 3, # 输入通道数
        embed_dim: int = 1280, # 嵌入维度
    ) -> None:
        super().__init__() # 调用父类初始化
        self.patch_size = patch_size # 保存Patch大小
        self.in_channels = in_chans # 保存输入通道数
        self.embed_dim = embed_dim # 保存嵌入维度

        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim, bias=False) # 线性投影，将展平的patch映射到嵌入空间

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # Ernie4.5视觉Patch嵌入前向传播
        target_dtype = self.proj.weight.dtype # 获取投影权重的数据类型
        hidden_states = hidden_states.to(target_dtype) # 转换为目标数据类型
        hidden_states = self.proj(hidden_states) # 线性投影

        return hidden_states # 返回嵌入向量


class VariableResolutionResamplerModel(nn.Module): # 可变分辨率重采样器模型，将视觉特征重采样为语言模型维度
    def __init__( # 可变分辨率重采样器初始化
        self,
        in_dim, # 输入维度
        out_dim, # 输出维度
        spatial_conv_size, # 空间卷积大小
        temporal_conv_size, # 时间卷积大小
        config, # 模型配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化
        self.in_dim = in_dim # 保存输入维度
        self.out_dim = out_dim # 保存输出维度
        self.config = config # 保存配置
        self.spatial_conv_size = spatial_conv_size # 保存空间卷积大小
        self.temporal_conv_size = temporal_conv_size # 保存时间卷积大小
        self.use_temporal_conv = config.use_temporal_conv # 是否使用时间卷积

        # compress 2d conv(picture) to 1d # 将2D卷积（图片）压缩为1D
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size # 空间维度大小
        # compress 3d conv(video) to 1d # 将3D卷积（视频）压缩为1D
        self.temporal_dim = ( # 时间维度大小
            self.in_dim
            * self.spatial_conv_size
            * self.spatial_conv_size
            * self.temporal_conv_size
        )

        self.spatial_linear1 = ColumnParallelLinear( # 空间线性层1
            self.spatial_dim, # 输入维度
            self.spatial_dim, # 输出维度
            bias=True, # 使用偏置
            gather_output=True, # 收集输出
            quant_config=getattr(config, "quant_config", None), # 量化配置
            prefix=f"{prefix}.spatial_linear1", # 参数前缀
        )

        self.spatial_gelu = nn.GELU() # 空间GELU激活

        self.spatial_linear2 = ColumnParallelLinear( # 空间线性层2
            self.spatial_dim, # 输入维度
            self.spatial_dim, # 输出维度
            bias=True, # 使用偏置
            gather_output=True, # 收集输出
            quant_config=getattr(config, "quant_config", None), # 量化配置
            prefix=f"{prefix}.spatial_linear2", # 参数前缀
        )

        self.spatial_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6) # 空间归一化

        if self.use_temporal_conv: # 使用时间卷积
            self.temporal_linear1 = ColumnParallelLinear( # 时间线性层1
                self.temporal_dim, # 输入维度
                self.spatial_dim, # 输出维度
                bias=True, # 使用偏置
                gather_output=True, # 收集输出
                quant_config=getattr(config, "quant_config", None), # 量化配置
                prefix=f"{prefix}.temporal_linear1", # 参数前缀
            )

            self.temporal_gelu = nn.GELU() # 时间GELU激活

            self.temporal_linear2 = ColumnParallelLinear( # 时间线性层2
                self.spatial_dim, # 输入维度
                self.spatial_dim, # 输出维度
                bias=True, # 使用偏置
                gather_output=True, # 收集输出
                quant_config=getattr(config, "quant_config", None), # 量化配置
                prefix=f"{prefix}.temporal_linear2", # 参数前缀
            )

            self.temporal_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6) # 时间归一化

        self.mlp = ColumnParallelLinear( # 最终MLP投影层
            self.spatial_dim, # 输入维度
            self.out_dim, # 输出维度
            bias=True, # 使用偏置
            gather_output=True, # 收集输出
            quant_config=getattr(config, "quant_config", None), # 量化配置
            prefix=f"{prefix}.mlp", # 参数前缀
        )

        self.after_norm = RMSNorm( # 最终归一化
            hidden_size=out_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def spatial_conv_reshape(self, x, spatial_conv_size): # 空间卷积重塑，将2D patch展平
        S, C = x.shape # 获取序列长度和通道数
        x = x.reshape([-1, C * (spatial_conv_size**2)]) # 展平为1D
        return x

    def forward(self, x, grid_thw): # 可变分辨率重采样器前向传播
        def fwd_spatial(x): # 空间前向处理
            x = self.spatial_conv_reshape(x, self.spatial_conv_size) # 重塑空间卷积

            x, _ = self.spatial_linear1(x) # 空间线性层1
            x = self.spatial_gelu(x) # GELU激活
            x, _ = self.spatial_linear2(x) # 空间线性层2
            x = self.spatial_norm(x) # 空间归一化

            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False): # 时间占位处理，将相邻时间步的token配对
            grid_thw_cpu = grid_thw.cpu().numpy() # 将grid_thw转为CPU NumPy数组
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:] # 拆分时间和空间维度
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2) # 卷积后的空间大小

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2) # 每个图像/视频的token数
            batch_offset = np.empty( # 批次偏移量
                tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype
            )
            batch_offset[0] = 0 # 第一个偏移为0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1] # 累积求和计算偏移

            slice_offsets = [] # 偶数时间步偏移列表
            for temporoal_size, spatial_size, b_offset in zip( # 遍历时间大小、空间大小和批次偏移
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2): # 偶数时间步
                    slice_offsets.append( # 添加偶数时间步的token索引
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = torch.tensor(np.concatenate(slice_offsets, axis=-1)).to( # 转为张量并移到设备
                x.device
            )

            slice_offsets2 = [] # 奇数时间步偏移列表
            for temporoal_size, spatial_size, b_offset in zip( # 遍历时间大小、空间大小和批次偏移
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range( # 奇数时间步
                    1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append( # 添加奇数时间步的token索引
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = torch.tensor(np.concatenate(slice_offsets2, axis=-1)).to( # 转为张量并移到设备
                x.device
            )

            x_timestep_1 = torch.index_select(x, dim=0, index=slice_offsets) # 选择偶数时间步token
            x_timestep_2 = torch.index_select(x, dim=0, index=slice_offsets2) # 选择奇数时间步token
            x = torch.concat([x_timestep_1, x_timestep_2], dim=-1) # 拼接偶数和奇数时间步
            return x

        def fwd_temporal(x): # 时间前向处理
            x, _ = self.temporal_linear1(x) # 时间线性层1
            x = self.temporal_gelu(x) # GELU激活
            x, _ = self.temporal_linear2(x) # 时间线性层2
            x = self.temporal_norm(x) # 时间归一化
            return x

        def fwd_mlp(x): # MLP前向处理
            x, _ = self.mlp(x) # MLP投影
            x = self.after_norm(x) # 最终归一化
            return x

        x = fwd_spatial(x) # 空间处理
        if self.use_temporal_conv: # 使用时间卷积
            x = fwd_placeholder(x, grid_thw) # 时间占位处理
            x = fwd_temporal(x) # 时间处理
        x = fwd_mlp(x) # MLP处理
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: # 加载重采样器权重
        params_dict = dict(self.named_parameters(remove_duplicate=False)) # 获取参数字典
        loaded_params: set[str] = set() # 已加载参数集合

        for name, loaded_weight in weights: # 遍历权重
            if name not in params_dict: # 参数名不存在则跳过
                continue
            param = params_dict[name] # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
            weight_loader(param, loaded_weight) # 加载权重
            loaded_params.add(name) # 记录已加载参数名
        return loaded_params # 返回已加载参数集合


class Ernie4_5_VisionRotaryEmbedding(nn.Module): # Ernie4.5视觉旋转位置嵌入

    def __init__(self, dim: int, theta: float = 10000.0) -> None: # Ernie4.5视觉旋转嵌入初始化
        super().__init__() # 调用父类初始化
        self.inv_freq = 1.0 / theta ** ( # 计算逆频率
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )

    def forward(self, seqlen: int) -> torch.Tensor: # Ernie4.5视觉旋转嵌入前向传播
        seq = torch.arange( # 生成序列位置
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(input=seq, vec2=self.inv_freq) # 计算频率矩阵
        return freqs # 返回频率矩阵


class Ernie4_5_VisionTransformer(nn.Module): # Ernie4.5视觉Transformer编码器

    def __init__( # Ernie4.5视觉Transformer初始化
        self,
        vision_config: PretrainedConfig, # 视觉配置
        norm_eps: float = 1e-6, # 归一化epsilon
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化

        patch_size: int = vision_config.patch_size # Patch大小
        spatial_merge_size: int = vision_config.spatial_merge_size # 空间合并大小
        in_chans: int = vision_config.in_chans # 输入通道数
        hidden_size: int = vision_config.hidden_size # 隐藏维度
        embed_dim: int = vision_config.embed_dim # 嵌入维度
        depth: int = vision_config.depth # Transformer深度（层数）
        num_heads: int = vision_config.num_heads # 注意力头数
        mlp_ratio: float = vision_config.mlp_ratio # MLP扩展比例

        self.spatial_merge_size = spatial_merge_size # 保存空间合并大小

        self.patch_embed = Ernie4_5_VisionPatchEmbed( # 创建Patch嵌入层
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps) # 创建归一化层构造函数
        head_dim = embed_dim // num_heads # 计算头维度
        self.rotary_pos_emb = get_rope( # 创建旋转位置嵌入
            head_size=head_dim, # 头维度
            rotary_dim=head_dim // 2, # 旋转维度为一半头维度
            max_position=8192, # 最大位置数
            base=10000.0, # 基频
            is_neox_style=True, # Neox风格
        )
        self.blocks = nn.ModuleList( # 创建Transformer块列表
            [
                Ernie4_5_VisionBlock(
                    dim=embed_dim, # 嵌入维度
                    num_heads=num_heads, # 注意力头数
                    mlp_ratio=mlp_ratio, # MLP扩展比例
                    norm_layer=norm_layer, # 归一化层
                    quant_config=quant_config, # 量化配置
                    prefix=add_prefix(f"blocks.{i}", prefix), # 参数前缀
                )
                for i in range(depth) # 遍历所有层
            ]
        )

        self.ln = nn.LayerNorm(hidden_size, eps=1e-6) # 最终层归一化

    @property
    def dtype(self) -> torch.dtype: # 获取模型数据类型
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device: # 获取模型设备
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb( # 计算旋转位置编码
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # 返回cos、sin和位置ID
        pos_ids = [] # 位置ID列表
        for i in range(grid_thw.size(0)): # 遍历每个网格
            t, h, w = grid_thw[i].tolist() # 获取时间、高度、宽度
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w) # 高度位置ID
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1) # 宽度位置ID
            hpos_ids = ( # 重排高度位置ID以适应空间合并
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3) # 置换维度
                .flatten() # 展平
            )
            wpos_ids = ( # 重排宽度位置ID以适应空间合并
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3) # 置换维度
                .flatten() # 展平
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)) # 堆叠并重复时间维度
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True) # 拼接所有位置ID
        max_grid_size = grid_thw[:, 1:].max() # 获取最大网格大小

        # Use pre-computed cos_sin_cache from RotaryEmbedding # 使用预计算的cos_sin缓存
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size) # 获取余弦和正弦值

        cos_combined = cos[pos_ids].flatten(1) # 组合余弦值
        sin_combined = sin[pos_ids].flatten(1) # 组合正弦值
        return cos_combined, sin_combined, pos_ids # 返回余弦、正弦和位置ID

    def forward( # Ernie4.5视觉Transformer前向传播
        self,
        x: torch.Tensor, # 输入像素特征
        grid_thw: torch.Tensor, # 网格时间-高度-宽度信息
    ) -> torch.Tensor: # 返回视觉特征
        # patchify # Patch化处理
        x = x.to(device=self.device, dtype=self.dtype) # 转换设备和数据类型
        x = self.patch_embed(x) # Patch嵌入

        # compute position embedding # 计算位置编码
        rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(
            grid_thw
        )
        rotary_pos_emb_cos = torch.cat([rotary_pos_emb_cos, rotary_pos_emb_cos], dim=-1) # 拼接余弦值以匹配头维度
        rotary_pos_emb_sin = torch.cat([rotary_pos_emb_sin, rotary_pos_emb_sin], dim=-1) # 拼接正弦值以匹配头维度
        # compute cu_seqlens # 计算累积序列长度
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32) # 累积求和
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens]) # 在开头添加0

        # transformers # Transformer块
        x = x.unsqueeze(1) # 增加批次维度
        for blk in self.blocks: # 遍历每个Transformer块
            x = blk(
                x,
                cu_seqlens=cu_seqlens, # 累积序列长度
                rotary_pos_emb_cos=rotary_pos_emb_cos, # 旋转位置编码余弦
                rotary_pos_emb_sin=rotary_pos_emb_sin, # 旋转位置编码正弦
            )

        final_output = self.ln(x) # 最终层归一化

        if final_output.ndim == 3: # 如果是3D输出
            final_output = final_output.squeeze(dim=1) # 去掉批次维度

        return final_output # 返回最终输出


cached_get_processor = lru_cache(get_processor) # 缓存处理器获取函数


class Ernie4_5_VLMoeForConditionalGeneration(nn.Module): # Ernie4.5 VL MoE条件生成模型，完整的视觉语言多模态模型
    # BitandBytes specific attributes # BitandBytes特定属性
    default_bitsandbytes_target_modules = [ # 默认BitandBytes目标模块
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = { # BitandBytes堆叠参数映射
        # shard_name, weight_name, index # 分片名，权重名，索引
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__( # Ernie4.5 VL MoE条件生成模型初始化
        self,
        config: PretrainedConfig, # 预训练配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数前缀
    ) -> None:
        super().__init__() # 调用父类初始化

        self.config = config # 保存配置
        self.vision_model = Ernie4_5_VisionTransformer( # 创建视觉Transformer编码器
            config.vision_config, # 视觉配置
            norm_eps=getattr(config, "rms_norm_eps", 1e-6), # 归一化epsilon
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("vision_model", prefix), # 参数前缀
        )

        self.model = Ernie4_5_VLMoeModel( # 创建语言模型主体（VL MoE模型）
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        self.resampler_model = VariableResolutionResamplerModel( # 创建可变分辨率重采样器
            self.config.pixel_hidden_size, # 像素隐藏维度
            self.config.hidden_size, # 语言模型隐藏维度
            self.config.spatial_conv_size, # 空间卷积大小
            self.config.temporal_conv_size, # 时间卷积大小
            config=self.config, # 模型配置
            prefix=add_prefix("resampler_model", prefix), # 参数前缀
        )

        if config.tie_word_embeddings: # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens # 语言模型头共享嵌入层
        else: # 否则
            self.lm_head = ParallelLMHead( # 创建并行语言模型头
                config.vocab_size, # 词表大小
                config.hidden_size, # 隐藏维度
                quant_config=quant_config, # 量化配置
                prefix=add_prefix("lm_head", prefix), # 参数前缀
            )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling # 判断是否启用多模态RoPE
        self.logits_processor = LogitsProcessor(config) # 创建logits处理器

        if getattr(self.config, "im_patch_id", None): # 如果配置了图像patch ID
            visual_token_ids = [ # 视觉token ID列表
                token_id
                for token_id in [
                    self.config.im_patch_id, # 图像patch token ID
                    getattr(self.config, "image_start_token_id", None), # 图像起始token ID
                    getattr(self.config, "image_end_token_id", None), # 图像结束token ID
                    getattr(self.config, "video_start_token_id", None), # 视频起始token ID
                    getattr(self.config, "video_end_token_id", None), # 视频结束token ID
                ]
                if token_id is not None # 过滤None值
            ]
            self._visual_token_ids_tensor_cache = torch.tensor( # 缓存视觉token ID张量
                visual_token_ids, dtype=torch.long
            )
        else: # 未配置图像patch ID
            self._visual_token_ids_tensor_cache = None # 不缓存

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs): # 填充输入ID，插入多模态token
        pattern = MultiModalityDataPaddingPatternMultimodalTokens() # 创建多模态数据填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs) # 填充输入token

    def _vision_forward( # 视觉前向传播，处理图像/视频特征
        self,
        pixel_values: torch.Tensor, # 像素值
        grid_thw: torch.Tensor, # 网格时间-高度-宽度信息
    ) -> torch.Tensor: # 返回图像特征
        if grid_thw is not None: # 有网格信息
            grid_thw = grid_thw[grid_thw > 0] # 过滤掉0值
            if grid_thw.numel() % 3 != 0: # 过滤后元素数不能被3整除
                raise ValueError(
                    f"grid_thw has {grid_thw.numel()} elements after filtering,"
                    "which is not divisible by 3." # grid_thw过滤后元素数不能被3整除
                )
            grid_thw = grid_thw.reshape(-1, 3) # 重塑为(N, 3)
            # example: [[1,64,64],[2,80,80]] -> [[1,64,64],[1,80,80],[1,80,80]] # 示例：展开时间维度
            grid_thw = F.pad( # 填充并展开时间维度
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0], # 在左侧填充1列
                value=1, # 填充值1
            )
        image_features = self.vision_model(pixel_values, grid_thw) # 通过视觉Transformer获取特征
        return image_features # 返回图像特征

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor: # 获取图像特征
        # in qwen-vl, last dim is the same # 在qwen-vl中，最后一维相同
        pixel_values = torch.cat([item.feature for item in items], dim=0).type( # 拼接所有图像特征
            self.vision_model.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0) # 拼接所有图像网格信息
        assert pixel_values.dim() == 2, pixel_values.dim() # 断言像素值为2D
        assert image_grid_thw.dim() == 2, image_grid_thw.dim() # 断言网格信息为2D
        image_feature = self._vision_forward(pixel_values, grid_thw=image_grid_thw) # 视觉前向传播
        image_embeds = self.resampler_model(image_feature, image_grid_thw) # 重采样器处理
        return image_embeds # 返回图像嵌入

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor: # 获取视频特征
        # in qwen-vl, last dim is the same # 在qwen-vl中，最后一维相同
        pixel_values = torch.cat([item.feature for item in items], dim=0).type( # 拼接所有视频特征
            self.vision_model.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0) # 拼接所有视频网格信息
        assert pixel_values.dim() == 2, pixel_values.dim() # 断言像素值为2D
        assert video_grid_thw.dim() == 2, video_grid_thw.dim() # 断言网格信息为2D
        video_feature = self._vision_forward(pixel_values, grid_thw=video_grid_thw) # 视觉前向传播
        video_embeds = self.resampler_model(video_feature, video_grid_thw) # 重采样器处理
        return video_embeds # 返回视频嵌入

    def _set_visual_token_mask( # 设置视觉token掩码
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        """Set mask for visual tokens (image/video patches and delimiters).""" # 为视觉token（图像/视频patch和分隔符）设置掩码
        if self._visual_token_ids_tensor_cache is None: # 无缓存视觉token ID
            self.visual_token_mask = None # 不设置掩码
            return
        # Create tensor on the correct device # 在正确的设备上创建张量
        visual_token_ids_tensor = self._visual_token_ids_tensor_cache.to( # 将缓存移到目标设备
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        pad_values = [] # 填充值列表
        if hasattr(forward_batch, "mm_inputs") and forward_batch.mm_inputs is not None: # 有多模态输入
            for mm_input in forward_batch.mm_inputs: # 遍历多模态输入
                if mm_input is None: # 跳过None
                    continue
                for item in mm_input.mm_items: # 遍历多模态数据项
                    pad_values.append(item.pad_value) # 收集填充值
        placeholder_tensor = torch.as_tensor( # 创建占位符张量
            pad_values,
            device=input_ids.device,
        )
        pad_visual_token_ids_tensor = torch.cat( # 拼接视觉token ID和占位符
            [visual_token_ids_tensor, placeholder_tensor], dim=0
        )
        self.visual_token_mask = torch.isin( # 创建视觉token掩码
            input_ids, pad_visual_token_ids_tensor
        ).reshape(-1, 1) # 重塑为(N, 1)

    def get_input_embeddings(self): # 获取输入嵌入层
        return self.model.embed_tokens

    def should_apply_lora(self, module_name: str) -> bool: # 判断是否应用LoRA
        # skip vision_model # 跳过视觉模型
        return not module_name.startswith("vision_model") # 不对视觉模型应用LoRA

    def forward( # Ernie4.5 VL MoE条件生成模型前向传播
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置编码
        forward_batch: ForwardBatch, # 前向批信息
        get_embedding: bool = False, # 是否获取嵌入
    ):
        """Run forward pass for Ernie45-VL. # 运行Ernie45-VL前向传播

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch. # 展平（拼接）的输入ID，对应一个批次
            positions: Flattened (concatenated) position ids corresponding to a
                batch. # 展平（拼接）的位置ID，对应一个批次
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,). # **注意**：如果启用了mrope（Qwen2-VL开源模型默认设置），形状为(3, seq_len)，否则为(seq_len,)
                (Use input_metadata.mrope_positions to replace it) # 使用input_metadata.mrope_positions替换
        """
        if self.is_mrope_enabled: # 启用多模态RoPE
            positions = forward_batch.mrope_positions # 使用mrope位置

        if not ( # 如果不是解码模式或包含图像输入
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled: # 启用mrope时检查位置维度
                assert positions.ndim == 2 and positions.size(0) == 3, ( # 断言位置为(3, seq_len)
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        self._set_visual_token_mask(input_ids, forward_batch) # 设置视觉token掩码

        assert ( # 断言输入ID和位置长度一致
            input_ids.numel() == positions.shape[-1]
        ), f"input_ids {input_ids.shape} and position_ids {positions.shape} should have the same length"

        hidden_states = general_mm_embed_routine( # 通用多模态嵌入处理
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model, # 语言模型
            multimodal_model=self, # 多模态模型（自身）
            positions=positions,
            visual_token_mask=self.visual_token_mask, # 视觉token掩码
        )

        self.visual_token_mask = None # 清除视觉token掩码

        return self.logits_processor( # 处理logits
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        stacked_params_mapping = [ # 堆叠参数映射
            # (param_name, shard_name, shard_id) # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"), # Q投影
            ("qkv_proj", "k_proj", "k"), # K投影
            ("qkv_proj", "v_proj", "v"), # V投影
            ("gate_up_proj", "up_proj", 1), # up投影
            ("gate_up_proj", "gate_proj", 0), # gate投影
        ]

        # resampler_weight_mappings # 重采样器权重映射
        resampler_weight_mapping = { # 重采样器权重名称映射
            "spatial_linear.0.": "spatial_linear1.", # 空间线性层0映射
            "spatial_linear.2.": "spatial_linear2.", # 空间线性层2映射
            "spatial_linear.3.": "spatial_norm.", # 空间归一化映射
            "temporal_linear.0.": "temporal_linear1.", # 时间线性层0映射
            "temporal_linear.2.": "temporal_linear2.", # 时间线性层2映射
            "temporal_linear.3.": "temporal_norm.", # 时间归一化映射
        }

        expert_params_mapping = FusedMoE.make_expert_params_mapping( # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj", # 检查点gate投影名
            ckpt_down_proj_name="down_proj", # 检查点down投影名
            ckpt_up_proj_name="up_proj", # 检查点up投影名
            num_experts=max(self.config.moe_num_experts), # 最大专家数
        )
        params_dict = dict(self.named_parameters(remove_duplicate=False)) # 获取参数字典
        for name, loaded_weight in weights: # 遍历权重
            if "rotary_emb.inv_freq" in name: # 跳过旋转嵌入逆频率
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name: # 绑定词嵌入时跳过lm_head
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射
                if weight_name not in name: # 权重名不匹配则跳过
                    continue

                if ("mlp.experts." in name) and name not in params_dict: # 专家权重且不在参数字典中
                    continue
                name = name.replace(weight_name, param_name) # 替换为堆叠参数名

                # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break
            else: # 非堆叠参数
                if "vision_model" in name: # 视觉模型权重
                    # adapt to VisionAttention # 适配VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.") # 替换注意力投影名
                if name.startswith("model.resampler_model"): # 重采样器权重
                    name = name.replace("model.resampler_model", "resampler_model") # 替换前缀

                for ( # 遍历重采样器权重映射
                    old_weight_name,
                    new_weight_name,
                ) in resampler_weight_mapping.items():
                    if old_weight_name in name: # 旧权重名在名称中
                        name = name.replace(old_weight_name, new_weight_name, 1) # 替换为新权重名
                        break

                # Distinguish between vision experts and text experts # 区分视觉专家和文本专家
                if "mlp.experts" in name: # 专家权重
                    moe_offset = int(name.split(".")[-3]) # 获取专家偏移量
                    vision_expert_start_idx = self.config.moe_num_experts[0] # 视觉专家起始索引
                    is_text_expert = moe_offset <= vision_expert_start_idx - 1 # 判断是否为文本专家
                    if is_text_expert: # 文本专家
                        name = name.replace(".experts.", ".text_experts.") # 替换为文本专家名
                    else: # 视觉专家
                        name = name.replace( # 替换为视觉专家名并调整索引
                            f".experts.{moe_offset}",
                            f".vision_experts.{moe_offset - vision_expert_start_idx}",
                        )

                for mapping in expert_params_mapping: # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping # 解包映射
                    if weight_name not in name: # 权重名不匹配则跳过
                        continue

                    # Distinguish between vision experts and text experts # 区分视觉专家和文本专家
                    moe_offset = int(name.split(".")[-3]) # 获取专家偏移量
                    is_text_expert = moe_offset <= self.config.moe_num_experts[0] - 1 # 判断是否为文本专家

                    name = name.replace(weight_name, param_name) # 替换权重名
                    if is_text_expert: # 文本专家
                        name = name.replace(".experts.", ".text_experts.") # 替换为文本专家名
                    else: # 视觉专家
                        name = name.replace(".experts.", ".vision_experts.") # 替换为视觉专家名

                    # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue

                    if name in params_dict.keys(): # 参数名存在于参数字典
                        param = params_dict[name] # 获取参数
                        weight_loader = param.weight_loader # 获取权重加载器
                        weight_loader( # 加载专家权重
                            param,
                            loaded_weight,
                            name, # 权重名称
                            shard_id=shard_id, # 分片ID
                            expert_id=expert_id, # 专家ID
                        )
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict") # 参数未找到警告
                    break
                else: # 非专家参数
                    # Distinguish between vision expert gate
                    # and text expert gate # 区分视觉专家门控和文本专家门控
                    if name.endswith("mlp.gate.weight"): # 文本专家门控
                        name = name.replace("gate.weight", "text_experts_gate.weight") # 替换为文本门控名
                        loaded_weight = loaded_weight.T # 转置权重
                    elif name.endswith("mlp.gate.weight_1"): # 视觉专家门控
                        name = name.replace(
                            "gate.weight_1", "vision_experts_gate.weight" # 替换为视觉门控名
                        )
                        loaded_weight = loaded_weight.T # 转置权重

                    if "e_score_correction_bias" in name: # 专家分数校正偏置
                        name = name.replace(".moe_statics.", ".") # 替换MoE统计前缀

                    # Skip loading extra bias for GPTQ models. # 跳过GPTQ模型的额外偏置
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue

                    if name in params_dict.keys(): # 参数名存在于参数字典
                        param = params_dict[name] # 获取参数
                        weight_loader = getattr( # 获取权重加载器
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight) # 加载权重
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict") # 参数未找到警告


EntryClass = [Ernie4_5_VLMoeForConditionalGeneration] # 模型入口类列表
