# GLM-4.1V 视觉语言模型文件
# 本文件实现了仅推理模式的 GLM-4.1V 多模态视觉语言模型，
# 包含视觉编码器、图像/视频特征提取、多模态嵌入融合等功能，
# 兼容 HuggingFace 权重格式。

# Copyright 2023-2024 SGLang Team # 版权所有 2023-2024 SGLang 团队
# Licensed under the Apache License, Version 2.0 (the "License"); # 根据 Apache 许可证 2.0 版本授权
# you may not use this file except in compliance with the License. # 除非遵守许可证，否则不得使用此文件。
# You may obtain a copy of the License at # 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS, # 依据许可证分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不附带任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and # 请参阅许可证以获取管理权限和
# limitations under the License. # 限制的具体条款
# ==============================================================================

# Modeling from:  # 建模参考：
# ./llama.py and  # ./llama.py 和
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4v/modular_glm4v.py  # HuggingFace GLM4V 模块化实现
"""Inference-only GLM-4.1V model compatible with HuggingFace weights."""  # 仅推理的 GLM-4.1V 模型，兼容 HuggingFace 权重

import logging  # 导入日志模块
from functools import lru_cache  # 导入 LRU 缓存装饰器
from typing import Iterable, List, Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from einops import rearrange  # 导入张量重排工具
from transformers.models.glm4v.configuration_glm4v import Glm4vConfig, Glm4vVisionConfig  # 导入 GLM4V 配置类

from sglang.srt.distributed import (  # 导入分布式相关函数
    get_tensor_model_parallel_rank,  # 获取张量并行秩
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.distributed.parallel_state import get_pp_group  # 导入获取流水线并行组的函数
from sglang.srt.layers.activation import SiluAndMul  # 导入 SiLU 与乘法激活函数
from sglang.srt.layers.attention import vision_utils  # 导入视觉注意力工具
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.conv import Conv3dLayer  # 导入 3D 卷积层
from sglang.srt.layers.layernorm import LayerNorm, RMSNorm  # 导入层归一化和 RMS 归一化
from sglang.srt.layers.linear import (  # 导入线性层
    MergedColumnParallelLinear,  # 合并列并行线性层
    ReplicatedLinear,  # 复制线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层和池化类型
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.rotary_embedding import get_rope  # 导入旋转位置编码
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs  # 导入多模态数据项和多模态输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors  # 导入前向批次信息和流水线代理张量
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.glm4 import Glm4Model  # 从 glm4 模型导入 GLM4 模型
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model  # 导入数据并行分片多节旋转位置编码视觉模型运行函数
from sglang.srt.server_args import get_global_server_args  # 导入获取全局服务器参数的函数
from sglang.srt.utils import add_prefix, is_npu  # 导入前缀添加和 NPU 判断工具
from sglang.srt.utils.hf_transformers_utils import get_processor  # 导入 HuggingFace 处理器获取函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

cached_get_processor = lru_cache(get_processor)  # 缓存处理器获取函数


class Glm4vRMSNorm(RMSNorm):  # GLM4V 专用 RMS 归一化层，继承自 RMSNorm
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        original_shape = x.shape  # 保存原始形状
        x_2d = x.contiguous().reshape(-1, original_shape[-1])  # 将输入重塑为2D，保持最后一维不变
        x_2d = super().forward(x_2d)  # 调用父类 RMSNorm 的前向传播
        x = x_2d.reshape(original_shape)  # 恢复原始形状
        return x  # 返回归一化后的张量


class Glm4vVisionMLP(nn.Module):  # GLM4V 视觉 MLP 层
    def __init__(  # 初始化方法
        self,
        in_features: int,  # 输入特征维度
        hidden_features: int,  # 隐藏层特征维度
        bias: bool = False,  # 是否使用偏置，默认为 False
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
    ):
        super().__init__()  # 调用父类初始化
        self.tp_size = (  # 张量并行大小
            1 if use_data_parallel else get_tensor_model_parallel_world_size()  # 数据并行时为1，否则取张量并行世界大小
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()  # 张量并行秩，数据并行时为0
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控-上投影合并线性层
            input_size=in_features,  # 输入大小
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj] # 输出大小为隐藏特征维度的2倍（门控投影和上投影）
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 添加前缀
            tp_size=self.tp_size,  # 张量并行大小
            tp_rank=self.tp_rank,  # 张量并行秩
        )
        self.down_proj = RowParallelLinear(  # 下投影行并行线性层
            hidden_features,  # 输入大小
            in_features,  # 输出大小
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 添加前缀
            tp_size=self.tp_size,  # 张量并行大小
            tp_rank=self.tp_rank,  # 张量并行秩
        )
        self.act_fn = SiluAndMul()  # SiLU 与乘法激活函数

    def forward(self, x: torch.Tensor):  # 前向传播方法
        gate_up, _ = self.gate_up_proj(x)  # 通过门控-上投影层
        x = self.act_fn(gate_up)  # 应用激活函数
        x, _ = self.down_proj(x)  # 通过下投影层
        return x  # 返回输出


class Glm4vVisionBlock(nn.Module):  # GLM4V 视觉 Transformer 块
    def __init__(  # 初始化方法
        self,
        dim: int,  # 嵌入维度
        intermediate_dim: int,  # 中间层维度
        num_heads: int,  # 注意力头数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
        attn_qkv_bias: bool = True,  # 注意力 QKV 是否使用偏置，默认为 True
        num_dummy_heads: int = 0,  # 虚拟头数量，默认为0
        rms_norm_eps: float = 1e-5,  # RMS 归一化 epsilon，默认为 1e-5
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)  # 第一个 RMS 归一化层
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)  # 第二个 RMS 归一化层

        self.attn = VisionAttention(  # 视觉注意力层
            embed_dim=dim,  # 嵌入维度
            num_heads=num_heads,  # 注意力头数
            projection_size=dim,  # 投影大小
            use_qkv_parallel=True,  # 使用 QKV 并行
            proj_bias=False,  # 投影不使用偏置
            qkv_bias=attn_qkv_bias,  # QKV 偏置
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 添加前缀
            num_dummy_heads=num_dummy_heads,  # 虚拟头数量
            use_data_parallel=use_data_parallel,  # 是否使用数据并行
        )
        self.mlp = Glm4vVisionMLP(  # 视觉 MLP 层
            dim,  # 嵌入维度
            intermediate_dim,  # 中间层维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 添加前缀
            use_data_parallel=use_data_parallel,  # 是否使用数据并行
        )

    def forward(  # 前向传播方法
        self,
        x: torch.Tensor,  # 输入张量
        cu_seqlens: torch.Tensor,  # 累计序列长度
        rotary_pos_emb_cos: torch.Tensor,  # 旋转位置编码余弦
        rotary_pos_emb_sin: torch.Tensor,  # 旋转位置编码正弦
    ) -> torch.Tensor:
        S, B, H = x.shape  # 获取序列长度、批次数和隐藏维度
        # norm1: flatten to 2D -> [S*B, H], then reshape back  # norm1：展平为2D -> [S*B, H]，然后恢复形状
        x2d = x.reshape(-1, H)  # 将输入重塑为2D
        hidden_states = self.norm1(x2d).reshape(S, B, H)  # 归一化后恢复形状

        # Attention expects [B, S, H]  # 注意力期望 [B, S, H] 格式
        hidden_states = rearrange(hidden_states, "s b h -> b s h")  # 重排为 [B, S, H]
        attn = self.attn(  # 通过注意力层
            hidden_states,  # 隐藏状态
            cu_seqlens=cu_seqlens,  # 累计序列长度
            rotary_pos_emb_cos=rotary_pos_emb_cos,  # 旋转位置编码余弦
            rotary_pos_emb_sin=rotary_pos_emb_sin,  # 旋转位置编码正弦
        )
        attn = rearrange(attn, "b s h -> s b h")  # 重排回 [S, B, H]

        # norm2 with fused residual-add: also 2D  # norm2 融合残差加法：也是2D
        attn2d = attn.reshape(-1, H)  # 将注意力输出展平为2D
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)  # 归一化并融合残差
        x_norm = x_norm_2d.reshape(S, B, H)  # 恢复归一化结果的形状
        x_after_add = x_after_add_2d.reshape(S, B, H)  # 恢复残差加法结果的形状

        # MLP and final residual  # MLP 和最终残差
        mlp_out = self.mlp(x_norm)  # 通过 MLP 层
        x = x_after_add + mlp_out  # 残差连接
        return x  # 返回输出


class Glm4vVisionPatchEmbed(nn.Module):  # GLM4V 视觉补丁嵌入层
    def __init__(  # 初始化方法
        self,
        patch_size: int = 14,  # 补丁大小，默认为14
        temporal_patch_size: int = 2,  # 时间补丁大小，默认为2
        in_channels: int = 3,  # 输入通道数，默认为3(RGB)
        hidden_size: int = 1536,  # 隐藏层大小，默认为1536
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.patch_size = patch_size  # 保存补丁大小
        self.temporal_patch_size = temporal_patch_size  # 保存时间补丁大小
        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.in_channels = in_channels  # 保存输入通道数

        kernel_size = (temporal_patch_size, patch_size, patch_size)  # 3D卷积核大小：(时间, 高度, 宽度)
        self.proj = Conv3dLayer(  # 3D卷积投影层
            in_channels,  # 输入通道数
            hidden_size,  # 输出通道数(隐藏层大小)
            kernel_size=kernel_size,  # 卷积核大小
            stride=kernel_size,  # 步幅与卷积核大小相同
            bias=True,  # 使用偏置
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        # Input x is 2-D: (num_patches, C * T * P * P)  # 输入 x 是2维的：(补丁数, C * T * P * P)
        # Reshape to 5-D for Conv3dLayer, then flatten back.  # 重塑为5维用于 Conv3dLayer，然后再展平回去
        x = x.view(  # 将输入重塑为5维
            -1,  # 自动推断补丁数
            self.in_channels,  # 通道数
            self.temporal_patch_size,  # 时间维度
            self.patch_size,  # 高度维度
            self.patch_size,  # 宽度维度
        )
        return self.proj(x).view(-1, self.hidden_size)  # 通过3D卷积后展平回2维


class Glm4vPatchMerger(nn.Module):  # GLM4V 补丁合并层，用于将视觉特征映射到语言模型空间
    def __init__(  # 初始化方法
        self,
        d_model: int,  # 模型维度
        context_dim: int,  # 上下文维度
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        bias: bool = False,  # 是否使用偏置，默认为 False
        prefix: str = "",  # 参数前缀，默认为空字符串
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.hidden_size = d_model  # 保存隐藏层大小
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()  # 张量并行大小
        tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()  # 张量并行秩
        self.proj = ReplicatedLinear(  # 复制线性投影层
            self.hidden_size,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("proj", prefix),  # 添加前缀
        )
        self.post_projection_norm = LayerNorm(self.hidden_size)  # 投影后层归一化
        self.gate_up_proj = MergedColumnParallelLinear(  # 门控-上投影合并线性层
            input_size=self.hidden_size,  # 输入大小
            output_sizes=[context_dim] * 2,  # 输出大小为上下文维度的2倍
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("gate_up_proj", prefix),  # 添加前缀
            tp_size=tp_size,  # 张量并行大小
            tp_rank=tp_rank,  # 张量并行秩
        )
        self.down_proj = RowParallelLinear(  # 下投影行并行线性层
            context_dim,  # 输入大小
            self.hidden_size,  # 输出大小
            bias=bias,  # 是否使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("down_proj", prefix),  # 添加前缀
            tp_size=tp_size,  # 张量并行大小
            tp_rank=tp_rank,  # 张量并行秩
        )
        self.extra_activation_func = nn.GELU()  # 额外的 GELU 激活函数

    def forward(self, x: torch.Tensor):  # 前向传播方法
        x, _ = self.proj(x)  # 通过投影层
        x = self.extra_activation_func(self.post_projection_norm(x))  # 归一化后应用 GELU 激活
        gate_up, _ = self.gate_up_proj(x)  # 通过门控-上投影层
        gate, up = gate_up.chunk(2, dim=-1)  # 将门控和上投影分离
        x = F.silu(gate) * up  # 应用 SiLU 门控激活
        x, _ = self.down_proj(x)  # 通过下投影层
        return x  # 返回输出


class Glm4vVisionEmbeddings(nn.Module):  # GLM4V 视觉位置嵌入层
    def __init__(self, config: Glm4vVisionConfig):  # 初始化方法，接收视觉配置
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.image_size = config.image_size  # 图像大小
        self.patch_size = config.patch_size  # 补丁大小

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 补丁数量
        self.num_positions = self.num_patches  # 位置数量等于补丁数量
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 位置嵌入层
        self.register_buffer(  # 注册缓冲区
            "position_ids",  # 位置ID
            torch.arange(self.num_positions).expand((1, -1)),  # 生成位置ID序列
            persistent=False,  # 不持久化
        )

    def forward(  # 前向传播方法
        self, embeddings, lengths, image_shapes, h_coords, w_coords  # 嵌入、长度、图像形状、高度坐标、宽度坐标
    ) -> torch.Tensor:
        pos_embed_weight = self.position_embedding.weight  # 获取位置嵌入权重
        hidden_size = pos_embed_weight.shape[1]  # 获取隐藏维度
        total_seq = h_coords.shape[0]  # 获取总序列长度
        device = pos_embed_weight.device  # 获取设备

        # Move coordinates to correct device  # 将坐标移动到正确的设备
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)  # 移动高度和宽度坐标到设备

        # Handle empty sequence case  # 处理空序列的情况
        if total_seq == 0:  # 如果序列为空
            adapted_pos_embed = torch.empty(  # 创建空的自适应位置嵌入
                0, hidden_size, device=device, dtype=pos_embed_weight.dtype  # 空张量，0行
            )
        else:  # 否则
            # Convert inputs to tensors if needed  # 如有需要将输入转换为张量
            if isinstance(lengths, list):  # 如果长度是列表
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)  # 转换为张量
            if not isinstance(image_shapes, torch.Tensor):  # 如果图像形状不是张量
                image_shapes = torch.tensor(  # 转换为张量
                    image_shapes, device=device, dtype=torch.long  # 指定设备和数据类型
                )

            # Prepare 2D position embedding  # 准备2D位置嵌入
            orig_size_sq = pos_embed_weight.shape[0]  # 原始大小的平方
            orig_size = int(orig_size_sq**0.5)  # 计算原始尺寸
            pos_embed_2d = (  # 2D位置嵌入
                pos_embed_weight.view(orig_size, orig_size, hidden_size)  # 重塑为2D网格
                .permute(2, 0, 1)  # 调整维度顺序为 [H, W, D] -> [D, H, W]
                .unsqueeze(0)  # 添加批次维度
                .to(device=device, dtype=torch.float32)  # 转换设备和数据类型
            )

            # Calculate target dimensions for each patch  # 计算每个补丁的目标维度
            target_h = torch.cat(  # 目标高度
                [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]  # 按长度重复每个图像的高度
            ).to(device=device, dtype=torch.float32)  # 转换设备和数据类型
            target_w = torch.cat(  # 目标宽度
                [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]  # 按长度重复每个图像的宽度
            ).to(device=device, dtype=torch.float32)  # 转换设备和数据类型

            # Normalize coordinates to [-1, 1] range for grid_sample  # 将坐标归一化到 [-1, 1] 范围以用于 grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)  # 转换高度坐标数据类型
            w_coords = w_coords.to(device=device, dtype=torch.float32)  # 转换宽度坐标数据类型
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1  # 归一化宽度坐标
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1  # 归一化高度坐标

            # Create sampling grid  # 创建采样网格
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)  # 堆叠为网格格式

            # Perform bicubic interpolation  # 执行双三次插值
            interpolated_embed_fp32 = F.grid_sample(  # 使用 grid_sample 进行插值
                pos_embed_2d,  # 2D位置嵌入
                grid,  # 采样网格
                mode="bicubic",  # 双三次插值模式
                align_corners=False,  # 不对齐角点
                padding_mode="border",  # 边界填充模式
            )

            # Reshape and convert back to original dtype  # 重塑并转换回原始数据类型
            adapted_pos_embed_fp32 = (  # 自适应位置嵌入(float32)
                interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)  # 移除多余维度并调整顺序
            )
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(  # 转换回原始数据类型
                embeddings.device  # 转换到嵌入的设备
            )

        # Add adapted position encoding to embeddings  # 将自适应位置编码添加到嵌入中
        embeddings = embeddings + adapted_pos_embed  # 加上位置嵌入
        return embeddings  # 返回添加位置编码后的嵌入


class Glm4vVisionModel(nn.Module):  # GLM4V 视觉模型，包含补丁嵌入、Transformer块和合并层
    def __init__(  # 初始化方法
        self,
        vision_config: Glm4vVisionConfig,  # 视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
        use_data_parallel: bool = False,  # 是否使用数据并行，默认为 False
    ) -> None:
        super().__init__()  # 调用父类初始化

        patch_size = vision_config.patch_size  # 获取补丁大小
        temporal_patch_size = vision_config.temporal_patch_size  # 获取时间补丁大小
        in_channels = vision_config.in_channels  # 获取输入通道数
        depth = vision_config.depth  # 获取Transformer深度（层数）
        self.hidden_size = vision_config.hidden_size  # 保存隐藏层大小
        self.num_heads = vision_config.num_heads  # 保存注意力头数

        self.patch_size = vision_config.patch_size  # 保存补丁大小
        self.spatial_merge_size = vision_config.spatial_merge_size  # 保存空间合并大小
        self.out_hidden_size = vision_config.out_hidden_size  # 保存输出隐藏层大小
        self.use_data_parallel = use_data_parallel  # 保存数据并行标志

        self.patch_embed = Glm4vVisionPatchEmbed(  # 创建补丁嵌入层
            patch_size=patch_size,  # 补丁大小
            temporal_patch_size=temporal_patch_size,  # 时间补丁大小
            in_channels=in_channels,  # 输入通道数
            hidden_size=self.hidden_size,  # 隐藏层大小
        )

        head_dim = self.hidden_size // self.num_heads  # 计算每个注意力头的维度
        self.rotary_pos_emb = get_rope(  # 创建旋转位置编码
            head_size=head_dim,  # 头维度
            rotary_dim=head_dim // 2,  # 旋转维度为头维度的一半
            max_position=8192,  # 最大位置数
            base=10000.0,  # 基数
            is_neox_style=True,  # 使用 NeoX 风格
        )

        self.blocks = nn.ModuleList(  # 创建 Transformer 块列表
            [
                Glm4vVisionBlock(  # 视觉 Transformer 块
                    dim=self.hidden_size,  # 嵌入维度
                    intermediate_dim=self.out_hidden_size,  # 中间层维度
                    num_heads=self.num_heads,  # 注意力头数
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),  # 添加前缀
                    num_dummy_heads=vision_config.num_dummy_heads,  # 虚拟头数量
                    rms_norm_eps=vision_config.rms_norm_eps,  # RMS 归一化 epsilon
                    attn_qkv_bias=vision_config.attention_bias,  # 注意力偏置
                    use_data_parallel=use_data_parallel,  # 是否使用数据并行
                )
                for layer_idx in range(depth)  # 遍历每一层
            ]
        )

        self.merger = Glm4vPatchMerger(  # 创建补丁合并层
            d_model=vision_config.out_hidden_size,  # 模型维度
            context_dim=vision_config.intermediate_size,  # 上下文维度
            quant_config=quant_config,  # 量化配置
            bias=False,  # 不使用偏置
            prefix=add_prefix("merger", prefix),  # 添加前缀
            use_data_parallel=use_data_parallel,  # 是否使用数据并行
        )

        self.embeddings = Glm4vVisionEmbeddings(vision_config)  # 创建视觉位置嵌入

        self.post_conv_layernorm = Glm4vRMSNorm(  # 卷积后 RMS 归一化
            vision_config.hidden_size, eps=vision_config.rms_norm_eps  # 隐藏层大小和 epsilon
        )
        self.downsample = nn.Conv2d(  # 2D下采样卷积层
            in_channels=vision_config.hidden_size,  # 输入通道数
            out_channels=vision_config.out_hidden_size,  # 输出通道数
            kernel_size=vision_config.spatial_merge_size,  # 卷积核大小等于空间合并大小
            stride=vision_config.spatial_merge_size,  # 步幅等于空间合并大小
        )
        self.post_layernorm = Glm4vRMSNorm(  # Transformer 后 RMS 归一化
            vision_config.hidden_size, eps=vision_config.rms_norm_eps  # 隐藏层大小和 epsilon
        )

    @property  # 属性装饰器
    def dtype(self) -> torch.dtype:  # 获取模型数据类型
        return self.patch_embed.proj.weight.dtype  # 返回补丁嵌入投影的权重数据类型

    @property  # 属性装饰器
    def device(self) -> torch.device:  # 获取模型设备
        return self.patch_embed.proj.weight.device  # 返回补丁嵌入投影的权重设备

    def rot_pos_emb(  # 计算旋转位置编码
        self, grid_thw: torch.Tensor  # 网格的时间-高度-宽度信息
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回余弦、正弦和位置ID
        pos_ids = []  # 初始化位置ID列表
        for t, h, w in grid_thw:  # 遍历每个网格
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  # 生成高度位置ID
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  # 生成宽度位置ID
            hpos_ids = (  # 重排高度位置ID以适应空间合并
                hpos_ids.reshape(  # 重塑形状
                    h // self.spatial_merge_size,  # 合并后的高度
                    self.spatial_merge_size,  # 合并大小
                    w // self.spatial_merge_size,  # 合并后的宽度
                    self.spatial_merge_size,  # 合并大小
                )
                .permute(0, 2, 1, 3)  # 交换维度顺序
                .flatten()  # 展平
            )
            wpos_ids = (  # 重排宽度位置ID以适应空间合并
                wpos_ids.reshape(  # 重塑形状
                    h // self.spatial_merge_size,  # 合并后的高度
                    self.spatial_merge_size,  # 合并大小
                    w // self.spatial_merge_size,  # 合并后的宽度
                    self.spatial_merge_size,  # 合并大小
                )
                .permute(0, 2, 1, 3)  # 交换维度顺序
                .flatten()  # 展平
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))  # 堆叠并按时间维度重复
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)  # 拼接所有位置ID
        max_grid_size = grid_thw[:, 1:].max()  # 获取最大网格尺寸

        # Use pre-computed cos_sin_cache from RotaryEmbedding  # 使用 RotaryEmbedding 预计算的 cos_sin 缓存
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)  # 获取余弦和正弦值

        cos_combined = cos[pos_ids].flatten(1)  # 根据位置ID索引余弦值并展平
        sin_combined = sin[pos_ids].flatten(1)  # 根据位置ID索引正弦值并展平
        return cos_combined, sin_combined, pos_ids  # 返回余弦、正弦和位置ID

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        # patchify  # 补丁化
        x = x.to(device=self.device, dtype=self.dtype)  # 将输入转换到正确的设备和数据类型
        x = self.patch_embed(x)  # 通过补丁嵌入层
        x = self.post_conv_layernorm(x)  # 卷积后归一化

        # compute position embedding  # 计算位置编码
        rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(  # 获取旋转位置编码
            grid_thw  # 网格信息
        )
        # compute cu_seqlens  # 计算累计序列长度
        cu_seqlens = torch.repeat_interleave(  # 按时间维度重复计算
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]  # 每个图像的补丁数 * 帧数
        ).cumsum(dim=0, dtype=torch.int32)  # 计算累积和
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])  # 在开头添加0

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()  # 计算每个序列的长度
        x = self.embeddings(  # 通过位置嵌入层
            x, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]  # 传入嵌入和坐标信息
        )

        rotary_pos_emb_cos = torch.cat([rotary_pos_emb_cos, rotary_pos_emb_cos], dim=-1)  # 拼接余弦编码（用于多头）
        rotary_pos_emb_sin = torch.cat([rotary_pos_emb_sin, rotary_pos_emb_sin], dim=-1)  # 拼接正弦编码（用于多头）

        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction  # cu_seqlens 必须在 CPU 上，因为 npu_flash_attention_unpad 算子限制
        if is_npu():  # 如果是 NPU 设备
            cu_seqlens = cu_seqlens.to("cpu")  # 将 cu_seqlens 移到 CPU

        # x.shape: (s, b, d) where b=1 for vision processing  # x.shape: (s, b, d)，其中视觉处理时 b=1
        # transformers  # Transformer 块
        x = x.unsqueeze(1)  # 添加批次维度
        for blk in self.blocks:  # 遍历每个 Transformer 块
            x = blk(  # 通过 Transformer 块
                x,  # 输入
                cu_seqlens=cu_seqlens,  # 累计序列长度
                rotary_pos_emb_cos=rotary_pos_emb_cos,  # 旋转位置编码余弦
                rotary_pos_emb_sin=rotary_pos_emb_sin,  # 旋转位置编码正弦
            )

        # adapter  # 适配器
        x = self.post_layernorm(x)  # Transformer 后归一化
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])  # 重塑为空间合并格式
        x = x.permute(0, 3, 1, 2)  # 调整维度顺序
        x = self.downsample(x).view(-1, self.out_hidden_size)  # 下采样并展平
        x = self.merger(x)  # 通过补丁合并层

        return x  # 返回视觉特征


class Glm4vForConditionalGeneration(nn.Module):  # GLM-4.1V 条件生成模型，整合视觉编码器和语言模型
    def __init__(  # 初始化方法
        self,
        config: Glm4vConfig,  # GLM4V 配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder  # 是否启用多模态数据并行编码器
        vision_utils.update_vit_attn_dummy_heads_config(self.config)  # 更新 ViT 注意力虚拟头配置
        self.visual = Glm4vVisionModel(  # 创建视觉模型
            config.vision_config,  # 视觉配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("visual", prefix),  # 添加前缀
            use_data_parallel=self.use_data_parallel,  # 是否使用数据并行
        )

        self.model = Glm4Model(  # 创建语言模型
            config,  # 模型配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("model", prefix),  # 添加前缀
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:  # 如果世界大小为1且绑定词嵌入
                self.lm_head = self.model.embed_tokens  # 语言模型头与嵌入层共享
            else:  # 否则
                self.lm_head = ParallelLMHead(  # 创建并行语言模型头
                    self.config.vocab_size,  # 词表大小
                    self.config.hidden_size,  # 隐藏层维度
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("lm_head", prefix),  # 添加前缀
                )
        else:  # 否则
            # ranks other than the last rank will have a placeholder layer  # 非最后一个秩将有一个占位层
            self.lm_head = PPMissingLayer()  # 流水线并行缺失层

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling  # 是否启用多节旋转位置编码

        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 创建池化层，使用最后token池化和归一化

        # For EAGLE3 support  # 用于 EAGLE3 支持
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态，默认为 False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID方法，将多模态token插入到输入序列中
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态数据填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式填充输入token

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取图像特征方法
        # in GLM-V, last dim is the same  # 在 GLM-V 中，最后一维相同
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接所有图像像素值
            self.visual.dtype  # 转换为视觉模型的数据类型
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)  # 拼接所有图像网格信息
        assert pixel_values.dim() == 2, pixel_values.dim()  # 断言像素值为2维
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()  # 断言网格信息为2维
        if self.use_data_parallel:  # 如果使用数据并行
            return run_dp_sharded_mrope_vision_model(  # 运行数据并行分片 mrope 视觉模型
                self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"  # 传入视觉模型、像素值和网格信息
            )
        else:  # 否则
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # 通过视觉模型获取图像嵌入
        return image_embeds  # 返回图像嵌入

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取视频特征方法
        # in GLM-V, last dim is the same  # 在 GLM-V 中，最后一维相同
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接所有视频像素值
            self.visual.dtype  # 转换为视觉模型的数据类型
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)  # 拼接所有视频网格信息

        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames  # 重塑视频网格信息 -> [b, 3] -> [1, h, w] * 帧数
        temp_frames_hw = []  # 临时帧高度宽度列表
        for t, h, w in video_grid_thw:  # 遍历每个视频的网格信息
            repeated_row = (  # 重复行
                torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)  # 每帧的网格信息 [1, h, w]，重复 t 次
            )
            temp_frames_hw.append(repeated_row)  # 添加到列表
        flattened_video_grid_thw = torch.cat(temp_frames_hw, dim=0)  # 拼接所有帧的网格信息

        assert pixel_values.dim() == 2, pixel_values.dim()  # 断言像素值为2维
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()  # 断言网格信息为2维
        if self.use_data_parallel:  # 如果使用数据并行
            return run_dp_sharded_mrope_vision_model(  # 运行数据并行分片 mrope 视觉模型
                self.visual,  # 视觉模型
                pixel_values,  # 像素值
                flattened_video_grid_thw.tolist(),  # 展平的视频网格信息
                rope_type="rope_3d",  # 旋转位置编码类型
            )
        else:  # 否则
            video_embeds = self.visual(pixel_values, grid_thw=flattened_video_grid_thw)  # 通过视觉模型获取视频嵌入
        return video_embeds  # 返回视频嵌入

    def get_input_embeddings(self):  # 获取输入嵌入方法
        return self.model.embed_tokens  # 返回语言模型的嵌入层

    @torch.no_grad()  # 禁用梯度计算装饰器
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID 张量
        positions: torch.Tensor,  # 位置编码张量
        forward_batch: ForwardBatch,  # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入，默认为 False
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线并行代理张量，可选
    ):
        """Run forward pass for GLM-4.1V.  # 运行 GLM-4.1V 的前向传播。

        Args:  # 参数：
            input_ids: Flattened (concatenated) input_ids corresponding to a  # input_ids：展平的（拼接的）输入ID，对应
                batch.  # 一个批次。
            positions: Flattened (concatenated) position ids corresponding to a  # positions：展平的（拼接的）位置ID，对应
                batch.  # 一个批次。
                **NOTE**: If mrope is enabled (default setting for GLM-4.1V  # **注意**：如果启用了 mrope（GLM-4.1V 的默认设置
                opensource models), the shape will be `(3, seq_len)`,  # 开源模型），形状将为 `(3, seq_len)`，
                otherwise it will be `(seq_len,).  # 否则将为 `(seq_len,)`。
                (Use input_metadata.mrope_positions to replace it)  # （使用 input_metadata.mrope_positions 替换）
        """
        if self.is_mrope_enabled:  # 如果启用了 mrope
            positions = forward_batch.mrope_positions  # 使用 mrope 位置

        if not (  # 如果不是
            forward_batch.forward_mode.is_decode()  # 解码模式
            or not forward_batch.contains_image_inputs()  # 或者不包含图像输入
        ):
            if self.is_mrope_enabled:  # 如果启用了 mrope
                assert positions.ndim == 2 and positions.size(0) == 3, (  # 断言位置为2维且第一维为3
                    "multimodal section rotary embedding requires "  # 多模态分段旋转位置编码需要
                    f"(3, seq_len) positions, but got {positions.size()}"  # `(3, seq_len)` 的位置，但得到 {positions.size()}
                )

        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次信息
            language_model=self.model,  # 语言模型
            multimodal_model=self,  # 多模态模型（自身）
            positions=positions,  # 位置编码
            pp_proxy_tensors=pp_proxy_tensors,  # 流水线并行代理张量
        )

        aux_hidden_states = None  # 初始化辅助隐藏状态为 None
        if self.capture_aux_hidden_states:  # 如果捕获辅助隐藏状态
            hidden_states, aux_hidden_states = hidden_states  # 分离隐藏状态和辅助隐藏状态

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            if not get_embedding:  # 如果不获取嵌入
                return self.logits_processor(  # 通过 logits 处理器返回 logits
                    input_ids,  # 输入ID
                    hidden_states,  # 隐藏状态
                    self.lm_head,  # 语言模型头
                    forward_batch,  # 前向批次信息
                )
            else:  # 否则
                return self.pooler(hidden_states, forward_batch)  # 通过池化层返回嵌入
        else:  # 否则
            return hidden_states  # 返回隐藏状态

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):  # 填充 ViT 注意力虚拟头方法
        """pad attn qkv weights for dummy heads"""  # 为虚拟头填充注意力 QKV 权重
        num_dummy_heads = self.config.vision_config.num_dummy_heads  # 获取虚拟头数量
        if num_dummy_heads == 0:  # 如果没有虚拟头
            return loaded_weight  # 直接返回原始权重
        head_dim = self.config.vision_config.head_dim  # 获取头维度

        if "attn.qkv_proj" in name:  # 如果是 QKV 投影权重
            wq, wk, wv = loaded_weight.chunk(3, dim=0)  # 将权重分为 Q、K、V 三部分
            if name.endswith(".weight"):  # 如果是权重
                dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]  # 虚拟头权重的形状
            elif name.endswith(".bias"):  # 如果是偏置
                dummy_shape = [num_dummy_heads, head_dim]  # 虚拟头偏置的形状
            else:  # 否则
                raise RuntimeError(f"Unsupported weight with name={name}")  # 抛出不支持的权重错误
            pad_func = lambda x: torch.cat(  # 填充函数
                [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0  # 将虚拟头零值拼接到原始权重
            ).flatten(0, 1)  # 展平
            wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)  # 对 Q、K、V 分别填充
            loaded_weight = torch.cat([wq, wk, wv], dim=0)  # 拼接回 QKV 权重
        elif "attn.proj.weight" in name:  # 如果是输出投影权重
            padded_weight = loaded_weight.new_zeros(  # 创建零值填充权重
                loaded_weight.shape[0], head_dim * num_dummy_heads  # 形状为 [输出维度, 虚拟头维度]
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)  # 在最后一维拼接
        elif "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:  # 如果是 Q 或 K 归一化权重
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)  # 创建零值填充权重
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)  # 在第一维拼接
        return loaded_weight  # 返回填充后的权重

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载权重方法
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # QKV 投影中的 Q
            (".qkv_proj", ".k_proj", "k"),  # QKV 投影中的 K
            (".qkv_proj", ".v_proj", "v"),  # QKV 投影中的 V
            (".gate_up_proj", ".up_proj", 1),  # 门控上投影中的上投影
            (".gate_up_proj", ".gate_proj", 0),  # 门控上投影中的门控投影
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取模型参数字典

        # For the PP case, we add special handling for lm_head.weight,  # 对于流水线并行情况，我们对 lm_head.weight 添加特殊处理，
        # - On non–last ranks: we continue, because this stage is supposed to  # - 在非最后一个秩上：我们继续，因为这个阶段应该是
        #   be just an empty PPMissingLayer shell.  # 只是一个空的 PPMissingLayer 外壳。
        # - On the last rank: params_dict is expected to contain lm_head.weight,  # - 在最后一个秩上：params_dict 应包含 lm_head.weight，
        #   so it will never hit the branch "if name not in params_dict".  # 所以不会进入 "if name not in params_dict" 分支。
        #
        # For all other parameters, such like  # 对于所有其他参数，例如
        # "model.visual.blocks.20.mlp.gate_proj.weight", the unified rule is:  # "model.visual.blocks.20.mlp.gate_proj.weight"，统一规则是：
        # If this name does not exist in the current rank's params_dict,  # 如果此名称不在当前秩的 params_dict 中，
        # it does not belong to this pipeline stage, thus we simply continue.  # 它不属于此流水线阶段，因此我们直接继续。
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转嵌入的逆频率
                continue  # 跳过
            if "language_model" in name:  # 如果名称包含 language_model
                name = name.replace(r"model.language_model.", r"model.")  # 替换为 model. 前缀
            if "model.visual." in name:  # 如果名称包含 model.visual.
                name = name.replace("model.visual.", "visual.")  # 替换为 visual. 前缀

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果分片名不在名称中
                    continue  # 跳过
                name = name.replace(weight_name, param_name)  # 替换分片名为参数名

                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载。
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过

                if name not in params_dict:  # 如果参数名不在参数字典中
                    continue  # 跳过

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果没有匹配的堆叠参数
                if "visual" in name:  # 如果是视觉模型参数
                    # adapt to VisionAttention  # 适配 VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换注意力 QKV 名称

                try:  # 尝试
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载。
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                        continue  # 跳过

                    if name not in params_dict:  # 如果参数名不在参数字典中
                        continue  # 跳过

                    param = params_dict[name]  # 获取参数
                except KeyError:  # 捕获键错误
                    print(params_dict.keys())  # 打印参数字典的所有键
                    raise  # 重新抛出异常

                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器，默认使用 default_weight_loader
                if "visual" in name:  # 如果是视觉模型参数
                    loaded_weight = vision_utils.pad_vit_attn_dummy_heads(  # 填充 ViT 注意力虚拟头
                        self.config, name, loaded_weight  # 传入配置、名称和权重
                    )
                weight_loader(param, loaded_weight)  # 加载权重

    def get_embed_and_head(self):  # 获取嵌入和语言模型头方法
        return self.model.embed_tokens.weight, self.lm_head.weight  # 返回嵌入权重和语言模型头权重

    def set_embed_and_head(self, embed, head):  # 设置嵌入和语言模型头方法
        del self.model.embed_tokens.weight  # 删除旧的嵌入权重
        self.model.embed_tokens.weight = embed  # 设置新的嵌入权重
        if self.config.tie_word_embeddings:  # 如果绑定词嵌入
            self.lm_head = self.model.embed_tokens  # 语言模型头与嵌入层共享
        else:  # 否则
            del self.lm_head.weight  # 删除旧的语言模型头权重
            self.lm_head.weight = head  # 设置新的语言模型头权重
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        torch.cuda.synchronize()  # 同步 CUDA


EntryClass = [Glm4vForConditionalGeneration]  # 入口类列表，用于模型注册
