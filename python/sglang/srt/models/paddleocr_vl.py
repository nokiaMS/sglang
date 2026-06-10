# PaddleOCR视觉语言模型推理实现文件
# 本文件实现了PaddleOCR-VL多模态模型的推理架构
# 包含投影器、SigLIP视觉编码器、视觉Transformer及条件生成模型等组件

# Reference: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Iterable  # 导入可迭代类型
from typing import List, Optional, Set, Tuple, Union  # 导入类型提示

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
from einops import rearrange  # 导入张量重排工具
from transformers.activations import GELUActivation  # 导入GELU激活函数
from transformers.utils import torch_int  # 导入torch整数工具

from sglang.srt.layers.activation import get_act_fn  # 导入获取激活函数的工具
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.conv import Conv2dLayer  # 导入2D卷积层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear  # 导入并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs  # 导入多模态数据项和输入
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.ernie4 import Ernie4_5_ForCausalLM  # 导入Ernie4.5因果语言模型
from sglang.srt.utils import add_prefix, is_npu  # 导入前缀添加和NPU检测工具


class Projector(nn.Module):  # 投影器模块，将视觉特征投影到文本空间

    def __init__(  # 初始化函数
        self,
        text_config,  # 文本配置
        vision_config,  # 视觉配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.text_config = text_config  # 保存文本配置
        self.vision_config = vision_config  # 保存视觉配置
        self.merge_kernel_size = (2, 2)  # 合并核大小

        self.hidden_size = (  # 计算隐藏层大小
            self.vision_config.hidden_size  # 视觉隐藏大小
            * self.merge_kernel_size[0]  # 乘以合并核高度
            * self.merge_kernel_size[1]  # 乘以合并核宽度
        )

        self.pre_norm = torch.nn.LayerNorm(self.vision_config.hidden_size, eps=1e-05)  # 预归一化层
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # 第一个线性层
        self.act = GELUActivation()  # GELU激活函数
        self.linear_2 = nn.Linear(  # 第二个线性层
            self.hidden_size, self.text_config.hidden_size, bias=True  # 投影到文本隐藏大小
        )

    def forward(  # 前向传播函数，将视觉特征投影到文本空间
        self,
        image_features: torch.Tensor,  # 图像特征
        image_grid_thw: List[Tuple[int, int, int]],  # 图像网格时间-高度-宽度
    ) -> torch.Tensor:
        m1, m2 = self.merge_kernel_size  # 获取合并核大小
        if isinstance(image_features, (list, tuple)):  # 如果图像特征是列表或元组
            processed_features = list()  # 初始化处理后的特征列表
            for image_feature, image_grid in zip(image_features, image_grid_thw):  # 遍历每个图像特征和网格
                image_feature = self.pre_norm(image_feature)  # 预归一化
                t, h, w = image_grid  # 获取时间、高度、宽度

                image_feature = rearrange(  # 重排图像特征
                    image_feature,
                    "(t h p1 w p2) d -> (t h w) (p1 p2 d)",  # 从扁平化重排为网格格式
                    t=t,  # 时间维度
                    h=h // m1,  # 高度除以合并核高度
                    p1=m1,  # 合并核高度
                    w=w // m2,  # 宽度除以合并核宽度
                    p2=m2,  # 合并核宽度
                )
                hidden_states = self.linear_1(image_feature)  # 通过第一个线性层
                hidden_states = self.act(hidden_states)  # 应用激活函数
                hidden_states = self.linear_2(hidden_states)  # 通过第二个线性层
                processed_features.append(hidden_states)  # 添加到处理后特征列表

            return processed_features  # 返回处理后的特征列表

        dims = image_features.shape[:-1]  # 获取除最后一维外的维度
        dim = image_features.shape[-1]  # 获取最后一维
        image_features = image_features.view(np.prod(dims), dim)  # 重塑图像特征
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)  # 预归一化并重塑
        hidden_states = self.linear_1(hidden_states)  # 通过第一个线性层
        hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states = self.linear_2(hidden_states)  # 通过第二个线性层

        return hidden_states.view(*dims, -1)  # 重塑并返回


class SiglipVisionEmbeddings(nn.Module):  # SigLIP视觉嵌入模块

    def __init__(self, config):  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.image_size = config.image_size  # 图像大小
        self.patch_size = config.patch_size  # 补丁大小

        self.patch_embedding = Conv2dLayer(  # 补丁嵌入卷积层
            in_channels=config.num_channels,  # 输入通道数
            out_channels=self.embed_dim,  # 输出通道数
            kernel_size=self.patch_size,  # 卷积核大小
            stride=self.patch_size,  # 步幅
            padding="valid",  # 不使用填充
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 补丁数量
        self.num_positions = self.num_patches  # 位置数量
        self.cache_position_embedding = dict()  # 位置嵌入缓存
        self.cache_position_count = dict()  # 位置缓存计数
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 位置嵌入层
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)  # 打包位置嵌入层

        self.register_buffer(  # 注册缓冲区
            "position_ids",  # 位置ID
            torch.arange(self.num_positions).expand((1, -1)),  # 位置ID范围
            persistent=False,  # 非持久化
        )

    def interpolate_pos_encoding(  # 插值位置编码函数
        self,
        embeddings: torch.Tensor,  # 嵌入张量
        height: int,  # 高度
        width: int,  # 宽度
        is_after_patchify: bool = False,  # 是否在补丁化之后
    ) -> torch.Tensor:

        num_positions = self.position_embedding.weight.shape[0]  # 位置数量

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)  # 获取位置嵌入权重

        dim = embeddings.shape[-1]  # 嵌入维度

        if is_after_patchify:  # 如果在补丁化之后
            new_height = height  # 新高度
            new_width = width  # 新宽度
        else:  # 否则
            new_height = height // self.patch_size  # 新高度除以补丁大小
            new_width = width // self.patch_size  # 新宽度除以补丁大小

        sqrt_num_positions = torch_int(num_positions**0.5)  # 位置数的平方根
        patch_pos_embed = patch_pos_embed.reshape(  # 重塑位置嵌入
            1, sqrt_num_positions, sqrt_num_positions, dim  # 二维网格形状
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # 转换为NCHW格式

        patch_pos_embed = nn.functional.interpolate(  # 双线性插值
            patch_pos_embed,
            size=(new_height, new_width),  # 目标大小
            mode="bilinear",  # 双线性模式
            align_corners=False,  # 不对齐角点
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)  # 转回NHWC并展平
        return patch_pos_embed  # 返回插值后的位置编码

    def fetch_position_embedding_lfu_cache(self, embeddings, h, w, max_cache: int = 20):  # 从LFU缓存获取位置嵌入
        grid = (h, w)  # 网格大小
        if grid in self.cache_position_embedding:  # 如果网格在缓存中
            self.cache_position_count[grid] += 1  # 增加计数
            return self.cache_position_embedding[grid]  # 返回缓存的位置嵌入

        if len(self.cache_position_embedding) >= max_cache:  # 如果缓存已满
            min_hit_grid = min(  # 找到最少使用的网格
                self.cache_position_count,
                key=self.cache_position_count.get,  # 按计数获取
            )
            self.cache_position_count.pop(min_hit_grid)  # 移除最少使用的计数
            self.cache_position_embedding.pop(min_hit_grid)  # 移除最少使用的嵌入

        position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)  # 计算位置嵌入
        self.cache_position_count[grid] = 1  # 初始化计数为1
        self.cache_position_embedding[grid] = position_embedding  # 存入缓存
        return position_embedding  # 返回位置嵌入

    def forward(  # 前向传播函数，计算视觉嵌入
        self,
        pixel_values: torch.FloatTensor,  # 像素值
        position_ids: Optional[torch.Tensor] = None,  # 位置ID，可选
        image_grid_thw: Optional[  # 图像网格THW，可选
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
        interpolate_pos_encoding=False,  # 是否插值位置编码
    ) -> torch.Tensor:
        if pixel_values.dim() == 4:  # 如果像素值是4维
            pixel_values = pixel_values.unsqueeze(0)  # 添加批次维度
        if pixel_values.dim() == 5:  # 如果像素值是5维
            if position_ids is None:  # 如果位置ID为空
                raise ValueError(  # 抛出值错误
                    "position_ids cannot be None when pixel_values.dim() is 5."  # 位置ID不能为空
                )
            (  # 解包像素值形状
                batch_size,  # 批次大小
                squence_len,  # 序列长度
                channel,  # 通道数
                height,  # 高度
                width,  # 宽度
            ) = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype  # 目标数据类型
            pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")  # 重排像素值
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 通过补丁嵌入
            embeddings = patch_embeds.flatten(-2).squeeze(-1)  # 展平并移除多余维度

            if interpolate_pos_encoding and image_grid_thw is not None:  # 如果需要插值位置编码且网格THW存在
                start = 0  # 起始索引
                tmp_embeddings = list()  # 临时嵌入列表
                for image_grid in image_grid_thw:  # 遍历每个图像网格
                    t, h, w = image_grid  # 获取时间、高度、宽度
                    end = start + t * h * w  # 结束索引
                    image_embeddings = embeddings[start:end, :]  # 获取当前图像嵌入
                    position_embedding = (  # 计算位置嵌入
                        self.interpolate_pos_encoding(image_embeddings, h, w, True)  # 插值位置编码
                        .squeeze(0)  # 移除批次维度
                        .repeat(t, 1)  # 重复时间维度
                    )
                    image_embeddings = image_embeddings + position_embedding  # 加上位置嵌入
                    tmp_embeddings.append(image_embeddings)  # 添加到临时列表
                    start = end  # 更新起始索引
                embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)  # 拼接并添加批次维度
            else:  # 否则
                embeddings = embeddings + self.packing_position_embedding(position_ids)  # 加上打包位置嵌入
            return embeddings  # 返回嵌入
        else:  # 否则
            raise ValueError(  # 抛出值错误
                "Unsupported pixel_values dimension:"  # 不支持的像素值维度
                f" {pixel_values.dim()}. Expected 4 or 5."  # 期望4或5维
            )


class SigLIPRotaryEmbedding(nn.Module):  # SigLIP旋转位置编码模块

    def __init__(self, dim: int, theta: float = 10000.0) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.dim = dim  # 保存维度
        self.theta = theta  # 保存theta
        self.rope_init()  # 初始化旋转位置编码

    def rope_init(self):  # 旋转位置编码初始化函数
        inv_freq = 1.0 / (  # 计算逆频率
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)  # 频率公式
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册逆频率缓冲区

    def forward(self, seqlen: int) -> torch.Tensor:  # 前向传播函数，计算旋转位置编码
        seq = torch.arange(  # 创建序列
            seqlen,  # 序列长度
            device=self.inv_freq.device,  # 设备
            dtype=self.inv_freq.dtype,  # 数据类型
        )
        freqs = torch.outer(seq, self.inv_freq)  # 计算外积得到频率
        return freqs  # 返回频率


class SiglipMLP(nn.Module):  # SigLIP的MLP模块

    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.activation_fn = get_act_fn(config.hidden_act)  # 获取激活函数
        if quant_config and quant_config.get_name() in ["bitsandbytes", "torchao"]:  # 如果是特定量化
            quantizable = True  # 可量化
        else:  # 否则
            quantizable = (  # 检查是否可量化
                config.hidden_size % 64 == 0 and config.intermediate_size % 64 == 0  # 维度需被64整除
            )
        self.fc1 = ColumnParallelLinear(  # 第一个列并行线性层
            config.hidden_size,  # 输入大小
            config.intermediate_size,  # 输出大小
            quant_config=quant_config if quantizable else None,  # 量化配置
            prefix=add_prefix("fc1", prefix),  # 添加前缀
        )
        self.fc2 = RowParallelLinear(  # 第二个行并行线性层
            config.intermediate_size,  # 输入大小
            config.hidden_size,  # 输出大小
            quant_config=quant_config if quantizable else None,  # 量化配置
            prefix=add_prefix("fc2", prefix),  # 添加前缀
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        hidden_states, _ = self.fc1(hidden_states)  # 通过第一个线性层
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states, _ = self.fc2(hidden_states)  # 通过第二个线性层
        return hidden_states  # 返回隐藏状态


class SiglipEncoderLayer(nn.Module):  # SigLIP编码器层

    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层归一化

        self.self_attn = VisionAttention(  # 视觉注意力层
            embed_dim=self.embed_dim,  # 嵌入维度
            num_heads=config.num_attention_heads,  # 注意力头数
            projection_size=self.embed_dim,  # 投影大小
            use_qkv_parallel=True,  # 使用QKV并行
            qkv_bias=True,  # 使用QKV偏置
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
        )

        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层归一化
        self.mlp = SiglipMLP(  # MLP模块
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)  # 传入配置
        )

    def forward(  # 前向传播函数，执行编码器层计算
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        cu_seqlens: Optional[List[torch.Tensor]] = None,  # 累积序列长度
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 旋转位置编码
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states  # 保存残差

        hidden_states = self.layer_norm1(hidden_states)  # 应用第一个层归一化

        hidden_states = self.self_attn(  # 通过自注意力层
            hidden_states,  # 隐藏状态
            cu_seqlens=cu_seqlens,  # 累积序列长度
            position_embeddings=rope_emb,  # 位置编码
        )

        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.layer_norm2(hidden_states)  # 应用第二个层归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP

        hidden_states = residual + hidden_states  # 残差连接

        return hidden_states  # 返回隐藏状态


class SiglipEncoder(nn.Module):  # SigLIP编码器

    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        embed_dim = config.hidden_size  # 嵌入维度
        num_heads = config.num_attention_heads  # 注意力头数
        head_dim = embed_dim // num_heads  # 每个头的维度
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                SiglipEncoderLayer(  # 编码器层
                    config,  # 配置
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),  # 添加前缀
                )
                for layer_idx in range(config.num_hidden_layers)  # 遍历隐藏层数
            ]
        )
        self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2)  # 旋转位置编码

    @staticmethod
    def flatten_list(image_grid_thw):  # 展平图像网格THW列表
        tmp_image_grid_thw = list()  # 临时列表
        for image_grid in image_grid_thw:  # 遍历每个图像网格
            if isinstance(image_grid, list):  # 如果是列表
                tmp_image_grid_thw.extend(image_grid)  # 扩展到临时列表
            else:  # 否则
                tmp_image_grid_thw.append(image_grid)  # 添加到临时列表
        return tmp_image_grid_thw  # 返回展平后的列表

    def forward(  # 前向传播函数，执行编码器计算
        self,
        inputs_embeds,  # 输入嵌入
        cu_seqlens: Optional[List[torch.Tensor]] = None,  # 累积序列长度
        image_grid_thw: Optional[  # 图像网格THW
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
        height_position_ids: Optional[torch.Tensor] = None,  # 高度位置ID
        width_position_ids: Optional[torch.Tensor] = None,  # 宽度位置ID
    ) -> torch.Tensor:
        device = inputs_embeds.device  # 获取设备
        hidden_states = inputs_embeds  # 初始化隐藏状态
        flatten_image_grid_thw = self.flatten_list(image_grid_thw)  # 展平图像网格THW

        if width_position_ids is None or height_position_ids is None:  # 如果位置ID为空
            split_hids = list()  # 高度位置ID列表
            split_wids = list()  # 宽度位置ID列表
            for t, h, w in flatten_image_grid_thw:  # 遍历每个图像网格
                image_pids = torch.arange(t * h * w, device=device) % (h * w)  # 计算图像位置ID
                sample_hids = image_pids // w  # 计算高度位置ID
                sample_wids = image_pids % w  # 计算宽度位置ID
                split_hids.append(sample_hids)  # 添加到高度列表
                split_wids.append(sample_wids)  # 添加到宽度列表
            width_position_ids = torch.concat(split_wids, dim=0)  # 拼接宽度位置ID
            height_position_ids = torch.concat(split_hids, dim=0)  # 拼接高度位置ID

        pids = torch.stack(  # 堆叠位置ID
            [height_position_ids, width_position_ids],  # 高度和宽度位置ID
            dim=-1,  # 最后一维
        )
        max_grid_size = pids.max() + 1  # 最大网格大小
        rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)  # 计算旋转位置编码
        rope_emb = rope_emb_max_grid[pids].flatten(1)  # 获取对应位置的编码并展平
        rope_emb = rope_emb.repeat(1, 2)  # 重复以匹配注意力头维度
        rope_emb = (rope_emb.cos(), rope_emb.sin())  # 转换为余弦和正弦
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction  # cu_seqlens必须在CPU上，因为NPU算子限制
        if is_npu() and isinstance(cu_seqlens, torch.Tensor):  # 如果是NPU且cu_seqlens是张量
            cu_seqlens = cu_seqlens.to("cpu")  # 转移到CPU
        attn_cu_seqlens = cu_seqlens  # 注意力cu_seqlens
        hidden_states = inputs_embeds  # 重新初始化隐藏状态

        for encoder_layer in self.layers:  # 遍历编码器层
            hidden_states = encoder_layer(  # 通过编码器层
                hidden_states,  # 隐藏状态
                cu_seqlens=attn_cu_seqlens,  # 累积序列长度
                rope_emb=rope_emb,  # 旋转位置编码
            )
        return hidden_states  # 返回隐藏状态


class SiglipVisionTransformer(nn.Module):  # SigLIP视觉Transformer模块

    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        embed_dim = config.hidden_size  # 嵌入维度

        self.embeddings = SiglipVisionEmbeddings(config)  # 视觉嵌入模块
        self.encoder = SiglipEncoder(  # 编码器模块
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("encoder", prefix),  # 添加前缀
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 后层归一化

    def forward(  # 前向传播函数，执行视觉Transformer计算
        self,
        pixel_values,  # 像素值
        interpolate_pos_encoding: Optional[bool] = False,  # 是否插值位置编码
        position_ids: Optional[torch.Tensor] = None,  # 位置ID
        height_position_ids: Optional[torch.Tensor] = None,  # 高度位置ID
        width_position_ids: Optional[torch.Tensor] = None,  # 宽度位置ID
        cu_seqlens: Optional[List[torch.Tensor]] = None,  # 累积序列长度
        image_grid_thw: Optional[  # 图像网格THW
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
    ) -> list[torch.Tensor]:

        hidden_states = self.embeddings(  # 通过嵌入层
            pixel_values,  # 像素值
            interpolate_pos_encoding=interpolate_pos_encoding,  # 插值位置编码
            position_ids=position_ids,  # 位置ID
            image_grid_thw=image_grid_thw,  # 图像网格THW
        )

        last_hidden_state = self.encoder(  # 通过编码器
            inputs_embeds=hidden_states,  # 输入嵌入
            cu_seqlens=cu_seqlens,  # 累积序列长度
            image_grid_thw=image_grid_thw,  # 图像网格THW
            height_position_ids=height_position_ids,  # 高度位置ID
            width_position_ids=width_position_ids,  # 宽度位置ID
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)  # 应用后层归一化

        sample_hidden_state = list()  # 样本隐藏状态列表
        if cu_seqlens is None:  # 如果cu_seqlens为空
            raise ValueError(  # 抛出值错误
                "cu_seqlens cannot be None for "  # cu_seqlens不能为空
                "SiglipVisionTransformer output processing."  # SiglipVisionTransformer输出处理
            )
        for i in range(cu_seqlens.shape[0] - 1):  # 遍历每个样本
            start = cu_seqlens[i]  # 起始索引
            end = cu_seqlens[i + 1]  # 结束索引
            tensor = last_hidden_state[:, start:end, :].squeeze(0)  # 获取当前样本的隐藏状态
            sample_hidden_state.append(tensor)  # 添加到列表

        return sample_hidden_state  # 返回样本隐藏状态列表


class SiglipVisionModel(nn.Module):  # SigLIP视觉模型
    config_class = "PaddleOCRVisionConfig"  # 配置类
    main_input_name = "pixel_values"  # 主输入名称

    def __init__(  # 初始化函数
        self,
        config,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 前缀
    ):
        super().__init__()  # 调用父类初始化

        self.vision_model = SiglipVisionTransformer(  # 视觉Transformer
            config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("vision_model", prefix),  # 添加前缀
        )
        self.quant_config = quant_config  # 保存量化配置

    @property
    def dtype(self) -> torch.dtype:  # 数据类型属性
        return self.vision_model.embeddings.patch_embedding.weight.dtype  # 返回补丁嵌入权重的数据类型

    @property
    def device(self) -> torch.device:  # 设备属性
        return self.vision_model.embeddings.patch_embedding.weight.device  # 返回补丁嵌入权重的设备

    def get_input_embeddings(self) -> nn.Module:  # 获取输入嵌入层
        return self.vision_model.embeddings.patch_embedding  # 返回补丁嵌入层

    def forward(  # 前向传播函数，执行视觉模型计算
        self,
        pixel_values,  # 像素值
        interpolate_pos_encoding: bool = False,  # 是否插值位置编码
        position_ids: Optional[torch.Tensor] = None,  # 位置ID
        image_grid_thw: Optional[  # 图像网格THW
            List[
                Union[
                    Tuple[int, int, int],
                    List[Tuple[int, int, int]],
                ]
            ]
        ] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,  # 累积序列长度
    ) -> list[torch.Tensor]:

        return self.vision_model(  # 通过视觉Transformer
            pixel_values=pixel_values,  # 像素值
            interpolate_pos_encoding=interpolate_pos_encoding,  # 插值位置编码
            position_ids=position_ids,  # 位置ID
            image_grid_thw=image_grid_thw,  # 图像网格THW
            cu_seqlens=cu_seqlens,  # 累积序列长度
        )


class PaddleOCRVLForConditionalGeneration(Ernie4_5_ForCausalLM):  # PaddleOCR视觉语言条件生成模型，继承自Ernie4.5

    def __init__(self, *, config, quant_config=None, prefix: str = ""):  # 初始化函数
        super().__init__(config=config, prefix=prefix)  # 调用父类初始化
        config = self.config  # 获取配置

        self.mlp_AR = Projector(  # 投影器模块
            config, config.vision_config, prefix=add_prefix("mlp_AR", prefix)  # 传入配置和视觉配置
        )
        self.visual = SiglipVisionModel(  # 视觉模型
            config=config.vision_config, prefix=add_prefix("visual", prefix)  # 传入视觉配置
        )
        if not hasattr(self.model, "get_input_embeddings"):  # 如果模型没有获取输入嵌入的方法
            import types  # 导入types模块

            self.model.get_input_embeddings = types.MethodType(  # 动态添加方法
                get_input_embeddings, self.model  # 绑定到模型
            )
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling  # 是否启用多维旋转位置编码

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入ID函数
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 填充输入token

    def get_input_embeddings(self):  # 获取输入嵌入层
        return self.model.embed_tokens  # 返回词表嵌入层

    def encode_image(self, pixel_values, image_grid_thw):  # 编码图像函数，将像素值转换为图像嵌入
        pixel_values = pixel_values.type(self.visual.dtype)  # 转换像素值数据类型
        siglip_position_ids = list()  # SigLIP位置ID列表
        image_grid_hws = list()  # 图像网格HWS列表
        cu_seqlens = [0]  # 累积序列长度，初始化为0

        for idx, grid_thw in enumerate(image_grid_thw):  # 遍历每个图像网格
            thw_tuple = tuple(grid_thw.detach().cpu().numpy().tolist())  # 转换为元组
            numel = np.prod(thw_tuple)  # 计算元素总数
            image_grid_hws.append(thw_tuple)  # 添加到网格列表
            image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])  # 计算图像位置ID
            siglip_position_ids.append(image_position_ids)  # 添加到位置ID列表
            cu_seqlens.append(cu_seqlens[-1] + numel)  # 更新累积序列长度

        siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(  # 拼接位置ID
            pixel_values.device  # 转移到像素值设备
        )
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)  # 转换为张量
        vision_outputs = self.visual(  # 通过视觉模型
            pixel_values=pixel_values,  # 像素值
            image_grid_thw=image_grid_hws,  # 图像网格HWS
            position_ids=siglip_position_ids,  # 位置ID
            interpolate_pos_encoding=True,  # 启用位置编码插值
            cu_seqlens=cu_seqlens,  # 累积序列长度
        )
        image_embeds = self.mlp_AR(vision_outputs, image_grid_thw)  # 通过投影器

        # image_embeds = torch.stack(image_embeds, dim=0)  # 图像嵌入堆叠（已注释）
        image_embeds = torch.cat(image_embeds, dim=0)  # 拼接图像嵌入

        return image_embeds  # 返回图像嵌入

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:  # 获取图像特征函数
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接并转换像素值
            self.visual.dtype  # 视觉模型数据类型
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)  # 拼接图像网格THW
        image_embeds = self.encode_image(pixel_values, image_grid_thw)  # 编码图像

        return image_embeds  # 返回图像嵌入

    def forward(  # 前向传播函数，执行条件生成模型计算
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置
        forward_batch: ForwardBatch,  # 前向批次信息
        get_embedding: bool = False,  # 是否获取嵌入
    ):
        if self.is_mrope_enabled:  # 如果启用了多维旋转位置编码
            positions = forward_batch.mrope_positions  # 使用多维旋转位置
        if not (  # 如果不是
            forward_batch.forward_mode.is_decode()  # 解码模式
            or not forward_batch.contains_image_inputs()  # 或不包含图像输入
        ):
            if self.is_mrope_enabled:  # 如果启用了多维旋转位置编码
                assert positions.ndim == 2 and positions.size(0) == 3, (  # 断言位置维度
                    "multimodal section rotary embedding requires "  # 多模态旋转位置编码需要
                    f"(3, seq_len) positions, but got {positions.size()}"  # (3, seq_len)维度
                )

        hidden_states = general_mm_embed_routine(  # 通过通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.model,  # 语言模型
            multimodal_model=self,  # 多模态模型
            positions=positions,  # 位置
        )

        return self.logits_processor(  # 通过logits处理器
            input_ids, hidden_states, self.lm_head, forward_batch  # 传入参数
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:  # 加载权重函数
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, weight_name, shard_id)  # (参数名, 权重名, 分片ID)
            (".qkv_proj", ".q_proj", "q"),  # Q投影映射
            (".qkv_proj", ".k_proj", "k"),  # K投影映射
            (".qkv_proj", ".v_proj", "v"),  # V投影映射
            (".gate_up_proj", ".gate_proj", 0),  # 门控投影映射
            (".gate_up_proj", ".up_proj", 1),  # 上投影映射
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典
        for name, loaded_weight in weights:  # 遍历权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转位置编码逆频率
                continue  # 跳过
            if "head.attention" in name or "head.layernorm" in name:  # 如果是头部注意力或层归一化
                continue  # 跳过
            if "head.mlp" in name or "head.probe" in name:  # 如果是头部MLP或探针
                continue  # 跳过

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue  # 继续
                name = name.replace(weight_name, param_name)  # 替换权重名
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出循环
            else:  # 如果没有匹配的堆叠参数
                if "vision_model" in name and "out_proj" in name:  # 如果是视觉模型输出投影
                    # adapt to VisionAttention  # 适配视觉注意力
                    name = name.replace(".self_attn.out_proj", ".self_attn.proj")  # 替换名称
                if name in params_dict.keys():  # 如果参数名在字典中
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认权重加载器
                    )
                    weight_loader(param, loaded_weight)  # 加载权重
                else:  # 否则
                    raise KeyError(f"Parameter '{name}' not found in model.")  # 抛出键错误


# monkey patch  # 猴子补丁
def get_input_embeddings(self) -> nn.Embedding:  # 获取输入嵌入的猴子补丁函数
    return self.embed_tokens  # 返回词表嵌入


EntryClass = [PaddleOCRVLForConditionalGeneration]  # 入口类列表
