# SigLIP视觉模型实现文件
# 本文件实现了SigLIP视觉模型，包括视觉嵌入、MLP、编码器层和编码器
# SigLIP是基于Transformer的视觉编码器，用于视觉-语言模型中的图像特征提取

# Adapted from
# https://github.com/huggingface/transformers/blob/af9b2eaa54c150741f298d6db939af6328e1dc38/src/transformers/models/siglip/modeling_siglip.py

from functools import partial  # 导入偏函数工具
from typing import Optional, Type, Union  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn as nn  # 导入PyTorch神经网络模块
from transformers import SiglipVisionConfig  # 导入SigLIP视觉配置

from sglang.srt.layers.activation import QuickGELU  # 导入QuickGELU激活函数
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.conv import Conv2dLayer  # 导入2D卷积层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear  # 导入并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding  # 导入词汇并行嵌入
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


# Adapted from transformers.models.siglip.modeling_siglip.SiglipVisionTransformer
# 改编自transformers的SiglipVisionTransformer
class SiglipVisionEmbeddings(nn.Module):  # SigLIP视觉嵌入类

    def __init__(self, config: SiglipVisionConfig):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.image_size = config.image_size  # 图像尺寸
        self.patch_size = config.patch_size  # 补丁尺寸

        self.patch_embedding = Conv2dLayer(  # 补丁嵌入卷积层
            in_channels=config.num_channels,  # 输入通道数
            out_channels=self.embed_dim,  # 输出通道数（嵌入维度）
            kernel_size=self.patch_size,  # 卷积核大小（补丁尺寸）
            stride=self.patch_size,  # 步长（补丁尺寸）
            padding="valid",  # 无填充
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 补丁数量
        self.num_positions = self.num_patches  # 位置数量等于补丁数量
        self.position_embedding = VocabParallelEmbedding(  # 位置嵌入
            self.num_positions, self.embed_dim  # 位置数量和嵌入维度
        )
        self.register_buffer(  # 注册缓冲区
            "position_ids",  # 位置ID
            torch.arange(self.num_positions).expand((1, -1)),  # 创建位置ID张量
            persistent=False,  # 不持久化
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        target_dtype = self.patch_embedding.weight.dtype  # 获取目标数据类型
        patch_embeds = self.patch_embedding(  # 通过补丁嵌入卷积层
            pixel_values.to(dtype=target_dtype)  # 转换为目标精度
        )  # shape = [*, width, grid, grid]  # 形状为[*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2).contiguous()  # 展平并转置
        # interpolate_pos_encoding is never used in sglang
        # interpolate_pos_encoding在sglang中从不使用
        embeddings = embeddings + self.position_embedding(self.position_ids)  # 加上位置嵌入

        return embeddings  # 返回嵌入


# Copied from sglang.srt.models.clip.CLIPMLP
# 复制自sglang的CLIPMLP
class SiglipMLP(nn.Module):  # SigLIP MLP类

    def __init__(  # 初始化方法
        self,
        config,  # 配置对象
        act_layer: Type[nn.Module] = QuickGELU,  # 激活层类型，默认QuickGELU
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.fc1 = ColumnParallelLinear(  # 第一个全连接层（列并行）
            config.hidden_size,  # 输入大小（隐藏维度）
            config.intermediate_size,  # 输出大小（中间维度）
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc1", prefix),  # 添加前缀
        )
        self.act = act_layer()  # 创建激活层实例
        self.fc2 = RowParallelLinear(  # 第二个全连接层（行并行）
            config.intermediate_size,  # 输入大小（中间维度）
            config.hidden_size,  # 输出大小（隐藏维度）
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc2", prefix),  # 添加前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播方法
        x_parallel, _ = self.fc1(x)  # 通过第一个全连接层
        x_parallel = self.act(x_parallel)  # 通过激活层
        x, _ = self.fc2(x_parallel)  # 通过第二个全连接层
        return x  # 返回输出


# Copied from sglang.srt.models.clip.CLIPEncoderLayer
# 复制自sglang的CLIPEncoderLayer
class SiglipEncoderLayer(nn.Module):  # SigLIP编码器层类

    def __init__(  # 初始化方法
        self,
        config: SiglipVisionConfig,  # SigLIP视觉配置
        act_layer: Type[nn.Module] = QuickGELU,  # 激活层类型
        norm_layer: Type[nn.Module] = None,  # 归一化层类型，可选
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化
        if norm_layer is None:  # 如果没有指定归一化层
            norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)  # 使用默认LayerNorm
        self.layer_norm1 = norm_layer(config.hidden_size)  # 第一个LayerNorm
        self.layer_norm2 = norm_layer(config.hidden_size)  # 第二个LayerNorm
        self.self_attn = VisionAttention(  # 自注意力层
            embed_dim=config.hidden_size,  # 嵌入维度
            num_heads=config.num_attention_heads,  # 注意力头数
            projection_size=config.hidden_size,  # 投影大小
            use_qkv_parallel=True,  # 使用QKV并行
            flatten_batch=True,  # 展平批次
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("self_attn", prefix),  # 添加前缀
        )
        self.mlp = SiglipMLP(  # MLP层
            config,  # 配置对象
            act_layer=act_layer,  # 激活层
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 添加前缀
        )

    def forward(  # 前向传播方法
        self,
        hidden_states: torch.Tensor,  # 隐藏状态
        attention_mask: torch.Tensor,  # 注意力掩码
        causal_attention_mask: torch.Tensor,  # 因果注意力掩码
    ) -> torch.Tensor:

        residual = hidden_states  # 保存残差
        hidden_states = self.layer_norm1(hidden_states)  # 通过第一个LayerNorm
        # Siglip text model uses both `causal_attention_mask` and `attention_mask`
        # Siglip文本模型同时使用causal_attention_mask和attention_mask
        if attention_mask is not None and causal_attention_mask is not None:  # 如果两个掩码都存在
            attn_mask = attention_mask + causal_attention_mask  # 合并掩码
        elif causal_attention_mask is not None:  # 如果只有因果掩码
            attn_mask = causal_attention_mask  # 使用因果掩码
        else:  # 否则
            attn_mask = attention_mask  # 使用注意力掩码
        hidden_states = self.self_attn(  # 通过自注意力层
            hidden_states,  # 隐藏状态
            attention_mask=attn_mask,  # 注意力掩码
            # causal_attention_mask=causal_attention_mask,
        )

        hidden_states = residual + hidden_states  # 残差连接
        residual = hidden_states  # 更新残差
        hidden_states = self.layer_norm2(hidden_states)  # 通过第二个LayerNorm
        hidden_states = self.mlp(hidden_states)  # 通过MLP层
        hidden_states = residual + hidden_states  # 残差连接
        return hidden_states  # 返回隐藏状态


# Copied from sglang.srt.models.clip.CLIPEncoder
# 复制自sglang的CLIPEncoder
class SiglipEncoder(nn.Module):  # SigLIP编码器类
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """
    # 由config.num_hidden_layers个自注意力层组成的Transformer编码器，每层为SiglipEncoderLayer

    def __init__(  # 初始化方法
        self,
        config: SiglipVisionConfig,  # SigLIP视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置

        num_hidden_layers = config.num_hidden_layers  # 隐藏层数量
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)  # 归一化层
        self.layers = nn.ModuleList(  # 创建模块列表
            [
                SiglipEncoderLayer(  # 创建编码器层
                    config=config,  # 配置
                    norm_layer=norm_layer,  # 归一化层
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),  # 添加前缀
                )
                for layer_idx in range(num_hidden_layers)  # 遍历每层
            ]
        )

    def forward(  # 前向传播方法
        self,
        inputs_embeds: torch.Tensor,  # 输入嵌入
        attention_mask: torch.Tensor = None,  # 注意力掩码，可选
        causal_attention_mask: torch.Tensor = None,  # 因果注意力掩码，可选
        return_all_hidden_states: bool = False,  # 是否返回所有隐藏状态
    ) -> Union[torch.Tensor, list[torch.Tensor]]:  # 返回张量或张量列表
        hidden_states_pool = [inputs_embeds]  # 隐藏状态池，初始化为输入嵌入
        hidden_states = inputs_embeds  # 初始化隐藏状态

        for encoder_layer in self.layers:  # 遍历编码器层
            hidden_states = encoder_layer(  # 通过编码器层
                hidden_states, attention_mask, causal_attention_mask  # 传入参数
            )
            if return_all_hidden_states:  # 如果需要返回所有隐藏状态
                hidden_states_pool.append(hidden_states)  # 添加到隐藏状态池
        if return_all_hidden_states:  # 如果需要返回所有隐藏状态
            return hidden_states_pool  # 返回隐藏状态池
        return hidden_states  # 返回最终隐藏状态


# Adapted from transformers.models.siglip.modeling_siglip.SiglipVisionTransformer
# 改编自transformers的SiglipVisionTransformer
class SiglipVisionTransformer(nn.Module):  # SigLIP视觉变换器类

    def __init__(  # 初始化方法
        self,
        config: SiglipVisionConfig,  # SigLIP视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ) -> None:
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        embed_dim = config.hidden_size  # 嵌入维度

        self.embeddings = SiglipVisionEmbeddings(config)  # 创建视觉嵌入层

        self.encoder = SiglipEncoder(  # 创建编码器
            config=config,  # 配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("encoder", prefix),  # 添加前缀
        )

        num_hidden_layers = config.num_hidden_layers  # 隐藏层数量
        if len(self.encoder.layers) > config.num_hidden_layers:  # 如果编码器层数超过配置
            raise ValueError(  # 抛出异常
                f"The original encoder only has {num_hidden_layers} "  # 原始编码器层数
                f"layers, but you requested {len(self.encoder.layers)} layers."  # 请求的层数
            )

        # VisionAttention in SiglipEncoderLayer is multihead attention
        # SiglipEncoderLayer中的VisionAttention是多头注意力
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 后LayerNorm

    @property  # 属性装饰器
    def device(self) -> torch.device:  # 设备属性
        return self.encoder.layers[0].layer_norm1.weight.device  # 返回第一层归一化权重所在设备

    def forward(  # 前向传播方法
        self,
        pixel_values: torch.Tensor,  # 像素值
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values.to(self.device))  # 通过嵌入层

        return_all_hidden_states = False  # 不返回所有隐藏状态

        last_hidden_state = self.encoder(  # 通过编码器
            inputs_embeds=hidden_states,  # 输入嵌入
            return_all_hidden_states=return_all_hidden_states,  # 是否返回所有隐藏状态
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)  # 通过后LayerNorm

        return last_hidden_state  # 返回最终隐藏状态


# Copied from sglang.srt.models.clip.CLIPVisionModel
# 复制自sglang的CLIPVisionModel
class SiglipVisionModel(nn.Module):  # SigLIP视觉模型类
    def __init__(  # 初始化方法
        self,
        config: SiglipVisionConfig,  # SigLIP视觉配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 前缀字符串
    ):
        super().__init__()  # 调用父类初始化
        self.vision_model = SiglipVisionTransformer(  # 创建视觉变换器
            config, quant_config, prefix=add_prefix("vision_model", prefix)  # 传入配置和前缀
        )

    @property  # 属性装饰器
    def device(self) -> torch.device:  # 设备属性
        return self.vision_model.device  # 返回视觉变换器的设备

    def forward(self, pixel_values: torch.Tensor):  # 前向传播方法
        return self.vision_model(pixel_values)  # 通过视觉变换器
