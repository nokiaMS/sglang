# Step3 VL 10B 视觉语言模型实现
# 本文件实现了 Step3 VL 10B 视觉语言模型，包含感知编码器（PerceptionEncoder）和视觉语言条件生成模型。
# 主要组件包括：2D旋转位置编码（RoPE2D）、感知编码器层、视觉变换器、图像特征提取与投影等。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This is basically a copy from perception_models/core/vision_encoder/pe.py"""

from functools import partial  # 导入partial用于创建带默认参数的函数
from typing import Callable, Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch核心库
from einops import rearrange, repeat  # 导入张量重排工具
from torch import nn  # 导入神经网络模块
from torch.nn import functional as F  # 导入神经网络函数式接口
from transformers.activations import ACT2FN  # 导入激活函数映射表

from sglang.srt.configs.step3_vl import Step3VLConfig  # 导入Step3 VL配置类
from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.conv import Conv2dLayer  # 导入2D卷积层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear  # 导入并行线性层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具函数
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向传播批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen3 import Qwen3ForCausalLM  # 导入Qwen3语言模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具

_DEFAULT_NORM_LAYER = partial(nn.LayerNorm, eps=1e-5)  # 默认归一化层，使用LayerNorm且eps=1e-5


def rotate_half(x):
    """对输入张量的最后一维进行半旋转操作，用于旋转位置编码"""
    x = rearrange(x, "... (d r) -> ... d r", r=2)  # 将最后一维拆分为d和r=2两个维度
    x1, x2 = x.unbind(dim=-1)  # 在最后一维上分离为x1和x2
    x = torch.stack((-x2, x1), dim=-1)  # 堆叠为[-x2, x1]，实现旋转
    return rearrange(x, "... d r -> ... (d r)")  # 将最后两个维度合并回一维


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    """应用旋转位置编码到输入张量上"""
    dtype = t.dtype  # 保存原始数据类型

    if t.ndim == 3:  # 如果输入是3维的
        seq_len = t.shape[seq_dim]  # 获取序列长度
        freqs = freqs[-seq_len:]  # 截取对应长度的频率

    rot_dim = freqs.shape[-1]  # 旋转维度大小
    end_index = start_index + rot_dim  # 旋转结束索引

    assert rot_dim <= t.shape[-1], (  # 断言旋转维度不超过特征维度
        "feature dimension {} is not of sufficient size to rotate in all the "
        "positions {}".format(t.shape[-1], rot_dim)
    )

    t_left, t, t_right = (  # 将张量分为左、中、右三部分
        t[..., :start_index],  # 旋转前的部分
        t[..., start_index:end_index],  # 需要旋转的部分
        t[..., end_index:],  # 旋转后的部分
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)  # 应用旋转位置编码公式
    out = torch.cat((t_left, t, t_right), dim=-1)  # 拼接回完整张量

    return out.type(dtype)  # 转换回原始数据类型


class PerceptionEncoderRope2D(nn.Module):
    """2D旋转位置编码模块，用于视觉编码器中的2D位置编码"""

    def __init__(
        self,
        dim: int,  # 嵌入维度
        max_grid_height: int,  # 最大网格高度
        max_grid_width: int,  # 最大网格宽度
        use_cls_token: bool = False,  # 是否使用分类token
        theta=10000,  # 旋转角度基数
        max_freq=10,  # 最大频率
        num_freqs=1,  # 频率数量
        theta_rescale_factor=1.0,  # theta缩放因子
    ):
        super().__init__()  # 调用父类初始化
        self.dim = dim  # 保存维度
        self.max_grid_height = max_grid_height  # 保存最大网格高度
        self.max_grid_width = max_grid_width  # 保存最大网格宽度
        self.use_cls_token = use_cls_token  # 保存是否使用cls token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))  # 计算缩放后的theta
        self.max_freq = max_freq  # 保存最大频率
        self.num_freqs = num_freqs  # 保存频率数量
        cache = self._compute_2d_freqs()  # 预计算2D频率
        self.register_buffer("freqs_cache", cache, persistent=False)  # 注册频率缓存为buffer

    def _compute_inv_freq(self, base: int | float, dim: int) -> torch.Tensor:
        """计算逆频率，用于旋转位置编码"""
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 计算逆频率公式
        return freqs  # 返回逆频率

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor):
        """根据位置和逆频率计算频率"""
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)  # 外积计算频率
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)  # 重复频率以匹配旋转编码维度
        return freqs  # 返回频率

    def _compute_2d_freqs(self) -> torch.Tensor:
        """计算2D旋转位置编码的频率缓存"""
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float)  # 高度方向位置范围
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float)  # 宽度方向位置范围
        if self.use_cls_token:  # 如果使用cls token
            grid_h_range += 1  # 高度范围偏移1
            grid_w_range += 1  # 宽度范围偏移1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)  # 计算逆频率
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(  # 计算高度方向频率并扩展
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(  # 计算宽度方向频率并扩展
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(  # 拼接宽度和高度频率并重塑
            self.max_grid_height * self.max_grid_width, -1
        )
        if self.use_cls_token:  # 如果使用cls token
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)  # 在前面添加全零频率
        freqs = freqs[None, None, ...]  # 添加batch和head维度
        return freqs  # 返回2D频率

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int], x_shape
    ):
        """对查询和键张量应用2D旋转位置编码"""
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:  # 如果网格大小不匹配
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)  # 行索引
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)  # 列索引
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)  # 计算扁平化位置
            if self.use_cls_token:  # 如果使用cls token
                positions = torch.cat(  # 在位置前添加cls token位置
                    [torch.zeros(1, device=q.device), positions + 1], dim=0
                )
                positions = positions.to(torch.long)  # 转换为长整型
            freqs = self.freqs_cache.index_select(2, positions)  # 根据位置索引选择频率
        else:
            freqs = self.freqs_cache  # 直接使用缓存的频率
        ori_shape = q.shape  # 保存原始形状
        bs, seq_len, _ = x_shape  # 解包批次大小和序列长度
        q = q.view(bs, seq_len, -1, self.dim).permute(0, 2, 1, 3)  # 重塑并转置q
        k = k.view(bs, seq_len, -1, self.dim).permute(0, 2, 1, 3)  # 重塑并转置k
        q = apply_rotary_emb(freqs, q)  # 对q应用旋转位置编码
        k = apply_rotary_emb(freqs, k)  # 对k应用旋转位置编码
        q = q.permute(0, 2, 1, 3).reshape(ori_shape)  # 恢复q的原始形状
        k = k.permute(0, 2, 1, 3).reshape(ori_shape)  # 恢复k的原始形状
        return q, k  # 返回编码后的q和k


class PerceptionEncoderLayerScale(nn.Module):
    """感知编码器层缩放模块，用于对层输出进行可学习的缩放"""

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()  # 调用父类初始化
        self.inplace = inplace  # 是否原地操作
        self.gamma = nn.Parameter(init_values * torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        """对输入张量进行缩放"""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma  # 原地或非原地缩放


class PerceptionEncoderMLP(nn.Module):
    """感知编码器的多层感知机（MLP）模块"""

    def __init__(
        self,
        input_dim: int,  # 输入维度
        hidden_dim: int,  # 隐藏层维度
        act_layer: Callable[[], nn.Module],  # 激活函数
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.fc1 = ColumnParallelLinear(  # 第一个全连接层（列并行）
            input_dim,  # 输入维度
            hidden_dim,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.fc1",  # 参数前缀
        )
        self.activation = act_layer  # 激活函数
        self.fc2 = RowParallelLinear(  # 第二个全连接层（行并行）
            hidden_dim,  # 输入维度
            input_dim,  # 输出维度
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.fc2",  # 参数前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP前向传播：fc1 -> 激活 -> fc2"""
        x, _ = self.fc1(x)  # 通过第一个全连接层
        x = self.activation(x)  # 应用激活函数
        x, _ = self.fc2(x)  # 通过第二个全连接层
        return x  # 返回输出


class PerceptionEncoderVisionBlock(nn.Module):
    """感知编码器的视觉块，包含自注意力和MLP"""

    def __init__(
        self,
        d_model: int,  # 模型维度
        n_head: int,  # 注意力头数
        max_grid_height: int,  # 最大网格高度
        max_grid_width: int,  # 最大网格宽度
        mlp_ratio: float = 4.0,  # MLP隐藏层维度比率
        ls_init_value: float = None,  # 层缩放初始值
        act_layer: Callable = nn.GELU,  # 激活函数
        norm_layer: Callable = nn.LayerNorm,  # 归一化层
        use_cls_token: bool = False,  # 是否使用cls token
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.head_dim = d_model // n_head  # 每个注意力头的维度
        self.rope = PerceptionEncoderRope2D(  # 2D旋转位置编码
            dim=self.head_dim,  # 头维度
            max_grid_height=max_grid_height,  # 最大网格高度
            max_grid_width=max_grid_width,  # 最大网格宽度
            use_cls_token=use_cls_token,  # 是否使用cls token
        )
        self.attn = VisionAttention(  # 视觉注意力层
            embed_dim=d_model,  # 嵌入维度
            num_heads=n_head,  # 注意力头数
            projection_size=d_model,  # 投影大小
            use_qkv_parallel=True,  # 使用QKV并行
            proj_bias=True,  # 投影使用偏置
            # flatten_batch=True,
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 参数前缀
            customized_position_embedding_applier=self.rope,  # 自定义位置编码应用器
        )
        self.ls_1 = (  # 注意力输出的层缩放
            PerceptionEncoderLayerScale(d_model, ls_init_value)  # 使用指定初始值
            if ls_init_value is not None  # 如果有初始值
            else nn.Identity()  # 否则使用恒等映射
        )
        self.ls_2 = (  # MLP输出的层缩放
            PerceptionEncoderLayerScale(d_model, ls_init_value)  # 使用指定初始值
            if ls_init_value is not None  # 如果有初始值
            else nn.Identity()  # 否则使用恒等映射
        )
        self.ln_1 = norm_layer(d_model)  # 注意力前的层归一化
        self.ln_2 = norm_layer(d_model)  # MLP前的层归一化
        hidden_dim = int(d_model * mlp_ratio)  # 计算MLP隐藏层维度
        self.mlp = PerceptionEncoderMLP(  # MLP模块
            d_model,  # 输入维度
            hidden_dim,  # 隐藏维度
            act_layer,  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.mlp",  # 参数前缀
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]):
        """视觉块前向传播：注意力残差连接 + MLP残差连接"""
        x = x + self.ls_1(self.attn(self.ln_1(x), position_embeddings=grid_hw))  # hacky # 自注意力残差连接
        x = x + self.ls_2(self.mlp(self.ln_2(x)))  # MLP残差连接
        return x  # 返回输出


class PerceptionEncoderVisionTransformer(nn.Module):
    """感知编码器的视觉变换器，由多个视觉块组成"""

    def __init__(
        self,
        width: int,  # 模型宽度
        layers: int,  # 层数
        heads: int,  # 注意力头数
        max_grid_height: int,  # 最大网格高度
        max_grid_width: int,  # 最大网格宽度
        mlp_ratio: float = 4.0,  # MLP比率
        ls_init_value: float = None,  # 层缩放初始值
        act_layer: Callable = nn.GELU,  # 激活函数
        norm_layer: Callable = nn.LayerNorm,  # 归一化层
        use_cls_token: bool = False,  # 是否使用cls token
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.width = width  # 保存模型宽度
        self.layers = layers  # 保存层数
        self.resblocks = nn.ModuleList(  # 残差块列表
            [
                PerceptionEncoderVisionBlock(  # 每个视觉块
                    d_model=width,  # 模型维度
                    n_head=heads,  # 注意力头数
                    max_grid_height=max_grid_height,  # 最大网格高度
                    max_grid_width=max_grid_width,  # 最大网格宽度
                    mlp_ratio=mlp_ratio,  # MLP比率
                    ls_init_value=ls_init_value,  # 层缩放初始值
                    act_layer=act_layer,  # 激活函数
                    norm_layer=norm_layer,  # 归一化层
                    use_cls_token=use_cls_token,  # 是否使用cls token
                    quant_config=quant_config,  # 量化配置
                    prefix=f"{prefix}.resblocks.{i}",  # 参数前缀
                )
                for i in range(layers)  # 遍历每一层
            ]
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]):
        """视觉变换器前向传播：依次通过所有残差块"""
        for block in self.resblocks:  # 遍历所有残差块
            x = block(x, grid_hw=grid_hw)  # 通过当前块
        return x  # 返回最终输出


class PerceptionEncoder(nn.Module):
    """感知编码器，包含卷积patch嵌入、视觉变换器和下采样器"""

    def __init__(
        self,
        config,  # 编码器配置
        act_layer: Callable,  # 激活函数
        norm_layer: Callable = _DEFAULT_NORM_LAYER,  # 归一化层
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化
        self.patch_size = config.patch_size  # patch大小

        self.output_dim = config.output_dim or config.width  # 输出维度
        self.heads = config.heads  # 注意力头数
        self.width = config.width  # 模型宽度
        self.layers = config.layers  # 层数

        self.use_abs_posemb = config.use_abs_posemb  # 是否使用绝对位置嵌入
        self.use_cls_token = config.use_cls_token  # 是否使用cls token
        self.use_rope2d = config.use_rope2d  # 是否使用2D旋转位置编码
        if not self.use_rope2d:  # 如果不使用2D旋转位置编码
            raise ValueError("use_rope2d must be True")  # 抛出异常
        self.image_size = config.image_size  # 图像大小

        self.conv1 = Conv2dLayer(  # 第一个卷积层（patch嵌入）
            in_channels=3,  # 输入通道数（RGB）
            out_channels=config.width,  # 输出通道数
            kernel_size=config.patch_size,  # 卷积核大小等于patch大小
            stride=config.patch_size,  # 步长等于patch大小
            bias=False,  # 不使用偏置
        )

        self.ln_pre = norm_layer(config.width) if config.use_ln_pre else nn.Identity()  # 前置归一化
        self.ln_post = norm_layer(self.width) if config.use_ln_post else nn.Identity()  # 后置归一化

        self.transformer = PerceptionEncoderVisionTransformer(  # 视觉变换器
            config.width,  # 宽度
            config.layers,  # 层数
            config.heads,  # 头数
            max_grid_height=self.image_size // self.patch_size,  # 最大网格高度
            max_grid_width=self.image_size // self.patch_size,  # 最大网格宽度
            mlp_ratio=config.mlp_ratio,  # MLP比率
            ls_init_value=config.ls_init_value,  # 层缩放初始值
            act_layer=act_layer,  # 激活函数
            norm_layer=norm_layer,  # 归一化层
            use_cls_token=self.use_cls_token,  # 是否使用cls token
            quant_config=quant_config,  # 量化配置
            prefix=f"{prefix}.transformer",  # 参数前缀
        )

        self.vit_downsampler1 = nn.Conv2d(  # 第一个下采样卷积
            config.width, config.width * 2, kernel_size=3, stride=2, padding=1  # 宽度翻倍，空间减半
        )
        self.vit_downsampler2 = nn.Conv2d(  # 第二个下采样卷积
            config.width * 2, config.width * 4, kernel_size=3, stride=2, padding=1  # 宽度再翻倍，空间再减半
        )

        if self.use_cls_token:  # 如果使用cls token
            self.class_embedding = nn.Parameter(  # 分类嵌入参数
                (self.width**-0.5) * torch.randn(self.width)  # 随机初始化并缩放
            )

        if self.use_abs_posemb:  # 如果使用绝对位置嵌入
            self.posemb_grid_size = self.image_size // self.patch_size  # 位置嵌入网格大小
            self.positional_embedding = nn.Parameter(  # 位置嵌入参数
                (self.width**-0.5)  # 缩放因子
                * torch.randn(  # 随机初始化
                    int(self.use_cls_token) + self.posemb_grid_size**2,  # 嵌入数量
                    self.width,  # 嵌入维度
                )
            )

    @property
    def dtype(self) -> torch.dtype:
        """获取编码器的数据类型"""
        return self.conv1.weight.dtype  # 返回卷积层权重的数据类型

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        """采样绝对位置嵌入，支持不同网格大小的插值"""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:  # 如果网格大小匹配
            return self.positional_embedding[None, ...]  # 直接返回位置嵌入

        pos_embed = self.positional_embedding  # 获取位置嵌入
        if self.use_cls_token:  # 如果使用cls token
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]  # 分离cls token嵌入

        pos_embed = (  # 重塑位置嵌入为2D网格
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)  # 调整维度顺序
            .contiguous()  # 确保内存连续
        )
        pos_embed = F.interpolate(  # 双线性插值到目标大小
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width)  # 恢复维度顺序并展平

        if self.use_cls_token:  # 如果使用cls token
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)  # 拼接cls token嵌入

        return pos_embed[None, ...]  # 添加batch维度并返回

    def forward_features(self, x: torch.Tensor):
        """提取图像特征：卷积嵌入 -> cls token -> 位置嵌入 -> 变换器 -> 归一化"""
        batch, _, h, w = x.shape  # 获取批次大小和图像尺寸
        grid_h, grid_w = h // self.patch_size, w // self.patch_size  # 计算网格大小

        x = self.conv1(x)  # 卷积patch嵌入
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)  # 展平为序列

        if self.use_cls_token:  # 如果使用cls token
            x = torch.cat(  # 在序列前添加cls token
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x], dim=1
            )

        if self.use_abs_posemb:  # 如果使用绝对位置嵌入
            x = x + self.sample_abs_posemb(grid_h, grid_w)  # 添加位置嵌入

        x = self.ln_pre(x)  # 前置归一化
        x = self.transformer(x, grid_hw=(grid_h, grid_w))  # 通过视觉变换器
        x = self.ln_post(x)  # 后置归一化

        if self.use_cls_token:  # 如果使用cls token
            x = x[:, 1:, :]  # 移除cls token

        return x  # 返回特征

    def forward(self, x: torch.Tensor):
        """感知编码器前向传播：提取特征后通过下采样器"""
        x = self.forward_features(x)  # 提取图像特征
        B, P, C = x.shape  # 获取批次大小、patch数、通道数
        T = int(P**0.5)  # 计算空间维度大小
        x = x.transpose(2, 1).contiguous()  # 转置
        x = x.view(B, C, T, T)  # 重塑为2D空间形状

        x = self.vit_downsampler1(x)  # 第一个下采样
        x = self.vit_downsampler2(x)  # 第二个下采样

        B, C, T, T = x.shape  # 获取下采样后的形状
        return x.view(B, -1, T * T).transpose(1, 2)  # 展平为序列格式


class StepVLForConditionalGeneration(nn.Module):
    """Step VL条件生成模型，包含视觉编码器、投影器和语言模型"""

    def __init__(
        self,
        config: Step3VLConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ):
        super().__init__()  # 调用父类初始化

        self.config = config  # 保存配置
        self.vision_model = PerceptionEncoder(  # 视觉编码器
            config.vision_config,  # 视觉配置
            ACT2FN[config.vision_config.hidden_act],  # 激活函数
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix(prefix, "vision_model"),  # 参数前缀
        )
        self.vit_large_projector = ColumnParallelLinear(  # 视觉到语言的投影层
            config.vision_config.width * 4,  # 输入维度（下采样后宽度x4）
            config.text_config.hidden_size,  # 输出维度（语言模型隐藏大小）
            bias=config.projector_bias,  # 是否使用偏置
            gather_output=True,  # 收集输出
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix(prefix, "vit_large_projector"),  # 参数前缀
        )

        self.language_model = Qwen3ForCausalLM(  # 语言模型（Qwen3）
            config=config.text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix(prefix, "language_model"),  # 参数前缀
        )

    def _get_vision_model_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """获取视觉模型输出"""
        return self.vision_model(input_tensor)  # 通过视觉编码器

    @property
    def device(self) -> torch.device:
        """获取模型所在设备"""
        return self.vit_large_projector.weight.device  # 返回投影层权重所在设备

    def _flatten_embeddings(self, embeddings) -> torch.Tensor:
        """将嵌入展平为一维序列"""
        if isinstance(embeddings, torch.Tensor):  # 如果是单个张量
            # Flatten all but the last dimension.
            return embeddings.flatten(0, -2)  # 展平除最后一维外的所有维度

        return torch.cat(tuple(self._flatten_embeddings(t) for t in embeddings))  # 递归展平并拼接

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """通过投影层处理图像特征"""
        image_features, _ = self.vit_large_projector(image_features)  # 通过投影层
        return image_features  # 返回投影后的特征

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """从多模态数据项中提取图像特征，支持高分辨率patch"""
        assert len(items) == 1  # We only have images. # 仅支持图像

        item = items[0]  # 获取第一个数据项
        pixel_values = item.feature.type(self.vision_model.dtype)  # 获取像素值并转换类型
        num_patches = item.model_specific_data.get("num_patches")  # 获取patch数量
        patch_pixel_values = item.model_specific_data.get("patch_pixel_values", None)  # 获取高分辨率patch像素值
        if patch_pixel_values is not None:  # 如果有高分辨率patch
            patch_pixel_values = patch_pixel_values.type(self.vision_model.dtype).to(  # 转换类型并移到设备
                self.device
            )

        image_features = self._get_vision_model_output(pixel_values)  # 获取基础图像特征
        patch_image_features = (  # 获取高分辨率patch特征
            self._get_vision_model_output(patch_pixel_values)  # 通过视觉编码器
            if patch_pixel_values is not None  # 如果有patch
            else None  # 否则为None
        )
        image_features = self._process_image_features(image_features)  # 投影基础图像特征
        patch_image_features = (  # 投影高分辨率patch特征
            self._process_image_features(patch_image_features)  # 通过投影层
            if patch_image_features is not None  # 如果有patch
            else None  # 否则为None
        )
        merged_image_features = []  # 合并后的图像特征列表
        cur_patch_idx = 0  # 当前patch索引
        for i, num_patch in enumerate(num_patches):  # 遍历每个图像的patch数
            cur_feature = []  # 当前图像的特征列表
            if num_patch > 0:  # 如果有高分辨率patch
                patch_slice = patch_image_features[  # 获取当前图像的patch特征
                    cur_patch_idx : cur_patch_idx + num_patch
                ]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))  # 展平patch特征
            cur_feature.append(image_features[i].view(-1, image_features.shape[-1]))  # 添加基础图像特征
            cur_patch_idx += num_patch  # 更新patch索引
            merged_image_features.append(  # 合并当前图像特征
                torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0]  # 拼接或直接使用
            )
        return self._flatten_embeddings(merged_image_features)  # 展平并返回

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """对输入ID进行多模态填充"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 执行填充

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向传播批次
        get_embedding: bool = False,  # 是否获取嵌入
    ):
        """模型前向传播：通过多模态嵌入例程处理图像和文本"""
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向传播批次
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.IMAGE: self.get_image_feature,  # 图像模态使用get_image_feature
            },
            positions=positions,  # 位置编码
        )

        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model, separating vision and language weights"""  # 加载模型权重，分离视觉和语言权重
        weights = list(weights)  # 将权重转为列表

        # Separate vision tower weights and language model weights
        vision_weights = []  # 视觉权重列表
        language_weights = []  # 语言权重列表

        for name, loaded_weight in weights:  # 遍历所有权重
            if "vision_model" in name or "vit_large_projector" in name:  # 如果是视觉相关权重
                name = name.replace(r".attn.in_proj_weight", r".attn.qkv_proj.weight")  # 重命名注意力投影权重
                name = name.replace(r".attn.in_proj_bias", r".attn.qkv_proj.bias")  # 重命名注意力投影偏置
                name = name.replace(r".attn.out_proj.bias", r".attn.proj.bias")  # 重命名注意力输出偏置
                name = name.replace(r".attn.out_proj.weight", r".attn.proj.weight")  # 重命名注意力输出权重
                name = name.replace(".mlp.c_fc", ".mlp.fc1")  # 重命名MLP第一层
                name = name.replace(".mlp.c_proj", ".mlp.fc2")  # 重命名MLP第二层
                vision_weights.append((name, loaded_weight))  # 添加到视觉权重列表
            else:
                # All other weights go to language model
                language_weights.append((name, loaded_weight))  # 添加到语言权重列表

        # Load vision tower weights
        vision_state_dict = dict(vision_weights)  # 将视觉权重转为字典
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取模型参数字典
        for name, loaded_weight in vision_state_dict.items():  # 遍历视觉权重
            if name not in params_dict:  # 如果参数名不存在
                raise ValueError(f"Weight {name} not found in params_dict")  # 抛出异常
            param = params_dict[name]  # 获取参数
            weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
            # loaded_weight = self._pad_vit_attn_dummy_heads(name, loaded_weight)
            weight_loader(param, loaded_weight)  # 加载权重

        # Load language model weights
        if language_weights:  # 如果有语言模型权重
            self.language_model.load_weights(language_weights)  # 加载语言模型权重


EntryClass = StepVLForConditionalGeneration  # 入口类
