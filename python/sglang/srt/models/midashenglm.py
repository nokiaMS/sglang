# MiDashengLM模型：基于达声(Dasheng)音频编码器和Qwen2语言模型的多模态音频-语言模型
# 该模型实现音频输入的编码、投影，并与语言模型进行融合推理
# 主要组件：音频前端(DashengFrontend)、音频Transformer编码器、音频投影器、Qwen2语言模型

import collections  # 导入collections模块，提供集合抽象基类
import collections.abc  # 导入collections.abc模块，提供集合抽象基类
import logging  # 导入logging模块，用于日志记录
from collections.abc import Callable, Sequence  # 从collections.abc导入Callable和Sequence类型
from typing import Iterable, List, Optional, Tuple, TypeAlias, cast  # 从typing导入类型注解

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torchaudio.functional as F  # 导入torchaudio功能模块
from transformers import PretrainedConfig  # 从transformers导入预训练配置类

from sglang.srt.layers.attention.vision import VisionAttention  # 导入视觉注意力层
from sglang.srt.layers.conv import Conv2dLayer  # 导入2D卷积层
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear  # 导入列并行和行并行线性层
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入Qwen2因果语言模型
from sglang.srt.utils import add_prefix  # 导入前缀添加工具函数

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器
_Tuple2: TypeAlias = int | tuple[int, int] | Sequence[int]  # 定义_Tuple2类型别名，表示整数或二元组或整数序列


def _resolve_tuple2(x: _Tuple2) -> tuple[int, int]:
    """将_Tuple2类型解析为二元组，如果输入是整数则复制为(整数, 整数)"""
    if isinstance(x, collections.abc.Sequence):  # 如果x是序列类型
        assert (
            len(x) == 2
        ), f"Expected a sequence of length 2, got {x} with length {len(x)}"  # 断言序列长度为2
        return cast(tuple[int, int], tuple(x))  # 将序列转换为二元组并返回
    return (x, x)  # 如果是整数，返回(x, x)二元组


def calculate_mel_frames_dasheng(
    audio_length_samples: int,
    n_fft: int = 512,
    hop_size: int = 160,
    dasheng_subsampling: int = 4,
    center=True,
    model_subsampling: int = 5,
) -> int:
    """计算达声模型的梅尔频谱帧数"""
    """Calculate the number of Mel-spectrogram frames."""
    if center:  # 如果使用中心填充
        audio_length_samples = audio_length_samples + n_fft  # 加上n_fft的填充长度

    return (  # 返回最终帧数
        int(1 + ((audio_length_samples - n_fft) / hop_size))  # 计算STFT帧数
        // dasheng_subsampling  # 除以达声子采样率
        // model_subsampling  # 除以模型子采样率
    )


class AudioPatchEmbed(nn.Module):
    """音频补丁嵌入层，将音频频谱图分割为补丁并投影到嵌入空间"""
    def __init__(
        self,
        input_size: _Tuple2 = 64,
        patch_size: _Tuple2 = 16,
        patch_stride: _Tuple2 = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten: bool = False,
    ):
        super().__init__()  # 调用父类初始化
        self.input_size = _resolve_tuple2(input_size)  # 解析输入尺寸为二元组
        self.patch_size = _resolve_tuple2(patch_size)  # 解析补丁尺寸为二元组
        self.patch_stride = _resolve_tuple2(patch_stride)  # 解析补丁步幅为二元组
        self.grid_size = (  # 计算网格尺寸
            self.input_size[0] // self.patch_stride[0],  # 高度方向的网格数
            self.input_size[1] // self.patch_stride[1],  # 宽度方向的网格数
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 补丁总数
        self.flatten = flatten  # 是否展平补丁
        self.proj = Conv2dLayer(  # 创建2D卷积投影层
            in_chans,  # 输入通道数
            embed_dim,  # 输出嵌入维度
            kernel_size=self.patch_size,  # 卷积核尺寸等于补丁尺寸
            stride=self.patch_stride,  # 步幅等于补丁步幅
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 归一化层或恒等映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：对输入进行卷积投影和归一化"""
        x = self.proj(x)  # 卷积投影
        if self.flatten:  # 如果需要展平
            x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))  # 展平空间维度并重排
        x = self.norm(x)  # 归一化
        return x  # 返回嵌入结果


class LayerScale(nn.Module):
    """层缩放模块，对输入进行可学习的逐通道缩放"""
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()  # 调用父类初始化
        self.inplace = inplace  # 是否原地操作
        self.gamma = nn.Parameter(init_values * torch.ones(dim))  # 可学习缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：对输入进行逐通道缩放"""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma  # 原地或非原地缩放


class DashengMlp(nn.Module):
    """达声模型的多层感知机(MLP)模块"""
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()  # 调用父类初始化
        out_features = out_features or in_features  # 输出特征数默认等于输入特征数
        hidden_features = hidden_features or in_features  # 隐藏特征数默认等于输入特征数
        self.fc1 = ColumnParallelLinear(  # 第一个全连接层（列并行）
            input_size=in_features,  # 输入大小
            output_size=hidden_features,  # 输出大小
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc1", prefix),  # 参数名前缀
        )
        self.act = nn.GELU()  # GELU激活函数
        self.fc2 = RowParallelLinear(  # 第二个全连接层（行并行）
            input_size=hidden_features,  # 输入大小
            output_size=out_features,  # 输出大小
            bias=True,  # 使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("fc2", prefix),  # 参数名前缀
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：两层全连接加激活函数"""
        x, _ = self.fc1(x)  # 第一个全连接层
        x = self.act(x)  # 激活函数
        x, _ = self.fc2(x)  # 第二个全连接层
        return x  # 返回输出


class DashengAttention(nn.Module):
    """音频编码器注意力层，使用VisionAttention以兼容"""
    """Audio encoder attention using VisionAttention for compatibility."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()  # 调用父类初始化
        assert dim % num_heads == 0, "dim should be divisible by num_heads"  # 断言维度可被头数整除
        self.embed_dim = dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = self.embed_dim // self.num_heads  # 每个头的维度
        self.scale = self.head_dim**-0.5  # 缩放因子

        self.attn = VisionAttention(  # 使用视觉注意力层实现音频注意力
            embed_dim=dim,  # 嵌入维度
            num_heads=num_heads,  # 头数
            projection_size=dim,  # 投影尺寸
            use_qkv_parallel=True,  # 使用QKV并行
            proj_bias=True,  # 投影偏置
            qkv_bias=qkv_bias,  # QKV偏置
            qkv_backend="sdpa",  # QKV后端使用SDPA
            softmax_in_single_precision=False,  # softmax不使用单精度
            flatten_batch=False,  # 不展平批次
            quant_config=quant_config,  # 量化配置
            prefix=prefix,  # 前缀
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """前向传播：计算带掩码的注意力"""
        """
        Args:
            x: [B, N, C] tensor
            mask: [B, N] boolean mask
        """
        attn_mask = None  # 初始化注意力掩码
        if mask is not None:  # 如果提供了掩码
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N] 扩展掩码维度
            attn_mask = attn_mask.float()  # 转换为浮点型
            attn_mask = (1.0 - attn_mask) * -10000.0  # 掩码位置设为-10000

        x = self.attn(x, attn_mask=attn_mask)  # 计算注意力
        return x  # 返回注意力输出


class DashengBlock(nn.Module):
    """达声Transformer块，包含注意力和MLP"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        init_values: float | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()  # 调用父类初始化
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # 第一个层归一化
        self.attn = DashengAttention(  # 注意力层
            dim,  # 维度
            num_heads=num_heads,  # 头数
            qkv_bias=qkv_bias,  # QKV偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("attn", prefix),  # 前缀
        )
        self.ls1 = (  # 第一个层缩放
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()  # 有初始值则使用层缩放，否则恒等
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)  # 第二个层归一化
        self.mlp = DashengMlp(  # MLP层
            in_features=dim,  # 输入特征维度
            hidden_features=int(dim * mlp_ratio),  # 隐藏特征维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("mlp", prefix),  # 前缀
        )
        self.ls2 = (  # 第二个层缩放
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()  # 有初始值则使用层缩放，否则恒等
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播：残差连接的注意力和MLP"""
        x = x + self.ls1(self.attn(self.norm1(x), mask))  # 注意力残差连接
        x = x + self.ls2(self.mlp(self.norm2(x)))  # MLP残差连接
        return x  # 返回输出


class DashengFrontend(nn.Module):
    """音频前端模块，将波形转换为对数梅尔频谱图"""
    """Audio frontend that converts waveforms to log mel-spectrograms."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()  # 调用父类初始化
        self.n_fft = config.n_fft  # FFT窗口大小
        self.hop_length = config.hop_length  # 帧移长度
        self.win_length = config.win_length  # 窗口长度
        self.center = config.center  # 是否中心填充
        spectrogram_window = torch.hann_window(config.win_length)  # 创建汉宁窗
        self.register_buffer(  # 注册缓冲区
            "spectrogram_window",  # 频谱图窗口名称
            spectrogram_window,  # 窗口张量
            persistent=False,  # 不持久化
        )
        self.spectrogram_window: torch.Tensor  # 类型注解
        melscale_fbanks = F.melscale_fbanks(  # 计算梅尔尺度滤波器组
            n_freqs=config.n_fft // 2 + 1,  # 频率数
            f_min=config.f_min,  # 最低频率
            f_max=config.f_max,  # 最高频率
            n_mels=config.n_mels,  # 梅尔频带数
            sample_rate=config.sample_rate,  # 采样率
        )
        self.register_buffer("melscale_fbanks", melscale_fbanks, persistent=False)  # 注册梅尔滤波器组缓冲区
        self.melscale_fbanks: torch.Tensor  # 类型注解

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """将波形转换为对数梅尔频谱图"""
        """Convert waveform to log mel-spectrogram.

        Args:
            waveform: [B, T] tensor of audio samples

        Returns:
            log_mel_spectrogram: [B, n_mels, time] tensor
        """
        spectrogram = F.spectrogram(  # 计算频谱图
            waveform=waveform.to(torch.float32),  # 转换为float32
            pad=0,  # 不填充
            window=self.spectrogram_window,  # 使用汉宁窗
            n_fft=self.n_fft,  # FFT大小
            hop_length=self.hop_length,  # 帧移
            win_length=self.win_length,  # 窗口长度
            power=2,  # 功率谱
            normalized=False,  # 不归一化
            center=self.center,  # 中心填充
        )
        mel_spectrogram = (spectrogram.mT @ self.melscale_fbanks.to(torch.float32)).mT  # 应用梅尔滤波器组
        log_mel_spectrogram = F.amplitude_to_DB(  # 将幅度转换为分贝
            mel_spectrogram.unsqueeze(1),  # 增加通道维度
            multiplier=10,  # 乘数
            amin=1e-10,  # 最小幅度
            db_multiplier=0,  # 分贝乘数
            top_db=120,  # 最大分贝范围
        ).squeeze(1)  # 移除通道维度
        return log_mel_spectrogram.to(waveform.dtype)  # 转换回原始数据类型


class DashengAudioTransformer(nn.Module):
    """达声音频Transformer编码器"""
    """Audio encoder transformer."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()  # 调用父类初始化
        self.target_length = config.target_length  # 目标长度
        self.hop_length = config.hop_length  # 帧移长度
        self.front_end = DashengFrontend(config)  # 音频前端模块
        self.init_bn = nn.BatchNorm2d(config.n_mels, momentum=0.01)  # 初始批归一化
        self.patch_embed = AudioPatchEmbed(  # 音频补丁嵌入
            input_size=(config.n_mels, config.target_length),  # 输入尺寸
            embed_dim=config.embed_dim,  # 嵌入维度
            in_chans=config.input_channels,  # 输入通道数
            patch_size=config.patch_size,  # 补丁尺寸
            flatten=False,  # 不展平
            patch_stride=config.patch_stride,  # 补丁步幅
        )
        self.time_pos_embed = nn.Parameter(  # 时间位置嵌入
            torch.empty(1, config.embed_dim, 1, self.patch_embed.grid_size[1])  # 参数形状
        )
        self.freq_pos_embed = nn.Parameter(  # 频率位置嵌入
            torch.empty(1, config.embed_dim, self.patch_embed.grid_size[0], 1)  # 参数形状
        )
        self.blocks = nn.ModuleList(  # Transformer块列表
            DashengBlock(  # 达声Transformer块
                dim=config.embed_dim,  # 维度
                num_heads=config.num_heads,  # 头数
                mlp_ratio=config.mlp_ratio,  # MLP比例
                qkv_bias=config.qkv_bias,  # QKV偏置
                init_values=config.init_values,  # 层缩放初始值
                quant_config=quant_config,  # 量化配置
                prefix=add_prefix(f"blocks.{i}", prefix),  # 前缀
            )
            for i in range(config.depth)  # 遍历层数
        )
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)  # 最终层归一化

    def forward_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播特征提取：添加位置编码并经过Transformer块"""
        t = x.shape[-1]  # 时间维度大小
        x = x + self.time_pos_embed[:, :, :, :t]  # 添加时间位置嵌入
        x = x + self.freq_pos_embed[:, :, :, :]  # 添加频率位置嵌入
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))  # 展平并重排维度
        for block in self.blocks:  # 遍历每个Transformer块
            x = block(x, mask)  # 通过块处理
        x = self.norm(x)  # 最终归一化
        return x  # 返回特征

    def _to_mask(self, lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        """根据长度生成布尔掩码"""
        batch_size = len(lengths)  # 批次大小
        idx = torch.arange(max_length, device=lengths.device)  # 生成索引
        idx = idx.repeat(batch_size).view(batch_size, max_length)  # 扩展并重塑索引
        mask = (idx < lengths.unsqueeze(-1)).bool()  # 生成布尔掩码
        return mask  # 返回掩码

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向传播：音频编码器主流程"""
        """
        Args:
            x: [B, T] audio waveform tensor
            x_length: [B] tensor of audio lengths

        Returns:
            x: [B, seq_len, embed_dim] encoded features
            mask: [B, seq_len] mask tensor
        """
        x = self.front_end(x)  # 通过前端提取频谱特征
        x = x.to(self.time_pos_embed.dtype)  # 转换为位置嵌入的数据类型
        target_length_in_patches = self.target_length // 4  # 计算补丁级目标长度
        x = x.unsqueeze(1)  # 增加通道维度
        x = torch.permute(x, (0, 2, 1, 3))  # 重排维度
        x = self.init_bn(x)  # 批归一化
        x = torch.permute(x, (0, 2, 1, 3))  # 重排维度回来
        x = self.patch_embed(x)  # 补丁嵌入
        t = x.shape[-1]  # 时间维度
        input_splits = x.split(target_length_in_patches, dim=-1)  # 按补丁目标长度分割
        if x_length is not None:  # 如果提供了长度信息
            assert len(x_length) == len(  # 断言批次大小匹配
                x
            ), "batchsizes of input x and x_length need to be same"
            assert x_length.ndim == 1, "Lengths are of size (B,)"  # 断言长度为一维
            scaled_lengths = (x_length / (self.hop_length * 4)).long()  # 缩放长度
            mask = self._to_mask(max_length=t, lengths=scaled_lengths)  # 生成掩码
            split_masks = mask.split(target_length_in_patches, dim=-1)  # 分割掩码
        else:  # 没有长度信息
            mask = None  # 掩码为空
            split_masks = [None] * len(input_splits)  # 分割掩码列表全为空
        outputs = []  # 输出列表
        for split_x, split_mask in zip(input_splits, split_masks):  # 遍历分割的输入和掩码
            forward_kwargs = {}  # 前向传播关键字参数
            forward_kwargs["mask"] = split_mask  # 设置掩码
            split_x = self.forward_features(split_x, **forward_kwargs)  # 特征提取
            outputs.append(split_x)  # 添加到输出列表
        x = torch.cat(outputs, dim=1)  # 拼接所有输出
        return x, mask  # 返回编码特征和掩码


class AudioProjectorSubsample(nn.Module):
    """带子采样的音频投影器"""
    """Audio projector with subsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate=5,
        dtype: torch.dtype | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()  # 调用父类初始化
        self.k = downsample_rate  # 下采样率
        self.fc1 = ColumnParallelLinear(  # 第一个全连接层
            input_size=in_dim * self.k,  # 输入大小为in_dim乘以下采样率
            output_size=out_dim,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("net.0", prefix),  # 参数名前缀
        )
        self.act = nn.GELU()  # GELU激活函数
        self.fc2 = RowParallelLinear(  # 第二个全连接层
            input_size=out_dim,  # 输入大小
            output_size=out_dim,  # 输出大小
            bias=False,  # 不使用偏置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("net.2", prefix),  # 参数名前缀
        )

    def forward(self, x, mask=None):
        """前向传播：子采样并投影音频特征"""
        batch_size, seq_len, dim = x.shape  # 获取批次大小、序列长度和维度
        num_frames_to_discard = seq_len % self.k  # 计算需要丢弃的帧数
        if num_frames_to_discard > 0:  # 如果有需要丢弃的帧
            x = x[:, :-num_frames_to_discard, :]  # 丢弃末尾帧
            if mask is not None:  # 如果有掩码
                mask = mask[:, :-num_frames_to_discard]  # 同步调整掩码
        if mask is None:  # 如果没有掩码
            mask = torch.ones(x.shape[:-1], dtype=torch.long, device=x.device)  # 创建全1掩码
        x = x.reshape(batch_size, -1, self.k * dim)  # 重塑为下采样后的形状
        x, _ = self.fc1(x)  # 第一个全连接层
        x = self.act(x)  # 激活函数
        x, _ = self.fc2(x)  # 第二个全连接层
        mask = mask.reshape(batch_size, -1, self.k)  # 重塑掩码
        mask = mask.any(dim=-1).long()  # 掩码取或操作
        return x, mask  # 返回投影后的特征和掩码


class MiDashengLMModel(nn.Module):
    """MiDashengLM模型，用于音频-语言处理"""
    """MiDashengLM model for audio-language processing."""

    default_bitsandbytes_target_modules = [  # bitsandbytes默认目标模块
        ".fc1.",  # fc1层
        ".fc2.",  # fc2层
        ".gate_up_proj.",  # 门控上投影层
        ".down_proj.",  # 下投影层
        ".q_proj.",  # Q投影层
        ".k_proj.",  # K投影层
        ".v_proj.",  # V投影层
        ".o_proj.",  # O投影层
    ]

    bitsandbytes_stacked_params_mapping = {  # bitsandbytes堆叠参数映射
        "q_proj": ("qkv_proj", 0),  # Q投影映射到QKV投影的第0部分
        "k_proj": ("qkv_proj", 1),  # K投影映射到QKV投影的第1部分
        "v_proj": ("qkv_proj", 2),  # V投影映射到QKV投影的第2部分
        "gate_proj": ("gate_up_proj", 0),  # 门控投影映射到门控上投影的第0部分
        "up_proj": ("gate_up_proj", 1),  # 上投影映射到门控上投影的第1部分
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        rope_scaling = config.text_config.rope_parameters  # 获取RoPE缩放参数
        if rope_scaling:  # 如果有RoPE缩放参数
            if "mrope_section" in rope_scaling:  # 如果包含mrope_section
                # Remove mrope_section from rope_parameters so downstream
                # code treats this as standard rotary embedding.
                del rope_scaling["mrope_section"]  # 删除mrope_section，使下游代码将其视为标准旋转嵌入
        self.audio_encoder = DashengAudioTransformer(  # 音频编码器
            config.audio_encoder_config,  # 音频编码器配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("audio_encoder", prefix),  # 前缀
        )
        self.audio_projector = AudioProjectorSubsample(  # 音频投影器
            in_dim=config.audio_encoder_config.embed_dim,  # 输入维度
            out_dim=config.text_config.hidden_size,  # 输出维度
            downsample_rate=config.subsample_factor,  # 下采样率
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("audio_projector", prefix),  # 前缀
        )
        self.language_model = Qwen2ForCausalLM(  # 语言模型
            config.text_config,  # 文本配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("decoder", prefix),  # 前缀
        )
        self.logits_processor = self.language_model.logits_processor  # logits处理器
        self.quant_config = quant_config  # 量化配置

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """使用多模态标记填充输入ID"""
        """Pad input IDs with multimodal tokens."""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建多模态填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 填充并返回

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """处理音频输入并返回嵌入"""
        """Process audio inputs and return embeddings.

        Args:
            items: List of multimodal data items containing audio features

        Returns:
            audio_embeddings: Concatenated audio embeddings
        """
        logger.debug("=" * 80)  # 打印分隔线
        logger.debug(f"get_audio_feature called with {len(items)} items")  # 打印调用信息
        logger.debug("=" * 80)  # 打印分隔线
        for i, item in enumerate(items):  # 遍历每个数据项
            logger.debug(f"Item {i} feature shape: {item.feature.shape}")  # 打印特征形状
            logger.debug(  # 打印音频长度
                f"Item {i} audio_length: {getattr(item, 'audio_length', 'NOT SET')}"
            )
            logger.debug(f"Item {i} pad_value: {getattr(item, 'pad_value', 'NOT SET')}")  # 打印填充值
            logger.debug(f"Item {i} hash: {getattr(item, 'hash', 'NOT SET')}")  # 打印哈希值
        input_values = torch.cat([item.feature for item in items], dim=0)  # 拼接所有特征
        logger.debug(f"Concatenated input_values shape: {input_values.shape}")  # 打印拼接后形状
        audio_lengths = []  # 音频长度列表
        for item in items:  # 遍历每个数据项
            if hasattr(item, "audio_length") and item.audio_length is not None:  # 如果有音频长度
                audio_lengths.append(item.audio_length)  # 添加音频长度
            else:  # 没有音频长度
                audio_lengths.append(item.feature.shape[-1])  # 使用特征最后一维作为长度
        audio_length = torch.tensor(audio_lengths, device=input_values.device)  # 转换为张量
        logger.debug(f"audio_length: {audio_length}")  # 打印音频长度
        encoder_out, encoder_atts = self.audio_encoder(input_values, audio_length)  # 音频编码
        logger.debug(f"Encoder output shape: {encoder_out.shape}")  # 打印编码器输出形状
        audio_embeddings, _ = self.audio_projector(encoder_out, encoder_atts)  # 音频投影
        audio_embeddings = audio_embeddings.to(input_values.dtype)  # 转换数据类型
        logger.debug(f"Projector output shape: {audio_embeddings.shape}")  # 打印投影器输出形状
        batch_size, max_audio_tokens, embed_dim = audio_embeddings.shape  # 获取形状信息
        logger.debug(f"Using all {max_audio_tokens} audio tokens from projector output")  # 打印音频标记数
        masked_audio_features = audio_embeddings.reshape(-1, embed_dim)  # 重塑为二维
        logger.debug(f"Final output shape: {masked_audio_features.shape}")  # 打印最终输出形状
        logger.debug(  # 打印统计信息
            f"Stats: min={masked_audio_features.min().item():.4f}, max={masked_audio_features.max().item():.4f}"
        )
        logger.debug(  # 打印数据类型和设备信息
            f"Audio embeddings dtype: {masked_audio_features.dtype}, device: {masked_audio_features.device}"
        )
        logger.debug(  # 打印前5个值
            f"First 5 values of first audio token: {masked_audio_features[0, :5].tolist()}"
        )
        logger.debug("=" * 80)  # 打印分隔线
        return masked_audio_features  # 返回掩码后的音频特征

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.language_model.model.embed_tokens  # 返回语言模型的词嵌入层

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """MiDashengLM前向传播"""
        """Run forward pass for MiDashengLM.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a batch.
            positions: Flattened (concatenated) position ids corresponding to a batch.
            forward_batch: Forward batch information including multimodal data.
        """
        if forward_batch.contains_mm_inputs():  # 如果包含多模态输入
            logger.debug("=" * 80)  # 打印分隔线
            logger.debug(f"input_ids shape: {input_ids.shape}")  # 打印输入ID形状
            logger.debug(f"input_ids first 20: {input_ids[:20].tolist()}")  # 打印前20个输入ID
            logger.debug(  # 打印唯一值数量
                f"input_ids unique values count: {len(torch.unique(input_ids))}"
            )
            if forward_batch.mm_inputs and len(forward_batch.mm_inputs) > 0:  # 如果有多模态输入
                mm_input = forward_batch.mm_inputs[0]  # 获取第一个多模态输入
                if mm_input and len(mm_input.mm_items) > 0:  # 如果有多模态数据项
                    pad_value = mm_input.mm_items[0].pad_value  # 获取填充值
                    logger.debug(f"Expected pad_value: {pad_value}")  # 打印预期填充值
                    logger.debug(  # 打印填充值出现次数
                        f"Count of pad_value in input_ids: {(input_ids == pad_value).sum().item()}"
                    )
                    if hasattr(mm_input, "audio_token_id") and mm_input.audio_token_id:  # 如果有音频标记ID
                        logger.debug(f"audio_token_id: {mm_input.audio_token_id}")  # 打印音频标记ID
                        logger.debug(  # 打印音频标记ID出现次数
                            f"Count of audio_token_id in input_ids: {(input_ids == mm_input.audio_token_id).sum().item()}"
                        )
            logger.debug("=" * 80)  # 打印分隔线

        return general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.language_model,  # 语言模型
            positions=positions,  # 位置ID
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},  # 音频特征提取函数
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        """Load model weights."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))  # 获取参数字典
        buffers_dict = dict(self.named_buffers())  # 获取缓冲区字典
        audio_encoder_loaded = []  # 已加载的音频编码器权重列表
        audio_projector_loaded = []  # 已加载的音频投影器权重列表
        skipped_weights = []  # 跳过的权重列表
        decoder_weights = []  # 解码器权重列表
        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入逆频率
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过旋转嵌入缓存
                continue
            if name.startswith("decoder"):  # 如果是解码器权重
                decoder_weights.append((name, loaded_weight))  # 添加到解码器权重列表
                continue
            original_name = name  # 保存原始名称
            if "audio_encoder.front_end" in name:  # 音频前端权重名称映射
                if ".mel_scale.fb" in name:  # 梅尔尺度滤波器组
                    name = name.replace(".mel_scale.fb", ".melscale_fbanks")  # 替换名称
                elif ".spectrogram.window" in name:  # 频谱图窗口
                    name = name.replace(".spectrogram.window", ".spectrogram_window")  # 替换名称
            if "audio_encoder" in name and ".attn.qkv." in name:  # 音频编码器注意力QKV
                name = name.replace(".attn.qkv.", ".attn.attn.qkv_proj.")  # 替换名称
            if "audio_encoder" in name and ".attn.proj." in name:  # 音频编码器注意力投影
                name = name.replace(".attn.proj.", ".attn.attn.proj.")  # 替换名称
            if "audio_projector" in name:  # 音频投影器权重名称映射
                name = name.replace(".net.0.", ".fc1.")  # 替换net.0为fc1
                name = name.replace(".net.2.", ".fc2.")  # 替换net.2为fc2
            if (  # 如果是偏置且不在参数或缓冲区字典中
                name.endswith(".bias")
                and name not in params_dict
                and name not in buffers_dict
            ):
                skipped_weights.append(f"{original_name} (bias not in params/buffers)")  # 记录跳过的权重
                continue
            if name in params_dict:  # 如果名称在参数字典中
                param = params_dict[name]  # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重
            elif name in buffers_dict:  # 如果名称在缓冲区字典中
                buffers_dict[name].copy_(loaded_weight)  # 复制权重到缓冲区
            else:  # 名称不在字典中
                if "audio_projector" in original_name:  # 如果是音频投影器权重
                    skipped_weights.append(f"{original_name} -> {name} (NOT IN MODEL)")  # 记录跳过
                else:  # 其他权重
                    skipped_weights.append(f"{original_name} (not in model)")  # 记录跳过
                continue

            if "audio_encoder" in original_name:  # 如果是音频编码器权重
                audio_encoder_loaded.append(original_name)  # 记录已加载
            elif "audio_projector" in original_name:  # 如果是音频投影器权重
                audio_projector_loaded.append(original_name)  # 记录已加载
        if decoder_weights:  # 如果有解码器权重
            logger.debug(  # 打印解码器权重数量
                f"Passing {len(decoder_weights)} decoder weights to language_model.load_weights()"
            )
            decoder_weights_stripped = [  # 去除decoder前缀
                (name.replace("decoder.", "", 1), weight)
                for name, weight in decoder_weights
            ]
            self.language_model.load_weights(decoder_weights_stripped)  # 加载解码器权重
        logger.debug("=" * 80)  # 打印分隔线
        logger.debug(f"Audio encoder weights loaded: {len(audio_encoder_loaded)}")  # 打印已加载的音频编码器权重数
        logger.debug(f"Audio projector weights loaded: {len(audio_projector_loaded)}")  # 打印已加载的音频投影器权重数
        logger.debug(  # 打印传递给语言模型的解码器权重数
            f"Decoder weights passed to language_model: {len(decoder_weights)}"
        )
        logger.debug(f"Skipped weights: {len(skipped_weights)}")  # 打印跳过的权重数
        encoder_skipped = [s for s in skipped_weights if "audio_encoder" in s]  # 音频编码器跳过的权重
        projector_skipped = [s for s in skipped_weights if "audio_projector" in s]  # 音频投影器跳过的权重
        if projector_skipped:  # 如果有跳过的音频投影器权重
            logger.debug("Skipped audio_projector weights:")  # 打印跳过信息
            for s in projector_skipped:  # 遍历跳过的权重
                logger.debug(f"  {s}")  # 打印详细信息
        if encoder_skipped:  # 如果有跳过的音频编码器权重
            logger.debug(f"Skipped audio_encoder weights: {len(encoder_skipped)}")  # 打印跳过数量
            non_bias_skipped = [s for s in encoder_skipped if "bias" not in s]  # 非偏置跳过的权重
            if non_bias_skipped:  # 如果有非偏置跳过的权重
                logger.debug("  First 10 non-bias skipped:")  # 打印前10个非偏置跳过
                for s in non_bias_skipped[:10]:  # 遍历前10个
                    logger.debug(f"    {s}")  # 打印详细信息
        logger.debug("=" * 80)  # 打印分隔线

    def get_embed_and_head(self):
        """获取嵌入层和语言模型头的权重"""
        return (  # 返回元组
            self.language_model.model.embed_tokens.weight,  # 词嵌入权重
            self.language_model.lm_head.weight,  # 语言模型头权重
        )


EntryClass = [MiDashengLMModel]  # 入口类列表
