# MiMo音频：分词器、编码工具和音频编码器
# 该模块实现了MiMo模型的音频处理组件，包括音频分词器（将音频转换为离散码本编码）、
# 残差向量量化器、音频编码器（将音频编码为LLM可用的嵌入表示）
# 核心组件：EuclideanCodebook、VectorQuantization、ResidualVectorQuantization、
# ResidualVectorQuantizer、MiMoAudioTokenizer、MiMoAudioEncoder
# 音频处理流程：梅尔频谱 -> 卷积编码器 -> 残差向量量化 -> 码本嵌入 -> 局部Transformer -> 投影到LLM维度

"""MiMo audio: tokenizer, encoding utilities, and audio encoder."""

# Audio tokenizer adapted from https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git

import logging  # 导入日志模块
import math  # 导入数学模块
import os  # 导入操作系统模块
import typing as tp  # 导入类型标注别名
from dataclasses import dataclass  # 导入数据类装饰器
from functools import wraps  # 导入装饰器工具
from typing import List, Optional, Tuple  # 导入类型注解

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口
from einops import rearrange  # 导入张量重排工具
from transformers.activations import ACT2FN  # 导入激活函数映射
from transformers.configuration_utils import PretrainedConfig  # 导入预训练配置基类
from transformers.modeling_utils import PreTrainedModel  # 导入预训练模型基类
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config  # 导入Qwen2配置
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model  # 导入Qwen2模型

from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数
from sglang.srt.utils import is_cuda  # 导入CUDA检测工具

if is_cuda():  # 如果是CUDA环境
    from sgl_kernel.flash_attn import flash_attn_varlen_func  # 导入Flash注意力变长函数
else:  # 非CUDA环境

    def flash_attn_varlen_func(*args, **kwargs):  # 不可用的占位函数
        raise RuntimeError("MiMoAudioTokenizer requires CUDA to run.")  # 抛出运行时错误


logger = logging.getLogger(__name__)  # 获取日志记录器


def _compute_default_rope_parameters(
    config=None, device=None, seq_len=None, **rope_kwargs
):
    """计算默认的旋转位置编码参数"""
    if config is not None and len(rope_kwargs) > 0:  # 参数互斥检查
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive"
        )
    if len(rope_kwargs) > 0:  # 从rope_kwargs获取参数
        base = rope_kwargs["base"]  # RoPE基频
        dim = rope_kwargs["dim"]  # 维度
    elif config is not None:  # 从config获取参数
        base = config.rope_theta  # RoPE基频
        partial_rotary_factor = (  # 部分旋转因子
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = getattr(config, "head_dim", None)  # 头维度
        if head_dim is None:  # 如果未设置头维度
            head_dim = config.hidden_size // config.num_attention_heads  # 从隐藏维度推导
            logger.info(
                "audio.head_dim not set; defaulting to hidden_size/num_heads = %d",
                head_dim,
            )
        dim = int(head_dim * partial_rotary_factor)  # 计算旋转维度
    attention_factor = 1.0  # 注意力缩放因子
    inv_freq = 1.0 / (  # 计算逆频率
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor  # 返回逆频率和缩放因子


_ROPE_INIT_FUNCTIONS = {  # RoPE初始化函数映射
    "default": _compute_default_rope_parameters,  # 默认初始化
}


def _dynamic_rope_update(rope_forward):
    """动态RoPE更新装饰器，支持dynamic和longrope类型"""
    def longrope_frequency_update(self, position_ids, device):
        """长序列RoPE频率更新"""
        seq_len = torch.max(position_ids) + 1  # 当前序列长度
        if hasattr(self.config, "original_max_position_embeddings"):  # 检查原始最大位置嵌入
            original_max_position_embeddings = (
                self.config.original_max_position_embeddings
            )
        else:  # 使用默认最大位置嵌入
            original_max_position_embeddings = self.config.max_position_embeddings
        if seq_len > original_max_position_embeddings:  # 超过原始最大长度
            if not hasattr(self, "long_inv_freq"):  # 如果没有长序列逆频率
                self.long_inv_freq, _ = self.rope_init_fn(
                    self.config, device, seq_len=original_max_position_embeddings + 1
                )
            self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)  # 注册长序列逆频率
        else:  # 未超过原始最大长度
            self.original_inv_freq = self.original_inv_freq.to(device)  # 转换设备
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)  # 使用原始逆频率

    def dynamic_frequency_update(self, position_ids, device):
        """动态频率更新"""
        seq_len = torch.max(position_ids) + 1  # 当前序列长度
        if seq_len > self.max_seq_len_cached:  # 增长情况
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # 更新逆频率
            self.max_seq_len_cached = seq_len  # 更新缓存的最大序列长度

        if (  # 序列长度收缩回原始长度
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.original_inv_freq = self.original_inv_freq.to(device)  # 转换设备
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)  # 恢复原始逆频率
            self.max_seq_len_cached = self.original_max_seq_len  # 恢复缓存长度

    @wraps(rope_forward)  # 保留原始函数签名
    def wrapper(self, x, position_ids):
        if "dynamic" in self.rope_type:  # 动态RoPE
            dynamic_frequency_update(self, position_ids, device=x.device)
        elif self.rope_type == "longrope":  # 长序列RoPE
            longrope_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)  # 调用原始前向传播

    return wrapper  # 返回装饰后的函数


class AudioRotaryEmbedding(nn.Module):
    """音频旋转位置编码嵌入层"""
    def __init__(self, base, dim, max_seq_len, rope_type="default", device=None):
        super().__init__()  # 调用父类初始化
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.rope_type = rope_type  # RoPE类型
        self.rope_init_fn = _ROPE_INIT_FUNCTIONS[self.rope_type]  # 初始化函数
        inv_freq, self.attention_scaling = self.rope_init_fn(  # 计算逆频率和缩放因子
            device=device, base=base, dim=dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册逆频率缓冲区
        self.original_inv_freq = self.inv_freq  # 保存原始逆频率

    @torch.no_grad()  # 禁用梯度计算
    @_dynamic_rope_update  # 应用动态RoPE更新
    def forward(self, x, position_ids):
        """计算旋转位置编码的cos和sin值"""
        inv_freq_expanded = self.inv_freq[:, None].float().expand(-1, 1).to(x.device)  # 扩展逆频率
        position_ids_expanded = position_ids[None, :].float()  # 扩展位置ID
        device_type = (  # 确定设备类型
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32  # 强制float32计算
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(0, 1)  # 计算频率矩阵
            emb = torch.cat((freqs, freqs), dim=-1)  # 拼接频率
            cos = emb.cos() * self.attention_scaling  # 计算cos值
            sin = emb.sin() * self.attention_scaling  # 计算sin值
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # 转换回原始数据类型


class EuclideanCodebook(nn.Module):
    """基于欧几里得距离的码本（仅推理）"""
    """Codebook with Euclidean distance (inference-only)."""

    def __init__(
        self, dim: int, codebook_size: int, kmeans_init: bool = False, **kwargs
    ):
        super().__init__()  # 调用父类初始化
        init_fn = self._uniform_init if not kmeans_init else torch.zeros  # 选择初始化函数
        embed = init_fn(codebook_size, dim)  # 初始化嵌入矩阵

        self.codebook_size = codebook_size  # 码本大小

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))  # 初始化标志
        self.register_buffer("cluster_size", torch.zeros(codebook_size))  # 簇大小
        self.register_buffer("embed", embed)  # 嵌入矩阵
        self.register_buffer("embed_avg", embed.clone())  # 嵌入均值

    def preprocess(self, x):
        """预处理：展平输入"""
        x = rearrange(x, "... d -> (...) d")  # 重排为二维
        return x

    def quantize(self, x):
        """量化：找到最近码本向量"""
        embed = self.embed.t()  # 转置嵌入矩阵
        dist_val = -(  # 计算负欧几里得距离
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist_val.max(dim=-1).indices  # 取最大距离对应的索引
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        """后处理：重塑嵌入索引"""
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        """反量化：从码本索引获取嵌入向量"""
        quantize = F.embedding(embed_ind, self.embed)  # 查找嵌入
        return quantize

    def encode(self, x):
        """编码：量化输入向量"""
        shape = x.shape
        x = self.preprocess(x)  # 预处理
        embed_ind = self.quantize(x)  # 量化
        embed_ind = self.postprocess_emb(embed_ind, shape)  # 后处理
        return embed_ind

    def decode(self, embed_ind):
        """解码：从索引重建向量"""
        quantize = self.dequantize(embed_ind)  # 反量化
        return quantize

    @staticmethod
    def _uniform_init(*shape: int):
        """均匀初始化"""
        t = torch.empty(shape)  # 创建空张量
        nn.init.kaiming_uniform_(t)  # Kaiming均匀初始化
        return t


class VectorQuantization(nn.Module):
    """基于欧几里得距离的向量量化（仅推理）"""
    """Vector quantization with euclidean distance (inference-only)."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        kmeans_init: bool = True,
        **kwargs,
    ):
        super().__init__()  # 调用父类初始化
        _codebook_dim: int = codebook_dim if codebook_dim is not None else dim  # 码本维度

        requires_projection = _codebook_dim != dim  # 是否需要投影
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )  # 输入投影
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )  # 输出投影

        self._codebook = EuclideanCodebook(  # 欧几里得码本
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
        )
        self.codebook_size = codebook_size  # 码本大小

    @property
    def codebook(self):
        """获取码本嵌入矩阵"""
        return self._codebook.embed

    def encode(self, x):
        """编码：投影后量化"""
        x = self.project_in(x)  # 输入投影
        embed_in = self._codebook.encode(x)  # 量化编码
        return embed_in

    def decode(self, embed_ind):
        """解码：反量化后投影"""
        quantize = self._codebook.decode(embed_ind)  # 反量化
        quantize = self.project_out(quantize)  # 输出投影
        return quantize


class ResidualVectorQuantization(nn.Module):
    """残差向量量化实现
    遵循论文 https://arxiv.org/pdf/2107.03312.pdf 的算法1
    """
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, codebook_size, **kwargs):
        super().__init__()  # 调用父类初始化
        if isinstance(codebook_size, int):  # 如果码本大小是整数
            codebook_size = [codebook_size] * num_quantizers  # 扩展为列表
        elif len(codebook_size) < num_quantizers:  # 如果列表不够长
            codebook_size += [codebook_size[-1]] * (num_quantizers - len(codebook_size))  # 用最后一个值填充
        self.layers = nn.ModuleList(  # 量化层列表
            [
                VectorQuantization(codebook_size=codebook_size[i], **kwargs)
                for i in range(num_quantizers)
            ]
        )

    def encode(
        self, x: torch.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int] = None
    ) -> torch.Tensor:
        """编码：逐层量化残差"""
        residual = x  # 初始残差为输入
        all_indices = []  # 所有层的量化索引
        n_q = len(self.layers) if n_q is None else n_q  # 量化器数量
        st = 0 if st is None else st  # 起始层
        for layer in self.layers[st:n_q]:  # 遍历量化层
            indices = layer.encode(residual)  # 编码当前残差
            quantized = layer.decode(indices)  # 解码获取量化值
            residual = residual - quantized  # 更新残差
            all_indices.append(indices)  # 添加索引
        out_indices = torch.stack(all_indices)  # 堆叠所有索引
        return out_indices

    def decode(self, q_indices: torch.Tensor, st: int = 0) -> torch.Tensor:
        """解码：累加各层量化输出"""
        quantized_out = self.layers[st].decode(q_indices[0])  # 第一层解码
        for i in range(1, len(q_indices)):  # 遍历后续层
            layer = self.layers[st + i]  # 获取当前层
            quantized = layer.decode(q_indices[i])  # 解码当前层
            quantized_out = quantized_out + quantized  # 累加量化输出
        return quantized_out


class ResidualVectorQuantizer(nn.Module):
    """残差向量量化器（仅推理）"""
    """Residual Vector Quantizer (inference-only)."""

    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int | list = 1024,
        kmeans_init: bool = True,
        **kwargs,
    ):
        super().__init__()  # 调用父类初始化
        self.n_q = n_q  # 量化器数量
        self.vq = ResidualVectorQuantization(  # 残差向量量化
            dim=dimension,
            codebook_size=bins,
            num_quantizers=n_q,
            kmeans_init=kmeans_init,
        )

    def encode(
        self, x: torch.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int] = None
    ) -> torch.Tensor:
        """编码输入"""
        n_q = n_q if n_q else self.n_q  # 使用默认量化器数量
        st = st or 0  # 默认起始层
        codes = self.vq.encode(x, n_q=n_q, st=st)  # 残差向量量化编码
        return codes

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        """解码码本编码"""
        quantized = self.vq.decode(codes, st=st)  # 残差向量量化解码
        return quantized


class MiMoAudioTokenizerConfig(PretrainedConfig):
    """MiMo音频分词器配置"""
    model_type = "mimo_audio_tokenizer"  # 模型类型

    def __init__(
        self,
        max_audio_seconds: int = 1800,  # 最大音频秒数
        stride_size: int = 2,  # 步幅大小
        avg_pooler: int = 1,  # 平均池化大小
        d_model: int = 768,  # 模型维度
        scale_embedding: bool = True,  # 是否缩放嵌入
        kernel_size: int = 3,  # 卷积核大小
        activation_function: str = "gelu",  # 激活函数
        encoder_layers: int = 8,  # 编码器层数
        encoder_skip_layer_id: int = None,  # 编码器跳跃连接层ID
        encoder_attention_heads: int = 12,  # 编码器注意力头数
        encoder_ffn_dim: int = 3072,  # 编码器FFN维度
        encoder_causal: bool = False,  # 编码器是否因果
        encoder_attn_window_size: list = None,  # 编码器注意力窗口大小
        decoder_layers: int = 8,  # 解码器层数
        decoder_attention_heads: int = 12,  # 解码器注意力头数
        decoder_ffn_dim: int = 3072,  # 解码器FFN维度
        decoder_kernel_size: int = 3,  # 解码器卷积核大小
        decoder_stride_size: int = 2,  # 解码器步幅大小
        decoder_causal: bool = True,  # 解码器是否因果
        decoder_attn_window_size: list = None,  # 解码器注意力窗口大小
        nfft: int = 1024,  # FFT大小
        vocoder_dim: int = 512,  # 声码器维度
        vocoder_intermediate_dim: int = 4096,  # 声码器中间维度
        vocoder_num_layers: int = 30,  # 声码器层数
        n_mels: int = 80,  # 梅尔频率数
        sampling_rate: int = 24000,  # 采样率
        hop_length: int = 240,  # 跳跃长度
        window_size: int = 1024,  # 窗口大小
        vocoder_padding: str = "same",  # 声码器填充模式
        fmin: int = 0,  # 最低频率
        fmax: int = None,  # 最高频率
        num_quantizers: int = 12,  # 量化器数量
        codebook_size: list = None,  # 码本大小
        threshold_ema_dead_code: int = 10,  # 死码阈值
        position_embedding_type: str = "rope",  # 位置嵌入类型
        rope_theta: int = 10000,  # RoPE基频
        rope_type: str = "default",  # RoPE类型
        ln_type: str = "LayerNorm",  # 层归一化类型
        vocoder_attention_heads: int = 4,  # 声码器注意力头数
        vocoder_attn_window_size: list = None,  # 声码器注意力窗口大小
        use_istft_only: bool = False,  # 是否仅使用iSTFT
        hybrid_attention: bool = False,  # 是否使用混合注意力
        hybrid_block_size: int = 8,  # 混合注意力块大小
        swa_per_block: int = 2,  # 每块的滑动窗口注意力数
        **kwargs,
    ):
        super().__init__(**kwargs)  # 调用父类初始化
        self.max_audio_seconds = max_audio_seconds  # 最大音频秒数
        self.stride_size = stride_size  # 步幅大小
        self.avg_pooler = avg_pooler  # 平均池化大小
        self.d_model = d_model  # 模型维度
        self.scale_embedding = scale_embedding  # 是否缩放嵌入
        self.kernel_size = kernel_size  # 卷积核大小
        self.activation_function = activation_function  # 激活函数
        self.encoder_layers = encoder_layers  # 编码器层数
        self.encoder_skip_layer_id = encoder_skip_layer_id  # 编码器跳跃连接层ID
        self.encoder_attention_heads = encoder_attention_heads  # 编码器注意力头数
        self.encoder_ffn_dim = encoder_ffn_dim  # 编码器FFN维度
        self.encoder_causal = encoder_causal  # 编码器是否因果
        self.encoder_attn_window_size = (  # 编码器注意力窗口大小
            encoder_attn_window_size
            if encoder_attn_window_size is not None
            else [-1, -1]  # 默认无窗口限制
        )
        self.decoder_layers = decoder_layers  # 解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 解码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 解码器FFN维度
        self.decoder_kernel_size = decoder_kernel_size  # 解码器卷积核大小
        self.decoder_stride_size = decoder_stride_size  # 解码器步幅大小
        self.decoder_causal = decoder_causal  # 解码器是否因果
        self.decoder_attn_window_size = (  # 解码器注意力窗口大小
            decoder_attn_window_size
            if decoder_attn_window_size is not None
            else [-1, -1]  # 默认无窗口限制
        )
        self.nfft = nfft  # FFT大小
        self.vocoder_dim = vocoder_dim  # 声码器维度
        self.vocoder_intermediate_dim = vocoder_intermediate_dim  # 声码器中间维度
        self.vocoder_num_layers = vocoder_num_layers  # 声码器层数
        self.n_mels = n_mels  # 梅尔频率数
        self.sampling_rate = sampling_rate  # 采样率
        self.hop_length = hop_length  # 跳跃长度
        self.window_size = window_size  # 窗口大小
        self.vocoder_padding = vocoder_padding  # 声码器填充模式
        self.fmin = fmin  # 最低频率
        self.fmax = fmax  # 最高频率
        self.num_quantizers = num_quantizers  # 量化器数量
        self.codebook_size = codebook_size if codebook_size is not None else [1024]  # 码本大小
        self.threshold_ema_dead_code = threshold_ema_dead_code  # 死码阈值
        self.position_embedding_type = position_embedding_type  # 位置嵌入类型
        self.rope_theta = rope_theta  # RoPE基频
        self.rope_type = rope_type  # RoPE类型
        self.ln_type = ln_type  # 层归一化类型
        self.vocoder_attention_heads = vocoder_attention_heads  # 声码器注意力头数
        self.vocoder_attn_window_size = (  # 声码器注意力窗口大小
            vocoder_attn_window_size
            if vocoder_attn_window_size is not None
            else [40, 10]  # 默认窗口大小
        )
        self.use_istft_only = use_istft_only  # 是否仅使用iSTFT
        self.hybrid_attention = hybrid_attention  # 是否使用混合注意力
        self.hybrid_block_size = hybrid_block_size  # 混合注意力块大小
        self.swa_per_block = swa_per_block  # 每块的滑动窗口注意力数


def get_sequence_mask(inputs, inputs_length):
    """获取序列掩码和解包索引"""
    if inputs.dim() == 3:  # 如果输入是三维
        bsz, tgt_len, _ = inputs.size()  # 获取批次大小和目标长度
    else:  # 二维输入
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)  # 从长度获取维度
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)  # 创建位置索引
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(  # 生成掩码
        bsz, tgt_len, 1
    )
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1  # 计算解包索引
    return sequence_mask, unpacking_index  # 返回掩码和解包索引


def unpack_hidden_states(
    hidden_states, lengths, sequence_mask=None, unpacking_index=None
):
    """解包隐藏状态：从打包格式恢复为填充批次格式"""
    bsz = lengths.shape[0]  # 批次大小
    if sequence_mask is None or unpacking_index is None:  # 如果没有预计算掩码
        sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)  # 计算掩码
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(  # 按索引恢复
        bsz, torch.max(lengths), hidden_states.shape[-1]
    )
    return torch.where(sequence_mask, hidden_states, 0)  # 应用掩码，填充位置置零


def get_position_ids(lengths):
    """根据长度生成位置ID"""
    total_len = lengths.sum()  # 总长度
    offset = torch.cat([torch.zeros(1).to(lengths), lengths[:-1].cumsum(dim=0)])  # 计算偏移
    offset = torch.repeat_interleave(offset, lengths)  # 重复偏移
    return torch.arange(0, total_len).to(offset) - offset  # 计算位置ID


LAYER_NORM = {"LayerNorm": nn.LayerNorm}  # 层归一化映射


class AudioEncoderAttention(nn.Module):
    """音频编码器注意力层，支持窗口注意力和因果注意力"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        causal: bool = False,
    ):
        super().__init__()  # 调用父类初始化
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 头数
        self.head_dim = embed_dim // num_heads  # 头维度
        self.window_size = window_size  # 窗口大小
        self.causal = causal  # 是否因果

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # K投影
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # V投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # Q投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 输出投影

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_position_embeddings=None,
    ):
        """音频编码器注意力前向传播"""
        bsz, _ = hidden_states.size()  # 获取批次大小

        query_states = self.q_proj(hidden_states).view(  # Q投影并重塑
            bsz, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)  # K投影并重塑
        value_states = self.v_proj(hidden_states).view(  # V投影并重塑
            bsz, self.num_heads, self.head_dim
        )

        if rope_position_embeddings is not None:  # 如果有旋转位置编码
            cos, sin = rope_position_embeddings  # 解包cos和sin
            query_states, key_states = self.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )  # 应用旋转位置编码

        attn_output = flash_attn_varlen_func(  # Flash注意力变长计算
            query_states,
            key_states,
            value_states,
            cu_seqlens,  # Q累计序列长度
            cu_seqlens,  # K累计序列长度
            max_seqlen,  # Q最大序列长度
            max_seqlen,  # K最大序列长度
            causal=self.causal,  # 是否因果
            window_size=self.window_size,  # 窗口大小
        )

        attn_output = attn_output.reshape(bsz, self.embed_dim)  # 重塑注意力输出
        attn_output = self.out_proj(attn_output)  # 输出投影
        return attn_output

    @staticmethod
    def _rotate_half(x):
        """旋转张量的后半部分"""
        x1 = x[..., : x.shape[-1] // 2]  # 前半部分
        x2 = x[..., x.shape[-1] // 2 :]  # 后半部分
        return torch.cat((-x2, x1), dim=-1)  # 交换并取反

    @classmethod
    def apply_rotary_pos_emb(cls, q, k, cos, sin, unsqueeze_dim=1):
        """应用旋转位置编码到Q和K"""
        cos = cos.unsqueeze(unsqueeze_dim)  # 增加维度
        sin = sin.unsqueeze(unsqueeze_dim)  # 增加维度
        q_embed = (q * cos) + (cls._rotate_half(q) * sin)  # Q旋转编码
        k_embed = (k * cos) + (cls._rotate_half(k) * sin)  # K旋转编码
        return q_embed, k_embed


class AudioEncoderTransformerLayer(nn.Module):
    """音频编码器Transformer层"""
    def __init__(
        self,
        config: MiMoAudioTokenizerConfig,
        causal: bool,
        attn_window_size: Tuple[int, int] = (-1, -1),
    ):
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.d_model  # 嵌入维度

        self.self_attn = AudioEncoderAttention(  # 自注意力层
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            window_size=attn_window_size,
            causal=causal,
        )
        self.self_attn_layer_norm = LAYER_NORM[config.ln_type](self.embed_dim)  # 自注意力层归一化

        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # FFN第一层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # FFN第二层
        self.final_layer_norm = LAYER_NORM[config.ln_type](self.embed_dim)  # FFN层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """音频编码器Transformer层前向传播"""
        residual = hidden_states  # 保存残差
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 层归一化
        hidden_states = self.self_attn(  # 自注意力
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rope_position_embeddings=rope_position_embeddings,
        )
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.final_layer_norm(hidden_states)  # FFN层归一化
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # FFN第一层+激活
        hidden_states = self.fc2(hidden_states)  # FFN第二层
        hidden_states = residual + hidden_states  # 残差连接

        return hidden_states


class AudioEncoder(nn.Module):
    """音频编码器：卷积+Transformer+量化"""
    def __init__(
        self,
        config: MiMoAudioTokenizerConfig,
    ):
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置
        self.max_source_positions = (  # 最大源位置数
            config.max_audio_seconds * config.sampling_rate // config.hop_length
        ) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 嵌入缩放
        self.skip_layer_idx = config.encoder_skip_layer_id  # 跳跃连接层索引

        self.conv1 = nn.Conv1d(  # 第一个卷积层
            config.n_mels,  # 输入通道：梅尔频率数
            config.d_model,  # 输出通道：模型维度
            kernel_size=config.kernel_size,  # 卷积核大小
            padding=1,  # 填充
        )
        self.conv2 = nn.Conv1d(  # 第二个卷积层（带下采样）
            config.d_model,  # 输入通道
            config.d_model,  # 输出通道
            kernel_size=config.kernel_size,  # 卷积核大小
            stride=config.stride_size,  # 步幅（下采样）
            padding=1,  # 填充
        )

        self.position_embedding = AudioRotaryEmbedding(  # 旋转位置编码
            config.rope_theta,  # RoPE基频
            config.d_model // config.encoder_attention_heads,  # 头维度
            self.max_source_positions,  # 最大源位置
            config.rope_type,  # RoPE类型
        )

        attn_window_sizes = []  # 每层的注意力窗口大小
        if config.hybrid_attention:  # 混合注意力模式
            for i in range(config.encoder_layers):  # 遍历每层
                if i % config.swa_per_block < config.swa_per_block - 1:  # 非最后一层用窗口
                    attn_window_sizes.append(tuple(config.encoder_attn_window_size))
                else:  # 最后一层用全局注意力
                    attn_window_sizes.append((-1, -1))
        else:  # 非混合模式：所有层用相同窗口
            attn_window_sizes = [
                tuple(config.encoder_attn_window_size)
            ] * config.encoder_layers

        self.layers = nn.ModuleList(  # 编码器层列表
            [
                AudioEncoderTransformerLayer(
                    config=config,
                    causal=config.encoder_causal,
                    attn_window_size=attn_window_sizes[i],
                )
                for i in range(config.encoder_layers)
            ]
        )

        self.layer_norm = LAYER_NORM[config.ln_type](config.d_model)  # 最终层归一化

        if config.avg_pooler != 1:  # 如果使用平均池化下采样
            self.down_sample_layer = nn.Sequential(  # 下采样层
                nn.Conv1d(
                    config.d_model,
                    config.d_model,
                    config.avg_pooler,  # 池化核大小
                    config.avg_pooler,  # 池化步幅
                    bias=False,
                ),
                nn.GELU(),  # GELU激活
            )
            self.down_sample_norm = LAYER_NORM[config.ln_type](config.d_model)  # 下采样归一化
        else:  # 不使用下采样
            self.down_sample_layer = None

        if config.num_quantizers != 0:  # 如果使用量化器
            self.quantizer = ResidualVectorQuantizer(  # 残差向量量化器
                dimension=config.d_model,
                n_q=config.num_quantizers,
                bins=config.codebook_size,
                threshold_ema_dead_code=config.threshold_ema_dead_code,
            )
        else:  # 不使用量化器
            self.quantizer = None

    def get_features(self, input_features, output_length):
        """提取音频特征：卷积+Transformer+可选下采样"""
        input_features = input_features.to(self.conv1.weight)  # 转换数据类型和设备
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # 第一个卷积+GELU
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # 第二个卷积+GELU
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # 转置：[B, D, T] -> [B, T, D]
        bsz, tgt_len, _ = inputs_embeds.size()  # 获取维度
        hidden_states = inputs_embeds  # 初始化隐藏状态

        position_ids = get_position_ids(output_length).long().to(input_features.device)  # 获取位置ID
        rope_position_embeddings = self.position_embedding(input_features, position_ids)  # 计算RoPE

        attention_mask, unpacking_index = get_sequence_mask(  # 获取序列掩码
            hidden_states, output_length
        )
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(  # 打包：移除填充
            torch.sum(output_length), self.config.d_model
        )

        cu_seqlens = F.pad(  # 计算累计序列长度
            torch.cumsum(output_length, dim=0), (1, 0), "constant", 0
        ).to(device=hidden_states.device, dtype=torch.int32)
        max_seqlen = torch.max(output_length).to(torch.int32).item()  # 最大序列长度

        skip_connect_hidden_states = 0.0  # 跳跃连接状态
        for idx, encoder_layer in enumerate(self.layers):  # 遍历编码器层
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                rope_position_embeddings=rope_position_embeddings,
            )
            if (self.skip_layer_idx is not None) and idx == self.skip_layer_idx - 1:  # 跳跃连接
                skip_connect_hidden_states = hidden_states.clone()  # 保存跳跃连接状态

        hidden_states += skip_connect_hidden_states  # 加上跳跃连接
        hidden_states = self.layer_norm(hidden_states)  # 层归一化

        if self.down_sample_layer is not None:  # 如果有下采样层
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(  # 解包
                bsz, tgt_len, self.config.d_model
            )
            if hidden_states.size(1) % self.config.avg_pooler:  # 需要填充
                pad_len = (
                    self.config.avg_pooler
                    - hidden_states.size(1) % self.config.avg_pooler
                )
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, 0, 0, pad_len), mode="constant", value=0.0
                )  # 填充
                tgt_len += pad_len  # 更新目标长度
            tgt_len = tgt_len // self.config.avg_pooler  # 更新目标长度
            hidden_states = self.down_sample_layer(hidden_states.transpose(1, 2))  # 卷积下采样
            output_length = (  # 更新输出长度
                output_length // self.config.avg_pooler
                + (output_length % self.config.avg_pooler != 0).int()
            )
            hidden_states = hidden_states.transpose(1, 2)  # 转置回来
            attention_mask, unpacking_index = get_sequence_mask(  # 重新计算掩码
                hidden_states, output_length
            )
            hidden_states = torch.masked_select(hidden_states, attention_mask).view(  # 重新打包
                torch.sum(output_length), self.config.d_model
            )
            hidden_states = self.down_sample_norm(hidden_states)  # 下采样归一化

        return (  # 返回特征、输出长度、掩码、解包索引、目标长度、批次大小
            hidden_states,
            output_length,
            attention_mask,
            unpacking_index,
            tgt_len,
            bsz,
        )

    def get_output_length(self, mel_len):
        """根据梅尔频谱长度计算输出长度"""
        tgt_len = mel_len + 3 - self.config.kernel_size  # 第一个卷积后的长度
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1  # 第二个卷积后的长度

    @torch.no_grad()  # 禁用梯度计算
    def encode(
        self,
        input_features,
        input_lens=None,
        output_length=None,
        return_codes_only=False,
        n_q=None,
        use_quantizer=True,
    ):
        """编码音频特征"""
        if output_length is None:  # 如果没有输出长度
            output_length = self.get_output_length(input_lens)  # 计算输出长度
        input_features = unpack_hidden_states(input_features, input_lens)  # 解包输入
        hidden_states, output_length, attention_mask, unpacking_index, tgt_len, bsz = (
            self.get_features(
                input_features=input_features.transpose(1, 2),
                output_length=output_length,
            )
        )

        dtype = hidden_states.dtype  # 保存原始数据类型
        if use_quantizer and self.quantizer is not None:  # 如果使用量化器
            self.quantizer.float()  # 量化器使用float32
            codes = self.quantizer.encode(hidden_states.float(), n_q=n_q)  # 编码
            if return_codes_only:  # 只返回码本编码
                return codes, output_length
            hidden_states = self.quantizer.decode(codes)  # 解码获取量化后的隐藏状态
            hidden_states = hidden_states.to(dtype)  # 恢复原始数据类型
        else:  # 不使用量化器
            codes = None

        hidden_states_packed = hidden_states.clone()  # 打包的隐藏状态副本
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(  # 解包
            bsz, tgt_len, self.config.d_model
        )
        hidden_states = torch.where(attention_mask, hidden_states, 0)  # 应用掩码
        return hidden_states, hidden_states_packed, output_length, codes  # 返回隐藏状态、打包状态、输出长度和编码

    @torch.no_grad()  # 禁用梯度计算
    def decode_vq(self, codes):
        """解码向量量化编码"""
        self.quantizer.float()  # 量化器使用float32
        return self.quantizer.decode(codes)  # 解码


class MiMoAudioTokenizer(PreTrainedModel):
    """MiMo音频分词器：将音频编码为离散码本表示"""
    config_class = MiMoAudioTokenizerConfig  # 配置类

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__(config)  # 调用父类初始化
        self.config = config  # 保存配置
        self.sampling_rate = config.sampling_rate  # 采样率
        self.encoder = AudioEncoder(config=config)  # 音频编码器
        self.downsample_rate = int(config.hop_length * 2 * config.avg_pooler)  # 下采样率

    def get_output_length(self, mel_len):
        """根据梅尔频谱长度计算输出长度"""
        tgt_len = mel_len + 3 - self.config.kernel_size  # 第一个卷积后的长度
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1  # 第二个卷积后的长度

    @torch.no_grad()  # 禁用梯度计算
    def encode(self, mels, input_lens, use_quantizer=True):
        """编码梅尔频谱为隐藏状态和码本编码"""
        input_features = mels  # 梅尔频谱特征
        encoder_output_length = self.get_output_length(input_lens)  # 计算编码器输出长度
        hidden_states, hidden_states_packed, encoder_output_length, codes = (
            self.encoder.encode(
                input_features, input_lens=input_lens, use_quantizer=use_quantizer
            )
        )
        return hidden_states, hidden_states_packed, encoder_output_length, codes  # 返回编码结果


def group_by_length(features: torch.Tensor, lengths: torch.Tensor, max_length: int):
    """按长度分组特征，每组总长度不超过max_length"""
    if features.size(0) != lengths.sum().item():  # 检查特征数和长度总和是否匹配
        raise ValueError(
            f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}"
        )

    split_points = []  # 分割点列表
    current_sum = 0  # 当前组总长度

    for i, seq_len in enumerate(lengths):  # 遍历每个序列长度
        if current_sum + seq_len > max_length and current_sum > 0:  # 超过最大长度
            split_points.append(i)  # 添加分割点
            current_sum = seq_len.item()  # 新组从当前序列开始
        else:  # 未超过
            current_sum += seq_len.item()  # 加入当前组

    # Convert split points to group sizes  # 将分割点转换为组大小
    group_sizes = []  # 组大小列表
    prev = 0  # 前一个分割点
    for point in split_points:  # 遍历分割点
        group_sizes.append(point - prev)  # 计算组大小
        prev = point
    if prev < len(lengths):  # 最后一组
        group_sizes.append(len(lengths) - prev)

    len_groups = torch.split(lengths, group_sizes)  # 按组大小分割长度
    feature_sizes = [group.sum().item() for group in len_groups]  # 每组的特征数
    feature_groups = torch.split(features, feature_sizes)  # 按特征数分割特征

    return feature_groups, len_groups  # 返回特征组和长度组


@torch.no_grad()  # 禁用梯度计算
def encode_batch(
    audio_tokenizer_encoder,
    input_features: torch.Tensor,
    input_lens: torch.Tensor,
    max_length: int = 256000,
):
    """批量编码音频特征"""
    feature_groups, len_groups = group_by_length(input_features, input_lens, max_length)  # 按长度分组

    encoded_parts = []  # 编码结果列表
    for features, lengths in zip(feature_groups, len_groups):  # 遍历每组
        codes, _ = audio_tokenizer_encoder.encode(  # codes are also packed  # 编码，码本编码也是打包格式
            input_features=features, input_lens=lengths, return_codes_only=True
        )
        encoded_parts.append(codes)  # 添加编码结果

    return torch.cat(encoded_parts, dim=-1)  # 拼接所有编码结果


def _segment_lengths_for_mel(mel: torch.Tensor, segment_size: int):
    """将梅尔频谱分割为指定大小的段"""
    """Split mel into segments of segment_size with a possible shorter remainder."""
    input_len = mel.size(0)  # 输入长度
    segs = [segment_size] * (input_len // segment_size)  # 完整段
    if input_len % segment_size > 0:  # 剩余段
        segs.append(input_len % segment_size)
    return segs


@torch.no_grad()  # 禁用梯度计算
def tokenize_audio_batch(mels, audio_tokenizer_encoder, segment_size=6000, device=None):
    """批量分词多个梅尔频谱"""
    """
    Tokenize multiple mels in one encode_batch call.
    Returns list of code tensors, each [T_i, C] for that mel.
    """
    if not mels:  # 空列表
        return []
    if device is None:  # 未指定设备
        device = next(audio_tokenizer_encoder.parameters()).device  # 使用编码器设备
    # Build segment lengths per mel  # 构建每个梅尔频谱的段长度
    input_len_seg_per_mel = [_segment_lengths_for_mel(m, segment_size) for m in mels]  # 每个梅尔的段长度
    input_lens_flat = [s for segs in input_len_seg_per_mel for s in segs]  # 展平段长度
    input_features = torch.cat([m.to(device) for m in mels], dim=0)  # 拼接所有梅尔频谱
    input_lens_t = torch.tensor(input_lens_flat, dtype=torch.long, device=device)  # 段长度张量
    codes_packed = encode_batch(  # 批量编码
        audio_tokenizer_encoder,
        input_features=input_features,
        input_lens=input_lens_t,
    )
    codes = codes_packed.transpose(0, 1).detach()  # [total_code_T, C]  # 转置并分离梯度
    # Code length per mel: must match encoder's actual output (get_output_length + optional avg_pooler downsampling)  # 每个梅尔的码本长度
    code_lengths = []  # 码本长度列表
    for segs in input_len_seg_per_mel:  # 遍历每个梅尔的段
        out_len = audio_tokenizer_encoder.get_output_length(
            torch.tensor(segs, dtype=torch.long, device=device)
        )
        if getattr(audio_tokenizer_encoder, "down_sample_layer", None) is not None:  # 有下采样层
            avg = audio_tokenizer_encoder.config.avg_pooler  # 池化大小
            out_len = out_len // avg + (out_len % avg != 0).long()  # 调整输出长度
        code_lengths.append(out_len.sum().item())  # 添加总长度
    code_list = torch.split(codes, code_lengths)  # 按长度分割码本
    return list(code_list)  # 返回码本列表


@dataclass
class MiMoAudioEncoderConfig:
    """MiMo音频编码器配置"""
    tokenizer_version: str = "v1"  # 分词器版本
    speech_vocab_size: str = "1025-1025-129-129-129-129-129-129"  # 语音词表大小
    speech_zeroemb_idx: str = "1024-1024-128-128-128-128-128-128"  # 语音零嵌入索引
    group_size: int = 4  # 分组大小
    audio_channels: int = 8  # 音频通道数
    input_local_layers: int = 6  # 输入局部Transformer层数
    input_local_dim: int = 1024  # 输入局部Transformer维度
    input_full_attention: bool = True  # 是否使用全注意力
    input_local_attn_heads: int = 64  # 输入局部注意力头数
    input_local_head_dim: int = 16  # 输入局部头维度
    input_local_intermediate_size: int = 4096  # 输入局部中间层大小
    input_local_hidden_dropout: float = 0.0  # 输入局部dropout
    out_hidden_size: int = 4096  # mimo vl hidden dim  # 输出隐藏维度（与VL模型对齐）
    rope_theta: float = 640000.0  # RoPE基频
    partial_rotary_factor: float = 0.334  # 部分旋转因子
    projection_layers: int = 1  # 投影层数
    add_post_norm: bool = False  # 是否添加后归一化
    audio_segment_size: int = 6000  # 音频段大小


class AudioProjection(nn.Module):
    """音频投影层：将音频嵌入投影到LLM维度"""
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.input_size = input_size  # 输入大小
        self.hidden_size = hidden_size  # 隐藏大小
        self.output_size = output_size  # 输出大小
        self.mlp = nn.Sequential(  # MLP序列
            nn.Linear(self.input_size, self.hidden_size, bias=False),  # 第一层
            nn.GELU(),  # GELU激活
            nn.Linear(self.hidden_size, self.output_size, bias=False),  # 第二层
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """音频投影前向传播"""
        return self.mlp(x)


class MiMoV2AudioConfig:
    """MiMo V2音频配置，支持V1和V2两种音频编码器"""
    def __init__(
        self,
        speech_vocab_size: str | int = "1280",  # 语音词表大小
        speech_lm_head_sizes: str | int | None = None,  # 语音LM头大小
        speech_zeroemb_idx: str | int = "1280",  # 语音零嵌入索引
        delay_pattern: str = "0-1-2-3-4-5-6-7-7-7-7-7-7-7-7-7-7-7-7-7",  # 延迟模式
        group_size: int = 4,  # 分组大小
        audio_channels: int = 20,  # 音频通道数
        input_local_dim: int = 1024,  # 输入局部维度
        input_local_layers: int = 6,  # 输入局部层数
        input_local_attn_heads: int = 16,  # 输入局部注意力头数
        input_local_intermediate_size: int = 4096,  # 输入局部中间层大小
        input_local_rope_theta: float = 640000.0,  # 输入局部RoPE基频
        input_local_partial_rotary_factor: float = 1.0,  # 输入局部部分旋转因子
        output_local_dim: int = 1024,  # 输出局部维度
        output_local_layers: int = 6,  # 输出局部层数
        output_local_attn_heads: int = 16,  # 输出局部注意力头数
        output_local_intermediate_size: int = 4096,  # 输出局部中间层大小
        output_local_rope_theta: float = 640000.0,  # 输出局部RoPE基频
        output_local_partial_rotary_factor: float = 1.0,  # 输出局部部分旋转因子
        input_projection_layers: int = 2,  # 输入投影层数
        output_projection_layers: int = 2,  # 输出投影层数
        add_encoder_post_norm: bool = True,  # 是否添加编码器后归一化
        audio_config: dict = None,  # 音频配置字典
        **kwargs,
    ):
        for key, value in kwargs.items():  # 设置额外参数
            setattr(self, key, value)

        if audio_config is not None:  # 如果提供了音频配置字典
            self._load_from_audio_config(audio_config)  # 从字典加载
        else:  # 直接设置参数
            self.speech_vocab_size = speech_vocab_size  # 语音词表大小
            self.speech_lm_head_sizes = (
                speech_lm_head_sizes
                if speech_lm_head_sizes is not None
                else speech_vocab_size
            )  # 语音LM头大小
            self.speech_zeroemb_idx = speech_zeroemb_idx  # 语音零嵌入索引
            self.delay_pattern = delay_pattern  # 延迟模式
            self.group_size = group_size  # 分组大小
            self.audio_channels = audio_channels  # 音频通道数
            self.input_local_dim = input_local_dim  # 输入局部维度
            self.input_local_layers = input_local_layers  # 输入局部层数
            self.input_local_attn_heads = input_local_attn_heads  # 输入局部注意力头数
            self.input_local_intermediate_size = input_local_intermediate_size  # 输入局部中间层大小
            self.input_local_rope_theta = input_local_rope_theta  # 输入局部RoPE基频
            self.input_local_partial_rotary_factor = input_local_partial_rotary_factor  # 输入局部部分旋转因子
            self.output_local_dim = output_local_dim  # 输出局部维度
            self.output_local_layers = output_local_layers  # 输出局部层数
            self.output_local_attn_heads = output_local_attn_heads  # 输出局部注意力头数
            self.output_local_intermediate_size = output_local_intermediate_size  # 输出局部中间层大小
            self.output_local_rope_theta = output_local_rope_theta  # 输出局部RoPE基频
            self.output_local_partial_rotary_factor = output_local_partial_rotary_factor  # 输出局部部分旋转因子
            self.input_projection_layers = input_projection_layers  # 输入投影层数
            self.output_projection_layers = output_projection_layers  # 输出投影层数
            self.add_encoder_post_norm = add_encoder_post_norm  # 是否添加编码器后归一化

        self._attn_implementation_internal = "sdpa"  # 注意力实现

    def _load_from_audio_config(self, audio_config: dict):
        """从audio_config字典加载音频参数"""
        """Load audio parameters from audio_config dict in checkpoint.

        Uses naming that matches megatron2hf conversion output to minimize manual mapping.
        """
        self.group_size = audio_config.get("group_size", 4)  # 分组大小
        self.audio_channels = audio_config.get("audio_channels", 20)  # 音频通道数
        self.speech_vocab_size = audio_config.get("speech_vocab_size", "1280")  # 语音词表大小
        self.speech_lm_head_sizes = audio_config.get(
            "speech_lm_head_sizes", self.speech_vocab_size
        )  # 语音LM头大小
        self.speech_zeroemb_idx = audio_config.get("speech_zeroemb_idx", "1280")  # 语音零嵌入索引
        # Per-channel decode delays; len must equal audio_channels.  # 每通道解码延迟，长度必须等于audio_channels
        self.delay_pattern = audio_config.get(
            "audio_output_delay_pattern", "0-1-2-3-4-5-6-7-7-7-7-7-7-7-7-7-7-7-7-7"
        )  # 延迟模式

        self.input_local_dim = audio_config.get("input_local_dim", 1024)  # 输入局部维度
        self.input_local_layers = audio_config.get("input_local_layers", 6)  # 输入局部层数
        self.input_local_attn_heads = audio_config.get("input_local_attn_heads", 16)  # 输入局部注意力头数
        self.input_local_intermediate_size = audio_config.get(
            "input_local_intermediate_size", 4096
        )  # 输入局部中间层大小
        self.input_local_rope_theta = audio_config.get(
            "input_local_rope_theta", 640000.0
        )  # 输入局部RoPE基频
        self.input_local_partial_rotary_factor = audio_config.get(
            "input_local_partial_rotary_factor", 1.0
        )  # 输入局部部分旋转因子

        self.output_local_dim = audio_config.get("output_local_dim", 1024)  # 输出局部维度
        self.output_local_layers = audio_config.get("output_local_layers", 6)  # 输出局部层数
        self.output_local_attn_heads = audio_config.get("output_local_attn_heads", 16)  # 输出局部注意力头数
        self.output_local_intermediate_size = audio_config.get(
            "output_local_intermediate_size", 4096
        )  # 输出局部中间层大小
        self.output_local_rope_theta = audio_config.get(
            "output_local_rope_theta", 640000.0
        )  # 输出局部RoPE基频
        self.output_local_partial_rotary_factor = audio_config.get(
            "output_local_partial_rotary_factor", 1.0
        )  # 输出局部部分旋转因子

        self.input_projection_layers = audio_config.get("input_projection_layers", 2)  # 输入投影层数
        self.output_projection_layers = audio_config.get("output_projection_layers", 2)  # 输出投影层数

        self.add_encoder_post_norm = audio_config.get("add_encoder_post_norm", True)  # 编码器后归一化

    def _parse_maybe_list(self, value: str | int, length: int) -> list[int]:
        """解析可能是连字符分隔列表的值"""
        if isinstance(value, str) and "-" in value:  # 连字符分隔的字符串
            return [int(s) for s in value.split("-")]  # 分割并转换为整数列表
        return [int(value)] * length  # 扩展为指定长度的列表

    def parsed_speech_empty_ids(self):
        """解析语音空嵌入ID"""
        return self._parse_maybe_list(self.speech_zeroemb_idx, self.audio_channels)

    def parsed_speech_vocab_sizes(self):
        """解析语音词表大小"""
        return self._parse_maybe_list(self.speech_vocab_size, self.audio_channels)

    def parsed_speech_lm_head_sizes(self):
        """解析语音LM头大小"""
        return self._parse_maybe_list(self.speech_lm_head_sizes, self.audio_channels)

    def parsed_delay_pattern(self):
        """解析延迟模式"""
        return self._parse_maybe_list(self.delay_pattern, self.audio_channels)

    def input_local_config(self):
        """创建输入局部Transformer的Qwen2配置"""
        """Create config for input local transformer."""
        config = Qwen2Config()  # 创建Qwen2配置
        for attr in dir(self):  # 遍历属性
            if not attr.startswith("_") and hasattr(config, attr):  # 跳过私有属性
                setattr(config, attr, getattr(self, attr))  # 复制属性

        config.hidden_size = self.input_local_dim  # 隐藏大小
        config.num_hidden_layers = self.input_local_layers  # 层数
        config.num_attention_heads = self.input_local_attn_heads  # 注意力头数
        config.num_key_value_heads = self.input_local_attn_heads  # KV头数
        config.head_dim = getattr(  # 头维度
            self,
            "input_local_head_dim",
            self.input_local_dim // self.input_local_attn_heads,
        )
        config.intermediate_size = self.input_local_intermediate_size  # 中间层大小
        config.rope_theta = self.input_local_rope_theta  # RoPE基频
        config.partial_rotary_factor = self.input_local_partial_rotary_factor  # 部分旋转因子
        config._attn_implementation_internal = "sdpa"  # 注意力实现

        return config

    def output_local_config(self):
        """创建输出局部Transformer的Qwen2配置"""
        """Create config for output local transformer."""
        config = Qwen2Config()  # 创建Qwen2配置
        for attr in dir(self):  # 遍历属性
            if not attr.startswith("_") and hasattr(config, attr):  # 跳过私有属性
                setattr(config, attr, getattr(self, attr))  # 复制属性

        config.hidden_size = self.output_local_dim  # 隐藏大小
        config.num_hidden_layers = self.output_local_layers  # 层数
        config.num_attention_heads = self.output_local_attn_heads  # 注意力头数
        config.num_key_value_heads = self.output_local_attn_heads  # KV头数
        config.head_dim = self.output_local_dim // self.output_local_attn_heads  # 头维度
        config.intermediate_size = self.output_local_intermediate_size  # 中间层大小
        config.rope_theta = self.output_local_rope_theta  # RoPE基频
        config.partial_rotary_factor = self.output_local_partial_rotary_factor  # 部分旋转因子
        config._attn_implementation_internal = "sdpa"  # 注意力实现

        return config


class MiMoAudioEncoder(nn.Module):
    """MiMo音频编码器：将音频分词并编码为LLM可用的嵌入"""
    config: MiMoAudioEncoderConfig

    def __init__(self, config):
        super().__init__()  # 调用父类初始化
        if not isinstance(config, MiMoV2AudioConfig):  # 如果不是V2配置
            config_dict = (
                vars(config) if hasattr(config, "__dict__") else config.__dict__
            )
            config = MiMoV2AudioConfig(**config_dict)  # 转换为V2配置
        self.config = config  # 保存配置
        self.server_args = get_global_server_args()  # 获取服务器参数
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder  # 是否使用数据并行编码器
        self.speech_empty_ids = self.parsed_speech_empty_ids()  # 语音空嵌入ID
        self.audio_channels = config.audio_channels  # 音频通道数
        self.audio_group_size = config.group_size  # 音频分组大小
        self.audio_segment_size = config.audio_segment_size  # 音频段大小
        speech_vocab_size = self._parse_maybe_list(
            self.config.speech_vocab_size, self.config.audio_channels
        )  # 解析语音词表大小
        input_local_config = Qwen2Config(  # 输入局部Transformer配置
            hidden_size=self.config.input_local_dim,
            num_hidden_layers=self.config.input_local_layers,
            num_attention_heads=self.config.input_local_attn_heads,
            num_key_value_heads=self.config.input_local_attn_heads,
            intermediate_size=self.config.input_local_intermediate_size,
            attention_dropout=self.config.input_local_hidden_dropout,
            rope_theta=self.config.rope_theta,
            partial_rotary_factor=self.config.partial_rotary_factor,
        )
        input_local_config.head_dim = self.config.input_local_head_dim  # 设置头维度

        self.input_local_transformer = Qwen2Model(input_local_config)  # 输入局部Transformer

        if not self.config.add_post_norm:  # 如果不添加后归一化
            self.input_local_transformer.norm = nn.Identity()  # 使用恒等映射

        self.speech_embeddings = nn.ModuleList(  # 语音嵌入层列表
            [
                nn.Embedding(
                    speech_vocab_size[i],  # 词表大小
                    self.config.input_local_dim,  # 嵌入维度
                    padding_idx=self.speech_empty_ids[i],  # 填充索引
                )
                for i in range(self.config.audio_channels)
            ]
        )

        if self.config.projection_layers == 1:  # 单层投影
            self.projection = nn.Linear(
                self.config.input_local_dim * self.config.group_size,
                self.config.out_hidden_size,
                bias=False,
            )
        elif self.config.projection_layers == 2:  # 两层投影
            self.projection = AudioProjection(
                self.config.input_local_dim * self.config.group_size,
                self.config.input_local_dim * self.config.group_size * 4,
                self.config.out_hidden_size,
            )
        else:  # 无效投影层数
            raise ValueError(
                f"Invalid projection layers: {self.config.projection_layers}"
            )

        model_path = self.server_args.model_path  # 模型路径
        if not os.path.isdir(model_path):  # 如果本地路径不存在
            from huggingface_hub import snapshot_download  # 导入HuggingFace下载工具

            model_path = snapshot_download(
                model_path,
                allow_patterns=["audio_tokenizer/*"],  # 只下载音频分词器
            )
        audio_tokenizer_path = os.path.join(model_path, "audio_tokenizer")  # 音频分词器路径
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")  # 当前CUDA设备
        self.audio_tokenizer = self._load_audio_tokenizer(audio_tokenizer_path, dev)  # 加载音频分词器

    @staticmethod
    def _load_audio_tokenizer(path: str, device: torch.device) -> MiMoAudioTokenizer:
        """手动加载MiMoAudioTokenizer以避免新版transformers兼容性问题"""
        """Load MiMoAudioTokenizer manually to avoid new-transformers compat issues."""
        import json  # 导入JSON模块
        import os  # 导入OS模块

        from safetensors.torch import load_file  # 导入safetensors加载器

        config_path = os.path.join(path, "config.json")  # 配置文件路径
        with open(config_path) as f:  # 打开配置文件
            config_dict = json.load(f)  # 加载配置字典
        config = MiMoAudioTokenizer.config_class(**config_dict)  # 创建配置对象
        model = MiMoAudioTokenizer(config)  # 创建模型
        # Load weights from safetensors or pytorch bin  # 从safetensors或pytorch bin加载权重
        safetensors_path = os.path.join(path, "model.safetensors")  # safetensors路径
        bin_path = os.path.join(path, "pytorch_model.bin")  # bin路径
        if os.path.exists(safetensors_path):  # 优先使用safetensors
            state_dict = load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):  # 其次使用bin
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:  # 找不到权重文件
            raise FileNotFoundError(
                f"No model weights found in {path} "
                "(expected model.safetensors or pytorch_model.bin)"
            )
        model.load_state_dict(state_dict, strict=False)  # 加载权重
        model = model.to(device=device, dtype=torch.bfloat16)  # 转换设备和数据类型
        model.eval()  # 设置为评估模式
        model.requires_grad_(False)  # 禁用梯度
        return model

    def parsed_speech_empty_ids(self):
        """解析语音空嵌入ID"""
        return self._parse_maybe_list(
            self.config.speech_zeroemb_idx, self.config.audio_channels
        )

    def _parse_maybe_list(self, value: str | int, length: int) -> List[int]:
        """解析可能是连字符分隔列表的值"""
        if isinstance(value, str) and "-" in value:  # 连字符分隔
            return [int(s) for s in value.split("-")]
        return [int(value)] * length  # 扩展为列表

    # adapted from mimo-audio  # 适配自mimo-audio
    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor):
        """应用输入局部Transformer"""
        output = self.input_local_transformer(
            inputs_embeds=speech_embeddings,
            return_dict=True,
            is_causal=not self.config.input_full_attention,  # for SDPA  # 为SDPA设置因果性
        )
        return output.last_hidden_state  # [T//group_size, group_size, input_local_dim]

    def apply_speech_embeddings(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """将音频码本编码转换为语音嵌入"""
        num_segments = audio_codes.shape[0]  # 段数
        _audio_embeddings = torch.zeros(  # 初始化嵌入张量
            (num_segments, self.config.group_size, self.config.input_local_dim),
            dtype=next(self.speech_embeddings[0].parameters()).dtype,
            device=audio_codes.device,
        )
        for i in range(self.config.audio_channels):  # 遍历每个音频通道
            _audio_embeddings.add_(self.speech_embeddings[i](audio_codes[:, :, i]))  # 累加各通道嵌入
        return _audio_embeddings

    def process_audio(self, audio):
        """处理音频：填充并分组"""
        T = audio.shape[0]  # 音频时间步数
        audio = audio[:, : self.audio_channels]  # 截取音频通道
        padded_T = (  # 计算填充后的长度
            (T + self.audio_group_size - 1)
            // self.audio_group_size
            * self.audio_group_size
        )
        padded_audio = torch.cat(  # 拼接填充
            [
                audio,
                torch.zeros(
                    padded_T - T,
                    self.audio_channels,
                    dtype=torch.int32,
                    device=audio.device,
                )
                + audio[-1, :],  # 用最后一个token填充
            ],
            dim=0,
        )  # pad using the last embedding  # 用最后一个嵌入填充
        padded_audio = padded_audio.reshape(  # 重塑为分组格式
            padded_T // self.audio_group_size,
            self.audio_group_size,
            self.audio_channels,
        )
        return padded_audio

    def get_audio_feature(self, items) -> torch.Tensor:
        """获取音频特征：分词+嵌入+局部Transformer+投影"""
        # items: already audio-only MultimodalDataItem list from caller.  # items：调用者提供的仅含音频的MultimodalDataItem列表
        # Each item.feature is either one mel tensor or a list of mel tensors (e.g. long audio split into chunks).  # 每个item.feature是一个mel张量或mel张量列表
        all_mels = []  # 所有梅尔频谱
        for item in items:  # 遍历每个数据项
            f = item.feature  # 获取特征
            if isinstance(f, (list, tuple)):  # 如果是列表（长音频分段）
                all_mels.extend(f)
            else:  # 单个mel张量
                all_mels.append(f)
        if not all_mels:  # 如果没有音频
            device = next(self.projection.parameters()).device  # 获取设备
            dtype = next(self.projection.parameters()).dtype  # 获取数据类型
            return torch.empty(
                0, self.config.out_hidden_size, device=device, dtype=dtype
            )
        # Batch tokenize: one encode_batch call for all mels  # 批量分词：一个encode_batch调用处理所有mel
        device = next(self.audio_tokenizer.encoder.parameters()).device  # 分词器设备
        code_list = tokenize_audio_batch(
            all_mels,
            self.audio_tokenizer.encoder,
            segment_size=self.audio_segment_size,
            device=device,
        )
        codecs_to_concat = []  # 待拼接的码本列表
        for codecs in code_list:  # 遍历每段码本
            padded_codes = self.process_audio(
                codecs
            )  # [T//group_size, group_size, audio_channels]  # 处理音频：填充并分组
            codecs_to_concat.append(padded_codes)
        audio_codes = torch.cat(
            codecs_to_concat, dim=0
        )  # [T//group_size, group_size, audio_channels]  # 拼接所有码本

        _audio_embeddings = self.apply_speech_embeddings(audio_codes)  # 应用语音嵌入
        audio_embeds = self.apply_input_local_transformer(
            _audio_embeddings
        )  #  [T//group_size,  group_size, input_local_dim]  # 应用输入局部Transformer
        B = audio_embeds.shape[0]  # 批次大小
        audio_embeds = self.projection(audio_embeds.reshape(B, -1))  # 投影到LLM维度
        return audio_embeds
