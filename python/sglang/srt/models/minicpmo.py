# MiniCPM-o多模态模型实现（视觉+音频+TTS）
# 该模块实现了MiniCPM-o多模态模型，支持视觉理解、音频理解和文本转语音（TTS）
# 核心组件：ConvNeXtBlock、DVAEDecoder、GFSQ、DVAE、ConditionalChatTTS、
# MiniCPMWhisperEncoderLayer、MiniCPMWhisperEncoder、MultiModalProjector、MiniCPMO
# 音频处理流程：Whisper编码器 -> 投影层 -> 平均池化 -> 嵌入
# TTS流程：LLM隐藏状态 -> 投影 -> 条件ChatTTS生成 -> DVAE解码 -> 梅尔频谱

# Copied and adapted from: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py

# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Inference-only MiniCPM-o model compatible with HuggingFace weights."""

import math  # 导入数学模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union  # 导入类型注解

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import torch.nn.functional as F  # 导入PyTorch函数式接口
import torch.nn.utils.parametrize as P  # 导入参数化工具
import torch.types  # 导入PyTorch类型
from torch import nn  # 导入PyTorch神经网络模块
from torch.nn.utils import parametrizations  # 导入参数化工具
from tqdm import tqdm  # 导入进度条
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel  # 导入Transformers组件
from transformers.activations import ACT2FN  # 导入激活函数映射
from transformers.cache_utils import DynamicCache, EncoderDecoderCache  # 导入缓存工具
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput  # 导入模型输出类
from transformers.models.whisper.modeling_whisper import (  # 导入Whisper组件
    WhisperAttention,  # Whisper注意力
    WhisperConfig,  # Whisper配置
    WhisperEncoder,  # Whisper编码器
)

from sglang.srt.layers.quantization import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternTokenPairs,  # 多模态token对填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputFormat,  # 多模态输入格式
    MultimodalInputs,  # 多模态输入
    flatten_nested_list,  # 展平嵌套列表
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.utils import set_default_torch_dtype  # 导入默认数据类型设置
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.idefics2 import Idefics2VisionTransformer  # 导入Idefics2视觉Transformer
from sglang.srt.models.minicpmv import MiniCPMBaseModel, Resampler2_5  # 导入MiniCPM-V基础模型和重采样器
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入Qwen2因果语言模型
from sglang.srt.utils import get_device, logger  # 导入工具函数

try:  # 尝试导入TTS依赖
    from transformers import LogitsWarper
    from vector_quantize_pytorch import GroupedResidualFSQ

    _tts_deps = True  # TTS依赖可用
except:  # TTS依赖不可用
    LogitsWarper = None
    _tts_deps = False


def apply_spk_emb(
    input_ids: torch.Tensor = None,
    spk_emb: torch.Tensor = None,
    input_embeds: torch.Tensor = None,
    spk_emb_token_id: int = 0,
    num_spk_embs: int = 1,
):
    """将说话人嵌入替换到输入嵌入的对应位置"""
    """
    Replace consecutive `num_spk_embs` speaker embedding placeholders in input_embeds with pre-prepared speaker embeddings. This is an in-place replacement, no new tensor is created, so no value is returned.

    Args:
        input_ids (torch.Tensor): Input ID tensor, shape [batch_size, seq_len_max]
        spk_emb (torch.Tensor): Speaker embedding tensor, shape [batch_size, num_spk_emb, hidden_dim]
        input_embeds (torch.Tensor): Input embedding tensor, shape [batch_size, seq_len_max, hidden_dim]
        spk_emb_token_id (int): ID of the speaker embedding token
        num_spk_embs (int): Number of speaker embeddings

    Returns:
        None
    """

    batch_size = input_ids.shape[0]  # 批次大小

    for idx in range(batch_size):  # 遍历每个批次
        input_ids_ = input_ids[idx]  # [seq_len_max]  # 当前输入ID
        spk_emb_ = spk_emb[idx]  # [num_spk_emb]  # 当前说话人嵌入
        mask_ = input_ids_ == spk_emb_token_id  # [batch_size, seq_len_max]  # 说话人token掩码
        nonzero_position_idx = mask_.nonzero(as_tuple=False)  # [num_spk_emb, 1]  # 非零位置索引
        assert nonzero_position_idx.shape[0] == num_spk_embs  # 数量必须匹配
        begin_idx = nonzero_position_idx.min()  # 起始索引
        end_idx = nonzero_position_idx.max()  # 结束索引
        input_embeds[idx, begin_idx : end_idx + 1, :] = spk_emb_  # 替换嵌入

    return


@dataclass
class ConditionalChatTTSGenerationOutput(ModelOutput):
    """条件ChatTTS生成输出"""
    """
    Output class for ConditionalChatTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.

    """

    new_ids: torch.LongTensor = None  # 新生成的音频码序列
    audio_input_ids: torch.LongTensor = None  # 更新后的输入ID
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # KV缓存
    finished: bool = None  # 生成是否完成


def make_streaming_chunk_mask_generation(
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int,
    streaming_tts_text_mask: torch.Tensor,
    streaming_reserved_length: int = 300,
    streaming_audio_chunk_size: int = 50,
    streaming_text_chunk_size: int = 10,
    num_spk_emb: int = 1,
    use_spk_emb: bool = True,
) -> torch.Tensor:
    """创建流式TTS生成的因果掩码"""
    """
    In streaming audio generation, determine which `text` positions the TTS model can attend to when generating each chunk of `audio` tokens.

    This function creates a mask that allows the model to attend to a specific chunk of text
    tokens when generating each chunk of audio tokens, enabling streaming TTS generation.

    Args:
        inputs_embeds (torch.Tensor): Input embeddings tensor.
        past_seen_tokens (int): Number of tokens already seen by the model.
        streaming_tts_text_mask (torch.Tensor): Mask for the text tokens.
        streaming_reserved_length (int, optional): Number of reserved tokens for streaming. Defaults to 300.
        streaming_text_chunk_size (int, optional): Size of each text chunk. Defaults to 7.

    Returns:
        torch.Tensor: Causal mask for streaming TTS generation, shape is [batch_size=1, 1, seq_len=1, past_seen_tokens+1]

    Raises:
        AssertionError: If the batch size is not 1 (only supports batch size of 1 for inference).
    """
    assert inputs_embeds.shape[0] == 1  # 仅支持批次大小1

    dtype = inputs_embeds.dtype  # 数据类型
    device = inputs_embeds.device  # 设备
    min_dtype = torch.finfo(dtype).min  # 最小值（用于掩码）

    # Add `1` to the past seen tokens to account for new `tokens` during `generate`  # 添加1以考虑生成时的新token
    causal_mask = torch.full(
        (1, past_seen_tokens + inputs_embeds.shape[1]),
        fill_value=0,
        dtype=dtype,
        device=device,
    )  # 初始化因果掩码

    # Calculate the start of invisible text tokens  # 计算不可见文本token的起始位置
    invisible_text_tokens_start = (
        min(
            math.ceil(
                (past_seen_tokens - streaming_reserved_length)
                / streaming_audio_chunk_size
            )
            * streaming_text_chunk_size,
            streaming_reserved_length,
        )
        + 1
        + num_spk_emb * use_spk_emb
    )  # Add 1 for [Stts] and N for [spk_emb] tokens if `use_spk_emb` is True  # 加1为[Stts]和N为[spk_emb]token

    invisible_text_tokens_end = (
        streaming_reserved_length + 1 + num_spk_emb * use_spk_emb + 1
    )  # Add 1 for [Ptts] (aka `audio_bos_token_id`)  # 加1为[Ptts]

    # Set invisible text tokens to min_dtype (effectively -inf)  # 设置不可见文本token为最小值（等效-inf）
    causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype

    # Mask padding positions in the text mask  # 掩码文本掩码中的填充位置
    causal_mask[
        0, 0 : 1 + num_spk_emb * use_spk_emb + streaming_reserved_length + 1
    ].masked_fill_(streaming_tts_text_mask == 0, min_dtype)

    # Add extra dimensions for batch and heads  # 添加批次和头的额外维度
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`  # 借用自ChatTTS的DVAE
class ConvNeXtBlock(nn.Module):
    """ConvNeXt块：深度可分离卷积+LayerNorm+MLP"""
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.  # 从Vocos复制的ConvNeXt块
        super().__init__()  # 调用父类初始化
        self.dwconv = nn.Conv1d(  # 深度可分离卷积
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,  # 深度卷积
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)  # 层归一化
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # 逐点卷积1
        self.act = nn.GELU()  # GELU激活
        self.pwconv2 = nn.Linear(intermediate_dim, dim)  # 逐点卷积2
        self.coef = (  # 层缩放系数
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        """ConvNeXt块前向传播"""
        residual = x  # 保存残差

        y = self.dwconv(x)  # 深度卷积
        y.transpose_(1, 2)  # (B, C, T) -> (B, T, C)  # 转置
        x = self.norm(y)  # 层归一化
        del y
        y = self.pwconv1(x)  # 逐点卷积1
        del x
        x = self.act(y)  # 激活
        del y
        y = self.pwconv2(x)  # 逐点卷积2
        del x
        if self.coef is not None:  # 应用层缩放
            y *= self.coef
        y.transpose_(1, 2)  # (B, T, C) -> (B, C, T)  # 转置回来

        x = y + residual  # 残差连接
        del y

        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`  # 借用自ChatTTS的DVAE
class DVAEDecoder(nn.Module):
    """DVAE解码器：卷积输入+ConvNeXt块堆叠+卷积输出"""
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer=12,
        bn_dim=64,
        hidden=256,
        kernel=7,
        dilation=2,
        up=False,
    ):
        super().__init__()  # 调用父类初始化
        self.up = up  # 是否上采样
        self.conv_in = nn.Sequential(  # 输入卷积
            nn.Conv1d(idim, bn_dim, 3, 1, 1),  # 3x3卷积
            nn.GELU(),  # GELU激活
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),  # 3x3卷积
        )
        self.decoder_block = nn.ModuleList(  # ConvNeXt块列表
            [
                ConvNeXtBlock(
                    hidden,
                    hidden * 4,  # 中间维度4倍扩展
                    kernel,
                    dilation,
                )
                for _ in range(n_layer)
            ]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)  # 输出1x1卷积

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        """DVAE解码器前向传播"""
        # B, C, T  # 批次，通道，时间
        y = self.conv_in(x)  # 输入卷积
        del x
        for f in self.decoder_block:  # 逐块处理
            y = f(y, conditioning)

        x = self.conv_out(y)  # 输出卷积
        del y
        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`  # 借用自ChatTTS的DVAE
class GFSQ(nn.Module):
    """分组残差有限标量量化"""
    def __init__(
        self,
        dim: int,
        levels: List[int],
        G: int,
        R: int,
        eps=1e-5,
        transpose=True,
    ):
        super(GFSQ, self).__init__()  # 调用父类初始化
        self.quantizer = GroupedResidualFSQ(  # 分组残差FSQ量化器
            dim=dim,
            levels=list(levels),
            num_quantizers=R,
            groups=G,
        )
        self.n_ind = math.prod(levels)  # 索引总数
        self.eps = eps  # 精度
        self.transpose = transpose  # 是否转置
        self.G = G  # 分组数
        self.R = R  # 量化器数

    def _embed(self, x: torch.Tensor):
        """从索引获取嵌入"""
        if self.transpose:  # 转置
            x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)  # 重塑和重排
        feat = self.quantizer.get_output_from_indices(x)  # 从索引获取输出
        return feat.transpose_(1, 2) if self.transpose else feat

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """调用前向传播"""
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GFSQ前向传播：量化"""
        if self.transpose:  # 转置
            x.transpose_(1, 2)
        _, ind = self.quantizer(x)  # 量化获取索引
        ind = ind.permute(1, 2, 0, 3).contiguous()  # 重排
        ind = ind.view(ind.size(0), ind.size(1), -1)  # 重塑
        return ind.transpose_(1, 2) if self.transpose else ind


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`  # 借用自ChatTTS的DVAE
class DVAE(nn.Module):
    """离散变分自编码器：编码+量化+解码"""
    def __init__(
        self,
    ):
        super().__init__()  # 调用父类初始化

        coef = torch.rand(100)  # 随机初始化系数
        self.coef = nn.Parameter(coef.unsqueeze(0).unsqueeze_(2))  # 系数参数

        self.downsample_conv = nn.Sequential(  # 下采样卷积
            nn.Conv1d(100, 512, 3, 1, 1),  # 3x3卷积
            nn.GELU(),  # GELU激活
            nn.Conv1d(512, 512, 4, 2, 1),  # 4x4卷积，步幅2
            nn.GELU(),  # GELU激活
        )

        self.encoder = DVAEDecoder(  # DVAE编码器
            idim=512,
            odim=1024,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.decoder = DVAEDecoder(  # DVAE解码器
            idim=512,
            odim=512,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.out_conv = nn.Conv1d(512, 100, 3, 1, 1, bias=False)  # 输出卷积

        self.vq_layer = GFSQ(  # 分组残差FSQ量化层
            dim=1024,
            levels=(5, 5, 5, 5),
            G=2,
            R=2,
        )

    @torch.inference_mode()  # 推理模式
    def forward(
        self, inp: torch.Tensor, mode: Literal["encode", "decode"] = "decode"
    ) -> torch.Tensor:
        """DVAE前向传播：编码或解码"""
        if mode == "encode" and hasattr(self, "encoder") and self.vq_layer is not None:  # 编码模式
            mel = inp.clone()  # 克隆输入
            x: torch.Tensor = self.downsample_conv(
                torch.div(mel, self.coef.view(100, 1).expand(mel.shape), out=mel),  # 除以系数
            ).unsqueeze_(0)  # 下采样卷积
            del mel
            x = self.encoder(x)  # 编码
            ind = self.vq_layer(x)  # 量化
            del x
            return ind

        if self.vq_layer is not None:  # 有量化层
            vq_feats = self.vq_layer._embed(inp)  # 从索引获取嵌入
        else:  # 无量化层
            vq_feats = inp

        vq_feats = (  # 重排特征
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
            )
            .permute(0, 2, 3, 1)
            .flatten(2)
        )

        dec_out = self.out_conv(  # 解码+输出卷积
            self.decoder(
                x=vq_feats,
            ),
        )

        del vq_feats

        return torch.mul(dec_out, self.coef, out=dec_out)  # 乘以系数


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/processors.py`  # 借用自ChatTTS的处理器
class CustomRepetitionPenaltyLogitsProcessorRepeat:
    """自定义重复惩罚logits处理器"""
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):  # 检查惩罚值
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty  # 惩罚值
        self.max_input_ids = max_input_ids  # 最大输入ID数
        self.past_window = past_window  # 过去窗口大小

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """应用重复惩罚"""
        if input_ids.size(1) > self.past_window:  # 截取窗口
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)  # 计算频率
        if freq.size(0) > self.max_input_ids:  # 超过最大ID数
            freq.narrow(
                0, self.max_input_ids, freq.size(0) - self.max_input_ids
            ).zero_()  # 清零
        alpha = torch.pow(self.penalty, freq)  # 计算惩罚因子
        scores = scores.contiguous()  # 确保连续
        inp = scores.multiply(alpha)  # 正值惩罚
        oth = scores.divide(alpha)  # 负值惩罚
        con = scores < 0  # 负值条件
        out = torch.where(con, inp, oth)  # 条件选择
        del inp, oth, scores, con, alpha
        return out


class ConditionalChatTTS(PreTrainedModel):
    """条件ChatTTS模型：支持说话人条件和流式生成的文本转语音模型"""
    """A conditional text-to-speech model that can generate speech from text with speaker conditioning.

    This model extends PreTrainedModel to provide text-to-speech capabilities with:
    - LLM hidden state conditioning
    - Streaming generation

    The model uses a transformer architecture with LLM hidden states and can operate in both
    streaming and non-streaming modes for flexible deployment.

    The model process sequence in the following format:
    | text bos token | LLM embedding projected to tts embedding space | text tokens (fixed length, reserved for future tokens) | audio bos token | audio tokens (audio token length is not fixed)| audio eos token |

    The format is designed to support LLM-conditioned streaming audio generation.

    Usage:
    To support streaming generation, two global variables should be maintained outside of the model.
        1. `audio_input_ids`: stores *discrete* audio codes. It is a tensor with shape [1, sequence length+1, num_vq].
        2. `past_key_values`: stores the KV cache for both text tokens and audio codes. It is a list of tuples, each tuple contains two tensors with shape [1, num_attention_heads, sequence length, hidden_size // num_attention_heads]

    where `num_vq` is the number of audio codebooks, in default setting, it is `4`.

    1. Create an empty `past_key_values` with
    ```python
    initial_kv_cache_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len # where `1` denotes the `bos` token
    dtype = model.emb_text.weight.dtype
    device = model.emb_text.weight.device
    past_key_values = [
        (
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device),
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device)
        )
        for _ in range(model.config.num_hidden_layers)
    ]

    2. At the same time, create an empty `audio_input_ids` with shape [1, sequence length, num_vq], `num_vq` denotes multiple layer audio codebooks. But here we also include text tokens in the sequence, but they will be zeros, and will not be used, just a placeholder.

    ```python
    initial_audio_input_ids_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len + 1
    # [bos token, speaker embeddings, text tokens, audio bos token]
    audio_input_ids = torch.zeros(batch_size=1, initial_audio_input_ids_length, model.num_vq)
    ```

    2. Prefill some text tokens to TTS model (for example, 10 tokens) using `prefill_text` method.

    ```python

    outputs = llm.generate(**kwargs)
    llm_tokens = some_function_to_extract_llm_tokens(outputs)
    lm_spk_emb_last_hidden_states = some_function_to_extract_lm_spk_emb_last_hidden_states(outputs)
    tts_text_input_ids = tts_tokenizer.encode(llm_tokenizer.decode(llm_tokens))
    # here assume we are prefilling text token 0 to text token 9 (included), totally 10 tokens.
    begin = 0
    end = 9+1
    position_ids = torch.arange(begin, end, dtype=torch.long, device=device)

    past_key_values = model.prefill_text(
        input_ids=tts_text_input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
    )
    ```

    3. Make a `streaming_tts_text_mask` to denote which position contains valid text tokens, similar to `attention_mask` in standard causal attention.

    ```python
    streaming_tts_text_mask = torch.zeros(model.streaming_reserved_length)
    streaming_tts_text_mask[0:end] = 1 # denotes these post
    ```

    3. Generate audio codes using `generate` method.

    ```python
    outputs = model.generate(
        input_ids=audio_input_ids,
        past_key_values=past_key_values,
        streaming_tts_text_mask=streaming_tts_text_mask,
        max_new_token=50,
    )

    # update past_key_values and input_ids
    past_key_values = outputs.past_key_values
    audio_input_ids = outputs.input_ids
    ```

    The `past_key_values` is extended by `max_new_token=50`, and `audio_input_ids` is also extended by `max_new_token=50` after `generate` calling.

    4. Notice that after prefilling `10` text tokens, the model can generate up to `50` audio tokens, if you want to generate more audio tokens, you need to prefill next `10` text tokens. And it is okay to only generate `25` audio tokens for faster initial response.

    5. Repeat steps `2,3,4` as needed in your streaming audio generation cases, but ensure usage complies with the following guidelines discussed above.
    """

    config_class = PretrainedConfig  # 配置类
    _no_split_modules = []  # 不分割模块

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)  # 调用父类初始化

        self.use_speaker_embedding = config.use_speaker_embedding  # 是否使用说话人嵌入
        self.use_llm_hidden_state = config.use_llm_hidden_state  # 是否使用LLM隐藏状态
        self.num_spk_embs = config.num_spk_embs  # 说话人嵌入数量
        self.spk_emb_token_id = config.spk_emb_token_id  # 说话人嵌入token ID

        self.use_text = config.use_text  # 是否使用文本
        self.streaming = config.streaming  # 是否流式
        self.streaming_text_chunk_size = config.streaming_text_chunk_size  # 流式文本块大小
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size  # 流式音频块大小
        self.streaming_text_reserved_len = config.streaming_text_reserved_len  # 流式文本预留长度
        self.audio_bos_token_id = config.audio_bos_token_id  # 音频BOS token ID
        self.num_mel_bins = config.num_mel_bins  # 梅尔频率数
        self.num_vq = config.num_vq  # VQ数量
        self.num_audio_tokens = config.num_audio_tokens  # 音频token数

        self.top_p = config.top_p  # top-p采样
        self.top_k = config.top_k  # top-k采样
        self.repetition_penalty = config.repetition_penalty  # 重复惩罚

        if self.config.use_mlp:  # 使用MLP投影
            self.projector = MultiModalProjector(config.llm_dim, config.hidden_size)
        else:  # 使用线性投影
            self.projector = nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        self.emb_code = nn.ModuleList(  # 音频码嵌入列表
            [
                nn.Embedding(config.num_audio_tokens, config.hidden_size)
                for _ in range(config.num_vq)
            ]
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)  # 文本嵌入
        self.head_code = nn.ModuleList(  # 音频码头列表
            [
                parametrizations.weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )  # 权重归一化
                for _ in range(config.num_vq)
            ]
        )

        dvae = DVAE()  # DVAE模型
        self.dvae = dvae

        model_config = LlamaConfig(  # Llama配置
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            attn_implementation=config.attn_implementation,
        )

        model = LlamaModel(model_config)  # Llama模型
        self.model = model

    @torch.inference_mode()  # 推理模式
    def merge_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """合并输入ID和LLM隐藏状态为输入嵌入"""
        """Merge `input_ids` and `lm_spk_emb_last_hidden_states` to `inputs_embeds`.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            lm_spk_emb_last_hidden_states (Optional[torch.Tensor], optional): Last hidden states of speaker embeddings from the language model. Defaults to None.

        Raises:
            NotImplementedError: If speaker embedding is not used and language model hidden states are not implemented.

        Returns:
            torch.Tensor: Prepared input embeddings for the model.
        """
        assert input_ids.shape[0] == 1  # 仅支持批次1

        # Embed input_ids to input_embeds  # 将输入ID嵌入为输入嵌入
        inputs_embeds = self.emb_text(input_ids)  # 文本嵌入

        # Inject speaker embedding to input_embeds if it exists  # 如果存在则注入说话人嵌入
        if self.use_speaker_embedding:  # 使用说话人嵌入
            spk_emb_mask = input_ids == self.spk_emb_token_id  # 说话人token掩码
            if spk_emb_mask.any():  # 有说话人token
                assert lm_spk_emb_last_hidden_states is not None
                # Project spk emb to tts hidden size first, [batch_size, num_spk_emb, llm_dim] -> [batch_size, num_spk_emb, self.hidden_size]  # 投影说话人嵌入到TTS隐藏大小
                lm_spk_emb_last_hidden_states = lm_spk_emb_last_hidden_states.to(
                    self.projector.linear1.weight.dtype
                )
                projected_spk_emb = self.projector(lm_spk_emb_last_hidden_states)  # 投影
                projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)  # L2归一化
                apply_spk_emb(
                    input_ids=input_ids,
                    spk_emb=projected_spk_emb,
                    input_embeds=inputs_embeds,
                    spk_emb_token_id=self.spk_emb_token_id,
                    num_spk_embs=self.num_spk_embs,
                )  # 应用说话人嵌入
        else:  # 不使用说话人嵌入
            raise NotImplementedError

        return inputs_embeds

    @torch.inference_mode()  # 推理模式
    def prefill_text(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """预填充文本token到KV缓存"""
        """Prefill a chunk of new text tokens in streaming setting.
        Specifically speaking, update `past_key_values` using new text tokens, then the model will read the new text tokens.

        Args:
            input_ids (Tensor): Tensor of shape [batch_size, seq_len]
            position_ids (LongTensor): Tensor of shape [batch_size, seq_len]
            past_key_values (List[Tuple[Tensor]]): KV Cache of all layers, each layer is a tuple (Tensor, Tensor) denoting keys and values. Each tensor is of seq_len = `self.streaming_text_reserved_len`. `past_key_values` will be updated.
            lm_spk_emb_last_hidden_states (Tensor, optional): Tensor of shape [batch_size, num_spk_emb, llm_dim]. Defaults to None.

        Note that all `batch_size` should be `1`.
        """
        assert input_ids.shape[0] == 1  # 仅支持批次1
        assert past_key_values is not None  # 必须有KV缓存

        # Merge text and LLM embeddings  # 合并文本和LLM嵌入
        inputs_embeds = self.merge_inputs_embeds(
            input_ids=input_ids,
            lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
        )

        # Clone KV Cache  # 克隆KV缓存
        past_key_values_for_prefill = []
        for i in range(len(past_key_values)):  # 遍历每层
            past_key_values_for_prefill.append(
                (
                    past_key_values[i][0][:, :, : position_ids[:, 0], :].clone(),  # K缓存
                    past_key_values[i][1][:, :, : position_ids[:, 0], :].clone(),  # V缓存
                )
            )

        # ModelMiniCPMVBaseModel  # 模型前向传播
        outputs_prefill: BaseModelOutputWithPast = self.model(
            attention_mask=None,  # because for text, it is standard causal attention mask, do nothing  # 文本使用标准因果注意力掩码
            position_ids=position_ids,  # position_ids denotes the position of new text tokens in the sequence  # 位置ID
            past_key_values=past_key_values_for_prefill,  # `past_key_values` will be updated by the model  # KV缓存将被模型更新
            inputs_embeds=inputs_embeds,  # contains text and language model embedding  # 包含文本和语言模型嵌入
            use_cache=True,
            output_attentions=False,
            cache_position=position_ids,  # which new positions will use this cache, basically the same as position_ids  # 使用缓存的新位置
        )

        # Get model updated KV Cache  # 获取模型更新的KV缓存
        past_key_values_for_prefill_updated = outputs_prefill.past_key_values

        # Update generated KV Cache to input `past_key_values`  # 更新生成的KV缓存到输入的past_key_values
        for layer_idx in range(len(past_key_values)):  # 遍历每层
            # Update keys  # 更新K
            past_key_values[layer_idx][0][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1, :
            ] = past_key_values_for_prefill_updated[layer_idx][0][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1
            ].clone()
            # Update values  # 更新V
            past_key_values[layer_idx][1][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1, :
            ] = past_key_values_for_prefill_updated[layer_idx][1][
                :, :, position_ids[:, 0] : position_ids[:, -1] + 1
            ].clone()

        # TODO: del past_key_values_for_prefill_updated recursively  # 待办：递归删除
        # TODO: del outputs_prefill recursively  # 待办：递归删除

        return past_key_values

    @torch.inference_mode()  # 推理模式
    def prefill_audio_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        streaming_tts_text_mask=None,
        add_audio_bos: bool = True,
    ):
        """预填充音频ID到模型，用于滑动窗口长音频生成"""
        """Prefill a chunk of audio ids to the model. Used in sliding-window long audio generation.
        Specifically, prefill many audio ids (typically from last window) to the model in the new window.

        Args:
            input_ids (torch.Tensor): (1, seq_len, num_vq) Audio input token ids.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
        """
        assert input_ids.shape[0] == 1  # 仅支持批次1
        assert past_key_values is not None  # 必须有KV缓存

        code_emb = [self.emb_code[i](input_ids[:, :, i]) for i in range(self.num_vq)]  # 各VQ层嵌入
        inputs_embeds = torch.stack(code_emb, 3).sum(3)  # [1,seq_len,768]  # 求和
        input_len = input_ids.shape[1]  # 输入长度

        if add_audio_bos:  # 添加音频BOS
            narrowed_input_ids = torch.tensor(
                [[self.audio_bos_token_id]], dtype=torch.long, device=self.device
            )
            bos_inputs_embeds = self.emb_text(narrowed_input_ids)  # BOS嵌入
            inputs_embeds = torch.cat([bos_inputs_embeds, inputs_embeds], dim=1)  # 拼接
            input_len += 1

        past_key_values_length = past_key_values[0][0].shape[2]  # KV缓存长度
        position_ids = torch.arange(
            past_key_values_length,
            past_key_values_length + input_len,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)  # 位置ID

        cache_position = position_ids.clone()  # 缓存位置
        causal_mask = make_streaming_chunk_mask_generation(
            inputs_embeds=inputs_embeds,
            past_seen_tokens=past_key_values[0][0].shape[2],
            streaming_tts_text_mask=streaming_tts_text_mask,
            streaming_reserved_length=self.streaming_text_reserved_len,
            streaming_text_chunk_size=self.streaming_text_chunk_size,
        )  # [1, 1, 1, past_key_values_length + input_len]  # 因果掩码

        # Model forward  # 模型前向传播
        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=cache_position,
        )
        past_key_values = outputs.past_key_values  # 更新KV缓存
        return past_key_values

    @torch.inference_mode()  # 推理模式
    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        streaming_tts_text_mask=None,
        force_no_stop=False,
        min_new_token=10,
        max_new_token=50,
        logits_warpers: Optional[List[LogitsWarper]] = None,
        logits_processors: Optional[
            List[CustomRepetitionPenaltyLogitsProcessorRepeat]
        ] = None,

        show_tqdm=False,
    ):
        """流式或非流式生成音频码"""
        """Generate audio codes in streaming setting or non-streaming setting.
        Specifically speaking, generate audio codes when not all text tokens are prefilled.

        Always pass a valid `past_key_values` to the method. The method does not do `prefill` by itself. It relies on `prefill_text` method to provide valid `past_key_values`. Please refer to docstring of this class for more details.

        In this method, we borrowed a lot of codes from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/gpt.py`.

        Args:
            input_ids (torch.Tensor): Input token ids.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
            temperature (torch.Tensor): Temperature for sampling.
            eos_token (Union[int, torch.Tensor]): End of sequence token.
            streaming_tts_text_mask (Optional[torch.Tensor], optional): Mask for streaming TTS text. Defaults to None.
            max_new_token (int, optional): Maximum number of new tokens to generate. Defaults to 50.
            logits_warpers (List[LogitsWarper], optional): List of logits warpers. Defaults to [].
            logits_processors (List[CustomRepetitionPenaltyLogitsProcessorRepeat], optional): List of logits processors. Defaults to [].
            show_tqdm (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            GenerationOutputs: Generation outputs.
        """

        # We only support batch size `1` for now  # 目前仅支持批次大小1
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        logits_warpers = logits_warpers or []  # 默认空列表
        logits_processors = logits_processors or []  # 默认空列表

        # fix: this should not be `input_ids.shape[1]`  # 修正：不应该是input_ids.shape[1]
        # start_idx = input_ids.shape[1]
        start_idx = (  # 音频token起始索引
            1
            + self.num_spk_embs * self.use_speaker_embedding
            + self.streaming_text_reserved_len
            + 1
        )

        finish = torch.zeros(input_ids.shape[0], device=input_ids.device).bool()  # 完成标志

        temperature = (  # 温度参数
            temperature.unsqueeze(0)
            .expand(input_ids.shape[0], -1)
            .contiguous()
            .view(-1, 1)
        )

        progress = input_ids.shape[1]  # 当前进度

        # Pre-allocate input_ids, shape is [batch_size=1, max_possible_seq_len, self.num_vqs]  # 预分配input_ids
        input_ids_buf = torch.zeros(
            input_ids.shape[0],  # batch_size
            progress
            + max_new_token,  # max_possible_seq_len = input_ids.shape[1] + max_new_token  # 最大可能序列长度
            input_ids.shape[2],  # self.num_vqs  # VQ数量
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        # Copy existing `input_ids` to `input_ids_buf`  # 复制现有input_ids到缓冲区
        input_ids_buf.narrow(1, 0, progress).copy_(input_ids)

        del input_ids
        input_ids = input_ids_buf.narrow(1, 0, progress)

        pbar: Optional[tqdm] = None  # 进度条
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        condition_length = (  # 条件长度
            1
            + self.num_spk_embs * self.use_speaker_embedding
            + self.streaming_text_reserved_len
            + 1
        )

        for i in range(max_new_token):  # 逐步生成
            # Prepare generation inputs  # 准备生成输入
            audio_bos = False

            # If this is the first audio token, the case is SPECIAL  # 第一个音频token是特殊情况
            if progress == condition_length:  # 第一个音频token
                audio_bos = True

            assert progress == (
                past_key_values[0][0].shape[2] + 1
            )  # If you are using according to the guidelines, this should be passed.  # 按指南使用时应通过此检查

            if audio_bos:  # 生成第一个token
                # Generate the first token, activate the model with `self.audio_bos_token_id`, the model will predict  # 生成第一个token，用audio_bos_token_id激活模型
                # a new audio token. This is a special case because without the `audio bos token`, it is impossible  # 模型将预测新的音频token。这是特殊情况，因为没有audio bos token
                # to generate the first audio token in our streaming setting.  # 在流式设置中无法生成第一个音频token
                narrowed_input_ids = torch.tensor(
                    [[self.audio_bos_token_id]], dtype=torch.long, device=self.device
                )
                inputs_embeds = self.emb_text(narrowed_input_ids)
                del narrowed_input_ids
            else:  # 生成后续token
                # Generate the following audio tokens, it is applicable to all other cases, including second and the  # 生成后续音频token，适用于所有其他情况
                # following calling of `generate`.  # 包括第二次及之后的generate调用
                narrowed_input_ids = input_ids.narrow(
                    dim=1, start=input_ids.shape[1] - 1, length=1
                )
                code_emb = [
                    self.emb_code[i](narrowed_input_ids[:, :, i])
                    for i in range(self.num_vq)
                ]
                inputs_embeds = torch.stack(code_emb, 3).sum(3)

            position_ids = torch.tensor(
                [past_key_values[0][0].shape[2]], dtype=torch.long, device=self.device
            ).unsqueeze(0)  # 位置ID

            cache_position = position_ids.clone()  # 缓存位置

            # Make causal mask  # 创建因果掩码
            causal_mask = make_streaming_chunk_mask_generation(
                inputs_embeds=inputs_embeds,
                past_seen_tokens=past_key_values[0][0].shape[2],
                streaming_tts_text_mask=streaming_tts_text_mask,
                streaming_reserved_length=self.streaming_text_reserved_len,
                streaming_text_chunk_size=self.streaming_text_chunk_size,
            )

            # Model forward  # 模型前向传播
            outputs: BaseModelOutputWithPast = self.model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                cache_position=cache_position,
            )

            del position_ids
            del inputs_embeds
            del cache_position
            del causal_mask

            hidden_states = outputs.last_hidden_state  # 隐藏状态
            past_key_values = outputs.past_key_values  # 更新KV缓存

            with P.cached():  # 缓存参数化
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )  # 预分配logits
                for num_vq_iter in range(self.num_vq):  # 遍历每个VQ层
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)  # 计算logits
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            # logits = logits[:, -1].float()
            logits = logits.narrow(1, -1, 1).squeeze_(1).float()  # 取最后一个token

            # logits = rearrange(logits, "b c n -> (b n) c")
            logits = logits.permute(0, 2, 1)  # 重排
            logits = logits.reshape(-1, logits.size(2))
            # logits_token = rearrange(input_ids[:, start_idx:], "b c n -> (b n) c")
            input_ids_sliced = input_ids.narrow(
                1,
                start_idx,
                input_ids.size(1) - start_idx,
            ).permute(0, 2, 1)
            logits_token = input_ids_sliced.reshape(
                input_ids_sliced.size(0) * input_ids_sliced.size(1),
                -1,
            ).to(self.device)
            del input_ids_sliced

            logits /= temperature  # 温度缩放

            if not audio_bos:  # 非第一个token应用处理器
                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)
            if not audio_bos:  # 非第一个token应用扭曲器
                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

            del logits_token

            if i < min_new_token:  # 最小新token数内禁止结束
                logits[:, eos_token] = -torch.inf

            if force_no_stop:  # 强制不停止
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)  # softmax

            del logits
            idx_next = torch.multinomial(scores, num_samples=1)  # 采样  # .to(finish.device)

            del scores

            # idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
            idx_next = idx_next.view(-1, self.num_vq)  # 重塑
            finish_or = idx_next.eq(eos_token).any(1)  # 检查是否遇到EOS
            finish.logical_or_(finish_or)

            del finish_or
            # Store new `token` into `input_ids_buf`  # 存储新token到缓冲区
            input_ids_buf.narrow(1, progress, 1).copy_(idx_next.unsqueeze_(1))

            if i == 0 and finish.any():  # 第一个token就结束
                # raise Exception
                break

            del idx_next
            progress += 1  # 更新进度
            input_ids = input_ids_buf.narrow(1, 0, progress)

            if finish.all():  # 所有序列都结束
                break

            if pbar is not None:  # 更新进度条
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():  # 未完成
            if show_tqdm:
                logger.info(f"incomplete result. hit max_new_token: {max_new_token}")

        del input_ids_buf

        if finish.all():  # 完成的情况
            # the last may contains eos token  # 最后可能包含EOS token
            genrated_input_ids = input_ids[:, condition_length:-1, :]
        else:  # 未完成的情况
            # there is no eos token  # 没有EOS token
            genrated_input_ids = input_ids[:, condition_length:, :]

        return ConditionalChatTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=input_ids,  # for update purpose  # 用于更新
            past_key_values=past_key_values,  # for update purpose  # 用于更新
            finished=finish.all(),
        )

    @torch.inference_mode()  # 推理模式
    def decode_to_mel_specs(
        self,
        result_list: List[torch.Tensor],
    ):
        """将离散音频码解码为梅尔频谱"""
        """Decode discrete audio codes to mel spectrograms.

        Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/core.py`

        Args:
            result_list (List[torch.Tensor]): Audio codes output from `generate`.

        Returns:
            torch.Tensor: Mel spectrograms.
        """

        decoder = self.dvae  # DVAE解码器
        max_x_len = -1
        if len(result_list) == 0:  # 空列表
            return np.array([], dtype=np.float32)
        for result in result_list:  # 找到最大长度
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(  # 批量结果
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):  # 填充批量
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
            del src

        mel_specs = decoder(batch_result)  # DVAE解码
        del batch_result
        return mel_specs


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer and add use_cache for streaming inference  # 从Whisper编码器层复制并添加use_cache用于流式推理
class MiniCPMWhisperEncoderLayer(nn.Module):
    """MiniCPM Whisper编码器层：带KV缓存的Whisper编码器层"""
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()  # 调用父类初始化
        self.embed_dim = config.d_model  # 嵌入维度
        self.self_attn = WhisperAttention(  # Whisper自注意力
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,

            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层归一化
        self.dropout = config.dropout  # dropout率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # FFN第一层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # FFN第二层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        """Whisper编码器层前向传播"""
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, embed_dim)`):
                Hidden states to be fed into the encoder layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, tgt_len, src_len)`):
                Attention mask where padding elements are indicated by large negative values.
            layer_head_mask (`torch.FloatTensor` of shape `(encoder_attention_heads,)`):
                Mask to nullify selected heads of the attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention weights.
            past_key_values (`EncoderDecoderCache`, *optional*):
                Past key-value pairs used for incremental decoding.
            use_cache (`bool`, *optional*):
                Whether or not to return updated `past_key_values` for caching.

        Returns:
            A tuple of shape `(hidden_states, optional(attn_weights), optional(past_key_values))`.
        """
        residual = hidden_states  # 保存残差
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 层归一化
        # TODO (lifuhuang): confirmed with Mick that the logic for past_key_values is copied from minicpmo official code,  # 已确认past_key_values逻辑来自minicpmo官方代码
        # currently we are not using past_key_values at all. We need to redesign the caching logic when we support streaming  # 目前完全未使用past_key_values。支持流式时需重新设计缓存逻辑
        # in the future.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )  # 自注意力
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )  # dropout
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差
        hidden_states = self.final_layer_norm(hidden_states)  # 层归一化
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # FFN第一层+激活
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=False
        )  # dropout
        hidden_states = self.fc2(hidden_states)  # FFN第二层
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )  # dropout
        hidden_states = residual + hidden_states  # 残差连接

        if hidden_states.dtype == torch.float16 and (  # float16溢出检查
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )  # 裁剪

        outputs = (hidden_states,)  # 输出

        if output_attentions:  # 输出注意力权重
            outputs += (attn_weights,)

        if use_cache:  # 输出KV缓存
            outputs += (past_key_values,)

        return outputs


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference  # 从Whisper编码器复制并添加use_cache用于流式推理
class MiniCPMWhisperEncoder(WhisperEncoder):
    """MiniCPM Whisper编码器：支持KV缓存的流式音频编码"""

    def __init__(self, config: WhisperConfig):
        super().__init__(config)  # 调用父类初始化
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                MiniCPMWhisperEncoderLayer(config, layer_idx=i)
                for i in range(config.encoder_layers)
            ]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
    ):
        """Whisper编码器前向传播"""
        r"""
        Forward pass of the Whisper encoder.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of log-mel features extracted from the raw audio waveform. Typically generated
                by a feature extractor (e.g., `WhisperFeatureExtractor`) that processes `.flac` or `.wav`
                files into padded 2D mel spectrogram frames. These features are projected via convolution layers
                (`conv1` and `conv2`) and then transformed into embeddings for the encoder.

            attention_mask (`torch.Tensor`, *optional*):
                Not used by Whisper for masking `input_features`, but included for API compatibility with
                other models. If provided, it is simply ignored within the model. By default, Whisper
                effectively ignores silence in the input log-mel spectrogram.

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected attention heads. The elements should be either 1 or 0, where:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked** (i.e., the attention head is dropped).

            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors of all encoder layers. If set to `True`, the
                returned tuple (or `BaseModelOutputWithPast`) will contain an additional element with
                attention weights for each encoder layer.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. If set to `True`, the returned
                tuple (or `BaseModelOutputWithPast`) will contain a tuple of hidden states, including the
                initial embedding output as well as the outputs of each layer.

            return_dict (`bool`, *optional*):
                Whether or not to return a `BaseModelOutputWithPast` (a subclass of `ModelOutput`) instead
                of a plain tuple. If set to `True`, the output will be a `BaseModelOutputWithPast` object,
                otherwise it will be a tuple.

            past_key_values (`EncoderDecoderCache`, *optional*):
                When using caching for faster inference, this is an object that stores the key-value pairs
                for attention states. If provided, the model will append new states to the existing cache
                and return the updated cache. This speeds up sequential decoding or chunked inference.

                - If `past_key_values` is `None`, no past states are used or returned.
                - If `past_key_values` is not `None` and `use_cache=True`, the model will use the provided
                cache and return the updated cache (as `next_encoder_cache`).

            use_cache (`bool`, *optional*):
                Whether or not the model should use caching (`past_key_values`) to speed up processing
                during inference. When set to `True`, the model will:
                - Inspect and use `past_key_values` if provided.
                - Return updated `past_key_values` (under the name `next_encoder_cache` in
                    `BaseModelOutputWithPast`).

        Returns:
            `BaseModelOutputWithPast` or `tuple` (depending on `return_dict`):
                If `return_dict=True`, a `BaseModelOutputWithPast` is returned, which contains:
                - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The output of the final encoder layer.
                - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True`):
                Hidden states of the model at each layer (including the initial projection).
                - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_attentions=True`):
                Attention weights from each encoder layer.
                - **past_key_values** (an object of type `EncoderDecoderCache` or `None`, *optional*):
                Updated cache of key-value pairs if `use_cache=True`.

                If `return_dict=False`, a tuple is returned, where the format is:
                `(last_hidden_state, hidden_states, attentions)`, with `hidden_states` and `attentions`
                only present if their respective `output_*` arguments are set to `True`.

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )  # 输出注意力
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )  # 输出隐藏状态
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )  # 返回字典

        # Ignore copy
        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )  # 转换类型和设备

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # 第一个卷积+GELU
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # 第二个卷积+GELU

        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # 转置

        embed_pos = self.embed_positions.weight  # 位置嵌入
        past_key_values_length = 0
        if use_cache:  # 使用缓存
            if past_key_values is None:  # 创建新缓存
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):  # 列表格式
                past_key_values = EncoderDecoderCache(
                    DynamicCache.from_legacy_cache(past_key_values), DynamicCache()
                )
            elif isinstance(past_key_values, DynamicCache):  # DynamicCache格式
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:  # 其他格式
                pass
            past_key_values_length = (
                past_key_values.self_attention_cache.get_usable_length(
                    inputs_embeds.shape[1]
                )
            )  # 可用缓存长度
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:  # 超出位置嵌入范围
                logger.warning(
                    "seems the audio is longer than 30s. repeating the last part of the audio"
                )  # 音频超过30秒
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1]
                            - embed_pos.shape[0]
                            + past_key_values_length,
                            dim=0,
                        ),
                    )
                )  # 重复最后一个位置
            else:  # 正常情况
                embed_pos = embed_pos[
                    past_key_values_length : inputs_embeds.shape[1]
                    + past_key_values_length,
                    :,
                ]
        else:  # 不使用缓存
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos  # 加位置嵌入
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=False
        )  # dropout

        encoder_states = () if output_hidden_states else None  # 编码器状态
        all_attentions = () if output_attentions else None  # 注意力权重

        # check if head_mask has a correct number of layers specified if desired  # 检查head_mask是否有正确的层数
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):  # 遍历编码器层
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)  # 添加LayerDrop
            to_drop = False

            # Ignore copy
            if to_drop:  # 丢弃层
                layer_outputs = (None, None)
            else:  # 正常前向传播
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]  # 更新隐藏状态

            if use_cache:  # 获取缓存
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:  # 收集注意力
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)  # 最终层归一化
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:  # 返回元组
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutputWithPast(  # 返回字典
            last_hidden_state=hidden_states,

            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )


class MultiModalProjector(nn.Module):
    """多模态投影层：两层MLP+ReLU"""
    def __init__(self, in_dim, out_dim):
        super().__init__()  # 调用父类初始化
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)  # 第一个线性层
        self.relu = nn.ReLU()  # ReLU激活
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)  # 第二个线性层

    def forward(self, audio_features):
        """多模态投影前向传播"""
        hidden_states = self.relu(self.linear1(audio_features))  # 第一个线性层+ReLU
        hidden_states = self.linear2(hidden_states)  # 第二个线性层
        return hidden_states


class MiniCPMO(MiniCPMBaseModel):
    """MiniCPM-o多模态模型：视觉+音频+TTS"""
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config, quant_config=quant_config)  # 调用父类初始化

        self.llm = self.init_llm(config=config, quant_config=quant_config)  # 初始化LLM

        self.embed_dim = self.llm.config.hidden_size  # 嵌入维度

        # init vision module  # 初始化视觉模块
        if self.config.init_vision:
            # print("vision-understanding enabled")
            self.vpm = self.init_vision_module(config=config, quant_config=quant_config)  # 视觉模块
            self.vision_dim = self.vpm.embed_dim  # 视觉维度
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)  # 重采样器

        # init audio module  # 初始化音频模块
        self.config.init_audio = True
        if self.config.init_audio:
            # print("audio-understanding enabled")
            self.apm = self.init_audio_module()  # 音频模块
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)  # 音频输出维度
            self.audio_avg_pooler = nn.AvgPool1d(  # 音频平均池化
                self.config.audio_pool_step, stride=self.config.audio_pool_step
            )
            self.audio_projection_layer = MultiModalProjector(  # 音频投影层
                in_dim=audio_output_dim, out_dim=self.embed_dim
            )
            self.audio_encoder_layer = -1  # 音频编码器层

        # init tts module  # 初始化TTS模块
        self.config.init_tts = False
        logger.info("TTS is disabled for now")  # TTS当前禁用
        if self.config.init_tts:  # 如果启用TTS
            # print("tts enabled")
            assert (
                _tts_deps
            ), "please make sure vector_quantize_pytorch and vocos are installed."
            self.tts = self.init_tts_module()  # TTS模块

    def init_tts_module(self):
        """初始化TTS模块"""
        model = ConditionalChatTTS(self.config.tts_config)
        return model

    def init_audio_module(self):
        """初始化音频模块"""
        model = MiniCPMWhisperEncoder(self.config.audio_config)
        return model

    def init_llm(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化语言模型"""
        return Qwen2ForCausalLM(config=config, quant_config=quant_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        """初始化视觉模块"""
        if self.config._attn_implementation == "flash_attention_2":  # Flash注意力
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:  # Eager注意力
            self.config.vision_config._attn_implementation = "eager"
        model = Idefics2VisionTransformer(
            config=config.vision_config, quant_config=quant_config, prefix=prefix
        )  # Idefics2视觉Transformer
        if self.config.drop_vision_last_layer:  # 丢弃最后一层
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)  # 设置嵌入维度
        setattr(model, "patch_size", model.embeddings.patch_size)  # 设置补丁大小

        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        """初始化重采样器"""
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.  # 2.6版本的重采样器与2.5版本一致
            resampler = Resampler2_5(
                num_queries=self.config.query_num,  # 查询数
                embed_dim=embed_dim,  # 嵌入维度
                num_heads=embed_dim // 128,  # 头数
                kv_dim=vision_dim,  # KV维度
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(device=get_device(), dtype=torch.get_default_dtype())

    def pad_input_ids(self, input_ids: List[int], mm_input: MultimodalInputs):
        """填充输入ID以适配多模态token"""
        # Get all special token IDs  # 获取所有特殊token ID
        im_start_id: int = mm_input.im_start_id
        im_end_id: int = mm_input.im_end_id
        slice_start_id: int = mm_input.slice_start_id
        slice_end_id: int = mm_input.slice_end_id

        data_token_pairs = [  # 数据token对
            (im_start_id, im_end_id),  # 图像
            (slice_start_id, slice_end_id),  # 切片
            (mm_input.audio_start_id, mm_input.audio_end_id),  # 音频
        ]
        data_start_token_ids = [im_start_id, mm_input.audio_start_id]  # 数据起始token ID
        pattern = MultiModalityDataPaddingPatternTokenPairs(
            data_token_pairs=data_token_pairs, data_start_token_ids=data_start_token_ids
        )

        return pattern.pad_input_tokens(input_ids, mm_input)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """计算卷积层和音频编码器的输出长度"""
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1  # 卷积后长度
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1  # 池化后长度
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_audio_embedding_streaming(self, items: List[MultimodalDataItem]):
        """流式提取音频嵌入"""
        r"""
        Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        wavforms = flatten_nested_list([item.feature for item in items if item.feature])  # 波形列表
        # list, [[x1, x2], [y1], [z1]]
        audio_feature_lens_raw = flatten_nested_list(
            [item.audio_feature_lens for item in items if item.audio_feature_lens]
        )  # 音频特征长度

        # exist audio  # 存在音频
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)  # 拼接长度
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1  # 仅支持批次1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1  # 最大序列长度

            if self.audio_past_key_values is not None:  # 有缓存
                cache_length = self.audio_past_key_values[0][0].shape[2]  # 缓存长度
                apm_max_len = self.apm.embed_positions.weight.shape[0]  # APM最大长度
                if cache_length + max_seq_len >= apm_max_len:  # 超出范围
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )  # 重置缓存
                    self.audio_past_key_values = None

            audio_outputs = self.apm(
                wavforms, past_key_values=self.audio_past_key_values, use_cache=True
            )  # Whisper编码器
            audio_states = (
                audio_outputs.last_hidden_state
            )  # [:, :audio_feat_lengths, :]  # 音频隐藏状态
            self.audio_past_key_values = audio_outputs.past_key_values  # 更新缓存

            audio_embeds = self.audio_projection_layer(audio_states)  # 音频投影

            audio_embeds = audio_embeds.transpose(1, 2)  # 转置
            audio_embeds = self.audio_avg_pooler(audio_embeds)  # 平均池化
            audio_embeds = audio_embeds.transpose(1, 2)  # 转置回来

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(
                audio_feature_lens
            )  # 池化后长度

            num_audio_tokens = feature_lens_after_pooling  # 音频token数

            final_audio_embeds = []  # 最终音频嵌入
            idx = 0
            for i in range(len(audio_feature_lens_raw)):  # 遍历每个音频
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):  # 遍历每段
                    target_audio_embeds.append(
                        audio_embeds[idx, : num_audio_tokens[idx], :]
                    )  # 截取有效token
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:  # 无音频
            return []

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """创建流式编码器的后续块掩码"""
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device

        Returns:
            torch.Tensor: mask

        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)  # 初始化掩码
        for i in range(size):  # 遍历每个位置
            if num_left_chunks < 0:  # 使用完整块
                start = 0
            else:  # 使用左侧块数
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)  # 结束位置
            ret[i, start:ending] = True  # 设置可见区域
        return ret

    def get_audio_embedding(self, items: List[MultimodalDataItem], chunk_length=-1):
        """非流式提取音频嵌入"""
        r"""
        Extract full audio embeddings with optional chunk-based attention.

        This method computes embeddings for all audio frames at once, either using full attention (when
        `chunk_length` is -1) or chunk-based attention (when `chunk_length` is a positive number). It does
        not use key-value caching and is suitable for non-streaming inference.

        Args:
            chunk_length (int, optional): Determines whether to use full attention (-1) or chunk-based
                attention (>0) during embedding computation.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        # (bs, 80, frames) or [], multi audios need filled in advance  # (bs, 80, 帧) 或 []，多音频需提前填充
        wavforms = flatten_nested_list([item.feature for item in items if item.feature])  # 波形列表
        # list, [[x1, x2], [y1], [z1]]
        audio_feature_lens_raw = flatten_nested_list(
            [item.audio_feature_lens for item in items if item.audio_feature_lens]
        )  # 音频特征长度

        # Ensure audio_feature_lens_raw is properly formatted as [[tensor], [tensor], ...]  # 确保格式正确
        if audio_feature_lens_raw:
            if isinstance(audio_feature_lens_raw[0], torch.Tensor):  # 张量列表
                # Flat list of tensors, wrap each in a list  # 扁平张量列表，每个包装为列表
                audio_feature_lens_raw = [[lens] for lens in audio_feature_lens_raw]
            elif isinstance(audio_feature_lens_raw[0], list):  # 嵌套列表
                # Already nested, ensure all elements are properly formatted  # 已嵌套，确保格式正确
                # Flatten if needed  # 需要时展平
                flattened = []
                for item in audio_feature_lens_raw:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                audio_feature_lens_raw = [
                    [item] if not isinstance(item, list) else item for item in flattened
                ]

        final_audio_embeds = []  # 最终音频嵌入

        assert isinstance(wavforms, list)
        assert isinstance(wavforms[0], torch.Tensor)
        # exist audio  # 存在音频
        for wavform in wavforms:  # 遍历每个波形
            if len(wavform) > 0:
                # Flatten audio_feature_lens_raw to get a list of tensors  # 展平音频特征长度
                flattened_lens = []
                for item in audio_feature_lens_raw:
                    if isinstance(item, list):
                        flattened_lens.extend(item)
                    else:
                        flattened_lens.append(item)
                audio_feature_lens = torch.hstack(flattened_lens)  # 拼接长度
                batch_size, _, max_mel_seq_len = wavform.shape
                max_seq_len = (max_mel_seq_len - 1) // 2 + 1  # 最大序列长度

                # Create a sequence tensor of shape (batch_size, max_seq_len)  # 创建序列张量
                seq_range = (
                    torch.arange(
                        0,
                        max_seq_len,
                        dtype=audio_feature_lens.dtype,
                        device=audio_feature_lens.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )  # 位置范围
                lengths_expand = audio_feature_lens.unsqueeze(1).expand(
                    batch_size, max_seq_len
                )  # 扩展长度
                # Create mask  # 创建掩码
                padding_mask = seq_range >= lengths_expand  # 1 for padded values  # 1表示填充值

                audio_attention_mask_ = padding_mask.view(
                    batch_size, 1, 1, max_seq_len
                ).expand(batch_size, 1, max_seq_len, max_seq_len)  # 注意力掩码
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.apm.conv1.weight.dtype,
                    device=self.apm.conv1.weight.device,
                )  # 转换类型和设备

                if chunk_length > 0:  # 分块注意力
                    chunk_num_frame = int(chunk_length * 50)  # 每块帧数
                    chunk_mask = self.subsequent_chunk_mask(
                        size=max_seq_len,
                        chunk_size=chunk_num_frame,
                        num_left_chunks=-1,
                        device=audio_attention_mask_.device,
                    )  # 块掩码
                    audio_attention_mask_ = torch.logical_or(
                        audio_attention_mask_, torch.logical_not(chunk_mask)
                    )  # 合并掩码

                audio_attention_mask[audio_attention_mask_] = float("-inf")  # 设置为-inf
                audio_states = self.apm(
                    wavform,
                    output_hidden_states=True,
                    attention_mask=audio_attention_mask,
                ).hidden_states[self.audio_encoder_layer]  # Whisper编码器
                audio_embeds = self.audio_projection_layer(audio_states)  # 音频投影

                audio_embeds = audio_embeds.transpose(1, 2)  # 转置
                audio_embeds = self.audio_avg_pooler(audio_embeds)  # 平均池化
                audio_embeds = audio_embeds.transpose(1, 2)  # 转置回来

                _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(
                    audio_feature_lens
                )  # 池化后长度

                num_audio_tokens = feature_lens_after_pooling  # 音频token数

                idx = 0
                for i in range(len(audio_feature_lens_raw)):  # 遍历每个音频
                    target_audio_embeds = []
                    for _ in range(len(audio_feature_lens_raw[i])):  # 遍历每段
                        target_audio_embeds.append(
                            audio_embeds[idx, : num_audio_tokens[idx], :]
                        )  # 截取有效token
                        idx += 1
                    final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """获取音频特征"""
        embedding = self.get_omni_embedding(
            items=items,
            chunk_length=self.config.audio_chunk_length,
            stream_input=False,
        )
        return embedding

    def get_omni_embedding(
        self,
        items: List[MultimodalDataItem],
        chunk_length=-1,
        stream_input=False,
    ):
        """获取全模态嵌入（音频）"""
        """
        Args:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding
        Returns:
            final embeddings with audio feature
        """

        if stream_input:  # 流式输入
            audio_embeddings = self.get_audio_embedding_streaming(items)
        else:  # 非流式输入
            audio_embeddings = self.get_audio_embedding(items, chunk_length)
        bs = len(audio_embeddings)  # 批次大小
        # batch size
        audio_embs = torch.cat(flatten_nested_list(audio_embeddings), dim=0)  # 拼接所有嵌入

        return audio_embs

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

        device = self.vpm.embeddings.position_embedding.weight.device  # 设备
        dtype = self.vpm.embeddings.position_embedding.weight.dtype  # 数据类型
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]  # 展平像素值

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()  # 最大补丁数
        assert isinstance(max_patches, int)
        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0
        )  # 填充序列

        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)  # 重塑
        patch_attn_mask = torch.zeros(
            (B, 1, max_patches), dtype=torch.bool, device=device
        )  # 补丁注意力掩码

        tgt_sizes_tensor = tgt_sizes.clone().to(device=patch_attn_mask.device)  # 目标尺寸
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        """MiniCPM-o前向传播"""
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm,
            multimodal_model=self,
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重"""
        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())  # 参数字典
        for name, loaded_weight in weights:  # 遍历权重

            if "rotary_emb.inv_freq~" in name or "projector" in name:  # 跳过
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:  # 跳过缓存
                # Models trained using ColossalAI may include these tensors in  # ColossalAI训练的模型可能包含
                # the checkpoint. Skip them.  # 跳过
                continue

            # For weight_norm parametrization, handle both old and new formats  # 权重归一化参数化，处理新旧格式
            if self.config.init_tts and "tts" in name:  # TTS权重
                # Handle loading from older checkpoints with weight_g/weight_v format  # 处理旧格式
                if ".weight_g" in name or ".weight_v" in name:
                    name = name.replace(
                        ".weight_g", ".parametrizations.weight.original0"
                    )
                    name = name.replace(
                        ".weight_v", ".parametrizations.weight.original1"
                    )
                elif ".weight" in name and name not in params_dict:  # 新格式
                    param_name = name.replace(
                        ".weight", ".parametrizations.weight.original0"
                    )
                    if param_name in params_dict:
                        name = param_name

            # adapt to VisionAttention  # 适配视觉注意力
            if "vpm" in name:
                name = name.replace(r"self_attn.out_proj", r"self_attn.proj")

            if not self.config.init_tts and "tts" in name:  # 不初始化TTS，跳过TTS权重
                continue
            if not self.config.init_audio and ("apm" in name or "audio" in name):  # 不初始化音频，跳过音频权重
                continue
            if not self.config.init_vision and "vpm" in name:  # 不初始化视觉，跳过视觉权重
                continue

            if (  # 直接加载的权重
                "sampler" in name
                or "apm" in name
                or ("tts" in name and "self_attn" in name)
                or ("tts.model.layers" in name and ".mlp" in name)
            ):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 匹配堆叠参数
                # replace the name and load with customized loader  # 替换名称并自定义加载
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 未匹配堆叠参数
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)  # 默认加载


EntryClass = [MiniCPMO]  # 入口类
