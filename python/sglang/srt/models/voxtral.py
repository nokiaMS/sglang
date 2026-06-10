# Voxtral语音转文本模型实现
# 本文件实现了Voxtral语音转文本模型，包含Whisper编码器、MLP投影器和Llama语言模型。
# 支持原始音频波形输入，通过STFT计算mel频谱图后编码。

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/voxtral.py
# https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
#
# Copyright 2025 Mistral AI and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Inference-only Voxtral (speech-to-text) model."""  # 仅推理的Voxtral语音转文本模型

import math  # 导入数学模块
from typing import Any, Iterable, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置

from sglang.srt.layers.activation import get_act_fn  # 导入激活函数获取
from sglang.srt.layers.linear import (  # 导入并行线性层
    ColumnParallelLinear,  # 列并行线性层
    QKVParallelLinear,  # QKV并行线性层
    RowParallelLinear,  # 行并行线性层
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.llama import LlamaForCausalLM  # 导入Llama模型


class AudioLanguageAdapter(nn.Module):
    """MLP projector: Linear -> GELU -> Linear (no bias)."""  # 音频-语言适配器：线性->GELU->线性

    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()  # 调用父类初始化
        self.w_in = nn.Linear(hidden_size, dim, bias=False)  # 输入线性层
        self.gelu = nn.GELU()  # GELU激活
        self.w_out = nn.Linear(dim, dim, bias=False)  # 输出线性层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """适配器前向传播"""
        return self.w_out(self.gelu(self.w_in(x)))  # 线性->GELU->线性


class VoxtralWhisperAttention(nn.Module):
    """Multi-headed self-attention using plain SDPA (no KV cache).

    Note: HF Voxtral has bias on q_proj, v_proj, out_proj but NOT on k_proj.
    We use QKVParallelLinear with bias=True and create a zero bias for k_proj
    during weight loading.
    """  # 使用SDPA的多头自注意力（无KV缓存）

    def __init__(
        self,
        embed_dim: int,  # 嵌入维度
        num_heads: int,  # 注意力头数
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ):
        super().__init__()  # 调用父类初始化
        self.head_dim = embed_dim // num_heads  # 头维度
        self.scaling = self.head_dim**-0.5  # 缩放因子

        self.qkv_proj = QKVParallelLinear(  # QKV并行投影
            embed_dim, self.head_dim, num_heads, quant_config=quant_config
        )
        # After TP split, the local head count lives on the linear layer
        self.num_heads = self.qkv_proj.num_heads  # TP后的头数
        self.out_proj = RowParallelLinear(  # 输出投影
            embed_dim, embed_dim, bias=True, quant_config=quant_config
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Whisper注意力前向传播，使用SDPA"""
        batch_size, seq_len, _ = hidden_states.shape  # 获取批次和序列长度
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影
        q, k, v = qkv.chunk(3, dim=-1)  # 分离Q、K、V
        q = q * self.scaling  # 缩放Q

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(  # 重塑并转置Q
            0, 2, 1, 3
        )
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(  # 重塑并转置K
            0, 2, 1, 3
        )
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(  # 重塑并转置V
            0, 2, 1, 3
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(  # SDPA注意力
            q, k, v, scale=1.0
        )
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(  # 恢复形状
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output, _ = self.out_proj(attn_output)  # 通过输出投影
        return attn_output  # 返回注意力输出


class VoxtralWhisperEncoderLayer(nn.Module):
    """Voxtral Whisper编码器层"""

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ):
        super().__init__()  # 调用父类初始化
        embed_dim = config.d_model  # 嵌入维度
        self.self_attn = VoxtralWhisperAttention(  # 自注意力
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            quant_config=quant_config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)  # 自注意力层归一化
        self.activation_fn = get_act_fn(  # 激活函数
            getattr(config, "activation_function", "gelu"),
            quant_config=quant_config,
        )
        self.fc1 = ColumnParallelLinear(embed_dim, config.encoder_ffn_dim)  # FFN第一层
        self.fc2 = RowParallelLinear(config.encoder_ffn_dim, embed_dim)  # FFN第二层
        self.final_layer_norm = nn.LayerNorm(embed_dim)  # FFN层归一化

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """编码器层前向传播：自注意力 + FFN"""
        residual = hidden_states  # 保存残差
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 归一化
        hidden_states = self.self_attn(hidden_states)  # 自注意力
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 更新残差
        hidden_states = self.final_layer_norm(hidden_states)  # 归一化
        hidden_states, _ = self.fc1(hidden_states)  # FFN第一层
        hidden_states = self.activation_fn(hidden_states)  # 激活
        hidden_states, _ = self.fc2(hidden_states)  # FFN第二层
        hidden_states = residual + hidden_states  # 残差连接

        if hidden_states.dtype == torch.float16:  # FP16精度保护
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(  # 限制范围
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states  # 返回输出


class VoxtralWhisperEncoder(nn.Module):
    """Whisper encoder (Conv1d + positional embed + transformer + layer norm)."""  # Whisper编码器

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
    ):
        super().__init__()  # 调用父类初始化
        embed_dim = config.d_model  # 嵌入维度

        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)  # 第一个卷积
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)  # 第二个卷积（下采样）
        self.embed_positions = nn.Embedding(config.max_source_positions, embed_dim)  # 位置嵌入
        self.layers = nn.ModuleList(  # 编码器层列表
            [
                VoxtralWhisperEncoderLayer(config, quant_config)
                for _ in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)  # 最终层归一化

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """编码器前向传播
        Args:
            input_features: [batch, num_mel_bins, seq_len]
        Returns:
            [batch, seq_len // 2, d_model]
        """
        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))  # 卷积1+GELU
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))  # 卷积2+GELU
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # 调整维度顺序

        seq_len = inputs_embeds.shape[1]  # 序列长度
        position_ids = torch.arange(seq_len, device=inputs_embeds.device)  # 位置ID
        hidden_states = inputs_embeds + self.embed_positions(position_ids)  # 添加位置嵌入

        for layer in self.layers:  # 遍历编码器层
            hidden_states = layer(hidden_states)  # 通过当前层

        hidden_states = self.layer_norm(hidden_states)  # 最终归一化
        return hidden_states  # 返回编码器输出


class VoxtralForConditionalGeneration(nn.Module):
    """Voxtral: Whisper encoder + MLP projector + Llama decoder.
    
    HF weight prefixes:
        audio_tower.*           -> self.audio_tower (VoxtralWhisperEncoder)
        multi_modal_projector.* -> self.multi_modal_projector (AudioLanguageAdapter)
        language_model.*        -> self.language_model (LlamaForCausalLM)
    """  # Voxtral条件生成模型

    def __init__(
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        prefix: str = "",  # 参数前缀
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存配置

        audio_config = config.audio_config  # 音频配置
        text_config = config.text_config  # 文本配置

        # Ensure text_config has rope_parameters (transformers v5 compatibility)
        if not hasattr(text_config, "rope_parameters"):  # 兼容性：添加rope_parameters
            text_config.rope_parameters = {
                "rope_type": getattr(text_config, "rope_type", "default"),
                "rope_theta": getattr(text_config, "rope_theta", 10000.0),
            }
            if getattr(text_config, "rope_scaling", None):  # 如果有缩放
                text_config.rope_parameters.update(text_config.rope_scaling)  # 更新参数

        # Infer downsample_factor: intermediate_size / hidden_size for HF format
        self.downsample_factor = getattr(  # 下采样因子
            audio_config,
            "downsample_factor",
            audio_config.intermediate_size // audio_config.hidden_size,  # 推断
        )

        # Encoder (named audio_tower to match HF weight prefix directly)
        self.audio_tower = VoxtralWhisperEncoder(audio_config, quant_config)  # 音频编码器

        # Projector: input = d_model * downsample_factor, output = text_hidden_size
        adapter_input_dim = audio_config.d_model * self.downsample_factor  # 适配器输入维度
        self.multi_modal_projector = AudioLanguageAdapter(  # 多模态投影器
            hidden_size=adapter_input_dim,
            dim=text_config.hidden_size,
        )

        # Language model
        self.language_model = LlamaForCausalLM(text_config, quant_config=quant_config)  # 语言模型

        # Mel filter bank for raw waveform -> mel spectrogram
        self._init_mel_filters(audio_config)  # 初始化mel滤波器

        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 填充模式

    def _init_mel_filters(self, audio_config: PretrainedConfig):
        """Initialize mel filter bank for mel spectrogram computation."""  # 初始化mel滤波器
        self._window_size = getattr(audio_config, "window_size", 400)  # 窗口大小
        self._hop_length = getattr(audio_config, "hop_length", 160)  # 跳跃长度
        self._sampling_rate = getattr(audio_config, "sampling_rate", 16000)  # 采样率

        try:
            from mistral_common.audio import mel_filter_bank  # 导入mel滤波器
        except ImportError:
            raise ImportError(  # 缺少依赖
                "mistral_common is required for Voxtral. "
                "Install it with: pip install mistral_common"
            )

        mel_filters = mel_filter_bank(  # 计算mel滤波器
            num_frequency_bins=1 + self._window_size // 2,  # 频率bins数
            num_mel_bins=audio_config.num_mel_bins,  # mel bins数
            min_frequency=0.0,  # 最小频率
            max_frequency=8000.0,  # 最大频率
            sampling_rate=self._sampling_rate,  # 采样率
        )
        self.register_buffer(  # 注册为buffer
            "mel_filters", torch.tensor(mel_filters, dtype=torch.float32)
        )

    @property
    def _conv_downsample_factor(self) -> int:
        """获取卷积下采样因子"""
        return self.audio_tower.conv1.stride[0] * self.audio_tower.conv2.stride[0]  # 两个卷积步长相乘

    @property
    def _chunk_size(self) -> int:
        """获取chunk大小"""
        return (
            self.config.audio_config.max_source_positions * self._conv_downsample_factor  # 最大源位置乘以下采样因子
        )

    def _compute_mel_spectrogram(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram from raw waveform using STFT."""  # 从原始波形计算log-mel频谱图
        window = torch.hann_window(self._window_size, device=audio_waveform.device)  # Hann窗
        stft = torch.stft(  # 短时傅里叶变换
            audio_waveform,
            self._window_size,  # 窗口大小
            self._hop_length,  # 跳跃长度
            window=window,  # 窗函数
            return_complex=True,  # 返回复数
        )
        magnitudes = stft[..., :-1].abs() ** 2  # 幅度平方
        mel_spec = self.mel_filters.T @ magnitudes  # mel滤波
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()  # log10
        log_spec_max = log_spec.max()  # 最大值
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)  # 限制最小值
        log_spec = (log_spec + 4.0) / 4.0  # 归一化
        return log_spec  # 返回log-mel频谱图

    def _encode_audio(self, audio_waveforms: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode raw audio waveforms through mel spectrogram + whisper encoder."""  # 编码原始音频波形
        dtype = self.audio_tower.conv1.weight.dtype  # 编码器数据类型
        device = self.audio_tower.conv1.weight.device  # 编码器设备

        chunked_features: List[torch.Tensor] = []  # 分块特征列表
        chunks_per_example: List[int] = []  # 每个样本的chunk数
        chunk_size = self._chunk_size  # chunk大小
        # Pad raw audio to a multiple of chunk_samples so that silence is
        # properly converted to mel features (matching HF VoxtralProcessor).
        chunk_samples = chunk_size * self._hop_length  # 每个chunk的采样数

        for waveform in audio_waveforms:  # 遍历每个音频波形
            waveform = waveform.to(device=device, dtype=torch.float32)  # 转换类型和设备
            n_samples = waveform.shape[-1]  # 采样数
            target_samples = chunk_samples * math.ceil(n_samples / chunk_samples)  # 目标采样数（向上取整）
            if target_samples > n_samples:  # 需要填充
                waveform = torch.nn.functional.pad(
                    waveform, (0, target_samples - n_samples)  # 填充静音
                )
            mel = self._compute_mel_spectrogram(waveform)  # 计算mel频谱图
            chunks = mel.split(chunk_size, dim=-1)  # 分块
            chunked_features.extend(chunks)  # 添加到列表
            chunks_per_example.append(len(chunks))  # 记录chunk数

        if not chunked_features:  # 没有特征
            return []

        input_embeds = torch.stack(chunked_features).to(dtype)  # 堆叠并转换类型
        encoder_out = self.audio_tower(input_embeds)  # 通过编码器

        results = []  # 结果列表
        chunk_idx = 0  # chunk索引
        for n_chunks in chunks_per_example:  # 遍历每个样本
            result = encoder_out[chunk_idx : chunk_idx + n_chunks].flatten(0, 1)  # 展平chunk
            results.append(result)  # 添加到结果
            chunk_idx += n_chunks  # 更新索引

        return results  # 返回编码结果

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """对输入ID进行多模态填充"""
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)  # 使用填充模式

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encode audio waveforms -> downsample -> project."""  # 编码音频波形 -> 下采样 -> 投影
        audio_waveforms = [item.feature for item in items]  # 获取音频波形
        audio_embeddings = self._encode_audio(audio_waveforms)  # 编码音频

        # Downsample: reshape to merge adjacent frames
        for i, emb in enumerate(audio_embeddings):  # 遍历嵌入
            seq_len, dim = emb.shape  # 获取形状
            audio_embeddings[i] = emb.reshape(  # 下采样重塑
                seq_len // self.downsample_factor,
                dim * self.downsample_factor,
            )

        # Project through adapter
        packed = torch.cat(audio_embeddings, dim=0)  # 拼接所有嵌入
        packed = self.multi_modal_projector(packed)  # 通过投影器

        return packed  # 返回投影后的特征

    def forward(
        self,
        input_ids: torch.Tensor,  # 输入ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批次
        **kwargs: Any,
    ) -> torch.Tensor:
        """模型前向传播：音频特征提取 -> 多模态嵌入 -> 语言模型"""
        hidden_states = general_mm_embed_routine(  # 通用多模态嵌入例程
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={  # 数据嵌入函数
                Modality.AUDIO: self.get_audio_feature,  # 音频模态
            },
            positions=positions,
        )
        return hidden_states  # 返回隐藏状态

    def get_language_model(self) -> nn.Module:
        """获取语言模型"""
        return self.language_model  # 返回语言模型

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理编码器、投影器和语言模型的权重"""
        encoder_stacked = [  # 编码器堆叠参数映射
            ("qkv_proj", "q_proj", "q"),  # Q投影
            ("qkv_proj", "k_proj", "k"),  # K投影
            ("qkv_proj", "v_proj", "v"),  # V投影
        ]

        encoder_dict = dict(self.audio_tower.named_parameters())  # 编码器参数字典
        projector_dict = dict(self.multi_modal_projector.named_parameters())  # 投影器参数字典

        # Collect all weights; synthesise missing k_proj bias as zeros.
        weights_list = list(weights)  # 转为列表
        extra_weights = []  # 额外权重列表
        for name, w in weights_list:  # 遍历权重
            if name.startswith("audio_tower.") and ".self_attn.k_proj.weight" in name:  # K投影权重
                bias_name = name.replace(".weight", ".bias")  # 构造偏置名
                if not any(n == bias_name for n, _ in weights_list):  # 偏置不存在
                    extra_weights.append(  # 创建零偏置
                        (bias_name, torch.zeros(w.shape[0], dtype=w.dtype))
                    )
        weights_list.extend(extra_weights)  # 添加额外权重

        def llm_weights_generator():  # 语言模型权重生成器
            for name, w in weights_list:  # 遍历权重
                # Encoder weights
                if name.startswith("audio_tower."):  # 编码器权重
                    trimmed = name[len("audio_tower.") :]  # 移除前缀
                    loaded = False  # 是否已加载
                    for param_name, weight_name, shard_id in encoder_stacked:  # 遍历堆叠映射
                        if f".{weight_name}." in trimmed:  # 匹配
                            stacked_name = trimmed.replace(weight_name, param_name)  # 替换名称
                            if stacked_name in encoder_dict:  # 参数存在
                                param = encoder_dict[stacked_name]  # 获取参数
                                param.weight_loader(param, w, shard_id)  # 加载
                                loaded = True  # 标记已加载
                                break
                    if not loaded and trimmed in encoder_dict:  # 未匹配堆叠但参数存在
                        param = encoder_dict[trimmed]  # 获取参数
                        weight_loader = getattr(  # 获取加载器
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, w)  # 加载
                    continue  # 继续下一个权重

                # Projector weights
                if name.startswith("multi_modal_projector."):  # 投影器权重
                    trimmed = name[len("multi_modal_projector.") :]  # 移除前缀
                    trimmed = trimmed.replace("linear_1.", "w_in.").replace(  # 重命名
                        "linear_2.", "w_out."
                    )
                    if trimmed in projector_dict:  # 参数存在
                        param = projector_dict[trimmed]  # 获取参数
                        default_weight_loader(param, w)  # 加载
                    continue  # 继续

                # LLM weights
                if name.startswith("language_model."):  # 语言模型权重
                    name = name[len("language_model.") :]  # 移除前缀
                yield (name, w)  # 生成权重元组

        self.language_model.load_weights(llm_weights_generator())  # 加载语言模型权重


EntryClass = [VoxtralForConditionalGeneration]  # 入口类列表
