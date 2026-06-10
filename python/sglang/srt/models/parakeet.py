# Parakeet音频编码器推理实现文件
# 本文件实现了Parakeet音频编码器及其投影模块
# 包含音频投影、编码器封装、特征提取器和音频分片等功能

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 SGLang Team
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/parakeet.py
#
# Audio encoder component used by models/nano_nemotron_vl.py  # 由nano_nemotron_vl模型使用的音频编码器组件

from collections.abc import Iterable  # 导入可迭代类型
from dataclasses import asdict  # 导入数据类转字典工具

import numpy as np  # 导入NumPy
import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
from transformers import ParakeetEncoder as HFParakeetEncoder  # 导入HuggingFace Parakeet编码器
from transformers import ParakeetFeatureExtractor, PretrainedConfig  # 导入特征提取器和预训练配置

from sglang.srt.configs.parakeet import ExtractorConfig, ParakeetConfig  # 导入Parakeet配置
from sglang.srt.layers.activation import ReLU2  # 导入ReLU2激活函数
from sglang.srt.layers.layernorm import RMSNorm  # 导入RMS归一化
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器


class ParakeetProjection(nn.Module):  # Parakeet投影模块，将音频特征投影到LLM空间
    def __init__(self, config: ParakeetConfig) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        sound_hidden_size = config.hidden_size  # 音频隐藏大小
        proj_hidden_size = config.projection_hidden_size  # 投影隐藏大小
        llm_hidden_size = config.llm_hidden_size  # LLM隐藏大小
        bias = config.projection_bias  # 投影偏置

        self.norm = RMSNorm(sound_hidden_size, eps=config.projection_eps)  # RMS归一化
        self.linear1 = nn.Linear(sound_hidden_size, proj_hidden_size, bias=bias)  # 第一个线性层
        self.activation = ReLU2()  # ReLU2激活函数
        self.linear2 = nn.Linear(proj_hidden_size, llm_hidden_size, bias=bias)  # 第二个线性层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        hidden_states = self.norm(hidden_states)  # 归一化
        hidden_states = self.linear1(hidden_states)  # 通过第一个线性层
        hidden_states = self.activation(hidden_states)  # 应用激活函数
        hidden_states = self.linear2(hidden_states)  # 通过第二个线性层
        return hidden_states  # 返回投影后的隐藏状态


class ProjectedParakeet(nn.Module):  # 带投影的Parakeet编码器封装
    def __init__(  # 初始化函数
        self,
        config: PretrainedConfig,  # 预训练配置
        *,  # 强制关键字参数
        dtype: torch.dtype,  # 数据类型
        llm_hidden_size: int,  # LLM隐藏大小
        max_model_len: int,  # 最大模型长度
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = ParakeetConfig.from_hf_config(  # 从HuggingFace配置创建Parakeet配置
            config, llm_hidden_size=llm_hidden_size, max_model_len=max_model_len  # 传入参数
        )
        self.encoder = HFParakeetEncoder(self.config)  # 创建HuggingFace Parakeet编码器
        self.encoder = self.encoder.to(dtype)  # 转换数据类型
        self.projection = ParakeetProjection(self.config)  # 创建投影模块
        self.projection = self.projection.to(dtype)  # 转换数据类型

    def forward(  # 前向传播函数，执行编码和投影
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None  # 输入特征和注意力掩码
    ) -> torch.Tensor:
        outputs = self.encoder(  # 通过编码器
            input_features=input_features, attention_mask=attention_mask  # 传入输入特征和掩码
        )
        outputs = outputs.last_hidden_state  # 获取最后一层隐藏状态
        outputs = self.projection(outputs)  # 通过投影
        return outputs  # 返回投影结果

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:  # 加载权重函数
        loaded_params: set[str] = set()  # 已加载参数集合
        params_dict = dict(self.named_parameters())  # 参数字典
        buffers_dict = dict(self.named_buffers())  # 缓冲区字典

        if isinstance(weights, dict):  # 如果权重是字典
            weights_list = list(weights.items())  # 转换为列表
        else:  # 否则
            weights_list = list(weights)  # 转换为列表

        for name, weight in weights_list:  # 遍历权重
            if name.startswith("sound_encoder.encoder.feature_extractor."):  # 如果是特征提取器权重
                continue  # 跳过
            if name.startswith("sound_encoder."):  # 如果是音频编码器权重
                target_name = name[len("sound_encoder.") :]  # 移除前缀
            elif name.startswith("sound_projection."):  # 如果是音频投影权重
                target_name = f"projection.{name[len('sound_projection.'):]}"  # 映射到投影模块
            else:  # 否则
                continue  # 跳过

            target = params_dict.get(target_name)  # 在参数字典中查找
            if target is None:  # 如果未找到
                target = buffers_dict.get(target_name)  # 在缓冲区字典中查找
            if target is None:  # 如果仍未找到
                continue  # 跳过
            weight_loader = getattr(target, "weight_loader", default_weight_loader)  # 获取权重加载器
            with torch.no_grad():  # 不计算梯度
                weight_loader(target, weight)  # 加载权重
            loaded_params.add(target_name)  # 添加到已加载集合

        return loaded_params  # 返回已加载参数集合


class ParakeetExtractor(ParakeetFeatureExtractor):  # Parakeet特征提取器，继承自HuggingFace
    def __init__(self, config: PretrainedConfig) -> None:  # 初始化函数
        self.config = ExtractorConfig.from_hf_config(config)  # 从HuggingFace配置创建提取器配置
        super().__init__(**asdict(self.config))  # 调用父类初始化
        self._clip_target_samples = int(  # 计算目标裁剪样本数
            round(self.config.clip_duration_s * self.sampling_rate)  # 裁剪时长乘以采样率
        )
        self._tail_min_samples = int(  # 计算最小尾部样本数
            round(self.config.clip_min_duration_s * self.sampling_rate)  # 最小裁剪时长乘以采样率
        )

    def _clip_sizes(self, audio_len: int) -> list[int]:  # 计算裁剪大小列表
        audio_len = max(audio_len, self._tail_min_samples)  # 确保音频长度不小于最小尾部
        num_full_clips, remainder = divmod(audio_len, self._clip_target_samples)  # 计算完整裁剪数和余数
        clip_sizes = [self._clip_target_samples] * num_full_clips  # 完整裁剪大小列表
        if remainder > 0:  # 如果有余数
            clip_sizes.append(max(remainder, self._tail_min_samples))  # 添加尾部裁剪
        return clip_sizes  # 返回裁剪大小列表

    def _subsampling_output_length(self, length: int) -> int:  # 计算子采样输出长度
        import math  # 导入数学模块

        kernel_size = self.config.subsampling_conv_kernel_size  # 子采样卷积核大小
        stride = self.config.subsampling_conv_stride  # 子采样步幅
        num_layers = int(math.log2(self.config.subsampling_factor))  # 子采样层数
        add_pad = (kernel_size - 1) // 2 * 2 - kernel_size  # 填充量
        for _ in range(num_layers):  # 遍历每一层
            length = int(math.floor((length + add_pad) / stride + 1.0))  # 计算输出长度
        return max(1, length)  # 返回输出长度，至少为1

    def audio_token_count(self, audio_len: int) -> int:  # 计算音频token数量
        total_tokens = 0  # 总token数
        for clip_size in self._clip_sizes(audio_len):  # 遍历每个裁剪
            num_frames = clip_size // self.hop_length  # 帧数
            total_tokens += self._subsampling_output_length(num_frames)  # 累加子采样输出长度
        return max(1, total_tokens)  # 返回总token数，至少为1

    def split_audio_into_clips(self, audio: np.ndarray) -> list[np.ndarray]:  # 将音频分割为多个裁剪片段
        assert audio.ndim == 1  # 断言音频是一维的
        audio_len = int(audio.shape[0])  # 音频长度
        clip_sizes = self._clip_sizes(audio_len)  # 裁剪大小列表
        target_len = sum(clip_sizes)  # 目标总长度
        if audio_len < target_len:  # 如果音频长度不足
            audio = np.pad(audio, (0, target_len - audio_len))  # 填充音频

        clips = list[np.ndarray]()  # 裁剪片段列表
        offset = 0  # 偏移量
        for clip_size in clip_sizes:  # 遍历每个裁剪大小
            clips.append(audio[offset : offset + clip_size])  # 添加裁剪片段
            offset += clip_size  # 更新偏移量
        return clips  # 返回裁剪片段列表

    def __call__(self, raw_speech: list[np.ndarray], *args, **kwargs):  # 调用函数，处理原始语音
        audio_clips = list[np.ndarray]()  # 音频裁剪列表
        audio_num_clips = list[int]()  # 每个音频的裁剪数量列表
        for audio in raw_speech:  # 遍历每段原始语音
            clips = self.split_audio_into_clips(audio)  # 分割为裁剪
            audio_clips.extend(clips)  # 扩展到裁剪列表
            audio_num_clips.append(len(clips))  # 记录裁剪数量

        outputs = super().__call__(audio_clips, *args, **kwargs)  # 调用父类处理
        outputs["audio_num_clips"] = audio_num_clips  # 添加裁剪数量信息
        return outputs  # 返回输出

    @staticmethod
    def audio_length(raw_config: PretrainedConfig, audio_tokens: int) -> int:  # 计算音频长度
        config = ExtractorConfig.from_hf_config(raw_config)  # 从配置创建提取器配置
        return int(audio_tokens * config.subsampling_factor * config.hop_length)  # 计算音频长度
