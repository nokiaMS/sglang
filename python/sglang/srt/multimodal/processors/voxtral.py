# Voxtral多模态处理器模块
# 本模块为Voxtral语音转文本模型提供音频数据处理功能
# 支持音频标记计算、输入ID构建和Mistral提示解析

"""Multimodal processor for Voxtral (speech-to-text) models."""

import math  # 导入数学模块
import re  # 导入正则表达式模块
from typing import Dict, List, Optional  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalProcessorOutput,  # 多模态处理器输出
)
from sglang.srt.models.voxtral import VoxtralForConditionalGeneration  # 导入Voxtral模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊标记类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

# Special token IDs for Voxtral audio (from tekken.json vocabulary)
AUDIO_TOKEN_ID = 24  # [AUDIO]，音频标记ID
BEGIN_AUDIO_TOKEN_ID = 25  # [BEGIN_AUDIO]，音频起始标记ID
INST_TOKEN_ID = 3  # [INST]，指令标记ID

# Placeholder for load_mm_data regex matching.
# encode("[AUDIO]") does NOT produce token 24; actual token insertion
# is handled in _build_input_ids_with_audio.
AUDIO_PLACEHOLDER = "[AUDIO]"  # 音频占位符，用于正则匹配
AUDIO_PLACEHOLDER_REGEX = re.compile(r"\[AUDIO\]")  # 音频占位符正则表达式


class VoxtralMultimodalProcessor(BaseMultimodalProcessor):  # Voxtral多模态处理器，继承基础多模态处理器
    models = [VoxtralForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Voxtral多模态处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        audio_config = getattr(hf_config, "audio_config", None)  # 获取音频配置
        self.audio_token_id = getattr(hf_config, "audio_token_id", AUDIO_TOKEN_ID)  # 音频标记ID
        self.sampling_rate = getattr(audio_config, "sampling_rate", 16000)  # 采样率
        self.hop_length = getattr(audio_config, "hop_length", 160)  # 跳跃长度
        self.max_source_positions = getattr(audio_config, "max_source_positions", 1500)  # 最大源位置数
        self.conv_downsample = 2  # conv1 stride=1 * conv2 stride=2，卷积下采样因子
        self.downsample_factor = getattr(  # 下采样因子
            audio_config,
            "downsample_factor",  # 优先从配置获取
            getattr(audio_config, "intermediate_size", 5120)  # 否则从中间尺寸计算
            // getattr(audio_config, "hidden_size", 1280),  # 中间尺寸除以隐藏尺寸
        )

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            audio_token=AUDIO_PLACEHOLDER,  # 音频占位符
            audio_token_regex=AUDIO_PLACEHOLDER_REGEX,  # 音频占位符正则
            audio_token_id=self.audio_token_id,  # 音频标记ID
        ).build(_processor)  # 构建标记对象

    def _compute_audio_token_count(self, n_samples: int) -> int:  # 计算给定音频长度对应的[AUDIO]标记数量
        """Compute the number of [AUDIO] tokens for a given audio length."""
        mel_frames = n_samples / self.hop_length  # 计算梅尔频谱帧数
        chunk_size = self.max_source_positions * self.conv_downsample  # 计算块大小
        n_chunks = math.ceil(mel_frames / chunk_size) if mel_frames > 0 else 1  # 计算块数
        tokens_per_chunk = self.max_source_positions // self.downsample_factor  # 每块标记数
        return n_chunks * tokens_per_chunk  # 返回总标记数

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data,  # 图像数据（未使用）
        audio_data,  # 音频数据
        input_text,  # 输入文本
        request_obj,  # 请求对象
        **kwargs,  # 其他关键字参数
    ) -> Optional[MultimodalProcessorOutput]:
        if not audio_data:  # 如果没有音频数据
            return None  # 返回None

        # Insert [AUDIO] placeholders into prompt for load_mm_data's regex
        prompt_with_placeholders = self._insert_audio_placeholders(  # 在提示中插入音频占位符
            input_text, len(audio_data)  # 输入文本和音频数量
        )

        # load_mm_data handles async loading, format detection, resampling.
        # process_and_combine_mm_data cannot be used: HF VoxtralProcessor.__call__
        # does not support audio (only apply_chat_template does).
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=prompt_with_placeholders,  # 带占位符的提示
            audio_data=audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
            audio_sample_rate=self.sampling_rate,  # 采样率
        )
        if base_output is None:  # 如果基础输出为None
            return None  # 返回None

        # Convert loaded audio to tensors
        waveforms: List[torch.Tensor] = []  # 波形张量列表
        for audio in base_output.audios:  # 遍历加载的音频
            wav = torch.as_tensor(audio, dtype=torch.float32)  # 转为float32张量
            if wav.dim() > 1:  # 如果是多通道
                wav = wav.mean(dim=0)  # 取均值转为单通道
            waveforms.append(wav)  # 添加到列表

        # Compute audio token counts and build input_ids with audio tokens
        audio_token_counts = [  # 计算每段音频的标记数
            self._compute_audio_token_count(wav.shape[-1]) for wav in waveforms  # 根据样本数计算
        ]
        tokenizer = getattr(self._processor, "tokenizer", self._processor)  # 获取分词器
        input_ids = self._build_input_ids_with_audio(  # 构建包含音频标记的输入ID
            tokenizer, input_text, audio_token_counts  # 分词器、文本和标记数
        )

        # Find offsets of [AUDIO] token runs and build mm_items
        audio_offsets = self._find_audio_offsets(input_ids, self.audio_token_id)  # 查找音频标记偏移量
        mm_items = []  # 多模态数据项列表
        for i, wav in enumerate(waveforms):  # 遍历波形
            item = MultimodalDataItem(feature=wav, modality=Modality.AUDIO)  # 创建音频数据项
            if i < len(audio_offsets):  # 如果有对应偏移量
                item.offsets = [audio_offsets[i]]  # 设置偏移量
            mm_items.append(item)  # 添加到列表

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids,  # 输入ID
            mm_items=mm_items,  # 多模态数据项
            audio_token_id=self.audio_token_id,  # 音频标记ID
        )

    @staticmethod
    def _insert_audio_placeholders(prompt: str, n_audio: int) -> str:  # 在提示中插入音频占位符
        """Insert [AUDIO] placeholder texts into the prompt for load_mm_data."""  # 在提示中插入[AUDIO]占位符文本
        placeholders = AUDIO_PLACEHOLDER * n_audio  # 生成占位符字符串
        # Insert after the last [INST] marker if present
        last_inst = prompt.rfind("[INST]")  # 查找最后一个[INST]标记
        if last_inst >= 0:  # 如果找到
            insert_pos = last_inst + len("[INST]")  # 计算插入位置
            return prompt[:insert_pos] + placeholders + prompt[insert_pos:]  # 插入占位符
        return placeholders + prompt  # 否则在开头插入

    @staticmethod
    def _find_audio_offsets(input_ids: List[int], audio_token_id: int) -> List[tuple]:  # 查找音频标记的连续运行偏移量
        """Find consecutive runs of audio_token_id in input_ids."""  # 在input_ids中查找audio_token_id的连续运行
        offsets = []  # 偏移量列表
        start = None  # 起始位置
        for i, tok_id in enumerate(input_ids):  # 遍历输入ID
            if tok_id == audio_token_id:  # 如果是音频标记
                if start is None:  # 如果未开始
                    start = i  # 记录起始位置
            elif start is not None:  # 如果已开始且遇到非音频标记
                offsets.append((start, i - 1))  # 添加偏移量
                start = None  # 重置起始位置
        if start is not None:  # 如果最后还有未结束的运行
            offsets.append((start, len(input_ids) - 1))  # 添加最后一个偏移量
        return offsets  # 返回偏移量列表

    def _build_input_ids_with_audio(  # 构建包含音频标记的输入ID
        self,
        tokenizer,  # 分词器
        input_text: str,  # 输入文本
        audio_token_counts: List[int],  # 音频标记数量列表
    ) -> List[int]:
        """Build input_ids by tokenizing text and inserting audio tokens.

        The input_text is a decoded Mistral prompt (from text-only
        apply_chat_template).  We re-tokenize to get proper special tokens
        (BOS, [INST], [/INST]), then insert [BEGIN_AUDIO] + [AUDIO]*N after
        the last [INST].
        """
        messages = self._parse_mistral_prompt(input_text)  # 解析Mistral格式提示
        try:  # 尝试使用聊天模板
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True)  # 应用聊天模板
        except (ValueError, KeyError):  # 如果失败
            # Fallback if prompt parsing produces malformed messages
            input_ids = tokenizer.encode(input_text)  # 回退到简单编码

        # Insert audio tokens after the last [INST]
        inst_positions = [i for i, t in enumerate(input_ids) if t == INST_TOKEN_ID]  # 查找所有[INST]位置
        insert_pos = (inst_positions[-1] + 1) if inst_positions else 1  # 在最后一个[INST]后插入

        audio_tokens = []  # 音频标记列表
        for count in audio_token_counts:  # 遍历每段音频的标记数
            audio_tokens.append(BEGIN_AUDIO_TOKEN_ID)  # 添加音频起始标记
            audio_tokens.extend([AUDIO_TOKEN_ID] * count)  # 添加音频内容标记

        return input_ids[:insert_pos] + audio_tokens + input_ids[insert_pos:]  # 拼接并返回

    @staticmethod
    def _parse_mistral_prompt(prompt: str) -> List[Dict[str, str]]:  # 解析Mistral格式提示为消息列表
        """Parse a Mistral-formatted prompt into a list of messages."""  # 将Mistral格式的提示解析为消息列表
        messages = []  # 消息列表
        text = prompt.strip()  # 去除首尾空白

        for marker in ["<s>", "</s>"]:  # 移除BOS和EOS标记
            text = text.replace(marker, "")  # 替换为空
        text = text.strip()  # 再次去除空白

        # Extract system prompt
        system_match = re.search(  # 查找系统提示
            r"\[SYSTEM_PROMPT\]\s*(.*?)\s*\[/SYSTEM_PROMPT\]", text, re.DOTALL  # 匹配系统提示标记
        )
        if system_match:  # 如果找到系统提示
            messages.append(  # 添加系统消息
                {"role": "system", "content": system_match.group(1).strip()}  # 系统消息
            )
            text = text[: system_match.start()] + text[system_match.end() :]  # 移除系统提示
            text = text.strip()  # 去除空白

        # Split by [INST] / [/INST]
        parts = re.split(r"\[/?INST\]", text)  # 按[INST]和[/INST]分割
        for i, part in enumerate(parts):  # 遍历分割后的部分
            part = part.strip()  # 去除空白
            if not part:  # 如果为空
                continue  # 跳过
            if i % 2 == 1:  # 奇数部分为用户消息
                messages.append({"role": "user", "content": part})  # 添加用户消息
            elif i > 0:  # 偶数部分为助手消息（跳过第一部分）
                messages.append({"role": "assistant", "content": part})  # 添加助手消息

        if not messages:  # 如果没有解析到消息
            messages.append({"role": "user", "content": text})  # 将整个文本作为用户消息

        return messages  # 返回消息列表
