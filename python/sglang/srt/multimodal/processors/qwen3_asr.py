# Qwen3-ASR多模态处理器模块
# 本模块为Qwen3自动语音识别模型提供音频数据处理功能
# 支持M-RoPE位置编码和音频特征提取

import re  # 导入正则表达式模块
from typing import Union  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput  # 导入模态枚举和处理器输出类
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration  # 导入Qwen3-ASR模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊标记类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"  # 音频占位符标记

DEFAULT_ASR_PROMPT = (  # 默认ASR提示模板
    f"<|im_start|>user\n"  # 用户消息开始
    f"{AUDIO_PLACEHOLDER}"  # 音频占位符
    f"<|im_end|>\n"  # 用户消息结束
    f"<|im_start|>assistant\n"  # 助手消息开始
)


class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):  # Qwen3-ASR多模态处理器，继承基础多模态处理器
    models = [Qwen3ASRForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Qwen3-ASR多模态处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.AUDIO_TOKEN = AUDIO_PLACEHOLDER  # 设置音频标记为占位符
        self.AUDIO_TOKEN_REGEX = re.compile(  # 编译音频标记的正则表达式
            r"<\|audio_start\|>(?:<\|audio_pad\|>)+<\|audio_end\|>"  # 匹配音频开始、填充和结束标记
        )
        tokenizer = self._processor.tokenizer  # 获取分词器
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")  # 获取音频开始标记ID
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")  # 获取音频填充标记ID
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")  # 获取音频结束标记ID

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            audio_token=self.AUDIO_TOKEN,  # 音频标记
            audio_token_regex=self.AUDIO_TOKEN_REGEX,  # 音频标记正则表达式
            audio_token_id=self.audio_token_id,  # 音频标记ID
        ).build(_processor)  # 构建标记对象

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})  # 将特征注意力掩码映射为音频模态

    def _build_transcription_prompt(self, input_text: Union[str, list]) -> str:  # 构建转录提示文本
        # TODO: support `force_language`
        if isinstance(input_text, list):  # 如果输入文本是列表（标记ID列表）
            input_text = self._tokenizer.decode(input_text)  # 解码为文本
        if not input_text or not input_text.strip():  # 如果输入文本为空或仅含空白
            return DEFAULT_ASR_PROMPT  # 返回默认ASR提示
        return input_text  # 返回输入文本

    def compute_mrope_positions(self, input_ids, mm_items):  # 计算M-RoPE位置编码
        if isinstance(input_ids, list):  # 如果输入ID是列表
            seq_len = len(input_ids)  # 获取序列长度
        else:  # 否则为张量
            seq_len = input_ids.shape[-1] if input_ids.dim() > 1 else input_ids.shape[0]  # 根据维度获取序列长度
        positions = torch.arange(seq_len, dtype=torch.long)  # 生成位置序列
        mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()  # 扩展为3维M-RoPE位置
        return mrope_positions, torch.tensor([0], dtype=torch.long)  # 返回位置和增量

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        audio_data=None,  # 音频数据
        input_text=None,  # 输入文本
        request_obj=None,  # 请求对象
        **kwargs,  # 其他关键字参数
    ):
        if not audio_data:  # 如果没有音频数据
            return None  # 返回None

        prompt = self._build_transcription_prompt(input_text)  # 构建转录提示

        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=prompt,  # 提示文本
            audio_data=audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
        )
        if base_output is None:  # 如果基础输出为None
            return None  # 返回None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和多模态标记
        )

        mrope_positions, mrope_position_delta = self.compute_mrope_positions(  # 计算M-RoPE位置
            input_ids, mm_items  # 输入ID和多模态数据项
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids.tolist(),  # 输入ID列表
            audio_start_id=self.audio_start_id,  # 音频开始标记ID
            audio_token_id=self.audio_token_id,  # 音频标记ID
            audio_end_id=self.audio_end_id,  # 音频结束标记ID
            mrope_positions=mrope_positions,  # M-RoPE位置
            mrope_position_delta=mrope_position_delta,  # M-RoPE位置增量
        )
