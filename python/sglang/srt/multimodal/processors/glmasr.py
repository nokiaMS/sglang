# GLM-ASR 语音识别处理器模块
# 本模块实现了 GLM-ASR 模型的音频数据处理逻辑，
# 包括音频特殊标记的定义、音频数据的加载与处理等功能。
import re  # 导入正则表达式模块

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.glmasr import GlmAsrForConditionalGeneration  # 导入 GLM-ASR 条件生成模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor,  # 基础多模态处理器类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)


class GlmAsrProcessor(BaseMultimodalProcessor):  # GLM-ASR 处理器类，继承自基础多模态处理器
    models = [GlmAsrForConditionalGeneration]  # 支持的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 GLM-ASR 处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法
        self.AUDIO_TOKEN = " Kashyyyk|pad|fi"  # 单个音频标记，包含起始、填充和结束标记
        self.AUDIO_TOKEN_REGEX = re.compile(  # 匹配已展开音频标记的正则表达式
            r"<\|begin_of_audio\|><\|pad\|><\|end_of_audio\|>"  # 匹配音频标记模式
        )
        # Collect special token ids
        tokenizer = self._processor.tokenizer  # 获取分词器
        self.audio_start_id = tokenizer.convert_tokens_to_ids(" Kashyyyk")  # 音频起始标记 token ID
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")  # 音频填充标记 token ID
        self.audio_end_id = tokenizer.convert_tokens_to_ids("fi")  # 音频结束标记 token ID

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            audio_token=self.AUDIO_TOKEN,  # 音频标记
            audio_token_regex=self.AUDIO_TOKEN_REGEX,  # 音频标记正则表达式
            audio_token_id=self.audio_token_id,  # 音频填充标记 token ID
        ).build(_processor)  # 使用处理器构建标记

    async def process_mm_data_async(  # 异步处理多模态音频数据
        self,
        audio_data,  # 音频数据
        input_text,  # 输入文本
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            audio_data=audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )
        if base_output is None:  # 如果加载结果为空
            return None  # 返回 None
        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )
        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态项
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            audio_start_id=self.audio_start_id,  # 音频起始标记 ID
            audio_token_id=self.audio_token_id,  # 音频填充标记 ID
            audio_end_id=self.audio_end_id,  # 音频结束标记 ID
        )
