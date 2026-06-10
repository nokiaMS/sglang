# Phi4MM多模态处理器模块
# 本模块为Microsoft Phi-4-multimodal-instruct模型提供多模态数据处理功能
# 支持图像和音频输入的预处理和标记化

import logging  # 导入日志模块
from typing import List, Union  # 导入类型提示

from transformers.processing_utils import ProcessorMixin  # 导入HuggingFace处理器混入类

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.phi4mm import Phi4MMForCausalLM  # 导入Phi4MM模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊标记类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)  # 创建日志记录器


# It is an adapter of hf phi4 mm processor to make it work for sglang
# Ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L693
class Phi4MMProcessorAdapter(ProcessorMixin):  # Phi4MM处理器适配器，将HuggingFace处理器适配为sglang兼容
    def __init__(self, _processor) -> None:  # 初始化适配器，接收原始HuggingFace处理器
        self._processor = _processor  # 保存原始处理器引用

    def __call__(self, **kwargs):  # 调用适配器，处理输入数据并映射输出键名
        result = self._processor(**kwargs)  # 调用原始处理器获取结果

        # Map HuggingFace output keys to sglang standard keys
        key_mapping = {  # 定义HuggingFace输出键到sglang标准键的映射
            "input_image_embeds": "pixel_values",  # 图像嵌入映射为像素值
            "input_audio_embeds": "audio_features",  # 音频嵌入映射为音频特征
            "audio_embed_sizes": "audio_feature_lens",  # 音频嵌入大小映射为音频特征长度
        }
        for hf_key, sglang_key in key_mapping.items():  # 遍历键映射
            if hf_key in result:  # 如果HuggingFace键存在于结果中
                result[sglang_key] = result[hf_key]  # 将值映射到sglang标准键
                del result[hf_key]  # 删除原始HuggingFace键

        # Filter out None or empty tensors from the result.
        # This prevents the sglang function base_processor.collect_mm_items_from_processor_output()
        # from misclassifying audio content as image content, and vice versa.
        filtered_result = {  # 过滤掉None或空张量的结果
            k: v  # 键值对
            for k, v in result.items()  # 遍历结果中的所有键值对
            if v is not None and (not hasattr(v, "numel") or v.numel() > 0)  # 保留非None且有内容的值
        }
        return filtered_result  # 返回过滤后的结果


class Phi4MMMultimodalProcessor(BaseMultimodalProcessor):  # Phi4MM多模态处理器，继承基础多模态处理器
    models = [Phi4MMForCausalLM]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Phi4MM多模态处理器
        self.processor = Phi4MMProcessorAdapter(_processor)  # 创建适配器包装原始处理器
        super().__init__(hf_config, server_args, self.processor, *args, **kwargs)  # 调用父类初始化

        # the following CONSTANTS come from hugging-face microsoft/Phi-4-multimodal-instruct's processing_phi4mm.py file
        # ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py
        self.IMAGE_TOKEN = "<|endoftext10|>"  # 图像标记符号
        self.AUDIO_TOKEN = "<|endoftext11|>"  # 音频标记符号
        self.IM_TOKEN_ID = 200010  # 图像标记ID
        self.AUDIO_TOKEN_ID = 200011  # 音频标记ID
        self.AUDIO_SAMPLE_RATE = 16000  # 音频采样率

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=self.IMAGE_TOKEN,  # 图像标记
            image_token_id=self.IM_TOKEN_ID,  # 图像标记ID
            audio_token=self.AUDIO_TOKEN,  # 音频标记
            audio_token_id=self.AUDIO_TOKEN_ID,  # 音频标记ID
        ).build(self.processor)  # 构建标记对象

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        audio_data,  # 音频数据
        input_text,  # 输入文本
        request_obj,  # 请求对象
        **kwargs,  # 其他关键字参数
    ):
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            audio_data=audio_data,  # 音频数据
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
            audio_sample_rate=self.AUDIO_SAMPLE_RATE,  # 音频采样率
        )

        if base_output.audios is not None:  # 如果存在音频数据
            # hugging-face microsoft/Phi-4-multimodal-instruct's processing_phi4mm.py file requires the audio input to be tuple of (audio, sample_rate)
            # ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py
            base_output.audios = [  # 将音频数据转换为(音频, 采样率)元组格式
                (audio, self.AUDIO_SAMPLE_RATE) for audio in base_output.audios  # 为每个音频添加采样率
            ]

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和多模态标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入ID列表
            mm_items=mm_items,  # 多模态数据项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
            audio_token_id=self.mm_tokens.audio_token_id,  # 音频标记ID
        )
