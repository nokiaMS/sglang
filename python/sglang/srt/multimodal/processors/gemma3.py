# Gemma3 多模态图像处理器模块
# 本模块实现了 Gemma3 模型的图像数据处理逻辑，
# 包括图像特殊标记的定义、图像数据的加载与处理等功能。
import re  # 导入正则表达式模块
from typing import Dict, List, Union  # 导入类型提示模块

from sglang.srt.managers.multimodal_processor import (  # 导入多模态处理器模块
    BaseMultimodalProcessor as SGLangBaseProcessor,  # 将基础处理器重命名为 SGLangBaseProcessor
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration  # 导入 Gemma3 条件生成模型类
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens  # 导入多模态特殊标记类

# Copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/image_processing_gemma3_fast.py
# will be removed in the future
# 从 HuggingFace transformers 仓库复制，未来将被移除


class Gemma3SGLangImageProcessor(SGLangBaseProcessor):  # Gemma3 SGLang 图像处理器类
    models = [Gemma3ForConditionalGeneration]  # 支持的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 Gemma3 图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法
        self.IM_START_TOKEN_ID = hf_config.boi_token_index  # 图像起始标记（beginning of image）的 token ID
        self.IM_END_TOKEN_ID = hf_config.eoi_token_index  # 图像结束标记（end of image）的 token ID
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            # The single, pre-expanded image token.
            image_token="<start_of_image>",  # 单个预展开的图像标记
            image_token_id=hf_config.image_token_index,  # 图像标记的 token ID
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(  # 匹配已展开图像标记的正则表达式
                r"<start_of_image>(?:(?:<image_soft_token>)*<end_of_image>)?"  # 匹配图像标记及其内容
            ),
        ).build(_processor)  # 使用处理器构建标记

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes, Dict]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
            discard_alpha_channel=True,  # 丢弃透明通道（RGBA 转 RGB）
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )
        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_start_id=self.IM_START_TOKEN_ID,  # 图像起始标记 ID
            im_end_id=self.IM_END_TOKEN_ID,  # 图像结束标记 ID
        )
