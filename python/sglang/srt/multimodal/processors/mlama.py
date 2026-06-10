# Mllama（Meta-Llama）多模态处理器模块
# 实现Mllama视觉语言模型的多模态数据处理
from typing import List, Union  # 导入类型提示

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.mllama import MllamaForConditionalGeneration  # 导入Mllama模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊令牌类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MllamaImageProcessor(BaseMultimodalProcessor):  # Mllama图像处理器类
    models = [MllamaForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Mllama图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            image_token=self._processor.image_token,  # 从处理器获取图像令牌
            image_token_id=self._processor.image_token_id,  # 从处理器获取图像令牌ID
        ).build(_processor)  # 构建令牌映射

    async def process_mm_data_async(  # 异步处理多模态数据
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_out = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_out, self.mm_tokens  # 基础输出和特殊令牌
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids.tolist(),  # 输入ID列表
            im_token_id=self.mm_tokens.image_token_id,  # 图像令牌ID
        )
