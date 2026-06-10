# Mllama4（Llama4）多模态处理器模块
# 实现Llama4视觉语言模型的多模态数据处理
from typing import List, Union  # 导入类型提示

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration  # 导入Llama4模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊令牌类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Mllama4ImageProcessor(BaseMultimodalProcessor):  # Mllama4图像处理器类
    models = [Llama4ForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Mllama4图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.vision_config = hf_config.vision_config  # 保存视觉配置
        self.text_config = hf_config.text_config  # 保存文本配置
        self.IM_START_TOKEN_ID = hf_config.boi_token_index  # 图像开始令牌ID
        self.IM_END_TOKEN_ID = hf_config.eoi_token_index  # 图像结束令牌ID
        self.IM_TOKEN_ID = hf_config.image_token_index  # 图像令牌ID
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            image_token=_processor.image_token,  # 从处理器获取图像令牌
            image_token_id=self.IM_TOKEN_ID,  # 图像令牌ID
        ).build(_processor)  # 构建令牌映射

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
        )

        # Process the prompt and images
        # 处理提示和图像
        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和特殊令牌
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入ID列表
            mm_items=mm_items,  # 多模态数据项
            im_start_id=self.IM_START_TOKEN_ID,  # 图像开始令牌ID
            im_end_id=self.IM_END_TOKEN_ID,  # 图像结束令牌ID
            im_token_id=self.IM_TOKEN_ID,  # 图像令牌ID
        )
