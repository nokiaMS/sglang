# 本文件实现 CLIP 模型的图像处理器，负责加载图像数据、
# 处理多模态输入并返回多模态处理器输出
from typing import List, Union  # 导入类型提示工具

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.clip import CLIPModel  # 导入 CLIP 模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基类和特殊 token 类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class ClipImageProcessor(BaseMultimodalProcessor):  # CLIP 图像处理器，继承自 BaseMultimodalProcessor
    models = [CLIPModel]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.mm_tokens = MultimodalSpecialTokens(image_token="<image>").build(
            _processor
        )  # 构建图像特殊 token 配置

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):  # 异步处理多模态数据
        """异步处理图像数据并返回多模态处理器输出"""
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并组合多模态数据
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(  # 返回处理器输出
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
        )
