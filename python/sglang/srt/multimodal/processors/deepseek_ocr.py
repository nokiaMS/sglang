# 本文件实现 DeepSeek OCR 模型的多模态处理器，负责配置 OCR 模式、
# 加载图像数据、处理多模态输入并返回带有图像 token ID 的处理器输出
from typing import List, Union  # 导入类型提示工具

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM  # 导入 DeepSeek OCR 模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基类和特殊 token 类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class DeepseekOCRProcessor(BaseMultimodalProcessor):  # DeepSeek OCR 处理器，继承自 BaseMultimodalProcessor
    models = [DeepseekOCRForCausalLM]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化
        _processor.image_size = 640  # 设置图像尺寸为 640
        _processor.ocr2_mode = (  # 判断是否为 OCR2 模式
            str(
                getattr(getattr(hf_config, "vision_config", None), "model_name", "")
            ).lower()
            == "deepencoderv2"  # 视觉配置的模型名是否为 DeepEncoderV2
            or getattr(getattr(hf_config, "projector_config", None), "input_dim", None)
            == 896  # 或投影器输入维度是否为 896
        )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.mm_tokens = MultimodalSpecialTokens(  # 构建特殊 token 配置
            image_token="<image>", image_token_id=self._processor.image_token_id
        ).build(_processor)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):  # 异步处理多模态数据
        """异步处理图像数据并返回带有图像 token ID 的处理器输出"""
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
            im_token_id=self.mm_tokens.image_token_id,  # 包含图像 token ID
        )
