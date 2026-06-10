# Kimi VL 多模态图像处理器模块
# 本模块实现了 Kimi VL 模型的图像数据处理逻辑，
# 包括图像特殊标记的定义、图像数据的异步加载与处理，
# 以及基于网格的多模态数据构建功能。
import re  # 导入正则表达式模块
from typing import Dict, List, Union  # 导入类型提示模块

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.kimi_vl import KimiVLForConditionalGeneration  # 导入 Kimi VL 条件生成模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor as SGLangBaseProcessor,  # 将基础处理器重命名为 SGLangBaseProcessor
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入多模态特殊标记类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin  # 导入 Kimi 网格多模态数据混入类


# Compatible with KimiVLForConditionalGeneration
class KimiVLImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):  # Kimi VL 图像处理器类，兼容 KimiVLForConditionalGeneration
    models = [KimiVLForConditionalGeneration]  # 支持的模型列表
    gpu_image_decode = False  # KimiVL HF processor does not support tensor inputs  # KimiVL HuggingFace 处理器不支持张量输入

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 Kimi VL 图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token="<|media_pad|>",  # 媒体填充标记
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,  # 媒体占位符标记 ID
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),  # 匹配媒体填充标记的正则表达式
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
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):  # 获取多模态数据，使用网格信息构建
        img_grid_thw = kwargs.get("img_grid_thw", None)  # 获取图像网格信息
        return self._build_kimi_mm_data_from_grids(  # 调用混入类的方法构建多模态数据
            prompt=prompt,  # 提示文本
            embeddings=embeddings,  # 嵌入
            image_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
            img_grid_thw=img_grid_thw,  # 图像网格信息
        )
