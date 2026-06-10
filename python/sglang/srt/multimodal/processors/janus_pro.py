# Janus Pro 多模态图像处理器模块
# 本模块实现了 DeepSeek Janus Pro 模型的图像数据处理逻辑，
# 包括图像特殊标记的定义和图像数据的异步加载与处理。
from typing import List, Union  # 导入类型提示模块

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.deepseek_janus_pro import MultiModalityCausalLM  # 导入 DeepSeek Janus Pro 多模态因果语言模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor,  # 基础多模态处理器类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)


class JanusProImageProcessor(BaseMultimodalProcessor):  # Janus Pro 图像处理器类，继承自基础多模态处理器
    models = [MultiModalityCausalLM]  # 支持的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 Janus Pro 图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=_processor.image_token,  # 图像标记，从处理器获取
            image_token_id=_processor.image_id,  # 图像标记 ID，从处理器获取
        ).build(_processor)  # 使用处理器构建标记

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        **kwargs,  # 关键字参数
    ):
        base_out = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_out, self.mm_tokens, prompt=base_out.input_text  # 传入基础输出、标记和提示文本
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态项
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            im_start_id=self._processor.image_start_id,  # 图像起始标记 ID
            im_end_id=self._processor.image_end_id,  # 图像结束标记 ID
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
        )
