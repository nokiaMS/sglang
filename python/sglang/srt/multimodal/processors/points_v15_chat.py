# Points-V15-Chat多模态处理器模块
# 本模块为Points-V15-Chat视觉模型提供图像数据处理功能
# 复用QwenVL图像处理器，并适配Points-V15-Chat的特殊配置

# Copy from qwen_vl.py, adapted for points-v15-chat

from typing import List, Union  # 导入类型提示

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.points_v15_chat import POINTSV15ChatModel  # 导入Points-V15-Chat模型类
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor  # 导入QwenVL图像处理器


class POINTSV15ChatProcessor(QwenVLImageProcessor):  # Points-V15-Chat处理器，继承QwenVL图像处理器
    models = [POINTSV15ChatModel]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Points-V15-Chat处理器
        # Compatible with POINTSV15Chat
        hf_config.vision_start_token_id = None  # 设置视觉起始标记ID为None，兼容Points-V15-Chat
        hf_config.vision_end_token_id = None  # 设置视觉结束标记ID为None，兼容Points-V15-Chat
        hf_config.video_token_id = None  # 设置视频标记ID为None，兼容Points-V15-Chat

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和多模态标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入ID列表
            mm_items=mm_items,  # 多模态数据项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
        )
