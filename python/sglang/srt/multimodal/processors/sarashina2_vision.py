# Sarashina2-Vision多模态处理器模块
# 本模块为Sarashina2视觉语言模型提供图像数据处理功能
# 包含图像预处理器参数兼容性修补

from typing import List, Union  # 导入类型提示

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.sarashina2_vision import Sarashina2VisionForCausalLM  # 导入Sarashina2视觉模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊标记类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Sarashina2VisionProcessor(BaseMultimodalProcessor):  # Sarashina2视觉处理器，继承基础多模态处理器
    models = [Sarashina2VisionForCausalLM]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Sarashina2视觉处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        # Sarashina2Vision specific tokens (default is <|file|>)
        self.IMAGE_TOKEN = "<|file|>"  # 图像标记符号，默认为<|file|>
        self.IM_TOKEN_ID = getattr(hf_config, "image_token_index", 14)  # 获取图像标记ID，默认为14
        self.IM_START_ID = getattr(hf_config, "start_image_token_index", 102397)  # 获取图像起始标记ID
        self.IM_END_ID = getattr(hf_config, "end_image_token_index", 102398)  # 获取图像结束标记ID

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=self.IMAGE_TOKEN,  # 图像标记
            image_token_id=self.IM_TOKEN_ID,  # 图像标记ID
        ).build(_processor)  # 构建标记对象

        # Patch the processor's image processor to handle parameter compatibility
        if hasattr(_processor, "image_processor") and hasattr(  # 检查处理器是否有图像处理器
            _processor.image_processor, "_preprocess"
        ):
            original_preprocess = _processor.image_processor._preprocess  # 保存原始预处理方法

            def patched_preprocess(*args, **kwargs):  # 定义修补后的预处理方法
                # Filter kwargs to only include parameters that the custom _preprocess method accepts
                # Based on Sarashina2VisionImageProcessor._preprocess signature
                allowed_params = {  # 允许的参数集合
                    "do_resize",  # 是否缩放
                    "resample",  # 重采样方法
                    "do_rescale",  # 是否重新缩放
                    "rescale_factor",  # 重新缩放因子
                    "do_normalize",  # 是否归一化
                    "image_mean",  # 图像均值
                    "image_std",  # 图像标准差
                    "do_convert_rgb",  # 是否转换为RGB
                    "data_format",  # 数据格式
                    "input_data_format",  # 输入数据格式
                }
                filtered_kwargs = {  # 过滤参数，只保留允许的参数
                    k: v for k, v in kwargs.items() if k in allowed_params  # 过滤条件
                }
                return original_preprocess(*args, **filtered_kwargs)  # 调用原始方法

            _processor.image_processor._preprocess = patched_preprocess  # 替换预处理方法

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        """Process image data for Sarashina2Vision model using standard SGLang pattern."""  # 使用标准SGLang模式处理Sarashina2Vision模型的图像数据
        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态标记
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output=base_output,  # 基础输出
            mm_tokens=self.mm_tokens,  # 多模态标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids.tolist(),  # 输入ID列表
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
            im_start_id=self.IM_START_ID,  # 图像起始标记ID
            im_end_id=self.IM_END_ID,  # 图像结束标记ID
        )
