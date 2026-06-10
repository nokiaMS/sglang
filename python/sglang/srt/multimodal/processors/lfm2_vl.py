# LFM2-VL多模态处理器模块
# 实现LFM2-VL视觉语言模型的多模态数据处理，支持SigLip2 NaFlex可变分辨率分块
# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal processor for LFM2-VL models with SigLip2 NaFlex support."""  # LFM2-VL模型的多模态处理器，支持SigLip2 NaFlex

from typing import List, Union  # 导入类型提示

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput  # 导入模态枚举和多模态处理器输出类
from sglang.srt.models.lfm2_vl import Lfm2VlForConditionalGeneration  # 导入LFM2-VL模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入多模态特殊令牌类
    MultimodalSpecialTokens,
)


class Lfm2VlImageProcessor(SGLangBaseProcessor):  # LFM2-VL图像处理器类，继承自基础多模态处理器
    """Multimodal processor for LFM2-VL vision-language models.
    # LFM2-VL视觉语言模型的多模态处理器

    Uses the base class load_mm_data + process_and_combine_mm_data flow.
    # 使用基类的load_mm_data + process_and_combine_mm_data流程
    The HF processor handles NaFlex variable-resolution tiling internally.
    # HF处理器内部处理NaFlex可变分辨率分块
    """

    models = [Lfm2VlForConditionalGeneration]  # 关联的模型列表
    gpu_image_decode = False  # 禁用GPU图像解码

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化LFM2-VL图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        self.IMAGE_TOKEN_ID = hf_config.image_token_id  # 从配置获取图像令牌ID
        self.IMAGE_TOKEN = "<image>"  # 图像令牌字符串

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            image_token=self.IMAGE_TOKEN,  # 图像令牌
            image_token_id=hf_config.image_token_id,  # 图像令牌ID
        ).build(_processor)  # 构建令牌映射

        # Register NaFlex-specific HF processor outputs so
        # collect_mm_items_from_processor_output picks them up
        # 注册NaFlex特有的HF处理器输出，以便collect_mm_items_from_processor_output能识别
        self.ATTR_NAME_TO_MODALITY["pixel_attention_mask"] = Modality.IMAGE  # 像素注意力掩码映射到图像模态
        self.ATTR_NAME_TO_MODALITY["spatial_shapes"] = Modality.IMAGE  # 空间形状映射到图像模态

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        audio_data,  # 音频数据
        input_text: str,  # 输入文本
        request_obj,  # 请求对象
        **kwargs,  # 其他关键字参数
    ):
        if not image_data:  # 如果没有图像数据
            input_ids = self._tokenizer(  # 仅对文本进行分词
                input_text, return_tensors="pt", add_special_tokens=False  # 不添加特殊令牌
            ).input_ids  # 获取输入ID
            return {  # 返回纯文本结果
                "input_ids": input_ids.squeeze(0).tolist(),  # 去除批次维度并转为列表
                "mm_items": [],  # 无多模态项
                "im_token_id": self.IMAGE_TOKEN_ID,  # 图像令牌ID
            }

        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和特殊令牌
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入ID列表
            mm_items=mm_items,  # 多模态数据项
            im_token_id=self.IMAGE_TOKEN_ID,  # 图像令牌ID
        )
