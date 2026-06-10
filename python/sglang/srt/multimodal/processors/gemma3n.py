# Gemma3n 多模态处理器模块
# 本模块实现了 Gemma3n 模型的多模态数据处理逻辑，
# 支持图像和音频输入，包括特殊标记的定义和数据异步处理。
# Copyright 2025 SGLang Team
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
# ==============================================================================

from typing import Dict, List, Optional, Union  # 导入类型提示模块

from sglang.srt.managers.multimodal_processor import (  # 导入多模态处理器模块
    BaseMultimodalProcessor as SGLangBaseProcessor,  # 将基础处理器重命名为 SGLangBaseProcessor
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.gemma3n_mm import Gemma3nForConditionalGeneration  # 导入 Gemma3n 条件生成模型类
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens  # 导入多模态特殊标记类


class Gemma3nSGLangProcessor(SGLangBaseProcessor):  # Gemma3n SGLang 处理器类
    """Multimodal processor for Gemma3n supporting image and audio inputs."""
    """Gemma3n 的多模态处理器，支持图像和音频输入。"""

    models = [Gemma3nForConditionalGeneration]  # 支持的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 Gemma3n 处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法

        self.IM_START_TOKEN_ID = hf_config.boi_token_id  # 图像起始标记（beginning of image）的 token ID
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id  # 图像结束标记（end of image）的 token ID

        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id  # 音频起始标记（beginning of audio）的 token ID
        self.AUDIO_END_TOKEN_ID = hf_config.eoa_token_id  # 音频结束标记（end of audio）的 token ID
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token="<image_soft_token>",  # 图像软标记
            image_token_id=hf_config.image_token_id,  # 图像标记 token ID
            audio_token="<audio_soft_token>",  # 音频软标记
            audio_token_id=hf_config.audio_token_id,  # 音频标记 token ID
        ).build(_processor)  # 使用处理器构建标记

    async def process_mm_data_async(  # 异步处理多模态数据，包括图像和音频
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,  # 可选的图像数据列表
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,  # 可选的音频数据列表
        input_text: str = "",  # 输入文本，默认为空字符串
        request_obj=None,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        """Process multimodal data including images and audio."""
        """处理包括图像和音频在内的多模态数据。"""
        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            audio_data=audio_data,  # 音频数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            mm_items=mm_items,  # 多模态项
            im_token_id=self.mm_tokens.image_token_id,  # 图像标记 token ID
            audio_token_id=self.mm_tokens.audio_token_id,  # 音频标记 token ID
        )
