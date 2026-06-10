# 本文件实现 DeepSeek VL2 模型的图像处理器，负责加载图像数据、
# 处理多模态输入并返回带有图像 token ID 的处理器输出
# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import List, Union  # 导入类型提示工具

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.deepseek_vl2 import DeepseekVL2ForCausalLM  # 导入 DeepSeek VL2 模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基类和特殊 token 类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class DeepseekVL2ImageProcessor(BaseMultimodalProcessor):  # DeepSeek VL2 图像处理器，继承自 BaseMultimodalProcessor
    models = [DeepseekVL2ForCausalLM]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.mm_tokens = MultimodalSpecialTokens(  # 构建特殊 token 配置
            image_token="<image>", image_token_id=self._processor.image_token_id
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        max_req_input_len,  # 最大请求输入长度
        *args,
        **kwargs,
    ):  # 异步处理多模态数据
        """异步处理图像数据并返回带有图像 token ID 的处理器输出"""
        base_output = await self.load_mm_data(  # 加载多模态数据
            input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并组合多模态数据
            base_output,
            self.mm_tokens,
            max_req_input_len=max_req_input_len,
            conversations=base_output.input_text,
        )

        return MultimodalProcessorOutput(  # 返回处理器输出
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self._processor.image_token_id,  # 包含图像 token ID
        )
