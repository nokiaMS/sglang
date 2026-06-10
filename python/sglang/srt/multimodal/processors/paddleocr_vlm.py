# PaddleOCR-VL多模态处理器模块
# 实现PaddleOCR视觉语言模型的多模态数据处理
# 继承自QwenVL图像处理器，自定义图像令牌格式
# Reference: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sglang.srt.models.paddleocr_vl import PaddleOCRVLForConditionalGeneration  # 导入PaddleOCR-VL模型类
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens  # 导入多模态特殊令牌类
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor  # 导入QwenVL图像处理器基类


class PaddleOCRVLImageProcessor(QwenVLImageProcessor):  # PaddleOCR-VL图像处理器类，继承自QwenVL图像处理器
    models = [PaddleOCRVLForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化PaddleOCR-VL图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            image_token="<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>",  # PaddleOCR-VL的图像令牌格式
            image_token_id=hf_config.image_token_id,  # 图像令牌ID
            video_token_id=hf_config.video_token_id,  # 视频令牌ID
        ).build(_processor)  # 构建令牌映射
