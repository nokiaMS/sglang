# LightOnOCR多模态处理器模块
# 实现lightonai/LightOnOCR-2-1B模型的多模态数据处理
# 与Pixtral处理器的关键区别：LightOnOCR不使用图像分隔/结束令牌
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

"""
Multimodal processor for lightonai/LightOnOCR-2-1B.
# lightonai/LightOnOCR-2-1B的多模态处理器

Key difference from Pixtral: LightOnOCR does NOT use image break/end tokens.
# 与Pixtral的关键区别：LightOnOCR不使用图像分隔/结束令牌
The parent PixtralProcessor inserts row-break and image-end tokens between
image patch rows. This processor removes them after the parent processing
to produce a single contiguous range of image tokens per image.
# 父类PixtralProcessor在图像块行之间插入行分隔和图像结束令牌。此处理器在父类处理后将它们移除，
# 以便每张图像生成一个连续的图像令牌范围
"""

from typing import List, Union  # 导入类型提示

from sglang.srt.models.lightonocr import LightOnOCRForConditionalGeneration  # 导入LightOnOCR模型类
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor  # 导入Pixtral处理器基类


class LightOnOCRProcessor(PixtralProcessor):  # LightOnOCR处理器类，继承自Pixtral处理器
    """Processor for LightOnOCR model."""  # LightOnOCR模型的处理器

    models = [LightOnOCRForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化LightOnOCR处理器
        # LightOnOCR uses image_token_id instead of image_token_index
        # LightOnOCR使用image_token_id而非image_token_index
        if not hasattr(hf_config, "image_token_index"):  # 如果配置中没有image_token_index属性
            hf_config.image_token_index = getattr(hf_config, "image_token_id", 151655)  # 从image_token_id获取，默认151655

        # Propagate spatial_merge_size from root config to vision_config
        # 将spatial_merge_size从根配置传播到vision_config
        spatial_merge_size = getattr(hf_config, "spatial_merge_size", 2)  # 获取空间合并大小，默认2
        if hasattr(hf_config, "vision_config"):  # 如果配置中有vision_config
            vc = hf_config.vision_config  # 获取视觉配置
            if not hasattr(vc, "spatial_merge_size") or vc.spatial_merge_size is None:  # 如果视觉配置中没有或为None
                vc.spatial_merge_size = spatial_merge_size  # 设置空间合并大小

        if hasattr(_processor, "patch_size"):  # 如果处理器有patch_size属性
            _processor.spatial_merge_size = spatial_merge_size  # 设置处理器的空间合并大小

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

        # Identify break/end token IDs for removal
        # 识别需要移除的分隔/结束令牌ID
        self._break_token_ids = set()  # 初始化分隔令牌ID集合
        for attr in ("image_break_token_id", "image_break_id"):  # 遍历图像分隔令牌属性名
            tid = getattr(_processor, attr, None)  # 获取令牌ID
            if tid is not None:  # 如果令牌ID存在
                self._break_token_ids.add(tid)  # 添加到集合
        for attr in ("image_end_token_id", "image_end_id"):  # 遍历图像结束令牌属性名
            tid = getattr(_processor, attr, None)  # 获取令牌ID
            if tid is not None:  # 如果令牌ID存在
                self._break_token_ids.add(tid)  # 添加到集合

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        result = await super().process_mm_data_async(  # 调用父类的异步处理方法
            image_data=image_data,  # 图像数据
            input_text=input_text,  # 输入文本
            request_obj=request_obj,  # 请求对象
            *args,  # 位置参数
            **kwargs,  # 关键字参数
        )

        if not result or not self._break_token_ids:  # 如果结果为空或没有分隔令牌
            return result  # 直接返回结果

        # Remove break/end tokens and fix multimodal item offsets
        # 移除分隔/结束令牌并修正多模态项偏移
        input_ids = result.input_ids or []  # 获取输入ID列表
        mm_items = result.mm_items or []  # 获取多模态项列表

        new_input_ids = []  # 新的输入ID列表
        old_to_new = {}  # 旧索引到新索引的映射
        for old_idx, token_id in enumerate(input_ids):  # 遍历所有令牌
            if token_id not in self._break_token_ids:  # 如果不是分隔令牌
                old_to_new[old_idx] = len(new_input_ids)  # 记录旧到新的索引映射
                new_input_ids.append(token_id)  # 保留该令牌

        if len(new_input_ids) == len(input_ids):  # 如果没有移除任何令牌
            return result  # 直接返回结果

        # Remap multimodal item offsets to account for removed tokens
        # 重新映射多模态项偏移以适应被移除的令牌
        for mm_item in mm_items:  # 遍历每个多模态项
            if not mm_item.offsets:  # 如果没有偏移信息
                continue  # 跳过
            new_indices = sorted(  # 计算新索引并排序
                old_to_new[idx]  # 旧索引映射到新索引
                for start, end in mm_item.offsets  # 遍历每个偏移范围
                for idx in range(start, end + 1)  # 遍历范围内的每个索引
                if idx in old_to_new  # 仅保留存在的索引
            )
            if new_indices:  # 如果有新索引
                mm_item.offsets = [(new_indices[0], new_indices[-1])]  # 更新偏移为新范围

        result.input_ids = new_input_ids  # 更新结果的输入ID
        return result  # 返回处理后的结果
