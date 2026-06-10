# DotsVLM 多模态图像处理器模块
# 本模块实现了 DotsVLM 和 DotsOCR 模型的图像数据处理逻辑，
# 包括图像特殊标记的定义、图像数据的加载与处理等功能。
import re  # 导入正则表达式模块
from typing import Dict, List, Union  # 导入类型提示模块

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput  # 导入多模态处理器输出类
from sglang.srt.models.dots_ocr import DotsOCRForCausalLM  # 导入 DotsOCR 模型类
from sglang.srt.models.dots_vlm import DotsVLMForCausalLM  # 导入 DotsVLM 模型类
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor,  # 基础多模态处理器类
    MultimodalSpecialTokens,  # 多模态特殊标记类
)


class DotsVLMImageProcessor(BaseMultimodalProcessor):  # DotsVLM 图像处理器类，继承自基础多模态处理器
    models = [DotsVLMForCausalLM, DotsOCRForCausalLM]  # 支持的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化 DotsVLM 图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化方法
        # The single, pre-expanded image token.
        self.IMAGE_TOKEN = "<|img|><|imgpad|><|endofimg|>"  # 单个预展开的图像标记
        # The regex that matches expanded image tokens.
        self.IMAGE_TOKEN_REGEX = re.compile(r"<\|img\|>(?:<\|imgpad\|>)+<\|endofimg\|>")  # 匹配已展开图像标记的正则表达式

        assert len(_processor.tokenizer.encode("<|img|>")) == 1  # 断言图像起始标记编码为单个 token
        self.im_start_id = _processor.tokenizer.encode("<|img|>")[0]  # 图像起始标记的 token ID
        self.im_end_id = _processor.tokenizer.encode("<|endofimg|>")[0]  # 图像结束标记的 token ID
        self.image_token_id = _processor.tokenizer.encode("<|imgpad|>")[0]  # 图像填充标记的 token ID
        self.IM_TOKEN_ID = self.image_token_id  # 图像标记 ID 别名
        self.IM_START_TOKEN_ID = self.im_start_id  # 图像起始标记 ID 别名
        self.IM_END_TOKEN_ID = self.im_end_id  # 图像结束标记 ID 别名

        vision_config = hf_config.vision_config  # 获取视觉编码器配置
        patch_size = vision_config.patch_size  # 获取 patch 大小
        merge_size = vision_config.spatial_merge_size  # 获取空间合并大小

        self.IMAGE_FACTOR = patch_size * merge_size  # 图像因子，由 patch 大小和合并大小计算
        self.MIN_PIXELS = getattr(  # 获取最小像素数
            _processor.image_processor,  # 从图像处理器获取
            "min_pixels",  # 尝试获取 min_pixels 属性
            getattr(_processor.image_processor, "size", {}).get("shortest_edge"),  # 否则从 size 中获取 shortest_edge
        )
        self.MAX_PIXELS = getattr(  # 获取最大像素数
            _processor.image_processor,  # 从图像处理器获取
            "max_pixels",  # 尝试获取 max_pixels 属性
            getattr(_processor.image_processor, "size", {}).get("longest_edge"),  # 否则从 size 中获取 longest_edge
        )
        self.MAX_RATIO = 200  # 最大宽高比限制
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=self.IMAGE_TOKEN,  # 图像标记
            image_token_id=self.image_token_id,  # 图像标记 ID
            image_token_regex=self.IMAGE_TOKEN_REGEX,  # 图像标记正则表达式
        ).build(_processor)  # 使用处理器构建标记

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes, Dict]],  # 图像数据列表，可以是字符串、字节或字典
        input_text,  # 输入文本
        request_obj,  # 请求对象
        max_req_input_len,  # 最大请求输入长度
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        if isinstance(image_data, str):  # 如果图像数据是单个字符串
            image_data = [image_data]  # 将其包装为列表

        if (  # 如果图像数据是嵌套列表
            isinstance(image_data, list)  # 检查是否为列表
            and image_data  # 检查列表是否非空
            and isinstance(image_data[0], list)  # 检查第一个元素是否也是列表
        ):
            image_data = sum(image_data, [])  # 将嵌套列表展平为一维列表

        base_output = await self.load_mm_data(  # 异步加载多模态数据
            prompt=input_text,  # 输入提示文本
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊标记
        )

        combined_mm_item, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 传入基础输出和标记
        )
        if combined_mm_item is None:  # 如果合并结果为空
            return None  # 返回 None

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=combined_mm_item,  # 合并后的多模态项
            input_ids=input_ids.tolist(),  # 输入 ID 列表
            im_token_id=self.image_token_id,  # 图像填充标记 ID
            im_start_id=self.im_start_id,  # 图像起始标记 ID
            im_end_id=self.im_end_id,  # 图像结束标记 ID
        )
