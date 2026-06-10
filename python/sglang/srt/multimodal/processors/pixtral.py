# Pixtral多模态处理器模块
# 本模块为Pixtral视觉模型提供图像数据处理功能
# 支持多图像输入的分割和处理

import math  # 导入数学模块
from typing import List, Union  # 导入类型提示

from transformers import PreTrainedTokenizerBase  # 导入预训练分词器基类
from transformers.models.pixtral.image_processing_pixtral import (  # 导入Pixtral图像处理工具
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput  # 导入模态枚举和处理器输出类
from sglang.srt.models.pixtral import (  # 导入Pixtral模型类
    PixtralForConditionalGeneration,
    PixtralVisionModel,
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊标记类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class PixtralProcessor(BaseMultimodalProcessor):  # Pixtral多模态处理器，继承基础多模态处理器
    models = [PixtralVisionModel, PixtralForConditionalGeneration]  # 关联的模型列表
    gpu_image_decode = False  # Pixtral processes loaded image as PIL image explicitly，Pixtral显式将图像作为PIL图像处理，不使用GPU图像解码

    PAD_TOKEN = "<pad>"  # 填充标记
    DEFAULT_IMAGE_TOKEN = "[IMG]"  # 默认图像标记

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Pixtral处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.IM_TOKEN_ID = getattr(  # 获取图像标记ID，默认使用PixtralVisionModel的默认值
            hf_config, "image_token_index", PixtralVisionModel.DEFAULT_IMAGE_TOKEN_ID
        )

        self.vision_config = hf_config.vision_config  # 获取视觉配置
        self.image_size = self.vision_config.image_size  # 获取图像尺寸
        self.patch_size = self.vision_config.patch_size  # 获取补丁尺寸

        # spatial_merge_size may live on vision_config (Mistral native) or
        # on the top-level config (HF native Mistral3Config).
        self._spatial_merge_size = getattr(  # 获取空间合并尺寸，可能在视觉配置或顶层配置中
            self.vision_config,  # 先在视觉配置中查找
            "spatial_merge_size",  # 属性名
            getattr(hf_config, "spatial_merge_size", 1),  # 否则在顶层配置中查找，默认为1
        )

        self._processor.patch_size = self.patch_size  # 设置处理器的补丁尺寸
        if self._spatial_merge_size > 1:  # 如果空间合并尺寸大于1
            self._processor.spatial_merge_size = self._spatial_merge_size  # 设置处理器的空间合并尺寸

        tokenizer = (  # 获取分词器
            _processor  # 如果处理器本身就是分词器
            if isinstance(_processor, PreTrainedTokenizerBase)  # 检查是否为分词器类型
            else _processor.tokenizer  # 否则从处理器中获取分词器
        )
        self.image_token = getattr(_processor, "image_token", self.DEFAULT_IMAGE_TOKEN)  # 获取图像标记，默认为[IMG]

        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=self.image_token,  # 图像标记
            image_token_id=self.IM_TOKEN_ID,  # 图像标记ID
        ).build(_processor)  # 构建标记对象
        tokenizer.add_special_tokens(  # 添加特殊标记到分词器
            {
                "pad_token": getattr(hf_config, "pad_token", self.PAD_TOKEN),  # 填充标记
            }
        )

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        mm_data = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 输入提示文本
            multimodal_tokens=self.mm_tokens,  # 多模态标记
            image_data=image_data,  # 图像数据
            return_text=True,  # 返回文本格式
        )
        if mm_data.images:  # 如果存在图像数据
            effective_patch = self.patch_size * self._spatial_merge_size  # 计算有效补丁尺寸
            image_nrows = []  # 存储每张图像的行数
            for img in mm_data.images:  # 遍历所有图像
                w, h = img.size  # 获取图像宽高
                ratio = max(w / self.image_size, h / self.image_size)  # 计算缩放比例
                if ratio > 1:  # 如果需要缩放
                    w = int(math.floor(w / ratio))  # 计算缩放后的宽度
                    h = int(math.floor(h / ratio))  # 计算缩放后的高度
                nrows, _ = _get_pixtral_hf_num_image_tokens(  # 计算图像标记的行数
                    (h, w), (effective_patch, effective_patch)  # 传入图像尺寸和有效补丁尺寸
                )
                image_nrows.append(nrows)  # 添加到行数列表

            mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
                mm_data, self.mm_tokens  # 多模态数据和标记
            )

            # For multi-image: split single IMAGE mm_item into per-image items
            if len(mm_data.images) > 1:  # 如果有多张图像
                from sglang.srt.managers.schedule_batch import MultimodalDataItem  # 导入多模态数据项类

                old_item = next(  # 找到图像类型的多模态数据项
                    item for item in mm_items if item.modality == Modality.IMAGE  # 查找图像模态项
                )
                all_offsets = old_item.offsets  # 获取所有偏移量
                old_feature = old_item.feature  # 获取特征数据
                old_image_sizes = getattr(old_item, "image_sizes", None)  # 获取图像尺寸信息

                mm_items = [  # 过滤掉原始图像项，保留其他模态项
                    item for item in mm_items if item.modality != Modality.IMAGE  # 保留非图像模态项
                ]
                offset_idx = 0  # 偏移量索引
                for i, img in enumerate(mm_data.images):  # 遍历每张图像
                    nr = image_nrows[i]  # 获取当前图像的行数
                    item_offsets = all_offsets[offset_idx : offset_idx + nr]  # 获取当前图像的偏移量
                    offset_idx += nr  # 更新偏移量索引
                    new_item = MultimodalDataItem(modality=Modality.IMAGE)  # 创建新的图像数据项
                    new_item.feature = old_feature[i : i + 1]  # 设置特征数据
                    new_item.offsets = item_offsets  # 设置偏移量
                    if old_image_sizes is not None:  # 如果存在图像尺寸信息
                        new_item.model_specific_data["image_sizes"] = old_image_sizes[  # 设置图像尺寸
                            i : i + 1  # 当前图像的尺寸
                        ]
                    mm_items.append(new_item)  # 添加新的图像数据项到列表
        else:  # 如果没有图像数据
            mm_items, input_ids, _ = self.process_and_combine_mm_data(  # 处理并合并多模态数据
                mm_data, self.mm_tokens  # 多模态数据和标记
            )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids.tolist(),  # 输入ID列表
            im_token_id=self.IM_TOKEN_ID,  # 图像标记ID
        )
