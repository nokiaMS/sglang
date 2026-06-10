# MiniCPM多模态处理器模块
# 实现MiniCPM-V和MiniCPM-O系列模型的多模态数据处理
# 兼容MiniCPM-O和MiniCPM-V两个版本，支持图像和音频输入
# 支持processor_output和precomputed_embedding特殊格式
from typing import List, Union  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.minicpmo import MiniCPMO  # 导入MiniCPM-O模型
from sglang.srt.models.minicpmv import MiniCPMV  # 导入MiniCPM-V模型
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器相关类
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)


# Compatible with both 'O' and 'V'
# 兼容MiniCPM-O和MiniCPM-V两个版本
class MiniCPMMultimodalProcessor(BaseMultimodalProcessor):  # MiniCPM多模态处理器类
    models = [MiniCPMV, MiniCPMO]  # 关联的模型列表
    support_dynamic_frame_expansion = True  # 支持动态帧扩展
    gpu_image_decode = False  # MiniCPM HF processor does not support tensor inputs
    # MiniCPM的HF处理器不支持张量输入，禁用GPU图像解码

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化MiniCPM多模态处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        # Collect special token ids
        # 收集特殊令牌ID
        tokenizer = self._processor.tokenizer  # 获取分词器
        self.slice_start_id = getattr(tokenizer, "slice_start_id", None)  # 切片开始令牌ID
        self.slice_end_id = getattr(tokenizer, "slice_end_id", None)  # 切片结束令牌ID
        self.audio_start_id = getattr(tokenizer, "audio_start_id", None)  # 音频开始令牌ID
        self.audio_end_id = getattr(tokenizer, "audio_end_id", None)  # 音频结束令牌ID
        self.im_start_id = getattr(tokenizer, "im_start_id", None)  # 图像开始令牌ID
        self.im_end_id = getattr(tokenizer, "im_end_id", None)  # 图像结束令牌ID
        self.im_token_id = getattr(tokenizer, "unk_id", None)  # 图像令牌ID
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊令牌
            image_token="(<image>./</image>)",  # 图像令牌格式
            audio_token="(<audio>./</audio>)",  # 音频令牌格式
            video_token="(<video>./</video>)",  # 视频令牌格式
            image_token_id=self.im_token_id,  # 图像令牌ID
        ).build(_processor)  # 构建令牌映射

    @staticmethod
    def _has_special_format(image_data, audio_data):  # 检查是否有特殊格式输入
        """Check if any input items use processor_output or precomputed_embedding format."""  # 检查是否有输入项使用processor_output或precomputed_embedding格式
        for data in list(image_data or []) + list(audio_data or []):  # 遍历所有图像和音频数据
            if isinstance(data, dict) and data.get("format") in (  # 如果是字典且格式为
                "processor_output",  # 处理器输出格式
                "precomputed_embedding",  # 预计算嵌入格式
            ):
                return True  # 返回True
        return False  # 没有特殊格式则返回False

    async def _process_special_format(  # 处理特殊格式输入
        self, image_data, audio_data, input_text, request_obj, **kwargs
    ):
        """Handle processor_output and precomputed_embedding input formats.
        # 处理processor_output和precomputed_embedding输入格式

        Delegates to the base class process_and_combine_mm_data which has
        built-in support for these formats.
        # 委托给基类的process_and_combine_mm_data，它内置支持这些格式
        """
        if isinstance(input_text, list):  # 如果输入文本是列表（预分词）
            user_input_ids = input_text  # 使用用户提供的input_ids
            prompt = ""  # 提示为空
        else:
            user_input_ids = None  # 不使用预分词
            prompt = input_text or ""  # 使用输入文本

        # Normalize dicts: the HF MiniCPM processor returns "tgt_sizes" (plural)
        # but the base class ATTR_NAME_TO_MODALITY maps "tgt_size" (singular).
        # Also flatten the nested batch dimension so the structure matches
        # what the NORMAL path produces (flat list of per-patch tensors).
        # 规范化字典：HF MiniCPM处理器返回"tgt_sizes"（复数），
        # 但基类ATTR_NAME_TO_MODALITY映射"tgt_size"（单数）。
        # 同时展平嵌套的批次维度，使结构与NORMAL路径产生的结果一致（扁平的逐补丁张量列表）。
        normalized_images = []  # 规范化后的图像列表
        for d in image_data or []:  # 遍历图像数据
            if isinstance(d, dict):  # 如果是字典
                d = dict(d)  # 复制字典以避免修改原数据
                if "tgt_sizes" in d and "tgt_size" not in d:  # 如果有tgt_sizes但没有tgt_size
                    d["tgt_size"] = d.pop("tgt_sizes")  # 重命名tgt_sizes为tgt_size
                if d.get("format") == "processor_output":  # 如果是处理器输出格式
                    pixel_values = d.get("pixel_values")  # 获取像素值
                    tgt_size = d.get("tgt_size")  # 获取目标尺寸
                    if pixel_values is not None and tgt_size is not None:  # 如果两者都存在
                        pv_flat, ts_flat = [], []  # 展平后的像素值和目标尺寸
                        for pixel_b, tgt_b in zip(pixel_values, tgt_size):  # 遍历批次
                            if isinstance(pixel_b, (list, tuple)):  # 如果批次元素是列表或元组
                                for pixel_n, tgt_n in zip(pixel_b, tgt_b):  # 遍历嵌套项
                                    pv_flat.append(pixel_n)  # 添加像素值
                                    ts_flat.append(tgt_n)  # 添加目标尺寸
                            else:
                                pv_flat.append(pixel_b)  # 直接添加像素值
                                ts_flat.append(tgt_b)  # 直接添加目标尺寸
                        d["pixel_values"] = pv_flat  # 更新展平后的像素值
                        d["tgt_size"] = ts_flat  # 更新展平后的目标尺寸
                normalized_images.append(d)  # 添加规范化后的图像
            else:
                normalized_images.append(d)  # 非字典直接添加

        normalized_audios = list(audio_data or [])  # 规范化音频数据

        if not prompt and (normalized_images or normalized_audios):  # 如果没有提示但有媒体数据
            images = [d for d in normalized_images if isinstance(d, dict)]  # 筛选字典格式的图像
            audios = [d for d in normalized_audios if isinstance(d, dict)]  # 筛选字典格式的音频

            raw_img_dropped = len(normalized_images) - len(images)  # 被丢弃的原始图像数
            raw_aud_dropped = len(normalized_audios) - len(audios)  # 被丢弃的原始音频数
            if raw_img_dropped > 0 or raw_aud_dropped > 0:  # 如果有丢弃的数据
                raise ValueError(  # 抛出异常
                    f"[minicpm] Cannot process raw media with pre-tokenized "
                    f"input_ids. Provide multimodal data in 'processor_output' or "
                    f"'precomputed_embedding' format, or use a text prompt instead. "
                    f"(raw images dropped: {raw_img_dropped}, "
                    f"raw audios dropped: {raw_aud_dropped})"
                )

            base_output = BaseMultiModalProcessorOutput(  # 创建基础输出
                input_text=prompt,  # 输入文本
                images=images,  # 图像列表
                audios=audios,  # 音频列表
            )
        else:
            base_output = await self.load_mm_data(  # 加载多模态数据
                prompt=prompt,  # 提示文本
                image_data=normalized_images,  # 图像数据
                audio_data=audio_data,  # 音频数据
                multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
            )

        if base_output is None:  # 如果基础输出为空
            return None  # 返回None

        mm_items, input_ids_tensor, ret = self.process_and_combine_mm_data(  # 处理并合并多模态数据
            base_output, self.mm_tokens  # 基础输出和特殊令牌
        )

        if user_input_ids is not None:  # 如果使用了预分词的input_ids
            input_ids_tensor = torch.tensor(user_input_ids, dtype=torch.long)  # 转为张量
            for mm_item in mm_items:  # 遍历多模态项
                if mm_item.modality == Modality.IMAGE:  # 如果是图像模态
                    image_offsets = self.get_mm_items_offset_by_pair(  # 获取图像偏移
                        input_ids=input_ids_tensor,  # 输入ID
                        mm_start_id=self.im_start_id,  # 图像开始令牌ID
                        mm_end_id=self.im_end_id,  # 图像结束令牌ID
                    )
                    slice_offsets = self.get_mm_items_offset_by_pair(  # 获取切片偏移
                        input_ids=input_ids_tensor,  # 输入ID
                        mm_start_id=self.slice_start_id,  # 切片开始令牌ID
                        mm_end_id=self.slice_end_id,  # 切片结束令牌ID
                    )
                    image_offsets.extend(slice_offsets)  # 合并偏移
                    mm_item.offsets = sorted(image_offsets)  # 排序偏移
                elif mm_item.modality == Modality.AUDIO:  # 如果是音频模态
                    if (  # 如果有音频开始和结束令牌
                        self.audio_start_id is not None
                        and self.audio_end_id is not None
                    ):
                        mm_item.offsets = self.get_mm_items_offset_by_pair(  # 获取音频偏移
                            input_ids=input_ids_tensor,  # 输入ID
                            mm_start_id=self.audio_start_id,  # 音频开始令牌ID
                            mm_end_id=self.audio_end_id,  # 音频结束令牌ID
                        )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
            input_ids=input_ids_tensor.flatten().tolist(),  # 输入ID列表
            audio_start_id=self.audio_start_id,  # 音频开始令牌ID
            audio_end_id=self.audio_end_id,  # 音频结束令牌ID
            im_token_id=self.im_token_id,  # 图像令牌ID
            im_start_id=self.im_start_id,  # 图像开始令牌ID
            im_end_id=self.im_end_id,  # 图像结束令牌ID
            slice_start_id=self.slice_start_id,  # 切片开始令牌ID
            slice_end_id=self.slice_end_id,  # 切片结束令牌ID
        )

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes]],  # 图像数据列表
        audio_data: List[Union[str, bytes]],  # 音频数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        **kwargs,  # 关键字参数
    ):
        if isinstance(input_text, list) or self._has_special_format(  # 如果输入是列表或有特殊格式
            image_data, audio_data
        ):
            return await self._process_special_format(  # 使用特殊格式处理
                image_data=image_data,  # 图像数据
                audio_data=audio_data,  # 音频数据
                input_text=input_text,  # 输入文本
                request_obj=request_obj,  # 请求对象
                **kwargs,  # 关键字参数
            )

        base_output = await self.load_mm_data(  # 加载多模态数据
            prompt=input_text,  # 提示文本
            audio_data=audio_data,  # 音频数据
            image_data=image_data,  # 图像数据
            multimodal_tokens=self.mm_tokens,  # 多模态特殊令牌
        )
        if base_output is None:  # 如果基础输出为空
            return None  # 返回None

        res = self.process_mm_data(  # 同步处理多模态数据
            input_text=base_output.input_text,  # 输入文本
            images=base_output.images,  # 图像列表
            audios=base_output.audios,  # 音频列表
        )

        pixel_values = res["pixel_values"]  # 获取像素值
        tgt_sizes = res["tgt_sizes"]  # 获取目标尺寸

        if not isinstance(pixel_values, (torch.Tensor, list)):  # 检查像素值类型
            raise ValueError(
                "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
            )

        if not isinstance(tgt_sizes, (torch.Tensor, list)):  # 检查目标尺寸类型
            raise ValueError(
                "Incorrect type of target sizes. " f"Got type: {type(tgt_sizes)}"
            )

        if len(pixel_values) != len(tgt_sizes):  # 检查像素值和目标尺寸长度一致性
            raise ValueError(
                "Inconsistent batch lengths, found: "
                f"{len(pixel_values)} vs. {len(tgt_sizes)}"
            )

        # Track slices per image (like vLLM's num_slices)
        # 跟踪每张图像的切片数（类似vLLM的num_slices）
        slices_per_image: List[int] = []  # 每张图像的切片数列表
        pixel_values_flat: List[torch.Tensor] = []  # 展平后的像素值列表
        tgt_sizes_flat: List[torch.Tensor] = []  # 展平后的目标尺寸列表
        for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):  # 遍历每张图像
            # per image
            # 每张图像
            if len(pixel_b) != len(tgt_b):  # 检查切片数量一致性
                raise ValueError(
                    "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                )
            slices_per_image.append(len(pixel_b))  # 记录切片数
            for pixel_n, tgt_n in zip(pixel_b, tgt_b):  # 遍历每个切片
                pixel_values_flat += [pixel_n]  # 添加像素值
                tgt_sizes_flat += [tgt_n]  # 添加目标尺寸

        pixel_values = pixel_values_flat  # 使用展平后的像素值

        items = []  # 多模态数据项列表
        input_ids = res["input_ids"].flatten()  # 展平输入ID
        image_offsets = self.get_mm_items_offset_by_pair(  # 获取图像偏移
            input_ids=input_ids, mm_start_id=self.im_start_id, mm_end_id=self.im_end_id
        )
        slice_offsets = self.get_mm_items_offset_by_pair(  # 获取切片偏移
            input_ids=input_ids,
            mm_start_id=self.slice_start_id,
            mm_end_id=self.slice_end_id,
        )
        image_offsets.extend(slice_offsets)  # 合并图像和切片偏移
        image_offsets = sorted(image_offsets)  # 排序偏移

        # Create one item per image, each with its own slices and offsets
        # 为每张图像创建一个数据项，每个有自己的切片和偏移
        if len(pixel_values) != 0:  # 如果有像素值
            pv_idx = 0  # 像素值索引
            offset_idx = 0  # 偏移索引
            for num_slices in slices_per_image:  # 遍历每张图像的切片数
                items.append(  # 添加多模态数据项
                    MultimodalDataItem(
                        feature=pixel_values[pv_idx : pv_idx + num_slices],  # 该图像的所有切片像素值
                        offsets=image_offsets[offset_idx : offset_idx + num_slices],  # 对应偏移
                        model_specific_data={
                            "tgt_size": tgt_sizes_flat[pv_idx : pv_idx + num_slices]  # 目标尺寸
                        },
                        modality=Modality.IMAGE,  # 图像模态
                    )
                )
                pv_idx += num_slices  # 更新像素值索引
                offset_idx += num_slices  # 更新偏移索引

        if (  # 如果有音频特征
            "audio_features" in res
            and res["audio_features"] is not None
            and len(res["audio_features"]) != 0
        ):
            if self.audio_start_id is not None and self.audio_end_id is not None:  # 如果有音频令牌ID
                audio_offsets = self.get_mm_items_offset_by_pair(  # 获取音频偏移
                    input_ids=input_ids,
                    mm_start_id=self.audio_start_id,
                    mm_end_id=self.audio_end_id,
                )
            else:
                audio_offsets = None  # 无音频偏移
            item = MultimodalDataItem(  # 创建音频数据项
                feature=[res["audio_features"]],  # 音频特征
                model_specific_data={"audio_feature_lens": res["audio_feature_lens"]},  # 音频特征长度
                offsets=audio_offsets,  # 偏移
                modality=Modality.AUDIO,  # 音频模态
            )
            items += [item]  # 添加音频数据项
        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=items,  # 多模态数据项
            input_ids=input_ids.tolist(),  # 输入ID列表
            audio_start_id=self.audio_start_id,  # 音频开始令牌ID
            audio_end_id=self.audio_end_id,  # 音频结束令牌ID
            im_token_id=self.im_token_id,  # 图像令牌ID
            im_start_id=self.im_start_id,  # 图像开始令牌ID
            im_end_id=self.im_end_id,  # 图像结束令牌ID
            slice_start_id=self.slice_start_id,  # 切片开始令牌ID
            slice_end_id=self.slice_end_id,  # 切片结束令牌ID
        )
