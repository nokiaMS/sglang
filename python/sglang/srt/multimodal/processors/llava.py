# LLaVA多模态处理器模块
# 实现LLaVA系列视觉语言模型（LLaVA-Llama、LLaVA-Qwen、LLaVA-Mistral、LLaVA-Vid、Mistral3）的多模态数据处理
# 支持多种图像宽高比模式：pad、anyres等
import asyncio  # 导入异步IO模块
import os  # 导入操作系统模块
from typing import Dict, List, Optional, Union  # 导入类型提示

import numpy as np  # 导入NumPy库
from transformers.models.auto.processing_auto import (  # 导入HF处理器映射名称
    PROCESSOR_MAPPING_NAMES as HF_MAPPING_NAMES,
)

import sglang.srt.managers.multimodal_processor as sgl_mm_processor_util  # 导入SGLang多模态处理器工具
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.llava import (  # 导入LLaVA模型类
    LlavaForConditionalGeneration,
    LlavaLlamaForCausalLM,
    LlavaMistralForCausalLM,
    LlavaQwenForCausalLM,
)
from sglang.srt.models.llavavid import LlavaVidForCausalLM  # 导入LLaVA-Vid模型
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration  # 导入Mistral3模型
from sglang.srt.multimodal.mm_utils import (  # 导入多模态工具函数
    ensure_numpy,
    expand2square,
    process_anyres_image,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor  # 导入基础多模态处理器
from sglang.srt.utils import ImageData, load_image, logger  # 导入图像数据类、图像加载函数和日志器
from sglang.utils import get_exception_traceback  # 导入异常回溯获取函数


class LlavaImageProcessor(BaseMultimodalProcessor):  # LLaVA图像处理器类，继承自基础多模态处理器
    models = [  # 关联的模型列表
        LlavaLlamaForCausalLM,  # LLaVA-Llama模型
        LlavaVidForCausalLM,  # LLaVA-Vid模型
        LlavaQwenForCausalLM,  # LLaVA-Qwen模型
        LlavaMistralForCausalLM,  # LLaVA-Mistral模型
    ]
    gpu_image_decode = False  # Llava processes loaded image as PIL image explicitly
    # LLaVA显式地将加载的图像作为PIL图像处理，禁用GPU图像解码

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化LLaVA图像处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化

    @staticmethod
    def _process_single_image_task(  # 处理单张图像的静态任务方法
        image_data: Union[str, bytes, ImageData],  # 图像数据
        image_aspect_ratio: Optional[str] = None,  # 图像宽高比模式
        image_grid_pinpoints: Optional[str] = None,  # 图像网格锚点
        processor=None,  # 处理器
    ):

        image_processor = processor.image_processor  # 获取图像预处理器

        try:
            url = image_data.url if isinstance(image_data, ImageData) else image_data  # 获取URL或原始数据
            image, image_size = load_image(url, False)  # 加载图像，不使用GPU
            if image_size is not None:  # 如果图像大小不为None
                # It is a video with multiple images
                # 这是一个包含多帧图像的视频
                image_hash = hash(url)  # 计算图像哈希值
                pixel_values = image_processor(image)["pixel_values"]  # 预处理图像获取像素值
                for i in range(len(pixel_values)):  # 遍历每帧像素值
                    pixel_values[i] = ensure_numpy(pixel_values[i]).astype(np.float16)  # 转为numpy数组并使用float16
                pixel_values = np.stack(pixel_values, axis=0)  # 堆叠所有帧
                return pixel_values, image_hash, image_size  # 返回像素值、哈希和图像大小
            else:
                # It is an image
                # 这是一张单张图像
                image_hash = hash(url)  # 计算图像哈希值
                if image_aspect_ratio == "pad":  # 如果宽高比模式为pad
                    image = expand2square(  # 将图像扩展为正方形
                        image,  # 原始图像
                        tuple(int(x * 255) for x in image_processor.image_mean),  # 使用均值颜色填充
                    )
                    pixel_values = image_processor(image.convert("RGB"))[  # 预处理RGB图像
                        "pixel_values"  # 获取像素值
                    ][0]  # 取第一个结果
                elif image_aspect_ratio == "anyres" or (  # 如果是anyres模式
                    image_aspect_ratio is not None  # 或宽高比不为None
                    and "anyres_max" in image_aspect_ratio  # 且包含anyres_max
                ):
                    pixel_values = process_anyres_image(  # 使用anyres方式处理图像
                        image, image_processor, image_grid_pinpoints  # 图像、预处理器和网格锚点
                    )
                else:
                    pixel_values = image_processor(image)["pixel_values"][0]  # 默认方式获取像素值

                pixel_values = ensure_numpy(pixel_values)  # 确保为numpy数组
                if isinstance(pixel_values, np.ndarray):  # 如果是numpy数组
                    pixel_values = pixel_values.astype(np.float16)  # 转为float16类型

                return pixel_values, image_hash, image.size  # 返回像素值、哈希和图像尺寸
        except Exception:  # 捕获异常
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())  # 记录错误日志

    async def _process_single_image(  # 异步处理单张图像
        self,
        image_data: Union[bytes, str, ImageData],  # 图像数据
        aspect_ratio: str,  # 宽高比模式
        grid_pinpoints: str,  # 网格锚点
    ):
        if self.cpu_executor is not None:  # 如果有CPU执行器
            loop = asyncio.get_running_loop()  # 获取当前事件循环
            fut = loop.run_in_executor(  # 在CPU执行器中运行任务
                self.cpu_executor,  # CPU执行器
                LlavaImageProcessor._process_single_image_task,  # 图像处理任务
                image_data,  # 图像数据
                aspect_ratio,  # 宽高比
                grid_pinpoints,  # 网格锚点
                self._processor,  # 处理器
            )
            timeout = int(os.environ.get("REQUEST_TIMEOUT", "10"))  # 获取超时时间，默认10秒
            return await asyncio.wait_for(fut, timeout=timeout)  # 等待任务完成或超时
        else:
            return self._process_single_image_task(  # 同步处理图像
                image_data,  # 图像数据
                aspect_ratio,  # 宽高比
                grid_pinpoints,  # 网格锚点
                self._processor.image_processor,  # 图像预处理器
            )

    def _process_precomputed_image_data(self, image_data: List[Dict]) -> Dict:  # 处理预计算的图像数据
        mm_items = []  # 多模态数据项列表
        for item in image_data:  # 遍历每个图像数据项
            # Infer size logic...
            # 推断尺寸逻辑...
            if "image_sizes" not in item:  # 如果没有image_sizes字段
                if "pixel_values" in item:  # 如果有pixel_values字段
                    pv = item["pixel_values"]  # 获取像素值
                    # Handle simplified if/else
                    # 处理简化的if/else逻辑
                    h, w = (  # 计算高度和宽度
                        (pv.shape[2], pv.shape[3])  # 4维张量的高宽
                        if len(pv.shape) == 4  # 如果是4维
                        else (pv.shape[1], pv.shape[2])  # 否则3维的高宽
                    )
                    item["image_sizes"] = [(w, h)]  # 设置图像尺寸
                else:
                    item["image_sizes"] = [(336, 336)]  # 默认尺寸336x336

            mm_items.append(  # 添加多模态数据项
                MultimodalDataItem(
                    feature=item["feature"],  # 特征数据
                    modality=Modality.IMAGE,  # 图像模态
                    model_specific_data=item,  # 模型特定数据
                )
            )
        return MultimodalProcessorOutput(mm_items=mm_items)  # 返回多模态处理器输出

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data: List[Union[str, bytes, ImageData]],  # 图像数据列表
        input_text,  # 输入文本
        request_obj,  # 请求对象
        *args,  # 位置参数
        **kwargs,  # 关键字参数
    ):
        # FIX: Handle precomputed embeddings (dictionaries)
        # If the input is already a dictionary, we skip the CPU image processor.
        # We also need to infer 'image_sizes' from 'pixel_values' if missing,
        # because pad_input_ids requires it.
        # 修复：处理预计算的嵌入（字典）
        # 如果输入已经是字典，则跳过CPU图像预处理器。
        # 如果缺少'image_sizes'，还需要从'pixel_values'推断，因为pad_input_ids需要它。
        if (
            isinstance(image_data, list)  # 如果图像数据是列表
            and len(image_data) > 0  # 且非空
            and isinstance(image_data[0], dict)  # 且第一个元素是字典
        ):
            return self._process_precomputed_image_data(image_data)  # 处理预计算数据

        modalities = request_obj.modalities or ["image"]  # 获取模态列表，默认为图像
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)  # 获取图像宽高比配置
        grid_pinpoints = (  # 获取网格锚点配置
            self.hf_config.image_grid_pinpoints  # 从配置获取
            if hasattr(self.hf_config, "image_grid_pinpoints")  # 如果配置中存在
            and "anyres" in aspect_ratio  # 且宽高比模式为anyres
            else None  # 否则为None
        )

        if isinstance(image_data, list) and len(image_data) > 0:  # 如果有图像数据
            if "multi-images" in modalities or "video" in modalities:  # 如果是多图像或视频模式
                # Multiple images
                # 多图像模式
                aspect_ratio = "pad"  # LLaVA OneVision Handling: more than one image --> interleaved image mode or video mode. We do not use anyres
                # LLaVA OneVision处理：多于一张图像时使用交错图像模式或视频模式，不使用anyres
                pixel_values, data_hashes, image_sizes = [], [], []  # 初始化结果列表
                res = []  # 异步任务结果列表
                for img_data in image_data:  # 遍历每张图像数据
                    res.append(  # 添加异步处理任务
                        self._process_single_image(
                            img_data, aspect_ratio, grid_pinpoints  # 图像数据、宽高比和网格锚点
                        )
                    )

                res = await asyncio.gather(*res)  # 并行等待所有任务完成
                for pixel_v, image_h, image_s in res:  # 遍历结果
                    pixel_values.append(pixel_v)  # 收集像素值
                    data_hashes.append(image_h)  # 收集哈希
                    image_sizes.append(image_s)  # 收集图像尺寸
            else:
                # A single image
                # 单张图像模式
                pixel_values, image_hash, image_size = await self._process_single_image(  # 处理单张图像
                    image_data[0], aspect_ratio, grid_pinpoints  # 第一张图像数据
                )
                pixel_values = [pixel_values]  # 包装为列表
                image_sizes = [image_size]  # 包装为列表
        else:
            raise ValueError(f"Invalid image data: {image_data}")  # 无效图像数据则抛出异常
        modality = Modality.IMAGE  # 默认模态为图像
        if isinstance(request_obj.modalities, list):  # 如果模态是列表
            if request_obj.modalities[0] == "video":  # 如果第一个模态是视频
                modality = Modality.VIDEO  # 设置模态为视频

        # Create one item per image for better cache granularity
        # 为每张图像创建一个数据项以获得更好的缓存粒度
        mm_items = []  # 多模态数据项列表
        for pixel_v, image_s in zip(pixel_values, image_sizes):  # 遍历像素值和图像尺寸
            # Ensure ndim=4 so the model forward takes the correct encode branch
            # 确保ndim=4以使模型前向传播走正确的编码分支
            if isinstance(pixel_v, np.ndarray) and pixel_v.ndim == 3:  # 如果是3维numpy数组
                pixel_v = np.expand_dims(pixel_v, 0)  # 扩展为4维
            mm_items.append(  # 添加多模态数据项
                MultimodalDataItem(
                    feature=pixel_v,  # 像素值特征
                    model_specific_data={  # 模型特定数据
                        "image_sizes": [image_s],  # 图像尺寸
                        "image_aspect_ratio": aspect_ratio,  # 宽高比模式
                    },
                    modality=modality,  # 模态类型
                )
            )

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            mm_items=mm_items,  # 多模态数据项
        )


class LlavaMultimodalProcessor(BaseMultimodalProcessor):  # LLaVA多模态处理器类
    """
    This is a wrapper class used to identify the multimodal processor for Llava architectures' vision model.
    # 这是一个包装类，用于识别LLaVA架构视觉模型的多模态处理器
    """

    models = [LlavaForConditionalGeneration, Mistral3ForConditionalGeneration]  # 关联的模型列表

    def _get_sgl_processor_cls(self, model_type: str):  # 根据模型类型获取SGLang处理器类
        if model_type == "clip_vision_model":  # 如果是CLIP视觉模型
            return LlavaImageProcessor  # 返回LLaVA图像处理器
        if hf_name := HF_MAPPING_NAMES.get(model_type):  # 获取HF映射名称
            sgl_mm_processor_set = sgl_mm_processor_util.PROCESSOR_MAPPING.values()  # 获取SGLang处理器映射
            sgl_processor_cls = list(  # 过滤查找匹配的处理器类
                filter(lambda p: p.__name__ == hf_name, sgl_mm_processor_set)
            )
            if sgl_processor_cls:  # 如果找到匹配的处理器
                return sgl_processor_cls[0]  # 返回第一个匹配的处理器
        raise ValueError(  # 找不到处理器则抛出异常
            f"Cannot find corresponding multimodal processor registered in sglang for model type `{model_type}`"
        )

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化LLaVA多模态处理器
        assert hasattr(hf_config, "vision_config")  # 断言配置中有vision_config
        assert hasattr(hf_config, "text_config")  # 断言配置中有text_config
        self.vision_config = hf_config.vision_config  # 保存视觉配置
        self.text_config = hf_config.text_config  # 保存文本配置
        self.hf_config = hf_config  # 保存HF配置

        if vision_type := getattr(self.vision_config, "model_type"):  # 获取视觉模型类型
            self.inner = self._get_sgl_processor_cls(vision_type)(  # 根据类型获取并实例化处理器
                hf_config, server_args, _processor, *args, **kwargs  # 传递初始化参数
            )
        else:
            raise ValueError(  # 找不到视觉模型类型则抛出异常
                f"Required `vision_config.model_type` is not found in hf_config: `{hf_config}`"
            )

    async def process_mm_data_async(self, *args, **kwargs):  # 异步处理多模态数据
        return await self.inner.process_mm_data_async(*args, **kwargs)  # 委托给内部处理器
