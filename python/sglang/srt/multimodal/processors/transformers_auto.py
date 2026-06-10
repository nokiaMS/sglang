# Transformers自动多模态处理器模块
# 本模块提供通用的多模态处理器，直接使用HuggingFace处理器处理多模态数据
# 适用于Gemma3、PaliGemma等使用非平凡标记扩展的模型

from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch

from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalProcessorOutput,  # 多模态处理器输出
)
from sglang.srt.multimodal.processors.base_processor import (  # 导入基础多模态处理器和特殊标记类
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_image  # 导入图像加载工具


def _first_attr(obj, names: tuple[str, ...], default=None):  # 从对象中获取第一个非None的属性值
    for name in names:  # 遍历属性名列表
        value = getattr(obj, name, None)  # 获取属性值
        if value is not None:  # 如果不为None
            return value  # 返回值
    return default  # 所有属性都为None则返回默认值


def _uses_mrope(hf_config) -> bool:  # 检查模型是否使用M-RoPE位置编码
    text_config = getattr(hf_config, "text_config", hf_config)  # 获取文本配置
    rope_scaling = getattr(text_config, "rope_scaling", None) or {}  # 获取RoPE缩放配置
    if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:  # 如果包含mrope_section
        return True  # 使用M-RoPE
    rope_type = str(getattr(text_config, "rope_type", "")).lower()  # 获取RoPE类型
    return "mrope" in rope_type  # 检查类型中是否包含mrope


class TransformersAutoMultimodalProcessor(BaseMultimodalProcessor):  # Transformers自动多模态处理器
    """Generic multimodal processor for the Transformers backend.

    Unlike model-specific processors that rely on regex-based token matching
    in the raw prompt, this processor applies the HF processor directly to
    the prompt text + raw media.  This handles models like Gemma3 where the
    chat template uses a marker (``<start_of_image>``) that the HF processor
    internally expands into placeholder tokens.
    """

    models = []  # 模型列表为空，由外部注册

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化自动多模态处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        self.mm_tokens = MultimodalSpecialTokens(  # 创建多模态特殊标记对象
            image_token=getattr(_processor, "image_token", None),  # 图像标记
            video_token=getattr(_processor, "video_token", None),  # 视频标记
            audio_token=getattr(_processor, "audio_token", None),  # 音频标记
            image_token_id=_first_attr(  # 图像标记ID
                hf_config,
                ("image_token_id", "image_token_index", "im_token_id"),  # 尝试多个属性名
            ),
            video_token_id=_first_attr(  # 视频标记ID
                hf_config,
                ("video_token_id",),  # 视频标记ID属性名
            ),
            audio_token_id=_first_attr(  # 音频标记ID
                hf_config,
                ("audio_token_id",),  # 音频标记ID属性名
            ),
        ).build(_processor)  # 构建标记对象

        self._is_mrope = _uses_mrope(hf_config)  # 检查是否使用M-RoPE
        if self._is_mrope:  # 如果使用M-RoPE
            vision_config = getattr(hf_config, "vision_config", None)  # 获取视觉配置
            self._spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)  # 空间合并尺寸
            self._tokens_per_second = getattr(vision_config, "tokens_per_second", None)  # 每秒标记数
            self._vision_start_token_id = _first_attr(  # 视觉起始标记ID
                hf_config, ("vision_start_token_id",)
            )
            self._model_type = getattr(hf_config, "model_type", "")  # 模型类型

    def _compute_mrope_positions(  # 计算M-RoPE位置编码
        self,
        input_ids: list[int],  # 输入ID列表
        image_grid_thw: Optional[torch.Tensor] = None,  # 图像网格
        video_grid_thw: Optional[torch.Tensor] = None,  # 视频网格
    ):
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding  # 导入M-RoPE嵌入

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # 转换为张量
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(  # 计算M-RoPE位置索引
            spatial_merge_size=self._spatial_merge_size,  # 空间合并尺寸
            image_token_id=self.mm_tokens.image_token_id,  # 图像标记ID
            video_token_id=self.mm_tokens.video_token_id or -1,  # 视频标记ID
            vision_start_token_id=self._vision_start_token_id,  # 视觉起始标记ID
            model_type=self._model_type,  # 模型类型
            input_ids=input_ids_tensor,  # 输入ID张量
            image_grid_thw=image_grid_thw,  # 图像网格
            video_grid_thw=video_grid_thw,  # 视频网格
            tokens_per_second=self._tokens_per_second,  # 每秒标记数
        )
        return mrope_positions.squeeze(1), mrope_position_delta  # 返回位置和增量

    def _load_images(self, image_data) -> list:  # 加载图像数据
        """Download / decode images from URLs, file paths, or base64."""  # 从URL、文件路径或base64下载/解码图像
        if not image_data:  # 如果没有图像数据
            return []  # 返回空列表
        images = []  # 图像列表
        for data in image_data:  # 遍历图像数据
            img, _ = load_image(data)  # 加载图像
            if img.mode != "RGB":  # 如果不是RGB模式
                img = img.convert("RGB")  # 转换为RGB
            images.append(img)  # 添加到列表
        return images  # 返回图像列表

    def _apply_hf_processor(self, text: str, images=None, videos=None):  # 应用HuggingFace处理器
        """Run the HF processor on text + media and return the full output.

        This is the key method that makes the generic processor work for
        models with non-trivial token expansion (Gemma3, PaliGemma, etc.).
        The HF processor handles chat-template expansion, image token
        insertion, and tokenization in one shot.
        """
        kwargs = {}  # 关键字参数
        if images:  # 如果有图像
            kwargs["images"] = images  # 添加图像参数
        if videos:  # 如果有视频
            kwargs["videos"] = videos  # 添加视频参数
        return self._processor(text=text, return_tensors="pt", **kwargs)  # 调用处理器

    def _build_mm_items(  # 构建多模态数据项
        self, processor_output: dict, input_ids: torch.Tensor  # 处理器输出和输入ID
    ) -> list[MultimodalDataItem]:
        """Extract MultimodalDataItem objects from the HF processor output."""  # 从HF处理器输出中提取多模态数据项
        items = self.collect_mm_items_from_processor_output(processor_output)  # 收集数据项

        modality_to_token_id = {  # 模态到标记ID的映射
            Modality.IMAGE: self.mm_tokens.image_token_id,  # 图像标记ID
            Modality.VIDEO: self.mm_tokens.video_token_id,  # 视频标记ID
            Modality.AUDIO: self.mm_tokens.audio_token_id,  # 音频标记ID
        }

        for item in items:  # 遍历数据项
            token_id = modality_to_token_id.get(item.modality)  # 获取对应标记ID
            if token_id is not None:  # 如果标记ID存在
                item.offsets = self.get_mm_items_offset(input_ids, token_id)  # 计算偏移量

        return items  # 返回数据项列表

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data,  # 图像数据
        audio_data,  # 音频数据
        input_text,  # 输入文本
        request_obj,  # 请求对象
        **kwargs,  # 其他关键字参数
    ):
        video_data = getattr(request_obj, "video_data", None)  # 获取视频数据
        if video_data is not None and not isinstance(video_data, list):  # 如果视频数据不是列表
            video_data = [video_data]  # 转为列表

        # Load raw media
        images = self._load_images(image_data)  # 加载原始图像
        # TODO: video / audio loading when needed

        # Apply HF processor — handles token expansion internally
        processor_output = self._apply_hf_processor(  # 应用HF处理器
            text=input_text,  # 输入文本
            images=images or None,  # 图像
            videos=video_data or None,  # 视频
        )

        input_ids = processor_output["input_ids"].flatten()  # 获取并展平输入ID

        # Build mm_items from processor output
        mm_items = self._build_mm_items(processor_output, input_ids)  # 构建多模态数据项

        ret = MultimodalProcessorOutput(  # 创建处理器输出
            input_ids=input_ids.tolist(),  # 输入ID列表
            mm_items=mm_items,  # 多模态数据项
        )

        # Propagate token_type_ids for models that need it (Gemma3, PaliGemma)
        token_type_key = (  # 确定token_type_ids的键名
            "mm_token_type_ids"
            if "mm_token_type_ids" in processor_output  # 优先使用mm_token_type_ids
            else "token_type_ids"  # 否则使用token_type_ids
        )
        if token_type_key in processor_output:  # 如果存在token_type_ids
            ret.token_type_ids = processor_output[token_type_key].flatten().tolist()  # 设置token类型ID

        if self.mm_tokens.image_token_id is not None:  # 如果图像标记ID存在
            ret.im_token_id = self.mm_tokens.image_token_id  # 设置图像标记ID
        if self.mm_tokens.video_token_id is not None:  # 如果视频标记ID存在
            ret.video_token_id = self.mm_tokens.video_token_id  # 设置视频标记ID
        if self.mm_tokens.audio_token_id is not None:  # 如果音频标记ID存在
            ret.audio_token_id = self.mm_tokens.audio_token_id  # 设置音频标记ID

        image_start_id = _first_attr(  # 获取图像起始标记ID
            self.hf_config,
            ("image_start_token_id", "vision_start_token_id", "im_start_id"),  # 尝试多个属性名
        )
        image_end_id = _first_attr(  # 获取图像结束标记ID
            self.hf_config,
            ("image_end_token_id", "vision_end_token_id", "im_end_id"),  # 尝试多个属性名
        )
        if image_start_id is not None:  # 如果图像起始标记ID存在
            ret.im_start_id = image_start_id  # 设置图像起始标记ID
        if image_end_id is not None:  # 如果图像结束标记ID存在
            ret.im_end_id = image_end_id  # 设置图像结束标记ID

        # M-RoPE positions (Qwen2.5-VL, Qwen3-VL)
        if self._is_mrope:  # 如果使用M-RoPE
            image_grid_thw = processor_output.get("image_grid_thw")  # 获取图像网格
            video_grid_thw = processor_output.get("video_grid_thw")  # 获取视频网格
            mrope_positions, mrope_position_delta = self._compute_mrope_positions(  # 计算M-RoPE位置
                ret.input_ids,  # 输入ID
                image_grid_thw=image_grid_thw,  # 图像网格
                video_grid_thw=video_grid_thw,  # 视频网格
            )
            ret.mrope_positions = mrope_positions  # 设置M-RoPE位置
            ret.mrope_position_delta = mrope_position_delta  # 设置M-RoPE位置增量

        return ret  # 返回处理器输出
